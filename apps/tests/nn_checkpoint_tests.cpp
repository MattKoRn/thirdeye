#include "gtest/gtest.h"
#include "../thirdeye/nn/online_policy.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>

namespace {
namespace fs = std::filesystem;
using THIRDEYE::nn::OnlinePolicy;

struct TempCheckpoint {
    fs::path dir;
    fs::path path;
    TempCheckpoint() {
        dir = fs::temp_directory_path() /
            ("thirdeye_nn_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
        path = dir / "online.ckpt";
    }
    ~TempCheckpoint() { std::error_code ec; fs::remove_all(dir, ec); }
};

std::string readAll(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
void writeAll(const fs::path &path, const std::string &bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
}
std::string withChecksum(const std::string &payload) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char c : payload) {
        hash ^= c;
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream out;
    out << payload << "checksum " << std::hex << std::setw(16)
        << std::setfill('0') << hash << '\n';
    return out.str();
}
const OnlinePolicy::Features kContext =
    OnlinePolicy::features(12, 10, 1, {false, true, false, true});
}

TEST(nnCheckpoint, LearningAndLoadedWeightsApplyImmediately) {
    TempCheckpoint tmp;
    OnlinePolicy live;
    std::string error;
    for (int i = 0; i < 150; ++i)
        ASSERT_GE(live.learn(kContext, 3), 0.0);
    ASSERT_EQ(live.predict(kContext).action, 3);
    EXPECT_GT(live.predict(kContext).confidence, 0.9);
    const auto savedWeights = live.weights();
    ASSERT_TRUE(live.save(tmp.path, error)) << error;

    for (int i = 0; i < 300; ++i)
        ASSERT_GE(live.learn(kContext, 5), 0.0);
    ASSERT_EQ(live.predict(kContext).action, 5);
    ASSERT_TRUE(live.load(tmp.path, error)) << error;
    EXPECT_EQ(live.samples(), 150u);
    EXPECT_EQ(live.weights(), savedWeights);
    EXPECT_EQ(live.predict(kContext).action, 3);
}

TEST(nnCheckpoint, FailedLoadNeverMutatesLiveModel) {
    TempCheckpoint tmp;
    OnlinePolicy live;
    std::string error;
    for (int i = 0; i < 60; ++i) live.learn(kContext, 2);
    const auto weights = live.weights();
    ASSERT_TRUE(live.save(tmp.path, error)) << error;
    const auto intact = readAll(tmp.path);

    writeAll(tmp.path, intact.substr(0, intact.size() / 2));
    EXPECT_FALSE(live.load(tmp.path, error));
    EXPECT_EQ(live.weights(), weights);
    EXPECT_EQ(live.samples(), 60u);

    auto changed = intact;
    changed[20] ^= 1;
    writeAll(tmp.path, changed);
    EXPECT_FALSE(live.load(tmp.path, error));
    EXPECT_EQ(live.weights(), weights);
    EXPECT_EQ(live.samples(), 60u);
}

TEST(nnCheckpoint, RejectsNonFiniteWeightsEvenWithValidChecksum) {
    TempCheckpoint tmp;
    OnlinePolicy live;
    std::string error;
    live.learn(kContext, 1);
    const auto old = live.weights();
    ASSERT_TRUE(live.save(tmp.path, error)) << error;
    const auto raw = readAll(tmp.path);
    auto data = raw.substr(0, raw.rfind("checksum "));
    const auto first = data.find('\n', data.find('\n') + 1) + 1;
    const auto end = data.find('\n', first);
    data.replace(first, end - first, "nan");
    writeAll(tmp.path, withChecksum(data));
    EXPECT_FALSE(live.load(tmp.path, error));
    EXPECT_EQ(live.weights(), old);
    EXPECT_EQ(live.samples(), 1u);
}

TEST(nnCheckpoint, AtomicOverwriteAndInvalidDestination) {
    TempCheckpoint tmp;
    OnlinePolicy live;
    std::string error;
    for (int i = 0; i < 40; ++i) live.learn(kContext, 0);
    ASSERT_TRUE(live.save(tmp.path, error)) << error;
    const auto initial = readAll(tmp.path);
    for (int i = 0; i < 40; ++i) live.learn(kContext, 4);
    ASSERT_TRUE(live.save(tmp.path, error)) << error;
    EXPECT_NE(readAll(tmp.path), initial);

    OnlinePolicy restored;
    ASSERT_TRUE(restored.load(tmp.path, error)) << error;
    EXPECT_EQ(restored.weights(), live.weights());
    EXPECT_EQ(restored.samples(), 80u);

    EXPECT_FALSE(live.save(tmp.dir / "missing" / "online.ckpt", error));
    ASSERT_TRUE(restored.load(tmp.path, error)) << error;
}

TEST(nnCheckpoint, InvalidFeaturesAndKeysCannotCorruptWeights) {
    OnlinePolicy live;
    auto features = kContext;
    features[1] = 1.5;
    EXPECT_LT(live.learn(features, 0), 0);
    EXPECT_EQ(live.predict(features).action, -1);
    EXPECT_LT(live.learn(kContext, 6), 0);
    EXPECT_EQ(live.samples(), 0u);
    EXPECT_EQ(OnlinePolicy::actionForKey(0x4800), 0);
    EXPECT_EQ(OnlinePolicy::actionForKey(0x1234), -1);
    EXPECT_EQ(OnlinePolicy::keyForAction(5), 0x4900);
    EXPECT_EQ(OnlinePolicy::keyForAction(6), -1);
}
