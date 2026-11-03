#include "gtest/gtest.h"
#include "../thirdeye/nn/online_policy.hpp"
#include "../thirdeye/nn/shared_brain.hpp"

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

TEST(nnCheckpoint, MergesLegacyEOB3AndDungeonHackBrainsExactlyOnce) {
    namespace mode = THIRDEYE::nn::mode;
    TempCheckpoint tmp;
    const auto shared = mode::brainFileForDocuments(tmp.dir);
    fs::create_directories(shared.parent_path());
    OnlinePolicy eob;
    OnlinePolicy hack;
    for (int i = 0; i < 40; ++i)
        ASSERT_TRUE(eob.reinforce(kContext, 0, 1.8, true));
    for (int i = 0; i < 60; ++i)
        ASSERT_TRUE(hack.reinforce(kContext, 4, -1.25, false));
    const auto eobLegacy = mode::legacyBrainFileForDocuments(tmp.dir, false);
    const auto hackLegacy = mode::legacyBrainFileForDocuments(tmp.dir, true);
    std::string error;
    ASSERT_TRUE(eob.save(eobLegacy, error)) << error;
    ASSERT_TRUE(hack.save(hackLegacy, error)) << error;
    const auto originalEob = readAll(eobLegacy);
    const auto originalHack = readAll(hackLegacy);

    OnlinePolicy merged;
    const auto migration = mode::restoreSharedBrain(merged, tmp.dir);
    ASSERT_TRUE(migration.ready) << migration.error;
    EXPECT_TRUE(migration.migrated);
    EXPECT_TRUE(migration.restored);
    EXPECT_TRUE(fs::exists(shared));
    EXPECT_EQ(merged.samples(), 100u);
    EXPECT_EQ(merged.successes(), 40u);
    EXPECT_EQ(merged.failures(), 60u);
    EXPECT_EQ(readAll(eobLegacy), originalEob);
    EXPECT_EQ(readAll(hackLegacy), originalHack);
    for (std::size_t i = 0; i < merged.weights().size(); ++i)
        EXPECT_NEAR(merged.weights()[i],
                    0.4 * eob.weights()[i] + 0.6 * hack.weights()[i], 1e-12);
    EXPECT_NEAR(merged.meanReward(),
                0.4 * eob.meanReward() + 0.6 * hack.meanReward(), 1e-12);

    // A fresh process (or a new game in the other executable) must use
    // shared learned weights immediately, not re-import legacy training.
    for (int i = 0; i < 12; ++i)
        ASSERT_TRUE(hack.reinforce(kContext, 1, 1.0, true));
    ASSERT_TRUE(hack.save(hackLegacy, error)) << error;
    OnlinePolicy newSession;
    const auto loaded = mode::restoreSharedBrain(newSession, tmp.dir);
    ASSERT_TRUE(loaded.ready) << loaded.error;
    EXPECT_FALSE(loaded.migrated);
    EXPECT_TRUE(loaded.restored);
    EXPECT_EQ(newSession.weights(), merged.weights());
    EXPECT_EQ(newSession.samples(), 100u);
}

TEST(nnCheckpoint, CorruptedSharedBrainIsNeverOverwrittenByLegacyMigration) {
    namespace mode = THIRDEYE::nn::mode;
    TempCheckpoint tmp;
    const auto shared = mode::brainFileForDocuments(tmp.dir);
    fs::create_directories(shared.parent_path());
    const std::string broken = "partial shared checkpoint";
    writeAll(shared, broken);
    OnlinePolicy oldBrain;
    ASSERT_TRUE(oldBrain.reinforce(kContext, 2, 1.8, true));
    std::string error;
    ASSERT_TRUE(oldBrain.save(
        mode::legacyBrainFileForDocuments(tmp.dir, false), error)) << error;
    OnlinePolicy running;
    const auto outcome = mode::restoreSharedBrain(running, tmp.dir);
    EXPECT_FALSE(outcome.ready);
    EXPECT_FALSE(outcome.restored);
    EXPECT_FALSE(outcome.migrated);
    EXPECT_EQ(readAll(shared), broken);
    EXPECT_EQ(running.samples(), 0u);
}

TEST(nnCheckpoint, OneBadLegacyBrainCannotEraseOtherGamesLearning) {
    namespace mode = THIRDEYE::nn::mode;
    TempCheckpoint tmp;
    const auto shared = mode::brainFileForDocuments(tmp.dir);
    fs::create_directories(shared.parent_path());
    OnlinePolicy eob;
    ASSERT_TRUE(eob.reinforce(kContext, 3, 1.8, true));
    std::string error;
    ASSERT_TRUE(eob.save(
        mode::legacyBrainFileForDocuments(tmp.dir, false), error)) << error;
    const auto badLegacy = mode::legacyBrainFileForDocuments(tmp.dir, true);
    writeAll(badLegacy, "not an ARJ or NN checkpoint");
    OnlinePolicy restored;
    const auto result = mode::restoreSharedBrain(restored, tmp.dir);
    ASSERT_TRUE(result.ready) << result.error;
    EXPECT_TRUE(result.migrated);
    EXPECT_TRUE(result.restored);
    EXPECT_FALSE(result.warning.empty());
    EXPECT_EQ(restored.weights(), eob.weights());
    EXPECT_EQ(restored.samples(), eob.samples());
    EXPECT_EQ(readAll(badLegacy), "not an ARJ or NN checkpoint");
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


TEST(nnCheckpoint, AutonomousRewardUpdatesWithoutPlayerDemonstrations) {
    OnlinePolicy brain;
    const std::array<bool, OnlinePolicy::kActions> allowed{true,true,true,true,true,true};
    EXPECT_EQ(brain.select(kContext, {false,false,false,false,true,false}, 0.1, 0.9), 4);
    EXPECT_EQ(brain.select(kContext, {false,false,false,false,false,false}, 0.1, 0.2), -1);
    EXPECT_EQ(brain.select(kContext, allowed, -0.5, 0.5), -1);

    for (int i = 0; i < 120; ++i)
        ASSERT_TRUE(brain.reinforce(kContext, 0, 1.8, true));
    EXPECT_EQ(brain.predict(kContext).action, 0);
    const auto before = brain.weights();
    for (int i = 0; i < 120; ++i)
        ASSERT_TRUE(brain.reinforce(kContext, 0, -1.25, false));
    EXPECT_NE(brain.weights(), before);
    EXPECT_EQ(brain.successes(), 120u);
    EXPECT_EQ(brain.failures(), 120u);
    EXPECT_DOUBLE_EQ(brain.successRate(), 0.5);
    EXPECT_EQ(brain.samples(), 240u);
}

TEST(nnCheckpoint, SelfLearnedProgressRoundTripsWithWeights) {
    TempCheckpoint tmp;
    OnlinePolicy brain;
    for (int i = 0; i < 30; ++i)
        ASSERT_TRUE(brain.reinforce(kContext, 1, i % 3 ? 1.8 : -1.25,
                                    i % 3 != 0));
    std::string error;
    ASSERT_TRUE(brain.save(tmp.path, error)) << error;
    OnlinePolicy restored;
    ASSERT_TRUE(restored.load(tmp.path, error)) << error;
    EXPECT_EQ(restored.weights(), brain.weights());
    EXPECT_EQ(restored.samples(), brain.samples());
    EXPECT_EQ(restored.successes(), brain.successes());
    EXPECT_EQ(restored.failures(), brain.failures());
    EXPECT_DOUBLE_EQ(restored.meanReward(), brain.meanReward());
    EXPECT_DOUBLE_EQ(restored.successRate(), brain.successRate());
}

TEST(nnCheckpoint, RejectsForgedAutonomousMetricsWithoutChangingBrain) {
    TempCheckpoint tmp;
    OnlinePolicy brain;
    ASSERT_TRUE(brain.reinforce(kContext, 2, 1.8, true));
    std::string error;
    ASSERT_TRUE(brain.save(tmp.path, error)) << error;
    const auto original = brain.weights();
    const auto raw = readAll(tmp.path);
    const auto headerEnd = raw.find('\n');
    const auto metricsEnd = raw.find('\n', headerEnd + 1);
    const auto originalPayload = raw.substr(0, raw.rfind("checksum "));
    std::string payload = originalPayload;
    payload.replace(headerEnd + 1, metricsEnd - headerEnd - 1,
                    "1 1 999 0 0");
    writeAll(tmp.path, withChecksum(payload));
    EXPECT_FALSE(brain.load(tmp.path, error));
    EXPECT_EQ(brain.weights(), original);
    EXPECT_EQ(brain.samples(), 1u);
    EXPECT_EQ(brain.successes(), 1u);
}

TEST(nnCheckpoint, LoadsLegacyV1BrainAndUpgradesOnNextSave) {
    TempCheckpoint tmp;
    OnlinePolicy previous;
    previous.learn(kContext, 3);
    std::string error;
    ASSERT_TRUE(previous.save(tmp.path, error)) << error;
    std::string saved = readAll(tmp.path);
    auto data = saved.substr(0, saved.rfind("checksum "));
    const auto nl = data.find('\n');
    const auto metricsEnd = data.find('\n', nl + 1);
    data.replace(0, nl, "THIRDEYE_NN 1");
    std::string metrics = data.substr(nl + 1, metricsEnd - nl - 1);
    const auto firstSpace = metrics.find(' ');
    metrics = metrics.substr(0, metrics.find(' ', firstSpace + 1));
    data.replace(nl + 1, metricsEnd - nl - 1, metrics);
    writeAll(tmp.path, withChecksum(data));
    OnlinePolicy restored;
    ASSERT_TRUE(restored.load(tmp.path, error)) << error;
    EXPECT_EQ(restored.weights(), previous.weights());
    EXPECT_EQ(restored.samples(), previous.samples());
    EXPECT_EQ(restored.successes(), 0u);
    EXPECT_EQ(restored.failures(), 0u);
}

TEST(nnCheckpoint, AdaptiveRateDecaysAndRestoresWithSamples) {
    TempCheckpoint tmp;
    OnlinePolicy brain;
    const auto initial = brain.learningRate();
    EXPECT_NEAR(initial, 0.10, 1e-12);
    for (int i = 0; i < 350; ++i)
        ASSERT_TRUE(brain.reinforce(kContext, 0, 1.8, true));
    const auto decayed = brain.learningRate();
    EXPECT_LT(decayed, initial);
    EXPECT_GT(decayed, 0.025);
    std::string error;
    ASSERT_TRUE(brain.save(tmp.path, error)) << error;
    OnlinePolicy restored;
    ASSERT_TRUE(restored.load(tmp.path, error)) << error;
    EXPECT_DOUBLE_EQ(restored.learningRate(), decayed);
    EXPECT_EQ(restored.weights(), brain.weights());
    ASSERT_TRUE(restored.reinforce(kContext, 2, -1.25, false));
    EXPECT_LT(restored.learningRate(), decayed);
}

TEST(nnCheckpoint, SeparateCombatPolicyLearnsAndRestoresWithoutMutatingMovement) {
    TempCheckpoint tmp;
    OnlinePolicy movement;
    OnlinePolicy combat;
    auto combatFeatures = kContext;
    combatFeatures[7] = 0.3;   // first party member HP fraction
    combatFeatures[8] = 0.8;   // second party member HP fraction
    combatFeatures[9] = 0.9;   // fourth party member HP fraction
    combatFeatures[10] = 0.25; // enemy HP context
    const std::array<bool, OnlinePolicy::kActions> available{
        true, true, true, false, false, false
    };
    const auto movementWeights = movement.weights();
    for (int n = 0; n < 150; ++n)
        ASSERT_TRUE(combat.reinforce(combatFeatures, 2, 1.6, true));
    EXPECT_EQ(combat.select(combatFeatures, available, 1.0, 0.5), 2);
    EXPECT_EQ(movement.weights(), movementWeights);
    const auto combatPath = tmp.dir / "combat.ckpt";
    std::string error;
    ASSERT_TRUE(combat.save(combatPath, error)) << error;
    ASSERT_TRUE(movement.save(tmp.path, error)) << error;
    OnlinePolicy restoredMovement;
    OnlinePolicy restoredCombat;
    ASSERT_TRUE(restoredMovement.load(tmp.path, error)) << error;
    ASSERT_TRUE(restoredCombat.load(combatPath, error)) << error;
    EXPECT_EQ(restoredCombat.weights(), combat.weights());
    EXPECT_EQ(restoredCombat.samples(), combat.samples());
    EXPECT_EQ(restoredCombat.select(combatFeatures, available, 1.0, 0.5), 2);
    EXPECT_EQ(restoredMovement.weights(), movementWeights);
    EXPECT_EQ(restoredMovement.samples(), 0u);
}
