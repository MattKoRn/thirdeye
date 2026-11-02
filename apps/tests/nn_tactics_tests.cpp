#include "gtest/gtest.h"
#include "../thirdeye/nn/tactical_memory.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
namespace tactics = THIRDEYE::nn::tactics;
struct TempMemory {
    std::filesystem::path dir;
    std::filesystem::path file;
    TempMemory() {
        dir = std::filesystem::temp_directory_path() /
            ("thirdeye_tactics_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(dir);
        file = dir / "weaknesses.ckpt";
    }
    ~TempMemory() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
};
}

TEST(nnTactics, ExperimentsWithUntriedAttacksThenExploitsObservedDamage) {
    tactics::Memory brain;
    const tactics::Signature sword{1904, 1324, tactics::Kind::Melee};
    const tactics::Signature scroll{1904, 1377, tactics::Kind::ScrollSpell, 12};
    const std::vector<tactics::Candidate> choices{{0, sword}, {3, scroll}};
    EXPECT_EQ(brain.select(choices, 0.0), 0);
    EXPECT_EQ(brain.select(choices, 0.99), 1);

    for (int i = 0; i < 16; ++i) {
        brain.learn(sword, 2);
        brain.learn(scroll, 12);
    }
    EXPECT_EQ(brain.select(choices, 0.0), 1);
    EXPECT_EQ(brain.experience(scroll).trials, 16u);
    EXPECT_EQ(brain.experience(scroll).hits, 16u);
    EXPECT_DOUBLE_EQ(brain.experience(scroll).damage, 192);
    EXPECT_EQ(brain.samples(), 32u);
    EXPECT_EQ(brain.entries(), 2u);
}

TEST(nnTactics, LearnsWeaknessesSeparatelyPerMonsterWeaponAndSpell) {
    tactics::Memory brain;
    const tactics::Signature vsMist{1904, 1377, tactics::Kind::ScrollSpell, 12};
    const tactics::Signature vsUndead{2000, 1377, tactics::Kind::ScrollSpell, 12};
    const tactics::Signature differentSpell{1904, 1377, tactics::Kind::ScrollSpell, 13};
    const tactics::Signature mace{1904, 1330, tactics::Kind::Melee};
    brain.learn(vsMist, 10);
    brain.learn(vsUndead, 0);
    brain.learn(differentSpell, 1);
    brain.learn(mace, 5);
    EXPECT_DOUBLE_EQ(brain.experience(vsMist).damage, 10);
    EXPECT_EQ(brain.experience(vsUndead).hits, 0u);
    EXPECT_DOUBLE_EQ(brain.experience(differentSpell).damage, 1);
    EXPECT_DOUBLE_EQ(brain.experience(mace).damage, 5);
    EXPECT_EQ(brain.entries(), 4u);
}

TEST(nnTactics, WeaknessesPersistAndApplyImmediatelyAfterReload) {
    TempMemory t;
    tactics::Memory brain;
    const tactics::Signature axe{1904, 1323, tactics::Kind::Melee};
    const tactics::Signature wand{1904, 1376, tactics::Kind::WandSpell, 5};
    const std::vector<tactics::Candidate> options{{0, axe}, {1, wand}};
    for (int i = 0; i < 12; ++i) {
        brain.learn(axe, 0);
        brain.learn(wand, 9);
    }
    std::string error;
    ASSERT_TRUE(brain.save(t.file, error)) << error;
    tactics::Memory restored;
    ASSERT_TRUE(restored.load(t.file, error)) << error;
    EXPECT_EQ(restored.samples(), 24u);
    EXPECT_EQ(restored.experience(wand).hits, 12u);
    EXPECT_EQ(restored.experience(axe).hits, 0u);
    EXPECT_EQ(restored.select(options, 0.75), 1);
}

TEST(nnTactics, DamagedCheckpointDoesNotEraseLiveLearning) {
    TempMemory t;
    tactics::Memory brain;
    const tactics::Signature sig{1904, 1330, tactics::Kind::Melee};
    brain.learn(sig, 8);
    std::string error;
    ASSERT_TRUE(brain.save(t.file, error)) << error;
    std::ifstream in(t.file, std::ios::binary);
    std::string bytes(std::istreambuf_iterator<char>{in},
                      std::istreambuf_iterator<char>{});
    ASSERT_GT(bytes.size(), 20u);
    bytes[18] = bytes[18] == '0' ? '1' : '0';
    {
        std::ofstream out(t.file, std::ios::binary | std::ios::trunc);
        out << bytes;
    }
    EXPECT_FALSE(brain.load(t.file, error));
    EXPECT_EQ(brain.samples(), 1u);
    EXPECT_DOUBLE_EQ(brain.experience(sig).damage, 8);
}

TEST(nnTactics, DoesNotLearnInvalidObservationsOrChooseWithoutCandidates) {
    tactics::Memory brain;
    const tactics::Signature sig{1904, 1324, tactics::Kind::Melee};
    brain.learn(sig, -1);
    brain.learn(sig, 40000);
    brain.learn({0, 1324, tactics::Kind::Melee}, 10);
    EXPECT_EQ(brain.samples(), 0u);
    EXPECT_EQ(brain.select({}, 0.1), -1);
    EXPECT_EQ(brain.select({{0, sig}}, -1.0), -1);
    EXPECT_STREQ(tactics::kindName(tactics::Kind::ScrollSpell), "scroll-spell");
}
