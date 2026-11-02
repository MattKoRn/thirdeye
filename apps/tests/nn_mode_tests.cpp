#include "gtest/gtest.h"
#include "../thirdeye/nn/online_mode.hpp"

#include <array>
#include <filesystem>

namespace mode = THIRDEYE::nn::mode;

TEST(nnMode, BrainLivesDirectlyInsideDocumentsThirdeyeNN) {
    const std::filesystem::path docs("my-documents");
    const auto file = mode::brainFileForDocuments(docs);
    EXPECT_EQ(file, docs / "ThirdeyeNN" / "brain.ckpt");
    EXPECT_EQ(file.parent_path().filename().string(), "ThirdeyeNN");
    EXPECT_EQ(file.filename().string(), "brain.ckpt");
}

TEST(nnMode, UntrainedTakeoverAvoidsKnownWalls) {
    EXPECT_EQ(mode::fallbackKey({false, false, false, false}), 0x4800);
    EXPECT_EQ(mode::fallbackKey({true, false, false, false}), 0x4900);
    EXPECT_EQ(mode::fallbackKey({true, true, false, false}), 0x4700);
    EXPECT_EQ(mode::fallbackKey({true, true, true, true}), 0x4700);
}
