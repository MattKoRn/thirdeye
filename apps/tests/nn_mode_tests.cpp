#include "gtest/gtest.h"
#include "../thirdeye/nn/online_mode.hpp"
#include "../thirdeye/nn/online_policy.hpp"

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

TEST(nnMode, RewardComesFromTheAgentsOwnObservedActions) {
    const mode::Pose start{10, 10, 0, 3};
    const auto moved = mode::evaluateOutcome(start, {10, 9, 0, 3}, 0, true);
    EXPECT_TRUE(moved.success);
    EXPECT_GT(moved.reward, 1.0);

    const auto revisited = mode::evaluateOutcome(start, {10, 9, 0, 3}, 0, false);
    EXPECT_TRUE(revisited.success);
    EXPECT_LT(revisited.reward, moved.reward);

    const auto blocked = mode::evaluateOutcome(start, start, 0, true);
    EXPECT_FALSE(blocked.success);
    EXPECT_LT(blocked.reward, 0.0);

    const auto rotated = mode::evaluateOutcome(start, {10, 10, 1, 3}, 5, false);
    EXPECT_TRUE(rotated.success);
    EXPECT_GT(rotated.reward, 0.0);

    const auto failedRotation = mode::evaluateOutcome(start, start, 4, false);
    EXPECT_FALSE(failedRotation.success);
    EXPECT_LT(failedRotation.reward, 0.0);
}

TEST(nnMode, PureOutcomeLearnerCanImproveWithoutPlayerExamples) {
    THIRDEYE::nn::OnlinePolicy learner;
    const auto context = THIRDEYE::nn::OnlinePolicy::features(
        10, 10, 0, {false, false, false, false});
    const mode::Pose before{10, 10, 0, 3};
    for (int i = 0; i < 150; ++i) {
        const auto outcome = mode::evaluateOutcome(
            before, {10, 9, 0, 3}, 0, true);
        ASSERT_TRUE(learner.reinforce(context, 0, outcome.reward, outcome.success));
    }
    EXPECT_EQ(learner.predict(context).action, 0);
    EXPECT_EQ(learner.samples(), 150u);
    EXPECT_EQ(learner.successes(), 150u);
    EXPECT_DOUBLE_EQ(learner.successRate(), 1.0);
}
