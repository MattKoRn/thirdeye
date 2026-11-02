#include "gtest/gtest.h"
#include "../thirdeye/nn/online_mode.hpp"
#include "../thirdeye/nn/online_policy.hpp"

#include <algorithm>
#include <array>
#include <unordered_set>
#include <chrono>
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

TEST(nnMode, AutonomousActionRateIsOnePerElapsedSecond) {
    mode::ActionPacer pacer;
    using Clock = mode::ActionPacer::Clock;
    const auto start = Clock::time_point{};
    pacer.reset(start);

    EXPECT_FALSE(pacer.ready(start));
    // Thousands of host frames cannot cause an early action.
    for (int frame = 0; frame < 1000; ++frame)
        EXPECT_FALSE(pacer.ready(start + std::chrono::milliseconds(999)));
    EXPECT_TRUE(pacer.ready(start + std::chrono::seconds(1)));
    EXPECT_FALSE(pacer.ready(start + std::chrono::seconds(1)));
    EXPECT_FALSE(pacer.ready(start + std::chrono::milliseconds(1999)));
    EXPECT_TRUE(pacer.ready(start + std::chrono::seconds(2)));
}

TEST(nnMode, ActionPacerDoesNotBurstAfterLongGameStall) {
    mode::ActionPacer pacer;
    using Clock = mode::ActionPacer::Clock;
    const auto start = Clock::time_point{};
    pacer.reset(start);

    const auto resumed = start + std::chrono::minutes(5);
    EXPECT_TRUE(pacer.ready(resumed));
    // Five minutes elapsed, but exactly one action is authorized.
    EXPECT_FALSE(pacer.ready(resumed));
    EXPECT_FALSE(pacer.ready(resumed + std::chrono::milliseconds(999)));
    EXPECT_TRUE(pacer.ready(resumed + std::chrono::seconds(1)));
}

TEST(nnMode, TogglingAutonomyOrExitingMenuRestartsActionCooldown) {
    mode::ActionPacer pacer;
    using Clock = mode::ActionPacer::Clock;
    const auto start = Clock::time_point{};
    pacer.reset(start);
    EXPECT_TRUE(pacer.ready(start + std::chrono::seconds(1)));

    // setActive() and a non-gameplay pump both reset the same pacer.
    const auto resumed = start + std::chrono::milliseconds(1500);
    pacer.reset(resumed);
    EXPECT_FALSE(pacer.ready(resumed));
    EXPECT_FALSE(pacer.ready(resumed + std::chrono::milliseconds(999)));
    EXPECT_TRUE(pacer.ready(resumed + std::chrono::seconds(1)));
}

TEST(nnMode, AgentPrefersUnvisitedAccessibleCellsAndKeepsLevelMemory) {
    using THIRDEYE::nn::OnlinePolicy;
    const mode::Pose pose{10, 10, 0, 3};
    const std::array<bool, OnlinePolicy::kActions> allowed{
        true, true, false, true, true, true
    };
    std::unordered_set<uint32_t> visited{
        mode::visitKey(pose),
        mode::visitKey({10, 9, 0, 3}),
        mode::visitKey({11, 10, 0, 3})
    };
    auto frontier = mode::frontierActions(pose, allowed, visited);
    EXPECT_FALSE(frontier[0]);
    EXPECT_TRUE(frontier[1]);
    EXPECT_FALSE(frontier[2]);
    EXPECT_FALSE(frontier[3]);
    EXPECT_FALSE(frontier[4]);
    EXPECT_FALSE(frontier[5]);
    const auto otherLevel = mode::frontierActions({10, 10, 0, 4}, allowed, visited);
    EXPECT_TRUE(otherLevel[0]);
    EXPECT_TRUE(otherLevel[1]);
    EXPECT_TRUE(otherLevel[3]);
    visited.insert(mode::visitKey({10, 11, 0, 3}));
    frontier = mode::frontierActions(pose, allowed, visited);
    EXPECT_TRUE(std::none_of(frontier.begin(), frontier.end(),
                             [](bool value) { return value; }));
}

TEST(nnMode, LearnsValidStairTransitionsButNotScriptedTurns) {
    const mode::Pose before{8, 8, 1, 2};
    const auto descended = mode::evaluateOutcome(before, {24, 20, 1, 3},
                                                 0, false);
    EXPECT_TRUE(descended.success);
    EXPECT_DOUBLE_EQ(descended.reward, 2.0);
    const auto turningDuringTransition = mode::evaluateOutcome(
        before, {24, 20, 2, 3}, 5, true);
    EXPECT_FALSE(turningDuringTransition.success);
    EXPECT_LT(turningDuringTransition.reward, 0);
}
