#include "gtest/gtest.h"
#include "../thirdeye/nn/online_mode.hpp"
#include "../thirdeye/nn/spell_probe.hpp"
#include "../thirdeye/nn/survival_mode.hpp"
#include "../thirdeye/nn/rolling_save.hpp"
#include "../thirdeye/nn/online_policy.hpp"

#include <algorithm>
#include <array>
#include <unordered_set>
#include <chrono>
#include <filesystem>

namespace mode = THIRDEYE::nn::mode;

TEST(nnSpell, ReadsOnlyActualPreparedSpellCountersNotCastIds) {
    namespace spell = THIRDEYE::nn::spell;
    std::array<uint8_t, 200> counts{};
    counts[10] = 2;   // mage level 1: spell 0, two copies
    counts[19] = 1;   // mage level 1: spell 9
    counts[99] = 1;   // mage level 9: spell 9
    counts[110] = 3;  // cleric level 1: spell 0
    counts[199] = 2;  // cleric level 9: spell 9
    counts[0] = 75;   // leading reserved bytes are NOT spell counters
    counts[100] = 77; // cleric leading reserved bytes also ignored
    EXPECT_EQ(spell::available(counts, 0), 4);
    EXPECT_EQ(spell::available(counts, 1), 5);
    EXPECT_EQ(spell::available(counts, -1), -1);
    EXPECT_EQ(spell::available(counts, 2), -1);
    counts[10] = 0xff; // game uses -1 for unmemorized
    EXPECT_EQ(spell::available(counts, 0), 2);
    counts[19] = 11; // invalid count: do not advertise it
    EXPECT_EQ(spell::available(counts, 0), -1);
}

TEST(nnSpell, NativeProbeOnlyReportsVerifiedOpenAndClose) {
    using Probe = THIRDEYE::nn::spell::Probe;
    Probe probe;
    EXPECT_STREQ(probe.status(), "idle");
    EXPECT_FALSE(probe.begin(1, 2)); // another UI owns the screen
    EXPECT_FALSE(probe.begin(0, 0)); // no prepared resource
    ASSERT_TRUE(probe.begin(0, 2));
    EXPECT_TRUE(probe.shouldOpen());
    EXPECT_FALSE(probe.begin(0, 2)); // cannot queue concurrent actions
    probe.opened(5);
    EXPECT_TRUE(probe.shouldClose());
    EXPECT_STREQ(probe.status(), "open");
    probe.closed(0);
    EXPECT_EQ(probe.stage(), Probe::Stage::Verified);
    EXPECT_STREQ(probe.status(), "menu-cycle-verified");
    EXPECT_FALSE(probe.busy());

    ASSERT_TRUE(probe.begin(0, 1));
    probe.opened(0); // activation failed to switch to native menu
    probe.closed(0);
    EXPECT_EQ(probe.stage(), Probe::Stage::Unverified);
    EXPECT_STREQ(probe.status(), "menu-transition-unverified");
    ASSERT_TRUE(probe.begin(0, 1));
    probe.fail();
    EXPECT_EQ(probe.stage(), Probe::Stage::Failed);
    EXPECT_FALSE(probe.shouldClose());
}

TEST(nnSurvival, RetreatsOnlyWithCriticalHPAndVerifiedEscape) {
    namespace v = THIRDEYE::nn::survival;
    v::Party party;
    party.complete = true;
    party.ids[0] = 50;
    party.ids[1] = 51;
    party.hp[0] = 8;
    party.maximum[0] = 20;
    party.hp[1] = 20;
    party.maximum[1] = 20;
    party.food[0] = party.food[1] = 60;

    EXPECT_TRUE(v::valid(party));
    EXPECT_DOUBLE_EQ(v::weakestHealth(party), 0.4);
    const std::array<bool, 4> passable{false, false, true, false};
    const auto allowed = v::exits(passable);
    EXPECT_FALSE(allowed[0]);   // forward toward an enemy is prohibited
    EXPECT_FALSE(allowed[1]);   // rear wall
    EXPECT_TRUE(allowed[2]);    // left strafe
    EXPECT_TRUE(allowed[3]);    // right strafe
    EXPECT_FALSE(allowed[4]);
    EXPECT_FALSE(allowed[5]);
    EXPECT_TRUE(v::shouldWithdraw(party, true, passable));
    EXPECT_FALSE(v::shouldWithdraw(party, false, passable));
    EXPECT_FALSE(v::shouldWithdraw(party, true, {true, true, true, true}));

    party.hp[0] = 12; // 60% HP; only retreats early if no resources carried
    party.rations = 1;
    EXPECT_FALSE(v::shouldWithdraw(party, true, passable));
    party.rations = 0;
    EXPECT_TRUE(v::shouldWithdraw(party, true, passable));
    party.hp[0] = 0;
    party.hp[1] = 0;
    EXPECT_FALSE(v::shouldWithdraw(party, true, passable));
}

TEST(nnSurvival, RewardsObservedEscapeAndPenalizesInjuryAndFoodCost) {
    namespace v = THIRDEYE::nn::survival;
    v::Party before;
    before.complete = true;
    before.ids[0] = 101;
    before.hp[0] = 6;
    before.maximum[0] = 20;
    before.food[0] = 90;
    before.rations = 1;
    before.potions = 2;

    auto after = before;
    auto safe = v::evaluate(before, after, true, false);
    ASSERT_TRUE(safe);
    EXPECT_TRUE(safe->success);
    EXPECT_NEAR(safe->reward, 1.0, 0.000001);
    EXPECT_EQ(safe->hpDelta, 0);
    EXPECT_EQ(safe->foodDelta, 0);

    after.hp[0] = 3;
    after.food[0] = 65;
    const auto injured = v::evaluate(before, after, true, false);
    ASSERT_TRUE(injured);
    EXPECT_FALSE(injured->success);
    EXPECT_LT(injured->reward, safe->reward);
    EXPECT_EQ(injured->hpDelta, -3);
    EXPECT_EQ(injured->foodDelta, -25);

    const auto failed = v::evaluate(before, before, false, true);
    ASSERT_TRUE(failed);
    EXPECT_FALSE(failed->success);
    EXPECT_LT(failed->reward, 0.0);

    after = before;
    after.ids[0] = 102; // new party after load/teleport: no stale reward
    EXPECT_FALSE(v::evaluate(before, after, true, false));
    after = before;
    after.maximum[0] = 19;
    EXPECT_FALSE(v::evaluate(before, after, true, false));
    before.complete = false;
    EXPECT_FALSE(v::evaluate(before, after, true, false));
}

TEST(nnSurvival, ContextAndCheckpointRetainLearnedEscapePreference) {
    namespace v = THIRDEYE::nn::survival;
    v::Party party;
    party.complete = true;
    party.ids[0] = 102;
    party.hp[0] = 7;
    party.maximum[0] = 20;
    party.food[0] = 45;
    party.potions = 2;
    const auto dungeon = THIRDEYE::nn::OnlinePolicy::features(
        12, 5, 0, {false, false, false, false});
    const auto learned = v::features(dungeon, party);
    EXPECT_NEAR(learned[7], 0.35, 0.000001);
    EXPECT_NEAR(learned[8], 0.45, 0.000001);
    EXPECT_DOUBLE_EQ(learned[9], 1.0);
    EXPECT_DOUBLE_EQ(learned[10], 0.0);

    THIRDEYE::nn::OnlinePolicy survivalPolicy;
    const auto allowed = v::exits({false, false, false, false});
    for (int i = 0; i < 180; ++i)
        ASSERT_TRUE(survivalPolicy.reinforce(learned, 3, 1.0, true));
    EXPECT_EQ(survivalPolicy.select(learned, allowed, 1.0, 0.5), 3);

    const auto dir = std::filesystem::temp_directory_path() /
        ("thirdeye_survival_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    const auto file = dir / "survival.ckpt";
    std::string error;
    ASSERT_TRUE(survivalPolicy.save(file, error)) << error;
    THIRDEYE::nn::OnlinePolicy loaded;
    ASSERT_TRUE(loaded.load(file, error)) << error;
    EXPECT_EQ(loaded.samples(), 180u);
    EXPECT_EQ(loaded.weights(), survivalPolicy.weights());
    EXPECT_EQ(loaded.select(learned, allowed, 1.0, 0.5), 3);
    EXPECT_EQ(loaded.successes(), survivalPolicy.successes());
    EXPECT_DOUBLE_EQ(loaded.meanReward(), survivalPolicy.meanReward());
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
}

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

TEST(nnMode, CombatTargetsOnlyLivingEnemiesInFrontCellOnSameFloor) {
    const mode::Pose north{10, 10, 0, 3};
    EXPECT_TRUE(mode::targetDirectlyAhead(north, {10, 9, -1, 3}));
    EXPECT_FALSE(mode::targetDirectlyAhead(north, {11, 10, -1, 3}));
    EXPECT_FALSE(mode::targetDirectlyAhead(north, {10, 10, -1, 3}));
    EXPECT_FALSE(mode::targetDirectlyAhead(north, {10, 9, -1, 4}));
    EXPECT_FALSE(mode::targetDirectlyAhead(north, {10, 8, -1, 3}));
    EXPECT_FALSE(mode::targetDirectlyAhead(north, {255, 255, -1, 3}));
    EXPECT_TRUE(mode::targetDirectlyAhead({10, 10, 1, 3}, {11, 10, -1, 3}));
    EXPECT_TRUE(mode::targetDirectlyAhead({10, 10, 2, 3}, {10, 11, -1, 3}));
    EXPECT_TRUE(mode::targetDirectlyAhead({10, 10, 3, 3}, {9, 10, -1, 3}));
}

TEST(nnMode, CombatRewardDependsOnObservedDamageNotClickedButton) {
    const auto dealt = mode::evaluateCombatOutcome(18, 10);
    ASSERT_TRUE(dealt);
    EXPECT_TRUE(dealt->success);
    EXPECT_GT(dealt->reward, 0);
    const auto blocked = mode::evaluateCombatOutcome(18, 18);
    ASSERT_TRUE(blocked);
    EXPECT_FALSE(blocked->success);
    EXPECT_LT(blocked->reward, 0);
    const auto killed = mode::evaluateCombatOutcome(18, 0);
    ASSERT_TRUE(killed);
    EXPECT_DOUBLE_EQ(killed->reward, 2.0);
    EXPECT_FALSE(mode::evaluateCombatOutcome(18, 20)); // healing/another NPC
    EXPECT_FALSE(mode::evaluateCombatOutcome(18, -1)); // missing reading
    EXPECT_FALSE(mode::evaluateCombatOutcome(0, 0)); // already dead
}

TEST(nnRollingSave, OneMinuteIntervalsNoBurstWhenReturningFromPause) {
    THIRDEYE::nn::rolling::Schedule clock;
    using Schedule = THIRDEYE::nn::rolling::Schedule;
    const auto t = Schedule::Time{};
    const auto alive = Schedule::Party::Alive;
    clock.enable(t);
    EXPECT_FALSE(clock.shouldSave(t + std::chrono::seconds(59), true, alive));
    EXPECT_FALSE(clock.shouldSave(t + std::chrono::seconds(60), false, alive));
    EXPECT_TRUE(clock.shouldSave(t + std::chrono::seconds(70), true, alive));
    EXPECT_FALSE(clock.shouldSave(t + std::chrono::seconds(70), true, alive));
    EXPECT_FALSE(clock.shouldSave(t + std::chrono::seconds(129), true, alive));
    EXPECT_TRUE(clock.shouldSave(t + std::chrono::seconds(130), true, alive));
    clock.disable();
    EXPECT_FALSE(clock.shouldSave(t + std::chrono::minutes(10), true, alive));
}

TEST(nnRollingSave, GameOverLoadsAfterExactlyFiveSecondsNotBefore) {
    using Schedule = THIRDEYE::nn::rolling::Schedule;
    Schedule clock;
    const auto t = Schedule::Time{};
    clock.enable(t);
    EXPECT_FALSE(clock.shouldRestore(t + std::chrono::seconds(10),
                                     Schedule::Party::Alive));
    EXPECT_FALSE(clock.shouldRestore(t + std::chrono::seconds(20),
                                     Schedule::Party::Dead));
    EXPECT_TRUE(clock.gameOverPending());
    // The character objects may disappear when the game-over screen opens.
    EXPECT_FALSE(clock.shouldRestore(t + std::chrono::seconds(24),
                                     Schedule::Party::Unknown));
    EXPECT_TRUE(clock.shouldRestore(t + std::chrono::seconds(25),
                                    Schedule::Party::Unknown));
    clock.afterRestore(t + std::chrono::seconds(25));
    EXPECT_FALSE(clock.gameOverPending());
    EXPECT_FALSE(clock.shouldSave(t + std::chrono::seconds(84), true,
                                  Schedule::Party::Alive));
    EXPECT_TRUE(clock.shouldSave(t + std::chrono::seconds(85), true,
                                 Schedule::Party::Alive));
}

TEST(nnRollingSave, ARevivedPartyCancelsTheGameOverCountdown) {
    using Schedule = THIRDEYE::nn::rolling::Schedule;
    Schedule clock;
    const auto t = Schedule::Time{};
    clock.enable(t);
    EXPECT_FALSE(clock.shouldRestore(t, Schedule::Party::Dead));
    EXPECT_FALSE(clock.shouldRestore(t + std::chrono::seconds(3),
                                     Schedule::Party::Alive));
    EXPECT_FALSE(clock.gameOverPending());
    EXPECT_FALSE(clock.shouldRestore(t + std::chrono::seconds(5),
                                     Schedule::Party::Dead));
    EXPECT_FALSE(clock.shouldRestore(t + std::chrono::seconds(9),
                                     Schedule::Party::Unknown));
    EXPECT_TRUE(clock.shouldRestore(t + std::chrono::seconds(10),
                                    Schedule::Party::Dead));
    clock.disable();
    EXPECT_FALSE(clock.shouldRestore(t + std::chrono::seconds(20),
                                     Schedule::Party::Dead));
}

TEST(nnRollingSave, NamesAndReservedSlotsAreStable) {
    using namespace THIRDEYE::nn::rolling;
    EXPECT_EQ(slotForIndex(0), 12);
    EXPECT_EQ(slotForIndex(1), 13);
    EXPECT_EQ(slotForIndex(2), 14);
    EXPECT_STREQ(nameForIndex(0), "nn-1");
    EXPECT_STREQ(nameForIndex(1), "nn-2");
    EXPECT_STREQ(nameForIndex(2), "nn-3");
    for (int index = 0; index < kSaveCount; ++index)
        EXPECT_EQ(indexForSlot(slotForIndex(index)), index);
}
