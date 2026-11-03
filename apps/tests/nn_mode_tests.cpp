#include "gtest/gtest.h"
#include "../thirdeye/nn/online_mode.hpp"
#include "../thirdeye/nn/spell_probe.hpp"
#include "../thirdeye/nn/survival_mode.hpp"
#include "../thirdeye/nn/food_learning.hpp"
#include "../thirdeye/nn/food_actuator.hpp"
#include "../thirdeye/nn/advice_menu.hpp"
#include "../thirdeye/game_boot.hpp"
#include "../thirdeye/nn/dungeon_hack_context.hpp"
#include "../thirdeye/nn/rolling_save.hpp"
#include "../thirdeye/nn/online_policy.hpp"

#include <algorithm>
#include <array>
#include <unordered_set>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>

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

TEST(nnAdvice, ModalPauseFreezesGameHeartbeatAndResumesWithoutCatchup) {
    THIRDEYE::nn::advice::GameClock clock;
    EXPECT_EQ(clock.activeMillis(1000), 1000u);
    clock.beginPause(1050);
    EXPECT_TRUE(clock.paused());
    EXPECT_EQ(clock.activeMillis(1500), 1050u);
    EXPECT_EQ(clock.activeMillis(40000), 1050u);
    clock.beginPause(5000); // repeated F1 redraw must not reset the offset
    EXPECT_EQ(clock.activeMillis(6000), 1050u);
    clock.endPause(6050);
    EXPECT_FALSE(clock.paused());
    EXPECT_EQ(clock.activeMillis(6050), 1050u);
    EXPECT_EQ(clock.activeMillis(6250), 1250u);
    clock.endPause(7000); // cannot subtract pause twice
    EXPECT_EQ(clock.activeMillis(7000), 2000u);
    clock.beginPause(7100);
    clock.endPause(7200);
    EXPECT_EQ(clock.activeMillis(7300), 2200u);
}

TEST(nnDungeonHack, UsesNativeLevelMapNotEOB3ClassOffsets) {
    namespace dh = THIRDEYE::nn::dungeon_hack;
    dh::View view;
    view.valid = true;
    view.x = 7; view.y = 8; view.facing = 0; view.level = 4;
    view.tiles.fill(0);           // conservatively blocked by default
    view.tiles[7 * 32 + 7] = 0xff; // north: open
    view.tiles[8 * 32 + 8] = 0xff; // east: open
    view.tiles[9 * 32 + 7] = 0x00; // south: wall
    view.tiles[8 * 32 + 6] = 0x18; // west: non-floor feature
    THIRDEYE::nn::OnlinePolicy::Features features{};
    std::array<bool, 4> blocked{};
    mode::Pose pose;
    ASSERT_TRUE(dh::observe(view, features, blocked, &pose));
    EXPECT_EQ(pose.x, 7);
    EXPECT_EQ(pose.y, 8);
    EXPECT_EQ(pose.facing, 0);
    EXPECT_EQ(pose.level, 4);
    EXPECT_FALSE(blocked[0]); // forward/north
    EXPECT_FALSE(blocked[1]); // right/east
    EXPECT_TRUE(blocked[2]);  // rear/south
    EXPECT_TRUE(blocked[3]);  // left/west
    const std::array<bool, THIRDEYE::nn::OnlinePolicy::kActions> allowed{
        !blocked[0], !blocked[2], !blocked[3], !blocked[1], true, true};
    THIRDEYE::nn::OnlinePolicy dhBrain;
    const std::array<bool, THIRDEYE::nn::OnlinePolicy::kActions> frontier{};
    const auto recommendations = THIRDEYE::nn::advice::generate(
        dhBrain, features, allowed, frontier, pose, 10);
    EXPECT_LE(recommendations.count, 5);
    for (int i = 0; i < recommendations.count; ++i) {
        const int action = recommendations.suggestions[static_cast<std::size_t>(i)].action;
        EXPECT_TRUE(allowed[static_cast<std::size_t>(action)]);
    }
    view.valid = false;
    EXPECT_FALSE(dh::observe(view, features, blocked, &pose));
    view.valid = true;
    view.x = 32;
    EXPECT_FALSE(dh::observe(view, features, blocked, &pose));
    view.x = 7;
    view.facing = 5;
    EXPECT_FALSE(dh::observe(view, features, blocked, &pose));
}

TEST(nnDungeonHack, SameNeuralOutcomeModelLearnsVerifiedDHMovement) {
    THIRDEYE::nn::OnlinePolicy dhBrain;
    const auto start = THIRDEYE::nn::OnlinePolicy::features(
        6, 6, 0, {false, true, true, true});
    const mode::Pose before{6, 6, 0, 2};
    const mode::Pose after{6, 5, 0, 2};
    const auto moved = mode::evaluateOutcome(before, after, 0, true);
    ASSERT_TRUE(moved.success);
    ASSERT_TRUE(dhBrain.reinforce(start, 0, moved.reward, moved.success));
    EXPECT_EQ(dhBrain.samples(), 1u);
    EXPECT_GT(dhBrain.meanReward(), 0.0);
    const auto stalled = mode::evaluateOutcome(after, after, 0, false);
    EXPECT_FALSE(stalled.success);
    ASSERT_TRUE(dhBrain.reinforce(start, 0, stalled.reward, stalled.success));
    EXPECT_EQ(dhBrain.samples(), 2u);
    EXPECT_EQ(dhBrain.successes(), 1u);
    EXPECT_EQ(dhBrain.failures(), 1u);
}

TEST(nnDungeonHack, BothGamesUseSamePersistentBrainWithoutTouchingLegacyFiles) {
    const std::filesystem::path documents("/user/Documents");
    const auto eob = mode::brainFileForDocuments(documents);
    const auto hack = mode::brainFileForDocuments(documents, true);
    EXPECT_EQ(eob, hack);
    EXPECT_EQ(eob.filename().string(), "brain-shared.ckpt");
    EXPECT_EQ(mode::legacyBrainFileForDocuments(documents, false),
              documents / "ThirdeyeNN" / "brain.ckpt");
    EXPECT_EQ(mode::legacyBrainFileForDocuments(documents, true),
              documents / "ThirdeyeNN" / "brain-dungeon-hack.ckpt");
    EXPECT_NE(mode::legacyBrainFileForDocuments(documents, false), eob);
    EXPECT_NE(mode::legacyBrainFileForDocuments(documents, true), eob);
}

TEST(nnAutoAdvice, TwoSecondDeadlinesDoNotBurstAfterStalls) {
    using THIRDEYE::nn::advice::AutoChoicePacer;
    using Clock = AutoChoicePacer::Clock;
    AutoChoicePacer pacer;
    const auto t0 = Clock::time_point{};
    pacer.reset(t0);
    EXPECT_FALSE(pacer.ready(t0));
    EXPECT_FALSE(pacer.ready(t0 + std::chrono::milliseconds(1999)));
    EXPECT_TRUE(pacer.ready(t0 + std::chrono::seconds(2)));
    EXPECT_FALSE(pacer.ready(t0 + std::chrono::seconds(2)));
    EXPECT_FALSE(pacer.ready(t0 + std::chrono::milliseconds(3999)));
    EXPECT_TRUE(pacer.ready(t0 + std::chrono::seconds(4)));
    pacer.reset(t0 + std::chrono::seconds(7)); // toggle F1 on again
    EXPECT_FALSE(pacer.ready(t0 + std::chrono::seconds(8)));
    EXPECT_TRUE(pacer.ready(t0 + std::chrono::seconds(9)));

    pacer.reset(t0);
    EXPECT_TRUE(pacer.ready(t0 + std::chrono::minutes(5)));
    EXPECT_FALSE(pacer.ready(t0 + std::chrono::minutes(5)));
    EXPECT_FALSE(pacer.ready(t0 + std::chrono::minutes(5) +
                             std::chrono::milliseconds(1999)));
    EXPECT_TRUE(pacer.ready(t0 + std::chrono::minutes(5) +
                            std::chrono::seconds(2)));
}

TEST(nnDungeonHack, StandardLaunchRoutesToNativePhaseOneWithoutIntroGff) {
    namespace boot = THIRDEYE::boot;
    const std::filesystem::path hack("C:/Games/Dungeon Hack/HACK.RES");
    const std::filesystem::path eob("C:/Games/Eye of the Beholder III/EYE.RES");
    EXPECT_TRUE(boot::isDungeonHackResource(hack));
    EXPECT_FALSE(boot::needsExternalIntro(hack));
    EXPECT_EQ(boot::initialObject(hack), "phase-one");
    EXPECT_EQ(boot::initialObject("c:/game/hack.res"), "phase-one");
    EXPECT_EQ(boot::initialObject("C:/game/OPEN.RES"), "opening");
    EXPECT_TRUE(boot::isDungeonHackResource("C:/game/OPEN.RES"));
    EXPECT_FALSE(boot::isDungeonHackResource(eob));
    EXPECT_TRUE(boot::needsExternalIntro(eob));
    EXPECT_EQ(boot::initialObject(eob), "start");
    EXPECT_EQ(boot::initialObject("C:/game/SAMPLE.RES"), "start");
}

TEST(nnAutoAdvice, PlannerChoosesObservableActionAndLearnsAfterVerification) {
    namespace advice = THIRDEYE::nn::advice;
    THIRDEYE::nn::OnlinePolicy brain;
    const mode::Pose start{8, 8, 0, 1};
    const auto features = THIRDEYE::nn::OnlinePolicy::features(
        start.x, start.y, start.facing, {false, true, true, true});
    const std::array<bool, 6> allowed{true, false, false, false, false, false};
    const std::array<bool, 6> frontier{true, false, false, false, false, false};
    const auto batch = advice::generate(brain, features, allowed, frontier,
                                        start, 1);
    ASSERT_EQ(batch.count, 1);
    ASSERT_EQ(batch.suggestions[0].action, 0);
    const auto target = batch.suggestions[0].target;
    ASSERT_EQ(target, (mode::Pose{8, 7, 0, 1}));
    const auto verified = advice::objectiveOutcome(
        batch.suggestions[0].action, start, target, target, true);
    ASSERT_TRUE(verified);
    EXPECT_TRUE(verified->success);
    ASSERT_TRUE(brain.reinforce(features, 0, verified->reward,
                                verified->success));
    EXPECT_EQ(brain.samples(), 1u);
    EXPECT_EQ(brain.successes(), 1u);
    EXPECT_FALSE(advice::objectiveOutcome(0, start, target,
                                          {19, 22, 0, 1}, false));
    EXPECT_EQ(brain.samples(), 1u); // unverified outcomes never train
}

TEST(nnAdvice, HoverRepaintsOnlyOnChoiceChanges) {
    namespace advice = THIRDEYE::nn::advice;
    const int y = advice::kRowY;
    EXPECT_FALSE(advice::hoverChanged(30, y + 3, 40, y + 6, 5));
    EXPECT_TRUE(advice::hoverChanged(-1, -1, 30, y + 3, 5));
    EXPECT_TRUE(advice::hoverChanged(30, y + 3,
                                     30, y + advice::kRowHeight + 3, 5));
    EXPECT_TRUE(advice::hoverChanged(30, y, 5, y, 5));
    EXPECT_FALSE(advice::hoverChanged(5, y, 5, y + 9, 5));
    EXPECT_FALSE(advice::hoverChanged(30, y, 30, y, 0));
}

TEST(nnAdvice, GeneratesAtMostFiveLiveReachableActions) {
    namespace advice = THIRDEYE::nn::advice;
    THIRDEYE::nn::OnlinePolicy brain;
    const auto features = THIRDEYE::nn::OnlinePolicy::features(
        13, 9, 1, {false, false, false, false});
    std::array<bool, THIRDEYE::nn::OnlinePolicy::kActions> allowed{};
    allowed.fill(true); // six possible movements
    std::array<bool, THIRDEYE::nn::OnlinePolicy::kActions> frontier{};
    frontier[0] = true;
    const mode::Pose pose{13, 9, 1, 0};
    const auto batch = advice::generate(
        brain, features, allowed, frontier, pose, 1);
    ASSERT_EQ(batch.count, 5);
    EXPECT_EQ(batch.learnedSamples, 0u);
    EXPECT_EQ(batch.generation, 1u);
    EXPECT_EQ(batch.suggestions[0].action, 0); // novel forward path
    std::array<bool, THIRDEYE::nn::OnlinePolicy::kActions> seen{};
    for (int i = 0; i < batch.count; ++i) {
        const int action = batch.suggestions[static_cast<std::size_t>(i)].action;
        ASSERT_GE(action, 0);
        ASSERT_LT(action, 6);
        EXPECT_FALSE(seen[static_cast<std::size_t>(action)]);
        seen[static_cast<std::size_t>(action)] = true;
        const auto &objective = batch.suggestions[static_cast<std::size_t>(i)];
        ASSERT_EQ(advice::targetFor(pose, action), objective.target);
        EXPECT_TRUE(objective.label.starts_with(
            action < 4 ? "Move to " : "Face "));
        EXPECT_NE(objective.label.find("(" + std::to_string(objective.target.x) +
                  "," + std::to_string(objective.target.y) + ")"),
                  std::string::npos);
        EXPECT_NE(THIRDEYE::nn::OnlinePolicy::keyForAction(action), -1);
    }
    EXPECT_EQ(advice::rowAt(35, advice::kRowY, batch.count), 0);
    EXPECT_EQ(advice::rowAt(35, advice::kRowY + 4 * advice::kRowHeight,
                            batch.count), 4);
    EXPECT_EQ(advice::rowAt(35, advice::kRowY + 5 * advice::kRowHeight,
                            batch.count), -1);
    EXPECT_EQ(advice::rowAt(10, advice::kRowY, batch.count), -1);
    EXPECT_EQ(advice::rowAt(35, advice::kRowY, 6), -1);
}

TEST(nnAdvice, RetrainsAfterObservedMoveAndChangesNextRanking) {
    namespace advice = THIRDEYE::nn::advice;
    THIRDEYE::nn::OnlinePolicy brain;
    const auto features = THIRDEYE::nn::OnlinePolicy::features(
        2, 3, 0, {false, false, false, false});
    const std::array<bool, 6> allowed{true, true, true, true, true, true};
    const std::array<bool, 6> frontier{};
    const mode::Pose origin{2, 3, 0, 0};
    const auto before = advice::generate(
        brain, features, allowed, frontier, origin, 1);
    ASSERT_EQ(before.count, 5);
    const mode::Pose turned{2, 3, 1, 0};
    const mode::Outcome actual = mode::evaluateOutcome(origin, turned, 5, false);
    ASSERT_TRUE(actual.success);
    ASSERT_TRUE(brain.reinforce(features, 5, actual.reward, actual.success));
    const auto regenerated = advice::generate(
        brain, features, allowed, frontier, origin, 2);
    EXPECT_EQ(regenerated.generation, 2u);
    EXPECT_EQ(regenerated.learnedSamples, 1u);
    ASSERT_EQ(regenerated.count, 5);
    // More confirmed successful experience increases learned preference.
    for (int i = 0; i < 90; ++i)
        ASSERT_TRUE(brain.reinforce(features, 5, 1.0, true));
    const auto learned = advice::generate(
        brain, features, allowed, frontier, origin, 3);
    ASSERT_EQ(learned.count, 5);
    EXPECT_EQ(learned.learnedSamples, 91u);
    EXPECT_EQ(learned.suggestions[0].action, 5);
    EXPECT_GT(brain.meanReward(), 0.0);
}

TEST(nnAdvice, RejectsBlockedOrInvalidRecommendations) {
    namespace advice = THIRDEYE::nn::advice;
    THIRDEYE::nn::OnlinePolicy brain;
    auto features = THIRDEYE::nn::OnlinePolicy::features(
        20, 8, 2, {false, false, false, false});
    const std::array<bool, 6> allowed{false, false, false, true, false, true};
    const std::array<bool, 6> frontier{};
    const mode::Pose pose{20, 8, 2, 1};
    const auto small = advice::generate(
        brain, features, allowed, frontier, pose, 4);
    ASSERT_EQ(small.count, 2);
    for (int i = 0; i < small.count; ++i) {
        const int action = small.suggestions[static_cast<std::size_t>(i)].action;
        EXPECT_TRUE(action == 3 || action == 5);
    }
    const std::array<bool, 6> none{};
    EXPECT_EQ(advice::generate(
        brain, features, none, frontier, pose, 5).count, 0);
    features[0] = std::numeric_limits<double>::infinity();
    EXPECT_EQ(advice::generate(
        brain, features, allowed, frontier, pose, 6).count, 0);
}

TEST(nnAdvice, TargetsLiveCellsAndExactFacingAcrossGames) {
    namespace advice = THIRDEYE::nn::advice;
    const mode::Pose east{13, 9, 1, 3};
    EXPECT_EQ(advice::targetFor(east, 0), (mode::Pose{14, 9, 1, 3}));
    EXPECT_EQ(advice::targetFor(east, 1), (mode::Pose{12, 9, 1, 3}));
    EXPECT_EQ(advice::targetFor(east, 2), (mode::Pose{13, 8, 1, 3}));
    EXPECT_EQ(advice::targetFor(east, 3), (mode::Pose{13, 10, 1, 3}));
    EXPECT_EQ(advice::targetFor(east, 4), (mode::Pose{13, 9, 0, 3}));
    EXPECT_EQ(advice::targetFor(east, 5), (mode::Pose{13, 9, 2, 3}));
    EXPECT_FALSE(advice::targetFor({0, 4, 3, 0}, 0));
    EXPECT_FALSE(advice::targetFor({3, 0, 0, 0}, 0));
    EXPECT_FALSE(advice::targetFor({3, 4, 5, 0}, 4));
    EXPECT_FALSE(advice::targetFor(east, 6));
    THIRDEYE::nn::OnlinePolicy brain;
    const auto features = THIRDEYE::nn::OnlinePolicy::features(
        0, 0, 0, {false, false, false, false});
    std::array<bool, 6> all{}; all.fill(true);
    const std::array<bool, 6> frontier{};
    const auto batch = advice::generate(
        brain, features, all, frontier, {0, 0, 0, 2}, 1);
    ASSERT_EQ(batch.count, 4); // north and west are off-map
    for (int i = 0; i < batch.count; ++i) {
        const auto &objective = batch.suggestions[i];
        EXPECT_TRUE(objective.label.starts_with("Move to ") ||
                    objective.label.starts_with("Face "));
        EXPECT_EQ(advice::targetFor({0, 0, 0, 2}, objective.action),
                  objective.target);
    }
    EXPECT_EQ(advice::generate(
        brain, features, all, frontier, {-1, 0, 0, 2}, 2).count, 0);
}

TEST(nnAdvice, LearnsOnlyExactObjectiveOutcomes) {
    namespace advice = THIRDEYE::nn::advice;
    const mode::Pose origin{7, 8, 0, 4};
    const auto destination = advice::targetFor(origin, 0);
    ASSERT_TRUE(destination);
    const auto reached = advice::objectiveOutcome(
        0, origin, *destination, *destination, true);
    ASSERT_TRUE(reached);
    EXPECT_TRUE(reached->success);
    const auto blocked = advice::objectiveOutcome(
        0, origin, *destination, origin, false);
    ASSERT_TRUE(blocked);
    EXPECT_FALSE(blocked->success);
    EXPECT_FALSE(advice::objectiveOutcome(
        0, origin, *destination, {7, 8, 1, 4}, false));
    EXPECT_FALSE(advice::objectiveOutcome(
        0, origin, *destination, {8, 8, 0, 4}, false));
    EXPECT_FALSE(advice::objectiveOutcome(
        0, origin, *destination, {7, 7, 0, 5}, false));
    EXPECT_FALSE(advice::objectiveOutcome(
        0, origin, {20, 20, 0, 4}, *destination, true));
    const auto turn = advice::targetFor(origin, 5);
    ASSERT_TRUE(turn);
    const auto turned = advice::objectiveOutcome(5, origin, *turn, *turn, false);
    ASSERT_TRUE(turned);
    EXPECT_TRUE(turned->success);
    THIRDEYE::nn::OnlinePolicy brain;
    const auto features = THIRDEYE::nn::OnlinePolicy::features(
        7, 8, 0, {false, false, false, false});
    EXPECT_TRUE(brain.reinforce(features, 0, reached->reward, reached->success));
    EXPECT_TRUE(brain.reinforce(features, 0, blocked->reward, blocked->success));
    EXPECT_EQ(brain.samples(), 2u);
    EXPECT_EQ(brain.failures(), 1u);
}

TEST(nnFood, NativeClickCalibrationRejectsAmbiguousOrUnrelatedTargets) {
    VM::EventSystem::LeftClick click;
    click.serial = 4;
    click.x = 120; click.y = 55;
    click.x0 = 115; click.y0 = 52;
    click.x1 = 125; click.y1 = 60;
    click.handle = 9;
    click.handlerClass = 1369;
    click.handlerObject = 81;
    click.message = 77;
    click.eventType = VM::SYS_LEFT_CLICK_REGION;
    ASSERT_TRUE(click.valid());
    THIRDEYE::nn::food::ActuatorMemory memory;
    THIRDEYE::nn::food::CalibratedClick candidate{0, 1347, 1, click};
    ASSERT_TRUE(memory.remember(candidate));
    ASSERT_EQ(memory.size(), 1u);
    ASSERT_NE(memory.find(0, 1347, 1), nullptr);
    EXPECT_EQ(memory.find(0, 1347, 1)->click.handlerObject, 81);
    EXPECT_EQ(memory.find(1, 1347, 1), nullptr);
    EXPECT_EQ(memory.find(0, 1451, 1), nullptr);
    EXPECT_EQ(memory.find(0, 1347, 2), nullptr);
    candidate.itemClass = 1357;  // potion never becomes food
    EXPECT_FALSE(memory.remember(candidate));
    candidate.itemClass = 1347;
    candidate.click.handlerObject = -1;
    EXPECT_FALSE(memory.remember(candidate));
    candidate.click = click;
    candidate.click.handle = 1; // page window, not a native UI region
    EXPECT_FALSE(memory.remember(candidate));
    candidate.click = click;
    candidate.click.x = 220; // outside registered rectangle
    EXPECT_FALSE(memory.remember(candidate));
    candidate.click = click;
    candidate.slot = 6;
    EXPECT_FALSE(memory.remember(candidate));
    EXPECT_EQ(memory.size(), 1u);
}

TEST(nnFood, CalibratedNativeActuatorCheckpointIsAtomicAndValidated) {
    namespace food = THIRDEYE::nn::food;
    const auto folder = std::filesystem::temp_directory_path() /
        ("thirdeye_food_actuator_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(folder);
    const auto path = folder / "food_actuator.ckpt";
    VM::EventSystem::LeftClick click;
    click.serial = 15;
    click.x = 222; click.y = 82;
    click.x0 = 213; click.y0 = 76;
    click.x1 = 233; click.y1 = 87;
    click.handle = 24;
    click.handlerClass = 1369;
    click.handlerObject = 919;
    click.message = 120;
    click.eventType = VM::SYS_CLICK_REGION;
    food::ActuatorMemory memory;
    ASSERT_TRUE(memory.remember({3, 1451, 2, click}));
    std::string error;
    ASSERT_TRUE(memory.save(path, error)) << error;
    food::ActuatorMemory loaded;
    ASSERT_TRUE(loaded.load(path, error)) << error;
    ASSERT_EQ(loaded.size(), 1u);
    const auto *saved = loaded.find(3, 1451, 2);
    ASSERT_NE(saved, nullptr);
    EXPECT_EQ(saved->click.x, 222);
    EXPECT_EQ(saved->click.handlerClass, 1369);
    EXPECT_EQ(saved->click.handlerObject, 919);
    EXPECT_EQ(saved->click.message, 120);
    EXPECT_EQ(saved->click.x0, 213);
    EXPECT_EQ(loaded.find(3, 1347, 2), nullptr);

    // Corrupt a single byte. Refusing load must NOT discard existing
    // validated calibrations; the caller must also preserve the bad file.
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(file);
        file.seekp(0);
        file.put('X');
    }
    EXPECT_FALSE(loaded.load(path, error));
    EXPECT_EQ(loaded.size(), 1u);
    EXPECT_NE(loaded.find(3, 1451, 2), nullptr);
    std::error_code ec;
    std::filesystem::remove_all(folder, ec);
}

TEST(nnFood, OnlyNativeFoodClassesCanProduceARecoveryTrial) {
    namespace food = THIRDEYE::nn::food;
    EXPECT_EQ(food::actionForClass(1347), 0);
    EXPECT_EQ(food::actionForClass(1451), 1);
    EXPECT_EQ(food::actionForClass(1456), 2);
    EXPECT_EQ(food::actionForClass(1357), -1); // unknown potion
    EXPECT_EQ(food::actionForClass(1377), -1); // scroll
    EXPECT_EQ(food::classForAction(0), 1347);
    EXPECT_EQ(food::classForAction(2), 1456);
    EXPECT_EQ(food::classForAction(5), -1);
}

TEST(nnFood, ConsumptionRequiresTheGameToChangeBothCursorAndFood) {
    namespace food = THIRDEYE::nn::food;
    food::Observation before;
    before.pc = 10;
    before.item = 999;
    before.itemClass = 1347;
    before.level = 3;
    before.food = 12;
    before.hitpoints = 18;
    before.maxHitpoints = 20;
    before.cursorItem = 999;
    auto after = before;
    after.food = 68;
    after.cursorItem = -1;
    const auto success = food::evaluate(before, after);
    ASSERT_TRUE(success);
    EXPECT_EQ(success->foodGain, 56);
    EXPECT_EQ(success->hpGain, 0);
    EXPECT_GT(success->reward, 0.0);

    after = before;
    after.cursorItem = -1; // item merely dropped or equipped
    EXPECT_FALSE(food::evaluate(before, after));
    after = before;
    after.food = 60; // non-consumptive food gain
    EXPECT_FALSE(food::evaluate(before, after));
    after = before;
    after.food = 60;
    after.cursorItem = -1;
    after.pc = 11; // character switched
    EXPECT_FALSE(food::evaluate(before, after));
    after = before;
    after.food = 60;
    after.cursorItem = -1;
    after.level = 4; // scene/restore transition
    EXPECT_FALSE(food::evaluate(before, after));
    after = before;
    after.food = 60;
    after.cursorItem = -1;
    after.maxHitpoints = 25; // unsafe statics
    EXPECT_FALSE(food::evaluate(before, after));
    before.itemClass = 1357; // potion is NOT assumed to heal/eat
    after = before;
    after.food = 60;
    after.cursorItem = -1;
    EXPECT_FALSE(food::evaluate(before, after));
}

TEST(nnFood, RealFoodOutcomeUpdatesAndRestoresClassPreference) {
    namespace food = THIRDEYE::nn::food;
    THIRDEYE::nn::survival::Party party;
    party.complete = true;
    party.ids[0] = 10;
    party.hp[0] = 17;
    party.maximum[0] = 20;
    party.food[0] = 12;
    party.rations = 2;
    const auto base = THIRDEYE::nn::OnlinePolicy::features(
        10, 10, 0, {false, false, false, false});
    const auto features = food::features(base, party, 0);
    EXPECT_NEAR(features[7], 0.12, 1e-9);
    EXPECT_NEAR(features[8], 0.85, 1e-9);
    THIRDEYE::nn::OnlinePolicy brain;
    for (int i = 0; i < 240; ++i)
        ASSERT_TRUE(brain.reinforce(features, 2, 1.5, true));
    const std::array<bool, THIRDEYE::nn::OnlinePolicy::kActions> available{
        true, true, true, false, false, false
    };
    EXPECT_EQ(brain.select(features, available, 1.0, 0.5), 2);
    const auto dir = std::filesystem::temp_directory_path() /
        ("thirdeye_food_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    const auto file = dir / "food.ckpt";
    std::string error;
    ASSERT_TRUE(brain.save(file, error)) << error;
    THIRDEYE::nn::OnlinePolicy restored;
    ASSERT_TRUE(restored.load(file, error)) << error;
    EXPECT_EQ(restored.samples(), 240u);
    EXPECT_EQ(restored.select(features, available, 1.0, 0.5), 2);
    EXPECT_DOUBLE_EQ(restored.meanReward(), brain.meanReward());
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
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
    EXPECT_EQ(file, docs / "ThirdeyeNN" / "brain-shared.ckpt");
    EXPECT_EQ(file.parent_path().filename().string(), "ThirdeyeNN");
    EXPECT_EQ(file.filename().string(), "brain-shared.ckpt");
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
