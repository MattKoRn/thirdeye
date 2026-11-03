#ifndef THIRDEYE_NN_ADVICE_MENU_HPP
#define THIRDEYE_NN_ADVICE_MENU_HPP

#include "online_mode.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace THIRDEYE::nn::advice {

// All advice is an observable, one-action coordinate or facing objective.
// Never advertise unverified item, door or combat interaction actuators.
constexpr int kMax = 5;
constexpr int kRowX = 22, kRowY = 51, kRowHeight = 22;
constexpr int kRowWidth = 276;
struct Suggestion {
    int action = -1;
    double score = 0.0;
    mode::Pose target{};
    std::string label;
};
struct Batch {
    std::array<Suggestion, kMax> suggestions{};
    int count = 0;
    uint64_t generation = 0;
    uint64_t learnedSamples = 0;
};

// Derive the exact objective pose from native N/E/S/W coordinates, including
// Dungeon Hack. Reject off-map targets even if a caller marks a move allowed.
inline std::optional<mode::Pose> targetFor(const mode::Pose &start, int action) {
    if (start.x < 0 || start.x >= 32 || start.y < 0 || start.y >= 32 ||
        start.facing < 0 || start.facing >= 4 || start.level < 0 ||
        action < 0 || action >= static_cast<int>(OnlinePolicy::kActions))
        return std::nullopt;
    mode::Pose target = start;
    if (action < 4) {
        constexpr std::array<int, 4> dx{0, 1, 0, -1};
        constexpr std::array<int, 4> dy{-1, 0, 1, 0};
        constexpr std::array<int, 4> offsets{0, 2, 3, 1};
        const int direction = (start.facing + offsets[action]) % 4;
        target.x += dx[direction];
        target.y += dy[direction];
        if (target.x < 0 || target.x >= 32 ||
            target.y < 0 || target.y >= 32) return std::nullopt;
    } else {
        target.facing = (start.facing + (action == 4 ? 3 : 1)) % 4;
    }
    return target;
}

inline std::string objectiveLabel(const mode::Pose &target, int action,
                                  bool frontier) {
    const std::string where = "(" + std::to_string(target.x) + "," +
        std::to_string(target.y) + ") L" + std::to_string(target.level);
    if (action < 4)
        return "Move to " + where + (frontier ? " [new]" : "");
    constexpr std::array<const char *, 4> directions{
        "north", "east", "south", "west"};
    return std::string("Face ") + directions[target.facing] + " at " + where;
}

// A blocked input is a negative outcome; an exact target is a positive
// outcome. Unexpected displacements, rotations, and level transitions are
// ambiguous and must never be attributed to this objective.
inline std::optional<mode::Outcome> objectiveOutcome(
    int action, const mode::Pose &before, const mode::Pose &target,
    const mode::Pose &after, bool firstVisit) {
    const auto expected = targetFor(before, action);
    if (!expected || *expected != target ||
        (after != before && after != target)) return std::nullopt;
    return mode::evaluateOutcome(before, after, action, firstVisit);
}

// Rank only live executable objectives from learned weights, passability and
// frontier exploration. No generic tips or invented interactions.
inline Batch generate(const OnlinePolicy &policy,
                      const OnlinePolicy::Features &features,
                      const std::array<bool, OnlinePolicy::kActions> &allowed,
                      const std::array<bool, OnlinePolicy::kActions> &frontier,
                      const mode::Pose &pose, uint64_t generation) {
    Batch batch;
    batch.generation = generation;
    batch.learnedSamples = policy.samples();
    if (!std::all_of(features.begin(), features.end(), [](double value) {
            return std::isfinite(value) && value >= 0.0 && value <= 1.0;
        })) return batch;
    std::array<Suggestion, OnlinePolicy::kActions> sorted{};
    int n = 0;
    for (int action = 0; action < static_cast<int>(OnlinePolicy::kActions); ++action) {
        if (!allowed[static_cast<std::size_t>(action)]) continue;
        const auto target = targetFor(pose, action);
        if (!target) continue;
        const auto &w = policy.weights();
        double score = 0.0;
        for (int i = 0; i < static_cast<int>(OnlinePolicy::kFeatures); ++i)
            score += w[static_cast<std::size_t>(action) *
                       OnlinePolicy::kFeatures + static_cast<std::size_t>(i)] *
                     features[static_cast<std::size_t>(i)];
        if (frontier[static_cast<std::size_t>(action)]) score += 0.9;
        // Small deterministic rotation; already learned preferences dominate.
        const int rotation = static_cast<int>((generation +
                                               static_cast<uint64_t>(action)) %
                                              OnlinePolicy::kActions);
        score += 0.008 * rotation;
        sorted[static_cast<std::size_t>(n++)] =
            Suggestion{action, score, *target,
                objectiveLabel(*target, action,
                               frontier[static_cast<std::size_t>(action)])};
    }
    std::stable_sort(sorted.begin(), sorted.begin() + n,
        [](const Suggestion &a, const Suggestion &b) {
            return a.score > b.score;
        });
    batch.count = std::min(n, kMax);
    for (int i = 0; i < batch.count; ++i)
        batch.suggestions[static_cast<std::size_t>(i)] =
            sorted[static_cast<std::size_t>(i)];
    return batch;
}

// Clock for SOP heartbeat pacing: elapsed time during an open advice menu
// is excluded, so monsters and spell timers cannot fast-forward on resume.
// Pure and deterministic, independently tested without SDL.
class GameClock {
public:
    void beginPause(uint64_t nowMs) {
        if (!paused_) { pauseBegin_ = nowMs; paused_ = true; }
    }
    void endPause(uint64_t nowMs) {
        if (!paused_) return;
        if (nowMs >= pauseBegin_) pausedMs_ += nowMs - pauseBegin_;
        paused_ = false;
    }
    uint64_t activeMillis(uint64_t nowMs) const {
        const uint64_t wall = paused_ ? pauseBegin_ : nowMs;
        return wall >= pausedMs_ ? wall - pausedMs_ : 0;
    }
    bool paused() const { return paused_; }
private:
    bool paused_ = false;
    uint64_t pausedMs_ = 0;
    uint64_t pauseBegin_ = 0;
};

// Pixel hit-test matches the overlay's own row rectangles. Its guard
// prevents clicks outside the menu from triggering a game action.
inline int rowAt(int x, int y, int count) {
    if (count < 1 || count > kMax || x < kRowX || x >= kRowX + kRowWidth ||
        y < kRowY || y >= kRowY + count * kRowHeight) return -1;
    return (y - kRowY) / kRowHeight;
}

} // namespace THIRDEYE::nn::advice
#endif
