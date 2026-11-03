#ifndef THIRDEYE_NN_ADVICE_MENU_HPP
#define THIRDEYE_NN_ADVICE_MENU_HPP

#include "online_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace THIRDEYE::nn::advice {

// All suggestions correspond to an actual, supported EOB3 movement key.
// There is never more than five at once, even if six moves are available.
constexpr int kMax = 5;
constexpr int kRowX = 22, kRowY = 51, kRowHeight = 22;
constexpr int kRowWidth = 276;
constexpr std::array<const char *, OnlinePolicy::kActions> kNames{
    "Explore forward", "Step backward", "Strafe left",
    "Strafe right", "Turn left", "Turn right"
};
struct Suggestion {
    int action = -1;
    double score = 0.0;
    std::string label;
};
struct Batch {
    std::array<Suggestion, kMax> suggestions{};
    int count = 0;
    uint64_t generation = 0;
    uint64_t learnedSamples = 0;
};

// Generate deterministic, individually actionable recommendations from the
// currently loaded policy's REAL weights, a live 11D observation, passability,
// and frontier exploration. Ties rotate on subsequent batches to encourage
// experimenting with other legal moves. Never invent an action beyond 0..5.
inline Batch generate(const OnlinePolicy &policy,
                      const OnlinePolicy::Features &features,
                      const std::array<bool, OnlinePolicy::kActions> &allowed,
                      const std::array<bool, OnlinePolicy::kActions> &frontier,
                      uint64_t generation) {
    Batch batch;
    batch.generation = generation;
    batch.learnedSamples = policy.samples();
    if (!OnlinePolicy::validFeatures(features)) return batch;
    std::array<Suggestion, OnlinePolicy::kActions> sorted{};
    int n = 0;
    for (int action = 0; action < static_cast<int>(OnlinePolicy::kActions); ++action) {
        if (!allowed[static_cast<std::size_t>(action)]) continue;
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
            Suggestion{action, score,
                frontier[static_cast<std::size_t>(action)]
                ? std::string(kNames[static_cast<std::size_t>(action)]) + " [new]"
                : std::string(kNames[static_cast<std::size_t>(action)])};
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

// Pixel hit-test matches the overlay's own row rectangles. Its guard
// prevents clicks outside the menu from triggering a game action.
inline int rowAt(int x, int y, int count) {
    if (count < 1 || count > kMax || x < kRowX || x >= kRowX + kRowWidth ||
        y < kRowY || y >= kRowY + count * kRowHeight) return -1;
    return (y - kRowY) / kRowHeight;
}

} // namespace THIRDEYE::nn::advice
#endif
