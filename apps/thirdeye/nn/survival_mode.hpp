#ifndef THIRDEYE_NN_SURVIVAL_MODE_HPP
#define THIRDEYE_NN_SURVIVAL_MODE_HPP

#include "online_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace THIRDEYE::nn::survival {

// EOB3 party snapshots. Negative object ids are empty party slots.
// A food/potion count means "carried", NOT "verified healing action".
struct Party {
    std::array<int, 6> ids{-1, -1, -1, -1, -1, -1};
    std::array<int, 6> hp{};
    std::array<int, 6> maximum{};
    std::array<int, 6> food{};
    int rations = 0;
    int potions = 0;
    bool complete = false;
};

inline bool valid(const Party &party) {
    if (!party.complete || party.rations < 0 || party.potions < 0) return false;
    bool found = false;
    for (std::size_t i = 0; i < party.ids.size(); ++i) {
        if (party.ids[i] < 0) continue;
        found = true;
        if (party.maximum[i] <= 0 || party.hp[i] < 0 ||
            party.hp[i] > party.maximum[i]) return false;
    }
    return found;
}

inline double weakestHealth(const Party &party) {
    if (!valid(party)) return -1.0;
    double lowest = 1.0;
    for (std::size_t i = 0; i < party.ids.size(); ++i)
        if (party.ids[i] >= 0)
            lowest = std::min(lowest,
                static_cast<double>(party.hp[i]) / party.maximum[i]);
    return lowest;
}

// Native movement actions: 0 forward, 1 back, 2 strafe left, 3 right,
// 4/5 turn. Never back into a known wall and never walk toward the threat.
inline std::array<bool, OnlinePolicy::kActions> exits(
    const std::array<bool, 4> &blocked) {
    return {false, !blocked[2], !blocked[3], !blocked[1], false, false};
}

inline bool shouldWithdraw(const Party &party, bool enemyAhead,
                           const std::array<bool, 4> &blocked) {
    if (!enemyAhead || !valid(party)) return false;
    const auto allowed = exits(blocked);
    if (!allowed[1] && !allowed[2] && !allowed[3]) return false;
    // When the party carries no known consumables, conserve the remaining
    // HP earlier. Potion subtype/healing power is deliberately NOT assumed.
    const double limit = party.rations == 0 && party.potions == 0 ? 0.65 : 0.50;
    const double worst = weakestHealth(party);
    return worst >= 0.0 && worst < limit &&
           std::any_of(party.hp.begin(), party.hp.end(),
                       [](int hp) { return hp > 0; });
}

inline OnlinePolicy::Features features(const OnlinePolicy::Features &dungeon,
                                       const Party &party) {
    auto f = dungeon;
    f[7] = std::clamp(weakestHealth(party), 0.0, 1.0);
    int totalFood = 0, members = 0;
    for (std::size_t i = 0; i < party.ids.size(); ++i) {
        if (party.ids[i] < 0) continue;
        ++members;
        totalFood += std::clamp(party.food[i], 0, 100);
    }
    f[8] = members ? totalFood / (100.0 * members) : 0.0;
    f[9] = party.potions > 0 ? 1.0 : 0.0;
    f[10] = party.rations > 0 ? 1.0 : 0.0;
    return f;
}

struct Outcome {
    bool success = false;
    double reward = 0.0;
    int hpDelta = 0;
    int foodDelta = 0;
};

// A retreat is learned only after a later SOP host tick confirms the party
// is the SAME party at a valid location. No damage/food credit is invented.
// A moved step which avoids the adjacent enemy without HP loss is rewarded;
// movement into danger, stalled moves, or HP loss are penalized.
inline std::optional<Outcome> evaluate(const Party &before,
                                       const Party &after, bool moved,
                                       bool enemyStillAhead) {
    if (!valid(before) || !valid(after) || before.ids != after.ids)
        return std::nullopt;
    Outcome out;
    for (std::size_t i = 0; i < before.ids.size(); ++i) {
        if (before.ids[i] < 0) continue;
        if (before.maximum[i] != after.maximum[i]) return std::nullopt;
        out.hpDelta += after.hp[i] - before.hp[i];
        out.foodDelta += after.food[i] - before.food[i];
    }
    out.success = moved && !enemyStillAhead && out.hpDelta >= 0;
    double reward = moved ? 0.25 : -1.25;
    if (!enemyStillAhead && moved) reward += 0.75;
    if (enemyStillAhead && moved) reward -= 0.50;
    if (out.hpDelta < 0) reward -= std::min(2.0, -out.hpDelta / 8.0);
    else if (out.hpDelta > 0) reward += std::min(0.75, out.hpDelta / 12.0);
    // Food decreases are observed survival costs, never healing credit.
    if (out.foodDelta < 0) reward -= std::min(0.40, -out.foodDelta / 100.0);
    out.reward = std::clamp(reward, -2.0, 2.0);
    return out;
}
} // namespace THIRDEYE::nn::survival
#endif
