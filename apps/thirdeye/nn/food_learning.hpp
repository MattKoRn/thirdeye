#ifndef THIRDEYE_NN_FOOD_LEARNING_HPP
#define THIRDEYE_NN_FOOD_LEARNING_HPP

#include "online_policy.hpp"
#include "survival_mode.hpp"

#include <algorithm>
#include <optional>

namespace THIRDEYE::nn::food {

// Exact EOB3 consumable food classes identified in docs/equipment_slots.md.
// No potion class is treated as healing or edible without separate proof.
inline int actionForClass(unsigned cls) {
    switch (cls) {
    case 1347: return 0; // rations
    case 1451: return 1; // apple
    case 1456: return 2; // iron rations
    default: return -1;
    }
}

struct Observation {
    int pc = -1;
    int item = -1;
    int itemClass = -1;
    int level = -1;
    int food = -1;
    int hitpoints = -1;
    int maxHitpoints = -1;
    int cursorItem = -1;
};

struct Result {
    int foodGain = 0;
    int hpGain = 0;
    double reward = 0.0;
};

// This is evidence of native FOOD consumption, not an inferred spell/potion
// effect. An in-hand food item must leave the cursor, the same PC must still
// exist, and the game's actual food counter must increase. Drops, movement,
// GUI changes and no-ops cannot generate training rewards.
inline std::optional<Result> evaluate(const Observation &before,
                                      const Observation &after) {
    if (before.pc < 0 || before.item < 0 ||
        actionForClass(static_cast<unsigned>(before.itemClass)) < 0 ||
        before.cursorItem != before.item ||
        before.food < 0 || before.food >= 100 ||
        before.maxHitpoints <= 0 ||
        before.hitpoints < 0 || before.hitpoints > before.maxHitpoints ||
        after.pc != before.pc || after.level != before.level ||
        after.maxHitpoints != before.maxHitpoints ||
        after.food <= before.food || after.food > 100 ||
        after.hitpoints < 0 || after.hitpoints > after.maxHitpoints ||
        after.cursorItem != -1)
        return std::nullopt;
    const int gain = after.food - before.food;
    return Result{gain, after.hitpoints - before.hitpoints,
                  std::clamp(0.25 + gain / 30.0, 0.25, 2.0)};
}

// The independent recovery policy learns which actual *item class* restored
// more food in the observed party context. The three other policy actions
// remain unavailable; this never relabels movement or combat actions.
inline OnlinePolicy::Features features(const OnlinePolicy::Features &pose,
                                       const survival::Party &party,
                                       int targetSlot) {
    auto result = survival::features(pose, party);
    if (targetSlot < 0 || targetSlot >= 6 || !survival::valid(party) ||
        party.ids[targetSlot] < 0) return result;
    result[7] = std::clamp(party.food[targetSlot] / 100.0, 0.0, 1.0);
    result[8] = party.maximum[targetSlot] > 0
        ? std::clamp(static_cast<double>(party.hp[targetSlot]) /
                     party.maximum[targetSlot], 0.0, 1.0) : 0.0;
    return result;
}

} // namespace THIRDEYE::nn::food
#endif
