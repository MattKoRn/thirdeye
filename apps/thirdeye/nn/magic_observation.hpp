#ifndef THIRDEYE_NN_MAGIC_OBSERVATION_HPP
#define THIRDEYE_NN_MAGIC_OBSERVATION_HPP

#include <cstdint>

namespace THIRDEYE::nn::magic {

// Resource-aware spell evidence: an unchanged HP reading alone is NOT proof
// that a scroll/wand fired, nor that an enemy resisted it. Do not attribute
// an unsuccessful magic experiment unless an actual item/charge cost was seen.
enum class Result : uint8_t {
    Invalid,        // the target's HP reading cannot be compared
    Damaged,        // observed NPC HP decrease (cost may be unobservable)
    NoDamageSpent,  // spell resource was spent, but no NPC HP loss was seen
    Unverified      // no damage AND no observable spell resource consumption
};

inline bool spentCharge(int before, int after) {
    return before > 0 && after >= 0 && after < before;
}

inline Result evaluate(int beforeHP, int afterHP, bool resourceSpent) {
    if (beforeHP <= 0 || afterHP < 0 || afterHP > beforeHP)
        return Result::Invalid;
    if (afterHP < beforeHP) return Result::Damaged;
    return resourceSpent ? Result::NoDamageSpent : Result::Unverified;
}

} // namespace THIRDEYE::nn::magic
#endif
