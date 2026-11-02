#ifndef THIRDEYE_NN_COMBAT_MODE_HPP
#define THIRDEYE_NN_COMBAT_MODE_HPP

#include "online_mode.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace VM { class ObjectSystem; class EventSystem; }

namespace THIRDEYE::nn::combat {

// EOB3-only autonomous combat. A separate v2 checkpoint stores the weapon
// selection policy; existing movement brain.ckpt files remain unchanged.
void initialize(const std::filesystem::path &path);
void shutdown();
void cancel();
void release(VM::EventSystem &events);
void observe(VM::ObjectSystem &objects, const mode::Pose &pose, bool training);
bool attack(VM::EventSystem &events, VM::ObjectSystem &objects,
            const mode::Pose &pose, const std::array<bool, 4> &blocked);
uint64_t samples();
double meanReward();
double successRate();
const char *checkpointStatus();

} // namespace THIRDEYE::nn::combat
#endif
