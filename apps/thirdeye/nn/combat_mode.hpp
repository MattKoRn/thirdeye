#ifndef THIRDEYE_NN_COMBAT_MODE_HPP
#define THIRDEYE_NN_COMBAT_MODE_HPP

#include "online_mode.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace VM { class ObjectSystem; class EventSystem; }

namespace THIRDEYE::nn::combat {

// EOB3-only autonomous combat. Preserve old movement/combat v2 checkpoints;
// weaknesses.ckpt independently stores per-enemy weapon and magic experiments.
void initialize(const std::filesystem::path &path);
void shutdown();
void cancel();
bool waiting();
void release(VM::EventSystem &events);
void observe(VM::ObjectSystem &objects, const mode::Pose &pose, bool training);
bool attack(VM::EventSystem &events, VM::ObjectSystem &objects,
            const mode::Pose &pose, const std::array<bool, 4> &blocked);
uint64_t samples();
uint64_t tacticalSamples();
std::size_t knownWeaknesses();
const char *tacticalCheckpointStatus();
double meanReward();
double successRate();
const char *checkpointStatus();

} // namespace THIRDEYE::nn::combat
#endif
