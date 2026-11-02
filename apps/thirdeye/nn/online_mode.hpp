#ifndef THIRDEYE_NN_ONLINE_MODE_HPP
#define THIRDEYE_NN_ONLINE_MODE_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace VM { class ObjectSystem; class EventSystem; }

namespace THIRDEYE::nn::mode {

// Pure helpers: testable without SDL or a running game.
inline std::filesystem::path brainFileForDocuments(
    const std::filesystem::path &documents) {
    return documents / "ThirdeyeNN" / "brain.ckpt";
}

inline int fallbackKey(const std::array<bool, 4> &blocked) {
    return !blocked[0] ? 0x4800 : (!blocked[1] ? 0x4900 : 0x4700);
}

// Per-process brain. It uses the OS Documents folder, not the game-data
// directory. All functions are called on the host (SDL) thread.
void initialize();
void shutdown();
std::filesystem::path checkpointPath();

bool active();
bool toggle();
void setActive(bool enabled);
bool consumesManualKey(int32_t code);

void observeKey(int32_t code, VM::ObjectSystem &objects);
double teach(int32_t code, VM::ObjectSystem &objects);
void setTraining(bool enabled);
bool training();
uint64_t samples();
double meanLoss();

// Returns a confident learned action (does not apply untrained fallback).
bool predict(VM::ObjectSystem &objects, int &key, double &confidence);

// Called once each host pump, whether a control socket exists or not.
void pump(VM::EventSystem &events, VM::ObjectSystem &objects, bool uiActive);

// Manual import/export for control clients. Automatic persistence remains in
// Documents/ThirdeyeNN/brain.ckpt, independent of these file paths.
bool saveTo(const std::filesystem::path &path, std::string &error);
bool loadFrom(const std::filesystem::path &path, std::string &error);

} // namespace THIRDEYE::nn::mode
#endif // THIRDEYE_NN_ONLINE_MODE_HPP
