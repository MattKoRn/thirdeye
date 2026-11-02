#ifndef THIRDEYE_NN_ONLINE_MODE_HPP
#define THIRDEYE_NN_ONLINE_MODE_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>

namespace VM { class ObjectSystem; class EventSystem; }

namespace THIRDEYE::nn::mode {

// Throttle gameplay actions using elapsed monotonic time, never the number of
// host pumps. Each action schedules its own next deadline, so a stalled game
// cannot emit a burst of catch-up inputs. Accept an injected time for tests.
class ActionPacer {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr std::chrono::seconds kInterval{1};

    void reset(Clock::time_point now) { next_ = now + kInterval; }
    bool ready(Clock::time_point now) {
        if (now < next_) return false;
        next_ = now + kInterval;
        return true;
    }

private:
    Clock::time_point next_ = Clock::time_point::max();
};

// A movement result is only learned after the next host tick has allowed the
// SOP runtime to process the key. This pure evaluator is unit tested.
struct Pose {
    int x = -1, y = -1, facing = -1, level = -1;
    bool operator==(const Pose &) const = default;
};
struct Outcome {
    bool success;
    double reward;
};
inline Outcome evaluateOutcome(const Pose &before, const Pose &after,
                               int action, bool firstVisit) {
    const bool moved = before.x != after.x || before.y != after.y ||
                       before.level != after.level;
    const bool turned = before.facing != after.facing;
    const bool success = action < 4 ? moved : turned;
    if (!success) return {false, -1.25};
    if (action >= 4) return {true, 0.08};
    return {true, firstVisit ? 1.8 : 0.20};
}

// Pure helpers: testable without SDL or a running game.
inline std::filesystem::path brainFileForDocuments(
    const std::filesystem::path &documents) {
    return documents / "ThirdeyeNN" / "brain.ckpt";
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

void setTraining(bool enabled);
bool training();
uint64_t samples();
double meanLoss();
double meanReward();
double successRate();

// Returns the currently preferred action for diagnostics only.
bool predict(VM::ObjectSystem &objects, int &key, double &confidence);

// Called once each host pump, whether a control socket exists or not.
void pump(VM::EventSystem &events, VM::ObjectSystem &objects, bool uiActive);

// Manual import/export for control clients. Automatic persistence remains in
// Documents/ThirdeyeNN/brain.ckpt, independent of these file paths.
bool saveTo(const std::filesystem::path &path, std::string &error);
bool loadFrom(const std::filesystem::path &path, std::string &error);

} // namespace THIRDEYE::nn::mode
#endif // THIRDEYE_NN_ONLINE_MODE_HPP
