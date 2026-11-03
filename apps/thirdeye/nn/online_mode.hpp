#ifndef THIRDEYE_NN_ONLINE_MODE_HPP
#define THIRDEYE_NN_ONLINE_MODE_HPP

#include "online_policy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_set>

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
    const bool changedLevel = before.level != after.level;
    const bool success = action < 4 ? moved : (turned && !changedLevel);
    if (!success) return {false, -1.25};
    if (action >= 4) return {true, 0.08};
    if (changedLevel) return {true, 2.0};
    return {true, firstVisit ? 1.8 : 0.20};
}

// Only a monster in the facing adjacent cell may be attacked.
inline bool targetDirectlyAhead(const Pose &party, const Pose &enemy) {
    if (party.level < 0 || party.facing < 0 || party.facing >= 4 ||
        party.level != enemy.level || enemy.x < 0 || enemy.x >= 32 ||
        enemy.y < 0 || enemy.y >= 32) return false;
    constexpr std::array<int, 4> dx{0, 1, 0, -1};
    constexpr std::array<int, 4> dy{-1, 0, 1, 0};
    return enemy.x == party.x + dx[party.facing] &&
           enemy.y == party.y + dy[party.facing];
}

// Measure reward from observed NPC HP loss, not from a click being sent.
inline std::optional<Outcome> evaluateCombatOutcome(int beforeHP, int afterHP) {
    if (beforeHP <= 0 || afterHP < 0 || afterHP > beforeHP)
        return std::nullopt;
    const int damage = beforeHP - afterHP;
    if (damage == 0) return Outcome{false, -0.4};
    return Outcome{true, std::min(2.0, 0.25 + damage / 8.0)};
}

// Prioritize accessible, unexplored neighbouring cells before backtracking.
// Visit keys are level-specific, preventing collisions between dungeon floors.
inline uint32_t visitKey(const Pose &pose) {
    return (static_cast<uint32_t>(pose.level) << 10) |
           (static_cast<uint32_t>(pose.y) << 5) |
           static_cast<uint32_t>(pose.x);
}
inline std::array<bool, OnlinePolicy::kActions> frontierActions(
    const Pose &pose,
    const std::array<bool, OnlinePolicy::kActions> &allowed,
    const std::unordered_set<uint32_t> &visited) {
    std::array<bool, OnlinePolicy::kActions> frontier{};
    if (pose.level < 0 || pose.facing < 0 || pose.facing >= 4) return frontier;
    constexpr std::array<int, 4> dx{0, 1, 0, -1};
    constexpr std::array<int, 4> dy{-1, 0, 1, 0};
    // Forward, back, strafe left, strafe right.
    constexpr std::array<int, 4> offsets{0, 2, 3, 1};
    for (std::size_t action = 0; action < offsets.size(); ++action) {
        if (!allowed[action]) continue;
        const int direction = (pose.facing + offsets[action]) % 4;
        const int x = pose.x + dx[direction];
        const int y = pose.y + dy[direction];
        if (x < 0 || x >= 32 || y < 0 || y >= 32) continue;
        if (!visited.contains(visitKey({x, y, pose.facing, pose.level})))
            frontier[action] = true;
    }
    return frontier;
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
// Preserve all confirmed movement, combat, and weakness observations before
// an nn-1..nn-3 native game restore without resetting live model weights.
void flushLearning();
std::filesystem::path checkpointPath();

bool active();
bool toggle();
void setActive(bool enabled);
bool consumesManualKey(int32_t code);
// F1 hands native in-game mouse movement and clicks to the NN. Manual
// movement is accepted only during a bounded calibration trial; F1 still
// interrupts the autopilot and its mouse button is released on host pump.
bool ownsMouse();
bool acceptsCalibrationMouse();
uint64_t mouseMoves();
uint64_t mouseClicks();
uint64_t mouseAborts();
uint64_t mouseScreens();
const char *mouseState();

// Read-only native EOB3 spell counters (mage/cleric) and active magic effects.
// Counters are NOT cast_spell message IDs; no arbitrary spells are fired.
std::string spellStockStatus(VM::ObjectSystem &objects, int partySlot);
const char *spellProbeStatus();
// Explicit one-shot native M:29 menu open, followed by M:172 close on the
// subsequent host pump. Requires neural autonomy, an alive caster and stock.
bool queueSpellProbe(VM::ObjectSystem &objects, int partySlot,
                     int spellType, std::string &error);

void setTraining(bool enabled);
bool training();
uint64_t samples();
double meanLoss();
double meanReward();
double successRate();
double learningRate();
// Separate resource-aware survival learner, persisted as survival.ckpt.
// Retains existing movement/combat/weakness and native game save formats.
uint64_t survivalSamples();
double survivalMeanReward();
double survivalSuccessRate();
const char *survivalCheckpointStatus();

// EOB3 food outcome verification: arm while holding a known ration/apple/
// iron ration in the kernel cursor; the native food-plate click is performed
// through the existing game UI. Reward only if food increases and cursor
// consumption is observed. No potion or guessed click coordinates are used.
bool queueFoodProbe(VM::ObjectSystem &objects, VM::EventSystem &events,
                    int partySlot, std::string &error);
// Click ONLY a previously verified native region, after matching the live
// SOP handler, exact PC object, screen, and equipped cursor food class.
bool useCalibratedFood(VM::ObjectSystem &objects, VM::EventSystem &events,
                       int partySlot, std::string &error);
std::size_t foodCalibrations();
const char *foodProbeStatus();
uint64_t foodSamples();
double foodMeanReward();
const char *foodCheckpointStatus();
// Apply saved item-class preferences to real carried food types; diagnostic
// recommendation only until a food-use actuator has been verified.
std::string foodRecommendation(VM::ObjectSystem &objects, int partySlot);
const char *checkpointStatus();
uint64_t combatSamples();
double combatMeanReward();
double combatSuccessRate();
const char *combatCheckpointStatus();
uint64_t tacticalSamples();
std::size_t knownWeaknesses();
const char *tacticalCheckpointStatus();

// Returns the currently preferred action for diagnostics only.
bool predict(VM::ObjectSystem &objects, int &key, double &confidence);

// Called once each host pump, whether a control socket exists or not.
// Known EOB3 weapon coordinates must never be sent to Dungeon Hack.
void pump(VM::EventSystem &events, VM::ObjectSystem &objects,
          bool uiActive, bool eob3);

// Manual import/export for control clients. Automatic persistence remains in
// Documents/ThirdeyeNN/brain.ckpt, independent of these file paths.
bool saveTo(const std::filesystem::path &path, std::string &error);
bool loadFrom(const std::filesystem::path &path, std::string &error);

} // namespace THIRDEYE::nn::mode
#endif // THIRDEYE_NN_ONLINE_MODE_HPP
