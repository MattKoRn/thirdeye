#include "online_mode.hpp"
#include "online_policy.hpp"

#include "../vm/events.hpp"
#include "../vm/objects.hpp"

#include <SDL3/SDL_filesystem.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <unordered_set>
#include <string>
#include <system_error>

namespace THIRDEYE::nn::mode {
namespace {

constexpr uint16_t kKernelClass = 1382;
constexpr uint16_t kDungeonClass = 1381;
constexpr uint64_t kAutoSaveInterval = 32;
constexpr int kActionIntervalPumps = 40;

struct State {
    OnlinePolicy brain;
    std::filesystem::path file;
    bool initialized = false;
    bool storageReady = false;
    bool trainingEnabled = true; // autonomous outcome training only
    bool autonomous = false;
    bool dirty = false;
    bool storageWarningLogged = false;
    int ticks = 0;
    uint64_t savedSamples = 0;
    struct Pending {
        OnlinePolicy::Features features{};
        Pose before{};
        int action = -1;
    };
    std::optional<Pending> pending;
    std::optional<Pose> failurePose;
    std::array<bool, OnlinePolicy::kActions> failedActions{};
    std::unordered_set<uint32_t> visited;
    std::mt19937 rng{std::random_device{}()};
};
State g;

void reportStorageError(const std::string &error) {
    // Successful automatic saves and loads are silent. Report a genuine
    // persistence error only once per session instead of per frame.
    if (!g.storageWarningLogged) {
        std::cerr << "[nn] Cannot persist brain: " << error << '\n';
        g.storageWarningLogged = true;
    }
}

std::filesystem::path documentsFolder() {
    // SDL resolves redirected Documents folders on Windows/macOS and follows
    // the desktop environment's XDG user-dir configuration on Linux.
    if (const char *p = SDL_GetUserFolder(SDL_FOLDER_DOCUMENTS); p && *p)
        return std::filesystem::u8path(p);
    if (const char *home = SDL_GetUserFolder(SDL_FOLDER_HOME); home && *home)
        return std::filesystem::u8path(home) / "Documents";
#ifdef _WIN32
    if (const char *home = std::getenv("USERPROFILE"); home && *home)
        return std::filesystem::u8path(home) / "Documents";
#else
    if (const char *home = std::getenv("HOME"); home && *home)
        return std::filesystem::u8path(home) / "Documents";
#endif
    return {};
}

void autoSave(bool force) {
    if (!g.storageReady || !g.dirty) return;
    if (!force && g.brain.samples() - g.savedSamples < kAutoSaveInterval)
        return;
    std::string error;
    if (g.brain.save(g.file, error)) {
        g.savedSamples = g.brain.samples();
        g.dirty = false;
    } else {
        reportStorageError(error);
    }
}

int readField(VM::ObjectSystem &objects, int obj, uint16_t cls,
              uint32_t off, int fallback = -1) {
    try {
        if (uint8_t *p = objects.classStaticPtr(obj, cls, off, 1))
            return *p;
    } catch (const std::exception &) {}
    return fallback;
}

bool context(VM::ObjectSystem &objects, OnlinePolicy::Features &features,
             std::array<bool, 4> &blocked, Pose *pose = nullptr) {
    const int kernel = objects.firstObjectOfClass(kKernelClass);
    const int dungeon = objects.firstObjectOfClass(kDungeonClass);
    if (kernel < 0 || dungeon < 0) return false;
    const int x = readField(objects, kernel, kKernelClass, 243);
    const int y = readField(objects, kernel, kKernelClass, 244);
    const int dir = readField(objects, kernel, kKernelClass, 245);
    if (x < 0 || x >= 32 || y < 0 || y >= 32 || dir < 0 || dir >= 4)
        return false;

    constexpr std::array<int, 4> dx{0, 1, 0, -1};
    constexpr std::array<int, 4> dy{-1, 0, 1, 0};
    for (int relative = 0; relative < 4; ++relative) {
        const int absolute = (dir + relative) % 4;
        const int nx = x + dx[absolute];
        const int ny = y + dy[absolute];
        blocked[relative] = nx < 0 || nx >= 32 || ny < 0 || ny >= 32 ||
            readField(objects, dungeon, kDungeonClass,
                      static_cast<uint32_t>(ny * 32 + nx), 0) != 0xff;
    }
    features = OnlinePolicy::features(x, y, dir, blocked);
    if (pose) *pose = {x, y, dir, readField(objects, kernel, kKernelClass, 246)};
    return true;
}

uint32_t locationKey(const Pose &p) {
    return (static_cast<uint32_t>(p.level) << 10) |
           (static_cast<uint32_t>(p.y) << 5) | static_cast<uint32_t>(p.x);
}

bool canMove(int action, const std::array<bool, 4> &blocked) {
    switch (action) {
    case 0: return !blocked[0]; // forward
    case 1: return !blocked[2]; // back
    case 2: return !blocked[3]; // strafe left
    case 3: return !blocked[1]; // strafe right
    case 4: case 5: return true; // rotations
    default: return false;
    }
}

} // namespace

void initialize() {
    if (g.initialized) return;
    g.initialized = true;
    const auto documents = documentsFolder();
    if (documents.empty()) {
        reportStorageError("Documents folder could not be located");
        return;
    }
    const auto file = brainFileForDocuments(documents);
    const auto folder = file.parent_path();
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    if (ec || !std::filesystem::is_directory(folder)) {
        reportStorageError("cannot create Documents/ThirdeyeNN");
        return;
    }
    g.file = file;
    g.storageReady = true;
    if (std::filesystem::exists(g.file, ec) && !ec) {
        std::string error;
        if (g.brain.load(g.file, error)) {
            g.savedSamples = g.brain.samples();
            g.dirty = false;
        } else {
            // A malformed brain must not be overwritten by an empty model.
            reportStorageError("existing brain rejected (" + error + ")");
            g.storageReady = false;
        }
    }
}

void shutdown() {
    if (!g.initialized) return;
    autoSave(true);
    g.autonomous = false;
    g.pending.reset();
}

std::filesystem::path checkpointPath() {
    initialize();
    return g.file;
}

bool active() { return g.autonomous; }

void setActive(bool enabled) {
    initialize();
    g.autonomous = enabled;
    g.ticks = 0;
    g.pending.reset();
    g.failurePose.reset();
    g.failedActions.fill(false);
}

bool toggle() {
    setActive(!active());
    return active();
}

bool consumesManualKey(int32_t code) {
    return active() && OnlinePolicy::actionForKey(code) >= 0;
}

void setTraining(bool enabled) { g.trainingEnabled = enabled; }
bool training() { return g.trainingEnabled; }
uint64_t samples() { initialize(); return g.brain.samples(); }
double meanLoss() { initialize(); return g.brain.meanLoss(); }

double meanReward() { initialize(); return g.brain.meanReward(); }
double successRate() { initialize(); return g.brain.successRate(); }

bool predict(VM::ObjectSystem &objects, int &key, double &confidence) {
    initialize();
    if (g.brain.samples() < 8) return false;
    OnlinePolicy::Features features{};
    std::array<bool, 4> blocked{};
    if (!context(objects, features, blocked)) return false;
    const auto prediction = g.brain.predict(features);
    key = OnlinePolicy::keyForAction(prediction.action);
    confidence = prediction.confidence;
    return key >= 0 && confidence >= 0.55 && canMove(prediction.action, blocked);
}

void pump(VM::EventSystem &events, VM::ObjectSystem &objects, bool uiActive) {
    // Never interpret changes caused by menus or manual control as rewards.
    if (!g.autonomous || uiActive) {
        g.pending.reset();
        return;
    }
    if (++g.ticks < kActionIntervalPumps) return;
    g.ticks = 0;

    OnlinePolicy::Features features{};
    std::array<bool, 4> blocked{};
    Pose current;
    if (!context(objects, features, blocked, &current) || current.level < 0) {
        g.pending.reset();
        return;
    }
    const bool firstVisit = g.visited.insert(locationKey(current)).second;

    // Only reward or penalize a move AFTER the SOP engine has processed it.
    if (g.pending) {
        const auto previous = *g.pending;
        g.pending.reset();
        const bool sameLevel = previous.before.level == current.level;
        const int travel = std::abs(previous.before.x - current.x) +
                           std::abs(previous.before.y - current.y);
        // Disregard teleports and scripted state resets (not our action).
        if (sameLevel && travel <= 1) {
            const Outcome result = evaluateOutcome(previous.before, current,
                                                   previous.action, firstVisit);
            if (result.success) {
                g.failurePose.reset();
                g.failedActions.fill(false);
            } else {
                if (!g.failurePose || *g.failurePose != previous.before)
                    g.failedActions.fill(false);
                g.failurePose = previous.before;
                g.failedActions[static_cast<std::size_t>(previous.action)] = true;
            }
            if (g.trainingEnabled &&
                g.brain.reinforce(previous.features, previous.action,
                                  result.reward, result.success)) {
                g.dirty = true;
                autoSave(false);
                if (g.brain.samples() % 50 == 0)
                    std::cout << "[nn] self-learning: " << g.brain.samples()
                              << " attempts, success=" << g.brain.successRate() * 100.0
                              << "%, reward=" << g.brain.meanReward() << '\n';
            }
        }
    }

    std::array<bool, OnlinePolicy::kActions> allowed = {
        !blocked[0], !blocked[2], !blocked[3], !blocked[1], true, true
    };
    // Exclude every failed action at the current pose; changing pose clears
    // the suppression so a previously blocked path may become available.
    if (g.failurePose && *g.failurePose == current) {
        for (std::size_t a = 0; a < allowed.size(); ++a)
            if (g.failedActions[a]) allowed[a] = false;
    } else {
        g.failurePose.reset();
        g.failedActions.fill(false);
    }
    std::uniform_real_distribution<double> random(0.0, 1.0);
    const int action = g.brain.select(features, allowed, random(g.rng), random(g.rng));
    if (action < 0) {
        g.failurePose.reset();
        g.failedActions.fill(false);
        return;
    }
    events.postEvent(0, VM::SYS_KEYDOWN, OnlinePolicy::keyForAction(action));
    g.pending = State::Pending{features, current, action};
}

bool saveTo(const std::filesystem::path &path, std::string &error) {
    initialize();
    return g.brain.save(path, error);
}

bool loadFrom(const std::filesystem::path &path, std::string &error) {
    initialize();
    if (!g.brain.load(path, error)) return false;
    // Treat an imported model as unsaved to the default Documents location.
    g.dirty = true;
    g.savedSamples = g.brain.samples();
    autoSave(true);
    return true;
}

} // namespace THIRDEYE::nn::mode
