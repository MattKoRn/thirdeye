#include "online_mode.hpp"
#include "online_policy.hpp"

#include "../vm/events.hpp"
#include "../vm/objects.hpp"

#include <SDL3/SDL_filesystem.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
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
    bool trainingEnabled = true;
    bool autonomous = false;
    bool dirty = false;
    bool storageWarningLogged = false;
    int ticks = 0;
    uint64_t savedSamples = 0;
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
             std::array<bool, 4> &blocked) {
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
    return true;
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

double teach(int32_t code, VM::ObjectSystem &objects) {
    initialize();
    const int action = OnlinePolicy::actionForKey(code);
    OnlinePolicy::Features features{};
    std::array<bool, 4> blocked{};
    if (action < 0 || !context(objects, features, blocked)) return -1.0;
    const double loss = g.brain.learn(features, action);
    if (loss >= 0.0) {
        g.dirty = true;
        autoSave(false);
    }
    return loss;
}

void observeKey(int32_t code, VM::ObjectSystem &objects) {
    if (g.trainingEnabled && !consumesManualKey(code))
        (void)teach(code, objects);
}

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
    if (!g.autonomous || ++g.ticks < kActionIntervalPumps) return;
    g.ticks = 0;
    // Never drive a menu, spellbook, camp panel, or any other UI screen.
    if (uiActive) return;

    OnlinePolicy::Features features{};
    std::array<bool, 4> blocked{};
    if (!context(objects, features, blocked)) return;

    int key = -1;
    double confidence = 0.0;
    if (!predict(objects, key, confidence)) {
        // A fresh brain still takes over when F1 is pressed. Conservative
        // exploration avoids stepping into a known wall until it is trained.
        key = fallbackKey(blocked);
    }
    events.postEvent(0, VM::SYS_KEYDOWN, key);
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
