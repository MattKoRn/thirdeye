#include "online_mode.hpp"
#include "combat_mode.hpp"
#include "online_policy.hpp"
#include "spell_probe.hpp"

#include "../vm/events.hpp"
#include "../vm/objects.hpp"

#include <SDL3/SDL_filesystem.h>

#include <algorithm>
#include <array>
#include <cstdint>
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
constexpr uint16_t kPcClass = 1369;
constexpr uint16_t kMagicClass = 2377;
constexpr uint64_t kAutoSaveInterval = 32;
constexpr std::array<const char *, OnlinePolicy::kActions> kActionNames{
    "forward", "back", "strafe-left", "strafe-right", "turn-left", "turn-right"
};

struct State {
    OnlinePolicy brain;
    std::filesystem::path file;
    bool initialized = false;
    bool storageReady = false;
    bool trainingEnabled = true; // autonomous outcome training only
    bool autonomous = false;
    bool dirty = false;
    bool checkpointPresent = false;
    bool storageWarningLogged = false;
    ActionPacer pacer;
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
    spell::Probe spellProbe;
    int spellCaster = -1;
    int spellSlot = -1;
    int spellType = -1;
    int spellPreparedBefore = -1;
    uint32_t spellEffectsBefore = 0;
    struct SpellBaseline {
        int pc = -1, mage = -1, cleric = -1;
        uint32_t effects = 0;
    } spellBaselines[6];
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
        g.checkpointPresent = true;
        std::cout << "[nn] checkpoint saved samples=" << g.savedSamples
                  << " path=" << g.file.string() << '\n';
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

// Native SOP statics documented by runtime/eye.cpp and internal.cpp.
int readWord(VM::ObjectSystem &objects, int obj, uint16_t cls, uint32_t offset) {
    if (obj < 0) return -1;
    try {
        if (auto *p = objects.classStaticPtr(obj, cls, offset, 2))
            return static_cast<int16_t>(static_cast<uint16_t>(p[0] | (p[1] << 8)));
    } catch (const std::exception &) {}
    return -1;
}

int spellScreen(VM::ObjectSystem &objects) {
    const int kernel = objects.firstObjectOfClass(kKernelClass);
    return readWord(objects, kernel, kKernelClass, 265);
}

struct SpellStock {
    int pc = -1, mage = -1, cleric = -1;
    uint32_t effects = 0;
    bool valid = false;
};

SpellStock spellStock(VM::ObjectSystem &objects, int partySlot) {
    SpellStock result;
    if (partySlot < 0 || partySlot >= 6) return result;
    const int kernel = objects.firstObjectOfClass(kKernelClass);
    if (kernel < 0) return result;
    const int pc = readWord(objects, kernel, kKernelClass, 229 + partySlot * 2);
    if (pc < 0 || objects.classOf(pc) == 0xffff ||
        !objects.isSubclassOf(objects.classOf(pc), kPcClass))
        return result;
    try {
        auto *counter = objects.classStaticPtr(pc, kPcClass, 209, 200);
        auto *effect = objects.classStaticPtr(pc, kPcClass, 199, 4);
        if (!counter || !effect) return result;
        std::array<uint8_t, 200> counts{};
        std::copy_n(counter, counts.size(), counts.begin());
        result.pc = pc;
        result.mage = spell::available(counts, 0);
        result.cleric = spell::available(counts, 1);
        result.effects = static_cast<uint32_t>(effect[0]) |
            (static_cast<uint32_t>(effect[1]) << 8) |
            (static_cast<uint32_t>(effect[2]) << 16) |
            (static_cast<uint32_t>(effect[3]) << 24);
        result.valid = result.mage >= 0 && result.cleric >= 0;
    } catch (const std::exception &) {}
    return result;
}

void observeSpellStock(VM::ObjectSystem &objects) {
    for (int slot = 0; slot < 6; ++slot) {
        const auto current = spellStock(objects, slot);
        auto &prev = g.spellBaselines[slot];
        if (!current.valid) { prev = {}; continue; }
        if (prev.pc == current.pc &&
            (prev.mage != current.mage || prev.cleric != current.cleric ||
             prev.effects != current.effects)) {
            std::cout << "[nn] spell-state observed slot=" << slot
                      << " mage_available=" << current.mage
                      << " mage_delta=" << current.mage - prev.mage
                      << " cleric_available=" << current.cleric
                      << " cleric_delta=" << current.cleric - prev.cleric
                      << " effects_before=" << prev.effects
                      << " effects_after=" << current.effects
                      << " attributed_cast=no trained=no\n";
        }
        prev = {current.pc, current.mage, current.cleric, current.effects};
    }
}

void closeSpellProbe(VM::ObjectSystem &objects) {
    const int magic = objects.firstObjectOfClass(kMagicClass);
    if (magic < 0) {
        g.spellProbe.fail();
        std::cerr << "[nn] spell probe cannot close: missing magic singleton\n";
        return;
    }
    try {
        objects.send(magic, 172, {}); // confirmed magic.deactivate
        g.spellProbe.closed(spellScreen(objects));
    } catch (const std::exception &e) {
        g.spellProbe.fail();
        std::cerr << "[nn] spell probe close failed: " << e.what() << '\n';
        return;
    }
    const auto after = spellStock(objects, g.spellSlot);
    std::cout << "[nn] spell probe=" << g.spellProbe.status()
              << " slot=" << g.spellSlot
              << " type=" << (g.spellType ? "cleric" : "mage")
              << " open_screen=" << g.spellProbe.openedScreen()
              << " closed_screen=" << g.spellProbe.closedScreen()
              << " available_before=" << g.spellPreparedBefore
              << " available_after=" << (after.valid
                  ? (g.spellType ? after.cleric : after.mage) : -1)
              << " effects_before=" << g.spellEffectsBefore
              << " effects_after=" << (after.valid ? after.effects : 0)
              << " cast=no trained=no\n";
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
            g.checkpointPresent = true;
            std::cout << "[nn] checkpoint restored samples=" << g.savedSamples
                      << " rate=" << g.brain.learningRate()
                      << " path=" << g.file.string() << '\n';
        } else {
            // A malformed brain must not be overwritten by an empty model.
            reportStorageError("existing brain rejected (" + error + ")");
            g.storageReady = false;
        }
    }
    if (g.storageReady && !g.checkpointPresent)
        std::cout << "[nn] new brain checkpoint=new path="
                  << g.file.string() << '\n';
    // Independent v2 sidecar; legacy movement checkpoints remain unchanged.
    combat::initialize(folder / "combat.ckpt");
}

void flushLearning() {
    if (!g.initialized) return;
    autoSave(true);
    combat::flushLearning();
}

void shutdown() {
    if (!g.initialized) return;
    flushLearning();
    combat::shutdown();
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
    combat::cancel();
    g.pacer.reset(ActionPacer::Clock::now());
    g.pending.reset();
    g.failurePose.reset();
    g.failedActions.fill(false);
    for (auto &value : g.spellBaselines) value = {};
    if (g.spellProbe.shouldOpen()) g.spellProbe.fail();
    // A probe that already opened must still run native close on next pump.
}

bool toggle() {
    setActive(!active());
    return active();
}

bool consumesManualKey(int32_t code) {
    return active() && OnlinePolicy::actionForKey(code) >= 0;
}

const char *spellProbeStatus() { return g.spellProbe.status(); }

std::string spellStockStatus(VM::ObjectSystem &objects, int slot) {
    const auto stock = spellStock(objects, slot);
    if (!stock.valid) return "err spell counters unavailable for party slot\n";
    return "nn spell slot=" + std::to_string(slot) +
           " pc=" + std::to_string(stock.pc) +
           " mage_available=" + std::to_string(stock.mage) +
           " cleric_available=" + std::to_string(stock.cleric) +
           " effects=" + std::to_string(stock.effects) +
           " screen=" + std::to_string(spellScreen(objects)) +
           " probe=" + g.spellProbe.status() + "\nok\n";
}

bool queueSpellProbe(VM::ObjectSystem &objects, int slot, int type,
                     std::string &error) {
    error.clear();
    if (!g.autonomous) { error = "enable nn auto first"; return false; }
    if (g.spellProbe.busy()) { error = "spell probe busy"; return false; }
    if (slot < 0 || slot >= 6 || (type != 0 && type != 1)) {
        error = "invalid party slot/type"; return false;
    }
    if (spellScreen(objects) != 0) {
        error = "spell probe needs normal gameplay screen"; return false;
    }
    if (objects.firstObjectOfClass(kMagicClass) < 0) {
        error = "EOB3 magic singleton not present"; return false;
    }
    const auto stock = spellStock(objects, slot);
    if (!stock.valid || readWord(objects, stock.pc, kPcClass, 171) <= 0) {
        error = "caster unavailable or not alive"; return false;
    }
    const int prepared = type == 0 ? stock.mage : stock.cleric;
    if (!g.spellProbe.begin(0, prepared)) {
        error = "no prepared spell counters for requested type"; return false;
    }
    g.spellSlot = slot;
    g.spellCaster = stock.pc;
    g.spellType = type;
    g.spellPreparedBefore = prepared;
    g.spellEffectsBefore = stock.effects;
    return true;
}

void setTraining(bool enabled) { g.trainingEnabled = enabled; }
bool training() { return g.trainingEnabled; }
uint64_t samples() { initialize(); return g.brain.samples(); }
double meanLoss() { initialize(); return g.brain.meanLoss(); }

double meanReward() { initialize(); return g.brain.meanReward(); }
double successRate() { initialize(); return g.brain.successRate(); }
double learningRate() { initialize(); return g.brain.learningRate(); }
uint64_t combatSamples() { initialize(); return combat::samples(); }
double combatMeanReward() { initialize(); return combat::meanReward(); }
double combatSuccessRate() { initialize(); return combat::successRate(); }
uint64_t tacticalSamples() { initialize(); return combat::tacticalSamples(); }
std::size_t knownWeaknesses() { initialize(); return combat::knownWeaknesses(); }
const char *tacticalCheckpointStatus() {
    initialize();
    return combat::tacticalCheckpointStatus();
}
const char *combatCheckpointStatus() {
    initialize();
    return combat::checkpointStatus();
}
const char *checkpointStatus() {
    initialize();
    if (!g.storageReady) return "unavailable";
    if (g.dirty) return "pending";
    return g.checkpointPresent ? "saved" : "new";
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

void pump(VM::EventSystem &events, VM::ObjectSystem &objects,
          bool uiActive, bool eob3) {
    combat::release(events); // always release pending click on next host pump
    if (g.spellProbe.shouldClose()) {
        closeSpellProbe(objects); // close even if autonomy toggled off or UI is up
        combat::cancel();
        g.pending.reset();
        g.pacer.reset(ActionPacer::Clock::now());
        return;
    }
    if (!eob3 && g.spellProbe.shouldOpen()) {
        g.spellProbe.fail(); // never send EOB3 magic messages to Dungeon Hack
    }
    if (eob3 && g.autonomous && !uiActive) observeSpellStock(objects);
    // Never interpret changes caused by menus or manual control as rewards.
    if (!g.autonomous || uiActive) {
        // A queued probe must never fire later after an unrelated menu closes.
        if (g.spellProbe.shouldOpen()) g.spellProbe.fail();
        combat::cancel();
        g.pending.reset();
        // Returning from a menu or restarting autonomous play must not
        // release an action accumulated while gameplay was unavailable.
        g.pacer.reset(ActionPacer::Clock::now());
        return;
    }
    // Enforce one action per elapsed second, independent of host frame rate.
    // ready() advances from *now*, never trying to catch up missed actions.
    if (!g.pacer.ready(ActionPacer::Clock::now())) return;

    OnlinePolicy::Features features{};
    std::array<bool, 4> blocked{};
    Pose current;
    if (!context(objects, features, blocked, &current) || current.level < 0) {
        combat::cancel();
        g.pending.reset();
        return;
    }
    const bool firstVisit = g.visited.insert(visitKey(current)).second;
    if (eob3) combat::observe(objects, current, g.trainingEnabled);
    else combat::cancel();
    // Do not issue another action while waiting for spell or attack damage.
    if (eob3 && combat::waiting()) return;

    // Only reward or penalize a move AFTER the SOP engine has processed it.
    if (g.pending) {
        const auto previous = *g.pending;
        g.pending.reset();
        const int levelDelta = std::abs(previous.before.level - current.level);
        const int travel = std::abs(previous.before.x - current.x) +
                           std::abs(previous.before.y - current.y);
        // Learn adjacent-floor exits but skip same-level teleports and
        // multi-floor scripted resets unrelated to normal movement.
        if ((levelDelta == 0 && travel <= 1) ||
            (levelDelta == 1 && previous.action < 4)) {
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
            if (g.trainingEnabled) {
                const double rate = g.brain.learningRate();
                if (g.brain.reinforce(previous.features, previous.action,
                                      result.reward, result.success)) {
                    g.dirty = true;
                    autoSave(false);
                    std::cout << "[nn] learn action="
                              << kActionNames[static_cast<std::size_t>(previous.action)]
                              << " success=" << (result.success ? "yes" : "no")
                              << " reward=" << result.reward
                              << " rate=" << rate
                              << " samples=" << g.brain.samples()
                              << " mean_reward=" << g.brain.meanReward()
                              << " checkpoint=" << checkpointStatus() << '\n';
                }
            }
        }
    }

    if (g.spellProbe.shouldOpen()) {
        // The existing runtime spellmenu probe verifies M:29 and M:172.
        // M:54 has no confirmed memorized spell ID; do not synthesize a cast.
        const int magic = objects.firstObjectOfClass(kMagicClass);
        const auto stock = spellStock(objects, g.spellSlot);
        if (!eob3 || magic < 0 || !stock.valid ||
            stock.pc != g.spellCaster ||
            (g.spellType == 0 ? stock.mage : stock.cleric) <= 0) {
            g.spellProbe.fail();
        } else {
            try {
                objects.send(magic, 29, {g.spellCaster, g.spellType});
                g.spellProbe.opened(spellScreen(objects));
                std::cout << "[nn] spell probe opened slot=" << g.spellSlot
                          << " type=" << (g.spellType ? "cleric" : "mage")
                          << " screen=" << g.spellProbe.openedScreen()
                          << " prepared=" << g.spellPreparedBefore
                          << " native_message=29 cast=no\n";
            } catch (const std::exception &e) {
                // An exception can occur after the SOP has changed UI state.
                // Best-effort native deactivation avoids leaving the HUD stuck.
                try { objects.send(magic, 172, {}); }
                catch (const std::exception &) {}
                g.spellProbe.fail();
                std::cerr << "[nn] spell probe open failed: " << e.what() << '\n';
            }
        }
        g.pacer.reset(ActionPacer::Clock::now());
        return;
    }

    // First live combat interaction: attack a verified front-cell monster.
    // No invented keyboard attack codes; actual EOB3 weapon hit regions.
    if (eob3 && combat::attack(events, objects, current, blocked))
        return;

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
    const auto frontier = frontierActions(current, allowed, g.visited);
    if (std::any_of(frontier.begin(), frontier.end(), [](bool f) { return f; }))
        allowed = frontier;
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
    combat::cancel();
    if (g.spellProbe.shouldOpen()) g.spellProbe.fail();
    // Do not train the imported weights on feedback from a pre-import action.
    g.pending.reset();
    g.failurePose.reset();
    g.failedActions.fill(false);
    g.pacer.reset(ActionPacer::Clock::now());
    // Treat an imported model as unsaved to the default Documents location.
    g.dirty = true;
    g.savedSamples = g.brain.samples();
    autoSave(true);
    std::cout << "[nn] checkpoint imported samples=" << g.brain.samples()
              << " rate=" << g.brain.learningRate()
              << " checkpoint=" << checkpointStatus() << '\n';
    return true;
}

} // namespace THIRDEYE::nn::mode
