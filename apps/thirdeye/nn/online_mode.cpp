#include "online_mode.hpp"
#include "shared_brain.hpp"
#include "combat_mode.hpp"
#include "food_learning.hpp"
#include "food_actuator.hpp"
#include "advice_menu.hpp"
#include "dungeon_hack_context.hpp"
#include "../runtime/internal.hpp"
#include "online_policy.hpp"
#include "spell_probe.hpp"
#include "survival_mode.hpp"

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
#include <set>
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
constexpr uint64_t kSurvivalSaveInterval = 8;
constexpr std::array<const char *, OnlinePolicy::kActions> kActionNames{
    "forward", "back", "strafe-left", "strafe-right", "turn-left", "turn-right"
};

struct State {
    OnlinePolicy brain;
    OnlinePolicy survivalBrain;
    OnlinePolicy foodBrain;
    food::ActuatorMemory foodActuator;
    std::set<food::ActuatorMemory::Key> failedFoodRegions;
    std::filesystem::path foodActuatorFile;
    bool foodActuatorReady = false;
    bool foodActuatorDirty = false;
    bool foodActuatorPresent = false;
    bool foodActuatorWarned = false;
    std::filesystem::path foodFile;
    bool foodStorageReady = false;
    bool foodDirty = false;
    bool foodCheckpointPresent = false;
    bool foodWarned = false;
    uint64_t savedFoodSamples = 0;
    std::filesystem::path file;
    std::filesystem::path survivalFile;
    bool survivalStorageReady = false;
    bool survivalDirty = false;
    bool survivalCheckpointPresent = false;
    bool survivalWarningLogged = false;
    uint64_t savedSurvivalSamples = 0;
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
    struct RetreatPending {
        OnlinePolicy::Features features{};
        survival::Party party;
        Pose before{};
        int action = -1;
    };
    std::optional<RetreatPending> retreatPending;
    struct FoodTrial {
        food::Observation before;
        OnlinePolicy::Features features{};
        int slot = -1;
        ActionPacer::Clock::time_point started{};
        uint64_t firstClickSerial = 0;
        VM::EventSystem::LeftClick click;
        int clickedScreen = -1;
        bool automatic = false;
    };
    std::optional<FoodTrial> foodTrial;
    bool foodClickDown = false;
    advice::Batch adviceBatch;
    bool adviceOpen = false;
    bool adviceReopen = false;
    bool autoAdvice = false;
    advice::AutoChoicePacer autoChoicePacer;
    uint64_t adviceGeneration = 0;
    Pose advicePose{};
    std::string adviceResult = "Choose a neural suggestion to start learning.";
    struct AdvicePending {
        OnlinePolicy::Features features{};
        Pose before{};
        int action = -1;
        Pose target{};
        ActionPacer::Clock::time_point sent{};
        uint64_t leftClickSerial = 0;
    };
    std::optional<AdvicePending> advicePending;
    const char *foodStatus = "idle";
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
    // Dungeon Hack's HACK.RES owns different SOP classes/statics from EOB3.
    // Its real native renderer publishes the party/map through verified
    // draw_walls and load_level_map runtime arguments.
    if (runtime::gDungeonHack) {
        // A cached renderer observation is not enough: a native DH dungeon
        // instance must still be alive. This prevents stale poses from
        // exposing movement advice on a title or game-over screen.
        uint16_t dungeonClass = 0;
        if (!objects.findClassByName("dungeon", dungeonClass) ||
            objects.firstObjectOfClass(dungeonClass) < 0)
            return false;
        const auto native = runtime::dh::neuralSnapshot();
        dungeon_hack::View view;
        view.x = native.x;
        view.y = native.y;
        view.facing = native.facing;
        view.level = native.level;
        view.tiles = native.tiles;
        view.valid = native.valid;
        return dungeon_hack::observe(view, features, blocked, pose);
    }
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


void autoSaveSurvival(bool force) {
    if (!g.survivalStorageReady || !g.survivalDirty) return;
    if (!force && g.survivalBrain.samples() - g.savedSurvivalSamples <
            kSurvivalSaveInterval) return;
    std::string error;
    if (g.survivalBrain.save(g.survivalFile, error)) {
        g.savedSurvivalSamples = g.survivalBrain.samples();
        g.survivalDirty = false;
        g.survivalCheckpointPresent = true;
        std::cout << "[nn] survival checkpoint saved samples="
                  << g.savedSurvivalSamples << " path="
                  << g.survivalFile.string() << '\n';
    } else if (!g.survivalWarningLogged) {
        std::cerr << "[nn] survival checkpoint error: " << error << '\n';
        g.survivalWarningLogged = true;
    }
}

survival::Party partyResources(VM::ObjectSystem &objects) {
    survival::Party party;
    const int kernel = objects.firstObjectOfClass(kKernelClass);
    if (kernel < 0) return party;
    std::unordered_set<int> seenItems;
    for (int slot = 0; slot < 6; ++slot) {
        const int pc = readWord(objects, kernel, kKernelClass, 229 + slot * 2);
        if (pc < 0 || objects.classOf(pc) == 0xffff ||
            !objects.isSubclassOf(objects.classOf(pc), kPcClass))
            continue;
        const int hp = readWord(objects, pc, kPcClass, 171);
        const int maximum = readWord(objects, pc, kPcClass, 173);
        const int food = readWord(objects, pc, kPcClass, 177);
        if (maximum <= 0 || hp > maximum || food < 0)
            return {}; // invalid statics; no safety-critical decisions
        party.ids[slot] = pc;
        // EOB3 can retain a negative HP value for a fallen PC. Treat that
        // member as at zero rather than suppressing retreat for survivors.
        party.hp[slot] = std::max(hp, 0);
        party.maximum[slot] = maximum;
        party.food[slot] = food;
        // Inventory pointers are not consumption commands. Distinct items
        // are counted only once, even if a 2-handed slot aliases an object.
        for (int inv = 0; inv < 26; ++inv) {
            const int item = readWord(objects, pc, kPcClass, 81 + inv * 2);
            if (item < 0 || !seenItems.insert(item).second) continue;
            const uint16_t cls = objects.classOf(item);
            if (cls == 0xffff) continue;
            // Documented EOB3 class families. Potion subtype/power is unknown.
            if (cls >= 1357 && cls <= 1364) ++party.potions;
            if (cls == 1347 || cls == 1451 || cls == 1456) ++party.rations;
        }
    }
    party.complete = true;
    return party;
}


void autoSaveFood(bool force) {
    if (!g.foodStorageReady || !g.foodDirty) return;
    if (!force && g.foodBrain.samples() - g.savedFoodSamples < 8)
        return;
    std::string error;
    if (g.foodBrain.save(g.foodFile, error)) {
        g.savedFoodSamples = g.foodBrain.samples();
        g.foodDirty = false;
        g.foodCheckpointPresent = true;
        std::cout << "[nn] verified food checkpoint saved samples="
                  << g.savedFoodSamples << " path=" << g.foodFile.string() << '\n';
    } else if (!g.foodWarned) {
        std::cerr << "[nn] food checkpoint error: " << error << '\n';
        g.foodWarned = true;
    }
}


void autoSaveFoodActuator(bool force) {
    if (!g.foodActuatorReady || !g.foodActuatorDirty || !force)
        return;
    std::string error;
    if (g.foodActuator.save(g.foodActuatorFile, error)) {
        g.foodActuatorDirty = false;
        g.foodActuatorPresent = true;
        std::cout << "[nn] calibrated food regions saved count="
                  << g.foodActuator.size()
                  << " path=" << g.foodActuatorFile.string() << '\n';
    } else if (!g.foodActuatorWarned) {
        std::cerr << "[nn] food actuator checkpoint error: " << error << '\n';
        g.foodActuatorWarned = true;
    }
}

food::Observation foodSnapshot(VM::ObjectSystem &objects, int slot) {
    food::Observation o;
    const int kernel = objects.firstObjectOfClass(kKernelClass);
    if (kernel < 0 || slot < 0 || slot >= 6) return o;
    o.pc = readWord(objects, kernel, kKernelClass, 229 + slot * 2);
    if (o.pc < 0 || objects.classOf(o.pc) == 0xffff ||
        !objects.isSubclassOf(objects.classOf(o.pc), kPcClass))
        return {};
    o.item = readWord(objects, kernel, kKernelClass, 241);
    o.cursorItem = o.item;
    o.itemClass = o.item >= 0 && objects.classOf(o.item) != 0xffff
        ? objects.classOf(o.item) : -1;
    o.level = readField(objects, kernel, kKernelClass, 246);
    o.food = readWord(objects, o.pc, kPcClass, 177);
    o.hitpoints = std::max(0, readWord(objects, o.pc, kPcClass, 171));
    o.maxHitpoints = readWord(objects, o.pc, kPcClass, 173);
    return o;
}

void observeFoodTrial(VM::ObjectSystem &objects, VM::EventSystem &events) {
    if (!g.foodTrial) return;
    auto &trial = *g.foodTrial;
    const auto observed = events.lastLeftClick();
    if (observed.serial != trial.firstClickSerial && trial.click.serial == 0) {
        // Only a single real left-click may be credited. Capture the target
        // while its native window still exists; unrelated clicks disqualify.
        if (observed.serial == trial.firstClickSerial + 1 &&
            observed.valid() && runtime::uiScreenActive(objects)) {
            trial.click = observed;
            trial.clickedScreen = spellScreen(objects);
            std::cout << "[nn] food clicked native_region="
                      << observed.handle << " handler_class="
                      << observed.handlerClass << " handler_obj="
                      << observed.handlerObject << " message="
                      << observed.message << " rect=(" << observed.x0 << ','
                      << observed.y0 << ',' << observed.x1 << ','
                      << observed.y1 << ") screen=" << trial.clickedScreen
                      << '\n';
        } else {
            if (trial.automatic) g.failedFoodRegions.insert(
                food::ActuatorMemory::key(trial.slot,
                       trial.before.itemClass, spellScreen(objects)));
            g.foodTrial.reset();
            g.foodStatus = "unverified";
            std::cout << "[nn] food trial rejected reason=ambiguous-or-unregistered-click learned=no\n";
            return;
        }
    }
    if (trial.click.serial &&
        observed.serial != trial.click.serial) {
        if (trial.automatic) g.failedFoodRegions.insert(
            food::ActuatorMemory::key(trial.slot,
                trial.before.itemClass, trial.clickedScreen));
        g.foodTrial.reset();
        g.foodStatus = "unverified";
        std::cout << "[nn] food trial rejected reason=multiple-clicks learned=no\n";
        return;
    }
    const auto pending = trial;
    const auto after = foodSnapshot(objects, pending.slot);
    // Since the cursor item may disappear on success, retain the original
    // item/class identity; it is the cursor transition that proves consumption.
    const auto outcome = food::evaluate(pending.before, after);
    if (outcome && pending.click.serial != 0) {
        // The observed item disappearance and food rise *corroborate* the
        // captured native click, but only a direct PC listener is replayable.
        food::CalibratedClick profile;
        profile.slot = pending.slot;
        profile.itemClass = pending.before.itemClass;
        profile.screen = pending.clickedScreen;
        profile.click = pending.click;
        const bool directPcListener =
            profile.click.handlerObject == pending.before.pc &&
            objects.isSubclassOf(
                static_cast<uint16_t>(profile.click.handlerClass), kPcClass);
        const bool learnedRegion =
            !pending.automatic && g.foodActuatorReady &&
            g.foodActuator.remember(profile);
        if (learnedRegion || pending.automatic)
            g.failedFoodRegions.erase(food::ActuatorMemory::key(
                pending.slot, pending.before.itemClass, pending.clickedScreen));
        if (learnedRegion) {
            g.foodActuatorDirty = true;
            autoSaveFoodActuator(true);
        }
        g.foodTrial.reset();
        g.foodStatus = "verified";
        const int action =
            food::actionForClass(static_cast<unsigned>(pending.before.itemClass));
        if (g.trainingEnabled && g.foodBrain.reinforce(
                pending.features, action, outcome->reward, true)) {
            g.foodDirty = true;
            autoSaveFood(false);
        }
        std::cout << "[nn] native food verified"
                  << " item_class=" << pending.before.itemClass
                  << " slot=" << pending.slot
                  << " actuator=" << (learnedRegion
                        ? (directPcListener ? "calibrated-direct-pc" : "observed-only")
                        : (pending.automatic ? "replayed" : "unverified-listener"))
                  << " regions=" << g.foodActuator.size()
                  << " food_gain=" << outcome->foodGain
                  << " hp_delta=" << outcome->hpGain
                  << " reward=" << outcome->reward
                  << " food_samples=" << g.foodBrain.samples()
                  << " food_mean_reward=" << g.foodBrain.meanReward()
                  << " checkpoint=" << (g.foodDirty ? "pending" :
                         (g.foodCheckpointPresent ? "saved" : "new"))
                  << '\n';
        return;
    }
    // Cursor changes without a food increase could be a dropped/equipped
    // item, not evidence of attempted feeding. NEVER train such an event.
    if (after.pc != pending.before.pc ||
        after.level != pending.before.level ||
        after.cursorItem != pending.before.cursorItem ||
        after.food < pending.before.food ||
        ActionPacer::Clock::now() - pending.started >=
            std::chrono::seconds(30)) {
        if (pending.automatic) g.failedFoodRegions.insert(
            food::ActuatorMemory::key(pending.slot,
                pending.before.itemClass, pending.clickedScreen));
        g.foodStatus = "unverified";
        g.foodTrial.reset();
        std::cout << "[nn] native food probe unverified"
                  << " item_class=" << pending.before.itemClass
                  << " slot=" << pending.slot
                  << " food_before=" << pending.before.food
                  << " food_after=" << after.food
                  << " cursor_after=" << after.cursorItem
                  << " learned=no\n";
    }
}


bool regenerateAdvice(VM::ObjectSystem &objects, bool eob3, bool uiActive,
                      bool showMenu = true) {
    if (uiActive || g.advicePending) return false;
    OnlinePolicy::Features features{};
    std::array<bool, 4> blocked{};
    Pose pose{};
    if (!context(objects, features, blocked, &pose) || pose.level < 0)
        return false;
    std::array<bool, OnlinePolicy::kActions> allowed = {
        !blocked[0], !blocked[2], !blocked[3], !blocked[1], true, true
    };
    if (g.failurePose && *g.failurePose == pose)
        for (std::size_t i = 0; i < allowed.size(); ++i)
            if (g.failedActions[i]) allowed[i] = false;
    const auto frontier = frontierActions(pose, allowed, g.visited);
    auto batch = advice::generate(g.brain, features, allowed, frontier,
                                  pose, ++g.adviceGeneration);
    if (batch.count == 0) return false;
    g.adviceBatch = std::move(batch);
    g.advicePose = pose;
    g.adviceOpen = showMenu;
    if (!showMenu) return true; // auto mode logs only attempted objectives
    std::cout << "[nn] advice batch=" << g.adviceGeneration
              << " count=" << g.adviceBatch.count
              << " trained_samples=" << g.brain.samples() << '\n';
    for (int i = 0; i < g.adviceBatch.count; ++i)
        std::cout << "[nn] advice " << (i + 1) << ": "
                  << g.adviceBatch.suggestions[static_cast<std::size_t>(i)].label
                  << " score="
                  << g.adviceBatch.suggestions[static_cast<std::size_t>(i)].score
                  << '\n';
    return true;
}

// Always return to the F1 panel once an action has resolved, including
// timeouts or a changed game scene. If gameplay cannot be observed yet,
// render a safe empty panel with a reason rather than making up actions.
void finishAdviceBatch(VM::ObjectSystem &objects, bool eob3, bool uiActive) {
    if (!g.adviceReopen) return;
    g.adviceReopen = false;
    if (regenerateAdvice(objects, eob3, uiActive)) return;
    g.adviceBatch = {};
    g.adviceOpen = true;
    g.adviceResult += " (F1 closes to return to the game)";
    std::cout << "[nn] advice returned with no safe actions; "
                 "close F1 and resume gameplay to regenerate\n";
}

void observeAdvice(VM::EventSystem &events, VM::ObjectSystem &objects,
                   bool eob3, bool uiActive) {
    if (!g.advicePending) return;
    const auto trial = *g.advicePending;
    const auto now = ActionPacer::Clock::now();
    const int key = OnlinePolicy::keyForAction(trial.action);
    // A queued key is not an executed movement. Give the VM an extra tick
    // after dispatch before evaluating a success or failure.
    const bool queued = events.findEvent(VM::SYS_KEYDOWN, key) >= 0;
    if (queued && now - trial.sent >= std::chrono::seconds(3)) {
        // A key still queued after three seconds was never dispatched.
        // Remove exactly our own input rather than rewarding a nonexistent
        // action or letting it execute much later behind the next menu.
        events.flushEventQueue(0, VM::SYS_KEYDOWN, key);
        g.advicePending.reset();
        g.adviceResult = "Advice timed out before execution; no learning.";
        std::cout << "[nn] advice timeout awaiting native key trained=no\n";
        finishAdviceBatch(objects, eob3, uiActive);
        return;
    }
    if (now - trial.sent < std::chrono::milliseconds(200) || queued) return;
    g.advicePending.reset();
    OnlinePolicy::Features currentFeatures{};
    std::array<bool, 4> blocked{};
    Pose current{};
    const bool observed = !uiActive &&
        context(objects, currentFeatures, blocked, &current);
    const int travel = std::abs(trial.before.x - current.x) +
                       std::abs(trial.before.y - current.y);
    const int floors = std::abs(trial.before.level - current.level);
    // User mouse clicks are never blocked outside the overlay, but an
    // overlapping native click could have caused the observed movement.
    // Refuse to attribute that outcome to the neural recommendation.
    const bool verified = observed &&
        events.lastLeftClick().serial == trial.leftClickSerial &&
        ((floors == 0 && travel <= 1) ||
         (floors == 1 && trial.action < 4));
    const bool firstVisit = observed && !g.visited.contains(visitKey(current));
    const auto result = verified
        ? advice::objectiveOutcome(trial.action, trial.before, trial.target,
                                   current, firstVisit)
        : std::optional<Outcome>{};
    if (result) {
        if (result->success) {
            if (trial.action < 4) g.visited.insert(visitKey(current));
            g.failurePose.reset();
            g.failedActions.fill(false);
        } else {
            if (!g.failurePose || *g.failurePose != trial.before)
                g.failedActions.fill(false);
            g.failurePose = trial.before;
            g.failedActions[static_cast<std::size_t>(trial.action)] = true;
        }
        if (g.trainingEnabled &&
            g.brain.reinforce(trial.features, trial.action,
                              result->reward, result->success)) {
            g.dirty = true;
            // F1 is user-directed and infrequent: persist each verified
            // objective instead of risking 31 unsaved samples on interruption.
            autoSave(true);
        }
        const std::string objective =
            advice::objectiveLabel(trial.target, trial.action, false);
        g.adviceResult = std::string(result->success ? "Reached: " : "Blocked: ") +
            objective;
        std::cout << "[nn] advice objective=" << objective
                  << " reached=" << (result->success ? "yes" : "no")
                  << " reward=" << result->reward
                  << " samples=" << g.brain.samples()
                  << " mean_reward=" << g.brain.meanReward() << '\n';
    } else {
        g.adviceResult = "Objective unverified; no neural reward applied.";
        std::cout << "[nn] advice objective unverified (unexpected pose/screen)"
                  << " trained=no\n";
    }
    finishAdviceBatch(objects, eob3, uiActive);
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
    const auto restored = restoreSharedBrain(g.brain, documents);
    g.storageReady = restored.ready;
    g.checkpointPresent = restored.restored;
    g.savedSamples = g.brain.samples();
    if (!restored.warning.empty())
        std::cerr << "[nn] shared brain migration: "
                  << restored.warning << '\n';
    if (!restored.ready)
        reportStorageError("shared brain unavailable (" + restored.error + ")");
    else if (restored.restored)
        std::cout << "[nn] " << (restored.migrated ? "merged legacy brains" :
                                "shared brain restored")
                  << " samples=" << g.savedSamples
                  << " rate=" << g.brain.learningRate()
                  << " path=" << g.file.string() << '\n';
    else
        std::cout << "[nn] new shared brain checkpoint=new path="
                  << g.file.string() << '\n';
    // Separate version-2 policy sidecar. Old movement/combat/weakness
    // checkpoints stay byte-for-byte compatible, and native nn saves do not
    // contain neural weights. A corrupt existing survival file is preserved.
    g.survivalFile = folder / "survival.ckpt";
    g.survivalStorageReady = true;
    std::error_code survivalError;
    if (std::filesystem::exists(g.survivalFile, survivalError) && !survivalError) {
        std::string error;
        if (g.survivalBrain.load(g.survivalFile, error)) {
            g.savedSurvivalSamples = g.survivalBrain.samples();
            g.survivalCheckpointPresent = true;
            std::cout << "[nn] survival checkpoint restored samples="
                      << g.savedSurvivalSamples << " path="
                      << g.survivalFile.string() << '\n';
        } else {
            g.survivalStorageReady = false;
            std::cerr << "[nn] existing survival checkpoint rejected: "
                      << error << '\n';
        }
    } else if (survivalError) {
        g.survivalStorageReady = false;
        std::cerr << "[nn] survival checkpoint unavailable: "
                  << survivalError.message() << '\n';
    }
    // A separate v2 food outcome model. Old weights and nn-1..3 native
    // game saves never need migration; corrupt existing sidecars are preserved.
    g.foodFile = folder / "food.ckpt";
    g.foodStorageReady = true;
    std::error_code foodError;
    if (std::filesystem::exists(g.foodFile, foodError) && !foodError) {
        std::string error;
        if (g.foodBrain.load(g.foodFile, error)) {
            g.savedFoodSamples = g.foodBrain.samples();
            g.foodCheckpointPresent = true;
            std::cout << "[nn] food checkpoint restored samples="
                      << g.savedFoodSamples << " path="
                      << g.foodFile.string() << '\n';
        } else {
            g.foodStorageReady = false;
            std::cerr << "[nn] existing food checkpoint rejected: "
                      << error << '\n';
        }
    } else if (foodError) {
        g.foodStorageReady = false;
        std::cerr << "[nn] food checkpoint unavailable: "
                  << foodError.message() << '\n';
    }
    // Verified-only UI actuator profile has its own stricter v1 checkpoint.
    g.foodActuatorFile = folder / "food_actuator.ckpt";
    g.foodActuatorReady = true;
    std::error_code actuatorError;
    if (std::filesystem::exists(g.foodActuatorFile, actuatorError) &&
        !actuatorError) {
        std::string error;
        if (g.foodActuator.load(g.foodActuatorFile, error)) {
            g.foodActuatorPresent = true;
            std::cout << "[nn] food regions restored count="
                      << g.foodActuator.size() << '\n';
        } else {
            g.foodActuatorReady = false; // never overwrite corrupted memory
            std::cerr << "[nn] food actuator checkpoint rejected: "
                      << error << '\n';
        }
    } else if (actuatorError) {
        g.foodActuatorReady = false;
        std::cerr << "[nn] actuator checkpoint unavailable: "
                  << actuatorError.message() << '\n';
    }
    // Independent v2 sidecar; legacy movement checkpoints remain unchanged.
    combat::initialize(folder / "combat.ckpt");
}

void flushLearning() {
    if (!g.initialized) return;
    autoSave(true);
    autoSaveSurvival(true);
    autoSaveFood(true);
    autoSaveFoodActuator(true);
    combat::flushLearning();
}

void resetForNewGame() {
    initialize();
    // Game saves and maze generation reset native state, NOT the persistent
    // policy. Commit outstanding evidence before clearing session-only map
    // visits, failed moves and queued actuators.
    flushLearning();
    combat::cancel();
    g.adviceOpen = false;
    g.adviceReopen = false;
    g.advicePending.reset();
    g.autoChoicePacer.reset(advice::AutoChoicePacer::Clock::now());
    g.adviceBatch = {};
    g.adviceResult = "New game started; neural learning was preserved.";
    g.pending.reset();
    g.retreatPending.reset();
    g.foodTrial.reset();
    g.foodStatus = "idle";
    g.failedFoodRegions.clear();
    g.failurePose.reset();
    g.failedActions.fill(false);
    g.visited.clear();
    if (runtime::gDungeonHack)
        runtime::dh::invalidateNeuralSnapshot();
    g.pacer.reset(ActionPacer::Clock::now());
    if (g.spellProbe.shouldOpen()) g.spellProbe.fail();
    for (auto &value : g.spellBaselines) value = {};
    std::cout << "[nn] new game: cleared route memory, retained shared brain"
              << " samples=" << g.brain.samples()
              << " checkpoint=" << (g.storageReady ? "ready" : "unavailable")
              << '\n';
}

void shutdown() {
    if (!g.initialized) return;
    flushLearning();
    combat::shutdown();
    g.autonomous = false;
    g.autoAdvice = false;
    g.adviceOpen = false;
    g.adviceReopen = false;
    g.advicePending.reset();
    g.pending.reset();
    g.retreatPending.reset();
    g.foodTrial.reset();
}

std::filesystem::path checkpointPath() {
    initialize();
    return g.file;
}

bool active() { return g.autonomous; }

bool adviceMenuOpen() { return g.adviceOpen; }
int adviceMenuCount() { return g.adviceOpen ? g.adviceBatch.count : 0; }
uint64_t adviceMenuGeneration() { return g.adviceGeneration; }
std::string adviceMenuLabel(int row) {
    if (!g.adviceOpen || row < 0 || row >= g.adviceBatch.count) return {};
    return g.adviceBatch.suggestions[static_cast<std::size_t>(row)].label;
}
std::string adviceLastResult() { return g.adviceResult; }
void closeAdviceMenu() {
    g.adviceOpen = false;
    g.adviceReopen = false;
}
bool toggleAdviceMenu(VM::ObjectSystem &objects, bool eob3, bool uiActive) {
    initialize();
    if (g.adviceOpen) {
        closeAdviceMenu();
        return false;
    }
    if (g.advicePending) {
        // F1 during execution cancels only the next automatic menu reopen.
        g.adviceReopen = false;
        return false;
    }
    // Explicit manual advice (e.g. the control socket) overrides F1 auto.
    // Never let the two planners queue native movement keys together.
    if (g.autoAdvice) setAutoAdvice(false);
    if (g.autonomous) setActive(false);
    if (regenerateAdvice(objects, eob3, uiActive)) return true;
    // Dungeon Hack's first native draw_walls snapshot may not have arrived
    // yet. Show a safe empty advice screen rather than silently refusing F1.
    // The player can close it to let the runtime finish loading the level.
    g.adviceBatch = {};
    g.adviceOpen = true;
    g.adviceResult = uiActive
        ? "Game menu active. F1 closes this advice screen."
        : "Waiting for the game map. F1 closes to resume.";
    return true;
}
bool chooseAdviceImpl(VM::EventSystem &events, VM::ObjectSystem &objects,
                      int row, bool eob3, bool uiActive, bool automatic) {
    if ((!g.adviceOpen && !(automatic && g.autoAdvice)) ||
        g.advicePending || uiActive ||
        row < 0 || row >= g.adviceBatch.count) return false;
    OnlinePolicy::Features features{};
    std::array<bool, 4> blocked{};
    Pose current{};
    if (!context(objects, features, blocked, &current) ||
        current != g.advicePose) {
        g.adviceResult = "Scene changed. Open F1 again for fresh advice.";
        closeAdviceMenu();
        return false;
    }
    const auto &objective =
        g.adviceBatch.suggestions[static_cast<std::size_t>(row)];
    const int action = objective.action;
    const auto target = advice::targetFor(current, action);
    if (!canMove(action, blocked) || !target ||
        *target != objective.target) {
        g.adviceResult = "Action no longer passable; refreshing advice.";
        regenerateAdvice(objects, eob3, uiActive, !automatic);
        return false;
    }
    const int key = OnlinePolicy::keyForAction(action);
    if (key < 0) return false;
    g.advicePending = State::AdvicePending{
        features, current, action, objective.target, ActionPacer::Clock::now(),
        events.lastLeftClick().serial};
    g.adviceOpen = false;
    g.adviceReopen = !automatic;
    events.postEvent(0, VM::SYS_KEYDOWN, key);
    std::cout << "[nn] " << (automatic ? "auto-advice" : "advice")
              << " execute=" << objective.label
              << " batch=" << g.adviceGeneration
              << " native_key=" << key << " awaiting_observed-outcome\n";
    return true;
}

bool chooseAdvice(VM::EventSystem &events, VM::ObjectSystem &objects,
                  int row, bool eob3, bool uiActive) {
    return chooseAdviceImpl(events, objects, row, eob3, uiActive, false);
}

bool autoAdviceEnabled() { return g.autoAdvice; }

void setAutoAdvice(bool enabled) {
    initialize();
    if (g.autoAdvice == enabled) return;
    if (enabled && g.autonomous)
        setActive(false); // only one movement learner emits native keys
    g.autoAdvice = enabled;
    g.adviceOpen = false;
    g.adviceReopen = false;
    g.adviceBatch = {};
    g.autoChoicePacer.reset(advice::AutoChoicePacer::Clock::now());
    if (!enabled) flushLearning();
    std::cout << "[nn] F1 auto-select advice " << (enabled ? "ON" : "OFF")
              << " interval=1s overlay=hidden samples=" << g.brain.samples()
              << " checkpoint=" << checkpointStatus() << '\n';
}

bool toggleAutoAdvice() {
    setAutoAdvice(!g.autoAdvice);
    return g.autoAdvice;
}

void setActive(bool enabled) {
    initialize();
    g.autonomous = enabled;
    if (enabled) {
        g.autoAdvice = false;
        closeAdviceMenu();
    }
    combat::cancel();
    g.pacer.reset(ActionPacer::Clock::now());
    g.pending.reset();
    g.retreatPending.reset();
    g.foodTrial.reset();
    g.foodStatus = "idle";
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
    // While a clicked recommendation is being measured, other movement
    // keys cannot be credited to its reward. F1/ESC still work normally.
    return (active() || g.advicePending.has_value()) &&
           OnlinePolicy::actionForKey(code) >= 0;
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

bool queueFoodProbe(VM::ObjectSystem &objects, VM::EventSystem &events,
                    int slot, std::string &error) {
    initialize();
    error.clear();
    if (!g.autonomous || !g.trainingEnabled) {
        error = "native food probe requires neural auto and training";
        return false;
    }
    if (g.foodTrial) { error = "native food probe already running"; return false; }
    if (events.leftButtonDown()) {
        error = "release the native left mouse button before calibration";
        return false;
    }
    if (!runtime::uiScreenActive(objects)) {
        error = "open the target PC's native equipment/food screen first";
        return false;
    }
    if (g.spellProbe.busy()) { error = "finish spell probe first"; return false; }
    if (objects.firstObjectOfClass(kMagicClass) < 0) {
        error = "native EOB3 game required"; return false;
    }
    if (slot < 0 || slot >= 6) { error = "invalid party slot"; return false; }
    const auto before = foodSnapshot(objects, slot);
    if (before.item < 0 || before.pc < 0 ||
        food::actionForClass(static_cast<unsigned>(before.itemClass)) < 0) {
        error = "hold a verified food-class item in the native cursor";
        return false;
    }
    if (before.food < 0 || before.food >= 100 ||
        before.hitpoints <= 0 || before.maxHitpoints <= 0) {
        error = "target must be alive and hungry";
        return false;
    }
    if (before.level < 0) { error = "invalid party level"; return false; }
    const auto party = partyResources(objects);
    if (!survival::valid(party) || party.ids[slot] != before.pc) {
        error = "party state unavailable";
        return false;
    }
    const int kernel = objects.firstObjectOfClass(kKernelClass);
    OnlinePolicy::Features base = OnlinePolicy::features(
        readField(objects, kernel, kKernelClass, 243),
        readField(objects, kernel, kKernelClass, 244),
        readField(objects, kernel, kKernelClass, 245),
        {false, false, false, false});
    g.foodTrial = State::FoodTrial{
        before, food::features(base, party, slot), slot,
        ActionPacer::Clock::now(), events.lastLeftClick().serial
    };
    g.foodStatus = "armed";
    std::cout << "[nn] native food trial armed slot=" << slot
              << " item_class=" << before.itemClass
              << " food_before=" << before.food
              << " hp_before=" << before.hitpoints
              << " action=use-native-food-plate"
              << " pending=confirm-consumption\n";
    return true;
}


std::string foodRecommendation(VM::ObjectSystem &objects, int slot) {
    initialize();
    if (slot < 0 || slot >= 6) return "err invalid party slot\n";
    if (g.foodBrain.samples() == 0)
        return "err food preference untrained: verify native food outcomes first\n";
    const int kernel = objects.firstObjectOfClass(kKernelClass);
    const auto party = partyResources(objects);
    if (kernel < 0 || !survival::valid(party) || party.ids[slot] < 0)
        return "err party state unavailable\n";
    std::array<bool, OnlinePolicy::kActions> allowed{};
    auto add = [&](int item) {
        if (item < 0 || objects.classOf(item) == 0xffff) return;
        const int action = food::actionForClass(objects.classOf(item));
        if (action >= 0)
            allowed[static_cast<std::size_t>(action)] = true;
    };
    add(readWord(objects, kernel, kKernelClass, 241)); // in-hand food
    for (int pc : party.ids) {
        if (pc < 0) continue;
        for (int i = 0; i < 26; ++i)
            add(readWord(objects, pc, kPcClass, 81 + i * 2));
    }
    const auto base = OnlinePolicy::features(
        readField(objects, kernel, kKernelClass, 243),
        readField(objects, kernel, kKernelClass, 244),
        readField(objects, kernel, kKernelClass, 245),
        {false, false, false, false});
    const auto features = food::features(base, party, slot);
    const int choice = g.foodBrain.select(features, allowed, 1.0, 0.5);
    if (choice < 0) return "err no known food class carried\n";
    return "nn food preferred_class=" +
           std::to_string(food::classForAction(choice)) +
           " target_slot=" + std::to_string(slot) +
           " food=" + std::to_string(party.food[slot]) +
           " samples=" + std::to_string(g.foodBrain.samples()) +
           " action=diagnostic-only-no-automatic-item-use\nok\n";
}

bool useCalibratedFood(VM::ObjectSystem &objects, VM::EventSystem &events,
                       int slot, std::string &error) {
    error.clear();
    if (!g.foodActuatorReady || g.foodActuator.size() == 0) {
        error = "no verified native food click calibration loaded";
        return false;
    }
    if (events.leftButtonDown()) {
        error = "native left mouse button is already pressed";
        return false;
    }
    if (!runtime::uiScreenActive(objects)) {
        error = "open the intended PC's native equipment screen";
        return false;
    }
    const auto before = foodSnapshot(objects, slot);
    const auto regionKey = food::ActuatorMemory::key(
        slot, before.itemClass, spellScreen(objects));
    if (g.failedFoodRegions.contains(regionKey)) {
        error = "unverified food click suppressed until a new manual calibration";
        return false;
    }
    const auto *profile = g.foodActuator.find(
        slot, before.itemClass, spellScreen(objects));
    if (!profile || profile->click.handlerObject != before.pc ||
        !objects.isSubclassOf(
            static_cast<uint16_t>(profile->click.handlerClass), kPcClass) ||
        !events.matchesLeftClickTarget(profile->click)) {
        error = "live food region/target PC does not match verified calibration";
        return false;
    }
    if (!queueFoodProbe(objects, events, slot, error)) return false;
    g.foodTrial->automatic = true;
    g.foodStatus = "executing";
    events.mouseMove(profile->click.x, profile->click.y);
    events.mouseButton(true, false);
    g.foodClickDown = true;
    return true;
}

std::size_t foodCalibrations() { initialize(); return g.foodActuator.size(); }
const char *foodProbeStatus() { return g.foodStatus; }
uint64_t foodSamples() { initialize(); return g.foodBrain.samples(); }
double foodMeanReward() { initialize(); return g.foodBrain.meanReward(); }
const char *foodCheckpointStatus() {
    initialize();
    if (!g.foodStorageReady) return "unavailable";
    if (g.foodDirty) return "pending";
    return g.foodCheckpointPresent ? "saved" : "new";
}

double survivalMeanReward() { initialize(); return g.survivalBrain.meanReward(); }
double survivalSuccessRate() { initialize(); return g.survivalBrain.successRate(); }
uint64_t survivalSamples() { initialize(); return g.survivalBrain.samples(); }
const char *survivalCheckpointStatus() {
    initialize();
    if (!g.survivalStorageReady) return "unavailable";
    if (g.survivalDirty) return "pending";
    return g.survivalCheckpointPresent ? "saved" : "new";
}

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
    if (g.foodClickDown) {
        events.mouseButton(false, false);
        g.foodClickDown = false;
    }
    // Observe first: a consumed item must not invalidate its own mouse release.
    observeFoodTrial(objects, events);
    if (g.adviceOpen && uiActive && g.adviceBatch.count > 0) {
        g.adviceResult = "Underlying game screen changed; no safe advice.";
        g.adviceBatch = {}; // stay in F1 panel, but never act on stale advice
    }
    if (g.advicePending) {
        // Turning F1 OFF must not leave an undelivered automatic key in
        // the VM queue. A key already consumed is still observed once, so
        // confirmed actions retain their learned success/failure outcome.
        if (!g.autoAdvice && !g.adviceReopen) {
            const int key = OnlinePolicy::keyForAction(g.advicePending->action);
            if (key >= 0 && events.findEvent(VM::SYS_KEYDOWN, key) >= 0) {
                events.flushEventQueue(0, VM::SYS_KEYDOWN, key);
                g.advicePending.reset();
                g.adviceResult = "F1 auto advice stopped before input; no learning.";
                std::cout << "[nn] F1 auto-advice queued key cancelled trained=no\n";
                return;
            }
        }
        observeAdvice(events, objects, eob3, uiActive);
        return; // never issue another automatic action during feedback
    }
    if (g.adviceOpen) return; // in-game overlay owns selection
    if (g.foodTrial) {
        // Preserve the food plate/UI sequence; no parallel autonomous keys
        // or combat clicks while awaiting real native consumption evidence.
        combat::cancel();
        g.pending.reset();
        g.retreatPending.reset();
        g.pacer.reset(ActionPacer::Clock::now());
        return;
    }
    if (g.spellProbe.shouldClose()) {
        closeSpellProbe(objects); // close even if autonomy toggled off or UI is up
        combat::cancel();
        g.pending.reset();
        g.retreatPending.reset();
        g.pacer.reset(ActionPacer::Clock::now());
        return;
    }
    if (!eob3 && g.spellProbe.shouldOpen()) {
        g.spellProbe.fail(); // never send EOB3 magic messages to Dungeon Hack
    }
    if (g.autoAdvice) {
        // F1 auto-select uses the SAME objective planner, native keyboard
        // actuator, measured pose feedback and autosave as manual F1 advice.
        // No SDL overlay, no mouse hijacking, and no parallel combat keys.
        if (uiActive) {
            g.autoChoicePacer.reset(advice::AutoChoicePacer::Clock::now());
            return;
        }
        if (!g.autoChoicePacer.ready(advice::AutoChoicePacer::Clock::now()))
            return;
        combat::cancel();
        if (!regenerateAdvice(objects, eob3, false, false))
            return; // waiting for a valid live map; no fabricated action
        if (!chooseAdviceImpl(events, objects, 0, eob3, false))
            g.autoChoicePacer.reset(advice::AutoChoicePacer::Clock::now());
        return;
    }
    if (eob3 && g.autonomous && !uiActive) observeSpellStock(objects);
    // Never interpret changes caused by menus or manual control as rewards.
    if (!g.autonomous || uiActive) {
        // A queued probe must never fire later after an unrelated menu closes.
        if (g.spellProbe.shouldOpen()) g.spellProbe.fail();
        combat::cancel();
        g.pending.reset();
        g.retreatPending.reset();
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
        g.retreatPending.reset();
        return;
    }
    const bool firstVisit = g.visited.insert(visitKey(current)).second;
    if (eob3) combat::observe(objects, current, g.trainingEnabled);
    else combat::cancel();
    // Do not issue another action while waiting for spell or attack damage.
    if (eob3 && combat::waiting()) return;

    // Outcome feedback for a verified backstep/strafe from imminent danger.
    // The pending action is evaluated only after the VM had a full action tick.
    if (g.retreatPending) {
        const auto pending = *g.retreatPending;
        g.retreatPending.reset();
        const auto nowParty = partyResources(objects);
        const bool moved = pending.before.level == current.level &&
            (pending.before.x != current.x || pending.before.y != current.y) &&
            std::abs(pending.before.x - current.x) +
            std::abs(pending.before.y - current.y) == 1;
        const auto result = survival::evaluate(
            pending.party, nowParty, moved, combat::threatAhead(objects, current));
        if (result && g.trainingEnabled &&
            g.survivalBrain.reinforce(pending.features, pending.action,
                                      result->reward, result->success)) {
            g.survivalDirty = true;
            autoSaveSurvival(false);
            std::cout << "[nn] survival learned action="
                      << kActionNames[static_cast<std::size_t>(pending.action)]
                      << " hp_delta=" << result->hpDelta
                      << " food_delta=" << result->foodDelta
                      << " potions_before=" << pending.party.potions
                      << " potions_after=" << nowParty.potions
                      << " rations_before=" << pending.party.rations
                      << " rations_after=" << nowParty.rations
                      << " escaped=" << (result->success ? "yes" : "no")
                      << " reward=" << result->reward
                      << " samples=" << g.survivalBrain.samples()
                      << " mean_reward=" << g.survivalBrain.meanReward()
                      << " checkpoint=" << survivalCheckpointStatus() << '\n';
        }
    }

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

    // Verified survival action: withdraw from an actual front-cell monster
    // when a living member is critically injured. No potion type, food icon,
    // spell menu, or unverified camp coordinate is ever guessed. Inventory
    // availability conditions decisions, but no consumables are auto-used.
    if (eob3 && !g.retreatPending) {
        const auto party = partyResources(objects);
        if (survival::shouldWithdraw(
                party, combat::threatAhead(objects, current), blocked)) {
            const auto allowed = survival::exits(blocked);
            const auto sf = survival::features(features, party);
            std::uniform_real_distribution<double> random(0.0, 1.0);
            const int action = g.survivalBrain.select(
                sf, allowed, random(g.rng), random(g.rng));
            if (action >= 1 && action <= 3) {
                events.postEvent(0, VM::SYS_KEYDOWN,
                                 OnlinePolicy::keyForAction(action));
                g.retreatPending = State::RetreatPending{sf, party, current, action};
                std::cout << "[nn] survival retreat="
                          << kActionNames[static_cast<std::size_t>(action)]
                          << " weakest_hp_ratio=" << survival::weakestHealth(party)
                          << " rations=" << party.rations
                          << " potions_unclassified=" << party.potions
                          << " reason=adjacent-threat-and-low-hp"
                          << " item_use=no\n";
                return;
            }
        }
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
    g.retreatPending.reset();
    g.foodTrial.reset();
    g.foodStatus = "idle";
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
