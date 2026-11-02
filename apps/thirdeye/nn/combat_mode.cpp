#include "combat_mode.hpp"
#include "tactical_memory.hpp"

#include "../vm/events.hpp"
#include "../vm/objects.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace THIRDEYE::nn::combat {
namespace {
constexpr uint16_t kKernelClass = 1382;
constexpr uint16_t kEntityClass = 1370;
constexpr uint16_t kPcClass = 1369;
constexpr uint16_t kNpcClass = 1622;
constexpr uint64_t kSaveEvery = 16;
constexpr uint64_t kTacticSaveEvery = 8;
constexpr std::array<int, 3> kPartySlots{0, 1, 3};
constexpr std::array<std::array<int, 2>, 3> kWeaponIcons{{
    {229, 20}, {296, 18}, {297, 70}
}};

struct Pending {
    OnlinePolicy::Features features{};
    mode::Pose pose{};
    int action = -1;
    int npc = -1;
    int hpBefore = 0;
    tactics::Signature signature{};
    int observationTicks = 0;
};
struct State {
    OnlinePolicy brain;
    tactics::Memory weaknesses;
    std::filesystem::path file;
    std::filesystem::path tacticsFile;
    bool ready = false;
    bool tacticalStorageReady = false;
    bool tacticalDirty = false;
    bool tacticalCheckpointExists = false;
    bool tacticalWarned = false;
    uint64_t savedTacticalSamples = 0;
    bool checkpointExists = false;
    bool dirty = false;
    bool warned = false;
    bool mouseDown = false;
    uint64_t savedSamples = 0;
    std::optional<Pending> pending;
    std::mt19937 rng{std::random_device{}()};
};
State g;

int byte(VM::ObjectSystem &objects, int id, uint16_t cls, uint32_t offset) {
    try {
        if (auto *ptr = objects.classStaticPtr(id, cls, offset, 1))
            return ptr[0];
    } catch (const std::exception &) {}
    return -1;
}

int word(VM::ObjectSystem &objects, int id, uint16_t cls, uint32_t offset) {
    try {
        if (auto *ptr = objects.classStaticPtr(id, cls, offset, 2)) {
            const uint16_t raw = static_cast<uint16_t>(ptr[0]) |
                                 (static_cast<uint16_t>(ptr[1]) << 8);
            return static_cast<int16_t>(raw);
        }
    } catch (const std::exception &) {}
    return -1;
}

void warn(const std::string &message) {
    if (!g.warned) {
        std::cerr << "[nn] combat checkpoint error: " << message << '\n';
        g.warned = true;
    }
}

void saveTactics(bool force) {
    if (!g.tacticalStorageReady || !g.tacticalDirty) return;
    if (!force && g.weaknesses.samples() - g.savedTacticalSamples <
                      kTacticSaveEvery) return;
    std::string error;
    if (g.weaknesses.save(g.tacticsFile, error)) {
        g.savedTacticalSamples = g.weaknesses.samples();
        g.tacticalCheckpointExists = true;
        g.tacticalDirty = false;
        std::cout << "[nn] enemy weakness memory saved samples="
                  << g.savedTacticalSamples
                  << " enemy-tactics=" << g.weaknesses.entries() << '\n';
    } else if (!g.tacticalWarned) {
        std::cerr << "[nn] weakness memory save failed: " << error << '\n';
        g.tacticalWarned = true;
    }
}

void save(bool force) {
    if (!g.ready || !g.dirty) return;
    if (!force && g.brain.samples() - g.savedSamples < kSaveEvery)
        return;
    std::string error;
    if (!g.brain.save(g.file, error)) {
        warn(error);
        return;
    }
    g.savedSamples = g.brain.samples();
    g.checkpointExists = true;
    g.dirty = false;
    std::cout << "[nn] combat checkpoint saved samples=" << g.savedSamples
              << " path=" << g.file.string() << '\n';
}

struct Target {
    int id;
    int hp;
    uint16_t enemyClass;
};

std::optional<Target> frontTarget(VM::ObjectSystem &objects,
                                  const mode::Pose &pose) {
    for (int id = 0; id < VM::ObjectSystem::kNumEntities; ++id) {
        const uint16_t cls = objects.classOf(id);
        if (cls == 0xffff || !objects.isSubclassOf(cls, kNpcClass))
            continue;
        const mode::Pose npc{
            byte(objects, id, kEntityClass, 2),
            byte(objects, id, kEntityClass, 3),
            -1,
            byte(objects, id, kEntityClass, 4)
        };
        if (!mode::targetDirectlyAhead(pose, npc)) continue;
        const int hp = word(objects, id, kNpcClass, 3);
        if (hp > 0) return Target{id, hp, cls};
    }
    return std::nullopt;
}

// Only attack with a weapon or use a spell from an equipped scroll/wand.
// Spellbooks and holy symbols launch a separate unverified spell menu and
// MUST NOT be clicked until its actual event regions are mapped.
uint16_t exportedSpellId(VM::ObjectSystem &objects, int item,
                         uint16_t itemClass) {
    const uint16_t type = itemClass;
    // Read a named exported field only. Never infer spell id from an
    // arbitrary static byte. Different wand/scroll spells can share a class.
    constexpr std::array<const char *, 8> tags{
        "B:spell", "W:spell", "B:spellnum", "W:spellnum",
        "B:spell_type", "W:spell_type", "B:spell_number", "W:spell_number"
    };
    uint16_t cls = type;
    for (int depth = 0; depth < 12; ++depth) {
        const auto *descriptor = objects.classByNumber(cls);
        if (!descriptor) break;
        for (const char *tag : tags) {
            const auto it = descriptor->exportedVars.find(tag);
            if (it == descriptor->exportedVars.end()) continue;
            const int size = tag[0] == 'B' ? 1 : 2;
            try {
                if (auto *p = objects.classStaticPtr(item, cls, it->second, size))
                    return size == 1 ? p[0] :
                        static_cast<uint16_t>(p[0] | (p[1] << 8));
            } catch (const std::exception &) {}
        }
        if (descriptor->header.parent == cls ||
            descriptor->header.parent > 65535) break;
        cls = static_cast<uint16_t>(descriptor->header.parent);
    }
    return 0;
}

std::optional<tactics::Signature> equippedTactic(
    VM::ObjectSystem &objects, int pc, uint16_t enemyClass) {
    const int item = word(objects, pc, kPcClass, 81 + 16 * 2);
    if (item < 0 || objects.classOf(item) == 0xffff)
        return std::nullopt;
    const auto itemClass = objects.classOf(item);
    tactics::Kind kind;
    if (objects.isSubclassOf(itemClass, 1377))
        kind = tactics::Kind::ScrollSpell;
    else if (objects.isSubclassOf(itemClass, 1376))
        kind = tactics::Kind::WandSpell;
    else if (itemClass == 1327 || itemClass == 1333 ||
             itemClass == 1334 || itemClass == 1335 || itemClass == 1336)
        kind = tactics::Kind::Ranged;
    else if (objects.isSubclassOf(itemClass, 1688))
        kind = tactics::Kind::Melee;
    else
        return std::nullopt;
    const uint16_t spellId =
        (kind == tactics::Kind::ScrollSpell ||
         kind == tactics::Kind::WandSpell) ?
        exportedSpellId(objects, item, itemClass) : 0;
    return tactics::Signature{enemyClass, itemClass, kind, spellId};
}

bool attackerFeatures(VM::ObjectSystem &objects, const mode::Pose &pose,
                      const std::array<bool, 4> &blocked, int npcHP,
                      uint16_t enemyClass,
                      OnlinePolicy::Features &features,
                      std::array<bool, OnlinePolicy::kActions> &allowed,
                      std::vector<tactics::Candidate> &candidates) {
    const int kernel = objects.firstObjectOfClass(kKernelClass);
    if (kernel < 0) return false;
    features = OnlinePolicy::features(pose.x, pose.y, pose.facing, blocked);
    allowed.fill(false);
    // Repurpose the independent policy's last four feature slots for
    // relative party health and observed enemy health.
    for (std::size_t index = 0; index < kPartySlots.size(); ++index) {
        const int slot = kPartySlots[index];
        const int pc = word(objects, kernel, kKernelClass, 229 + slot * 2);
        if (pc < 0 || objects.classOf(pc) == 0xffff ||
            !objects.isSubclassOf(objects.classOf(pc), kPcClass))
            continue;
        const int hp = word(objects, pc, kPcClass, 171);
        const int maxHP = word(objects, pc, kPcClass, 173);
        if (hp <= 0 || maxHP <= 0) continue;
        const auto tactic = equippedTactic(objects, pc, enemyClass);
        if (!tactic) continue;
        // EOB3 rear rank cannot strike with ordinary melee weapons.
        if (slot >= 2 && tactic->kind == tactics::Kind::Melee) continue;
        candidates.push_back({slot, *tactic});
        allowed[index] = true;
        features[7 + index] = std::clamp(static_cast<double>(hp) / maxHP,
                                          0.0, 1.0);
    }
    features[10] = std::clamp(npcHP / 100.0, 0.0, 1.0);
    return !candidates.empty();
}
} // namespace

void initialize(const std::filesystem::path &path) {
    if (g.ready || !g.file.empty() || path.empty()) return;
    g.file = path;
    g.tacticsFile = path.parent_path() / "weaknesses.ckpt";
    g.tacticalStorageReady = true;
    std::error_code tacticalError;
    if (std::filesystem::exists(g.tacticsFile, tacticalError) && !tacticalError) {
        std::string error;
        if (g.weaknesses.load(g.tacticsFile, error)) {
            g.savedTacticalSamples = g.weaknesses.samples();
            g.tacticalCheckpointExists = true;
            std::cout << "[nn] enemy weakness memory loaded samples="
                      << g.savedTacticalSamples << '\n';
        } else {
            // Preserve a damaged checkpoint; live learning may continue.
            g.tacticalStorageReady = false;
            std::cerr << "[nn] weakness memory rejected: " << error << '\n';
        }
    } else if (tacticalError) {
        g.tacticalStorageReady = false;
        std::cerr << "[nn] weakness memory unavailable: "
                  << tacticalError.message() << '\n';
    }
    g.ready = true;
    std::error_code ec;
    if (std::filesystem::exists(path, ec) && !ec) {
        std::string error;
        if (g.brain.load(path, error)) {
            g.savedSamples = g.brain.samples();
            g.checkpointExists = true;
            std::cout << "[nn] combat checkpoint loaded samples="
                      << g.savedSamples << " path=" << path.string() << '\n';
        } else {
            // Never silently overwrite a damaged existing combat policy.
            warn(error);
            g.ready = false;
        }
    } else if (ec) {
        warn(ec.message());
        g.ready = false;
    }
}

void flushLearning() {
    save(true);
    saveTactics(true);
}

void shutdown() {
    flushLearning();
    g.pending.reset();
}

void cancel() { g.pending.reset(); }
bool waiting() { return g.pending.has_value(); }

void release(VM::EventSystem &events) {
    if (g.mouseDown) {
        events.mouseButton(false, false);
        g.mouseDown = false;
    }
}

void observe(VM::ObjectSystem &objects, const mode::Pose &pose, bool training) {
    if (!g.pending) return;
    Pending p = *g.pending;
    g.pending.reset();
    if (!training || pose != p.pose ||
        objects.classOf(p.npc) == 0xffff ||
        !objects.isSubclassOf(objects.classOf(p.npc), kNpcClass))
        return;
    const int hpAfter = word(objects, p.npc, kNpcClass, 3);
    // Wait for delayed weapon and magic projectiles before declaring a miss.
    if (hpAfter == p.hpBefore && p.observationTicks++ < 2) {
        g.pending = p;
        return;
    }
    const auto outcome = mode::evaluateCombatOutcome(p.hpBefore, hpAfter);
    if (!outcome) return;
    const double rate = g.brain.learningRate();
    if (!g.brain.reinforce(p.features, p.action, outcome->reward,
                           outcome->success))
        return;
    g.dirty = true;
    const int damage = p.hpBefore - hpAfter;
    g.weaknesses.learn(p.signature, damage);
    g.tacticalDirty = true;
    saveTactics(false);
    const auto experience = g.weaknesses.experience(p.signature);
    std::cout << "[nn] weakness learned enemy_class=" << p.signature.enemy
              << " tactic=" << tactics::kindName(p.signature.kind)
              << " item_class=" << p.signature.itemClass
              << " spell_id=" << p.signature.spellId
              << " damage=" << damage
              << " mean_damage=" << (experience.trials ? experience.damage / experience.trials : 0.0)
              << " hits=" << experience.hits
              << " trials=" << experience.trials
              << " tactical_checkpoint=" << tacticalCheckpointStatus()
              << '\n';
    save(false);
    std::cout << "[nn] combat action=weapon-pc" << kPartySlots[p.action]
              << " target=" << p.npc
              << " damage=" << p.hpBefore - hpAfter
              << " reward=" << outcome->reward
              << " success=" << (outcome->success ? "yes" : "no")
              << " rate=" << rate
              << " samples=" << g.brain.samples()
              << " mean_reward=" << g.brain.meanReward()
              << " success_rate=" << g.brain.successRate()
              << " checkpoint=" << checkpointStatus() << '\n';
}

bool attack(VM::EventSystem &events, VM::ObjectSystem &objects,
            const mode::Pose &pose, const std::array<bool, 4> &blocked) {
    const auto enemy = frontTarget(objects, pose);
    if (!enemy) return false;
    OnlinePolicy::Features features{};
    std::array<bool, OnlinePolicy::kActions> allowed{};
    std::vector<tactics::Candidate> candidates;
    if (!attackerFeatures(objects, pose, blocked, enemy->hp, enemy->enemyClass,
                          features, allowed, candidates))
        return false;
    std::uniform_real_distribution<double> random(0.0, 1.0);
    // Both models learn from measured NPC HP changes. The enemy-specific UCB
    // selector still explores distinct weapons/spells; the online combat
    // policy chooses which party member should use equivalent equipment.
    const int learnedAction = g.brain.select(
        features, allowed, random(g.rng), random(g.rng));
    const int preferredSlot = learnedAction >= 0 &&
                              learnedAction < static_cast<int>(kPartySlots.size())
        ? kPartySlots[static_cast<std::size_t>(learnedAction)] : -1;
    const int choice = g.weaknesses.select(
        candidates, random(g.rng), preferredSlot);
    if (choice < 0) return false;
    const int selectedSlot = candidates[static_cast<std::size_t>(choice)].partySlot;
    const auto it = std::find(kPartySlots.begin(), kPartySlots.end(), selectedSlot);
    if (it == kPartySlots.end()) return false;
    const int action = static_cast<int>(it - kPartySlots.begin());
    if (action < 0 || action >= static_cast<int>(kWeaponIcons.size()))
        return false;
    const auto &icon = kWeaponIcons[static_cast<std::size_t>(action)];
    events.mouseMove(icon[0], icon[1]);
    events.mouseButton(false, true);
    g.mouseDown = true;
    const auto signature = candidates[static_cast<std::size_t>(choice)].signature;
    g.pending = Pending{features, pose, action, enemy->id, enemy->hp, signature};
    std::cout << "[nn] combat experiment="
              << tactics::kindName(signature.kind)
              << " item_class=" << signature.itemClass
              << " spell_id=" << signature.spellId
              << " attacker=pc" << kPartySlots[action]
              << " enemy_class=" << signature.enemy
              << " target=" << enemy->id << " hp=" << enemy->hp
              << " previous_trials=" << g.weaknesses.experience(signature).trials
              << " policy_preferred_pc=" << preferredSlot
              << '\n';
    return true;
}

uint64_t samples() { return g.brain.samples(); }
uint64_t tacticalSamples() { return g.weaknesses.samples(); }
std::size_t knownWeaknesses() { return g.weaknesses.entries(); }
const char *tacticalCheckpointStatus() {
    if (!g.tacticalStorageReady) return "unavailable";
    if (g.tacticalDirty) return "pending";
    return g.tacticalCheckpointExists ? "saved" : "new";
}
double meanReward() { return g.brain.meanReward(); }
double successRate() { return g.brain.successRate(); }
const char *checkpointStatus() {
    if (!g.ready) return "unavailable";
    if (g.dirty) return "pending";
    return g.checkpointExists ? "saved" : "new";
}
} // namespace THIRDEYE::nn::combat
