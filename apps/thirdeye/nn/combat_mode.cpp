#include "combat_mode.hpp"

#include "../vm/events.hpp"
#include "../vm/objects.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
#include <random>
#include <string>

namespace THIRDEYE::nn::combat {
namespace {
constexpr uint16_t kKernelClass = 1382;
constexpr uint16_t kEntityClass = 1370;
constexpr uint16_t kPcClass = 1369;
constexpr uint16_t kNpcClass = 1622;
constexpr uint64_t kSaveEvery = 16;
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
};
struct State {
    OnlinePolicy brain;
    std::filesystem::path file;
    bool ready = false;
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
        if (hp > 0) return Target{id, hp};
    }
    return std::nullopt;
}

bool attackerFeatures(VM::ObjectSystem &objects, const mode::Pose &pose,
                      const std::array<bool, 4> &blocked, int npcHP,
                      OnlinePolicy::Features &features,
                      std::array<bool, OnlinePolicy::kActions> &allowed) {
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
        allowed[index] = true;
        features[7 + index] = std::clamp(static_cast<double>(hp) / maxHP,
                                          0.0, 1.0);
    }
    features[10] = std::clamp(npcHP / 100.0, 0.0, 1.0);
    return std::any_of(allowed.begin(), allowed.end(),
                       [](bool value) { return value; });
}
} // namespace

void initialize(const std::filesystem::path &path) {
    if (g.ready || !g.file.empty() || path.empty()) return;
    g.file = path;
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

void shutdown() {
    save(true);
    g.pending.reset();
}

void cancel() { g.pending.reset(); }

void release(VM::EventSystem &events) {
    if (g.mouseDown) {
        events.mouseButton(false, false);
        g.mouseDown = false;
    }
}

void observe(VM::ObjectSystem &objects, const mode::Pose &pose, bool training) {
    if (!g.pending) return;
    const Pending p = *g.pending;
    g.pending.reset();
    if (!training || pose != p.pose ||
        objects.classOf(p.npc) == 0xffff ||
        !objects.isSubclassOf(objects.classOf(p.npc), kNpcClass))
        return;
    const int hpAfter = word(objects, p.npc, kNpcClass, 3);
    const auto outcome = mode::evaluateCombatOutcome(p.hpBefore, hpAfter);
    if (!outcome) return;
    const double rate = g.brain.learningRate();
    if (!g.brain.reinforce(p.features, p.action, outcome->reward,
                           outcome->success))
        return;
    g.dirty = true;
    save(false);
    std::cout << "[nn] combat action=weapon-pc" << kPartySlots[p.action]
              << " target=" << p.npc
              << " damage=" << p.hpBefore - hpAfter
              << " reward=" << outcome->reward
              << " success=" << (outcome->success ? "yes" : "no")
              << " rate=" << rate
              << " samples=" << g.brain.samples()
              << " checkpoint=" << checkpointStatus() << '\n';
}

bool attack(VM::EventSystem &events, VM::ObjectSystem &objects,
            const mode::Pose &pose, const std::array<bool, 4> &blocked) {
    const auto enemy = frontTarget(objects, pose);
    if (!enemy) return false;
    OnlinePolicy::Features features{};
    std::array<bool, OnlinePolicy::kActions> allowed{};
    if (!attackerFeatures(objects, pose, blocked, enemy->hp, features, allowed))
        return false;
    std::uniform_real_distribution<double> random(0.0, 1.0);
    const int action = g.brain.select(features, allowed,
                                      random(g.rng), random(g.rng));
    if (action < 0 || action >= static_cast<int>(kWeaponIcons.size()))
        return false;
    const auto &icon = kWeaponIcons[static_cast<std::size_t>(action)];
    events.mouseMove(icon[0], icon[1]);
    events.mouseButton(false, true);
    g.mouseDown = true;
    g.pending = Pending{features, pose, action, enemy->id, enemy->hp};
    std::cout << "[nn] combat attack=weapon-pc" << kPartySlots[action]
              << " target=" << enemy->id << " hp=" << enemy->hp << '\n';
    return true;
}

uint64_t samples() { return g.brain.samples(); }
double meanReward() { return g.brain.meanReward(); }
double successRate() { return g.brain.successRate(); }
const char *checkpointStatus() {
    if (!g.ready) return "unavailable";
    if (g.dirty) return "pending";
    return g.checkpointExists ? "saved" : "new";
}
} // namespace THIRDEYE::nn::combat
