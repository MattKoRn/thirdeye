#include "rolling_save.hpp"

#include "../automap.hpp"
#include "../resources/res.hpp"
#include "../runtime/internal.hpp"
#include "../savegame/items_tmp.hpp"
#include "../savegame/lvl_tmp.hpp"
#include "../savegame/savegame_dir.hpp"
#include "../vm/objects.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace THIRDEYE::nn::rolling {
namespace {
namespace fs = std::filesystem;
constexpr uint16_t kKernel = 1382, kPc = 1369;
struct State {
    Schedule schedule;
    fs::path dir;
    int nextIndex = 0;
    bool recoveryAttempted = false;
    bool warned = false;
};
State g;

int field(VM::ObjectSystem &objects, int obj, uint16_t cls,
          uint32_t offset, int bytes) {
    try {
        if (auto *p = objects.classStaticPtr(obj, cls, offset, bytes)) {
            if (bytes == 1) return p[0];
            const uint16_t raw = static_cast<uint16_t>(p[0]) |
                                 (static_cast<uint16_t>(p[1]) << 8);
            return static_cast<int16_t>(raw);
        }
    } catch (const std::exception &) {}
    return -32768; // invalid statics sentinel; negative HP stays observable
}

fs::path itemsPath(const fs::path &dir, int slot) {
    char name[32];
    std::snprintf(name, sizeof(name), "ITEMS_%02d.BIN", slot);
    return dir / name;
}
fs::path levelPath(const fs::path &dir, int level, int slot) {
    char name[32];
    std::snprintf(name, sizeof(name), "LVL%02d_%02d.BIN", level, slot);
    return dir / name;
}

bool healthySave(const fs::path &dir, int slot) {
    std::error_code ec;
    if (!fs::is_regular_file(itemsPath(dir, slot), ec) || ec) return false;
    bool levelPresent = false;
    for (int level = 1; level <= 14; ++level) {
        if (fs::is_regular_file(levelPath(dir, level, slot), ec) && !ec) {
            levelPresent = true;
            break;
        }
    }
    if (!levelPresent) return false;
    const auto items = savegame::loadItemsTmp(itemsPath(dir, slot));
    if (items.position.level < 1 || items.position.level > 14) return false;
    for (const auto &pc : items.characters)
        if (pc.classNumber == kPc && pc.hpCurrent > 0)
            return true;
    return false;
}

bool namedSlot(const fs::path &dir, int index) {
    const auto entries = savegame::loadSaveDir(dir / "SAVEGAME.DIR");
    const int slot = slotForIndex(index);
    return slot < static_cast<int>(entries.size()) &&
           entries[static_cast<std::size_t>(slot)].name == nameForIndex(index);
}

bool hasExistingLevel(const fs::path &dir, int slot) {
    for (int level = 1; level <= 14; ++level)
        if (fs::exists(levelPath(dir, level, slot))) return true;
    return false;
}

bool slotAvailable(const fs::path &dir, int index) {
    const int slot = slotForIndex(index);
    const auto entries = savegame::loadSaveDir(dir / "SAVEGAME.DIR");
    if (slot < static_cast<int>(entries.size()) &&
        entries[static_cast<std::size_t>(slot)].used &&
        entries[static_cast<std::size_t>(slot)].name != nameForIndex(index))
        return false;
    // Never overwrite an existing unnamed backup; it may be a user save.
    return namedSlot(dir, index) ||
           (!fs::exists(itemsPath(dir, slot)) &&
            !hasExistingLevel(dir, slot));
}

bool replace(const fs::path &temp, const fs::path &dest) {
#ifdef _WIN32
    return MoveFileExW(temp.c_str(), dest.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::error_code ec;
    fs::rename(temp, dest, ec);
    return !ec;
#endif
}

// Preserve ordinary player names; NN entries live at indices 12..14, beyond
// the game's 12-slot manual picker. Native writer preserves these tail entries.
bool writeName(const fs::path &dir, int index) {
    const fs::path path = dir / "SAVEGAME.DIR";
    auto entries = savegame::loadSaveDir(path);
    if (entries.size() < 15) entries.resize(15);
    const int slot = slotForIndex(index);
    entries[static_cast<std::size_t>(slot)] = {nameForIndex(index), true};
    fs::path temp = path;
    temp += ".nn-stage";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        for (const auto &entry : entries) {
            if (entry.used && !entry.name.empty()) {
                out.write(entry.name.data(),
                          static_cast<std::streamsize>(std::min<std::size_t>(
                              entry.name.size(), 26)));
            } else {
                out << "__________________________";
            }
            out << "\r\n";
        }
        out.put('\x1a');
        out.flush();
        if (!out) return false;
    }
    if (replace(temp, path)) return true;
    std::error_code ignored;
    fs::remove(temp, ignored);
    return false;
}

std::optional<int> newestHealthy(const fs::path &dir) {
    std::optional<int> best;
    fs::file_time_type newest = fs::file_time_type::min();
    for (int index = 0; index < kSaveCount; ++index) {
        if (!namedSlot(dir, index) || !healthySave(dir, slotForIndex(index)))
            continue;
        std::error_code ec;
        const auto time = fs::last_write_time(itemsPath(dir, slotForIndex(index)), ec);
        if (!ec && (!best || time > newest)) {
            best = index;
            newest = time;
        }
    }
    return best;
}

void discover(const fs::path &dir) {
    if (dir == g.dir) return;
    g.dir = dir;
    g.recoveryAttempted = false;
    if (const auto latest = newestHealthy(dir))
        g.nextIndex = (*latest + 1) % kSaveCount;
    else
        g.nextIndex = 0;
}

bool saveNow(runtime::Context &ctx, int level) {
    for (int attempt = 0; attempt < kSaveCount; ++attempt) {
        const int index = (g.nextIndex + attempt) % kSaveCount;
        if (!slotAvailable(g.dir, index)) continue;
        const int slot = slotForIndex(index);
        VM::Value result = 0;
        if (!runtime::eye::tryHandle(ctx, "save_game",
                                    {slot, level}, result) || result != 1)
            return false;
        if (!writeName(g.dir, index)) return false;
        g.nextIndex = (index + 1) % kSaveCount;
        std::cout << "[nn] in-game autosave " << nameForIndex(index)
                  << " (slot " << slot << ", level " << level << ") OK\n";
        return true;
    }
    if (!g.warned) {
        std::cerr << "[nn] no free NN slots (12..14); refusing to overwrite "
                     "unrelated game saves\n";
        g.warned = true;
    }
    return false;
}
} // namespace

Schedule::Party partyState(VM::ObjectSystem &objects) {
    const int kernel = objects.firstObjectOfClass(kKernel);
    if (kernel < 0) return Schedule::Party::Unknown;
    int members = 0;
    for (int slot = 0; slot < 6; ++slot) {
        const int pc = field(objects, kernel, kKernel, 229 + slot * 2, 2);
        if (pc < 0 || objects.classOf(pc) == 0xffff ||
            !objects.isSubclassOf(objects.classOf(pc), kPc))
            continue;
        const int hp = field(objects, pc, kPc, 171, 2);
        if (hp < -10000) continue;
        ++members;
        if (hp > 0) return Schedule::Party::Alive;
    }
    return members ? Schedule::Party::Dead : Schedule::Party::Unknown;
}

void tick(runtime::Context &ctx, bool autonomy, bool gameplay) {
    const auto now = Schedule::Clock::now();
    if (!autonomy) {
        g.schedule.disable();
        g.recoveryAttempted = false;
        return;
    }
    discover(ctx.res.resourcePath().parent_path() / "SAVEGAME");
    if (!g.schedule.enabled()) g.schedule.enable(now);
    const auto party = partyState(ctx.objects);
    if (!g.recoveryAttempted && g.schedule.shouldRestore(now, party)) {
        g.recoveryAttempted = true;
        if (const auto best = newestHealthy(g.dir)) {
            const int slot = slotForIndex(*best);
            std::cout << "[nn] game over: reloading recommended "
                      << nameForIndex(*best) << " after 5 seconds\n";
            throw runtime::NeuralRecovery{slot};
        }
        std::cerr << "[nn] game over: no healthy nn-1/nn-2/nn-3 "
                     "in-game save is available for recovery\n";
    }
    if (!g.schedule.gameOverPending() &&
        g.schedule.shouldSave(now, gameplay, party)) {
        const int kernel = ctx.objects.firstObjectOfClass(kKernel);
        const int level = field(ctx.objects, kernel, kKernel, 246, 1);
        if (level >= 1 && level <= 14 && !saveNow(ctx, level))
            std::cerr << "[nn] in-game autosave failed (previous saves retained)\n";
    }
}

bool restore(int slot, RESOURCES::Resource &resource) {
    const int index = indexForSlot(slot);
    if (index < 0 || index >= kSaveCount) return false;
    const auto dir = resource.resourcePath().parent_path() / "SAVEGAME";
    if (!namedSlot(dir, index) || !healthySave(dir, slot)) return false;
    if (!savegame::restoreItems(dir, slot)) return false;
    if (savegame::restoreLevels(dir, slot) < 1) return false;
    char name[32];
    std::snprintf(name, sizeof(name), "MAPS_%02d.BIN", slot);
    std::error_code ec;
    const auto src = dir / name;
    const auto live = dir / "MAPS.TMP";
    if (fs::exists(src, ec) && !ec &&
        fs::copy_file(src, live, fs::copy_options::overwrite_existing, ec) &&
        !ec) {
        if (!automap::loadFrom(live)) automap::reset();
    } else {
        automap::reset();
    }
    return true;
}

void afterRestore() {
    g.recoveryAttempted = false;
    g.schedule.afterRestore(Schedule::Clock::now());
}
void reset() {
    g.schedule.disable();
    g.recoveryAttempted = false;
}
} // namespace THIRDEYE::nn::rolling
