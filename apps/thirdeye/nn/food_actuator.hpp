#ifndef THIRDEYE_NN_FOOD_ACTUATOR_HPP
#define THIRDEYE_NN_FOOD_ACTUATOR_HPP

#include "food_learning.hpp"
#include "../vm/events.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <tuple>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace THIRDEYE::nn::food {

// Learned ONLY after an actual native left-click causes confirmed food
// consumption. Object IDs/window handles are unstable across game restores,
// so the live region is revalidated by geometry + handler class + message.
struct CalibratedClick {
    int slot = -1, itemClass = -1, screen = -1;
    VM::EventSystem::LeftClick click;
    bool valid() const {
        return slot >= 0 && slot < 6 && actionForClass(itemClass) >= 0 &&
               screen >= 0 && screen <= 255 && click.valid() &&
               click.handlerClass <= 65534 && click.handlerObject < 200000 &&
               click.message <= 65535;
    }
};

class ActuatorMemory {
public:
    using Key = std::tuple<int, int, int>;
    static Key key(int slot, int cls, int screen) {
        return {slot, cls, screen};
    }
    bool remember(const CalibratedClick &value) {
        if (!value.valid()) return false;
        if (data_.size() >= 48 && !data_.contains(key(value.slot,
                            value.itemClass, value.screen))) return false;
        data_[key(value.slot, value.itemClass, value.screen)] = value;
        return true;
    }
    const CalibratedClick *find(int slot, int cls, int screen) const {
        const auto it = data_.find(key(slot, cls, screen));
        return it == data_.end() ? nullptr : &it->second;
    }
    std::size_t size() const { return data_.size(); }

    static uint64_t checksum(const std::string &s) {
        uint64_t h = UINT64_C(14695981039346656037);
        for (unsigned char ch : s) {
            h ^= ch;
            h *= UINT64_C(1099511628211);
        }
        return h;
    }
    static std::string hex(uint64_t x) {
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << x;
        return out.str();
    }

    bool save(const std::filesystem::path &path, std::string &error) const {
        error.clear();
        std::ostringstream text;
        text << "THIRDEYE_FOOD_ACTUATOR 1\n" << data_.size() << '\n';
        for (const auto &[_, p] : data_) {
            const auto &c = p.click;
            text << p.slot << ' ' << p.itemClass << ' ' << p.screen << ' '
                 << c.x << ' ' << c.y << ' ' << c.x0 << ' ' << c.y0 << ' '
                 << c.x1 << ' ' << c.y1 << ' ' << c.handlerClass << ' '
                 << c.handlerObject << ' ' << c.message << ' ' << c.eventType
                 << ' ' << c.handle << '\n';
        }
        const std::string payload = text.str();
        const auto staging = std::filesystem::path(path.string() + ".tmp");
        {
            std::ofstream out(staging, std::ios::binary | std::ios::trunc);
            if (!out || !(out << payload << "checksum "
                              << hex(checksum(payload)) << '\n')) {
                error = "cannot write actuator staging checkpoint";
                return false;
            }
            out.flush();
            if (!out) { error = "cannot flush actuator checkpoint"; return false; }
        }
#ifdef _WIN32
        if (!MoveFileExW(staging.c_str(), path.c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            error = "atomic actuator checkpoint replacement failed";
            return false;
        }
#else
        std::error_code ec;
        std::filesystem::rename(staging, path, ec);
        if (ec) { error = ec.message(); return false; }
#endif
        return true;
    }

    bool load(const std::filesystem::path &path, std::string &error) {
        error.clear();
        std::error_code ec;
        const auto length = std::filesystem::file_size(path, ec);
        if (ec || length == 0 || length > 16384) {
            error = "actuator checkpoint absent or oversized";
            return false;
        }
        std::ifstream in(path, std::ios::binary);
        if (!in) { error = "cannot open actuator checkpoint"; return false; }
        const std::string raw(std::istreambuf_iterator<char>{in}, {});
        if (in.bad() || raw.size() != length) {
            error = "truncated actuator checkpoint"; return false;
        }
        const auto suffix = raw.rfind("checksum ");
        if (suffix == std::string::npos || suffix == 0 ||
            raw.size() != suffix + 9 + 16 + 1 || raw.back() != '\n' ||
            raw.substr(suffix + 9, 16) != hex(checksum(raw.substr(0, suffix)))) {
            error = "actuator checksum mismatch"; return false;
        }
        std::istringstream data(raw.substr(0, suffix));
        std::string header;
        std::size_t n = 0;
        if (!std::getline(data, header) || header != "THIRDEYE_FOOD_ACTUATOR 1" ||
            !(data >> n) || n > 48) {
            error = "actuator header invalid"; return false;
        }
        std::map<Key, CalibratedClick> staged;
        for (std::size_t i = 0; i < n; ++i) {
            CalibratedClick c;
            if (!(data >> c.slot >> c.itemClass >> c.screen >>
                  c.click.x >> c.click.y >> c.click.x0 >> c.click.y0 >>
                  c.click.x1 >> c.click.y1 >> c.click.handlerClass >>
                  c.click.handlerObject >> c.click.message >> c.click.eventType >> c.click.handle) ||
                !c.valid() || !staged.emplace(
                    key(c.slot, c.itemClass, c.screen), c).second) {
                error = "invalid/duplicate actuator region"; return false;
            }
        }
        std::string trailing;
        if (data >> trailing) { error = "actuator extra data"; return false; }
        data_.swap(staged); // fully validate before mutating live state
        return true;
    }
private:
    std::map<Key, CalibratedClick> data_;
};

} // namespace THIRDEYE::nn::food
#endif
