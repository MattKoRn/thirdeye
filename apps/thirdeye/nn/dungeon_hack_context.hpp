#ifndef THIRDEYE_NN_DUNGEON_HACK_CONTEXT_HPP
#define THIRDEYE_NN_DUNGEON_HACK_CONTEXT_HPP

#include "online_mode.hpp"

#include <array>
#include <cstdint>

namespace THIRDEYE::nn::dungeon_hack {

// Native DH runtime draw_walls(x,y,facing,...,lvlmap,...) observation.
// Unlike EOB3, DH has different SOP object classes. Only the verified
// 32x32 tile buffer and party pose passed to the real runtime are used.
struct View {
    int x = -1, y = -1, facing = -1, level = -1;
    std::array<uint8_t, 1024> tiles{};
    bool valid = false;
};

inline bool observe(const View &view, OnlinePolicy::Features &features,
                    std::array<bool, 4> &blocked, mode::Pose *pose = nullptr) {
    if (!view.valid || view.x < 0 || view.x >= 32 ||
        view.y < 0 || view.y >= 32 || view.facing < 0 ||
        view.facing >= 4 || view.level < 0 || view.level > 255)
        return false;
    constexpr std::array<int, 4> dx{0, 1, 0, -1};
    constexpr std::array<int, 4> dy{-1, 0, 1, 0};
    for (int relative = 0; relative < 4; ++relative) {
        const int direction = (view.facing + relative) % 4;
        const int nx = view.x + dx[direction];
        const int ny = view.y + dy[direction];
        // DH's native maze uses 0xFF for walkable passages; all other
        // types, including unknown dynamic features, are conservative.
        blocked[relative] = nx < 0 || nx >= 32 ||
            ny < 0 || ny >= 32 ||
            view.tiles[static_cast<std::size_t>(ny * 32 + nx)] != 0xff;
    }
    features = OnlinePolicy::features(view.x, view.y, view.facing, blocked);
    if (pose) *pose = {view.x, view.y, view.facing, view.level};
    return true;
}

} // namespace THIRDEYE::nn::dungeon_hack
#endif
