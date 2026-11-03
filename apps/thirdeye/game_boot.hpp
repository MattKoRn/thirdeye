#ifndef THIRDEYE_GAME_BOOT_HPP
#define THIRDEYE_GAME_BOOT_HPP

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

namespace THIRDEYE::boot {

// Dungeon Hack ships HACK.RES (phase-one / phase-two) and OPEN.RES
// (opening), not the EOB3 start class or INTRO.GFF. The ordinary launcher
// must route both game assets through the actual native VM entrypoint.
inline std::string resourceName(const std::filesystem::path &resource) {
    std::string name = resource.filename().string();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return name;
}

inline bool isDungeonHackResource(const std::filesystem::path &resource) {
    const std::string name = resourceName(resource);
    return name == "hack.res" || name == "open.res";
}

inline std::string initialObject(const std::filesystem::path &resource) {
    const std::string name = resourceName(resource);
    if (name == "hack.res") return "phase-one";
    if (name == "open.res") return "opening";
    return "start";
}

// Only EOB3 uses the separately shipped intro cinematic as its standard
// entry gate. HACK.RES is valid without INTRO.GFF in the same directory.
inline bool needsExternalIntro(const std::filesystem::path &resource) {
    return !isDungeonHackResource(resource);
}

} // namespace THIRDEYE::boot
#endif
