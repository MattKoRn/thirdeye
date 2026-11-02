// Standalone ARJ extractor for scripts/install.py. Reuses the launcher's
// legacy ARJ decoder without pulling Qt into the installer build.
#include "../apps/launcher/unarj.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr std::uint64_t MAX_ARCHIVE_BYTES = 2ULL * 1024 * 1024 * 1024;

std::string upperAscii(std::string s) {
    for (char& ch : s)
        if (ch >= 'a' && ch <= 'z')
            ch = char(ch - 'a' + 'A');
    return s;
}

bool safeBasename(const std::string& name) {
    if (name.empty() || name == "." || name == ".."
        || name.back() == '.' || name.back() == ' '
        || name.find("..") != std::string::npos)
        return false;
    // Accept the DOS names these archives use; reject path separators,
    // drive prefixes, NTFS alternate data streams, and non-ASCII controls.
    for (unsigned char ch : name)
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')
              || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-'
              || ch == '.' || ch == ' '))
            return false;
    const std::string stem = upperAscii(name.substr(0, name.find('.')));
    return stem != "CON" && stem != "PRN" && stem != "AUX"
        && stem != "NUL" && stem != "COM1" && stem != "COM2"
        && stem != "COM3" && stem != "COM4" && stem != "COM5"
        && stem != "COM6" && stem != "COM7" && stem != "COM8"
        && stem != "COM9" && stem != "LPT1" && stem != "LPT2"
        && stem != "LPT3" && stem != "LPT4" && stem != "LPT5"
        && stem != "LPT6" && stem != "LPT7" && stem != "LPT8"
        && stem != "LPT9";
}

std::vector<std::uint8_t> readVolume(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open ARJ volume: " + path.string());
    const auto size = file.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > MAX_ARCHIVE_BYTES)
        throw std::runtime_error("ARJ volume too large: " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!bytes.empty() && !file.read(reinterpret_cast<char*>(bytes.data()), size))
        throw std::runtime_error("Cannot read entire ARJ volume: " + path.string());
    return bytes;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: thirdeye-unarj OUTPUT_DIR GAME DATA1.ARJ [DATA2.ARJ ...]\n"
                     "GAME is eob3 or hack\n";
        return 2;
    }
    try {
        const fs::path destination = fs::path(argv[1]);
        const std::string game = argv[2];
        if (game != "eob3" && game != "hack")
            throw std::runtime_error("Unsupported game identifier: " + game);

        std::vector<std::vector<std::uint8_t>> volumes;
        std::uint64_t total = 0;
        for (int i = 3; i < argc; ++i) {
            auto vol = readVolume(fs::path(argv[i]));
            if (vol.size() > MAX_ARCHIVE_BYTES - total)
                throw std::runtime_error("Combined ARJ archive size exceeds limit");
            total += vol.size();
            volumes.push_back(std::move(vol));
        }

        std::vector<ArjEntry> files;
        std::string err;
        if (!unarjExtract(volumes, files, err))
            throw std::runtime_error("ARJ extraction failed: " + err);

        // Reject all unsafe paths before writing any files, including
        // case-insensitive collisions on Windows.
        std::set<std::string> names;
        for (const ArjEntry& file : files) {
            if (!safeBasename(file.name))
                throw std::runtime_error("Unsafe ARJ member name: " + file.name);
            if (file.volume < 0 || file.volume + 3 >= argc)
                throw std::runtime_error("Invalid ARJ volume index");

            fs::path subdir;
            if (game == "eob3") {
                const std::string sourceStem =
                    upperAscii(fs::path(argv[3 + file.volume]).stem().string());
                if (sourceStem == "DATA5")
                    subdir = "SAVEGAME";
                else if (sourceStem == "DATA6")
                    subdir = "CHARGEN";
            }
            const fs::path rel = subdir / file.name;
            if (!names.insert(upperAscii(rel.generic_string())).second)
                throw std::runtime_error("Duplicate ARJ member: " + rel.string());
        }

        fs::create_directories(destination);
        for (const ArjEntry& file : files) {
            fs::path parent = destination;
            if (game == "eob3") {
                const std::string sourceStem =
                    upperAscii(fs::path(argv[3 + file.volume]).stem().string());
                if (sourceStem == "DATA5")
                    parent /= "SAVEGAME";
                else if (sourceStem == "DATA6")
                    parent /= "CHARGEN";
            }
            fs::create_directories(parent);
            const fs::path target = parent / fs::path(file.name);
            std::ofstream out(target, std::ios::binary | std::ios::trunc);
            if (!out || !out.write(reinterpret_cast<const char*>(file.data.data()),
                                   static_cast<std::streamsize>(file.data.size())))
                throw std::runtime_error("Cannot write: " + target.string());
        }
        std::cout << "Extracted " << files.size()
                  << " ARJ members (all pieces CRC-checked)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "thirdeye-unarj: " << e.what() << '\n';
        return 1;
    }
}
