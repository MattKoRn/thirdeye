// Standalone ARJ extractor for scripts/install.py. Reuses the launcher's
// legacy ARJ decoder without pulling Qt into the installer build.
#include "../apps/launcher/unarj.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <system_error>
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

// ARJ file names can contain DOS or Unix directory separators (Dungeon
// Hack's DATA1.ARJ has SAVEGAME/PC.DAT). Validate *each* component before
// joining it to the extraction root. In particular, never accept drive
// prefixes, absolute paths, empty/dot components or traversal.
bool safeRelativePath(const std::string& name, fs::path& relative) {
    relative.clear();
    std::string component;
    for (char ch : name) {
        if (ch == '/' || ch == '\\') {
            if (!safeBasename(component))
                return false;
            relative /= component;
            component.clear();
        } else {
            component += ch;
        }
    }
    if (!safeBasename(component))
        return false;
    relative /= component;
    return true;
}

void rejectSymlink(const fs::path& path) {
    std::error_code ec;
    const fs::file_status status = fs::symlink_status(path, ec);
    if (ec && ec != std::errc::no_such_file_or_directory)
        throw fs::filesystem_error("Cannot inspect ARJ output path", path, ec);
    if (fs::is_symlink(status))
        throw std::runtime_error("Unsafe ARJ output symlink: " + path.string());
}

void rejectSymlinkChain(const fs::path& root, const fs::path& relative) {
    fs::path current = root;
    rejectSymlink(current);
    for (const auto& part : relative) {
        current /= part;
        rejectSymlink(current);
    }
}

fs::path outputSubdir(const std::string& game, int volume, char** argv) {
    if (game == "eob3") {
        const std::string stem =
            upperAscii(fs::path(argv[3 + volume]).stem().string());
        if (stem == "DATA5")
            return "SAVEGAME";
        if (stem == "DATA6")
            return "CHARGEN";
    }
    return {};
}

struct OutputEntry {
    const ArjEntry* file;
    fs::path relative;
};

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

        // Validate every member before writing anything: allow DOS-style
        // relative subdirectories, but reject traversal, unsafe components,
        // case-insensitive collisions, and file/directory conflicts.
        std::set<std::string> names;
        std::vector<OutputEntry> entries;
        entries.reserve(files.size());
        for (const ArjEntry& file : files) {
            fs::path member;
            if (!safeRelativePath(file.name, member))
                throw std::runtime_error("Unsafe ARJ member name: " + file.name);
            if (file.volume < 0 || file.volume + 3 >= argc)
                throw std::runtime_error("Invalid ARJ volume index");

            const fs::path relative =
                outputSubdir(game, file.volume, argv) / member;
            if (!names.insert(upperAscii(relative.generic_string())).second)
                throw std::runtime_error("Duplicate ARJ member: " + relative.string());
            entries.push_back({&file, relative});
        }
        for (const OutputEntry& entry : entries) {
            for (fs::path parent = entry.relative.parent_path(); !parent.empty();
                 parent = parent.parent_path()) {
                if (names.count(upperAscii(parent.generic_string())))
                    throw std::runtime_error("ARJ file/directory conflict: "
                                             + entry.relative.string());
            }
            // A preexisting link could redirect an otherwise safe relative path
            // outside the extraction tree. Check before creating any output.
            rejectSymlinkChain(destination, entry.relative);
        }

        fs::create_directories(destination);
        for (const OutputEntry& entry : entries) {
            const fs::path target = destination / entry.relative;
            rejectSymlinkChain(destination, entry.relative);
            fs::create_directories(target.parent_path());
            rejectSymlinkChain(destination, entry.relative);
            std::ofstream out(target, std::ios::binary | std::ios::trunc);
            if (!out || !out.write(
                            reinterpret_cast<const char*>(entry.file->data.data()),
                            static_cast<std::streamsize>(entry.file->data.size())))
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
