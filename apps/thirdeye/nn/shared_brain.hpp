#ifndef THIRDEYE_NN_SHARED_BRAIN_HPP
#define THIRDEYE_NN_SHARED_BRAIN_HPP

#include "online_mode.hpp"

#include <array>
#include <filesystem>
#include <string>

namespace THIRDEYE::nn::mode {

// Restores a shared checkpoint, or migrates the two independent legacy brains
// once into brain-shared.ckpt. Never writes to either legacy file. If the
// shared checkpoint is corrupt, refuses to replace it with an empty model.
// This is deliberately independent of SDL/VM to allow disk-based tests.
struct SharedBrainRestore {
    bool ready = true;    // false: do not overwrite an existing bad checkpoint
    bool restored = false;
    bool migrated = false;
    std::string error;
    std::string warning;
};

inline SharedBrainRestore restoreSharedBrain(
    OnlinePolicy &brain, const std::filesystem::path &documents) {
    namespace fs = std::filesystem;
    SharedBrainRestore result;
    const fs::path shared = brainFileForDocuments(documents);
    std::error_code ec;
    const bool sharedExists = fs::exists(shared, ec);
    if (ec) {
        result.ready = false;
        result.error = "cannot inspect shared brain: " + ec.message();
        return result;
    }
    if (sharedExists) {
        if (!brain.load(shared, result.error)) {
            result.ready = false;
            return result;
        }
        result.restored = true;
        return result;
    }

    // Migrate before saving anything. If both games have been played, blend
    // their trained policies weighted by the number of learning samples so
    // neither old game-specific brain is silently discarded.
    const std::array<fs::path, 2> oldFiles{
        legacyBrainFileForDocuments(documents, false),
        legacyBrainFileForDocuments(documents, true)
    };
    OnlinePolicy imported;
    for (const fs::path &file : oldFiles) {
        ec.clear();
        const bool exists = fs::exists(file, ec);
        if (ec) {
            result.warning += "Cannot inspect " + file.string() +
                              ": " + ec.message() + "; ";
            continue;
        }
        if (!exists) continue;
        OnlinePolicy old;
        std::string error;
        if (!old.load(file, error)) {
            result.warning += "Legacy brain rejected " + file.string() +
                              " (" + error + "); ";
            continue;
        }
        if (!imported.mergeFrom(old)) {
            result.ready = false;
            result.error = "combined legacy brain learning count overflow";
            return result;
        }
        result.migrated = true;
    }
    if (!result.migrated) return result;
    brain = imported;
    if (!brain.save(shared, result.error)) {
        result.ready = false;
        return result;
    }
    result.restored = true;
    return result;
}

} // namespace THIRDEYE::nn::mode
#endif
