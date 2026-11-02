#include "online_policy.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <locale>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace THIRDEYE::nn {
namespace {

uint64_t checksum(const std::string &data) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char c : data) {
        hash ^= c;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::string hexChecksum(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

// Existing good checkpoints are never deleted before replacement.
bool replaceFile(const std::filesystem::path &temp,
                 const std::filesystem::path &dest, std::string &error) {
#ifdef _WIN32
    if (!MoveFileExW(temp.c_str(), dest.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        error = "atomic replacement failed (Windows error " +
                std::to_string(GetLastError()) + ")";
        return false;
    }
#else
    std::error_code ec;
    std::filesystem::rename(temp, dest, ec);
    if (ec) {
        error = "atomic replacement failed: " + ec.message();
        return false;
    }
    auto parent = dest.parent_path();
    if (parent.empty()) parent = ".";
    int fd = ::open(parent.c_str(), O_RDONLY);
    if (fd >= 0) { (void)::fsync(fd); ::close(fd); }
#endif
    return true;
}

} // namespace

bool OnlinePolicy::save(const std::filesystem::path &path, std::string &error) const {
    error.clear();
    if (path.empty() || !std::isfinite(meanLoss_)) {
        error = "invalid checkpoint path or metrics";
        return false;
    }
    std::ostringstream data;
    data.imbue(std::locale::classic());
    data << std::setprecision(17) << "THIRDEYE_NN 1\n"
         << samples_ << ' ' << meanLoss_ << '\n';
    for (double weight : weights_) {
        if (!std::isfinite(weight) || std::abs(weight) > 100.0) {
            error = "invalid model weights";
            return false;
        }
        data << weight << '\n';
    }
    const std::string payload = data.str();
    const std::string bytes =
        payload + "checksum " + hexChecksum(checksum(payload)) + "\n";
    // Unique sibling temp file: rename is atomic on the same filesystem.
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto temp = path;
    temp += ".tmp." + std::to_string(stamp);
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out || !out.write(bytes.data(), static_cast<std::streamsize>(bytes.size())) ||
            !(out.flush())) {
            error = "could not write temporary checkpoint";
            out.close();
            std::error_code ignored;
            std::filesystem::remove(temp, ignored);
            return false;
        }
    }
#ifndef _WIN32
    int fd = ::open(temp.c_str(), O_RDONLY);
    if (fd < 0 || ::fsync(fd) != 0) {
        error = "could not sync temporary checkpoint";
        if (fd >= 0) ::close(fd);
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        return false;
    }
    ::close(fd);
#endif
    if (!replaceFile(temp, path, error)) {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        return false;
    }
    return true;
}

bool OnlinePolicy::load(const std::filesystem::path &path, std::string &error) {
    error.clear();
    std::error_code ec;
    const auto length = std::filesystem::file_size(path, ec);
    if (ec || length > 32768) {
        error = "checkpoint missing or too large";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "could not open checkpoint"; return false; }
    const std::string bytes((std::istreambuf_iterator<char>(in)), {});
    if (in.bad() || bytes.size() != length) {
        error = "could not read complete checkpoint";
        return false;
    }
    const auto marker = bytes.rfind("checksum ");
    if (marker == std::string::npos || marker == 0 ||
        bytes.size() != marker + 9 + 16 + 1 || bytes.back() != '\n') {
        error = "checkpoint truncated or malformed";
        return false;
    }
    const std::string payload = bytes.substr(0, marker);
    if (payload.back() != '\n' ||
        bytes.substr(marker + 9, 16) != hexChecksum(checksum(payload))) {
        error = "checkpoint checksum mismatch";
        return false;
    }
    std::istringstream input(payload);
    input.imbue(std::locale::classic());
    std::string magic;
    int version = 0;
    uint64_t samples = 0;
    double meanLoss = 0.0;
    Weights incoming{};
    if (!(input >> magic >> version) || magic != "THIRDEYE_NN" || version != 1 ||
        !(input >> samples >> meanLoss) || !std::isfinite(meanLoss) ||
        meanLoss < 0.0 || meanLoss > 10000.0) {
        error = "incompatible checkpoint header or metrics";
        return false;
    }
    for (double &w : incoming) {
        if (!(input >> w) || !std::isfinite(w) || std::abs(w) > 100.0) {
            error = "checkpoint has invalid weights";
            return false;
        }
    }
    input >> std::ws;
    if (!input.eof()) {
        error = "checkpoint has unexpected trailing data";
        return false;
    }
    // Commit only after every field has been checked. Inference on the next
    // pump sees these weights, not the stale ones.
    weights_ = incoming;
    samples_ = samples;
    meanLoss_ = meanLoss;
    return true;
}

} // namespace THIRDEYE::nn
