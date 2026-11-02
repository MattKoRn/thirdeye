#include "tactical_memory.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace THIRDEYE::nn::tactics {
namespace {
constexpr std::size_t kMaxEntries = 8192;
constexpr uint64_t kMaxTrials = UINT64_C(1000000000);

uint64_t checksum(const std::string &s) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (unsigned char c : s) {
        h ^= c;
        h *= UINT64_C(1099511628211);
    }
    return h;
}

bool parseEntry(const std::string &line, Signature &sig, Experience &e) {
    std::istringstream in(line);
    int enemy = -1, item = -1, kind = -1, spellId = -1;
    std::string trailing;
    if (!(in >> enemy >> item >> kind >> spellId >> e.trials >> e.hits >> e.damage) ||
        (in >> trailing) || enemy < 0 || enemy > 65535 ||
        item < 0 || item > 65535 || kind < 0 || kind > 4 ||
        spellId < 0 || spellId > 65535 ||
        e.trials == 0 || e.trials > kMaxTrials || e.hits > e.trials ||
        !std::isfinite(e.damage) || e.damage < 0.0 ||
        e.damage > static_cast<double>(e.trials) * 32767.0)
        return false;
    sig = {static_cast<uint16_t>(enemy), static_cast<uint16_t>(item),
           static_cast<Kind>(kind), static_cast<uint16_t>(spellId)};
    return true;
}
} // namespace

const char *kindName(Kind kind) {
    switch (kind) {
    case Kind::Melee: return "weapon";
    case Kind::Ranged: return "ranged";
    case Kind::ScrollSpell: return "scroll-spell";
    case Kind::WandSpell: return "wand-spell";
    case Kind::Unarmed: return "unarmed";
    }
    return "unknown";
}

Experience Memory::experience(const Signature &s) const {
    const auto it = data_.find(s);
    return it == data_.end() ? Experience{} : it->second;
}

int Memory::select(const std::vector<Candidate> &candidates, double random) const {
    if (candidates.empty() || !std::isfinite(random) ||
        random < 0.0 || random > 1.0)
        return -1;
    // Exhaust unknown tactics first, deliberately testing distinct weapon and
    // magical-item capabilities before ranking them by confirmed damage.
    std::vector<int> untried;
    for (std::size_t i = 0; i < candidates.size(); ++i)
        if (experience(candidates[i].signature).trials == 0)
            untried.push_back(static_cast<int>(i));
    const auto pick = [&](std::size_t size) {
        return std::min(size - 1, static_cast<std::size_t>(random * size));
    };
    if (!untried.empty()) return untried[pick(untried.size())];

    uint64_t total = 0;
    for (const auto &candidate : candidates)
        total += experience(candidate.signature).trials;
    const double exploration = std::log(2.0 + static_cast<double>(total));
    int best = -1;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto e = experience(candidates[i].signature);
        const double meanDamage = e.damage / static_cast<double>(e.trials);
        const double bonus = 2.0 * std::sqrt(exploration / e.trials);
        const double score = meanDamage + bonus;
        if (score > bestScore) {
            bestScore = score;
            best = static_cast<int>(i);
        }
    }
    return best;
}

void Memory::learn(const Signature &sig, int observedDamage) {
    if (sig.enemy == 0 || observedDamage < 0 || observedDamage > 32767)
        return;
    auto it = data_.find(sig);
    if (it == data_.end()) {
        if (data_.size() >= kMaxEntries) return;
        it = data_.emplace(sig, Experience{}).first;
    }
    if (it->second.trials >= kMaxTrials || samples_ >= kMaxTrials) return;
    ++it->second.trials;
    if (observedDamage > 0) ++it->second.hits;
    it->second.damage += observedDamage;
    ++samples_;
}

bool Memory::save(const std::filesystem::path &path, std::string &error) const {
    std::ostringstream content;
    content << "THIRDEYE_TACTICS 1\n" << samples_ << ' ' << data_.size() << '\n';
    content << std::setprecision(17);
    for (const auto &[s, e] : data_)
        content << s.enemy << ' ' << s.itemClass << ' '
                << static_cast<int>(s.kind) << ' ' << s.spellId << ' ' << e.trials << ' '
                << e.hits << ' ' << e.damage << '\n';
    const std::string payload = content.str();
    content << "checksum " << std::hex << std::setw(16) << std::setfill('0')
            << checksum(payload) << '\n';

    std::filesystem::path staging = path;
    staging += ".tmp";
    {
        std::ofstream out(staging, std::ios::binary | std::ios::trunc);
        if (!out || !(out << content.str())) {
            error = "cannot write tactical checkpoint";
            return false;
        }
        out.flush();
        if (!out) {
            error = "cannot flush tactical checkpoint";
            return false;
        }
    }
#ifdef _WIN32
    if (!MoveFileExW(staging.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        error = "atomic tactical checkpoint replace failed";
        return false;
    }
#else
    std::error_code ec;
    std::filesystem::rename(staging, path, ec);
    if (ec) {
        error = ec.message();
        return false;
    }
#endif
    return true;
}

bool Memory::load(const std::filesystem::path &path, std::string &error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open tactical checkpoint"; return false; }
    const std::string raw(std::istreambuf_iterator<char>{in},
                          std::istreambuf_iterator<char>{});
    const auto end = raw.rfind("checksum ");
    if (end == std::string::npos || end == 0) {
        error = "missing checksum";
        return false;
    }
    const std::string payload = raw.substr(0, end);
    const std::string suffix = raw.substr(end);
    std::istringstream verifier(suffix);
    std::string marker, hex, extra;
    if (!(verifier >> marker >> hex) || marker != "checksum" ||
        (verifier >> extra) || hex.size() != 16) {
        error = "invalid checksum record";
        return false;
    }
    uint64_t expected = 0;
    std::istringstream number(hex);
    number >> std::hex >> expected;
    if (!number || expected != checksum(payload)) {
        error = "tactical checksum mismatch";
        return false;
    }
    std::istringstream input(payload);
    std::string header, counts;
    if (!std::getline(input, header) || header != "THIRDEYE_TACTICS 1" ||
        !std::getline(input, counts)) {
        error = "invalid tactical checkpoint version";
        return false;
    }
    uint64_t samples = 0;
    std::size_t count = 0;
    {
        std::istringstream numbers(counts);
        if (!(numbers >> samples >> count) || (numbers >> extra) ||
            count > kMaxEntries || samples > kMaxTrials) {
            error = "invalid tactical entry count";
            return false;
        }
    }
    std::map<Signature, Experience> loaded;
    uint64_t total = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::string line;
        Signature sig;
        Experience exp;
        if (!std::getline(input, line) || !parseEntry(line, sig, exp) ||
            !loaded.emplace(sig, exp).second) {
            error = "invalid or duplicate tactical record";
            return false;
        }
        total += exp.trials;
    }
    if (total != samples || (input >> extra)) {
        error = "tactical sample count mismatch or trailing data";
        return false;
    }
    data_.swap(loaded); // validate everything before mutating live memory
    samples_ = samples;
    return true;
}
} // namespace THIRDEYE::nn::tactics
