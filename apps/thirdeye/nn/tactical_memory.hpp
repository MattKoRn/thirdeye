#ifndef THIRDEYE_NN_TACTICAL_MEMORY_HPP
#define THIRDEYE_NN_TACTICAL_MEMORY_HPP

#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace THIRDEYE::nn::tactics {

// Tactics are keyed by actual NPC SOP class and equipped item/spell source.
// Spell scroll and wand classes execute their native in-hand use action;
// spellbook/holy-symbol menus are deliberately not guessed.
enum class Kind : uint8_t { Melee = 0, Ranged = 1, ScrollSpell = 2,
                            WandSpell = 3, Unarmed = 4 };
struct Signature {
    uint16_t enemy = 0;
    uint16_t itemClass = 0;
    Kind kind = Kind::Melee;
    uint16_t spellId = 0; // if exported by the item class, else unspecified
    auto operator<=>(const Signature &) const = default;
};
struct Candidate {
    int partySlot = -1;
    Signature signature{};
};
struct Experience {
    uint64_t trials = 0;
    uint64_t hits = 0;
    double damage = 0.0;  // sum of confirmed damage; zero-hits count
};
class Memory {
public:
    // UCB1 exploration, with each unseen tactic tried before exploitation.
    // No action is sent by this pure selector. Candidate list comes only from
    // currently equipped, living party members and a visible enemy.
    // A separately trained combat policy may prefer a party slot when two
    // characters offer the same tactic. Never override an enemy-specific
    // decision to try or use a DIFFERENT weapon/spell.
    int select(const std::vector<Candidate> &candidates, double random,
               int preferredPartySlot = -1) const;
    void learn(const Signature &signature, int observedDamage);
    Experience experience(const Signature &signature) const;
    uint64_t samples() const { return samples_; }
    std::size_t entries() const { return data_.size(); }
    bool save(const std::filesystem::path &path, std::string &error) const;
    bool load(const std::filesystem::path &path, std::string &error);
private:
    std::map<Signature, Experience> data_;
    uint64_t samples_ = 0;
};

const char *kindName(Kind kind);
} // namespace THIRDEYE::nn::tactics
#endif
