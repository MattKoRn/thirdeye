#ifndef THIRDEYE_NN_ROLLING_SAVE_HPP
#define THIRDEYE_NN_ROLLING_SAVE_HPP

#include <chrono>
#include <optional>

namespace RESOURCES { class Resource; }
namespace VM { class ObjectSystem; }
namespace THIRDEYE::runtime { struct Context; }

namespace THIRDEYE::nn::rolling {

// Reserved native EOB3 save slots: outside the game's 0..11 picker, so NN
// rotation never overwrites the player's ordinary in-game saves.
inline constexpr int kFirstSlot = 12;
inline constexpr int kSaveCount = 3;
inline constexpr int slotForIndex(int index) { return kFirstSlot + index; }
inline constexpr int indexForSlot(int slot) { return slot - kFirstSlot; }
inline constexpr const char *nameForIndex(int index) {
    return index == 0 ? "nn-1" : index == 1 ? "nn-2" :
           index == 2 ? "nn-3" : "";
}

// Pure clock for regression testing. Save time is measured by monotonic wall
// time while neural play is active, NOT by host frames. UI screens defer saves,
// and death only starts a restore timer once an actual all-dead party is seen.
class Schedule {
public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;
    static constexpr auto kSaveEvery = std::chrono::minutes(1);
    static constexpr auto kRestoreAfter = std::chrono::seconds(5);
    enum class Party { Unknown, Alive, Dead };

    void enable(Time now) {
        enabled_ = true;
        nextSave_ = now + kSaveEvery;
        deadSince_.reset();
    }
    void disable() {
        enabled_ = false;
        deadSince_.reset();
    }
    void afterRestore(Time now) { enable(now); }
    bool enabled() const { return enabled_; }
    bool shouldSave(Time now, bool gameplay, Party party) {
        if (!enabled_ || !gameplay || party != Party::Alive ||
            deadSince_ || now < nextSave_) return false;
        nextSave_ = now + kSaveEvery; // no catch-up burst after pause
        return true;
    }
    bool shouldRestore(Time now, Party party) {
        if (!enabled_) return false;
        if (party == Party::Alive) {
            deadSince_.reset();
            return false;
        }
        if (party == Party::Dead && !deadSince_) deadSince_ = now;
        return deadSince_ && now - *deadSince_ >= kRestoreAfter;
    }
    bool gameOverPending() const { return deadSince_.has_value(); }
private:
    bool enabled_ = false;
    Time nextSave_ = Time::max();
    std::optional<Time> deadSince_;
};

Schedule::Party partyState(VM::ObjectSystem &objects);
// Called in pumpHost while the live runtime context is available.
// May throw runtime::NeuralRecovery after five seconds of observed game-over.
void tick(THIRDEYE::runtime::Context &ctx, bool autonomy, bool gameplay);
// After the boot handler unwinds, validate then copy native save files to the
// game's live TMP files and re-enter the normal EOB3 resume_level boot.
bool restore(int slot, RESOURCES::Resource &resource);
void afterRestore();
void reset();

} // namespace THIRDEYE::nn::rolling
#endif
