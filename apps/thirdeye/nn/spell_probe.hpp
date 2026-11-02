#ifndef THIRDEYE_NN_SPELL_PROBE_HPP
#define THIRDEYE_NN_SPELL_PROBE_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace THIRDEYE::nn::spell {

// Mirrors runtime/eye.cpp spell_list's mage/cleric slot offsets. Counts
// prepared copies only; a slot index is NOT a spell class or M:54 dispatch key.
inline int available(const std::array<uint8_t, 200> &counters, int type) {
    if (type < 0 || type > 1) return -1;
    const std::size_t start = type == 0 ? 10u : 110u;
    int sum = 0;
    for (std::size_t i = start; i < start + 90; ++i) {
        const int count = static_cast<int8_t>(counters[i]);
        if (count > 10) return -1; // malformed; don't offer a castable slot
        if (count > 0) sum += count;
    }
    return sum;
}

// Permits only M:29 (native open) followed by M:172 (native close).
// There is deliberately no cast, menu key, or coordinate transition.
class Probe {
public:
    enum class Stage { Idle, Queued, Open, Verified, Unverified, Failed };
    bool begin(int currentScreen, int stock) {
        if (busy() || currentScreen != 0 || stock <= 0) return false;
        state_ = Stage::Queued;
        openScreen_ = -1;
        closedScreen_ = -1;
        return true;
    }
    bool busy() const { return shouldOpen() || shouldClose(); }
    bool shouldOpen() const { return state_ == Stage::Queued; }
    bool shouldClose() const { return state_ == Stage::Open; }
    void opened(int screen) {
        if (shouldOpen()) { openScreen_ = screen; state_ = Stage::Open; }
    }
    void closed(int screen) {
        if (!shouldClose()) return;
        closedScreen_ = screen;
        state_ = openScreen_ > 0 && screen == 0
            ? Stage::Verified : Stage::Unverified;
    }
    void fail() { state_ = Stage::Failed; }
    Stage stage() const { return state_; }
    int openedScreen() const { return openScreen_; }
    int closedScreen() const { return closedScreen_; }
    const char *status() const {
        switch (state_) {
        case Stage::Idle: return "idle";
        case Stage::Queued: return "queued";
        case Stage::Open: return "open";
        case Stage::Verified: return "menu-cycle-verified";
        case Stage::Unverified: return "menu-transition-unverified";
        case Stage::Failed: return "failed";
        }
        return "failed";
    }
private:
    Stage state_ = Stage::Idle;
    int openScreen_ = -1, closedScreen_ = -1;
};

} // namespace THIRDEYE::nn::spell
#endif
