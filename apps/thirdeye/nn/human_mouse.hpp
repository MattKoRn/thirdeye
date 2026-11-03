#ifndef THIRDEYE_NN_HUMAN_MOUSE_HPP
#define THIRDEYE_NN_HUMAN_MOUSE_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <utility>

namespace THIRDEYE::nn::mouse {

// Timed in-game logical cursor motion. Drives the existing native AESOP
// mouseMove/mouseButton path, never the OS-wide cursor or another application.
// UI targets are supplied ONLY after caller-specific live-region verification.
class HumanMouse {
public:
    using Clock = std::chrono::steady_clock;
    enum class Kind { Idle, Move, Press, Release };
    struct Step {
        Kind kind = Kind::Idle;
        int x = -1, y = -1;
    };

    bool start(int fromX, int fromY, int targetX, int targetY,
               Clock::time_point now, uint32_t seed = 0) {
        if (busy() || !onScreen(fromX, fromY) ||
            !onScreen(targetX, targetY)) return false;
        fromX_ = lastX_ = fromX;
        fromY_ = lastY_ = fromY;
        toX_ = targetX; toY_ = targetY;
        const double dx = static_cast<double>(targetX - fromX);
        const double dy = static_cast<double>(targetY - fromY);
        const double distance = std::hypot(dx, dy);
        const int milliseconds = std::clamp(
            250 + static_cast<int>(distance * 2.1) +
            static_cast<int>(seed % 81), 300, 780);
        moveEnd_ = now + std::chrono::milliseconds(milliseconds);
        start_ = now;
        hoverEnd_ = moveEnd_ + std::chrono::milliseconds(95 + seed % 80);
        releaseAt_ = hoverEnd_ + std::chrono::milliseconds(90 + seed % 55);
        curve_ = (seed & 1u ? 1.0 : -1.0) *
            std::min(16.0, distance / 8.0);
        phase_ = Phase::Moving;
        return true;
    }

    Step advance(Clock::time_point now) {
        if (phase_ == Phase::Idle) return {};
        if (phase_ == Phase::Moving) {
            if (now >= moveEnd_) {
                phase_ = Phase::Hover;
                lastX_ = toX_;
                lastY_ = toY_;
                return {Kind::Move, lastX_, lastY_};
            }
            const auto span = std::chrono::duration<double>(moveEnd_ - start_).count();
            const double t = std::clamp(
                std::chrono::duration<double>(now - start_).count() / span,
                0.0, 1.0);
            const double eased = t * t * (3.0 - 2.0 * t);
            const double dx = static_cast<double>(toX_ - fromX_);
            const double dy = static_cast<double>(toY_ - fromY_);
            const double length = std::max(1.0, std::hypot(dx, dy));
            const double bulge = curve_ * std::sin(3.141592653589793 * t);
            lastX_ = std::clamp(static_cast<int>(std::lround(
                fromX_ + dx * eased - dy / length * bulge)), 0, 319);
            lastY_ = std::clamp(static_cast<int>(std::lround(
                fromY_ + dy * eased + dx / length * bulge)), 0, 199);
            return {Kind::Move, lastX_, lastY_};
        }
        if (phase_ == Phase::Hover && now >= hoverEnd_) {
            phase_ = Phase::Pressed;
            return {Kind::Press, toX_, toY_};
        }
        if (phase_ == Phase::Pressed && now >= releaseAt_) {
            phase_ = Phase::Idle;
            return {Kind::Release, toX_, toY_};
        }
        return {};
    }

    // If AUTO is disabled mid-click, caller MUST send native mouse release.
    bool cancel() {
        const bool held = phase_ == Phase::Pressed;
        phase_ = Phase::Idle;
        return held;
    }
    bool busy() const { return phase_ != Phase::Idle; }
    bool buttonHeld() const { return phase_ == Phase::Pressed; }
    int x() const { return lastX_; }
    int y() const { return lastY_; }
    static bool onScreen(int x, int y) {
        return x >= 0 && x < 320 && y >= 0 && y < 200;
    }
private:
    enum class Phase { Idle, Moving, Hover, Pressed };
    Phase phase_ = Phase::Idle;
    int fromX_ = 0, fromY_ = 0, toX_ = 0, toY_ = 0;
    int lastX_ = 0, lastY_ = 0;
    double curve_ = 0;
    Clock::time_point start_{}, moveEnd_{}, hoverEnd_{}, releaseAt_{};
};
} // namespace THIRDEYE::nn::mouse
#endif
