#include "online_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace THIRDEYE::nn {
namespace {
constexpr std::array<int, OnlinePolicy::kActions> kKeys = {
    0x4800, 0x5000, 0x4b00, 0x4d00, 0x4700, 0x4900
};
}

OnlinePolicy::Features OnlinePolicy::features(
    int x, int y, int direction, const std::array<bool, 4> &blocked) {
    Features f{};
    f[0] = 1.0;
    f[1] = std::clamp(x, 0, 31) / 31.0;
    f[2] = std::clamp(y, 0, 31) / 31.0;
    if (direction >= 0 && direction < 4)
        f[3 + direction] = 1.0;
    for (std::size_t i = 0; i < blocked.size(); ++i)
        f[7 + i] = blocked[i] ? 1.0 : 0.0;
    return f;
}

int OnlinePolicy::actionForKey(int key) {
    auto it = std::find(kKeys.begin(), kKeys.end(), key);
    return it == kKeys.end() ? -1 : static_cast<int>(it - kKeys.begin());
}

int OnlinePolicy::keyForAction(int action) {
    return action < 0 || action >= static_cast<int>(kActions) ?
        -1 : kKeys[static_cast<std::size_t>(action)];
}

bool OnlinePolicy::validFeatures(const Features &features) {
    return std::all_of(features.begin(), features.end(), [](double v) {
        return std::isfinite(v) && v >= 0.0 && v <= 1.0;
    });
}

std::array<double, OnlinePolicy::kActions> OnlinePolicy::probabilities(
    const Features &features, const Weights &weights) {
    std::array<double, kActions> scores{};
    for (std::size_t a = 0; a < kActions; ++a)
        for (std::size_t i = 0; i < kFeatures; ++i)
            scores[a] += weights[a * kFeatures + i] * features[i];
    const double maxScore = *std::max_element(scores.begin(), scores.end());
    double sum = 0.0;
    for (double &score : scores) {
        score = std::exp(score - maxScore);
        sum += score;
    }
    for (double &score : scores) score /= sum;
    return scores;
}

OnlinePolicy::Prediction OnlinePolicy::predict(const Features &features) const {
    if (!validFeatures(features)) return {};
    const auto p = probabilities(features, weights_);
    const auto it = std::max_element(p.begin(), p.end());
    return {static_cast<int>(it - p.begin()), *it};
}

double OnlinePolicy::learn(const Features &features, int action) {
    if (action < 0 || action >= static_cast<int>(kActions) ||
        !validFeatures(features) || samples_ == std::numeric_limits<uint64_t>::max())
        return -1.0;
    const auto p = probabilities(features, weights_);
    const double loss = -std::log(std::max(p[action], 1e-300));
    constexpr double rate = 0.12;
    for (std::size_t a = 0; a < kActions; ++a) {
        const double gradient =
            (a == static_cast<std::size_t>(action) ? 1.0 : 0.0) - p[a];
        for (std::size_t i = 0; i < kFeatures; ++i) {
            double &w = weights_[a * kFeatures + i];
            w = std::clamp(w + rate * gradient * features[i], -100.0, 100.0);
        }
    }
    ++samples_;
    meanLoss_ = (samples_ == 1) ? loss : 0.95 * meanLoss_ + 0.05 * loss;
    return loss;
}

} // namespace THIRDEYE::nn
