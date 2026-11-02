#ifndef THIRDEYE_NN_ONLINE_POLICY_HPP
#define THIRDEYE_NN_ONLINE_POLICY_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace THIRDEYE::nn {

// Opt-in online softmax movement policy; pure C++ for deterministic tests.
class OnlinePolicy {
public:
    static constexpr std::size_t kFeatures = 11;
    static constexpr std::size_t kActions = 6;
    using Features = std::array<double, kFeatures>;
    using Weights = std::array<double, kFeatures * kActions>;

    struct Prediction {
        int action = -1;
        double confidence = 0.0;
    };

    // Directions: 0=N, 1=E, 2=S, 3=W. Blocked is relative to
    // the character: front, right, back, left.
    static Features features(int x, int y, int direction,
                             const std::array<bool, 4> &blocked);
    static int actionForKey(int key);
    static int keyForAction(int action);

    Prediction predict(const Features &features) const;
    // Choose only permitted actions. 0..1 exploration rolls make decisions
    // reproducible in tests; the live host supplies its own random samples.
    int select(const Features &features, const std::array<bool, kActions> &allowed,
               double exploreRoll, double choiceRoll) const;
    // Learn from the observed result of OUR action (not the player's keys).
    // Reward is clipped to [-2,2]; success counts actual movement/turns.
    bool reinforce(const Features &features, int action, double reward, bool success);
    // Returns negative on invalid input; otherwise cross-entropy loss.
    double learn(const Features &features, int action);

    bool save(const std::filesystem::path &path, std::string &error) const;
    // Failed loads must never overwrite the running model.
    bool load(const std::filesystem::path &path, std::string &error);

    uint64_t samples() const { return samples_; }
    double meanLoss() const { return meanLoss_; }
    double meanReward() const { return meanReward_; }
    // Derived from the persisted sample count so training resumes seamlessly.
    double learningRate() const {
        return 0.025 + 0.075 / (1.0 + static_cast<double>(samples_) / 250.0);
    }
    uint64_t successes() const { return successes_; }
    uint64_t failures() const { return failures_; }
    double successRate() const {
        const auto total = successes_ + failures_;
        return total ? static_cast<double>(successes_) / static_cast<double>(total) : 0.0;
    }
    const Weights &weights() const { return weights_; }

private:
    static bool validFeatures(const Features &features);
    static std::array<double, kActions> probabilities(const Features &features,
                                                      const Weights &weights);
    Weights weights_{};
    uint64_t samples_ = 0;
    double meanLoss_ = 0.0;
    double meanReward_ = 0.0;
    uint64_t successes_ = 0;
    uint64_t failures_ = 0;
};

} // namespace THIRDEYE::nn
#endif // THIRDEYE_NN_ONLINE_POLICY_HPP
