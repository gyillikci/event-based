// Sequential Bayesian sensor fusion and harmonic-aware frequency
// cross-validation. Port of BayesianDroneDetector and cross_validate_frequency
// from detect_drone_fused.py.
#pragma once

#include <optional>
#include <tuple>

namespace ndrone {

// Result of harmonic cross-validation between the visual and acoustic BPF.
struct FreqMatch {
    bool matched = false;
    double confidence = 0.0;
    int harmonic_v = 0;
    int harmonic_a = 0;
};

// Check whether the visual and acoustic frequencies are consistent, allowing
// one sensor to have detected a harmonic of the fundamental (up to
// max_harmonic). Returns confidence 1/(n_v*n_a) for the matching pair.
FreqMatch cross_validate_frequency(double visual_freq_hz, double acoustic_freq_hz,
                                   double tolerance_ratio = 0.05,
                                   int max_harmonic = 4);

// Visual evidence for a Bayesian update.
struct VisualEvidence {
    double snr = 0.0;
    double freq_hz = 0.0;
};

// Acoustic evidence for a Bayesian update. elevation is optional (planar array
// cannot resolve it, so it is usually absent).
struct AcousticEvidence {
    double snr = 0.0;
    double freq_hz = 0.0;
    std::optional<double> elevation_deg;
};

// Sequential Bayesian fusion for real-time drone detection. Updates P(drone)
// as visual and acoustic evidence arrive, with a cross-modal frequency-match
// bonus and belief decay toward the prior when no evidence is seen.
class BayesianDroneDetector {
public:
    BayesianDroneDetector(double prior = 0.001, double decay_rate = 0.995,
                          double min_freq = 50.0, double max_freq = 500.0,
                          double detection_threshold = 0.8);

    // Update with optionally-present visual and/or acoustic evidence. Returns
    // the updated P(drone).
    double update(const std::optional<VisualEvidence>& visual,
                  const std::optional<AcousticEvidence>& acoustic);

    // Drift toward the prior when no evidence arrives this frame.
    void decay();

    bool detected() const { return p_drone_ > detection_threshold_; }
    double confidence() const { return p_drone_; }

private:
    bool freq_in_range(double f) const { return f >= min_freq_ && f <= max_freq_; }
    std::pair<double, double> visual_likelihood(double snr, double freq_hz) const;
    std::pair<double, double> acoustic_likelihood(double snr, double freq_hz,
                                                  std::optional<double> elevation_deg) const;

    double p_drone_;
    double base_prior_;
    double decay_rate_;
    double min_freq_, max_freq_;
    double detection_threshold_;
};

}  // namespace ndrone
