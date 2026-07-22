#include "bayesian_fusion.hpp"

#include <algorithm>
#include <cmath>

namespace ndrone {

namespace {
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}  // namespace

FreqMatch cross_validate_frequency(double visual_freq_hz, double acoustic_freq_hz,
                                   double tolerance_ratio, int max_harmonic) {
    for (int n_v = 1; n_v <= max_harmonic; ++n_v) {
        for (int n_a = 1; n_a <= max_harmonic; ++n_a) {
            const double fundamental_v = visual_freq_hz / n_v;
            const double fundamental_a = acoustic_freq_hz / n_a;
            const double ratio =
                std::abs(fundamental_v - fundamental_a) / std::max(fundamental_a, 1e-6);
            if (ratio < tolerance_ratio) {
                FreqMatch m;
                m.matched = true;
                m.confidence = 1.0 / (n_v * n_a);
                m.harmonic_v = n_v;
                m.harmonic_a = n_a;
                return m;
            }
        }
    }
    return FreqMatch{};
}

BayesianDroneDetector::BayesianDroneDetector(double prior, double decay_rate,
                                             double min_freq, double max_freq,
                                             double detection_threshold)
    : p_drone_(prior),
      base_prior_(prior),
      decay_rate_(decay_rate),
      min_freq_(min_freq),
      max_freq_(max_freq),
      detection_threshold_(detection_threshold) {}

std::pair<double, double> BayesianDroneDetector::visual_likelihood(double snr,
                                                                   double freq_hz) const {
    if (!freq_in_range(freq_hz)) return {0.1, 0.3};
    const double p_v_d = sigmoid(snr - 5.0);
    const double p_v_nd = 0.01 * std::exp(-0.5 * snr);
    return {p_v_d, std::max(p_v_nd, 1e-9)};
}

std::pair<double, double> BayesianDroneDetector::acoustic_likelihood(
    double snr, double freq_hz, std::optional<double> elevation_deg) const {
    if (!freq_in_range(freq_hz)) return {0.05, 0.2};
    const double p_a_d = sigmoid(snr - 4.0);
    double p_a_nd = 0.05;
    if (elevation_deg.has_value() && elevation_deg.value() > 10.0) {
        p_a_nd = 0.005;  // few elevated non-drone sources
    }
    return {p_a_d, std::max(p_a_nd, 1e-9)};
}

double BayesianDroneDetector::update(const std::optional<VisualEvidence>& visual,
                                     const std::optional<AcousticEvidence>& acoustic) {
    double p_d = p_drone_;
    double p_nd = 1.0 - p_d;

    if (visual.has_value()) {
        auto [pv_d, pv_nd] = visual_likelihood(visual->snr, visual->freq_hz);
        const double denom = pv_d * p_d + pv_nd * p_nd;
        p_d = pv_d * p_d / denom;
        p_nd = 1.0 - p_d;
    }

    if (acoustic.has_value()) {
        auto [pa_d, pa_nd] =
            acoustic_likelihood(acoustic->snr, acoustic->freq_hz, acoustic->elevation_deg);
        const double denom = pa_d * p_d + pa_nd * p_nd;
        p_d = pa_d * p_d / denom;
        p_nd = 1.0 - p_d;
    }

    // Cross-modal frequency-match bonus.
    if (visual.has_value() && acoustic.has_value()) {
        FreqMatch m = cross_validate_frequency(visual->freq_hz, acoustic->freq_hz);
        if (m.matched) {
            const double pf_d = 0.99 * m.confidence;
            const double pf_nd = 0.001;
            const double denom = pf_d * p_d + pf_nd * p_nd;
            p_d = pf_d * p_d / denom;
        }
    }

    p_drone_ = std::clamp(p_d, 1e-9, 1.0 - 1e-9);
    return p_drone_;
}

void BayesianDroneDetector::decay() {
    p_drone_ = decay_rate_ * p_drone_ + (1.0 - decay_rate_) * base_prior_;
}

}  // namespace ndrone
