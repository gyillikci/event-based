// Synthetic signal sources used when the Metavision SDK is not available
// (e.g. this Linux build) and for tests/demos. These mirror the repo's
// vibration_pattern_generator.py role: exercise the full pipeline end-to-end
// without live hardware.
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace ndrone {

// Renders a per-pixel frequency map (CV_32F, Hz) containing one or more
// rotating "propeller" disks, plus sprinkled single-pixel noise. This is the
// same kind of map FrequencyMapAsyncAlgorithm produces, so the downstream
// analyzer/tracker code is identical whether the source is real or synthetic.
class SyntheticFrequencyMapSource {
public:
    struct Prop {
        int cx, cy, radius;
        float freq_hz;
    };

    SyntheticFrequencyMapSource(int width, int height, std::vector<Prop> props,
                                double noise_pixels_frac = 0.0002, unsigned seed = 1);

    // Advance by delta_t microseconds and fill `freq_map`. Returns the current
    // timestamp (us). Propellers appear only after `warmup_frames` so the
    // detector's confirmation dynamics are visible.
    int64_t next(cv::Mat& freq_map, int delta_t_us = 50000);

private:
    int width_, height_;
    std::vector<Prop> props_;
    double noise_frac_;
    int64_t ts_us_ = 0;
    int frame_ = 0;
    std::mt19937 rng_;
};

// Generates a shaking-centroid signal: a global oscillation at a set frequency
// and amplitude, for exercising the hand-shake FFT analyzer.
struct SyntheticShake {
    double freq_hz = 5.0;
    double amp_px = 8.0;
    double cx0 = 640.0, cy0 = 360.0;
    // Centroid at time t (seconds).
    void at(double t, double& cx, double& cy) const;
};

// Generates n_channels of a synthetic tone at `tone_hz` arriving from a given
// azimuth (deg), with per-channel time delays consistent with the ReSpeaker
// 64 mm opposing-pair geometry. Used to drive the acoustic analyzer in tests.
std::vector<std::vector<double>> synth_tone_array(int n_samples, int sample_rate,
                                                  double tone_hz, double azimuth_deg,
                                                  double amplitude = 3000.0,
                                                  double noise = 50.0, unsigned seed = 7);

}  // namespace ndrone
