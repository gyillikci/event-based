#include "synthetic.hpp"

#include <cmath>

namespace ndrone {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kSpeedOfSound = 343.0;
}  // namespace

SyntheticFrequencyMapSource::SyntheticFrequencyMapSource(int width, int height,
                                                         std::vector<Prop> props,
                                                         double noise_pixels_frac, unsigned seed)
    : width_(width),
      height_(height),
      props_(std::move(props)),
      noise_frac_(noise_pixels_frac),
      rng_(seed) {}

int64_t SyntheticFrequencyMapSource::next(cv::Mat& freq_map, int delta_t_us) {
    if (freq_map.rows != height_ || freq_map.cols != width_ || freq_map.type() != CV_32F)
        freq_map.create(height_, width_, CV_32F);
    freq_map.setTo(0.0f);

    constexpr int kWarmupFrames = 2;
    std::uniform_real_distribution<float> jitter(-1.5f, 1.5f);

    if (frame_ >= kWarmupFrames) {
        for (const Prop& p : props_) {
            for (int dy = -p.radius; dy <= p.radius; ++dy) {
                for (int dx = -p.radius; dx <= p.radius; ++dx) {
                    if (dx * dx + dy * dy > p.radius * p.radius) continue;
                    const int x = p.cx + dx, y = p.cy + dy;
                    if (x < 0 || x >= width_ || y < 0 || y >= height_) continue;
                    // Small per-pixel jitter so freq_cv is realistic (non-zero).
                    freq_map.at<float>(y, x) = p.freq_hz + jitter(rng_);
                }
            }
        }
    }

    // Sprinkle single-pixel noise at random frequencies (the transient
    // false-positive source the tracker must reject).
    const int n_noise = static_cast<int>(noise_frac_ * width_ * height_);
    std::uniform_int_distribution<int> xd(0, width_ - 1), yd(0, height_ - 1);
    std::uniform_real_distribution<float> fd(10.0f, 300.0f);
    for (int i = 0; i < n_noise; ++i) freq_map.at<float>(yd(rng_), xd(rng_)) = fd(rng_);

    ++frame_;
    ts_us_ += delta_t_us;
    return ts_us_;
}

void SyntheticShake::at(double t, double& cx, double& cy) const {
    cx = cx0 + amp_px * std::sin(2.0 * kPi * freq_hz * t);
    cy = cy0 + 0.5 * amp_px * std::sin(2.0 * kPi * freq_hz * t + 1.0);
}

std::vector<std::vector<double>> synth_tone_array(int n_samples, int sample_rate,
                                                  double tone_hz, double azimuth_deg,
                                                  double amplitude, double noise,
                                                  unsigned seed) {
    // ReSpeaker opposing-pair geometry: pair 0-2 on the x-axis, 1-3 on y-axis,
    // 64 mm apart. Delay across a pair for a plane wave from `azimuth_deg`.
    const double az = azimuth_deg * kPi / 180.0;
    const double baseline = 0.064;
    const double tdoa_x = (baseline * std::sin(az)) / kSpeedOfSound;  // sec, mic0 vs mic2
    const double tdoa_y = 0.0;  // broadside for the y pair in this simple model

    const double ch_delay[4] = {+tdoa_x / 2.0, +tdoa_y / 2.0, -tdoa_x / 2.0, -tdoa_y / 2.0};

    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, noise);

    std::vector<std::vector<double>> ch(4, std::vector<double>(n_samples));
    for (int c = 0; c < 4; ++c) {
        for (int i = 0; i < n_samples; ++i) {
            const double t = static_cast<double>(i) / sample_rate - ch_delay[c];
            ch[c][i] = amplitude * std::sin(2.0 * kPi * tone_hz * t) + nd(rng);
        }
    }
    return ch;
}

}  // namespace ndrone
