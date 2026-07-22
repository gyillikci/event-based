// Acoustic BPF extraction and GCC-PHAT direction-of-arrival for the ReSpeaker
// 4-mic array. Port of AcousticProcessor from detect_drone_fused.py.
//
// The analysis core (analyze / gcc_phat_doa) is dependency-free and operates on
// plain sample buffers, so it is unit-testable without any audio hardware. Live
// capture is provided by start()/stop() when built WITH_PORTAUDIO.
#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace ndrone {

// One window's acoustic analysis result.
struct AcousticResult {
    double freq_hz = 0.0;        // dominant blade-pass peak
    double snr = 0.0;            // peak / median noise floor
    double magnitude = 0.0;
    double doa_azimuth = 0.0;    // smoothed bearing (deg)
    std::optional<double> doa_azimuth_raw;
    std::optional<double> doa_spread_deg;
    bool doa_valid = false;
    std::optional<double> doa_elevation;  // planar array: always empty
};

class AcousticProcessor {
public:
    static constexpr double kSpeedOfSound = 343.0;  // m/s

    AcousticProcessor(int sample_rate = 16000, int chunk_size = 1024, int n_channels = 4,
                      double fft_window_sec = 0.2, double min_freq = 50.0,
                      double max_freq = 500.0, double min_snr = 3.0,
                      double doa_min_confidence = 1.5, int doa_smoothing_window = 7,
                      int device_index = -1);

    // Analyze one full window of de-interleaved channel data. Returns nullopt
    // when there is no confident peak in the target band.
    std::optional<AcousticResult> analyze(const std::vector<std::vector<double>>& channel_data);

    // Thread-safe snapshot of the most recent result (for the live path).
    std::optional<AcousticResult> latest() const;

#ifdef WITH_PORTAUDIO
    // Start/stop the background capture+analysis thread (ReSpeaker array).
    bool start();
    void stop();
#endif

    int sample_rate() const { return sample_rate_; }
    int fft_window_samples() const { return buffer_maxlen_; }

private:
    // Fuse the two opposing mic pairs (0-2 and 1-3, 64 mm baseline).
    std::optional<double> multi_pair_doa_fusion(
        const std::vector<std::vector<double>>& channel_data) const;

    // Frequency-weighted GCC-PHAT with 4x zero-padded TDOA and a
    // peak/second-peak confidence gate. Returns nullopt when low-confidence.
    std::optional<double> gcc_phat_doa(const std::vector<double>& sig1,
                                       const std::vector<double>& sig2,
                                       double mic_dist) const;

    int sample_rate_, chunk_size_, n_channels_;
    double min_freq_, max_freq_, min_snr_;
    double doa_min_confidence_;
    int device_index_;
    int buffer_maxlen_;

    std::deque<double> doa_history_;
    int doa_smoothing_window_;
    std::optional<double> smoothed_doa_;

    mutable std::mutex result_mutex_;
    std::optional<AcousticResult> latest_result_;

#ifdef WITH_PORTAUDIO
    void run();
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::vector<std::deque<double>> buffers_;
#endif
};

}  // namespace ndrone
