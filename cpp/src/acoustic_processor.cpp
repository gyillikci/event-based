#include "acoustic_processor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "fft.hpp"

#ifdef WITH_PORTAUDIO
#include <portaudio.h>
#include <cstdint>
#include <cstdio>
#endif

namespace ndrone {

AcousticProcessor::AcousticProcessor(int sample_rate, int chunk_size, int n_channels,
                                     double fft_window_sec, double min_freq, double max_freq,
                                     double min_snr, double doa_min_confidence,
                                     int doa_smoothing_window, int device_index)
    : sample_rate_(sample_rate),
      chunk_size_(chunk_size),
      n_channels_(n_channels),
      min_freq_(min_freq),
      max_freq_(max_freq),
      min_snr_(min_snr),
      doa_min_confidence_(doa_min_confidence),
      device_index_(device_index),
      buffer_maxlen_(static_cast<int>(fft_window_sec * sample_rate)),
      doa_smoothing_window_(std::max(1, doa_smoothing_window)) {
#ifdef WITH_PORTAUDIO
    buffers_.resize(n_channels_);
#endif
}

std::optional<AcousticResult> AcousticProcessor::latest() const {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return latest_result_;
}

std::optional<AcousticResult> AcousticProcessor::analyze(
    const std::vector<std::vector<double>>& channel_data) {
    const int n = channel_data.empty() ? 0 : static_cast<int>(channel_data[0].size());
    if (n < 64) return std::nullopt;

    // Channel-averaged magnitude spectrum for frequency detection.
    const std::vector<double> window = hanning(n);
    std::vector<double> avg_spectrum;
    for (int ch = 0; ch < n_channels_; ++ch) {
        const std::vector<double>& sig = channel_data[ch];
        const double mean = std::accumulate(sig.begin(), sig.end(), 0.0) / n;
        std::vector<double> windowed(n);
        for (int i = 0; i < n; ++i) windowed[i] = (sig[i] - mean) * window[i];
        std::vector<Complex> spec = rfft(windowed);
        if (avg_spectrum.empty()) avg_spectrum.assign(spec.size(), 0.0);
        for (size_t i = 0; i < spec.size(); ++i) avg_spectrum[i] += std::abs(spec[i]);
    }
    const double scale = 2.0 / (static_cast<double>(n) * n_channels_);
    for (double& v : avg_spectrum) v *= scale;

    const std::vector<double> freqs = rfftfreq(n, 1.0 / sample_rate_);

    // Peak in target band.
    double peak_mag = -1.0, peak_freq = 0.0;
    std::vector<double> valid_mags;
    for (size_t i = 0; i < freqs.size(); ++i) {
        if (freqs[i] >= min_freq_ && freqs[i] <= max_freq_) {
            valid_mags.push_back(avg_spectrum[i]);
            if (avg_spectrum[i] > peak_mag) {
                peak_mag = avg_spectrum[i];
                peak_freq = freqs[i];
            }
        }
    }
    if (valid_mags.empty()) return std::nullopt;

    std::vector<double> sorted_mags = valid_mags;
    std::sort(sorted_mags.begin(), sorted_mags.end());
    const size_t m = sorted_mags.size();
    const double noise_floor =
        (m % 2 == 1) ? sorted_mags[m / 2]
                     : 0.5 * (sorted_mags[m / 2 - 1] + sorted_mags[m / 2]);
    const double snr = noise_floor > 0.0 ? peak_mag / noise_floor : peak_mag;
    if (snr < min_snr_) return std::nullopt;

    // DOA via multi-pair GCC-PHAT.
    std::optional<double> raw_doa = multi_pair_doa_fusion(channel_data);

    // Temporal median smoothing + spread metric.
    std::optional<double> doa_spread;
    if (raw_doa.has_value()) {
        doa_history_.push_back(raw_doa.value());
        while (static_cast<int>(doa_history_.size()) > doa_smoothing_window_)
            doa_history_.pop_front();
    }
    if (!doa_history_.empty()) {
        std::vector<double> hist(doa_history_.begin(), doa_history_.end());
        std::sort(hist.begin(), hist.end());
        smoothed_doa_ = hist[hist.size() / 2];
        doa_spread = hist.back() - hist.front();
    }

    AcousticResult res;
    res.freq_hz = peak_freq;
    res.snr = snr;
    res.magnitude = peak_mag;
    res.doa_azimuth = smoothed_doa_.value_or(0.0);
    res.doa_azimuth_raw = raw_doa;
    res.doa_spread_deg = doa_spread;
    res.doa_valid = raw_doa.has_value();
    res.doa_elevation = std::nullopt;
    return res;
}

std::optional<double> AcousticProcessor::multi_pair_doa_fusion(
    const std::vector<std::vector<double>>& channel_data) const {
    // Pair 0-2 (x-axis) and 1-3 (y-axis), both 64 mm baselines.
    std::optional<double> doa_x = gcc_phat_doa(channel_data[0], channel_data[2], 0.064);
    std::optional<double> doa_y = gcc_phat_doa(channel_data[1], channel_data[3], 0.064);

    if (doa_x.has_value() && doa_y.has_value())
        return 0.5 * (doa_x.value() + doa_y.value());
    if (doa_x.has_value()) return doa_x;
    if (doa_y.has_value()) return doa_y;
    return std::nullopt;
}

std::optional<double> AcousticProcessor::gcc_phat_doa(const std::vector<double>& sig1,
                                                      const std::vector<double>& sig2,
                                                      double mic_dist) const {
    const int n = static_cast<int>(sig1.size());
    std::vector<Complex> S1 = rfft(sig1);
    std::vector<Complex> S2 = rfft(sig2);
    const std::vector<double> freqs = rfftfreq(n, 1.0 / sample_rate_);

    // Frequency-band weighting: emphasize the target band.
    std::vector<double> weight(freqs.size(), 0.0);
    double wmax = 0.0;
    for (size_t i = 0; i < freqs.size(); ++i) {
        if (freqs[i] >= min_freq_ && freqs[i] <= max_freq_) {
            weight[i] = std::abs(S1[i]) + std::abs(S2[i]);
            wmax = std::max(wmax, weight[i]);
        }
    }
    for (double& w : weight) w /= (wmax + 1e-10);

    // GCC-PHAT with band emphasis.
    std::vector<Complex> gcc_phat(S1.size());
    for (size_t i = 0; i < S1.size(); ++i) {
        Complex cross = S1[i] * std::conj(S2[i]);
        cross *= (1.0 + 4.0 * weight[i]);
        double mag = std::abs(cross);
        if (mag < 1e-10) mag = 1e-10;
        gcc_phat[i] = cross / mag;
    }

    // High-resolution TDOA via 4x zero-padded inverse FFT.
    const int n_padded = n * 4;
    const std::vector<double> gcc = irfft(gcc_phat, n_padded);

    const int max_delay_samples =
        static_cast<int>(mic_dist / kSpeedOfSound * sample_rate_) + 1;
    const int search_range = std::min(max_delay_samples + 2, n_padded / 2);

    // Assemble [-search_range .. +search_range] around zero lag.
    std::vector<double> gcc_shifted;
    gcc_shifted.reserve(2 * search_range + 1);
    for (int i = n_padded - search_range; i < n_padded; ++i) gcc_shifted.push_back(gcc[i]);
    for (int i = 0; i <= search_range; ++i) gcc_shifted.push_back(gcc[i]);

    // Main peak.
    int peak_idx = 0;
    double peak_val = -1.0;
    for (int i = 0; i < static_cast<int>(gcc_shifted.size()); ++i) {
        const double a = std::abs(gcc_shifted[i]);
        if (a > peak_val) {
            peak_val = a;
            peak_idx = i;
        }
    }

    // Second peak (exclude a window around the main peak) for confidence.
    const int search_width = std::max(5, search_range / 20);
    double second_peak = 0.0;
    for (int i = 0; i < static_cast<int>(gcc_shifted.size()); ++i) {
        if (i >= peak_idx - search_width && i < peak_idx + search_width) continue;
        second_peak = std::max(second_peak, std::abs(gcc_shifted[i]));
    }
    const double confidence = peak_val / (second_peak + 1e-6);

    // Sub-sample delay via parabolic interpolation.
    double delay_samples;
    if (peak_idx > 0 && peak_idx < static_cast<int>(gcc_shifted.size()) - 1) {
        const double y0 = std::abs(gcc_shifted[peak_idx - 1]);
        const double y1 = std::abs(gcc_shifted[peak_idx]);
        const double y2 = std::abs(gcc_shifted[peak_idx + 1]);
        const double denom = 2.0 * (y0 - 2.0 * y1 + y2);
        if (std::abs(denom) > 1e-10) {
            const double frac = (y2 - y0) / denom;
            delay_samples = peak_idx - search_range + frac;
        } else {
            delay_samples = peak_idx - search_range;
        }
    } else {
        delay_samples = peak_idx - search_range;
    }
    delay_samples /= 4.0;  // undo zero-padding

    double delay_sec = delay_samples / sample_rate_;
    const double max_delay = mic_dist / kSpeedOfSound;
    delay_sec = std::clamp(delay_sec, -max_delay, max_delay);

    double sin_theta = delay_sec * kSpeedOfSound / mic_dist;
    sin_theta = std::clamp(sin_theta, -1.0, 1.0);

    if (confidence < doa_min_confidence_) return std::nullopt;
    return std::asin(sin_theta) * 180.0 / 3.14159265358979323846;
}

#ifdef WITH_PORTAUDIO

bool AcousticProcessor::start() {
    if (Pa_Initialize() != paNoError) {
        std::fprintf(stderr, "[Audio] Pa_Initialize failed.\n");
        return false;
    }
    running_ = true;
    thread_ = std::thread(&AcousticProcessor::run, this);
    return true;
}

void AcousticProcessor::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    Pa_Terminate();
}

void AcousticProcessor::run() {
    PaStreamParameters in_params;
    in_params.device = device_index_ >= 0 ? device_index_ : Pa_GetDefaultInputDevice();
    if (in_params.device == paNoDevice) {
        std::fprintf(stderr, "[Audio] No input device.\n");
        return;
    }
    in_params.channelCount = n_channels_;
    in_params.sampleFormat = paInt16;
    in_params.suggestedLatency = Pa_GetDeviceInfo(in_params.device)->defaultLowInputLatency;
    in_params.hostApiSpecificStreamInfo = nullptr;

    PaStream* stream = nullptr;
    if (Pa_OpenStream(&stream, &in_params, nullptr, sample_rate_, chunk_size_, paNoFlag,
                      nullptr, nullptr) != paNoError) {
        std::fprintf(stderr, "[Audio] Failed to open mic stream.\n");
        return;
    }
    Pa_StartStream(stream);
    std::printf("[Audio] Streaming from ReSpeaker (%d Hz, %d ch, chunk=%d)\n", sample_rate_,
                n_channels_, chunk_size_);

    std::vector<int16_t> raw(chunk_size_ * n_channels_);
    while (running_) {
        if (Pa_ReadStream(stream, raw.data(), chunk_size_) != paNoError) continue;
        for (int i = 0; i < chunk_size_; ++i) {
            for (int ch = 0; ch < n_channels_; ++ch)
                buffers_[ch].push_back(static_cast<double>(raw[i * n_channels_ + ch]));
        }
        for (int ch = 0; ch < n_channels_; ++ch)
            while (static_cast<int>(buffers_[ch].size()) > buffer_maxlen_)
                buffers_[ch].pop_front();

        if (static_cast<int>(buffers_[0].size()) >= buffer_maxlen_) {
            std::vector<std::vector<double>> channel_data(n_channels_);
            for (int ch = 0; ch < n_channels_; ++ch)
                channel_data[ch].assign(buffers_[ch].begin(), buffers_[ch].end());
            std::optional<AcousticResult> res = analyze(channel_data);
            std::lock_guard<std::mutex> lock(result_mutex_);
            latest_result_ = res;
        }
    }
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
}

#endif  // WITH_PORTAUDIO

}  // namespace ndrone
