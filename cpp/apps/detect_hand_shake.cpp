// Hand-shake detector (C++ port of detect_hand_shake.py).
//
// Tracks the centroid of events over time and uses FFT on the centroid
// displacement to estimate the shake frequency (Hz) and magnitude (px p-p).
// Built WITH_METAVISION it reads a live camera / event file; otherwise a
// synthetic shaking-centroid source drives the identical FFT analyzer.

#include <cmath>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

#include "fft.hpp"
#include "synthetic.hpp"

#ifdef WITH_METAVISION
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#endif

namespace {

struct ShakeResult {
    double freq_x = 0, mag_x = 0;
    double freq_y = 0, mag_y = 0;
    double freq_combined = 0, mag_combined = 0;
    bool valid = false;
};

// Dominant frequency (>= 1 Hz) and peak-to-peak magnitude of a signal.
void dominant_peak(const std::vector<double>& sig, double fs, double& freq, double& mag) {
    const int n = static_cast<int>(sig.size());
    double mean = 0;
    for (double v : sig) mean += v;
    mean /= n;
    std::vector<double> w = ndrone::hanning(n);
    std::vector<double> windowed(n);
    for (int i = 0; i < n; ++i) windowed[i] = (sig[i] - mean) * w[i];

    std::vector<ndrone::Complex> spec = ndrone::rfft(windowed);
    std::vector<double> freqs = ndrone::rfftfreq(n, 1.0 / fs);
    freq = 0;
    mag = 0;
    double best = -1;
    for (size_t i = 0; i < spec.size(); ++i) {
        if (freqs[i] < 1.0) continue;
        const double m = std::abs(spec[i]) * 2.0 / n;
        if (m > best) {
            best = m;
            freq = freqs[i];
            mag = m * 2.0;  // peak-to-peak
        }
    }
}

ShakeResult analyze_shake(const std::vector<double>& ts_us, const std::vector<double>& cx,
                          const std::vector<double>& cy) {
    ShakeResult r;
    const int n = static_cast<int>(ts_us.size());
    if (n < 8) return r;
    const double duration = (ts_us.back() - ts_us.front()) / 1e6;
    if (duration <= 0) return r;
    const double fs = (n - 1) / duration;

    dominant_peak(cx, fs, r.freq_x, r.mag_x);
    dominant_peak(cy, fs, r.freq_y, r.mag_y);

    double mcx = 0, mcy = 0;
    for (int i = 0; i < n; ++i) {
        mcx += cx[i];
        mcy += cy[i];
    }
    mcx /= n;
    mcy /= n;
    std::vector<double> combined(n);
    for (int i = 0; i < n; ++i) {
        const double dx = cx[i] - mcx, dy = cy[i] - mcy;
        combined[i] = std::sqrt(dx * dx + dy * dy);
    }
    dominant_peak(combined, fs, r.freq_combined, r.mag_combined);
    r.valid = true;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::string event_file;
    double analysis_window_sec = 2.0;
    double print_interval_sec = 0.5;
    int delta_t = 1000;
    int synthetic_frames = 400;
    double shake_freq = 5.0, shake_amp = 8.0;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "-i" || k == "--input-event-file") event_file = next();
        else if (k == "--analysis-window") analysis_window_sec = std::atof(next());
        else if (k == "--print-interval") print_interval_sec = std::atof(next());
        else if (k == "--delta-t") delta_t = std::atoi(next());
        else if (k == "--synthetic-frames") synthetic_frames = std::atoi(next());
        else if (k == "--shake-freq") shake_freq = std::atof(next());
        else if (k == "--shake-amp") shake_amp = std::atof(next());
    }

    const int window_samples = static_cast<int>(analysis_window_sec * 1e6 / delta_t);
    const int print_samples = static_cast<int>(print_interval_sec * 1e6 / delta_t);

    std::deque<double> ts_buf, cx_buf, cy_buf;
    int sample_count = 0;

    auto emit = [&](double cx, double cy, double ts) {
        ts_buf.push_back(ts);
        cx_buf.push_back(cx);
        cy_buf.push_back(cy);
        while (static_cast<int>(ts_buf.size()) > window_samples) {
            ts_buf.pop_front();
            cx_buf.pop_front();
            cy_buf.pop_front();
        }
        if (++sample_count % print_samples == 0 && static_cast<int>(ts_buf.size()) >= 8) {
            ShakeResult r = analyze_shake({ts_buf.begin(), ts_buf.end()},
                                          {cx_buf.begin(), cx_buf.end()},
                                          {cy_buf.begin(), cy_buf.end()});
            if (r.valid)
                std::printf("Shake  |  X: freq=%6.1f Hz, mag=%5.1f px  |  "
                            "Y: freq=%6.1f Hz, mag=%5.1f px  |  "
                            "Combined: freq=%6.1f Hz, mag=%5.1f px\n",
                            r.freq_x, r.mag_x, r.freq_y, r.mag_y, r.freq_combined,
                            r.mag_combined);
        }
    };

    std::printf("=================================================================\n");
    std::printf("  Hand-Shake Detector (C++)\n");
    std::printf("  Sampling rate: %.0f Hz (delta_t=%d us)\n", 1e6 / delta_t, delta_t);
    std::printf("  FFT window: %.1fs (%d samples)\n", analysis_window_sec, window_samples);
    std::printf("=================================================================\n\n");

#ifdef WITH_METAVISION
    Metavision::Camera camera = event_file.empty()
                                    ? Metavision::Camera::from_first_available()
                                    : Metavision::Camera::from_file(event_file);
    camera.cd().add_callback(
        [&](const Metavision::EventCD* begin, const Metavision::EventCD* end) {
            if (begin == end) return;
            double sx = 0, sy = 0;
            int n = 0;
            for (const Metavision::EventCD* e = begin; e != end; ++e, ++n) {
                sx += e->x;
                sy += e->y;
            }
            emit(sx / n, sy / n, static_cast<double>((end - 1)->t));
        });
    camera.start();
    while (camera.is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    camera.stop();
#else
    std::printf("  [synthetic] shaking at %.1f Hz, amplitude %.1f px\n\n", shake_freq, shake_amp);
    ndrone::SyntheticShake shake;
    shake.freq_hz = shake_freq;
    shake.amp_px = shake_amp;
    for (int f = 0; f < synthetic_frames; ++f) {
        const double t = f * (delta_t / 1e6);
        double cx, cy;
        shake.at(t, cx, cy);
        emit(cx, cy, t * 1e6);
    }
    std::printf("\nDone (%d synthetic frames).\n", synthetic_frames);
#endif
    return 0;
}
