// Known-answer tests for the SDK-independent core. Builds and runs without the
// Metavision SDK, so CI on any platform can verify the ported algorithms.

#include <cmath>
#include <cstdio>
#include <vector>

#include "acoustic_processor.hpp"
#include "azimuth.hpp"
#include "bayesian_fusion.hpp"
#include "fft.hpp"
#include "propeller_detector.hpp"
#include "synthetic.hpp"

namespace {

int g_failures = 0;

void check(bool cond, const char* name) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
    if (!cond) ++g_failures;
}

// FFT round-trip: irfft(rfft(x)) == x.
void test_fft() {
    std::printf("FFT\n");
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};  // odd length -> Bluestein
    std::vector<ndrone::Complex> spec = ndrone::rfft(x);
    std::vector<double> y = ndrone::irfft(spec, static_cast<int>(x.size()));
    double max_err = 0;
    for (size_t i = 0; i < x.size(); ++i) max_err = std::max(max_err, std::abs(x[i] - y[i]));
    check(max_err < 1e-9, "rfft/irfft round-trip (arbitrary length)");

    // A pure tone should peak at its bin.
    const int n = 256;
    std::vector<double> tone(n);
    for (int i = 0; i < n; ++i) tone[i] = std::sin(2 * M_PI * 10.0 * i / n);
    std::vector<ndrone::Complex> ts = ndrone::rfft(tone);
    int peak = 0;
    double best = 0;
    for (size_t i = 1; i < ts.size(); ++i)
        if (std::abs(ts[i]) > best) {
            best = std::abs(ts[i]);
            peak = static_cast<int>(i);
        }
    check(peak == 10, "rfft locates a pure tone bin");
}

// The visual pipeline confirms a rotating propeller and rejects noise-only frames.
void test_propeller_pipeline() {
    std::printf("Propeller pipeline\n");
    const int w = 1280, h = 720;
    ndrone::FrequencyMapAnalyzer analyzer(w, h, 10, 300, 3, /*min_pixels=*/2,
                                          /*dilate=*/5, /*max_cv=*/0.3);
    ndrone::PropellerTracker tracker(100, 0.3, /*min_hits=*/5, 8, 0.95);

    // Frames with a 270 Hz, 3-blade propeller (-> 5400 RPM).
    ndrone::SyntheticFrequencyMapSource source(w, h, {{430, 220, 9, 270.0f}});
    cv::Mat fm;
    bool confirmed = false;
    ndrone::Track best;
    for (int f = 0; f < 30; ++f) {
        source.next(fm, 50000);
        std::vector<ndrone::Detection> cand = analyzer.analyze(fm);
        std::vector<ndrone::Track> conf = tracker.update(cand);
        if (!conf.empty()) {
            confirmed = true;
            best = conf.front();
        }
    }
    check(confirmed, "confirms a rotating propeller");
    check(std::abs(best.freq_hz - 270.0) < 10.0, "recovers ~270 Hz blade-pass frequency");
    check(std::abs(best.rpm - 5400.0) < 300.0, "reports ~5400 RPM (3 blades)");

    // Noise-only frames (no propeller) must not confirm anything.
    ndrone::FrequencyMapAnalyzer analyzer2(w, h, 10, 300, 3, 2, 5, 0.3);
    ndrone::PropellerTracker tracker2(100, 0.3, 5, 8, 0.95);
    ndrone::SyntheticFrequencyMapSource noise(w, h, /*no props=*/{}, /*noise=*/0.0005);
    bool any = false;
    for (int f = 0; f < 30; ++f) {
        noise.next(fm, 50000);
        if (!tracker2.update(analyzer2.analyze(fm)).empty()) any = true;
    }
    check(!any, "rejects single-pixel noise (no false confirmations)");
}

// Harmonic cross-validation matches fundamentals and harmonics.
void test_cross_validation() {
    std::printf("Cross-validation\n");
    check(ndrone::cross_validate_frequency(180.0, 180.0).matched, "equal frequencies match");
    ndrone::FreqMatch harm = ndrone::cross_validate_frequency(180.0, 90.0);
    check(harm.matched && harm.harmonic_v == 2 && harm.harmonic_a == 1,
          "180 Hz (2nd harmonic) matches 90 Hz fundamental");
    // 205 Hz shares no low-order harmonic with 180 Hz within 5% tolerance.
    check(!ndrone::cross_validate_frequency(180.0, 205.0).matched, "unrelated freqs do not match");
}

// Bayesian fusion drives P(drone) high with corroborating evidence, then decays.
void test_bayesian() {
    std::printf("Bayesian fusion\n");
    ndrone::BayesianDroneDetector det(0.001, 0.995, 140, 200, 0.8);
    for (int i = 0; i < 5; ++i) {
        ndrone::VisualEvidence v{9.0, 180.0};
        ndrone::AcousticEvidence a{9.0, 180.0, std::nullopt};
        det.update(v, a);
    }
    check(det.detected(), "P(drone) exceeds threshold with matched V+A evidence");
    for (int i = 0; i < 2000; ++i) det.decay();
    check(!det.detected(), "belief decays back below threshold without evidence");
}

// Acoustic analyzer recovers a synthetic tone frequency.
void test_acoustic() {
    std::printf("Acoustic\n");
    ndrone::AcousticProcessor proc(16000, 1024, 4, 0.2, 50, 500, 3.0);
    std::vector<std::vector<double>> ch =
        ndrone::synth_tone_array(proc.fft_window_samples(), 16000, 162.7, 15.0);
    std::optional<ndrone::AcousticResult> r = proc.analyze(ch);
    check(r.has_value(), "detects a tone above the SNR floor");
    if (r.has_value()) check(std::abs(r->freq_hz - 162.7) < 5.0, "recovers ~162.7 Hz tone");
}

// Azimuth mapping round-trips through the pinhole model.
void test_azimuth() {
    std::printf("Azimuth\n");
    const int w = 1280;
    const double hfov = 63.8;
    const double az = ndrone::pixel_x_to_visual_azimuth_deg(w / 2.0, w, hfov);
    check(std::abs(az) < 0.2, "image center maps to ~0 deg azimuth");
    const double x = ndrone::visual_azimuth_deg_to_pixel_x(10.0, w, hfov);
    const double az2 = ndrone::pixel_x_to_visual_azimuth_deg(x, w, hfov);
    check(std::abs(az2 - 10.0) < 0.5, "pixel<->azimuth round-trips");
}

}  // namespace

int main() {
    std::printf("=== ndrone core tests ===\n");
    test_fft();
    test_propeller_pipeline();
    test_cross_validation();
    test_bayesian();
    test_acoustic();
    test_azimuth();
    std::printf("=========================\n");
    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
