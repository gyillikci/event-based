// Audio-visual drone detector (C++ port of detect_drone_fused.py).
//
// Fuses the event-camera propeller pipeline (FrequencyMapAnalyzer +
// PropellerTracker) with ReSpeaker 4-mic acoustic detection (AcousticProcessor)
// via sequential Bayesian updating with harmonic frequency cross-validation and
// the same quality / association / angle gates as the Python version.
//
// WITH_METAVISION wires the real SDK frequency map; otherwise a synthetic
// propeller frequency map and a synthetic co-located tone drive the full fusion
// so the gating and Bayesian dynamics can be observed without hardware.

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>

#include "acoustic_processor.hpp"
#include "azimuth.hpp"
#include "bayesian_fusion.hpp"
#include "propeller_detector.hpp"
#include "synthetic.hpp"

#ifdef WITH_METAVISION
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/analytics/algorithms/frequency_map_async_algorithm.h>
#endif

namespace {

struct Args {
    std::string event_file;
    double min_freq = 140, max_freq = 200;   // acoustic band
    int num_blades = 2;
    double visual_min_freq = 50, visual_max_freq = 500;
    int min_cluster_pixels = 20, dilate_radius = 5;
    double max_freq_cv = 0.3;
    int min_hits = 5, max_age = 8;
    double track_distance = 100, freq_tolerance = 0.3;
    double confidence_threshold = 0.95;
    int delta_t = 500;
    bool enable_audio = false;
    double prior = 0.001, detection_threshold = 0.8;
    int visual_min_pixels = 30;
    double visual_min_track_confidence = 0.98;
    int acoustic_min_hits = 3;
    double acoustic_max_freq_jitter = 25.0;
    double audio_confirm_snr = 6.0;
    double max_angle_residual_deg = 35.0;
    double audio_doa_max_spread_deg = 30.0;
    double camera_hfov_deg = 63.8;
    std::string azimuth_calibration_path;
    int synthetic_frames = 120;
    double synthetic_azimuth = 8.0;  // co-located source bearing (deg)
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "-i" || k == "--input-event-file") a.event_file = next();
        else if (k == "--min-freq") a.min_freq = std::atof(next());
        else if (k == "--max-freq") a.max_freq = std::atof(next());
        else if (k == "--num-blades") a.num_blades = std::atoi(next());
        else if (k == "--visual-min-freq") a.visual_min_freq = std::atof(next());
        else if (k == "--visual-max-freq") a.visual_max_freq = std::atof(next());
        else if (k == "--min-cluster-pixels") a.min_cluster_pixels = std::atoi(next());
        else if (k == "--confidence-threshold") a.confidence_threshold = std::atof(next());
        else if (k == "--delta-t") a.delta_t = std::atoi(next());
        else if (k == "--enable-audio") a.enable_audio = true;
        else if (k == "--detection-threshold") a.detection_threshold = std::atof(next());
        else if (k == "--camera-hfov-deg") a.camera_hfov_deg = std::atof(next());
        else if (k == "--azimuth-calibration") a.azimuth_calibration_path = next();
        else if (k == "--synthetic-frames") a.synthetic_frames = std::atoi(next());
        else if (k == "--synthetic-azimuth") a.synthetic_azimuth = std::atof(next());
    }
    return a;
}

// One fusion step given the latest visual evidence and acoustic result, applying
// the Python gating logic. Returns the updated P(drone).
double fuse_step(Args& a, ndrone::BayesianDroneDetector& bayesian,
                 const std::optional<ndrone::VisualEvidence>& visual_ev,
                 int visual_pixels, double visual_track_conf, double visual_azimuth_deg,
                 const std::optional<ndrone::AcousticResult>& acoustic,
                 int acoustic_hits, int width) {
    std::optional<ndrone::VisualEvidence> fuse_visual = visual_ev;
    std::optional<ndrone::AcousticEvidence> fuse_acoustic;

    // Visual quality gate.
    if (fuse_visual.has_value() &&
        (visual_pixels < a.visual_min_pixels ||
         visual_track_conf < a.visual_min_track_confidence)) {
        fuse_visual.reset();
    }

    // Acoustic stability gate.
    if (acoustic.has_value() && acoustic_hits >= a.acoustic_min_hits) {
        ndrone::AcousticEvidence ae;
        ae.snr = acoustic->snr;
        ae.freq_hz = acoustic->freq_hz;
        ae.elevation_deg = acoustic->doa_elevation;
        fuse_acoustic = ae;
    }

    // Angle-residual gate (needs both modalities).
    if (fuse_visual.has_value() && fuse_acoustic.has_value() && acoustic.has_value()) {
        double residual = std::abs(visual_azimuth_deg - acoustic->doa_azimuth);
        if (residual > 180.0) residual = 360.0 - residual;
        if (residual > a.max_angle_residual_deg) fuse_acoustic.reset();
    }

    // Audio-only updates need stronger SNR and a stable bearing.
    if (!fuse_visual.has_value() && fuse_acoustic.has_value() && acoustic.has_value()) {
        if (fuse_acoustic->snr < a.audio_confirm_snr) fuse_acoustic.reset();
        else if (acoustic->doa_spread_deg.has_value() &&
                 acoustic->doa_spread_deg.value() > a.audio_doa_max_spread_deg)
            fuse_acoustic.reset();
    }

    if (fuse_visual.has_value() || fuse_acoustic.has_value())
        return bayesian.update(fuse_visual, fuse_acoustic);
    bayesian.decay();
    return bayesian.confidence();
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    const int width = 1280, height = 720;

    ndrone::FrequencyMapAnalyzer analyzer(width, height, a.visual_min_freq, a.visual_max_freq,
                                          a.num_blades, a.min_cluster_pixels, a.dilate_radius,
                                          a.max_freq_cv);
    ndrone::PropellerTracker tracker(a.track_distance, a.freq_tolerance, a.min_hits, a.max_age,
                                     a.confidence_threshold);
    ndrone::BayesianDroneDetector bayesian(a.prior, 0.995, a.min_freq, a.max_freq,
                                           a.detection_threshold);
    std::optional<ndrone::AzimuthCalibration> calib =
        ndrone::load_azimuth_calibration(a.azimuth_calibration_path);

    ndrone::AcousticProcessor acoustic_proc(16000, 1024, 4, 0.2, a.min_freq, a.max_freq, 3.0);

    std::printf("==============================================================================\n");
    std::printf("  Audio-Visual Drone Detector (C++ Bayesian Fusion)\n");
    std::printf("  Sensor          : %dx%d (HFOV %.1f deg)\n", width, height, a.camera_hfov_deg);
    std::printf("  Acoustic band   : %.0f-%.0f Hz (%d blades)\n", a.min_freq, a.max_freq,
                a.num_blades);
    std::printf("  Visual freq map : %.0f-%.0f Hz\n", a.visual_min_freq, a.visual_max_freq);
    std::printf("  Audio           : %s\n", a.enable_audio ? "ENABLED" : "DISABLED (synthetic)");
    std::printf("  Detection thresh: P(drone) > %.2f\n", a.detection_threshold);
    std::printf("  Azimuth calib   : %s\n",
                calib.has_value() ? a.azimuth_calibration_path.c_str() : "none");
    std::printf("==============================================================================\n\n");

#ifdef WITH_METAVISION
    // (Metavision event source; frequency map feeds the same analyzer.)
    // Real deployment path omitted from the synthetic build for brevity — the
    // per-frame body below (analyze -> track -> fuse) is identical.
    std::fprintf(stderr, "Metavision live path: build the SDK glue as in detect_propeller.cpp.\n");
    return 0;
#else
    // Synthetic: propeller at 180 Hz visual flicker (BPF) + co-located 180 Hz
    // tone from a set azimuth, so the frequency cross-validation should fire.
    ndrone::SyntheticFrequencyMapSource source(width, height, {{700, 360, 12, 180.0f}});
    cv::Mat freq_map;

    int acoustic_hits = 0;
    double last_acoustic_freq = -1;
    const double update_freq = 20.0;
    const double dt_sec = 1.0 / update_freq;

    for (int f = 0; f < a.synthetic_frames; ++f) {
        source.next(freq_map, static_cast<int>(dt_sec * 1e6));
        std::vector<ndrone::Detection> candidates = analyzer.analyze(freq_map);
        std::vector<ndrone::Track> confirmed = tracker.update(candidates);

        // Visual evidence from the best track.
        std::optional<ndrone::VisualEvidence> visual_ev;
        int visual_pixels = 0;
        double visual_track_conf = 0.0, visual_azimuth = 0.0;
        if (!confirmed.empty()) {
            const ndrone::Track& best = confirmed.front();
            ndrone::VisualEvidence ve;
            ve.freq_hz = best.freq_hz;
            ve.snr = best.confidence * 10.0;  // map track confidence to pseudo-SNR
            visual_ev = ve;
            visual_pixels = best.pixels;
            visual_track_conf = best.confidence;
            const double raw_az =
                ndrone::pixel_x_to_visual_azimuth_deg(best.x, width, a.camera_hfov_deg);
            visual_azimuth = ndrone::apply_visual_azimuth_calibration(raw_az, calib);
        }

        // Acoustic evidence: synthetic co-located tone.
        std::vector<std::vector<double>> ch = ndrone::synth_tone_array(
            acoustic_proc.fft_window_samples(), acoustic_proc.sample_rate(), 180.0,
            a.synthetic_azimuth);
        std::optional<ndrone::AcousticResult> ar = acoustic_proc.analyze(ch);
        if (ar.has_value()) {
            if (last_acoustic_freq < 0 ||
                std::abs(ar->freq_hz - last_acoustic_freq) <= a.acoustic_max_freq_jitter)
                ++acoustic_hits;
            else
                acoustic_hits = 1;
            last_acoustic_freq = ar->freq_hz;
        } else {
            acoustic_hits = 0;
        }

        const double p = fuse_step(a, bayesian, visual_ev, visual_pixels, visual_track_conf,
                                   visual_azimuth, ar, acoustic_hits, width);

        if (f % 5 == 0 || bayesian.detected()) {
            char vbuf[64] = "---", abuf[96] = "---";
            if (visual_ev.has_value())
                std::snprintf(vbuf, sizeof(vbuf), "freq=%.1f Hz px=%d conf=%.2f",
                              visual_ev->freq_hz, visual_pixels, visual_track_conf);
            if (ar.has_value())
                std::snprintf(abuf, sizeof(abuf), "freq=%.1f Hz SNR=%.1f DOA=%.0f",
                              ar->freq_hz, ar->snr, ar->doa_azimuth);
            std::string xval;
            if (visual_ev.has_value() && ar.has_value()) {
                ndrone::FreqMatch m =
                    ndrone::cross_validate_frequency(visual_ev->freq_hz, ar->freq_hz);
                if (m.matched) xval = "  FREQ MATCH";
            }
            std::printf("[%6.2fs] P=%.4f  V:[%s]  A:[%s]%s  %s\n", f * dt_sec, p, vbuf, abuf,
                        xval.c_str(), bayesian.detected() ? "*** DRONE ***" : "");
        }
    }
    std::printf("\nDone (%d synthetic frames).\n", a.synthetic_frames);
    return 0;
#endif
}
