// Drone propeller rotation detector (C++ port of detect_propeller.py, v3).
//
// Pipeline: per-pixel frequency map -> connected-component clustering
// (FrequencyMapAnalyzer) -> temporal tracking with Bayesian confidence
// (PropellerTracker). This headless C++ build removes the interpreter/GIL/GC
// overhead that bounded the Python version's latency.
//
// The per-pixel frequency map comes from Metavision's FrequencyMapAsyncAlgorithm
// when built WITH_METAVISION; otherwise a synthetic rotating-propeller source
// drives the identical downstream pipeline so the detector can be exercised and
// benchmarked without the SDK.

#include <cstdio>
#include <cstring>
#include <string>

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
    double min_freq = 10.0;
    double max_freq = 300.0;
    int num_blades = 3;
    int filter_length = 4;
    int max_period_diff = 1500;
    int min_cluster_pixels = 2;
    int dilate_radius = 5;
    double max_freq_cv = 0.3;
    int min_hits = 5;
    int max_age = 8;
    double track_distance = 100.0;
    double freq_tolerance = 0.3;
    int delta_t = 500;
    double update_freq = 20.0;
    double confidence_threshold = 0.95;
    int synthetic_frames = 120;  // synthetic mode only
};

double darg(const char* v) { return std::atof(v); }
int iarg(const char* v) { return std::atoi(v); }

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "-i" || k == "--input-event-file") a.event_file = next();
        else if (k == "--min-freq") a.min_freq = darg(next());
        else if (k == "--max-freq") a.max_freq = darg(next());
        else if (k == "--num-blades") a.num_blades = iarg(next());
        else if (k == "--filter-length") a.filter_length = iarg(next());
        else if (k == "--max-period-diff") a.max_period_diff = iarg(next());
        else if (k == "--min-cluster-pixels") a.min_cluster_pixels = iarg(next());
        else if (k == "--dilate-radius") a.dilate_radius = iarg(next());
        else if (k == "--max-freq-cv") a.max_freq_cv = darg(next());
        else if (k == "--min-hits") a.min_hits = iarg(next());
        else if (k == "--max-age") a.max_age = iarg(next());
        else if (k == "--track-distance") a.track_distance = darg(next());
        else if (k == "--freq-tolerance") a.freq_tolerance = darg(next());
        else if (k == "--delta-t") a.delta_t = iarg(next());
        else if (k == "--update-freq") a.update_freq = darg(next());
        else if (k == "--confidence-threshold") a.confidence_threshold = darg(next());
        else if (k == "--synthetic-frames") a.synthetic_frames = iarg(next());
    }
    return a;
}

void print_header(const Args& a, int width, int height, const char* source) {
    const double nyquist = 1e6 / a.delta_t / 2.0;
    std::printf("==============================================================================\n");
    std::printf("  Drone Propeller Rotation Detector (C++ v3)\n");
    std::printf("  Source          : %s\n", source);
    std::printf("  Sensor          : %dx%d\n", width, height);
    std::printf("  Frequency range : %.0f - %.0f Hz\n", a.min_freq, a.max_freq);
    std::printf("  Blades assumed  : %d\n", a.num_blades);
    std::printf("  Min cluster px  : %d\n", a.min_cluster_pixels);
    std::printf("  Max freq CV     : %.2f\n", a.max_freq_cv);
    std::printf("  Tracking        : min_hits=%d, max_age=%d, dist=%.0f, freq_tol=%.2f\n",
                a.min_hits, a.max_age, a.track_distance, a.freq_tolerance);
    std::printf("  Confidence      : Bayesian threshold=%.0f%%\n", a.confidence_threshold * 100);
    std::printf("  Update rate     : %.0f Hz\n", a.update_freq);
    std::printf("  delta_t         : %d us (Nyquist=%.0f Hz)\n", a.delta_t, nyquist);
    std::printf("==============================================================================\n\n");
}

// Analyze one frequency map, update tracks, print confirmations. Shared by the
// synthetic and Metavision paths.
void process_freq_map(double ts_sec, const cv::Mat& freq_map,
                      ndrone::FrequencyMapAnalyzer& analyzer,
                      ndrone::PropellerTracker& tracker, int num_blades) {
    std::vector<ndrone::Detection> candidates = analyzer.analyze(freq_map);
    std::vector<ndrone::Track> confirmed = tracker.update(candidates);
    if (!confirmed.empty()) {
        std::printf("[%7.2fs] === %zu CONFIRMED PROPELLER(S) === (%zu cand, analyze=%.2fms)\n",
                    ts_sec, confirmed.size(), candidates.size(), analyzer.analysis_time_ms());
        for (const ndrone::Track& t : confirmed) {
            std::printf("    >> ID%3d: freq=%6.1f Hz, RPM=%7.0f, pixels=%4d, conf=%.1f%%, "
                        "hits=%3d, pos=(%.0f,%.0f)\n",
                        t.id, t.freq_hz, t.rpm, t.pixels, t.confidence * 100, t.hits, t.x, t.y);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    const double nyquist = 1e6 / a.delta_t / 2.0;
    if (a.max_freq > nyquist) {
        std::fprintf(stderr, "--max-freq (%.0f Hz) exceeds Nyquist (%.0f Hz); reduce --delta-t.\n",
                     a.max_freq, nyquist);
        return 1;
    }

#ifdef WITH_METAVISION
    // ---- Real per-pixel frequency map from the Metavision SDK ----
    Metavision::Camera camera;
    if (a.event_file.empty())
        camera = Metavision::Camera::from_first_available();
    else
        camera = Metavision::Camera::from_file(a.event_file);

    const int width = camera.geometry().width();
    const int height = camera.geometry().height();

    ndrone::FrequencyMapAnalyzer analyzer(width, height, a.min_freq, a.max_freq, a.num_blades,
                                          a.min_cluster_pixels, a.dilate_radius, a.max_freq_cv);
    ndrone::PropellerTracker tracker(a.track_distance, a.freq_tolerance, a.min_hits, a.max_age,
                                     a.confidence_threshold);
    print_header(a, width, height, "Metavision SDK");

    Metavision::FrequencyMapAsyncAlgorithm freq_algo(width, height, a.filter_length, a.min_freq,
                                                     a.max_freq, a.max_period_diff);
    freq_algo.set_update_frequency(static_cast<float>(a.update_freq));
    // The SDK delivers a cv::Mat_<float> frequency map (Hz per pixel); feed it
    // straight into the identical analyzer/tracker used by the synthetic path.
    freq_algo.set_output_callback([&](Metavision::timestamp ts, cv::Mat& freq_map) {
        process_freq_map(ts / 1e6, freq_map, analyzer, tracker, a.num_blades);
    });

    camera.cd().add_callback(
        [&](const Metavision::EventCD* begin, const Metavision::EventCD* end) {
            freq_algo.process_events(begin, end);
        });

    camera.start();
    while (camera.is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    camera.stop();
    return 0;
#else
    // ---- Synthetic rotating-propeller source (no SDK required) ----
    const int width = 1280, height = 720;
    ndrone::FrequencyMapAnalyzer analyzer(width, height, a.min_freq, a.max_freq, a.num_blades,
                                          a.min_cluster_pixels, a.dilate_radius, a.max_freq_cv);
    ndrone::PropellerTracker tracker(a.track_distance, a.freq_tolerance, a.min_hits, a.max_age,
                                     a.confidence_threshold);
    print_header(a, width, height, "synthetic (SDK not built in)");
    std::printf("  [synthetic] one 3-blade propeller at 270 Hz BPF + single-pixel noise.\n\n");

    // 270 Hz blade-pass -> 5400 RPM for a 3-blade prop (as in the dev logs).
    ndrone::SyntheticFrequencyMapSource source(width, height,
                                               {{430, 220, 9, 270.0f}});
    cv::Mat freq_map;
    const double dt_sec = 1.0 / a.update_freq;
    for (int f = 0; f < a.synthetic_frames; ++f) {
        source.next(freq_map, static_cast<int>(dt_sec * 1e6));
        process_freq_map(f * dt_sec, freq_map, analyzer, tracker, a.num_blades);
    }
    std::printf("\nDone (%d synthetic frames).\n", a.synthetic_frames);
    return 0;
#endif
}
