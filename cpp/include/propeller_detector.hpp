// Visual propeller detector: connected-component clustering on a per-pixel
// frequency map + temporal tracking with Bayesian confidence.
//
// This is a faithful C++ port of the FrequencyMapAnalyzer and PropellerTracker
// classes in detect_propeller.py (the v3 production-optimized detector). The
// input is a per-pixel frequency map (CV_32F, Hz per pixel, 0 = no vibration),
// exactly as produced by Metavision's FrequencyMapAsyncAlgorithm. Keeping the
// input identical to the Python version means the SDK glue is the only part
// that differs between platforms.
//
// The C++ port exists to remove the interpreter, GIL and garbage-collection
// overheads identified in PROPELLER_DETECTOR_DEVELOPMENT.md as the remaining
// latency source after the v3 Python optimizations.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

namespace ndrone {

// One candidate propeller region extracted from a single frequency map.
struct Detection {
    double x = 0.0;         // centroid (full-res pixels)
    double y = 0.0;
    double freq_hz = 0.0;   // median frequency within the cluster
    double rpm = 0.0;       // (freq_hz / num_blades) * 60
    int pixels = 0;         // active pixel count
    cv::Rect bbox;          // full-res bounding box
    double freq_cv = 0.0;   // coefficient of variation of frequency
};

// A tracked propeller, persisted across frames.
struct Track {
    int id = 0;
    double x = 0.0, y = 0.0;
    double vx = 0.0, vy = 0.0;   // velocity (px/frame) for predictive matching
    double freq_hz = 0.0;
    double rpm = 0.0;
    int pixels = 0;
    cv::Rect bbox;
    double freq_cv = 0.0;
    int hits = 0;
    int age = 0;
    double avg_p = 0.0;          // running-average per-observation likelihood
    double confidence = 0.0;     // Bayesian confidence P = 1 - (1-p)^n
};

// Connected-component spatial clustering on the per-pixel frequency map.
//
// Mirrors FrequencyMapAnalyzer: threshold -> 4x downscale -> dilate ->
// connected components -> per-cluster ROI frequency stats -> size/CV filter ->
// consolidation of nearby same-frequency clusters. All working buffers are
// pre-allocated once (no per-frame allocation).
class FrequencyMapAnalyzer {
public:
    static constexpr int kDownscale = 4;

    FrequencyMapAnalyzer(int width, int height, double min_freq, double max_freq,
                         int num_blades, int min_pixels = 3, int dilate_radius = 5,
                         double max_freq_cv = 0.3);

    // Extract candidate propeller regions from a CV_32F frequency map.
    std::vector<Detection> analyze(const cv::Mat& freq_map);

    // Exponential moving average of analyze() cost, in milliseconds.
    double analysis_time_ms() const { return analysis_time_ms_; }

private:
    std::vector<Detection> consolidate_clusters(std::vector<Detection> dets,
                                                double merge_distance = 150.0,
                                                double freq_tolerance = 0.15) const;

    double min_freq_, max_freq_;
    int num_blades_, min_pixels_;
    double max_freq_cv_;
    int w_full_, h_full_, w_ds_, h_ds_;

    cv::Mat mask_full_;     // CV_8U full-res binary mask
    cv::Mat mask_small_;    // CV_8U downscaled mask
    cv::Mat mask_dilated_;  // CV_8U dilated mask
    cv::Mat labels_, stats_, centroids_;  // connectedComponents outputs
    cv::Mat kernel_;

    double analysis_time_ms_ = 0.0;
};

// Temporal tracker: greedy association with velocity prediction and Bayesian
// confidence scoring. Mirrors PropellerTracker.
class PropellerTracker {
public:
    // Per-observation likelihood bounds (calibrated as in the Python version).
    static constexpr double kBaseP = 0.65;
    static constexpr double kMaxP = 0.85;

    PropellerTracker(double max_distance = 100.0, double freq_tolerance = 0.3,
                     int min_hits = 2, int max_age = 8,
                     double confidence_threshold = 0.95);

    // Update tracks with this frame's detections; return confirmed tracks
    // (confidence >= threshold AND hits >= min_hits), sorted by pixel count.
    std::vector<Track> update(const std::vector<Detection>& detections);

    const std::vector<Track>& tracks() const { return tracks_; }

private:
    static double detection_likelihood(const Detection& d);
    static double bayesian_confidence(int hits, double avg_p);

    double max_distance_, freq_tolerance_;
    int min_hits_, max_age_;
    double confidence_threshold_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

}  // namespace ndrone
