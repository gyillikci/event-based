/**********************************************************************************************************************
 * Native C++ port of the v3 production propeller detector (detect_propeller.py).
 *
 * Pipeline:
 *   [optional] Metavision::ProximityFilterAlgorithm   -> focus region (C++-only)
 *              Metavision::FrequencyMapAsyncAlgorithm -> per-pixel frequency map (cv::Mat1f)
 *              FrequencyMapAnalyzer                   -> threshold / downscale / dilate /
 *                                                        connected components / ROI stats
 *              PropellerTracker                       -> greedy NN + Bayesian confirmation
 *
 * Differs from cpp/propeller_detector, which implements the SDK-clustering path
 * (FrequencyAlgorithm + FrequencyClusteringAlgorithm) instead.
 *
 * CLI names and defaults mirror detect_propeller.py so results can be compared directly.
 **********************************************************************************************************************/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// timestamp.h must come first: the analytics headers use Metavision::timestamp without including it.
#include <metavision/sdk/base/utils/timestamp.h>

#include <metavision/sdk/analytics/algorithms/dominant_value_map_algorithm.h>
#include <metavision/sdk/analytics/algorithms/frequency_map_async_algorithm.h>
#include <metavision/sdk/analytics/algorithms/heat_map_frame_generator_algorithm.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/colors.h>
#include <metavision/sdk/cv/algorithms/proximity_filter_algorithm.h>
#include <metavision/sdk/cv/configs/frequency_estimation_config.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/file_config_hints.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/mt_window.h>

namespace {

// ---------------------------------------------------------------------------
//  Command line
// ---------------------------------------------------------------------------

/// Minimal command-line parser: supports "--key value" and boolean "--flag".
class Args {
public:
    Args(int argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a.rfind("--", 0) == 0) {
                const std::string key = a.substr(2);
                if (i + 1 < argc) {
                    const std::string next = argv[i + 1];
                    if (next.rfind("--", 0) != 0) {
                        kv_[key] = next;
                        ++i;
                        continue;
                    }
                }
                flags_.insert(key);
            } else if (a.rfind("-", 0) == 0 && a.size() == 2) {
                const std::string key = a.substr(1);
                if (i + 1 < argc) {
                    const std::string next = argv[i + 1];
                    if (next.rfind("-", 0) != 0) {
                        kv_[key] = next;
                        ++i;
                        continue;
                    }
                }
                flags_.insert(key);
            }
        }
    }

    bool has(const std::string &k) const {
        return kv_.count(k) != 0 || flags_.count(k) != 0;
    }
    std::string get(const std::string &k, const std::string &d = "") const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : it->second;
    }
    int geti(const std::string &k, int d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stoi(it->second);
    }
    float getf(const std::string &k, float d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stof(it->second);
    }
    bool getb(const std::string &k, bool d) const {
        if (flags_.count(k) != 0) {
            return true;
        }
        auto it = kv_.find(k);
        if (it == kv_.end()) {
            return d;
        }
        return it->second == "1" || it->second == "true" || it->second == "on" || it->second == "yes";
    }

private:
    std::map<std::string, std::string> kv_;
    std::set<std::string> flags_;
};

void print_usage() {
    std::cout <<
        "drone_propeller_detector - C++ port of detect_propeller.py (v3 pipeline)\n\n"
        "FrequencyMapAsyncAlgorithm -> connected-component analyzer -> Bayesian tracker.\n"
        "Two windows: events with detection overlay, and the frequency heat map.\n"
        "Press 'q' or ESC in a window to quit (killing the process can wedge EVK4 USB).\n\n"
        "Input:\n"
        "  -i, --input <file>              .raw/.dat/.hdf5 recording (omit to use a live camera)\n"
        "      --realtime-playback-speed   play files back at real time (default on)\n\n"
        "Frequency range:\n"
        "      --min-freq <hz>             minimum propeller frequency (default 10)\n"
        "      --max-freq <hz>             maximum propeller frequency (default 300)\n"
        "      --num-blades <n>            blades per propeller, RPM = freq/blades*60 (default 3)\n\n"
        "Frequency map tuning:\n"
        "      --filter-length <n>         successive periods to confirm a vibration (default 4)\n"
        "      --max-period-diff <us>      max period difference, same signal (default 1500)\n"
        "      --freq-precision <hz>       heat map / dominant value bin width (default 5.0)\n"
        "      --update-freq <hz>          frequency map update rate (default 20)\n\n"
        "Spatial clustering:\n"
        "      --min-cluster-pixels <n>    minimum vibrating pixels per candidate (default 2)\n"
        "      --dilate-radius <px>        dilation radius bridging nearby pixels (default 5)\n"
        "      --max-freq-cv <f>           max frequency coefficient of variation (default 0.3)\n\n"
        "Temporal tracking:\n"
        "      --min-hits <n>              minimum detections before confirmation (default 5)\n"
        "      --max-age <n>               frames without detection before pruning (default 8)\n"
        "      --track-distance <px>       max matching distance (default 100)\n"
        "      --freq-tolerance <f>        max relative frequency difference (default 0.3)\n"
        "      --confidence-threshold <f>  Bayesian confirmation threshold (default 0.95)\n\n"
        "Focus region (ProximityFilterAlgorithm, C++-only; enabled if any is set):\n"
        "      --focus-x <px>              center x\n"
        "      --focus-y <px>              center y\n"
        "      --focus-radius <px>         max distance from center (default 100)\n\n"
        "Output:\n"
        "      --no-display                headless, no windows\n"
        "      --session-log <path>        write a JSONL trace, one record per frequency map\n"
        "      --benchmark                 headless timing run, prints a summary at exit\n"
        "      --self-test                 run the analyzer/tracker on a synthetic map and exit\n"
        "  -h, --help                      show this help\n";
}

// ---------------------------------------------------------------------------
//  Spatial analysis on the per-pixel frequency map
// ---------------------------------------------------------------------------

struct Detection {
    double x        = 0.0;
    double y        = 0.0;
    double freq_hz  = 0.0;
    double rpm      = 0.0;
    int pixels      = 0;
    cv::Rect bbox;
    double freq_cv = 0.0;
};

/// Reorders v; matches numpy's even-length median (mean of the two central values).
float median_inplace(std::vector<float> &v) {
    const size_t n   = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const float hi = v[mid];
    if (n % 2 == 1) {
        return hi;
    }
    const float lo = *std::max_element(v.begin(), v.begin() + mid);
    return 0.5f * (lo + hi);
}

/// Finds propeller candidate regions in the SDK's per-pixel frequency map.
class FrequencyMapAnalyzer {
public:
    static constexpr int DOWNSCALE = 4;

    FrequencyMapAnalyzer(int width, int height, float min_freq, float max_freq, int num_blades, int min_pixels,
                         int dilate_radius, float max_freq_cv) :
        min_freq_(min_freq),
        max_freq_(max_freq),
        num_blades_(num_blades),
        min_pixels_(min_pixels),
        max_freq_cv_(max_freq_cv),
        w_full_(width),
        h_full_(height),
        w_ds_(width / DOWNSCALE),
        h_ds_(height / DOWNSCALE) {
        const int ds_radius = std::max(1, dilate_radius / DOWNSCALE);
        const int k         = 2 * ds_radius + 1;
        kernel_             = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));

        mask_full_.create(h_full_, w_full_);
        mask_small_.create(h_ds_, w_ds_);
        mask_dilated_.create(h_ds_, w_ds_);
    }

    double analysis_time_ms() const {
        return analysis_time_ms_;
    }

    std::vector<Detection> analyze(const cv::Mat1f &freq_map) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Detection> detections;

        cv::inRange(freq_map, cv::Scalar(min_freq_), cv::Scalar(max_freq_), mask_full_);
        if (cv::countNonZero(mask_full_) < min_pixels_) {
            update_timing(t0);
            return detections;
        }

        cv::resize(mask_full_, mask_small_, cv::Size(w_ds_, h_ds_), 0, 0, cv::INTER_NEAREST);
        cv::dilate(mask_small_, mask_dilated_, kernel_);

        const int num_labels  = cv::connectedComponentsWithStats(mask_dilated_, labels_, stats_, centroids_, 8);
        const int min_area_ds = std::max(1, min_pixels_ / (DOWNSCALE * DOWNSCALE));

        for (int i = 1; i < num_labels; ++i) { // label 0 is the background
            if (stats_.at<int>(i, cv::CC_STAT_AREA) < min_area_ds) {
                continue;
            }

            const int x_full = stats_.at<int>(i, cv::CC_STAT_LEFT) * DOWNSCALE;
            const int y_full = stats_.at<int>(i, cv::CC_STAT_TOP) * DOWNSCALE;
            const int x2     = std::min(x_full + stats_.at<int>(i, cv::CC_STAT_WIDTH) * DOWNSCALE, w_full_);
            const int y2     = std::min(y_full + stats_.at<int>(i, cv::CC_STAT_HEIGHT) * DOWNSCALE, h_full_);
            if (x2 <= x_full || y2 <= y_full) {
                continue;
            }

            // Only the component's bounding box is read from the full-resolution map.
            const cv::Rect roi(x_full, y_full, x2 - x_full, y2 - y_full);
            cv::Mat1b roi_valid = mask_full_(roi);
            cv::Mat1f roi_freqs = freq_map(roi);

            cluster_freqs_.clear();
            for (int r = 0; r < roi.height; ++r) {
                const uchar *mrow = roi_valid[r];
                const float *frow = roi_freqs[r];
                for (int c = 0; c < roi.width; ++c) {
                    if (mrow[c]) {
                        cluster_freqs_.push_back(frow[c]);
                    }
                }
            }

            const int actual_pixels = static_cast<int>(cluster_freqs_.size());
            if (actual_pixels < min_pixels_) {
                continue;
            }

            const double median_freq = median_inplace(cluster_freqs_);
            if (median_freq <= 0.0) {
                continue;
            }

            double freq_cv = 0.0;
            if (actual_pixels > 1) {
                double sum = 0.0, sum2 = 0.0;
                for (const float f : cluster_freqs_) {
                    sum += f;
                    sum2 += static_cast<double>(f) * f;
                }
                const double mean = sum / actual_pixels;
                const double var  = std::max(0.0, sum2 / actual_pixels - mean * mean);
                freq_cv           = std::sqrt(var) / median_freq;
            }
            if (freq_cv > max_freq_cv_) {
                continue;
            }

            Detection d;
            d.x       = centroids_.at<double>(i, 0) * DOWNSCALE;
            d.y       = centroids_.at<double>(i, 1) * DOWNSCALE;
            d.freq_hz = median_freq;
            d.rpm     = (median_freq / num_blades_) * 60.0;
            d.pixels  = actual_pixels;
            d.bbox    = roi;
            d.freq_cv = freq_cv;
            detections.push_back(d);
        }

        std::sort(detections.begin(), detections.end(),
                  [](const Detection &a, const Detection &b) { return a.pixels > b.pixels; });

        detections = consolidate_clusters(detections);
        update_timing(t0);
        return detections;
    }

private:
    /// Merges fragments of the same propeller (nearby, similar frequency) into one candidate.
    std::vector<Detection> consolidate_clusters(const std::vector<Detection> &detections,
                                                double merge_distance = 150.0,
                                                double freq_tolerance = 0.15) const {
        if (detections.size() <= 1) {
            return detections;
        }

        std::vector<Detection> consolidated;
        std::vector<bool> used(detections.size(), false);

        for (size_t i = 0; i < detections.size(); ++i) {
            if (used[i]) {
                continue;
            }
            Detection merged = detections[i];
            used[i]          = true;

            for (size_t j = i + 1; j < detections.size(); ++j) {
                if (used[j]) {
                    continue;
                }
                const Detection &other = detections[j];

                const double dx   = other.x - merged.x;
                const double dy   = other.y - merged.y;
                const double dist = std::sqrt(dx * dx + dy * dy);
                if (dist > merge_distance) {
                    continue;
                }

                const double f1         = merged.freq_hz;
                const double f2         = other.freq_hz;
                const double freq_ratio = std::max(f1, f2) / std::max(std::min(f1, f2), 1e-6);
                if (freq_ratio > 1.0 + freq_tolerance) {
                    continue;
                }

                const double total_px = merged.pixels + other.pixels;
                const double w1       = merged.pixels / total_px;
                const double w2       = other.pixels / total_px;

                merged.x       = w1 * merged.x + w2 * other.x;
                merged.y       = w1 * merged.y + w2 * other.y;
                merged.freq_hz = w1 * merged.freq_hz + w2 * other.freq_hz;
                merged.rpm     = (merged.freq_hz / num_blades_) * 60.0;
                merged.pixels  = merged.pixels + other.pixels;
                merged.bbox    = merged.bbox | other.bbox;
                merged.freq_cv = (merged.freq_cv + other.freq_cv) / 2.0;

                used[j] = true;
            }

            consolidated.push_back(merged);
        }

        return consolidated;
    }

    void update_timing(const std::chrono::steady_clock::time_point &t0) {
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        analysis_time_ms_ = (analysis_time_ms_ == 0.0) ? elapsed_ms : 0.2 * elapsed_ms + 0.8 * analysis_time_ms_;
    }

    float min_freq_;
    float max_freq_;
    int num_blades_;
    int min_pixels_;
    double max_freq_cv_;
    int w_full_, h_full_, w_ds_, h_ds_;

    cv::Mat kernel_;
    cv::Mat1b mask_full_, mask_small_, mask_dilated_;
    cv::Mat labels_, stats_, centroids_;
    std::vector<float> cluster_freqs_;
    double analysis_time_ms_ = 0.0;
};

// ---------------------------------------------------------------------------
//  Temporal tracking
// ---------------------------------------------------------------------------

struct Track {
    int id      = 0;
    double x    = 0.0;
    double y    = 0.0;
    double vx   = 0.0;
    double vy   = 0.0;
    double freq_hz = 0.0;
    double rpm     = 0.0;
    int pixels     = 0;
    cv::Rect bbox;
    double freq_cv    = 0.0;
    int hits          = 0;
    int age           = 0;
    double avg_p      = 0.0;
    double confidence = 0.0;
};

/// Tracks candidates across frequency maps and confirms them with a Bayesian score.
class PropellerTracker {
public:
    static constexpr double BASE_P = 0.65; // baseline likelihood of a minimal detection
    static constexpr double MAX_P  = 0.85; // likelihood of a strong detection

    PropellerTracker(double max_distance, double freq_tolerance, int min_hits, int max_age,
                     double confidence_threshold) :
        max_distance_(max_distance),
        freq_tolerance_(freq_tolerance),
        min_hits_(min_hits),
        max_age_(max_age),
        confidence_threshold_(confidence_threshold) {}

    const std::vector<Track> &tracks() const {
        return tracks_;
    }

    std::vector<Track> update(const std::vector<Detection> &detections) {
        std::set<size_t> matched_tracks;
        std::set<size_t> matched_dets;

        for (size_t di = 0; di < detections.size(); ++di) {
            const Detection &det = detections[di];

            bool found          = false;
            size_t best_idx     = 0;
            double best_dist    = max_distance_;

            for (size_t ti = 0; ti < tracks_.size(); ++ti) {
                if (matched_tracks.count(ti)) {
                    continue;
                }
                const Track &track = tracks_[ti];

                const double pred_x = track.x + track.vx;
                const double pred_y = track.y + track.vy;
                const double dx     = det.x - pred_x;
                const double dy     = det.y - pred_y;
                const double dist   = std::sqrt(dx * dx + dy * dy);
                if (dist > max_distance_) {
                    continue;
                }

                const double f1         = det.freq_hz;
                const double f2         = track.freq_hz;
                const double freq_ratio = std::max(f1, f2) / std::max(std::min(f1, f2), 1e-6);
                if (freq_ratio > 1.0 + freq_tolerance_) {
                    continue;
                }

                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx  = ti;
                    found     = true;
                }
            }

            if (found) {
                Track &track       = tracks_[best_idx];
                constexpr double a = 0.3;

                const double raw_vx = det.x - track.x;
                const double raw_vy = det.y - track.y;
                track.vx            = a * raw_vx + (1 - a) * track.vx;
                track.vy            = a * raw_vy + (1 - a) * track.vy;

                track.x       = a * det.x + (1 - a) * track.x;
                track.y       = a * det.y + (1 - a) * track.y;
                track.freq_hz = a * det.freq_hz + (1 - a) * track.freq_hz;
                track.rpm     = a * det.rpm + (1 - a) * track.rpm;
                track.pixels  = det.pixels;
                track.bbox    = det.bbox;
                track.freq_cv = det.freq_cv;
                track.hits += 1;
                track.age = 0;

                const double p  = detection_likelihood(det);
                track.avg_p     = (track.avg_p * (track.hits - 1) + p) / track.hits;
                track.confidence = bayesian_confidence(track.hits, track.avg_p);

                matched_tracks.insert(best_idx);
                matched_dets.insert(di);
            }
        }

        for (size_t di = 0; di < detections.size(); ++di) {
            if (matched_dets.count(di)) {
                continue;
            }
            const Detection &det = detections[di];
            const double p       = detection_likelihood(det);

            Track t;
            t.id         = next_id_++;
            t.x          = det.x;
            t.y          = det.y;
            t.freq_hz    = det.freq_hz;
            t.rpm        = det.rpm;
            t.pixels     = det.pixels;
            t.bbox       = det.bbox;
            t.freq_cv    = det.freq_cv;
            t.hits       = 1;
            t.age        = 0;
            t.avg_p      = p;
            t.confidence = p;
            tracks_.push_back(t);
        }

        for (size_t ti = 0; ti < tracks_.size(); ++ti) {
            if (!matched_tracks.count(ti)) {
                tracks_[ti].age += 1;
                tracks_[ti].confidence *= 0.8;
            }
        }

        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [this](const Track &t) { return t.age > max_age_; }),
                      tracks_.end());

        std::vector<Track> confirmed;
        for (const Track &t : tracks_) {
            if (t.confidence >= confidence_threshold_ && t.hits >= min_hits_) {
                confirmed.push_back(t);
            }
        }
        std::sort(confirmed.begin(), confirmed.end(),
                  [](const Track &a, const Track &b) { return a.pixels > b.pixels; });
        return confirmed;
    }

private:
    /// Stronger detections (more pixels, lower CV) confirm in fewer frames.
    static double detection_likelihood(const Detection &det) {
        const double px_score = std::min(1.0, (det.pixels - 1) / 49.0);
        const double cv_score = std::max(0.0, 1.0 - det.freq_cv / 0.3);
        const double quality  = 0.6 * px_score + 0.4 * cv_score;
        return BASE_P + quality * (MAX_P - BASE_P);
    }

    static double bayesian_confidence(int hits, double avg_p) {
        if (hits <= 0 || avg_p <= 0.0) {
            return 0.0;
        }
        return 1.0 - std::pow(1.0 - avg_p, hits);
    }

    double max_distance_;
    double freq_tolerance_;
    int min_hits_;
    int max_age_;
    double confidence_threshold_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

// ---------------------------------------------------------------------------
//  Rendering
// ---------------------------------------------------------------------------

/// Draws in-place; the frame generator overwrites the frame on the next callback anyway.
void draw_detections_on_frame(cv::Mat &frame, const std::vector<Track> &confirmed_tracks) {
    const bool use_full_labels = confirmed_tracks.size() <= 8;

    for (const Track &track : confirmed_tracks) {
        const int margin = std::max(10, static_cast<int>(std::max(track.bbox.width, track.bbox.height) * 0.3));
        const int x1     = std::max(0, track.bbox.x - margin);
        const int y1     = std::max(0, track.bbox.y - margin);
        const int x2     = std::min(frame.cols, track.bbox.x + track.bbox.width + margin);
        const int y2     = std::min(frame.rows, track.bbox.y + track.bbox.height + margin);

        const int brightness = std::min(255, 100 + std::min(track.hits, 10) * 15);
        const cv::Scalar color(0, brightness, 0);
        constexpr int thickness = 2;

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);

        const int font       = cv::FONT_HERSHEY_SIMPLEX;
        constexpr int t_thick = 1;

        if (use_full_labels) {
            std::ostringstream l1, l2;
            l1 << "ID" << track.id << ": " << std::fixed << std::setprecision(0) << track.freq_hz << "Hz "
               << track.rpm << "RPM";
            l2 << track.pixels << "px conf=" << std::fixed << std::setprecision(0) << (track.confidence * 100.0)
               << "%";

            constexpr double font_scale = 0.5;
            int baseline                = 0;
            const cv::Size s1           = cv::getTextSize(l1.str(), font, font_scale, t_thick, &baseline);
            const cv::Size s2           = cv::getTextSize(l2.str(), font, font_scale, t_thick, &baseline);
            const int tw                = std::max(s1.width, s2.width);

            int ty = y1 - 5;
            if (ty - s1.height - s2.height - 8 < 0) {
                ty = y2 + s1.height + 5;
            }

            cv::rectangle(frame, cv::Point(x1, ty - s1.height - s2.height - 8), cv::Point(x1 + tw + 6, ty + 4),
                          cv::Scalar(0, 0, 0), -1);
            cv::putText(frame, l1.str(), cv::Point(x1 + 3, ty - s2.height - 4), font, font_scale, color, t_thick,
                        cv::LINE_AA);
            cv::putText(frame, l2.str(), cv::Point(x1 + 3, ty), font, font_scale, color, t_thick, cv::LINE_AA);
        } else {
            std::ostringstream label;
            label << "ID" << track.id << ": " << std::fixed << std::setprecision(0) << track.freq_hz << "Hz";
            cv::putText(frame, label.str(), cv::Point(x1 + 3, y1 - 3), font, 0.4, color, t_thick, cv::LINE_AA);
        }

        cv::drawMarker(frame, cv::Point(static_cast<int>(track.x), static_cast<int>(track.y)), color,
                       cv::MARKER_CROSS, 12, thickness);
    }
}

// ---------------------------------------------------------------------------
//  JSONL session log
// ---------------------------------------------------------------------------

void write_log_record(std::ofstream &log, double ts_s, double wall_ms, size_t n_candidates, size_t n_tracks,
                      const std::vector<Track> &confirmed, double analyze_ms) {
    log << std::fixed;
    log << "{\"ts_s\":" << std::setprecision(6) << ts_s << ",\"wall_ms\":" << std::setprecision(3) << wall_ms
        << ",\"n_candidates\":" << n_candidates << ",\"n_tracks\":" << n_tracks
        << ",\"n_confirmed\":" << confirmed.size() << ",\"analyze_ms\":" << std::setprecision(4) << analyze_ms
        << ",\"tracks\":[";
    for (size_t i = 0; i < confirmed.size(); ++i) {
        const Track &t = confirmed[i];
        if (i) {
            log << ",";
        }
        log << "{\"id\":" << t.id << ",\"x\":" << std::setprecision(2) << t.x << ",\"y\":" << t.y
            << ",\"freq_hz\":" << std::setprecision(3) << t.freq_hz << ",\"rpm\":" << std::setprecision(1) << t.rpm
            << ",\"pixels\":" << t.pixels << ",\"confidence\":" << std::setprecision(4) << t.confidence
            << ",\"hits\":" << t.hits << ",\"freq_cv\":" << t.freq_cv << ",\"bbox\":[" << t.bbox.x << "," << t.bbox.y
            << "," << t.bbox.width << "," << t.bbox.height << "]}";
    }
    log << "]}\n";
}

// ---------------------------------------------------------------------------
//  Self test
// ---------------------------------------------------------------------------

void paint_blob(cv::Mat1f &map, int cx, int cy, int radius, float freq_hz) {
    for (int y = std::max(0, cy - radius); y < std::min(map.rows, cy + radius); ++y) {
        for (int x = std::max(0, cx - radius); x < std::min(map.cols, cx + radius); ++x) {
            const int dx = x - cx;
            const int dy = y - cy;
            if (dx * dx + dy * dy <= radius * radius) {
                map(y, x) = freq_hz;
            }
        }
    }
}

/// Runs the analyzer + tracker on a synthetic frequency map. No camera required.
int run_self_test(int num_blades) {
    constexpr int W = 1280;
    constexpr int H = 720;

    cv::Mat1f freq_map(H, W, 0.f);
    paint_blob(freq_map, 400, 300, 12, 100.f);
    paint_blob(freq_map, 900, 450, 9, 150.f);

    FrequencyMapAnalyzer analyzer(W, H, 10.f, 300.f, num_blades, 2, 5, 0.3f);
    PropellerTracker tracker(100.0, 0.3, 5, 8, 0.95);

    std::vector<Track> confirmed;
    std::vector<Detection> candidates;
    for (int frame = 0; frame < 10; ++frame) {
        candidates = analyzer.analyze(freq_map);
        confirmed  = tracker.update(candidates);
    }

    std::cout << "self-test: " << candidates.size() << " candidate(s), " << confirmed.size()
              << " confirmed after 10 frames (analyze=" << std::fixed << std::setprecision(2)
              << analyzer.analysis_time_ms() << "ms)\n";

    bool ok = (confirmed.size() == 2);
    if (!ok) {
        std::cerr << "self-test FAILED: expected 2 confirmed tracks\n";
    }

    std::vector<double> freqs;
    for (const Track &t : confirmed) {
        std::cout << "  ID" << t.id << ": freq=" << std::setprecision(1) << t.freq_hz
                  << " Hz, RPM=" << std::setprecision(0) << t.rpm << ", pixels=" << t.pixels
                  << ", conf=" << std::setprecision(1) << (t.confidence * 100.0) << "%, pos=(" << t.x << "," << t.y
                  << ")\n";
        freqs.push_back(t.freq_hz);
    }
    std::sort(freqs.begin(), freqs.end());

    const std::vector<double> expected = {100.0, 150.0};
    if (freqs.size() == expected.size()) {
        for (size_t i = 0; i < freqs.size(); ++i) {
            if (std::fabs(freqs[i] - expected[i]) / expected[i] > 0.01) {
                std::cerr << "self-test FAILED: frequency " << freqs[i] << " != " << expected[i] << " (+/-1%)\n";
                ok = false;
            }
        }
    } else {
        ok = false;
    }

    std::cout << (ok ? "self-test PASSED\n" : "self-test FAILED\n");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
//  Benchmark accumulator
// ---------------------------------------------------------------------------

struct BenchStats {
    std::vector<double> samples;

    void add(double ms) {
        samples.push_back(ms);
    }

    void print(const std::string &title, double total_wall_s) {
        if (samples.empty()) {
            std::cout << title << ": no samples\n";
            return;
        }
        std::vector<double> s = samples;
        std::sort(s.begin(), s.end());
        double sum = 0.0;
        for (const double v : s) {
            sum += v;
        }
        const double p95 = s[static_cast<size_t>(0.95 * (s.size() - 1))];
        std::cout << "\n" << title << " over " << s.size() << " frequency maps\n"
                  << std::fixed << std::setprecision(3) << "  min  : " << s.front() << " ms\n"
                  << "  mean : " << (sum / s.size()) << " ms\n"
                  << "  p95  : " << p95 << " ms\n"
                  << "  max  : " << s.back() << " ms\n"
                  << "  wall : " << std::setprecision(2) << total_wall_s << " s\n";
    }
};

} // namespace

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    Args args(argc, argv);
    if (args.has("help") || args.has("h")) {
        print_usage();
        return 0;
    }

    const int num_blades = std::max(1, args.geti("num-blades", 3));
    if (args.has("self-test")) {
        return run_self_test(num_blades);
    }

    const std::string input = args.get("input", args.get("i", ""));
    const bool realtime     = args.getb("realtime-playback-speed", true);

    const float min_freq              = args.getf("min-freq", 10.f);
    const float max_freq              = args.getf("max-freq", 300.f);
    const unsigned int filter_length  = static_cast<unsigned int>(args.geti("filter-length", 4));
    const unsigned int diff_thresh_us = static_cast<unsigned int>(args.geti("max-period-diff", 1500));
    const float freq_precision        = args.getf("freq-precision", 5.f);
    const float update_freq           = args.getf("update-freq", 20.f);

    const int min_cluster_pixels = args.geti("min-cluster-pixels", 2);
    const int dilate_radius      = args.geti("dilate-radius", 5);
    const float max_freq_cv      = args.getf("max-freq-cv", 0.3f);

    const int min_hits                 = args.geti("min-hits", 5);
    const int max_age                  = args.geti("max-age", 8);
    const float track_distance         = args.getf("track-distance", 100.f);
    const float freq_tolerance         = args.getf("freq-tolerance", 0.3f);
    const float confidence_threshold   = args.getf("confidence-threshold", 0.95f);

    const bool use_focus = args.has("focus-x") || args.has("focus-y") || args.has("focus-radius");
    const float focus_x  = args.getf("focus-x", 0.f);
    const float focus_y  = args.getf("focus-y", 0.f);
    const float focus_r  = args.getf("focus-radius", 100.f);

    const bool benchmark        = args.getb("benchmark", false);
    const bool display          = !args.getb("no-display", false) && !benchmark;
    const std::string log_path  = args.get("session-log", "");

    if (max_freq <= min_freq) {
        std::cerr << "--max-freq must be greater than --min-freq\n";
        return 1;
    }

    Metavision::Camera camera;
    try {
        if (input.empty()) {
            camera = Metavision::Camera::from_first_available();
        } else {
            camera = Metavision::Camera::from_file(input, Metavision::FileConfigHints().real_time_playback(realtime));
        }
    } catch (const std::exception &e) {
        std::cerr << "Camera init failed: " << e.what() << "\n";
        return 1;
    }

    const int width  = camera.geometry().width();
    const int height = camera.geometry().height();

    Metavision::FrequencyEstimationConfig freq_cfg(filter_length, min_freq, max_freq, diff_thresh_us, false);
    Metavision::FrequencyMapAsyncAlgorithm freq_algo(width, height, freq_cfg);
    freq_algo.set_update_frequency(update_freq);

    Metavision::DominantValueMapAlgorithm<float> dominant_algo(min_freq, max_freq, freq_precision,
                                                               min_cluster_pixels);
    Metavision::HeatMapFrameGeneratorAlgorithm heat_gen(min_freq, max_freq, freq_precision, width, height, "Hz");
    const int freq_full_height = heat_gen.get_full_height();
    cv::Mat freq_img(freq_full_height, width, CV_8UC3, cv::Scalar::all(0));

    FrequencyMapAnalyzer analyzer(width, height, min_freq, max_freq, num_blades, min_cluster_pixels, dilate_radius,
                                  max_freq_cv);
    PropellerTracker tracker(track_distance, freq_tolerance, min_hits, max_age, confidence_threshold);

    std::optional<Metavision::ProximityFilterAlgorithm> prox;
    if (use_focus) {
        prox.emplace(Eigen::Vector2f(focus_x, focus_y), focus_r);
    }

    std::ofstream session_log;
    if (!log_path.empty()) {
        session_log.open(log_path, std::ios::out | std::ios::trunc);
        if (!session_log) {
            std::cerr << "Cannot open --session-log " << log_path << "\n";
            return 1;
        }
    }

    std::cout << "==============================================================================\n"
              << "  Drone Propeller Rotation Detector  (C++ port of detect_propeller.py v3)\n"
              << "  Sensor          : " << width << "x" << height << "\n"
              << "  Frequency range : " << min_freq << " - " << max_freq << " Hz\n"
              << "  Blades assumed  : " << num_blades << "\n"
              << "  Clustering      : custom connected-component\n"
              << "  Min cluster px  : " << min_cluster_pixels << "\n"
              << "  Dilate radius   : " << dilate_radius << "\n"
              << "  Max freq CV     : " << max_freq_cv << "\n"
              << "  Tracking        : min_hits=" << min_hits << ", max_age=" << max_age
              << ", dist=" << track_distance << ", freq_tol=" << freq_tolerance << "\n"
              << "  Confidence      : Bayesian threshold=" << confidence_threshold << "\n"
              << "  Update rate     : " << update_freq << " Hz  (cycle=" << std::fixed << std::setprecision(0)
              << (1000.0 / update_freq) << " ms)\n"
              << "  Focus           : " << (use_focus ? "ProximityFilterAlgorithm ON" : "off");
    if (use_focus) {
        std::cout << " center=(" << focus_x << "," << focus_y << ") r=" << focus_r;
    }
    std::cout << "\n  Display         : " << (display ? "2 windows (q/ESC to quit)" : "headless") << "\n"
              << "  Source          : " << (input.empty() ? "live camera" : input) << "\n"
              << "==============================================================================\n\n";

    std::unique_ptr<Metavision::MTWindow> ev_window;
    std::unique_ptr<Metavision::MTWindow> freq_window;
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm> frame_gen;
    std::atomic<bool> stop_requested{false};

    std::mutex state_mutex;
    std::vector<Track> confirmed_propellers;

    if (display) {
        ev_window = std::make_unique<Metavision::MTWindow>("Propeller Detector - Events + Detections", width, height,
                                                           Metavision::BaseWindow::RenderMode::BGR);
        freq_window = std::make_unique<Metavision::MTWindow>("Frequency Map", width, freq_full_height,
                                                             Metavision::BaseWindow::RenderMode::BGR);

        auto keyboard_cb = [&](Metavision::UIKeyEvent key, int, Metavision::UIAction action, int) {
            if (action == Metavision::UIAction::RELEASE &&
                (key == Metavision::UIKeyEvent::KEY_Q || key == Metavision::UIKeyEvent::KEY_ESCAPE)) {
                stop_requested = true;
            }
        };
        ev_window->set_keyboard_callback(keyboard_cb);
        freq_window->set_keyboard_callback(keyboard_cb);

        frame_gen = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(width, height, 10000, 25.,
                                                                                   Metavision::ColorPalette::Dark);
        frame_gen->set_output_callback([&](Metavision::timestamp, cv::Mat &cd_frame) {
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                if (!confirmed_propellers.empty()) {
                    draw_detections_on_frame(cd_frame, confirmed_propellers);
                }
            }
            ev_window->show_async(cd_frame);
        });
    }

    BenchStats bench;
    const auto wall_start = std::chrono::steady_clock::now();

    double last_print_ts = 0.0;
    std::set<int> prev_confirmed_ids;

    freq_algo.set_output_callback([&](Metavision::timestamp ts, cv::Mat1f &freq_map) {
        const auto t_cycle = std::chrono::steady_clock::now();

        const std::vector<Detection> candidates = analyzer.analyze(freq_map);
        std::vector<Track> confirmed            = tracker.update(candidates);

        const double cycle_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cycle).count();
        if (benchmark) {
            bench.add(cycle_ms);
        }

        const size_t n_candidates = candidates.size();
        const size_t n_confirmed  = confirmed.size();
        const size_t n_tracks     = tracker.tracks().size();

        if (display) {
            heat_gen.generate_bgr_heat_map(freq_map, freq_img);

            for (const Track &track : confirmed) {
                const int margin =
                    std::max(8, static_cast<int>(std::max(track.bbox.width, track.bbox.height) * 0.2));
                const int x1 = std::max(0, track.bbox.x - margin);
                const int y1 = std::max(0, track.bbox.y - margin);
                const int x2 = std::min(width, track.bbox.x + track.bbox.width + margin);
                const int y2 = std::min(height, track.bbox.y + track.bbox.height + margin);
                cv::rectangle(freq_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255), 2);

                std::ostringstream label;
                label << std::fixed << std::setprecision(0) << track.freq_hz << "Hz " << track.rpm << "RPM";
                cv::putText(freq_img, label.str(), cv::Point(x1, std::max(y1 - 5, 12)), cv::FONT_HERSHEY_SIMPLEX,
                            0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }

            float dominant_freq = 0.f;
            if (dominant_algo.compute_dominant_value(freq_map, dominant_freq)) {
                std::ostringstream text;
                text << "Dominant: " << std::fixed << std::setprecision(0) << dominant_freq << " Hz  ("
                     << ((dominant_freq / num_blades) * 60.0) << " RPM, " << num_blades << " blades)";
                cv::putText(freq_img, text.str(), cv::Point(10, height - 10), cv::FONT_HERSHEY_PLAIN, 1.0,
                            cv::Scalar(255, 255, 255), 1);
            }

            std::ostringstream status;
            status << "Candidates: " << n_candidates << "  Tracks: " << n_tracks << "  Confirmed: " << n_confirmed;
            cv::putText(freq_img, status.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

            freq_window->show_async(freq_img);
        }

        const double ts_sec = ts / 1e6;

        if (session_log.is_open()) {
            const double wall_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
            write_log_record(session_log, ts_sec, wall_ms, n_candidates, n_tracks, confirmed,
                             analyzer.analysis_time_ms());
        }

        std::set<int> current_ids;
        for (const Track &t : confirmed) {
            current_ids.insert(t.id);
        }
        const bool changed  = (current_ids != prev_confirmed_ids);
        const bool periodic = (ts_sec - last_print_ts >= 2.0);

        if (changed || (periodic && !confirmed.empty())) {
            prev_confirmed_ids = current_ids;
            last_print_ts      = ts_sec;

            if (!confirmed.empty()) {
                std::cout << "[" << std::fixed << std::setw(7) << std::setprecision(2) << ts_sec << "s] === "
                          << n_confirmed << " CONFIRMED PROPELLER(S) ===  (" << n_candidates << " cand, " << n_tracks
                          << " trk, analyze=" << std::setprecision(1) << analyzer.analysis_time_ms() << "ms)\n";
                for (const Track &t : confirmed) {
                    std::cout << "    >> ID" << std::setw(3) << t.id << ": freq=" << std::setw(6)
                              << std::setprecision(1) << t.freq_hz << " Hz, RPM=" << std::setw(7)
                              << std::setprecision(0) << t.rpm << ", pixels=" << std::setw(4) << t.pixels
                              << ", conf=" << std::setprecision(1) << (t.confidence * 100.0) << "%, hits="
                              << std::setw(3) << t.hits << ", pos=(" << std::setprecision(0) << t.x << "," << t.y
                              << ")\n";
                }
                std::cout.flush();
            }
        }

        std::lock_guard<std::mutex> lock(state_mutex);
        confirmed_propellers = std::move(confirmed);
    });

    std::vector<Metavision::EventCD> focused;
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        const Metavision::EventCD *b = begin;
        const Metavision::EventCD *e = end;
        if (prox) {
            focused.clear();
            prox->process_events(begin, end, std::back_inserter(focused));
            b = focused.data();
            e = focused.data() + focused.size();
        }
        freq_algo.process_events(b, e);

        // The display keeps the full field of view even when a focus region is set.
        if (frame_gen) {
            frame_gen->process_events(begin, end);
        }
    });

    try {
        camera.start();
        while (camera.is_running() && !stop_requested) {
            if (display) {
                Metavision::EventLoop::poll_and_dispatch(20);
                if (ev_window->should_close() || freq_window->should_close()) {
                    break;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        camera.stop();
    } catch (const std::exception &e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return 1;
    }

    if (session_log.is_open()) {
        session_log.close();
    }

    if (benchmark) {
        const double total_wall_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        bench.print("Analyzer + tracker cost", total_wall_s);
    }

    std::cout << "Done.\n";
    return 0;
}
