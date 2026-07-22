#include "propeller_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

namespace ndrone {

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Median of a vector (numpy.median semantics: average the two central values
// for an even count). Sorts in place.
double median_inplace(std::vector<float>& v) {
    const size_t n = v.size();
    if (n == 0) return 0.0;
    std::sort(v.begin(), v.end());
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (static_cast<double>(v[n / 2 - 1]) + static_cast<double>(v[n / 2]));
}

// Population standard deviation about the mean (numpy.std default, ddof=0).
double stddev(const std::vector<float>& v) {
    const size_t n = v.size();
    if (n < 2) return 0.0;
    double mean = 0.0;
    for (float x : v) mean += x;
    mean /= static_cast<double>(n);
    double acc = 0.0;
    for (float x : v) {
        double d = x - mean;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(n));
}

}  // namespace

// ---------------------------------------------------------------------------
// FrequencyMapAnalyzer
// ---------------------------------------------------------------------------

FrequencyMapAnalyzer::FrequencyMapAnalyzer(int width, int height, double min_freq,
                                           double max_freq, int num_blades,
                                           int min_pixels, int dilate_radius,
                                           double max_freq_cv)
    : min_freq_(min_freq),
      max_freq_(max_freq),
      num_blades_(num_blades),
      min_pixels_(min_pixels),
      max_freq_cv_(max_freq_cv),
      w_full_(width),
      h_full_(height),
      w_ds_(width / kDownscale),
      h_ds_(height / kDownscale) {
    const int ds_radius = std::max(1, dilate_radius / kDownscale);
    const int k = 2 * ds_radius + 1;
    kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));

    // Pre-allocated buffers (v3: eliminates per-frame allocation).
    mask_full_.create(h_full_, w_full_, CV_8U);
    mask_small_.create(h_ds_, w_ds_, CV_8U);
    mask_dilated_.create(h_ds_, w_ds_, CV_8U);
}

std::vector<Detection> FrequencyMapAnalyzer::analyze(const cv::Mat& freq_map) {
    const double t0 = now_ms();
    std::vector<Detection> detections;
    constexpr int DS = kDownscale;

    // Step 1: binary mask of pixels vibrating in the target range.
    cv::inRange(freq_map, cv::Scalar(min_freq_), cv::Scalar(max_freq_), mask_full_);

    // Early exit: too few active pixels to be a cluster.
    const int active_count = cv::countNonZero(mask_full_);
    if (active_count < min_pixels_) {
        analysis_time_ms_ = analysis_time_ms_ == 0.0
                                ? (now_ms() - t0)
                                : 0.2 * (now_ms() - t0) + 0.8 * analysis_time_ms_;
        return detections;
    }

    // Step 2: downscale for fast morphology + connected components.
    cv::resize(mask_full_, mask_small_, cv::Size(w_ds_, h_ds_), 0, 0, cv::INTER_NEAREST);

    // Step 3: dilate to bridge nearby pixels (cheap on the small image).
    cv::dilate(mask_small_, mask_dilated_, kernel_);

    // Step 4: connected components on the small image.
    const int num_labels = cv::connectedComponentsWithStats(
        mask_dilated_, labels_, stats_, centroids_, 8, CV_32S);

    const int min_area_ds = std::max(1, min_pixels_ / (DS * DS));

    for (int i = 1; i < num_labels; ++i) {  // label 0 = background
        const int area_dilated = stats_.at<int>(i, cv::CC_STAT_AREA);
        if (area_dilated < min_area_ds) continue;

        // Map bounding box back to full resolution.
        const int x_full = stats_.at<int>(i, cv::CC_STAT_LEFT) * DS;
        const int y_full = stats_.at<int>(i, cv::CC_STAT_TOP) * DS;
        const int w_box = stats_.at<int>(i, cv::CC_STAT_WIDTH) * DS;
        const int h_box = stats_.at<int>(i, cv::CC_STAT_HEIGHT) * DS;
        const int x2 = std::min(x_full + w_box, w_full_);
        const int y2 = std::min(y_full + h_box, h_full_);
        if (x2 <= x_full || y2 <= y_full) continue;

        const cv::Rect roi(x_full, y_full, x2 - x_full, y2 - y_full);

        // ROI-only analysis: gather frequencies of active pixels in this box.
        const cv::Mat freq_roi = freq_map(roi);
        const cv::Mat mask_roi = mask_full_(roi);
        std::vector<float> cluster_freqs;
        cluster_freqs.reserve(static_cast<size_t>(roi.area()));
        for (int r = 0; r < freq_roi.rows; ++r) {
            const float* fp = freq_roi.ptr<float>(r);
            const uint8_t* mp = mask_roi.ptr<uint8_t>(r);
            for (int c = 0; c < freq_roi.cols; ++c) {
                if (mp[c]) cluster_freqs.push_back(fp[c]);
            }
        }

        const int actual_pixels = static_cast<int>(cluster_freqs.size());
        if (actual_pixels < min_pixels_) continue;

        // Step 5: frequency statistics on the ROI only.
        const double std_freq = stddev(cluster_freqs);
        const double median_freq = median_inplace(cluster_freqs);
        if (median_freq <= 0.0) continue;

        const double freq_cv = actual_pixels > 1 ? std_freq / median_freq : 0.0;
        if (freq_cv > max_freq_cv_) continue;

        Detection det;
        det.x = centroids_.at<double>(i, 0) * DS;
        det.y = centroids_.at<double>(i, 1) * DS;
        det.freq_hz = median_freq;
        det.rpm = (median_freq / num_blades_) * 60.0;
        det.pixels = actual_pixels;
        det.bbox = cv::Rect(x_full, y_full, x2 - x_full, y2 - y_full);
        det.freq_cv = freq_cv;
        detections.push_back(det);
    }

    // Sort by pixel count (largest = most confident).
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.pixels > b.pixels; });

    detections = consolidate_clusters(std::move(detections));

    const double elapsed = now_ms() - t0;
    analysis_time_ms_ =
        analysis_time_ms_ == 0.0 ? elapsed : 0.2 * elapsed + 0.8 * analysis_time_ms_;
    return detections;
}

std::vector<Detection> FrequencyMapAnalyzer::consolidate_clusters(
    std::vector<Detection> dets, double merge_distance, double freq_tolerance) const {
    if (dets.size() <= 1) return dets;

    std::vector<Detection> out;
    std::vector<bool> used(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (used[i]) continue;
        Detection merged = dets[i];
        used[i] = true;

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (used[j]) continue;
            const Detection& d2 = dets[j];

            const double dx = d2.x - merged.x;
            const double dy = d2.y - merged.y;
            if (std::sqrt(dx * dx + dy * dy) > merge_distance) continue;

            const double f1 = merged.freq_hz, f2 = d2.freq_hz;
            const double ratio = std::max(f1, f2) / std::max(std::min(f1, f2), 1e-6);
            if (ratio > 1.0 + freq_tolerance) continue;

            // Merge: pixel-count-weighted average.
            const double total = merged.pixels + d2.pixels;
            const double w1 = merged.pixels / total;
            const double w2 = d2.pixels / total;
            merged.x = w1 * merged.x + w2 * d2.x;
            merged.y = w1 * merged.y + w2 * d2.y;
            merged.freq_hz = w1 * merged.freq_hz + w2 * d2.freq_hz;
            merged.rpm = (merged.freq_hz / num_blades_) * 60.0;
            merged.pixels = static_cast<int>(total);
            merged.bbox |= d2.bbox;  // cv::Rect union
            merged.freq_cv = 0.5 * (merged.freq_cv + d2.freq_cv);
            used[j] = true;
        }
        out.push_back(merged);
    }
    return out;
}

// ---------------------------------------------------------------------------
// PropellerTracker
// ---------------------------------------------------------------------------

PropellerTracker::PropellerTracker(double max_distance, double freq_tolerance,
                                   int min_hits, int max_age,
                                   double confidence_threshold)
    : max_distance_(max_distance),
      freq_tolerance_(freq_tolerance),
      min_hits_(min_hits),
      max_age_(max_age),
      confidence_threshold_(confidence_threshold) {}

double PropellerTracker::detection_likelihood(const Detection& d) {
    // Pixel score ramps 0 (1 px) -> 1 (50+ px); CV score ramps 1 (cv=0) -> 0.
    const double px_score = std::min(1.0, (d.pixels - 1) / 49.0);
    const double cv_score = std::max(0.0, 1.0 - d.freq_cv / 0.3);
    const double quality = 0.6 * px_score + 0.4 * cv_score;
    return kBaseP + quality * (kMaxP - kBaseP);
}

double PropellerTracker::bayesian_confidence(int hits, double avg_p) {
    if (hits <= 0 || avg_p <= 0.0) return 0.0;
    return 1.0 - std::pow(1.0 - avg_p, hits);
}

std::vector<Track> PropellerTracker::update(const std::vector<Detection>& detections) {
    std::set<int> matched_tracks;
    std::set<int> matched_dets;

    // Greedy matching against predicted (velocity-extrapolated) positions.
    for (int di = 0; di < static_cast<int>(detections.size()); ++di) {
        const Detection& det = detections[di];
        int best_track = -1;
        double best_dist = max_distance_;

        for (int ti = 0; ti < static_cast<int>(tracks_.size()); ++ti) {
            if (matched_tracks.count(ti)) continue;
            const Track& tr = tracks_[ti];

            const double pred_x = tr.x + tr.vx;
            const double pred_y = tr.y + tr.vy;
            const double dist =
                std::sqrt((det.x - pred_x) * (det.x - pred_x) +
                          (det.y - pred_y) * (det.y - pred_y));
            if (dist > max_distance_) continue;

            const double f1 = det.freq_hz, f2 = tr.freq_hz;
            const double ratio = std::max(f1, f2) / std::max(std::min(f1, f2), 1e-6);
            if (ratio > 1.0 + freq_tolerance_) continue;

            if (dist < best_dist) {
                best_dist = dist;
                best_track = ti;
            }
        }

        if (best_track >= 0) {
            Track& tr = tracks_[best_track];
            constexpr double alpha = 0.3;

            const double raw_vx = det.x - tr.x;
            const double raw_vy = det.y - tr.y;
            tr.vx = alpha * raw_vx + (1 - alpha) * tr.vx;
            tr.vy = alpha * raw_vy + (1 - alpha) * tr.vy;

            tr.x = alpha * det.x + (1 - alpha) * tr.x;
            tr.y = alpha * det.y + (1 - alpha) * tr.y;
            tr.freq_hz = alpha * det.freq_hz + (1 - alpha) * tr.freq_hz;
            tr.rpm = alpha * det.rpm + (1 - alpha) * tr.rpm;
            tr.pixels = det.pixels;
            tr.bbox = det.bbox;
            tr.freq_cv = det.freq_cv;
            tr.hits += 1;
            tr.age = 0;

            const double p = detection_likelihood(det);
            tr.avg_p = (tr.avg_p * (tr.hits - 1) + p) / tr.hits;
            tr.confidence = bayesian_confidence(tr.hits, tr.avg_p);

            matched_tracks.insert(best_track);
            matched_dets.insert(di);
        }
    }

    // Create new tracks for unmatched detections.
    for (int di = 0; di < static_cast<int>(detections.size()); ++di) {
        if (matched_dets.count(di)) continue;
        const Detection& det = detections[di];
        const double p = detection_likelihood(det);
        Track tr;
        tr.id = next_id_++;
        tr.x = det.x;
        tr.y = det.y;
        tr.freq_hz = det.freq_hz;
        tr.rpm = det.rpm;
        tr.pixels = det.pixels;
        tr.bbox = det.bbox;
        tr.freq_cv = det.freq_cv;
        tr.hits = 1;
        tr.age = 0;
        tr.avg_p = p;
        tr.confidence = p;
        tracks_.push_back(tr);
    }

    // Age unmatched tracks; decay their confidence.
    for (int ti = 0; ti < static_cast<int>(tracks_.size()); ++ti) {
        if (!matched_tracks.count(ti)) {
            tracks_[ti].age += 1;
            tracks_[ti].confidence *= 0.8;
        }
    }

    // Prune dead tracks.
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [this](const Track& t) { return t.age > max_age_; }),
                  tracks_.end());

    // Return confirmed tracks (Bayesian confidence AND minimum hits).
    std::vector<Track> confirmed;
    for (const Track& t : tracks_) {
        if (t.confidence >= confidence_threshold_ && t.hits >= min_hits_)
            confirmed.push_back(t);
    }
    std::sort(confirmed.begin(), confirmed.end(),
              [](const Track& a, const Track& b) { return a.pixels > b.pixels; });
    return confirmed;
}

}  // namespace ndrone
