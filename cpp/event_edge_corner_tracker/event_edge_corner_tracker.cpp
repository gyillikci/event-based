/**********************************************************************************************************************
 * event_edge_corner_tracker - real-time C++ tracking of EDGES and CORNERS from an event stream.
 *
 * Two geometric primitives are extracted from the same Surface of Active Events (SAE, a
 * per-pixel "time surface" holding the timestamp of the last event):
 *
 *   CORNERS  eFAST / Arc* (Mueggler 2017; Alzugaray & Chli 2018). For each incoming event,
 *            inspect the 16-pixel Bresenham circle (radius 3) of the same-polarity SAE. The
 *            pixel is a corner if the newest timestamps form a CONTIGUOUS ARC whose length is
 *            in [3,6] and whose minimum exceeds the maximum of the rest of the ring. The
 *            optional radius-4 ring (20 px, arc [4,8]) suppresses false positives on edges.
 *
 *   EDGES    Pixels whose SAE age is below a threshold form the "active edge" mask; a
 *            probabilistic Hough transform turns that mask into line segments. Segments are
 *            associated across slices by midpoint distance and orientation, which converts
 *            per-slice detections into persistent, identified edge tracks.
 *
 * Both primitive types are then tracked the same way: nearest-neighbour association under a
 * spatial (and, for edges, angular) gate, with a hit counter for confirmation and an age
 * counter for termination.
 *
 * REAL-TIME BEHAVIOUR is a first-class concern here, because an event camera trivially
 * out-produces any host-side consumer in a bright scene. Two mechanisms guard it:
 *   * --erc-rate arms the sensor's hardware event-rate controller, capping events/s at the
 *     source. Measured on this workspace's EVK4: above ~15 Mev/s a Python consumer's display
 *     drift grew ~348 ms/s without bound, while a capped 8 Mev/s stream held drift flat.
 *   * the --stats line reports the DRIFT between wall-clock and stream time. A drift that
 *     keeps growing means the pipeline is slower than the source; a drift that stays constant
 *     means it is keeping up, whatever its absolute value.
 *
 * Input: live camera or .raw/.hdf5 file. Output: CSV track updates and/or a live overlay.
 **********************************************************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/file_config_hints.h>
#include <metavision/hal/facilities/i_erc_module.h>
#include <metavision/hal/facilities/i_ll_biases.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// ---------------------------------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------------------------------

class Args {
public:
    Args(int argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a.rfind("--", 0) != 0) continue;
            const std::string key = a.substr(2);
            if (i + 1 < argc) {
                const std::string next = argv[i + 1];
                if (next.rfind("--", 0) != 0) {
                    kv_[key] = next;
                    ++i;
                    continue;
                }
            }
            kv_[key] = "1";
        }
    }

    bool has(const std::string &k) const { return kv_.count(k) != 0; }
    std::string str(const std::string &k, const std::string &d = "") const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : it->second;
    }
    long long num(const std::string &k, long long d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stoll(it->second);
    }
    double real(const std::string &k, double d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stod(it->second);
    }
    bool flag(const std::string &k) const { return kv_.count(k) != 0; }

private:
    std::unordered_map<std::string, std::string> kv_;
};

// ---------------------------------------------------------------------------------------------------------------
// Surface of Active Events + eFAST / Arc* corner test
// ---------------------------------------------------------------------------------------------------------------

// Bresenham circle of radius 3 (16 px) and radius 4 (20 px), in contiguous ring order.
const int kRing3[16][2] = {{0, 3},  {1, 3},   {2, 2},   {3, 1},   {3, 0},   {3, -1},  {2, -2},  {1, -3},
                           {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0},  {-3, 1},  {-2, 2},  {-1, 3}};
const int kRing4[20][2] = {{0, 4},   {1, 4},   {2, 3},   {3, 2},   {4, 1},   {4, 0},  {4, -1}, {3, -2}, {2, -3}, {1, -4},
                           {0, -4},  {-1, -4}, {-2, -3}, {-3, -2}, {-4, -1}, {-4, 0}, {-4, 1}, {-3, 2}, {-2, 3}, {-1, 4}};

/// True if some contiguous arc of length in [lo,hi] has a minimum timestamp strictly greater
/// than the maximum over the rest of the ring. Sliding min/max via monotonic deques keeps this
/// O(N) per arc length instead of the naive O(N^2), which matters at Mev/s rates.
bool arc_is_corner(const int64_t *ring, int n, int lo, int hi) {
    int64_t dbl[40];
    for (int i = 0; i < 2 * n; ++i) dbl[i] = ring[i % n];

    for (int len = lo; len <= hi; ++len) {
        const int comp = n - len; // complement length
        if (comp <= 0) continue;

        // Sliding minimum over windows of size `len`.
        int64_t wmin[40];
        int dq[40], head = 0, tail = 0;
        for (int i = 0; i < 2 * n; ++i) {
            while (tail > head && dbl[dq[tail - 1]] >= dbl[i]) --tail;
            dq[tail++] = i;
            if (dq[head] <= i - len) ++head;
            if (i >= len - 1) wmin[i - len + 1] = dbl[dq[head]];
        }

        // Sliding maximum over windows of size `comp`.
        int64_t wmax[40];
        head = tail = 0;
        for (int i = 0; i < 2 * n; ++i) {
            while (tail > head && dbl[dq[tail - 1]] <= dbl[i]) --tail;
            dq[tail++] = i;
            if (dq[head] <= i - comp) ++head;
            if (i >= comp - 1) wmax[i - comp + 1] = dbl[dq[head]];
        }

        for (int s = 0; s < n; ++s) {
            if (wmin[s] > wmax[s + len]) return true; // arc [s,s+len) newer than the rest
        }
    }
    return false;
}

class TimeSurface {
public:
    TimeSurface(int width, int height) : w_(width), h_(height) {
        for (int p = 0; p < 2; ++p) sae_[p].assign(std::size_t(w_) * h_, std::numeric_limits<int64_t>::min() / 4);
    }

    void update(int x, int y, int p, int64_t t) { sae_[p][std::size_t(y) * w_ + x] = t; }

    int64_t at(int x, int y, int p) const { return sae_[p][std::size_t(y) * w_ + x]; }

    int64_t newest(int x, int y) const { return std::max(at(x, y, 0), at(x, y, 1)); }

    bool is_corner(int x, int y, int p, bool double_ring) const {
        if (x < 4 || y < 4 || x >= w_ - 4 || y >= h_ - 4) return false;

        int64_t ring[20];
        for (int i = 0; i < 16; ++i) ring[i] = at(x + kRing3[i][0], y + kRing3[i][1], p);
        if (!arc_is_corner(ring, 16, 3, 6)) return false;

        if (!double_ring) return true;
        for (int i = 0; i < 20; ++i) ring[i] = at(x + kRing4[i][0], y + kRing4[i][1], p);
        return arc_is_corner(ring, 20, 4, 8);
    }

    int width() const { return w_; }
    int height() const { return h_; }

private:
    int w_, h_;
    std::vector<int64_t> sae_[2];
};

// ---------------------------------------------------------------------------------------------------------------
// Corner tracks
// ---------------------------------------------------------------------------------------------------------------

struct CornerTrack {
    int id = 0;
    float x = 0.f, y = 0.f;
    int hits = 0;
    int age = 0; // slices since the last association
    int64_t last_t = 0;
};

class CornerTracker {
public:
    CornerTracker(float gate_px, int max_age, int min_hits, float smoothing)
        : gate_(gate_px), max_age_(max_age), min_hits_(min_hits), alpha_(smoothing) {}

    void update(const std::vector<cv::Point2f> &detections, int64_t t) {
        std::vector<bool> used(detections.size(), false);

        for (auto &tr : tracks_) {
            int best = -1;
            float best_d2 = gate_ * gate_;
            for (std::size_t i = 0; i < detections.size(); ++i) {
                if (used[i]) continue;
                const float dx = detections[i].x - tr.x, dy = detections[i].y - tr.y;
                const float d2 = dx * dx + dy * dy;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best = int(i);
                }
            }
            if (best >= 0) {
                used[best] = true;
                tr.x += alpha_ * (detections[best].x - tr.x);
                tr.y += alpha_ * (detections[best].y - tr.y);
                ++tr.hits;
                tr.age = 0;
                tr.last_t = t;
            } else {
                ++tr.age;
            }
        }

        for (std::size_t i = 0; i < detections.size(); ++i) {
            if (used[i]) continue;
            CornerTrack tr;
            tr.id = next_id_++;
            tr.x = detections[i].x;
            tr.y = detections[i].y;
            tr.hits = 1;
            tr.last_t = t;
            tracks_.push_back(tr);
        }

        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [this](const CornerTrack &tr) { return tr.age > max_age_; }),
                      tracks_.end());
    }

    std::vector<CornerTrack> confirmed() const {
        std::vector<CornerTrack> out;
        for (const auto &tr : tracks_)
            if (tr.hits >= min_hits_) out.push_back(tr);
        return out;
    }

private:
    float gate_;
    int max_age_, min_hits_;
    float alpha_;
    int next_id_ = 1;
    std::vector<CornerTrack> tracks_;
};

// ---------------------------------------------------------------------------------------------------------------
// Edge tracks
// ---------------------------------------------------------------------------------------------------------------

struct EdgeTrack {
    int id = 0;
    cv::Point2f a, b;   // endpoints
    cv::Point2f mid;
    float angle = 0.f;  // [0, pi)
    int hits = 0;
    int age = 0;
    int64_t last_t = 0;
};

/// Angular distance on the half-circle: orientations differing by pi are the same line.
float angle_delta(float a, float b) {
    float d = std::fabs(a - b);
    if (d > float(CV_PI) / 2.f) d = float(CV_PI) - d;
    return d;
}

class EdgeTracker {
public:
    EdgeTracker(float gate_px, float angle_gate_rad, int max_age, int min_hits, float smoothing)
        : gate_(gate_px), angle_gate_(angle_gate_rad), max_age_(max_age), min_hits_(min_hits), alpha_(smoothing) {}

    void update(const std::vector<cv::Vec4i> &segments, int64_t t) {
        struct Det {
            cv::Point2f a, b, mid;
            float angle;
        };
        std::vector<Det> dets;
        dets.reserve(segments.size());
        for (const auto &s : segments) {
            Det d;
            d.a = cv::Point2f(float(s[0]), float(s[1]));
            d.b = cv::Point2f(float(s[2]), float(s[3]));
            d.mid = (d.a + d.b) * 0.5f;
            float ang = std::atan2(d.b.y - d.a.y, d.b.x - d.a.x);
            if (ang < 0.f) ang += float(CV_PI);
            d.angle = ang;
            dets.push_back(d);
        }

        std::vector<bool> used(dets.size(), false);

        for (auto &tr : tracks_) {
            int best = -1;
            float best_d2 = gate_ * gate_;
            for (std::size_t i = 0; i < dets.size(); ++i) {
                if (used[i]) continue;
                if (angle_delta(dets[i].angle, tr.angle) > angle_gate_) continue;
                const float dx = dets[i].mid.x - tr.mid.x, dy = dets[i].mid.y - tr.mid.y;
                const float d2 = dx * dx + dy * dy;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best = int(i);
                }
            }
            if (best >= 0) {
                used[best] = true;
                const Det &d = dets[best];
                tr.a = tr.a + alpha_ * (d.a - tr.a);
                tr.b = tr.b + alpha_ * (d.b - tr.b);
                tr.mid = (tr.a + tr.b) * 0.5f;
                tr.angle += alpha_ * (d.angle - tr.angle);
                ++tr.hits;
                tr.age = 0;
                tr.last_t = t;
            } else {
                ++tr.age;
            }
        }

        for (std::size_t i = 0; i < dets.size(); ++i) {
            if (used[i]) continue;
            EdgeTrack tr;
            tr.id = next_id_++;
            tr.a = dets[i].a;
            tr.b = dets[i].b;
            tr.mid = dets[i].mid;
            tr.angle = dets[i].angle;
            tr.hits = 1;
            tr.last_t = t;
            tracks_.push_back(tr);
        }

        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [this](const EdgeTrack &tr) { return tr.age > max_age_; }),
                      tracks_.end());
    }

    std::vector<EdgeTrack> confirmed() const {
        std::vector<EdgeTrack> out;
        for (const auto &tr : tracks_)
            if (tr.hits >= min_hits_) out.push_back(tr);
        return out;
    }

private:
    float gate_, angle_gate_;
    int max_age_, min_hits_;
    float alpha_;
    int next_id_ = 1;
    std::vector<EdgeTrack> tracks_;
};

// ---------------------------------------------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------------------------------------------

struct Config {
    int64_t slice_dt_us = 10000;
    int64_t edge_tau_us = 20000;
    int corner_stride = 1;
    bool double_ring = false;
    int hough_threshold = 30;
    int hough_min_length = 25;
    int hough_max_gap = 6;
    float corner_gate = 12.f;
    int corner_max_age = 4;
    int corner_min_hits = 3;
    float edge_gate = 40.f;
    float edge_angle_gate = 0.25f;
    int edge_max_age = 4;
    int edge_min_hits = 3;
    float smoothing = 0.5f;
    bool display = true;
    double stats_period_s = 1.0;
};

class Pipeline {
public:
    Pipeline(int width, int height, const Config &cfg, std::ostream *csv)
        : cfg_(cfg), sae_(width, height), fresh_(height, width, uint8_t(0)),
          corners_(cfg.corner_gate, cfg.corner_max_age, cfg.corner_min_hits, cfg.smoothing),
          edges_(cfg.edge_gate, cfg.edge_angle_gate, cfg.edge_max_age, cfg.edge_min_hits, cfg.smoothing),
          csv_(csv) {}

    void process(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            const int64_t t = int64_t(ev->t);
            sae_.update(ev->x, ev->y, ev->p, t);

            if (++event_counter_ % cfg_.corner_stride == 0) {
                if (sae_.is_corner(ev->x, ev->y, ev->p, cfg_.double_ring))
                    slice_corners_.emplace_back(float(ev->x), float(ev->y));
            }

            if (slice_start_us_ < 0) slice_start_us_ = t;
            last_t_us_ = t;
            ++events_in_slice_;

            if (t - slice_start_us_ >= cfg_.slice_dt_us) {
                finish_slice(t);
                slice_start_us_ = t;
            }
        }
    }

    /// Copies the latest overlay for the main thread to show. Returns false if none is ready.
    bool take_frame(cv::Mat &out) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (frame_.empty()) return false;
        frame_.copyTo(out);
        return true;
    }

private:
    void finish_slice(int64_t t) {
        // Active-edge mask: pixels whose most recent event is younger than edge_tau.
        const int w = sae_.width(), h = sae_.height();
        for (int y = 0; y < h; ++y) {
            uint8_t *row = fresh_.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x) row[x] = (t - sae_.newest(x, y) <= cfg_.edge_tau_us) ? 255 : 0;
        }

        std::vector<cv::Vec4i> segments;
        cv::HoughLinesP(fresh_, segments, 1.0, CV_PI / 180.0, cfg_.hough_threshold, double(cfg_.hough_min_length),
                        double(cfg_.hough_max_gap));

        corners_.update(slice_corners_, t);
        edges_.update(segments, t);
        slice_corners_.clear();

        const auto ctracks = corners_.confirmed();
        const auto etracks = edges_.confirmed();

        if (csv_) {
            for (const auto &tr : ctracks)
                *csv_ << "corner," << tr.id << ',' << t << ',' << tr.x << ',' << tr.y << ",,," << tr.hits << '\n';
            for (const auto &tr : etracks)
                *csv_ << "edge," << tr.id << ',' << t << ',' << tr.a.x << ',' << tr.a.y << ',' << tr.b.x << ','
                      << tr.b.y << ',' << tr.hits << '\n';
        }

        if (cfg_.display) render(ctracks, etracks);
        report(t);
    }

    void render(const std::vector<CornerTrack> &ctracks, const std::vector<EdgeTrack> &etracks) {
        cv::Mat bgr;
        cv::cvtColor(fresh_, bgr, cv::COLOR_GRAY2BGR);
        bgr *= 0.35; // dim the raw events so the overlays stand out

        for (const auto &tr : etracks) {
            cv::line(bgr, cv::Point(tr.a), cv::Point(tr.b), cv::Scalar(255, 160, 0), 2);
            cv::putText(bgr, "E" + std::to_string(tr.id), cv::Point(tr.mid), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(255, 200, 120), 1);
        }
        for (const auto &tr : ctracks) {
            cv::circle(bgr, cv::Point(int(tr.x), int(tr.y)), 5, cv::Scalar(0, 255, 0), 1);
            cv::putText(bgr, "C" + std::to_string(tr.id), cv::Point(int(tr.x) + 6, int(tr.y) - 6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        }
        cv::putText(bgr, "corners:" + std::to_string(ctracks.size()) + "  edges:" + std::to_string(etracks.size()),
                    cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        std::lock_guard<std::mutex> lock(frame_mutex_);
        frame_ = bgr;
    }

    /// Drift = wall time elapsed minus stream time elapsed. Constant => keeping up.
    void report(int64_t t) {
        if (cfg_.stats_period_s <= 0) return;
        const auto now = std::chrono::steady_clock::now();
        const auto period_dur =
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(cfg_.stats_period_s));
        if (first_report_) {
            wall_start_ = now;
            stream_start_us_ = t;
            next_report_ = now + period_dur;
            first_report_ = false;
            events_since_report_ = 0;
        }
        events_since_report_ += events_in_slice_;
        events_in_slice_ = 0;

        if (now < next_report_) return;
        const double wall = std::chrono::duration<double>(now - wall_start_).count();
        const double stream = double(t - stream_start_us_) / 1e6;
        const double period = std::chrono::duration<double>(now - next_report_).count() + cfg_.stats_period_s;
        std::cout << "in " << std::fixed << std::setprecision(2) << (events_since_report_ / period / 1e6)
                  << " Mev/s   drift " << std::setprecision(0) << ((wall - stream) * 1e3) << " ms\n"
                  << std::flush;
        events_since_report_ = 0;
        next_report_ = now + period_dur;
    }

    Config cfg_;
    TimeSurface sae_;
    cv::Mat1b fresh_;
    CornerTracker corners_;
    EdgeTracker edges_;
    std::ostream *csv_;

    std::vector<cv::Point2f> slice_corners_;
    int64_t slice_start_us_ = -1;
    int64_t last_t_us_ = 0;
    unsigned long long event_counter_ = 0;

    std::mutex frame_mutex_;
    cv::Mat frame_;

    bool first_report_ = true;
    std::chrono::steady_clock::time_point wall_start_, next_report_;
    int64_t stream_start_us_ = 0;
    unsigned long long events_since_report_ = 0, events_in_slice_ = 0;
};

void print_help() {
    std::cout
        << "event_edge_corner_tracker - real-time edge and corner tracking from events\n\n"
        << "Input:\n"
        << "  --input PATH           RAW/HDF5 file. Default: first available live camera.\n\n"
        << "Sensor throughput (live camera only):\n"
        << "  --erc-rate N           Hardware event-rate cap in ev/s (0 = off). Default 8000000.\n"
        << "  --bias NAME=VALUE      Pixel bias, repeatable via comma list, e.g. bias_hpf=30,bias_diff_on=120.\n\n"
        << "Processing:\n"
        << "  --slice-dt-us N        Slice length driving detection/tracking updates. Default 10000.\n"
        << "  --edge-tau-us N        Max SAE age for a pixel to join the active-edge mask. Default 20000.\n"
        << "  --corner-stride N      Run the corner test on every Nth event. Default 1.\n"
        << "  --double-ring          Also require the radius-4 arc test (fewer false corners).\n"
        << "  --hough-threshold N    Hough accumulator threshold. Default 30.\n"
        << "  --hough-min-length N   Minimum segment length in px. Default 25.\n"
        << "  --hough-max-gap N      Maximum gap joined into one segment. Default 6.\n\n"
        << "Tracking:\n"
        << "  --corner-gate PX       Association radius for corners. Default 12.\n"
        << "  --edge-gate PX         Midpoint association radius for edges. Default 40.\n"
        << "  --edge-angle-gate RAD  Orientation gate for edges. Default 0.25.\n"
        << "  --max-age N            Slices without association before a track dies. Default 4.\n"
        << "  --min-hits N           Associations before a track is reported. Default 3.\n\n"
        << "Output:\n"
        << "  --csv PATH             Write track updates as CSV.\n"
        << "  --no-display           Headless.\n"
        << "  --stats-period S       Seconds between throughput/drift lines (0 = off). Default 1.\n";
}

} // namespace

int main(int argc, char **argv) {
    const Args args(argc, argv);
    if (args.flag("help") || args.flag("h")) {
        print_help();
        return 0;
    }

    Config cfg;
    cfg.slice_dt_us = args.num("slice-dt-us", cfg.slice_dt_us);
    cfg.edge_tau_us = args.num("edge-tau-us", cfg.edge_tau_us);
    cfg.corner_stride = std::max<int>(1, int(args.num("corner-stride", cfg.corner_stride)));
    cfg.double_ring = args.flag("double-ring");
    cfg.hough_threshold = int(args.num("hough-threshold", cfg.hough_threshold));
    cfg.hough_min_length = int(args.num("hough-min-length", cfg.hough_min_length));
    cfg.hough_max_gap = int(args.num("hough-max-gap", cfg.hough_max_gap));
    cfg.corner_gate = float(args.real("corner-gate", cfg.corner_gate));
    cfg.edge_gate = float(args.real("edge-gate", cfg.edge_gate));
    cfg.edge_angle_gate = float(args.real("edge-angle-gate", cfg.edge_angle_gate));
    cfg.corner_max_age = cfg.edge_max_age = int(args.num("max-age", cfg.corner_max_age));
    cfg.corner_min_hits = cfg.edge_min_hits = int(args.num("min-hits", cfg.corner_min_hits));
    cfg.display = !args.flag("no-display");
    cfg.stats_period_s = args.real("stats-period", cfg.stats_period_s);

    const std::string input = args.str("input");
    const long long erc_rate = args.num("erc-rate", 8000000);

    Metavision::Camera camera;
    try {
        if (input.empty()) {
            camera = Metavision::Camera::from_first_available();
        } else {
            camera = Metavision::Camera::from_file(input, Metavision::FileConfigHints().real_time_playback(true));
        }
    } catch (const std::exception &e) {
        std::cerr << "Camera init failed: " << e.what() << "\n";
        return 1;
    }

    const bool is_live = input.empty();
    const int width = camera.geometry().width();
    const int height = camera.geometry().height();

    bool erc_on = false;
    if (is_live && erc_rate > 0) {
        try {
            auto *erc = camera.get_device().get_facility<Metavision::I_ErcModule>();
            if (erc) {
                erc->set_cd_event_rate(uint32_t(erc_rate));
                erc->enable(true);
                erc_on = true;
            }
        } catch (const std::exception &) { /* not available on this source */ }
    }

    if (is_live && args.has("bias")) {
        try {
            auto *biases = camera.get_device().get_facility<Metavision::I_LL_Biases>();
            std::string spec = args.str("bias");
            std::size_t pos = 0;
            while (pos <= spec.size() && biases) {
                const std::size_t comma = spec.find(',', pos);
                const std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                const std::size_t eq = item.find('=');
                if (eq != std::string::npos) {
                    const std::string name = item.substr(0, eq);
                    biases->set(name, std::stoi(item.substr(eq + 1)));
                    std::cout << "bias " << name << " = " << item.substr(eq + 1) << "\n";
                }
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } catch (const std::exception &e) {
            std::cerr << "bias setup failed: " << e.what() << "\n";
        }
    }

    std::ofstream csv;
    if (args.has("csv")) {
        csv.open(args.str("csv"));
        if (csv) csv << "type,id,t_us,x1,y1,x2,y2,hits\n";
        else std::cerr << "warning: could not open CSV '" << args.str("csv") << "'\n";
    }

    Pipeline pipeline(width, height, cfg, csv.is_open() ? &csv : nullptr);

    std::cout << "event_edge_corner_tracker\n"
              << "  source : " << (is_live ? "LIVE CAMERA" : input) << "\n"
              << "  sensor : " << width << "x" << height << "\n"
              << "  erc    : " << (erc_on ? std::to_string(erc_rate) + " ev/s" : std::string("off")) << "\n"
              << "  slice  : " << cfg.slice_dt_us << " us   edge tau " << cfg.edge_tau_us << " us\n"
              << "  corners: stride " << cfg.corner_stride << (cfg.double_ring ? ", double ring" : ", single ring")
              << "\n"
              << (cfg.display ? "  press q or ESC in the window to stop\n" : "  headless\n")
              << std::flush;

    camera.cd().add_callback(
        [&pipeline](const Metavision::EventCD *begin, const Metavision::EventCD *end) { pipeline.process(begin, end); });

    camera.start();

    const std::string win = "Edge + Corner Tracker";
    if (cfg.display) cv::namedWindow(win, cv::WINDOW_NORMAL);

    cv::Mat frame;
    while (camera.is_running()) {
        if (cfg.display) {
            if (pipeline.take_frame(frame)) cv::imshow(win, frame);
            const int key = cv::waitKey(1) & 0xff;
            if (key == 'q' || key == 27) break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    camera.stop();
    if (cfg.display) cv::destroyAllWindows();
    if (csv.is_open()) csv.close();
    std::cout << "Done.\n";
    return 0;
}
