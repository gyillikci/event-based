/**********************************************************************************************************************
 * event_feature_tracker - native C++ event-based corner FRONT-END for tracking / SLAM.
 *
 * A deterministic replacement for the "learned corners" front-end (which needs trained
 * weights not shipped here). It implements the well-established event corner detector
 * on the Surface of Active Events (time surface):
 *
 *   eFAST / Arc* (Mueggler 2017; Alzugaray & Chli 2018): for each incoming event,
 *   inspect the 16-pixel Bresenham circle (radius 3) of the same-polarity SAE. The
 *   pixel is a corner if the newest events form a CONTIGUOUS ARC of length in [3,6]
 *   (or its complement) whose timestamps are all greater than the rest of the ring.
 *
 * Detected corners are linked into tracks by nearest-neighbour association (spatial
 * gate + temporal gate), giving the point tracks a VO/SLAM front-end consumes.
 *
 * Input: live camera or .raw/.hdf5. Output: a CSV of track updates (track_id,t_us,x,y)
 * and/or a live overlay. Headless unless --display/--output set.
 **********************************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/file_config_hints.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

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
                    if (!next.empty() && next[0] != '-') {
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
                    if (!next.empty() && next[0] != '-') {
                        kv_[key] = next;
                        ++i;
                        continue;
                    }
                }
                flags_.insert(key);
            }
        }
    }
    bool has(const std::string &k) const { return kv_.count(k) != 0 || flags_.count(k) != 0; }
    std::string get(const std::string &k, const std::string &d = "") const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : it->second;
    }
    int geti(const std::string &k, int d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stoi(it->second);
    }
    long long getll(const std::string &k, long long d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stoll(it->second);
    }
    double getd(const std::string &k, double d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stod(it->second);
    }
    bool getb(const std::string &k, bool d) const {
        if (flags_.count(k) != 0) return true;
        auto it = kv_.find(k);
        if (it == kv_.end()) return d;
        return it->second == "1" || it->second == "true" || it->second == "on" || it->second == "yes";
    }

private:
    std::map<std::string, std::string> kv_;
    std::set<std::string> flags_;
};

void print_usage() {
    std::cout <<
        "event_feature_tracker - native C++ event corner front-end (eFAST/Arc*) + tracking\n\n"
        "Deterministic corner detection on the time surface + nearest-neighbour tracking.\n\n"
        "Input:\n"
        "  -i, --input <file>          .raw/.hdf5 recording (omit for a live camera)\n\n"
        "Detection:\n"
        "      --arc-min <n>           min contiguous newest-arc length (default 3)\n"
        "      --arc-max <n>           max contiguous newest-arc length (default 6)\n"
        "      --sae-decay-us <us>     ignore ring pixels older than this (default 50000)\n\n"
        "Tracking:\n"
        "      --match-radius <px>     spatial gate to extend a track (default 4)\n"
        "      --track-timeout-us <us> drop a track after this idle time (default 100000)\n"
        "      --ema <a>               position smoothing 0..1 (default 0.5)\n"
        "      --refractory-us <us>    per-pixel activity dead-time (default 1000)\n"
        "      --max-tracks <n>        cap active tracks (default 4000)\n\n"
        "Eclipse / illumination transition monitor (no tracking, cannot saturate):\n"
        "      --illum-monitor         detect global light ON/OFF (sunrise/eclipse) only\n"
        "      --illum-window-us <us>  polarity integration window (default 10000)\n"
        "      --illum-balance <b>     min |ON-OFF|/total for a flip 0..1 (default 0.5)\n"
        "      --illum-min-events <n>  min events in a window to evaluate (default 500)\n"
        "      --illum-hold-us <us>    min dwell before flipping back (default 300000)\n"
        "      --illum-decay-us <us>   event-view persistence in the panel (default 30000)\n\n"
        "Illumination robustness (common-mode glare guard):\n"
        "      --glare-guard           reject sudden-illumination slices + hot regions\n"
        "      --slice-dt-us <us>      guard slice cadence (default 20000)\n"
        "      --cell-size <px>        common-mode cell size (default 16)\n"
        "      --cm-index-threshold <v> contamination threshold 0..1 (default 0.55)\n"
        "      --cm-dispersion-scale <v> balance-dispersion scale (default 0.5)\n"
        "      --mask-rate-sigma <s>   hot-cell density sigma (default 6)\n"
        "      --mask-persistence <n>  slices before a cell is masked (default 4)\n"
        "      --mask-release <n>      hit decay per quiet slice (default 2)\n"
        "      --mask-halo-px <px>     dilate mask by this halo (default 48)\n\n"
        "Output:\n"
        "  -o, --output <file>         CSV track updates track_id,t_us,x,y\n"
        "      --display               show a live corner/track overlay\n"
        "      --max-duration <s>      stop after s seconds (0 = unlimited)\n"
        "  -h, --help                  this message\n";
}

// 16-pixel Bresenham circle, radius 3 (clockwise from top), for eFAST/Arc*.
constexpr int kRing = 16;
const int kCircle[kRing][2] = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
    {0, 3},  {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

struct Track {
    int id;
    float x;
    float y;
    Metavision::timestamp t_last;
    long long n;
    bool active;
    int gcell;
};

// --------------------------------------------------------------------------- //
// Common-mode glare guard (median-mode) + persistent hot-region mask.
// Ported from the validated solar-array monitor (glare_guard.py / solar_array_
// monitor.cpp). A sudden global illumination change (orbital sunrise, glare sweep)
// makes every pixel fire at once, flooding the time surface with uniformly fresh
// timestamps and destroying the arc contrast the corner test relies on. This guard
// detects that surge (per-cell polarity balance is globally coherent -> high
// common-mode index) and flags the slice as contaminated, and separately masks a
// persistently over-bright region so a moving glare patch is excluded even when the
// slice as a whole is not contaminated.
// --------------------------------------------------------------------------- //
class GlareGuard {
public:
    GlareGuard(int width, int height, const Args &a) : width_(width), height_(height) {
        cell_ = std::max(1, a.geti("cell-size", 16));
        n_cols_ = (width_ + cell_ - 1) / cell_;
        n_rows_ = (height_ + cell_ - 1) / cell_;
        min_events_per_cell_ = a.geti("min-events-per-cell", 4);
        index_threshold_ = a.getd("cm-index-threshold", 0.55);
        dispersion_scale_ = a.getd("cm-dispersion-scale", 0.5);
        rate_sigma_ = a.getd("mask-rate-sigma", 6.0);
        persistence_ = a.geti("mask-persistence", 4);
        release_ = a.geti("mask-release", 2);
        halo_cells_ = std::max(0, static_cast<int>(std::lround(a.getd("mask-halo-px", 48.0) / cell_)));
        on_.assign(static_cast<size_t>(n_rows_) * n_cols_, 0);
        off_.assign(static_cast<size_t>(n_rows_) * n_cols_, 0);
        px_map_.assign(static_cast<size_t>(n_rows_) * n_cols_, 0);
        for (int r = 0; r < n_rows_; ++r)
            for (int c = 0; c < n_cols_; ++c) {
                const int ph = std::min((r + 1) * cell_, height_) - r * cell_;
                const int pw = std::min((c + 1) * cell_, width_) - c * cell_;
                px_map_[static_cast<size_t>(r) * n_cols_ + c] = ph * pw;
            }
        hits_.assign(static_cast<size_t>(n_rows_) * n_cols_, 0);
        mask_.assign(static_cast<size_t>(n_rows_) * n_cols_, 0);
        mask_tmp_.assign(static_cast<size_t>(n_rows_) * n_cols_, 0);
        nominal_px_ = cell_ * cell_;
    }

    // Analyse one accumulated slice; updates contaminated flag and the hot mask.
    void analyze(const std::vector<Metavision::EventCD> &ev) {
        std::fill(on_.begin(), on_.end(), 0);
        std::fill(off_.begin(), off_.end(), 0);
        for (const auto &e : ev) {
            const size_t ci = static_cast<size_t>(e.y / cell_) * n_cols_ + (e.x / cell_);
            if (e.p) ++on_[ci]; else ++off_[ci];
        }
        double balance = 0.0, dispersion = 0.0;
        int valid = 0, judged = 0;
        balances_.clear();
        for (size_t i = 0; i < on_.size(); ++i) {
            const long tot   = on_[i] + off_[i];
            const bool msk   = mask_[i] != 0;
            if (!msk) ++judged;
            if (tot >= min_events_per_cell_ && !msk) {
                ++valid;
                balances_.push_back((on_[i] - off_[i]) / static_cast<double>(tot));
            }
        }
        if (!balances_.empty()) {
            balance = median(balances_);
            devs_.clear();
            for (double b : balances_) devs_.push_back(std::fabs(b - balance));
            dispersion = median(devs_);
        }
        const double coverage   = judged ? static_cast<double>(valid) / judged : 0.0;
        const double uniformity = 1.0 - std::min(1.0, dispersion / std::max(dispersion_scale_, 1e-9));
        cm_index_     = std::fabs(balance) * uniformity * coverage;
        contaminated_ = cm_index_ >= index_threshold_;
        update_mask();
    }

    bool contaminated() const { return contaminated_; }
    double cm_index() const { return cm_index_; }
    bool masked(int x, int y) const {
        return mask_[static_cast<size_t>(y / cell_) * n_cols_ + (x / cell_)] != 0;
    }
    int cell() const { return cell_; }
    int n_cols() const { return n_cols_; }
    int n_rows() const { return n_rows_; }
    int n_masked_cells() const {
        int n = 0;
        for (auto m : mask_) n += (m != 0);
        return n;
    }
    const std::vector<uint8_t> &mask() const { return mask_; }

private:
    static double median(std::vector<double> &v) {
        if (v.empty()) return 0.0;
        const size_t n = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + n, v.end());
        double m = v[n];
        if (v.size() % 2 == 0) {
            const double lo = *std::max_element(v.begin(), v.begin() + n);
            m = 0.5 * (m + lo);
        }
        return m;
    }

    void update_mask() {
        dens_.clear();
        for (size_t i = 0; i < on_.size(); ++i)
            if (px_map_[i] >= 0.5 * nominal_px_)
                dens_.push_back((on_[i] + off_[i]) / static_cast<double>(std::max<long>(1, px_map_[i])));
        const double med = dens_.empty() ? 0.0 : median(dens_);
        devs_.clear();
        for (double d : dens_) devs_.push_back(std::fabs(d - med));
        const double mad   = devs_.empty() ? 0.0 : median(devs_);
        const double sigma = std::max(1.4826 * mad, 1e-3);
        const double thr   = med + rate_sigma_ * sigma;
        for (size_t i = 0; i < on_.size(); ++i) {
            const bool eligible = px_map_[i] >= 0.5 * nominal_px_;
            const double d = (on_[i] + off_[i]) / static_cast<double>(std::max<long>(1, px_map_[i]));
            const bool hot = eligible && d > thr;
            hits_[i] = hot ? hits_[i] + 1 : std::max(0, hits_[i] - release_);
        }
        for (size_t i = 0; i < mask_.size(); ++i) mask_tmp_[i] = (hits_[i] >= persistence_) ? 1 : 0;
        box_dilate(mask_tmp_, mask_, n_rows_, n_cols_, halo_cells_);
    }

    static void box_dilate(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int rows, int cols,
                           int radius) {
        if (radius <= 0) { dst = src; return; }
        std::vector<uint8_t> tmp(src);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                uint8_t v = 0;
                for (int dc = -radius; dc <= radius && !v; ++dc) {
                    const int cc = c + dc;
                    if (cc >= 0 && cc < cols && src[static_cast<size_t>(r) * cols + cc]) v = 1;
                }
                tmp[static_cast<size_t>(r) * cols + c] = v;
            }
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                uint8_t v = 0;
                for (int dr = -radius; dr <= radius && !v; ++dr) {
                    const int rr = r + dr;
                    if (rr >= 0 && rr < rows && tmp[static_cast<size_t>(rr) * cols + c]) v = 1;
                }
                dst[static_cast<size_t>(r) * cols + c] = v;
            }
    }

    int width_, height_, cell_, n_cols_, n_rows_;
    int min_events_per_cell_, persistence_, release_, halo_cells_, nominal_px_;
    double index_threshold_, dispersion_scale_, rate_sigma_;
    bool contaminated_ = false;
    double cm_index_ = 0.0;
    std::vector<long> on_, off_, px_map_;
    std::vector<int> hits_;
    std::vector<uint8_t> mask_, mask_tmp_;
    std::vector<double> balances_, devs_, dens_;
};

} // namespace

int main(int argc, char **argv) {
    Args args(argc, argv);
    if (args.getb("help", false) || args.getb("h", false)) {
        print_usage();
        return 0;
    }

    const std::string input   = !args.get("input").empty() ? args.get("input") : args.get("i");
    const std::string out_csv = !args.get("output").empty() ? args.get("output") : args.get("o");

    const int arc_min                   = args.geti("arc-min", 3);
    const int arc_max                   = args.geti("arc-max", 6);
    const Metavision::timestamp sae_decay = args.getll("sae-decay-us", 50000);
    const float match_radius            = static_cast<float>(args.getd("match-radius", 4.0));
    const float match_radius2           = match_radius * match_radius;
    const Metavision::timestamp track_to = args.getll("track-timeout-us", 100000);
    const float ema                     = static_cast<float>(args.getd("ema", 0.5));
    const bool display                  = args.getb("display", false);
    const double max_dur                = args.getd("max-duration", 0.0);

    const bool guard_on                 = args.getb("glare-guard", false);
    const Metavision::timestamp slice_dt = args.getll("slice-dt-us", 20000);
    const Metavision::timestamp refractory_us = args.getll("refractory-us", 1000);
    const int max_tracks                = args.geti("max-tracks", 4000);

    const bool illum_on                 = args.getb("illum-monitor", false);
    const Metavision::timestamp illum_win = args.getll("illum-window-us", 10000);
    const double illum_balance          = args.getd("illum-balance", 0.5);
    const long long illum_min_events    = args.getll("illum-min-events", 500);
    const Metavision::timestamp illum_hold = args.getll("illum-hold-us", 300000);
    const Metavision::timestamp illum_decay = args.getll("illum-decay-us", 30000);

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

    const int W = camera.geometry().width();
    const int H = camera.geometry().height();
    const size_t N = static_cast<size_t>(W) * H;

    // Surface of Active Events, one timestamp per pixel per polarity.
    std::vector<Metavision::timestamp> sae[2] = {
        std::vector<Metavision::timestamp>(N, std::numeric_limits<Metavision::timestamp>::min() / 2),
        std::vector<Metavision::timestamp>(N, std::numeric_limits<Metavision::timestamp>::min() / 2)};

    std::ofstream out;
    if (!out_csv.empty()) {
        out.open(out_csv);
        if (!out.is_open()) {
            std::cerr << "error: cannot open output '" << out_csv << "'\n";
            return 1;
        }
        out << (illum_on ? "t_us,type,total,on,off,balance\n" : "track_id,t_us,x,y\n");
    }

    std::vector<Track> pool;
    pool.reserve(static_cast<size_t>(std::max(1, max_tracks)));
    std::vector<int> free_slots;
    int active_count = 0;
    int peak_active = 0;
    int next_id = 0;
    long long n_corners = 0;
    Metavision::timestamp cur_t = 0;
    Metavision::timestamp last_purge = 0;

    // Illumination / eclipse transition monitor state. A global light change makes
    // almost every pixel fire the SAME polarity at once (balance -> +-1); ordinary
    // motion (a swinging hand) fires both polarities in roughly equal numbers
    // (balance ~ 0), so the polarity balance cleanly separates the two.
    long long il_on = 0, il_off = 0;             // running counts in the open window
    long long il_on_win = 0, il_off_win = 0;      // counts from the last closed window
    long long il_total_win = 0;
    double il_balance_win = 0.0, il_rate_win = 0.0;
    Metavision::timestamp il_win_start = -1;
    int il_state = 0;                             // 0=unknown, +1=bright, -1=dark
    long long il_sunrise = 0, il_eclipse = 0;
    Metavision::timestamp il_last_change = std::numeric_limits<Metavision::timestamp>::min() / 2;
    std::string il_last_label;
    Metavision::timestamp il_flash_until = 0;

    // Per-pixel latest event (time + polarity) so the display can render the live
    // event activity even in the dark - the reason for an event sensor.
    std::vector<Metavision::timestamp> ev_ts;
    std::vector<uint8_t> ev_pol;
    if (illum_on && display) {
        ev_ts.assign(N, std::numeric_limits<Metavision::timestamp>::min() / 2);
        ev_pol.assign(N, 0);
    }

    // Per-pixel refractory (activity) filter: drop redundant burst events.
    std::vector<Metavision::timestamp> refr_last(N, std::numeric_limits<Metavision::timestamp>::min() / 2);

    // Occupancy grid for O(1) nearest-neighbour association (cell ~= match radius).
    const int gcell = std::max(1, static_cast<int>(std::lround(match_radius)));
    const int gw = (W + gcell - 1) / gcell;
    const int gh = (H + gcell - 1) / gcell;
    std::vector<int> occ(static_cast<size_t>(gw) * gh, -1);
    auto cell_of = [&](float x, float y) -> int {
        int gx = static_cast<int>(x) / gcell;
        int gy = static_cast<int>(y) / gcell;
        if (gx < 0) gx = 0; else if (gx >= gw) gx = gw - 1;
        if (gy < 0) gy = 0; else if (gy >= gh) gy = gh - 1;
        return gy * gw + gx;
    };

    cv::Mat overlay;
    if (display) overlay = cv::Mat::zeros(H, W, CV_8UC3);

    // eFAST / Arc* corner test on a same-polarity SAE around (x,y).
    auto is_corner = [&](const std::vector<Metavision::timestamp> &s, int x, int y,
                         Metavision::timestamp t_now) -> bool {
        if (x < 3 || y < 3 || x >= W - 3 || y >= H - 3) return false;
        Metavision::timestamp ring[kRing];
        for (int k = 0; k < kRing; ++k) {
            const int xx = x + kCircle[k][0];
            const int yy = y + kCircle[k][1];
            ring[k] = s[static_cast<size_t>(yy) * W + xx];
        }
        // For each arc length in [arc_min, arc_max] and each start, test whether the
        // arc's minimum timestamp exceeds the maximum of all pixels outside the arc,
        // and that every arc pixel is recent (within sae_decay of now).
        for (int len = arc_min; len <= arc_max; ++len) {
            for (int start = 0; start < kRing; ++start) {
                Metavision::timestamp arc_min_t = std::numeric_limits<Metavision::timestamp>::max();
                bool recent = true;
                for (int j = 0; j < len; ++j) {
                    const Metavision::timestamp v = ring[(start + j) % kRing];
                    if (v < arc_min_t) arc_min_t = v;
                    if (t_now - v > sae_decay) { recent = false; break; }
                }
                if (!recent) continue;
                Metavision::timestamp out_max_t = std::numeric_limits<Metavision::timestamp>::min();
                for (int j = len; j < kRing; ++j) {
                    const Metavision::timestamp v = ring[(start + j) % kRing];
                    if (v > out_max_t) out_max_t = v;
                }
                if (arc_min_t > out_max_t) return true;
            }
        }
        return false;
    };

    std::cout << "event_feature_tracker - eFAST/Arc* corners + NN tracking (C++)\n"
              << "  source   : " << (input.empty() ? "LIVE CAMERA" : input) << "\n"
              << "  sensor   : " << W << "x" << H << "\n"
              << "  arc      : [" << arc_min << "," << arc_max << "]  sae_decay=" << sae_decay << "us\n"
              << "  tracking : match_radius=" << match_radius << "px timeout=" << track_to << "us\n"
              << "  filter   : refractory=" << refractory_us << "us  max_tracks=" << max_tracks << "\n"
              << "  guard    : " << (guard_on ? "GLARE-GUARD ON (common-mode + hot-mask)" : "off")
              << (guard_on ? ("  slice=" + std::to_string(slice_dt) + "us") : "") << "\n"
              << (out_csv.empty() ? "" : ("  output   : " + out_csv + "\n"));
    if (display) std::cout << "press Q or Esc in the window to stop\n";
    if (illum_on)
        std::cout << "  mode     : ILLUM-MONITOR (eclipse on/off) window=" << illum_win
                  << "us balance>=" << illum_balance << " min_events=" << illum_min_events
                  << " hold=" << illum_hold << "us\n";
    std::cout << "\n";

    GlareGuard guard(W, H, args);
    std::vector<Metavision::EventCD> slice;
    Metavision::timestamp slice_start = -1;
    long long n_contaminated = 0;
    long long n_masked_skip  = 0;

    // Per-event corner detection + nearest-neighbour tracking (shared by both paths).
    auto handle_event = [&](const Metavision::EventCD *ev) {
        const int p    = ev->p ? 1 : 0;
        const size_t i = static_cast<size_t>(ev->y) * W + ev->x;
        sae[p][i]      = ev->t;

        if (!is_corner(sae[p], ev->x, ev->y, ev->t)) return;
        ++n_corners;
        const float fx = static_cast<float>(ev->x);
        const float fy = static_cast<float>(ev->y);
        const int gc   = cell_of(fx, fy);
        const int gx   = gc % gw;
        const int gy   = gc / gw;

        // O(1) association: only the corner's grid cell + 8 neighbours.
        int best = -1;
        float best_d2 = match_radius2;
        for (int dyc = -1; dyc <= 1; ++dyc)
            for (int dxc = -1; dxc <= 1; ++dxc) {
                const int cx = gx + dxc, cy = gy + dyc;
                if (cx < 0 || cy < 0 || cx >= gw || cy >= gh) continue;
                const int s = occ[static_cast<size_t>(cy) * gw + cx];
                if (s < 0) continue;
                Track &tk = pool[s];
                if (!tk.active || ev->t - tk.t_last > track_to) continue;
                const float dx = tk.x - fx, dy = tk.y - fy;
                const float d2 = dx * dx + dy * dy;
                if (d2 <= best_d2) {
                    best_d2 = d2;
                    best    = s;
                }
            }

        if (best >= 0) {
            Track &tk = pool[best];
            tk.x      = ema * fx + (1.f - ema) * tk.x;
            tk.y      = ema * fy + (1.f - ema) * tk.y;
            tk.t_last = ev->t;
            ++tk.n;
            const int nc = cell_of(tk.x, tk.y);
            if (nc != tk.gcell) {
                if (occ[tk.gcell] == best) occ[tk.gcell] = -1;
                occ[nc]  = best;
                tk.gcell = nc;
            }
            if (out.is_open()) out << tk.id << ',' << ev->t << ',' << tk.x << ',' << tk.y << '\n';
        } else if (active_count < max_tracks) {
            int slot;
            if (!free_slots.empty()) {
                slot = free_slots.back();
                free_slots.pop_back();
                pool[slot] = {next_id, fx, fy, ev->t, 1, true, gc};
            } else {
                slot = static_cast<int>(pool.size());
                pool.push_back({next_id, fx, fy, ev->t, 1, true, gc});
            }
            occ[gc] = slot; // evicts any prior grid occupant (it will expire naturally)
            ++active_count;
            if (active_count > peak_active) peak_active = active_count;
            if (out.is_open()) out << next_id << ',' << ev->t << ',' << fx << ',' << fy << '\n';
            ++next_id;
        }
    };

    // Flush one accumulated slice through the glare guard, then track survivors.
    auto flush_slice = [&]() {
        if (slice.empty()) return;
        guard.analyze(slice);
        if (guard.contaminated()) {
            ++n_contaminated;   // freeze: do not poison the time surface with the surge
            slice.clear();
            return;
        }
        for (const Metavision::EventCD &e : slice) {
            if (guard.masked(e.x, e.y)) {
                ++n_masked_skip;   // skip a persistently over-bright (glare) region
                continue;
            }
            handle_event(&e);
        }
        slice.clear();
    };

    // Close one illumination window: evaluate polarity balance and flag a global
    // light ON (SUNRISE) or OFF (ECLIPSE) transition, with state debounce so a
    // single flip yields a single marker.
    auto il_flush = [&](Metavision::timestamp now) {
        const long long tot = il_on + il_off;
        const double bal = tot > 0 ? static_cast<double>(il_on - il_off) / tot : 0.0;
        il_on_win = il_on;
        il_off_win = il_off;
        il_total_win = tot;
        il_balance_win = bal;
        il_rate_win = tot / (illum_win / 1e6);
        if (tot >= illum_min_events && now - il_last_change >= illum_hold) {
            if (bal >= illum_balance && il_state != 1) {
                il_state = 1;
                il_last_change = now;
                ++il_sunrise;
                il_last_label = "SUNRISE  (light ON)";
                il_flash_until = now + 1000000;
                std::printf("  [%9.3f s] SUNRISE  (light ON)   balance=%+.2f  events=%lld\n",
                            now / 1e6, bal, tot);
                std::fflush(stdout);
                if (out.is_open())
                    out << now << ",SUNRISE," << tot << ',' << il_on << ',' << il_off << ',' << bal << '\n';
            } else if (bal <= -illum_balance && il_state != -1) {
                il_state = -1;
                il_last_change = now;
                ++il_eclipse;
                il_last_label = "ECLIPSE  (light OFF)";
                il_flash_until = now + 1000000;
                std::printf("  [%9.3f s] ECLIPSE  (light OFF)  balance=%+.2f  events=%lld\n",
                            now / 1e6, bal, tot);
                std::fflush(stdout);
                if (out.is_open())
                    out << now << ",ECLIPSE," << tot << ',' << il_on << ',' << il_off << ',' << bal << '\n';
            }
        }
        il_on = 0;
        il_off = 0;
        il_win_start = now;
    };

    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (begin == end) return;
        if (illum_on) {
            for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
                if (ev->p) ++il_on; else ++il_off;
                if (!ev_ts.empty()) {
                    const size_t i = static_cast<size_t>(ev->y) * W + ev->x;
                    ev_ts[i]  = ev->t;
                    ev_pol[i] = ev->p ? 1 : 0;
                }
                if (il_win_start < 0) il_win_start = ev->t;
                if (ev->t - il_win_start >= illum_win) il_flush(ev->t);
            }
            cur_t = (end - 1)->t;
            return;
        }
        if (!guard_on) {
            for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
                const size_t i = static_cast<size_t>(ev->y) * W + ev->x;
                if (refractory_us > 0 && ev->t - refr_last[i] < refractory_us) { refr_last[i] = ev->t; continue; }
                refr_last[i] = ev->t;
                cur_t = ev->t;
                handle_event(ev);
            }
        } else {
            for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
                const size_t i = static_cast<size_t>(ev->y) * W + ev->x;
                if (refractory_us > 0 && ev->t - refr_last[i] < refractory_us) { refr_last[i] = ev->t; continue; }
                refr_last[i] = ev->t;
                if (slice_start < 0) slice_start = ev->t;
                if (ev->t - slice_start >= slice_dt) {
                    flush_slice();
                    slice_start = ev->t;
                }
                slice.push_back(*ev);
            }
            cur_t = (end - 1)->t;
        }

        // Periodically expire stale tracks (pool slots return to the free list).
        if (cur_t - last_purge > track_to) {
            last_purge = cur_t;
            for (int s = 0; s < static_cast<int>(pool.size()); ++s) {
                Track &tk = pool[s];
                if (tk.active && cur_t - tk.t_last > track_to) {
                    tk.active = false;
                    if (occ[tk.gcell] == s) occ[tk.gcell] = -1;
                    free_slots.push_back(s);
                    --active_count;
                }
            }
        }
    });

    try {
        camera.start();
        const Metavision::timestamp end_t =
            max_dur > 0 ? static_cast<Metavision::timestamp>(max_dur * 1e6) : 0;
        Metavision::timestamp next_draw = 0;
        const Metavision::timestamp draw_period = 33000; // ~30 fps overlay

        while (camera.is_running()) {
            if (display && cur_t >= next_draw) {
                next_draw = cur_t + draw_period;
                if (illum_on) {
                    // Live event activity: ON=green, OFF=red, brightness=recency.
                    // A frame camera only reports "dark"; here you still see what
                    // moves/vibrates during the eclipse.
                    cv::Vec3b *op = overlay.ptr<cv::Vec3b>(0);
                    for (size_t i = 0; i < N; ++i) {
                        const Metavision::timestamp age = cur_t - ev_ts[i];
                        if (age < 0 || age > illum_decay) { op[i] = cv::Vec3b(18, 18, 18); continue; }
                        const double f = 1.0 - static_cast<double>(age) / illum_decay;
                        const uchar v = static_cast<uchar>(40 + 200 * f);
                        op[i] = ev_pol[i] ? cv::Vec3b(30, v, 30) : cv::Vec3b(30, 30, v);
                    }
                    const int cx = W / 2;
                    const cv::Scalar scol = il_state == 1 ? cv::Scalar(90, 240, 90)
                                          : il_state == -1 ? cv::Scalar(60, 170, 240)
                                          : cv::Scalar(200, 200, 200);
                    const std::string st = il_state == 1 ? "SUNLIT"
                                         : il_state == -1 ? "ECLIPSE (event view)"
                                         : "waiting for a light change...";
                    cv::rectangle(overlay, cv::Rect(0, 0, W - 1, H - 1), scol, 3);
                    cv::putText(overlay, st, cv::Point(12, 32),
                                cv::FONT_HERSHEY_SIMPLEX, 0.9, scol, 2, cv::LINE_AA);
                    const int bw = std::min(W - 40, 640);
                    const int by = H - 34;
                    cv::rectangle(overlay, cv::Rect(cx - bw / 2, by, bw, 18), cv::Scalar(90, 90, 90), 1);
                    cv::line(overlay, cv::Point(cx, by - 6), cv::Point(cx, by + 24),
                             cv::Scalar(160, 160, 160), 1);
                    const int bl = static_cast<int>(il_balance_win * (bw / 2));
                    const cv::Scalar barcol = il_balance_win >= 0 ? cv::Scalar(80, 220, 80)
                                                                  : cv::Scalar(80, 80, 235);
                    if (bl >= 0) cv::rectangle(overlay, cv::Rect(cx, by, bl, 18), barcol, cv::FILLED);
                    else cv::rectangle(overlay, cv::Rect(cx + bl, by, -bl, 18), barcol, cv::FILLED);
                    char info[192];
                    std::snprintf(info, sizeof(info),
                                  "balance=%+.2f  rate=%.0f ev/s  ON=%lld OFF=%lld  sunrise=%lld eclipse=%lld",
                                  il_balance_win, il_rate_win, il_on_win, il_off_win, il_sunrise, il_eclipse);
                    cv::putText(overlay, info, cv::Point(12, H - 44),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
                    if (cur_t < il_flash_until && !il_last_label.empty()) {
                        const cv::Scalar fc = il_state == 1 ? cv::Scalar(90, 240, 90)
                                                            : cv::Scalar(90, 90, 245);
                        cv::putText(overlay, il_last_label, cv::Point(cx - 200, 66),
                                    cv::FONT_HERSHEY_SIMPLEX, 1.0, fc, 2, cv::LINE_AA);
                    }
                    cv::imshow("event_feature_tracker", overlay);
                    const int k = cv::waitKey(1);
                    if (k == 27 || k == 'q' || k == 'Q') break;
                    if (end_t > 0 && cur_t >= end_t) break;
                    continue;
                }
                overlay.setTo(cv::Scalar(20, 20, 20));
                if (guard_on) {
                    // Red wash over persistently over-bright (masked) cells.
                    const int cs = guard.cell();
                    const std::vector<uint8_t> &m = guard.mask();
                    for (int r = 0; r < guard.n_rows(); ++r)
                        for (int c = 0; c < guard.n_cols(); ++c)
                            if (m[static_cast<size_t>(r) * guard.n_cols() + c]) {
                                const cv::Rect roi(c * cs, r * cs, std::min(cs, W - c * cs),
                                                   std::min(cs, H - r * cs));
                                overlay(roi) += cv::Scalar(0, 0, 60);
                            }
                }
                for (const Track &t : pool) {
                    if (!t.active || cur_t - t.t_last > track_to) continue;
                    const cv::Point c(static_cast<int>(t.x), static_cast<int>(t.y));
                    const cv::Scalar col = t.n > 5 ? cv::Scalar(60, 220, 60) : cv::Scalar(60, 160, 220);
                    cv::circle(overlay, c, 3, col, 1, cv::LINE_AA);
                }
                std::string hud = "tracks=" + std::to_string(active_count) +
                                  "  corners=" + std::to_string(n_corners);
                if (guard_on) {
                    char cb[64];
                    std::snprintf(cb, sizeof(cb), "  cm=%.2f", guard.cm_index());
                    hud += cb;
                }
                cv::putText(overlay, hud, cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
                if (guard_on && guard.contaminated())
                    cv::putText(overlay, "CONTAMINATED - tracking frozen", cv::Point(8, 44),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(60, 60, 255), 2, cv::LINE_AA);
                cv::imshow("event_feature_tracker", overlay);
                const int k = cv::waitKey(1);
                if (k == 27 || k == 'q' || k == 'Q') break;
            } else {
                std::this_thread::yield();
            }
            if (end_t > 0 && cur_t >= end_t) break;
        }
        camera.stop();
    } catch (const std::exception &e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return 1;
    }

    if (guard_on) flush_slice(); // drain the final partial slice
    if (illum_on && il_win_start >= 0) il_flush(cur_t);

    if (out.is_open()) out.close();
    if (display) cv::destroyAllWindows();
    std::cout << "\n=== done ===\n";
    if (illum_on) {
        std::cout << "  illum    : " << il_sunrise << " sunrise, " << il_eclipse
                  << " eclipse transitions\n";
    } else {
        std::cout << "  corners  : " << n_corners << "\n"
                  << "  tracks   : " << next_id << " created, peak " << peak_active << " active (cap "
                  << max_tracks << ")\n";
        if (guard_on)
            std::cout << "  guard    : " << n_contaminated << " slices frozen (contaminated), "
                      << n_masked_skip << " events dropped in hot mask\n";
    }
    if (!out_csv.empty()) std::cout << "  csv      : " << out_csv << "\n";
    return 0;
}
