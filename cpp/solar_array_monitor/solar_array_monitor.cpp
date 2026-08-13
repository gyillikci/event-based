/**********************************************************************************************************************
 * Native real-time solar-array vibration monitor (Metavision SDK, C++).
 *
 * IAC 2026 -- Event-Based Vibration Monitoring of Flexible Solar Arrays Through
 * Orbital Sunrise.
 *
 * WHY C++: the Python monitor (solar_array/monitor_solar_array.py) runs the same
 * pipeline at real time when headless, but rendering every 1280x720 slice through
 * the Python/OpenCV bindings costs more than the 20 ms slice budget, so the
 * camera's buffer backs up and the live view drifts ~20 s behind. This native
 * port keeps acquisition + glare guard + vibrometer on one thread and pushes only
 * the finished frame to a display thread, so the view stays current.
 *
 * It is a faithful port of the *validated median-mode* pipeline:
 *   L0  hardware ERC rate cap                          (I_ErcModule)
 *   L1  common-mode rejection (per-cell polarity balance, spatial-median subtract)
 *   L2  solar-disk / limb masking (persistent hot + halo dilation)
 *   L3  event vibrometer (per-gauge signed integration + 2nd-order drift high-pass)
 *
 * Modal identification (Welch PSD / log-decrement) is deliberately NOT ported: it
 * only becomes possible ~459 s in (159 s filter warm-up + 300 s Welch window) and
 * is not latency-sensitive, so it stays in the numpy-only Python modules and runs
 * offline on the JSONL this tool writes (--session-log), or on a RAW recording.
 *
 * The numeric constants mirror solar_array/solar_array_config.json; override them
 * on the command line for a lab scene whose panel is not at the configured pixels.
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
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/file_config_hints.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/facilities/i_erc_module.h>

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
                    if (next.rfind("--", 0) != 0) {
                        kv_[key] = next;
                        ++i;
                        continue;
                    }
                }
                flags_.insert(key);
            }
        }
    }
    bool has(const std::string &k) const { return kv_.count(k) || flags_.count(k); }
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
    double getf(const std::string &k, double d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stod(it->second);
    }
    bool getb(const std::string &k, bool d) const {
        if (flags_.count(k)) return true;
        auto it = kv_.find(k);
        if (it == kv_.end()) return d;
        return it->second == "1" || it->second == "true" || it->second == "on";
    }

private:
    std::map<std::string, std::string> kv_;
    std::set<std::string> flags_;
};

// --------------------------------------------------------------------------- //
// L3 -- cascaded first-order high-pass (drift rejection). Two poles reject the
// linear drift that residual bias / threshold asymmetry integrate into.
// --------------------------------------------------------------------------- //
class HighPassFilter {
public:
    HighPassFilter(double corner_hz, double dt_s, int order)
        : order_(std::max(1, order)), y_(order_, 0.0), x_(order_, 0.0) {
        corner_hz_ = corner_hz;
        if (corner_hz <= 0.0) {
            alpha_ = 1.0;
        } else {
            const double rc = 1.0 / (2.0 * M_PI * corner_hz);
            alpha_ = rc / (rc + dt_s);
        }
    }
    double operator()(double x) {
        if (!primed_) {
            std::fill(x_.begin(), x_.end(), x);
            std::fill(y_.begin(), y_.end(), 0.0);
            primed_ = true;
            return 0.0;
        }
        double value = x;
        for (int k = 0; k < order_; ++k) {
            y_[k] = alpha_ * (y_[k] + value - x_[k]);
            x_[k] = value;
            value = y_[k];
        }
        return value;
    }
    double warmup_s() const {
        if (corner_hz_ <= 0.0) return 0.0;
        const double rc = 1.0 / (2.0 * M_PI * corner_hz_);
        return 5.0 * rc * order_;
    }

private:
    int order_;
    double corner_hz_ = 0.0;
    double alpha_ = 1.0;
    bool primed_ = false;
    std::vector<double> y_, x_;
};

// --------------------------------------------------------------------------- //
// Virtual strain gauge straddling a panel edge.
// --------------------------------------------------------------------------- //
struct Gauge {
    std::string name;
    int x0, x1, y0, y1;   // pixel bounds [x0,x1) x [y0,y1)
    int cx, cy;           // centre pixel (for cell lookup)
    double xi;            // normalised span coordinate
    double sign;          // edge sign (top -1, bottom +1, times contrast sign)
    double area;          // pixel area
    // running state
    double raw = 0.0;
    long long activity_total = 0;
    long long n_dropped = 0;
};

// --------------------------------------------------------------------------- //
// HUD / shared-frame state handed from the processing thread to the display.
// --------------------------------------------------------------------------- //
struct SharedFrame {
    std::mutex m;
    cv::Mat bgr;
    std::atomic<bool> ready{false};
    std::atomic<bool> stop{false};
};

// --------------------------------------------------------------------------- //
// The whole pipeline for one sensor, driven slice by slice.
// --------------------------------------------------------------------------- //
class Monitor {
public:
    Monitor(int width, int height, const Args &args)
        : width_(width), height_(height) {
        cell_ = std::max(1, args.geti("cell-size", 16));
        n_cols_ = (width_ + cell_ - 1) / cell_;
        n_rows_ = (height_ + cell_ - 1) / cell_;
        min_events_per_cell_ = args.geti("min-events-per-cell", 4);
        index_threshold_ = args.getf("cm-index-threshold", 0.55);
        dispersion_scale_ = args.getf("cm-dispersion-scale", 0.5);
        rate_sigma_ = args.getf("mask-rate-sigma", 6.0);
        persistence_ = args.geti("mask-persistence", 4);
        release_ = args.geti("mask-release", 2);
        halo_cells_ = std::max(0, (int)std::lround(args.getf("mask-halo-px", 48.0) / cell_));
        slice_dt_us_ = args.getll("slice-dt-us", 20000);
        const double dt_s = slice_dt_us_ * 1e-6;

        // Gauge geometry (defaults mirror solar_array_config.json).
        const double root_x = args.getf("root-x", 90.0);
        const double tip_x = args.getf("tip-x", 1210.0);
        const double centre_y = args.getf("centre-y", 300.0);
        const double chord = args.getf("chord", 150.0);
        const int half_w = args.geti("gauge-half-width", 40);
        const int half_h = args.geti("gauge-half-height", 18);
        const double contrast_sign = args.getf("contrast-sign", 1.0);
        const std::vector<double> xis = {0.18, 0.34, 0.50, 0.66, 0.82, 0.96};

        const double panel_refl = args.getf("panel-reflectance", 0.075);
        const double bg_refl = args.getf("background-reflectance", 0.004);
        const double c_thr = args.getf("contrast-threshold", 0.20);
        const double roi_w = 2.0 * half_w + 1.0;
        const double log_contrast = std::log(std::max(panel_refl, 1e-9) / std::max(bg_refl, 1e-9));
        gain_ = std::max(roi_w * log_contrast / std::max(c_thr, 1e-9), 1e-9);

        min_mean_activity_ = args.getf("min-mean-activity", 0.5);
        const double hp_hz = args.getf("highpass-hz", 0.01);
        const int hp_order = args.geti("highpass-order", 2);

        gauge_map_.assign((size_t)width_ * height_, -1);
        int gi = 0;
        for (double xi : xis) {
            const double x = root_x + xi * (tip_x - root_x);
            for (int edge = 0; edge < 2; ++edge) {   // 0 top, 1 bottom
                const double offset = (edge == 0 ? -0.5 : 0.5) * chord;
                Gauge g;
                std::ostringstream nm;
                nm << (edge == 0 ? "T" : "B") << std::fixed << std::setprecision(2) << xi;
                g.name = nm.str();
                g.cx = (int)std::lround(x);
                g.cy = (int)std::lround(centre_y + offset);
                g.x0 = std::max(0, g.cx - half_w);
                g.x1 = std::min(width_, g.cx + half_w + 1);
                g.y0 = std::max(0, g.cy - half_h);
                g.y1 = std::min(height_, g.cy + half_h + 1);
                g.xi = xi;
                g.sign = contrast_sign * (edge == 0 ? -1.0 : 1.0);
                g.area = std::max(0, g.x1 - g.x0) * (double)std::max(0, g.y1 - g.y0);
                gauges_.push_back(g);
                filters_.emplace_back(hp_hz, dt_s, hp_order);
                for (int y = g.y0; y < g.y1; ++y)
                    for (int xx = g.x0; xx < g.x1; ++xx)
                        gauge_map_[(size_t)y * width_ + xx] = gi;
                ++gi;
            }
        }

        // Per-cell scratch.
        on_.assign((size_t)n_rows_ * n_cols_, 0);
        off_.assign((size_t)n_rows_ * n_cols_, 0);
        px_map_.assign((size_t)n_rows_ * n_cols_, 0);
        for (int r = 0; r < n_rows_; ++r)
            for (int c = 0; c < n_cols_; ++c) {
                const int ph = std::min((r + 1) * cell_, height_) - r * cell_;
                const int pw = std::min((c + 1) * cell_, width_) - c * cell_;
                px_map_[(size_t)r * n_cols_ + c] = ph * pw;
            }
        hits_.assign((size_t)n_rows_ * n_cols_, 0);
        mask_.assign((size_t)n_rows_ * n_cols_, 0);
        mask_tmp_.assign((size_t)n_rows_ * n_cols_, 0);
        nominal_px_ = cell_ * cell_;

        // Per-gauge slice accumulators.
        g_signed_.assign(gauges_.size(), 0.0);
        g_total_.assign(gauges_.size(), 0);
        disp_.assign(gauges_.size(), 0.0);
        warmup_s_ = filters_.empty() ? 0.0 : filters_[0].warmup_s();

        render_frame_ = args.getb("no-display", false) ? cv::Mat() : cv::Mat(height_, width_, CV_8UC3);
        max_display_w_ = args.geti("max-display-width", 1280);
    }

    double gain() const { return gain_; }
    double warmup_s() const { return warmup_s_; }
    int n_gauges() const { return (int)gauges_.size(); }

    // Process one fully-accumulated slice of events (sorted or not).
    void process_slice(Metavision::timestamp ts_us,
                       const std::vector<Metavision::EventCD> &events,
                       bool want_frame, SharedFrame *shared,
                       std::ofstream *log) {
        std::fill(on_.begin(), on_.end(), 0);
        std::fill(off_.begin(), off_.end(), 0);
        std::fill(g_signed_.begin(), g_signed_.end(), 0.0);
        std::fill(g_total_.begin(), g_total_.end(), 0);

        double raw_rate = events.size() / (slice_dt_us_ * 1e-6);
        peak_rate_ = std::max(peak_rate_, raw_rate);

        // Bin into cells and gauges in one pass.
        for (const auto &e : events) {
            const int cellr = e.y / cell_, cellc = e.x / cell_;
            const size_t ci = (size_t)cellr * n_cols_ + cellc;
            if (e.p) ++on_[ci]; else ++off_[ci];
            const int gi = gauge_map_[(size_t)e.y * width_ + e.x];
            if (gi >= 0) {
                g_signed_[gi] += e.p ? 1.0 : -1.0;
                ++g_total_[gi];
            }
        }

        // ---- L1 common mode (median mode) ---- //
        double balance = 0.0, dispersion = 0.0, coverage = 0.0, common_per_px = 0.0;
        {
            balances_.clear();
            signed_px_.clear();
            int valid = 0, judged = 0;
            for (size_t i = 0; i < on_.size(); ++i) {
                const long tot = on_[i] + off_[i];
                const double px = std::max<long>(1, px_map_[i]);
                const bool masked = mask_[i] != 0;
                if (!masked) ++judged;
                if (tot >= min_events_per_cell_ && !masked) {
                    ++valid;
                    balances_.push_back((on_[i] - off_[i]) / (double)tot);
                    signed_px_.push_back((on_[i] - off_[i]) / px);
                }
            }
            if (!balances_.empty()) {
                balance = median(balances_);
                devs_.clear();
                for (double b : balances_) devs_.push_back(std::fabs(b - balance));
                dispersion = median(devs_);
                common_per_px = median(signed_px_);
            }
            coverage = judged ? (double)valid / judged : 0.0;
        }
        const double uniformity = 1.0 - std::min(1.0, dispersion / std::max(dispersion_scale_, 1e-9));
        const double cm_index = std::fabs(balance) * uniformity * coverage;
        const bool contaminated = cm_index >= index_threshold_;

        // ---- L2 solar-disk mask (persistent hot + halo) ---- //
        update_mask();

        // ---- L3 vibrometer ---- //
        int n_healthy = 0;
        ++n_slices_;
        for (size_t i = 0; i < gauges_.size(); ++i) {
            Gauge &g = gauges_[i];
            const int cellr = std::min(std::max(g.cy / cell_, 0), n_rows_ - 1);
            const int cellc = std::min(std::max(g.cx / cell_, 0), n_cols_ - 1);
            const bool masked = mask_[(size_t)cellr * n_cols_ + cellc] != 0;
            const bool dropout = contaminated || masked;
            const double common = common_per_px * g.area;
            const double corrected = g_signed_[i] - common;
            const double increment = dropout ? 0.0 : g.sign * corrected / gain_;
            g.raw += increment;
            disp_[i] = filters_[i](g.raw);
            g.activity_total += g_total_[i];
            if (dropout) ++g.n_dropped;
            const double mean_activity = (double)g.activity_total / n_slices_;
            const double drop_frac = (double)g.n_dropped / n_slices_;
            if (mean_activity >= min_mean_activity_ && drop_frac <= 0.35) ++n_healthy;
        }

        // ---- log ---- //
        if (log) write_log(*log, ts_us, events.size(), raw_rate, cm_index, balance,
                           coverage, contaminated, n_healthy);

        // ---- render ---- //
        if (want_frame && shared && !render_frame_.empty()) {
            render(events, cm_index, balance, coverage, raw_rate, n_healthy, ts_us);
            std::lock_guard<std::mutex> lk(shared->m);
            display_.copyTo(shared->bgr);
            shared->ready = true;
        }
        last_index_ = cm_index;
        last_healthy_ = n_healthy;
    }

private:
    static double median(std::vector<double> &v) {
        if (v.empty()) return 0.0;
        const size_t n = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + n, v.end());
        double m = v[n];
        if (v.size() % 2 == 0) {
            double lo = *std::max_element(v.begin(), v.begin() + n);
            m = 0.5 * (m + lo);
        }
        return m;
    }

    void update_mask() {
        // Scene density statistics over eligible (near-full) cells.
        dens_.clear();
        for (size_t i = 0; i < on_.size(); ++i) {
            if (px_map_[i] >= 0.5 * nominal_px_)
                dens_.push_back((on_[i] + off_[i]) / (double)std::max<long>(1, px_map_[i]));
        }
        double med = dens_.empty() ? 0.0 : median(dens_);
        devs_.clear();
        for (double d : dens_) devs_.push_back(std::fabs(d - med));
        double mad = devs_.empty() ? 0.0 : median(devs_);
        const double sigma = std::max(1.4826 * mad, 1e-3);
        const double thr = med + rate_sigma_ * sigma;

        for (size_t i = 0; i < on_.size(); ++i) {
            const bool eligible = px_map_[i] >= 0.5 * nominal_px_;
            const double d = (on_[i] + off_[i]) / (double)std::max<long>(1, px_map_[i]);
            const bool hot = eligible && d > thr;
            hits_[i] = hot ? hits_[i] + 1 : std::max(0, hits_[i] - release_);
        }
        // core = hits >= persistence, then box-dilate by halo_cells.
        for (size_t i = 0; i < mask_.size(); ++i) mask_tmp_[i] = (hits_[i] >= persistence_) ? 1 : 0;
        box_dilate(mask_tmp_, mask_, n_rows_, n_cols_, halo_cells_);
    }

    static void box_dilate(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst,
                           int rows, int cols, int radius) {
        if (radius <= 0) { dst = src; return; }
        std::vector<uint8_t> tmp(src);
        // horizontal
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                uint8_t v = 0;
                for (int dc = -radius; dc <= radius && !v; ++dc) {
                    const int cc = c + dc;
                    if (cc >= 0 && cc < cols && src[(size_t)r * cols + cc]) v = 1;
                }
                tmp[(size_t)r * cols + c] = v;
            }
        }
        // vertical
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                uint8_t v = 0;
                for (int dr = -radius; dr <= radius && !v; ++dr) {
                    const int rr = r + dr;
                    if (rr >= 0 && rr < rows && tmp[(size_t)rr * cols + c]) v = 1;
                }
                dst[(size_t)r * cols + c] = v;
            }
        }
    }

    void render(const std::vector<Metavision::EventCD> &events, double cm_index,
                double balance, double coverage, double raw_rate, int n_healthy,
                Metavision::timestamp ts_us) {
        render_frame_.setTo(cv::Scalar(32, 32, 32));
        cv::Vec3b *img = render_frame_.ptr<cv::Vec3b>();
        for (const auto &e : events) {
            img[(size_t)e.y * width_ + e.x] = e.p ? cv::Vec3b(60, 220, 60)
                                                  : cv::Vec3b(200, 90, 60);
        }
        // Masked cells: translucent red wash.
        for (int r = 0; r < n_rows_; ++r) {
            for (int c = 0; c < n_cols_; ++c) {
                if (!mask_[(size_t)r * n_cols_ + c]) continue;
                cv::Rect cell(c * cell_, r * cell_,
                              std::min(cell_, width_ - c * cell_),
                              std::min(cell_, height_ - r * cell_));
                cv::Mat roi = render_frame_(cell);
                roi = roi * 0.6 + cv::Scalar(0, 0, 110) * 0.4;
            }
        }
        // Gauge boxes.
        for (size_t i = 0; i < gauges_.size(); ++i) {
            const Gauge &g = gauges_[i];
            const double mean_activity = (double)g.activity_total / std::max(1LL, n_slices_);
            const double drop_frac = (double)g.n_dropped / std::max(1LL, n_slices_);
            const bool healthy = mean_activity >= min_mean_activity_ && drop_frac <= 0.35;
            cv::rectangle(render_frame_, cv::Point(g.x0, g.y0), cv::Point(g.x1 - 1, g.y1 - 1),
                          healthy ? cv::Scalar(0, 210, 0) : cv::Scalar(0, 165, 255), 1);
        }

        // HUD.
        char buf[160];
        auto put = [&](const std::string &s, int y, cv::Scalar col) {
            cv::putText(render_frame_, s, cv::Point(8, y), cv::FONT_HERSHEY_SIMPLEX,
                        0.45, col, 1, cv::LINE_AA);
        };
        put("SOLAR-ARRAY VIBRATION MONITOR (C++)", 18, cv::Scalar(0, 255, 255));
        std::snprintf(buf, sizeof(buf), "t=%8.1fs  rate=%6.2f Mev/s", ts_us * 1e-6, raw_rate / 1e6);
        put(buf, 36, cv::Scalar(235, 235, 235));
        std::snprintf(buf, sizeof(buf), "common-mode idx=%5.2f  bal=%+5.2f  cov=%4.2f",
                      cm_index, balance, coverage);
        put(buf, 54, cv::Scalar(235, 235, 235));
        int masked_cells = 0; for (uint8_t m : mask_) masked_cells += m;
        std::snprintf(buf, sizeof(buf), "gauges healthy %d/%d   mask %4.1f%%",
                      n_healthy, (int)gauges_.size(),
                      100.0 * masked_cells / std::max(1, n_rows_ * n_cols_));
        put(buf, 72, cv::Scalar(235, 235, 235));
        put("modal ID runs offline from --session-log (needs ~459 s)", 90,
            cv::Scalar(180, 180, 180));

        // Scale down for display if the sensor is wider than the target.
        if (max_display_w_ > 0 && width_ > max_display_w_) {
            const double s = (double)max_display_w_ / width_;
            cv::resize(render_frame_, display_, cv::Size(), s, s, cv::INTER_NEAREST);
        } else {
            render_frame_.copyTo(display_);
        }
    }

    void write_log(std::ofstream &log, Metavision::timestamp ts_us, size_t n_ev,
                   double raw_rate, double cm_index, double balance, double coverage,
                   bool contaminated, int n_healthy) {
        log << "{\"ts_us\":" << ts_us
            << ",\"t_s\":" << std::fixed << std::setprecision(4) << ts_us * 1e-6
            << ",\"raw_events\":" << n_ev
            << ",\"raw_rate_evs\":" << std::setprecision(1) << raw_rate
            << ",\"glare_guard\":{\"common_mode_index\":" << std::setprecision(4) << cm_index
            << ",\"polarity_balance\":" << balance
            << ",\"coverage\":" << coverage
            << ",\"contaminated\":" << (contaminated ? "true" : "false") << "}"
            << ",\"vibrometer\":{\"n_healthy\":" << n_healthy
            << ",\"n_gauges\":" << gauges_.size()
            << ",\"displacement_px\":[";
        for (size_t i = 0; i < disp_.size(); ++i)
            log << (i ? "," : "") << std::setprecision(5) << disp_[i];
        log << "]}}" << "\n";
        log.flush();
    }

    int width_, height_, cell_, n_cols_, n_rows_;
    int min_events_per_cell_, persistence_, release_, halo_cells_;
    long nominal_px_;
    double index_threshold_, dispersion_scale_, rate_sigma_;
    double gain_, min_mean_activity_, warmup_s_;
    long long slice_dt_us_;

    std::vector<Gauge> gauges_;
    std::vector<HighPassFilter> filters_;
    std::vector<int> gauge_map_;

    std::vector<long> on_, off_, px_map_;
    std::vector<int> hits_;
    std::vector<uint8_t> mask_, mask_tmp_;
    std::vector<double> g_signed_, disp_;
    std::vector<long> g_total_;
    std::vector<double> balances_, signed_px_, devs_, dens_;

    long long n_slices_ = 0;
    double peak_rate_ = 0.0, last_index_ = 0.0;
    int last_healthy_ = 0;

    cv::Mat render_frame_, display_;
    int max_display_w_ = 1280;

public:
    double peak_rate() const { return peak_rate_; }
    long long n_slices() const { return n_slices_; }
};

void print_usage() {
    std::cout <<
        "solar_array_monitor - native real-time event-based solar-array vibration monitor\n\n"
        "Runs the validated glare-guard + vibrometer pipeline over a live Prophesee\n"
        "camera or a RAW/HDF5 recording, with a lag-free OpenCV overlay. Modal ID runs\n"
        "offline in Python from the JSONL this writes.\n\n"
        "Input:\n"
        "  -i, --input <file>            .raw/.hdf5 recording (omit for a live camera)\n"
        "      --slice-dt-us <us>        slice cadence (default 20000)\n"
        "      --erc-rate <ev/s>         hardware ERC target, live only (default 20000000; 0=off)\n\n"
        "Display / logging:\n"
        "      --no-display              headless (no OpenCV window)\n"
        "      --max-display-width <px>  downscale wide sensors for display (default 1280)\n"
        "      --session-log <path>      write a JSONL trace (displacement series + guard stats)\n"
        "      --max-slices <n>          stop after n slices (0 = unlimited)\n\n"
        "Scene geometry (defaults mirror solar_array_config.json):\n"
        "      --root-x --tip-x --centre-y --chord --gauge-half-width --gauge-half-height\n"
        "      --contrast-sign --cell-size --mask-halo-px\n";
}

} // namespace

int main(int argc, char **argv) {
    Args args(argc, argv);
    if (args.has("help") || args.has("h")) {
        print_usage();
        return 0;
    }

    const std::string input = args.has("input") ? args.get("input") : args.get("i");
    const bool no_display = args.getb("no-display", false);
    const long long erc_rate = args.getll("erc-rate", 20000000);
    const long long slice_dt_us = args.getll("slice-dt-us", 20000);
    const long long max_slices = args.getll("max-slices", 0);
    const std::string log_path = args.get("session-log");

    Metavision::Camera camera;
    try {
        if (input.empty()) {
            camera = Metavision::Camera::from_first_available();
        } else {
            camera = Metavision::Camera::from_file(
                input, Metavision::FileConfigHints().real_time_playback(true));
        }
    } catch (const std::exception &e) {
        std::cerr << "Camera init failed: " << e.what() << "\n";
        return 1;
    }

    const int width = camera.geometry().width();
    const int height = camera.geometry().height();
    const bool is_live = input.empty();

    // L0 -- enable the hardware event-rate controller on a live camera.
    bool erc_on = false;
    if (is_live && erc_rate > 0) {
        try {
            auto *erc = camera.get_device().get_facility<Metavision::I_ErcModule>();
            if (erc) {
                erc->set_cd_event_rate((uint32_t)erc_rate);
                erc->enable(true);
                erc_on = true;
            }
        } catch (const std::exception &) { /* facility not available on this source */ }
    }

    std::ofstream log;
    if (!log_path.empty()) {
        log.open(log_path);
        if (!log) std::cerr << "warning: could not open session log '" << log_path << "'\n";
    }

    Monitor monitor(width, height, args);

    std::cout << "IAC 2026 - native solar-array vibration monitor (C++)\n"
              << "  source : " << (is_live ? "LIVE CAMERA" : input) << "\n"
              << "  sensor : " << width << "x" << height
              << "  erc=" << (erc_on ? "on" : "off")
              << "  gain=" << std::fixed << std::setprecision(1) << monitor.gain()
              << " ev/px  warmup=" << std::setprecision(0) << monitor.warmup_s() << "s\n"
              << "  gauges : " << monitor.n_gauges() << "\n"
              << "  log    : " << (log_path.empty() ? "disabled" : log_path) << "\n"
              << "running - press Q or Esc in the window to stop\n\n";

    SharedFrame shared;
    const std::string win = "solar-array vibration monitor (C++)";

    // Slice accumulation state (all touched only in the camera callback thread).
    std::vector<Metavision::EventCD> slice;
    slice.reserve(1 << 16);
    Metavision::timestamp slice_start = -1;
    long long slice_count = 0;
    std::atomic<bool> done{false};

    camera.cd().add_callback(
        [&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            for (const Metavision::EventCD *e = begin; e != end; ++e) {
                if (slice_start < 0) slice_start = e->t;
                if (e->t - slice_start >= slice_dt_us) {
                    const bool want_frame = !no_display;
                    monitor.process_slice(slice_start, slice, want_frame, &shared,
                                          log.is_open() ? &log : nullptr);
                    slice.clear();
                    slice_start = e->t;
                    ++slice_count;
                    if (max_slices && slice_count >= max_slices) { done = true; return; }
                }
                slice.push_back(*e);
            }
        });

    try {
        camera.start();
        if (no_display) {
            auto t0 = std::chrono::steady_clock::now();
            auto last = t0;
            while (camera.is_running() && !done) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - last).count() >= 2.0) {
                    last = now;
                    std::cout << "  slices=" << monitor.n_slices()
                              << "  peak=" << std::fixed << std::setprecision(1)
                              << monitor.peak_rate() / 1e6 << " Mev/s\n";
                }
            }
        } else {
            cv::namedWindow(win, cv::WINDOW_NORMAL);
            while (camera.is_running() && !done) {
                if (shared.ready.exchange(false)) {
                    std::lock_guard<std::mutex> lk(shared.m);
                    if (!shared.bgr.empty()) cv::imshow(win, shared.bgr);
                }
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') break;
            }
            cv::destroyAllWindows();
        }
        camera.stop();
    } catch (const std::exception &e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n=== summary ===\n"
              << "  slices   : " << monitor.n_slices() << "\n"
              << "  peak rate: " << std::fixed << std::setprecision(1)
              << monitor.peak_rate() / 1e6 << " Mev/s\n"
              << "  modal ID : run offline -> python modal_analysis on "
              << (log_path.empty() ? "<session-log>" : log_path) << "\n";
    return 0;
}
