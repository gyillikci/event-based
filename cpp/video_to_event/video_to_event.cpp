/**********************************************************************************************************************
 * video_to_event - native C++ event-camera SIMULATOR (deterministic, no ML weights).
 *
 * Turns an ordinary video (or image sequence) into a DVS event stream using the
 * classic ESIM / log-intensity threshold model - the same principle as the Python
 * metavision_core_ml/video_to_event simulator, but without any trained network:
 *
 *   L(x,y,t) = log(I/255 + eps)                          (log photoreceptor)
 *   an event fires every time L moves by +/- C from the       (contrast threshold C)
 *   pixel's last-event level; polarity = sign of the move.
 *   Event timestamps are LINEARLY INTERPOLATED between the two source frames so
 *   sub-frame timing is preserved (temporal upsampling of the discrete video).
 *
 * Output: a CSV event list (t_us,x,y,p) with p in {0,1}, monotonic in time, plus an
 * optional accumulated-event preview window. Headless unless --display is given.
 **********************************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
        "video_to_event - native C++ ESIM event-camera simulator (deterministic)\n\n"
        "Converts a video into a DVS event stream via the log-intensity threshold model,\n"
        "with linear sub-frame timestamp interpolation. No trained model required.\n\n"
        "Input:\n"
        "  -i, --input <file>        source video (.mp4/.avi/...) REQUIRED\n"
        "      --fps <hz>            override source fps (default: read from file, else 30)\n\n"
        "Simulation:\n"
        "      --contrast <C>        log-intensity threshold per event (default 0.15)\n"
        "      --eps <e>             log offset log(I/255 + eps) (default 1e-3)\n"
        "      --refractory-us <us>  per-pixel dead time after an event (default 0)\n"
        "      --max-events <n>      stop after n events (0 = unlimited)\n\n"
        "Output:\n"
        "  -o, --output <file>       CSV event list t_us,x,y,p (default events.csv)\n"
        "      --display             show accumulated-event preview\n"
        "      --quiet               suppress per-frame progress\n"
        "  -h, --help                this message\n";
}

} // namespace

int main(int argc, char **argv) {
    Args args(argc, argv);
    if (args.getb("help", false) || args.getb("h", false) || argc == 1) {
        print_usage();
        return 0;
    }

    const std::string input = !args.get("input").empty() ? args.get("input") : args.get("i");
    if (input.empty()) {
        std::cerr << "error: --input <video> is required\n";
        return 1;
    }
    const std::string out_path =
        !args.get("output").empty() ? args.get("output") : (!args.get("o").empty() ? args.get("o") : "events.csv");

    const double C          = args.getd("contrast", 0.15);
    const double eps        = args.getd("eps", 1e-3);
    const long long refr_us = args.getll("refractory-us", 0);
    const long long max_ev  = args.getll("max-events", 0);
    const bool display      = args.getb("display", false);
    const bool quiet        = args.getb("quiet", false);

    cv::VideoCapture cap(input);
    if (!cap.isOpened()) {
        std::cerr << "error: cannot open video '" << input << "'\n";
        return 1;
    }
    double fps = args.has("fps") ? args.getd("fps", 30.0) : cap.get(cv::CAP_PROP_FPS);
    if (!(fps > 0.0)) fps = 30.0;
    const double frame_dt_us = 1e6 / fps;

    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::cerr << "error: cannot open output '" << out_path << "'\n";
        return 1;
    }
    out << "t_us,x,y,p\n";

    std::cout << "video_to_event - ESIM simulator (C++)\n"
              << "  input    : " << input << "\n"
              << "  fps      : " << fps << "  (frame dt " << frame_dt_us << " us)\n"
              << "  contrast : " << C << "  eps=" << eps << "  refractory=" << refr_us << "us\n"
              << "  output   : " << out_path << "\n\n";

    cv::Mat frame, gray, logf;
    std::vector<float> ref_level;   // per-pixel log level at last emitted event
    std::vector<float> prev_log;    // per-pixel log intensity of previous frame
    std::vector<long long> last_ev; // per-pixel timestamp of last emitted event (for refractory)
    int W = 0, H = 0;

    long long frame_idx = 0;
    long long total_events = 0;
    long long t_prev_us = 0;

    // Reusable per-frame event buffer (sorted by time before writing).
    struct Ev { long long t; int x; int y; int p; };
    std::vector<Ev> buf;

    cv::Mat preview;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(logf, CV_32F, 1.0 / 255.0);
        // log(I/255 + eps)
        cv::log(logf + static_cast<float>(eps), logf);

        if (frame_idx == 0) {
            W = logf.cols;
            H = logf.rows;
            ref_level.assign(static_cast<size_t>(W) * H, 0.f);
            prev_log.assign(static_cast<size_t>(W) * H, 0.f);
            last_ev.assign(static_cast<size_t>(W) * H, std::numeric_limits<long long>::min() / 2);
            const float *lp = logf.ptr<float>(0);
            for (size_t i = 0; i < ref_level.size(); ++i) {
                ref_level[i] = lp[i];
                prev_log[i]  = lp[i];
            }
            if (display) preview = cv::Mat::zeros(H, W, CV_8UC3);
            t_prev_us = 0;
            frame_idx = 1;
            continue;
        }

        const long long t_cur_us = static_cast<long long>(std::llround(frame_idx * frame_dt_us));
        const float *Lc = logf.ptr<float>(0);
        buf.clear();

        for (int y = 0; y < H; ++y) {
            const size_t row = static_cast<size_t>(y) * W;
            for (int x = 0; x < W; ++x) {
                const size_t i = row + x;
                const double cur = Lc[i];
                const double prv = prev_log[i];
                if (cur == prv) continue;
                const int pol = (cur > prv) ? +1 : -1;
                double level  = ref_level[i];
                const double denom = (cur - prv);
                // Emit one event per threshold crossing between prev and cur.
                while (true) {
                    const double next = level + pol * C;
                    const bool crossed = (pol > 0) ? (next <= cur) : (next >= cur);
                    if (!crossed) break;
                    // Linear-interpolate the crossing time within [t_prev, t_cur].
                    double frac = (next - prv) / denom;
                    if (frac < 0.0) frac = 0.0;
                    if (frac > 1.0) frac = 1.0;
                    const long long te =
                        t_prev_us + static_cast<long long>(std::llround(frac * (t_cur_us - t_prev_us)));
                    if (refr_us <= 0 || te - last_ev[i] >= refr_us) {
                        buf.push_back({te, x, y, pol > 0 ? 1 : 0});
                        last_ev[i] = te;
                    }
                    level = next;
                }
                ref_level[i] = static_cast<float>(level);
            }
            std::copy(Lc + row, Lc + row + W, prev_log.begin() + row);
        }

        std::sort(buf.begin(), buf.end(), [](const Ev &a, const Ev &b) { return a.t < b.t; });
        for (const Ev &e : buf) {
            out << e.t << ',' << e.x << ',' << e.y << ',' << e.p << '\n';
            ++total_events;
            if (max_ev > 0 && total_events >= max_ev) break;
        }

        if (display) {
            preview.setTo(cv::Scalar(0, 0, 0));
            for (const Ev &e : buf) {
                cv::Vec3b &px = preview.at<cv::Vec3b>(e.y, e.x);
                if (e.p) { px[1] = 220; px[2] = 60; } else { px[0] = 220; px[1] = 90; }
            }
            cv::imshow("video_to_event (events)", preview);
            if (cv::waitKey(1) == 27) break;
        }

        if (!quiet) {
            std::cout << "\r  frame " << frame_idx << "  t=" << (t_cur_us / 1e6) << "s  events=" << total_events
                      << "        " << std::flush;
        }

        t_prev_us = t_cur_us;
        ++frame_idx;
        if (max_ev > 0 && total_events >= max_ev) break;
    }

    out.flush();
    out.close();
    std::cout << "\n\n=== done ===\n"
              << "  frames : " << frame_idx << "\n"
              << "  events : " << total_events << "\n"
              << "  sensor : " << W << "x" << H << "\n"
              << "  file   : " << out_path << "\n";
    if (display) cv::destroyAllWindows();
    return 0;
}
