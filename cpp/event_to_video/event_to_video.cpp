/**********************************************************************************************************************
 * event_to_video - native C++ intensity reconstruction from events (deterministic).
 *
 * Reconstructs a conventional grayscale video from a Prophesee event stream WITHOUT
 * a trained network, using the Scheerlinck et al. "complementary filter" / high-pass
 * direct-integration model - the classical, real-time alternative to the learned
 * E2VID in metavision_core_ml/event_to_video:
 *
 *   per pixel, log-intensity estimate L obeys
 *       dL/dt = -alpha * L  +  C * sum_k p_k * delta(t - t_k)
 *   i.e. each event nudges L by +/- C (contrast) and L leaks toward 0 at rate alpha
 *   (rad/s cutoff). The displayed pixel is  clip(0.5 + gain * L).
 *
 * Input: live camera or a .raw/.hdf5 recording (Metavision SDK). Output: an .avi/.mp4
 * via OpenCV VideoWriter and/or a live window. Headless unless --display/--output set.
 **********************************************************************************************************************/

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/file_config_hints.h>

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
    double getd(const std::string &k, double d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stod(it->second);
    }
    long long getll(const std::string &k, long long d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stoll(it->second);
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
        "event_to_video - native C++ intensity reconstruction from events (deterministic)\n\n"
        "Scheerlinck complementary-filter / high-pass reconstruction. No trained model.\n\n"
        "Input:\n"
        "  -i, --input <file>        .raw/.hdf5 recording (omit for a live camera)\n\n"
        "Reconstruction:\n"
        "      --contrast <C>        log-intensity step per event (default 0.15)\n"
        "      --cutoff <rad/s>      leak / high-pass cutoff alpha (default 5.0)\n"
        "      --gain <g>            display gain on log-intensity (default 0.6)\n"
        "      --fps <hz>            output/render frame rate (default 30)\n\n"
        "Output:\n"
        "  -o, --output <pattern>    write a PNG frame sequence; give a printf pattern\n"
        "                            like results/recon_%06d.png (a plain name/dir gets\n"
        "                            _%06d.png appended). No video codec needed.\n"
        "      --display             show a live reconstruction window\n"
        "      --max-duration <s>    stop after s seconds (0 = unlimited)\n"
        "  -h, --help                this message\n";
}

} // namespace

int main(int argc, char **argv) {
    Args args(argc, argv);
    if (args.getb("help", false) || args.getb("h", false)) {
        print_usage();
        return 0;
    }

    const std::string input   = !args.get("input").empty() ? args.get("input") : args.get("i");
    const std::string out_vid = !args.get("output").empty() ? args.get("output") : args.get("o");
    const double C        = args.getd("contrast", 0.15);
    const double alpha    = args.getd("cutoff", 5.0);
    const double gain     = args.getd("gain", 0.6);
    const double fps      = args.getd("fps", 30.0);
    const double max_dur  = args.getd("max-duration", 0.0);
    const bool display    = args.getb("display", false);

    if (out_vid.empty() && !display) {
        std::cerr << "note: neither --output nor --display given; defaulting to --display\n";
    }
    const bool do_display = display || out_vid.empty();

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

    std::vector<float> L(N, 0.f);            // log-intensity estimate
    std::vector<Metavision::timestamp> tlast(N, 0); // last update time per pixel
    std::mutex state_mtx;
    std::atomic<Metavision::timestamp> cur_t{0};
    std::atomic<bool> stop{false};

    const double alpha_us = alpha * 1e-6; // per-microsecond leak rate

    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::lock_guard<std::mutex> lk(state_mtx);
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            const size_t i = static_cast<size_t>(ev->y) * W + ev->x;
            const double dt = static_cast<double>(ev->t - tlast[i]);
            if (dt > 0) L[i] *= static_cast<float>(std::exp(-alpha_us * dt));
            L[i] += static_cast<float>((ev->p ? +1.0 : -1.0) * C);
            tlast[i] = ev->t;
        }
        if (end != begin) cur_t.store((end - 1)->t, std::memory_order_relaxed);
    });

    // The bundled OpenCV ships no video-encoder backend, so persist frames as a PNG
    // sequence via imwrite (always available). Build a printf pattern from --output.
    std::string seq_pattern;
    if (!out_vid.empty()) {
        if (out_vid.find('%') != std::string::npos) {
            seq_pattern = out_vid;
        } else {
            const size_t dot = out_vid.find_last_of('.');
            const std::string stem = (dot == std::string::npos) ? out_vid : out_vid.substr(0, dot);
            seq_pattern = stem + "_%06d.png";
        }
    }
    long long frame_no = 0;
    char name_buf[1024];
    const Metavision::timestamp frame_period_us = static_cast<Metavision::timestamp>(1e6 / fps);

    std::cout << "event_to_video - complementary-filter reconstruction (C++)\n"
              << "  source   : " << (input.empty() ? "LIVE CAMERA" : input) << "\n"
              << "  sensor   : " << W << "x" << H << "\n"
              << "  contrast : " << C << "  cutoff=" << alpha << " rad/s  gain=" << gain << "\n"
              << "  render   : " << fps << " fps"
              << (out_vid.empty() ? "" : ("  -> " + out_vid)) << "\n\n";
    if (do_display) std::cout << "press Q or Esc in the window to stop\n";

    cv::Mat img8(H, W, CV_8UC1);

    try {
        camera.start();
        Metavision::timestamp next_frame = frame_period_us;
        const Metavision::timestamp end_t =
            max_dur > 0 ? static_cast<Metavision::timestamp>(max_dur * 1e6) : 0;

        while (camera.is_running() && !stop.load()) {
            const Metavision::timestamp now = cur_t.load(std::memory_order_relaxed);
            if (now < next_frame) {
                std::this_thread::yield();
                continue;
            }
            next_frame += frame_period_us;

            {
                std::lock_guard<std::mutex> lk(state_mtx);
                for (int y = 0; y < H; ++y) {
                    uchar *row = img8.ptr<uchar>(y);
                    const size_t base = static_cast<size_t>(y) * W;
                    for (int x = 0; x < W; ++x) {
                        const size_t i = base + x;
                        const double dt = static_cast<double>(now - tlast[i]);
                        if (dt > 0) {
                            L[i] *= static_cast<float>(std::exp(-alpha_us * dt));
                            tlast[i] = now;
                        }
                        double v = 0.5 + gain * L[i];
                        if (v < 0.0) v = 0.0;
                        if (v > 1.0) v = 1.0;
                        row[x] = static_cast<uchar>(v * 255.0 + 0.5);
                    }
                }
            }

            if (!seq_pattern.empty()) {
                std::snprintf(name_buf, sizeof(name_buf), seq_pattern.c_str(), frame_no++);
                cv::imwrite(name_buf, img8);
            }
            if (do_display) {
                cv::imshow("event_to_video (reconstruction)", img8);
                const int k = cv::waitKey(1);
                if (k == 27 || k == 'q' || k == 'Q') stop.store(true);
            }
            if (end_t > 0 && now >= end_t) stop.store(true);
        }
        camera.stop();
    } catch (const std::exception &e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return 1;
    }

    if (do_display) cv::destroyAllWindows();
    std::cout << "\nDone.";
    if (!seq_pattern.empty()) std::cout << "  wrote " << frame_no << " frames: " << seq_pattern;
    std::cout << "\n";
    return 0;
}
