/**********************************************************************************************************************
 * Native frequency-based active-marker detector (Metavision SDK, C++).
 *
 * C++ port of detect_active_markers.py: identifies blinking LEDs by their fixed
 * blink frequency (the Arduino Nano 33 BLE firmware in arduino/active_markers/).
 * Unlike cpp/active_marker_detector (ID-encoded modulated light), this matches
 * each detected frequency to the nearest entry in active_markers.json.
 *
 * Pipeline (all native C++ SDK algorithms, no Python loop overhead):
 *   Metavision::FrequencyAlgorithm            -> per-pixel flicker frequency
 *   Metavision::FrequencyClusteringAlgorithm  -> group pixels into marker clusters
 *   nearest-frequency match                   -> assign a marker id/name/color
 *
 * Live OpenCV window (on by default) draws each identified marker; --no-display
 * runs headless with console reports only. Press q/ESC in the window to quit.
 **********************************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/cv/configs/frequency_estimation_config.h>
#include <metavision/sdk/cv/configs/frequency_clustering_algorithm_config.h>
#include <metavision/sdk/cv/events/event_frequency.h>
#include <metavision/sdk/cv/events/event_frequency_cluster.h>
#include <metavision/sdk/cv/algorithms/frequency_algorithm.h>
#include <metavision/sdk/cv/algorithms/frequency_clustering_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/file_config_hints.h>

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
    long long getll(const std::string &k, long long d) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? d : std::stoll(it->second);
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

struct Marker {
    int id            = 0;
    std::string name  = "";
    float freq_hz     = 0.f;
    cv::Scalar color  = cv::Scalar(0, 255, 0); // BGR
};

struct MarkerConfig {
    std::vector<Marker> markers;
    float min_freq  = 500.f;
    float max_freq  = 2000.f;
    float match_tol = 120.f;
};

/// Load the LED marker frequency table (same active_markers.json used by Python).
MarkerConfig load_marker_config(const std::string &path) {
    boost::property_tree::ptree root;
    boost::property_tree::read_json(path, root);

    MarkerConfig cfg;
    cfg.min_freq  = root.get<float>("band.min_freq_hz", 500.f);
    cfg.max_freq  = root.get<float>("band.max_freq_hz", 2000.f);
    cfg.match_tol = root.get<float>("match_tolerance_hz", 120.f);

    for (const auto &item : root.get_child("markers")) {
        Marker m;
        m.id      = item.second.get<int>("id", 0);
        m.name    = item.second.get<std::string>("name", "");
        m.freq_hz = item.second.get<float>("freq_hz", 0.f);
        std::vector<int> bgr;
        if (auto c = item.second.get_child_optional("color_bgr")) {
            for (const auto &cc : *c) {
                bgr.push_back(cc.second.get_value<int>());
            }
        }
        if (bgr.size() == 3) {
            m.color = cv::Scalar(bgr[0], bgr[1], bgr[2]);
        }
        cfg.markers.push_back(m);
    }
    return cfg;
}

/// Match a frequency to the nearest known marker within tolerance (nullptr if none).
const Marker *assign_marker(float freq_hz, const MarkerConfig &cfg) {
    const Marker *best = nullptr;
    float best_diff    = cfg.match_tol;
    for (const auto &m : cfg.markers) {
        const float d = std::fabs(freq_hz - m.freq_hz);
        if (d <= best_diff) {
            best_diff = d;
            best      = &m;
        }
    }
    return best;
}

void print_usage() {
    std::cout <<
        "active_marker_freq_detector - native frequency-based active-marker detector\n\n"
        "C++ port of detect_active_markers.py: FrequencyAlgorithm + FrequencyClustering\n"
        "identify blinking LEDs by fixed blink frequency, matched to active_markers.json.\n\n"
        "Input:\n"
        "  -i, --input <file>              .raw/.hdf5 recording (omit to use a live camera)\n"
        "      --realtime-playback-speed   play files at real time (default on)\n\n"
        "Markers:\n"
        "  -c, --config <file>             marker table JSON (default active_markers.json)\n\n"
        "Frequency estimation (FrequencyAlgorithm):\n"
        "      --min-freq <hz>             minimum frequency (default from config band)\n"
        "      --max-freq <hz>             maximum frequency (default from config band)\n"
        "      --filter-length <n>         periods before output (default 4)\n"
        "      --max-period-diff <us>      same-period threshold us (default 200)\n\n"
        "Clustering (FrequencyClusteringAlgorithm):\n"
        "      --min-cluster-size <px>     minimum cluster size (default 10)\n"
        "      --max-freq-diff <hz>        freq diff to join a cluster (default 100)\n"
        "      --max-time-diff <us>        time diff to join a cluster (default 100000)\n"
        "      --filter-alpha <f>          cluster position smoothing (default 0.1)\n"
        "      --inactivity-us <us>        drop a marker after this idle time (default 200000)\n\n"
        "Display:\n"
        "      --no-display                disable the live OpenCV window (headless)\n"
        "      --display-fps <f>           window refresh rate (default 25)\n"
        "      --accumulation-time-us <us> event accumulation per frame (default 10000)\n\n"
        "Output:\n"
        "      --report-period <s>         console report interval seconds (default 1.0)\n"
        "  -h, --help                      show this help\n";
}

} // namespace

int main(int argc, char **argv) {
    Args args(argc, argv);
    if (args.has("help") || args.has("h")) {
        print_usage();
        return 0;
    }

    const std::string input      = args.get("input", args.get("i", ""));
    const std::string config_path = args.get("config", args.get("c", "active_markers.json"));
    const bool realtime          = args.getb("realtime-playback-speed", true);

    MarkerConfig cfg;
    try {
        cfg = load_marker_config(config_path);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load marker config '" << config_path << "': " << e.what() << "\n";
        return 1;
    }
    if (cfg.markers.empty()) {
        std::cerr << "No markers found in " << config_path << "\n";
        return 1;
    }

    const float min_freq              = args.getf("min-freq", cfg.min_freq);
    const float max_freq              = args.getf("max-freq", cfg.max_freq);
    const unsigned int filter_length  = static_cast<unsigned int>(args.geti("filter-length", 4));
    const unsigned int diff_thresh_us = static_cast<unsigned int>(args.geti("max-period-diff", 200));

    const int min_cluster_size = args.geti("min-cluster-size", 10);
    const float max_freq_diff  = args.getf("max-freq-diff", 100.f);
    const Metavision::timestamp max_time_diff =
        static_cast<Metavision::timestamp>(args.getll("max-time-diff", 100000));
    const float filter_alpha = args.getf("filter-alpha", 0.1f);
    const Metavision::timestamp inactivity_us =
        static_cast<Metavision::timestamp>(args.getll("inactivity-us", 200000));

    const double report_period_s = args.getf("report-period", 1.0f);

    Metavision::Camera camera;
    try {
        if (input.empty()) {
            camera = Metavision::Camera::from_first_available();
        } else {
            camera = Metavision::Camera::from_file(
                input, Metavision::FileConfigHints().real_time_playback(realtime));
        }
    } catch (const std::exception &e) {
        std::cerr << "Camera init failed: " << e.what() << "\n";
        return 1;
    }

    const int width  = camera.geometry().width();
    const int height = camera.geometry().height();

    Metavision::FrequencyEstimationConfig freq_cfg(filter_length, min_freq, max_freq, diff_thresh_us, false);
    Metavision::FrequencyAlgorithm freq_algo(width, height, freq_cfg);

    Metavision::FrequencyClusteringAlgorithmConfig clus_cfg(min_cluster_size, max_freq_diff, max_time_diff,
                                                            filter_alpha);
    Metavision::FrequencyClusteringAlgorithm clustering(width, height, clus_cfg);

    std::cout << "Active-marker detector (C++, frequency-based, native SDK)\n"
              << "  sensor    : " << width << "x" << height << "\n"
              << "  frequency : " << min_freq << "-" << max_freq << " Hz, match_tol=+/-"
              << cfg.match_tol << " Hz\n"
              << "  clustering: min_size=" << min_cluster_size << " max_freq_diff=" << max_freq_diff
              << " max_time_diff=" << max_time_diff << "us\n"
              << "  markers   : ";
    for (const auto &m : cfg.markers) {
        std::cout << "M" << m.id << " " << m.name << "=" << m.freq_hz << "Hz  ";
    }
    std::cout << "\n  algorithms: FrequencyAlgorithm + FrequencyClusteringAlgorithm\n"
              << "  source    : " << (input.empty() ? "live camera" : input) << "\n\n";

    std::vector<Metavision::Event2dFrequency<float>> freqs;
    std::vector<Metavision::Event2dFrequencyCluster<float>> clusters;
    std::map<std::uint32_t, Metavision::Event2dFrequencyCluster<float>> latest;
    Metavision::timestamp last_report = 0;
    const Metavision::timestamp report_period_us =
        static_cast<Metavision::timestamp>(report_period_s * 1e6);

    // Optional live visualization.
    const bool display = !args.getb("no-display", false);
    std::mutex frame_mtx;
    cv::Mat display_frame;
    bool has_new_frame = false;
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm> frame_gen;
    if (display) {
        frame_gen.reset(new Metavision::PeriodicFrameGenerationAlgorithm(
            width, height, static_cast<std::uint32_t>(args.geti("accumulation-time-us", 10000)),
            args.getf("display-fps", 25.f)));
        frame_gen->set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
            for (const auto &kv : latest) {
                const auto &c    = kv.second;
                const Marker *m  = assign_marker(c.frequency, cfg);
                const cv::Scalar color = m ? m->color : cv::Scalar(150, 150, 150);
                const int r = std::min(60, std::max(8, static_cast<int>(std::sqrt(
                                  static_cast<double>(c.n_pixels)) * 2.0)));
                const cv::Point ctr(static_cast<int>(c.x), static_cast<int>(c.y));
                cv::circle(frame, ctr, r, color, 2);
                cv::drawMarker(frame, ctr, color, cv::MARKER_CROSS, 12, 2);
                std::ostringstream lbl;
                lbl << std::fixed << std::setprecision(0);
                if (m) {
                    lbl << "M" << m->id << " " << m->name << " " << c.frequency << "Hz";
                } else {
                    lbl << "? " << c.frequency << "Hz";
                }
                cv::putText(frame, lbl.str(), cv::Point(ctr.x - r, ctr.y - r - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
            }
            std::lock_guard<std::mutex> lk(frame_mtx);
            frame.copyTo(display_frame);
            has_new_frame = true;
        });
    }

    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        freqs.clear();
        freq_algo.process_events(begin, end, std::back_inserter(freqs));

        clusters.clear();
        // The SDK only exports the const-pointer instantiation of process_events.
        const Metavision::Event2dFrequency<float> *freqs_begin = freqs.data();
        clustering.process_events(freqs_begin, freqs_begin + freqs.size(), std::back_inserter(clusters));
        for (const auto &c : clusters) {
            latest[c.id] = c;
        }

        if (begin == end) {
            return;
        }
        const Metavision::timestamp t = (end - 1)->t;

        // Expire markers that have not been updated recently.
        for (auto it = latest.begin(); it != latest.end();) {
            if (t - it->second.t > inactivity_us) {
                it = latest.erase(it);
            } else {
                ++it;
            }
        }

        if (frame_gen) {
            frame_gen->process_events(begin, end);
        }

        if (t - last_report >= report_period_us) {
            last_report = t;
            if (latest.empty()) {
                std::cout << "[" << std::fixed << std::setprecision(2) << (t / 1e6)
                          << "s] 0 marker(s)\n";
            } else {
                std::cout << "[" << std::fixed << std::setprecision(2) << (t / 1e6) << "s] "
                          << latest.size() << " marker(s):";
                for (const auto &kv : latest) {
                    const auto &c   = kv.second;
                    const Marker *m = assign_marker(c.frequency, cfg);
                    std::cout << "  ";
                    if (m) {
                        std::cout << "M" << m->id << " " << m->name;
                    } else {
                        std::cout << "?";
                    }
                    std::cout << " " << std::setprecision(1) << c.frequency << "Hz n_px=" << c.n_pixels
                              << " @(" << std::setprecision(0) << c.x << "," << c.y << ")";
                }
                std::cout << std::endl;
            }
        }
    });

    try {
        camera.start();
        if (display) {
            const std::string win = "Active Marker Detector (frequency)";
            cv::namedWindow(win, cv::WINDOW_NORMAL);
            cv::resizeWindow(win, width, height);
            cv::Mat shown;
            while (camera.is_running()) {
                bool got = false;
                {
                    std::lock_guard<std::mutex> lk(frame_mtx);
                    if (has_new_frame) {
                        display_frame.copyTo(shown);
                        has_new_frame = false;
                        got           = true;
                    }
                }
                if (got && !shown.empty()) {
                    cv::imshow(win, shown);
                }
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    break;
                }
            }
            cv::destroyAllWindows();
        } else {
            while (camera.is_running()) {
                std::this_thread::yield();
            }
        }
        camera.stop();
    } catch (const std::exception &e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Done.\n";
    return 0;
}
