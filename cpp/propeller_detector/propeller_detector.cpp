/**********************************************************************************************************************
 * Native propeller / rotating-source detector (Metavision SDK, C++).
 *
 * Pipeline (all native C++ SDK algorithms):
 *   [optional] Metavision::ProximityFilterAlgorithm    -> focus on a region (C++-only: no Python binding)
 *              Metavision::FrequencyAlgorithm          -> per-pixel flicker frequency (Event2dFrequency)
 *              Metavision::FrequencyClusteringAlgorithm -> group pixels into rotating-object clusters
 *
 * For each cluster it reports frequency (Hz), RPM (= freq / blades * 60), pixel
 * count and position. Headless: console output only, no OpenCV / UI dependency.
 *
 * This mirrors the Python detect_propeller.py --use-sdk-clustering path, but the
 * ProximityFilterAlgorithm focus stage is only available here (C++ has no wrapper).
 **********************************************************************************************************************/

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/cv/configs/frequency_estimation_config.h>
#include <metavision/sdk/cv/configs/frequency_clustering_algorithm_config.h>
#include <metavision/sdk/cv/events/event_frequency.h>
#include <metavision/sdk/cv/events/event_frequency_cluster.h>
#include <metavision/sdk/cv/algorithms/frequency_algorithm.h>
#include <metavision/sdk/cv/algorithms/frequency_clustering_algorithm.h>
#include <metavision/sdk/cv/algorithms/proximity_filter_algorithm.h>
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

void print_usage() {
    std::cout <<
        "propeller_detector - native Metavision propeller / rotating-source detector\n\n"
        "FrequencyAlgorithm + FrequencyClusteringAlgorithm, with optional C++-only\n"
        "ProximityFilterAlgorithm focus. Reports frequency, RPM, pixels and position.\n\n"
        "Input:\n"
        "  -i, --input <file>              .raw/.hdf5 recording (omit to use a live camera)\n"
        "      --realtime-playback-speed   play files at real time (default on)\n\n"
        "Frequency estimation (FrequencyAlgorithm):\n"
        "      --min-freq <hz>             minimum frequency (default 10)\n"
        "      --max-freq <hz>             maximum frequency (default 300)\n"
        "      --filter-length <n>         periods before output (default 4)\n"
        "      --max-period-diff <us>      same-period threshold us (default 1500)\n"
        "      --num-blades <n>            blades per propeller for RPM (default 3)\n\n"
        "Clustering (FrequencyClusteringAlgorithm):\n"
        "      --min-cluster-size <px>     minimum cluster size (default 3)\n"
        "      --max-freq-diff <hz>        freq diff to join a cluster (default 10)\n"
        "      --max-time-diff <us>        time diff to join a cluster (default 100000)\n"
        "      --filter-alpha <f>          cluster position smoothing (default 0.1)\n\n"
        "Focus region (ProximityFilterAlgorithm, C++-only; enabled if any is set):\n"
        "      --focus-x <px>              center x\n"
        "      --focus-y <px>              center y\n"
        "      --focus-radius <px>         max distance from center (default 100)\n\n"
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

    const std::string input = args.get("input", args.get("i", ""));
    const bool realtime     = args.getb("realtime-playback-speed", true);

    const float min_freq              = args.getf("min-freq", 10.f);
    const float max_freq              = args.getf("max-freq", 300.f);
    const unsigned int filter_length  = static_cast<unsigned int>(args.geti("filter-length", 4));
    const unsigned int diff_thresh_us = static_cast<unsigned int>(args.geti("max-period-diff", 1500));
    const int num_blades              = std::max(1, args.geti("num-blades", 3));

    const int min_cluster_size          = args.geti("min-cluster-size", 3);
    const float max_freq_diff           = args.getf("max-freq-diff", 10.f);
    const Metavision::timestamp max_time_diff =
        static_cast<Metavision::timestamp>(args.getll("max-time-diff", 100000));
    const float filter_alpha = args.getf("filter-alpha", 0.1f);

    const bool use_focus = args.has("focus-x") || args.has("focus-y") || args.has("focus-radius");
    const float focus_x  = args.getf("focus-x", 0.f);
    const float focus_y  = args.getf("focus-y", 0.f);
    const float focus_r  = args.getf("focus-radius", 100.f);

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

    std::optional<Metavision::ProximityFilterAlgorithm> prox;
    if (use_focus) {
        prox.emplace(Eigen::Vector2f(focus_x, focus_y), focus_r);
    }

    std::cout << "Propeller detector (C++, native SDK)\n"
              << "  sensor    : " << width << "x" << height << "\n"
              << "  frequency : " << min_freq << "-" << max_freq << " Hz, blades=" << num_blades << "\n"
              << "  clustering: min_size=" << min_cluster_size << " max_freq_diff=" << max_freq_diff
              << " max_time_diff=" << max_time_diff << "us\n"
              << "  focus     : " << (use_focus ? "ProximityFilterAlgorithm ON" : "off");
    if (use_focus) {
        std::cout << " center=(" << focus_x << "," << focus_y << ") r=" << focus_r;
    }
    std::cout << "\n  algorithms: FrequencyAlgorithm + FrequencyClusteringAlgorithm"
              << (use_focus ? " + ProximityFilterAlgorithm (C++-only)" : "") << "\n"
              << "  source    : " << (input.empty() ? "live camera" : input) << "\n\n";

    std::vector<Metavision::EventCD> focused;
    std::vector<Metavision::Event2dFrequency<float>> freqs;
    std::vector<Metavision::Event2dFrequencyCluster<float>> clusters;
    std::map<std::uint32_t, Metavision::Event2dFrequencyCluster<float>> latest;
    Metavision::timestamp last_report = 0;
    const Metavision::timestamp report_period_us =
        static_cast<Metavision::timestamp>(report_period_s * 1e6);

    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        const Metavision::EventCD *b = begin;
        const Metavision::EventCD *e = end;
        if (prox) {
            focused.clear();
            prox->process_events(begin, end, std::back_inserter(focused));
            b = focused.data();
            e = focused.data() + focused.size();
        }

        freqs.clear();
        freq_algo.process_events(b, e, std::back_inserter(freqs));

        clusters.clear();
        // The SDK only exports the const-pointer instantiation of process_events.
        const Metavision::Event2dFrequency<float> *freqs_begin = freqs.data();
        clustering.process_events(freqs_begin, freqs_begin + freqs.size(), std::back_inserter(clusters));
        for (const auto &c : clusters) {
            latest[c.id] = c;
        }

        if (begin != end) {
            const Metavision::timestamp t = (end - 1)->t;
            if (t - last_report >= report_period_us) {
                last_report = t;
                if (latest.empty()) {
                    std::cout << "[" << std::fixed << std::setprecision(2) << (t / 1e6)
                              << "s] no rotating source\n";
                } else {
                    std::cout << "[" << std::fixed << std::setprecision(2) << (t / 1e6) << "s] "
                              << latest.size() << " candidate(s):";
                    for (const auto &kv : latest) {
                        const auto &c   = kv.second;
                        const double rpm = (static_cast<double>(c.frequency) / num_blades) * 60.0;
                        std::cout << "  id" << c.id << " " << std::setprecision(1) << c.frequency << "Hz "
                                  << std::setprecision(0) << rpm << "rpm n_px=" << c.n_pixels << " @("
                                  << c.x << "," << c.y << ")";
                    }
                    std::cout << std::endl;
                }
                latest.clear();
            }
        }
    });

    try {
        camera.start();
        while (camera.is_running()) {
            std::this_thread::yield();
        }
        camera.stop();
    } catch (const std::exception &e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Done.\n";
    return 0;
}
