/**********************************************************************************************************************
 * Native active-marker detector (Metavision SDK, C++).
 *
 * WHY C++: this tool is built on two SDK Pro algorithms that ship WITHOUT Python
 * bindings, so they cannot be used from the workspace's Python scripts:
 *   - Metavision::ModulatedLightDetectorAlgorithm  (metavision_sdk_cv)
 *   - Metavision::ActiveMarkerTrackerAlgorithm      (metavision_sdk_cv)
 *
 * These decode LEDs that transmit an ID via *inter-blink timing* (base period p):
 *   p -> '0', 2p -> '1', 3p -> start bit. This is DIFFERENT from the frequency-
 *   encoded firmware used by detect_active_markers.py (fixed Hz per LED). To use
 *   this detector the markers must run ID-encoded (modulated-light) firmware.
 *
 * Headless: reports tracked markers to stdout. No OpenCV / UI dependency.
 **********************************************************************************************************************/

#include <cctype>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/cv/events/event_source_id.h>
#include <metavision/sdk/cv/events/event_active_track.h>
#include <metavision/sdk/cv/algorithms/modulated_light_detector_algorithm.h>
#include <metavision/sdk/cv/algorithms/active_marker_tracker_algorithm.h>
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

std::set<std::uint32_t> parse_led_ids(const std::string &csv) {
    std::set<std::uint32_t> ids;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const auto a = tok.find_first_not_of(" \t");
        if (a == std::string::npos) {
            continue;
        }
        const auto b = tok.find_last_not_of(" \t");
        ids.insert(static_cast<std::uint32_t>(std::stoul(tok.substr(a, b - a + 1))));
    }
    return ids;
}

/// Dependency-free JSON id scanner: collects every integer value bound to an
/// "id" key. Handles both the SDK "active marker" format ({"led":{"id":N}}) and
/// the workspace active_markers.json format ({"markers":[{"id":N,...}]}).
std::set<std::uint32_t> scan_ids_from_json(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open JSON file: " + path);
    }
    std::stringstream buf;
    buf << f.rdbuf();
    const std::string s = buf.str();

    std::set<std::uint32_t> ids;
    const std::string key = "\"id\"";
    size_t pos = 0;
    while ((pos = s.find(key, pos)) != std::string::npos) {
        size_t i = pos + key.size();
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == ':' || s[i] == '\n' || s[i] == '\r')) {
            ++i;
        }
        size_t j = i;
        while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) {
            ++j;
        }
        if (j > i) {
            ids.insert(static_cast<std::uint32_t>(std::stoul(s.substr(i, j - i))));
        }
        pos = pos + key.size();
    }
    return ids;
}

void print_usage() {
    std::cout <<
        "active_marker_detector - native Metavision active-marker detector (C++-only algorithms)\n\n"
        "Uses ModulatedLightDetectorAlgorithm + ActiveMarkerTrackerAlgorithm, which have no\n"
        "Python bindings. Markers must run ID-encoded modulated-light firmware (inter-blink\n"
        "timing), NOT the fixed-frequency firmware used by detect_active_markers.py.\n\n"
        "Input:\n"
        "  -i, --input <file>              .raw/.hdf5 recording (omit to use a live camera)\n"
        "  -j, --camera-config <file>      optional camera settings file (live only)\n"
        "      --realtime-playback-speed   play files at real time (default on)\n\n"
        "Markers to track (choose one):\n"
        "      --led-ids 1,2,3,4           comma-separated marker IDs\n"
        "  -a, --am-json <file>            JSON listing marker IDs (SDK or workspace format)\n\n"
        "Detector (ModulatedLightDetectorAlgorithm):\n"
        "      --detector-num-bits <n>          word size in bits (default 8)\n"
        "      --detector-base-period-us <us>   base blink period (default 200)\n"
        "      --detector-tolerance <f>         blink tolerance (default 0.1)\n\n"
        "Tracker (ActiveMarkerTrackerAlgorithm):\n"
        "      --tracker-radius <f>                influence radius px (default 30)\n"
        "      --tracker-inactivity-period-us <us> lost after inactivity (default 1000)\n"
        "      --tracker-monitoring-frequency <hz> monitoring rate (default 30)\n"
        "      --tracker-distance-percentage <f>   radius update pct (default 0.3)\n"
        "      --tracker-alpha-pos <f>             position smoothing (default 0.05)\n"
        "      --tracker-update-radius <0|1>       auto radius update (default 1)\n\n"
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

    const std::string input   = args.get("input", args.get("i", ""));
    const std::string cam_cfg = args.get("camera-config", args.get("j", ""));
    const std::string am_json = args.get("am-json", args.get("a", ""));

    std::set<std::uint32_t> led_ids;
    try {
        if (args.has("led-ids")) {
            led_ids = parse_led_ids(args.get("led-ids"));
        } else if (!am_json.empty()) {
            led_ids = scan_ids_from_json(am_json);
        }
    } catch (const std::exception &e) {
        std::cerr << "Error loading marker IDs: " << e.what() << "\n";
        return 1;
    }
    if (led_ids.empty()) {
        std::cerr << "No LED IDs provided. Use --am-json <file> or --led-ids 1,2,3.\n\n";
        print_usage();
        return 1;
    }

    Metavision::ModulatedLightDetectorAlgorithm::Params det;
    det.num_bits       = static_cast<std::uint8_t>(args.geti("detector-num-bits", 8));
    det.base_period_us = static_cast<std::uint32_t>(args.geti("detector-base-period-us", 200));
    det.tolerance      = args.getf("detector-tolerance", 0.1f);

    Metavision::ActiveMarkerTrackerAlgorithm::Params trk;
    trk.update_radius          = args.getb("tracker-update-radius", true);
    trk.inactivity_period_us   = static_cast<Metavision::timestamp>(args.getll("tracker-inactivity-period-us", 1000));
    trk.monitoring_frequency_hz = args.getf("tracker-monitoring-frequency", 30.f);
    trk.radius                 = args.getf("tracker-radius", 30.f);
    trk.distance_pct           = args.getf("tracker-distance-percentage", 0.3f);
    trk.alpha_pos              = args.getf("tracker-alpha-pos", 0.05f);

    const double report_period_s = args.getf("report-period", 1.0f);
    const bool realtime          = args.getb("realtime-playback-speed", true);

    Metavision::Camera camera;
    try {
        if (input.empty()) {
            camera = Metavision::Camera::from_first_available();
            if (!cam_cfg.empty()) {
                camera.load(cam_cfg);
            }
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
    det.width  = static_cast<std::uint16_t>(width);
    det.height = static_cast<std::uint16_t>(height);

    std::cout << "Active-marker detector (C++, SDK-native, no Python wrapper)\n"
              << "  sensor    : " << width << "x" << height << "\n"
              << "  markers   : ";
    for (auto id : led_ids) {
        std::cout << id << " ";
    }
    std::cout << "\n  detector  : num_bits=" << static_cast<int>(det.num_bits)
              << " base_period_us=" << det.base_period_us << " tol=" << det.tolerance << "\n"
              << "  algorithms: ModulatedLightDetectorAlgorithm + ActiveMarkerTrackerAlgorithm\n"
              << "  source    : " << (input.empty() ? "live camera" : input) << "\n\n";

    Metavision::ModulatedLightDetectorAlgorithm detector(det);
    Metavision::ActiveMarkerTrackerAlgorithm tracker(trk, led_ids);

    std::vector<Metavision::EventSourceId> src_ids;
    std::vector<Metavision::EventActiveTrack> tracks;
    std::map<std::uint32_t, Metavision::EventActiveTrack> latest;
    Metavision::timestamp last_report = 0;
    const Metavision::timestamp report_period_us =
        static_cast<Metavision::timestamp>(report_period_s * 1e6);

    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        src_ids.clear();
        detector.process_events(begin, end, std::back_inserter(src_ids));

        tracks.clear();
        tracker.process_events(src_ids.cbegin(), src_ids.cend(), std::back_inserter(tracks));
        for (const auto &tr : tracks) {
            if (tr.status == Metavision::EventActiveTrack::Status::Lost) {
                latest.erase(tr.id);
            } else {
                latest[tr.id] = tr;
            }
        }

        if (begin != end) {
            const Metavision::timestamp t = (end - 1)->t;
            tracker.notify_elapsed_time(t);
            if (t - last_report >= report_period_us) {
                last_report = t;
                std::cout << "[" << std::fixed << std::setprecision(2) << (t / 1e6) << "s] "
                          << latest.size() << " marker(s) tracked";
                for (const auto &kv : latest) {
                    const auto &tr = kv.second;
                    std::cout << "  id" << kv.first << "@(" << std::setprecision(0) << tr.x << ","
                              << tr.y << ") r=" << std::setprecision(1) << tr.radius;
                }
                std::cout << std::endl;
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
