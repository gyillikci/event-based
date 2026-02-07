/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <thread>
#include <opencv2/opencv.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>
#include <boost/circular_buffer.hpp>

#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/cv/events/event_source_id.h>
#include <metavision/sdk/cv/algorithms/modulated_light_detector_algorithm.h>
#include <metavision/sdk/cv/algorithms/active_marker_tracker_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/mt_window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

struct Config {
    std::string event_file_path;
    std::string cam_config_path;
    std::string am_json_path;
    bool no_display;
    bool realtime_playback_speed;
    bool display_ids;
    std::uint32_t track_length;

    Metavision::ModulatedLightDetectorAlgorithm::Params detector_params;
    Metavision::ActiveMarkerTrackerAlgorithm::Params tracker_params;
};

/// @brief Class in charge of printing the trail of several tracks
class TrackPrinter {
public:
    TrackPrinter(std::uint16_t width, std::uint16_t height, const std::set<std::uint32_t> &sources,
                 std::uint32_t track_length, bool display_ids) :
        width_(width), height_(height), display_ids_(display_ids) {
        size_t i = 0;
        for (const auto &id : sources) {
            tracks_history_[id] = boost::circular_buffer<cv::Point2f>(track_length);
            update_map_[id]     = false;
        }

        generate_random_colors();
    }

    template<typename InputIt>
    void process_events(InputIt begin, InputIt end) {
        for (auto it = begin; it != end; ++it) {
            auto &track_history = tracks_history_.at(it->id);

            if (it->status == Metavision::EventActiveTrack::Status::Lost) {
                continue;
            }

            track_history.push_back(cv::Point2f(it->x, it->y));

            radius_by_id_[it->id] = it->radius;
            update_map_[it->id]   = true;
        }
    }

    void print(cv::Mat &frame) {
        remove_tracks_from_non_updated_histories();
        print_histories(frame);
        reset_update_map();
    }

private:
    static constexpr size_t kNumberOfTracksToRemove = 1000;
    static constexpr size_t kNumColors              = 500;

    void generate_random_colors() {
        colors_.resize(kNumColors);
        for (auto &c : colors_) {
            c = cv::Vec3b(static_cast<double>(std::rand()) / RAND_MAX * 255,
                          static_cast<double>(std::rand()) / RAND_MAX * 255,
                          static_cast<double>(std::rand()) / RAND_MAX * 255);
        }
    }
    void remove_tracks_from_non_updated_histories() {
        for (const auto &p : update_map_) {
            if (!p.second) {
                auto &history = tracks_history_[p.first];

                history.erase_begin(std::min(kNumberOfTracksToRemove, history.size()));
            }
        }
    }
    void print_histories(cv::Mat &frame) {
        for (const auto &p : tracks_history_) {
            const auto &id      = p.first;
            const auto &history = p.second;
            const auto &color   = colors_[id];

            if (history.empty())
                continue;

            for (const auto &pos : history) {
                frame.at<cv::Vec3b>(pos) = color;
            }

            if (display_ids_) {
                cv::putText(frame, std::to_string(id), history.back(), cv::FONT_HERSHEY_DUPLEX, 0.5, color);
            } else if (radius_by_id_.count(id) > 0) {
                const auto radius = radius_by_id_.at(id);
                cv::circle(frame, history.back(), radius, color);
            }
        }
    }
    void reset_update_map() {
        for (auto &p : update_map_) {
            p.second = false;
        }
    }

    const std::uint16_t width_;
    const std::uint16_t height_;
    const bool display_ids_;
    std::unordered_map<std::uint32_t, boost::circular_buffer<cv::Point2f>> tracks_history_;
    std::unordered_map<std::uint32_t, bool> update_map_;
    std::vector<cv::Vec3b> colors_;

    std::unordered_map<std::uint32_t, float> radius_by_id_;
};

/// @brief Class in charge of displaying tracks
class DisplayManager {
public:
    DisplayManager(std::uint16_t width, std::uint16_t height, const std::set<std::uint32_t> &ids,
                   std::uint32_t track_length, bool display_ids) :
        window_("Active Marker Tracking", width, height, Metavision::BaseWindow::RenderMode::BGR),
        printer_(width, height, ids, track_length, display_ids) {
        using Condition       = Metavision::EventBufferReslicerAlgorithm::Condition;
        using ConditionStatus = Metavision::EventBufferReslicerAlgorithm::ConditionStatus;
        const auto condition  = Condition::make_n_us(20000);

        slicer_.set_slicing_condition(condition);
        slicer_.set_on_new_slice_callback(
            [&](ConditionStatus, Metavision::timestamp, std::size_t) { display_new_frame(); });

        visu_8uc3_.create(height, width, CV_8UC3);
        visu_8uc3_.setTo(0);

        window_.set_keyboard_callback([&](auto key, int scancode, auto action, int) {
            if (action == Metavision::UIAction::RELEASE) {
                if (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)
                    window_.set_close_flag();
            }
        });
    }

    bool should_close() const {
        return window_.should_close();
    }

    template<typename TrackIt>
    void process_events(TrackIt track_begin, TrackIt track_end) {
        slicer_.process_events(track_begin, track_end, [&](const auto &slice_begin, const auto &slice_end) {
            printer_.process_events(slice_begin, slice_end);
        });
    }

    void notify_elapsed_time(Metavision::timestamp t) {
        slicer_.notify_elapsed_time(t);
    }

private:
    void display_new_frame() {
        printer_.print(visu_8uc3_);
        window_.show_async(visu_8uc3_);

        visu_8uc3_.setTo(0);
    }

    Metavision::MTWindow window_;
    Metavision::EventBufferReslicerAlgorithm slicer_;
    TrackPrinter printer_;
    cv::Mat visu_8uc3_;
};

std::optional<Config> parse_command_line(int argc, char *argv[]) {
    namespace po = boost::program_options;

    const std::string program_desc("Code sample showing how to use Metavision SDK to track active markers in 2D");

    Config config;
    po::options_description options_desc;
    po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&config.event_file_path), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j", po::value<std::string>(&config.cam_config_path), "Path to a JSON file containing camera settings. Should be used to tune biases so that modulated light is detected")
        ("am-json-file,a", po::value<std::string>(&config.am_json_path)->required(), "Path to a JSON file describing an active marker.")
        ;
    // clang-format on

    po::options_description display_options("Display options");
    // clang-format off
    display_options.add_options()
        ("no-display,d", po::bool_switch(&config.no_display)->default_value(false), "Disable output display window")
        ("realtime-playback-speed", po::value<bool>(&config.realtime_playback_speed)->default_value(true), "Replay events at speed of recording if true, otherwise as fast as possible")
        ("display-ids", po::value<bool>(&config.display_ids)->default_value(true), "Displays LED IDs if true, otherwise displays their influence radius")
        ("track-length", po::value<std::uint32_t>(&config.track_length)->default_value(5000), "Maximal displayed track length")
        ;
    // clang-format on

    po::options_description detector_options("Modulated light detection options");
    int num_bits = 8; // Workaround because the program option library interprets 8-bits integers as string characters
                      // which causes the wrong value to be stored
    // clang-format off
    detector_options.add_options()
        ("detector-num-bits", po::value<int>(&num_bits)->default_value(8), "Number of bits encoding a light ID")
        ("detector-base-period-us", po::value<std::uint32_t>(&config.detector_params.base_period_us)->default_value(200), "Base period for modulated light decoding (us)")
        ("detector-tolerance", po::value<float>(&config.detector_params.tolerance)->default_value(0.1f), "Tolerance percentage on the blink period measurement")
        ;
    // clang-format on

    po::options_description tracker_options("Active marker tracker options");
    // clang-format off
    tracker_options.add_options()
        ("tracker-update-radius", po::value<bool>(&config.tracker_params.update_radius)->default_value(true), "Update the radius of the blobs during tracking")
        ("tracker-inactivity-period-us", po::value<Metavision::timestamp>(&config.tracker_params.inactivity_period_us)->default_value(1000), "Duration above which a LED tracking is considered lost")
        ("tracker-monitoring-frequency", po::value<float>(&config.tracker_params.monitoring_frequency_hz)->default_value(30.f), "Frequency at which the tracking monitoring process is executed")
        ("tracker-radius", po::value<float>(&config.tracker_params.radius)->default_value(30.f), "Radius used to associate events to a track")
        ("tracker-distance-percentage", po::value<float>(&config.tracker_params.distance_pct)->default_value(0.3f), "Percentage on the closest distance between two blobs used to update the radius of the blobs")
        ("tracker-alpha-pos", po::value<float>(&config.tracker_params.alpha_pos)->default_value(0.05f), "Weight of an event when updating a track's position")
        ;
    // clang-format on

    options_desc.add(base_options).add(display_options).add(detector_options).add(tracker_options);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return std::nullopt;
    }

    config.detector_params.num_bits = static_cast<std::uint8_t>(num_bits);

    if (config.event_file_path.empty() && config.cam_config_path.empty()) {
        MV_LOG_ERROR() << "A camera config file setting the biases is required for online execution";
        return std::nullopt;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return std::nullopt;
    }

    return config;
}

std::set<std::uint32_t> load_active_marker(const std::string &path) {
    namespace pt = boost::property_tree;

    std::set<std::uint32_t> ids;

    pt::ptree root;
    pt::read_json(path, root);

    const auto marker_node = root.get_child("active marker");

    for (const auto &led_node : marker_node) {
        ids.emplace(led_node.second.get<std::uint32_t>("led.id"));
    }

    return ids;
}

int main(int argc, char *argv[]) {
    auto opt_config = parse_command_line(argc, argv);
    if (!opt_config)
        return 1;

    /// [AM_2D_TRACKING_CAMERA_BEGIN]
    Metavision::Camera camera;
    if (opt_config->event_file_path.empty()) {
        Metavision::DeviceConfig device_config;
        device_config.enable_biases_range_check_bypass(true);
        camera = Metavision::Camera::from_first_available(device_config);
        camera.load(opt_config->cam_config_path);
    } else {
        const auto cam_config = Metavision::FileConfigHints().real_time_playback(opt_config->realtime_playback_speed);
        camera                = Metavision::Camera::from_file(opt_config->event_file_path, cam_config);
    }
    /// [AM_2D_TRACKING_CAMERA_END]

    /// [AM_2D_TRACKING_LOAD_AM_BEGIN]
    const auto led_ids = load_active_marker(opt_config->am_json_path);
    /// [AM_2D_TRACKING_LOAD_AM_END]

    const auto eb_w                    = static_cast<std::uint16_t>(camera.geometry().width());
    const auto eb_h                    = static_cast<std::uint16_t>(camera.geometry().height());
    opt_config->detector_params.width  = eb_w;
    opt_config->detector_params.height = eb_h;

    /// [AM_2D_TRACKING_INIT_DISPLAY_BEGIN]
    std::unique_ptr<DisplayManager> display;

    if (!opt_config->no_display) {
        display =
            std::make_unique<DisplayManager>(eb_w, eb_h, led_ids, opt_config->track_length, opt_config->display_ids);
    }
    /// [AM_2D_TRACKING_INIT_DISPLAY_END]

    /// [AM_2D_TRACKING_INIT_ALGOS_BEGIN]
    std::vector<Metavision::EventSourceId> source_id_events;
    std::vector<Metavision::EventActiveTrack> active_tracks;
    Metavision::ModulatedLightDetectorAlgorithm modulated_light_detector(opt_config->detector_params);
    Metavision::ActiveMarkerTrackerAlgorithm tracker(opt_config->tracker_params, led_ids);
    /// [AM_2D_TRACKING_INIT_ALGOS_END]

    /// [AM_2D_TRACKING_TIME_CB_BEGIN]
    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();

    decoder->add_time_callback([&](Metavision::timestamp t) {
        tracker.notify_elapsed_time(t);
        if (display)
            display->notify_elapsed_time(t);
    });
    /// [AM_2D_TRACKING_TIME_CB_END]

    /// [AM_2D_TRACKING_CAMERA_CB_BEGIN]
    camera.cd().add_callback([&](const auto begin, const auto end) {
        source_id_events.clear();
        modulated_light_detector.process_events(begin, end, std::back_inserter(source_id_events));

        active_tracks.clear();
        tracker.process_events(source_id_events.cbegin(), source_id_events.cend(), std::back_inserter(active_tracks));

        if (display) {
            display->process_events(active_tracks.cbegin(), active_tracks.cend());
        }
    });

    /// [AM_2D_TRACKING_CAMERA_CB_END]

    /// [AM_2D_TRACKING_MAIN_LOOP_BEGIN]
    camera.start();

    while (camera.is_running()) {
        if (!display) {
            std::this_thread::yield();
            continue;
        }

        if (display->should_close())
            break;

        Metavision::EventLoop::poll_and_dispatch(20);
    }

    camera.stop();
    /// [AM_2D_TRACKING_MAIN_LOOP_END]

    return 0;
}