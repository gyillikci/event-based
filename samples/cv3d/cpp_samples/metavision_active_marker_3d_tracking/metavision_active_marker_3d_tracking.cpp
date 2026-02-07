/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <Eigen/LU>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>

#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/cv/events/event_source_id.h>
#include <metavision/sdk/cv/utils/camera_geometry_factory.h>
#include <metavision/sdk/cv/algorithms/modulated_light_detector_algorithm.h>
#include <metavision/sdk/cv3d/algorithms/active_marker_pose_estimator_algorithm.h>
#include <metavision/sdk/driver/camera.h>

#include "viewer_3d.h"

namespace std {
std::istream &operator>>(std::istream &in, Metavision::ActiveMarkerPoseEstimatorAlgorithm::EstimatorType &type) {
    std::string type_str;
    in >> type_str;

    if (type_str == "CameraClock")
        type = Metavision::ActiveMarkerPoseEstimatorAlgorithm::EstimatorType::CameraClock;
    else
        type = Metavision::ActiveMarkerPoseEstimatorAlgorithm::EstimatorType::SystemClock;

    return in;
}

std::ostream &operator<<(std::ostream &out, const Metavision::ActiveMarkerPoseEstimatorAlgorithm::EstimatorType &type) {
    switch (type) {
    case Metavision::ActiveMarkerPoseEstimatorAlgorithm::EstimatorType::CameraClock:
        out << "CameraClock";
        break;
    case Metavision::ActiveMarkerPoseEstimatorAlgorithm::EstimatorType::SystemClock:
        out << "SystemClock";
        break;
    }

    return out;
}

std::istream &operator>>(std::istream &in, Metavision::Viewer3d::UpdateMode &mode) {
    std::string type_str;
    in >> type_str;

    if (type_str == "Camera")
        mode = Metavision::Viewer3d::UpdateMode::Camera;
    else
        mode = Metavision::Viewer3d::UpdateMode::Object;

    return in;
}

std::ostream &operator<<(std::ostream &out, const Metavision::Viewer3d::UpdateMode &mode) {
    switch (mode) {
    case Metavision::Viewer3d::UpdateMode::Camera:
        out << "Camera";
        break;
    case Metavision::Viewer3d::UpdateMode::Object:
        out << "Object";
        break;
    }

    return out;
}

} // namespace std

namespace detail {
using ActiveMarker = std::unordered_map<std::uint32_t, cv::Point3f>;
ActiveMarker load_active_marker(const std::string &path) {
    namespace pt = boost::property_tree;

    ActiveMarker active_marker;

    pt::ptree root;
    pt::read_json(path, root);

    const auto marker_node = root.get_child("active marker");

    std::vector<float> xyz;
    for (const auto &led_node : marker_node) {
        const auto id = led_node.second.get<std::uint32_t>("led.id");

        xyz.clear();
        const auto xyz_node = led_node.second.get_child("led.xyz");
        for (const auto &v : xyz_node) {
            xyz.emplace_back(v.second.get_value<float>());
        }

        active_marker[id] = cv::Point3f{xyz[0], xyz[1], xyz[2]};
    }

    return active_marker;
}
} // namespace detail

struct Config {
    std::string event_file_path;
    std::string cam_config_path;
    std::string am_json_path;
    std::string calib_json_path;
    bool realtime_playback_speed;

    Metavision::ModulatedLightDetectorAlgorithm::Params detector_params;
    Metavision::ActiveMarkerPoseEstimatorAlgorithm::Params pose_estimator_params;
    Metavision::Viewer3d::Params viewer_params;
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
       ("input-camera-config,j", po::value<std::string>(&config.cam_config_path), "Path to a JSON file containing camera settings. Should be used to tune biases for this application")
       ("am-json-file,a", po::value<std::string>(&config.am_json_path)->required(), "Path to a JSON file describing an active marker.")
       ("calib-json-file,c", po::value<std::string>(&config.calib_json_path)->required(), "Path to a JSON file containing the camera calibration.")
       ;
    // clang-format on

    po::options_description display_options("Display options");
    // clang-format off
   display_options.add_options()
       ("realtime-playback-speed", po::value<bool>(&config.realtime_playback_speed)->default_value(true), "Replay events at speed of recording if true, otherwise as fast as possible")
       ("am-3d-model-path,m", po::value<std::string>(&config.viewer_params.model_3d_path)->required(), "Path to a 3d model that will be associated with the active marker")
       ("update-mode", po::value<Metavision::Viewer3d::UpdateMode>(&config.viewer_params.update_mode)->default_value(Metavision::Viewer3d::UpdateMode::Object), "Target of the pose updates (Camera or Object)")
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
       ("tracker-update-radius", po::value<bool>(&config.pose_estimator_params.tracker_params.update_radius)->default_value(true), "Update the radius of the blobs during tracking")
       ("tracker-inactivity-period-us", po::value<Metavision::timestamp>(&config.pose_estimator_params.tracker_params.inactivity_period_us)->default_value(1000), "Duration above which a LED tracking is considered lost")
       ("tracker-monitoring-frequency", po::value<float>(&config.pose_estimator_params.tracker_params.monitoring_frequency_hz)->default_value(30.f), "Frequency at which the tracking monitoring process is executed")
       ("tracker-radius2", po::value<float>(&config.pose_estimator_params.tracker_params.radius)->default_value(30.f), "Squared radius used to associate events to a tracked blob")
       ("tracker-distance-percentage", po::value<float>(&config.pose_estimator_params.tracker_params.distance_pct)->default_value(0.3f), "Percentage on the closest distance between two blobs used to update the radius of the blobs")
       ("tracker-alpha-pos", po::value<float>(&config.pose_estimator_params.tracker_params.alpha_pos)->default_value(0.05f), "Weight of an event when updating a blob's position")
       ;
    // clang-format on

    using PoseEstimatorType = Metavision::ActiveMarkerPoseEstimatorAlgorithm::EstimatorType;
    po::options_description pose_estimation_options("Pose estimation options");
    // clang-format off
   pose_estimation_options.add_options()
       ("pose-estimation-type", po::value<PoseEstimatorType>(&config.pose_estimator_params.type)->default_value(PoseEstimatorType::CameraClock), "Clock used to compute the active marker pose (CameraClock, SystemClock)")
       ("pose-estimation-n-events", po::value<size_t>(&config.pose_estimator_params.delta_n_events)->default_value(5000), "Number of events between two pose estimates in the Camera's clock")
       ("pose-estimation-n-us", po::value<Metavision::timestamp>(&config.pose_estimator_params.delta_ts)->default_value(0), "Time interval (in us) between two pose estimates in the Camera's clock")
       ("pose-estimation-frequency-hz", po::value<float>(&config.pose_estimator_params.estimation_rate_hz)->default_value(0.f), "Pose estimation rate (in Hz) in the System's clock")
       ;
    // clang-format on

    options_desc.add(base_options)
        .add(display_options)
        .add(detector_options)
        .add(tracker_options)
        .add(pose_estimation_options);

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
        MV_LOG_ERROR() << "A camera setting file setting the biases is required for online execution";
        return std::nullopt;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return std::nullopt;
    }

    return config;
}

int main(int argc, char *argv[]) {
    auto opt_config = parse_command_line(argc, argv);
    if (!opt_config)
        return 1;

    /// [AM_3D_TRACKING_CAMERA_BEGIN]
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
    /// [AM_3D_TRACKING_CAMERA_END]

    /// [AM_3D_TRACKING_LOAD_AM_BEGIN]
    const auto active_marker = detail::load_active_marker(opt_config->am_json_path);
    /// [AM_3D_TRACKING_LOAD_AM_END]

    /// [AM_3D_TRACKING_LOAD_CALIB_BEGIN]
    const auto camera_geometry = Metavision::load_camera_geometry<float>(opt_config->calib_json_path);
    /// [AM_3D_TRACKING_LOAD_CALIB_END]

    const auto eb_w                    = static_cast<std::uint16_t>(camera.geometry().width());
    const auto eb_h                    = static_cast<std::uint16_t>(camera.geometry().height());
    opt_config->detector_params.width  = eb_w;
    opt_config->detector_params.height = eb_h;
    opt_config->viewer_params.width    = eb_w;
    opt_config->viewer_params.height   = eb_h;

    /// [AM_3D_TRACKING_INIT_ALGOS_BEGIN]
    Metavision::Viewer3d viewer(opt_config->viewer_params);
    Metavision::ModulatedLightDetectorAlgorithm modulated_light_detector(opt_config->detector_params);
    Metavision::ActiveMarkerPoseEstimatorAlgorithm pose_estimator(opt_config->pose_estimator_params, *camera_geometry,
                                                                  active_marker);
    /// [AM_3D_TRACKING_INIT_ALGOS_END]

    /// [AM_3D_TRACKING_TIME_CB_BEGIN]
    auto decoder = camera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    decoder->add_time_callback([&](Metavision::timestamp t) { pose_estimator.notify_elapsed_time(t); });
    /// [AM_3D_TRACKING_TIME_CB_END]

    /// [AM_3D_TRACKING_CAMERA_CB_BEGIN]
    std::vector<Metavision::EventSourceId> source_id_events;
    camera.cd().add_callback([&](const auto begin, const auto end) {
        source_id_events.clear();
        modulated_light_detector.process_events(begin, end, std::back_inserter(source_id_events));

        pose_estimator.process_events(source_id_events.cbegin(), source_id_events.cend());
    });
    /// [AM_3D_TRACKING_CAMERA_CB_END]

    /// [AM_3D_TRACKING_VIEWER_CB_BEGIN]
    pose_estimator.set_pose_update_callback(
        [&](Metavision::timestamp t, const Metavision::Viewer3d::PoseUpdate &T_w_m) {
            viewer.apply_pose_update(T_w_m);
        });
    /// [AM_3D_TRACKING_VIEWER_CB_END]

    /// [AM_3D_TRACKING_MAIN_LOOP_BEGIN]
    camera.start();
    viewer.run();
    camera.stop();
    /// [AM_3D_TRACKING_MAIN_LOOP_END]

    return 0;
}