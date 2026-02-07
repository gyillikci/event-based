/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/core/algorithms/shared_cd_events_buffer_producer_algorithm.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/cv/utils/camera_geometry.h>
#include <metavision/sdk/cv/utils/camera_geometry_factory.h>
#include <metavision/sdk/cv/algorithms/proximity_filter_algorithm.h>
#include <metavision/sdk/core/utils/mostrecent_timestamp_buffer.h>
#include <metavision/sdk/cv3d/algorithms/model_3d_tracking_algorithm.h>
#include <metavision/sdk/cv3d/utils/model_3d_processing.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/mt_window.h>
#include "aruco_marker_detection_algorithm.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

struct Config {
    std::string event_file_path;
    std::string cam_config_path;
    std::string calibration_path;
    float marker_size;
    std::uint32_t display_acc_time_us_;
    std::uint32_t n_detections_;
    std::uint32_t n_events_;
    Metavision::timestamp n_us_;
    Metavision::timestamp detection_period_us_;
    Metavision::Model3dTrackingAlgorithm::Parameters tracking_params_;
    int id_to_track_;
    // display parameters
    float display_fps_;
    bool no_display;
    bool realtime_playback_speed;
    std::string output_video_;
    std::string output_path_;
};

void load_aruco_dictionary(const std::string &json_file, std::vector<std::vector<std::vector<int>>> &corners_list,
                           std::vector<std::vector<std::vector<int>>> &edges_list) {
    boost::property_tree::ptree aruco_dictionary;
    try {
        // Load the JSON file into the property tree
        boost::property_tree::read_json(json_file, aruco_dictionary);

        // Retrieve the list of corners
        for (const auto &corner : aruco_dictionary.get_child("corners_list")) {
            std::vector<std::vector<int>> aruco_corners;
            for (const auto &coordinate : corner.second) {
                std::vector<int> corner_coord;
                for (const auto &value : coordinate.second) {
                    corner_coord.emplace_back(value.second.get_value<int>());
                }
                aruco_corners.emplace_back(corner_coord);
            }
            corners_list.emplace_back(aruco_corners);
        }

        // Retrieve the list of edges
        for (const auto &edge : aruco_dictionary.get_child("edges_list")) {
            std::vector<std::vector<int>> marker_edges;
            for (const auto &coordinate : edge.second) {
                std::vector<int> edge_coord;
                for (const auto &value : coordinate.second) {
                    edge_coord.emplace_back(value.second.get_value<int>());
                }
                marker_edges.emplace_back(edge_coord);
            }
            edges_list.emplace_back(marker_edges);
        }
    } catch (const boost::property_tree::json_parser_error &e) {
        throw std::runtime_error("Error reading Aruco dictionary JSON file: " + std::string(e.what()));
    }
};

class Pipeline {
public:
    Pipeline(const Config &config) : config_(config) {
        is_tracking_ = false;
        n_detection_ = 0;

        if (config_.event_file_path.empty()) {
            camera_ = Metavision::Camera::from_first_available();

            if (!config_.cam_config_path.empty()) {
                camera_.load(config_.cam_config_path);
            }
        } else {
            camera_ = Metavision::Camera::from_file(
                config_.event_file_path,
                Metavision::FileConfigHints().real_time_playback(config_.realtime_playback_speed));
        }

        const auto width  = camera_.geometry().width();
        const auto height = camera_.geometry().height();

        cam_geometry_ = Metavision::load_camera_geometry<float>(config.calibration_path);
        if (!cam_geometry_)
            throw std::runtime_error("Impossible to load the camera calibration from " + config.calibration_path);

        time_surface_.create(height, width, 2);

        // Load the Aruco Markers dictionary
        const std::string dict_file = (fs::path(__FILE__).parent_path() / "aruco_dictionary_6x6.json").string();
        std::vector<std::vector<std::vector<int>>> corners_list, edges_list;
        load_aruco_dictionary(dict_file, corners_list, edges_list);

        /// [INSTANTIATE_ALGOS_BEGIN]
        detection_algo_ = std::make_unique<Metavision::ArucoMarkerDetectionAlgorithm>(*cam_geometry_, corners_list,
                                                                                      edges_list, config.marker_size);
        tracking_algo_  = std::make_unique<Metavision::Model3dTrackingAlgorithm>(*cam_geometry_, model_, time_surface_,
                                                                                config.tracking_params_);
        /// [INSTANTIATE_ALGOS_END]

        frame_generation_algo_ = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
            width, height, config.display_acc_time_us_, config.display_fps_);

        frame_generation_algo_->set_output_callback(
            std::bind(&Pipeline::frame_callback, this, std::placeholders::_1, std::placeholders::_2));

        /// [CD_PRODUCER_CALLBACK_BEGIN]
        Metavision::SharedEventsBufferProducerParameters params;
        auto cb   = [&](Metavision::timestamp ts, const Buffer &b) { buffers_.emplace(b); };
        producer_ = std::make_unique<Metavision::SharedCdEventsBufferProducerAlgorithm>(params, cb);
        /// [CD_PRODUCER_CALLBACK_END]

        set_detection_params();

        camera_.cd().add_callback(
            std::bind(&Pipeline::cd_processing_callback, this, std::placeholders::_1, std::placeholders::_2));

        if (!config_.no_display || !config_.output_video_.empty()) {
            window_ = std::make_unique<Metavision::MTWindow>("3D Model tracking", width, height,
                                                             Metavision::BaseWindow::RenderMode::BGR);
            window_->set_keyboard_callback(std::bind(&Pipeline::ui_key_callback, this, std::placeholders::_1,
                                                     std::placeholders::_2, std::placeholders::_3,
                                                     std::placeholders::_4));
            if (!config_.output_video_.empty()) {
                video_out_ = std::make_unique<cv::VideoWriter>(
                    config_.output_video_, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height));
            }
        }
    }

    void run() {
        camera_.start();
        while (camera_.is_running()) {
            if (!window_) {
                std::this_thread::yield();
                continue;
            }

            if (window_->should_close())
                break;

            Metavision::EventLoop::poll_and_dispatch(10); // no need to rush
        }
        camera_.stop();
        if (video_out_)
            MV_LOG_INFO() << "Wrote video file:" << config_.output_video_;
    }

private:
    void process_buffers_queue() {
        // As the async_algorithm class is not reentrant (i.e. calling flush inside a process_async call would
        // create internal conflicts) buffers are not processed directly when created (i.e. inside the callback
        // passed to the producer) but added to a queue and processed afterwards instead.
        // After processing the buffer, we check whether the tracking status has changed. If so, we change the
        // parameters of the events buffers producer accordingly. As a consequence a flush is called in the
        // async_algorithm which causes the creation of a new buffer that can safely be added to the queue and
        // processed.
        /// [BUFFERS_QUEUE_PROCESSING_LOOP_BEGIN]
        while (!buffers_.empty()) {
            const bool prev_is_tracking = is_tracking_;

            const auto buffer = buffers_.front();
            buffers_.pop();

            const auto begin = buffer->cbegin();
            const auto end   = buffer->cend();

            bool enabled_tracking = false;
            if (is_tracking_) {
                is_tracking_ = tracking_algo_->process_events(begin, end, T_c_w_);
            } else {
                if (detection_algo_->process_events(begin, end, detected_markers_)) {
                    // tries to find the marker to track
                    // if found, updates the model and the camera pose
                    if (process_detected_markers()) {
                        // we wait for several consecutive detections before considering the model as detected to avoid
                        // false positive detections
                        enabled_tracking = (++n_detection_ >= config_.n_detections_);
                    }
                }
                // filling time surface manually since the tracking will use it
                for (auto it = begin; it != end; ++it) {
                    time_surface_.at(it->y, it->x, it->p) = it->t;
                }
            }

            // the frame generation algorithm processing can trigger a call to show_async which can trigger a reset of
            // the tracking if the space bar has been pressed.
            frame_generation_algo_->process_events(begin, end);

            // we want to update the tracking state AFTER the frame_generation_algo_ produced the frames. Indeed, after
            // the successful detection of the marker, its pose corresponds to its last pose observed during the batch
            // of events, while several frames might be produced during this batch. However, we don't want to display
            // the marker on intermediate frames where it wouldn't match the events drawn.
            if (enabled_tracking) {
                is_tracking_ = true;
                frame_generation_algo_->force_generate();
            }

            if (prev_is_tracking != is_tracking_) {
                if (is_tracking_)
                    set_tracking_params(std::prev(buffer->cend())->t); // triggers the creation of a new buffer
                else
                    set_detection_params(); // triggers the creation of a new buffer
            }
        }
        /// [BUFFERS_QUEUE_PROCESSING_LOOP_END]
    }

    void set_detection_params() {
        MV_LOG_INFO() << "set detection param";
        detection_algo_->reset();
        producer_->set_processing_n_us(config_.detection_period_us_);
    }

    void set_tracking_params(Metavision::timestamp ts) {
        MV_LOG_INFO() << "set tracking param";
        tracking_algo_->set_previous_camera_pose(ts, T_c_w_);
        n_detection_ = 0;
        producer_->set_processing_mixed(config_.n_events_, config_.n_us_);

        if (!config_.output_path_.empty())
            output_to_json(ts);
    }

    bool process_detected_markers() {
        // if should not detect a particular marker: take the first one
        if (config_.id_to_track_ < 0) {
            model_ = detected_markers_[0].model;
            T_c_w_ = detected_markers_[0].T_c_am;
            return true;
        }

        auto it = std::find_if(detected_markers_.begin(), detected_markers_.end(),
                               [&](const Metavision::ArucoMarker &m) { return m.id == config_.id_to_track_; });
        if (it != detected_markers_.end()) {
            model_ = it->model;
            T_c_w_ = it->T_c_am;
            return true;
        }
        return false;
    }

    void output_to_json(Metavision::timestamp ts) {
        Metavision::write_model_3d_to_json(config_.output_path_ + "/marker.json", model_);
        const Eigen::Matrix4f pose = T_c_w_.inverse();
        write_init_pose(ts, pose);
    }

    bool write_init_pose(Metavision::timestamp ts, const Eigen::Matrix4f &T) {
        Eigen::Matrix4f T_w_c = T.inverse();
        pt::ptree root;

        pt::ptree cam_node;

        pt::ptree ts_node;
        ts_node.put_value(ts);
        cam_node.push_back(std::make_pair("ts", ts_node));

        pt::ptree matrix;
        for (int i = 0; i < 4; ++i) {
            pt::ptree row;
            for (int j = 0; j < 4; ++j) {
                pt::ptree cell;
                cell.put_value(T_w_c(i, j));
                row.push_back(std::make_pair("", cell));
            }
            matrix.push_back(std::make_pair("", row));
        }
        cam_node.add_child("T_w_c", matrix);

        root.put_child("camera_pose", cam_node);

        try {
            pt::write_json(config_.output_path_ + "/initial_pose.json", root);
        } catch (const pt::json_parser_error &err) {
            MV_SDK_LOG_ERROR() << err.message();
            return false;
        }
        MV_SDK_LOG_INFO() << "Pose written in " + config_.output_path_;
        return true;
    }

    /// [FRAME_GENERATOR_CALLBACK_BEGIN]
    void frame_callback(Metavision::timestamp ts, cv::Mat &frame) {
        cv::putText(frame, std::to_string(ts), cv::Point(0, 10), cv::FONT_HERSHEY_DUPLEX, 0.5,
                    (is_tracking_ ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255)));

        // Be careful, here the events and the 3D model are not rendered in a tightly synchronized way, meaning that
        // some shifts might occur. However, most of the time they should not be noticeable
        if (is_tracking_) {
            Metavision::select_visible_edges(T_c_w_, model_, visible_edges_);
            Metavision::draw_edges(*cam_geometry_, T_c_w_, model_, visible_edges_, frame, cv::Scalar(0, 255, 0));
            cv::putText(frame, "tracking", cv::Point(0, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0));

        } else {
            cv::putText(frame, "detecting", cv::Point(0, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255));
        }
        if (video_out_)
            video_out_->write(frame);
        if (window_)
            window_->show_async(frame);
    }
    /// [FRAME_GENERATOR_CALLBACK_END]

    /// [CD_CAMERA_CALLBACK_BEGIN]
    void cd_processing_callback(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        producer_->process_events(begin, end);
        process_buffers_queue();
    }
    /// [CD_CAMERA_CALLBACK_END]

    void ui_key_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
        if (action == Metavision::UIAction::RELEASE) {
            if (key == Metavision::UIKeyEvent::KEY_SPACE)
                is_tracking_ = false;
            else if (key == Metavision::UIKeyEvent::KEY_Q || key == Metavision::UIKeyEvent::KEY_ESCAPE)
                window_->set_close_flag();
        }
    }

    using Buffer = Metavision::SharedCdEventsBufferProducerAlgorithm::SharedEventsBuffer;

    const Config config_;
    Metavision::Model3d model_;
    Metavision::Camera camera_;
    Metavision::MostRecentTimestampBufferT<Metavision::timestamp> time_surface_;
    std::unique_ptr<Metavision::CameraGeometryBase<float>> cam_geometry_;
    std::unique_ptr<Metavision::ArucoMarkerDetectionAlgorithm> detection_algo_;
    std::unique_ptr<Metavision::Model3dTrackingAlgorithm> tracking_algo_;
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm> frame_generation_algo_;
    std::unique_ptr<Metavision::MTWindow> window_;

    std::unique_ptr<Metavision::SharedCdEventsBufferProducerAlgorithm> producer_;
    std::queue<Buffer> buffers_;

    volatile bool is_tracking_;
    std::uint32_t n_detection_;
    Eigen::Matrix4f T_c_w_;
    std::vector<Metavision::ArucoMarker> detected_markers_;
    std::set<size_t> visible_edges_;

    std::unique_ptr<cv::VideoWriter> video_out_;
};

bool parse_command_line(int argc, char *argv[], Config &config) {
    const std::string program_desc("3D model detection and tracking\n");

    po::options_description options_desc;
    po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&config.event_file_path), "ath to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j", po::value<std::string>(&config.cam_config_path), "Path to a JSON file containing camera settings. Should be used to tune biases for this application")
        ("input-calibration-file,c", po::value<std::string>(&config.calibration_path), "Path to a JSON file containing the camera's calibration");
    // clang-format on

    po::options_description detection_options("Detection options");
    // clang-format off
    detection_options.add_options()
        ("input-marker-size,s", po::value<float>(&config.marker_size)->default_value(1.f), "Size of the ArUco marker, used to scale the camera poses")
        ("num-detections", po::value<std::uint32_t>(&config.n_detections_)->default_value(1), "Number of successive valid detection to consider the model as detected")
        ("detection-period", po::value<Metavision::timestamp>(&config.detection_period_us_)->default_value(10000), "Integration time (in us) before attempting a detection")
        ("id-to-track", po::value<int>(&config.id_to_track_)->default_value(-1), "Specific ArUco marker ID to track. If not specified, will track the first marker detected")
        ;
    // clang-format on

    po::options_description tracking_options("Tracking options");
    // clang-format off
    tracking_options.add_options()
        ("n-events", po::value<std::uint32_t>(&config.n_events_)->default_value(5000), "Number of events after which a tracking step is attempted")
        ("n-us", po::value<Metavision::timestamp>(&config.n_us_)->default_value(10000), "Delay (in us) between tracking steps")
        ("search-radius", po::value<std::uint32_t>(&config.tracking_params_.search_radius_)->default_value(5), "Search radius for matches of each support point")
        ("support-point-step", po::value<std::uint32_t>(&config.tracking_params_.support_point_step_)->default_value(10), "Distance (in pixels in the distorted image) between two support points")
        ("n-last-poses", po::value<std::uint32_t>(&config.tracking_params_.n_last_poses_)->default_value(5), "Number of past poses to consider when computing the accumulation time")
        ("default-acc-time", po::value<Metavision::timestamp>(&config.tracking_params_.default_acc_time_us_)->default_value(5000), "Default accumulation time used when the N last poses have not been estimated yet")
        ("oldest-weight", po::value<float>(&config.tracking_params_.oldest_weight_)->default_value(0.1f), "Weight attributed to oldest matches")
        ("most-recent-weight", po::value<float>(&config.tracking_params_.most_recent_weight_)->default_value(1.f), "Weight attributed to most recent matches")
        ("n-directional-axes", po::value<std::uint32_t>(&config.tracking_params_.nb_directional_axes_)->default_value(8), "Number of axes used to discretize the possible directions of the edges' normals")
        ;
    // clang-format on

    po::options_description display_options("Display options");
    // clang-format off
    display_options.add_options()
        ("disp-accumulation-time,a", po::value<std::uint32_t>(&config.display_acc_time_us_)->default_value(5000), "Accumulation time in us used to generate frames for display")
        ("fps,f", po::value<float>(&config.display_fps_)->default_value(30.f), "Display's fps")
        ("no-display,d", po::bool_switch(&config.no_display)->default_value(false), "Disable output display window")
        ("realtime-playback-speed", po::value<bool>(&config.realtime_playback_speed)->default_value(true), "Replay events at speed of recording if true, otherwise as fast as possible")
        ("output-video,o", po::value<std::string>(&config.output_video_)->default_value(""), "Path to save a video of the display window in a .avi format")
        ("detection-output-dir", po::value<std::string>(&config.output_path_)->default_value(""), "Directory in which to save the marker JSON file and the first pose estimate");
    // clang-format on

    options_desc.add(base_options);
    options_desc.add(detection_options);
    options_desc.add(tracking_options);
    options_desc.add(display_options);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return false;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return false;
    }

    if (!config.event_file_path.empty()) {
        const fs::path file_path(config.event_file_path);
        if (!fs::exists(file_path)) {
            MV_LOG_ERROR() << "Invalid event file:" << file_path.string();
            return false;
        }
    }

    if (config.calibration_path.empty()) {
        MV_LOG_ERROR() << "A calibration file is required.";
        return false;
    }

    if (!fs::exists(config.calibration_path)) {
        MV_LOG_ERROR() << "The path to the camera's calibration file is invalid:" << config.calibration_path;
        return false;
    }

    if (!config.output_path_.empty() && !fs::is_directory(fs::path(config.output_path_))) {
        MV_LOG_ERROR() << "Invalid output path. Provide a directory.";
        config.output_path_ = "";
    }

    MV_LOG_INFO() << "event file:" << config.event_file_path;
    MV_LOG_INFO() << "camera settings file:" << config.cam_config_path;
    MV_LOG_INFO() << "calibration path:" << config.calibration_path;

    return true;
};

int main(int argc, char *argv[]) {
    Config config;
    if (!parse_command_line(argc, argv, config))
        return 1;

    try {
        Pipeline p(config);

        p.run();
    } catch (const std::runtime_error &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    return 0;
};
