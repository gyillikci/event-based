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
#include <metavision/sdk/cv/algorithms/trail_filter_algorithm.h>
#include <metavision/sdk/cv/utils/camera_geometry.h>
#include <metavision/sdk/cv/utils/camera_geometry_factory.h>
#include <metavision/sdk/core/utils/mostrecent_timestamp_buffer.h>
#include <metavision/sdk/cv3d/algorithms/model_3d_tracking_algorithm.h>
#include <metavision/sdk/cv3d/algorithms/model_3d_detection_algorithm.h>
#include <metavision/sdk/cv3d/utils/model_3d_processing.h>
#include <metavision/sdk/cv/algorithms/proximity_filter_algorithm.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/mt_window.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

struct Config {
    std::string base_path;
    std::string raw_path;
    std::string cam_config_path;
    std::string model_path;
    std::string model_pose_path;
    std::string calibration_path;
    std::uint32_t display_acc_time_us_;
    std::uint32_t n_detections_;
    std::uint32_t n_events_;
    Metavision::timestamp n_us_;
    Metavision::timestamp detection_period_us_;
    Metavision::Model3dDetectionAlgorithm::Parameters detection_params_;
    Metavision::Model3dTrackingAlgorithm::Parameters tracking_params_;
    float display_fps_;
    bool no_display;
    bool realtime_playback_speed;
    std::string output_video_;
    std::string save_data_file_;
    // roi filtering
    bool use_roi_filtering_;
};

class Pipeline {
public:
    Pipeline(const Config &config) : config_(config) {
        is_tracking_ = false;
        n_detection_ = 0;

        /// [LOAD_3D_MODEL_BEGIN]
        if (!Metavision::load_model_3d_from_json(config.model_path, model_))
            throw std::runtime_error("Impossible to load the 3D model from " + config.model_path);
        /// [LOAD_3D_MODEL_END]

        if (config_.raw_path.empty()) {
            camera_ = Metavision::Camera::from_first_available();

            if (!config_.cam_config_path.empty()) {
                camera_.load(config_.cam_config_path);
            }
        } else {
            camera_ = Metavision::Camera::from_file(
                config_.raw_path, Metavision::FileConfigHints().real_time_playback(config_.realtime_playback_speed));
        }

        const auto width  = camera_.geometry().width();
        const auto height = camera_.geometry().height();

        cam_geometry_ = Metavision::load_camera_geometry<float>(config.calibration_path);
        if (!cam_geometry_)
            throw std::runtime_error("Impossible to load the camera calibration from " + config.calibration_path);

        load_init_pose(config.model_pose_path);

        T_c_w_ = T_c_w_init_;

        time_surface_.create(height, width, 2);

        /// [INSTANTIATE_ALGOS_BEGIN]
        detection_algo_ = std::make_unique<Metavision::Model3dDetectionAlgorithm>(*cam_geometry_, model_, time_surface_,
                                                                                  config.detection_params_);

        tracking_algo_ = std::make_unique<Metavision::Model3dTrackingAlgorithm>(*cam_geometry_, model_, time_surface_,
                                                                                config.tracking_params_);
        /// [INSTANTIATE_ALGOS_END]

        frame_generation_algo_ = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
            width, height, config.display_acc_time_us_, config.display_fps_);

        frame_generation_algo_->set_output_callback(
            std::bind(&Pipeline::frame_callback, this, std::placeholders::_1, std::placeholders::_2));

        if (config.use_roi_filtering_) {
            model_center_ = Eigen::Vector2f(width * 0.5f, height * 0.5f);
            model_radius_ = 2 * width;
            // adaptive radius margin is computed so that a movement going up to search_radius_ still find useful
            // timestamps
            model_margin_ = config_.tracking_params_.n_last_poses_ * config_.tracking_params_.search_radius_;
            evt_filter_   = std::make_unique<Metavision::ProximityFilterAlgorithm>(model_center_, model_radius_);
        }

        /// [CD_PRODUCER_CALLBACK_BEGIN]
        Metavision::SharedEventsBufferProducerParameters params;
        auto cb   = [&](Metavision::timestamp ts, const Buffer &b) { buffers_.emplace(b); };
        producer_ = std::make_unique<Metavision::SharedCdEventsBufferProducerAlgorithm>(params, cb);
        /// [CD_PRODUCER_CALLBACK_END]

        set_detection_params();
        if (evt_filter_) {
            update_filtering_params();
        }

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
        if (!config_.save_data_file_.empty())
            camera_.start_recording(config_.save_data_file_);

        while (camera_.is_running()) {
            if (!window_) {
                std::this_thread::yield();
                continue;
            }

            if (window_->should_close())
                break;

            Metavision::EventLoop::poll_and_dispatch(10); // no need to rush
        }
        if (!config_.save_data_file_.empty()) {
            camera_.stop_recording();
        }
        camera_.stop();
        if (video_out_)
            MV_LOG_INFO() << "Wrote video file:" << config_.output_video_;
    }

private:
    void load_init_pose(const std::string &path) {
        pt::ptree root;

        pt::read_json(path, root);

        const auto &cam_node = root.get_child("camera_pose");

        Eigen::Matrix4f T_w_c;

        int row = 0;
        for (const auto &row_node : cam_node.get_child("T_w_c")) {
            int col = 0;
            for (const auto &col_node : row_node.second) {
                T_w_c(row, col) = col_node.second.get_value<float>();
                ++col;
            }
            ++row;
        }

        T_c_w_init_ = T_w_c.inverse();
    }

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

            if (is_tracking_) {
                is_tracking_ = tracking_algo_->process_events(begin, end, T_c_w_);
            } else {
                if (detection_algo_->process_events(begin, end, T_c_w_, &visible_edges_, &detected_edges_)) {
                    // we wait for several consecutives detections before considering the model as detected to avoid
                    // false positive detections
                    is_tracking_ = (++n_detection_ > config_.n_detections_);
                } else {
                    n_detection_ = 0;
                }
            }

            // the frame generation algorithm processing can trigger a call to show_async which can trigger a reset of
            // the tracking if the space bar has been pressed.
            frame_generation_algo_->process_events(begin, end);

            if (prev_is_tracking != is_tracking_) {
                if (is_tracking_)
                    set_tracking_params(std::prev(buffer->cend())->t); // triggers the creation of a new buffer
                else
                    set_detection_params(); // triggers the creation of a new buffer
            }
            if (evt_filter_)
                update_filtering_params();
        }
        /// [BUFFERS_QUEUE_PROCESSING_LOOP_END]
    }

    void set_detection_params() {
        MV_LOG_INFO() << "set detection param";
        T_c_w_ = T_c_w_init_;
        detection_algo_->set_init_pose(T_c_w_);
        producer_->set_processing_n_us(config_.detection_period_us_);
    }

    void set_tracking_params(Metavision::timestamp ts) {
        MV_LOG_INFO() << "set tracking param";
        tracking_algo_->set_previous_camera_pose(ts, T_c_w_);
        n_detection_ = 0;
        producer_->set_processing_mixed(config_.n_events_, config_.n_us_);
    }

    void update_filtering_params() {
        find_model_center_radius(model_center_, model_radius_);
        evt_filter_->set_center(model_center_);
        model_radius_ += model_margin_;
        evt_filter_->set_distance(model_radius_);
    }

    void find_model_center_radius(Eigen::Vector2f &center, float &dist) {
        const Eigen::Vector4f origin(0, 0, 0, 1);
        const Eigen::Vector4f center_cam = T_c_w_ * origin; // the center of the model is at world origin
        // if the center of the model is not the origin, the approximation will still be sufficient for an adaptive ROI
        // filter. The ROI may not be correctly centered on the projection of the model so it will filter less tightly
        cam_geometry_->camera_to_img(center_cam.hnormalized(), center);

        float max_sqr_dist = 0;
        for (const auto &pt_w : model_.vertices_) {
            // find the furthest vertex in the model
            const auto pt_c = T_c_w_ * pt_w.homogeneous();
            Eigen::Vector2f pt_img;
            cam_geometry_->camera_to_img(pt_c.hnormalized().eval(), pt_img);
            max_sqr_dist = std::max(max_sqr_dist, (pt_img - center).squaredNorm());
        }

        dist = std::sqrt(max_sqr_dist);
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
            Metavision::draw_edges(*cam_geometry_, T_c_w_, model_, visible_edges_, frame, cv::Scalar(0, 0, 255));
            Metavision::draw_edges(*cam_geometry_, T_c_w_, model_, detected_edges_, frame, cv::Scalar(0, 255, 0));
            cv::putText(frame, "detecting", cv::Point(0, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255));
        }
        if (evt_filter_)
            cv::circle(frame, cv::Point(model_center_(0), model_center_(1)), static_cast<int>(model_radius_),
                       cv::Scalar(150, 0, 255), 1);
        if (video_out_)
            video_out_->write(frame);
        if (window_)
            window_->show_async(frame);
    }
    /// [FRAME_GENERATOR_CALLBACK_END]

    /// [CD_CAMERA_CALLBACK_BEGIN]
    void cd_processing_callback(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (evt_filter_) {
            filtered_evts_.clear();
            evt_filter_->process_events(begin, end, std::back_inserter(filtered_evts_));
            producer_->process_events(filtered_evts_.cbegin(), filtered_evts_.cend());
        } else {
            producer_->process_events(begin, end);
        }

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
    std::unique_ptr<Metavision::Model3dDetectionAlgorithm> detection_algo_;
    std::unique_ptr<Metavision::Model3dTrackingAlgorithm> tracking_algo_;
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm> frame_generation_algo_;
    std::unique_ptr<Metavision::MTWindow> window_;

    std::unique_ptr<Metavision::SharedCdEventsBufferProducerAlgorithm> producer_;
    std::queue<Buffer> buffers_;

    volatile bool is_tracking_;
    std::uint32_t n_detection_;
    Eigen::Matrix4f T_c_w_init_;
    Eigen::Matrix4f T_c_w_;
    std::set<size_t> visible_edges_, detected_edges_;

    std::unique_ptr<cv::VideoWriter> video_out_;
    std::vector<Metavision::EventCD> filtered_evts_;
    std::unique_ptr<Metavision::ProximityFilterAlgorithm> evt_filter_;
    Eigen::Vector2f model_center_;
    float model_radius_, model_margin_;
};

bool parse_command_line(int argc, char *argv[], Config &config) {
    const std::string program_desc("3D model detection and tracking\n");

    po::options_description options_desc;
    po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-base-path,i", po::value<std::string>(&config.base_path), "Base path used to load a recording, a 3D model, a camera calibration and an initialization pose. Apart from the recording, every file can also be set individually")
        ("input-raw-file,r", po::value<std::string>(&config.raw_path), "Path to a RAW file. Ignored if a base path is set")
        ("input-camera-config,j", po::value<std::string>(&config.cam_config_path), "Path to a JSON file containing camera settings. Should be used to tune biases for this application")
        ("input-model-file,m", po::value<std::string>(&config.model_path), "Path to a JSON file containing the description of a 3D model")
        ("input-pose-file,p", po::value<std::string>(&config.model_pose_path), "Path to a JSON file containing a camera pose used to detect the 3D model")
        ("input-calibration-file,c", po::value<std::string>(&config.calibration_path), "Path to a JSON file containing the camera's calibration")
        ("use-roi-filtering", po::value<bool>(&config.use_roi_filtering_)->default_value(false), "Use a circular ROI centered on the projection of the object on the image to filter out incoming events")
        ;
    // clang-format on

    po::options_description detection_options("Detection options");
    // clang-format off
    detection_options.add_options()
        ("num-detections", po::value<std::uint32_t>(&config.n_detections_)->default_value(10), "Number of successive valid detection to consider the model as detected")
        ("detection-period", po::value<Metavision::timestamp>(&config.detection_period_us_)->default_value(10000), "Amount of time after which a detection is attempted")
        ("detection-support-point-step", po::value<std::uint32_t>(&config.detection_params_.support_point_step_)->default_value(10), "Distance (in pixels in the distorted image) between two support points")
        ("detection-search-radius", po::value<std::uint32_t>(&config.detection_params_.search_radius_)->default_value(10), "Search radius for matches of each support point")
        ("flow-radius", po::value<std::uint32_t>(&config.detection_params_.flow_radius_)->default_value(3), "Radius used to estimate the normal flow of the edge")
        ("n-fitting-pts", po::value<std::uint32_t>(&config.detection_params_.n_fitting_pts_)->default_value(3), "Minimum number of matches required for line fitting")
        ("variance-threshold", po::value<float>(&config.detection_params_.variance_threshold_)->default_value(2e-5f), "Threshold of the variance of the support points around the fitted line below which an edge is considered matched")
        ("fitted-edges-ratio", po::value<float>(&config.detection_params_.fitted_edges_ratio_)->default_value(0.5f), "Matched edges to visible edges ratio above which a pose estimation is attempted")
        ;
    // clang-format on

    po::options_description tracking_options("Tracking options");
    // clang-format off
    tracking_options.add_options()
        ("n-events", po::value<std::uint32_t>(&config.n_events_)->default_value(5000), "Number of events after which a tracking step is attempted")
        ("n-us", po::value<Metavision::timestamp>(&config.n_us_)->default_value(10000), "Amount of time after which a tracking step is attempted")
        ("tracking-search-radius", po::value<std::uint32_t>(&config.tracking_params_.search_radius_)->default_value(5), "Search radius for matches of each support point")
        ("tracking-support-point-step", po::value<std::uint32_t>(&config.tracking_params_.support_point_step_)->default_value(10), "Distance (in pixels in the distorted image) between two support points")
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
        ("output-video,o", po::value<std::string>(&config.output_video_)->default_value(""), "Save display window in a .avi format")
        ("save-data-file", po::value<std::string>(&config.save_data_file_)->default_value(""), "Save input data in case of live camera")
        ;
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

    fs::path root_path;
    fs::path base_path;
    if (config.base_path.empty()) {
        if (config.raw_path.empty()) {
            MV_LOG_ERROR() << "You should provide either a base path or a path to a RAW file";
            return false;
        }

        const fs::path raw_path(config.raw_path);
        if (!fs::exists(raw_path)) {
            MV_LOG_ERROR() << "Invalid RAW path:" << raw_path.string();
            return false;
        }

        root_path = raw_path.parent_path();
        base_path = root_path / raw_path.stem();
    } else {
        base_path = fs::path(config.base_path);
        root_path = base_path.parent_path();

        const std::string extension = base_path.extension().string();
        if (extension != "") {
            MV_LOG_ERROR() << "Invalid base path. Remove the extension" << extension;
            if (extension == ".raw")
                MV_LOG_ERROR() << "Or use -r instead of -i to run the app on a RAW file.";
            return false;
        }

        if (!fs::is_directory(root_path)) {
            MV_LOG_ERROR() << "Invalid base path:" << base_path.string();
            return false;
        }

        if (!config.raw_path.empty())
            config.raw_path.clear();
    }

    if (config.model_path.empty())
        config.model_path = base_path.string() + ".json";

    if (!fs::exists(config.model_path)) {
        MV_LOG_ERROR() << "The path to the model file is invalid:" << config.model_path;
        return false;
    }

    if (config.model_pose_path.empty())
        config.model_pose_path = base_path.string() + "_init_pose.json";

    if (!fs::exists(config.model_pose_path)) {
        MV_LOG_ERROR() << "The path to the model's pose file is invalid:" << config.model_pose_path;
        return false;
    }

    if (config.calibration_path.empty())
        config.calibration_path = (root_path / fs::path("calibration.json")).string();

    if (!fs::exists(config.calibration_path)) {
        MV_LOG_ERROR() << "The path to the camera's calibration file is invalid:" << config.calibration_path;
        return false;
    }

    MV_LOG_INFO() << "root path:" << root_path;
    MV_LOG_INFO() << "base path:" << config.base_path;
    MV_LOG_INFO() << "raw path:" << config.raw_path;
    MV_LOG_INFO() << "camera settings path:" << config.cam_config_path;
    MV_LOG_INFO() << "model path:" << config.model_path;
    MV_LOG_INFO() << "model pose path:" << config.model_pose_path;
    MV_LOG_INFO() << "calibration path:" << config.calibration_path;

    return true;
}

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
}
