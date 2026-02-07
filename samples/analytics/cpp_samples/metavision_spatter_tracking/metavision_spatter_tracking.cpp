/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// Example of using SDK to track simple, non colliding objects.

#include <mutex>
#include <condition_variable>
#include <vector>
#include <boost/program_options.hpp>

#include <metavision/sdk/analytics/utils/cluster_trajectories.h>
#include <metavision/sdk/analytics/utils/spatter_tracker_csv_logger.h>
#include <metavision/sdk/analytics/utils/tracking_drawing.h>
#include <metavision/sdk/analytics/events/event_spatter_cluster.h>
#include <metavision/sdk/analytics/configs/spatter_tracker_algorithm_config.h>
#include <metavision/sdk/analytics/algorithms/spatter_tracker_algorithm_sync.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

#include "simple_video_writer.h"
#include "simple_timer.h"

// Utility namespace used to parse command line arguments
namespace boost_po = boost::program_options;

/// @brief Class for spatter tracking pipeline
class Pipeline {
public:
    Pipeline() = default;

    ~Pipeline() = default;

    /// @brief Parses command line arguments
    bool parse_command_line(int argc, char *argv[]);

    /// @brief Initializes the camera
    bool initialize_camera();

    /// @brief Initializes the filters
    void initialize_tracker();

    /// @brief Starts the pipeline and the camera
    bool start();

    /// @brief Stops the pipeline and the camera
    void stop();

    /// @brief Waits until the end of the file or until the exit of the display
    void run();

private:
    /// @brief Processes the output of the spatter tracker
    ///
    /// @param ts Current timestamp
    void tracker_callback(Metavision::timestamp ts);

    /// @brief Draws the zones in which no clusters are tracked
    /// @param img The image to draw on
    void draw_no_track_zones(cv::Mat &img);

    Metavision::timestamp trajectory_duration_us_;
    std::unique_ptr<Metavision::ClusterTrajectories> tracked_clusters_;

    // Spatter tracking parameters
    int cell_width_;                          ///< Cell width used for clustering
    int cell_height_;                         ///< Cell height used for clustering
    int activation_threshold_;                ///< Number of events in a cell to consider it as active
    Metavision::timestamp min_tracking_time_; ///< Minimum time necessary for a cluster to be tracked
    bool apply_filter_; ///< If true, then the activation threshold considers only one event per pixel
    Metavision::timestamp accumulation_time_; ///< Processing accumulation time, in us
    int max_distance_;                        ///< Maximum distance for clusters association
    int untracked_threshold_;             ///< Maximum number of times a cluster can stay untracked before being removed
    Metavision::timestamp static_memory_; ///< Time after which the static constraint of detected clusters is relaxed
    int min_dist_moving_obj_; ///< Minimum distance to consider a cluster moving (else, it is considered a static object
                              ///< which we don't want to track)
    int max_size_variation_;  ///< Maximum size variation allowed to match two clusters
    Metavision::EvtFilterType filter_polarity_; ///< Type of event polarity filter to implement
    std::vector<Metavision::EventSpatterCluster> clusters_;

    // No Tracking zones
    std::vector<int> center_x_;       ///< X coordinates of no-tracking zones centers
    std::vector<int> center_y_;       ///< Y coordinates of no-tracking zones centers
    std::vector<int> radius_;         ///< Radius of no-tracking zones
    std::vector<bool> filter_inside_; ///< Indicates whether the no-track zone is inside or outside the circle

    // Min object size to track
    int min_size_;
    // Max object size to track
    int max_size_;

    // Camera parameters
    std::string event_file_path_ = "";
    std::string cam_config_path_ = "";
    uint32_t stc_threshold_;
    std::vector<Metavision::EventCD> filtered_events_;

    // CSV filename to save the tracking output
    std::string res_csv_file_ = "";

    // Display parameters
    // If we display data on the screen or not
    bool display_ = true;
    // If display as fast as possible
    bool as_fast_as_possible_ = false;
    // Filename to save the resulted video
    std::string output_video_ = "";
    // If we want to measure computation time
    bool time_ = false;
    // Current image
    cv::Mat back_img_;

    // Conditional variables to notify the end of the processing
    std::condition_variable process_cond_;
    std::mutex process_mutex_;
    bool is_processing_ = true;

    Metavision::timestamp write_from_ = 0;
    Metavision::timestamp write_to_   = std::numeric_limits<Metavision::timestamp>::max();

    std::unique_ptr<Metavision::Camera> camera_;                                     ///< Pointer to Camera class
    std::unique_ptr<Metavision::SpatterTrackerAlgorithmSync> tracker_;               ///< Instance of the cluster maker
    std::unique_ptr<Metavision::SpatterTrackerCsvLogger> tracker_logger_;            ///< SpatterTracker csv logger
    std::unique_ptr<Metavision::OnDemandFrameGenerationAlgorithm> frame_generation_; ///< Frame generator
    std::unique_ptr<Metavision::SpatioTemporalContrastAlgorithm> stc_filter_;
    std::unique_ptr<Metavision::EventBufferReslicerAlgorithm> slicer_;
    std::unique_ptr<Metavision::Window> window_;      ///< Display window
    std::unique_ptr<SimpleVideoWriter> video_writer_; ///< Video writer
    std::unique_ptr<SimpleTimer> timer_;              ///< SpatterTracker timer
};

bool Pipeline::initialize_camera() {
    // If the filename is set, then read from the file
    if (event_file_path_ != "") {
        try {
            camera_ = std::make_unique<Metavision::Camera>(Metavision::Camera::from_file(
                event_file_path_, Metavision::FileConfigHints().real_time_playback(!as_fast_as_possible_)));
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return false;
        }
        // Otherwise, set the input source to the first available camera
    } else {
        try {
            camera_ = std::make_unique<Metavision::Camera>(Metavision::Camera::from_first_available());

            if (!cam_config_path_.empty()) {
                camera_->load(cam_config_path_);
            }

        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return false;
        }
    }

    // Add camera runtime error callback
    camera_->add_runtime_error_callback([](const Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); });

    return true;
}

bool Pipeline::parse_command_line(int argc, char *argv[]) {
    const std::string program_desc("Code sample using Metavision SDK to track simple, non colliding objects.\n"
                                   "By default, only ON events are tracked.\n");

    // clang-format off
    boost_po::options_description options_desc("Options");

    boost_po::options_description io_options_desc("Input/Output options");
    io_options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i",           boost_po::value<std::string>(&event_file_path_), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j",        boost_po::value<std::string>(&cam_config_path_), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("stc-threshold",                boost_po::value<uint32_t>(&stc_threshold_)->default_value(0), "Software STC: filtering threshold delay (in us). 0 means no filtering will be applied")
        ("display,d",                    boost_po::value(&display_), "Activate display or not")
        ("out-video,o",                  boost_po::value<std::string>(&output_video_), "Path to an output AVI file to save the resulting video. "
                                                                                       "A frame is generated every time the tracking callback is called.")
        ("process-from,s",               boost_po::value<Metavision::timestamp>(&write_from_), "Start time to save the video (in us)")
        ("process-to,e",                 boost_po::value<Metavision::timestamp>(&write_to_), "End time to save the video (in us)")
        ("time,t",                       boost_po::value(&time_), "Measure the time of processing in the interest time range")
        ("log-results,l",                boost_po::value<std::string>(&res_csv_file_), "File to save the output of tracking")
        ("traj-duration",                boost_po::value<Metavision::timestamp>(&trajectory_duration_us_)->default_value(0), "Last X us of trajectory to draw for each tracked cluster. 0 means no trajectory will be drawn.")
        ;
    boost_po::options_description parameters_options_desc("Processing parameters options");
    parameters_options_desc.add_options()
        ("cell-width",                   boost_po::value<int>(&cell_width_)->default_value(7), "Cell width used for clustering, in pixels")
        ("cell-height",                  boost_po::value<int>(&cell_height_)->default_value(7), "Cell height used for clustering, in pixels")
        ("max-distance,D",               boost_po::value<int>(&max_distance_)->default_value(50), "Maximum distance for clusters association, in pixels")
        ("activation-threshold,a",       boost_po::value<int>(&activation_threshold_)->default_value(10), "Minimum number of events in a cell to consider it as active")
        ("min-track-time",               boost_po::value<Metavision::timestamp>(&min_tracking_time_)->default_value(1000), "Minimum time necessary for a cluster to be tracked (us)")
        ("apply-filter",                 boost_po::value<bool>(&apply_filter_)->default_value(true), "If true, then the cell activation threshold considers only one event per pixel")
        ("min-size",                     boost_po::value<int>(&min_size_)->default_value(10), "Minimum object size, in pixels")
        ("max-size",                     boost_po::value<int>(&max_size_)->default_value(300), "Maximum object size, in pixels")
        ("untracked-threshold",          boost_po::value<int>(&untracked_threshold_)->default_value(5),"Maximum number of times a cluster can stay untracked before being removed")
        ("static-memory",                boost_po::value<Metavision::timestamp>(&static_memory_)->default_value(1000000),"Time after which the static constraint of detected clusters is reset (us)")
        ("max-size-variation",           boost_po::value<int>(&max_size_variation_)->default_value(100), "Maximum size variation (in pixels) allowed to match two clusters")
        ("min-dist-moving-obj",          boost_po::value<int>(&min_dist_moving_obj_)->default_value(0), "Minimum distance in pixels to consider a cluster moving (else, it is considered a static object which we don't want to track)")
        ("polarity-filter",              boost_po::value<Metavision::EvtFilterType>(&filter_polarity_)->default_value(Metavision::EvtFilterType::FILTER_NEG),"Polarity of events to filter out for the tracking [NO_FILTER, FILTER_NEG, FILTER_POS, SEPARATE_FILTER]")
        ("processing-accumulation-time", boost_po::value<Metavision::timestamp>(&accumulation_time_)->default_value(5000),"Processing accumulation time (in us)")
        ;
    boost_po::options_description no_track_zones_options_desc("No-track zones options (same number of values expected for all arguments)");
    no_track_zones_options_desc.add_options()
        ("x", boost_po::value<std::vector<int>>(&center_x_)->multitoken(), "X coordinates of no-track zone center")
        ("y", boost_po::value<std::vector<int>>(&center_y_)->multitoken(), "Y coordinates of no-track zone center")
        ("radius", boost_po::value<std::vector<int>>(&radius_)->multitoken(), "Radius of no-track zone")
        ("inside", boost_po::value<std::vector<bool>>(&filter_inside_)->multitoken(), "Inside flags of no-track zone: true/1 to filter inside of circle, false/0 to filter outside")
        ;
    // clang-format on

    options_desc.add(io_options_desc);
    options_desc.add(parameters_options_desc);
    options_desc.add(no_track_zones_options_desc);

    boost_po::variables_map vm;
    try {
        boost_po::store(boost_po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        boost_po::notify(vm);
    } catch (boost_po::error &e) {
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

    // When running offline on a recording, we want to replay data as fast as possible when the display is disabled
    const bool is_offline = !event_file_path_.empty();
    if (is_offline && !display_) {
        as_fast_as_possible_ = true;
    }

    if (vm.count("x") || vm.count("y") || vm.count("radius") || vm.count("inside")) {
        if (center_x_.size() != center_y_.size() || center_x_.size() != radius_.size() ||
            center_x_.size() != filter_inside_.size()) {
            MV_LOG_ERROR()
                << "To indicate no-tracking zones, the user must provide all the following information FOR EACH ZONE :";
            MV_LOG_ERROR() << "x (pixels), y(pixels), radius (pixels), inside (bool)";
            return false;
        }
    }

    return true;
}

bool Pipeline::start() {
    const bool started = camera_->start();
    if (!started)
        MV_LOG_ERROR() << "The camera could not be started.";
    return started;
}

void Pipeline::stop() {
    // Show the number of counted trackers
    MV_LOG_INFO() << "Counter =" << tracker_->get_cluster_count();

    try {
        camera_->stop();
    } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
}

void Pipeline::run() {
    if (window_) {
        while (!window_->should_close()) {
            // we sleep the main thread a bit to avoid using 100% of a CPU's core
            static constexpr std::int64_t kSleepPeriodMs = 20;

            Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
        }
    } else {
        // Wait until the end of the file
        std::unique_lock<std::mutex> lock(process_mutex_);
        process_cond_.wait(lock, [this] { return !is_processing_; });
    }
}

/// [TRACKING_TRACKER_CALLBACK_BEGIN]
void Pipeline::tracker_callback(Metavision::timestamp ts) {
    tracker_->get_results(ts, clusters_);

    if (tracker_logger_)
        tracker_logger_->log_output(ts, clusters_);

    if (frame_generation_) {
        frame_generation_->generate(ts, back_img_);

        tracked_clusters_->update_trajectories(ts, clusters_);

        Metavision::draw_tracking_results(ts, clusters_.cbegin(), clusters_.cend(), back_img_);
        tracked_clusters_->draw(back_img_);
        draw_no_track_zones(back_img_);

        if (video_writer_)
            video_writer_->write_frame(ts, back_img_);

        if (window_)
            window_->show(back_img_);
    }

    if (timer_)
        timer_->update_timer(ts);
}
/// [TRACKING_TRACKER_CALLBACK_END]

void Pipeline::draw_no_track_zones(cv::Mat &img) {
    for (size_t i = 0; i < center_x_.size(); ++i)
        cv::circle(img, cv::Point(center_x_[i], center_y_[i]), radius_[i],
                   filter_inside_[i] ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0));
}

void Pipeline::initialize_tracker() {
    const auto &geometry    = camera_->geometry();
    const int sensor_width  = geometry.width();
    const int sensor_height = geometry.height();

    // Creates filters
    Metavision::SpatterTrackerAlgorithmConfig tracker_config(
        cell_width_, cell_height_, accumulation_time_, untracked_threshold_, activation_threshold_, apply_filter_,
        max_distance_, min_size_, max_size_, min_tracking_time_, static_memory_, max_size_variation_,
        min_dist_moving_obj_, filter_polarity_);

    tracker_ = std::make_unique<Metavision::SpatterTrackerAlgorithmSync>(sensor_width, sensor_height, tracker_config);

    for (size_t i = 0; i < center_x_.size(); ++i)
        tracker_->add_nozone(cv::Point(center_x_[i], center_y_[i]), radius_[i], filter_inside_[i]);

    tracked_clusters_ = std::make_unique<Metavision::ClusterTrajectories>(trajectory_duration_us_);

    if (!res_csv_file_.empty()) {
        tracker_logger_.reset(new Metavision::SpatterTrackerCsvLogger(res_csv_file_));
    }

    if (!output_video_.empty() || display_) {
        frame_generation_.reset(
            new Metavision::OnDemandFrameGenerationAlgorithm(sensor_width, sensor_height, accumulation_time_));
    }

    if (!output_video_.empty()) {
        video_writer_.reset(new SimpleVideoWriter(sensor_width, sensor_height, accumulation_time_, 30, output_video_));
        video_writer_->set_write_range(write_from_, write_to_);
    }

    if (display_) {
        window_ = std::make_unique<Metavision::Window>("Tracking result", sensor_width, sensor_height,
                                                       Metavision::BaseWindow::RenderMode::BGR);
        // Notify the pipeline when the window is exited
        window_->set_keyboard_callback(
            [this](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE) {
                    if (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q) {
                        {
                            std::lock_guard<std::mutex> lock(process_mutex_);
                            is_processing_ = false;
                            process_cond_.notify_all();
                        }
                        window_->set_close_flag();
                    } else {
                        MV_LOG_INFO() << key;
                    }
                }
            });
    }

    if (time_) {
        timer_.reset(new SimpleTimer());
        timer_->set_time_range(write_from_, write_to_);
    }

    /// [SLICER_INIT_BEGIN]
    using ConditionStatus  = Metavision::EventBufferReslicerAlgorithm::ConditionStatus;
    using Condition        = Metavision::EventBufferReslicerAlgorithm::Condition;
    const auto tracking_cb = [this](ConditionStatus, Metavision::timestamp ts, std::size_t n) { tracker_callback(ts); };
    const auto slicing_condition = Condition::make_n_us(accumulation_time_);
    slicer_ = std::make_unique<Metavision::EventBufferReslicerAlgorithm>(tracking_cb, slicing_condition);
    /// [SLICER_INIT_END]

    /// [CAMERA_CALLBACK_BEGIN]
    Metavision::EventsCDCallback camera_callback;
    if (stc_threshold_ > 0) {
        stc_filter_.reset(
            new Metavision::SpatioTemporalContrastAlgorithm(sensor_width, sensor_height, stc_threshold_, false));
        camera_callback = [this](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            filtered_events_.clear();
            stc_filter_->process_events(begin, end, std::back_inserter(filtered_events_));

            // Frame generator must be called first
            if (frame_generation_)
                frame_generation_->process_events(filtered_events_.cbegin(), filtered_events_.cend());

            slicer_->process_events(filtered_events_.cbegin(), filtered_events_.cend(),
                                    [&](auto begin_it, auto end_it) { tracker_->process_events(begin_it, end_it); });
        };
    } else {
        camera_callback = [this](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            // Frame generator must be called first
            if (frame_generation_)
                frame_generation_->process_events(begin, end);
            slicer_->process_events(begin, end,
                                    [&](auto begin_it, auto end_it) { tracker_->process_events(begin_it, end_it); });
        };
    }

    camera_->cd().add_callback(camera_callback);
    /// [CAMERA_CALLBACK_END]

    // Stops the pipeline when the camera is stopped
    camera_->add_status_change_callback([this](const Metavision::CameraStatus &status) {
        if (status == Metavision::CameraStatus::STOPPED) {
            std::lock_guard<std::mutex> lock(process_mutex_);
            is_processing_ = false;

            if (window_)
                window_->set_close_flag();

            process_cond_.notify_all();
        }
    });
}

/// Main function
int main(int argc, char *argv[]) {
    Pipeline pipeline;

    // Parse command line
    if (!pipeline.parse_command_line(argc, argv))
        return 1;

    // Initialize the camera
    if (!pipeline.initialize_camera())
        return 2;

    // Initialize tracker
    pipeline.initialize_tracker();

    // Start the camera
    if (!pipeline.start())
        return 3;

    // Wait until the end of the pipeline
    pipeline.run();

    pipeline.stop();

    return 0;
}
