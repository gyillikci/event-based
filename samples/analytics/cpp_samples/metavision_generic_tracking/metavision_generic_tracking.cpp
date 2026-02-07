/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// Example of using SDK to track objects.

#include <mutex>
#include <condition_variable>
#include <set>
#include <boost/program_options.hpp>

#include <metavision/sdk/analytics/algorithms/tracking_algorithm.h>
#include <metavision/sdk/analytics/configs/tracking_algorithm_config.h>
#include <metavision/sdk/analytics/utils/tracking_drawing.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/rolling_event_buffer.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

#include "simple_video_writer.h"

// Utility function to parse command line
namespace boost_po = boost::program_options;

using RollingEventBuffer = Metavision::RollingEventBuffer<Metavision::EventCD>;
using TrackingBuffer     = std::vector<Metavision::EventTrackingData>;

class Pipeline {
public:
    Pipeline() = default;

    ~Pipeline() = default;

    bool parse_command_line(int argc, char *argv[]);

    /// @brief Initializes the camera
    bool initialize_camera();

    /// @brief Initializes the filters
    void initialize_tracker();

    /// @brief Starts the camera
    bool start();

    /// @brief Stops the pipeline
    void stop();

    /// @brief Waits until the end of the file or until the exit of the display
    void run();

private:
    /// @brief Processing of the output of the @ref Metavision::TrackingAlgorithm.
    void tracker_callback(Metavision::EventBufferReslicerAlgorithm::ConditionStatus status, Metavision::timestamp ts,
                          std::size_t n_events);

    /// @brief Callback called by the @ref Metavision::Window when a key is pressed.
    void keyboard_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods);

    /// Tracker's update frequency.
    float update_frequency_;

    /// Min and max size of an object to track.
    int min_size_;
    int max_size_;

    /// Camera's parameters.
    std::string event_file_path_;
    std::string cam_config_path_;

    /// Display's parameters.
    /// If we display data on the screen or not.
    bool display_;
    /// Filename to save the resulted video.
    std::string output_video_;
    /// Current image in which the result of the tracking is drawn.
    cv::Mat back_img_;

    // Conditional variables to notify the end of the processing.
    std::condition_variable process_cond_;
    std::mutex process_mutex_;
    volatile bool is_processing_ = true;

    /// Time at which to start the video writing.
    /// 0 meaning from the beginning.
    Metavision::timestamp write_from_;
    /// Time at which to stop the video writing.
    /// Max meaning until the end.
    Metavision::timestamp write_to_;

    /// Software SpatioTemporalContrast threshold (if 0 STC is disabled)
    int sw_stc_threshold_;

    /// Time slice duration accumulated in the rolling event buffer
    int accumulation_time_us_;

    std::unique_ptr<Metavision::Camera> camera_;                            ///< Pointer to the camera class
    std::unique_ptr<Metavision::TrackingAlgorithm> tracker_;                ///< Instance of the tracker
    std::unique_ptr<SimpleVideoWriter> video_writer_;                       ///< Video writer
    std::unique_ptr<Metavision::Window> window_;                            ///< Display window
    std::unique_ptr<Metavision::SpatioTemporalContrastAlgorithm> stc_algo_; ///< Noise filter

    /// Ids of all the tracked objects.
    TrackingBuffer tracked_objects_;
    std::set<size_t> tracked_object_ids_;

    /// Vector of events used to filter events
    std::vector<Metavision::EventCD> filtered_events_;
    RollingEventBuffer event_buffer_;
    std::unique_ptr<Metavision::EventBufferReslicerAlgorithm> slicer_;
};

bool Pipeline::initialize_camera() {
    // If the filename is set, then read from the file
    if (event_file_path_ != "") {
        try {
            camera_ = std::make_unique<Metavision::Camera>(
                Metavision::Camera::from_file(event_file_path_, Metavision::FileConfigHints().real_time_playback(false)));
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return false;
        }
        // Otherwise, set the input source to the first available camera
    } else {
        try {
            camera_ = std::make_unique<Metavision::Camera>(Metavision::Camera::from_first_available());

            if (!cam_config_path_.empty()) {
               try {
                 camera_->load(cam_config_path_);
              } catch (const Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
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
    const std::string short_program_desc("Code sample tracking generic moving objects in a stream of events from an "
                                         "event-based device or recorded data.\n");

    const std::string long_program_desc(
        short_program_desc + "Press 'q' to leave the program.\n"
                             "Press 'a' to increase the minimum size of the object to track.\n"
                             "Press 'b' to decrease the minimum size of the object to track.\n"
                             "Press 'c' to increase the maximum size of the object to track.\n"
                             "Press 'd' to decrease the maximum size of the object to track.\n");

    boost_po::options_description options_desc;
    boost_po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i",    boost_po::value<std::string>(&event_file_path_), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j", boost_po::value<std::string>(&cam_config_path_), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("update-frequency",      boost_po::value<float>(&update_frequency_)->default_value(200.f), "Tracker's update frequency, in Hz")
        ("min-size",              boost_po::value<int>(&min_size_)->default_value(10), "Minimal size of an object to track, in pixels")
        ("max-size",              boost_po::value<int>(&max_size_)->default_value(300), "Maximal size of an object to track, in pixels")
        ("sw-stc-threshold",      boost_po::value<int>(&sw_stc_threshold_)->default_value(10000), "Software STC threshold in us (10000 by default, disabled if sensor's STC is enabled or if the threshold is set to 0)")
        ("acc-time,a",            boost_po::value<int>(&accumulation_time_us_)->default_value(10000), "Duration of the time slice to store in the rolling event buffer at each tracking step")
        ;
    // clang-format on

    boost_po::options_description outcome_options("Outcome options");
    // clang-format off
    outcome_options.add_options()   
        ("display,d",           boost_po::value(&display_)->default_value(true), "Activate display or not")
        ("output-video-file,o", boost_po::value<std::string>(&output_video_), "Path to an output AVI file to save the resulting video. A frame is generated every time the tracking callback is called.")
        ("write-from",          boost_po::value<Metavision::timestamp>(&write_from_)->default_value(0), "Start time to save video (in us)")
        ("write-to",            boost_po::value<Metavision::timestamp>(&write_to_)->default_value(std::numeric_limits<Metavision::timestamp>::max()), "End time to save video (in us)")
        ;
    // clang-format on

    options_desc.add(base_options).add(outcome_options);

    boost_po::variables_map vm;
    try {
        boost_po::store(boost_po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        boost_po::notify(vm);
    } catch (boost_po::error &e) {
        MV_LOG_ERROR() << short_program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return false;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << short_program_desc;
        MV_LOG_INFO() << options_desc;
        return false;
    }

    MV_LOG_INFO() << long_program_desc;

    return true;
}

/// @brief Starts the camera.
bool Pipeline::start() {
    const bool started = camera_->start();
    if (!started) {
        MV_LOG_ERROR() << "The camera could not be started.";
    }

    return started;
}

/// @brief Stops the camera when finished.
void Pipeline::stop() {
    try {
        camera_->stop();
    } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }
}

/// @brief Runs the pipeline.
///
/// Displays the window if enabled, otherwise waits for the tracking thread to complete (i.e. the camera's one).
void Pipeline::run() {
    if (window_) {
        while (!window_->should_close()) {
            // we sleep the main thread a bit to avoid using 100% of a CPU's core
            static constexpr std::int64_t kSleepPeriodMs = 20;

            Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
        }
    } else {
        // Waits until the end of the file or until the user presses 'q'.
        std::unique_lock<std::mutex> lock(process_mutex_);
        process_cond_.wait(lock, [this] { return !is_processing_; });
    }

    MV_LOG_INFO() << "Number of tracked objects:" << tracked_object_ids_.size();
}

/// [GENERIC_TRACKING_TRACKER_CALLBACK_BEGIN]
void Pipeline::tracker_callback(Metavision::EventBufferReslicerAlgorithm::ConditionStatus, Metavision::timestamp ts,
                                std::size_t) {
    tracked_objects_.clear();
    tracker_->process_events(event_buffer_.cbegin(), event_buffer_.cend(), std::back_inserter(tracked_objects_));

    for (const auto &obj : tracked_objects_)
        tracked_object_ids_.insert(obj.object_id_);

    if (display_) {
        Metavision::BaseFrameGenerationAlgorithm::generate_frame_from_events(event_buffer_.cbegin(),
                                                                             event_buffer_.cend(), back_img_);
        Metavision::draw_tracking_results(ts, tracked_objects_.cbegin(), tracked_objects_.cend(), back_img_);

        if (video_writer_)
            video_writer_->write_frame(ts, back_img_);

        if (window_)
            window_->show(back_img_);
    }
}
/// [GENERIC_TRACKING_TRACKER_CALLBACK_END]

void Pipeline::keyboard_callback(Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
    static constexpr int kSizeStep = 10;

    if (action == Metavision::UIAction::RELEASE) {
        switch (key) {
        case Metavision::UIKeyEvent::KEY_Q:
        case Metavision::UIKeyEvent::KEY_ESCAPE: {
            std::lock_guard<std::mutex> lock(process_mutex_);
            is_processing_ = false;
            process_cond_.notify_all();
        }
            window_->set_close_flag();
            break;
        case Metavision::UIKeyEvent::KEY_A:
            if (min_size_ + kSizeStep <= max_size_) {
                min_size_ += kSizeStep;
                MV_LOG_INFO() << "Setting min size to" << min_size_;
                tracker_->set_min_size(min_size_);
            }
            break;
        case Metavision::UIKeyEvent::KEY_B:
            if (min_size_ - kSizeStep >= 0) {
                min_size_ -= kSizeStep;
                MV_LOG_INFO() << "Setting min size to" << min_size_;
                tracker_->set_min_size(min_size_);
            }
            break;
        case Metavision::UIKeyEvent::KEY_C:
            if (max_size_ <= std::numeric_limits<int>::max() - kSizeStep) {
                max_size_ += kSizeStep;
                MV_LOG_INFO() << "Setting max size to" << max_size_;
                tracker_->set_max_size(max_size_);
            }
            break;
        case Metavision::UIKeyEvent::KEY_D:
            if (max_size_ - kSizeStep >= min_size_) {
                max_size_ -= kSizeStep;
                MV_LOG_INFO() << "Setting max size to" << max_size_;
                tracker_->set_max_size(max_size_);
            }
            break;
        default:
            break;
        }
    }
}

void Pipeline::initialize_tracker() {
    /// [GENERIC_TRACKING_CREATE_ROLLING_BUFFER_BEGIN]
    using RollingEventBufferConfig = Metavision::RollingEventBufferConfig;
    event_buffer_                  = RollingEventBuffer(
        RollingEventBufferConfig::make_n_us(static_cast<Metavision::timestamp>(accumulation_time_us_)));
    /// [GENERIC_TRACKING_CREATE_ROLLING_BUFFER_END]

    const auto &geometry    = camera_->geometry();
    const int sensor_width  = geometry.width();
    const int sensor_height = geometry.height();

    auto *event_trail_filter = camera_->get_device().get_facility<Metavision::I_EventTrailFilterModule>();
    const bool hardware_stc_enabled = event_trail_filter && event_trail_filter->is_enabled();

    // We prepare a software STC algo to use if sensor's hardware STC is not enabled and software threshold not 0
    const bool bypass_software_stc = hardware_stc_enabled || sw_stc_threshold_ == 0;
    if (!bypass_software_stc)
        stc_algo_ = std::make_unique<Metavision::SpatioTemporalContrastAlgorithm>(sensor_width, sensor_height,
                                                                                  sw_stc_threshold_);

    // Creates filters.
    Metavision::TrackingConfig tracking_config;
    tracker_ = std::make_unique<Metavision::TrackingAlgorithm>(sensor_width, sensor_height, tracking_config);
    tracker_->set_min_size(min_size_);
    tracker_->set_max_size(max_size_);

    if (!output_video_.empty()) {
        video_writer_.reset(new SimpleVideoWriter(sensor_width, sensor_height,
                                                  static_cast<int>(1'000'000.f / update_frequency_), output_video_));
        video_writer_->set_write_range(write_from_, write_to_);
    }

    if (display_) {
        back_img_.create(sensor_height, sensor_width, CV_8UC3);
        window_ = std::make_unique<Metavision::Window>("Metavision Tracking sample", sensor_width, sensor_height,
                                                       Metavision::BaseWindow::RenderMode::BGR);
        // Notifies the pipeline when the window is exited.
        window_->set_keyboard_callback(std::bind(&Pipeline::keyboard_callback, this, std::placeholders::_1,
                                                 std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
    }

    /// [GENERIC_TRACKING_SET_CAMERA_CALLBACK_BEGIN]
    camera_->cd().add_callback([this](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        const Metavision::EventCD *begin_it = begin;
        const Metavision::EventCD *end_it   = end;
        filtered_events_.clear();

        if (stc_algo_) {
            filtered_events_.reserve(std::distance(begin_it, end_it));

            const auto last = stc_algo_->process_events(begin_it, end_it, filtered_events_.begin());
            const auto size = std::distance(filtered_events_.begin(), last);

            begin_it = filtered_events_.data();
            end_it   = begin_it + size;
        }

        slicer_->process_events(begin_it, end_it, [&](const auto sub_slice_begin_it, const auto sub_slice_end_it) {
            event_buffer_.insert_events(sub_slice_begin_it, sub_slice_end_it);
        });
    });
    /// [GENERIC_TRACKING_SET_CAMERA_CALLBACK_END]

    /// [GENERIC_TRACKING_CREATE_SLICER_BEGIN]
    const auto update_period = static_cast<Metavision::timestamp>(1'000'000.f / update_frequency_);
    const auto cond          = Metavision::EventBufferReslicerAlgorithm::Condition::make_n_us(update_period);
    slicer_                  = std::make_unique<Metavision::EventBufferReslicerAlgorithm>(
        [&](Metavision::EventBufferReslicerAlgorithm::ConditionStatus status, Metavision::timestamp ts,
            std::size_t n_events) { tracker_callback(status, ts, n_events); },
        cond);
    /// [GENERIC_TRACKING_CREATE_SLICER_END]

    // Ends the pipeline when the camera is stopped.
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

    // Parses command line.
    if (!pipeline.parse_command_line(argc, argv))
        return 1;

    // Initializes the camera.
    if (!pipeline.initialize_camera())
        return 2;

    // Initializes the tracker.
    pipeline.initialize_tracker();

    // Starts the camera.
    if (!pipeline.start())
        return 3;

    // Waits until the end of the pipeline.
    pipeline.run();

    pipeline.stop();

    return 0;
}
