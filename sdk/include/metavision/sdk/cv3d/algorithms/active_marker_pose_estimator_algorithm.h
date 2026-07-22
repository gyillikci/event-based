/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_ACTIVE_MARKER_POSE_ESTIMATOR_H
#define METAVISION_SDK_CV3D_ACTIVE_MARKER_POSE_ESTIMATOR_H

#include <opencv2/core.hpp>
#include <Eigen/Core>

#include "metavision/sdk/cv/utils/camera_geometry.h"
#include "metavision/sdk/cv/algorithms/active_marker_tracker_algorithm.h"
#include "metavision/sdk/base/utils/log.h"

namespace Metavision {

/// @brief A class that estimates the 3D pose of an active marker with respect to the camera
///
/// An active marker is a rigid object composed of several blinking LEDs. Knowing the 3D location of each LED in the
/// coordinates system made by the active marker itself, its 3D pose can be recovered by tracking the 2D location of the
/// LEDs in the sensor plane.
/// The 3D pose can be computed either in the camera's clock (every N events or every N us) or in the system's clock (at
/// a given frequency).
class ActiveMarkerPoseEstimatorAlgorithm {
public:
    class PoseEstimator;
    using PoseUpdate         = std::optional<Eigen::Matrix4f>;
    using PoseUpdateCallback = std::function<void(timestamp, const PoseUpdate &)>;

    /// @brief Clock in which the 3D pose is estimated
    enum class EstimatorType { CameraClock, SystemClock };

    /// @brief Parameters for configuring the ActiveMarkerPoseEstimator
    ///
    /// In the camera's clock mode, if @p delta_n_events and @p delta_ts parameters are valid, then @p delta_n_events
    /// will be used
    struct Params {
        EstimatorType type       = EstimatorType::CameraClock; ///< Clock in which the 3D pose is estimated
        size_t delta_n_events    = 5000; ///< Number of events between two pose estimates in the Camera's clock
        timestamp delta_ts       = 0;    ///< Time interval (in us) between two pose estimates in the Camera's clock
        float estimation_rate_hz = 0.f;  ///< Pose estimation rate (in Hz) in the System's clock
        ActiveMarkerTrackerAlgorithm::Params
            tracker_params; ///< Parameters for the underlying @ref ActiveMarkerTrackerAlgorithm

        /// @brief Creates parameters for a camera clock-based estimator using the number of events as the trigger for
        /// pose updates
        /// @param delta_n_events The number of events between pose updates
        /// @param tracker_params Parameters for the @ref ActiveMarkerTrackerAlgorithm
        /// @return The parameters
        static Params make_n_events_camera_clock(size_t delta_n_events,
                                                 const ActiveMarkerTrackerAlgorithm::Params &tracker_params);

        /// @brief Creates parameters for a camera clock-based estimator using a time interval as the trigger for
        /// pose updates
        /// @param delta_ts The time interval (in us) between pose updates
        /// @param tracker_params Parameters for the @ref ActiveMarkerTrackerAlgorithm
        /// @return The parameters
        static Params make_n_us_camera_clock(timestamp delta_ts,
                                             const ActiveMarkerTrackerAlgorithm::Params &tracker_params);

        /// @brief Creates parameters for a system clock-based pose estimator
        /// @param estimation_rate The rate (in Hz) at which pose updates are estimated
        /// @param tracker_params Parameters for the @ref ActiveMarkerTrackerAlgorithm
        /// @return The parameters
        static Params make_system_clock(float estimation_rate,
                                        const ActiveMarkerTrackerAlgorithm::Params &tracker_params);
    };

    /// @brief Constructor
    /// @param params The parameters for configuring the estimator
    /// @param camera_geometry The camera geometry of the event-based camera
    /// @param active_marker_leds The map of LED IDs to their 3D locations in the active marker's coordinate system
    ActiveMarkerPoseEstimatorAlgorithm(const Params &params, const CameraGeometryBase<float> &camera_geometry,
                                       const std::unordered_map<std::uint32_t, cv::Point3f> &active_marker_leds);

    /// @brief Destructor
    ~ActiveMarkerPoseEstimatorAlgorithm();

    /// @brief Sets the callback function for receiving pose updates
    /// @param cb The callback function to be called
    /// @note If the pose estimation fails (e.g. not enough visible LEDs), the callback is still called but with an
    /// empty pose update. This is a way to get notified when the tracking gets lost.
    void set_pose_update_callback(PoseUpdateCallback cb);

    /// @brief Processes a range of @ref EventSourceId events for updating the pose estimation
    /// @tparam InputIt Input event iterator type, works for iterators over containers of @ref EventSourceId
    /// @param begin Iterator pointing to the first event in the stream
    /// @param end Iterator pointing to the past-the-end element in the stream
    template<typename InputIt>
    void process_events(InputIt begin, InputIt end);

    /// @brief Notifies the pose estimator that time has elapsed without new events, which may trigger several calls
    /// to the internal monitoring process
    /// @param ts Current timestamp
    void notify_elapsed_time(timestamp ts);

private:
    void process_active_tracks();

    ActiveMarkerTrackerAlgorithm tracker_;
    std::unique_ptr<PoseEstimator> pose_estimator_;

    std::vector<EventActiveTrack> active_tracks_;
};

} // namespace Metavision

#include "metavision/sdk/cv3d/algorithms/detail/active_marker_pose_estimator_algorithm_impl.h"

#endif // METAVISION_SDK_CV3D_ACTIVE_MARKER_POSE_ESTIMATOR_H
