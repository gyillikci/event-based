/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_SYNC_H
#define METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_SYNC_H

#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>
#include <vector>

#include "metavision/sdk/analytics/configs/spatter_tracker_algorithm_config.h"
#include "metavision/sdk/analytics/events/event_spatter_cluster.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

class SpatterTrackerAlgorithmInternal;

/// @brief Class that tracks spatter clusters using Metavision SpatterTracking API
class SpatterTrackerAlgorithmSync {
public:
    /// @brief Builds a new @ref SpatterTrackerAlgorithm object
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param config Spatter tracker's configuration
    SpatterTrackerAlgorithmSync(int width, int height, const SpatterTrackerAlgorithmConfig &config);

    /// @brief Default destructor
    ~SpatterTrackerAlgorithmSync();

    [[deprecated("This function is deprecated since version 4.3.0. Please use add_nozone() instead.")]] void
        set_nozone(cv::Point center, int radius);

    /// @brief Adds a region that isn't used for tracking
    /// @param center Center of the region
    /// @param radius Radius of the region
    /// @param inside True if the region to filter is inside the defined shape, false otherwise
    void add_nozone(const cv::Point &center, int radius, bool inside = true);

    /// @brief Returns the current number of clusters
    /// @return The current number of clusters
    int get_cluster_count() const;

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Processes a buffer of events and retrieves the detected clusters in one call
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param[in] it_begin Iterator to the first input event
    /// @param[in] it_end Iterator to the past-the-end event
    /// @param[in] ts Upper bound timestamp of the events time slice
    /// @param[out] clusters Detected clusters
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end, timestamp ts, std::vector<EventSpatterCluster> &clusters);

    /// @brief Retrieves the detected clusters since the last call to this method
    /// @param[in] ts Upper bound timestamp of the processed events
    /// @param[out] clusters Detected clusters
    void get_results(timestamp ts, std::vector<EventSpatterCluster> &clusters);

private:
    std::unique_ptr<SpatterTrackerAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_SYNC_H
