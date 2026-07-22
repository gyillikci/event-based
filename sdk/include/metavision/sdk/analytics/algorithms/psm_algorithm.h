/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_PSM_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_PSM_ALGORITHM_H

#include <functional>
#include <map>
#include <memory>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/analytics/configs/line_particle_tracking_config.h"
#include "metavision/sdk/analytics/configs/line_cluster_tracking_config.h"
#include "metavision/sdk/analytics/utils/line_cluster.h"
#include "metavision/sdk/analytics/utils/line_particle_tracking_output.h"

namespace Metavision {

class PsmAlgorithmInternal;

/// @brief Class that both counts objects and estimates their size using Metavision Particle Size Measurement API
///
class PsmAlgorithm {
public:
    using LineClustersOutput = std::vector<LineClusterWithId>;

    /// @brief Constructor
    /// @param sensor_width Sensor's width (in pixels)
    /// @param sensor_height Sensor's height (in pixels)
    /// @param rows Rows on which to instantiate line cluster trackers
    /// @param detection_config Detection config
    /// @param tracking_config Tracking config
    PsmAlgorithm(int sensor_width, int sensor_height, const std::vector<int> &rows,
                 const LineClusterTrackingConfig &detection_config, const LineParticleTrackingConfig &tracking_config);

    /// @brief Destructor
    ~PsmAlgorithm();

    /// @brief Processes a buffer of events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end);

    /// @brief Processes a buffer of events and retrieves the detected particles in one call
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @param[in] it_begin Iterator to the first input event
    /// @param[in] it_end Iterator to the past-the-end event
    /// @param[in] ts Upper bound timestamp of the events time slice
    /// @param[out] tracks Detected particles for each line
    /// @param[out] line_clusters Detected clusters for each line
    template<typename InputIt>
    void process_events(InputIt it_begin, InputIt it_end, const timestamp ts, LineParticleTrackingOutput &tracks,
                        LineClustersOutput &line_clusters);

    /// @brief Retrieves the detected particles since the last call to this method
    /// @param[in] ts Upper bound timestamp of the processed events
    /// @param[out] tracks Detected particles for each line
    /// @param[out] line_clusters Detected clusters for each line
    void get_results(const timestamp ts, LineParticleTrackingOutput &tracks, LineClustersOutput &line_clusters);

    /// @brief Resets line cluster trackers and line particle trackers
    void reset();

private:
    std::unique_ptr<PsmAlgorithmInternal> algo_;
};

template<typename InputIt>
void PsmAlgorithm::process_events(InputIt it_begin, InputIt it_end, const timestamp ts,
                                  LineParticleTrackingOutput &tracks, LineClustersOutput &line_clusters) {
    process_events(it_begin, it_end);
    get_results(ts, tracks, line_clusters);
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_PSM_ALGORITHM_H
