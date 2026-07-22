/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_CONFIG_H
#define METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_CONFIG_H

#include <limits>

#include "metavision/sdk/analytics/utils/evt_filter_type.h"
#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Configuration to instantiate a SpatterTrackerAlgorithmConfig
struct SpatterTrackerAlgorithmConfig {
    /// @brief Constructor
    /// @param cell_width Width of cells used for clustering
    /// @param cell_height Height of cells used for clustering
    /// @param accumulation_time Time to accumulate events and process
    /// @param untracked_threshold Maximum number of times a spatter_cluster can stay untracked before being removed
    /// @param activation_threshold Threshold distinguishing an active cell from inactive cells (i.e. minimum number of
    /// events in a cell to consider it as active)
    /// @param apply_filter If true, then a simple filter to remove crazy pixels will be applied
    /// @param max_distance Max distance for clusters association (in pixels)
    /// @param min_size Minimum object size (in pixels) - minimum of the object's width and height should be larger than
    /// this value
    /// @param max_size Maximum object size (in pixels) - maximum of the object's width and height should be smaller
    /// than this value
    /// @param min_time_tracked Minimum delay to track a custer. Before this delay, it is tracked internally only.
    /// @param static_objects_memory Time after which the static constraint of detected clusters is relaxed
    /// @param max_size_variation Maximum size variation allowed for two cluster to be associated
    /// @param max_dist_static_obj Maximum displacement allowed for a static cluster (else, it is considered a moving
    /// cluster)
    /// @param filtered_polarity Polarity of events to filter out
    SpatterTrackerAlgorithmConfig(int cell_width, int cell_height, timestamp accumulation_time,
                                  int untracked_threshold = 5, int activation_threshold = 10, bool apply_filter = true,
                                  int max_distance = 50, const int min_size = 1,
                                  const int max_size = std::numeric_limits<int>::max(), timestamp min_time_tracked = 0,
                                  timestamp static_objects_memory = 0, int max_size_variation = 100,
                                  int max_dist_static_obj         = 0,
                                  EvtFilterType filtered_polarity = EvtFilterType::FILTER_NEG) :
        cell_width_(cell_width),
        cell_height_(cell_height),
        accumulation_time_(accumulation_time),
        untracked_threshold_(untracked_threshold),
        activation_threshold_(activation_threshold),
        apply_filter_(apply_filter),
        max_distance_(max_distance),
        min_size_(min_size),
        max_size_(max_size),
        min_time_tracked_us_(min_time_tracked),
        static_memory_(static_objects_memory),
        max_size_variation_(max_size_variation),
        max_dist_static_obj_(max_dist_static_obj),
        filter_type_(filtered_polarity) {}

    int cell_width_;  ///< Width of the cells used for clustering
    int cell_height_; ///< Height of the cells used for clustering
    [[deprecated("This attribute is deprecated since version 4.5.2 and will be removed in a future release")]] timestamp
        accumulation_time_;    ///< Accumulation time for asynchronous processing of events
    int untracked_threshold_;  ///< Maximum number of times a spatter cluster can stay untracked before being forgotten
    int activation_threshold_; ///< Minimum number of events in a cell to consider it active during clustering
    bool apply_filter_;        ///< If true then will consider only one event per pixel
    int max_distance_;         ///< Maximum distance (in pixel) allowed for clusters association
    int min_size_;             ///< Minimum object size (in pixel)
    int max_size_;             ///< Maximum object size (in pixel)
    timestamp min_time_tracked_us_; ///< Minimum duration a new cluster should be tracked before being output (in us)
    int max_dist_static_obj_;       ///< Maximum displacement allowed to consider a cluster static (in pixel)
    int max_size_variation_;        ///< Maximum size variation allowed for two clusters to be matched (in pixel)
    timestamp static_memory_;   ///< Time after which the static constraint of the detected clusters is relaxed (in us)
    EvtFilterType filter_type_; ///< Polarity of events to filter out
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_SPATTER_TRACKER_ALGORITHM_CONFIG_H
