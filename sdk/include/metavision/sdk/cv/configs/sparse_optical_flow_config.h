/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_SPARSE_OPTICAL_FLOW_CONFIG_H
#define METAVISION_SDK_CV_SPARSE_OPTICAL_FLOW_CONFIG_H

#include "metavision/sdk/base/events/event_cd.h"

namespace Metavision {

/// @brief Configuration to use with the sparse optical flow algorithm
struct SparseOpticalFlowConfig {
    enum class Preset { SlowObjects, FastObjects };

    SparseOpticalFlowConfig(Preset preset) {
        if (preset == Preset::FastObjects) {
            distance_update_rate = 0.05f;
            damping              = 0.707f;
            omega_cutoff         = 10.f;
            min_cluster_size     = 10;
            max_link_time        = 20000;
            match_polarity       = true;
            use_simple_match     = true;
            full_square          = false;
            last_event_only      = false;
            size_threshold       = 100000000;
        } else if (preset == Preset::SlowObjects) {
            distance_update_rate = 0.05f;
            damping              = 0.707f;
            omega_cutoff         = 5.f;
            min_cluster_size     = 5;
            max_link_time        = 50000;
            match_polarity       = true;
            use_simple_match     = true;
            full_square          = true;
            last_event_only      = false;
            size_threshold       = 100000000;
        }
    }

    /// @brief Constructor
    /// @param distance_gain Distance gain of the low pass filter to compute the size of a cluster
    /// @param damping Damping parameter of the Luenberger estimator, will determine how fast the speed computation of a
    /// cluster will converge. It is related to the eigenvalues of the observer. The damping ratio influences the decay
    /// of the oscillations in the observer response. A value of 1 is for critical damping, while a value of 0 produces
    /// perfect oscillations. The system is underdamped for values lesser than 1 and overdamped for values greater
    /// than 1.
    /// @param omega_cutoff Parameter of the Luenberger estimator, will determine how fast the speed computation of a
    /// cluster will respond to changes in the system. It is related to the eigenvalues of the observer. A higher
    /// frequency results in faster convergence but may introduce more oscillations.
    /// @param min_cluster_size Minimal number of events hitting a cluster before it gets outputted
    /// @param max_link_time Maximum time in us for two events to be linked in time
    /// @param match_polarity Set to true to create mono-polarity clusters, otherwise it will use multi-polarity ones
    /// @param use_simple_match Set to false to use a costlier constant velocity match strategy
    /// @param full_square Set to true to check connectivity on the full 3x3 square around the events, otherwise it will
    /// use the 3x3 cross around it
    /// @param last_event_only Set to true to only check the connectivity with the last event added to the cluster
    /// referenced at every pixel positions
    /// @param size_threshold Threshold on the spatial size of a cluster before being outputted
    SparseOpticalFlowConfig(float distance_gain = 0.05f, float damping = 0.707f, float omega_cutoff = 5.f,
                            unsigned int min_cluster_size = 5, timestamp max_link_time = 50000,
                            bool match_polarity = true, bool use_simple_match = true, bool full_square = true,
                            bool last_event_only = false, unsigned int size_threshold = 100000000) :
        distance_update_rate(distance_gain),
        damping(damping),
        omega_cutoff(omega_cutoff),
        min_cluster_size(min_cluster_size),
        max_link_time(max_link_time),
        match_polarity(match_polarity),
        use_simple_match(use_simple_match),
        full_square(full_square),
        last_event_only(last_event_only),
        size_threshold(size_threshold) {}

    float distance_update_rate; ///< Update rate on the distance between the center of the cluster and new events. In
                                ///< other words, it is the gain of the low pass filter to compute the size of a CCL. It
                                ///< defines how fast it will converge. Between 0 and 1 : the closer to 0, the more
                                ///< filtered (thus slower) the size estimation will be. The closer to 1, the more
                                ///< sensitive to new events it will be.
    float damping; ///< Parameter of the Luenberger estimator, will determine how fast the speed computation of a CCL
                   ///< will converge. The damping ratio influences the decay of the oscillations in the observer
                   ///< response. A value of 1 is for critical damping, while a value of 0 produces
                   ///< perfect oscillations. The system is underdamped for values lesser than 1 and overdamped
                   ///< for values greater than 1.
    float omega_cutoff; ///< Parameter of the Luenberger estimator, will determine how fast the speed computation of a
                        ///< CCL will respond to changes in the system. A higher frequency results in faster convergence
                        ///< but may introduce more oscillations.
    unsigned int min_cluster_size; ///< Minimal number of events hitting a cluster before it gets outputted (depends on
                                   ///< the size of the clusters to track)
    timestamp max_link_time; ///< Maximum time difference in us for two events to be associated to the same CCL. The
                             ///< higher the link time, the more events can be associated, the more wrong associations
                             ///< can be made. On the contrary, a lower link time will limit wrong associations but may
                             ///< also prevent the formation of correct clusters.
    bool match_polarity;     ///< Decide if we have multi-polarity cluster or mono polarity ones. If True, positive and
                             ///< negative events will trigger same CCLs.
    bool use_simple_match;   ///< If false, will use a constant velocity match strategy, i.e. we ensure each new event
                             ///< matches the estimated cluster speeds
    bool full_square; ///< If true, connectivity is checked on the full 3x3 square around the events, otherwise just the
                      ///< 3x3 cross around it (top, bottom, left and right pixels)
    bool last_event_only;        ///< If true will only check the connectivity with the last event added to the cluster,
                                 ///< instead of all the cluster pixels
    unsigned int size_threshold; ///< Threshold on the spatial size of a cluster (in pixels). Maximum distance allowed
                                 ///< from the estimated center of the cluster to the last event added to it
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_SPARSE_OPTICAL_FLOW_CONFIG_H
