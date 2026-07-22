/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_CLUSTER_TRAJECTORIES_IMPL_H
#define METAVISION_SDK_ANALYTICS_CLUSTER_TRAJECTORIES_IMPL_H

#include "metavision/sdk/analytics/utils/cluster_trajectories.h"

namespace Metavision {

template<typename InputIt>
void ClusterTrajectories::update_trajectories(Metavision::timestamp ts, InputIt begin, InputIt end) {
    last_seen_clusters_.clear();

    // Update trajectories or add new ones
    for (auto it = begin; it != end; ++it) {
        bool processed_new = false;
        const int id_new   = it->id;
        for (auto it_tracked = tracked_clusters_.begin(); it_tracked != tracked_clusters_.end(); ++it_tracked) {
            if (it_tracked->id() == id_new) {
                it_tracked->add_pose(ts, it->get_centroid());
                processed_new = true;
                break;
            }
        }
        if (!processed_new) {
            tracked_clusters_.emplace_back(ClusterTrajectory(id_new, duration_, ts, it->get_centroid()));
        }
        last_seen_clusters_.emplace_back(id_new);
    }

    // Keep only recent points in the trajectories
    auto it_tracked = tracked_clusters_.begin();
    while (it_tracked != tracked_clusters_.end()) {
        if (std::count(last_seen_clusters_.begin(), last_seen_clusters_.end(), it_tracked->id())) {
            it_tracked->remove_old_poses(ts);
            it_tracked++;
        } else {
            it_tracked = tracked_clusters_.erase(it_tracked);
        }
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_CLUSTER_TRAJECTORIES_IMPL_H
