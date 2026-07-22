/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_CLUSTER_TRAJECTORIES_H
#define METAVISION_SDK_ANALYTICS_CLUSTER_TRAJECTORIES_H

#include "metavision/sdk/analytics/events/event_spatter_cluster.h"

namespace Metavision {

/// @brief Class to store the recent history of a cluster's trajectory
class ClusterTrajectory {
public:
    using Path = std::deque<std::pair<timestamp, cv::Point>>;

    /// @brief Constructor
    /// @param id Cluster's id
    /// @param duration Maximum memory for the trajectory's points
    /// @param t Timestamp of the first point to store
    /// @param p First point to store
    ClusterTrajectory(int id, timestamp duration, timestamp t, const cv::Point &p);

    /// @brief Adds a point to the trajectory
    /// @param t The timestamp of the new point
    /// @param p The new point
    void add_pose(timestamp t, const cv::Point &p);

    /// @brief Removes all points in the trajectory with a timestamp older than t - duration
    /// @param t Current timestamp. Points with timestamps lower than 't-duration' will be removed
    void remove_old_poses(timestamp t);

    /// @brief Gets the current trajectory
    /// @returns The current trajectory
    const Path &get_path() const;

    /// @brief Provide the trajectory id (same as cluster tracked)
    /// @returns The cluster (and trajectory) id
    int id() const;

private:
    int id_;             ///< Cluster's ID
    timestamp duration_; ///< Trajectory of the tracker with x,y position of the tracker's center and the corresponding
                         ///< timestamp
    Path traj_;          ///< (t,(x,y)) points of the trajectory
};

/// @brief Class to store and manage all tracked trajectories with a limited memory
class ClusterTrajectories {
public:
    /// @brief Constructor
    /// @param duration_us The maximum memory of trajectories to monitor. Points older than t_current - duration_us will
    /// be forgotten.
    ClusterTrajectories(timestamp duration_us);

    /// @brief Updates the trajectories of tracked clusters with new clusters positions
    /// Removes the trajectory points that are too old
    /// @param ts The current timestamp
    /// @param new_clusters Clusters detected at the current timestamp
    void update_trajectories(Metavision::timestamp ts,
                             const std::vector<Metavision::EventSpatterCluster> &new_clusters);

    /// @brief Updates the trajectories of tracked clusters with new clusters positions
    /// Removes the trajectory points that are too old
    /// @tparam InputIt Iterator type over a tracking result
    /// @param ts The current timestamp
    /// @param begin First cluster detected at the current timestamp
    /// @param end Last cluster detected at the current timestamp
    template<typename InputIt>
    void update_trajectories(Metavision::timestamp ts, InputIt begin, InputIt end);

    /// @brief Draws the recent trajectory if tracked clusters
    /// @param output The image to draw trajectories on
    void draw(cv::Mat &output) const;

private:
    const timestamp duration_;
    std::vector<ClusterTrajectory> tracked_clusters_;
    std::vector<int> last_seen_clusters_;
};

} // namespace Metavision

#include "metavision/sdk/analytics/utils/detail/cluster_trajectories_impl.h"

#endif // METAVISION_SDK_ANALYTICS_CLUSTER_TRAJECTORIES_H
