/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_ALGORITHMS_PROXIMITY_FILTER_ALGORITHM_H
#define METAVISION_SDK_CV_ALGORITHMS_PROXIMITY_FILTER_ALGORITHM_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

/// @brief Class filter that only propagates events close enough to a particular point in the sensor
class ProximityFilterAlgorithm {
public:
    /// @brief Constructor
    /// @param center Point with which the distance is computed
    /// @param max_distance Maximal distance to center allowed to keep an event
    inline explicit ProximityFilterAlgorithm(const Eigen::Vector2f &center, float max_distance);

    /// @brief Default destructor
    ~ProximityFilterAlgorithm() = default;

    /// @brief Applies the Proximity filter to the given input buffer storing the result in the output buffer
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<class InputIt, class OutputIt>
    inline OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
        return Metavision::detail::insert_if(it_begin, it_end, inserter, std::ref(*this));
    }

    /// @brief Basic operator to check if an event is accepted
    /// @param ev Event2D to be tested
    inline bool operator()(const Event2d &ev) const;

    /// @brief Sets the center point of the filter
    /// @param center Point to be used in the filtering process
    inline void set_center(const Eigen::Vector2f &center);

    /// @brief Sets the maximal allowed distance of the filter
    /// @param max_distance Maximal distance to be used in the filtering process
    inline void set_distance(float max_distance);

    /// @brief Returns the center point used to filter the events
    /// @return Current reference point used in the filtering process
    inline Eigen::Vector2f center() const;

    /// @brief Returns the maximal distance used to filter the events
    /// @return Current maximal distance used in the filtering process
    inline float max_distance() const;

private:
    Eigen::Vector2f center_;
    float sqrd_max_dist_;
};

inline ProximityFilterAlgorithm::ProximityFilterAlgorithm(const Eigen::Vector2f &center, float max_distance) :
    center_(center), sqrd_max_dist_(max_distance * max_distance) {}

inline void ProximityFilterAlgorithm::set_center(const Eigen::Vector2f &center) {
    center_ = center;
}

inline void ProximityFilterAlgorithm::set_distance(float max_distance) {
    sqrd_max_dist_ = max_distance * max_distance;
}

inline Eigen::Vector2f ProximityFilterAlgorithm::center() const {
    return center_;
}

inline float ProximityFilterAlgorithm::max_distance() const {
    return std::sqrt(sqrd_max_dist_);
}

inline bool ProximityFilterAlgorithm::operator()(const Event2d &ev) const {
    return (Eigen::Vector2f{ev.x, ev.y} - center_).squaredNorm() < sqrd_max_dist_;
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_ALGORITHMS_PROXIMITY_FILTER_ALGORITHM_H
