/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_TIME_GRADIENT_FLOW_ALGORITHM_H
#define METAVISION_SDK_CV_TIME_GRADIENT_FLOW_ALGORITHM_H

#include "metavision/sdk/core/utils/mostrecent_timestamp_buffer.h"
#include "metavision/sdk/cv/configs/time_gradient_flow_algorithm_config.h"

namespace Metavision {

/// @brief This class is a local and dense implementation of Optical Flow from events
///
/// It computes the optical flow along the edge's normal by analyzing the recent timestamps at only the left, right,
/// top and down K-pixel far neighbors (i.e. not the whole neighborhood). Thus, the estimated flow results are still
/// quite sensitive to noise. The algorithm is run for each input event, generating a dense stream of flow events, but
/// making it relatively costly on high event-rate scenes. The bit size of the timestamp representation can be reduced
/// to accelerate the processing.
/// @note This approach is dense in the sense that it processes events at the sensor resolution and produces
/// OpticalFlowEvents potentially on the whole sensor matrix.
/// @tparam timestamp_type Type of the timestamp used in to compute the optical flow. Typically @ref
/// Metavision::timestamp. Can be used with lighter type (like std::uint32_t) to lower processing time when critical.
template<typename timestamp_type>
class TimeGradientFlowAlgorithmT {
public:
    /// @brief Constructor
    /// @param input_width Maximum width of input events
    /// @param input_height Maximum height of input events
    /// @param config Configuration for the algorithm
    TimeGradientFlowAlgorithmT(int input_width, int input_height, const TimeGradientFlowAlgorithmConfig &config);

    /// @brief Processes each input event to estimate the optical flow from them into the output buffer
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of EventOpticalFlow
    /// or equivalent
    /// @param[in] it_begin Iterator to first input event
    /// @param[in] it_end Iterator to the past-the-end event
    /// @param[out] inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<typename InputIt, typename OutputIt>
    OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter);

private:
    /// @brief Timestamp reduction for optimization. Original timestamp is 64 bits, corresponding to a range of
    /// 5124095576 hours It can be reduced to fewer bits. The bit_cut_ lowest bits are ignored, as well as the useless
    /// higher bits, which results in a lower timerange available for timestamps. Time looping/overflow is handled in
    /// the process_events method.
    /// @param t The input timestamp to degrade
    /// @returns The lower resolution bit-cut timestamp
    inline timestamp_type get_lower_resolution_ts(Metavision::timestamp t);

    // Input parameters
    const int width_, height_; ///< Maximal dimensions of input events
    const float inv_squared_min_norm_;
    const int flow_radius_;
    const float radius_inv_;

    // Timestamp management
    const uint8_t bit_cut_;
    long long timestamp_mask_;

    // Recent timestamps memory
    MostRecentTimestampBufferT<timestamp_type> last_ts_; ///< Most recent timestamp matrix

    // Temporary variables
    timestamp_type last_seen_ts_ = 0; ///< Last seen timestamp, used to detect time looping
    float factor_ts_;
};

/// @brief Instantiation of the @ref TimeGradientFlowAlgorithmT class using 32 bits unsigned integers
using TimeGradientFlowAlgorithm = TimeGradientFlowAlgorithmT<std::uint32_t>;

} // namespace Metavision

#endif // METAVISION_SDK_CV_TIME_GRADIENT_FLOW_ALGORITHM_H
