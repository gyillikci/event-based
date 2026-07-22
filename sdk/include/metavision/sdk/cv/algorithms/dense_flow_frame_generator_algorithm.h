/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_DENSE_FLOW_FRAME_GENERATOR_ALGORITHM_H
#define METAVISION_SDK_CV_DENSE_FLOW_FRAME_GENERATOR_ALGORITHM_H

#include <vector>
#include <opencv2/core.hpp>

#include "metavision/sdk/cv/events/event_optical_flow.h"

namespace Metavision {

/// @brief Algorithm used to generate visualization images of dense optical flow streams.
class DenseFlowFrameGeneratorAlgorithm {
public:
    /// @brief Policy for accumulating multiple flow events at a given pixel
    enum class AccumulationPolicy {
        Average,       ///< Computes the average flow from the observations at the pixel
        PeakMagnitude, ///< Keeps the highest magnitude flow amongst the observations at the pixel
        Last           ///< Keeps the most recent flow amongst the observations at the pixel
    };

    enum class VisualizationMethod {
        DenseColorMap, ///< Visualization using a dense colormap where hue represents orientation, and value
                       ///< represents magnitude
        Arrows ///< Visualization using colored arrows where hue represents orientation, and value & length represent
               ///< magnitude
    };

    /// @brief Constructor
    /// @param width Input stream width
    /// @param height Input stream height
    /// @param max_flow_magnitude Maximum flow magnitude to display.
    /// @param flow_magnitude_scale Scale for flow magnitude in the visualization.
    /// @param visualization_method Method used to visualize flow field
    /// @param accumulation_policy Method used to accumulate multiple flow values at the same pixel.
    /// @param resolution_subsampling For Arrows representation, subsampling factor used to accumulate flow events. If
    /// negative, accumulation is done at resolution 1/8th. For DenseColorMap representation, this parameter is ignored
    /// and flow events are always accumulated in full resolution.
    DenseFlowFrameGeneratorAlgorithm(int width, int height, float max_flow_magnitude, float flow_magnitude_scale,
                                     VisualizationMethod visualization_method = VisualizationMethod::Arrows,
                                     AccumulationPolicy accumulation_policy   = AccumulationPolicy::Last,
                                     int resolution_subsampling               = -1);

    [[deprecated("This function is deprecated since version 4.4.0. Please use the other constructor "
                 "instead")]] DenseFlowFrameGeneratorAlgorithm(int width, int height, float maximum_flow_magnitude,
                                                               AccumulationPolicy accumulation_policy =
                                                                   AccumulationPolicy::Last);

    /// @brief Processes a buffer of flow events
    /// @tparam EventIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventOpticalFlow
    /// or equivalent
    /// @param it_begin Iterator to the first input event
    /// @param it_end Iterator to the past-the-end event
    /// @note Successive calls to process_events will accumulate data at each pixel until @ref generate or @ref reset
    /// is called.
    template<typename EventIt>
    void process_events(EventIt it_begin, EventIt it_end);

    /// @brief Generates a flow visualization frame
    /// @param frame Frame that will contain the flow visualization
    /// @param allocate Allocates the frame if true. Otherwise, the user must ensure the validity of the input frame.
    /// This is to be used when the data ptr must not change (external allocation, ROI over another cv::Mat, ...).
    /// @note In DenseColorMap mode, the frame will be reset to zero prior to being filled with the flow visualization.
    /// In Arrows mode, the flow visualization will be overlaid on top of the input frame.
    /// @throw invalid_argument if the frame doesn't have the expected type and geometry
    void generate(cv::Mat &frame, bool allocate = true);

    /// @brief Generates a legend image for the flow visualization
    /// @param legend_frame Frame that will contain the flow visualization legend
    /// @param square_size Size of the generated image
    /// @param allocate Allocates the frame if true. Otherwise, the user must ensure the validity of the input frame.
    /// This is to be used when the data ptr must not change (external allocation, ROI over another cv::Mat, ...)
    /// @throw invalid_argument if the frame doesn't have the expected type
    void generate_legend_image(cv::Mat &legend_frame, int square_size = 0, bool allocate = true);

    /// @brief Resets the internal states
    void reset();

private:
    void flow_map_to_color_map(const cv::Mat_<float> &vx, const cv::Mat_<float> &vy, cv::Mat &rgb);
    void flow_map_to_arrows(const cv::Mat_<float> &vx, const cv::Mat_<float> &vy, float flow_scale, cv::Mat &rgb);

    struct State {
        void reset();
        void accumulate(const EventOpticalFlow &ev_flow, AccumulationPolicy policy, float &vx, float &vy);
        int n = 0;
    };

    const int in_width_, in_height_;
    const float max_flow_magnitude_;
    const float flow_magnitude_scale_;
    const VisualizationMethod visualization_method_;
    const AccumulationPolicy accumulation_policy_;
    const int subsampling_;
    const int acc_width_, acc_height_;
    const int min_observations_;
    cv::Mat_<float> vx_, vy_, mag_, ang_;
    std::vector<State> states_;
};

} // namespace Metavision

#include "metavision/sdk/cv/algorithms/detail/dense_flow_frame_generator_algorithm_impl.h"

#endif // METAVISION_SDK_CV_DENSE_FLOW_FRAME_GENERATOR_ALGORITHM_H
