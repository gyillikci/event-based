/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <cstdint>

#ifndef METAVISION_SDK_CV_TIME_GRADIENT_FLOW_ALGORITHM_CONFIG_H
#define METAVISION_SDK_CV_TIME_GRADIENT_FLOW_ALGORITHM_CONFIG_H

namespace Metavision {

/// @brief Structure representing the configuration of the time gradient flow algorithm.
struct TimeGradientFlowAlgorithmConfig {
    /// @brief Initializing constructor
    /// @param radius Spatial radius for flow analysis
    /// @param min_flow_mag Minimum flow magnitude to be observed
    /// @param bit_cut Number of bits to remove to compare timestamps (used to accelerate processing)
    TimeGradientFlowAlgorithmConfig(uint8_t radius = 3, float min_flow_mag = 0.f, uint8_t bit_cut = 0) :
        local_radius(radius), min_norm(min_flow_mag), bit_cut(bit_cut) {}

    uint8_t local_radius; ///< Matching spatial search radius. The event received will provide the center of the
                          ///< analyzed neighborhood. The pixels at the local_radius distance on the left, right, top
                          ///< and bottom will be used to compute optical flow, not the pixels
                          /// in-between. The further the distance, the more the flow estimate is regularized but also
                          /// the more subject to association error.
    float min_norm;       ///< Minimal flow norm estimated to produce an opticalFlowEvent (the actual value used
                          ///< is 1.f/(min_norm*min_norm) to get a value in pix/s)
    uint8_t bit_cut;      ///< Number of bits to remove to compare timestamps (used to accelerate processing)
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_TIME_GRADIENT_FLOW_ALGORITHM_CONFIG_H
