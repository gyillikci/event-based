/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_MODULATED_LIGHT_DETECTOR_ALGORITHM_H
#define METAVISION_SDK_CV_MODULATED_LIGHT_DETECTOR_ALGORITHM_H

#include <cstdint>
#include <vector>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief A class that detects modulated lights per pixel
///
/// Modulated light sources encode and transmit their ID by using frequency modulation: symbols can be encoded
/// using the time elapsed between two blinks. Every symbol duration is a multiple of a base period p:
/// - p : logical '0'
/// - 2p: logical '1'
/// - 3p: start bit
/// The size of a word (i.e. the number of bits used to encode the ID) is configurable and limited to 32 bits.
class ModulatedLightDetectorAlgorithm {
public:
    /// @brief Parameters for configuring the @ref ModulatedLightDetectorAlgorithm
    struct Params {
        std::uint16_t width          = 0;    ///< EB field of view's width in pixels
        std::uint16_t height         = 0;    ///< EB field of view's height in pixels
        std::uint8_t num_bits        = 8;    ///< Size of a word
        std::uint32_t base_period_us = 200;  ///< Base period for encoding bits
        float tolerance              = 0.1f; ///< Tolerance used for blink measurement
    };

    /// @brief Constructs a @ref ModulatedLightDetectorAlgorithm object with the specified parameters
    /// @param params The parameters for configuring the ModulatedLightDetectorAlgorithm
    /// @throw std::invalid_argument if the resolution is not valid or if the size of a word is greater than 32
    explicit ModulatedLightDetectorAlgorithm(const Params &params);

    /// @brief Tries to detect modulated light sources per pixel from events
    ///
    /// One @ref EventSourceId is produced per input @ref EventCD. The id field of an output event will be valid only if
    /// the corresponding input event allowed decoding a light ID.
    /// @tparam InputIt Input event iterator type
    /// @tparam OutputIt Output iterator type, works for iterators over containers of @ref EventSourceId
    /// @param[in] begin_it Iterator pointing to the first event in the stream
    /// @param[in] end_it Iterator pointing to the past-the-end element in the stream
    /// @param[out] output_it Iterator to the first source id event
    /// @return The iterator to the past-the-last source id event
    template<typename InputIt, typename OutputIt>
    OutputIt process_events(InputIt begin_it, InputIt end_it, OutputIt output_it);

private:
    const Params params_;
    std::vector<std::uint8_t> pixel_states_;   ///< Number of bits decoded per pixel
    std::vector<std::uint32_t> id_candidates_; ///< Contains the decoded bits per pixel
    std::vector<timestamp> last_blinks_;
};

} // namespace Metavision

#include "metavision/sdk/cv/algorithms/detail/modulated_light_detector_algorithm_impl.h"

#endif // METAVISION_SDK_CV_MODULATED_LIGHT_DETECTOR_ALGORITHM_H
