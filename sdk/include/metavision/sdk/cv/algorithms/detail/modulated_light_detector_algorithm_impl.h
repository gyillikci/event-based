/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_MODULATED_LIGHT_DETECTOR_ALGORITHM_IMPL_H
#define METAVISION_SDK_CV_MODULATED_LIGHT_DETECTOR_ALGORITHM_IMPL_H

#include "metavision/sdk/cv/events/event_source_id.h"

namespace Metavision {
template<typename InputIt, typename OutputIt>
OutputIt ModulatedLightDetectorAlgorithm::process_events(InputIt begin_it, InputIt end_it, OutputIt output_it) {
    const auto tolerance = static_cast<std::uint32_t>(params_.base_period_us * params_.tolerance);

    for (auto it = begin_it; it != end_it; ++it) {
        if (it->p == 0)
            continue;

        EventSourceId id_ev{it->x, it->y, it->t, EventSourceId::kInvalidId};

        const auto ev_idx   = it->y * params_.width + it->x;
        auto &last_blink_ts = last_blinks_[ev_idx];
        const auto dt       = static_cast<std::uint32_t>(it->t - last_blink_ts);

        if (dt > params_.base_period_us) {
            last_blink_ts = it->t;

            // we measure the time between two positive events (i.e. two rising edges) meaning that there is one
            // additional period to take into account
            if (dt > 2 * params_.base_period_us - tolerance && dt < 2 * params_.base_period_us + tolerance) {
                //'0'
                pixel_states_[ev_idx] += 1;
            } else if (dt > 3 * params_.base_period_us - tolerance && dt < 3 * params_.base_period_us + tolerance) {
                // '1'
                id_candidates_[ev_idx] += (1 << pixel_states_[ev_idx]++);
            } else if (dt > 4 * params_.base_period_us - tolerance && dt < 4 * params_.base_period_us + tolerance) {
                // start
                if (pixel_states_[ev_idx] == params_.num_bits) {
                    id_ev.id = id_candidates_[ev_idx];
                }
                pixel_states_[ev_idx]  = 0;
                id_candidates_[ev_idx] = 0;
            } else {
                pixel_states_[ev_idx]  = 0;
                id_candidates_[ev_idx] = 0;
            }
        }

        *output_it = id_ev;
        ++output_it;
    }

    return output_it;
}
} // namespace Metavision

#endif // METAVISION_SDK_CV_MODULATED_LIGHT_DETECTOR_ALGORITHM_IMPL_H
