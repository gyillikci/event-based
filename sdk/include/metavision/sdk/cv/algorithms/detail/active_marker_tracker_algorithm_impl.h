/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_ACTIVE_MARKER_TRACKER_IMPL_H
#define METAVISION_SDK_CV_ACTIVE_MARKER_TRACKER_IMPL_H

#include "metavision/sdk/cv/events/event_active_track.h"

namespace Metavision {
template<typename InputIt, typename OutputIt>
void ActiveMarkerTrackerAlgorithm::process_events(InputIt begin, InputIt end, OutputIt inserter) {
    slicer_.process_events(
        begin, end, [&](const auto &cbegin, const auto &cend) { process_events_internal(cbegin, cend, inserter); });
}

template<typename InputIt, typename OutputIt>
void ActiveMarkerTrackerAlgorithm::process_events_internal(InputIt begin, InputIt end, OutputIt inserter) {
    for (const auto &lost_track : lost_track_events_) {
        *inserter = lost_track;
        ++inserter;
    }

    lost_track_events_.clear();

    for (auto it = begin; it != end; ++it) {
        if (const auto track_idx = try_find_closest_track(*it); track_idx) {
            *inserter = update_track(tracks_[*track_idx], *it);
            ++inserter;
        } else if (it->id != EventSourceId::kInvalidId) {
            if (const auto new_track = try_start_new_track(*it); new_track) {
                *inserter = *new_track;
                ++inserter;
            }
        }
    }
}

} // namespace Metavision

#endif // METAVISION_SDK_CV_ACTIVE_MARKER_TRACKER_IMPL_H
