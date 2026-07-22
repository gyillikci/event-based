/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_ACTIVE_MARKER_TRACKER_ALGORITHM_H
#define METAVISION_SDK_CV_ACTIVE_MARKER_TRACKER_ALGORITHM_H

#include <set>
#include <optional>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/cv/events/event_source_id.h"
#include "metavision/sdk/cv/events/event_active_track.h"
#include "metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h"

namespace Metavision {

/// @brief A class that tracks an active marker in 2D
///
/// An active marker is a rigid object composed of several blinking LEDs. It is tracked by tracking its LEDs based on
/// the events they generate (i.e. @ref EventSourceId), the latter of which form blobs in the sensor array. The blobs
/// are not directly detected and tracked. Instead, events are associated to tracks using a radius criterion. A
/// monitoring process is then regularly executed to:
/// - re-assess this radius criterion (i.e. proportionally to the lowest distance between the tracks)
/// - remove the tracks that are no longer tracked (i.e. based on an inactivity period, meaning with no update)
/// The first time an event with a valid ID that doesn't match any existing track is received, a new one is created.
class ActiveMarkerTrackerAlgorithm {
public:
    /// @brief Parameters for configuring the @ref ActiveMarkerTrackerAlgorithm
    struct Params {
        using Duration = timestamp;

        bool update_radius = true; ///< Flag indicating whether to update the radius during the monitoring process
        Duration inactivity_period_us = 10000; ///< Inactivity period above which a track is considered as lost
        float monitoring_frequency_hz = 30.f;  ///< Frequency (in camera time) at which the monitoring process is
                                               /// executed
        float radius       = 30.f;             ///< Influence radius used to associate events to tracks
        float distance_pct = 0.3f; ///< Percentage on the lowest distance between two tracks used to update the radius
                                   /// criterion
        float alpha_pos = 0.01f;   ///< Influence of a new associated event on the position update of a track
    };

    /// @brief Constructor
    /// @param params The configuration parameters
    /// @param sources_to_track List of unique IDs of the LEDs that constitute the active marker to be tracked
    ActiveMarkerTrackerAlgorithm(const Params &params, const std::set<std::uint32_t> &sources_to_track);

    /// @brief Tracks the 2D position of the active marker's LEDs from @ref EventSourceId events
    /// @tparam InputIt Input event iterator type, works for iterators over containers of @ref EventSourceId
    /// @tparam OutputIt Output iterator type, works for iterators over containers of @ref EventActiveTrack
    /// @param[in] begin Iterator pointing to the first event in the stream
    /// @param[in] end Iterator pointing to the past-the-end element in the stream
    /// @param[out] inserter Back inserter
    template<typename InputIt, typename OutputIt>
    void process_events(InputIt begin, InputIt end, OutputIt inserter);

    /// @brief Notifies the tracker that time has elapsed without new events, which may trigger several calls
    /// to the internal monitoring process
    /// @param ts Current timestamp
    void notify_elapsed_time(timestamp ts);

private:
    struct ActiveTrack {
        std::uint32_t id;
        float x;
        float y;
        timestamp last_update;
    };

    std::optional<size_t> try_find_closest_track(const EventSourceId &id_ev);
    EventActiveTrack update_track(ActiveTrack &track, const EventSourceId &id_ev);
    std::optional<EventActiveTrack> try_start_new_track(const EventSourceId &id_ev);
    void do_monitoring(timestamp t);
    void remove_inactive_tracks(timestamp t);
    void estimate_influence_radius();

    template<typename InputIt, typename OutputIt>
    void process_events_internal(InputIt begin, InputIt end, OutputIt inserter);

    Params params_;
    float radius2_;
    EventBufferReslicerAlgorithm slicer_;
    std::vector<ActiveTrack> tracks_;
    std::vector<std::uint32_t> sources_to_track_;
    std::vector<EventActiveTrack> lost_track_events_;
};

} // namespace Metavision

#include "metavision/sdk/cv/algorithms/detail/active_marker_tracker_algorithm_impl.h"

#endif // METAVISION_SDK_CV_ACTIVE_MARKER_TRACKER_ALGORITHM_H
