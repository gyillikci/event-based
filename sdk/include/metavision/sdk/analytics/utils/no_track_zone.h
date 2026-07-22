/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_NO_TRACK_ZONE_H
#define METAVISION_SDK_ANALYTICS_NO_TRACK_ZONE_H

#include <opencv2/core.hpp>

#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

struct NoTrackZone {
    NoTrackZone() : x(0), y(0), radius(0), filter_inside(true) {}
    NoTrackZone(float x, float y, int roi_radius, bool filter_inside) :
        x(x), y(y), radius(roi_radius), filter_inside(filter_inside) {}

    /// @brief Returns the center of the cluster
    /// @return The center of the cluster
    const cv::Point2f get_center() const {
        return cv::Point2f{x, y};
    }

    float x, y;
    int radius;         ///< Radius of the no-tracking zone
    bool filter_inside; ///< Indicates whether to filter events inside or outside the zone

    /// @brief Writes EventSpatterCluster to buffer
    void write_event(void *buf, timestamp) const {
        RawEvent *buffer      = (RawEvent *)buf;
        buffer->x             = x;
        buffer->y             = y;
        buffer->radius        = radius;
        buffer->filter_inside = filter_inside;
    }

    /// @brief Reads event 2D from buffer
    /// @return Event spatter cluster
    static NoTrackZone read_event(void *buf, const timestamp &) {
        RawEvent *buffer = static_cast<RawEvent *>(buf);
        return NoTrackZone(buffer->x, buffer->y, buffer->radius, buffer->filter_inside);
    }

    /// @brief Returns the size of the RawEvent
    /// @return The size of the RawEvent
    static size_t get_raw_event_size() {
        return sizeof(RawEvent);
    }

    /// @brief Operator <<
    friend std::ostream &operator<<(std::ostream &output, const NoTrackZone &e) {
        output << "NoZoneArea: (";
        output << e.x << ", " << e.y << ", " << e.radius << ", " << e.filter_inside;
        output << ")";
        return output;
    }

    FORCE_PACK(
        /// @brief Structure representing one event
        struct RawEvent {
            float x;
            float y;
            int radius;
            bool filter_inside;
        });
};

} // namespace Metavision

METAVISION_DEFINE_EVENT_TRAIT(Metavision::NoTrackZone, 255, "NoTrackZone")

#endif // METAVISION_SDK_ANALYTICS_NO_TRACK_ZONE_H
