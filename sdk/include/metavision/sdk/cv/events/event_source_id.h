/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_EVENT_SOURCE_ID_H
#define METAVISION_SDK_CV_EVENT_SOURCE_ID_H

#include <cstdint>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Event triggered by a source identified with a unique ID
struct EventSourceId {
    static constexpr std::uint32_t kInvalidId = static_cast<std::uint32_t>(-1);

    std::uint16_t x;  ///< Column position in the sensor at which the event happened
    std::uint16_t y;  ///< Row position in the sensor at which the event happened
    timestamp t;      ///< Timestamp at which the event happened (in us)
    std::uint32_t id; ///< ID of the source that generated the event
};
} // namespace Metavision

#endif // METAVISION_SDK_CV_EVENT_SOURCE_ID_H
