/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_EVT_FILTER_TYPE_H
#define METAVISION_SDK_ANALYTICS_EVT_FILTER_TYPE_H

#include <unordered_map>

namespace Metavision {

/// @brief Type of event polarity filter to implement
enum EvtFilterType {
    NO_FILTER,      ///< Keep both polarities, count all events
    FILTER_NEG,     ///< Filter negative events
    FILTER_POS,     ///< Filter positive events
    SEPARATE_FILTER ///< Keep both polarities, count positive and negative events separately
};

static std::unordered_map<std::string, EvtFilterType> const stringToFilterTypeMap = {
    {"NO_FILTER", EvtFilterType::NO_FILTER},
    {"FILTER_NEG", EvtFilterType::FILTER_NEG},
    {"FILTER_POS", EvtFilterType::FILTER_POS},
    {"SEPARATE_FILTER", EvtFilterType::SEPARATE_FILTER}};

static std::unordered_map<EvtFilterType, std::string> const filterTypeToStringMap = {
    {EvtFilterType::NO_FILTER, "NO_FILTER"},
    {EvtFilterType::FILTER_NEG, "FILTER_NEG"},
    {EvtFilterType::FILTER_POS, "FILTER_POS"},
    {EvtFilterType::SEPARATE_FILTER, "SEPARATE_FILTER"}};

} // namespace Metavision

namespace std {
std::istream &operator>>(std::istream &in, Metavision::EvtFilterType &filter_type);
std::ostream &operator<<(std::ostream &os, const Metavision::EvtFilterType &filter_type);
} // namespace std

#endif // METAVISION_SDK_ANALYTICS_EVT_FILTER_TYPE_H
