/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV3D_ACTIVE_MARKER_POSE_ESTIMATOR_IMPL_H
#define METAVISION_SDK_CV3D_ACTIVE_MARKER_POSE_ESTIMATOR_IMPL_H

namespace Metavision {
template<typename InputIt>
void Metavision::ActiveMarkerPoseEstimatorAlgorithm::process_events(InputIt begin, InputIt end) {
    active_tracks_.clear();
    tracker_.process_events(begin, end, std::back_inserter(active_tracks_));
    process_active_tracks();
}
} // namespace Metavision

#endif // METAVISION_SDK_CV3D_ACTIVE_MARKER_POSE_ESTIMATOR_IMPL_H
