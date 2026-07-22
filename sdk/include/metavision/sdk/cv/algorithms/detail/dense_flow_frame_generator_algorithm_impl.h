/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

namespace Metavision {

template<typename EventIt>
void DenseFlowFrameGeneratorAlgorithm::process_events(EventIt it_begin, EventIt it_end) {
    for (auto it = it_begin; it != it_end; ++it) {
        const int accx = it->x >> subsampling_;
        const int accy = it->y >> subsampling_;
        auto &px       = states_[accy * acc_width_ + accx];
        px.accumulate(*it, accumulation_policy_, vx_.at<float>(accy, accx), vy_.at<float>(accy, accx));
    }
}

} // namespace Metavision
