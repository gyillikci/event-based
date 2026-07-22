/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#ifndef METAVISION_HAL_PSEE_PLUGINS_DEVICES_EVK3D_PROJECTOR_H
#define METAVISION_HAL_PSEE_PLUGINS_DEVICES_EVK3D_PROJECTOR_H

#include <memory>
#include <string>
#include <vector>

#include "metavision/hal/facilities/i_registrable_facility.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

class EVK3DProjector : public I_RegistrableFacility<EVK3DProjector> {
public:
    EVK3DProjector(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix);

    virtual bool enable(bool b);
    virtual bool is_enabled() const;

    virtual uint32_t get_number_channels() const {
        return 8;
    }

    virtual bool set_pattern_sequence(const std::vector<bool> &channels);
    virtual std::vector<bool> get_pattern_sequence() const;
    virtual bool enable_channel(uint32_t chan, bool b);
    virtual bool channel_is_enabled(uint32_t chan) const;

    virtual bool set_channel_period(unsigned int period_us);
    virtual unsigned int get_channel_period() const;
    virtual unsigned int get_min_channel_period() const;
    virtual unsigned int get_max_channel_period() const;

    virtual bool set_pulse_level(unsigned int level);
    virtual unsigned int get_pulse_level() const;
    virtual unsigned int get_max_pulse_level() const;

    virtual bool set_power(unsigned int power);

private:
    std::shared_ptr<RegisterMap> register_map_;
    const std::string prefix_;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_DEVICES_EVK3D_PROJECTOR_H
