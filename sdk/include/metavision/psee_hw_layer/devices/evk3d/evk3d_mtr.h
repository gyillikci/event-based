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

#ifndef METAVISION_HAL_PSEE_PLUGINS_DEVICES_EVK3D_MTR_H
#define METAVISION_HAL_PSEE_PLUGINS_DEVICES_EVK3D_MTR_H

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "metavision/hal/facilities/i_registrable_facility.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

class EVK3DMTR : public I_RegistrableFacility<EVK3DMTR> {
public:
    EVK3DMTR(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix);

    virtual std::set<std::string> get_calibration_keys() const;
    virtual bool load_calibration(const std::map<std::string, std::vector<std::vector<float>>> &calibration_data);

    virtual bool set_x_inf_data(uint32_t ch_id, uint32_t id, uint32_t y, uint32_t x_inf);
    virtual uint32_t get_x_inf_data(uint32_t ch_id, uint32_t id, uint32_t y) const;

    virtual bool set_bf_positive(bool state);
    virtual bool get_bf_positive() const;

    virtual bool set_bf_data(uint32_t ch_id, uint32_t id, uint32_t y, uint32_t bf);
    virtual uint32_t get_bf_data(uint32_t ch_id, uint32_t id, uint32_t y) const;

    virtual bool has_undistortion() const;
    virtual bool load_undistortion_calibration(const std::map<std::string, std::vector<float>> &calibration_data);

    virtual bool set_kinv_coeff(uint32_t line, uint32_t column, float val);
    virtual float get_kinv_coeff(uint32_t line, uint32_t column) const;

    virtual bool set_dist_coeff(uint32_t idx, float val);
    virtual float get_dist_coeff(uint32_t idx) const;

    virtual bool set_x_scale_factor(float scale_factor);
    virtual float get_x_scale_factor() const;
    virtual bool set_y_scale_factor(float scale_factor);
    virtual float get_y_scale_factor() const;
    virtual bool set_z_scale_factor(float scale_factor);
    virtual float get_z_scale_factor() const;

private:
    std::shared_ptr<RegisterMap> register_map_;
    const std::string prefix_;

    const int LUT_READ_WAIT          = 5;
    const unsigned int SENSOR_HEIGHT = 720;
    const unsigned int NUM_CHANNELS  = 8;
    const unsigned int NUM_LINES     = 6;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_DEVICES_EVK3D_MTR_H
