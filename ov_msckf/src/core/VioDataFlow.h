/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2022 Patrick Geneva
 * Copyright (C) 2018-2022 Guoquan Huang
 * Copyright (C) 2018-2022 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_VIODATAFLOW_H
#define OV_MSCKF_VIODATAFLOW_H

#include <Eigen/StdVector>
#include <algorithm>
#include <atomic>
#include <boost/filesystem.hpp>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "VioManagerOptions.h"

namespace ov_core {
struct ImuData;
struct CameraData;
} // namespace ov_core

namespace ov_msckf {

class VioManager;

class VioDataFlow {

public:
  VioDataFlow(VioManagerOptions &params_);

  // void feed_measurement_imu(const ov_core::ImuData &message);

  // void feed_measurement_camera(const ov_core::CameraData &message) { track_image_and_update(message); }

  // void feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
  //                                  const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats);
};

} // namespace ov_msckf

#endif // OV_MSCKF_VIODATAFLOW_H
