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

#ifndef OV_MSCKF_STATE_H
#define OV_MSCKF_STATE_H

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "StateOptions.h"
#include "cam/CamBase.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "types/Type.h"
#include "types/Vec.h"

namespace ov_msckf {

/**
 * @brief State of our filter
 *
 * This state has all the current estimates for the filter.
 * This system is modeled after the MSCKF filter, thus we have a sliding window of clones.
 * We additionally have more parameters for online estimation of calibration and SLAM features.
 * We also have the covariance of the system, which should be managed using the StateHelper class.
 */
class State {

public:
  /**
   * @brief Default Constructor (will initialize variables to defaults)
   * @param options_ Options structure containing filter options
   */
  State(const StateOptions &options_);

  ~State() {}

  /**
   * @brief Will return the timestep that we will marginalize next.
   * As of right now, since we are using a sliding window, this is the oldest clone.
   * But if you wanted to do a keyframe system, you could selectively marginalize clones.
   * @return timestep of clone we will marginalize
   */
  double margtimestep() const {
    double time = INFINITY;
    for (const auto &clone_imu : _clones_IMU) {
      if (clone_imu.first < time) {
        time = clone_imu.first;
      }
    }
    return time;
  }

  /**
   * @brief Calculates the current max size of the covariance
   * @return Size of the current covariance matrix
   */
  int max_covariance_size() const { return (int)_Cov.rows(); }


  std::shared_ptr<State> clone() const {
    auto clone = std::make_shared<State>(_options);
    clone->_timestamp = _timestamp;
    clone->_Cov = _Cov;

    // clone _cam_intrinsics_cameras
    clone->_cam_intrinsics_cameras.clear();
    for (auto& pair : _cam_intrinsics_cameras) {
      clone->_cam_intrinsics_cameras[pair.first] = pair.second->clone();
    }

    // clone _variables, and record the old_to_new map
    std::map<std::shared_ptr<ov_type::Type>, std::shared_ptr<ov_type::Type>> old_to_new;
    clone->_variables.clear();
    for (auto& variable : _variables) {
      auto new_variable = variable->clone();
      new_variable->set_local_id(variable->id());
      clone->_variables.push_back(new_variable);
      old_to_new[variable] = new_variable;
    }

    // _imu and _clones_IMU
    clone->_imu = std::dynamic_pointer_cast<ov_type::IMU>(old_to_new[_imu]);
    clone->_clones_IMU.clear();
    for (auto& pair : _clones_IMU) {
      clone->_clones_IMU[pair.first] = std::dynamic_pointer_cast<ov_type::PoseJPL>(old_to_new[pair.second]);
    }

    // _features_SLAM
    clone->_features_SLAM.clear();
    for (auto& pair : _features_SLAM) {
      clone->_features_SLAM[pair.first] = std::dynamic_pointer_cast<ov_type::Landmark>(old_to_new[pair.second]);
    }

    // _calib_dt_CAMtoIMU
    clone->_calib_dt_CAMtoIMU = std::dynamic_pointer_cast<ov_type::Vec>(old_to_new[_calib_dt_CAMtoIMU]);

    // _calib_IMUtoCAM
    clone->_calib_IMUtoCAM.clear();
    for (auto& pair : _calib_IMUtoCAM) {
      clone->_calib_IMUtoCAM[pair.first] = std::dynamic_pointer_cast<ov_type::PoseJPL>(old_to_new[pair.second]);
    }

    // _cam_intrinsics
    clone->_cam_intrinsics.clear();
    for (auto& pair : _cam_intrinsics) {
      clone->_cam_intrinsics[pair.first] = std::dynamic_pointer_cast<ov_type::Vec>(old_to_new[pair.second]);
    }

    return clone;
  }

  /// Current timestamp (should be the last update time!)
  double _timestamp = -1;

  /// Struct containing filter options
  StateOptions _options;

  /// Pointer to the "active" IMU state (q_GtoI, p_IinG, v_IinG, bg, ba)
  std::shared_ptr<ov_type::IMU> _imu;

  /// Map between imaging times and clone poses (q_GtoIi, p_IiinG)
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> _clones_IMU;

  /// Our current set of SLAM features (3d positions)
  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> _features_SLAM;

  /// Time offset base IMU to camera (t_imu = t_cam + t_off)
  std::shared_ptr<ov_type::Vec> _calib_dt_CAMtoIMU;

  /// Calibration poses for each camera (R_ItoC, p_IinC)
  std::unordered_map<size_t, std::shared_ptr<ov_type::PoseJPL>> _calib_IMUtoCAM;

  /// Camera intrinsics
  std::unordered_map<size_t, std::shared_ptr<ov_type::Vec>> _cam_intrinsics;

  /// Camera intrinsics camera objects
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> _cam_intrinsics_cameras;

private:
  // Define that the state helper is a friend class of this class
  // This will allow it to access the below functions which should normally not be called
  // This prevents a developer from thinking that the "insert clone" will actually correctly add it to the covariance
  friend class StateHelper;

  /// Covariance of all active variables
  Eigen::MatrixXd _Cov;

  /// Vector of variables
  std::vector<std::shared_ptr<ov_type::Type>> _variables;
};

} // namespace ov_msckf

#endif // OV_MSCKF_STATE_H