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

#include "State.h"
#include "utils/colors.h"
#include "utils/print.h"

#include <sstream>
#include <glog/logging.h>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

State::State(const StateOptions &options) {

  // Save our options
  _options = options;

  // Append the imu to the state and covariance
  int current_id = 0;
  _imu = std::make_shared<IMU>();
  _imu->set_local_id(current_id);
  _variables.push_back(_imu);
  current_id += _imu->size();

  // Camera to IMU time offset
  _calib_dt_CAMtoIMU = std::make_shared<Vec>(1);
  if (_options.do_calib_camera_timeoffset) {
    _calib_dt_CAMtoIMU->set_local_id(current_id);
    _variables.push_back(_calib_dt_CAMtoIMU);
    current_id += _calib_dt_CAMtoIMU->size();
  }

  // Loop through each camera and create extrinsic and intrinsics
  for (int i = 0; i < _options.num_cameras; i++) {

    // Allocate extrinsic transform
    auto pose = std::make_shared<PoseJPL>();

    // Allocate intrinsics for this camera
    auto intrin = std::make_shared<Vec>(8);

    // Add these to the corresponding maps
    _calib_IMUtoCAM.insert({i, pose});
    _cam_intrinsics.insert({i, intrin});

    // If calibrating camera-imu pose, add to variables
    if (_options.do_calib_camera_pose) {
      pose->set_local_id(current_id);
      _variables.push_back(pose);
      current_id += pose->size();
    }

    // If calibrating camera intrinsics, add to variables
    if (_options.do_calib_camera_intrinsics) {
      intrin->set_local_id(current_id);
      _variables.push_back(intrin);
      current_id += intrin->size();
    }
  }

  if (_options.use_rgbd) {
    assert(_options.num_cameras == 1);
    int i = _options.num_cameras;

    // Allocate extrinsic transform
    auto pose = std::make_shared<PoseJPL>();

    // Allocate intrinsics for this camera
    auto intrin = std::make_shared<Vec>(8);

    // Add these to the corresponding maps
    _calib_IMUtoCAM.insert({i, pose});
    _cam_intrinsics.insert({i, intrin});

    // We don't calibrate camera-imu pose or intrinsics for the virtual right camera.
    // So we don't need to set_local_id for pose and intrin and won't add them to _variables.
  }

  // Finally initialize our covariance to small value
  _Cov = std::pow(1e-3, 2) * Eigen::MatrixXd::Identity(current_id, current_id);

  // Finally, set some of our priors for our calibration parameters
  if (_options.do_calib_camera_timeoffset) {
    _Cov(_calib_dt_CAMtoIMU->id(), _calib_dt_CAMtoIMU->id()) = std::pow(0.01, 2);
  }
  if (_options.do_calib_camera_pose) {
    for (int i = 0; i < _options.num_cameras; i++) {
      _Cov.block(_calib_IMUtoCAM.at(i)->id(), _calib_IMUtoCAM.at(i)->id(), 3, 3) = std::pow(0.005, 2) * Eigen::MatrixXd::Identity(3, 3);
      _Cov.block(_calib_IMUtoCAM.at(i)->id() + 3, _calib_IMUtoCAM.at(i)->id() + 3, 3, 3) =
          std::pow(0.01, 2) * Eigen::MatrixXd::Identity(3, 3);
    }
  }
  if (_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < _options.num_cameras; i++) {
      _Cov.block(_cam_intrinsics.at(i)->id(), _cam_intrinsics.at(i)->id(), 4, 4) = std::pow(1.0, 2) * Eigen::MatrixXd::Identity(4, 4);
      _Cov.block(_cam_intrinsics.at(i)->id() + 4, _cam_intrinsics.at(i)->id() + 4, 4, 4) =
          std::pow(0.005, 2) * Eigen::MatrixXd::Identity(4, 4);
    }
  }
}

std::shared_ptr<State> State::clone(bool print_variable_types) const {
  auto clone = std::make_shared<State>(_options);
  clone->_timestamp = _timestamp;
  clone->_Cov = _Cov;

  // clone _cam_intrinsics_cameras
  clone->_cam_intrinsics_cameras.clear();
  for (auto& pair : _cam_intrinsics_cameras) {
    clone->_cam_intrinsics_cameras[pair.first] = pair.second->clone();
  }

  // clone _T_CtoIs
  // std::map<size_t, std::shared_ptr<Eigen::Matrix4d>> _T_CtoIs;
  clone->_T_CtoIs.clear();
  for (auto& pair : _T_CtoIs) {
    clone->_T_CtoIs[pair.first] = std::make_shared<Eigen::Matrix4d>(*pair.second);
  }

  // clone _variables, and record the old_to_new map
  std::map<std::shared_ptr<ov_type::Type>, std::shared_ptr<ov_type::Type>> old_to_new;
  clone->_variables.clear();

  std::ostringstream oss;
  size_t ivar = 0;
  size_t n_landmark = 0;
  if (print_variable_types) {
    oss << "Variable types: ";
  }
  for (auto& variable : _variables) {
    auto new_variable = variable->clone();
    new_variable->set_local_id(variable->id());
    clone->_variables.push_back(new_variable);
    old_to_new[variable] = new_variable;

    if (print_variable_types) {
      if (std::dynamic_pointer_cast<IMU>(variable)) {
        oss << ivar << "(IMU) ";
      } else if (std::dynamic_pointer_cast<PoseJPL>(variable)) {
        oss << ivar << "(PoseJPL) ";
      } else if (std::dynamic_pointer_cast<Landmark>(variable)) {
        oss << ivar << "(Landmark) ";
        n_landmark ++;
      } else if (std::dynamic_pointer_cast<Vec>(variable)) {
        oss << ivar << "(Vec) ";
      } else {
        oss << ivar << "(Unknown) ";
      }
      ivar++;
    }

  }
  if (print_variable_types) {
    oss << ",  _features_SLAM.size() = " << _features_SLAM.size()
        << ", n_landmark = " << n_landmark << std::endl;
    PRINT_INFO("%s", oss.str().c_str());
    CHECK_EQ(n_landmark, _features_SLAM.size());
  }

  // _imu and _clones_IMU
  clone->_imu = std::dynamic_pointer_cast<ov_type::IMU>(old_to_new.at(_imu));
  clone->_clones_IMU.clear();
  for (auto& pair : _clones_IMU) {
    clone->_clones_IMU[pair.first] = std::dynamic_pointer_cast<ov_type::PoseJPL>(old_to_new.at(pair.second));
  }

  // _features_SLAM
  clone->_features_SLAM.clear();
  for (const auto& pair : _features_SLAM) {
    size_t n_marginalized = 0;
    size_t n_null = 0;
    if(old_to_new.count(pair.second)) {
      clone->_features_SLAM[pair.first] = std::dynamic_pointer_cast<ov_type::Landmark>(old_to_new.at(pair.second));
    } else {
      if (pair.second) {
        n_marginalized ++;
        clone->_features_SLAM[pair.first] = std::dynamic_pointer_cast<ov_type::Landmark>(pair.second->clone());
        CHECK(clone->_features_SLAM.at(pair.first));
      } else {
        n_null ++;
        PRINT_WARNING(YELLOW "State::clone(): Might be a bug? get null landmark for feature id %d\n" RESET, pair.first);
        clone->_features_SLAM[pair.first] = nullptr;
      }
    }

    if (n_marginalized > 0 || n_null > 0) {
      // Some slam features have already been marginalized but still stay in _features_SLAM?
      PRINT_WARNING(YELLOW "State::clone(): Might be a bug? Some slam features are null (%d) or have already been "
                            "marginalized but still stay in _features_SLAM (%d).\n" RESET, n_null, n_marginalized)
    }
  }

  // _calib_dt_CAMtoIMU
  if (old_to_new.count(_calib_dt_CAMtoIMU)) {
    clone->_calib_dt_CAMtoIMU = std::dynamic_pointer_cast<ov_type::Vec>(old_to_new.at(_calib_dt_CAMtoIMU));
  } else {
    clone->_calib_dt_CAMtoIMU = std::dynamic_pointer_cast<ov_type::Vec>(_calib_dt_CAMtoIMU->clone());
  }

  // _calib_IMUtoCAM
  clone->_calib_IMUtoCAM.clear();
  for (auto& pair : _calib_IMUtoCAM) {
    if (old_to_new.count(pair.second)) {
      clone->_calib_IMUtoCAM[pair.first] = std::dynamic_pointer_cast<ov_type::PoseJPL>(old_to_new.at(pair.second));
    } else {
      clone->_calib_IMUtoCAM[pair.first] = std::dynamic_pointer_cast<ov_type::PoseJPL>(pair.second->clone());
    }
  }

  // _cam_intrinsics
  clone->_cam_intrinsics.clear();
  for (auto& pair : _cam_intrinsics) {
    if (old_to_new.count(pair.second)) {
      clone->_cam_intrinsics[pair.first] = std::dynamic_pointer_cast<ov_type::Vec>(old_to_new.at(pair.second));
    } else {
      clone->_cam_intrinsics[pair.first] = std::dynamic_pointer_cast<ov_type::Vec>(pair.second->clone());
    }
  }

  return clone;
}
