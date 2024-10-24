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

#include "VisualizerHelper.h"

#include "core/VioManager.h"
#include "sim/Simulator.h"
#include "state/State.h"
#include "state/StateHelper.h"

#include "types/PoseJPL.h"

using namespace ov_msckf;
using namespace std;


void VisualizerHelper::init_total_state_files(
    const std::string& filepath_est, const std::string& filepath_std, const std::string& filepath_gt,
    std::shared_ptr<Simulator> sim, 
    std::ofstream &of_state_est, std::ofstream &of_state_std, std::ofstream &of_state_gt) {

  // If it exists, then delete it
  if (std::filesystem::exists(filepath_est))
    std::filesystem::remove(filepath_est);
  if (std::filesystem::exists(filepath_std))
    std::filesystem::remove(filepath_std);

  // Create folder path to this location if not exists
  auto filepath_est_parent = std::filesystem::path(filepath_est.c_str()).parent_path();
  auto filepath_std_parent = std::filesystem::path(filepath_std.c_str()).parent_path();
  if (!filepath_est_parent.empty()) {
    std::filesystem::create_directories(filepath_est_parent);
  }
  if (!filepath_std_parent.empty()) {
    std::filesystem::create_directories(filepath_std_parent);
  }

  // Open the files
  of_state_est.open(filepath_est.c_str());
  of_state_std.open(filepath_std.c_str());
  of_state_est << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;
  of_state_std << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;

  // Groundtruth if we are simulating
  if (sim != nullptr) {
    if (std::filesystem::exists(filepath_gt))
      std::filesystem::remove(filepath_gt);
    auto filepath_gt_parent = std::filesystem::path(filepath_gt.c_str()).parent_path();
    if (!filepath_gt_parent.empty()) {
      std::filesystem::create_directories(filepath_gt_parent);
    }
    of_state_gt.open(filepath_gt.c_str());
    of_state_gt << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;
  }
}

void VisualizerHelper::sim_save_total_state_to_file(std::shared_ptr<State> state, std::shared_ptr<Simulator> sim,
                                                       std::ofstream &of_state_est, std::ofstream &of_state_std,
                                                       std::ofstream &of_state_gt) {

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = state->_timestamp + t_ItoC;

  // If we have our simulator, then save it to our groundtruth file
  if (sim != nullptr) {

    // Note that we get the true time in the IMU clock frame
    // NOTE: we record both the estimate and groundtruth with the same "true" timestamp if we are doing simulation
    Eigen::Matrix<double, 17, 1> state_gt;
    timestamp_inI = state->_timestamp + sim->get_true_parameters().calib_camimu_dt;
    if (sim->get_state(timestamp_inI, state_gt)) {
      // STATE: write current true state
      of_state_gt.precision(5);
      of_state_gt.setf(std::ios::fixed, std::ios::floatfield);
      of_state_gt << state_gt(0) << " ";
      of_state_gt.precision(6);
      of_state_gt << state_gt(1) << " " << state_gt(2) << " " << state_gt(3) << " " << state_gt(4) << " ";
      of_state_gt << state_gt(5) << " " << state_gt(6) << " " << state_gt(7) << " ";
      of_state_gt << state_gt(8) << " " << state_gt(9) << " " << state_gt(10) << " ";
      of_state_gt << state_gt(11) << " " << state_gt(12) << " " << state_gt(13) << " ";
      of_state_gt << state_gt(14) << " " << state_gt(15) << " " << state_gt(16) << " ";

      // TIMEOFF: Get the current true time offset
      of_state_gt.precision(7);
      of_state_gt << sim->get_true_parameters().calib_camimu_dt << " ";
      of_state_gt.precision(0);
      of_state_gt << state->_options.num_cameras << " ";
      of_state_gt.precision(6);

      // CALIBRATION: Write the camera values to file
      assert(state->_options.num_cameras == sim->get_true_parameters().state_options.num_cameras);
      for (int i = 0; i < state->_options.num_cameras; i++) {
        // Intrinsics values
        of_state_gt << sim->get_true_parameters().camera_intrinsics.at(i)->get_value()(0) << " "
                    << sim->get_true_parameters().camera_intrinsics.at(i)->get_value()(1) << " "
                    << sim->get_true_parameters().camera_intrinsics.at(i)->get_value()(2) << " "
                    << sim->get_true_parameters().camera_intrinsics.at(i)->get_value()(3) << " ";
        of_state_gt << sim->get_true_parameters().camera_intrinsics.at(i)->get_value()(4) << " "
                    << sim->get_true_parameters().camera_intrinsics.at(i)->get_value()(5) << " "
                    << sim->get_true_parameters().camera_intrinsics.at(i)->get_value()(6) << " "
                    << sim->get_true_parameters().camera_intrinsics.at(i)->get_value()(7) << " ";
        // Rotation and position
        of_state_gt << sim->get_true_parameters().camera_extrinsics.at(i)(0) << " " << sim->get_true_parameters().camera_extrinsics.at(i)(1)
                    << " " << sim->get_true_parameters().camera_extrinsics.at(i)(2) << " "
                    << sim->get_true_parameters().camera_extrinsics.at(i)(3) << " ";
        of_state_gt << sim->get_true_parameters().camera_extrinsics.at(i)(4) << " " << sim->get_true_parameters().camera_extrinsics.at(i)(5)
                    << " " << sim->get_true_parameters().camera_extrinsics.at(i)(6) << " ";
      }

      // New line
      of_state_gt << endl;
    }
  }

  //==========================================================================
  //==========================================================================
  //==========================================================================

  // Get the covariance of the whole system
  Eigen::MatrixXd cov = StateHelper::get_full_covariance(state);

  // STATE: Write the current state to file
  of_state_est.precision(5);
  of_state_est.setf(std::ios::fixed, std::ios::floatfield);
  of_state_est << timestamp_inI << " ";
  of_state_est.precision(6);
  of_state_est << state->_imu->quat()(0) << " " << state->_imu->quat()(1) << " " << state->_imu->quat()(2) << " " << state->_imu->quat()(3)
               << " ";
  of_state_est << state->_imu->pos()(0) << " " << state->_imu->pos()(1) << " " << state->_imu->pos()(2) << " ";
  of_state_est << state->_imu->vel()(0) << " " << state->_imu->vel()(1) << " " << state->_imu->vel()(2) << " ";
  of_state_est << state->_imu->bias_g()(0) << " " << state->_imu->bias_g()(1) << " " << state->_imu->bias_g()(2) << " ";
  of_state_est << state->_imu->bias_a()(0) << " " << state->_imu->bias_a()(1) << " " << state->_imu->bias_a()(2) << " ";

  // STATE: Write current uncertainty to file
  of_state_std.precision(5);
  of_state_std.setf(std::ios::fixed, std::ios::floatfield);
  of_state_std << timestamp_inI << " ";
  of_state_std.precision(6);
  int id = state->_imu->q()->id();
  of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
  id = state->_imu->p()->id();
  of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
  id = state->_imu->v()->id();
  of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
  id = state->_imu->bg()->id();
  of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";
  id = state->_imu->ba()->id();
  of_state_std << std::sqrt(cov(id + 0, id + 0)) << " " << std::sqrt(cov(id + 1, id + 1)) << " " << std::sqrt(cov(id + 2, id + 2)) << " ";

  // TIMEOFF: Get the current estimate time offset
  of_state_est.precision(7);
  of_state_est << state->_calib_dt_CAMtoIMU->value()(0) << " ";
  of_state_est.precision(0);
  of_state_est << state->_options.num_cameras << " ";
  of_state_est.precision(6);

  // TIMEOFF: Get the current std values
  if (state->_options.do_calib_camera_timeoffset) {
    of_state_std << std::sqrt(cov(state->_calib_dt_CAMtoIMU->id(), state->_calib_dt_CAMtoIMU->id())) << " ";
  } else {
    of_state_std << 0.0 << " ";
  }
  of_state_std.precision(0);
  of_state_std << state->_options.num_cameras << " ";
  of_state_std.precision(6);

  // CALIBRATION: Write the camera values to file
  for (int i = 0; i < state->_options.num_cameras; i++) {
    // Intrinsics values
    of_state_est << state->_cam_intrinsics.at(i)->value()(0) << " " << state->_cam_intrinsics.at(i)->value()(1) << " "
                 << state->_cam_intrinsics.at(i)->value()(2) << " " << state->_cam_intrinsics.at(i)->value()(3) << " ";
    of_state_est << state->_cam_intrinsics.at(i)->value()(4) << " " << state->_cam_intrinsics.at(i)->value()(5) << " "
                 << state->_cam_intrinsics.at(i)->value()(6) << " " << state->_cam_intrinsics.at(i)->value()(7) << " ";
    // Rotation and position
    of_state_est << state->_calib_IMUtoCAM.at(i)->value()(0) << " " << state->_calib_IMUtoCAM.at(i)->value()(1) << " "
                 << state->_calib_IMUtoCAM.at(i)->value()(2) << " " << state->_calib_IMUtoCAM.at(i)->value()(3) << " ";
    of_state_est << state->_calib_IMUtoCAM.at(i)->value()(4) << " " << state->_calib_IMUtoCAM.at(i)->value()(5) << " "
                 << state->_calib_IMUtoCAM.at(i)->value()(6) << " ";
    // Covariance
    if (state->_options.do_calib_camera_intrinsics) {
      int index_in = state->_cam_intrinsics.at(i)->id();
      of_state_std << std::sqrt(cov(index_in + 0, index_in + 0)) << " " << std::sqrt(cov(index_in + 1, index_in + 1)) << " "
                   << std::sqrt(cov(index_in + 2, index_in + 2)) << " " << std::sqrt(cov(index_in + 3, index_in + 3)) << " ";
      of_state_std << std::sqrt(cov(index_in + 4, index_in + 4)) << " " << std::sqrt(cov(index_in + 5, index_in + 5)) << " "
                   << std::sqrt(cov(index_in + 6, index_in + 6)) << " " << std::sqrt(cov(index_in + 7, index_in + 7)) << " ";
    } else {
      of_state_std << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " ";
      of_state_std << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " ";
    }
    if (state->_options.do_calib_camera_pose) {
      int index_ex = state->_calib_IMUtoCAM.at(i)->id();
      of_state_std << std::sqrt(cov(index_ex + 0, index_ex + 0)) << " " << std::sqrt(cov(index_ex + 1, index_ex + 1)) << " "
                   << std::sqrt(cov(index_ex + 2, index_ex + 2)) << " ";
      of_state_std << std::sqrt(cov(index_ex + 3, index_ex + 3)) << " " << std::sqrt(cov(index_ex + 4, index_ex + 4)) << " "
                   << std::sqrt(cov(index_ex + 5, index_ex + 5)) << " ";
    } else {
      of_state_std << 0.0 << " " << 0.0 << " " << 0.0 << " ";
      of_state_std << 0.0 << " " << 0.0 << " " << 0.0 << " ";
    }
  }

  // Done with the estimates!
  of_state_est << endl;
  of_state_std << endl;
}
