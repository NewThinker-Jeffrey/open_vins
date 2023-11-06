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

#ifndef OV_MSCKF_VisualizerForViCapture_H
#define OV_MSCKF_VisualizerForViCapture_H

#include <fstream>
#include <string>
#include <memory>
#include <thread>

#include "slam_dataset/vi_capture.h"
#include "ov_interface/VIO.h"
#include "core/VioManager.h"

namespace ov_msckf {

class Simulator;
class Viewer;

/**
 * @brief Helper class that will publish results onto the ROS framework.
 *
 * Also save to file the current total state and covariance along with the groundtruth if we are simulating.
 * We visualize the following things:
 * - State of the system on TF, pose message, and path
 * - Image of our tracker
 * - Our different features (SLAM, MSCKF, ARUCO)
 * - Groundtruth trajectory if we have it
 */
class VisualizerForViCapture {

public:
  /**
   * @brief Default constructor
   * @param node ROS node pointer
   * @param app Core estimator manager
   * @param sim Simulator if we are simulating
   */

  VisualizerForViCapture(
    std::shared_ptr<ov_interface::VIO> app,
    std::shared_ptr<slam_dataset::ViCapture> capture,
    std::shared_ptr<Viewer> gl_viewer = nullptr,
    const std::string& output_dir = "",
    bool save_feature_images = false,
    bool save_total_state = true);

  ~VisualizerForViCapture();

  void stop_visualization_thread();

protected:
  void visualize();

  void visualize_odometry(double timestamp);


protected:
  std::string output_dir = "";
  bool save_feature_images = false;
  std::string feature_image_save_dir = "";
  bool save_total_state = false;
  std::ofstream of_state_est, of_state_std, of_state_gt;

  std::shared_ptr<std::thread> vis_thread_;
  std::shared_ptr<VioManager::Output> vis_output_;
  std::atomic<bool> stop_viz_request_;

  double last_visualization_timestamp = 0;

  std::shared_ptr<ov_interface::VIO> sys_;
  std::shared_ptr<Viewer> gl_viewer_;
};

} // namespace ov_msckf

#endif // OV_MSCKF_VisualizerForViCapture_H
