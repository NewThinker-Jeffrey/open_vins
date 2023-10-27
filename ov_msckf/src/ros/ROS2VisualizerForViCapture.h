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

#ifndef OV_MSCKF_ROS2VisualizerForViCapture_H
#define OV_MSCKF_ROS2VisualizerForViCapture_H

#include "ROS2Visualizer.h"
#include "utils/sensor_data.h"

#include "slam_dataset/vi_capture.h"
#include "ov_interface/VIO.h"

namespace ov_msckf {

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
class ROS2VisualizerForViCapture : public ROS2Visualizer {

public:
  /**
   * @brief Default constructor
   * @param node ROS node pointer
   * @param app Core estimator manager
   * @param sim Simulator if we are simulating
   */
  ROS2VisualizerForViCapture(
    std::shared_ptr<rclcpp::Node> node,
    std::shared_ptr<ov_interface::VIO> app,
    std::shared_ptr<slam_dataset::ViCapture> capture,
    std::shared_ptr<Viewer> gl_viewer = nullptr,
    const std::string& output_dir = "",
    bool save_feature_images = false,
    bool save_total_state = true);

  ~ROS2VisualizerForViCapture();

protected:
  std::shared_ptr<ov_interface::VIO> sys_;
};

} // namespace ov_msckf

#endif // OV_MSCKF_ROS2VisualizerForViCapture_H
