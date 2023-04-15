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

#ifndef OV_MSCKF_ROS2VisualizerForFolderBasedDataset_H
#define OV_MSCKF_ROS2VisualizerForFolderBasedDataset_H

#include "ROS2Visualizer.h"
#include "utils/sensor_data.h"


namespace ov_msckf {


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
class ROS2VisualizerForFolderBasedDataset : public ROS2Visualizer {

public:
  /**
   * @brief Default constructor
   * @param node ROS node pointer
   * @param app Core estimator manager
   * @param sim Simulator if we are simulating
   */
  ROS2VisualizerForFolderBasedDataset(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim = nullptr) :
  ROS2Visualizer(node, app, sim) {}

  ~ROS2VisualizerForFolderBasedDataset();

  void setup_player(std::shared_ptr<ov_core::YamlParser> parser);

protected:
  std::vector<ov_core::ImuData> imu_data_;
  std::vector<double> image_times_;
  std::vector<std::string> left_image_files_;
  std::vector<std::string> right_image_files_;
  std::shared_ptr<std::thread> data_play_thread_;
  void load_dataset(const std::string& dataset_folder);
  void data_play();
};

} // namespace ov_msckf

#endif // OV_MSCKF_ROS2VisualizerForFolderBasedDataset_H
