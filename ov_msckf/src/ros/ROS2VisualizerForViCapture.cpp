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

#include "ROS2VisualizerForViCapture.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

#include "core/VioManager.h"
#include "ros/ROSVisualizerHelper.h"
#include "no_ros/Viewer.h"
#include "sim/Simulator.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

extern std::shared_ptr<ov_msckf::VioManager> getVioManagerFromVioInterface(ov_interface::VIO*);

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;


ROS2VisualizerForViCapture::ROS2VisualizerForViCapture(
    std::shared_ptr<rclcpp::Node> node,
    std::shared_ptr<ov_interface::VIO> app,
    std::shared_ptr<slam_dataset::ViCapture> capture,
    std::shared_ptr<Viewer> gl_viewer,
    const std::string& output_dir,
    bool save_feature_images,
    bool save_total_state) :
  ROS2Visualizer(node, getVioManagerFromVioInterface(app.get()), nullptr, gl_viewer, output_dir, save_feature_images, save_total_state),
  sys_(app) {
  
  auto print_queue = [this](int image_idx, double image_timestamp) {
    auto internal_sys = getVioManagerFromVioInterface(sys_.get());
    std::cout << "play image: " << image_idx << ", queue_size "
              << "(camera_sync = " << internal_sys->get_camera_sync_queue_size()
              << ", feature_tracking = " << internal_sys->get_feature_tracking_queue_size()
              << ", state_update = " << internal_sys->get_state_update_queue_size() << ")"
              // << ", sensor dt = " << image_timestamp - capture->sensor_start_time() << ", play dt = " 
              // << (std::chrono::high_resolution_clock::now() - capture->play_start_time()).count() / 1e9
              << std::endl;
  };

  auto cam_data_cb = [this, print_queue](int image_idx, const ov_interface::IMG_MSG& msg) {
    print_queue(image_idx, msg.timestamp);
    sys_->ReceiveCamera(msg);
  };

  // auto stereo_cam_data_cb = [this, print_queue](int image_idx, const ov_interface::STEREO_IMG_MSG& msg) {
  //   print_queue(image_idx, msg.timestamp);
  //   sys_->ReceiveStereoCamera(msg);
  // };

  auto imu_data_cb = [this](int imu_idx, const ov_interface::IMU_MSG& msg) {
    // std::cout << "play imu: " << imu_idx << std::endl;
    sys_->ReceiveImu(msg);
    visualize_odometry(msg.timestamp);
  };

  auto internal_sys = getVioManagerFromVioInterface(sys_.get());
  bool stereo = (internal_sys->get_params().state_options.num_cameras == 2);
  capture->setImageCallback(cam_data_cb);
  capture->setImuCallback(imu_data_cb);
  if (stereo) {
    capture->setVisualSensorType(
        slam_dataset::ViCapture::VisualSensorType::STEREO);
  } else {
    capture->setVisualSensorType(
        slam_dataset::ViCapture::VisualSensorType::MONO);
  }
}

ROS2VisualizerForViCapture::~ROS2VisualizerForViCapture() {}
