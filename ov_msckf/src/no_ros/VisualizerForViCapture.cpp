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

#include "VisualizerForViCapture.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

#include "no_ros/VisualizerHelper.h"
#include "no_ros/Viewer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"

#include "utils/print.h"
#include "utils/sensor_data.h"

extern std::shared_ptr<ov_msckf::VioManager> getVioManagerFromVioInterface(ov_interface::VIO*);

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

VisualizerForViCapture::VisualizerForViCapture(
    std::shared_ptr<ov_interface::VIO> app,
    std::shared_ptr<slam_dataset::ViCapture> capture,
    std::shared_ptr<Viewer> gl_viewer,
    const std::string& tmp_output_dir,
    bool tmp_save_feature_images,
    bool tmp_save_total_state) :
    
    gl_viewer_(gl_viewer), sys_(app), output_dir(tmp_output_dir),
    save_feature_images(tmp_save_feature_images),
    save_total_state(tmp_save_total_state) {

  if (save_feature_images && !output_dir.empty()) {
    feature_image_save_dir = output_dir + "/" + "feature_images";
    std::filesystem::create_directories(std::filesystem::path(feature_image_save_dir.c_str()));
  }

  // Load if we should save the total state to file
  // If so, then open the file and create folders as needed
  if (save_total_state) {
    // files we will open
    std::string filepath_est = "state_estimate.txt";
    std::string filepath_std = "state_deviation.txt";
    std::string filepath_gt = "state_groundtruth.txt";

    if (!output_dir.empty()) {
      // override the file paths if output_dir is set
      filepath_est = output_dir + "/" + "state_estimate.txt";
      filepath_std = output_dir + "/" + "state_deviation.txt";
      filepath_gt = output_dir + "/" + "state_groundtruth.txt";
    }
    VisualizerHelper::init_total_state_files(
        filepath_est, filepath_std, filepath_gt,
        nullptr, 
        of_state_est, of_state_std, of_state_gt);
  }

  // Start thread for the visualizing
  stop_viz_request_ = false;
  vis_thread_ = std::make_shared<std::thread>([&] {
    pthread_setname_np(pthread_self(), "ov_visualize");
    assert(gl_viewer);
    gl_viewer->init();
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // use a high rate to ensure the vis_output_ to update in time (which is also needed in visualize_odometry()).
    while (!stop_viz_request_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(25));  // todo: make the period configurable.
      visualize();
    }
  });


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
    //visualize_odometry(msg.timestamp);
  };

  capture->registerImageCallback(cam_data_cb);
  capture->registerImuCallback(imu_data_cb);
}

VisualizerForViCapture::~VisualizerForViCapture() {
  // dataset_->stop_play();
}

void VisualizerForViCapture::visualize() {
  auto internal_sys = getVioManagerFromVioInterface(sys_.get());
  auto simple_output = internal_sys->getLastOutput(false, false);
  if (simple_output->status.timestamp <= 0 || last_visualization_timestamp == simple_output->status.timestamp && simple_output->status.initialized)
    return;

  vis_output_ = internal_sys->getLastOutput(true, true);
  // last_visualization_timestamp = vis_output_->state_clone->_timestamp;
  last_visualization_timestamp = vis_output_->status.timestamp;

  if (gl_viewer_) {
    gl_viewer_->show(vis_output_);
  }

  // Save total state
  if (save_total_state) {
    VisualizerHelper::sim_save_total_state_to_file(vis_output_->state_clone, nullptr, of_state_est, of_state_std, of_state_gt);
  }
}

void VisualizerForViCapture::visualize_odometry(double timestamp) {
  // do nothing
}

void VisualizerForViCapture::stop_visualization_thread() {
  stop_viz_request_ = true;
  if (vis_thread_ && vis_thread_->joinable()) {
    vis_thread_->join();
    vis_thread_.reset();
  }
  std::cout << "visualization_thread stoped." << std::endl;
}
