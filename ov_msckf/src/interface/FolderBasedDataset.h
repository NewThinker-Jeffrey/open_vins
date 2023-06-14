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

#ifndef HEISENBERG_FolderBasedDataset_H
#define HEISENBERG_FolderBasedDataset_H

#include "HeisenbergSensor.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

namespace heisenberg_algo {

class FolderBasedDataset {

public:
  FolderBasedDataset() {}

  ~FolderBasedDataset();

  void setup_player(const std::string& dataset,
                    double track_frequency,
                    std::function<void(int image_idx, IMG_MSG msg)> cam_data_cb,
                    std::function<void(int image_idx, STEREO_IMG_MSG msg)> stereo_cam_data_cb,
                    std::function<void(int imu_idx, IMU_MSG msg)> imu_data_cb,
                    double play_rate = 1.0,
                    bool stereo = false,
                    int cam_id0 = 0,
                    int cam_id1 = 1);

  void request_stop_play();

  void wait_play_over();

  double sensor_start_time() {return sensor_start_time_;}

  std::chrono::high_resolution_clock::time_point play_start_time() {return play_start_time_;}


protected:
  std::vector<IMU_MSG> imu_data_;
  std::vector<double> image_times_;
  std::vector<std::string> left_image_files_;
  std::vector<std::string> right_image_files_;
  std::shared_ptr<std::thread> data_play_thread_;
  std::atomic<bool> stop_request_;

  double sensor_start_time_;
  std::chrono::high_resolution_clock::time_point play_start_time_;


  double track_frequency_;
  std::function<void(int image_idx, const IMG_MSG& msg)> cam_data_cb_;
  std::function<void(int image_idx, const STEREO_IMG_MSG& msg)> stereo_cam_data_cb_;
  std::function<void(int imu_idx, const IMU_MSG& msg)> imu_data_cb_;

  bool stereo_;
  double play_rate_;
  int cam_id0_, cam_id1_;

  void load_dataset(const std::string& dataset_folder);
  void data_play();
};

} // namespace ov_msckf

#endif // HEISENBERG_FolderBasedDataset_H
