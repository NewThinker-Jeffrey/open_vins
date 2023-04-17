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

#include "ROS2VisualizerForFolderBasedDataset.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

#include "core/VioManager.h"
#include "ros/ROSVisualizerHelper.h"
#include "sim/Simulator.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {

bool checkFileExistence(const std::string& path) {
  std::ifstream f(path.c_str());
  return f.good();
}

void LoadImages(const std::string &strPathLeft, const std::string &strPathRight, const std::string &strPathTimes,
                std::vector<std::string> &vstrImageLeft, std::vector<std::string> &vstrImageRight, std::vector<double> &vTimeStamps)
{
    std::ifstream fTimes;
    std::string pic_suffix;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while(!fTimes.eof())
    {
        std::string s;
        std::getline(fTimes,s);
        int pos;
        if ((pos = s.find(',')) != string::npos) {
          s = s.substr(0, pos);
        }

        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            if (pic_suffix.empty()) {
              std::string testfile = strPathLeft + "/" + ss.str();
              std::vector <std::string> suffix_list = {".png", ".jpg"};
              for (const std::string & suffix : suffix_list) {
                if (checkFileExistence(testfile + suffix)) {
                  pic_suffix = suffix;
                  break;
                }
              }
            }

            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + pic_suffix);
            if (!strPathRight.empty()) {
              vstrImageRight.push_back(strPathRight + "/" + ss.str() + pic_suffix);
            }
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);
        }
    }
}

void LoadIMU(const std::string &strImuPath, std::vector<ov_core::ImuData>& imu_data)
{
    std::ifstream fImu;
    fImu.open(strImuPath.c_str());
    imu_data.reserve(5000);

    while(!fImu.eof())
    {
        std::string s;
        std::getline(fImu,s);
        if (s[0] == '#')
            continue;

        if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            ov_core::ImuData imu;
            imu.timestamp = data[0]/1e9;
            imu.wm = Eigen::Vector3d(data[1],data[2],data[3]);
            imu.am = Eigen::Vector3d(data[4],data[5],data[6]);
            imu_data.push_back(imu);
        }
    }
}
}  // namespace

ROS2VisualizerForFolderBasedDataset::~ROS2VisualizerForFolderBasedDataset() {
  if (data_play_thread_) {
    data_play_thread_->join();  // todo: stop the thread 
  }
}

void ROS2VisualizerForFolderBasedDataset::setup_player(std::shared_ptr<ov_core::YamlParser> parser) {

  // We need a valid parser
  assert(parser != nullptr);

  if (_node->has_parameter("dataset")) {
    std::string dataset;
    _node->get_parameter<std::string>("dataset", dataset);
    load_dataset(dataset);
    data_play_thread_.reset(new std::thread([this](){
      pthread_setname_np(pthread_self(), "ov_play");
      this->data_play();
    }));
  } else {
    PRINT_WARNING("dataset is not set!\n");
  }
}

void ROS2VisualizerForFolderBasedDataset::wait_play_over() {
  if (data_play_thread_ && data_play_thread_->joinable()) {
    data_play_thread_->join();
    data_play_thread_.reset();
  }
}

void ROS2VisualizerForFolderBasedDataset::stop_algo_threads() {
  _app->stop_threads();
}

void ROS2VisualizerForFolderBasedDataset::load_dataset(const std::string& dataset_folder) {
  LoadIMU(dataset_folder + "/imu0/data.csv", imu_data_);
  if (_app->get_params().state_options.num_cameras == 2) {
    LoadImages(dataset_folder + "/cam0/data", dataset_folder + "/cam1/data", dataset_folder + "/cam0/data.csv",
               left_image_files_, right_image_files_, image_times_);
  } else {
    LoadImages(dataset_folder + "/cam0/data", "", dataset_folder + "/cam0/data.csv",
               left_image_files_, right_image_files_, image_times_);
  }
}

void ROS2VisualizerForFolderBasedDataset::data_play() {
  size_t imu_idx = 0;
  size_t image_idx = 0;
  double imu_start_time = imu_data_[imu_idx].timestamp;
  double image_start_time = image_times_[image_idx];
  double sensor_start_time = imu_start_time; // std::min(imu_start_time, image_start_time);
  auto play_start_time = std::chrono::high_resolution_clock::now();
  std::mutex mutex;
  std::condition_variable cond;
  double play_rate = 1.0;

  // std::cout << "image_start_time - imu_start_time = " << image_start_time - imu_start_time << std::endl;
  // for (size_t i=0; i<10; i++) {
  //   std::cout << "imu " << i << ":" << (imu_data_[i].timestamp - imu_data_[0].timestamp) << std::endl;
  // }
  // for (size_t i=0; i<10; i++) {    
  //   std::cout << "image " << i << ":" << (image_times_[i] - image_times_[0]) << std::endl;
  // }


  if (_node->has_parameter("play_rate")) {
    std::string play_rate_str;
    _node->get_parameter<std::string>("play_rate", play_rate_str);
    play_rate = std::stod(play_rate_str);
  }
  std::cout << "play_rate: " << play_rate << std::endl;
  std::cout << "imu frames: " << imu_data_.size() << std::endl;
  std::cout << "image frames: " << image_times_.size() << std::endl;

  double next_sensor_time;
  auto get_next_sensor_time = [&]() {
    double next_imu_time = imu_idx < imu_data_.size() ? imu_data_[imu_idx].timestamp : -1;
    double next_image_time = image_idx < image_times_.size() ? image_times_[image_idx] : -1;
    if (next_imu_time < 0) {
      return next_image_time;
    } else if (next_image_time < 0) {
      return next_imu_time;
    } else {
      return std::min(next_image_time, next_imu_time);
    }
  };

  // std::cout << "sensor_start_time " << sensor_start_time << std::endl;

  while(rclcpp::ok() && (next_sensor_time = get_next_sensor_time()) > 0) {
    double sensor_dt = next_sensor_time - sensor_start_time;
    auto next_play_time = play_start_time + std::chrono::milliseconds(int(sensor_dt / play_rate * 1000));
    // std::cout << "sensor dt = " << sensor_dt << ", play dt = " << (next_play_time - play_start_time).count() / 1e9 << std::endl;

    {
      std::unique_lock<std::mutex> locker(mutex);
      while (cond.wait_until(locker, next_play_time) != std::cv_status::timeout);
    }

    if (image_idx < image_times_.size() && image_times_[image_idx] <= next_sensor_time) {
      if (image_idx % 2 == 0) {
        std::lock_guard<std::mutex> lck(camera_queue_mtx);
        std::cout << "play image: " << image_idx << ", queue_size "
                  << "(camera_sync = " << _app->get_camera_sync_queue_size()
                  << ", feature_tracking = " << _app->get_feature_tracking_queue_size()
                  << ", state_update = " << _app->get_state_update_queue_size()
                  << "), sensor dt = " << sensor_dt << ", play dt = " 
                  << (std::chrono::high_resolution_clock::now() - play_start_time).count() / 1e9 << std::endl;
      }

      const int cam_id0 = 0, cam_id1 = 1;
      bool control_imread_frequency = true;  // todo: try false
      double timestamp = image_times_[image_idx];
      double time_delta = 1.0 / _app->get_params().track_frequency;      
      if (!control_imread_frequency || camera_last_timestamp.find(cam_id0) == camera_last_timestamp.end() || timestamp >= camera_last_timestamp.at(cam_id0) + time_delta) {
        camera_last_timestamp[cam_id0] = timestamp;

        // Read left and right images from file
        cv::Mat img_left, img_right;
        // img_left = cv::imread(left_image_files_[image_idx],cv::IMREAD_UNCHANGED);

        img_left = cv::imread(left_image_files_[image_idx],cv::IMREAD_GRAYSCALE);

        // Create the measurement
        ov_core::CameraData message;
        message.timestamp = image_times_[image_idx];
        message.sensor_ids.push_back(cam_id0);
        message.images.push_back(img_left.clone());

        if (_app->get_params().state_options.num_cameras == 2) {
          // img_right = cv::imread(right_image_files_[image_idx],cv::IMREAD_UNCHANGED);
          img_right = cv::imread(right_image_files_[image_idx],cv::IMREAD_GRAYSCALE);
          message.sensor_ids.push_back(cam_id1);
          message.images.push_back(img_right.clone());
        }

        // Load the mask if we are using it, else it is empty
        // TODO: in the future we should get this from external pixel segmentation
        if (_app->get_params().use_mask) {
          message.masks.push_back(_app->get_params().masks.at(cam_id0));
          if (_app->get_params().state_options.num_cameras == 2) {
            message.masks.push_back(_app->get_params().masks.at(cam_id1));
          }
        } else {
          // message.masks.push_back(cv::Mat(img_left.rows, img_left.cols, CV_8UC1, cv::Scalar(255)));
          message.masks.push_back(cv::Mat::zeros(img_left.rows, img_left.cols, CV_8UC1));
          if (_app->get_params().state_options.num_cameras == 2) {
            message.masks.push_back(cv::Mat::zeros(img_right.rows, img_right.cols, CV_8UC1));
          }
        }

        _app->feed_measurement_camera(std::move(message));  // todo: run this in another thread
      }

      image_idx ++;
    }

    if (imu_idx < imu_data_.size() && imu_data_[imu_idx].timestamp <= next_sensor_time) {
      // std::cout << "play imu: " << imu_idx << ", " << sensor_dt << std::endl;
      // callback_inertial(imu_data_[imu_idx]);

      // send it to our VIO system
      auto& message = imu_data_[imu_idx];
      _app->feed_measurement_imu(message);
      visualize_odometry(message.timestamp);

      imu_idx ++;
    }
  }
  std::cout << "**** play over! *****" << std::endl;
}
