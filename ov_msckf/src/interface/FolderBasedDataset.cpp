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

#include "FolderBasedDataset.h"

#include <iostream>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <fstream>
#include <thread>
#include <opencv2/opencv.hpp>


// #define LOAD_INTERNAL_SENSOR_DATA
#ifdef LOAD_INTERNAL_SENSOR_DATA
#include "utils/sensor_data.h"
namespace {
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
            std::string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != std::string::npos) {
                item = s.substr(0, pos);
                data[count++] = std::stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = std::stod(item);

            ov_core::ImuData imu;
            imu.timestamp = data[0]/1e9;
            imu.wm = Eigen::Vector3d(data[1],data[2],data[3]);
            imu.am = Eigen::Vector3d(data[4],data[5],data[6]);
            imu_data.push_back(imu);
        }
    }
}

void readImg(const std::string& img_path, ov_core::CameraData& message) {
  cv::Mat img;
  // img = cv::imread(img_path,cv::IMREAD_UNCHANGED);
  img = cv::imread(img_path,cv::IMREAD_GRAYSCALE);
  message.images.push_back(img.clone());
}

void readStereoImg(const std::string& img_path_l, const std::string& img_path_r, ov_core::CameraData& message) {
  cv::Mat img_left, img_right;

  // img_left = cv::imread(img_path_l,cv::IMREAD_UNCHANGED);
  // img_right = cv::imread(img_path_r,cv::IMREAD_UNCHANGED);

  img_left = cv::imread(img_path_l,cv::IMREAD_GRAYSCALE);
  img_right = cv::imread(img_path_r,cv::IMREAD_GRAYSCALE);

  message.images.push_back(img_left.clone());
  message.images.push_back(img_right.clone());
}
}
#endif

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
        if ((pos = s.find(',')) != std::string::npos) {
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

void LoadIMU(const std::string &strImuPath, std::vector<heisenberg_algo::IMU_MSG>& imu_data)
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
            std::string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != std::string::npos) {
                item = s.substr(0, pos);
                data[count++] = std::stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = std::stod(item);

            heisenberg_algo::IMU_MSG imu;
            imu.timestamp = data[0]/1e9;
            imu.angle_velocity[0] = data[1];
            imu.angle_velocity[1] = data[2];
            imu.angle_velocity[2] = data[3];
            imu.linear_acceleration[0] = data[4];
            imu.linear_acceleration[1] = data[5];
            imu.linear_acceleration[2] = data[6];
            imu.valid = true;
            imu_data.push_back(imu);
        }
    }
}

void readImg(const std::string& img_path, heisenberg_algo::IMG_MSG& message) {
  cv::Mat img;
  // img = cv::imread(img_path,cv::IMREAD_UNCHANGED);
  img = cv::imread(img_path,cv::IMREAD_GRAYSCALE);

  message.channel = img.channels();
  message.width = img.cols;
  message.height = img.rows;
  // std::cout << "readImg():  channel = " << message.channel
  //           << ", width = " << message.width
  //           << ", height = " << message.height << std::endl;
  size_t img_size = message.channel * message.width * message.height;
  assert(img_size <= heisenberg_algo::kMaxImageSize);
  message.size = img_size;
  message.valid = true;
  memcpy(message.data, img.ptr(), img_size);
}

void readStereoImg(const std::string& img_path_l, const std::string& img_path_r, heisenberg_algo::STEREO_IMG_MSG& message) {
  cv::Mat img_left, img_right;

  // img_left = cv::imread(img_path_l,cv::IMREAD_UNCHANGED);
  // img_right = cv::imread(img_path_r,cv::IMREAD_UNCHANGED);

  img_left = cv::imread(img_path_l,cv::IMREAD_GRAYSCALE);
  img_right = cv::imread(img_path_r,cv::IMREAD_GRAYSCALE);

  assert(img_left.channels() == img_right.channels());
  assert(img_left.cols == img_right.cols);
  assert(img_left.rows == img_right.rows);

  message.channel = img_left.channels();
  message.width = img_left.cols;
  message.height = img_left.rows;

  // std::cout << "readStereoImg():  channel = " << message.channel
  //           << ", width = " << message.width
  //           << ", height = " << message.height << std::endl;

  size_t img_size = message.channel * message.width * message.height;
  assert(img_size <= heisenberg_algo::kMaxImageSize);
  message.size = img_size;
  message.valid = true;

  memcpy(message.data_l, img_left.ptr(), img_size);
  memcpy(message.data_r, img_right.ptr(), img_size);
}

}  // namespace

namespace heisenberg_algo {

FolderBasedDataset::~FolderBasedDataset() {
  stop_play();
}

void FolderBasedDataset::setup_player(
        const std::string& dataset,
        double track_frequency,
        std::function<void(int image_idx, IMG_MSG msg)> cam_data_cb,
        std::function<void(int image_idx, STEREO_IMG_MSG msg)> stereo_cam_data_cb,
        std::function<void(int imu_idx, IMU_MSG msg)> imu_data_cb,
        double play_rate,
        bool stereo, int cam_id0, int cam_id1) {

  stereo_ = stereo;
  cam_id0_ = cam_id0;
  cam_id1_ = cam_id1;
  cam_data_cb_ = cam_data_cb;
  stereo_cam_data_cb_ = stereo_cam_data_cb;
  imu_data_cb_ = imu_data_cb;
  track_frequency_ = track_frequency;
  play_rate_ = play_rate;
  std::cout << "FolderBasedDataset::setup_player(): "
            << "dataset='" << dataset << "', "
            << "track_frequency=" << track_frequency << ", "
            << "stereo=" << stereo << ", "
            << "play_rate=" << play_rate << std::endl;
  
  load_dataset(dataset);
  stop_request_ = false;
  data_play_thread_.reset(new std::thread([this](){
    pthread_setname_np(pthread_self(), "ov_play");
    this->data_play();
  }));
}

void FolderBasedDataset::stop_play() {
  stop_request_ = true;
  wait_play_over();
}

void FolderBasedDataset::wait_play_over() {
  if (data_play_thread_ && data_play_thread_->joinable()) {
    data_play_thread_->join();
    data_play_thread_.reset();
  }
}

void FolderBasedDataset::load_dataset(const std::string& dataset_folder) {
  LoadIMU(dataset_folder + "/imu0/data.csv", imu_data_);
  if (stereo_) {
    LoadImages(dataset_folder + "/cam0/data", dataset_folder + "/cam1/data", dataset_folder + "/cam0/data.csv",
               left_image_files_, right_image_files_, image_times_);
  } else {
    LoadImages(dataset_folder + "/cam0/data", "", dataset_folder + "/cam0/data.csv",
               left_image_files_, right_image_files_, image_times_);
  }
  std::cout << "FolderBasedDataset::load_dataset():  imu frames: " << imu_data_.size() << std::endl;
  std::cout << "FolderBasedDataset::load_dataset():  image frames: " << image_times_.size() << std::endl;
}

void FolderBasedDataset::data_play() {
  std::cout << "**FolderBasedDataset::data_play()**" << std::endl;
  size_t imu_idx = 0;
  size_t image_idx = 0;
  std::map<int, double> camera_last_timestamp;
  double imu_start_time = imu_data_[imu_idx].timestamp;
  double image_start_time = image_times_[image_idx];
  sensor_start_time_ = imu_start_time; // std::min(imu_start_time, image_start_time);
  play_start_time_ = std::chrono::high_resolution_clock::now();
  std::mutex mutex;
  std::condition_variable cond;

  // std::cout << "image_start_time - imu_start_time = " << image_start_time - imu_start_time << std::endl;
  // for (size_t i=0; i<10; i++) {
  //   std::cout << "imu " << i << ":" << (imu_data_[i].timestamp - imu_data_[0].timestamp) << std::endl;
  // }
  // for (size_t i=0; i<10; i++) {    
  //   std::cout << "image " << i << ":" << (image_times_[i] - image_times_[0]) << std::endl;
  // }


  std::cout << "FolderBasedDataset::data_play():  play_rate: " << play_rate_ << std::endl;
  std::cout << "FolderBasedDataset::data_play():  imu frames: " << imu_data_.size() << std::endl;
  std::cout << "FolderBasedDataset::data_play():  image frames: " << image_times_.size() << std::endl;

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

  // std::cout << "sensor_start_time " << sensor_start_time_ << std::endl;

  while(!stop_request_ && (next_sensor_time = get_next_sensor_time()) > 0) {
    double sensor_dt = next_sensor_time - sensor_start_time_;
    auto next_play_time = play_start_time_ + std::chrono::milliseconds(int(sensor_dt / play_rate_ * 1000));
    {
      std::unique_lock<std::mutex> locker(mutex);
      while (cond.wait_until(locker, next_play_time) != std::cv_status::timeout);
    }

    if (image_idx < image_times_.size() && image_times_[image_idx] <= next_sensor_time) {

      const int& cam_id0 = cam_id0_;
      const int& cam_id1 = cam_id1_;
      bool control_imread_frequency = true;  // todo: try false
      double timestamp = image_times_[image_idx];
      double time_delta = 1.0 / track_frequency_;

      if (!control_imread_frequency || camera_last_timestamp.find(cam_id0) == camera_last_timestamp.end() || timestamp >= camera_last_timestamp.at(cam_id0) + time_delta) {
        camera_last_timestamp[cam_id0] = timestamp;
        if (stereo_) {
          auto message = std::make_shared<heisenberg_algo::STEREO_IMG_MSG>();
          message->timestamp = timestamp;
          message->cam_id_left = cam_id0;
          message->cam_id_right = cam_id1;
          readStereoImg(left_image_files_[image_idx], right_image_files_[image_idx], *message);
          if (stereo_cam_data_cb_) {
std::cout << "FolderBasedDataset::data_play():  DEBUG 4.3.4" << std::endl;
            // std::cout << "stereo_cam_data_cb_ " << image_idx << std::endl;
            stereo_cam_data_cb_(image_idx, *message);
std::cout << "FolderBasedDataset::data_play():  DEBUG 4.3.5" << std::endl;
          }
        } else {
          auto message = std::make_shared<heisenberg_algo::IMG_MSG>();
          message->timestamp = timestamp;
          message->cam_id = cam_id0;
          readImg(left_image_files_[image_idx], *message);
          if (cam_data_cb_) {
            cam_data_cb_(image_idx, *message);
          }
        }
      }

      image_idx ++;
    }

    if (imu_idx < imu_data_.size() && imu_data_[imu_idx].timestamp <= next_sensor_time) {
      auto& message = imu_data_[imu_idx];
      if (imu_data_cb_) {
        imu_data_cb_(imu_idx, std::move(message));
      }

      imu_idx ++;
    }
  }
  std::cout << "**** play over! *****" << std::endl;
}

}  // namespace heisenberg_algo
