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

#ifndef OV_CORE_SENSOR_DATA_H
#define OV_CORE_SENSOR_DATA_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <vector>

#ifdef USE_HEAR_SLAM
#include "hear_slam/common/sensors/imu.h"
#include "hear_slam/common/sensors/camera.h"
#endif

namespace ov_core {

#ifdef USE_HEAR_SLAM
using ImuData = hear_slam::Imu::Data;

using CameraData = hear_slam::Camera::Data;
#else

/**
 * @brief Struct for a single imu measurement (time, wm, am)
 */
struct ImuData {

  /// Timestamp of the reading
  double timestamp;

  /// Gyroscope reading, angular velocity (rad/s)
  Eigen::Matrix<double, 3, 1> wm;

  /// Accelerometer reading, linear acceleration (m/s^2)
  Eigen::Matrix<double, 3, 1> am;

  /// Sort function to allow for using of STL containers
  bool operator<(const ImuData &other) const { return timestamp < other.timestamp; }

};

/**
 * @brief Struct for a collection of camera measurements.
 *
 * For each image we have a camera id and timestamp that it occured at.
 * If there are multiple cameras we will treat it as pair-wise stereo tracking.
 */
struct CameraData {

  /// Timestamp of the reading
  double timestamp;

  /// Camera ids for each of the images collected
  std::vector<int> sensor_ids;

  /// Raw image we have collected for each camera
  std::vector<cv::Mat> images;

  /// Tracking masks for each camera we have
  std::vector<cv::Mat> masks;

  /// Sort function to allow for using of STL containers
  bool operator<(const CameraData &other) const {
    if (timestamp == other.timestamp) {
      int id = *std::min_element(sensor_ids.begin(), sensor_ids.end());
      int id_other = *std::min_element(other.sensor_ids.begin(), other.sensor_ids.end());
      return id < id_other;
    } else {
      return timestamp < other.timestamp;
    }
  }
};

#endif  // USE_HEAR_SLAM


struct LocalizationData {

  /// Timestamp of the Localization
  double timestamp;

  /// Position reading, position of imu frame expressed in world frame
  Eigen::Matrix<double, 3, 1> pm;

  /// Quaternion reading, rotation from world frame to imu frame (in JPL convention)
  Eigen::Matrix<double, 4, 1> qm;

  /// Cov matrix. q first, and then p, which agrees with openvins convention (see ov_type::PoseJPL::set_local_id())
  Eigen::Matrix<double, 6, 6> qp_cov;

  /// Sort function to allow for using of STL containers
  bool operator<(const LocalizationData &other) const { return timestamp < other.timestamp; }
};


/**
 * @brief Nice helper function that will linearly interpolate between two imu messages.
 *
 * This should be used instead of just "cutting" imu messages that bound the camera times
 * Give better time offset if we use this function, could try other orders/splines if the imu is slow.
 *
 * @param imu_1 imu at begining of interpolation interval
 * @param imu_2 imu at end of interpolation interval
 * @param timestamp Timestamp being interpolated to
 */
ImuData interpolate_data(const ImuData &imu_1, const ImuData &imu_2, double timestamp);


std::vector<ImuData> fill_imu_data_gaps(const std::vector<ImuData>& in_data, double max_gap = 0.011);

/**
 * @brief Helper function that given current imu data, will select imu readings between the two times.
 *
 * This will create measurements that we will integrate with, and an extra measurement at the end.
 * We use the @ref interpolate_data() function to "cut" the imu readings at the begining and end of the integration.
 * The timestamps passed should already take into account the time offset values.
 *
 * @param imu_data IMU data we will select measurements from
 * @param time0 Start timestamp
 * @param time1 End timestamp
 * @param warn If we should warn if we don't have enough IMU to propagate with (e.g. fast prop will get warnings otherwise)
 * @return Vector of measurements (if we could compute them)
 */
std::vector<ImuData> select_imu_readings(const std::vector<ImuData> &imu_data, double time0, double time1,
                                                bool warn = true);

} // namespace ov_core

#endif // OV_CORE_SENSOR_DATA_H