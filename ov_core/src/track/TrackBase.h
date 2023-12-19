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

#ifndef OV_CORE_TRACK_BASE_H
#define OV_CORE_TRACK_BASE_H

#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils/colors.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace ov_core {

class Feature;
class CamBase;
class FeatureDatabase;

/**
 * @brief Visual feature tracking base class
 *
 * This is the base class for all our visual trackers.
 * The goal here is to provide a common interface so all underlying trackers can simply hide away all the complexities.
 * We have something called the "feature database" which has all the tracking information inside of it.
 * The user can ask this database for features which can then be used in an MSCKF or batch-based setting.
 * The feature tracks store both the raw (distorted) and undistorted/normalized values.
 * Right now we just support two camera models, see: undistort_point_brown() and undistort_point_fisheye().
 *
 * @m_class{m-note m-warning}
 *
 * @par A Note on Multi-Threading Support
 * There is some support for asynchronous multi-threaded feature tracking of independent cameras.
 * The key assumption during implementation is that the user will not try to track on the same camera in parallel, and instead call on
 * different cameras. For example, if I have two cameras, I can either sequentially call the feed function, or I spin each of these into
 * separate threads and wait for their return. The @ref currid is atomic to allow for multiple threads to access it without issue and ensure
 * that all features have unique id values. We also have mutex for access for the calibration and previous images and tracks (used during
 * visualization). It should be noted that if a thread calls visualization, it might hang or the feed thread might, due to acquiring the
 * mutex for that specific camera id / feed.
 *
 * This base class also handles most of the heavy lifting with the visualization, but the sub-classes can override
 * this and do their own logic if they want (i.e. the TrackAruco has its own logic for visualization).
 * This visualization needs access to the prior images and their tracks, thus must synchronise in the case of multi-threading.
 * This shouldn't impact performance, but high frequency visualization calls can negatively effect the performance.
 */
class TrackBase {

public:
  /**
   * @brief Desired pre-processing image method.
   */
  enum HistogramMethod { NONE, HISTOGRAM, CLAHE };

  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   */
  TrackBase(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
            HistogramMethod histmethod,
            bool rgbd = false,
            double rgbd_depth_unit = 0.001,
            std::map<size_t, std::shared_ptr<Eigen::Matrix4d>> T_CtoIs=std::map<size_t, std::shared_ptr<Eigen::Matrix4d>>(),
            bool keypoint_predict = true, bool high_frequency_log = false);

  virtual ~TrackBase() {}


  void set_t_d(double t_d) {this->t_d = t_d;}

  void set_gyro_bias(const Eigen::Vector3d& gyro_bias) {this->gyro_bias = gyro_bias;}

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  virtual void feed_new_camera(const CameraData &message) = 0;

  /**
   * @brief Shows features extracted in the last image
   * @param img_out image to which we will overlayed features on
   * @param r1,g1,b1 first color to draw in
   * @param r2,g2,b2 second color to draw in
   * @param overlay Text overlay to replace to normal "cam0" in the top left of screen
   */
  virtual void display_active(double timestamp, cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::string overlay = "");

  /**
   * @brief Shows a "trail" for each feature (i.e. its history)
   * @param img_out image to which we will overlayed features on
   * @param r1,g1,b1 first color to draw in
   * @param r2,g2,b2 second color to draw in
   * @param highlighted unique ids which we wish to highlight (e.g. slam feats)
   * @param overlay Text overlay to replace to normal "cam0" in the top left of screen
   */
  virtual void display_history(double timestamp, cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::vector<size_t> highlighted = {},
                               std::string overlay = "");

  /**
   * @brief Get the feature database with all the track information
   * @return FeatureDatabase pointer that one can query for features
   */
  std::shared_ptr<FeatureDatabase> get_feature_database() { return database; }

  /**
   * @brief Changes the ID of an actively tracked feature to another one.
   *
   * This function can be helpfull if you detect a loop-closure with an old frame.
   * One could then change the id of an active feature to match the old feature id!
   *
   * @param id_old Old id we want to change
   * @param id_new Id we want to change the old id to
   */
  void change_feat_id(size_t id_old, size_t id_new);

  /// Getter method for active features in the last frame (observations per camera)
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> get_last_obs(double timestamp) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    return history_vars.at(timestamp).pts;
  }

  /// Getter method for active features in the last frame (ids per camera)
  std::unordered_map<size_t, std::vector<size_t>> get_last_ids(double timestamp) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    return history_vars.at(timestamp).ids;
  }

  /// Getter method for number of active features
  int get_num_features() { return num_features; }

  /// Setter method for number of active features
  void set_num_features(int _num_features) { num_features = _num_features; }

  void clear_older_history(double timestamp) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    internal_clear_older_history(timestamp);
  }

  void feed_imu(const ov_core::ImuData &message, double oldest_time = -1) {
    // Append it to our vector
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    imu_data.emplace_back(message);

    // Clean old measurements
    // std::cout << "PROP: imu_data.size() " << imu_data.size() << std::endl;
    clean_old_imu_measurements(oldest_time - 0.10);
  }

  void clean_old_imu_measurements(double oldest_time) {
    if (oldest_time < 0)
      return;
    auto it0 = imu_data.begin();
    while (it0 != imu_data.end()) {
      if (it0->timestamp < oldest_time) {
        it0 = imu_data.erase(it0);
      } else {
        it0++;
      }
    }
  }

protected:
  Eigen::Matrix3d integrate_gryo(double old_time, double new_time);

  Eigen::Matrix3d predict_rotation(size_t cam_id, double new_time);

  void predict_keypoints(
      size_t cam_id0, size_t cam_id1, const std::vector<cv::KeyPoint>& kpts0, 
      const Eigen::Matrix3d& R_0_in_1, std::vector<cv::KeyPoint>& kpts1_predict);
  
  void predict_keypoints_temporally(
      size_t cam_id, double new_time,
      const std::vector<cv::KeyPoint>& kpts_old,
      std::vector<cv::KeyPoint>& kpts_new_predict,
      Eigen::Matrix3d& output_R_old_in_new);

  void predict_keypoints_stereo(
      size_t cam_id_left, size_t cam_id_right,
      const std::vector<cv::KeyPoint>& kpts_left,
      std::vector<cv::KeyPoint>& kpts_right_predict,
      Eigen::Matrix3d& output_R_left_in_right,
      Eigen::Vector3d& output_t_left_in_right);

  void select_masked(const std::vector<uchar>& mask, std::vector<size_t>& selected_indices);

  void apply_selected_mask(const std::vector<uchar>& selected_mask, const std::vector<size_t>& selected_indices, std::vector<uchar>& total_mask);

  double get_coeffs_mat_for_essential_test(
      const Eigen::Matrix3d& R_0_in_1,
      const std::vector<cv::Point2f>& pts0_n,
      const std::vector<cv::Point2f>& pts1_n,
      Eigen::MatrixXd& coeffs_mat,
      std::vector<double>& disparities);

  Eigen::Vector3d solve_essential(const Eigen::MatrixXd& coeffs_mat, const std::vector<size_t>& used_rows);

  std::vector<size_t> get_essential_inliers(const Eigen::MatrixXd& coeffs_mat, const Eigen::Vector3d& t, const double thr = 0.02);

  void two_point_ransac(
      const Eigen::Matrix3d& R_0_in_1,
      const std::vector<cv::Point2f>& pts0_n,
      const std::vector<cv::Point2f>& pts1_n,
      std::vector<uchar> & inliers_mask,
      const double disparity_thr = 1.0 * M_PI / 180.0 / 3.0,  // moving or stationary ?
      const double essential_inlier_thr = 1.0 * M_PI / 180.0,
      int max_iter = 30);

  void two_point_ransac(
      const Eigen::Matrix3d& R_0_in_1,
      size_t cam_id0, size_t cam_id1,
      const std::vector<cv::KeyPoint>& kpts0,
      const std::vector<cv::KeyPoint>& kpts1,
      std::vector<uchar> & inliers_mask,
      const double disparity_thr = 1.0 * M_PI / 180.0 / 3.0,  // moving or stationary ?
      const double essential_inlier_thr = 1.0 * M_PI / 180.0,
      int max_iter = 30);

  void known_essential_check(
      const Eigen::Matrix3d& R_0_in_1,
      const Eigen::Vector3d& t_0_in_1,
      const std::vector<cv::Point2f>& pts0_n,
      const std::vector<cv::Point2f>& pts1_n,
      std::vector<uchar> & inliers_mask,
      const double essential_inlier_thr = 1.0 * M_PI / 180.0);

  void select_common_id(const std::vector<size_t>& ids0, const std::vector<size_t>& ids1,
                        std::vector<size_t>& common_ids,
                        std::vector<size_t>& selected_indices0,
                        std::vector<size_t>& selected_indices1);

  std::vector<cv::KeyPoint> select_keypoints(const std::vector<size_t>& selected_indices, const std::vector<cv::KeyPoint>& keypoints);

  void fundamental_ransac(
      const std::vector<cv::Point2f>& pts0_n,
      const std::vector<cv::Point2f>& pts1_n,
      const double fundamental_inlier_thr,
      std::vector<uchar> & inliers_mask);

  void fundamental_ransac(
      size_t cam_id0, size_t cam_id1,
      const std::vector<cv::KeyPoint>& kpts0,
      const std::vector<cv::KeyPoint>& kpts1,
      const double fundamental_inlier_thr,
      std::vector<uchar> & inliers_mask);


  void add_rgbd_virtual_keypoints_nolock(
      const CameraData &message,
      const std::vector<size_t>& good_ids_left,
      const std::vector<cv::KeyPoint>& good_left,
      std::vector<size_t>& good_ids_right,
      std::vector<cv::KeyPoint>& good_right);

  void add_rgbd_last_cache_nolock(
      const CameraData &message,
      std::vector<size_t>& good_ids_right,
      std::vector<cv::KeyPoint>& good_right);

protected:
  /// Camera object which has all calibration in it
  std::unordered_map<size_t, std::shared_ptr<CamBase>> camera_calib;

  /// Database with all our current features
  std::shared_ptr<FeatureDatabase> database;

  /// If we are a fisheye model or not
  std::map<size_t, bool> camera_fisheye;

  /// Number of features we should try to track frame to frame
  int num_features;

  /// If we should use binocular tracking or stereo tracking for multi-camera
  bool use_stereo;

  /// Whether our mono-camera supports rgb-d.
  bool use_rgbd;
  double depth_unit_for_rgbd;

  /// What histogram equalization method we should pre-process images with?
  HistogramMethod histogram_method;

  /// Mutexs for our last set of image storage (img_last, pts_last, and ids_last)
  std::vector<std::mutex> mtx_feeds;

  /// Mutex for editing the *_last variables
  std::mutex mtx_last_vars;

  std::map<size_t, double> img_time_last;

  /// Last set of images (use map so all trackers render in the same order)
  std::map<size_t, cv::Mat> img_last;

  /// Last set of images (use map so all trackers render in the same order)
  std::map<size_t, cv::Mat> img_mask_last;

  /// Last set of tracked points
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> pts_last;

  /// Set of IDs of each current feature in the database
  std::unordered_map<size_t, std::vector<size_t>> ids_last;

  /// Last set of doubly_verified_stereo observations
  std::map<size_t, std::set<size_t>> doubly_verified_stereo_last;


  struct HistoryVars {
    std::map<size_t, cv::Mat> img;
    std::map<size_t, cv::Mat> img_mask;
    std::unordered_map<size_t, std::vector<cv::KeyPoint>> pts;
    std::unordered_map<size_t, std::vector<size_t>> ids;
  };

  std::map<double, HistoryVars> history_vars;  // time to history vars

  void internal_add_last_to_history(double timestamp) {
    auto& h = history_vars[timestamp];
    h.img = img_last;
    h.img_mask = img_mask_last;
    h.pts = pts_last;
    h.ids = ids_last;
  }

  void internal_clear_older_history(double timestamp) {
    auto it = history_vars.begin();
    while (it != history_vars.end()) {
      if (it->first < timestamp) {
        it = history_vars.erase(it);
      } else {
        it ++;
      }
    }
  }

  // double last_img_time;



  /// Master ID for this tracker (atomic to allow for multi-threading)
  std::atomic<size_t> currid;

  // Timing variables (most children use these...)
  std::chrono::high_resolution_clock::time_point rT1, rT2, rT3, rT4, rT5, rT6, rT7;


  /// Our history of IMU messages (time, angular, linear)
  std::vector<ImuData> imu_data;
  std::mutex imu_data_mtx;

  /// Map between camid and camera extrinsics
  std::map<size_t, std::shared_ptr<Eigen::Matrix4d>> T_CtoIs;

  double t_d;
  Eigen::Vector3d gyro_bias;
  bool enable_high_frequency_log;
  bool enable_keypoint_predict;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_BASE_H */
