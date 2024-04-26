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

#ifndef OV_MSCKF_VIOMANAGER_H
#define OV_MSCKF_VIOMANAGER_H

#include <Eigen/StdVector>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <string>

#include "VioManagerOptions.h"
#include "state/State.h"
#include "utils/sensor_data.h"


namespace ov_core {
class TrackBase;
class FeatureInitializer;
} // namespace ov_core
namespace ov_init {
class InertialInitializer;
} // namespace ov_init

namespace ov_msckf {

class State;
class StateHelper;
class UpdaterMSCKF;
class UpdaterSLAM;
class UpdaterZeroVelocity;
class Propagator;

namespace dense_mapping {
struct SimpleDenseMapOutput;
class SimpleDenseMapBuilder;
}  // namespace dense_mapping

/**
 * @brief Core class that manages the entire system
 *
 * This class contains the state and other algorithms needed for the MSCKF to work.
 * We feed in measurements into this class and send them to their respective algorithms.
 * If we have measurements to propagate or update with, this class will call on our state to do that.
 */
class VioManager {

public:

  struct Output {
    /////// Status ///////
    // the timestamp of the image used for this output
    struct {
      double timestamp = -1;
      double prev_timestamp = -1;
      bool initialized = false;
      double initialized_time = -1;
      bool drift = false;
      double distance = 0.0;
      bool localized = false;
      Eigen::Matrix4d T_MtoG;
      int accepted_localization_cnt = 0;
      Eigen::Matrix4d last_accepted_reloc_TItoG;
    } status;

    /////// The state ///////
    // A clone for our latest internal state.
    // std::shared_ptr<const State> state_clone;
    std::shared_ptr<State> state_clone;

    /////// Variables used for visualization  ////////
    struct {
      // Good features that where used in the last update (used in visualization)
      std::vector<Eigen::Vector3d> good_features_MSCKF;
      std::vector<size_t> good_feature_ids_MSCKF;

      /// Returns 3d SLAM features in the global frame
      std::vector<Eigen::Vector3d> features_SLAM;
      std::vector<size_t> feature_ids_SLAM;

      /// Returns 3d ARUCO features in the global frame
      std::vector<Eigen::Vector3d> features_ARUCO;
      std::vector<size_t> feature_ids_ARUCO;
      
      /// Returns active tracked features in the current frame
      std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
      std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;

      /// Return the image used when projecting the active tracks
      cv::Mat active_cam0_image;

      /// rgbd map
      std::shared_ptr<const dense_mapping::SimpleDenseMapBuilder> rgbd_dense_map_builder;
    } visualization;
  };

  std::shared_ptr<Output> getLastOutput(bool need_state=true, bool need_visualization=false);

  void setUpdateCallback(std::function<void(const Output& output)> cb) {
    update_callback_ = cb;
  }

  /**
   * @brief Default constructor, will load all configuration variables
   * @param params_ Parameters loaded from either ROS or CMDLINE
   */
  VioManager(VioManagerOptions &params_);

  ~VioManager() { stop_threads(); }

  void stop_threads();

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief Feed function for camera measurements
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_measurement_camera(ov_core::CameraData message);

  /**
   * @brief Feed function for localization measurements (visual-relocal)
   * @param message Contains our timestamp, pose, and cov
   */
  void feed_measurement_localization(ov_core::LocalizationData message);


  /**
   * @brief Feed function for a synchronized simulated cameras
   * @param timestamp Time that this image was collected
   * @param camids Camera ids that we have simulated measurements for
   * @param feats Raw uv simulated measurements
   */
  void feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                   const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats);

  /**
   * @brief Given a state, this will initialize our IMU state.
   * @param imustate State in the MSCKF ordering: [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
   */
  void initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate);

  /// Accessor for current system parameters
  VioManagerOptions get_params() { return params; }

  /// Accessor to get the current propagator
  std::shared_ptr<Propagator> get_propagator() { return propagator; }

  /// Get a nice visualization image of what tracks we have
  cv::Mat get_historical_viz_image(std::shared_ptr<Output> output);

  // clear older tracking cache (used for visualization)
  void clear_older_tracking_cache(double timestamp);

  size_t get_camera_sync_queue_size() {
    std::unique_lock<std::mutex> locker(camera_queue_mutex_);
    return camera_queue_.size();
  }

  size_t get_feature_tracking_queue_size() {
    std::unique_lock<std::mutex> locker(feature_tracking_task_queue_mutex_);
    return feature_tracking_task_queue_.size();
  }

  size_t get_state_update_queue_size() {
    std::unique_lock<std::mutex> locker(update_task_queue_mutex_);
    return update_task_queue_.size();
  }

  void begin_rgbd_mapping();

  void stop_rgbd_mapping();

  void clear_rgbd_map();

  void set_rgbd_map_update_callback(std::function<void(std::shared_ptr<dense_mapping::SimpleDenseMapOutput>)> cb);

protected:
  struct ImgProcessContext {
    std::chrono::high_resolution_clock::time_point rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    std::shared_ptr<ov_core::CameraData> message;
  };  
  using ImgProcessContextPtr = std::shared_ptr<ImgProcessContext>;
  using ImgProcessContextQueue = std::deque<ImgProcessContextPtr>;

  void do_semantic_masking(ImgProcessContextPtr c);
  void do_feature_tracking(ImgProcessContextPtr c);
  void do_update(ImgProcessContextPtr c);
  void update_rgbd_map(ImgProcessContextPtr c);
  void dealwith_localizations();
  void dealwith_one_localization(const ov_core::LocalizationData& reloc, std::shared_ptr<ov_type::PoseJPL> target_clone);

  /// Returns 3d SLAM features in the global frame
  std::vector<Eigen::Vector3d> get_features_SLAM();
  std::vector<size_t> get_feature_ids_SLAM();

  /// Returns 3d ARUCO features in the global frame
  std::vector<Eigen::Vector3d> get_features_ARUCO();
  std::vector<size_t> get_feature_ids_ARUCO();

  /// Return the image used when projecting the active tracks
  void get_active_image(double &timestamp, cv::Mat &image) {
    timestamp = active_tracks_time;
    image = active_image;
  }

  /// Returns active tracked features in the current frame
  void get_active_tracks(double &timestamp, std::unordered_map<size_t, Eigen::Vector3d> &feat_posinG,
                         std::unordered_map<size_t, Eigen::Vector3d> &feat_tracks_uvd) {
    timestamp = active_tracks_time;
    feat_posinG = active_tracks_posinG;
    feat_tracks_uvd = active_tracks_uvd;
  }

  /**
   * @brief Given a new set of camera images, this will track them.
   *
   * If we are having stereo tracking, we should call stereo tracking functions.
   * Otherwise we will try to track on each of the images passed.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void track_image_and_update(ov_core::CameraData &&message);

  /**
   * @brief This will do the propagation and feature updates to the state
   * @param message Contains our timestamp, images, and camera ids
   */
  void do_feature_propagate_update(ImgProcessContextPtr c);

  /**
   * @brief This function will try to initialize the state.
   *
   * This should call on our initializer and try to init the state.
   * In the future we should call the structure-from-motion code from here.
   * This function could also be repurposed to re-initialize the system after failure.
   *
   * @param message Contains our timestamp, images, and camera ids
   * @return True if we have successfully initialized
   */
  bool try_to_initialize(const ov_core::CameraData &message);

  /**
   * @brief This function will will re-triangulate all features in the current frame
   *
   * For all features that are currently being tracked by the system, this will re-triangulate them.
   * This is useful for downstream applications which need the current pointcloud of points (e.g. loop closure).
   * This will try to triangulate *all* points, not just ones that have been used in the update.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void retriangulate_active_tracks(const ov_core::CameraData &message);

  /// Manager parameters
  VioManagerOptions params;

  /// Our master state object :D
  std::shared_ptr<State> state;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// Our sparse feature tracker (klt or descriptor)
  std::shared_ptr<ov_core::TrackBase> trackFEATS;

  /// Our aruoc tracker
  std::shared_ptr<ov_core::TrackBase> trackARUCO;

  /// State initializer
  std::shared_ptr<ov_init::InertialInitializer> initializer;

  /// Boolean if we are initialized or not
  bool is_initialized_vio = false;

  /// Our MSCKF feature updater
  std::shared_ptr<UpdaterMSCKF> updaterMSCKF;

  /// Our SLAM/ARUCO feature updater
  std::shared_ptr<UpdaterSLAM> updaterSLAM;

  /// Our zero velocity tracker
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// This is the queue of measurement times that have come in since we starting doing initialization
  /// After we initialize, we will want to prop & update to the latest timestamp quickly
  std::vector<double> camera_queue_init;
  std::mutex camera_queue_init_mtx;

  // // Timing statistic file and variables
  // std::ofstream of_statistics;
  // // std::chrono::high_resolution_clock::time_point rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  // Track how much distance we have traveled
  double timelastupdate = -1;
  double distance = 0;

  // Startup time of the filter
  double startup_time = -1;

  // Threads and their atomics
  std::atomic<bool> thread_init_running, thread_init_success;

  // If we did a zero velocity update
  bool did_zupt_update = false;
  bool has_moved_since_zupt = false;

  // Good features that where used in the last update (used in visualization)
  std::vector<Eigen::Vector3d> good_features_MSCKF;
  std::vector<size_t> good_feature_ids_MSCKF;

  /// Feature initializer used to triangulate all active tracks
  std::shared_ptr<ov_core::FeatureInitializer> active_tracks_initializer;

  // Re-triangulated features 3d positions seen from the current frame (used in visualization)
  // For each feature we have a linear system A * p_FinG = b we create and increment their costs
  double active_tracks_time = -1;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  cv::Mat active_image;
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b;
  std::map<size_t, int> active_feat_linsys_count;

  std::mutex camera_queue_mutex_;
  std::deque<ov_core::CameraData> camera_queue_;

  std::mutex localization_queue_mutex_;
  std::deque<ov_core::LocalizationData> localization_queue_;
  bool localized_ = false;
  int accepted_localization_cnt_ = 0;
  Eigen::Matrix<double, 4, 1> q_GtoM_;  // in JPL convention
  Eigen::Matrix3d R_GtoM_;
  Eigen::Matrix<double, 3, 1> p_MinG_;
  struct LocalizationAnchor {
    Eigen::Matrix<double, 4, 1> q_GtoM;  // in JPL convention
    Eigen::Matrix3d R_GtoM;
    Eigen::Matrix<double, 3, 1> p_MinG;

    Eigen::Matrix<double, 4, 1> q_MtoI;
    Eigen::Matrix3d R_MtoI;
    Eigen::Matrix<double, 3, 1> p_IinM;
    Eigen::Matrix<double, 4, 1> q_GtoI;
    Eigen::Matrix3d R_GtoI;
    Eigen::Matrix<double, 3, 1> p_IinG;
  };
  std::deque<LocalizationAnchor> initial_loc_buffer_;
  Eigen::Matrix4d last_accepted_reloc_TItoG_;

  ImgProcessContextQueue semantic_masking_task_queue_;
  std::mutex semantic_masking_task_queue_mutex_;
  std::condition_variable semantic_masking_task_queue_cond_;
  std::shared_ptr<std::thread> semantic_masking_thread_;
  void semantic_masking_thread_func();

  ImgProcessContextQueue feature_tracking_task_queue_;
  std::mutex feature_tracking_task_queue_mutex_;
  std::condition_variable feature_tracking_task_queue_cond_;
  std::shared_ptr<std::thread> feature_tracking_thread_;
  void feature_tracking_thread_func();

  ImgProcessContextQueue update_task_queue_;
  std::mutex update_task_queue_mutex_;
  std::condition_variable update_task_queue_cond_;
  std::shared_ptr<std::thread> update_thread_;
  void update_thread_func();


  std::mutex imu_sync_mutex_;
  std::condition_variable imu_sync_cond_;
  double last_imu_time_ = -1;


  std::atomic<bool> stop_request_;


  std::shared_ptr<std::thread> initialization_thread_;


  std::mutex output_mutex_;
  Output output;
    // This callback will be invoked when the output is updated.
  std::function<void(const Output& output)> update_callback_;
  void update_output(double timestamp);

  // Last camera message timestamps we have received (mapped by cam id)
  std::map<int, double> camera_last_timestamp;

  // drift check
  int drift_alarm_count = 0;
  double last_drift_check_time = -1;
  double last_drift_check_distance = -1;
  bool has_drift = false;

  bool check_drift();

  // for disparity computation.
  std::map<double, Eigen::Matrix3d>  gyro_integrated_rotations_window;
  Eigen::Matrix3d cur_gyro_integrated_rotation;
  double cur_gyro_integrated_time;
  void update_gyro_integrated_rotations(double time, const Eigen::Matrix3d& new_rotation);
  void clear_old_gyro_integrated_rotations(double time);
  double compute_disparity_square(
      std::shared_ptr<ov_core::Feature> feat, const std::vector<double>& cloned_times,
      const std::vector<std::unordered_map<size_t, Eigen::Matrix3d>>& R_Cold_in_Ccurs,
      size_t ref_cam_id);

  // imu filter
  std::deque<ov_core::ImuData> imu_filter_buffer;

  // rgbd dense mapping
  std::shared_ptr<dense_mapping::SimpleDenseMapBuilder> rgbd_dense_map_builder;
  std::function<void(std::shared_ptr<dense_mapping::SimpleDenseMapOutput>)> rgbd_dense_map_update_cb;
};

} // namespace ov_msckf

#endif // OV_MSCKF_VIOMANAGER_H
