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

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "utils/quat_ops.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

VioManager::VioManager(VioManagerOptions &params_) : 
    thread_init_running(false), thread_init_success(false), 
    stop_request_(false), 
    cur_gyro_integrated_time(-1.0),
    cur_gyro_integrated_rotation(Eigen::Matrix3d::Identity()) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // override the record_timing_filepath if output_dir is set.
    if (!params.output_dir.empty()) {
      params.record_timing_filepath = params.output_dir + "/" + "ov_msckf_timing.txt";
    }

    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                         state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                         params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist,
                                                         params.camera_extrinsics,
                                                         params.klt_left_major_stereo, params.klt_strict_stereo, params.klt_force_fundamental,
                                                         params.feattrack_predict_keypoints,
                                                         params.feattrack_high_frequency_log));
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio, params.camera_extrinsics,
        params.feattrack_predict_keypoints,
        params.feattrack_high_frequency_log));
  }

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(state->_cam_intrinsics_cameras, state->_options.max_aruco_features,
                                                           params.use_stereo, params.histogram_method, params.downsize_aruco));
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // Our state initialize
  initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options, trackFEATS->get_feature_database());
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options, trackFEATS->get_feature_database());

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity);
  }

  // Feature initializer for active tracks
  active_tracks_initializer = std::make_shared<FeatureInitializer>(params.featinit_options);

  // sync the output with our state
  update_output(-1);

  if (params.async_img_process)  {
    feature_tracking_thread_.reset(new std::thread(std::bind(&VioManager::feature_tracking_thread_func, this)));
    update_thread_.reset(new std::thread(std::bind(&VioManager::update_thread_func, this)));
  }
}

void VioManager::stop_threads() {
  stop_request_ = true;
  if (initialization_thread_ && initialization_thread_->joinable()) {
    initialization_thread_->join();
    initialization_thread_.reset();
  }
  std::cout << "initialization_thread stoped." << std::endl;

  if (feature_tracking_thread_ && feature_tracking_thread_->joinable()) {
    {
      std::unique_lock<std::mutex> locker(feature_tracking_task_queue_mutex_);
      feature_tracking_task_queue_cond_.notify_one();
    }
    feature_tracking_thread_->join();
    feature_tracking_thread_.reset();
  }
  std::cout << "feature_tracking_thread stoped." << std::endl;

  if (update_thread_ && update_thread_->joinable()) {
    {
      std::unique_lock<std::mutex> locker(update_task_queue_mutex_);
      update_task_queue_cond_.notify_one();
    }
    update_thread_->join();
    update_thread_.reset();
  }
  std::cout << "update_thread stoped." << std::endl;
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {
  if (stop_request_) {
    PRINT_WARNING(YELLOW "VioManager::feed_measurement_imu called after the stop_request!\n" RESET);
    return;
  }

  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  // double oldest_time = state->margtimestep();  // not thread-safe

  double oldest_time;
  {
    std::unique_lock<std::mutex> locker(output_mutex_);
    // it's ok to get these timestamps from the last cloned state.
    oldest_time = output.state_clone->margtimestep();
    if (oldest_time > output.state_clone->_timestamp) {
      oldest_time = -1;
    }
    if (!output.status.initialized) {
      oldest_time = message.timestamp - params.init_options.init_window_time + output.state_clone->_calib_dt_CAMtoIMU->value()(0) - 0.10;
    }
  }

  propagator->feed_imu(message, oldest_time);
  trackFEATS->feed_imu(message, oldest_time);
  // trackARUCO->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }

  {
    std::unique_lock<std::mutex> locker(imu_sync_mutex_);
    last_imu_time_ = message.timestamp;
    imu_sync_cond_.notify_all();
  }
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {
  if (stop_request_) {
    PRINT_WARNING(YELLOW "VioManager::feed_measurement_simulation called after the stop_request!\n" RESET);
    return;
  }

  ImgProcessContextPtr c(new ImgProcessContext());

  // Start timing
  c->rT1 = boost::posix_time::microsec_clock::local_time();

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    // Need to also replace it in init and zv-upt since it points to the trackFEATS db pointer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
    if (params.try_zupt) {
      updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                          propagator, params.gravity_mag, params.zupt_max_velocity,
                                                          params.zupt_noise_multiplier, params.zupt_max_disparity);
    }
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  c->rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == timestamp);
      propagator->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }

  c->message.reset(new ov_core::CameraData(std::move(message)));
  do_feature_propagate_update(c);
}

void VioManager::feature_tracking_thread_func() {
  pthread_setname_np(pthread_self(), "ov_track");

  while(1) {
    ImgProcessContextPtr c;
    size_t queue_size;
    {
      std::unique_lock<std::mutex> locker(feature_tracking_task_queue_mutex_);
      feature_tracking_task_queue_cond_.wait(locker, [this](){
        return ! feature_tracking_task_queue_.empty() || stop_request_;
      });
      queue_size = feature_tracking_task_queue_.size();
      if (queue_size > 0) {
        c = feature_tracking_task_queue_.front();
        feature_tracking_task_queue_.pop_front();
      } else {  // stop_request_ is true and we've finished the queue
        return;
      }
    }

    if (queue_size > 2) {
      PRINT_WARNING(YELLOW "too many feature tracking tasks in the queue!! (queue size = %d)\n" RESET,
                    (queue_size));
    }

    do_feature_tracking(c);

    {
      std::unique_lock<std::mutex> locker(update_task_queue_mutex_);
      update_task_queue_.push_back(c);
      update_task_queue_cond_.notify_one();
    }
  }
}

void VioManager::update_thread_func() {
  pthread_setname_np(pthread_self(), "ov_update");

  while(1) {
    ImgProcessContextPtr c;
    int abandon = 0;
    size_t queue_size;
    {
      std::unique_lock<std::mutex> locker(update_task_queue_mutex_);
      update_task_queue_cond_.wait(locker, [this](){
        return ! update_task_queue_.empty() || stop_request_;
      });
      while (update_task_queue_.size() > 2) {
        update_task_queue_.pop_front();
        abandon ++;
      }
      queue_size = update_task_queue_.size();
      if (queue_size > 0) {
        c = update_task_queue_.front();
        update_task_queue_.pop_front();
      } else {  // stop_request_ is true and we've finished the queue
        return;
      }
    }

    if (abandon > 0) {
      PRINT_WARNING(YELLOW "Abandon some updating tasks!! (abandon %d)\n" RESET,
                    (abandon));
    }

    do_update(c);
    has_drift = check_drift();
    update_output(c->message->timestamp);
  }
}

void VioManager::do_feature_tracking(ImgProcessContextPtr c) {
  ov_core::CameraData &message = *(c->message);
  std::shared_ptr<Output> output = getLastOutput();
  double t_d = output->state_clone->_calib_dt_CAMtoIMU->value()(0);
  Eigen::Vector3d gyro_bias = output->state_clone->_imu->bias_g();

  double timestamp_imu_inC;
  {
    // We are able to process if we have at least one IMU measurement greater than the camera time
    std::unique_lock<std::mutex> locker(imu_sync_mutex_);
    imu_sync_cond_.wait(locker, [&](){
      timestamp_imu_inC = last_imu_time_ - t_d;
      return  message.timestamp < timestamp_imu_inC || stop_request_;
    });
  }

  c->rT1 = boost::posix_time::microsec_clock::local_time();
  // Assert we have valid measurement data and ids
  assert(!message.sensor_ids.empty());
  assert(message.sensor_ids.size() == message.images.size());
  for (size_t i = 0; i < message.sensor_ids.size() - 1; i++) {
    assert(message.sensor_ids.at(i) != message.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // Perform our feature tracking!
  trackFEATS->set_t_d(t_d);
  trackFEATS->set_gyro_bias(gyro_bias);
  trackFEATS->feed_new_camera(message);

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }
  c->rT2 = boost::posix_time::microsec_clock::local_time();
}


void VioManager::do_update(ImgProcessContextPtr c) {
  ov_core::CameraData &message = *(c->message);
  double timestamp_imu_inC;
  {
    // We are able to process if we have at least one IMU measurement greater than the camera time
    std::unique_lock<std::mutex> locker(imu_sync_mutex_);
    imu_sync_cond_.wait(locker, [&](){
      timestamp_imu_inC = last_imu_time_ - state->_calib_dt_CAMtoIMU->value()(0);
      return  message.timestamp < timestamp_imu_inC || stop_request_;
    });
  }

  if (message.timestamp >= timestamp_imu_inC) {
    return;   // stop_request_ is true and we don't have newer imu data.
  }

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, message.timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == message.timestamp);
      propagator->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      double time_track = (c->rT2 - c->rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }

  // Call on our propagate and update function
  do_feature_propagate_update(c);
}

bool VioManager::check_drift() {
  constexpr double acc_bias_thr = 3.0;  // m / s^2
  constexpr double gyro_bias_thr = 1.0;  // rad / s
  constexpr double velocity_thr = 2.0;  // m / s
  constexpr double std_p_thr = 2.0;  // m  (larger std_p means we've lost confidence about the estimate)
  constexpr double std_q_thr = 0.5;  // rad


  if (!is_initialized_vio || !state || state->_timestamp <= 0) {
    return false;  // we have not initialized.
  }

  if (last_drift_check_time >= state->_timestamp) {
    return false;  // prevent duplicate checks.
  }

  double dt = state->_timestamp - last_drift_check_time;
  double ds = 0.0;
  if (last_drift_check_distance > 0 ) {
    ds = distance - last_drift_check_distance;
  }

  last_drift_check_time = state->_timestamp;
  last_drift_check_distance = distance;

  Eigen::Vector3d acc_bias = state->_imu->bias_a();
  Eigen::Vector3d gyro_bias = state->_imu->bias_g();
  Eigen::Vector3d vel = state->_imu->vel();

  double newest_clone_time = -1.0;
  std::shared_ptr<ov_type::PoseJPL> newest_clone_pose;
  for (auto it = state->_clones_IMU.begin(); it != state->_clones_IMU.end(); it ++) {
    if (it->first > newest_clone_time) {
      newest_clone_time = it->first;
      newest_clone_pose = it->second;
    }
  }

  double std_p = 0.0;
  double std_q = 0.0;
  if (newest_clone_pose) {
    Eigen::MatrixXd cov =
    StateHelper::get_marginal_covariance(state, {newest_clone_pose});
    Eigen::MatrixXd q_cov = cov.block(0,0,3,3);
    Eigen::MatrixXd p_cov = cov.block(3,3,3,3);
    std_q = std::sqrt(q_cov.trace());
    std_p = std::sqrt(p_cov.trace());
  }
  
  bool drift_alarm = false;
  if (vel.norm() > velocity_thr || ds / dt > velocity_thr ||
      acc_bias.norm() > acc_bias_thr || gyro_bias.norm() > gyro_bias_thr ||
      std_q > std_q_thr || std_p > std_p_thr) {
    drift_alarm = true;
  }

  if (drift_alarm) {
    drift_alarm_count ++;
  } else {
    drift_alarm_count = 0;
  }

  PRINT_INFO("DriftCheck: vel=%.4f,  ds/dt=%.4f, acc_bias=%.4f,  gyro_bias=%.4f, "
             "std_q=%.4f, std_p=%.4f, drift_alarm=%d, drift_alarm_count=%d\n",
                           vel.norm(), ds/dt,    acc_bias.norm(), gyro_bias.norm(),
              std_q,      std_p,       drift_alarm,    drift_alarm_count);

  if (drift_alarm_count > 10) {
    return true;
  } else {
    return false;
  }
}

void VioManager::update_output(double timestamp) {

  Output output;
  if (timestamp > 0) {
    std::unique_lock<std::mutex> locker(output_mutex_);
    output.status.prev_timestamp = this->output.status.timestamp;
  } else {
    output.status.prev_timestamp = -1;
  }

  output.status.timestamp = timestamp;
  output.status.initialized = is_initialized_vio;
  output.status.initialized_time = startup_time;
  output.status.distance = distance;
  output.status.drift = has_drift;
  // output.state_clone = std::const_pointer_cast<const State>(state->clone());
  output.state_clone = std::const_pointer_cast<State>(state->clone());
  output.visualization.good_features_MSCKF = good_features_MSCKF;
  output.visualization.good_feature_ids_MSCKF = good_feature_ids_MSCKF;
  output.visualization.features_SLAM = get_features_SLAM();
  output.visualization.feature_ids_SLAM = get_feature_ids_SLAM();
  output.visualization.features_ARUCO = get_features_ARUCO();
  output.visualization.feature_ids_ARUCO = get_feature_ids_ARUCO();
  output.visualization.active_tracks_posinG = active_tracks_posinG;
  output.visualization.active_tracks_uvd = active_tracks_uvd;
  output.visualization.active_cam0_image = active_image;
  std::unique_lock<std::mutex> locker(output_mutex_);
  this->output = std::move(output);
}

void VioManager::clear_older_tracking_cache(double timestamp) {
  trackFEATS->clear_older_history(timestamp);
  trackFEATS->get_feature_database()->cleanup_measurements_cache(timestamp - 2.0);  // 2s

  if (trackARUCO) {
    trackARUCO->clear_older_history(timestamp);
    trackARUCO->get_feature_database()->cleanup_measurements_cache(timestamp - 2.0);  // 2s
  }  
}

void VioManager::feed_measurement_camera(ov_core::CameraData &&message) {
  if (stop_request_) {
    PRINT_WARNING(YELLOW "VioManager::feed_measurement_camera called after the stop_request!\n" RESET);
    return;
  }

  std::deque<ov_core::CameraData> pending_messages;
  double time_delta = 1.0 / params.track_frequency;

  {
    std::unique_lock<std::mutex> locker(camera_queue_mutex_);
    camera_queue_.push_back(message);
    std::sort(camera_queue_.begin(), camera_queue_.end());
    size_t num_unique_cameras = (params.state_options.num_cameras == 2) ? 1 : params.state_options.num_cameras;

    while(1) {
      // Count how many unique image streams
      std::map<int, bool> unique_cam_ids;
      for (const auto &cam_msg : camera_queue_) {
        unique_cam_ids[cam_msg.sensor_ids.at(0)] = true;
      }
  
      // We should wait till we have one of each camera to ensure we propagate in the correct order
      if (unique_cam_ids.size() == num_unique_cameras) {
        pending_messages.push_back(std::move(camera_queue_.at(0)));
        camera_queue_.pop_front();
      } else {
        // If we do not have enough unique cameras then we need to wait (for the nexe message)
        break;
      }
    }
  }

  for (auto & msg : pending_messages) {
    // Check if we should drop this image
    int cam_id0 = msg.sensor_ids[0];
    if (camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && msg.timestamp < camera_last_timestamp.at(cam_id0) + time_delta) {
      return;
    }
    track_image_and_update(std::move(msg));
  }
}

void VioManager::track_image_and_update(ov_core::CameraData &&message_in) {
  // Start timing
  auto c = std::make_shared<ImgProcessContext>();
  c->message = std::make_shared<ov_core::CameraData>(std::move(message_in));
  if (params.async_img_process)  {
    PRINT_DEBUG("Run feature tracking and state update in separate threads\n");
    std::unique_lock<std::mutex> locker(feature_tracking_task_queue_mutex_);
    feature_tracking_task_queue_.push_back(c);
    feature_tracking_task_queue_cond_.notify_one();
  } else {
    PRINT_DEBUG("Run feature tracking and state update in the same thread\n");
    do_feature_tracking(c);
    do_update(c);
    has_drift = check_drift();
    update_output(c->message->timestamp);
  }
}

void VioManager::do_feature_propagate_update(ImgProcessContextPtr c) {
  ov_core::CameraData &message = *(c->message);
  auto tmp_rT2 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != message.timestamp) {
    Eigen::Matrix3d new_gyro_rotation = Eigen::Matrix3d::Identity();
    propagator->propagate_and_clone(state, message.timestamp, &new_gyro_rotation);
    update_gyro_integrated_rotations(message.timestamp, new_gyro_rotation);
  }
  c->rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...

  // int min_clones = 5;
  int min_clones = 3;
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, min_clones)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, min_clones));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  double marg_timestamp = state->margtimestep();
  clear_old_gyro_integrated_rotations(marg_timestamp - 1.0);
  double current_timestamp = state->_timestamp;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(current_timestamp, false, true);

  // Don't need to get the oldest features until we reach our max number of clones
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > min_clones) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(marg_timestamp, false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    Feature cloned_feature;
    {
      std::unique_lock<std::mutex> lck(trackFEATS->get_feature_database()->get_mutex());
      cloned_feature = *(*it1);
    }
    cloned_feature.clean_future_measurements(message.timestamp);

    for (const auto &camuvpair : cloned_feature.uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    Feature cloned_feature;
    {
      std::unique_lock<std::mutex> lck(trackFEATS->get_feature_database()->get_mutex());
      cloned_feature = *(*it2);
    }
    cloned_feature.clean_future_measurements(message.timestamp);

    for (const auto &cams : cloned_feature.timestamps) {
      if ((int)cams.second.size() > state->_options.max_clone_size &&
          cams.second.front() <= marg_timestamp &&
          cams.second.back() >= current_timestamp) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  StateHelper::marginalize_slam(state);


  // add new slam features.
  
  // prepare variables for calculating disparities
  std::map<std::shared_ptr<Feature>, double> feat_to_disparity_square;
  std::vector<double> cloned_times;
  std::vector<Eigen::Matrix3d> R_Cold_in_Ccurs;
  std::set<std::shared_ptr<Feature>> feats_maxtracks_set;
  const auto ref_K = params.camera_intrinsics.at(message.sensor_ids[0])->get_K();
  const double ref_focallength = std::max(ref_K(0, 0), ref_K(1, 1));

  Eigen::Matrix3d cur_imu_rotation = gyro_integrated_rotations_window.at(message.timestamp);
  Eigen::Matrix3d R_C_in_I = Eigen::Matrix3d::Identity();
  const auto & extrinsic = params.camera_extrinsics.at(message.sensor_ids[0]);
  R_C_in_I = ov_core::quat_2_Rot(extrinsic.block(0,0,4,1)).transpose();
  if (params.vio_manager_high_frequency_log) {
    std::ostringstream oss;
    oss << R_C_in_I << std::endl;
    PRINT_ALL(YELLOW "%s" RESET, oss.str().c_str());
  }

  for (const auto &clone_imu : state->_clones_IMU) {
    cloned_times.push_back(clone_imu.first);
    if (clone_imu.first == message.timestamp) {
      R_Cold_in_Ccurs.push_back(Eigen::Matrix3d::Identity());
      continue;
    }
    const Eigen::Matrix3d& old_imu_rotation = gyro_integrated_rotations_window.at(clone_imu.first);
    Eigen::Matrix3d R_Iold_in_Icur = cur_imu_rotation.transpose() * old_imu_rotation;
    Eigen::Matrix3d R_Cold_in_Ccur = R_C_in_I.transpose() * R_Iold_in_Icur * R_C_in_I;
    R_Cold_in_Ccurs.push_back(R_Cold_in_Ccur);
  }

  assert(cloned_times.size() == R_Cold_in_Ccurs.size());
  assert(cloned_times.back() == message.timestamp);

  for (auto feat : feats_maxtracks) {
    feats_maxtracks_set.insert(feat);
  }
  auto compare_feat_disparity = [&](std::shared_ptr<Feature> a, std::shared_ptr<Feature> b) -> bool {
    return feat_to_disparity_square.at(a) < feat_to_disparity_square.at(b);
  };

  if (params.choose_new_landmark_by_disparity) {
    for (auto feat : feats_maxtracks) {
      feat_to_disparity_square[feat] = compute_disparity_square(feat, cloned_times, R_Cold_in_Ccurs, message.sensor_ids[0]);
    }
    if (!feats_maxtracks.empty()) {
      std::sort(feats_maxtracks.begin(), feats_maxtracks.end(), compare_feat_disparity);
    }
    // todo(isaac): set a threshold for the disparities?
    if (params.vio_manager_high_frequency_log) {
      std::ostringstream oss;
      oss << "DEBUG feats_slam: for feats_maxtracks (total " <<  feats_maxtracks.size() << "):  ";
      if (!feats_maxtracks.empty()) {
        for (auto feat : feats_maxtracks) {
          oss << sqrt(feat_to_disparity_square.at(feat)) * ref_focallength << "  ";
        }
      }
      oss << std::endl;
      PRINT_DEBUG(YELLOW "%s" RESET, oss.str().c_str());
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && 
      (params.enable_early_landmark || message.timestamp - startup_time >= params.dt_slam_delay) &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      if (params.choose_new_landmark_by_disparity) {
        PRINT_DEBUG(YELLOW "DEBUG feats_slam: add feats_maxtracks (%d) with disparities from %f to %f\n" RESET,
                      valid_amount,
                      sqrt(feat_to_disparity_square.at(*(feats_maxtracks.end() - valid_amount))) * ref_focallength,
                      sqrt(feat_to_disparity_square.at(feats_maxtracks.back())) * ref_focallength);
      }
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());      
    }
    // PRINT_DEBUG(YELLOW "DEBUG feats_slam: new feats_slam size %d, amount_to_add=%d, valid_amount=%d\n" RESET,
    //             feats_slam.size(), amount_to_add, valid_amount);



    // int reserve_for_future_maxtrack = 0;
    // int most_quick_feats_slam_per_frame = state->_options.max_slam_features;
    int reserve_for_future_maxtrack = std::min(5, state->_options.max_slam_features / 4);
    int most_quick_feats_slam_per_frame = state->_options.max_slam_features;  // std::min(15, state->_options.max_slam_features / 4);
    if (valid_amount < amount_to_add - reserve_for_future_maxtrack && params.enable_early_landmark) {
      std::vector<std::shared_ptr<Feature>> quick_feats_slam = 
          trackFEATS->get_feature_database()->features_containing(message.timestamp, false, true);
      const double disparity_thr = std::min(
        1.0 / 180.0 * M_PI,   // 1 degree
        10.0 / ref_focallength  // 10 pixels
      );
      const double disparity_square_thr = disparity_thr * disparity_thr;
      // std::cout << "DEBUG feats_slam:  initial quick_feats_slam.size() = " << quick_feats_slam.size() 
      //           << ",  state->_timestamp - message.timestamp = " << state->_timestamp - message.timestamp << std::endl;
      if (!quick_feats_slam.empty()) {
        auto it = quick_feats_slam.begin();

        while (it != quick_feats_slam.end()) {
          auto feat = *it;
          // if (!feat->timestamps.count(message.sensor_ids[0])) {
          //   // feature belongs to other camera.
          //   it = quick_feats_slam.erase(it);
          //   continue;
          // }
          auto it2 = feats_maxtracks_set.find(feat);
          if (it2 != feats_maxtracks_set.end()) {
            // feature already in feats_maxtracks
            it = quick_feats_slam.erase(it);
            feats_maxtracks_set.erase(it2);
            continue;
          }
          double disparity_square = compute_disparity_square(feat, cloned_times, R_Cold_in_Ccurs, message.sensor_ids[0]);
          if (disparity_square < disparity_square_thr) {
            it = quick_feats_slam.erase(it);
            continue;
          }
          feat_to_disparity_square[feat] = disparity_square;
          it ++;
        }

        std::sort(quick_feats_slam.begin(), quick_feats_slam.end(), compare_feat_disparity);

        if (params.vio_manager_high_frequency_log) {
          std::ostringstream oss;
          oss << "DEBUG feats_slam: for quick_feats_slam (total " <<  quick_feats_slam.size() << "):  ";
          if (!quick_feats_slam.empty()) {
            for (auto feat : quick_feats_slam) {
              oss << sqrt(feat_to_disparity_square.at(feat)) * ref_focallength << "  ";
            }
          }
          oss << std::endl;
          PRINT_DEBUG(YELLOW "%s" RESET, oss.str().c_str());
        }
      }
      int amount_to_add2 = amount_to_add - reserve_for_future_maxtrack - valid_amount;
      amount_to_add2 = amount_to_add2 > most_quick_feats_slam_per_frame ? most_quick_feats_slam_per_frame : amount_to_add2;
      int valid_amount2 = (amount_to_add2 > (int)quick_feats_slam.size()) ? (int)quick_feats_slam.size() : amount_to_add2;
      if (valid_amount2 > 0) {
        feats_slam.insert(feats_slam.end(), quick_feats_slam.end() - valid_amount2, quick_feats_slam.end());
        PRINT_DEBUG(YELLOW "DEBUG feats_slam: add quick_feats_slam (%d) with disparities from %f to %f (disparity_thr = %f)\n" RESET,
                    valid_amount2,
                    sqrt(feat_to_disparity_square.at(*(quick_feats_slam.end() - valid_amount2))) * ref_focallength,
                    sqrt(feat_to_disparity_square.at(quick_feats_slam.back())) * ref_focallength,
                    disparity_thr * ref_focallength);
      }
      PRINT_DEBUG(YELLOW "DEBUG feats_slam: new feats_slam size %d, amount_to_add=%d, valid_amount=%d, amount_to_add2=%d, valid_amount2=%d\n" RESET,
                    feats_slam.size(), amount_to_add, valid_amount, amount_to_add2, valid_amount2);
    }
  }

  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);
  size_t msckf_features_used = featsup_MSCKF.size();
  size_t msckf_features_outliers = 0;
  updaterMSCKF->update(state, featsup_MSCKF);
  msckf_features_outliers = msckf_features_used - featsup_MSCKF.size();
  msckf_features_used = featsup_MSCKF.size();
  c->rT4 = boost::posix_time::microsec_clock::local_time();

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  size_t slam_features_used = feats_slam_UPDATE.size();
  size_t slam_features_outliers = 0;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    updaterSLAM->update(state, featsup_TEMP);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  slam_features_outliers = slam_features_used - feats_slam_UPDATE.size();
  slam_features_used = feats_slam_UPDATE.size();
  c->rT5 = boost::posix_time::microsec_clock::local_time();

  size_t delayed_features_used = feats_slam_DELAYED.size();
  size_t delayed_features_outliers = 0;
  updaterSLAM->delayed_init(state, feats_slam_DELAYED);
  delayed_features_outliers = delayed_features_used - feats_slam_DELAYED.size();
  delayed_features_used = feats_slam_DELAYED.size();
  c->rT6 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
    good_feature_ids_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    good_feature_ids_MSCKF.push_back(feat->featid);
    feat->to_delete = true;
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  updaterSLAM->change_anchors(state);

  // Cleanup any features older than the marginalization time
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
    }
  }

  // Finally marginalize the oldest clone if needed
  StateHelper::marginalize_old_clone(state);
  c->rT7 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (c->rT2 - c->rT1).total_microseconds() * 1e-6;
  double time_switch_thread = (tmp_rT2 - c->rT2).total_microseconds() * 1e-6;
  double time_prop = (c->rT3 - tmp_rT2).total_microseconds() * 1e-6;
  double time_msckf = (c->rT4 - c->rT3).total_microseconds() * 1e-6;
  double time_slam_update = (c->rT5 - c->rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (c->rT6 - c->rT5).total_microseconds() * 1e-6;
  double time_marg = (c->rT7 - c->rT6).total_microseconds() * 1e-6;
  double time_total = (c->rT7 - c->rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_INFO(BLUE "[used_features_and_time]: msckf(%d + %d, %.4f), slam(%d + %d, %.4f), delayed(%d + %d, %.4f), total(%d + %d, %.4f), timestampe: %.6f\n" RESET,
                    msckf_features_used, msckf_features_outliers, time_msckf,
                    slam_features_used, slam_features_outliers, time_slam_update,
                    delayed_features_used, delayed_features_outliers, time_slam_delay,
                    msckf_features_used + slam_features_used + delayed_features_used,
                    msckf_features_outliers + slam_features_outliers + delayed_features_outliers,
                    time_msckf + time_slam_update + time_slam_delay,
                    message.timestamp
                    );

  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for switch thread\n" RESET, time_switch_thread);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
             state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
             state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
             state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

  // Debug for camera imu offset
  if (state->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state->_calib_dt_CAMtoIMU->value()(0));
  }

  // Debug for camera intrinsics
  if (state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }
}

void VioManager::clear_old_gyro_integrated_rotations(double time) {
  const double window_time = 1.5;  // seconds
  auto it = gyro_integrated_rotations_window.begin();
  while (it != gyro_integrated_rotations_window.end() && it->first < time) {
    it = gyro_integrated_rotations_window.erase(it);
  }
}

void VioManager::update_gyro_integrated_rotations(double time, const Eigen::Matrix3d& new_rotation) {
  if (time <= cur_gyro_integrated_time) {
    return;
  }
  cur_gyro_integrated_rotation = cur_gyro_integrated_rotation * new_rotation;
  gyro_integrated_rotations_window[time] = cur_gyro_integrated_rotation;
}

double VioManager::compute_disparity_square(
    std::shared_ptr<ov_core::Feature> feat, const std::vector<double>& cloned_times,
    const std::vector<Eigen::Matrix3d>& R_Cold_in_Ccurs, size_t cam_id) {

  if (!params.camera_extrinsics.count(cam_id)) {
    return 0.0;
  }

  std::vector<double> feat_times;
  std::vector<Eigen::VectorXf> feat_uvs_norm;
  std::vector<Eigen::VectorXf> feat_uvs;
  {
    std::unique_lock<std::mutex> lck(trackFEATS->get_feature_database()->get_mutex());
    if (!feat->timestamps.count(cam_id)) {
      return 0.0;
    }
    // only compute disparities for features with 3 or more observations.
    if (feat->timestamps.at(cam_id).size() < 3) {
      return 0.0;
    }

    feat_times = feat->timestamps.at(cam_id);
    feat_uvs_norm = feat->uvs_norm.at(cam_id);
    if (params.vio_manager_high_frequency_log) {
      feat_uvs = feat->uvs.at(cam_id);
    }
  }
  assert(!feat_times.empty());

  std::vector<int> indices(cloned_times.size());
  int j = 0;
  for (size_t i=0; i<cloned_times.size(); i++) {
    const auto& clone_time = cloned_times[i];
    while (j < feat_times.size() && feat_times[j] < clone_time) {
      j ++;
    }
    assert(j < feat_times.size());  // "j >= feat_times.size()" shouldn't happen.
    if (feat_times[j] > clone_time) {
      indices[i] = -1;
    } else {
      // assert(feat_times[j] == clone_time);
      indices[i] = j;
    }
  }

  assert(indices.back() >= 0);
  assert(feat_times[indices.back()] == cloned_times.back());

  double cur_time = feat_times[indices.back()];
  Eigen::Vector2f cur_uv_norm = feat_uvs_norm[indices.back()];
  Eigen::Vector3d cur_homo(cur_uv_norm.x(), cur_uv_norm.y(), 1.0);
  double cur_homo_square = cur_homo.dot(cur_homo);
  double max_disparity_square = 0.0;

  // for debug
  Eigen::Vector2f cur_uv;  // = feat_uvs[indices.back()];
  Eigen::Vector2f best_old_uv;
  Eigen::Vector2f best_old_uv_norm;
  Eigen::Vector2f best_predict_uv_norm;
  Eigen::Matrix3d best_R_Cold_in_Ccur;
  double best_old_time;

  for (size_t i=0; i<cloned_times.size() - 1; i++) {
    // otherwise, feat_times[j] == clone_time
    int j = indices[i];
    if (j < 0) {
      continue;
    }
    assert(cloned_times[i] == feat_times[j]);
    const Eigen::Vector2f& old_uv_norm = feat_uvs_norm[j];
    const Eigen::Matrix3d& R_Cold_in_Ccur = R_Cold_in_Ccurs[i];
    Eigen::Vector3d old_homo(old_uv_norm.x(), old_uv_norm.y(), 1.0);
    Eigen::Vector3d predict_homo = R_Cold_in_Ccur * old_homo;

#if 0  // bad performance.
    Eigen::Vector2f predict_uv_norm(predict_homo.x() / predict_homo.z(), predict_homo.y() / predict_homo.z());
    Eigen::Vector2f uv_norm_diff = cur_uv_norm - predict_uv_norm;
    double disparity_square = uv_norm_diff.dot(uv_norm_diff);
#else
    double predict_homo_square = predict_homo.dot(predict_homo);
    double dot = predict_homo.dot(cur_homo);
    double disparity_square = 1 - (dot * dot) / (predict_homo_square * cur_homo_square);
#endif

    if (disparity_square > max_disparity_square) {
      max_disparity_square = disparity_square;
      if (params.vio_manager_high_frequency_log) {
        best_old_uv_norm = old_uv_norm;
        best_R_Cold_in_Ccur = R_Cold_in_Ccur;
        best_old_uv = feat_uvs[j];
        best_predict_uv_norm = Eigen::Vector2f(predict_homo.x() / predict_homo.z(), predict_homo.y() / predict_homo.z());
        best_old_time = feat_times[j];
      }
    }
  }

  if (params.vio_manager_high_frequency_log) {
    std::ostringstream oss;
    cur_uv = feat_uvs[indices.back()];
    Eigen::Vector2f predict_uv = params.camera_intrinsics.at(cam_id)->distort_f(Eigen::Vector2f(best_predict_uv_norm.x(), best_predict_uv_norm.y()));
    double raw_uv_norm_diff = (best_old_uv_norm - cur_uv_norm).norm();
    double raw_uv_diff = (best_old_uv - cur_uv).norm();
    double predict_uv_norm_diff = (best_predict_uv_norm - cur_uv_norm).norm();
    double predict_uv_diff = (predict_uv - cur_uv).norm();
    oss << "DEBUG compute_disparity_square: , max_disparity = " << sqrt(max_disparity_square)
        << ",  old_time = " << int64_t((best_old_time - cur_time) * 1000) << "[ms]"
        << ",  raw_uv_diff = " << raw_uv_diff
        << ",  predict_uv_diff = " << predict_uv_diff
        << ",  raw_uv_norm_diff = " << raw_uv_norm_diff
        << ",  predict_uv_norm_diff = " << predict_uv_norm_diff
        << ",  cur_uv_norm = " << cur_uv_norm.transpose()
        << ",  R_Cold_in_Ccur = " << std::endl;
    oss << best_R_Cold_in_Ccur << std::endl;
    PRINT_ALL(YELLOW "%s" RESET, oss.str().c_str());
  }

  return max_disparity_square;
}
