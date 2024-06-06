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
#include "SimpleDenseMapping.h"

#include <glog/logging.h>

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

// headers used by reloc update
// todo(jeffrey): maybe we should put the implementation of localization update in a signle .cpp file)
#include "utils/chi_square/chi_squared_quantile_table_0_95.h"
#include "update/UpdaterHelper.h"

#ifdef USE_HEAR_SLAM
#include "hear_slam/basic/logging.h"
#include "hear_slam/basic/time.h"
#endif


#if ENABLE_MMSEG
#include "mmdeploy/segmentor.hpp"

class SemanticSegmentorWrapper {
 public:
  // Example:  model_path="/slam/seg_model",  profiler_path="/tmp/mmseg_profile.bin"
  SemanticSegmentorWrapper(const std::string model_path,
                           const std::string profiler_path,
                           const std::set<int>& labels_to_mask) :
      profiler(profiler_path) {

    context.Add(mmdeploy::Device("cuda"));
    context.Add(profiler);
    segmentor.reset(new mmdeploy::Segmentor(mmdeploy::Model{model_path}, context));

    lookup_table = cv::Mat::zeros(256, 1, CV_8UC1);
    for (int i=0; i<256; i++) {
      if (labels_to_mask.find(i) != labels_to_mask.end()) {
        lookup_table.at<uchar>(i) = 255;
      }
    }
  }

  mmdeploy::Segmentor* getSegmentor() const {
    return segmentor.get();
  }

  const cv::Mat& getLookupTable() const {
    return lookup_table;
  }

 private:
  mmdeploy::Profiler profiler;
  mmdeploy::Context context;
  std::unique_ptr<mmdeploy::Segmentor> segmentor;
  cv::Mat lookup_table;
};

// static mmdeploy::Segmentor* getSegmentor() {
//   static mmdeploy::Segmentor* segmentor = nullptr;
//   if (!segmentor) {
//     static mmdeploy::Profiler profiler("/tmp/mmseg_profile.bin");
//     static mmdeploy::Context context;
//     context.Add(mmdeploy::Device("cuda"));
//     context.Add(profiler);
//     segmentor = new mmdeploy::Segmentor(mmdeploy::Model{"/slam/seg_model"}, context);
//   }
//   return segmentor;
// }

#else

class SemanticSegmentorWrapper{};

#endif


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

#if ENABLE_MMSEG
  if (params.use_semantic_masking) {
    semantic_segmentor_wrapper = 
      std::make_shared<SemanticSegmentorWrapper>(params.semantic_masking_model_path,
                                                 params.semantic_masking_profiler_path,
                                                 params.semantic_masking_labels_to_mask);
  }
#endif

  // If rgbd_mapping is enabled
  begin_rgbd_mapping();

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
  state->_T_CtoIs = params.T_CtoIs;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // // If we are recording statistics, then open our file
  // if (params.record_timing_information) {
  //   // override the record_timing_filepath if output_dir is set.
  //   if (!params.output_dir.empty()) {
  //     params.record_timing_filepath = params.output_dir + "/" + "ov_msckf_timing.txt";
  //   }

  //   // If the file exists, then delete it
  //   if (std::filesystem::exists(params.record_timing_filepath)) {
  //     std::filesystem::remove(params.record_timing_filepath);
  //     PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
  //   }
  //   // Create the directory that we will open the file in
  //   std::filesystem::path p(params.record_timing_filepath);
  //   std::filesystem::create_directories(p.parent_path());
  //   // Open our statistics file!
  //   of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
  //   // Write the header information into it
  //   of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
  //   if (state->_options.max_slam_features > 0) {
  //     of_statistics << "slam update,slam delayed,";
  //   }
  //   of_statistics << "re-tri & marg,total" << std::endl;
  // }

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
                                                         params.state_options.use_rgbd, params.state_options.virtual_baseline_for_rgbd, params.state_options.depth_unit_for_rgbd,
                                                         state->_T_CtoIs,
                                                         params.klt_left_major_stereo, params.klt_strict_stereo, params.klt_force_fundamental,
                                                         params.feattrack_predict_keypoints,
                                                         params.feattrack_high_frequency_log));
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio,
        params.state_options.use_rgbd, params.state_options.virtual_baseline_for_rgbd, params.state_options.depth_unit_for_rgbd,
        state->_T_CtoIs,
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
    semantic_masking_thread_.reset(new std::thread(std::bind(&VioManager::semantic_masking_thread_func, this)));
    feature_tracking_thread_.reset(new std::thread(std::bind(&VioManager::feature_tracking_thread_func, this)));
    update_thread_.reset(new std::thread(std::bind(&VioManager::update_thread_func, this)));
  }
}

void VioManager::begin_rgbd_mapping() {
  if (params.state_options.use_rgbd && params.rgbd_mapping && !rgbd_dense_map_builder) {
    rgbd_dense_map_builder = std::make_shared<dense_mapping::SimpleDenseMapBuilder>(
        params.rgbd_mapping_resolution,
        params.rgbd_mapping_max_voxels,
        params.rgbd_mapping_max_height,
        params.rgbd_mapping_min_height);
    const size_t color_cam_id = 0;
    auto intrin = params.camera_intrinsics.at(color_cam_id)->clone();
    rgbd_dense_map_builder->registerCamera(color_cam_id, intrin->w(), intrin->h(),
      [=](const Eigen::Vector2i& ixy, Eigen::Vector2f& nxy){
        nxy = intrin->undistort_f(ixy.cast<float>());
        return true;
      },
      [=](const Eigen::Vector2f& nxy, Eigen::Vector2i& ixy){
        ixy = intrin->distort_f(nxy).cast<int>();
        return true;
      }
    );
    rgbd_dense_map_builder->set_output_update_callback(rgbd_dense_map_update_cb);
  }
}

void VioManager::stop_rgbd_mapping() {
  if (rgbd_dense_map_builder) {
    rgbd_dense_map_builder->clear_map();
    rgbd_dense_map_builder.reset();
  }
}

void VioManager::clear_rgbd_map() {
  if (rgbd_dense_map_builder) {
    rgbd_dense_map_builder->clear_map();
  }
}

void VioManager::set_rgbd_map_update_callback(std::function<void(std::shared_ptr<dense_mapping::SimpleDenseMapOutput>)> cb) {
  rgbd_dense_map_update_cb = cb;
  if (rgbd_dense_map_builder) {
    rgbd_dense_map_builder->set_output_update_callback(rgbd_dense_map_update_cb);
  }
}


void VioManager::stop_threads() {
  stop_request_ = true;
  if (initialization_thread_ && initialization_thread_->joinable()) {
    initialization_thread_->join();
    initialization_thread_.reset();
  }
  std::cout << "initialization_thread stoped." << std::endl;

  if (semantic_masking_thread_ && semantic_masking_thread_->joinable()) {
    {
      std::unique_lock<std::mutex> locker(semantic_masking_task_queue_mutex_);
      semantic_masking_task_queue_cond_.notify_one();
    }
    semantic_masking_thread_->join();
    semantic_masking_thread_.reset();
  }
  std::cout << "semantic_masking_thread stoped." << std::endl;


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
  CHECK_GT(message.timestamp, 0);
  if (stop_request_) {
    PRINT_WARNING(YELLOW "VioManager::feed_measurement_imu called after the stop_request!\n" RESET);
    return;
  }

  {
    std::unique_lock<std::mutex> locker(imu_sync_mutex_);
    if (last_imu_time_ > 0) {
      double dt = message.timestamp - last_imu_time_;
      CHECK_GT(dt, 0.0);
    }
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

  static int imu_count = 0;
  imu_count ++;
  double last_propagate_time = -1.0;
  if (params.imu_acc_filter_param >= 2.0) {
    int filter_window = params.imu_acc_filter_param;
    if (filter_window % 2 == 0) {
      ++filter_window;  // make sure it's odd
    }

    if (imu_count % 200 == 0) {
      PRINT_INFO("ImuFilter: filter_window = %d\n", filter_window);
    }
    // do mean filter after a median filter.
    imu_filter_buffer.emplace_back(message);
    while (imu_filter_buffer.size() > filter_window) {
      imu_filter_buffer.pop_front();
    }
    if (imu_filter_buffer.size() == filter_window) {
      ov_core::ImuData median_msg = imu_filter_buffer.at(filter_window/2);
      std::vector<double> xs, ys, zs;
      xs.reserve(filter_window);
      ys.reserve(filter_window);
      zs.reserve(filter_window);
      for (const auto & msg : imu_filter_buffer) {
        xs.push_back(msg.am.x());
        ys.push_back(msg.am.y());
        zs.push_back(msg.am.z());
      }
      auto median = [](std::vector<double>& arr) {
        auto m = arr.begin() + arr.size() / 2;
        std::nth_element(arr.begin(), m, arr.end());
        return arr[arr.size() / 2];
      };
      double x = median(xs);
      double y = median(ys);
      double z = median(zs);
      median_msg.am = Eigen::Vector3d(x,y,z);

      imu_filter_buffer2.emplace_back(median_msg);
      while (imu_filter_buffer2.size() > filter_window) {
        imu_filter_buffer2.pop_front();
      }
      if (imu_filter_buffer2.size() == filter_window) {
        ov_core::ImuData filtered_msg = imu_filter_buffer2.at(filter_window/2);
        filtered_msg.am = Eigen::Vector3d(0,0,0);
        for (const auto & msg : imu_filter_buffer2) {
          filtered_msg.am += msg.am;
        }
        filtered_msg.am /= filter_window;
        propagator->feed_imu(filtered_msg, oldest_time);
        last_propagate_time = filtered_msg.timestamp;
      }
    } else {
      // do not feed anything before the filter buffer is full
    }    
  } else {
    if (imu_count % 200 == 0) {
      PRINT_INFO("ImuFilter: NO_FILTER\n");
    }
    propagator->feed_imu(message, oldest_time);
    last_propagate_time = message.timestamp;
  }

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
    if (last_imu_time_ > 0) {
      auto dt = message.timestamp - last_imu_time_;
      // const double dt_thr = 0.1;  // 100ms
      const double dt_thr = 0.05;  // 50ms
      if (dt >= dt_thr) {  
        PRINT_WARNING(YELLOW "VioManager::feed_measurement_imu(): TOO LARGE IMU_GAP! %d ms (thr: %d)\n" RESET, int(dt*1000), int (dt_thr*1000));
      }
      // assert(dt < dt_thr);  // issue a crash for debug
    }

    last_imu_time_ = message.timestamp;
    last_propagate_time_ = last_propagate_time;
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
  c->rT1 = std::chrono::high_resolution_clock::now();
  c->rT0 = c->rT1;

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
  c->rT2 = std::chrono::high_resolution_clock::now();

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


void VioManager::semantic_masking_thread_func() {
  pthread_setname_np(pthread_self(), "ov_semantic");

  // bool no_drop = true;
  bool no_drop = false;

  while(1) {
    ImgProcessContextPtr c;
    int abandon = 0;
    size_t queue_size;
    {
      std::unique_lock<std::mutex> locker(semantic_masking_task_queue_mutex_);
      semantic_masking_task_queue_cond_.wait(locker, [this](){
        return ! semantic_masking_task_queue_.empty() || stop_request_;
      });

      if (!no_drop) {
        while (semantic_masking_task_queue_.size() > 2) {
          semantic_masking_task_queue_.pop_front();
          abandon ++;
        }
      }

      queue_size = semantic_masking_task_queue_.size();
      if (queue_size > 0) {
        c = semantic_masking_task_queue_.front();
        semantic_masking_task_queue_.pop_front();
      } else {  // stop_request_ is true and we've finished the queue
        return;
      }
    }

    if (!no_drop) {
      if (abandon > 0) {
        PRINT_WARNING(YELLOW "Abandon some semantic_masking tasks!! (abandon %d)\n" RESET,
                      (abandon));
      }
    } else if (queue_size > 2) {
      PRINT_WARNING(YELLOW "too many semantic_masking tasks in the queue!! (queue size = %d)\n" RESET,
                    (queue_size));
    }

    c->rT0 = std::chrono::high_resolution_clock::now();
    do_semantic_masking(c);
    c->rT1 = std::chrono::high_resolution_clock::now();
    double time_mask = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT1 - c->rT0).count();
    PRINT_INFO(GREEN "[TIME]: %.4f seconds for semantic masking\n" RESET, time_mask);

    {
      std::unique_lock<std::mutex> locker(feature_tracking_task_queue_mutex_);
      feature_tracking_task_queue_.push_back(c);
      feature_tracking_task_queue_cond_.notify_one();
    }
  }
}


void VioManager::feature_tracking_thread_func() {
  pthread_setname_np(pthread_self(), "ov_track");

  // bool no_drop = true;
  bool no_drop = false;

  while(1) {
    ImgProcessContextPtr c;
    int abandon = 0;
    size_t queue_size;
    {
      std::unique_lock<std::mutex> locker(feature_tracking_task_queue_mutex_);
      feature_tracking_task_queue_cond_.wait(locker, [this](){
        return ! feature_tracking_task_queue_.empty() || stop_request_;
      });

      if (!no_drop) {
        while (feature_tracking_task_queue_.size() > 2) {
          feature_tracking_task_queue_.pop_front();
          abandon ++;
        }
      }

      queue_size = feature_tracking_task_queue_.size();
      if (queue_size > 0) {
        c = feature_tracking_task_queue_.front();
        feature_tracking_task_queue_.pop_front();
      } else {  // stop_request_ is true and we've finished the queue
        return;
      }
    }

    if (!no_drop) {
      if (abandon > 0) {
        PRINT_WARNING(YELLOW "Abandon some feature tracking tasks!! (abandon %d)\n" RESET,
                      (abandon));
      }
    } else if (queue_size > 2) {
      PRINT_WARNING(YELLOW "too many feature tracking tasks in the queue!! (queue size = %d)\n" RESET,
                    (queue_size));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    do_feature_tracking(c);
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_track = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    PRINT_INFO(GREEN "[TIME]: %.4f seconds for feature tracking\n" RESET, time_track);

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
      // while (update_task_queue_.size() > 5) {
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

#ifdef USE_HEAR_SLAM
    using hear_slam::TimeCounter;
    TimeCounter tc;
#endif

    do_update(c);

#ifdef USE_HEAR_SLAM
    tc.tag("do_update_Done");
#endif

    assert(!is_initialized_vio || !state->_clones_IMU.empty());
    has_drift = check_drift();

#ifdef USE_HEAR_SLAM
    tc.tag("check_drift_Done");
#endif

    dealwith_localizations();

#ifdef USE_HEAR_SLAM
    tc.tag("dealwith_localizations_Done");
#endif

    update_rgbd_map(c);

#ifdef USE_HEAR_SLAM
    tc.tag("update_rgbd_map_Done");
#endif

    update_output(c->message->timestamp);

#ifdef USE_HEAR_SLAM
    tc.tag("update_output_Done");
    tc.report("UpdateTiming: ", true);
#endif
  }
}

void VioManager::do_semantic_masking(ImgProcessContextPtr c) {
#if ENABLE_MMSEG
  if (!params.use_semantic_masking) {
    return;
  }

#ifdef USE_HEAR_SLAM
    using hear_slam::TimeCounter;
    TimeCounter tc;
#endif

  ASSERT(semantic_segmentor_wrapper);

  ov_core::CameraData &message = *(c->message);

  // semantic masking for the 1st image
  cv::Mat img = message.images.at(0);
  cv::Mat img_temp;
  cv::pyrDown(img, img_temp, cv::Size(img.cols / 2, img.rows / 2));

#ifdef USE_HEAR_SLAM
    tc.tag("PreProcessDone");
#endif

  // apply the detector, the result is an array-like class holding a reference to
  // `mmdeploy_segmentation_t`, will be released automatically on destruction
  mmdeploy::Segmentor::Result seg = semantic_segmentor_wrapper->getSegmentor()->Apply(img_temp);

#ifdef USE_HEAR_SLAM
    tc.tag("SegmentationDone");
#endif

  // cv::Mat mask_temp(img.rows / 2, img.cols / 2, CV_8UC1);
  cv::Mat mask_cls(img.rows / 2, img.cols / 2, CV_32SC1, seg->mask);

  // const int person_cls_id = 15;
  // cv::Mat mask_temp = (mask_cls == person_cls_id);

  // // Check max and min value in mask_cls. (Only for debug)
  // double min_val, max_val;
  // cv::minMaxLoc(mask_cls, &min_val, &max_val);
  // PRINT_INFO(GREEN "Semantic labels range: %f to %f\n" RESET, min_val, max_val);
  // if (min_val < 0 || max_val > 255) {
  //   // PRINT_INFO(GREEN "Semantic labels range: %f to %f\n" RESET, min_val, max_val);
  //   ASSERT(min_val >= 0 && max_val <= 255);
  // }

  // convert to 8-bit
  cv::Mat mask_cls_u8;
  mask_cls.convertTo(mask_cls_u8, CV_8UC1);
  cv::Mat mask_temp = cv::Mat::zeros(mask_cls.size(), CV_8UC1);
  cv::LUT(mask_cls_u8, semantic_segmentor_wrapper->getLookupTable(), mask_temp);

  // do dilating
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::Mat dilated_mask_temp;
  cv::dilate(mask_temp, dilated_mask_temp, kernel);

  cv::Mat semantic_mask;
  cv::pyrUp(dilated_mask_temp, semantic_mask, cv::Size(img.cols, img.rows));

  // Compose the semantic_mask and the original mask (i.e. message.masks.at(0))
  message.masks.at(0) |= semantic_mask;

#ifdef USE_HEAR_SLAM
    tc.tag("PostProcessDone");
    tc.report("SemanticSegTiming: ", true);
#endif

#endif
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
      timestamp_imu_inC = last_propagate_time_ - t_d;
      return  message.timestamp < timestamp_imu_inC || stop_request_;
    });
  }

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
  trackFEATS->set_camera_calib(output->state_clone->_cam_intrinsics_cameras);
  trackFEATS->set_T_CtoIs(output->state_clone->_T_CtoIs);

  trackFEATS->feed_new_camera(message);

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }

  c->rT2 = std::chrono::high_resolution_clock::now();
}


void VioManager::do_update(ImgProcessContextPtr c) {
  ov_core::CameraData &message = *(c->message);
  double timestamp_imu_inC;
  {
    // We are able to process if we have at least one IMU measurement greater than the camera time
    std::unique_lock<std::mutex> locker(imu_sync_mutex_);
    imu_sync_cond_.wait(locker, [&](){
      timestamp_imu_inC = last_propagate_time_ - state->_calib_dt_CAMtoIMU->value()(0);
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
      double time_track = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT2 - c->rT1).count();
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }

  // Call on our propagate and update function
  do_feature_propagate_update(c);
}

void VioManager::update_rgbd_map(ImgProcessContextPtr c) {
  // this->rgbd_dense_map_builder might be reset in other threads,
  // so we need to loat it atomicly.
  auto rgbd_dense_map_builder = this->rgbd_dense_map_builder;

  if (!is_initialized_vio || !state || state->_clones_IMU.empty()) {
    return;  // vio not initialized yet.
  }

  if (rgbd_dense_map_builder) {
    const size_t color_cam_id = 0;
    const size_t depth_cam_id = 1;
    const cv::Mat& color = c->message->images.at(color_cam_id);
    const cv::Mat& depth = c->message->images.at(depth_cam_id);

    if (cv::countNonZero(depth) > 0) {
      static int depth_count = 0;
      ++ depth_count;
      if (depth_count % 3 != 0) {
        return;
      }

      auto jpl_q = state->_imu->quat();
      auto pos = state->_imu->pos();
      Eigen::Quaternionf q(jpl_q[3], jpl_q[0], jpl_q[1], jpl_q[2]);
      Eigen::Vector3f p(pos[0], pos[1], pos[2]);
      Eigen::Isometry3f T_M_I = Eigen::Isometry3f::Identity();
      T_M_I.translation() = p;
      T_M_I.rotate(q);
      Eigen::Isometry3f T_I_C;
      T_I_C.matrix() = params.T_CtoIs.at(color_cam_id)->cast<float>();
      Eigen::Isometry3f T_M_C = T_M_I * T_I_C;


      rgbd_dense_map_builder->feed_rgbd_frame(color, depth,
                                color_cam_id,
                                T_M_C, c->message->timestamp,
                                params.rgbd_mapping_pixel_downsample,
                                params.rgbd_mapping_max_depth,
                                params.rgbd_mapping_pixel_start_row,
                                params.rgbd_mapping_pixel_end_row,
                                params.rgbd_mapping_pixel_start_col,
                                params.rgbd_mapping_pixel_end_col);
    }
  }
}

void VioManager::dealwith_one_localization(const ov_core::LocalizationData& reloc, std::shared_ptr<ov_type::PoseJPL> target_clone) {

// std::cout << "DEBUG dealwith_localization: input,  q = [" << reloc.qm.transpose() << "],  p = [" << reloc.pm.transpose() << "]" << std::endl;


  std::shared_ptr<ov_type::PoseJPL> predict_pose = target_clone;

  Eigen::Matrix<double, 4, 1> q_MtoI_predict = predict_pose->quat();
  Eigen::Matrix<double, 3, 1> p_IinM_predict = predict_pose->pos();
  Eigen::Matrix3d R_MtoI_predict = predict_pose->Rot();

  Eigen::Matrix<double, 4, 1> q_GtoI_observe = reloc.qm;
  Eigen::Matrix<double, 3, 1> p_IinG_observe = reloc.pm;
  Eigen::Matrix3d R_GtoI_observe = ov_core::quat_2_Rot(reloc.qm);

  // check gravity direction disparity
  Eigen::Vector3d gravity_vec_diff = R_GtoI_observe * Eigen::Vector3d::UnitZ() - R_MtoI_predict * Eigen::Vector3d::UnitZ();
  double approx_gravity_angle_diff = gravity_vec_diff.norm();
  const double gravity_angle_diff_thr = 0.08;  // about 5°
  PRINT_INFO(BLUE "dealwith_localization: check gravity: gravity_angle_diff = %.4f\n" RESET, approx_gravity_angle_diff);
  if (approx_gravity_angle_diff > gravity_angle_diff_thr) {
    PRINT_WARNING(YELLOW "dealwith_localization: Reject the localization since the gravity check failed!\n" RESET);
    return;
  }

  if (!localized_) {
    // todo(jeffrey):
    //     We may need to collect a series of localizations and then do some consistency check
    //     before accept the initial localization.

    const Eigen::Matrix<double, 4, 1>& q_MtoI = q_MtoI_predict;
    const Eigen::Matrix<double, 3, 1>& p_IinM = p_IinM_predict;
    const Eigen::Matrix<double, 4, 1>& q_GtoI = q_GtoI_observe;
    const Eigen::Matrix<double, 3, 1>& p_IinG = p_IinG_observe;
    const Eigen::Matrix3d& R_MtoI = R_MtoI_predict;
    const Eigen::Matrix3d& R_GtoI = R_GtoI_observe;

    Eigen::Matrix<double, 4, 1> q_MtoI_inv;
    q_MtoI_inv << -q_MtoI[0], -q_MtoI[1], -q_MtoI[2], q_MtoI[3];

// std::cout << "DEBUG dealwith_localization: q_MtoI = [" << q_MtoI.transpose() << "]" << std::endl;
// std::cout << "DEBUG dealwith_localization: p_IinM = [" << p_IinM.transpose() << "]" << std::endl;
// std::cout << "DEBUG dealwith_localization: q_GtoI = [" << q_GtoI.transpose() << "]" << std::endl;
// std::cout << "DEBUG dealwith_localization: p_IinG = [" << p_IinG.transpose() << "]" << std::endl;
// std::cout << "DEBUG dealwith_localization: q_MtoI_inv = [" << q_MtoI_inv.transpose() << "]" << std::endl;

    LocalizationAnchor base;
    base.q_MtoI = q_MtoI;
    base.R_MtoI = R_MtoI;
    base.p_IinM = p_IinM;
    base.q_GtoI = q_GtoI;
    base.R_GtoI = R_GtoI;
    base.p_IinG = p_IinG;
    base.R_GtoM = R_MtoI.transpose() * R_GtoI;  // R_M_G = R_M_I * R_I_G
    base.q_GtoM = ov_core::quat_multiply(q_MtoI_inv, q_GtoI);
    base.p_MinG = p_IinG - base.R_GtoM.transpose() * p_IinM;

// std::cout << "DEBUG dealwith_localization: base.q_GtoM = [" << base.q_GtoM.transpose() << "],  base.p_MinG = [" << base.p_MinG.transpose() << "]" << std::endl;


    initial_loc_buffer_.push_back(base);
    const size_t initial_loc_buffer_size = 3;
    while(initial_loc_buffer_.size() > initial_loc_buffer_size) {
      initial_loc_buffer_.pop_front();
    }
    if (initial_loc_buffer_.size() < initial_loc_buffer_size) {
      PRINT_INFO(BLUE "dealwith_localization: We need to collect %d localization measurements before we "
                 "initialize the localization. now we have %d\n" RESET,
                 initial_loc_buffer_size, initial_loc_buffer_.size());
      return;
    }

    // check consistency
    Eigen::Matrix<double, 4, 1> base_q_GtoM_inv;
    base_q_GtoM_inv << -base.q_GtoM[0], -base.q_GtoM[1], -base.q_GtoM[2], base.q_GtoM[3];
    double max_p_diff = 0.0;
    double max_theta_diff = 0.0;
    for (const LocalizationAnchor& a : initial_loc_buffer_) {
      Eigen::Matrix<double, 4, 1> q_diff = ov_core::quat_multiply(a.q_GtoM, base_q_GtoM_inv);
      // Eigen::Matrix<double, 3, 1> p_diff = a.p_MinG - a.R_GtoM.transpose() * base.R_GtoM * base.p_MinG;
      // Eigen::Matrix<double, 3, 1> p_diff = a.p_MinG - base.p_MinG;
      Eigen::Matrix<double, 3, 1> p_diff =
          base.R_GtoI * (a.p_IinG - base.p_IinG)  // measured vec3 'base -> a' (expressed in base)
            -
          base.R_MtoI * (a.p_IinM - base.p_IinM);  // predicted vec3 'base -> a' (expressed in base)

      Eigen::Matrix<double, 3, 1> theta_diff(2.0 * q_diff[0], 2.0 * q_diff[1], 2.0 * q_diff[2]);

      
      double theta_diff_norm = theta_diff.norm();
      double p_diff_norm = p_diff.norm();
      PRINT_INFO(BLUE "dealwith_localization: check initial: theta_diff_norm = %.4f, p_diff_norm = %.4f\n" RESET, theta_diff_norm, p_diff_norm);
      if (theta_diff_norm > max_theta_diff) {
        max_theta_diff = theta_diff_norm;
      }
      if (p_diff_norm > max_p_diff) {
        max_p_diff = p_diff_norm;
      }
    }
    PRINT_INFO(BLUE "dealwith_localization: check initial: max_theta_diff = %.4f, max_p_diff = %.4f\n" RESET, max_theta_diff, max_p_diff);
    const double initial_loc_theta_diff_thr = 0.1;
    const double initial_loc_pos_diff_thr = 0.5;
    if (max_theta_diff > initial_loc_theta_diff_thr
        // todo(jeffrey): find a better method to deal with p_diff (which is coupled with orientation err and the distance from current position to the origin point)
        || max_p_diff > initial_loc_pos_diff_thr
        ) {
      PRINT_WARNING(YELLOW "dealwith_localization: localization not initilized since the recent localizations differ too much\n" RESET);
      return;
    }
    
    // After q_GtoM_ & p_MinG is initialized, we'll fix it.
    R_GtoM_ = base.R_GtoM;
    q_GtoM_ = base.q_GtoM;
    p_MinG_ = base.p_MinG;
    localized_ = true;
    initial_loc_buffer_.clear();
    PRINT_INFO(BLUE "dealwith_localization: localization initialized!\n" RESET);
  }

  Eigen::Matrix3d R_MtoG = R_GtoM_.transpose();
  Eigen::Matrix3d R_GtoI_predict = R_MtoI_predict * R_GtoM_;  // R_I_G = R_I_M * R_M_G
  // q_GtoI = q_MtoI * q_GtoM;
  // p_IinG = p_MinG + R_MtoG * p_IinM;
  Eigen::Matrix<double, 4, 1> q_GtoI_predict = ov_core::quat_multiply(q_MtoI_predict, q_GtoM_);
  Eigen::Matrix<double, 3, 1> p_IinG_predict = p_MinG_ + R_MtoG * p_IinM_predict;

  // q_err * q_predict = q_observe
  // q_err = q_observe * q_predict.inv()
  Eigen::Matrix<double, 4, 1> q_GtoI_predict_inv;
  q_GtoI_predict_inv << -q_GtoI_predict[0], -q_GtoI_predict[1], -q_GtoI_predict[2], q_GtoI_predict[3];
  Eigen::Matrix<double, 4, 1> q_err = ov_core::quat_multiply(q_GtoI_observe, q_GtoI_predict_inv);
  Eigen::Matrix<double, 3, 1> theta_err(2.0 * q_err[0], 2.0 * q_err[1], 2.0 * q_err[2]);
  Eigen::Matrix<double, 3, 1> p_err = p_IinG_observe - p_IinG_predict;
  double theta_err_norm = theta_err.norm();
  double p_err_norm = p_err.norm();
  PRINT_INFO(BLUE "dealwith_localization: theta_err.norm() = %.4f, p_err.norm() = %.4f\n" RESET, theta_err_norm, p_err_norm);
  // todo(jeffrey): Check theta_err_norm & p_err_norm, they should be small.
  //                If not, we may need to disgard (or reinitialize?) the localization.
  Eigen::Matrix<double, 6, 1> residual;
  residual << theta_err[0], theta_err[1], theta_err[2],
              p_err[0], p_err[1], p_err[2];
  PRINT_INFO(BLUE "dealwith_localization: residual.theta = (%.4f, %.4f, %.4f),  residual.pos = (%.4f, %.4f, %.4f)\n" RESET,
              theta_err[0], theta_err[1], theta_err[2], p_err[0], p_err[1], p_err[2]);
  
  //
  // q_GtoI = q_MtoI * q_GtoM;
  // p_IinG = p_MinG + R_MtoG * p_IinM;
  //
  // We use left perturbation for quaternion in openvins, 
  // so,
  //
  // err_q_GtoI = err_q_MtoI
  // err_p_IinG = R_MtoG * err_p_IinM
  //
  // then we get the H matrix:  [I_{33},  R_G_M]^t
  //

  Eigen::Matrix<double, 6, 6> P_prior =
      StateHelper::get_marginal_covariance(state, {predict_pose});
  {
    Eigen::MatrixXd q_cov = P_prior.block(0,0,3,3);
    Eigen::MatrixXd p_cov = P_prior.block(3,3,3,3);
    double std_q = std::sqrt(q_cov.trace());
    double std_p = std::sqrt(p_cov.trace());
    PRINT_INFO(BLUE "dealwith_localization: predict uncertainty: std_q = %.4f, std_p = %.4f\n" RESET, std_q, std_p);
    // If the uncertainty of predict pose is larger than a thresheld, we re-initialize the localization;
    // Otherwise, we use the cov of predict pose to detect outlier localizations.

    const double q_uncertainty_thr_for_localization_reset = 0.05;
    const double p_uncertainty_thr_for_localization_reset = 0.5;
    if (std_q > q_uncertainty_thr_for_localization_reset ||
        std_p > p_uncertainty_thr_for_localization_reset) {
      PRINT_WARNING(YELLOW "dealwith_localization: Will re-initialize the localization since the current pose estimate has high uncertainty!\n" RESET);
      localized_ = false;
      dealwith_one_localization(reloc, target_clone);
      return;
    }
  }
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  H.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
  H.block<3,3>(3,3) = R_MtoG;
  Eigen::Matrix<double, 6, 6> residual_cov = H * P_prior * H.transpose() + reloc.qp_cov;
  Eigen::Matrix<double, 6, 6> info = residual_cov.inverse();
  double mh_distance_square = residual.transpose() * info * residual;
  const double mh_thr = ::chi_squared_quantile_table_0_95[6];
  PRINT_INFO(BLUE "dealwith_localization: mh_distance_square = %.4f, mh_thr = %.4f\n" RESET, mh_distance_square, mh_thr);
  // check mh_distance, it should be small. If not, we may need to disgard the localization.
  if (mh_distance_square > mh_thr) {
    PRINT_WARNING(YELLOW "dealwith_localization: Will reject the localization @%f since the mh_distance check failed!\n" RESET, reloc.timestamp);
    return;
  }
  
  // do the update
  accepted_localization_cnt_ ++;
  PRINT_INFO(BLUE "dealwith_localization: accecp the localization @%f, total accepted %d!\n" RESET, reloc.timestamp, accepted_localization_cnt_);
  StateHelper::EKFUpdate(state, {predict_pose}, H, residual, reloc.qp_cov);

  // update last_accepted_reloc_TItoG_
  last_accepted_reloc_TItoG_ = Eigen::Matrix4d::Identity();
  last_accepted_reloc_TItoG_.block<3, 3>(0, 0) = R_GtoI_observe.transpose();
  last_accepted_reloc_TItoG_.block<3, 1>(0, 3) = p_IinG_observe;
}

void VioManager::dealwith_localizations() {
  // find the closest clone
  std::vector<double> cloned_times;
  double state_time = state->_timestamp;
  for (const auto &clone_imu : state->_clones_IMU) {
    cloned_times.push_back(clone_imu.first);
  }

  assert(!is_initialized_vio || !cloned_times.empty());
  if (cloned_times.empty()) {  // vio not initialized yet.
    return;
  }
  assert(did_zupt_update || cloned_times.back() == state_time);

  const double TIME_PRECISION = 0.000001;  // in seconds.

  while(1) {
    ov_core::LocalizationData oldest_loc;
    {
      std::unique_lock<std::mutex> locker(localization_queue_mutex_);
      if (localization_queue_.empty()) {
        return;
      }
      oldest_loc = localization_queue_.front();

      if (oldest_loc.timestamp > state_time + TIME_PRECISION) {
        PRINT_WARNING(YELLOW "Localization earlier than vio estimates! But it's ok, "
                      "we'll deal with it later when newer estimates become available. "
                      "(time advanced: %f)\n" RESET, oldest_loc.timestamp - state_time);
        return;
      }
      if (oldest_loc.timestamp + TIME_PRECISION < cloned_times.front()) {
        PRINT_WARNING(YELLOW "Localization comes too late! We'll disgard it!"
                      "(time behind: %f)\n" RESET, cloned_times.front() - oldest_loc.timestamp);
        localization_queue_.pop_front();
        continue;
      }
      
      localization_queue_.pop_front();
    }

    // find the corresponding clone time.
    double target_clone_time = -1.0;
    if (oldest_loc.timestamp > cloned_times.back() + TIME_PRECISION) {
      assert(did_zupt_update && oldest_loc.timestamp <= state_time + TIME_PRECISION);
      target_clone_time = cloned_times.back();
    } else {
      for (double clone_time : cloned_times) {
        if (fabs(clone_time - oldest_loc.timestamp) <= TIME_PRECISION) {
          target_clone_time = clone_time;
          break;
        }
      }
    }

    if (target_clone_time < 0) {
      PRINT_WARNING(YELLOW "dealwith_localization: Localization time doesn't match the vio-estimate times! We'll disgard it!");
      continue;
    }
    PRINT_INFO(BLUE "dealwith_localization: find target clone @%f, reloc.time %f\n" RESET, target_clone_time, oldest_loc.timestamp);

    std::shared_ptr<ov_type::PoseJPL> target_clone = state->_clones_IMU.at(target_clone_time);

    dealwith_one_localization(oldest_loc, target_clone);
  }
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
  output.status.localized = localized_;
  output.status.accepted_localization_cnt = accepted_localization_cnt_;
  output.status.T_MtoG = Eigen::Matrix4d::Identity();
  if (localized_) {
    output.status.T_MtoG.block<3,3>(0,0) = R_GtoM_.transpose();
    output.status.T_MtoG.block<3,1>(0,3) = p_MinG_;
    output.status.last_accepted_reloc_TItoG = last_accepted_reloc_TItoG_;
  }

  // output.state_clone = std::const_pointer_cast<const State>(state->clone());
  output.state_clone = std::const_pointer_cast<State>(state->clone(true));
  output.visualization.good_features_MSCKF = good_features_MSCKF;
  output.visualization.good_feature_ids_MSCKF = good_feature_ids_MSCKF;
  output.visualization.maxtrack_feature_ids = maxtrack_feature_ids;
  output.visualization.features_SLAM = get_features_SLAM();
  output.visualization.feature_ids_SLAM = get_feature_ids_SLAM();
  output.visualization.features_ARUCO = get_features_ARUCO();
  output.visualization.feature_ids_ARUCO = get_feature_ids_ARUCO();
  output.visualization.active_tracks_posinG = active_tracks_posinG;
  output.visualization.active_tracks_uvd = active_tracks_uvd;
  output.visualization.active_cam0_image = active_image;
  output.visualization.rgbd_dense_map_builder = this->rgbd_dense_map_builder;  // Note this->rgbd_dense_map_builder might be reset in other threads.
  std::unique_lock<std::mutex> locker(output_mutex_);
  this->output = std::move(output);

  if (timestamp > 0 && update_callback_) {
    update_callback_(this->output);
  }
}

std::shared_ptr<VioManager::Output> VioManager::getLastOutput(bool need_state, bool need_visualization) {
  std::unique_lock<std::mutex> locker(output_mutex_);
  auto output = std::make_shared<Output>();
  output->status = this->output.status;
  if (need_state && this->output.state_clone) {
    output->state_clone = this->output.state_clone->clone();
    if (output->status.initialized) {
      CHECK_EQ(output->status.timestamp, output->state_clone->_timestamp);  // time in camera_clock
      CHECK_GT(output->state_clone->_timestamp, 0);
      // std::cout << "output->status.timestamp - output->state_clone->_timestamp = " << output->status.timestamp - output->state_clone->_timestamp << std::endl;  // camera_clock vs imu_clock ?
    }
  }
  if (need_visualization) {
    output->visualization = this->output.visualization;
  }
  return output;
}


void VioManager::clear_older_tracking_cache(double timestamp) {
  trackFEATS->clear_older_history(timestamp - 2.0);
  trackFEATS->get_feature_database()->cleanup_measurements_cache(timestamp - 2.0);  // 2s

  if (trackARUCO) {
    trackARUCO->clear_older_history(timestamp - 2.0);
    trackARUCO->get_feature_database()->cleanup_measurements_cache(timestamp - 2.0);  // 2s
  }  
}

void VioManager::feed_measurement_camera(ov_core::CameraData message) {
  CHECK_GT(message.timestamp, 0);
  
  if (stop_request_) {
    PRINT_WARNING(YELLOW "VioManager::feed_measurement_camera called after the stop_request!\n" RESET);
    return;
  }

  // Force the sensor_ids to be 0 (for color) and 1 (for depth or virtual right-camera) for rgbd camera.
  if (params.state_options.use_rgbd) {
    assert(message.sensor_ids.size() == 2);
    message.sensor_ids[0] = 0;
    message.sensor_ids[1] = 1;
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

    // Ensure frames with depth image be accepted.
    //
    // NOTE this might make the actual tracking freq larger than the parameter 'track_frequency' (when
    // the freq of depth images is higher than the parameter 'track_frequency')
    bool has_depth = false;
    if (params.state_options.use_rgbd) {
      if (!msg.images[1].empty() && cv::countNonZero(msg.images[1]) > 0) {
        has_depth = true;
      }
    }

    if (!has_depth && camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && msg.timestamp < camera_last_timestamp.at(cam_id0) + time_delta) {
      return;
    }
    track_image_and_update(std::move(msg));
  }
}


void VioManager::feed_measurement_localization(ov_core::LocalizationData message) {
  if (stop_request_) {
    PRINT_WARNING(YELLOW "VioManager::feed_measurement_localization: called after the stop_request!\n" RESET);
    return;
  }

  PRINT_WARNING(YELLOW "VioManager::feed_measurement_localization: Receive Localization with timestamp %f\n" RESET, message.timestamp);

  {
    std::unique_lock<std::mutex> locker(localization_queue_mutex_);
    localization_queue_.push_back(message);
    std::sort(localization_queue_.begin(), localization_queue_.end());
  }
}

void VioManager::track_image_and_update(ov_core::CameraData &&message_in) {
  // Start timing
  auto c = std::make_shared<ImgProcessContext>();
  c->message = std::make_shared<ov_core::CameraData>(std::move(message_in));
  if (params.async_img_process)  {
    PRINT_DEBUG("Run feature tracking and state update in separate threads\n");
    std::unique_lock<std::mutex> locker(semantic_masking_task_queue_mutex_);
    semantic_masking_task_queue_.push_back(c);
    semantic_masking_task_queue_cond_.notify_one();
  } else {
    PRINT_DEBUG("Run feature tracking and state update in the same thread\n");
    do_semantic_masking(c);
    do_feature_tracking(c);
    do_update(c);
    assert(!is_initialized_vio || !state->_clones_IMU.empty());
    has_drift = check_drift();
    dealwith_localizations();
    update_rgbd_map(c);
    update_output(c->message->timestamp);
  }
}

std::vector<std::shared_ptr<StereoFeatureForPropagation>>
VioManager::choose_stereo_feature_for_propagation(
    double prev_image_time,
    const ov_core::CameraData &message,
    const Eigen::Matrix3d& R_I1toI0) {
  // Choose a stereo feature for state propagation.
  ASSERT(message.sensor_ids.size() == 2);  // we need stereo camera or rgb-d camemra.

  // double prev_image_time = state->_timestamp;
  double cur_image_time = message.timestamp;
  std::vector<std::shared_ptr<Feature>> feats = trackFEATS->get_feature_database()->features_containing(prev_image_time, false, true);
  auto cam_id0 = message.sensor_ids[0];
  auto cam_id1 = message.sensor_ids[1];

  struct StereoPair {
    Eigen::Vector2f uv_norm_cam0[2];
    Eigen::Vector2f uv_norm_cam1[2];

    size_t featid;

    double disparity_square_cam0;
    // double disparity_square_cam1;

    Eigen::Vector3d feat_pos_frame0;
    Eigen::Matrix3d feat_pos_frame0_cov;

    Eigen::Vector3d feat_pos_frame1;
    Eigen::Matrix3d feat_pos_frame1_cov;

    Eigen::Vector3d p1in0;
    Eigen::Matrix3d p1in0_cov;
  };

  std::vector<StereoPair> stereo_pairs;

  {
    std::unique_lock<std::mutex> lck(trackFEATS->get_feature_database()->get_mutex());
    for (auto feat : feats) {
      if (!feat->timestamps.count(cam_id0) || !feat->timestamps.count(cam_id1)) {
        continue;  // skip non-stereo features
      }
      if (!feat->uvs_norm.count(cam_id0) || !feat->uvs_norm.count(cam_id1)) {
        continue;  // skip non-stereo features
      }

      // NOTE:
      //    Though theoretically we should skip those features that're used for propagation
      //    at the previous image time, remaining them might produce better performance in
      //    practice.
      if (params.propagation_feature_skip_latest_used) {
        if (prev_propagation_feat_ids.count(feat->featid)) {
          continue;  // skip features used for prop last time.
        }
      }

      StereoPair pair;

      auto iter_cam0_cur = std::find(feat->timestamps[cam_id0].rbegin(), feat->timestamps[cam_id0].rend(), cur_image_time);
      if (iter_cam0_cur == feat->timestamps[cam_id0].rend()) {
        continue;
      }
      auto ridx_cam0_cur = iter_cam0_cur - feat->timestamps[cam_id0].rbegin();
      pair.uv_norm_cam0[1] = *(feat->uvs_norm[cam_id0].rbegin() + ridx_cam0_cur);

      auto iter_cam1_cur = std::find(feat->timestamps[cam_id1].rbegin(), feat->timestamps[cam_id1].rend(), cur_image_time);
      if (iter_cam1_cur == feat->timestamps[cam_id1].rend()) {
        continue;
      }
      auto ridx_cam1_cur = iter_cam1_cur - feat->timestamps[cam_id1].rbegin();
      pair.uv_norm_cam1[1] = *(feat->uvs_norm[cam_id1].rbegin() + ridx_cam1_cur);

      auto iter_cam0_prev = std::find(feat->timestamps[cam_id0].rbegin(), feat->timestamps[cam_id0].rend(), prev_image_time);
      if (iter_cam0_prev == feat->timestamps[cam_id0].rend()) {
        continue;
      }
      auto ridx_cam0_prev = iter_cam0_prev - feat->timestamps[cam_id0].rbegin();
      pair.uv_norm_cam0[0] = *(feat->uvs_norm[cam_id0].rbegin() + ridx_cam0_prev);

      auto iter_cam1_prev = std::find(feat->timestamps[cam_id1].rbegin(), feat->timestamps[cam_id1].rend(), prev_image_time);
      if (iter_cam1_prev == feat->timestamps[cam_id1].rend()) {
        continue;
      }
      auto ridx_cam1_prev = iter_cam1_prev - feat->timestamps[cam_id1].rbegin();
      pair.uv_norm_cam1[0] = *(feat->uvs_norm[cam_id1].rbegin() + ridx_cam1_prev);

      pair.featid = feat->featid;
      stereo_pairs.push_back(pair);
    }
  }
  prev_propagation_feat_ids.clear();

  Eigen::Matrix4d T_C0toI = *params.T_CtoIs.at(cam_id0);
  Eigen::Matrix3d R_C0toI = T_C0toI.block<3,3>(0,0);
  Eigen::Vector3d p_C0inI = T_C0toI.block<3,1>(0,3);

  Eigen::Matrix4d T_C1toI;
  Eigen::Matrix3d R_C1toI;
  Eigen::Vector3d p_C1inI;
  if (params.state_options.use_rgbd) {
    T_C1toI = T_C0toI;
    Eigen::Vector3d p_C1inC0(params.state_options.virtual_baseline_for_rgbd, 0, 0);
    R_C1toI = T_C1toI.block<3,3>(0,0);
    p_C1inI = p_C0inI + R_C0toI * p_C1inC0;
    T_C1toI.block<3,1>(0,3) = p_C1inI;
  } else {
    T_C1toI = *params.T_CtoIs.at(cam_id1);
    R_C1toI = T_C1toI.block<3,3>(0,0);
    p_C1inI = T_C1toI.block<3,1>(0,3);
  }

  Eigen::Matrix4d T_C1toC0 = T_C0toI.inverse() * T_C1toI;
  Eigen::Matrix3d R_C1toC0 = T_C1toC0.block<3,3>(0,0);
  Eigen::Vector3d p_C1inC0 = T_C1toC0.block<3,1>(0,3);

  for (auto& pair : stereo_pairs) {
    Eigen::Vector3d homo0_cam0(pair.uv_norm_cam0[0].x(), pair.uv_norm_cam0[0].y(), 1.0);
    Eigen::Vector3d homo0_cam1(pair.uv_norm_cam1[0].x(), pair.uv_norm_cam1[0].y(), 1.0);
    homo0_cam1 = R_C1toC0 * homo0_cam1;
    double homo0_cam0_square = homo0_cam0.dot(homo0_cam0);
    double homo0_cam1_square = homo0_cam1.dot(homo0_cam1);
    double dot = homo0_cam0.dot(homo0_cam1);
    pair.disparity_square_cam0 = 1 - (dot * dot) / (homo0_cam0_square * homo0_cam1_square);
  }

  // select "n_select" pairs with the max disparities.
  size_t n_select = std::min(size_t(params.propagation_feature_n_select), stereo_pairs.size());
  auto comp_disparity = [](const StereoPair& a, const StereoPair& b) {
        return a.disparity_square_cam0 > b.disparity_square_cam0;
      };
  std::nth_element(
      stereo_pairs.begin(), stereo_pairs.begin() + n_select, stereo_pairs.end(),
      comp_disparity);
  std::sort(stereo_pairs.begin(), stereo_pairs.begin() + n_select, comp_disparity);

  { // print debug info
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(4);
    for (size_t i=0; i<n_select; i++) {
      const auto& pair = stereo_pairs[i];
      oss << " " << sqrt(pair.disparity_square_cam0) * 180 / M_PI << "° ";
    }
    if (n_select < stereo_pairs.size()) {
      oss << " |(more) ";
      size_t more = std::min(size_t(2), stereo_pairs.size() - n_select);
      for (size_t i=0; i<more; i++) {
        const auto& pair = stereo_pairs[i+n_select];
        oss << " " << sqrt(pair.disparity_square_cam0) * 180 / M_PI << "° ";
      }
      oss << " ...";
    }
    PRINT_INFO("DEBUG_STEREO_PROPAGATION: dt = %.4f, disparities: %s\n", cur_image_time - prev_image_time, oss.str().c_str());
  }


  auto triangulate = [&](const Eigen::Vector2f& uvn_cam0, const Eigen::Vector2f& uvn_cam1) {
    Eigen::Vector3d homo0(uvn_cam0.x(), uvn_cam0.y(), 1.0);
    Eigen::Vector3d homo1(uvn_cam1.x(), uvn_cam1.y(), 1.0);

    // homo0 = R_C0toI * homo0;
    // homo1 = R_C1toI * homo1;
    homo1 = R_C1toC0 * homo1;

    homo0.normalize();
    homo1.normalize();
    Eigen::Matrix3d skew_homo0 = skew_x(homo0);
    Eigen::Matrix3d skew_homo1 = skew_x(homo1);
    Eigen::Matrix3d square_skew_homo0 = skew_homo0.transpose() * skew_homo0;
    Eigen::Matrix3d square_skew_homo1 = skew_homo1.transpose() * skew_homo1;
    Eigen::Matrix3d A = square_skew_homo0 + square_skew_homo1;

    // Eigen::Vector3d b = square_skew_homo0 * p_C0inI +  square_skew_homo1 * p_C1inI;
    Eigen::Vector3d b = square_skew_homo1 * p_C1inC0;

    // Eigen::Vector3d p_feat_in_I = A.colPivHouseholderQr().solve(b);
    // return p_feat_in_I;
    Eigen::Vector3d p_feat_in_C0 = A.colPivHouseholderQr().solve(b);
    return p_feat_in_C0;
  };

  const double bearing_sigma = params.propagation_feature_bearing_sigma;  // rad

  auto refine_triangulate =
      [&](const Eigen::Vector2f& uvn_cam0, const Eigen::Vector2f& uvn_cam1,
          Eigen::Vector3d& p_feat_in_C0, Eigen::Matrix3d& cov) {
    p_feat_in_C0 = triangulate(uvn_cam0, uvn_cam1);
    auto jaco_proj = [](const Eigen::Vector3d& p) {
      Eigen::Matrix<double, 2, 3> J;
      const double& x = p.x();
      const double& y = p.y();
      const double& z = p.z();
      double z2 = z*z;
      J << 1/z, 0, -x/z2,
           0, 1/z, -y/z2;
      return J;
    };

    Eigen::Vector3d p_feat_in_C1 = R_C1toC0.transpose() * (p_feat_in_C0 - p_C1inC0);

    Eigen::Matrix<double, 4, 3> J;
    J << jaco_proj(p_feat_in_C0), jaco_proj(p_feat_in_C1);
    Eigen::Matrix<double, 4, 1> err;
    err << p_feat_in_C0.x() / p_feat_in_C0.z() - uvn_cam0.x(),
           p_feat_in_C0.y() / p_feat_in_C0.z() - uvn_cam0.y(),
           p_feat_in_C1.x() / p_feat_in_C1.z() - uvn_cam1.x(),
           p_feat_in_C1.y() / p_feat_in_C1.z() - uvn_cam1.y();
    Eigen::Matrix3d JTJ = J.transpose() * J;
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(JTJ);
    // Eigen::Matrix3d inverse_JTJ = eigensolver.operatorInverse();
    Eigen::Matrix3d inverse_JTJ = JTJ.inverse();
    cov = inverse_JTJ * (bearing_sigma * bearing_sigma);
    p_feat_in_C0 = p_feat_in_C0 + inverse_JTJ * J.transpose() * err;
  };

  // Estimate p1in0 with each selected pair
  for (size_t i=0; i<n_select; i++) {
    auto& pair = stereo_pairs[i];

    // estimate feat_pos in frame0 and in frame1
    refine_triangulate(pair.uv_norm_cam0[0], pair.uv_norm_cam1[0],
                       pair.feat_pos_frame0, pair.feat_pos_frame0_cov);
    refine_triangulate(pair.uv_norm_cam0[1], pair.uv_norm_cam1[1],
                       pair.feat_pos_frame1, pair.feat_pos_frame1_cov);

    // estimate pos of frame1 in frame0 (with known rotation).
    pair.p1in0 = pair.feat_pos_frame0 - R_I1toI0 * pair.feat_pos_frame1;
    pair.p1in0_cov = pair.feat_pos_frame0_cov + R_I1toI0 * pair.feat_pos_frame1_cov * R_I1toI0.transpose();
  }

  { // print debug info
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(4);
    for (size_t i=0; i<n_select; i++) {
      const auto& pair = stereo_pairs[i];
      oss << "(" << pair.p1in0.transpose() << ")  ";
    }
    PRINT_INFO("DEBUG_STEREO_PROPAGATION: estimated p_1_in_0: %s\n", oss.str().c_str());
  }

  std::vector<std::shared_ptr<StereoFeatureForPropagation>> ret_vec;

  const double pos_diff_thr = params.propagation_feature_con_trans_diff_thr;
  const size_t con_thr = params.propagation_feature_n_con_thr;
  const size_t con_thr2 = params.propagation_feature_n_con_thr2;
  ASSERT(con_thr2 >= con_thr);

  int best_idx = -1;
  int best_n_con = 0;
  if (n_select > con_thr) {
    for (size_t i=0; i<n_select-con_thr; i++) {
      size_t n_con = 0;
      // for (size_t j=i+1; j<n_select; j++) {
      for (size_t j=0; j<n_select; j++) {
        if (j == i) {
          continue;
        }

        // double pos_diff = (stereo_pairs[i].p1in0 - stereo_pairs[j].p1in0).norm();
        // if (pos_diff < pos_diff_thr) {
        //   ++n_con;
        // }

        Eigen::Vector3d err = stereo_pairs[i].p1in0 - stereo_pairs[j].p1in0;
        Eigen::Matrix3d err_cov = stereo_pairs[i].p1in0_cov + stereo_pairs[j].p1in0_cov;
        double mal_square = err.transpose() * err_cov.inverse() * err;
        const double mal_dis_thr = 2.0;
        const double mal_square_thr = mal_dis_thr * mal_dis_thr;
        if (mal_square < mal_square_thr) {
          ++n_con;
        }
      }
      if (n_con > best_n_con) {
        best_idx = i;
        best_n_con = n_con;
      }
      if (best_n_con >= con_thr2) {
        break;
      }
    }
  }

  PRINT_INFO("DEBUG_STEREO_PROPAGATION: best_idx=%d, best_n_con = %d\n", best_idx, best_n_con);
  if (best_n_con < con_thr) {
    PRINT_WARNING("DEBUG_STEREO_PROPAGATION: best_n_con(%d) less than con_thr(%d)!!!!!\n", best_n_con, con_thr);
  }

  if (params.propagation_feature_force_psuedo_stationary || best_idx < 0) {
    if (params.propagation_feature_force_psuedo_stationary) {
      PRINT_INFO("DEBUG_STEREO_PROPAGATION: force stationary propagation!!\n");
    }

    const double feat_pos_cov = params.propagation_feature_psuedo_stationary_sigma * params.propagation_feature_psuedo_stationary_sigma;

    // makeup data
    auto ret = std::make_shared<StereoFeatureForPropagation>();
    ret->feat_pos_frame1 = Eigen::Vector3d(5, 5, 5);
    ret->feat_pos_frame0 = R_I1toI0 * ret->feat_pos_frame1;
    ret->feat_pos_frame0_cov = feat_pos_cov * Eigen::Matrix3d::Identity();
    ret->feat_pos_frame1_cov = feat_pos_cov * Eigen::Matrix3d::Identity();
    ret_vec.emplace_back(std::move(ret));
    return ret_vec;
  }

  // Use extrinsics to compute disparity and feature position.
  double baseline = p_C1inC0.norm();

  auto get_cov = [&](const Eigen::Vector3d& feat_in_C0) {
#if 0    
    double distance = feat_in_C0.norm();
    const double tangent_sigma = distance * bearing_sigma;
    const double radial_sigma = distance * bearing_sigma / (baseline / distance);

    Eigen::Vector3d tan_vec_0;
    if (fabs(feat_in_C0.z()) < fabs(feat_in_C0.x()) &&
        fabs(feat_in_C0.z()) < fabs(feat_in_C0.y())) {
      tan_vec_0 = Eigen::Vector3d(0,0,1);
    } else if (fabs(feat_in_C0.x()) < fabs(feat_in_C0.y())) {
      tan_vec_0 = Eigen::Vector3d(1,0,0);
    } else {
      tan_vec_0 = Eigen::Vector3d(0,1,0);
    }

    Eigen::Vector3d radial_vec = feat_in_C0 / feat_in_C0.norm();
    Eigen::Vector3d tan_vec_1 = tan_vec_0.cross(radial_vec);  // tan_vec_1 perp to radial_vec
    tan_vec_1.normalize();
    tan_vec_0 = tan_vec_1.cross(radial_vec);  // tan_vec_0 perp to feat_in_C0 and tan_vec_1
    Eigen::Matrix3d tan_rad_cov = Eigen::Matrix3d::Identity();
    tan_rad_cov(0,0) = tangent_sigma * tangent_sigma;
    tan_rad_cov(1,1) = tangent_sigma * tangent_sigma;
    tan_rad_cov(2,2) = radial_sigma * radial_sigma;

    Eigen::Matrix3d tan_rad_to_cam0;
    tan_rad_to_cam0 << tan_vec_0, tan_vec_1, radial_vec;
    Eigen::Matrix3d cov_in_cam0 = tan_rad_to_cam0 * tan_rad_cov * tan_rad_to_cam0.transpose();
#else
    double z = feat_in_C0.z();
    const double xy_sigma = z * bearing_sigma;
    const double z_sigma = z * bearing_sigma / (baseline / z);
    Eigen::Matrix3d cov_in_cam0 = Eigen::Matrix3d::Identity();
    cov_in_cam0(0,0) = xy_sigma * xy_sigma;
    cov_in_cam0(1,1) = xy_sigma * xy_sigma;
    cov_in_cam0(2,2) = z_sigma * z_sigma;
#endif

    Eigen::Matrix3d cov_in_imu = R_C0toI * cov_in_cam0 * R_C0toI.transpose();
    return cov_in_imu;
  };

  std::vector<int> inliers;
  inliers.reserve(best_n_con + 1);
  inliers.push_back(best_idx);

  // If we want to use multi-feature propagation:
  bool use_multi_feature_propagation = (params.propagation_feature_n_max_adopt > 1);
  if (use_multi_feature_propagation) {
    for (size_t j=0; j<n_select && inliers.size() < params.propagation_feature_n_max_adopt; j++) {
      if (j == best_idx) {
        continue;
      }

      // double pos_diff = (stereo_pairs[best_idx].p1in0 - stereo_pairs[j].p1in0).norm();
      // if (pos_diff < pos_diff_thr) {
      //   ++n_con;
      // }

      Eigen::Vector3d err = stereo_pairs[best_idx].p1in0 - stereo_pairs[j].p1in0;
      Eigen::Matrix3d err_cov = stereo_pairs[best_idx].p1in0_cov + stereo_pairs[j].p1in0_cov;
      double mal_square = err.transpose() * err_cov.inverse() * err;
      const double mal_dis_thr = 2.0;
      const double mal_square_thr = mal_dis_thr * mal_dis_thr;
      if (mal_square < mal_square_thr) {
        inliers.push_back(j);
      }
    }
  }

  ASSERT(prev_propagation_feat_ids.empty());
  for (const int i : inliers) {
    auto ret = std::make_shared<StereoFeatureForPropagation>();
    ret->feat_pos_frame0 = R_C0toI * stereo_pairs[i].feat_pos_frame0 + p_C0inI;
    // ret->feat_pos_frame0_cov = get_cov(stereo_pairs[i].feat_pos_frame0);
    ret->feat_pos_frame0_cov = R_C0toI * stereo_pairs[i].feat_pos_frame0_cov * R_C0toI.transpose();

    ret->feat_pos_frame1 = R_C0toI * stereo_pairs[i].feat_pos_frame1 + p_C0inI;
    // ret->feat_pos_frame1_cov = get_cov(stereo_pairs[i].feat_pos_frame1);
    ret->feat_pos_frame1_cov = R_C0toI * stereo_pairs[i].feat_pos_frame1_cov * R_C0toI.transpose();

    ret_vec.emplace_back(std::move(ret));
    prev_propagation_feat_ids.insert(stereo_pairs[best_idx].featid);
  }

  ASSERT(ret_vec.size() == inliers.size());
  PRINT_INFO("DEBUG_STEREO_PROPAGATION: number of features for propagation: %d\n", ret_vec.size());


  { // print debug info
    const auto& ret = ret_vec[0];
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(4);
    oss << "pos0: (" << ret->feat_pos_frame0.transpose();
    oss << "), cov0: ";
    oss << "(" << ret->feat_pos_frame0_cov.row(0) << ") "
        << "(" << ret->feat_pos_frame0_cov.row(1) << ") "
        << "(" << ret->feat_pos_frame0_cov.row(2) << ") ";
    PRINT_INFO("DEBUG_STEREO_PROPAGATION: pos_and_cov0: %s\n", oss.str().c_str());
  }
  { // print debug info
    const auto& ret = ret_vec[0];
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(4);
    oss << "pos1: (" << ret->feat_pos_frame1.transpose();
    oss << "), cov1: ";
    oss << "(" << ret->feat_pos_frame1_cov.row(0) << ") "
        << "(" << ret->feat_pos_frame1_cov.row(1) << ") "
        << "(" << ret->feat_pos_frame1_cov.row(2) << ") ";
    PRINT_INFO("DEBUG_STEREO_PROPAGATION: pos_and_cov1: %s\n", oss.str().c_str());
  }



  return ret_vec;
}

void VioManager::do_feature_propagate_update(ImgProcessContextPtr c) {
  ov_core::CameraData &message = *(c->message);
  auto tmp_rT2 = std::chrono::high_resolution_clock::now();

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
    if (params.propagate_with_stereo_feature) {
      propagator->propagate_and_clone_with_stereo_feature(
          state, message.timestamp,
          [this, &message](double prev_image_time, const Eigen::Matrix3d& R_I1toI0) {
            return choose_stereo_feature_for_propagation(prev_image_time, message, R_I1toI0);          
          }, &new_gyro_rotation);
    } else {
      propagator->propagate_and_clone(state, message.timestamp, &new_gyro_rotation);
    }
    update_gyro_integrated_rotations(message.timestamp, new_gyro_rotation);
  }
  c->rT3 = std::chrono::high_resolution_clock::now();

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
  maxtrack_feature_ids.clear();
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
      maxtrack_feature_ids.insert((*it2)->featid);
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
  int n_old_slam_feats = 0;
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1) {
      PRINT_WARNING(YELLOW "Marginalize outlier feature (id %d)" RESET, landmark.second->_featid);
      landmark.second->should_marg = true;
    }
    if (feat2 != nullptr && !landmark.second->should_marg) {
      feats_slam.push_back(feat2);
      n_old_slam_feats ++;
    }
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  StateHelper::marginalize_slam(state);

  // add new slam features.
  
  // prepare variables for calculating disparities
  std::map<std::shared_ptr<Feature>, double> feat_to_disparity_square;
  std::vector<double> cloned_times;
  std::vector<std::unordered_map<size_t, Eigen::Matrix3d>> R_Cold_in_Ccurs;
  std::set<std::shared_ptr<Feature>> feats_maxtracks_set;
  const auto ref_K = params.camera_intrinsics.at(message.sensor_ids[0])->get_K();
  const double ref_focallength = std::max(ref_K(0, 0), ref_K(1, 1));

  Eigen::Matrix3d cur_imu_rotation = gyro_integrated_rotations_window.at(message.timestamp);
  Eigen::Matrix3d R_Cref_in_I = params.T_CtoIs.at(message.sensor_ids[0])->block(0,0,3,3);
  // const auto & extrinsic = params.camera_extrinsics.at(message.sensor_ids[0]);
  // R_Cref_in_I = ov_core::quat_2_Rot(extrinsic.block(0,0,4,1)).transpose();
  if (params.vio_manager_high_frequency_log) {
    std::ostringstream oss;
    oss << R_Cref_in_I << std::endl;
    PRINT_ALL(YELLOW "%s" RESET, oss.str().c_str());
  }

  for (const auto &clone_imu : state->_clones_IMU) {
    cloned_times.push_back(clone_imu.first);
    const Eigen::Matrix3d& old_imu_rotation = gyro_integrated_rotations_window.at(clone_imu.first);
    Eigen::Matrix3d R_Iold_in_Icur = cur_imu_rotation.transpose() * old_imu_rotation;

    std::unordered_map<size_t, Eigen::Matrix3d> camid_to_r;
    for (const auto& item : params.T_CtoIs) {
      Eigen::Matrix3d R_C_in_I = item.second->block(0,0,3,3);
      camid_to_r [item.first] = R_Cref_in_I.transpose() * R_Iold_in_Icur * R_C_in_I;
    }
    R_Cold_in_Ccurs.push_back(camid_to_r);
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
    PRINT_DEBUG("DEBUG compute_disparity_square: computre for feats_maxtracks ...\n");
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

  CHECK_EQ(n_old_slam_feats + curr_aruco_tags, state->_features_SLAM.size());

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
      assert(cloned_times.back() == message.timestamp);
      std::vector<std::shared_ptr<Feature>> quick_feats_slam = 
          trackFEATS->get_feature_database()->features_containing(message.timestamp, false, true);
      const double disparity_thr = std::min(
        params.early_landmark_disparity_thr,
        10.0 / ref_focallength  // 10 pixels
      );
      const double disparity_square_thr = disparity_thr * disparity_thr;
      // std::cout << "DEBUG feats_slam:  initial quick_feats_slam.size() = " << quick_feats_slam.size() 
      //           << ",  state->_timestamp - message.timestamp = " << state->_timestamp - message.timestamp << std::endl;
      if (!quick_feats_slam.empty()) {
        auto it = quick_feats_slam.begin();

        PRINT_DEBUG("DEBUG compute_disparity_square: computre for potential early landmarks ...\n");
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


  if (feats_slam.size() > state->_options.max_slam_features) {
    PRINT_WARNING(YELLOW "VioManager::do_feature_propagate_update():  Might be a bug???? "
                         "We have feats_slam.size()=%d > %d=state->_options.max_slam_features!" RESET,
                         feats_slam.size(), state->_options.max_slam_features);
  }

  assert(feats_slam.size() <= state->_options.max_slam_features);

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
  
  if (params.state_options.use_rgbd) {
    // If a feature is only observed twice, then it can produce a 1d constraint for
    // the relative pose between the previous and the current frame (mono) or for the
    // extrinsics between two cameras (stereo).
    //
    // However for RGB-D mode, we're assuming the extrinsics between the main camera and the
    // virtual right camera is fixed, so a stereo feature with two observations at a single
    // image time provide almost no information. so we just remove those features which are
    // observed less than 3 times.
    //
    // Note featsup_MSCKF is already a sorted vector that features with less observations
    // are at the front.
    for (auto it=featsup_MSCKF.begin(); it != featsup_MSCKF.end(); it++) {
      size_t asize = 0;  // number of observations
      for (const auto &pair : (*it)->timestamps) {
        asize += pair.second.size();
      }
      if (asize >= 3) {
        featsup_MSCKF.erase(featsup_MSCKF.begin(), it);
        break;
      }
      if (it == featsup_MSCKF.end() - 1) {
        featsup_MSCKF.clear();
        break;
      }
    }
  }

  size_t msckf_features_used = featsup_MSCKF.size();
  size_t msckf_features_outliers = 0;
  if (!params.disable_visual_update) {
    updaterMSCKF->update(state, featsup_MSCKF);
  }
  msckf_features_outliers = msckf_features_used - featsup_MSCKF.size();
  msckf_features_used = featsup_MSCKF.size();
  c->rT4 = std::chrono::high_resolution_clock::now();

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
    if (!params.disable_visual_update) {
      updaterSLAM->update(state, featsup_TEMP);
    }
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  slam_features_outliers = slam_features_used - feats_slam_UPDATE.size();
  slam_features_used = feats_slam_UPDATE.size();
  c->rT5 = std::chrono::high_resolution_clock::now();

  size_t delayed_features_used = feats_slam_DELAYED.size();
  size_t delayed_features_outliers = 0;
  if (!params.disable_visual_update) {
    updaterSLAM->delayed_init(state, feats_slam_DELAYED);
  }
  delayed_features_outliers = delayed_features_used - feats_slam_DELAYED.size();
  delayed_features_used = feats_slam_DELAYED.size();
  c->rT6 = std::chrono::high_resolution_clock::now();


  if (params.propagate_with_stereo_feature && params.grivaty_update_after_propagate_with_stereo_feature) {
    propagator->gravity_update(state);
  }

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  // note(jeffrey): we assume 'sensor_ids' is a sorted vector (small id frist), and there is always a
  //                camera with sensor_id=0;
  // std::cout << "DEBUG sensor_ids: " << message.sensor_ids[0] << ", " << message.sensor_ids[1] << std::endl;
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
  {
    std::unique_lock<std::mutex> lck(trackFEATS->get_feature_database()->get_mutex());

    for (auto const &feat : featsup_MSCKF) {
      good_features_MSCKF.push_back(feat->p_FinG);
      good_feature_ids_MSCKF.push_back(feat->featid);

      feat->to_delete = true;
      // feat->clean_older_measurements(message.timestamp);
    }
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
  c->rT7 = std::chrono::high_resolution_clock::now();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_mask = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT1 - c->rT0).count();
  double time_track = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT2 - c->rT1).count();
  double time_switch_thread = std::chrono::duration_cast<std::chrono::duration<double>>(tmp_rT2 - c->rT2).count();
  double time_prop = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT3 - tmp_rT2).count();
  double time_msckf = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT4 - c->rT3).count();
  double time_slam_update = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT5 - c->rT4).count();
  double time_slam_delay = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT6 - c->rT5).count();
  double time_marg = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT7 - c->rT6).count();
  double time_total = std::chrono::duration_cast<std::chrono::duration<double>>(c->rT7 - c->rT0).count();

  // Timing information
  PRINT_INFO(GREEN "[used_features_and_time]: msckf(%d + %d, %.4f), slam(%d + %d, %.4f), delayed(%d + %d, %.4f), total(%d + %d, %.4f), timestampe: %.6f\n" RESET,
                    msckf_features_used, msckf_features_outliers, time_msckf,
                    slam_features_used, slam_features_outliers, time_slam_update,
                    delayed_features_used, delayed_features_outliers, time_slam_delay,
                    msckf_features_used + slam_features_used + delayed_features_used,
                    msckf_features_outliers + slam_features_outliers + delayed_features_outliers,
                    time_msckf + time_slam_update + time_slam_delay,
                    message.timestamp
                    );

  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for semantic masking\n" RESET, time_mask);
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

  // // Finally if we are saving stats to file, lets save it to file
  // if (params.record_timing_information && of_statistics.is_open()) {
  //   // We want to publish in the IMU clock frame
  //   // The timestamp in the state will be the last camera time
  //   double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  //   double timestamp_inI = state->_timestamp + t_ItoC;
  //   // Append to the file
  //   of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
  //                 << time_prop << "," << time_msckf << ",";
  //   if (state->_options.max_slam_features > 0) {
  //     of_statistics << time_slam_update << "," << time_slam_delay << ",";
  //   }
  //   of_statistics << time_marg << "," << time_total << std::endl;
  //   of_statistics.flush();
  // }

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
    const std::vector<std::unordered_map<size_t, Eigen::Matrix3d>>& R_Cold_in_Ccurs,
    size_t ref_cam_id) {

  bool is_stereo = false;
  // bool only_ref_cam = true;
  bool only_ref_cam = false;
  int total_obs = 0;

  std::unordered_map<size_t, std::vector<double>> camid_to_feat_times;
  std::unordered_map<size_t, std::vector<Eigen::VectorXf>> camid_to_feat_uvs_norm;
  std::unordered_map<size_t, std::vector<Eigen::VectorXf>> camid_to_feat_uvs;
  {
    std::unique_lock<std::mutex> lck(trackFEATS->get_feature_database()->get_mutex());
    if (!feat->timestamps.count(ref_cam_id)) {
      return 0.0;
    }

    if (!only_ref_cam) {
      is_stereo = (feat->uvs.size() > 1);
    }

    // only compute disparities for stereo features possessing 2 or more observations.
    if (is_stereo && feat->timestamps.at(ref_cam_id).size() < 2) {
      return 0.0;
    }

    // only compute disparities for mono features possessing 3 or more observations.
    if (!is_stereo && feat->timestamps.at(ref_cam_id).size() < 3) {
      return 0.0;
    }
    
    // feat_times = feat->timestamps.at(ref_cam_id);
    // feat_uvs_norm = feat->uvs_norm.at(ref_cam_id);
    // if (params.vio_manager_high_frequency_log) {
    //   feat_uvs = feat->uvs.at(ref_cam_id);
    // }
    
    camid_to_feat_times = feat->timestamps;
    camid_to_feat_uvs_norm = feat->uvs_norm;
    if (params.vio_manager_high_frequency_log) {
      camid_to_feat_uvs = feat->uvs;
    }
  }
  assert(!camid_to_feat_times.empty());
  assert(!camid_to_feat_times.at(ref_cam_id).empty());


  std::unordered_map<size_t, std::vector<int>> camid_to_indices;
  Eigen::Vector2f cur_uv_norm;
  Eigen::Vector2f cur_uv;  // for debug only. = feat_uvs[indices.back()];
  double cur_time;
  for (const auto & item : params.T_CtoIs) {
    const auto & camid = item.first;
    if (only_ref_cam && camid != ref_cam_id) {
      continue;
    }
    if (!camid_to_feat_times.count(camid)) {
      continue;
    }
    const auto & feat_times = camid_to_feat_times.at(camid);

    camid_to_indices[camid] = std::vector<int>(cloned_times.size());    
    std::vector<int>& indices = camid_to_indices[camid];
    int j = 0;
    for (size_t i=0; i<cloned_times.size(); i++) {
      const auto& clone_time = cloned_times[i];
      while (j < feat_times.size() && feat_times[j] < clone_time) {
        j ++;
      }

      if (camid == ref_cam_id) {
        assert(j < feat_times.size());  // "j >= feat_times.size()" shouldn't happen for ref_cam.
      }

      if (j >=  feat_times.size()) {
        indices[i] = -1;
      } else if (feat_times[j] > clone_time) {
        indices[i] = -1;
      } else {
        assert(feat_times[j] == clone_time);
        indices[i] = j;
      }
    }

    if (camid == ref_cam_id) {
      if (indices.empty() || indices.back() < 0 || feat_times[indices.back()] != cloned_times.back()) {
        std::unique_lock<std::mutex> lck(trackFEATS->get_feature_database()->get_mutex());

        PRINT_WARNING(YELLOW "compute_disparity_square():  indices.empty() || indices.back() < 0 || "
                             "feat_times[indices.back()] == cloned_times.back() !!\n" RESET);
        PRINT_WARNING(YELLOW "compute_disparity_square():  current state time: %.3f.   ref-cam feat n_observation = %d / %d.  ",
                      cloned_times.back(), feat->timestamps.at(ref_cam_id).size(), feat->uvs_norm.at(ref_cam_id).size());
        printf("ref-cam feat times (feat_time - state_time) :  ");
        for (const auto& time : feat->timestamps.at(ref_cam_id)) {
          printf("%.3f   ", time - cloned_times.back());
        }
        printf(".  feature-id=%d, is_stereo=%d\n" RESET, feat->featid, is_stereo);
        PRINT_WARNING(YELLOW "compute_disparity_square():  all feat times of feature [%d] (feat_time - state_time) :  ", feat->featid);
        for (const auto & pair : feat->timestamps) {
          const auto & camid = pair.first;
          printf("| cam%d:  ", camid);
          for (const auto& time : feat->timestamps.at(camid)) {
            printf("%.3f   ", time - cloned_times.back());
          }
        }
        printf("\n" RESET);

        //// It's not a bug. observations from left camera might be used and deleted before
        //// its right counterpart become available.
        ////
        // PRINT_WARNING(YELLOW "compute_disparity_square():  Might be a bug?? " RESET);
        // assert(!indices.empty());  // report the bug
        // assert(indices.back() >= 0);
        // assert(feat_times[indices.back()] == cloned_times.back());

        return 0.0;
      }

      cur_uv_norm = camid_to_feat_uvs_norm.at(ref_cam_id)[indices.back()];
      if (params.vio_manager_high_frequency_log) {
        cur_uv = camid_to_feat_uvs.at(ref_cam_id)[indices.back()];
        cur_time = feat_times[indices.back()];
      }
    }
  }

  Eigen::Vector3d cur_homo(cur_uv_norm.x(), cur_uv_norm.y(), 1.0);
  double cur_homo_square = cur_homo.dot(cur_homo);
  double max_disparity_square = 0.0;

  // for debug
  size_t best_camid;
  Eigen::Vector2f best_old_uv;
  Eigen::Vector2f best_old_uv_norm;
  Eigen::Vector2f best_predict_uv_norm;
  Eigen::Matrix3d best_R_Cold_in_Ccur;
  double best_old_time;

  for (const auto & item : params.T_CtoIs) {
    const auto & camid = item.first;
    if (only_ref_cam && camid != ref_cam_id) {
      continue;
    }

    if (!camid_to_feat_times.count(camid) ||
        !camid_to_feat_uvs_norm.count(camid) ||
        !camid_to_indices.count(camid)) {
      continue;
    }

    const auto & feat_times = camid_to_feat_times.at(camid);
    const auto & feat_uvs_norm = camid_to_feat_uvs_norm.at(camid);
    const auto & indices = camid_to_indices.at(camid);

    for (size_t i=0; i<cloned_times.size() - 1; i++) {
      // otherwise, feat_times[j] == clone_time
      int j = indices[i];
      if (j < 0) {
        continue;
      }
      assert(cloned_times[i] == feat_times[j]);
      const Eigen::Vector2f& old_uv_norm = feat_uvs_norm[j];
      const Eigen::Matrix3d& R_Cold_in_Ccur = R_Cold_in_Ccurs[i].at(camid);
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
      total_obs ++;
      if (disparity_square > max_disparity_square) {
        max_disparity_square = disparity_square;
        if (params.vio_manager_high_frequency_log) {
          const auto & feat_uvs = camid_to_feat_uvs.at(camid);
          best_camid = camid;
          best_old_uv_norm = old_uv_norm;
          best_R_Cold_in_Ccur = R_Cold_in_Ccur;
          best_old_uv = feat_uvs[j];
          best_predict_uv_norm = Eigen::Vector2f(predict_homo.x() / predict_homo.z(), predict_homo.y() / predict_homo.z());
          best_old_time = feat_times[j];
        }
      }
    }
  }


  if (params.vio_manager_high_frequency_log) {
    std::ostringstream oss;
    // Eigen::Vector2f predict_uv = params.camera_intrinsics.at(ref_cam_id)->distort_f(Eigen::Vector2f(best_predict_uv_norm.x(), best_predict_uv_norm.y()));
    Eigen::Vector2f predict_uv = params.camera_intrinsics.at(best_camid)->distort_f(Eigen::Vector2f(best_predict_uv_norm.x(), best_predict_uv_norm.y()));
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

  // Only output non-zero disparities for features with at least three observations.
  if (total_obs < 3) {
    return 0.0;
  }
  return max_disparity_square;
}
