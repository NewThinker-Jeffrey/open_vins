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

#include "UpdaterSLAM.h"

#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "feat/FeatureDatabase.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/chi_square/chi_squared_quantile_table_0_95.h"

#include <chrono>
// #include <boost/math/distributions/chi_squared.hpp>

#ifdef USE_HEAR_SLAM
#include "hear_slam/basic/logging.h"
#include "hear_slam/basic/time.h"
#include "hear_slam/basic/thread_pool.h"
#include "hear_slam/basic/work_queue.h"
#define PARALLEL_SLAM_UPDATE
#endif


using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterSLAM::UpdaterSLAM(UpdaterOptions &options_slam, UpdaterOptions &options_mappoint, UpdaterOptions &options_aruco, ov_core::FeatureInitializerOptions &feat_init_options, std::shared_ptr<ov_core::FeatureDatabase> db)
    : _options_slam(options_slam), _options_mappoint(options_mappoint), _options_aruco(options_aruco), _db(db) {

  // Save our raw pixel noise squared
  _options_slam.sigma_pix_sq = std::pow(_options_slam.sigma_pix, 2);
  _options_mappoint.sigma_pix_sq = std::pow(_options_mappoint.sigma_pix, 2);
  _options_aruco.sigma_pix_sq = std::pow(_options_aruco.sigma_pix, 2);

#ifdef PARALLEL_SLAM_UPDATE
  if (!landmark_initialzing_queue) {
    landmark_initialzing_queue.reset(new hear_slam::TaskQueue("ov_landmk_init"));
  }
#endif

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  // for (int i = 1; i < 500; i++) {
  //   boost::math::chi_squared chi_squared_dist(i);
  //   chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  // }
}

void UpdaterSLAM::delayed_init(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

#ifdef USE_HEAR_SLAM
  hear_slam::TimeCounter tc;
#endif

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  std::chrono::high_resolution_clock::time_point rT0, rT1, rT2, rT3;
  rT0 = std::chrono::high_resolution_clock::now();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }
  double max_clonetime = *std::max_element(clonetimes.begin(), clonetimes.end());

  // create cloned features containing no 'future' observations.
  std::map<std::shared_ptr<Feature>, std::shared_ptr<Feature>> cloned_features;

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {
    {
      std::unique_lock<std::mutex> lck(_db->get_mutex());
      // Clean the feature
      (*it0)->clean_old_measurements(clonetimes);
      cloned_features[*it0] = std::make_shared<Feature>(*(*it0));
    }
    auto& cloned_feature = cloned_features[*it0];
    cloned_feature->clean_future_measurements(max_clonetime);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : cloned_feature->timestamps) {
      // ct_meas += (*it0)->timestamps[pair.first].size();
      ct_meas += cloned_feature->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {
      {
        std::unique_lock<std::mutex> lck(_db->get_mutex());
        (*it0)->to_delete = true;
      }
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = std::chrono::high_resolution_clock::now();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  if (state->_options.use_rgbd) {
    assert(state->_calib_IMUtoCAM.size() == 1);
    const auto &clone_calib = *(state->_calib_IMUtoCAM.begin());
    assert(clone_calib.first == 0);
    size_t virtual_rightcamera_id = 1;

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_camr;
    for (const auto &clone_imu : state->_clones_IMU) {
      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCr = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CroinG = clone_imu.second->pos() - R_GtoCr.transpose() * clone_calib.second->pos();
      p_CroinG += R_GtoCr.transpose() * Eigen::Vector3d(state->_options.virtual_baseline_for_rgbd, 0, 0);

      // Append to our map
      clones_camr.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCr, p_CroinG)});
    }

    // Append to our map
    clones_cam.insert({virtual_rightcamera_id, clones_camr});
  }

#ifdef USE_HEAR_SLAM
  tc.tag("prepareDone");
#endif

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  auto triang_one = [&](decltype(feature_vec.begin()) it1) {
    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    auto& cloned_feature = cloned_features[(*it1)];
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(cloned_feature, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(cloned_feature, clones_cam);
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(cloned_feature, clones_cam);
    }

    {
      // copy back the triangulaion result
      std::unique_lock<std::mutex> lck(_db->get_mutex());
      (*it1)->anchor_cam_id = cloned_feature->anchor_cam_id;
      (*it1)->anchor_clone_timestamp = cloned_feature->anchor_clone_timestamp;
      (*it1)->p_FinA = cloned_feature->p_FinA;
      (*it1)->p_FinG = cloned_feature->p_FinG;
    }

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      {
        std::unique_lock<std::mutex> lck(_db->get_mutex());
        (*it1)->to_delete = true;
      }
      // it1 = feature_vec.erase(it1);    // we'll erase it later.
      // continue;
    }
  };

#ifdef PARALLEL_SLAM_UPDATE
  {
    auto pool = hear_slam::ThreadPool::getNamed("ov_visual_updt");
    int n_workers = pool->numThreads();
    int segment_size = feature_vec.size() / n_workers + 1;
    using IterType = decltype(feature_vec.begin());
    auto process_segment = [&](IterType begin, IterType end) {
      auto it = begin;
      while (it != end) {
        triang_one(it);
        it++;
      }
    };
    for (size_t i=0; i<n_workers; i++) {
      if (i * segment_size < feature_vec.size()) {
        IterType begin = feature_vec.begin() + i * segment_size;
        IterType end;
        if ((i + 1) * segment_size < feature_vec.size()) {
          end = feature_vec.begin() + (i + 1) * segment_size;
        } else {
          end = feature_vec.end();
        }
        pool->schedule([&process_segment, begin, end](){process_segment(begin, end);});
      }
    }
    pool->waitUntilAllTasksDone();
  }
#else
  {
    auto it1 = feature_vec.begin();
    while (it1 != feature_vec.end()) {
      triang_one(it1);
      it1++;
    }    
  }
#endif
  {
    auto it1 = feature_vec.begin();
    while (it1 != feature_vec.end()) {
      if ((*it1)->to_delete) {
        it1 = feature_vec.erase(it1);
      } else {
        it1++;
      }
    }
  }

  rT2 = std::chrono::high_resolution_clock::now();

#ifdef USE_HEAR_SLAM
  tc.tag("triangDone");
#endif

  // 4. Compute linear system for each feature, nullspace project, and reject
  auto compute_one = [&](decltype(feature_vec.begin()) it2) {
    // Convert our feature into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    auto& cloned_feature = cloned_features[*it2];
    feat.featid = cloned_feature->featid;
    feat.uvs = cloned_feature->uvs;
    feat.uvs_norm = cloned_feature->uvs_norm;
    feat.timestamps = cloned_feature->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    auto feat_rep =
        ((int)feat.featid < state->_options.max_aruco_features) ? state->_options.feat_rep_aruco : state->_options.feat_rep_slam;
    feat.feat_representation = feat_rep;
    if (feat_rep == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = cloned_feature->anchor_cam_id;
      feat.anchor_clone_timestamp = cloned_feature->anchor_clone_timestamp;
      feat.p_FinA = cloned_feature->p_FinA;
      feat.p_FinA_fej = cloned_feature->p_FinA;
    } else {
      feat.p_FinG = cloned_feature->p_FinG;
      feat.p_FinG_fej = cloned_feature->p_FinG;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    double absolute_residual_thr =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.absolute_residual_thr : _options_slam.absolute_residual_thr;    
    if (absolute_residual_thr > 0.0) {
      const double absolute_residual_square_thr = absolute_residual_thr * absolute_residual_thr;
      std::cout << "absolute_residuals: ";
      double max_error_square = 0.0;
      for (size_t i=0; i<res.size()/2; i++) {
        double& px = res[2*i];
        double& py = res[2*i+1];
        double error_square = px * px + py * py;
        if (error_square > max_error_square) {
          max_error_square = error_square;
        }
        std::cout << sqrt(error_square) << "  ";
        if (error_square > absolute_residual_square_thr) {
          break;
        }
      }
      std::cout << std::endl;
      if (max_error_square > absolute_residual_square_thr) {
        {
          std::unique_lock<std::mutex> lck(_db->get_mutex());      
          (*it2)->to_delete = true;
        }
        // it2 = feature_vec.erase(it2);  // we'll erase it later
        std::cout << "absolute_residual_check_failed!! res = " << res.transpose() << std::endl;
        // continue;
        return;
      }
    }
  

    // If we are doing the single feature representation, then we need to remove the bearing portion
    // To do so, we project the bearing portion onto the state and depth Jacobians and the residual.
    // This allows us to directly initialize the feature as a depth-old feature
    if (feat_rep == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

      // Append the Jacobian in respect to the depth of the feature
      Eigen::MatrixXd H_xf = H_x;
      H_xf.conservativeResize(H_x.rows(), H_x.cols() + 1);
      H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0, H_f.cols() - 1, H_f.rows(), 1);
      H_f.conservativeResize(H_f.rows(), H_f.cols() - 1);

      // Nullspace project the bearing portion
      // This takes into account that we have marginalized the bearing already
      // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
      UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

      // Split out the state portion and feature portion
      H_x = H_xf.block(0, 0, H_xf.rows(), H_xf.cols() - 1);
      H_f = H_xf.block(0, H_xf.cols() - 1, H_xf.rows(), 1);
    }

    // Create feature pointer (we will always create it of size three since we initialize the single invese depth as a msckf anchored
    // representation)
    int landmark_size = (feat_rep == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 1 : 3;
    auto landmark = std::make_shared<Landmark>(landmark_size);
    landmark->_featid = feat.featid;
    landmark->_feat_representation = feat_rep;
    landmark->_unique_camera_id = cloned_feature->anchor_cam_id;
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      landmark->_anchor_cam_id = feat.anchor_cam_id;
      landmark->_anchor_clone_timestamp = feat.anchor_clone_timestamp;
      landmark->set_from_xyz(feat.p_FinA, false);
      landmark->set_from_xyz(feat.p_FinA_fej, true);
    } else {
      landmark->set_from_xyz(feat.p_FinG, false);
      landmark->set_from_xyz(feat.p_FinG_fej, true);
    }

    // Measurement noise matrix
    double sigma_pix_sq =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
    Eigen::MatrixXd R = sigma_pix_sq * Eigen::MatrixXd::Identity(res.rows(), res.rows());

    // Try to initialize, delete new pointer if we failed
    double chi2_multipler =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
    auto run_initialize = [=]() mutable {
      if (StateHelper::initialize(state, landmark, Hx_order, H_x, H_f, R, res, chi2_multipler)) {
        state->_features_SLAM.insert({cloned_feature->featid, landmark});
        {
          std::unique_lock<std::mutex> lck(_db->get_mutex());      
          // (*it2)->to_delete = true;
          (*it2)->clean_older_measurements(max_clonetime);
        }
        // it2++;
      } else {
        {
          std::unique_lock<std::mutex> lck(_db->get_mutex());      
          (*it2)->to_delete = true;
        }
        // it2 = feature_vec.erase(it2);  // we'll erase it later.
      }
    };
#ifdef PARALLEL_SLAM_UPDATE
    landmark_initialzing_queue->enqueue(run_initialize);
#else
    run_initialize();
#endif
  };


#ifdef PARALLEL_SLAM_UPDATE
  {
    auto pool = hear_slam::ThreadPool::getNamed("ov_visual_updt");
    int n_workers = pool->numThreads();
    int segment_size = feature_vec.size() / n_workers + 1;
    using IterType = decltype(feature_vec.begin());
    auto process_segment = [&](IterType begin, IterType end) {
      auto it = begin;
      while (it != end) {
        compute_one(it);
        it++;
      }
    };
    for (size_t i=0; i<n_workers; i++) {
      if (i * segment_size < feature_vec.size()) {
        IterType begin = feature_vec.begin() + i * segment_size;
        IterType end;
        if ((i + 1) * segment_size < feature_vec.size()) {
          end = feature_vec.begin() + (i + 1) * segment_size;
        } else {
          end = feature_vec.end();
        }
        pool->schedule([&process_segment, begin, end](){process_segment(begin, end);});
      }
    }
    pool->waitUntilAllTasksDone();
  }

  landmark_initialzing_queue->waitUntilAllJobsDone();
#else
  {
    auto it2 = feature_vec.begin();
    while (it2 != feature_vec.end()) {
      compute_one(it2);
      it2++;
    }
  }
#endif
  {
    auto it2 = feature_vec.begin();
    while (it2 != feature_vec.end()) {
      if ((*it2)->to_delete) {
        it2 = feature_vec.erase(it2);
      } else {
        it2++;
      }
    }
  }

#ifdef USE_HEAR_SLAM
  tc.tag("allDone");
  tc.report("delayInitUpTiming: ", true);
#endif

  rT3 = std::chrono::high_resolution_clock::now();

  // Debug print timing information
  if (!feature_vec.empty()) {
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds to clean\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT1 - rT0).count());
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds to triangulate\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT2 - rT1).count());
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds initialize (%d features)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT2).count(), (int)feature_vec.size());
    PRINT_ALL("[SLAM-DELAY]: %.4f seconds total\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT1).count());
  }
}

void UpdaterSLAM::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {


#ifdef USE_HEAR_SLAM
  hear_slam::TimeCounter tc;
#endif

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  std::chrono::high_resolution_clock::time_point rT0, rT1, rT2, rT3;
  rT0 = std::chrono::high_resolution_clock::now();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }
  double max_clonetime = *std::max_element(clonetimes.begin(), clonetimes.end());

  // create cloned features containing no 'future' observations.
  std::map<std::shared_ptr<Feature>, std::shared_ptr<Feature>> cloned_features;

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {
    {
      std::unique_lock<std::mutex> lck(_db->get_mutex());
      // Clean the feature
      (*it0)->clean_old_measurements(clonetimes);
      cloned_features[*it0] = std::make_shared<Feature>(*(*it0));
    }
    auto& cloned_feature = cloned_features[*it0];
    cloned_feature->clean_future_measurements(max_clonetime);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : cloned_feature->timestamps) {
      // ct_meas += (*it0)->timestamps[pair.first].size();
      ct_meas += cloned_feature->timestamps[pair.first].size();
    }

    // Get the landmark and its representation
    // For single depth representation we need at least two measurement
    // This is because we do nullspace projection
    std::shared_ptr<Landmark> landmark = state->_features_SLAM.at(cloned_feature->featid);
    int required_meas = (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 2 : 1;

    // Remove if we don't have enough
    if (ct_meas < 1) {
      {
        std::unique_lock<std::mutex> lck(_db->get_mutex());
        (*it0)->to_delete = true;
      }
      it0 = feature_vec.erase(it0);
    } else if (ct_meas < required_meas) {
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = std::chrono::high_resolution_clock::now();

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    // for (const auto &pair : feature_vec.at(i)->timestamps) {
    //   max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    // }
    for (const auto &pair : cloned_features[feature_vec.at(i)]->timestamps) {
      max_meas_size += 2 * cloned_features[feature_vec.at(i)]->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  size_t max_hx_size = state->max_covariance_size();

  // Large Jacobian, residual, and measurement noise of *all* features for this update
  std::mutex mutex_big;
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size, max_meas_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

#ifdef USE_HEAR_SLAM
  tc.tag("prepareDone");
#endif


  // 4. Compute linear system for each feature, nullspace project, and reject
  auto compute_one = [&](decltype(feature_vec.begin()) it2) {
    auto& cloned_feature = cloned_features[*it2];

    // Ensure we have the landmark and it is the same
    assert(state->_features_SLAM.find(cloned_feature->featid) != state->_features_SLAM.end());
    assert(state->_features_SLAM.at(cloned_feature->featid)->_featid == cloned_feature->featid);

    // Get our landmark from the state
    std::shared_ptr<Landmark> landmark = state->_features_SLAM.at(cloned_feature->featid);

    // Convert the state landmark into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = cloned_feature->featid;
    feat.uvs = cloned_feature->uvs;
    feat.uvs_norm = cloned_feature->uvs_norm;
    feat.timestamps = cloned_feature->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = landmark->_feat_representation;
    if (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = landmark->_anchor_cam_id;
      feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
      feat.p_FinA = landmark->get_xyz(false);
      feat.p_FinA_fej = landmark->get_xyz(true);
    } else {
      feat.p_FinG = landmark->get_xyz(false);
      feat.p_FinG_fej = landmark->get_xyz(true);
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    double absolute_residual_thr =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.absolute_residual_thr : _options_slam.absolute_residual_thr;
    if (absolute_residual_thr > 0.0) {
      const double absolute_residual_square_thr = absolute_residual_thr * absolute_residual_thr;
      std::cout << "absolute_residuals: ";
      double max_error_square = 0.0;
      for (size_t i=0; i<res.size()/2; i++) {
        double& px = res[2*i];
        double& py = res[2*i+1];
        double error_square = px * px + py * py;
        if (error_square > max_error_square) {
          max_error_square = error_square;
        }
        std::cout << sqrt(error_square) << "  ";
        if (error_square > absolute_residual_square_thr) {
          break;
        }
      }
      std::cout << std::endl;
      if (max_error_square > absolute_residual_square_thr) {
        if ((int)feat.featid < state->_options.max_aruco_features) {
          PRINT_WARNING(YELLOW "[SLAM-UP]: rejecting aruco tag %d for absolute_residual_check (%.3f > %.3f)\n" RESET, (int)feat.featid, sqrt(max_error_square),
                        absolute_residual_thr);
        } else {
          landmark->update_fail_count++;
        }
        {
          std::unique_lock<std::mutex> lck(_db->get_mutex());
          (*it2)->to_delete = true;
        }
        // it2 = feature_vec.erase(it2);  // we'll erase it later.
        std::cout << "absolute_residual_check_failed!! res = " << res.transpose() << std::endl;
        // continue;
        return;
      }
    }

    // Place Jacobians in one big Jacobian, since the landmark is already in our state vector
    Eigen::MatrixXd H_xf = H_x;
    if (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

      // Append the Jacobian in respect to the depth of the feature
      H_xf.conservativeResize(H_x.rows(), H_x.cols() + 1);
      H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0, H_f.cols() - 1, H_f.rows(), 1);
      H_f.conservativeResize(H_f.rows(), H_f.cols() - 1);

      // Nullspace project the bearing portion
      // This takes into account that we have marginalized the bearing already
      // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
      UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

    } else {

      // Else we have the full feature in our state, so just append it
      H_xf.conservativeResize(H_x.rows(), H_x.cols() + H_f.cols());
      H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;
    }

    // Append to our Jacobian order vector
    std::vector<std::shared_ptr<Type>> Hxf_order = Hx_order;
    Hxf_order.push_back(landmark);

    // Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hxf_order);
    Eigen::MatrixXd S = H_xf * P_marg * H_xf.transpose();
    double sigma_pix_sq =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
    S.diagonal() += sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    // if (res.rows() < 500) {
    //   chi2_check = chi_squared_table[res.rows()];
    // } else {
    //   boost::math::chi_squared chi_squared_dist(res.rows());
    //   chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
    //   PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    // }
    chi2_check = ::chi_squared_quantile_table_0_95[res.rows()];

    // Check if we should delete or not
    double chi2_multipler =
        ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
    if (chi2 > chi2_multipler * chi2_check) {
      if ((int)feat.featid < state->_options.max_aruco_features) {
        PRINT_WARNING(YELLOW "[SLAM-UP]: rejecting aruco tag %d for chi2 thresh (%.3f > %.3f)\n" RESET, (int)feat.featid, chi2,
                      chi2_multipler * chi2_check);
      } else {
        landmark->update_fail_count++;
      }
      {
        std::unique_lock<std::mutex> lck(_db->get_mutex());
        (*it2)->to_delete = true;
      }
      // it2 = feature_vec.erase(it2);  // we'll erase it later.
      // continue;
      return;
    }

    // Debug print when we are going to update the aruco tags
    if ((int)feat.featid < state->_options.max_aruco_features) {
      PRINT_DEBUG("[SLAM-UP]: accepted aruco tag %d for chi2 thresh (%.3f < %.3f)\n", (int)feat.featid, chi2, chi2_multipler * chi2_check);
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;

#ifdef PARALLEL_SLAM_UPDATE  // #ifdef USE_HEAR_SLAM
    std::unique_lock<std::mutex> lck(mutex_big);
#endif
    for (const auto &var : Hxf_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_xf.rows(), var->size()) = H_xf.block(0, ct_hx, H_xf.rows(), var->size());
      ct_hx += var->size();
    }

    // Our isotropic measurement noise
    R_big.block(ct_meas, ct_meas, res.rows(), res.rows()) *= sigma_pix_sq;

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();    
  };

#ifdef PARALLEL_SLAM_UPDATE  // #ifdef USE_HEAR_SLAM
  {
    auto pool = hear_slam::ThreadPool::getNamed("ov_visual_updt");
    int n_workers = pool->numThreads();
    int segment_size = feature_vec.size() / n_workers + 1;
    PRINT_INFO("ov_visual_updt: n_workers=%d, segment_size=%d, feature_vec.size()=%d\n",
               n_workers, segment_size, feature_vec.size());
    using IterType = decltype(feature_vec.begin());
    auto process_segment = [&](IterType begin, IterType end) {
      auto it = begin;
      while (it != end) {
        compute_one(it);
        it++;
      }
    };
    for (size_t i=0; i<n_workers; i++) {
      if (i * segment_size < feature_vec.size()) {
        IterType begin = feature_vec.begin() + i * segment_size;
        IterType end;
        if ((i + 1) * segment_size < feature_vec.size()) {
          end = feature_vec.begin() + (i + 1) * segment_size;
        } else {
          end = feature_vec.end();
        }
        pool->schedule([&process_segment, begin, end](){process_segment(begin, end);});
      }
    }
    pool->waitUntilAllTasksDone();
  }
#else
  {
    auto it2 = feature_vec.begin();
    while (it2 != feature_vec.end()) {
      compute_one(it2);
      it2++;
    }
  }
#endif
  {
    auto it2 = feature_vec.begin();
    while (it2 != feature_vec.end()) {
      if ((*it2)->to_delete) {
        it2 = feature_vec.erase(it2);
      } else {
        it2++;
      }
    }
  }

  rT2 = std::chrono::high_resolution_clock::now();

#ifdef USE_HEAR_SLAM
  tc.tag("JacobianDone");
#endif

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  {
    std::unique_lock<std::mutex> lck(_db->get_mutex());
    for (size_t f = 0; f < feature_vec.size(); f++) {
      // feature_vec[f]->to_delete = true;
      feature_vec[f]->clean_older_measurements(max_clonetime);
    }
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);
  R_big.conservativeResize(ct_meas, ct_meas);

  // 5. With all good SLAM features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT3 = std::chrono::high_resolution_clock::now();

#ifdef USE_HEAR_SLAM
  tc.tag("EKFDone");
  tc.report("SlamUpTiming: ", true);
#endif

  // Debug print timing information
  PRINT_ALL("[SLAM-UP]: %.4f seconds to clean\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT1 - rT0).count());
  PRINT_ALL("[SLAM-UP]: %.4f seconds creating linear system\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT2 - rT1).count());
  PRINT_ALL("[SLAM-UP]: %.4f seconds to update (%d feats of %d size)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT2).count(), (int)feature_vec.size(),
            (int)Hx_big.rows());
  PRINT_ALL("[SLAM-UP]: %.4f seconds total\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT1).count());
}

void UpdaterSLAM::change_anchors(std::shared_ptr<State> state) {

  // Return if we do not have enough clones
  if ((int)state->_clones_IMU.size() <= state->_options.max_clone_size) {
    return;
  }

  // Get the marginalization timestep, and change the anchor for any feature seen from it
  // NOTE: for now we have anchor the feature in the same camera as it is before
  // NOTE: this also does not change the representation of the feature at all right now
  double marg_timestep = state->margtimestep();
  for (auto &f : state->_features_SLAM) {
    // Skip any features that are in the global frame
    if (f.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D ||
        f.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH)
      continue;
    // Else lets see if it is anchored in the clone that will be marginalized
    assert(marg_timestep <= f.second->_anchor_clone_timestamp);
    if (f.second->_anchor_clone_timestamp == marg_timestep) {
      perform_anchor_change(state, f.second, state->_timestamp, f.second->_anchor_cam_id);
    }
  }
}

void UpdaterSLAM::perform_anchor_change(std::shared_ptr<State> state, std::shared_ptr<Landmark> landmark, double new_anchor_timestamp,
                                        size_t new_cam_id) {

  // Assert that this is an anchored representation
  assert(LandmarkRepresentation::is_relative_representation(landmark->_feat_representation));
  assert(landmark->_anchor_cam_id != -1);

  // Create current feature representation
  UpdaterHelper::UpdaterHelperFeature old_feat;
  old_feat.featid = landmark->_featid;
  old_feat.feat_representation = landmark->_feat_representation;
  old_feat.anchor_cam_id = landmark->_anchor_cam_id;
  old_feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
  old_feat.p_FinA = landmark->get_xyz(false);
  old_feat.p_FinA_fej = landmark->get_xyz(true);

  // Get Jacobians of p_FinG wrt old representation
  Eigen::MatrixXd H_f_old;
  std::vector<Eigen::MatrixXd> H_x_old;
  std::vector<std::shared_ptr<Type>> x_order_old;
  UpdaterHelper::get_feature_jacobian_representation(state, old_feat, H_f_old, H_x_old, x_order_old);

  // Create future feature representation
  UpdaterHelper::UpdaterHelperFeature new_feat;
  new_feat.featid = landmark->_featid;
  new_feat.feat_representation = landmark->_feat_representation;
  new_feat.anchor_cam_id = new_cam_id;
  new_feat.anchor_clone_timestamp = new_anchor_timestamp;

  //==========================================================================
  //==========================================================================

  // OLD: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoIOLD = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->Rot();
  Eigen::Matrix<double, 3, 3> R_GtoOLD = state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->Rot() * R_GtoIOLD;
  Eigen::Matrix<double, 3, 1> p_OLDinG = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->pos() -
                                         R_GtoOLD.transpose() * state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->pos();

  // NEW: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoINEW = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->Rot();
  Eigen::Matrix<double, 3, 3> R_GtoNEW = state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->Rot() * R_GtoINEW;
  Eigen::Matrix<double, 3, 1> p_NEWinG = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->pos() -
                                         R_GtoNEW.transpose() * state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->pos();

  // Calculate transform between the old anchor and new one
  Eigen::Matrix<double, 3, 3> R_OLDtoNEW = R_GtoNEW * R_GtoOLD.transpose();
  Eigen::Matrix<double, 3, 1> p_OLDinNEW = R_GtoNEW * (p_OLDinG - p_NEWinG);
  new_feat.p_FinA = R_OLDtoNEW * landmark->get_xyz(false) + p_OLDinNEW;

  //==========================================================================
  //==========================================================================

  // OLD: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoIOLD_fej = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->Rot_fej();
  Eigen::Matrix<double, 3, 3> R_GtoOLD_fej = state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->Rot() * R_GtoIOLD_fej;
  Eigen::Matrix<double, 3, 1> p_OLDinG_fej = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->pos_fej() -
                                             R_GtoOLD_fej.transpose() * state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->pos();

  // NEW: anchor camera position and orientation
  Eigen::Matrix<double, 3, 3> R_GtoINEW_fej = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->Rot_fej();
  Eigen::Matrix<double, 3, 3> R_GtoNEW_fej = state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->Rot() * R_GtoINEW_fej;
  Eigen::Matrix<double, 3, 1> p_NEWinG_fej = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->pos_fej() -
                                             R_GtoNEW_fej.transpose() * state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->pos();

  // Calculate transform between the old anchor and new one
  Eigen::Matrix<double, 3, 3> R_OLDtoNEW_fej = R_GtoNEW_fej * R_GtoOLD_fej.transpose();
  Eigen::Matrix<double, 3, 1> p_OLDinNEW_fej = R_GtoNEW_fej * (p_OLDinG_fej - p_NEWinG_fej);
  new_feat.p_FinA_fej = R_OLDtoNEW_fej * landmark->get_xyz(true) + p_OLDinNEW_fej;

  // Get Jacobians of p_FinG wrt new representation
  Eigen::MatrixXd H_f_new;
  std::vector<Eigen::MatrixXd> H_x_new;
  std::vector<std::shared_ptr<Type>> x_order_new;
  UpdaterHelper::get_feature_jacobian_representation(state, new_feat, H_f_new, H_x_new, x_order_new);

  //==========================================================================
  //==========================================================================

  // New phi order is just the landmark
  std::vector<std::shared_ptr<Type>> phi_order_NEW;
  phi_order_NEW.push_back(landmark);

  // Loop through all our orders and append them
  std::vector<std::shared_ptr<Type>> phi_order_OLD;
  int current_it = 0;
  std::map<std::shared_ptr<Type>, int> Phi_id_map;
  for (const auto &var : x_order_old) {
    if (Phi_id_map.find(var) == Phi_id_map.end()) {
      Phi_id_map.insert({var, current_it});
      phi_order_OLD.push_back(var);
      current_it += var->size();
    }
  }
  for (const auto &var : x_order_new) {
    if (Phi_id_map.find(var) == Phi_id_map.end()) {
      Phi_id_map.insert({var, current_it});
      phi_order_OLD.push_back(var);
      current_it += var->size();
    }
  }
  Phi_id_map.insert({landmark, current_it});
  phi_order_OLD.push_back(landmark);
  current_it += landmark->size();

  // Anchor change Jacobian
  int phisize = (new_feat.feat_representation != LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
  Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(phisize, current_it);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(phisize, phisize);

  // Inverse of our new representation
  // pf_new_error = Hfnew^{-1}*(Hfold*pf_olderror+Hxold*x_olderror-Hxnew*x_newerror)
  Eigen::MatrixXd H_f_new_inv;
  if (phisize == 1) {
    H_f_new_inv = 1.0 / H_f_new.squaredNorm() * H_f_new.transpose();
  } else {
    H_f_new_inv = H_f_new.colPivHouseholderQr().solve(Eigen::Matrix<double, 3, 3>::Identity());
  }

  // Place Jacobians for old anchor
  for (size_t i = 0; i < H_x_old.size(); i++) {
    Phi.block(0, Phi_id_map.at(x_order_old[i]), phisize, x_order_old[i]->size()).noalias() += H_f_new_inv * H_x_old[i];
  }

  // Place Jacobians for old feat
  Phi.block(0, Phi_id_map.at(landmark), phisize, phisize) = H_f_new_inv * H_f_old;

  // Place Jacobians for new anchor
  for (size_t i = 0; i < H_x_new.size(); i++) {
    Phi.block(0, Phi_id_map.at(x_order_new[i]), phisize, x_order_new[i]->size()).noalias() -= H_f_new_inv * H_x_new[i];
  }

  // Perform covariance propagation
  StateHelper::EKFPropagation(state, phi_order_NEW, phi_order_OLD, Phi, Q);

  // Set state from new feature
  landmark->_featid = new_feat.featid;
  landmark->_feat_representation = new_feat.feat_representation;
  landmark->_anchor_cam_id = new_feat.anchor_cam_id;
  landmark->_anchor_clone_timestamp = new_feat.anchor_clone_timestamp;
  landmark->set_from_xyz(new_feat.p_FinA, false);
  landmark->set_from_xyz(new_feat.p_FinA_fej, true);
  landmark->has_had_anchor_change = true;
}

void UpdaterSLAM::mappoint_update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec, const FeatToMappointMatches& featid_to_mappoint) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  std::chrono::high_resolution_clock::time_point rT0, rT1, rT2, rT3;
  rT0 = std::chrono::high_resolution_clock::now();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }
  double max_clonetime = *std::max_element(clonetimes.begin(), clonetimes.end());

  // create cloned features containing no 'future' observations.
  std::map<std::shared_ptr<Feature>, std::shared_ptr<Feature>> cloned_features;

// std::cout << "DEBUG_mappoint_udpate: 0, feature_vec.size = " << feature_vec.size() << ", featid_to_mappoint.size = " << featid_to_mappoint.size() << std::endl;


  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {
    {
      std::unique_lock<std::mutex> lck(_db->get_mutex());
      // Clean the feature
      (*it0)->clean_old_measurements(clonetimes);
      cloned_features[*it0] = std::make_shared<Feature>(*(*it0));
    }
    auto& cloned_feature = cloned_features[*it0];
    cloned_feature->clean_future_measurements(max_clonetime);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : cloned_feature->timestamps) {
      // ct_meas += (*it0)->timestamps[pair.first].size();
      ct_meas += cloned_feature->timestamps[pair.first].size();
    }

    // // Get the landmark and its representation
    // // For single depth representation we need at least two measurement
    // // This is because we do nullspace projection
    // std::shared_ptr<Landmark> landmark = state->_features_SLAM.at(cloned_feature->featid);
    // int required_meas = (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 2 : 1;
    int required_meas = 1;

    // Remove if we don't have enough
    if (ct_meas < 1) {
      {
        std::unique_lock<std::mutex> lck(_db->get_mutex());
        (*it0)->to_delete = true;
      }
      it0 = feature_vec.erase(it0);
    } else if (ct_meas < required_meas) {
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }

// std::cout << "DEBUG_mappoint_udpate: 1, feature_vec.size = " << feature_vec.size() << ", featid_to_mappoint.size = " << featid_to_mappoint.size() << std::endl;

  rT1 = std::chrono::high_resolution_clock::now();

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    // for (const auto &pair : feature_vec.at(i)->timestamps) {
    //   max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    // }
    for (const auto &pair : cloned_features[feature_vec.at(i)]->timestamps) {
      max_meas_size += 2 * cloned_features[feature_vec.at(i)]->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  size_t max_hx_size = state->max_covariance_size();

  // Large Jacobian, residual, and measurement noise of *all* features for this update
  std::mutex mutex_big;
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  // Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size, max_meas_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

// std::cout << "DEBUG_mappoint_udpate: 2, max_meas_size = " << max_meas_size << ", max_hx_size = " << max_hx_size << std::endl;


  // 4. Compute linear system for each feature, nullspace project, and reject
  auto compute_one = [&](decltype(feature_vec.begin()) it2) {
    auto& cloned_feature = cloned_features[*it2];
    auto feat_id = cloned_feature->featid;
    if (featid_to_mappoint.count(feat_id) == 0) {
      {
        std::unique_lock<std::mutex> lck(_db->get_mutex());
        (*it2)->to_delete = true;
      }
      // it2 = feature_vec.erase(it2);  // we'll erase it later.
      // continue;
      return;
    }

    // // Ensure we have the landmark and it is the same
    // assert(state->_features_SLAM.find(cloned_feature->featid) != state->_features_SLAM.end());
    // assert(state->_features_SLAM.at(cloned_feature->featid)->_featid == cloned_feature->featid);

    // // Get our landmark from the state
    // std::shared_ptr<Landmark> landmark = state->_features_SLAM.at(cloned_feature->featid);

    // Convert the state landmark into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = feat_id;
    feat.uvs = cloned_feature->uvs;
    feat.uvs_norm = cloned_feature->uvs_norm;
    feat.timestamps = cloned_feature->timestamps;

    // // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    // feat.feat_representation = landmark->_feat_representation;
    // if (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
    //   feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    // }
    feat.feat_representation = ov_type::LandmarkRepresentation::Representation::GLOBAL_3D;

    // // Save the position and its fej value
    // if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
    //   feat.anchor_cam_id = landmark->_anchor_cam_id;
    //   feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
    //   feat.p_FinA = landmark->get_xyz(false);
    //   feat.p_FinA_fej = landmark->get_xyz(true);
    // } else {
    //   feat.p_FinG = landmark->get_xyz(false);
    //   feat.p_FinG_fej = landmark->get_xyz(true);
    // }

    const Eigen::Vector3d& mappoint = featid_to_mappoint.at(feat_id).p;
    const Eigen::Matrix3d& mappoint_cov = featid_to_mappoint.at(feat_id).cov;

    feat.p_FinG = mappoint;
    feat.p_FinG_fej = mappoint;

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // double absolute_residual_thr =
    //     ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.absolute_residual_thr : _options_slam.absolute_residual_thr;
    double absolute_residual_thr = _options_mappoint.absolute_residual_thr;
    if (absolute_residual_thr > 0.0) {
      const double absolute_residual_square_thr = absolute_residual_thr * absolute_residual_thr;
      std::cout << "absolute_residuals: ";
      double max_error_square = 0.0;
      for (size_t i=0; i<res.size()/2; i++) {
        double& px = res[2*i];
        double& py = res[2*i+1];
        double error_square = px * px + py * py;
        if (error_square > max_error_square) {
          max_error_square = error_square;
        }
        std::cout << sqrt(error_square) << "  ";
        if (error_square > absolute_residual_square_thr) {
          break;
        }
      }
      std::cout << std::endl;
      if (max_error_square > absolute_residual_square_thr) {
        // if ((int)feat.featid < state->_options.max_aruco_features) {
        //   PRINT_WARNING(YELLOW "[SLAM-UP]: rejecting aruco tag %d for absolute_residual_check (%.3f > %.3f)\n" RESET, (int)feat.featid, sqrt(max_error_square),
        //                 absolute_residual_thr);
        // } else {
        //   landmark->update_fail_count++;
        // }
        {
          std::unique_lock<std::mutex> lck(_db->get_mutex());
          (*it2)->to_delete = true;
        }
        // it2 = feature_vec.erase(it2);  // we'll erase it later.
        std::cout << "absolute_residual_check_failed!! res = " << res.transpose() << std::endl;
        // continue;
        return;
      }
    }

    // Place Jacobians in one big Jacobian, since the landmark is already in our state vector
    // Eigen::MatrixXd H_xf = H_x;

    // if (landmark->_feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
    //   // Append the Jacobian in respect to the depth of the feature
    //   H_xf.conservativeResize(H_x.rows(), H_x.cols() + 1);
    //   H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0, H_f.cols() - 1, H_f.rows(), 1);
    //   H_f.conservativeResize(H_f.rows(), H_f.cols() - 1);

    //   // Nullspace project the bearing portion
    //   // This takes into account that we have marginalized the bearing already
    //   // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
    //   UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

    // } else {

    //   // Else we have the full feature in our state, so just append it
    //   H_xf.conservativeResize(H_x.rows(), H_x.cols() + H_f.cols());
    //   H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;
    // }

    // Append to our Jacobian order vector
    // std::vector<std::shared_ptr<Type>> Hxf_order = Hx_order;
    // Hxf_order.push_back(landmark);

    // Chi2 distance check
    // Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hxf_order);
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    
    // double sigma_pix_sq =
    //     ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
    double sigma_pix_sq = _options_mappoint.sigma_pix_sq;
    Eigen::MatrixXd R = H_f * mappoint_cov * H_f.transpose();
    R.diagonal() += sigma_pix_sq * Eigen::VectorXd::Ones(R.rows());
    S += R;
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    // if (res.rows() < 500) {
    //   chi2_check = chi_squared_table[res.rows()];
    // } else {
    //   boost::math::chi_squared chi_squared_dist(res.rows());
    //   chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
    //   PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    // }
    chi2_check = ::chi_squared_quantile_table_0_95[res.rows()];

    // Check if we should delete or not
    // double chi2_multipler =
    //     ((int)feat.featid < state->_options.max_aruco_features) ? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
    double chi2_multipler = _options_mappoint.chi2_multipler;
    if (chi2 > chi2_multipler * chi2_check) {
      // if ((int)feat.featid < state->_options.max_aruco_features) {
      //   PRINT_WARNING(YELLOW "[SLAM-UP]: rejecting aruco tag %d for chi2 thresh (%.3f > %.3f)\n" RESET, (int)feat.featid, chi2,
      //                 chi2_multipler * chi2_check);
      // } else {
      //   landmark->update_fail_count++;
      // }
      {
        std::unique_lock<std::mutex> lck(_db->get_mutex());
        (*it2)->to_delete = true;
      }
      // it2 = feature_vec.erase(it2);  // we'll erase it later.
      // continue;
      return;
    }

    Eigen::MatrixXd sqrt_info = R.llt().matrixL().solve(Eigen::MatrixXd::Identity(R.rows(), R.cols()));
    Eigen::MatrixXd normalized_Hx =  sqrt_info* H_x;
    Eigen::MatrixXd normalized_res =  sqrt_info* res;

    // std::cout << "DEBUG_mappoint_udpate: H_x.cols() = " << H_x.cols() << std::endl;

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
#ifdef PARALLEL_SLAM_UPDATE  // #ifdef USE_HEAR_SLAM
    std::unique_lock<std::mutex> lck(mutex_big);
#endif
    // for (const auto &var : Hxf_order) {
    for (const auto &var : Hx_order) {
      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      // Hx_big.block(ct_meas, Hx_mapping[var], H_xf.rows(), var->size()) = H_xf.block(0, ct_hx, H_xf.rows(), var->size());
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = normalized_Hx.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = normalized_res;
    ct_meas += res.rows();
    // it2++;
  };

#ifdef PARALLEL_SLAM_UPDATE  // #ifdef USE_HEAR_SLAM
  {
    auto pool = hear_slam::ThreadPool::getNamed("ov_visual_updt");
    int n_workers = pool->numThreads();
    int segment_size = feature_vec.size() / n_workers + 1;
    using IterType = decltype(feature_vec.begin());
    auto process_segment = [&](IterType begin, IterType end) {
      auto it = begin;
      while (it != end) {
        compute_one(it);
        it++;
      }
    };
    for (size_t i=0; i<n_workers; i++) {
      if (i * segment_size < feature_vec.size()) {
        IterType begin = feature_vec.begin() + i * segment_size;
        IterType end;
        if ((i + 1) * segment_size < feature_vec.size()) {
          end = feature_vec.begin() + (i + 1) * segment_size;
        } else {
          end = feature_vec.end();
        }
        pool->schedule([&process_segment, begin, end](){process_segment(begin, end);});
      }
    }
    pool->waitUntilAllTasksDone();
  }
#else
  {
    auto it2 = feature_vec.begin();
    while (it2 != feature_vec.end()) {
      compute_one(it2);
      it2++;
    }
  }
#endif
  {
    auto it2 = feature_vec.begin();
    while (it2 != feature_vec.end()) {
      if ((*it2)->to_delete) {
        it2 = feature_vec.erase(it2);
      } else {
        it2++;
      }
    }
  }

  rT2 = std::chrono::high_resolution_clock::now();

std::cout << "DEBUG_mappoint_udpate: 3, feature_vec.size = " << feature_vec.size() << ", featid_to_mappoint.size = " << featid_to_mappoint.size() << std::endl;

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  {
    std::unique_lock<std::mutex> lck(_db->get_mutex());
    for (size_t f = 0; f < feature_vec.size(); f++) {
      // feature_vec[f]->to_delete = true;
      feature_vec[f]->clean_older_measurements(max_clonetime);
    }
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);

std::cout << "DEBUG_mappoint_udpate: 4, ct_meas = " << ct_meas << ", ct_jacob = " << ct_jacob << std::endl;

  // 5. Perform measurement compression
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);

std::cout << "DEBUG_mappoint_udpate: 5, Hx_big.rows() = " << Hx_big.rows() << std::endl;

  if (Hx_big.rows() < 1) {
    return;
  }

  // Our noise is isotropic, so make it here after our compression
  Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

  // 6. With all good mappoint features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);

// std::cout << "DEBUG_mappoint_udpate: 6" << std::endl;

  rT3 = std::chrono::high_resolution_clock::now();

  // Debug print timing information
  PRINT_ALL("[MAPPOINT-UP]: %.4f seconds to clean\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT1 - rT0).count());
  PRINT_ALL("[MAPPOINT-UP]: %.4f seconds creating linear system\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT2 - rT1).count());
  PRINT_ALL("[MAPPOINT-UP]: %.4f seconds to update (%d feats of %d size)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT2).count(), (int)feature_vec.size(),
            (int)Hx_big.rows());
  PRINT_ALL("[MAPPOINT-UP]: %.4f seconds total\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT1).count());
}
