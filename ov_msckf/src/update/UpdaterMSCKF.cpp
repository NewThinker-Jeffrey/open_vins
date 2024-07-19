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

#include "UpdaterMSCKF.h"

#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "feat/FeatureDatabase.h"
#include "state/State.h"
#include "state/StateHelper.h"
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
// #define PARALLEL_MSCKF_UPDATE
#endif

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterMSCKF::UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options, std::shared_ptr<ov_core::FeatureDatabase> db) : _options(options), _db(db) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  // for (int i = 1; i < 500; i++) {
  //   boost::math::chi_squared chi_squared_dist(i);
  //   chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  //   // std::cout << "chi_squared_table  " << i << ":  " << chi_squared_table[i] << std::endl;
  // }
}

void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

#ifdef USE_HEAR_SLAM
  hear_slam::TimeCounter tc;
#endif

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  std::chrono::high_resolution_clock::time_point rT0, rT1, rT2, rT3, rT4, rT5;
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
      // it1 = feature_vec.erase(it1);  // we'll erase it later.
      // continue;
    }
    // it1++;
  };

#ifdef PARALLEL_MSCKF_UPDATE
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
  // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
  size_t max_hx_size = state->max_covariance_size();
  for (auto &landmark : state->_features_SLAM) {
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  std::mutex mutex_big;
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

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
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
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
    
    if (_options.absolute_residual_thr > 0.0) {
      const double absolute_residual_square_thr = _options.absolute_residual_thr * _options.absolute_residual_thr;
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
        // it2 = feature_vec.erase(it2);  // we'll erase it later.
        std::cout << "absolute_residual_check_failed!! res = " << res.transpose() << std::endl;
        // continue;
        return;
      }
    }

    // Nullspace project
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

    /// Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
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
    if (chi2 > _options.chi2_multipler * chi2_check) {
      {
        std::unique_lock<std::mutex> lck(_db->get_mutex());
        (*it2)->to_delete = true;
      }
      // it2 = feature_vec.erase(it2);  // we'll erase it later.

      // PRINT_DEBUG("featid = %d\n", feat.featid);
      // PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler*chi2_check);
      // std::stringstream ss;
      // ss << "res = " << std::endl << res.transpose() << std::endl;
      // PRINT_DEBUG(ss.str().c_str());

      // continue;
      return;
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
#ifdef PARALLEL_MSCKF_UPDATE  // #ifdef USE_HEAR_SLAM
    std::unique_lock<std::mutex> lck(mutex_big);
#endif
    for (const auto &var : Hx_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    // it2++;
  };

#ifdef PARALLEL_MSCKF_UPDATE  // #ifdef USE_HEAR_SLAM
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


  rT3 = std::chrono::high_resolution_clock::now();
#ifdef USE_HEAR_SLAM
  tc.tag("JacobianDone");
#endif

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  {
    std::unique_lock<std::mutex> lck(_db->get_mutex());
    for (size_t f = 0; f < feature_vec.size(); f++) {
      feature_vec[f]->to_delete = true;
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

  // 5. Perform measurement compression
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
  if (Hx_big.rows() < 1) {
    return;
  }
  rT4 = std::chrono::high_resolution_clock::now();

#ifdef USE_HEAR_SLAM
  tc.tag("CompressDone");
#endif

  // Our noise is isotropic, so make it here after our compression
  Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

  // 6. With all good features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT5 = std::chrono::high_resolution_clock::now();

#ifdef USE_HEAR_SLAM
  tc.tag("EKFDone");
  tc.report("MSCKFUpTiming: ", true);
#endif


  // Debug print timing information
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT1 - rT0).count());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT2 - rT1).count());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT2).count(), (int)feature_vec.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT4 - rT3).count());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT5 - rT4).count(), (int)res_big.rows());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT5 - rT1).count());
}
