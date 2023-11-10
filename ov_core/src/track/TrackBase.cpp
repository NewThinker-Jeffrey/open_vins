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

#include "TrackBase.h"

#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/quat_ops.h"
#include "utils/ransac_helper.h"
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <algorithm>

using namespace ov_core;

TrackBase::TrackBase(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                     HistogramMethod histmethod, std::map<size_t, Eigen::VectorXd> inp_camera_extrinsics,
                     bool keypoint_predict, bool high_frequency_log)
    : camera_calib(cameras), database(new FeatureDatabase()), num_features(numfeats), 
      use_stereo(stereo), histogram_method(histmethod), 
      t_d(0), gyro_bias(0,0,0), enable_high_frequency_log(high_frequency_log), enable_keypoint_predict(keypoint_predict) {

  for (const auto & item : inp_camera_extrinsics) {
    camera_extrinsics[item.first] = Eigen::Isometry3d::Identity();
    camera_extrinsics[item.first].linear() = quat_2_Rot(item.second.block(0,0,4,1));
    camera_extrinsics[item.first].translation() = item.second.block(4,0,3,1);
    camera_extrinsics[item.first] = camera_extrinsics[item.first].inverse();  // todo(isaac): check the transformation.
    {
      std::ostringstream oss;
      oss << "TrackBase::TrackBase():  extrinsics for camera " << item.first << ": " << std::endl;
      oss << camera_extrinsics[item.first].matrix() << std::endl;
      PRINT_INFO("%s", oss.str().c_str());
    }
  }

  // Our current feature ID should be larger then the number of aruco tags we have (each has 4 corners)
  currid = 4 * (size_t)numaruco + 1;
  // Create our mutex array based on the number of cameras we have
  // See https://stackoverflow.com/a/24170141/7718197
  if (mtx_feeds.empty() || mtx_feeds.size() != camera_calib.size()) {
    std::vector<std::mutex> list(camera_calib.size());
    mtx_feeds.swap(list);
  }
}

void TrackBase::display_active(double timestamp, cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::string overlay) {

  // Cache the images to prevent other threads from editing while we viz (which can be slow)
  std::map<size_t, cv::Mat> img_last_cache, img_mask_last_cache;
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> pts_last_cache;
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last_cache = history_vars.at(timestamp).img;
    img_mask_last_cache = history_vars.at(timestamp).img_mask;
    pts_last_cache = history_vars.at(timestamp).pts;
  }

  // Get the largest width and height
  int max_width = -1;
  int max_height = -1;
  for (auto const &pair : img_last_cache) {
    if (max_width < pair.second.cols)
      max_width = pair.second.cols;
    if (max_height < pair.second.rows)
      max_height = pair.second.rows;
  }

  // Return if we didn't have a last image
  if (img_last_cache.empty() || max_width == -1 || max_height == -1)
    return;

  // If the image is "small" thus we should use smaller display codes
  bool is_small = (std::min(max_width, max_height) < 400);

  // If the image is "new" then draw the images from scratch
  // Otherwise, we grab the subset of the main image and draw on top of it
  bool image_new = ((int)img_last_cache.size() * max_width != img_out.cols || max_height != img_out.rows);

  // If new, then resize the current image
  if (image_new)
    img_out = cv::Mat(max_height, (int)img_last_cache.size() * max_width, CV_8UC3, cv::Scalar(0, 0, 0));

  // Loop through each image, and draw
  int index_cam = 0;
  for (auto const &pair : img_last_cache) {
    // select the subset of the image
    cv::Mat img_temp;
    if (image_new)
      cv::cvtColor(img_last_cache[pair.first], img_temp, cv::COLOR_GRAY2RGB);
    else
      img_temp = img_out(cv::Rect(max_width * index_cam, 0, max_width, max_height));
    // draw, loop through all keypoints
    for (size_t i = 0; i < pts_last_cache[pair.first].size(); i++) {
      // Get bounding pts for our boxes
      cv::Point2f pt_l = pts_last_cache[pair.first].at(i).pt;
      // Draw the extracted points and ID
      cv::circle(img_temp, pt_l, (is_small) ? 1 : 2, cv::Scalar(r1, g1, b1), cv::FILLED);
      // cv::putText(img_out, std::to_string(ids_left_last.at(i)), pt_l, cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),1,cv::LINE_AA);
      // Draw rectangle around the point
      cv::Point2f pt_l_top = cv::Point2f(pt_l.x - 3, pt_l.y - 3);
      cv::Point2f pt_l_bot = cv::Point2f(pt_l.x + 3, pt_l.y + 3);
      cv::rectangle(img_temp, pt_l_top, pt_l_bot, cv::Scalar(r2, g2, b2), 1);
    }
    // Draw what camera this is
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);
    if (overlay == "") {
      cv::putText(img_temp, "CAM:" + std::to_string((int)pair.first), txtpt, cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0,
                  cv::Scalar(0, 255, 0), 3);
    } else {
      cv::putText(img_temp, overlay, txtpt, cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0, cv::Scalar(0, 0, 255), 3);
    }
    // Overlay the mask
    cv::Mat mask = cv::Mat::zeros(img_mask_last_cache[pair.first].rows, img_mask_last_cache[pair.first].cols, CV_8UC3);
    mask.setTo(cv::Scalar(0, 0, 255), img_mask_last_cache[pair.first]);
    cv::addWeighted(mask, 0.1, img_temp, 1.0, 0.0, img_temp);
    // Replace the output image
    img_temp.copyTo(img_out(cv::Rect(max_width * index_cam, 0, img_last_cache[pair.first].cols, img_last_cache[pair.first].rows)));
    index_cam++;
  }
}

void TrackBase::display_history(double timestamp, cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::vector<size_t> highlighted,
                                std::string overlay) {

  // Cache the images to prevent other threads from editing while we viz (which can be slow)
  std::map<size_t, cv::Mat> img_last_cache, img_mask_last_cache;
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> pts_last_cache;
  std::unordered_map<size_t, std::vector<size_t>> ids_last_cache;
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last_cache = history_vars.at(timestamp).img;
    img_mask_last_cache = history_vars.at(timestamp).img_mask;
    pts_last_cache = history_vars.at(timestamp).pts;
    ids_last_cache = history_vars.at(timestamp).ids;
  }

  // Get the largest width and height
  int max_width = -1;
  int max_height = -1;
  for (auto const &pair : img_last_cache) {
    if (max_width < pair.second.cols)
      max_width = pair.second.cols;
    if (max_height < pair.second.rows)
      max_height = pair.second.rows;
  }

  // Return if we didn't have a last image
  if (img_last_cache.empty() || max_width == -1 || max_height == -1)
    return;

  // If the image is "small" thus we shoudl use smaller display codes
  // bool is_small = (std::min(max_width, max_height) < 400);
  bool is_small = (std::min(max_width, max_height) < 600);

  // If the image is "new" then draw the images from scratch
  // Otherwise, we grab the subset of the main image and draw on top of it
  bool image_new = ((int)img_last_cache.size() * max_width != img_out.cols || max_height != img_out.rows);

  // If new, then resize the current image
  if (image_new)
    img_out = cv::Mat(max_height, (int)img_last_cache.size() * max_width, CV_8UC3, cv::Scalar(0, 0, 0));

  // Max tracks to show (otherwise it clutters up the screen)
  size_t maxtracks = 50;
  // size_t maxtracks = 11;

  // get time_str
  std::string time_str;
  {
    bool use_utc_time = false;  // use local time
    int64_t ts = int64_t(timestamp*1e9);
    ts += 8 * 3600 * 1e9;  // add 8 hours
    std::chrono::nanoseconds time_since_epoch(ts);
    std::chrono::time_point
        <std::chrono::system_clock, std::chrono::nanoseconds>
        time_point(time_since_epoch);
    std::time_t tt = std::chrono::system_clock::to_time_t(time_point);
    struct tm* ptm;
    if (use_utc_time) {
      ptm = gmtime(&tt);
    } else {
      ptm = localtime(&tt);
    }
    struct tm& tm = *ptm;
    int64_t sub_seconds_in_nano = ts % 1000000000;
    char dt[100];
    sprintf(  // NOLINT
        dt, "%04d-%02d-%02d %02d:%02d:%02d", tm.tm_year + 1900, tm.tm_mon + 1,
        tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    // sprintf(dt, "%02d:%02d:%02d", tm.tm_hour, tm.tm_min, tm.tm_sec);
    time_str = std::string(dt) + " " + std::to_string(sub_seconds_in_nano/1000000);
  }

  // Loop through each image, and draw
  int index_cam = 0;
  for (auto const &pair : img_last_cache) {
    // select the subset of the image
    cv::Mat img_temp;
    if (image_new)
      cv::cvtColor(img_last_cache[pair.first], img_temp, cv::COLOR_GRAY2RGB);
    else
      img_temp = img_out(cv::Rect(max_width * index_cam, 0, max_width, max_height));

    // draw, loop through all keypoints

    bool highlighted_only = false;

    // bool no_tail = true;
    bool no_tail = false;

    for (size_t i = 0; i < ids_last_cache[pair.first].size(); i++) {
      // Get the feature from the database
      Feature feat;
      if (!database->get_feature_clone(ids_last_cache[pair.first].at(i), feat, true))
        continue;
      feat.clean_future_measurements(timestamp);

      // If a highlighted point, then put a nice box around it
      if (std::find(highlighted.begin(), highlighted.end(), ids_last_cache[pair.first].at(i)) != highlighted.end()) {
        cv::Point2f pt_c = pts_last_cache[pair.first].at(i).pt;
        // cv::Point2f pt_l_top = cv::Point2f(pt_c.x - ((is_small) ? 3 : 5), pt_c.y - ((is_small) ? 3 : 5));
        // cv::Point2f pt_l_bot = cv::Point2f(pt_c.x + ((is_small) ? 3 : 5), pt_c.y + ((is_small) ? 3 : 5));
        cv::Point2f pt_l_top = cv::Point2f(pt_c.x - ((is_small) ? 5 : 7), pt_c.y - ((is_small) ? 5 : 7));
        cv::Point2f pt_l_bot = cv::Point2f(pt_c.x + ((is_small) ? 5 : 7), pt_c.y + ((is_small) ? 5 : 7));
        bool is_stereo = (feat.uvs.size() > 1);
        cv::Scalar color = is_stereo ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);
        cv::rectangle(img_temp, pt_l_top, pt_l_bot, color, 1);
        // cv::circle(img_temp, pt_c, (is_small) ? 1 : 2, color, cv::FILLED);
        cv::circle(img_temp, pt_c, (is_small) ? 2 : 3, color, cv::FILLED);
      } else if (highlighted_only) {
        continue;
      }

      if (no_tail) {
        continue;
      }

      // if (feat.uvs.empty() || feat.uvs[pair.first].empty() || feat.to_delete)
      if (feat.uvs.empty() || feat.uvs[pair.first].empty())
        continue;
      // Draw the history of this point (start at the last inserted one)
      for (size_t z = feat.uvs[pair.first].size() - 1; z > 0; z--) {
        // Check if we have reached the max
        if (feat.uvs[pair.first].size() - z > maxtracks)
          break;
        // Calculate what color we are drawing in
        bool is_stereo = (feat.uvs.size() > 1);
        int color_r = (is_stereo ? b2 : r2) - (int)(1.0 * (is_stereo ? b1 : r1) / feat.uvs[pair.first].size() * z);
        int color_g = (is_stereo ? r2 : g2) - (int)(1.0 * (is_stereo ? r1 : g1) / feat.uvs[pair.first].size() * z);
        int color_b = (is_stereo ? g2 : b2) - (int)(1.0 * (is_stereo ? g1 : b1) / feat.uvs[pair.first].size() * z);
        // Draw current point
        cv::Point2f pt_c(feat.uvs[pair.first].at(z)(0), feat.uvs[pair.first].at(z)(1));
        cv::circle(img_temp, pt_c, (is_small) ? 1 : 2, cv::Scalar(color_r, color_g, color_b), cv::FILLED);
        // If there is a next point, then display the line from this point to the next
        if (z + 1 < feat.uvs[pair.first].size()) {
          cv::Point2f pt_n(feat.uvs[pair.first].at(z + 1)(0), feat.uvs[pair.first].at(z + 1)(1));
          cv::line(img_temp, pt_c, pt_n, cv::Scalar(color_r, color_g, color_b));
        }
        // If the first point, display the ID
        if (z == feat.uvs[pair.first].size() - 1) {
          // cv::putText(img_out0, std::to_string(feat->featid), pt_c, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1,
          // cv::LINE_AA); cv::circle(img_out0, pt_c, 2, cv::Scalar(color,color,255), CV_FILLED);
        }
      }
    }
    // Draw what camera this is
    auto txtpt = (is_small) ? cv::Point(10, 30) : cv::Point(30, 60);


    if (index_cam == 0) {
      cv::putText(img_temp, time_str, txtpt,
                  cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0,
                  cv::Scalar(0, 255, 0), (is_small) ? 2.0 : 3);
      if (overlay != "") {
        cv::putText(img_temp, overlay, txtpt + cv::Point(0, (is_small) ? 30 : 60),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, (is_small) ? 1.5 : 3.0,
                    cv::Scalar(0, 0, 255), (is_small) ? 2.0 : 3);
      }
    }

    // Overlay the mask
    cv::Mat mask = cv::Mat::zeros(img_mask_last_cache[pair.first].rows, img_mask_last_cache[pair.first].cols, CV_8UC3);
    mask.setTo(cv::Scalar(0, 0, 255), img_mask_last_cache[pair.first]);
    cv::addWeighted(mask, 0.1, img_temp, 1.0, 0.0, img_temp);
    // Replace the output image
    img_temp.copyTo(img_out(cv::Rect(max_width * index_cam, 0, img_last_cache[pair.first].cols, img_last_cache[pair.first].rows)));
    index_cam++;
  }
}

void TrackBase::change_feat_id(size_t id_old, size_t id_new) {

  // If found in db then replace
  database->change_feat_id(id_old, id_new);

  // Update current track IDs
  //// TODO(isaac): also update the history_vars.ids (for visualization)
  for (auto &cam_ids_pair : ids_last) {
    for (size_t i = 0; i < cam_ids_pair.second.size(); i++) {
      if (cam_ids_pair.second.at(i) == id_old) {
        ids_last.at(cam_ids_pair.first).at(i) = id_new;
      }
    }
  }
}

Eigen::Matrix3d TrackBase::integrate_gryo(double old_time, double new_time) {
  if (enable_high_frequency_log) {
    PRINT_ALL("DEBUG integrate_gryo: old_time=%f,   new_time=%f,   new-old=%f,  t_d=%f\n", old_time, new_time, new_time - old_time, t_d);  
  }
  double old_imu_time = old_time + t_d;
  double new_imu_time = new_time + t_d;
  std::vector<ImuData> prop_data;
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    prop_data = select_imu_readings(imu_data, old_imu_time, new_imu_time);
  }
  if (prop_data.size() < 2) {
    PRINT_WARNING(YELLOW "TrackBase::integrate_gryo(): no prop_data!\n" RESET);
    return Eigen::Matrix3d::Identity();
  }

  // check time gap
  double max_time_gap = 0.021;
  for (size_t i=0; i<prop_data.size()-1; i++) {
    const auto & d0 = prop_data[i];
    const auto & d1 = prop_data[i+1];
    double dt = d1.timestamp - d0.timestamp;
    if (dt > max_time_gap) {
      PRINT_WARNING(YELLOW "TrackBase::integrate_gryo(): large imu gap (%.3f)!\n" RESET, dt);
      return Eigen::Matrix3d::Identity();
    }
  }

  prop_data = fill_imu_data_gaps(prop_data, 0.0051);

  Eigen::Quaterniond q(1,0,0,0);
  for (size_t i=0; i<prop_data.size()-1; i++) {
    const auto & d0 = prop_data[i];
    const auto & d1 = prop_data[i+1];
    double dt = d1.timestamp - d0.timestamp;
    Eigen::Vector3d w_ave = 0.5 * (d0.wm + d1.wm) - gyro_bias;
    Eigen::Vector3d im = 0.5 * w_ave * dt;    
    Eigen::Quaterniond dq(1, im.x(), im.y(), im.z());
    q = q * dq;
  }
  q.normalize();
  return q.toRotationMatrix();
}

Eigen::Matrix3d TrackBase::predict_rotation(size_t cam_id, double new_time) {
  if (img_time_last.count(cam_id) == 0) {
    PRINT_WARNING(YELLOW "TrackBase::predict_rotation(): no previous frame!\n" RESET);
    return Eigen::Matrix3d::Identity();
  }

  double old_time = img_time_last.at(cam_id);
  Eigen::Matrix3d R_I1_in_I0 = integrate_gryo(old_time, new_time);
  Eigen::Matrix3d R_C_in_I = camera_extrinsics[cam_id].linear();
  Eigen::Matrix3d R_C1_in_C0 = R_C_in_I.transpose() * R_I1_in_I0 * R_C_in_I;
  Eigen::Matrix3d R_C0_in_C1 = R_C1_in_C0.transpose();
  return R_C0_in_C1;
}

void TrackBase::predict_keypoints(
  size_t cam_id0, size_t cam_id1, const std::vector<cv::KeyPoint>& kpts0, 
  const Eigen::Matrix3d& R_0_in_1, std::vector<cv::KeyPoint>& kpts1_predict) {
  kpts1_predict = kpts0;
  if (!enable_keypoint_predict) {
    if (enable_high_frequency_log) {
      PRINT_ALL("DEBUG TrackBase::predict_keypoints: Skip predict keypoints since it's disabled!\n");
    }
    return;
  }  

  cv::Matx33f cv_R_0_in_1;
  Eigen::Matrix3f R_0_in_1f = R_0_in_1.cast<float>();
  cv::eigen2cv(R_0_in_1f, cv_R_0_in_1);

  if (enable_high_frequency_log) {
    std::ostringstream oss;
    oss << "DEBUG TrackBase::predict_keypoints:  R_0_in_1:" << std::endl;
    oss << R_0_in_1 << std::endl;
    oss << "DEBUG TrackBase::predict_keypoints:  cv_R_0_in_1:" << std::endl;
    oss << cv_R_0_in_1 << std::endl;
    PRINT_ALL("%s", oss.str().c_str());
  }

  for (size_t i=0; i<kpts0.size(); i++) {
    cv::Point2f npt_0 = camera_calib.at(cam_id0)->undistort_cv(kpts0.at(i).pt);
    cv::Point3f homogenous_npt_0(npt_0.x, npt_0.y, 1.0);
    cv::Point3f pt3d_1 = cv_R_0_in_1 * homogenous_npt_0;
    cv::Point2f npt_1(pt3d_1.x / pt3d_1.z, pt3d_1.y / pt3d_1.z);
    kpts1_predict.at(i).pt = camera_calib.at(cam_id1)->distort_cv(npt_1);
  }
}

void TrackBase::predict_keypoints_temporally(
    size_t cam_id, double new_time,
    const std::vector<cv::KeyPoint>& kpts_old,
    std::vector<cv::KeyPoint>& kpts_new_predict,
    Eigen::Matrix3d& R_old_in_new) {
  R_old_in_new = predict_rotation(cam_id, new_time);
  predict_keypoints(cam_id, cam_id, kpts_old, R_old_in_new, kpts_new_predict);
}

void TrackBase::predict_keypoints_stereo(
    size_t cam_id_left, size_t cam_id_right,
    const std::vector<cv::KeyPoint>& kpts_left,
    std::vector<cv::KeyPoint>& kpts_right_predict,
    Eigen::Matrix3d& R_left_in_right,
    Eigen::Vector3d& t_left_in_right) {
  
  Eigen::Isometry3d T_Cleft_in_I = camera_extrinsics[cam_id_left];
  Eigen::Isometry3d T_Cright_in_I = camera_extrinsics[cam_id_right];
  Eigen::Isometry3d T_left_in_right = T_Cright_in_I.inverse() * T_Cleft_in_I;
  // R_left_in_right = R_Cright_in_I.transpose() * R_Cleft_in_I;
  R_left_in_right = T_left_in_right.linear();
  t_left_in_right = T_left_in_right.translation();

  predict_keypoints(
      cam_id_left, cam_id_right, kpts_left, R_left_in_right, kpts_right_predict);
}

void TrackBase::select_masked(const std::vector<uchar> & mask, std::vector<size_t>& selected_indices) {
  selected_indices.reserve(mask.size());
  for (size_t i=0; i<mask.size(); i++) {
    if (mask[i]) {
      selected_indices.push_back(i);
    }
  }
}

void apply_selected_mask(const std::vector<uchar> & selected_mask, const std::vector<size_t> selected_indices, std::vector<uchar> & total_mask) {
  // reset total_mask
  for (size_t i=0; i<total_mask.size(); i++) {
    total_mask[i] = 0;
  }
  // apply selected_mask
  for (size_t i=0; i<selected_mask.size(); i++) {
    total_mask[selected_indices[i]] = selected_mask[i];
  }
}


double TrackBase::get_coeffs_mat_for_essential_test(
    const Eigen::Matrix3d& R_0_in_1,
    const std::vector<cv::Point2f>& pts0_n,
    const std::vector<cv::Point2f>& pts1_n,
    Eigen::MatrixXd& out_coeffs_mat,
    std::vector<double>& disparities) {
  assert(pts0_n.size() == pts1_n.size());

  // std::vector<double> disparities;
  disparities.resize(pts0_n.size());
  Eigen::MatrixXd coeffs_mat (pts0_n.size(), 3);
  cv::Matx33f cv_R_0_in_1;
  Eigen::Matrix3f R_0_in_1f = R_0_in_1.cast<float>();
  cv::eigen2cv(R_0_in_1f, cv_R_0_in_1);
  for (size_t i=0; i<pts0_n.size(); i++) {
    cv::Point3f homo_pt0(pts0_n[i].x, pts0_n[i].y, 1);
    cv::Point3f rot_pt0 = cv_R_0_in_1 * homo_pt0;
    cv::Point2f rot_pt0n;
    rot_pt0n.x = rot_pt0.x/rot_pt0.z;
    rot_pt0n.y = rot_pt0.y/rot_pt0.z;
    float x0 = rot_pt0n.x;
    float y0 = rot_pt0n.y;
    float x1 = pts1_n[i].x;
    float y1 = pts1_n[i].y;
    // (x0, y0, 1) X (x1, y1, 1) = (y0 - y1,  x1 - x0,  x0*y1 - x1*y0)
    coeffs_mat(i, 0) = y0 - y1;
    coeffs_mat(i, 1) = x1 - x0;
    coeffs_mat(i, 2) = x0*y1 - x1*y0;
    cv::Point2f diff = rot_pt0n - pts1_n[i];
    disparities[i] = sqrt(diff.dot(diff));
  }

  // size_t representative_index = disparities.size() / 3;  // the first 1/3 point
  size_t representative_index = disparities.size() / 2;  // the median
  std::nth_element(disparities.begin(), disparities.begin() + representative_index, disparities.end());
  double representative_disparity = disparities[representative_index];
  PRINT_DEBUG("get_coeffs_mat_for_essential_test.representative_disparity: %f\n", representative_disparity);
  out_coeffs_mat = std::move(coeffs_mat);
  return representative_disparity;
}


Eigen::Vector3d TrackBase::solve_essential(const Eigen::MatrixXd& coeffs_mat, const std::vector<size_t>& used_rows) {
  Eigen::MatrixXd used_coeffs(used_rows.size(), 3);
  for (size_t i=0; i<used_rows.size(); i++) {
    used_coeffs.row(i) = coeffs_mat.row(used_rows[i]);
  }
  Eigen::VectorXd X = used_coeffs.col(0);
  Eigen::VectorXd Y = used_coeffs.col(1);
  Eigen::VectorXd Z = used_coeffs.col(2);
  double X_norm = X.norm();
  double Y_norm = Y.norm();
  double Z_norm = Z.norm();
  double min_norm = std::min(X_norm, std::min(Y_norm, Z_norm));

  Eigen::MatrixXd A(used_rows.size(), 2);
  Eigen::VectorXd b;
  auto solveAb = [](const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::VectorXd solution;
    if (A.rows() == 2) {
      solution = A.inverse() * b;
    } else {
      solution = (A.transpose() * A).inverse() * A.transpose() * b;
    }
    return solution;
  };

  if (min_norm == X_norm) {
    A << used_coeffs.col(1), used_coeffs.col(2);
    b = - used_coeffs.col(0);
    auto r = solveAb(A,b);
    return Eigen::Vector3d(1.0, r[0], r[1]);
  } else if (min_norm == Y_norm) {
    A << used_coeffs.col(0), used_coeffs.col(2);
    b = - used_coeffs.col(1);
    auto r = solveAb(A,b);
    return Eigen::Vector3d(r[0], 1.0, r[1]);
  } else { // if (min_norm == Z_norm)
    A << used_coeffs.col(0), used_coeffs.col(1);
    b = - used_coeffs.col(2);
    auto r = solveAb(A,b);
    return Eigen::Vector3d(r[0], r[1], 1.0);
  }
}

std::vector<size_t> TrackBase::get_essential_inliers(const Eigen::MatrixXd& coeffs_mat, const Eigen::Vector3d& t, const double thr) {
  Eigen::VectorXd err_vec = coeffs_mat * t;
  std::vector<size_t> inliers;
  inliers.reserve(err_vec.rows());
  for (size_t i=0; i<err_vec.rows(); i++) {
    if (std::abs(err_vec[i]) <= thr) {
      inliers.push_back(i);
    }
  }
  return inliers;
}

void TrackBase::two_point_ransac(
    const Eigen::Matrix3d& R_0_in_1,
    const std::vector<cv::Point2f>& pts0_n,
    const std::vector<cv::Point2f>& pts1_n,
    std::vector<uchar> & inliers_mask,
    const double disparity_thr,
    const double essential_inlier_thr,
    int max_iter) {

  // const double essential_inlier_thr = 1.0 * M_PI / 180.0;
  // const double disparity_thr = essential_inlier_thr / 3.0;  // moving or stationary

  if (pts0_n.size() < 5) {
    // too few keypoints. mark all as outliers.
    inliers_mask.resize(pts0_n.size(), 0);
    PRINT_WARNING(YELLOW "two_point_ransac[too few points]: inliers/total = 0/%d\n" RESET, pts0_n.size());
    return;
  }

  Eigen::MatrixXd coeffs_mat;
  std::vector<double> disparities;
  double representative_disparity = get_coeffs_mat_for_essential_test(
    R_0_in_1, pts0_n, pts1_n, coeffs_mat, disparities);

  if (representative_disparity < disparity_thr) {
    // the stationary case.
    // So features with large disparities are outliers.
    // (What if most of the features are far but only a minor part of them are near features? We'll mark all near features as outliers?)
    inliers_mask.resize(pts0_n.size(), 1);
    int cnt_inliers = 0;
    Eigen::VectorXd disparities_vec(disparities.size());
    for (size_t i=0; i<pts0_n.size(); i++) {
      disparities_vec[i] = disparities[i];
      if (disparities[i] > essential_inlier_thr) {
        inliers_mask[i] = 0;
      } else {
        cnt_inliers ++;
      }
    }

    if (enable_high_frequency_log) {
      std::ostringstream oss;
      oss << "two_point_ransac[stationary].err_vec(thr=" << essential_inlier_thr << "): " << disparities_vec.transpose() << std::endl;
      PRINT_ALL("%s", oss.str().c_str());
    }
    PRINT_DEBUG("two_point_ransac[stationary]: inliers/total =  %d/%d\n", cnt_inliers, coeffs_mat.rows());

    return;
  }

  // motion case. the model works.

  std::vector<size_t> best_inliers;
  Eigen::Vector3d best_t;
  for (int iter = 0; iter < max_iter; iter ++) {
    std::vector<size_t> used_rows = select_samples(coeffs_mat.rows(), 2);
    Eigen::Vector3d t = solve_essential(coeffs_mat, used_rows);
    if (enable_high_frequency_log) {
      std::ostringstream oss;
      oss << "two_point_ransac: t = " << t.transpose() << ", t.norm(): " << t.norm() << std::endl;
      PRINT_ALL("%s", oss.str().c_str());
    }
    // if (t.norm() < 1e-6) {
    //   continue;
    // }
    t.normalize();
    std::vector<size_t> inliers = get_essential_inliers(coeffs_mat, t, essential_inlier_thr);
    if (inliers.size() > best_inliers.size()) {
      best_inliers = inliers;
      best_t = t;
    }
  }
  
  inliers_mask = std::vector<uchar>(coeffs_mat.rows(), 0);
  for (size_t inlier_idx : best_inliers) {
    inliers_mask[inlier_idx] = 1;
  }

  Eigen::VectorXd err_vec = coeffs_mat * best_t;
  if (enable_high_frequency_log) {
    std::ostringstream oss;
    oss << "two_point_ransac[normal].err_vec(thr=" << essential_inlier_thr << "): " << err_vec.transpose() << std::endl;
    PRINT_ALL("%s", oss.str().c_str());
  }
  PRINT_DEBUG("two_point_ransac[normal]: inliers/total =  %d/%d\n", best_inliers.size(), coeffs_mat.rows());

  return;
}

void TrackBase::known_essential_check(
    const Eigen::Matrix3d& R_0_in_1,
    const Eigen::Vector3d& t_0_in_1,
    const std::vector<cv::Point2f>& pts0_n,
    const std::vector<cv::Point2f>& pts1_n,
    std::vector<uchar> & inliers_mask,
    const double essential_inlier_thr) {
  Eigen::Vector3d t = t_0_in_1;
  t.normalize();
  Eigen::MatrixXd coeffs_mat;
  std::vector<double> disparities_dummy;
  get_coeffs_mat_for_essential_test(R_0_in_1, pts0_n, pts1_n, coeffs_mat, disparities_dummy);
  std::vector<size_t> inliers = get_essential_inliers(coeffs_mat, t, essential_inlier_thr);

  Eigen::VectorXd err_vec = coeffs_mat * t;
  if (enable_high_frequency_log) {
    std::ostringstream oss;
    oss << "known_essential_check.err_vec(thr=" << essential_inlier_thr << "): " << err_vec.transpose() << std::endl;
    PRINT_ALL("%s", oss.str().c_str());
  }
  PRINT_DEBUG("known_essential_check: inliers/total =  %d/%d\n", inliers.size(), coeffs_mat.rows());
  inliers_mask = std::vector<uchar>(coeffs_mat.rows(), 0);
  for (size_t inlier_idx : inliers) {
    inliers_mask[inlier_idx] = 1;
  }
  return;  
}

void TrackBase::select_common_id(
    const std::vector<size_t>& ids0, const std::vector<size_t>& ids1,
    std::vector<size_t>& common_ids,
    std::vector<size_t>& selected_indices0,
    std::vector<size_t>& selected_indices1) {

  std::map<size_t, size_t> id_to_idx0, id_to_idx1;
  std::set<size_t> ids0_set, ids1_set;
  for (size_t i=0; i<ids0.size(); i++) {
    id_to_idx0[ids0[i]] = i;
    ids0_set.insert(ids0[i]);
  }
  for (size_t i=0; i<ids1.size(); i++) {
    id_to_idx1[ids1[i]] = i;
    ids1_set.insert(ids1[i]);
  }

  common_ids.clear();
  std::set_intersection(ids0_set.begin(), ids0_set.end(), ids1_set.begin(), ids1_set.end(), 
                        std::inserter(common_ids, common_ids.end()));

  selected_indices0.clear();
  selected_indices1.clear();
  selected_indices0.reserve(common_ids.size());
  selected_indices1.reserve(common_ids.size());
  for (auto id : common_ids) {
    selected_indices0.push_back(id_to_idx0[id]);
    selected_indices1.push_back(id_to_idx1[id]);
  }
}

std::vector<cv::KeyPoint> TrackBase::select_keypoints(
    const std::vector<size_t>& selected_indices, const std::vector<cv::KeyPoint>& keypoints) {
  std::vector<cv::KeyPoint> selected_keypoints(selected_indices.size());
  for (size_t i=0; i<selected_indices.size(); i++) {
    selected_keypoints[i] = keypoints[selected_indices[i]];
  }
  return selected_keypoints;
}


void TrackBase::two_point_ransac(
    const Eigen::Matrix3d& R_0_in_1,
    size_t cam_id0, size_t cam_id1,
    const std::vector<cv::KeyPoint>& kpts0,
    const std::vector<cv::KeyPoint>& kpts1,
    std::vector<uchar> & inliers_mask,
    const double disparity_thr,
    const double essential_inlier_thr,
    int max_iter) {

  std::vector<cv::Point2f> pts0_n(kpts0.size());
  std::vector<cv::Point2f> pts1_n(kpts1.size());

  for (size_t i=0; i<kpts0.size(); i++) {
    pts0_n[i] = camera_calib.at(cam_id0)->undistort_cv(kpts0.at(i).pt);
  }

  for (size_t i=0; i<kpts1.size(); i++) {
    pts1_n[i] = camera_calib.at(cam_id1)->undistort_cv(kpts1.at(i).pt);
  }

  two_point_ransac(R_0_in_1, pts0_n, pts1_n, inliers_mask, disparity_thr, essential_inlier_thr, max_iter);
}

void TrackBase::fundamental_ransac(
    const std::vector<cv::Point2f>& pts0_n,
    const std::vector<cv::Point2f>& pts1_n,
    const double fundamental_inlier_thr,
    std::vector<uchar> & inliers_mask) {

  // If we don't have enough points for ransac just return empty
  // We set the mask to be all zeros since all points failed RANSAC
  if (pts0_n.size() < 15) {  // or < 10 ?
    for (size_t i = 0; i < pts0_n.size(); i++)
      inliers_mask.push_back((uchar)0);
    PRINT_DEBUG("fundamental_ransac: Too few points (%d)\n", pts0_n.size());
    return;
  }

  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, fundamental_inlier_thr, 0.999, inliers_mask);

  int cnt_inliers = 0;
  for (auto v : inliers_mask) {
    if (v) {
      cnt_inliers ++;
    }
  }
  PRINT_DEBUG("fundamental_ransac: inliers/total =  %d/%d\n", cnt_inliers, pts0_n.size());
}

void TrackBase::fundamental_ransac(
      size_t cam_id0, size_t cam_id1,
      const std::vector<cv::KeyPoint>& kpts0,
      const std::vector<cv::KeyPoint>& kpts1,
      const double fundamental_inlier_thr,
      std::vector<uchar> & inliers_mask) {
  std::vector<cv::Point2f> pts0_n(kpts0.size());
  std::vector<cv::Point2f> pts1_n(kpts1.size());

  for (size_t i=0; i<kpts0.size(); i++) {
    pts0_n[i] = camera_calib.at(cam_id0)->undistort_cv(kpts0.at(i).pt);
  }

  for (size_t i=0; i<kpts1.size(); i++) {
    pts1_n[i] = camera_calib.at(cam_id1)->undistort_cv(kpts1.at(i).pt);
  }
  fundamental_ransac(pts0_n, pts1_n, fundamental_inlier_thr, inliers_mask);
}