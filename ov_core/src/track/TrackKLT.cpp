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

#include "TrackKLT.h"

#include "Grider_FAST.h"
#include "Grider_GRID.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"

using namespace ov_core;

void TrackKLT::feed_new_camera(const CameraData &message) {
  if (enable_high_frequency_log) {
    std::ostringstream oss;
    oss << "DEBUG TrackKLT::feed_new_camera: t_d = " << t_d << ", gyro_bias = (" << gyro_bias.transpose() << ")" << std::endl;
    PRINT_ALL("%s", oss.str().c_str());
  }

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // Preprocessing steps that we do not parallelize
  // NOTE: DO NOT PARALLELIZE THESE!
  // NOTE: These seem to be much slower if you parallelize them...
  rT1 = std::chrono::high_resolution_clock::now();
  size_t num_images = message.images.size();
  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {

    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize
    cv::Mat img;
    if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images.at(msg_id), img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      double eq_clip_limit = 10.0;
      cv::Size eq_win_size = cv::Size(8, 8);
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
      clahe->apply(message.images.at(msg_id), img);
    } else {
      img = message.images.at(msg_id);
    }

    // Extract image pyramid
    std::vector<cv::Mat> imgpyr;
    cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);

    // Save!
    img_curr[cam_id] = img;
    img_pyramid_curr[cam_id] = imgpyr;
  }

  // Either call our stereo or monocular version
  // If we are doing binocular tracking, then we should parallize our tracking
  if (num_images == 1) {
    feed_monocular(message, 0);
  } else if (num_images == 2 && use_stereo) {
    if (left_major_stereo) {
      feed_stereo2(message, 0, 1);
      PRINT_DEBUG("left_major_stereo = %d, use feed_stereo2()\n", left_major_stereo);
    } else {
      feed_stereo(message, 0, 1);
      PRINT_DEBUG("left_major_stereo = %d, use feed_stereo()\n", left_major_stereo);
    }
  } else if (!use_stereo) {
    parallel_for_(cv::Range(0, (int)num_images), LambdaBody([&](const cv::Range &range) {
                    for (int i = range.start; i < range.end; i++) {
                      feed_monocular(message, i);
                    }
                  }));
  } else {
    PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
    std::exit(EXIT_FAILURE);
  }
}

void TrackKLT::feed_monocular(const CameraData &message, size_t msg_id) {

  // Lock this data feed for this camera
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Get our image objects for this image
  cv::Mat img = img_curr.at(cam_id);

  // const cv::Mat& vis_img = img;
  const cv::Mat& vis_img = message.images.at(msg_id);

  std::vector<cv::Mat> imgpyr = img_pyramid_curr.at(cam_id);
  cv::Mat mask = message.masks.at(msg_id);
  rT2 = std::chrono::high_resolution_clock::now();

  int pts_before_detect = 0;
  int pts_after_detect = 0;
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;

  if (pts_last[cam_id].empty()) {
    // If we didn't have any successful tracks last time, just extract this time
    // This also handles, the tracking initalization on the first call to this extractor
    rT3 = std::chrono::high_resolution_clock::now();
    perform_detection_monocular(imgpyr, mask, good_left, good_ids_left);
  } else {
    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    auto pts_left_old = pts_last[cam_id];
    auto ids_left_old = ids_last[cam_id];

    // Our return success masks, and predicted new features
    std::vector<uchar> mask_ll;
    // std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
    Eigen::Matrix3d R_old_in_new;
    std::vector<cv::KeyPoint> pts_left_new;
    predict_keypoints_temporally(
        cam_id, message.timestamp,
        pts_left_old, pts_left_new, R_old_in_new);

    std::vector<cv::KeyPoint> pts_left_new_predict  =  pts_left_new;

    // Lets track temporally
    perform_matching(img_pyramid_last[cam_id], imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll, &R_old_in_new);
    assert(pts_left_new.size() == ids_left_old.size());
    rT3 = std::chrono::high_resolution_clock::now();
    
    if (enable_high_frequency_log) {
      std::ostringstream oss;
      oss << "DEBUG temporal predict(predict_err, prior_error): ";
      for (size_t i=0; i<pts_left_new.size(); i++) {
        cv::Point2f predict_err = pts_left_new[i].pt - pts_left_new_predict[i].pt;
        cv::Point2f prior_err = pts_left_new[i].pt - pts_left_old[i].pt;
        oss << "(" << sqrt(predict_err.dot(predict_err)) << ", " << sqrt(prior_err.dot(prior_err)) << ")   ";
      }
      oss << std::endl;
      PRINT_ALL("%s", oss.str().c_str());
    }
    
    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty()) {
      std::lock_guard<std::mutex> lckv(mtx_last_vars);
      img_last[cam_id] = vis_img;
      img_time_last[cam_id] = message.timestamp;
      img_pyramid_last[cam_id] = imgpyr;
      img_mask_last[cam_id] = mask;
      pts_last[cam_id].clear();
      ids_last[cam_id].clear();
      // internal_add_last_to_history(message.timestamp);
      PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
      return;
    }

    // Get our "good tracks"
    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++) {
      // Ensure we do not have any bad KLT tracks (i.e., points are negative)
      if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
          (int)pts_left_new.at(i).pt.y >= img.rows)
        continue;
      // Check if it is in the mask
      // NOTE: mask has max value of 255 (white) if it should be
      if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
        continue;
      // If it is a good track, and also tracked from left to right
      if (mask_ll[i]) {
        good_left.push_back(pts_left_new[i]);
        good_ids_left.push_back(ids_left_old[i]);
      }
    }

    pts_before_detect = good_left.size();
    perform_detection_monocular(imgpyr, mask, good_left, good_ids_left);
  }
  pts_after_detect = good_left.size();
  PRINT_ALL("DEBUG: pts_before_detect = %d,   pts_after_detect = %d\n", pts_before_detect, pts_after_detect);

  rT4 = std::chrono::high_resolution_clock::now();


  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
  }

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_time_last[cam_id] = message.timestamp;
    img_last[cam_id] = vis_img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    internal_add_last_to_history(message.timestamp);
  }
  rT5 = std::chrono::high_resolution_clock::now();

  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT2 - rT1).count());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT2).count());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT4 - rT3).count(),
            pts_after_detect - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT5 - rT4).count(),
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT5 - rT1).count());
}

void TrackKLT::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Get our image objects for this image
  cv::Mat img_left = img_curr.at(cam_id_left);
  cv::Mat img_right = img_curr.at(cam_id_right);

  // const cv::Mat& vis_img_left = img_left;
  // const cv::Mat& vis_img_right = img_right;
  const cv::Mat& vis_img_left = message.images.at(msg_id_left);
  const cv::Mat& vis_img_right = message.images.at(msg_id_right);

  std::vector<cv::Mat> imgpyr_left = img_pyramid_curr.at(cam_id_left);
  std::vector<cv::Mat> imgpyr_right = img_pyramid_curr.at(cam_id_right);
  cv::Mat mask_left = message.masks.at(msg_id_left);
  cv::Mat mask_right = message.masks.at(msg_id_right);
  rT2 = std::chrono::high_resolution_clock::now();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
    // Track into the new image
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;
    perform_detection_stereo(imgpyr_left, imgpyr_right, mask_left, mask_right, cam_id_left, cam_id_right, good_left, good_right,
                             good_ids_left, good_ids_right);
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_time_last[cam_id_left] = message.timestamp;
    img_time_last[cam_id_right] = message.timestamp;
    img_last[cam_id_left] = vis_img_left;
    img_last[cam_id_right] = vis_img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    internal_add_last_to_history(message.timestamp);
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  int pts_before_detect = (int)pts_last[cam_id_left].size();
  auto pts_left_old = pts_last[cam_id_left];
  auto pts_right_old = pts_last[cam_id_right];
  auto ids_left_old = ids_last[cam_id_left];
  auto ids_right_old = ids_last[cam_id_right];
  perform_detection_stereo(img_pyramid_last[cam_id_left], img_pyramid_last[cam_id_right], img_mask_last[cam_id_left],
                           img_mask_last[cam_id_right], cam_id_left, cam_id_right, pts_left_old, pts_right_old, ids_left_old,
                           ids_right_old);
  rT3 = std::chrono::high_resolution_clock::now();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll, mask_rr;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
  std::vector<cv::KeyPoint> pts_right_new = pts_right_old;

  // Lets track temporally
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    perform_matching(img_pyramid_last[is_left ? cam_id_left : cam_id_right], is_left ? imgpyr_left : imgpyr_right,
                                     is_left ? pts_left_old : pts_right_old, is_left ? pts_left_new : pts_right_new,
                                     is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                                     is_left ? mask_ll : mask_rr);
                  }
                }));
  rT4 = std::chrono::high_resolution_clock::now();

  //===================================================================================
  //===================================================================================

  // left to right matching
  // TODO: we should probably still do this to reject outliers
  // TODO: maybe we should collect all tracks that are in both frames and make they pass this?
  // std::vector<uchar> mask_lr;
  // perform_matching(imgpyr_left, imgpyr_right, pts_left_new, pts_right_new, cam_id_left, cam_id_right, mask_lr);
  rT5 = std::chrono::high_resolution_clock::now();

  //===================================================================================
  //===================================================================================

  // If any of our masks are empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty() && mask_rr.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_time_last[cam_id_left] = message.timestamp;
    img_time_last[cam_id_right] = message.timestamp;
    img_last[cam_id_left] = vis_img_left;
    img_last[cam_id_right] = vis_img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left].clear();
    pts_last[cam_id_right].clear();
    ids_last[cam_id_left].clear();
    ids_last[cam_id_right].clear();
    // internal_add_last_to_history(message.timestamp);
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x > img_left.cols ||
        (int)pts_left_new.at(i).pt.y > img_left.rows)
      continue;
    // See if we have the same feature in the right
    bool found_right = false;
    size_t index_right = 0;
    for (size_t n = 0; n < ids_right_old.size(); n++) {
      if (ids_left_old.at(i) == ids_right_old.at(n)) {
        found_right = true;
        index_right = n;
        break;
      }
    }
    // If it is a good track, and also tracked from left to right
    // Else track it as a mono feature in just the left image
    if (mask_ll[i] && found_right && mask_rr[index_right]) {
      // Ensure we do not have any bad KLT tracks (i.e., points are negative)
      if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
          (int)pts_right_new.at(index_right).pt.x >= img_right.cols || (int)pts_right_new.at(index_right).pt.y >= img_right.rows)
        continue;
      good_left.push_back(pts_left_new.at(i));
      good_right.push_back(pts_right_new.at(index_right));
      good_ids_left.push_back(ids_left_old.at(i));
      good_ids_right.push_back(ids_right_old.at(index_right));
      // PRINT_DEBUG("adding to stereo - %u , %u\n", ids_left_old.at(i), ids_right_old.at(index_right));
    } else if (mask_ll[i]) {
      good_left.push_back(pts_left_new.at(i));
      good_ids_left.push_back(ids_left_old.at(i));
      // PRINT_DEBUG("adding to left - %u \n",ids_left_old.at(i));
    }
  }

  // Loop through all right points
  for (size_t i = 0; i < pts_right_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= img_right.cols ||
        (int)pts_right_new.at(i).pt.y >= img_right.rows)
      continue;
    // See if we have the same feature in the right
    bool added_already = (std::find(good_ids_right.begin(), good_ids_right.end(), ids_right_old.at(i)) != good_ids_right.end());
    // If it has not already been added as a good feature, add it as a mono track
    if (mask_rr[i] && !added_already) {
      good_right.push_back(pts_right_new.at(i));
      good_ids_right.push_back(ids_right_old.at(i));
      // PRINT_DEBUG("adding to right - %u \n", ids_right_old.at(i));
    }
  }

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y);
  }
  for (size_t i = 0; i < good_right.size(); i++) {
    cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
    database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y);
  }

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_time_last[cam_id_left] = message.timestamp;
    img_time_last[cam_id_right] = message.timestamp;
    img_last[cam_id_left] = vis_img_left;
    img_last[cam_id_right] = vis_img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    internal_add_last_to_history(message.timestamp);
  }
  rT6 = std::chrono::high_resolution_clock::now();

  //  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT2 - rT1).count());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT2).count(),
            (int)pts_last[cam_id_left].size() - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT4 - rT3).count());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for stereo klt\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT5 - rT4).count());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT6 - rT5).count(),
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT6 - rT1).count());
}

void TrackKLT::feed_stereo2(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Get our image objects for this image
  cv::Mat img_left = img_curr.at(cam_id_left);
  cv::Mat img_right = img_curr.at(cam_id_right);

  // const cv::Mat& vis_img_left = img_left;
  // const cv::Mat& vis_img_right = img_right;
  const cv::Mat& vis_img_left = message.images.at(msg_id_left);
  const cv::Mat& vis_img_right = message.images.at(msg_id_right);

  std::vector<cv::Mat> imgpyr_left = img_pyramid_curr.at(cam_id_left);
  std::vector<cv::Mat> imgpyr_right = img_pyramid_curr.at(cam_id_right);
  cv::Mat mask_left = message.masks.at(msg_id_left);
  cv::Mat mask_right = message.masks.at(msg_id_right);
  rT2 = std::chrono::high_resolution_clock::now();

  // Get our "good tracks"
  int pts_before_detect = 0;
  int pts_after_detect = 0;
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;
  std::set<size_t> doubly_verified_stereo_ids;

  if (pts_last[cam_id_left].empty()) {
    // If we didn't have any successful tracks last time, just extract this time
    // This also handles, the tracking initalization on the first call to this extractor.
    rT3 = std::chrono::high_resolution_clock::now();
    perform_detection_monocular(imgpyr_left, mask_left, good_left, good_ids_left);
  } else {
    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    auto pts_left_old = pts_last[cam_id_left];
    auto ids_left_old = ids_last[cam_id_left];


    // Our return success masks, and predicted new features
    std::vector<uchar> mask_ll;
    // std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
    Eigen::Matrix3d R_old_in_new;
    std::vector<cv::KeyPoint> pts_left_new;
    predict_keypoints_temporally(
        cam_id_left, message.timestamp,
        pts_left_old, pts_left_new, R_old_in_new);

    std::vector<cv::KeyPoint> pts_left_new_predict  =  pts_left_new;

    perform_matching(img_pyramid_last[cam_id_left], imgpyr_left,
                    pts_left_old, pts_left_new,
                    cam_id_left, cam_id_left,
                    mask_ll, &R_old_in_new);
    assert(pts_left_new.size() == ids_left_old.size());

    if (enable_high_frequency_log) {
      std::ostringstream oss;
      oss << "DEBUG temporal predict(predict_err, prior_error): ";
      for (size_t i=0; i<pts_left_new.size(); i++) {
        cv::Point2f predict_err = pts_left_new[i].pt - pts_left_new_predict[i].pt;
        cv::Point2f prior_err = pts_left_new[i].pt - pts_left_old[i].pt;
        oss << "(" << sqrt(predict_err.dot(predict_err)) << ", " << sqrt(prior_err.dot(prior_err)) << ")   ";
      }
      oss << std::endl;
      PRINT_ALL("%s", oss.str().c_str());
    }


    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty()) {
      std::lock_guard<std::mutex> lckv(mtx_last_vars);
      img_time_last[cam_id_left] = message.timestamp;
      img_time_last[cam_id_right] = message.timestamp;
      img_last[cam_id_left] = vis_img_left;
      img_last[cam_id_right] = vis_img_right;
      img_pyramid_last[cam_id_left] = imgpyr_left;
      img_pyramid_last[cam_id_right] = imgpyr_right;
      img_mask_last[cam_id_left] = mask_left;
      img_mask_last[cam_id_right] = mask_right;
      pts_last[cam_id_left].clear();
      pts_last[cam_id_right].clear();
      ids_last[cam_id_left].clear();
      ids_last[cam_id_right].clear();
      doubly_verified_stereo_last[cam_id_left].clear();
      doubly_verified_stereo_last[cam_id_right].clear();

      // internal_add_last_to_history(message.timestamp);
      PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
      return;
    }


    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++) {
      // Ensure we do not have any bad KLT tracks (i.e., points are negative)
      if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img_left.cols ||
          (int)pts_left_new.at(i).pt.y >= img_left.rows)
        continue;
      // Check if it is in the mask
      // NOTE: mask has max value of 255 (white) if it should be
      if ((int)mask_left.at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
        continue;
      // If it is a good track
      if (mask_ll[i]) {
        good_left.push_back(pts_left_new[i]);
        good_ids_left.push_back(ids_left_old[i]);
      }
    }

    rT3 = std::chrono::high_resolution_clock::now();
    pts_before_detect = good_left.size();
    perform_detection_monocular(imgpyr_left, mask_left, good_left, good_ids_left);
  }
  pts_after_detect = good_left.size();


  rT4 = std::chrono::high_resolution_clock::now();


  //===================================================================================
  //===================================================================================

  // left to right matching

  std::vector<uchar> mask_lr;
  // std::vector<cv::KeyPoint> pts_right_new = good_left;
  std::vector<cv::KeyPoint> pts_right_new;
  Eigen::Matrix3d R_left_in_right;
  Eigen::Vector3d t_left_in_right;
  predict_keypoints_stereo(
      cam_id_left, cam_id_right, good_left, pts_right_new, R_left_in_right, t_left_in_right);

  std::vector<cv::KeyPoint> pts_right_new_predict  =  pts_right_new;

  perform_matching(imgpyr_left, imgpyr_right, good_left, pts_right_new, cam_id_left, cam_id_right, mask_lr, &R_left_in_right, &t_left_in_right);
  assert(pts_right_new.size() == good_left.size());
  rT5 = std::chrono::high_resolution_clock::now();

  if (enable_high_frequency_log) {
    std::ostringstream oss;
    oss << "DEBUG stereo predict(predict_err, prior_error): ";
    for (size_t i=0; i<pts_right_new.size(); i++) {
      cv::Point2f predict_err = pts_right_new[i].pt - pts_right_new_predict[i].pt;
      cv::Point2f prior_err = pts_right_new[i].pt - good_left[i].pt;
      oss << "(" << sqrt(predict_err.dot(predict_err)) << ", " << sqrt(prior_err.dot(prior_err)) << ")   ";
    }
    oss << std::endl;
    PRINT_ALL("%s", oss.str().c_str());
  }



  //===================================================================================
  //===================================================================================

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_right;
  std::vector<size_t> good_ids_right;

  // Loop through all right points
  for (size_t i = 0; i < pts_right_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= img_right.cols ||
        (int)pts_right_new.at(i).pt.y >= img_right.rows)
      continue;
    // Check if it is in the mask
    // NOTE: mask has max value of 255 (white) if it should be
    if ((int)mask_right.at<uint8_t>((int)pts_right_new.at(i).pt.y, (int)pts_right_new.at(i).pt.x) > 127)
      continue;
    // If it is a good track
    if (mask_lr[i]) {
      good_right.push_back(pts_right_new[i]);
      good_ids_right.push_back(good_ids_left[i]);
    }
  }

  // right to right matching (only do essential check)
  if (!pts_last[cam_id_right].empty()) {
    std::vector<uchar> mask_rr;
    auto pts_right_old = pts_last[cam_id_right];
    auto ids_right_old = ids_last[cam_id_right];
    std::vector<size_t> common_ids;
    std::vector<size_t> selected_indices_old;
    std::vector<size_t> selected_indices_new;
    select_common_id(ids_right_old, good_ids_right, common_ids, selected_indices_old, selected_indices_new);
    std::vector<cv::KeyPoint> selected_kpts_old = select_keypoints(selected_indices_old, pts_right_old);
    std::vector<cv::KeyPoint> selected_kpts_new = select_keypoints(selected_indices_new, good_right);
    Eigen::Matrix3d R_old_in_new_right = predict_rotation(cam_id_right, message.timestamp);
    std::vector<uchar> selected_mask_rr;
    double max_focallength = std::max(camera_calib.at(cam_id_right)->get_K()(0, 0), camera_calib.at(cam_id_right)->get_K()(1, 1));
    const double success_probability = 0.99;
    const int max_iter = ceil(log(1-success_probability) / log(1-0.7*0.7));  // = 7
    if (force_fundamental) {
      fundamental_ransac(cam_id_right, cam_id_right, selected_kpts_old, selected_kpts_new, 2.0 / max_focallength, selected_mask_rr);
    } else {
      two_point_ransac(
          R_old_in_new_right,
          cam_id_right, cam_id_right,
          selected_kpts_old, selected_kpts_new,
          selected_mask_rr, 1.0 / max_focallength, 3.0 / max_focallength, max_iter);
    }

#if 0
    // disable rr
    doubly_verified_stereo_ids.insert(common_ids.begin(), common_ids.end());
#else        
    for (size_t i=0; i<selected_mask_rr.size(); i++) {
      if (selected_mask_rr[i]) {
        doubly_verified_stereo_ids.insert(common_ids[i]);
      }
    }
#endif
    PRINT_DEBUG("DEBUG doubly_verified_stereo_ids/prev_left_total=%d/%d\n", doubly_verified_stereo_ids.size(), ids_last[cam_id_left].size());
  }

  rT6 = std::chrono::high_resolution_clock::now();

  //===================================================================================
  //===================================================================================

  // Update our feature database, with theses new observations
  for (size_t i=0; i<pts_last[cam_id_left].size(); i++) {
    if (strict_stereo && doubly_verified_stereo_ids.count(ids_last[cam_id_left].at(i))
        && !doubly_verified_stereo_last[cam_id_left].count(ids_last[cam_id_left].at(i))) {
      // some keypoints (e.g. newly extracted ones) in the last frame had not been marked as doubly_verified,
      // but now we know they're good, so add them to database now (for the left-camera, only needed when strict_stereo=true).
      cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(pts_last[cam_id_left].at(i).pt);
      database->update_feature(ids_last[cam_id_left].at(i), img_time_last[cam_id_left], cam_id_left, 
                               pts_last[cam_id_left].at(i).pt.x, pts_last[cam_id_left].at(i).pt.y, 
                               npt_l.x, npt_l.y);
    }
  }

  for (size_t i=0; i<pts_last[cam_id_right].size(); i++) {
    if (doubly_verified_stereo_ids.count(ids_last[cam_id_right].at(i))
        && !doubly_verified_stereo_last[cam_id_right].count(ids_last[cam_id_right].at(i))) {
      // some keypoints (e.g. newly extracted ones) in the last frame had not been marked as doubly_verified,
      // but now we know they're good, so add them to database now.
      cv::Point2f npt_l = camera_calib.at(cam_id_right)->undistort_cv(pts_last[cam_id_right].at(i).pt);
      database->update_feature(ids_last[cam_id_right].at(i), img_time_last[cam_id_right], cam_id_right, 
                               pts_last[cam_id_right].at(i).pt.x, pts_last[cam_id_right].at(i).pt.y, 
                               npt_l.x, npt_l.y);
    }
  }

  for (size_t i = 0; i < good_left.size(); i++) {
    if (!strict_stereo || doubly_verified_stereo_ids.count(good_ids_left.at(i))) {
      cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
      database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                              npt_l.y);
    }
  }
  for (size_t i = 0; i < good_right.size(); i++) {
    if (doubly_verified_stereo_ids.count(good_ids_right.at(i))) {
      cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
      database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                              npt_r.y);
    }
  }

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_time_last[cam_id_left] = message.timestamp;
    img_time_last[cam_id_right] = message.timestamp;
    img_last[cam_id_left] = vis_img_left;
    img_last[cam_id_right] = vis_img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    doubly_verified_stereo_last[cam_id_left] = doubly_verified_stereo_ids;
    doubly_verified_stereo_last[cam_id_right] = doubly_verified_stereo_ids;
    internal_add_last_to_history(message.timestamp);
  }
  rT7 = std::chrono::high_resolution_clock::now();

  //  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT2 - rT1).count());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt (left)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT3 - rT2).count());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT4 - rT3).count(),
            pts_after_detect - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for stereo klt\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT5 - rT4).count());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal ransac (right)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT6 - rT5).count());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT7 - rT6).count(),
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", std::chrono::duration_cast<std::chrono::duration<double>>(rT7 - rT1).count());
}


void TrackKLT::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less than grid_px_size points away then existing features
  cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                      (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x);
    int y_grid = std::floor(kpt.pt.y / size_y);
    if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // isaac: skip this check for old features. (we'd like to keep all the old features even they're near each other)
    // Check if this keypoint is near another point
    // if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
    //   it0 = pts0.erase(it0);
    //   it1 = ids0.erase(it1);
    //   continue;
    // }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255));
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  // If we don't need any features, just return
  double min_feat_percent = 0.50;
  int num_featsneeded = num_features - (int)pts0.size();
  if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features))) {
    return;
  }

  // This is old extraction code that would extract from the whole image
  // This can be slow as this will recompute extractions for grid areas that we have max features already
  // std::vector<cv::KeyPoint> pts0_ext;
  // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // We also check a downsampled mask such that we don't extract in areas where it is all masked!
  cv::Mat mask0_grid;
  cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

  // Create grids we need to extract from and then extract our features (use fast with griding)
  int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
  int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
  std::vector<std::pair<int, int>> valid_locs;
  for (int x = 0; x < grid_2d_grid.cols; x++) {
    for (int y = 0; y < grid_2d_grid.rows; y++) {
      if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
        valid_locs.emplace_back(x, y);
      }
    }
  }
  std::vector<cv::KeyPoint> pts0_ext;
  Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // Now, reject features that are close a current feature
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : pts0_ext) {
    // Check that it is in bounds
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
      continue;
    // See if there is a point at this location
    if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
      continue;
    // Else lets add it!
    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Loop through and record only ones that are valid
  // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
  // NOTE: this is due to the fact that we select update features based on feat id
  // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
  // NOTE: not sure how to remove... maybe a better way?
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // update the uv coordinates
    kpts0_new.at(i).pt = pts0_new.at(i);
    // append the new uv coordinate
    pts0.push_back(kpts0_new.at(i));
    // move id foward and append this new point
    size_t temp = ++currid;
    ids0.push_back(temp);
  }
}

void TrackKLT::perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                        const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                        std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size_close0((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                       (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
  float size_x0 = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y0 = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid0(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x0);
    int y_grid = std::floor(kpt.pt.y / size_y0);
    if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // isaac: skip this check for old features. (we'd like to keep all the old features even they're near each other)
    // Check if this keypoint is near another point
    // if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
    //   it0 = pts0.erase(it0);
    //   it1 = ids0.erase(it1);
    //   continue;
    // }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255));
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  double min_feat_percent = 0.50;
  int num_featsneeded_0 = num_features - (int)pts0.size();

  // LEFT: if we need features we should extract them in the current frame
  // LEFT: we will also try to track them from this frame over to the right frame
  // LEFT: in the case that we have two features that are the same, then we should merge them
  if (num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts0_ext;
    // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid0, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid0.cols; x++) {
      for (int y = 0; y < grid_2d_grid0.rows; y++) {
        if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    // std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
      kpts0_new.push_back(kpt);
      // pts0_new.push_back(kpt.pt);
    }

    // TODO: Project points from the left frame into the right frame
    // TODO: This will not work for large baseline systems.....
    // TODO: If we had some depth estimates we could do a better projection
    // TODO: Or project and search along the epipolar line??
    std::vector<cv::KeyPoint> kpts1_new;
    // std::vector<cv::Point2f> pts1_new;
    kpts1_new = kpts0_new;
    // pts1_new = pts0_new;

    // If we have points, do KLT tracking to get the valid projections into the right image
    if (!kpts0_new.empty()) {

      // Do our KLT tracking from the left to the right frame of reference
      // Note: we have a pretty big window size here since our projection might be bad
      // Note: but this might cause failure in cases of repeated textures (eg. checkerboard)
      std::vector<uchar> mask;
      perform_matching(img0pyr, img1pyr, kpts0_new, kpts1_new, cam_id_left, cam_id_right, mask);
      // std::vector<float> error;
      // cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
      // cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0_new, pts1_new, mask, error, win_size, pyr_levels, term_crit,
      //                          cv::OPTFLOW_USE_INITIAL_FLOW);


      // Loop through and record only ones that are valid
      for (size_t i = 0; i < kpts0_new.size(); i++) {

        // Check that it is in bounds
        if ((int)kpts0_new.at(i).pt.x < 0 || (int)kpts0_new.at(i).pt.x >= img0pyr.at(0).cols || (int)kpts0_new.at(i).pt.y < 0 ||
            (int)kpts0_new.at(i).pt.y >= img0pyr.at(0).rows) {
          continue;
        }

        // isaac: we adopt all points in kpts0_new (except those failed the bound check above) dispite 
        // their matching result to the right image.

        // update the uv coordinates
        // kpts0_new.at(i).pt = pts0_new.at(i);
        // append the new uv coordinate
        pts0.push_back(kpts0_new.at(i));
        // move id forward and append this new point
        size_t temp = ++currid;
        ids0.push_back(temp);


        if ((int)kpts1_new.at(i).pt.x < 0 || (int)kpts1_new.at(i).pt.x >= img1pyr.at(0).cols || (int)kpts1_new.at(i).pt.y < 0 ||
            (int)kpts1_new.at(i).pt.y >= img1pyr.at(0).rows) {
          continue;
        }

        if (mask[i] != 1) {
          continue;
        }

        // Check to see if it there is already a feature in the right image at this location
        //  1) If this is not already in the right image, then we should treat it as a stereo
        //  2) Otherwise we will treat this as just a monocular track of the feature
        // TODO: we should check to see if we can combine this new feature and the one in the right
        // TODO: seems if reject features which overlay with right features already we have very poor tracking perf

        // update the uv coordinates
        // kpts1_new.at(i).pt = pts1_new.at(i);
        // append the new uv coordinate
        pts1.push_back(kpts1_new.at(i));
        // append this new point
        ids1.push_back(temp);
      }
    }
  }

  // RIGHT: Now summarise the number of tracks in the right image
  // RIGHT: We will try to extract some monocular features if we have the room
  // RIGHT: This will also remove features if there are multiple in the same location
  cv::Size size_close1((int)((float)img1pyr.at(0).cols / (float)min_px_dist), (int)((float)img1pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
  float size_x1 = (float)img1pyr.at(0).cols / (float)grid_x;
  float size_y1 = (float)img1pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid1(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
  it0 = pts1.begin();
  it1 = ids1.begin();
  while (it0 != pts1.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img1pyr.at(0).cols - edge || y < edge || y >= img1pyr.at(0).rows - edge) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x1);
    int y_grid = std::floor(kpt.pt.y / size_y1);
    if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Check if this is a stereo point
    bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
    // isaac: skip this check for old features. (we'd like to keep all the old features even they're near each other)
    // Check if this keypoint is near another point
    // NOTE: if it is *not* a stereo point, then we will not delete the feature
    // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
    // if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127 && !is_stereo) {
    //   it0 = pts1.erase(it0);
    //   it1 = ids1.erase(it1);
    //   continue;
    // }

    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask1.at<uint8_t>(y, x) > 127) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
    if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
    }
    it0++;
    it1++;
  }

  // RIGHT: if we need features we should extract them in the current frame
  // RIGHT: note that we don't track them to the left as we already did left->right tracking above
  int num_featsneeded_1 = num_features - (int)pts1.size();
  if (num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts1_ext;
    // Grider_FAST::perform_griding(img1pyr.at(0), mask1, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask1_grid;
    cv::resize(mask1, mask1_grid, size_grid1, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid1.cols; x++) {
      for (int y = 0; y < grid_2d_grid1.rows; y++) {
        if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts1_ext;
    Grider_GRID::perform_griding(img1pyr.at(0), mask1, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    for (auto &kpt : pts1_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      pts1.push_back(kpt);
      size_t temp = ++currid;
      ids1.push_back(temp);
      grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
    }
  }
}

void TrackKLT::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out,
                                const Eigen::Matrix3d* R_0_in_1, const Eigen::Vector3d* t_0_in_1) {
  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1;
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // // If we don't have enough points for ransac just return empty
  // // We set the mask to be all zeros since all points failed RANSAC
  // if (pts0.size() < 10) {
  //   for (size_t i = 0; i < pts0.size(); i++)
  //     mask_out.push_back((uchar)0);
  //   return;
  // }

  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

  // Normalize these points, so we can then do ransac
  // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
  std::vector<cv::Point2f> pts0_n, pts1_n;
  for (size_t i = 0; i < pts0.size(); i++) {
    pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
    pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
  }

  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
  std::vector<uchar> mask_rsc;
  double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
  double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);

  if (force_fundamental || !R_0_in_1) {
    // cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);
    fundamental_ransac(pts0_n, pts1_n, 2.0 / max_focallength, mask_rsc);
  } else {
    if (t_0_in_1) {
      known_essential_check(*R_0_in_1, *t_0_in_1, pts0_n, pts1_n, mask_rsc, 3.0 / max_focallength);
    } else {
      const double success_probability = 0.99;
      const int max_iter = ceil(log(1-success_probability) / log(1-0.7*0.7));  // = 7
      two_point_ransac(*R_0_in_1, pts0_n, pts1_n, mask_rsc, 1.0 / max_focallength, 3.0 / max_focallength, max_iter);
    }

    // std::vector<size_t> selected_indices;
    // select_masked(mask_klt, selected_indices);
    // std::vector<cv::Point2f> selected_pts0_n, selected_pts1_n;
    // selected_pts0_n.reserve(selected_indices.size());
    // selected_pts1_n.reserve(selected_indices.size());
    // for (size_t idx : selected_indices) {
    //   selected_pts0_n.push_back(pts0_n[idx]);
    //   selected_pts1_n.push_back(pts1_n[idx]);
    // }
    // std::vector<uchar> selected_mask;

    // if (t_0_in_1) {
    //   known_essential_check(*R_0_in_1, *t_0_in_1, selected_pts0_n, selected_pts1_n, selected_mask, 3.0 / max_focallength);
    // } else {
    //   const double success_probability = 0.99;
    //   const int max_iter = ceil(log(1-success_probability) / log(1-0.7*0.7));  // = 7
    //   two_point_ransac(*R_0_in_1, selected_pts0_n, selected_pts1_n, selected_mask, 1.0 / max_focallength, 3.0 / max_focallength, max_iter);
    // }

    // apply_selected_mask(selected_mask, selected_indices, mask_rsc);
  }

  // Loop through and record only ones that are valid
  for (size_t i = 0; i < mask_klt.size(); i++) {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
    mask_out.push_back(mask);
  }

  // Copy back the updated positions
  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}
