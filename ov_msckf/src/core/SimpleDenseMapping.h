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

#ifndef OV_MSCKF_SIMPLE_DENSE_MAPPING_H
#define OV_MSCKF_SIMPLE_DENSE_MAPPING_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <glog/logging.h>

#include <Eigen/Geometry>
#include "cam/CamBase.h"

namespace ov_msckf {
namespace dense_mapping {


using Color = Eigen::Matrix<uint8_t, 3, 1>;

using Timestamp = double;

struct Position : public Eigen::Matrix<int32_t, 3, 1> {
  Position(int32_t x=0, int32_t y=0, int32_t z=0) : Eigen::Matrix<int32_t, 3, 1>(x, y, z) {}
  
  bool operator< (const Position& other) const {
    if (x() == other.x()) {
      if (y() == other.y()) {
        return z() < other.z();
      }
      return y() < other.y();
    }
    return x() < other.x();
  }
};

struct Voxel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Position p;
  Color c;
  Timestamp time;
};

struct SimpleDenseMap {
  std::vector<Voxel> voxels;
  double resolution;
  Timestamp time;
};

class SimpleDenseMapBuilder {

public:

  using Voxel = ov_msckf::dense_mapping::Voxel;
  SimpleDenseMapBuilder(
    size_t max_voxels=1000000,
    float resolution = 0.01,
    float max_depth = 5.0,
    float max_height = 1.0,
    float min_height = -1.0,
    int max_display = -1) :
      stop_insert_thread_request_(false),
      max_display_(max_display),
      output_mutex_(new std::mutex()),
      max_voxels_(max_voxels),
      resolution_(resolution),
      max_height_(max_height),
      min_height_(min_height),
      max_depth_(max_depth) {
    voxels_.resize(max_voxels_);
    for (size_t i=0; i<max_voxels_; i++) {
      unused_entries_.insert(i);
    }
    insert_thread_ = std::thread(&SimpleDenseMapBuilder::insert_thread_func, this);
  }

  ~SimpleDenseMapBuilder() {
    stop_insert_thread();
  }

  void set_output_update_callback(std::function<void(std::shared_ptr<const SimpleDenseMap>)> cb) {
    output_update_callback_ = cb;
  }

  float resolution() const {
    return resolution_;
  }

  void feed_rgbd_frame(const cv::Mat& color, const cv::Mat& depth,
                       ov_core::CamBase&& cam,
                       const Eigen::Isometry3f& T_W_C,
                       const Timestamp& time,
                       int pixel_downsample = 1,
                       int start_row = 0,
                       int end_row = -1,
                       int start_col = 0,
                       int end_col = -1) {
    std::unique_lock<std::mutex> lk(insert_thread_mutex_);
    auto cam_clone = cam.clone();
    insert_tasks_.push_back([=](){
      LOG(INFO) << "RgbdMapping:  task - begin";
      std::shared_ptr<const SimpleDenseMap> output;
      {
        std::unique_lock<std::mutex> lk(mapping_mutex_);
        LOG(INFO) << "RgbdMapping:  task - get mutex";
        insert_rgbd_frame(color, depth, std::move(*cam_clone), T_W_C, time, pixel_downsample, start_row, end_row, start_col, end_col);
        if (time > time_) {
          time_ = time;
        }
        LOG(INFO) << "RgbdMapping:  task - map updated";
        output = update_output();
      }
      LOG(INFO) << "RgbdMapping:  task - output updated";
      if (output_update_callback_) {        
        output_update_callback_(output);
      }
      LOG(INFO) << "RgbdMapping:  task - callback invoked";
    });
    LOG(INFO) << "RgbdMapping:  add new task - task size = " << insert_tasks_.size();
    insert_thread_cv_.notify_one();
  }

  // Return occupied voxels sorted by latest update time.
  std::shared_ptr<const SimpleDenseMap>
  get_output_map() const {
    std::unique_lock<std::mutex> lk(*output_mutex_);
    return output_;
  }

  std::shared_ptr<const SimpleDenseMap>
  get_display_map() const {
    auto output = get_output_map();
    if (!output) {
      return nullptr;
    }
    if (max_display_ > 0 && output->voxels.size() > max_display_) {
      SimpleDenseMap map_copy;
      map_copy.voxels.reserve(max_display_);
      // Add the newest half max_display_ voxels
      map_copy.voxels.insert(map_copy.voxels.end(), output->voxels.end() - max_display_ / 2, output->voxels.end());

      // and half max_display_ sampled older voxels
      for (size_t i=0; i<max_display_ / 2; i++) {
        int idx = rand() % (output->voxels.size() - max_display_ / 2);
        map_copy.voxels.push_back(output->voxels.at(idx));
      }
      map_copy.resolution = output->resolution;
      map_copy.time = output->time;
      return std::make_shared<const SimpleDenseMap>(std::move(map_copy));
    } else {
      return output;
    }
  }

  void clear_map() {
    LOG(INFO) << "RgbdMapping::clear_map: begin";
    std::deque<std::function<void()>> insert_tasks;
    std::set<size_t> unused_entries;
    for (size_t i=0; i<max_voxels_; i++) {
      unused_entries.insert(i);
    }
    std::map<Timestamp, std::set<size_t>> time_to_voxels;
    std::map<Position, size_t> pos_to_voxel;

    LOG(INFO) << "RgbdMapping::clear_map: swapping tasks";
    auto output = std::make_shared<const SimpleDenseMap>();
    {
      std::unique_lock<std::mutex> lk1(insert_thread_mutex_);
      std::swap(insert_tasks, insert_tasks_);
    }
    LOG(INFO) << "RgbdMapping::clear_map: swapping data";
    {
      std::unique_lock<std::mutex> lk2(mapping_mutex_);
      std::unique_lock<std::mutex> lk3(*output_mutex_);
      std::swap(unused_entries, unused_entries_);
      std::swap(time_to_voxels, time_to_voxels_);
      std::swap(pos_to_voxel, pos_to_voxel_);
      std::swap(output, output_);
    }

    LOG(INFO) << "RgbdMapping::clear_map: clearing tasks and data";
    insert_tasks.clear();
    unused_entries.clear();
    time_to_voxels.clear();
    pos_to_voxel.clear();
    LOG(INFO) << "RgbdMapping::clear_map: done";
  }

protected:

  void insert_thread_func() {
    pthread_setname_np(pthread_self(), "ov_map_rgbd");
    while (1) {
      std::function<void()> insert_task;
      {
        std::unique_lock<std::mutex> lk(insert_thread_mutex_);
        insert_thread_cv_.wait(lk, [this] {return stop_insert_thread_request_ || insert_tasks_.empty() == false;});
        if (stop_insert_thread_request_) {
          return;
        }
        const int MAX_INSERT_TASK_CACHE_SIZE = 3;
        int abandon_count = 0;
        while (insert_tasks_.size() > MAX_INSERT_TASK_CACHE_SIZE + 1) {
          insert_tasks_.pop_front();
          abandon_count++;
        }
        if (abandon_count > 0) {
          LOG(INFO) << "RgbdMapping:  Abandoned " << abandon_count << " tasks.";
        }
        insert_task = insert_tasks_.front();
        insert_tasks_.pop_front();
        LOG(INFO) << "RgbdMapping:  remain " << insert_tasks_.size() << " tasks.";
      }

      LOG(INFO) << "RgbdMapping:  Processing  begein ... ";
      insert_task();
      LOG(INFO) << "RgbdMapping:  Processing finished.";
    }
  }

  void stop_insert_thread() {
    {
      std::unique_lock<std::mutex> lk(insert_thread_mutex_);
      stop_insert_thread_request_ = true;
      insert_thread_cv_.notify_all();
    }
    if (insert_thread_.joinable()) {
      insert_thread_.join();
    }
  }

  void insert_voxel(const Position& p, const Color& c, const Timestamp& time) {
    if (pos_to_voxel_.count(p)) {
      // update the existing voxel
      size_t idx = pos_to_voxel_.at(p);
      time_to_voxels_.at(voxels_[idx].time).erase(idx);
      voxels_[idx].c = c;
      voxels_[idx].time = time;
      time_to_voxels_[time].insert(idx);
      return;
    }

    // add new voxel
    while (unused_entries_.empty()) {
      for (size_t idx : time_to_voxels_.begin()->second) {
        pos_to_voxel_.erase(voxels_[idx].p);
      }
      unused_entries_ = std::move(time_to_voxels_.begin()->second);
      time_to_voxels_.erase(time_to_voxels_.begin());
    }

    size_t idx = *unused_entries_.begin();
    unused_entries_.erase(unused_entries_.begin());
    voxels_[idx].p = p;
    voxels_[idx].c = c;
    voxels_[idx].time = time;
    pos_to_voxel_[p] = idx;
    time_to_voxels_[time].insert(idx);
  }

  void insert_rgbd_frame(const cv::Mat& color, const cv::Mat& depth,
                         ov_core::CamBase&& cam,
                         const Eigen::Isometry3f& T_W_C,
                         const Timestamp& time,
                         int pixel_downsample = 1,
                         int start_row = 0,
                         int end_row = -1,
                         int start_col = 0,
                         int end_col = -1) {
    if (end_row < 0) {
      end_row = color.rows;
    }
    if (end_col < 0) {
      end_col = color.cols;
    }
    for (size_t y=start_row; y<end_row; y+=pixel_downsample) {
      for (size_t x=0; x<end_col; x+=pixel_downsample) {
        const uint16_t d = depth.at<uint16_t>(y,x);
        if (d == 0) continue;
        float depth = d / 1000.0f;
        if (max_depth_ < depth) {
          continue;
        }

        Eigen::Vector2f p_normal = cam.undistort_f(Eigen::Vector2f(x, y));
        Eigen::Vector3f p3d_c(p_normal.x(), p_normal.y(), 1.0f);
        p3d_c = p3d_c * depth;
        Eigen::Vector3f p3d_w = T_W_C * p3d_c;
        if (p3d_w.z() > max_height_ || p3d_w.z() < min_height_) {
          continue;
        }
        p3d_w /= resolution_;
        Position pos(round(p3d_w.x()), round(p3d_w.y()), round(p3d_w.z()));
        auto rgb = color.at<cv::Vec3b>(y,x);
        uint8_t r = rgb[0];
        uint8_t g = rgb[1];
        uint8_t b = rgb[2];
        Color c(r,g,b);
        insert_voxel(pos, c, time);
      }
    }
  }

  std::shared_ptr<const SimpleDenseMap> update_output() {
    SimpleDenseMap output;
    output.voxels.reserve(voxels_.size() - unused_entries_.size());
    for (const auto& pair : time_to_voxels_) {
      for (size_t i : pair.second) {
        output.voxels.push_back(voxels_[i]);
      }
    }
    output.resolution = resolution_;
    output.time = time_;

    std::unique_lock<std::mutex> lk(*output_mutex_);
    output_ = std::make_shared<const SimpleDenseMap>(std::move(output));
    return output_;
  }

protected:
  std::mutex mapping_mutex_;
  std::vector<Voxel> voxels_;
  std::set<size_t> unused_entries_;
  std::map<Timestamp, std::set<size_t>> time_to_voxels_;
  std::map<Position, size_t> pos_to_voxel_;

  // insert_thread
  std::thread insert_thread_;
  std::mutex insert_thread_mutex_;
  std::condition_variable insert_thread_cv_;
  std::deque<std::function<void()>> insert_tasks_;
  bool stop_insert_thread_request_;
  
  // output voxels
  std::shared_ptr<std::mutex> output_mutex_;
  std::shared_ptr<const SimpleDenseMap> output_;

  // output update callback
  std::function<void(std::shared_ptr<const SimpleDenseMap>)> output_update_callback_;

  const size_t max_voxels_ = 1000000;
  const float max_depth_ = 5.0;
  const float max_height_ = 5.0;
  const float min_height_ = 5.0;
  const float resolution_ = 0.01;
  int max_display_ = -1;
  Timestamp time_ = -1.0;
};



}  // namespace dense_mapping

} // namespace ov_msckf


#endif // OV_MSCKF_SIMPLE_DENSE_MAPPING_H
