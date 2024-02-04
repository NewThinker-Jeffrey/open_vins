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

#ifdef USE_HEAR_SLAM
#include "hear_slam/basic/thread_pool.h"
#include "hear_slam/basic/logging.h"
#endif

namespace ov_msckf {
namespace dense_mapping {


using VoxColor = Eigen::Matrix<uint8_t, 3, 1>;

using Timestamp = double;

struct VoxPosition : public Eigen::Matrix<int32_t, 3, 1> {
  VoxPosition(int32_t x=0, int32_t y=0, int32_t z=0) : Eigen::Matrix<int32_t, 3, 1>(x, y, z) {}
  
  inline bool operator< (const VoxPosition& other) const {
    return x() < other.x() ||
           (x() == other.x() && y() < other.y()) ||
           (x() == other.x() && y() == other.y() && z() < other.z());
  }
};

struct Position2 : public Eigen::Matrix<int32_t, 2, 1> {
  Position2(int32_t x=0, int32_t y=0) : Eigen::Matrix<int32_t, 2, 1>(x, y) {}
  
  inline bool operator< (const Position2& other) const {
    return x() < other.x() ||
           (x() == other.x() && y() < other.y());
  }
};


struct Voxel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VoxPosition p;
  VoxColor c;
  Timestamp time;
  bool valid;
};

struct SimpleDenseMap {
  std::vector<Voxel> voxels_;
  double resolution;
  Timestamp time;

  using ZSortedVoxels = std::map<VoxPosition::Scalar, size_t>;
  std::map<Position2, ZSortedVoxels> xy_to_voxels;
  

  inline const Voxel* voxels() const {
    // return blocks[0].voxels_;    
    return voxels_.data();
  }

  inline size_t reservedVoxelSize() const {
    return voxels_.size();
  }

  cv::Mat getHeightMap(float center_x, float center_y, float orientation,
                       float center_z = 0.0,
                       float min_h = -3.0,
                       float max_h = 3.0,
                       float hmap_resolution = 0.01,
                       float discrepancy_thr = 0.1) const {
    static constexpr int rows = 256;
    static constexpr int cols = 256;
    float c = cos(orientation);
    float s = sin(orientation);
    cv::Mat height_map(rows, cols, CV_16UC1, cv::Scalar(0));
    for (int i=0; i<rows; i++) {
      for (int j=0; j<cols; j++) {
        float local_x = (j - cols/2) * hmap_resolution;
        float local_y = (i - rows/2) * hmap_resolution;
        float global_x = center_x + local_x * c - local_y * s;
        float global_y = center_y + local_x * s + local_y * c;
        Position2 p(round(global_x / this->resolution), round(global_y / this->resolution));
        auto it = xy_to_voxels.find(p);
        if (it != xy_to_voxels.end()) {
          const auto& z_to_voxel = it->second;
          ASSERT(z_to_voxel.size() > 0);
          if (z_to_voxel.size() > 0) {
            float min_z = z_to_voxel.begin()->first * this->resolution;
            float max_z = z_to_voxel.rbegin()->first * this->resolution;
            float delta_z = max_z - min_z;
            if (delta_z > discrepancy_thr) {
              // LOGW("Heights of voxels with (x,y) = (%f, %f) are not consistent: min_z = %f, max_z = %f, delta_z = %f",
              //      p.x() * this->resolution, p.y() * this->resolution,\
              //      min_z, max_z, delta_z);
              continue;
            }

            float h = min_z + (max_z - min_z) / 2 - center_z;
            h = std::min(std::max(h, min_h), max_h); // Clamp to min/max height
            height_map.at<uint16_t>(i, j) = h / hmap_resolution + 32768;
          }
        }
      }
    }
    return height_map;
  }

  cv::Mat getHeightMap(const Eigen::Isometry3f& pose,
                       float min_h = -1.5,
                       float max_h = 0.0,
                       float hmap_resolution = 0.01,
                       float discrepancy_thr = 0.1) const {
    float center_x = pose.translation().x();
    float center_y = pose.translation().y();
    float center_z = pose.translation().z();
    // float center_z = 0.0;  // Fix the z-center plane?
    // min_h = pose.translation().z() - min_h;
    // max_h = pose.translation().z() + max_h;
    auto Zc = pose.rotation().col(2);
    float orientation = atan2(Zc.y(), Zc.x());
    return getHeightMap(center_x, center_y, orientation, center_z, min_h, max_h, hmap_resolution);
  }
};

struct SimpleDenseMapOutput {
  std::shared_ptr<const dense_mapping::SimpleDenseMap> ro_map;
};

class SimpleDenseMapBuilder {

public:

  using Voxel = ov_msckf::dense_mapping::Voxel;
  SimpleDenseMapBuilder(
    float voxel_resolution = 0.01,
    size_t max_voxels=1000000,
    float max_height = FLT_MAX,
    float min_height = -FLT_MAX) :
      stop_insert_thread_request_(false),
      output_mutex_(new std::mutex()),
      output_(new SimpleDenseMapOutput()),
      max_voxels_(max_voxels),
      resolution_(voxel_resolution),
      max_height_(max_height),
      min_height_(min_height) {
    voxels_.resize(max_voxels_);
    output_->ro_map = std::make_shared<SimpleDenseMap>();
    for (size_t i=0; i<max_voxels_; i++) {
      unused_entries_.insert(i);
    }
#ifdef USE_HEAR_SLAM
    thread_pool_group_.createNamed("ov_map_rgbd_w", std::thread::hardware_concurrency());
#endif
    insert_thread_ = std::thread(&SimpleDenseMapBuilder::insert_thread_func, this);
  }

  ~SimpleDenseMapBuilder() {
    stop_insert_thread();
  }

  using UndistortFunction = std::function<Eigen::Vector2f(size_t x, size_t y)>;
  void registerCamera(size_t cam_id, int width, int height, UndistortFunction undistort_func) {
    // todo: use width and height to construct a table to accelerate undistortion.
    cam_to_undistort_[cam_id] = undistort_func;
  }

  void set_output_update_callback(std::function<void(std::shared_ptr<SimpleDenseMapOutput>)> cb) {
    output_update_callback_ = cb;
  }

  float resolution() const {
    return resolution_;
  }

  void feed_rgbd_frame(const cv::Mat& color, const cv::Mat& depth,
                       size_t cam,
                       const Eigen::Isometry3f& T_W_C,
                       const Timestamp& time,
                       int pixel_downsample = 1,
                       float max_depth = 5.0,
                       int start_row = 0,
                       int end_row = -1,
                       int start_col = 0,
                       int end_col = -1) {
    std::unique_lock<std::mutex> lk(insert_thread_mutex_);
    insert_tasks_.push_back([=](){
      LOG(INFO) << "RgbdMapping:  task - begin";

      insert_rgbd_frame(color, depth, cam, T_W_C, time, pixel_downsample, max_depth, start_row, end_row, start_col, end_col);

      if (time > time_) {
        time_ = time;
      }
      
      LOG(INFO) << "RgbdMapping:  task - map updated";
      update_output();

      LOG(INFO) << "RgbdMapping:  task - output updated";
      if (output_update_callback_) {        
        output_update_callback_(output_);
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
    return output_->ro_map;
  }

  std::shared_ptr<const SimpleDenseMap>
  get_display_map(int max_display = -1) const {
    auto output = get_output_map();
    if (!output) {
      return nullptr;
    }
    return output;
    // if (max_display > 0 && output->voxels.size() > max_display) {
    //   SimpleDenseMap map_copy;
    //   map_copy.voxels_.reserve(max_display);
    //   // Add the newest half max_display voxels
    //   map_copy.voxels_.insert(map_copy.voxels_.end(), output->voxels.end() - max_display / 2, output->voxels.end());

    //   // and half max_display sampled older voxels
    //   for (size_t i=0; i<max_display / 2; i++) {
    //     int idx = rand() % (output->voxels.size() - max_display / 2);
    //     map_copy.voxels_.push_back(output->voxels.at(idx));
    //   }
    //   map_copy.resolution = output->resolution;
    //   map_copy.time = output->time;
    //   return std::make_shared<const SimpleDenseMap>(std::move(map_copy));
    // } else {
    //   return output;
    // }
  }

  void clear_map() {
    LOG(INFO) << "RgbdMapping::clear_map: begin";
    std::deque<std::function<void()>> insert_tasks;
    std::set<size_t> unused_entries;
    for (size_t i=0; i<max_voxels_; i++) {
      unused_entries.insert(i);
    }
    std::map<Timestamp, std::set<size_t>> time_to_voxels;
    std::map<VoxPosition, size_t> pos_to_voxel;

    LOG(INFO) << "RgbdMapping::clear_map: swapping tasks";
    auto output = std::make_shared<SimpleDenseMapOutput>();
    output->ro_map = std::make_shared<const SimpleDenseMap>();
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
      // std::swap(map, map_);
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
        // const int MAX_INSERT_TASK_BUFFER_SIZE = 3;
        const int MAX_INSERT_TASK_BUFFER_SIZE = 0;
        int abandon_count = 0;
        while (insert_tasks_.size() > MAX_INSERT_TASK_BUFFER_SIZE + 1) {
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

  void insert_voxel(const VoxPosition& p, const VoxColor& c, const Timestamp& time) {
    std::unique_lock<std::mutex> lk(mapping_mutex_);
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
    voxels_[idx].valid = true;
    pos_to_voxel_[p] = idx;
    time_to_voxels_[time].insert(idx);
  }

  void insert_rgbd_frame(const cv::Mat& color, const cv::Mat& depth,
                         size_t cam,
                         const Eigen::Isometry3f& T_W_C,
                         const Timestamp& time,
                         int pixel_downsample = 1,
                         float max_depth = 5.0,
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

    auto& undistort = cam_to_undistort_.at(cam);
    auto insert_row_range = [&](int first_row, int last_row) {
      last_row = std::min(end_row, last_row);
      for (size_t y=first_row; y<last_row; y+=pixel_downsample) {
        for (size_t x=0; x<end_col; x+=pixel_downsample) {
          const uint16_t d = depth.at<uint16_t>(y,x);
          if (d == 0) continue;
          float depth = d / 1000.0f;
          if (max_depth < depth) {
            continue;
          }

          Eigen::Vector2f p_normal = undistort(x, y);
          Eigen::Vector3f p3d_c(p_normal.x(), p_normal.y(), 1.0f);
          p3d_c = p3d_c * depth;
          Eigen::Vector3f p3d_w = T_W_C * p3d_c;
          if (p3d_w.z() > max_height_ || p3d_w.z() < min_height_) {
            continue;
          }
          p3d_w /= resolution_;
          VoxPosition pos(round(p3d_w.x()), round(p3d_w.y()), round(p3d_w.z()));
          auto rgb = color.at<cv::Vec3b>(y,x);
          uint8_t& r = rgb[0];
          uint8_t& g = rgb[1];
          uint8_t& b = rgb[2];
          VoxColor c(r,g,b);
          insert_voxel(pos, c, time);
        }
      }
    };

#ifdef USE_HEAR_SLAM
    auto pool = thread_pool_group_.getNamed("ov_map_rgbd_w");
    ASSERT(pool);
    ASSERT(pool->numThreads() == std::thread::hardware_concurrency());

    size_t row_block_size = 10 * pixel_downsample;
    using hear_slam::TaskID;
    using hear_slam::INVALID_TASK;
    std::vector<TaskID> task_ids;
    task_ids.reserve((end_row - start_row) / row_block_size + 1);
    for (size_t y=start_row; y<end_row; y+=row_block_size) {
      auto new_task = pool->schedule(
        [&insert_row_range, row_block_size, y](){
          insert_row_range(y, y+row_block_size);
        });
      task_ids.emplace_back(new_task);
    }
    pool->waitTasks(task_ids.rbegin(), task_ids.rend());
    // pool->freeze();
    // pool->waitUntilAllTasksDone();
    // pool->unfreeze();
#else
    insert_row_range(start_row, end_row);
#endif
  }

  void update_output() {
    auto output_ptr = std::make_shared<SimpleDenseMap>();
    auto& output = *output_ptr;

    if (unused_entries_.empty()) {
      output.voxels_ = voxels_;
    } else {
      output.voxels_.reserve(voxels_.size() - unused_entries_.size());
      std::vector<size_t> unused_entries;
      unused_entries.reserve(unused_entries_.size());
      for (auto it=unused_entries_.begin(); it!=unused_entries_.end(); it++) {
        unused_entries.push_back(*it);
      }

      auto next_unused_it = unused_entries.begin();
      size_t next_unused = *next_unused_it;
      for (size_t i=0; i<voxels_.size(); i++) {
        if (i != next_unused) {
          output.voxels_.push_back(voxels_[i]);
        } else {
          next_unused_it++;
          if (next_unused_it != unused_entries.end()) {
            next_unused = *next_unused_it;
          }
        }
      }
    }

    // for (const auto& pair : time_to_voxels_) {
    //   for (size_t i : pair.second) {
    //     output.voxels_.push_back(voxels_[i]);
    //   }
    // }

    for (size_t i=0; i<output.voxels_.size(); i++) {
      Position2 p2(output.voxels_[i].p.x(), output.voxels_[i].p.y());
      output.xy_to_voxels[p2][output.voxels_[i].p.z()] = i;
    }

    ASSERT(output.voxels_.size() == voxels_.size() - unused_entries_.size());
    output.resolution = resolution_;
    output.time = time_;

    std::unique_lock<std::mutex> lk(*output_mutex_);
    output_->ro_map = output_ptr;
  }

protected:
  std::mutex mapping_mutex_;
  std::vector<Voxel> voxels_;
  std::set<size_t> unused_entries_;
  std::map<Timestamp, std::set<size_t>> time_to_voxels_;
  std::map<VoxPosition, size_t> pos_to_voxel_;

  // insert_thread
  std::thread insert_thread_;
  std::mutex insert_thread_mutex_;
  std::condition_variable insert_thread_cv_;
  std::deque<std::function<void()>> insert_tasks_;
  bool stop_insert_thread_request_;
  
  // output voxels
  std::shared_ptr<std::mutex> output_mutex_;
  std::shared_ptr<SimpleDenseMapOutput> output_;

  // output update callback
  std::function<void(std::shared_ptr<SimpleDenseMapOutput>)> output_update_callback_;

  const size_t max_voxels_;
  const float max_height_;
  const float min_height_;
  const float resolution_;
  Timestamp time_ = -1.0;

#ifdef USE_HEAR_SLAM
  hear_slam::ThreadPoolGroup thread_pool_group_;
#endif

  std::map<size_t, UndistortFunction> cam_to_undistort_;
};
}  // namespace dense_mapping

} // namespace ov_msckf


#endif // OV_MSCKF_SIMPLE_DENSE_MAPPING_H
