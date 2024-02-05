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
#include <unordered_set>

#include <glog/logging.h>

#include <Eigen/Geometry>
#include "cam/CamBase.h"

#ifdef USE_HEAR_SLAM
#include "hear_slam/basic/thread_pool.h"
#include "hear_slam/basic/logging.h"
#include "hear_slam/basic/time.h"
#endif

namespace ov_msckf {
namespace dense_mapping {


using Timestamp = double;
using VoxColor = Eigen::Matrix<uint8_t, 3, 1>;
using VoxPosition = Eigen::Matrix<int32_t, 3, 1>;
using PixPosition = Eigen::Matrix<int32_t, 2, 1>;
using VoxelKey = VoxPosition;
using BlockKey3 = VoxPosition;
using PixelKey = PixPosition;
using BlockKey2 = PixPosition;
static const BlockKey3 InvalidBlockKey3 = BlockKey3(INT_MAX, INT_MAX, INT_MAX);

template <size_t SIZE>
struct AlignedBufT {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EIGEN_ALIGN16 char buf[SIZE];
};

struct Voxel final {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EIGEN_ALIGN16 VoxPosition p;
  VoxColor c;
  Timestamp time;
  bool valid;
};

union VoxelSimple {
  uint8_t v[8];
  struct {
    uint8_t x;
    uint8_t y;
    uint8_t z;
    uint8_t s;  // state

    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
  };
};

struct SpatialHash3 {
  inline size_t operator()(const BlockKey3& p) const {
    return size_t(((p[0]) * 73856093) ^ ((p[1]) * 471943) ^ ((p[2]) * 83492791));
  }
};

struct SpatialHash2 {
  inline size_t operator()(const BlockKey2& p) const {
    return size_t(((p[0]) * 73856093) ^ ((p[1]) * 471943));
  }
};

struct CubeBlock final {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // static constexpr size_t kMaxVoxelsPow = 10;
  // static constexpr size_t kMaxVoxels = 1 << kMaxVoxelsPow;  // 1024

  // static constexpr size_t kMaxVoxelsPow = 9;
  // static constexpr size_t kMaxVoxels = 1 << kMaxVoxelsPow;  // 512

  static constexpr size_t kMaxVoxelsPow = 8;
  static constexpr size_t kMaxVoxels = 1 << kMaxVoxelsPow;  // 256

  // static constexpr size_t kMaxVoxelsPow = 6;
  // static constexpr size_t kMaxVoxels = 1 << kMaxVoxelsPow;  // 64



  static constexpr size_t kMaxVoxelsMask = kMaxVoxels - 1;
  static constexpr size_t kVoxelsBytes = kMaxVoxels * sizeof(Voxel);

  static constexpr size_t kSideLengthPow = 3;
  static constexpr size_t kSideLength = 1 << kSideLengthPow;  // 8
  static SpatialHash3 hash;

  Voxel voxels[kMaxVoxels]; // 64 voxels per block
  double time;

  CubeBlock() : time(0.0) {}

  inline size_t getIndex(const VoxelKey& p) const {
    return hash(p) & kMaxVoxelsMask;
  }

  inline void put(const Voxel& v) {
    size_t idx = getIndex(v.p);
    auto& vox = voxels[idx];
    vox = v;
    vox.valid = true;
  }

  inline void reset() {
    std::memset(voxels, 0, sizeof(voxels));
    time = 0.0;
  }

  std::shared_ptr<CubeBlock> clone() {
    using AlignedBuf = AlignedBufT<sizeof(CubeBlock)>;
    auto* buf = new AlignedBuf;
    std::memcpy(buf->buf, this, sizeof(CubeBlock));
    std::shared_ptr<CubeBlock> clone(
        reinterpret_cast<CubeBlock*>(buf),
        [](CubeBlock* ptr){delete reinterpret_cast<AlignedBuf*>(ptr);});
    return clone;
  }
};

template<
    size_t _reserved_blocks_pow = 18  // reserved_blocks = 2^18 = 256K by default.
    // size_t _reserved_blocks_pow = 16  // reserved_blocks = 2^16 = 64K by default.
  >
struct SimpleDenseMapT final {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static constexpr size_t kReservedBlocksPow = _reserved_blocks_pow;
  static constexpr size_t kReservedBlocks = 1 << kReservedBlocksPow;
  static constexpr size_t kReservedBlocksMask = kReservedBlocks - 1;
  static constexpr size_t kMaxBlocks = kReservedBlocks * 25 / 100;  // load factor 0.25
  // static constexpr size_t kMaxBlocks = kReservedBlocks * 75 / 100;  // load factor 0.75
  using BlockKey3 = VoxPosition;

  double resolution;
  Timestamp time;

  struct OutputVoxels;

 private:
  std::shared_ptr<OutputVoxels> output;
  std::unordered_map<BlockKey3, std::shared_ptr<CubeBlock>, SpatialHash3> blocks_map;
  std::map<double, std::unordered_set<BlockKey3, SpatialHash3>> time_to_blocks;
  std::unordered_map<BlockKey2, std::unordered_set<BlockKey3::Scalar>, SpatialHash2> xy_to_z_blocks;

  std::unordered_map<BlockKey3, double, SpatialHash3> updated_block_to_old_time;

 public:

  struct OutputVoxels {
    Voxel* output_voxels;
    size_t output_voxels_size;

    using AlignedBuf = AlignedBufT<sizeof(CubeBlock)>;
    OutputVoxels(size_t n_blocks) {
      output_voxels_size = 0;
      char* buf = reinterpret_cast<char*>(new AlignedBuf[n_blocks]);
      output_voxels = reinterpret_cast<Voxel*>(buf);
    }
    ~OutputVoxels() {
      delete[] reinterpret_cast<AlignedBuf*>(output_voxels);
    }
  };


  SimpleDenseMapT() :
    time(-1.0),
    thread_pool_group(nullptr),
    resolution(0.01) {
    blocks_map.rehash(kReservedBlocks);
    // output = std::make_shared<OutputVoxels>(kMaxBlocks);
    output = std::make_shared<OutputVoxels>(kReservedBlocks);
  }

  ~SimpleDenseMapT() {
  }

  template<typename Vec>
  inline std::pair<BlockKey3, VoxelKey>
  getKeysOfPoint(const Vec& point) const {
    VoxelKey  vk = (point / resolution).template cast<typename VoxelKey::Scalar>();
    BlockKey3 bk(vk.x() >> CubeBlock::kSideLengthPow,
                 vk.y() >> CubeBlock::kSideLengthPow,
                 vk.z() >> CubeBlock::kSideLengthPow);
    return std::make_pair(bk, vk);
  }

  void generateOutput() {
    // ASSERT(blocks_map.size() < kMaxBlocks);
    ASSERT(blocks_map.size() < kReservedBlocks);

    // return;  // skip

    if (blocks_map.empty()) {
      return;
    }

    // auto new_output = std::make_shared<OutputVoxels>(blocks_map.size());
    // char* buf = reinterpret_cast<char*>(new_output->output_voxels);
    char* buf = reinterpret_cast<char*>(output->output_voxels);
    size_t i = 0;
    char* cur = buf;

#ifdef USE_HEAR_SLAM
    // std::memcpy(buf->buf, this, sizeof(SimpleDenseMapT<_reserved_blocks_pow>));
    auto pool = thread_pool_group->getNamed("ov_map_rgbd_w");
    ASSERT(pool);
    ASSERT(pool->numThreads() == std::thread::hardware_concurrency());
    using hear_slam::TaskID;
    using hear_slam::INVALID_TASK;
    std::vector<TaskID> task_ids;

    auto enqueue_jobs = [&]() {
      for (auto it = blocks_map.begin(); it != blocks_map.end(); ++it) {
        // size_t n_bytes = sizeof(item.second->voxels);
        size_t n_bytes = CubeBlock::kMaxVoxels * sizeof(Voxel);
        auto new_task = pool->schedule([=](){
          std::memcpy(cur, it->second->voxels, n_bytes);
        });
        cur += n_bytes;
        task_ids.emplace_back(new_task);
      }
    };

    pool->wait(pool->schedule(enqueue_jobs));
    pool->waitTasks(task_ids.rbegin(), task_ids.rend());
#else
    for (auto it = blocks_map.begin(); it != blocks_map.end(); ++it) {
      // size_t n_bytes = sizeof(item.second->voxels);
      size_t n_bytes = CubeBlock::kMaxVoxels * sizeof(Voxel);
      std::memcpy(cur, it->second->voxels, n_bytes);
      cur += n_bytes;
    }
#endif
    output->output_voxels_size = blocks_map.size() * CubeBlock::kMaxVoxels;

    // new_output->output_voxels_size = blocks_map.size() * CubeBlock::kMaxVoxels;
    // std::swap(output, new_output);
  }

  inline const Voxel* voxels() const {
    // return blocks[0].voxels;    
    return output ? output->output_voxels : nullptr;
  }

  inline size_t reservedVoxelSize() const {
    return output ? output->output_voxels_size : 0;
  }

  template<typename Vec>
  inline void insert(const Vec& p, const VoxColor& c, const Timestamp& time) {
    auto keys = getKeysOfPoint(p);
    auto& bk = keys.first;
    auto& vk = keys.second;

    Voxel v;
    v.c = c;
    v.time = time;
    v.p = vk;

    std::unique_lock<std::mutex> lk(mapping_mutex_);

    auto it_blk = blocks_map.find(bk);
    if (it_blk == blocks_map.end()) {
      it_blk = blocks_map.insert({bk, std::make_shared<CubeBlock>()}).first;
    }

    it_blk->second->put(v);
    time_to_blocks[time].insert(bk);
    xy_to_z_blocks[BlockKey2(bk.x(), bk.y())].insert(bk.z());

    // update the block's time.
    auto old_time = it_blk->second->time;
    bool need_record_old_time = (updated_block_to_old_time.find(bk) == updated_block_to_old_time.end());
    if (time != old_time) {
      it_blk->second->time = time;
      ASSERT(need_record_old_time);
      updated_block_to_old_time[bk] = old_time;
    } else if (need_record_old_time) {
      updated_block_to_old_time[bk] = -1.0;
    }
  }

  void removeOldBlocksIfNeeded() {
    for (auto& item : updated_block_to_old_time) {
      auto& bk = item.first;
      auto& old_time = item.second;
      if (old_time > 0) {
        auto it = time_to_blocks.find(old_time);
        ASSERT(it != time_to_blocks.end());
        it->second.erase(bk);
        // if (it->second.empty()) {
        //   time_to_blocks.erase(it);
        // }
      }
    }

    while (blocks_map.size() > kMaxBlocks) {
      if (time_to_blocks.size() <= 1) {  // keep at least 1 frame
        break;
      }
      for (const auto& bk_to_remove : time_to_blocks.begin()->second) {
        blocks_map.erase(bk_to_remove);

        BlockKey2 bk2(bk_to_remove.x(), bk_to_remove.y());
        auto it = xy_to_z_blocks.find(bk2);
        ASSERT(it != xy_to_z_blocks.end());
        it->second.erase(bk_to_remove.z());
        if (it->second.empty()) {
          xy_to_z_blocks.erase(it);
        }
      }
      time_to_blocks.erase(time_to_blocks.begin());
    }

    updated_block_to_old_time.clear();
    return;
  }


  cv::Mat getHeightMap(float center_x, float center_y, float orientation,
                       float center_z = 0.0,
                       float min_h = -3.0,
                       float max_h = 3.0,
                       float hmap_resolution = 0.01,
                       float discrepancy_thr = 0.1) const {
#ifdef USE_HEAR_SLAM
    using hear_slam::TimeCounter;
    TimeCounter tc;
#endif

    static constexpr int rows = 256;
    static constexpr int cols = 256;
    float c = cos(orientation);
    float s = sin(orientation);

    cv::Mat height_map(rows, cols, CV_16UC1, cv::Scalar(0));

    cv::Mat hmax(rows, cols, CV_32SC1, cv::Scalar(INT_MIN));
    cv::Mat hmin(rows, cols, CV_32SC1, cv::Scalar(INT_MAX));
    std::unordered_set<BlockKey2, SpatialHash2> involved_xy_blocks;
    std::unordered_map<PixelKey, std::pair<int, int>, SpatialHash2> voxelxy_to_imgxy;
    float resolution_ratio = hmap_resolution / this->resolution;
    const size_t approx_max_xys = (rows * cols) * (resolution_ratio * resolution_ratio);
    involved_xy_blocks.rehash(approx_max_xys);
    voxelxy_to_imgxy.rehash(approx_max_xys * 4);

    float block_resolution = this->resolution * CubeBlock::kSideLength;

    for (int i=0; i<rows; i++) {
      for (int j=0; j<cols; j++) {
        float local_x = (j - cols/2) * hmap_resolution;
        float local_y = (i - rows/2) * hmap_resolution;
        float global_x = center_x + local_x * c - local_y * s;
        float global_y = center_y + local_x * s + local_y * c;
        BlockKey2 bk2(global_x / block_resolution, global_y / block_resolution);
        PixelKey voxelxy(global_x / this->resolution, global_y / this->resolution);
        voxelxy_to_imgxy[voxelxy] = std::make_pair(j, i);  // x, y
        involved_xy_blocks.insert(bk2);
      }
    }

    std::vector<std::shared_ptr<CubeBlock>> block_data;
    block_data.reserve(involved_xy_blocks.size() * ((max_h - min_h) / block_resolution + 1) * 2);

#ifdef USE_HEAR_SLAM
    tc.tag("BeforeRetrieveBlocks");
#endif
    {
      std::unique_lock<std::mutex> lk(mapping_mutex_);
      for (auto & bk2 : involved_xy_blocks) {
        auto zit = xy_to_z_blocks.find(bk2);
        if (zit != xy_to_z_blocks.end()) {
          for (auto& z : zit->second) {
            block_data.push_back(blocks_map.at(BlockKey3(bk2.x(), bk2.y(), z))->clone());
          }
        }
      }
    }
#ifdef USE_HEAR_SLAM
    tc.tag("AfterRetrieveBlocks");
#endif

    const int32_t center_zi = center_z / this->resolution;
    const int32_t discrepancy_thr_i = discrepancy_thr / this->resolution;
    for (auto & block : block_data) {
      for (size_t i=0; i<CubeBlock::kMaxVoxels; i++) {
        auto &vexel = block->voxels[i];
        if (vexel.valid) {
          auto& vx = vexel.p.x();
          auto& vy = vexel.p.y();
          auto vit = voxelxy_to_imgxy.find(PixelKey(vx, vy));
          if (vit != voxelxy_to_imgxy.end()) {
            auto& vzi = vexel.p.z();
            auto& imgx = vit->second.first;
            auto& imgy = vit->second.second;
            auto& max_zi = hmax.at<int32_t>(imgy, imgx);
            auto& min_zi = hmin.at<int32_t>(imgy, imgx);
            if (vzi > max_zi) {
              max_zi = vzi;
            }
            if (vzi < min_zi) {
              min_zi = vzi;
            }
            if (max_zi - min_zi > discrepancy_thr_i) {
              // height_map.at<uint16_t>(imgy, imgx) = 1;
              height_map.at<uint16_t>(imgy, imgx) = 0;
            } else {
              float h = (min_zi + (max_zi - min_zi) / 2 - center_zi) * this->resolution;
              h = std::min(std::max(h, min_h), max_h); // Clamp to min/max height
              height_map.at<uint16_t>(imgy, imgx) = h / hmap_resolution + 32768;
            }
          }
        }
      }
    }
#ifdef USE_HEAR_SLAM
    tc.tag("HeightmapDone");
    tc.report("HeightmapTiming: ", true);
#endif
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
    Eigen::Vector3f Zc = pose.rotation().col(2);
    float orientation = atan2f(Zc.y(), Zc.x());

    return getHeightMap(center_x, center_y, orientation, center_z, min_h, max_h, hmap_resolution);
  }


  hear_slam::ThreadPoolGroup* thread_pool_group;
  mutable std::mutex mapping_mutex_;

  // EIGEN_ALIGN16 uint8_t _[16];
};

using SimpleDenseMap = SimpleDenseMapT<>;

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
      map_(new SimpleDenseMap()),
      output_(new SimpleDenseMapOutput()),
      max_voxels_(max_voxels),
      resolution_(voxel_resolution),
      max_height_(max_height),
      min_height_(min_height) {
#ifdef USE_HEAR_SLAM
    thread_pool_group_.createNamed("ov_map_rgbd_w", std::thread::hardware_concurrency());
    thread_pool_group_.createNamed("ov_map_height", std::thread::hardware_concurrency());
#endif
    insert_thread_ = std::thread(&SimpleDenseMapBuilder::insert_thread_func, this);
    map_->thread_pool_group = &thread_pool_group_;
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

      LOG(INFO) << "RgbdMapping:  DEBUG_SIZE: sizeof(Voxel)=" << sizeof(Voxel)
                << ", sizeof(CubeBlock)=" << sizeof(CubeBlock)
                << ", CubeBlock::kMaxVoxels=" << CubeBlock::kMaxVoxels
                << ", sizeof(VoxelSimple)=" << sizeof(VoxelSimple);

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
    //   map_copy.voxels.reserve(max_display);
    //   // Add the newest half max_display voxels
    //   map_copy.voxels.insert(map_copy.voxels.end(), output->voxels.end() - max_display / 2, output->voxels.end());

    //   // and half max_display sampled older voxels
    //   for (size_t i=0; i<max_display / 2; i++) {
    //     int idx = rand() % (output->voxels.size() - max_display / 2);
    //     map_copy.voxels.push_back(output->voxels.at(idx));
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
    std::shared_ptr<SimpleDenseMap> map(new SimpleDenseMap());

    LOG(INFO) << "RgbdMapping::clear_map: swapping tasks";
    auto output = std::make_shared<SimpleDenseMapOutput>();
    output->ro_map = std::make_shared<const SimpleDenseMap>();
    {
      std::unique_lock<std::mutex> lk1(insert_thread_mutex_);
      std::swap(insert_tasks, insert_tasks_);
    }
    LOG(INFO) << "RgbdMapping::clear_map: swapping data";
    {
      std::unique_lock<std::mutex> lk2(*output_mutex_);
      std::swap(map, map_);
      std::swap(output, output_);
    }

    LOG(INFO) << "RgbdMapping::clear_map: clearing tasks and data";
    insert_tasks.clear();
    map.reset();
    LOG(INFO) << "RgbdMapping::clear_map: done";
  }

protected:

  void insert_thread_func() {
    pthread_setname_np(pthread_self(), "ov_map_rgbd");
    while (1) {
      LOG(INFO) << "RgbdMapping:  new loop.";
      std::function<void()> insert_task;
      {
        std::unique_lock<std::mutex> lk(insert_thread_mutex_);
        insert_thread_cv_.wait(lk, [this] {LOG(INFO) << "RgbdMapping:  check condition."; return stop_insert_thread_request_ || insert_tasks_.empty() == false;});
        if (stop_insert_thread_request_) {
          return;
        }
        LOG(INFO) << "RgbdMapping:  retrieve task ";
        
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

  template <typename Vec>
  void insert_voxel(const Vec& p, const VoxColor& c, const Timestamp& time) {
    map_->insert(p, c, time);
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
          auto rgb = color.at<cv::Vec3b>(y,x);
          uint8_t& r = rgb[0];
          uint8_t& g = rgb[1];
          uint8_t& b = rgb[2];
          VoxColor c(r,g,b);
          // p3d_w /= resolution_;
          // VoxPosition pos(round(p3d_w.x()), round(p3d_w.y()), round(p3d_w.z()));
          // insert_voxel(pos, c, time);
          insert_voxel(p3d_w, c, time);
        }
      }
    };

#ifdef USE_HEAR_SLAM
    // insert_row_range(start_row, end_row);
    auto pool = thread_pool_group_.getNamed("ov_map_rgbd_w");
    ASSERT(pool);
    ASSERT(pool->numThreads() == std::thread::hardware_concurrency());

    size_t row_group_size = 10 * pixel_downsample;
    using hear_slam::TaskID;
    using hear_slam::INVALID_TASK;
    std::vector<TaskID> task_ids;
    task_ids.reserve((end_row - start_row) / row_group_size + 1);
    for (size_t y=start_row; y<end_row; y+=row_group_size) {
      auto new_task = pool->schedule(
        [&insert_row_range, row_group_size, y](){
          insert_row_range(y, y+row_group_size);
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
    map_->removeOldBlocksIfNeeded();
  }

  void update_output() {
    map_->resolution = resolution_;
    map_->time = time_;
    map_->generateOutput();

    std::unique_lock<std::mutex> lk(*output_mutex_);
    output_->ro_map = map_;
  }

protected:
  std::shared_ptr<SimpleDenseMap> map_;
  // std::vector<Voxel> voxels_;
  // std::set<size_t> unused_entries_;
  // std::map<Timestamp, std::set<size_t>> time_to_voxels_;
  // std::map<VoxPosition, size_t> pos_to_voxel_;

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
