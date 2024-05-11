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
#include "hear_slam/basic/atomic_hash_table.h"
#include "hear_slam/basic/circular_queue.h"
#endif

#define USE_ATOMIC_BLOCK_MAP

namespace ov_msckf {
namespace dense_mapping {


using Timestamp = double;
using VoxColor = Eigen::Matrix<uint8_t, 3, 1>;
using VoxPosition = Eigen::Matrix<int32_t, 3, 1>;
using PixPosition = Eigen::Matrix<int32_t, 2, 1>;

using hear_slam::HashTable;
using hear_slam::HashMap;
using hear_slam::CircularQueue;

struct ComparableVoxPosition : public VoxPosition {
  using Base = VoxPosition;
  ComparableVoxPosition() : Base() {}
  ComparableVoxPosition(int32_t x, int32_t y, int32_t z) : Base(x, y, z) {}
  template <typename T> ComparableVoxPosition(T&& t) : Base(std::forward<T>(t)) {}
  bool operator==(const ComparableVoxPosition& other) const {
    return x() == other.x() && y() == other.y() && z() == other.z();
  }
  bool operator<(const ComparableVoxPosition& other) const {
    return x() < other.x()  ||
          (x() == other.x() && y() < other.y()) ||
          (x() == other.x() && y() == other.y() && z() < other.z());
  }
};

struct ComparablePixPosition : public PixPosition {
  using Base = PixPosition;
  ComparablePixPosition() : Base() {}
  ComparablePixPosition(int32_t x, int32_t y) : Base(x, y) {}
  template <typename T> ComparablePixPosition(T&& t) : Base(std::forward<T>(t)) {}
  bool operator==(const ComparablePixPosition& other) const {
    return x() == other.x() && y() == other.y();
  }
  bool operator<(const ComparablePixPosition& other) const {
    return x() < other.x()  ||
          (x() == other.x() && y() < other.y());
  }
};

using VoxelKey = ComparableVoxPosition;
using BlockKey3 = ComparableVoxPosition;
using RegionKey3 = ComparableVoxPosition;
using PixelKey = ComparablePixPosition;
using BlockKey2 = ComparablePixPosition;

static const BlockKey3 InvalidBlockKey3 = BlockKey3(INT_MAX, INT_MAX, INT_MAX);

template <size_t SIZE>
struct AlignedBufT {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EIGEN_ALIGN16 char buf[SIZE];
};

// struct Voxel final {
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//   EIGEN_ALIGN16 VoxPosition p;
//   VoxColor c;
//   Timestamp time;
//   bool valid;
// };

struct CubeBlock;
struct alignas(8) Voxel {
  alignas(8) union {
    uint64_t v;
    struct {
      uint8_t x;
      uint8_t y;
      uint8_t z;
      uint8_t status;  // state

      uint8_t r;
      uint8_t g;
      uint8_t b;
      uint8_t a;
    };
  } u_;

  Voxel() {}
  Voxel(const Voxel& other) {u_.v = other.u_.v;}
  Voxel(Voxel&& other) {u_.v = other.u_.v;}
  Voxel& operator=(const Voxel& other) {u_.v = other.u_.v; return *this;}
  Voxel& operator=(Voxel&& other) {u_.v = other.u_.v; return *this;}
  inline Voxel atomicClone() const {
    Voxel v;
    v.u_.v = u_.v;
    return v;
  }
  VoxColor c() const { return VoxColor(u_.r, u_.g, u_.b); }
};

struct alignas(8) CubeBlock final {
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
  static constexpr size_t kSideLengthMask = kSideLength - 1;

  Voxel voxels[kMaxVoxels]; // 64 voxels per block
  alignas(8) std::atomic<double> time;
  const BlockKey3 bk;
  const VoxelKey vk0;

  CubeBlock(const BlockKey3& k=BlockKey3(0, 0, 0)) :
      bk(k), vk0(
        k.x() << kSideLengthPow,
        k.y() << kSideLengthPow,
        k.z() << kSideLengthPow
      ) {
    time.store(0.0, std::memory_order_relaxed);
    clearVoxels();
  }

  inline void put(const Voxel& v, size_t idx) {
    Voxel vox = v;
    vox.u_.status = 1;
    voxels[idx] = vox;
  }

  inline void clearVoxels() {
    std::memset(voxels, 0, sizeof(voxels));
  }

  // std::shared_ptr<CubeBlock> clone() {
  //   using AlignedBuf = AlignedBufT<sizeof(CubeBlock)>;
  //   auto* buf = new AlignedBuf;
  //   std::memcpy(buf->buf, this, sizeof(CubeBlock));
  //   std::shared_ptr<CubeBlock> clone(
  //       reinterpret_cast<CubeBlock*>(buf),
  //       [](CubeBlock* ptr){delete reinterpret_cast<AlignedBuf*>(ptr);});
  //   return clone;
  // }

  inline VoxPosition getVoxPosition(const Voxel& v) const {
    return VoxPosition(
        vk0.x() + v.u_.x,
        vk0.y() + v.u_.y,
        vk0.z() + v.u_.z);
  }

  inline void foreachVoxel(const std::function<void(const VoxPosition& p, const VoxColor& c)>& f) const {
    for (int i = 0; i < kMaxVoxels; ++i) {
      auto v = voxels[i].atomicClone();  // atomicly copy
      if (v.u_.status == 1) {
        f(getVoxPosition(v), v.c());
      }
    }
  }

  bool operator==(const CubeBlock& other) const {
    return bk == other.bk;
  }
  bool operator<(const CubeBlock& other) const {
    return bk < other.bk;
  }
  bool operator==(const BlockKey3& other_bk) const {
    return bk == other_bk;
  }
  bool operator<(const BlockKey3& other_bk) const {
    return bk < other_bk;
  }
};

struct SpatialHash3 {
  inline size_t operator()(const BlockKey3& p) const {
    return size_t(((p[0]) * 73856093) ^ ((p[1]) * 471943) ^ ((p[2]) * 83492791));
  }
  inline size_t operator()(const CubeBlock& c) const {
    return operator()(c.bk);
  }
};

struct SpatialHash2 {
  inline size_t operator()(const BlockKey2& p) const {
    return size_t(((p[0]) * 73856093) ^ ((p[1]) * 471943));
  }
};

#ifdef USE_ATOMIC_BLOCK_MAP
  using BlockHashTable = HashTable<CubeBlock, SpatialHash3, false, 14>;  // page size is 2^14=16K
  using CubeBlockRefPtr = typename BlockHashTable::UnderlyingMemPool::Ptr;
#else
  using CubeBlockRefPtr = std::shared_ptr<CubeBlock>;
#endif  

template<
    size_t _reserved_blocks_pow = 18  // reserved_blocks = 2^18 = 256K by default.
    // size_t _reserved_blocks_pow = 16  // reserved_blocks = 2^16 = 64K by default.
    ,
    size_t _raycast_region_sidelength_in_blocks = 64
  >
struct SimpleDenseMapT final {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static constexpr size_t kReservedBlocksPow = _reserved_blocks_pow;
  static constexpr size_t kReservedBlocks = 1 << kReservedBlocksPow;
  static constexpr size_t kReservedBlocksMask = kReservedBlocks - 1;
  static constexpr size_t kMaxBlocks = kReservedBlocks * 25 / 100;  // load factor 0.25
  // static constexpr size_t kMaxBlocks = kReservedBlocks * 75 / 100;  // load factor 0.75

  static constexpr size_t kRaycastRegionSideLengthInBlocks = _raycast_region_sidelength_in_blocks;
  static constexpr size_t kRaycastRegionBlocks =
      kRaycastRegionSideLengthInBlocks * kRaycastRegionSideLengthInBlocks * kRaycastRegionSideLengthInBlocks;

  double resolution;
  Timestamp time;

  struct OutputVoxels;

 private:
  std::shared_ptr<OutputVoxels> output;
  mutable std::mutex output_mutex;

#ifdef USE_ATOMIC_BLOCK_MAP
  BlockHashTable blocks_map;
#else
  std::unordered_map<BlockKey3, CubeBlockRefPtr, SpatialHash3> blocks_map;
#endif

  std::unordered_map<RegionKey3, std::unordered_set<BlockKey3, SpatialHash3>, SpatialHash3> raycast_regions;
  std::map<double, std::unordered_set<BlockKey3, SpatialHash3>> time_to_blocks;

  mutable std::shared_mutex mutex_xy_to_z_blocks;
  std::unordered_map<BlockKey2, std::unordered_set<BlockKey3::Scalar>, SpatialHash2> xy_to_z_blocks;

 public:

  struct OutputVoxels {
    // Voxel* output_voxels;
    // size_t output_voxels_size;

#ifndef USE_ATOMIC_BLOCK_MAP
    std::shared_ptr<std::vector<CubeBlockRefPtr>> output_blocks;
#endif    
    // std::shared_ptr<std::unordered_map<BlockKey2, std::unordered_set<BlockKey3::Scalar>, SpatialHash2>> output_xy_to_z_blocks;    

    // using AlignedBuf = AlignedBufT<sizeof(CubeBlock)>;
    // OutputVoxels(size_t n_blocks) {
    //   output_voxels_size = 0;
    //   char* buf = reinterpret_cast<char*>(new AlignedBuf[n_blocks]);
    //   output_voxels = reinterpret_cast<Voxel*>(buf);
    // }
    // ~OutputVoxels() {
    //   delete[] reinterpret_cast<AlignedBuf*>(output_voxels);
    // }
  };


  SimpleDenseMapT() :
    time(-1.0),
#ifdef USE_HEAR_SLAM
    thread_pool_group(nullptr),
#endif
#ifdef USE_ATOMIC_BLOCK_MAP
    blocks_map(kReservedBlocks),
#endif
    resolution(0.01) {

#ifndef USE_ATOMIC_BLOCK_MAP
    blocks_map.rehash(kReservedBlocks);
#endif

    // output = std::make_shared<OutputVoxels>(kMaxBlocks);
    // output = std::make_shared<OutputVoxels>(kReservedBlocks);
    output = std::make_shared<OutputVoxels>();
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

  template<typename Vec>
  inline std::vector<BlockKey3>
  getRayCastingBlocks(const Vec& p, const std::function<bool(const RegionKey3&)>& check_region_observability=nullptr) const {
    auto bk_vk = getKeysOfPoint(p);
    auto& bk = bk_vk.first;
    RegionKey3 rk(bk.x() / kRaycastRegionSideLengthInBlocks,
                  bk.y() / kRaycastRegionSideLengthInBlocks,
                  bk.z() / kRaycastRegionSideLengthInBlocks);

    std::vector<BlockKey3> blocks;

    blocks.reserve(27 * kRaycastRegionBlocks);
    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        for (int z = -1; z <= 1; z++) {
          RegionKey3 rk_delta(x, y, z);
          RegionKey3 rk_new = rk + rk_delta;
          auto it = raycast_regions.find(rk_new);
          if (it != raycast_regions.end()) {
            if (check_region_observability && !check_region_observability(rk_new)) {
              continue;
            }
            for (auto& bk : it->second) {
              blocks.push_back(bk);
            }
          }
        }
      }
    }
    return blocks;
  }

  void generateOutput() {
    // ASSERT(blocks_map.size() < kMaxBlocks);
    ASSERT(blocks_map.size() < kReservedBlocks);

    // return;  // skip

    if (blocks_map.size() == 0) {
      return;
    }

    // auto copy_xy_to_z_blocks = [&]() {
    //   auto output_xy_to_z_blocks = std::make_shared<
    //       std::unordered_map<BlockKey2, std::unordered_set<BlockKey3::Scalar>, SpatialHash2>
    //     >();
    //   *output_xy_to_z_blocks = xy_to_z_blocks;  // copy xy_to_z_blocks

    //   std::unique_lock<std::mutex> lock(output_mutex);
    //   std::swap(output->output_xy_to_z_blocks, output_xy_to_z_blocks);
    // };

    // copy_xy_to_z_blocks();

#ifndef USE_ATOMIC_BLOCK_MAP
    auto collect_blocks = [&]() {
      auto output_blocks = std::make_shared<std::vector<CubeBlockRefPtr>>();
      output_blocks->reserve(blocks_map.size());
      for (auto it = blocks_map.begin(); it != blocks_map.end(); it++) {
        output_blocks->emplace_back(it->second);
      }

      std::unique_lock<std::mutex> lock(output_mutex);
      std::swap(output->output_blocks, output_blocks);      
    };

    collect_blocks();
#endif

    // new_output->output_voxels_size = blocks_map.size() * CubeBlock::kMaxVoxels;
    // std::swap(output, new_output);
  }

  // inline const Voxel* voxels() const {
  //   return output ? output->output_voxels : nullptr;
  // }

  inline size_t reservedVoxelSize() const {
    // return output ? output->output_voxels_size : 0;
#ifdef USE_ATOMIC_BLOCK_MAP
    return kReservedBlocks * CubeBlock::kMaxVoxels;
    // return blocks_map.size() * CubeBlock::kMaxVoxels;
#else
    if (output && output->output_blocks) {
      return output->output_blocks->size() * CubeBlock::kMaxVoxels;
    } else {
      return 0;
    }
#endif
  }

  // process internal blcok
  inline bool processBlock(const BlockKey3& bk, std::function<void(CubeBlock&)>& f) {
#ifdef USE_ATOMIC_BLOCK_MAP
    auto node_ptr = blocks_map.find(bk);
    if (node_ptr == nullptr) {
      return false;
    }
    CubeBlock& block = node_ptr->data;
#else
    CubeBlockRefPtr block_ptr = blocks_map[bk];
    if (!block_ptr) {
      return false;
    }
    CubeBlock& block = *block_ptr;
#endif
    f(block);
    return true;
  }

  // process internal blcok
  inline bool processBlock(const BlockKey3& bk, const std::function<void(const CubeBlock&)>& f) const {
#ifdef USE_ATOMIC_BLOCK_MAP
    auto node_ptr = blocks_map.find(bk);
    if (node_ptr == nullptr) {
      return false;
    }
    const CubeBlock& block = node_ptr->data;
#else
    auto it = blocks_map.find(bk);
    if (it == blocks_map.end() || it->second == nullptr) {
      return false;
    }
    const CubeBlock& block = *(it->second);
#endif
    f(block);
    return true;
  }

  // foreach output block
  inline void foreachBlock(const std::function<void(const CubeBlock&)>& f) const {
#ifdef USE_ATOMIC_BLOCK_MAP
    blocks_map.foreachAlongMemory(f);
#else
    std::unique_lock<std::mutex> lock(output_mutex);
    if (output) {
      auto output_blocks = output->output_blocks;
      lock.unlock();

      if (output_blocks) {
        for (auto b : *output_blocks) {
          f(*b);
        }
      }
    }
#endif
  }

  inline void foreachVoxel(const std::function<void(const VoxPosition& p, const VoxColor& c)>& f) const {
    auto bf = [&](const CubeBlock& b) {
      b.foreachVoxel(f);
    };
    foreachBlock(bf);
  }

  struct BlockUpdateInfo {
    double old_time;
    bool is_new;
  };

  template<typename Vec>
  inline void insert_voxel(
      const Vec& p, const VoxColor& c, const Timestamp& time,
      std::unordered_map<BlockKey3, BlockUpdateInfo, SpatialHash3>& updated_blocks,
      std::unordered_map<BlockKey3, CubeBlockRefPtr, SpatialHash3>& blocks_map_cache) {
    auto keys = getKeysOfPoint(p);
    auto& bk = keys.first;
    auto& vk = keys.second;
    Voxel v;
    v.u_.x = (vk.x() & CubeBlock::kSideLengthMask);
    v.u_.y = (vk.y() & CubeBlock::kSideLengthMask);
    v.u_.z = (vk.z() & CubeBlock::kSideLengthMask);
    v.u_.r = c[0];
    v.u_.g = c[1];
    v.u_.b = c[2];


    bool is_new_block = false;
    CubeBlock* blkp = nullptr;    
    auto cache_it_blk = blocks_map_cache.find(bk);

#ifdef USE_ATOMIC_BLOCK_MAP
    if (cache_it_blk != blocks_map_cache.end()) {
      blkp = &(cache_it_blk->second->data);
    } else {
      CubeBlockRefPtr node_ptr = blocks_map.find(bk);
      if (!node_ptr) {
        auto insert_res = blocks_map.insert(bk);
        node_ptr = insert_res.second;
        ASSERT(node_ptr);
        is_new_block = insert_res.first;
      }
      blkp = &(node_ptr->data);
      blocks_map_cache[bk] = std::move(node_ptr);
    }
#else
    if (cache_it_blk != blocks_map_cache.end()) {
      blkp = cache_it_blk->second.get();
    } else {
      CubeBlockRefPtr blkptr;
      std::unique_lock<std::shared_mutex> lk(mutex_blocks_map_);
      auto it_blk = blocks_map.find(bk);
      if (it_blk == blocks_map.end()) {
        it_blk = blocks_map.insert({bk, std::make_shared<CubeBlock>(bk)}).first;
        is_new_block = true;
      }
      blkptr = it_blk->second;
      lk.unlock();

      blkp = blkptr.get();
      blocks_map_cache[bk] = std::move(blkptr);
    }
#endif

    CubeBlock& blk = *blkp;

    blk.put(v, hash3(vk) & CubeBlock::kMaxVoxelsMask);

    // update the block info.
    bool need_record_update_info = (updated_blocks.find(bk) == updated_blocks.end());
    auto old_time = blk.time.load(std::memory_order_relaxed);
    if (need_record_update_info) {
      BlockUpdateInfo& update_info = updated_blocks[bk];
      update_info.is_new = is_new_block;
      if (time != old_time) {
        update_info.old_time = old_time;
        blk.time.store(time, std::memory_order_relaxed);
      } else {
        update_info.old_time = -1.0;
      }
    }
  }

  void removeOldBlocksIfNeeded(
    const Timestamp& time,
    const std::vector<
        std::unordered_map<BlockKey3, BlockUpdateInfo, SpatialHash3>>&
      updated_blocks_array) {

    using UpdateInfoHandler
        = std::function<void(const BlockKey3&, const BlockUpdateInfo&)>;
    auto foreach_updated_block = [&](const UpdateInfoHandler& handler) {
      for (const auto& updated_blocks : updated_blocks_array) {
        for (const auto& item : updated_blocks) {
          const BlockKey3& bk = item.first;
          const BlockUpdateInfo& update_info = item.second;
          handler(bk, update_info);
        }
      }
    };

    auto update_time_to_blocks = [&]() {
      // size_t updated_blocks = 0;  // include duplicates
      foreach_updated_block([&] (const BlockKey3& bk, const BlockUpdateInfo& u) {
        if (u.old_time > 0) {
          auto it = time_to_blocks.find(u.old_time);
          ASSERT(it != time_to_blocks.end());
          it->second.erase(bk);
        }
        // updated_blocks ++;
        time_to_blocks[time].insert(bk);
      });
      // LOGI("update_time_to_blocks: updated_blocks = %d", updated_blocks);
    };

    auto update_region_to_blocks = [&]() {
      foreach_updated_block([&](const BlockKey3& bk, const BlockUpdateInfo& u) {
        if (u.is_new) {
          RegionKey3 rk(bk.x() / kRaycastRegionSideLengthInBlocks,
                        bk.y() / kRaycastRegionSideLengthInBlocks,
                        bk.z() / kRaycastRegionSideLengthInBlocks);

          auto insert_res = raycast_regions[rk].insert(bk);
          ASSERT(insert_res.second);
        }
      });
    };

    auto update_xy_to_z_blocks = [&]() {
      std::unique_lock<std::shared_mutex> lock(mutex_xy_to_z_blocks);
      foreach_updated_block([&](const BlockKey3& bk, const BlockUpdateInfo& u) {
        if (u.is_new) {
          BlockKey2 bk2(bk.x(), bk.y());
          auto insert_res = xy_to_z_blocks[bk2].insert(bk.z());
          ASSERT(insert_res.second);
        }
      });
    };

#ifdef USE_HEAR_SLAM
    // std::memcpy(buf->buf, this, sizeof(SimpleDenseMapT<_reserved_blocks_pow>));
    auto pool = thread_pool_group->getNamed("ov_map_rgbd_w");
    std::vector<hear_slam::TaskID> task_ids;
    task_ids.push_back(pool->schedule(update_time_to_blocks));
    task_ids.push_back(pool->schedule(update_region_to_blocks));
    task_ids.push_back(pool->schedule(update_xy_to_z_blocks));
    pool->waitTasks(task_ids.begin(), task_ids.end());
    task_ids.clear();
#else
    update_time_to_blocks();
    update_region_to_blocks();
    update_xy_to_z_blocks();    
#endif

    // collect the blocks to remove
    std::map<double, std::unordered_set<BlockKey3, SpatialHash3>> to_remove_time_to_blocks;

    size_t to_remove_size = (blocks_map.size() > kMaxBlocks ?
                             blocks_map.size() - kMaxBlocks : 0);
    size_t tmp_size = 0;
    while (time_to_blocks.size() > 1  // keep at least 1 frame
            && tmp_size < to_remove_size) {
      auto begin = time_to_blocks.begin();
      tmp_size += begin->second.size();
      to_remove_time_to_blocks[begin->first] = std::move(begin->second);
      time_to_blocks.erase(begin);
    }
    if (tmp_size > 0) {
      LOGI("NeedRemoveOldBlocks: blocks_map.size()=%d, kMaxBlocks=%d, time_to_blocks.size()=%d, to_remove_size=%d, newly_updated_blocks.size()=%d",
            blocks_map.size(), kMaxBlocks, time_to_blocks.size(), tmp_size, time_to_blocks[time].size());
    }

    auto foreach_block_to_remove = [&](const std::function<void(const BlockKey3& bk)>& f) {
      for (const auto& item : to_remove_time_to_blocks) {
        for (const auto& bk : item.second) {
          f(bk);
        }
      }
    };

    auto remove_from_region_to_blocks = [&]() {
      foreach_block_to_remove([&](const BlockKey3& bk){
        RegionKey3 rk(bk.x() / kRaycastRegionSideLengthInBlocks,
                      bk.y() / kRaycastRegionSideLengthInBlocks,
                      bk.z() / kRaycastRegionSideLengthInBlocks);
        auto rit = raycast_regions.find(rk);
        rit->second.erase(bk);
        if (rit->second.empty()) {
          raycast_regions.erase(rit);
        }
      });
    };

    auto remove_from_xy_to_z_blocks = [&]() {
      std::unique_lock<std::shared_mutex> lock(mutex_xy_to_z_blocks);
      foreach_block_to_remove([&](const BlockKey3& bk){
          BlockKey2 bk2(bk.x(), bk.y());
          auto it = xy_to_z_blocks.find(bk2);
          it->second.erase(bk.z());
          if (it->second.empty()) {
            xy_to_z_blocks.erase(it);
          }
      });
    };

    auto remove_from_blocks_map = [&]() {
#ifndef USE_ATOMIC_BLOCK_MAP
      std::unique_lock<std::shared_mutex> lk(mutex_blocks_map_);
#endif
      foreach_block_to_remove([&](const BlockKey3& bk){
#ifdef USE_ATOMIC_BLOCK_MAP        
        ASSERT(blocks_map.erase(bk));
#else
        auto it = blocks_map.find(bk);
        ASSERT(it != blocks_map.end());
        blocks_map.erase(it);
#endif        
      });
    };


#ifdef USE_HEAR_SLAM
    task_ids.push_back(pool->schedule(remove_from_region_to_blocks));
    task_ids.push_back(pool->schedule(remove_from_xy_to_z_blocks));
    task_ids.push_back(pool->schedule(remove_from_blocks_map));
    pool->waitTasks(task_ids.begin(), task_ids.end());
    task_ids.clear();
#else
    remove_from_region_to_blocks();
    remove_from_xy_to_z_blocks();
    remove_from_blocks_map();
#endif

    ASSERT(blocks_map.size() <= kMaxBlocks);
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
    cv::Mat upperbound(rows, cols, CV_32SC1, cv::Scalar(INT_MAX));

    float resolution_ratio = hmap_resolution / this->resolution;
    float block_resolution = this->resolution * CubeBlock::kSideLength;
    const float upperbound_theta_a = 50.0;
    const float upperbound_tan = tan(upperbound_theta_a * M_PI / 180.0);
    const float upperbound_b = 0.1;
    const float upperbound_c = -0.5;
    const float upperbound_d = 0.0;
    const int32_t center_zi = center_z / this->resolution;

    auto fill_row_range = [&](int row_start, int row_end) {
      if (row_end > rows) {
        row_end = rows;
      }
      int local_rows = row_end - row_start;


      std::unordered_set<BlockKey2, SpatialHash2> involved_xy_blocks;
      std::unordered_map<PixelKey, std::pair<int, int>, SpatialHash2> voxelxy_to_imgxy;
      const size_t approx_max_xys = (local_rows * cols) * (resolution_ratio * resolution_ratio);
      involved_xy_blocks.rehash(approx_max_xys);
      voxelxy_to_imgxy.rehash(approx_max_xys * 4);

      for (int i=row_start; i<row_end; i++) {
        for (int j=0; j<cols; j++) {
          float local_x = (j - cols/2) * hmap_resolution;
          float local_y = (i - rows/2) * hmap_resolution;
          float xy_dist = sqrt(local_x * local_x + local_y * local_y);
          float truncated_dist = xy_dist - upperbound_b;
          float z_thr = upperbound_c;
          if (truncated_dist > 0) {
            z_thr += upperbound_tan * truncated_dist;
          }
          z_thr = std::min(z_thr, upperbound_d);
          upperbound.at<int32_t>(i, j) = z_thr / this->resolution;

          float global_x = center_x + local_x * c - local_y * s;
          float global_y = center_y + local_x * s + local_y * c;
          BlockKey2 bk2(global_x / block_resolution, global_y / block_resolution);
          if (global_x < 0.0) {
            bk2.x() -= 1;
          }
          if (global_y < 0.0) {
            bk2.y() -= 1;
          }

          PixelKey voxelxy(global_x / this->resolution, global_y / this->resolution);
          voxelxy_to_imgxy[voxelxy] = std::make_pair(j, i);  // x, y
          involved_xy_blocks.insert(bk2);
        }
      }

#ifdef USE_ATOMIC_BLOCK_MAP

      auto foreach_block = [&](const std::function<void(const CubeBlock&)>& f) {

        std::vector<BlockKey2> block_key2s;
        // convert set involved_xy_blocks to vector block_key2s
        block_key2s.reserve(involved_xy_blocks.size());
        block_key2s.insert(block_key2s.end(), involved_xy_blocks.begin(), involved_xy_blocks.end());

        // size_t n_executors = pool->numThreads();
        size_t n_executors = 1;

        std::atomic<size_t> n_processed(0);
        std::vector<std::vector<BlockKey3>> blocks_array(n_executors);
        auto retrieve_blocks = [&](size_t executor_idx) {
          std::vector<BlockKey3>& blocks = blocks_array[executor_idx];
          blocks.reserve(block_key2s.size() * 20 / n_executors);
          size_t key2_idx;

          std::shared_lock<std::shared_mutex> lock(mutex_xy_to_z_blocks);
          while((key2_idx=n_processed.fetch_add(1)) < block_key2s.size()) {
            const auto& bk2 = block_key2s[key2_idx];
            auto zit = xy_to_z_blocks.find(bk2);
            if (zit != xy_to_z_blocks.end()) {
              for (auto& z : zit->second) {
                blocks.emplace_back(bk2.x(), bk2.y(), z);
              }
            }
          }
        };

        // tc.tag("BeforeRetrieveBlocks");
        retrieve_blocks(0);
        // tc.tag("AfterRetrieveBlocks");

        for (size_t i=0; i<n_executors; ++i) {
          for (const auto& bk : blocks_array[i]) {
            processBlock(bk, f);
          }
        }
      };

#else
      auto foreach_block = [&](const std::function<void(const CubeBlock&)>& f) {
        std::vector<CubeBlockRefPtr> block_data;
        block_data.reserve(involved_xy_blocks.size() * ((max_h - min_h) / block_resolution + 1) * 2);

// #ifdef USE_HEAR_SLAM
//         tc.tag("BeforeRetrieveBlocks");
// #endif

        for (auto & bk2 : involved_xy_blocks) {
          std::shared_lock<std::shared_mutex> lk(mutex_blocks_map_);
          std::shared_lock<std::shared_mutex> lk2(mutex_xy_to_z_blocks);
          auto zit = xy_to_z_blocks.find(bk2);
          if (zit != xy_to_z_blocks.end()) {
            for (auto& z : zit->second) {
              auto it = blocks_map.find(BlockKey3(bk2.x(), bk2.y(), z));
              if (it != blocks_map.end()) {
                auto block_ptr = it->second;
                if (block_ptr) {
                  block_data.push_back(block_ptr);
                }              
              }
            }
          }
        }

// #ifdef USE_HEAR_SLAM
//         tc.tag("AfterRetrieveBlocks");
// #endif

        for (auto & block : block_data) {
          f(*block);
        }
      };

#endif

      const int32_t discrepancy_thr_i = discrepancy_thr / this->resolution;
      // const bool adopt_max_zi = true;
      const bool adopt_max_zi = false;

      foreach_block([&](const CubeBlock& block){
        for (size_t i=0; i<CubeBlock::kMaxVoxels; i++) {
          auto voxel = block.voxels[i].atomicClone();
          if (voxel.u_.status) {
            auto p = block.getVoxPosition(voxel);
            auto& vx = p.x();
            auto& vy = p.y();
            auto vit = voxelxy_to_imgxy.find(PixelKey(vx, vy));
            if (vit != voxelxy_to_imgxy.end()) {
              auto& vzi = p.z();
              auto& imgx = vit->second.first;
              auto& imgy = vit->second.second;

              // may race if parallelized
              auto& max_zi = hmax.at<int32_t>(imgy, imgx);
              auto& min_zi = hmin.at<int32_t>(imgy, imgx);
              const auto& upper_b = upperbound.at<int32_t>(imgy, imgx);
              if (vzi > upper_b) {
                continue;
              }

              if (vzi > max_zi) {
                max_zi = vzi;
              }
              if (vzi < min_zi) {
                min_zi = vzi;
              }

              if (adopt_max_zi) {
                float h = (max_zi - center_zi) * this->resolution;
                h = std::min(std::max(h, min_h), max_h); // Clamp to min/max height
                height_map.at<uint16_t>(imgy, imgx) = h / hmap_resolution + 32768;
              } else {
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
      });
    };

#ifdef USE_HEAR_SLAM
    auto pool = thread_pool_group->getNamed("height_map_w");
    ASSERT(pool);
    std::vector<hear_slam::TaskID> task_ids;

    const int row_block = 32;
    for (int start_row=0; start_row<rows; start_row+=row_block) {
      task_ids.push_back(pool->schedule([&fill_row_range, &row_block, start_row](){
        fill_row_range(start_row, start_row + row_block);
      }));
    }

    pool->waitTasks(task_ids.begin(), task_ids.end());
    task_ids.clear();

    // fill_row_range(0, rows);

#else
    fill_row_range(0, rows);
#endif



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


#ifdef USE_HEAR_SLAM
  hear_slam::ThreadPoolGroup* thread_pool_group;
#endif
  mutable std::shared_mutex mutex_blocks_map_;
  static SpatialHash3 hash3;

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
    thread_pool_group_.createNamed("height_map_w", std::thread::hardware_concurrency());
#endif
    insert_thread_ = std::thread(&SimpleDenseMapBuilder::insert_thread_func, this);
    map_->thread_pool_group = &thread_pool_group_;
  }

  ~SimpleDenseMapBuilder() {
    stop_insert_thread();
  }

  using UndistortFunction = std::function<bool(const Eigen::Vector2i& ixy, Eigen::Vector2f& nxy)>;
  using DistortFunction = std::function<bool(const Eigen::Vector2f& nxy, Eigen::Vector2i& ixy)>;
  void registerCamera(size_t cam_id, int width, int height, UndistortFunction undistort_func, DistortFunction distort_func) {
    cam_to_param_[cam_id] = std::make_shared<CameraParam>(width, height, undistort_func, distort_func);
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
      // LOG(INFO) << "RgbdMapping:  task - begin";
#ifdef USE_HEAR_SLAM
      tc_.reset();
#endif
      raycast(depth, cam, T_W_C, time, pixel_downsample, max_depth, start_row, end_row, start_col, end_col);

#ifdef USE_HEAR_SLAM
      tc_.tag("RayCastDone");
#endif
      // LOG(INFO) << "RgbdMapping:  task - raycast done";

      insert_rgbd_frame(color, depth, cam, T_W_C, time, pixel_downsample, max_depth, start_row, end_row, start_col, end_col);
#ifdef USE_HEAR_SLAM
      tc_.tag("MapUpdated");
#endif

      if (time > time_) {
        time_ = time;
      }

      LOG(INFO) << "RgbdMapping:  DEBUG_SIZE: sizeof(Voxel)=" << sizeof(Voxel)
                << ", sizeof(CubeBlock)=" << sizeof(CubeBlock)
                << ", CubeBlock::kMaxVoxels=" << CubeBlock::kMaxVoxels
                << ", sizeof(VoxelSimple)=" << sizeof(Voxel);

      // LOG(INFO) << "RgbdMapping:  task - map updated";
      update_output();
#ifdef USE_HEAR_SLAM
      tc_.tag("OutputUpdated");
#endif
      // LOG(INFO) << "RgbdMapping:  task - output updated";
      if (output_update_callback_) {        
        output_update_callback_(output_);
      }
#ifdef USE_HEAR_SLAM
      tc_.tag("CallbackFinished");
      tc_.report("RgbdMappingTiming: ", true);
#endif
      // LOG(INFO) << "RgbdMapping:  task - callback invoked";
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

#ifdef USE_HEAR_SLAM
    size_t row_group_size = 10 * pixel_downsample;
    size_t total_groups = (end_row - start_row) / row_group_size + 1;

    std::vector<
        std::unordered_map<BlockKey3, SimpleDenseMap::BlockUpdateInfo, SpatialHash3>>
      updated_blocks_array;
    updated_blocks_array.resize(total_groups);
#else
    std::vector<
        std::unordered_map<BlockKey3, SimpleDenseMap::BlockUpdateInfo, SpatialHash3>>
      updated_blocks_array;
    updated_blocks_array.resize(1);
#endif

    // auto& undistort = cam_to_undistort_.at(cam);
    auto& undistort = cam_to_param_.at(cam)->quick_undistort_func;    
    auto insert_row_range = [&](int first_row, int last_row, size_t group_id=0) {
      std::unordered_map<BlockKey3, SimpleDenseMap::BlockUpdateInfo, SpatialHash3>&
          updated_blocks = updated_blocks_array.at(group_id);
      std::unordered_map<BlockKey3, CubeBlockRefPtr, SpatialHash3> blocks_map_cache;
      blocks_map_cache.rehash(100000);

      last_row = std::min(end_row, last_row);
      for (size_t y=first_row; y<last_row; y+=pixel_downsample) {
        for (size_t x=0; x<end_col; x+=pixel_downsample) {
          const uint16_t d = depth.at<uint16_t>(y,x);
          if (d == 0) continue;
          float depth = d / 1000.0f;
          if (max_depth < depth) {
            continue;
          }

          Eigen::Vector2f p_normal;
          ASSERT(undistort(Eigen::Vector2i(x, y), p_normal));
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
          map_->insert_voxel(p3d_w, c, time, updated_blocks, blocks_map_cache);
        }
      }
    };

#ifdef USE_HEAR_SLAM
    // insert_row_range(start_row, end_row);
    auto pool = thread_pool_group_.getNamed("ov_map_rgbd_w");
    ASSERT(pool);
    ASSERT(pool->numThreads() == std::thread::hardware_concurrency());

    using hear_slam::TaskID;
    using hear_slam::INVALID_TASK;
    std::vector<TaskID> task_ids;
    task_ids.reserve(total_groups);
    size_t group_id = 0;
    for (size_t y=start_row; y<end_row; y+=row_group_size) {
      auto new_task = pool->schedule(
        [&insert_row_range, row_group_size, y, group_id](){
          insert_row_range(y, y+row_group_size, group_id);
        });
      task_ids.emplace_back(new_task);
      group_id++;
    }
    pool->waitTasks(task_ids.rbegin(), task_ids.rend());
    // pool->freeze();
    // pool->waitUntilAllTasksDone();
    // pool->unfreeze();
#else
    insert_row_range(start_row, end_row);
#endif
    tc_.tag("InsertDone");
    map_->removeOldBlocksIfNeeded(time, updated_blocks_array);
    tc_.tag("OldBlocksRemoved");
  }



  void raycast(const cv::Mat& depth,
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
      end_row = depth.rows;
    }
    if (end_col < 0) {
      end_col = depth.cols;
    }


    Eigen::Vector3f p_W_C = T_W_C.translation();
    Eigen::Isometry3f T_C_W = T_W_C.inverse();

    // auto& distort = cam_to_distort_.at(cam);
    auto& distort = cam_to_param_.at(cam)->quick_distort_func;

    auto project_point = [&](const Eigen::Vector3f& p_W_V, Eigen::Vector2i& ixy, float& depth_V) -> bool {
      Eigen::Vector3f p_C_V = T_C_W * p_W_V;
      if (p_C_V.z() <= 0.0) {  // only eliminate voxels whose depth are positive
        return false;
      }
      if (p_C_V.z() > max_depth) {  // only eliminate voxels whose depth are less than max_depth
        return false;
      }
      if (!distort(Eigen::Vector2f(p_C_V.x()/p_C_V.z(), p_C_V.y()/p_C_V.z()), ixy)) {
        return false;
      }
      if (ixy.x() >= start_col && ixy.x() < end_col && ixy.y() >= start_row && ixy.y() < end_row) {
        depth_V = p_C_V.z();
        return true;
      } else {
        return false;
      }
    };

    auto is_point_observable = [&](const Eigen::Vector3f& p) {
      Eigen::Vector2i ixy;
      float depth_V;
      return project_point(p, ixy, depth_V);
    };

    auto is_cube_observable = [&](const Eigen::Vector3f& min_corner, float side_length) {
      std::vector <Eigen::Vector3f> corners;
      corners.reserve(8);
      for (int i=0; i<8; i++) {
        corners.emplace_back(min_corner + Eigen::Vector3f(side_length*(i&1), side_length*((i>>1)&1), side_length*((i>>2)&1)));
      }
      return std::any_of(corners.begin(), corners.end(), is_point_observable);
    };

    auto is_block_observable = [&](const BlockKey3& bk) {
      float side_length = CubeBlock::kSideLength * resolution_;
      Eigen::Vector3f block_min_corner(
        (bk.x() << CubeBlock::kSideLengthPow) * resolution_,
        (bk.y() << CubeBlock::kSideLengthPow) * resolution_,
        (bk.z() << CubeBlock::kSideLengthPow) * resolution_
      );
      return is_cube_observable(block_min_corner, side_length);
    };

    auto cube_observable_corners = [&](const Eigen::Vector3f& min_corner, float side_length) {
      std::vector <Eigen::Vector3f> observable_corners;
      observable_corners.reserve(8);
      for (int i=0; i<8; i++) {
        Eigen::Vector3f corner = 
          min_corner + Eigen::Vector3f(side_length*(i&1), side_length*((i>>1)&1), side_length*((i>>2)&1));
        if (is_point_observable(corner)) {
          observable_corners.push_back(corner);
        }
      }
      return observable_corners;
    };

    auto block_observable_corners = [&](const BlockKey3& bk) {
      float side_length = CubeBlock::kSideLength * resolution_;
      Eigen::Vector3f block_min_corner(
        (bk.x() << CubeBlock::kSideLengthPow) * resolution_,
        (bk.y() << CubeBlock::kSideLengthPow) * resolution_,
        (bk.z() << CubeBlock::kSideLengthPow) * resolution_
      );
      return cube_observable_corners(block_min_corner, side_length);
    };

    auto check_region_observability = [&](const RegionKey3& rk) {
      static constexpr int kRegionSideLength = SimpleDenseMap::kRaycastRegionSideLengthInBlocks * CubeBlock::kSideLength;
      float side_length = kRegionSideLength * resolution_;
      Eigen::Vector3f region_min_corner(
        (rk.x() * kRegionSideLength) * resolution_,
        (rk.y() * kRegionSideLength) * resolution_,
        (rk.z() * kRegionSideLength) * resolution_
      );
      std::vector <Eigen::Vector3f> corners;
      corners.reserve(8);
      for (int i=0; i<8; i++) {
        corners.emplace_back(region_min_corner + Eigen::Vector3f(side_length*(i&1), side_length*((i>>1)&1), side_length*((i>>2)&1)));
      }
      return std::any_of(corners.begin(), corners.end(), [&](const Eigen::Vector3f& corner) {
        Eigen::Vector3f corner_in_C= T_C_W * corner;
        if (corner_in_C.z() <= 0.0) {
          return false;
        } else {
          return true;
        }
      });
    };

    std::vector<BlockKey3> blocks = map_->getRayCastingBlocks(p_W_C, check_region_observability);
    // std::atomic<size_t> n_empty_blocks(0);
    // std::atomic<size_t> n_unobservable_blocks(0);
    // std::atomic<size_t> n_whole_cleared_blocks(0);
    // std::atomic<size_t> n_normal(0);
    // std::atomic<size_t> n_new_empty(0);

    std::function<void(CubeBlock&)> raycast_one_block = [&](CubeBlock& block) {

      // bool all_corners_observable = false;
      // size_t valid_voxels = 0;

      // {
      //   // TODO(jeffrey):
      //   //   Consider whether the checks are worth.
      //   // Example log:
      //   //   total 45498: empty 20591, unobservable 10449, cleared 498, normal 13960, new empty 2639
      //   //   total 45505: empty 20588, unobservable 10452, cleared 448, normal 14017, new empty 2586
      //   //   total 45506: empty 20563, unobservable 10387, cleared 483, normal 14073, new empty 2564


      //   // Check and skip.
      //   for (size_t i=0; i<CubeBlock::kMaxVoxels; i++) {
      //     valid_voxels += (block.voxels[i].u_.status & 1);
      //   }
      //   if (valid_voxels == 0) {
      //     n_empty_blocks.fetch_add(1, std::memory_order_relaxed);
      //     return;
      //   }

      //   size_t n_observable_corners = block_observable_corners(block.bk).size();
      //   if (n_observable_corners == 0) {
      //     n_unobservable_blocks.fetch_add(1, std::memory_order_relaxed);
      //     return;
      //   }
      //   all_corners_observable = (n_observable_corners == 8);
      // }

      if (!is_block_observable(block.bk)) {
        return;
      }

      Voxel* voxels = block.voxels;
      static const float depth_noise_level = 0.02;
      // float block_width = CubeBlock::kSideLength * resolution_;
      // float clear_block_thr = depth_noise_level + 2 * block_width;
      // valid_voxels = 0;
      for (size_t i=0; i<CubeBlock::kMaxVoxels; i++) {
        Voxel& voxel = voxels[i];
        if (voxel.u_.status) {
          Eigen::Vector3f p_W_V = block.getVoxPosition(voxel).cast<float>() * resolution_;
          Eigen::Vector2i ixy;
          float depth_V;
          if (project_point(p_W_V, ixy, depth_V)) {
            const uint16_t d = depth.at<uint16_t>(ixy.y(), ixy.x());
            if (d == 0) continue;
            float fd = d / 1000.0f;
            // if (all_corners_observable && depth_V < fd - clear_block_thr) {
            //   block.clearVoxels();
            //   n_whole_cleared_blocks.fetch_add(1, std::memory_order_relaxed);
            //   // LOGD("Clear CubeBlock (%d, %d, %d): fd = %f, depth_V = %f", 
            //   //      block.bk.x(), block.bk.y(), block.bk.z(), fd, depth_V);
            //   n_new_empty.fetch_add(1, std::memory_order_relaxed);
            //   return;
            // }
            if (depth_V < fd - depth_noise_level) {  // eliminate the voxel if it can be cast through
              voxel.u_.status = 0;
            }
          }
          // valid_voxels += (voxel.u_.status & 1);
        }
      }
      // if (valid_voxels == 0) {
      //   n_new_empty.fetch_add(1, std::memory_order_relaxed);
      // }
      // n_normal.fetch_add(1, std::memory_order_relaxed);
    };

    // This implementation may present bad load balance since different block
    // groups may differ significantly in computation cost (some group may
    // contain many unobservable blocks which can be skipped quickly but
    // others may consists of all observable blocks for which every voxel
    // needs computation)
    //
    // // size_t block_group_size = 32;    
    // size_t block_group_size = blocks.size() / (2 * pool->numThreads()) + 1;
    // auto raycast_blocks = [&](size_t first, size_t last) {
    //   last = std::min(last, blocks.size());
    //   for (size_t i=first; i<last; ++i) {
    //     map_->processBlock(blocks[i], raycast_one_block);
    //   }
    // };


    // Balance the load
    std::atomic<size_t> n_processed(0);
    auto raycast_blocks = [&](){
      size_t blk_idx;
      while ((blk_idx = n_processed.fetch_add(1)) < blocks.size()) {
        map_->processBlock(blocks[blk_idx], raycast_one_block);
      }
    };

#ifdef USE_HEAR_SLAM
    auto pool = thread_pool_group_.getNamed("ov_map_rgbd_w");
    ASSERT(pool);
    ASSERT(pool->numThreads() == std::thread::hardware_concurrency());

    // using hear_slam::TaskID;
    // using hear_slam::INVALID_TASK;
    // std::vector<TaskID> task_ids;
    // task_ids.reserve(blocks.size() / block_group_size + 1);
    // for (size_t i=0; i<blocks.size(); i+=block_group_size) {
    //   auto new_task = pool->schedule(
    //     [&raycast_blocks, block_group_size, i](){
    //       raycast_blocks(i, i+block_group_size);
    //     });
    //   task_ids.emplace_back(new_task);
    // }

    using hear_slam::TaskID;
    std::vector<TaskID> task_ids(pool->numThreads());
    for (size_t i=0; i<pool->numThreads(); ++i) {
      auto new_task = pool->schedule(raycast_blocks);
      task_ids[i] = new_task;
    }

    pool->waitTasks(task_ids.rbegin(), task_ids.rend());
    // pool->freeze();
    // pool->waitUntilAllTasksDone();
    // pool->unfreeze();
#else
    // raycast_blocks(0, blocks.size());
    raycast_blocks();
#endif
    // LOGI("RayCastStatistics: total %d: empty %d, unobservable %d, cleared %d, normal %d, new empty %d",
    //      blocks.size(), n_empty_blocks.load(), n_unobservable_blocks.load(),
    //      n_whole_cleared_blocks.load(), n_normal.load(), n_new_empty.load());
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
  hear_slam::TimeCounter tc_;
#endif



  struct CameraParam {
    int width;
    int height;
    std::vector<Eigen::Vector2f> undistortion_table;
    UndistortFunction quick_undistort_func;

    int distortion_width;
    int distortion_height;
    std::vector<Eigen::Vector2i> distortion_table;
    DistortFunction quick_distort_func;
    float distortion_resolution;
    float distortion_startx;
    float distortion_starty;

    CameraParam(int width, int height, UndistortFunction undistort_func, DistortFunction distort_func) {
      this->width = width;
      this->height = height;
      undistortion_table.resize(width * height);
      float min_x = std::numeric_limits<float>::max();
      float min_y = std::numeric_limits<float>::max();
      float max_x = - std::numeric_limits<float>::max();
      float max_y = - std::numeric_limits<float>::max();
      for (size_t y=0; y<height; y++) {
        for (size_t x=0; x<width; x++) {
          Eigen::Vector2f p;
          ASSERT(undistort_func(Eigen::Vector2i(x, y), p));
          undistortion_table[y * width + x] = p;
          min_x = std::min(min_x, p.x());
          min_y = std::min(min_y, p.y());
          max_x = std::max(max_x, p.x());
          max_y = std::max(max_y, p.y());
        }
      }
      quick_undistort_func = [this](const Eigen::Vector2i& ixy, Eigen::Vector2f& nxy) {
        const int& x = ixy.x();
        const int& y = ixy.y();
        if (x < 0 || x >= this->width || y < 0 || y >= this->height) {
          return false;
        }
        nxy = undistortion_table[y * this->width + x];
        return true;
      };


      distortion_resolution = (max_x - min_x) / (3 * width);
      distortion_startx = min_x;
      distortion_starty = min_y;
      distortion_width = std::ceil((max_x - min_x) / distortion_resolution);
      distortion_height = std::ceil((max_y - min_y) / distortion_resolution);
      distortion_table.resize(distortion_width * distortion_height);
      for (size_t y=0; y<distortion_height; y++) {
        for (size_t x=0; x<distortion_width; x++) {
          Eigen::Vector2i ixy;
          ASSERT(distort_func(Eigen::Vector2f(
                      distortion_startx + x * distortion_resolution,
                      distortion_starty + y * distortion_resolution), ixy));
          distortion_table[y * distortion_width + x] = ixy;
        }
      }

      quick_distort_func = [this](const Eigen::Vector2f& nxy, Eigen::Vector2i& ixy) {
        const int x = std::round((nxy.x() - distortion_startx) / distortion_resolution);
        const int y = std::round((nxy.y() - distortion_starty) / distortion_resolution);
        if (x < 0 || x >= distortion_width || y < 0 || y >= distortion_height) {
          return false;
        }
        ixy = distortion_table[y * distortion_width + x];
        // if (ixy.x() < 0 || ixy.x() >= this->width || ixy.y() < 0 || ixy.y() >= this->height) {
        //   return false;  // a more strict check will be done outside this function.
        // }
        return true;
      };
    }
  };
  std::map<size_t, std::shared_ptr<CameraParam>> cam_to_param_;

  // std::map<size_t, UndistortFunction> cam_to_undistort_;
  // std::map<size_t, DistortFunction> cam_to_distort_;

};
}  // namespace dense_mapping

} // namespace ov_msckf


#endif // OV_MSCKF_SIMPLE_DENSE_MAPPING_H
