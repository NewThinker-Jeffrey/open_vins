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

#ifndef OV_MSCKF_SIMPLE_RGBD_MAP_H
#define OV_MSCKF_SIMPLE_RGBD_MAP_H

#include <Eigen/Geometry>
#include "cam/CamBase.h"

namespace ov_msckf {

class SimpleRgbdMap {

public:

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

  using Color = Eigen::Matrix<uint8_t, 3, 1>;
  using Timestamp = double;

  struct Voxel {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Position p;
    Color c;
    Timestamp time;
  };

  SimpleRgbdMap(size_t max_voxels=1000000, float resolution = 0.01) : 
      max_voxels_(max_voxels), resolution_(resolution) {
    voxels_.resize(max_voxels);
    for (size_t i=0; i<voxels_.size(); i++) {
      unused_entries_.insert(i);
    }
  }

  ~SimpleRgbdMap() {}

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
                         ov_core::CamBase* cam,
                         const Eigen::Isometry3f& T_W_C,
                         const Timestamp& time,
                         int pixel_downsample = 1,
                         int start_row = 0) {
    for (size_t y=start_row; y<color.rows; y+=pixel_downsample) {
      for (size_t x=0; x<color.cols; x+=pixel_downsample) {
        const uint16_t d = depth.at<uint16_t>(y,x);
        if (d == 0) continue;
        float depth = d / 1000.0f;

        Eigen::Vector2f p_normal = cam->undistort_f(Eigen::Vector2f(x, y));
        Eigen::Vector3f p3d_c(p_normal.x(), p_normal.y(), 1.0f);
        p3d_c = p3d_c * depth;
        Eigen::Vector3f p3d_w = T_W_C * p3d_c;
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

  std::vector<Voxel> get_occupied_voxels() const {
    std::vector<Voxel> output;
    output.reserve(voxels_.size() - unused_entries_.size());
    for (size_t i=0; i<voxels_.size(); i++) {
      if (unused_entries_.count(i) == 0) {
        output.push_back(voxels_[i]);
      }
    }
    return output;
  }

  float resolution() const {
    return resolution_;
  }

protected:
  std::vector<Voxel> voxels_;
  std::set<size_t> unused_entries_;
  std::map<Timestamp, std::set<size_t>> time_to_voxels_;
  std::map<Position, size_t> pos_to_voxel_;

  const size_t max_voxels_ = 1000000;
  const float resolution_ = 0.01;
};


} // namespace ov_msckf


#endif // OV_MSCKF_SIMPLE_RGBD_MAP_H
