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

#ifndef OV_MSCKF_VIEWER_H
#define OV_MSCKF_VIEWER_H

#include "core/VioManager.h"
#include <mutex>


#if ENABLE_PANGOLIN
#include <pangolin/pangolin.h>
#include "slam_viz/pangolin_helper_types.h"
 
namespace ov_msckf {

class Viewer {
public:
  Viewer(VioManager* interal_app);
  void init();
  void show(std::shared_ptr<VioManager::Output> task);

private:

  void classifyPoints(std::shared_ptr<VioManager::Output> task);

  void drawRobotAndMap(std::shared_ptr<VioManager::Output> task);


  /// Core application of the filter system
  VioManager* _interal_app;

  std::shared_ptr<pangolin::OpenGlRenderState> s_cam1, s_cam2;
  std::deque <Eigen::Vector3f> _traj;
  std::map<size_t, Eigen::Vector3d> _map_points;
  slam_viz::pangolin_helper::PointdPtrSet _slam_points, _msckf_points, _old_points, _active_points;
  Eigen::Isometry3f _imu_pose, _predicted_imu_pose;
};


} // namespace ov_msckf

#else

namespace ov_msckf {

class Viewer {
public:
  Viewer(VioManager* interal_app) {
    std::cout << "Viewer::Viewer():  No Pangolin!" << std::endl;
  }
  void init() {}
  void show(std::shared_ptr<VioManager::Output> task) {}
};

} // namespace ov_msckf

#endif


#endif // OV_MSCKF_VIEWER_H
