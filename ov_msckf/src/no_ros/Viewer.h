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
#include "interface/HeisenbergVIO.h"
#include <mutex>


#if ENABLE_PANGOLIN
#include <pangolin/pangolin.h>

namespace ov_msckf {

class Viewer {
public:
  Viewer(std::shared_ptr<heisenberg_algo::VIO> app);
  void init();
  void show(std::shared_ptr<VioManager::Output> task);

private:

  /// Core application of the filter system
  std::shared_ptr<heisenberg_algo::VIO> _app;

  std::shared_ptr<pangolin::OpenGlRenderState> s_cam;
  std::deque <Eigen::Vector3f> _traj;
  std::map<size_t, Eigen::Vector3f> _map_points;
};


} // namespace ov_msckf

#else

namespace ov_msckf {

class Viewer {
public:
  Viewer(std::shared_ptr<heisenberg_algo::VIO> app) {
    std::cout << "Viewer::Viewer():  No Pangolin!" << std::endl;
  }
  void init() {}
  void show(std::shared_ptr<VioManager::Output> task) {}
};

} // namespace ov_msckf

#endif


#endif // OV_MSCKF_VIEWER_H
