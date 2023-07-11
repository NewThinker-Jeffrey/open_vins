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

#ifndef OV_MSCKF_VISUALIZER_HELPER_H
#define OV_MSCKF_VISUALIZER_HELPER_H

#include <Eigen/Eigen>
#include <memory>

namespace ov_type {
class PoseJPL;
}

namespace ov_msckf {

class State;
class VioManager;
class Simulator;

/**
 * @brief Helper class that handles some common versions into and out of ROS formats
 */
class VisualizerHelper {
public:

  static void init_total_state_files(const std::string& filepath_est, const std::string& filepath_std, const std::string& filepath_gt,
                                     std::shared_ptr<Simulator> sim, 
                                     std::ofstream &of_state_est, std::ofstream &of_state_std, std::ofstream &of_state_gt);


  /**
   * @brief Save current estimate state and groundtruth including calibration
   * @param state Pointer to the state
   * @param sim Pointer to the simulator (or null)
   * @param of_state_est Output file for state estimate
   * @param of_state_std Output file for covariance
   * @param of_state_gt Output file for groundtruth (if we have it from sim)
   */
  static void sim_save_total_state_to_file(std::shared_ptr<State> state, std::shared_ptr<Simulator> sim, std::ofstream &of_state_est,
                                           std::ofstream &of_state_std, std::ofstream &of_state_gt);
private:
  // Cannot create this class
  VisualizerHelper() = default;
};

} // namespace ov_msckf

#endif // OV_MSCKF_ROSVISUALIZER_HELPER_H
