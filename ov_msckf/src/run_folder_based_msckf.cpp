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

#include <memory>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "no_ros/Viewer.h"
#include "utils/dataset_reader.h"
#include "interface/HeisenbergVIO.h"


#if ROS_AVAILABLE == 2
#include "ros/ROS2VisualizerForFolderBasedDataset.h"
#include <rclcpp/rclcpp.hpp>
std::shared_ptr<ov_msckf::ROS2VisualizerForFolderBasedDataset> viz;
extern std::weak_ptr<rclcpp::Node> unique_parser_node;

#elif ROS_AVAILABLE == 0
#include "no_ros/VisualizerForFolderBasedDataset.h"
std::shared_ptr<ov_msckf::VisualizerForFolderBasedDataset> viz;

// #define USE_GFLAGS
#ifdef USE_GFLAGS
DEFINE_string(config_path, "", "config_path");
DEFINE_string(dataset, "", "dataset");
DEFINE_double(play_rate, 1.0, "play_rate");
DEFINE_string(output_dir, "", "output_dir");
DEFINE_bool(save_feature_images, false, "save_feature_images");
DEFINE_bool(save_total_state, false, "save_total_state");
#endif

#endif

std::shared_ptr<heisenberg_algo::VIO> sys;
extern std::shared_ptr<ov_msckf::VioManager> getVioManagerFromVioInterface(heisenberg_algo::VIO*);

// Main function
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

#ifdef USE_GFLAGS  // ROS_AVAILABLE == 0  
  google::ParseCommandLineFlags(&argc, &argv, false);
#endif

  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;
  // FLAGS_run_register = false;


  std::string config_path = "";
  std::string dataset = "";
  double play_rate = 1.0;
  std::string output_dir = "";
  bool save_feature_images = false;
  bool save_total_state = true;

#if ROS_AVAILABLE == 2
  std::cout << "ROS_AVAILABLE == 2" << std::endl;
  if (argc > 1) {
    config_path = argv[1];
  }

  // Launch our ros node
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("run_folder_based_msckf", options);
  node->get_parameter<std::string>("config_path", config_path);

  if (node->has_parameter("dataset")) {
    node->get_parameter<std::string>("dataset", dataset);
  }
  if (node->has_parameter("play_rate")) {
    std::string play_rate_str;
    node->get_parameter<std::string>("play_rate", play_rate_str);
    play_rate = std::stod(play_rate_str);
  }
  if (node->has_parameter("output_dir")) {
    node->get_parameter<std::string>("output_dir", output_dir);
  }
  if (node->has_parameter("save_feature_images")) {
    node->get_parameter<bool>("save_feature_images", save_feature_images);
  }
  if (node->has_parameter("save_total_state")) {
    node->get_parameter<bool>("save_total_state", save_total_state);
  }  

  unique_parser_node = node;
#elif ROS_AVAILABLE == 0
  std::cout << "ROS_AVAILABLE == 0" << std::endl;
#ifdef USE_GFLAGS  // ROS_AVAILABLE == 0  
  config_path = FLAGS_config_path;
  dataset = FLAGS_dataset;
  play_rate = FLAGS_play_rate;
  output_dir = FLAGS_output_dir;
  save_feature_images = FLAGS_save_feature_images;
  save_total_state = FLAGS_save_total_state;
#else
  if (argc <= 3) {
    PRINT_ERROR(RED "Too few arguments!\n" RESET);
    std::cout << "Usage: " << std::endl;
    std::cout << argv[0] << "  <config_path>  <dataset>  <output_dir>  [play_rate]  [save_feature_images: 0 or 1]  [save_total_state:  0 or 1]" << std::endl;
    return -1;
  }
  config_path = argv[1];
  dataset = argv[2];
  output_dir = argv[3];
  if (argc > 4) {
    play_rate = std::stod(argv[4]);
  }
  if (argc > 5) {
    save_feature_images = std::stoi(argv[5]);
  }
  if (argc > 6) {
    save_total_state = std::stoi(argv[5]);
  }
#endif  

#endif

  if (config_path.empty()) {
    PRINT_ERROR(RED "config_path is not set!\n" RESET);
    return 1;
  }
  if (dataset.empty()) {
    PRINT_ERROR(RED "dataset is not set!\n" RESET);
    return 2;
  }

  sys = std::make_shared<heisenberg_algo::VIO>(config_path);
  auto gl_viewer = std::make_shared<ov_msckf::Viewer>(sys);


  // auto internal_sys = getVioManagerFromVioInterface(sys.get());
  // bool stereo = (internal_sys->get_params().state_options.num_cameras == 2);
  // internal_sys.reset();

#if ROS_AVAILABLE == 2
  viz = std::make_shared<ov_msckf::ROS2VisualizerForFolderBasedDataset>(node, sys, gl_viewer, output_dir, save_feature_images, save_total_state);
#elif ROS_AVAILABLE == 0
  viz = std::make_shared<ov_msckf::VisualizerForFolderBasedDataset>(sys, gl_viewer, output_dir, save_feature_images, save_total_state);
#endif

  viz->setup_player(dataset, play_rate);
  viz->wait_play_over();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  sys->Shutdown();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  viz->stop_visualization_thread();


#if ROS_AVAILABLE == 2
  // Final visualization
  viz->visualize_final();

  rclcpp::shutdown();
#endif

  std::cout << "Destroying VIO ..." << std::endl;
  sys.reset();
  std::cout << "Destroying Visualizer ..." << std::endl;
  viz.reset();
  std::cout << "All done." << std::endl;

  // Done!
  return EXIT_SUCCESS;
}
