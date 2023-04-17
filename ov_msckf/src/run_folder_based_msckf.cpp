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

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"
#include "ros/ROS2VisualizerForFolderBasedDataset.h"
#include <rclcpp/rclcpp.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace ov_msckf;

std::shared_ptr<VioManager> sys;
std::shared_ptr<ROS2VisualizerForFolderBasedDataset> viz;

// Main function
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  //google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;
  // FLAGS_run_register = false;


  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
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

  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  parser->set_node(node);

  // Verbosity
  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = true;
  sys = std::make_shared<VioManager>(params);
  viz = std::make_shared<ROS2VisualizerForFolderBasedDataset>(node, sys);
  viz->setup_player(parser);

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  viz->wait_play_over();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  viz->stop_algo_threads();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  viz->stop_visualization_thread();

  // // Spin off to ROS
  // PRINT_DEBUG("done...spinning to ros\n");
  // // rclcpp::spin(node);
  // rclcpp::executors::MultiThreadedExecutor executor;
  // executor.add_node(node);
  // executor.spin();

  // Final visualization
  viz->visualize_final();
  rclcpp::shutdown();


  std::cout << "Destroying VioManager ..." << std::endl;
  sys.reset();
  std::cout << "Destroying ROS2Visualizer ..." << std::endl;
  viz.reset();
  std::cout << "All done." << std::endl;

  // Done!
  return EXIT_SUCCESS;
}
