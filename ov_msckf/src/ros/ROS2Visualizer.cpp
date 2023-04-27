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

#include "ROS2Visualizer.h"

#include "core/VioManager.h"
#include "ros/ROSVisualizerHelper.h"
#include "sim/Simulator.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

ROS2Visualizer::ROS2Visualizer(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim)
    : _node(node), _app(app), _sim(sim), thread_update_running(false) {

  // Setup our transform broadcaster
  mTfBr = std::make_shared<tf2_ros::TransformBroadcaster>(node);

  // Create image transport
  image_transport::ImageTransport it(node);

  // Setup pose and path publisher
  pub_poseimu = node->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("poseimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_poseimu->get_topic_name());
  pub_odomimu = node->create_publisher<nav_msgs::msg::Odometry>("odomimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_odomimu->get_topic_name());
  pub_pathimu = node->create_publisher<nav_msgs::msg::Path>("pathimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_pathimu->get_topic_name());

  // 3D points publishing
  pub_points_msckf = node->create_publisher<sensor_msgs::msg::PointCloud2>("points_msckf", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_msckf->get_topic_name());
  pub_points_slam = node->create_publisher<sensor_msgs::msg::PointCloud2>("points_slam", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_msckf->get_topic_name());
  pub_points_aruco = node->create_publisher<sensor_msgs::msg::PointCloud2>("points_aruco", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_aruco->get_topic_name());
  pub_points_sim = node->create_publisher<sensor_msgs::msg::PointCloud2>("points_sim", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_sim->get_topic_name());

  // Our tracking image
  it_pub_tracks = it.advertise("trackhist", 2);
  PRINT_DEBUG("Publishing: %s\n", it_pub_tracks.getTopic().c_str());

  // Groundtruth publishers
  pub_posegt = node->create_publisher<geometry_msgs::msg::PoseStamped>("posegt", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_posegt->get_topic_name());
  pub_pathgt = node->create_publisher<nav_msgs::msg::Path>("pathgt", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_pathgt->get_topic_name());

  // Loop closure publishers
  pub_loop_pose = node->create_publisher<nav_msgs::msg::Odometry>("loop_pose", 2);
  pub_loop_point = node->create_publisher<sensor_msgs::msg::PointCloud>("loop_feats", 2);
  pub_loop_extrinsic = node->create_publisher<nav_msgs::msg::Odometry>("loop_extrinsic", 2);
  pub_loop_intrinsics = node->create_publisher<sensor_msgs::msg::CameraInfo>("loop_intrinsics", 2);
  it_pub_loop_img_depth = it.advertise("loop_depth", 2);
  it_pub_loop_img_depth_color = it.advertise("loop_depth_colored", 2);

  // option to enable publishing of global to IMU transformation
  if (node->has_parameter("publish_global_to_imu_tf")) {
    node->get_parameter<bool>("publish_global_to_imu_tf", publish_global2imu_tf);
  }
  if (node->has_parameter("publish_calibration_tf")) {
    node->get_parameter<bool>("publish_calibration_tf", publish_calibration_tf);
  }

  // Load groundtruth if we have it and are not doing simulation
  // NOTE: needs to be a csv ASL format file
  std::string path_to_gt;
  bool has_gt = node->get_parameter("path_gt", path_to_gt);
  if (has_gt && _sim == nullptr && !path_to_gt.empty()) {
    DatasetReader::load_gt_file(path_to_gt, gt_states);
    PRINT_DEBUG("gt file path is: %s\n", path_to_gt.c_str());
  }

  // Load if we should save the total state to file
  // If so, then open the file and create folders as needed
  if (node->has_parameter("save_total_state")) {
    node->get_parameter<bool>("save_total_state", save_total_state);
  }
  if (save_total_state) {

    // files we will open
    std::string filepath_est = "state_estimate.txt";
    std::string filepath_std = "state_deviation.txt";
    std::string filepath_gt = "state_groundtruth.txt";
    if (node->has_parameter("filepath_est")) {
      node->get_parameter<std::string>("filepath_est", filepath_est);
    }
    if (node->has_parameter("filepath_std")) {
      node->get_parameter<std::string>("filepath_std", filepath_std);
    }
    if (node->has_parameter("filepath_gt")) {
      node->get_parameter<std::string>("filepath_gt", filepath_gt);
    }

    // If it exists, then delete it
    if (boost::filesystem::exists(filepath_est))
      boost::filesystem::remove(filepath_est);
    if (boost::filesystem::exists(filepath_std))
      boost::filesystem::remove(filepath_std);

    // Create folder path to this location if not exists
    boost::filesystem::create_directories(boost::filesystem::path(filepath_est.c_str()).parent_path());
    boost::filesystem::create_directories(boost::filesystem::path(filepath_std.c_str()).parent_path());

    // Open the files
    of_state_est.open(filepath_est.c_str());
    of_state_std.open(filepath_std.c_str());
    of_state_est << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;
    of_state_std << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;

    // Groundtruth if we are simulating
    if (_sim != nullptr) {
      if (boost::filesystem::exists(filepath_gt))
        boost::filesystem::remove(filepath_gt);
      boost::filesystem::create_directories(boost::filesystem::path(filepath_gt.c_str()).parent_path());
      of_state_gt.open(filepath_gt.c_str());
      of_state_gt << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans .... etc" << std::endl;
    }
  }

  // Start thread for the visualizing
  stop_viz_request_ = false;
  _vis_thread = std::make_shared<std::thread>([&] {
    pthread_setname_np(pthread_self(), "ov_visualize");

    // use a high rate to ensure the _vis_output to update in time (which is also needed in visualize_odometry()).
    // rclcpp::Rate loop_rate(20);
    rclcpp::Rate loop_rate(40);
    while (rclcpp::ok() && !stop_viz_request_) {
      visualize();
      loop_rate.sleep();
    }
  });
}

void ROS2Visualizer::stop_visualization_thread() {
  stop_viz_request_ = true;
  if (_vis_thread && _vis_thread->joinable()) {
    _vis_thread->join();
    _vis_thread.reset();
  }
  std::cout << "visualization_thread stoped." << std::endl;
}

void ROS2Visualizer::setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser) {

  // We need a valid parser
  assert(parser != nullptr);

  if (_node->has_parameter("heisenberg_dataset")) {
    std::string heisenberg_dataset;
    _node->get_parameter<std::string>("heisenberg_dataset", heisenberg_dataset);
    load_heisenberg_img_queues(heisenberg_dataset);
  }

  // Create imu subscriber (handle legacy ros param info)
  std::string topic_imu;
  _node->declare_parameter<std::string>("topic_imu", "/imu0");
  _node->get_parameter("topic_imu", topic_imu);
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  sub_imu = _node->create_subscription<sensor_msgs::msg::Imu>(topic_imu, rclcpp::SensorDataQoS(),
                                                              std::bind(&ROS2Visualizer::callback_inertial, this, std::placeholders::_1));

  PRINT_INFO("subscribing to IMU: %s\n", topic_imu.c_str());

  // Logic for sync stereo subscriber
  // https://answers.ros.org/question/96346/subscribe-to-two-image_raws-with-one-function/?answer=96491#post-id-96491
  if (_app->get_params().state_options.num_cameras == 2) {
    // Read in the topics
    std::string cam_topic0, cam_topic1;
    _node->declare_parameter<std::string>("topic_camera" + std::to_string(0), "/cam" + std::to_string(0) + "/image_raw");
    _node->get_parameter("topic_camera" + std::to_string(0), cam_topic0);
    _node->declare_parameter<std::string>("topic_camera" + std::to_string(1), "/cam" + std::to_string(1) + "/image_raw");
    _node->get_parameter("topic_camera" + std::to_string(1), cam_topic0);
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(0), "rostopic", cam_topic0);
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(1), "rostopic", cam_topic1);
    // Create sync filter (they have unique pointers internally, so we have to use move logic here...)
    auto image_sub0 = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(_node, cam_topic0);
    auto image_sub1 = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(_node, cam_topic1);
    auto sync = std::make_shared<message_filters::Synchronizer<sync_pol>>(sync_pol(10), *image_sub0, *image_sub1);
    sync->registerCallback(std::bind(&ROS2Visualizer::callback_stereo, this, std::placeholders::_1, std::placeholders::_2, 0, 1));
    // sync->registerCallback([](const sensor_msgs::msg::Image::SharedPtr msg0, const sensor_msgs::msg::Image::SharedPtr msg1)
    // {callback_stereo(msg0, msg1, 0, 1);});
    // sync->registerCallback(&callback_stereo2); // since the above two alternatives fail to compile for some reason
    // Append to our vector of subscribers
    sync_cam.push_back(sync);
    sync_subs_cam.push_back(image_sub0);
    sync_subs_cam.push_back(image_sub1);
    PRINT_INFO("subscribing to cam (stereo): %s\n", cam_topic0.c_str());
    PRINT_INFO("subscribing to cam (stereo): %s\n", cam_topic1.c_str());
  } else {
    // Now we should add any non-stereo callbacks here
    for (int i = 0; i < _app->get_params().state_options.num_cameras; i++) {
      // read in the topic
      std::string cam_topic;
      _node->declare_parameter<std::string>("topic_camera" + std::to_string(i), "/cam" + std::to_string(i) + "/image_raw");
      _node->get_parameter("topic_camera" + std::to_string(i), cam_topic);
      parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
      // create subscriber
      // auto sub = _node->create_subscription<sensor_msgs::msg::Image>(
      //    cam_topic, rclcpp::SensorDataQoS(), std::bind(&ROS2Visualizer::callback_monocular, this, std::placeholders::_1, i));
      auto sub = _node->create_subscription<sensor_msgs::msg::Image>(
          cam_topic, 10, [this, i](const sensor_msgs::msg::Image::SharedPtr msg0) { callback_monocular(msg0, i); });
      subs_cam.push_back(sub);
      PRINT_INFO("subscribing to cam (mono): %s\n", cam_topic.c_str());
    }
  }
}

void ROS2Visualizer::visualize() {
  // Return if we have already visualized
  // The check is fast.
  auto simple_output = _app->getLastOutput(false, false);
  if (simple_output->status.timestamp <= 0 || last_visualization_timestamp == simple_output->status.timestamp && simple_output->status.initialized)
    return;
  _app->clear_older_tracking_cache(last_visualization_timestamp);

  _vis_output = _app->getLastOutput(true, true);
  // last_visualization_timestamp = _vis_output->state_clone->_timestamp;
  last_visualization_timestamp = _vis_output->status.timestamp;

  // Start timing
  // boost::posix_time::ptime rT0_1, rT0_2;
  // rT0_1 = boost::posix_time::microsec_clock::local_time();

  // publish current image
  publish_images();

  // Return if we have not inited
  if (!_vis_output->status.initialized)
    return;

  // Save the start time of this dataset
  if (!start_time_set) {
    rT1 = boost::posix_time::microsec_clock::local_time();
    start_time_set = true;
  }

  // publish state
  publish_state();

  // publish points
  publish_features();

  // Publish gt if we have it
  publish_groundtruth();

  // Publish keyframe information
  publish_loopclosure_information();

  // Save total state
  if (save_total_state) {
    ROSVisualizerHelper::sim_save_total_state_to_file(_vis_output->state_clone, _sim, of_state_est, of_state_std, of_state_gt);
  }

  // Print how much time it took to publish / displaying things
  // rT0_2 = boost::posix_time::microsec_clock::local_time();
  // double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
  // PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for visualization\n" RESET, time_total);
}

void ROS2Visualizer::visualize_odometry(double timestamp) {

  if (timestamp - last_visualization_timestamp_odom < 0.019) {
    // don't visulize too frequently (50Hz at most)
    return;
  }
  last_visualization_timestamp_odom = timestamp;

  // Return if we have not inited and a second has passes
  if (!_vis_output || !_vis_output->status.initialized || (timestamp - _vis_output->status.initialized_time) < 1) {
    return;
  }

  // Get fast propagate state at the desired timestamp
  std::shared_ptr<State> state = _vis_output->state_clone;  // shared_ptr itself is thread-safe.
  Eigen::Matrix<double, 13, 1> state_plus = Eigen::Matrix<double, 13, 1>::Zero();
  Eigen::Matrix<double, 12, 12> cov_plus = Eigen::Matrix<double, 12, 12>::Zero();
  if (!_app->get_propagator()->fast_state_propagate(state, timestamp, state_plus, cov_plus))
    return;

  // Publish our odometry message if requested
  if (pub_odomimu->get_subscription_count() != 0) {

    // Our odometry message
    nav_msgs::msg::Odometry odomIinM;
    odomIinM.header.stamp = ROSVisualizerHelper::get_time_from_seconds(timestamp);
    odomIinM.header.frame_id = "global";

    // The POSE component (orientation and position)
    odomIinM.pose.pose.orientation.x = state_plus(0);
    odomIinM.pose.pose.orientation.y = state_plus(1);
    odomIinM.pose.pose.orientation.z = state_plus(2);
    odomIinM.pose.pose.orientation.w = state_plus(3);
    odomIinM.pose.pose.position.x = state_plus(4);
    odomIinM.pose.pose.position.y = state_plus(5);
    odomIinM.pose.pose.position.z = state_plus(6);

    // The TWIST component (angular and linear velocities)
    odomIinM.child_frame_id = "imu";
    odomIinM.twist.twist.linear.x = state_plus(7);   // vel in local frame
    odomIinM.twist.twist.linear.y = state_plus(8);   // vel in local frame
    odomIinM.twist.twist.linear.z = state_plus(9);   // vel in local frame
    odomIinM.twist.twist.angular.x = state_plus(10); // we do not estimate this...
    odomIinM.twist.twist.angular.y = state_plus(11); // we do not estimate this...
    odomIinM.twist.twist.angular.z = state_plus(12); // we do not estimate this...

    // Finally set the covariance in the message (in the order position then orientation as per ros convention)
    Eigen::Matrix<double, 12, 12> Phi = Eigen::Matrix<double, 12, 12>::Zero();
    Phi.block(0, 3, 3, 3).setIdentity();
    Phi.block(3, 0, 3, 3).setIdentity();
    Phi.block(6, 6, 6, 6).setIdentity();
    cov_plus = Phi * cov_plus * Phi.transpose();
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        odomIinM.pose.covariance[6 * r + c] = cov_plus(r, c);
      }
    }
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        odomIinM.twist.covariance[6 * r + c] = cov_plus(r + 6, c + 6);
      }
    }
    pub_odomimu->publish(odomIinM);
  }

  // Publish our transform on TF
  // NOTE: since we use JPL we have an implicit conversion to Hamilton when we publish
  // NOTE: a rotation from GtoI in JPL has the same xyzw as a ItoG Hamilton rotation
  auto odom_pose = std::make_shared<ov_type::PoseJPL>();
  odom_pose->set_value(state_plus.block(0, 0, 7, 1));
  geometry_msgs::msg::TransformStamped trans = ROSVisualizerHelper::get_stamped_transform_from_pose(_node, odom_pose, false);
  trans.header.stamp = _node->now();
  trans.header.frame_id = "global";
  trans.child_frame_id = "imu";
  if (publish_global2imu_tf) {
    mTfBr->sendTransform(trans);
  }

  // Loop through each camera calibration and publish it
  for (const auto &calib : state->_calib_IMUtoCAM) {
    geometry_msgs::msg::TransformStamped trans_calib = ROSVisualizerHelper::get_stamped_transform_from_pose(_node, calib.second, true);
    trans_calib.header.stamp = _node->now();
    trans_calib.header.frame_id = "imu";
    trans_calib.child_frame_id = "cam" + std::to_string(calib.first);
    if (publish_calibration_tf) {
      mTfBr->sendTransform(trans_calib);
    }
  }
}

void ROS2Visualizer::visualize_final() {

  // Final time offset value
  if (_vis_output->state_clone->_options.do_calib_camera_timeoffset) {
    PRINT_INFO(REDPURPLE "camera-imu timeoffset = %.5f\n\n" RESET, _vis_output->state_clone->_calib_dt_CAMtoIMU->value()(0));
  }

  // Final camera intrinsics
  if (_vis_output->state_clone->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < _vis_output->state_clone->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = _vis_output->state_clone->_cam_intrinsics.at(i);
      PRINT_INFO(REDPURPLE "cam%d intrinsics:\n" RESET, (int)i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n" RESET, calib->value()(0), calib->value()(1), calib->value()(2), calib->value()(3));
      PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,%.5f\n\n" RESET, calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Final camera extrinsics
  if (_vis_output->state_clone->_options.do_calib_camera_pose) {
    for (int i = 0; i < _vis_output->state_clone->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = _vis_output->state_clone->_calib_IMUtoCAM.at(i);
      Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
      T_CtoI.block(0, 0, 3, 3) = quat_2_Rot(calib->quat()).transpose();
      T_CtoI.block(0, 3, 3, 1) = -T_CtoI.block(0, 0, 3, 3) * calib->pos();
      PRINT_INFO(REDPURPLE "T_C%dtoI:\n" RESET, i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(0, 0), T_CtoI(0, 1), T_CtoI(0, 2), T_CtoI(0, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(1, 0), T_CtoI(1, 1), T_CtoI(1, 2), T_CtoI(1, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(2, 0), T_CtoI(2, 1), T_CtoI(2, 2), T_CtoI(2, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n\n" RESET, T_CtoI(3, 0), T_CtoI(3, 1), T_CtoI(3, 2), T_CtoI(3, 3));
    }
  }

  // Publish RMSE if we have it
  if (!gt_states.empty()) {
    PRINT_INFO(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_INFO(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
  }

  // Publish RMSE and NEES if doing simulation
  if (_sim != nullptr) {
    PRINT_INFO(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_INFO(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
    PRINT_INFO(REDPURPLE "NEES: %.3f (deg) orientation\n" RESET, summed_nees_ori / summed_number);
    PRINT_INFO(REDPURPLE "NEES: %.3f (m) position\n\n" RESET, summed_nees_pos / summed_number);
  }

  // Print the total time
  rT2 = boost::posix_time::microsec_clock::local_time();
  PRINT_INFO(REDPURPLE "TIME: %.3f seconds\n\n" RESET, (rT2 - rT1).total_microseconds() * 1e-6);
}

void ROS2Visualizer::callback_inertial(const sensor_msgs::msg::Imu::SharedPtr msg) {

  heisenberg_imu_hook_for_img_publishing(msg);
  
  // convert into correct format
  ov_core::ImuData message;
  message.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  message.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  message.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;

  // send it to our VIO system
  _app->feed_measurement_imu(message);
  visualize_odometry(message.timestamp);
}

void ROS2Visualizer::callback_monocular(const sensor_msgs::msg::Image::SharedPtr msg0, int cam_id0) {

  // Get the image
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Create the measurement
  ov_core::CameraData message;
  message.timestamp = cv_ptr->header.stamp.sec + cv_ptr->header.stamp.nanosec * 1e-9;
  message.sensor_ids.push_back(cam_id0);
  message.images.push_back(cv_ptr->image.clone());

  // Load the mask if we are using it, else it is empty
  // TODO: in the future we should get this from external pixel segmentation
  if (_app->get_params().use_mask) {
    message.masks.push_back(_app->get_params().masks.at(cam_id0));
  } else {
    message.masks.push_back(cv::Mat::zeros(cv_ptr->image.rows, cv_ptr->image.cols, CV_8UC1));
  }

  _app->feed_measurement_camera(std::move(message));
}

void ROS2Visualizer::callback_stereo(const sensor_msgs::msg::Image::ConstSharedPtr msg0, const sensor_msgs::msg::Image::ConstSharedPtr msg1,
                                     int cam_id0, int cam_id1) {

  // Get the image
  cv_bridge::CvImageConstPtr cv_ptr0;
  try {
    cv_ptr0 = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Get the image
  cv_bridge::CvImageConstPtr cv_ptr1;
  try {
    cv_ptr1 = cv_bridge::toCvShare(msg1, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Create the measurement
  ov_core::CameraData message;
  message.timestamp = cv_ptr0->header.stamp.sec + cv_ptr0->header.stamp.nanosec * 1e-9;
  message.sensor_ids.push_back(cam_id0);
  message.sensor_ids.push_back(cam_id1);
  message.images.push_back(cv_ptr0->image.clone());
  message.images.push_back(cv_ptr1->image.clone());

  // Load the mask if we are using it, else it is empty
  // TODO: in the future we should get this from external pixel segmentation
  if (_app->get_params().use_mask) {
    message.masks.push_back(_app->get_params().masks.at(cam_id0));
    message.masks.push_back(_app->get_params().masks.at(cam_id1));
  } else {
    // message.masks.push_back(cv::Mat(cv_ptr0->image.rows, cv_ptr0->image.cols, CV_8UC1, cv::Scalar(255)));
    message.masks.push_back(cv::Mat::zeros(cv_ptr0->image.rows, cv_ptr0->image.cols, CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv_ptr1->image.rows, cv_ptr1->image.cols, CV_8UC1));
  }

  _app->feed_measurement_camera(std::move(message));
}

void ROS2Visualizer::publish_state() {
  // Get the current state
  std::shared_ptr<State> state = _vis_output->state_clone;

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = state->_timestamp + t_ItoC;
  // std::cout << "t_ItoC = " << t_ItoC << ", state->_timestamp = " << state->_timestamp << ", timestamp_inI = " << timestamp_inI << std::endl;

  // Create pose of IMU (note we use the bag time)
  geometry_msgs::msg::PoseWithCovarianceStamped poseIinM;
  poseIinM.header.stamp = ROSVisualizerHelper::get_time_from_seconds(timestamp_inI);
  poseIinM.header.frame_id = "global";
  poseIinM.pose.pose.orientation.x = state->_imu->quat()(0);
  poseIinM.pose.pose.orientation.y = state->_imu->quat()(1);
  poseIinM.pose.pose.orientation.z = state->_imu->quat()(2);
  poseIinM.pose.pose.orientation.w = state->_imu->quat()(3);
  poseIinM.pose.pose.position.x = state->_imu->pos()(0);
  poseIinM.pose.pose.position.y = state->_imu->pos()(1);
  poseIinM.pose.pose.position.z = state->_imu->pos()(2);

  // Finally set the covariance in the message (in the order position then orientation as per ros convention)
  std::vector<std::shared_ptr<Type>> statevars;
  statevars.push_back(state->_imu->pose()->p());
  statevars.push_back(state->_imu->pose()->q());
  Eigen::Matrix<double, 6, 6> covariance_posori = StateHelper::get_marginal_covariance(_vis_output->state_clone, statevars);
  for (int r = 0; r < 6; r++) {
    for (int c = 0; c < 6; c++) {
      poseIinM.pose.covariance[6 * r + c] = covariance_posori(r, c);
    }
  }
  pub_poseimu->publish(poseIinM);

  //=========================================================
  //=========================================================

  // Append to our pose vector
  geometry_msgs::msg::PoseStamped posetemp;
  posetemp.header = poseIinM.header;
  posetemp.pose = poseIinM.pose.pose;
  poses_imu.push_back(posetemp);

  // Create our path (imu)
  // NOTE: We downsample the number of poses as needed to prevent rviz crashes
  // NOTE: https://github.com/ros-visualization/rviz/issues/1107
  nav_msgs::msg::Path arrIMU;
  arrIMU.header.stamp = _node->now();
  arrIMU.header.frame_id = "global";
  for (size_t i = 0; i < poses_imu.size(); i += std::floor((double)poses_imu.size() / 16384.0) + 1) {
    arrIMU.poses.push_back(poses_imu.at(i));
  }
  pub_pathimu->publish(arrIMU);
}

void ROS2Visualizer::publish_images() {
  if (_vis_output->status.timestamp <= 0)
    return;

  // Return if we have already visualized
  // double cur_state_timestamp = _vis_output->state_clone->_timestamp;
  double cur_state_timestamp = _vis_output->status.timestamp;

  if (last_visualization_timestamp_image == cur_state_timestamp && _vis_output->status.initialized)
    return;
  last_visualization_timestamp_image = cur_state_timestamp;

  // Check if we have subscribers
  if (it_pub_tracks.getNumSubscribers() == 0)
    return;

  // Get our image of history tracks
  cv::Mat img_history = _app->get_historical_viz_image(cur_state_timestamp);
  if (img_history.empty())
    return;

  // Create our message
  std_msgs::msg::Header header;
  header.stamp = _node->now();
  header.frame_id = "cam0";
  sensor_msgs::msg::Image::SharedPtr exl_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

  // Publish
  it_pub_tracks.publish(exl_msg);
}

void ROS2Visualizer::publish_features() {
  // Check if we have subscribers
  if (pub_points_msckf->get_subscription_count() == 0 && pub_points_slam->get_subscription_count() == 0 &&
      pub_points_aruco->get_subscription_count() == 0 && pub_points_sim->get_subscription_count() == 0)
    return;

  // Get our good MSCKF features
  std::vector<Eigen::Vector3d>& feats_msckf = _vis_output->visualization.good_features_MSCKF;
  sensor_msgs::msg::PointCloud2 cloud = ROSVisualizerHelper::get_ros_pointcloud(_node, feats_msckf);
  pub_points_msckf->publish(cloud);

  // Get our good SLAM features
  std::vector<Eigen::Vector3d>& feats_slam = _vis_output->visualization.features_SLAM;
  sensor_msgs::msg::PointCloud2 cloud_SLAM = ROSVisualizerHelper::get_ros_pointcloud(_node, feats_slam);
  pub_points_slam->publish(cloud_SLAM);

  // Get our good ARUCO features
  std::vector<Eigen::Vector3d>& feats_aruco = _vis_output->visualization.features_ARUCO;
  sensor_msgs::msg::PointCloud2 cloud_ARUCO = ROSVisualizerHelper::get_ros_pointcloud(_node, feats_aruco);
  pub_points_aruco->publish(cloud_ARUCO);

  // Skip the rest of we are not doing simulation
  if (_sim == nullptr)
    return;

  // Get our good SIMULATION features
  std::vector<Eigen::Vector3d> feats_sim = _sim->get_map_vec();
  sensor_msgs::msg::PointCloud2 cloud_SIM = ROSVisualizerHelper::get_ros_pointcloud(_node, feats_sim);
  pub_points_sim->publish(cloud_SIM);
}

void ROS2Visualizer::publish_groundtruth() {
  // Our groundtruth state
  Eigen::Matrix<double, 17, 1> state_gt;

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = _vis_output->state_clone->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = _vis_output->state_clone->_timestamp + t_ItoC;

  // Check that we have the timestamp in our GT file [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
  if (_sim == nullptr && (gt_states.empty() || !DatasetReader::get_gt_state(timestamp_inI, state_gt, gt_states))) {
    return;
  }

  // Get the simulated groundtruth
  // NOTE: we get the true time in the IMU clock frame
  if (_sim != nullptr) {
    timestamp_inI = _vis_output->state_clone->_timestamp + _sim->get_true_parameters().calib_camimu_dt;
    if (!_sim->get_state(timestamp_inI, state_gt))
      return;
  }

  // Get the GT and system state state
  Eigen::Matrix<double, 16, 1> state_ekf = _vis_output->state_clone->_imu->value();

  // Create pose of IMU
  geometry_msgs::msg::PoseStamped poseIinM;
  poseIinM.header.stamp = ROSVisualizerHelper::get_time_from_seconds(timestamp_inI);
  poseIinM.header.frame_id = "global";
  poseIinM.pose.orientation.x = state_gt(1, 0);
  poseIinM.pose.orientation.y = state_gt(2, 0);
  poseIinM.pose.orientation.z = state_gt(3, 0);
  poseIinM.pose.orientation.w = state_gt(4, 0);
  poseIinM.pose.position.x = state_gt(5, 0);
  poseIinM.pose.position.y = state_gt(6, 0);
  poseIinM.pose.position.z = state_gt(7, 0);
  pub_posegt->publish(poseIinM);

  // Append to our pose vector
  poses_gt.push_back(poseIinM);

  // Create our path (imu)
  // NOTE: We downsample the number of poses as needed to prevent rviz crashes
  // NOTE: https://github.com/ros-visualization/rviz/issues/1107
  nav_msgs::msg::Path arrIMU;
  arrIMU.header.stamp = _node->now();
  arrIMU.header.frame_id = "global";
  for (size_t i = 0; i < poses_gt.size(); i += std::floor((double)poses_gt.size() / 16384.0) + 1) {
    arrIMU.poses.push_back(poses_gt.at(i));
  }
  pub_pathgt->publish(arrIMU);

  // Publish our transform on TF
  geometry_msgs::msg::TransformStamped trans;
  trans.header.stamp = _node->now();
  trans.header.frame_id = "global";
  trans.child_frame_id = "truth";
  trans.transform.rotation.x = state_gt(1, 0);
  trans.transform.rotation.y = state_gt(2, 0);
  trans.transform.rotation.z = state_gt(3, 0);
  trans.transform.rotation.w = state_gt(4, 0);
  trans.transform.translation.x = state_gt(5, 0);
  trans.transform.translation.y = state_gt(6, 0);
  trans.transform.translation.z = state_gt(7, 0);
  if (publish_global2imu_tf) {
    mTfBr->sendTransform(trans);
  }

  //==========================================================================
  //==========================================================================

  // Difference between positions
  double dx = state_ekf(4, 0) - state_gt(5, 0);
  double dy = state_ekf(5, 0) - state_gt(6, 0);
  double dz = state_ekf(6, 0) - state_gt(7, 0);
  double err_pos = std::sqrt(dx * dx + dy * dy + dz * dz);

  // Quaternion error
  Eigen::Matrix<double, 4, 1> quat_gt, quat_st, quat_diff;
  quat_gt << state_gt(1, 0), state_gt(2, 0), state_gt(3, 0), state_gt(4, 0);
  quat_st << state_ekf(0, 0), state_ekf(1, 0), state_ekf(2, 0), state_ekf(3, 0);
  quat_diff = quat_multiply(quat_st, Inv(quat_gt));
  double err_ori = (180 / M_PI) * 2 * quat_diff.block(0, 0, 3, 1).norm();

  //==========================================================================
  //==========================================================================

  // Get covariance of pose
  std::vector<std::shared_ptr<Type>> statevars;
  statevars.push_back(_vis_output->state_clone->_imu->q());
  statevars.push_back(_vis_output->state_clone->_imu->p());
  Eigen::Matrix<double, 6, 6> covariance = StateHelper::get_marginal_covariance(_vis_output->state_clone, statevars);

  // Calculate NEES values
  // NOTE: need to manually multiply things out to make static asserts work
  // NOTE: https://github.com/rpng/open_vins/pull/226
  // NOTE: https://github.com/rpng/open_vins/issues/236
  // NOTE: https://gitlab.com/libeigen/eigen/-/issues/1664
  Eigen::Vector3d quat_diff_vec = quat_diff.block(0, 0, 3, 1);
  Eigen::Vector3d cov_vec = covariance.block(0, 0, 3, 3).inverse() * 2 * quat_diff.block(0, 0, 3, 1);
  double ori_nees = 2 * quat_diff_vec.dot(cov_vec);
  Eigen::Vector3d errpos = state_ekf.block(4, 0, 3, 1) - state_gt.block(5, 0, 3, 1);
  double pos_nees = errpos.transpose() * covariance.block(3, 3, 3, 3).inverse() * errpos;

  //==========================================================================
  //==========================================================================

  // Update our average variables
  if (!std::isnan(ori_nees) && !std::isnan(pos_nees)) {
    summed_mse_ori += err_ori * err_ori;
    summed_mse_pos += err_pos * err_pos;
    summed_nees_ori += ori_nees;
    summed_nees_pos += pos_nees;
    summed_number++;
  }

  // Nice display for the user
  PRINT_INFO(REDPURPLE "error to gt => %.3f, %.3f (deg,m) | rmse => %.3f, %.3f (deg,m) | called %d times\n" RESET, err_ori, err_pos,
             std::sqrt(summed_mse_ori / summed_number), std::sqrt(summed_mse_pos / summed_number), (int)summed_number);
  PRINT_INFO(REDPURPLE "nees => %.1f, %.1f (ori,pos) | avg nees = %.1f, %.1f (ori,pos)\n" RESET, ori_nees, pos_nees,
             summed_nees_ori / summed_number, summed_nees_pos / summed_number);

  //==========================================================================
  //==========================================================================
}

void ROS2Visualizer::publish_loopclosure_information() {
  // Get the current tracks in this frame

  double active_tracks_time1 = _vis_output->status.timestamp;

  // double active_tracks_time1 = -1;
  // double active_tracks_time2 = -1;
  // std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  // std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  // cv::Mat active_cam0_image;
  // _app->get_active_tracks(active_tracks_time1, active_tracks_posinG, active_tracks_uvd);
  // _app->get_active_image(active_tracks_time2, active_cam0_image);

  cv::Mat& active_cam0_image = _vis_output->visualization.active_cam0_image;
  auto& active_tracks_posinG = _vis_output->visualization.active_tracks_posinG;
  auto& active_tracks_uvd = _vis_output->visualization.active_tracks_uvd;

  if (active_tracks_time1 == -1)
    return;
  if (_vis_output->state_clone->_clones_IMU.find(active_tracks_time1) == _vis_output->state_clone->_clones_IMU.end())
    return;
  // if (active_tracks_time1 != active_tracks_time2)
  //   return;

  // Default header
  std_msgs::msg::Header header;
  header.stamp = ROSVisualizerHelper::get_time_from_seconds(active_tracks_time1);

  //======================================================
  // Check if we have subscribers for the pose odometry, camera intrinsics, or extrinsics
  if (pub_loop_pose->get_subscription_count() != 0 || pub_loop_extrinsic->get_subscription_count() != 0 ||
      pub_loop_intrinsics->get_subscription_count() != 0) {

    // PUBLISH HISTORICAL POSE ESTIMATE
    nav_msgs::msg::Odometry odometry_pose;
    odometry_pose.header = header;
    odometry_pose.header.frame_id = "global";
    odometry_pose.pose.pose.position.x = _vis_output->state_clone->_clones_IMU.at(active_tracks_time1)->pos()(0);
    odometry_pose.pose.pose.position.y = _vis_output->state_clone->_clones_IMU.at(active_tracks_time1)->pos()(1);
    odometry_pose.pose.pose.position.z = _vis_output->state_clone->_clones_IMU.at(active_tracks_time1)->pos()(2);
    odometry_pose.pose.pose.orientation.x = _vis_output->state_clone->_clones_IMU.at(active_tracks_time1)->quat()(0);
    odometry_pose.pose.pose.orientation.y = _vis_output->state_clone->_clones_IMU.at(active_tracks_time1)->quat()(1);
    odometry_pose.pose.pose.orientation.z = _vis_output->state_clone->_clones_IMU.at(active_tracks_time1)->quat()(2);
    odometry_pose.pose.pose.orientation.w = _vis_output->state_clone->_clones_IMU.at(active_tracks_time1)->quat()(3);
    pub_loop_pose->publish(odometry_pose);

    // PUBLISH IMU TO CAMERA0 EXTRINSIC
    // need to flip the transform to the IMU frame
    Eigen::Vector4d q_ItoC = _vis_output->state_clone->_calib_IMUtoCAM.at(0)->quat();
    Eigen::Vector3d p_CinI = -_vis_output->state_clone->_calib_IMUtoCAM.at(0)->Rot().transpose() * _vis_output->state_clone->_calib_IMUtoCAM.at(0)->pos();
    nav_msgs::msg::Odometry odometry_calib;
    odometry_calib.header = header;
    odometry_calib.header.frame_id = "imu";
    odometry_calib.pose.pose.position.x = p_CinI(0);
    odometry_calib.pose.pose.position.y = p_CinI(1);
    odometry_calib.pose.pose.position.z = p_CinI(2);
    odometry_calib.pose.pose.orientation.x = q_ItoC(0);
    odometry_calib.pose.pose.orientation.y = q_ItoC(1);
    odometry_calib.pose.pose.orientation.z = q_ItoC(2);
    odometry_calib.pose.pose.orientation.w = q_ItoC(3);
    pub_loop_extrinsic->publish(odometry_calib);

    // PUBLISH CAMERA0 INTRINSICS
    bool is_fisheye = (std::dynamic_pointer_cast<ov_core::CamEqui>(_app->get_params().camera_intrinsics.at(0)) != nullptr);
    sensor_msgs::msg::CameraInfo cameraparams;
    cameraparams.header = header;
    cameraparams.header.frame_id = "cam0";
    cameraparams.distortion_model = is_fisheye ? "equidistant" : "plumb_bob";
    Eigen::VectorXd cparams = _vis_output->state_clone->_cam_intrinsics.at(0)->value();
    cameraparams.d = {cparams(4), cparams(5), cparams(6), cparams(7)};
    cameraparams.k = {cparams(0), 0, cparams(2), 0, cparams(1), cparams(3), 0, 0, 1};
    pub_loop_intrinsics->publish(cameraparams);
  }

  //======================================================
  // PUBLISH FEATURE TRACKS IN THE GLOBAL FRAME OF REFERENCE
  if (pub_loop_point->get_subscription_count() != 0) {

    // Construct the message
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header = header;
    point_cloud.header.frame_id = "global";
    for (const auto &feattimes : active_tracks_posinG) {

      // Get this feature information
      size_t featid = feattimes.first;
      Eigen::Vector3d uvd = Eigen::Vector3d::Zero();
      if (active_tracks_uvd.find(featid) != active_tracks_uvd.end()) {
        uvd = active_tracks_uvd.at(featid);
      }
      Eigen::Vector3d pFinG = active_tracks_posinG.at(featid);

      // Push back 3d point
      geometry_msgs::msg::Point32 p;
      p.x = pFinG(0);
      p.y = pFinG(1);
      p.z = pFinG(2);
      point_cloud.points.push_back(p);

      // Push back the uv_norm, uv_raw, and feature id
      // NOTE: we don't use the normalized coordinates to save time here
      // NOTE: they will have to be re-normalized in the loop closure code
      sensor_msgs::msg::ChannelFloat32 p_2d;
      p_2d.values.push_back(0);
      p_2d.values.push_back(0);
      p_2d.values.push_back(uvd(0));
      p_2d.values.push_back(uvd(1));
      p_2d.values.push_back(featid);
      point_cloud.channels.push_back(p_2d);
    }
    pub_loop_point->publish(point_cloud);
  }

  //======================================================
  // Depth images of sparse points and its colorized version
  if (it_pub_loop_img_depth.getNumSubscribers() != 0 || it_pub_loop_img_depth_color.getNumSubscribers() != 0) {

    // Create the images we will populate with the depths
    std::pair<int, int> wh_pair = {active_cam0_image.cols, active_cam0_image.rows};
    cv::Mat depthmap = cv::Mat::zeros(wh_pair.second, wh_pair.first, CV_16UC1);
    cv::Mat depthmap_viz = active_cam0_image;

    // Loop through all points and append
    for (const auto &feattimes : active_tracks_uvd) {

      // Get this feature information
      size_t featid = feattimes.first;
      Eigen::Vector3d uvd = active_tracks_uvd.at(featid);

      // Skip invalid points
      double dw = 4;
      if (uvd(0) < dw || uvd(0) > wh_pair.first - dw || uvd(1) < dw || uvd(1) > wh_pair.second - dw) {
        continue;
      }

      // Append the depth
      // NOTE: scaled by 1000 to fit the 16U
      // NOTE: access order is y,x (stupid opencv convention stuff)
      depthmap.at<uint16_t>((int)uvd(1), (int)uvd(0)) = (uint16_t)(1000 * uvd(2));

      // Taken from LSD-SLAM codebase segment into 0-4 meter segments:
      // https://github.com/tum-vision/lsd_slam/blob/d1e6f0e1a027889985d2e6b4c0fe7a90b0c75067/lsd_slam_core/src/util/globalFuncs.cpp#L87-L96
      float id = 1.0f / (float)uvd(2);
      float r = (0.0f - id) * 255 / 1.0f;
      if (r < 0)
        r = -r;
      float g = (1.0f - id) * 255 / 1.0f;
      if (g < 0)
        g = -g;
      float b = (2.0f - id) * 255 / 1.0f;
      if (b < 0)
        b = -b;
      uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
      uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
      uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);
      cv::Scalar color(255 - rc, 255 - gc, 255 - bc);

      // Small square around the point (note the above bound check needs to take into account this width)
      cv::Point p0(uvd(0) - dw, uvd(1) - dw);
      cv::Point p1(uvd(0) + dw, uvd(1) + dw);
      cv::rectangle(depthmap_viz, p0, p1, color, -1);
    }

    // Create our messages
    header.frame_id = "cam0";
    sensor_msgs::msg::Image::SharedPtr exl_msg1 =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_16UC1, depthmap).toImageMsg();
    it_pub_loop_img_depth.publish(exl_msg1);
    header.stamp = _node->now();
    header.frame_id = "cam0";
    sensor_msgs::msg::Image::SharedPtr exl_msg2 = cv_bridge::CvImage(header, "bgr8", depthmap_viz).toImageMsg();
    it_pub_loop_img_depth_color.publish(exl_msg2);
  }
}

ROS2Visualizer::HeisenbergImgFileQueue
ROS2Visualizer::load_heisenberg_img_queue(const std::string& img_folder) {
  ifstream ifs;
  HeisenbergImgFileQueue file_queue;
  std::string ts_file = img_folder + ".timestamps.txt";
  ifs.open(ts_file.c_str());

  while(!ifs.eof())
  {
      string line;
      getline(ifs, line);
      if(!line.empty())
      {
          HeisenbergImgFile img_file;
          stringstream ss;
          ss << line;
          ss >> img_file.ts;
          
          // ss << line;
          img_file.full_path = img_folder + "/" + std::to_string(img_file.ts) + ".jpg";
          // std::cout << "load_heisenberg_img_queue: " << img_file.ts << ", " << img_file.full_path << std::endl;
          file_queue.push_back(std::move(img_file));
      }
  }
  return file_queue;
}

void ROS2Visualizer::load_heisenberg_img_queues(const std::string& dataset) {
  camera_name_to_img_queue.clear();
  camera_name_to_img_publisher.clear();
  image_transport::ImageTransport it(_node);
  std::string camera_name, topic;

  camera_name = "cam_front";
  topic = "/cam0/image_raw";
  camera_name_to_img_queue[camera_name] = load_heisenberg_img_queue(dataset + "/" + camera_name);
  camera_name_to_img_publisher[camera_name] = it.advertise(topic, 2);
  PRINT_DEBUG("Publishing: %s\n", camera_name_to_img_publisher[camera_name].getTopic().c_str());
}

void ROS2Visualizer::heisenberg_imu_hook_for_img_publishing(const sensor_msgs::msg::Imu::SharedPtr msg) {
  if (!_node->has_parameter("heisenberg_dataset")) {
    return;
  }
  msg->angular_velocity.x *= M_PI / 180.0;
  msg->angular_velocity.y *= M_PI / 180.0;
  msg->angular_velocity.z *= M_PI / 180.0;
  msg->linear_acceleration.x *= 9.8;
  msg->linear_acceleration.y *= 9.8;
  msg->linear_acceleration.z *= 9.8;

  int64_t imu_ts = msg->header.stamp.sec;
  imu_ts = imu_ts * 1000000000 + msg->header.stamp.nanosec;
  for (auto & item : camera_name_to_img_queue) {
    const std::string & camera_name = item.first;
    auto & img_queue = item.second;

    if (!img_queue.empty() && img_queue.front().ts <= imu_ts) {  // publish 1 img at most per imu callback
    // while (!img_queue.empty() && img_queue.front().ts <= imu_ts) {
      std::string& img_path = img_queue.front().full_path;
      cv::Mat img;
      // img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
      img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

      std_msgs::msg::Header header;
      header.stamp.sec = img_queue.front().ts / 1000000000;
      header.stamp.nanosec = img_queue.front().ts % 1000000000;
      header.frame_id = camera_name;
      // sensor_msgs::msg::Image::SharedPtr img_msg = cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
      sensor_msgs::msg::Image::SharedPtr img_msg = cv_bridge::CvImage(header, "mono8", img).toImageMsg();
      camera_name_to_img_publisher[camera_name].publish(img_msg);
      std::cout << "publish image for " << camera_name << "@" << img_queue.front().ts << std::endl;

      img_queue.pop_front();
    }
  }
}
