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

#include "ROSVisualizerHelper.h"

#include "core/VioManager.h"
#include "sim/Simulator.h"
#include "state/State.h"
#include "state/StateHelper.h"

#include "types/PoseJPL.h"

using namespace ov_msckf;
using namespace std;

#if ROS_AVAILABLE == 1
sensor_msgs::PointCloud2 ROSVisualizerHelper::get_ros_pointcloud(const std::vector<Eigen::Vector3d> &feats) {

  // Declare message and sizes
  sensor_msgs::PointCloud2 cloud;
  cloud.header.frame_id = "global";
  cloud.header.stamp = ros::Time::now();
  cloud.width = 3 * feats.size();
  cloud.height = 1;
  cloud.is_bigendian = false;
  cloud.is_dense = false; // there may be invalid points

  // Setup pointcloud fields
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(3 * feats.size());

  // Iterators
  sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");

  // Fill our iterators
  for (const auto &pt : feats) {
    *out_x = (float)pt(0);
    ++out_x;
    *out_y = (float)pt(1);
    ++out_y;
    *out_z = (float)pt(2);
    ++out_z;
  }

  return cloud;
}

tf::StampedTransform ROSVisualizerHelper::get_stamped_transform_from_pose(const std::shared_ptr<ov_type::PoseJPL> &pose, bool flip_trans) {

  // Need to flip the transform to the IMU frame
  Eigen::Vector4d q_ItoC = pose->quat();
  Eigen::Vector3d p_CinI = pose->pos();
  if (flip_trans) {
    p_CinI = -pose->Rot().transpose() * pose->pos();
  }

  // publish our transform on TF
  // NOTE: since we use JPL we have an implicit conversion to Hamilton when we publish
  // NOTE: a rotation from ItoC in JPL has the same xyzw as a CtoI Hamilton rotation
  tf::StampedTransform trans;
  trans.stamp_ = ros::Time::now();
  tf::Quaternion quat(q_ItoC(0), q_ItoC(1), q_ItoC(2), q_ItoC(3));
  trans.setRotation(quat);
  tf::Vector3 orig(p_CinI(0), p_CinI(1), p_CinI(2));
  trans.setOrigin(orig);
  return trans;
}
#endif

#if ROS_AVAILABLE == 2
sensor_msgs::msg::PointCloud2 ROSVisualizerHelper::get_ros_pointcloud(std::shared_ptr<rclcpp::Node> node,
                                                                      const std::vector<Eigen::Vector3d> &feats) {

  // Declare message and sizes
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = "global";
  cloud.header.stamp = node->now();
  cloud.width = 3 * feats.size();
  cloud.height = 1;
  cloud.is_bigendian = false;
  cloud.is_dense = false; // there may be invalid points

  // Setup pointcloud fields
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(3 * feats.size());

  // Iterators
  sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");

  // Fill our iterators
  for (const auto &pt : feats) {
    *out_x = (float)pt(0);
    ++out_x;
    *out_y = (float)pt(1);
    ++out_y;
    *out_z = (float)pt(2);
    ++out_z;
  }

  return cloud;
}

geometry_msgs::msg::TransformStamped ROSVisualizerHelper::get_stamped_transform_from_pose(std::shared_ptr<rclcpp::Node> node,
                                                                                          const std::shared_ptr<ov_type::PoseJPL> &pose,
                                                                                          bool flip_trans) {

  // Need to flip the transform to the IMU frame
  Eigen::Vector4d q_ItoC = pose->quat();
  Eigen::Vector3d p_CinI = pose->pos();
  if (flip_trans) {
    p_CinI = -pose->Rot().transpose() * pose->pos();
  }

  // publish our transform on TF
  // NOTE: since we use JPL we have an implicit conversion to Hamilton when we publish
  // NOTE: a rotation from ItoC in JPL has the same xyzw as a CtoI Hamilton rotation
  geometry_msgs::msg::TransformStamped trans;
  trans.header.stamp = node->now();
  trans.transform.rotation.x = q_ItoC(0);
  trans.transform.rotation.y = q_ItoC(1);
  trans.transform.rotation.z = q_ItoC(2);
  trans.transform.rotation.w = q_ItoC(3);
  trans.transform.translation.x = p_CinI(0);
  trans.transform.translation.y = p_CinI(1);
  trans.transform.translation.z = p_CinI(2);
  return trans;
}
#endif
