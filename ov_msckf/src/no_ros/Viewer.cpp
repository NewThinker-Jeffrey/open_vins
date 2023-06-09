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


#if ! OPENVINS_FOR_TROS

#include "Viewer.h"
#include <pangolin/pangolin.h>

#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

extern std::shared_ptr<ov_msckf::VioManager> getVioManagerFromVioInterface(heisenberg_algo::VIO*);

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

const double viewpoint_height = 10.0;

Viewer::Viewer(std::shared_ptr<heisenberg_algo::VIO> app) : _app(app) {

}

void Viewer::init() {
  pangolin::CreateWindowAndBind("Heisenberg Robotics VIO Demo",1024,768);
#if 0
  pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",false,true);
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
          .SetHandler(new pangolin::Handler3D(s_cam));
#endif 

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  // Issue specific OpenGl we might need
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(640,480,320,320,320,240,0.1,1000);
  // pangolin::OpenGlRenderState s_cam(proj, pangolin::ModelViewLookAt(1,0.5,-2,0,0,0, pangolin::AxisY) );
  s_cam = std::make_shared<pangolin::OpenGlRenderState>(proj, pangolin::ModelViewLookAt(0, 0, viewpoint_height, 0, 0, 0, pangolin::AxisY));

  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam1 = pangolin::Display("cam1")
    .SetAspect(640.0f/480.0f)
    .SetHandler(new pangolin::Handler3D(*s_cam))
    .SetBounds(0.0, 1.0, 0.0, 0.5);

  pangolin::View& d_feat_track_img = pangolin::Display("feature_tracking")
    .SetAspect(640.0f/480.0f)
    .SetBounds(0.0, 1.0, 0.5, 1.0);
}

void Viewer::show(std::shared_ptr<VioManager::Output> output) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  auto internal_app = getVioManagerFromVioInterface(_app.get());
  cv::Mat img_history = internal_app->get_historical_viz_image(output);
  cv::Mat flipped_img_history;
  cv::cvtColor(img_history, flipped_img_history, cv::COLOR_BGR2RGB);
  cv::flip(flipped_img_history, flipped_img_history, 0);
  pangolin::GlTexture imageTexture(img_history.cols, img_history.rows, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
  imageTexture.Upload(flipped_img_history.ptr<uchar>(), GL_RGB, GL_UNSIGNED_BYTE);

  pangolin::Display("feature_tracking").SetAspect(img_history.cols/(float)img_history.rows);
  pangolin::Display("feature_tracking").Activate();
  glColor4f(1.0f,1.0f,1.0f,1.0f);
  imageTexture.RenderToViewport();

  Eigen::Vector3f new_pos = output->state_clone->_imu->pos().cast<float>();
  Eigen::Isometry3f imu_pose(output->state_clone->_imu->Rot().inverse().cast<float>());
  imu_pose.translation() = new_pos;
  Eigen::Matrix4f imu_pose_mat = imu_pose.matrix();  // column major
  pangolin::OpenGlMatrix Twi;
  for (int i = 0; i<4; i++) {
    Twi.m[4*i] = imu_pose_mat(0,i);
    Twi.m[4*i+1] = imu_pose_mat(1,i);
    Twi.m[4*i+2] = imu_pose_mat(2,i);
    Twi.m[4*i+3] = imu_pose_mat(3,i);
  }

  s_cam->SetModelViewMatrix(pangolin::ModelViewLookAt(new_pos(0), new_pos(1), viewpoint_height, new_pos(0), new_pos(1), 0, pangolin::AxisY));
  // s_cam->Follow(Twi);

  pangolin::Display("cam1").Activate(*s_cam);
  // pangolin::glDrawColouredCube();

  // draw grid
  int x_begin = new_pos(0) - 2 * viewpoint_height;
  int y_begin = new_pos(1) - 2 * viewpoint_height;
  int n_lines = 4 * viewpoint_height;
  glLineWidth(1.0);
  glColor4f(1.0f, 1.0f, 1.0f, 0.15f);
  glBegin(GL_LINES);
  for (int i=0; i<n_lines; i++) {
    // draw the ith vertical line
    glVertex3f(x_begin + i, y_begin, 0);
    glVertex3f(x_begin + i, y_begin + n_lines, 0);
    // draw the ith horizontal line
    glVertex3f(x_begin, y_begin + i, 0);
    glVertex3f(x_begin + n_lines, y_begin + i, 0);
  }
  glEnd();

  // Return if we have not inited
  if (!output->status.initialized) {
    pangolin::FinishFrame();
    return;
  }

  // draw points
  std::set<size_t> new_slam_ids;
  std::set<size_t> new_msckf_ids;
  std::set<size_t> new_ids;
  for (size_t i=0; i<output->visualization.feature_ids_SLAM.size(); i++) {
    auto id = output->visualization.feature_ids_SLAM[i];
    auto p = output->visualization.features_SLAM[i];
    _map_points[id] = p.cast<float>();
    new_slam_ids.insert(id);
    new_ids.insert(id);
  }
  for (size_t i=0; i<output->visualization.good_feature_ids_MSCKF.size(); i++) {
    auto id = output->visualization.good_feature_ids_MSCKF[i];
    auto p = output->visualization.good_features_MSCKF[i];
    _map_points[id] = p.cast<float>();
    new_msckf_ids.insert(id);
    new_ids.insert(id);
  }

  glPointSize(3.0);
  glBegin(GL_POINTS);
  glColor4f(0.5, 0.5, 0.5, 0.3f);
  for (const auto& item : _map_points) {
    if (new_ids.count(item.first) == 0) {
      const Eigen::Vector3f& p = item.second;
      glVertex3f(p(0), p(1), p(2));
    }
  }

  glColor4f(1.0, 1.0, 1.0, 0.3f);
  for (const auto& item : output->visualization.active_tracks_posinG) {
    if (new_ids.count(item.first) == 0) {
      Eigen::Vector3f p = item.second.cast<float>();
      glVertex3f(p(0), p(1), p(2));
    }
  }

  glEnd();
  glPointSize(5.0);
  glBegin(GL_POINTS);

  glColor3f(1.0, 0.0, 0.0);
  for (auto id : new_slam_ids) {
    Eigen::Vector3f& p = _map_points.at(id);
    glVertex3f(p(0), p(1), p(2));
  }

  glColor3f(0.0, 0.0, 1.0);
  for (auto id : new_msckf_ids) {
    Eigen::Vector3f& p = _map_points.at(id);
    glVertex3f(p(0), p(1), p(2));
  }
  glEnd();


  // draw traj
  _traj.push_back(new_pos);

  glLineWidth(4.0);
  glColor4f(1.0f, 0.0f, 0.0f, 0.3f);
  glBegin(GL_LINE_STRIP);
  for (const auto& p : _traj) {
    glVertex3f(p(0), p(1), p(2));
  }
  glEnd();


  // draw imu frame
  glPushMatrix();
  glMultMatrixf(imu_pose_mat.data());

  glLineWidth(4.0);
  glColor4f(1.0f, 1.0f, 0.0f, 0.3f);
  glBegin(GL_LINE_LOOP);
  double half_w = 0.15;
  double l = 0.40;
  glVertex3f(-half_w, 0, -l);
  glVertex3f( half_w, 0, -l);
  glVertex3f(      0, 0,  0);
  glEnd();
  glPopMatrix();


  pangolin::FinishFrame();
}


#endif