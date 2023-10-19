#ifndef SLAM_VIZ_PANGOLIN_HELPER_H
#define SLAM_VIZ_PANGOLIN_HELPER_H

#include "pangolin_helper_types.h"

#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace slam_viz {

namespace pangolin_helper {


//// Fonts

// The loading might be slow.
bool loadChineseFont();

// NOTE:
//   If loadChineseFont() has not been called yet or it returned false,
//   then the default_font() of pangolin will be returned.
pangolin::GlFont* getChineseFont();


//// Draw text

void drawTextLine(
    const TextLine& line,
    const Eigen::Vector3f& translation = Eigen::Vector3f(0,0,0),
    const Eigen::Matrix3f& mult_rotation = Eigen::Matrix3f::Identity(),
    float scale = 1.0 / 36.0);

void drawTextLineFacingScreen(
    const TextLine& line,
    const Eigen::Vector3f& translation = Eigen::Vector3f(0,0,0),
    float scale = 1.0);

void drawTextLineInViewCoord(
    const TextLine& line,
    int start_pix_x,  int start_pix_y,
    float scale = 1.0);

void drawTextLineInWindowCoord(
    const TextLine& line,
    int start_pix_x,  int start_pix_y,
    float scale = 1.0);

void drawMultiTextLines(const TextLines& lines);

void drawMultiTextLines(
    const TextLines& lines,
    const Eigen::Vector3f& translation = Eigen::Vector3f(0,0,0),
    const Eigen::Matrix3f& mult_rotation = Eigen::Matrix3f::Identity(),
    float scale = 1.0 / 36.0);

void drawMultiTextLinesFacingScreen(
    const TextLines& lines,
    const Eigen::Vector3f& translation = Eigen::Vector3f(0,0,0),
    float scale = 1.0);

void drawMultiTextLinesInViewCoord(
    const TextLines& lines,
    int start_pix_x,  int start_pix_y,
    float scale = 1.0);

void drawMultiTextLinesInWindowCoord(
    const TextLines& lines,
    int start_pix_x,  int start_pix_y,
    float scale = 1.0);


//// Draw images

void drawCvImageOnView(
    const cv::Mat& img_in,
    pangolin::View& view,  // `view` will be activated in the call
    bool need_bgr2rgb = false);


//// Make 4x4 Matrix
//   Matrices used by OpenGL are in column-major order, and luckily,
//   Eigen use the same order by default. So it's ok to write code like
//   `glMultMatirxf (eigen_float_mat.data());`.

Eigen::Matrix4f makeMatrixf(
    const Eigen::Vector3f& translation = Eigen::Vector3f(0,0,0),
    const Eigen::Matrix3f& rotation = Eigen::Matrix3f::Identity(),
    float scale = 1.0);

void multMatrixfAndDraw(
    const Eigen::Matrix4f& matrix,
    std::function<void()> do_draw);

pangolin::OpenGlMatrix
makeGlMatrix(const Eigen::Matrix4f& eigen_float_mat);



//// Draw objects

void drawFrame(
    float axis_len = 1.0,
    float line_width = 4.0,
    uint8_t alpha = 255);

void drawGrids2D(
    int center_x = 0, int center_y = 0,
    int n_x = 20, int n_y = 20,
    Color c = Color(255, 255, 255, 40),
    float line_width = 1.0f);

// void drawGrids3D() ???? 

// void drawChessBoard() ???? 

void drawCamera(
    double s = 1.0,
    Color c = Color(0,0,255,255),
    float line_width = 1.0f);

void drawVehicle(  // A triangle
    double s = 1.0,
    const Eigen::Vector3f& front = Eigen::Vector3f(1, 0, 0),
    const Eigen::Vector3f& up = Eigen::Vector3f(0, 0, 1),
    Color c = Color(255,255,0,80),
    float line_width = 4.0f);


template <class PointTrajectory>
void drawPointTrajectory(
    const PointTrajectory& traj,
    Color c = Color(255,0,0,127),
    float line_width = 4.0f) {

  glLineWidth(line_width);
  glColor4ub(c.r, c.g, c.b, c.a);
  glBegin(GL_LINE_STRIP);
  for (const auto& p : traj) {
    glVertex3f(p.x(), p.y(), p.z());
  }
  glEnd();
}

template <class PointPtrTrajectory>
void drawPointTrajectory2(
    const PointPtrTrajectory& traj,
    Color c = Color(255,0,0,127),
    float line_width = 4.0f) {

  glLineWidth(line_width);
  glColor4ub(c.r, c.g, c.b, c.a);
  glBegin(GL_LINE_STRIP);
  for (const auto& p : traj) {
    glVertex3f(p->x(), p->y(), p->z());
  }
  glEnd();
}

template <class PointSet>
void drawPointCloud(
    const PointSet& point_cloud,
    Color c = Color(255,255,255,80),
    float point_size = 3.0) {

  glPointSize(point_size);
  glBegin(GL_POINTS);

  glColor4ub(c.r, c.g, c.b, c.a);
  for (const auto& p : point_cloud) {
    glVertex3f(p.x(), p.y(), p.z());
  }
  glEnd();
}

template <class PointPtrSet>
void drawPointCloud2(
    const PointPtrSet& point_cloud,
    Color c = Color(255,255,255,80),
    float point_size = 3.0) {

  glPointSize(point_size);
  glBegin(GL_POINTS);

  glColor4ub(c.r, c.g, c.b, c.a);
  for (const auto& p : point_cloud) {
    glVertex3f(p->x(), p->y(), p->z());
  }
  glEnd();
}

} // namespace pangolin_helper

} // namespace slam_viz


#endif // SLAM_VIZ_PANGOLIN_HELPER_H
