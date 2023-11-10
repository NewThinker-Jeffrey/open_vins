#ifndef SLAM_VIZ_PANGOLIN_HELPER_TYPES_H
#define SLAM_VIZ_PANGOLIN_HELPER_TYPES_H

#ifdef USE_HEAR_SLAM

#include "hear_slam/viz/pangolin_helper.h"

namespace slam_viz {
  using namespace hear_slam;
}

#else

#include <stdint.h>
#include <string>
#include <deque>
#include <Eigen/Geometry>
#include <pangolin/pangolin.h>
#include <pangolin/display/default_font.h>

namespace slam_viz {

namespace pangolin_helper {

//// Color

struct Color {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
  Color(uint8_t _r=0, uint8_t _g=0, uint8_t _b=0, uint8_t _a=255) :
      r(_r), g(_g), b(_b), a(_a) {}
};


//// Text line

struct TextLine {
  std::string text;
  Color c;
  pangolin::GlFont* font;

  TextLine(const std::string& str = "",
           bool hightlight = false,
           pangolin::GlFont* font = &pangolin::default_font());

  TextLine(
      const std::string& str,
      Color c,
      pangolin::GlFont* font = &pangolin::default_font());
};

using TextLines = std::vector<TextLine>;


//// Trajectory

using PointfTrajectory = std::deque<Eigen::Vector3f>;

using PointfPtrTrajectory = std::deque<const Eigen::Vector3f*>;

using PointdTrajectory = std::deque<Eigen::Vector3d>;

using PointdPtrTrajectory = std::deque<const Eigen::Vector3d*>;


//// Point cloud

using PointfSet = std::deque<Eigen::Vector3f>;

using PointfPtrSet = std::deque<const Eigen::Vector3f*>;

using PointdSet = std::deque<Eigen::Vector3d>;

using PointdPtrSet = std::deque<const Eigen::Vector3d*>;


} // namespace pangolin_helper

} // namespace slam_viz

#endif // USE_HEAR_SLAM

#endif // SLAM_VIZ_PANGOLIN_HELPER_TYPES_H
