#ifndef SLAM_VIZ_PANGOLIN_HELPER_H
#define SLAM_VIZ_PANGOLIN_HELPER_H

#include <stdint.h>
#include <vector>
#include <string>
#include <memory>

#include <Eigen/Geometry>
#include <pangolin/pangolin.h>
#include <pangolin/display/default_font.h>

namespace slam_viz {

namespace pangolin_helper {


//// Draw poses

void drawTrianglePose();  // todo



//// Fonts

// The loading might be slow.
bool loadChineseFont();

// NOTE:
//   If loadChineseFont() has not been called yet or it returned false,
//   then the default_font() of pangolin will be returned.
pangolin::GlFont* getChineseFont();



//// Draw text

struct TextLine {
  std::string text;
  uint8_t color[4];  // in rgba order.
  pangolin::GlFont* font;

  TextLine(const std::string& str = "",
           bool hightlight = false,
           pangolin::GlFont* font = &pangolin::default_font());

  TextLine(
      const std::string& str,
      uint8_t r,
      uint8_t g,
      uint8_t b,
      uint8_t a = 255,
      pangolin::GlFont* font = &pangolin::default_font());
};

using TextLines = std::vector<TextLine>;

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

} // namespace pangolin_helper

} // namespace slam_viz


#endif // SLAM_VIZ_PANGOLIN_HELPER_H
