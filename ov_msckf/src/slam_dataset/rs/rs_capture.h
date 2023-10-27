#ifndef SLAM_DATASET_RS_CAPTURE_H
#define SLAM_DATASET_RS_CAPTURE_H

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../vi_capture.h"

namespace slam_dataset {

struct RsHelper;

/**
 * @brief Realsense Visual-Inertial dataset capture
 *
 * This class captures Visual-Inertial data and invoke
 * user-defined callbacks when new data frames become available. 
 */
class RsCapture : public ViCapture {

public:

  static constexpr int kDefaultImageFramerate = 30;
  static constexpr int kDefaultImageWidth = 640;
  static constexpr int kDefaultImageHeight = 480;

  struct BasicSettings {

    int infra_framerate = kDefaultImageFramerate;
    int infra_width     = kDefaultImageWidth;
    int infra_height    = kDefaultImageHeight;

    int depth_framerate = kDefaultImageFramerate;
    int depth_width     = kDefaultImageWidth;
    int depth_height    = kDefaultImageHeight;

    int color_framerate = kDefaultImageFramerate;
    int color_width     = kDefaultImageWidth;
    int color_height    = kDefaultImageHeight;

    int gyro_framerate  = 0;  // 0 for sensor's default rate
    int accel_framerate = 0;  // 0 for sensor's default rate

    BasicSettings() {}

  };

  RsCapture(VisualSensorType type = VisualSensorType::STEREO,
            bool capture_imu = true,
            CameraCallback image_cb = nullptr,
            ImuCallback imu_cb = nullptr,
            const BasicSettings& basic_settings = BasicSettings())
            // const std::string& device_name = "",
            // const std::string& config_file = "");
              :
            ViCapture(type, capture_imu, image_cb, imu_cb),
            rs_(nullptr), bs_(basic_settings) {}

  virtual ~RsCapture();

  virtual bool startStreaming();

  virtual void stopStreaming();

  virtual bool isStreaming() const;

protected:

  virtual bool initSersors();

  std::shared_ptr<RsHelper> rs_;

  BasicSettings bs_;
};

} // namespace slam_dataset

#endif // SLAM_DATASET_RS_CAPTURE_H