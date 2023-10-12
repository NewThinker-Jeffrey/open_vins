#ifndef SLAM_DATASET_VI_CAPTURE_H
#define SLAM_DATASET_VI_CAPTURE_H

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "sensor_data.h"

namespace slam_dataset {

constexpr int LEFT_CAM_ID = 0;
constexpr int RIGHT_CAM_ID = 1;
constexpr int DEPTH_CAM_ID = 2
constexpr int COLOR_CAM_ID = 3;

/**
 * @brief Visual-Inertial dataset capture
 *
 * Subclasses of this interface class capture Visual-Inertial data
 * and invoke user-defined callbacks when new data frames become
 * available. 
 */
class ViCapture {

public:

  enum class VisualSensorType : uint64_t {
    NONE = 0,
    MONO = 1,
    STEREO = 2,
    DEPTH = 3,
    RGBD = 4
  };

  ViCapture(VisualSensorType type,
            bool capture_imu,
            std::function<void(int image_idx, CameraData msg)> image_cb,
            std::function<void(int imu_idx, ImuData msg)> imu_cb)
            :
            vsensor_type_(type),
            capture_imu_(capture_imu),
            image_cb_(image_cb),
            imu_cb_(imu_cb) {}

  virtual ~ViCapture() {}

  virtual bool startStreaming() = 0;

  virtual void stopStreaming() = 0;

  virtual bool isStreaming() const = 0;

  void waitStreamingOver(double polling_period_ms = 500);

  void setImageCallback(
      std::function<void(int image_idx, CameraData msg)> image_cb) {
    image_cb_ = image_cb;
  }

  void setImuCallback(
      std::function<void(int imu_idx, ImuData msg)> imu_cb) {
    imu_cb_ = imu_cb;
  }

  void setVisualSensorType(VisualSensorType type) {
    vsensor_type_ = type;
  }

protected:

  const VisualSensorType vsensor_type_;

  const bool capture_imu_;

  std::function<void(int image_idx, CameraData msg)> image_cb_;

  std::function<void(int imu_idx, ImuData msg)> imu_cb_;

};

} // namespace slam_dataset

#endif // SLAM_DATASET_VI_CAPTURE_H