#ifndef SLAM_DATASET_VI_RECORDER_H
#define SLAM_DATASET_VI_RECORDER_H

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

#include "vi_capture.h"

namespace slam_dataset {

class ViRecorder {

public:

  using VisualSensorType = ViCapture::VisualSensorType;

  ViRecorder(VisualSensorType type,
             bool record_imu)
            :
            vsensor_type_(type),
            record_imu_(record_imu) {}

  virtual ~ViRecorder() {}

  bool startRecord(const std::string& save_folder = "");

  void stopRecord();

  void feedCameraData(CameraData cam);

  void feedImuData(ImuData imu);

protected:

  const VisualSensorType vsensor_type_;

  const bool record_imu_;

  struct Context;

  std::shared_ptr<Context> c_;
};

} // namespace slam_dataset

#endif // SLAM_DATASET_VI_RECORDER_H