#ifndef SLAM_DATASET_VI_RECORDER_H
#define SLAM_DATASET_VI_RECORDER_H

#include "sensor_data.h"

#ifndef USE_HEAR_SLAM

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

  ViRecorder() : vsensor_type_(VisualSensorType::NONE),
                 record_imu_(true), c_(nullptr) {}

  virtual ~ViRecorder() {}

  // If `capture` is not set, you should manually call the "feedXXX()"
  // methods (see beblow) to record the data frames.
  bool startRecord(
      ViCapture* capture = nullptr, const std::string& save_folder = "",
       // when `capture` is not set, manually specify sensor settings
      VisualSensorType type = VisualSensorType::NONE, bool record_imu = true,
      const std::string& thread_name="slam_record");
      // thread_name should consist of less than 15 characters

  void stopRecord();

  void enableImageWindow();

  void disableImageWindow();

public:

  void feedCameraData(CameraData cam);

  void feedImuData(ImuData imu);

protected:

  VisualSensorType vsensor_type_;

  bool record_imu_;

  struct Context;

  std::shared_ptr<Context> c_;
};

} // namespace slam_dataset

#endif // USE_HEAR_SLAM

#endif // SLAM_DATASET_VI_RECORDER_H