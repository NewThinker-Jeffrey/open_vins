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

  RsCapture(VisualSensorType type = VisualSensorType::STEREO,
            bool capture_imu = true,
            CameraCallback image_cb = nullptr,
            ImuCallback imu_cb = nullptr)
            // const std::string& device_name = "",
            // const std::string& config_file = "");
              :
            ViCapture(type, capture_imu, image_cb, imu_cb),
            rs_(nullptr) {}

  virtual ~RsCapture();

  virtual bool startStreaming();

  virtual void stopStreaming();

  virtual bool isStreaming() const;

protected:

  virtual bool initSersors();

  std::shared_ptr<RsHelper> rs_;
};

} // namespace slam_dataset

#endif // SLAM_DATASET_RS_CAPTURE_H