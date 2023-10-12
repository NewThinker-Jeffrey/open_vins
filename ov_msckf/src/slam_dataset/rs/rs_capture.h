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

  RsCapture(VisualSensorType type,
            bool capture_imu,
            std::function<void(int image_idx, CameraData msg)> image_cb,
            std::function<void(int imu_idx, ImuData msg)> imu_cb)
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

  virtual bool init_sersors();

  RsHelper* rs_;
};

} // namespace slam_dataset

#endif // SLAM_DATASET_RS_CAPTURE_H