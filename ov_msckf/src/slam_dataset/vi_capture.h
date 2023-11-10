#ifndef SLAM_DATASET_VI_CAPTURE_H
#define SLAM_DATASET_VI_CAPTURE_H

#include "sensor_data.h"

#ifndef USE_HEAR_SLAM

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>

namespace slam_dataset {

constexpr int LEFT_CAM_ID = 0;
constexpr int RIGHT_CAM_ID = 1;
constexpr int DEPTH_CAM_ID = 2;
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
    RGBD = 4,
  };

  using CameraCallback = std::function<void(int image_idx, CameraData msg)>;

  using RoCameraCallback = std::function<void(int image_idx, const CameraData& msg)>;

  using ImuCallback = std::function<void(int imu_idx, ImuData msg)>;

  ViCapture(VisualSensorType type = VisualSensorType::STEREO,
            bool capture_imu = true,
            CameraCallback image_cb = nullptr,
            ImuCallback imu_cb = nullptr)
            :
            vsensor_type_(type),
            capture_imu_(capture_imu) {
    if (image_cb) {registerImageCallback(image_cb);}  // NOLINT
    if (imu_cb) {registerImuCallback(imu_cb);}  // NOLINT
  }

  virtual ~ViCapture() {}

  virtual bool startStreaming() = 0;

  virtual void stopStreaming() = 0;

  virtual bool isStreaming() const = 0;

  void waitStreamingOver(int polling_period_ms = 500);

public:

  // Add new callbacks (thread-safe).
  //
  // Keep in mind that all callbacks should be quick so that the internal mutex
  // could be released ASAP and new-coming data frames can be processed in time.
  //
  // Remember NOT to register new callbacks inside a callback!! Because that will
  // cause a dead-lock!!

  void registerImageCallback(CameraCallback image_cb);

  void registerRoImageCallback(RoCameraCallback ro_image_cb);

  void registerImuCallback(ImuCallback imu_cb);

public:

  // Methods to change settings.
  //
  // Note these methods shouldn't be used anymore after the streaming is started.

  void changeVisualSensorType(VisualSensorType type);

  void enableImu(bool enable = true);



  // Read settings.

  VisualSensorType getVisualSensorType() const {return vsensor_type_;}  // NOLINT

  bool isImuEnabled() const {return capture_imu_;}  // NOLINT


protected:

  void runImageCallbacks(int image_idx, CameraData&& msg);

  void runImuCallbacks(int imu_idx, ImuData&& msg);

protected:

  // sensor settings

  VisualSensorType vsensor_type_;

  bool capture_imu_;


  // image callbacks

  std::mutex mutex_image_cbs_;

  std::vector<CameraCallback> image_cbs_;

  std::vector<RoCameraCallback> ro_image_cbs_;

  // imu callbacks

  std::mutex mutex_imu_cbs_;

  std::vector<ImuCallback> imu_cbs_;

};

} // namespace slam_dataset

#endif // USE_HEAR_SLAM

#endif // SLAM_DATASET_VI_CAPTURE_H