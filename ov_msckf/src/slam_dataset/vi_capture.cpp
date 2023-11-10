#ifndef USE_HEAR_SLAM

#include "vi_capture.h"

#include <thread>

namespace slam_dataset {

void ViCapture::waitStreamingOver(int polling_period_ms) {
  while (isStreaming()) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(polling_period_ms));
  }
}

void ViCapture::registerImageCallback(CameraCallback image_cb) {
  std::unique_lock<std::mutex> locker(mutex_image_cbs_);
  image_cbs_.push_back(image_cb);
}

void ViCapture::registerRoImageCallback(RoCameraCallback ro_image_cb) {
  std::unique_lock<std::mutex> locker(mutex_image_cbs_);
  ro_image_cbs_.push_back(ro_image_cb);
}

void ViCapture::registerImuCallback(ImuCallback imu_cb) {
  std::unique_lock<std::mutex> locker(mutex_imu_cbs_);
  imu_cbs_.push_back(imu_cb);
}

void ViCapture::changeVisualSensorType(VisualSensorType type) {
  vsensor_type_ = type;
}

void ViCapture::enableImu(bool enable) {
  capture_imu_ = enable;
}

void ViCapture::runImageCallbacks(int image_idx, CameraData&& msg) {
  std::unique_lock<std::mutex> locker(mutex_image_cbs_);
  // std::cout << "runImageCallbacks: ro(" << ro_image_cbs_.size() << "), normal(" << image_cbs_.size() << ")" << std::endl;
  for (auto & cb : ro_image_cbs_) {
    cb(image_idx, msg);
  }

  for (size_t i=0; i<image_cbs_.size(); i++) {
    auto & cb = image_cbs_[i];
    if (i == image_cbs_.size() - 1) {
      // For the last callback, we can use the move-assignement
      cb(image_idx, std::move(msg));
    } else {
      cb(image_idx, msg);
    }
  }
}

void ViCapture::runImuCallbacks(int imu_idx, ImuData&& msg) {
  std::unique_lock<std::mutex> locker(mutex_imu_cbs_);
  // std::cout << "runImuCallbacks: (" << imu_cbs_.size() << ")" << std::endl;
  for (auto & cb : imu_cbs_) {
    cb(imu_idx, msg);
  }
}


}  // namespace slam_dataset

#endif // USE_HEAR_SLAM
