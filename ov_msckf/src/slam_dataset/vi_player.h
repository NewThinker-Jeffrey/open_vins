#ifndef SLAM_DATASET_VI_PLYAER_H
#define SLAM_DATASET_VI_PLYAER_H

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <functional>

#include "vi_capture.h"

namespace slam_dataset {

class ViPlayer : public ViCapture {

public:

  using VisualSensorType = ViCapture::VisualSensorType;

  ViPlayer(const std::string& dataset,
           VisualSensorType type,
           bool capture_imu,
           std::function<void(int image_idx, CameraData msg)> image_cb,
           std::function<void(int imu_idx, ImuData msg)> imu_cb,
           double play_rate = 1.0) 
             :
           ViCapture(type, capture_imu, image_cb, imu_cb),
           dataset_(dataset),
           play_rate_(play_rate)
           {}

  virtual ~ViPlayer();

  virtual bool startStreaming();

  virtual void stopStreaming();

  virtual bool isStreaming() const;

  // double sensorStartTime() {
  //   return sensor_start_time_;
  // }

  // std::chrono::high_resolution_clock::time_point playStartTime() {
  //   return play_start_time_;
  // }

protected:

  void loadDataset();

  void dataPlay();

protected:
  std::string dataset_;

  std::vector<ImuData> imu_data_;

  std::vector<double> image_times_;

  using ImageFileList = std::vector<std::string>;

  std::map<std::string, ImageFileList> camname_to_imgfiles_;
  // camname: "left"  "right"  "color"  "depth"

  std::shared_ptr<std::thread> data_play_thread_;
  std::atomic<bool> stop_request_;

  double sensor_start_time_;
  std::chrono::high_resolution_clock::time_point play_start_time_;

  double play_rate_;
};

} // namespace slam_dataset

#endif // SLAM_DATASET_VI_RECORDER_H