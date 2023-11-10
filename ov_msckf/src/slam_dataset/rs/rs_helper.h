#ifndef SLAM_DATASET_RS_HELPER_H
#define SLAM_DATASET_RS_HELPER_H

#ifndef USE_HEAR_SLAM

#include <memory>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <Eigen/Geometry>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

namespace slam_dataset {

struct RsHelper {

  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;

  // Create a configuration for configuring the pipeline with a non default profile
  rs2::config cfg;

  rs2::pipeline_profile pipe_profile;


  //// frame (imu & image) process
  std::mutex frame_mutex;
  double first_frame_time = -1;

  // image process
  std::condition_variable cond_image_rec;
  double timestamp_image = -1.0;
  bool image_ready = false;
  std::shared_ptr<rs2::frameset> fs;
  int count_im_buffer = 0;
  std::shared_ptr<std::thread> image_process_thread;
  std::atomic<bool> image_process_thread_stop_request;
  int image_count = 0;
  int image_framerate = 1;

  // RGBD specified
  rs2_stream align_to;  // rs2_stream is a enum type.
  std::shared_ptr<rs2::align> align;

  // imu sync
  std::deque<std::pair<double, rs2_vector>> gyro_wait_list;
  std::deque<std::pair<double, rs2_vector>> acc_list;  // size <= 2
  int imu_count = 0;
  int acc_count = 0;
  int gyro_count = 0;


  static rs2_option getSensorOption(const rs2::sensor& sensor);

  static rs2_stream findStreamToAlign(const std::vector<rs2::stream_profile>& streams);

  static bool profileChanged(
      const std::vector<rs2::stream_profile>& current,
      const std::vector<rs2::stream_profile>& prev);

  static rs2_vector interpolateMeasure(
      const double target_time,
      const rs2_vector current_data, const double current_time,
      const rs2_vector prev_data, const double prev_time);

  static Eigen::Isometry3d getExtrinsics(
      rs2::pipeline_profile& pipe_profile,
      rs2_stream from_cam_type, rs2_stream to_cam_type,
      int from_cam_index = -1, int to_cam_index = -1,  // -1 for any matching.
      bool print = false);

  static Eigen::Isometry3d getExtrinsics(
      rs2::stream_profile& from_stream,
      rs2::stream_profile& to_stream,
      bool print = false);

  static rs2_intrinsics getCameraIntrinsics(
      rs2::pipeline_profile& pipe_profile,
      rs2_stream cam_type,
      int cam_index = -1,  // -1 for any matching.
      bool print = false);

  static rs2_intrinsics getCameraIntrinsics(
      rs2::stream_profile& cam_stream,
      bool print = false);
};

} // namespace slam_dataset

#endif // USE_HEAR_SLAM

#endif // SLAM_DATASET_RS_HELPER_H