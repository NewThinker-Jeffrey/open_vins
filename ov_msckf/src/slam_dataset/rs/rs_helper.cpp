#ifndef USE_HEAR_SLAM

#include "rs_helper.h"

#include <iostream>

namespace slam_dataset {

rs2_option RsHelper::getSensorOption(const rs2::sensor& sensor) {
  // Sensors usually have several options to control their properties
  // such as Exposure, Brightness etc.
  std::cout << "Sensor supports the following options:\n" << std::endl;

  // The following loop shows how to iterate over all available options.
  // Starting from 0 until RS2_OPTION_COUNT (exclusive)
  for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++) {
    rs2_option option_type = static_cast<rs2_option>(i);

    // To control an option, use the following api:
    // First, verify that the sensor actually supports this option
    if (sensor.supports(option_type)) {
      // SDK enum types can be streamed to get a string that represents them
      std::cout << "  " << i << ": " << option_type;
      std::cout << std::endl;

      // Get a human readable description of the option
      const char* description = sensor.get_option_description(option_type);
      std::cout << "       Description   : " << description << std::endl;

      // Get the current value of the option
      float current_value = sensor.get_option(option_type);
      std::cout << "       Current Value : " << current_value << std::endl;

      // To change the value of an option, please follow the
      // change_sensor_option() function
    } else {
        // //SDK enum types can be streamed to get a string that represents them
        // std::cout << "  " << i << ": " << option_type;
        // std::cout << " is not supported" << std::endl;
    }
  }

  uint32_t selected_sensor_option = 0;
  return static_cast<rs2_option>(selected_sensor_option);
}


rs2_stream RsHelper::findStreamToAlign(const std::vector<rs2::stream_profile>& streams) {
  // Given a vector of streams, we try to find a depth stream and
  // another stream to align depth with.
  // We prioritize color streams to make the view look better.
  // If color is not available, we take another stream that (other than depth)
  rs2_stream align_to = RS2_STREAM_ANY;
  bool depth_stream_found = false;
  bool color_stream_found = false;
  for (rs2::stream_profile sp : streams) {
    rs2_stream profile_stream = sp.stream_type();
    if (profile_stream != RS2_STREAM_DEPTH) {
      if (!color_stream_found) {
        // Prefer color
        align_to = profile_stream;
      }
      if (profile_stream == RS2_STREAM_COLOR) {
        color_stream_found = true;
      }
    } else {
      depth_stream_found = true;
    }
  }

  if(!depth_stream_found) {
    throw std::runtime_error("No Depth stream available");
  }
  if (align_to == RS2_STREAM_ANY) {
    throw std::runtime_error("No stream found to align with Depth");
  }
  return align_to;
}

bool RsHelper::profileChanged(
    const std::vector<rs2::stream_profile>& current,
    const std::vector<rs2::stream_profile>& prev) {
  for (auto&& sp : prev) {
    //If previous profile is in current (maybe just added another)
    auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
    if (itr == std::end(current)) {
      //If it previous stream wasn't found in current
      return true;
    }
  }
  return false;
}

rs2_vector RsHelper::interpolateMeasure(
    const double target_time,
    const rs2_vector current_data, const double current_time,
    const rs2_vector prev_data, const double prev_time) {

  // If there are not previous information, the current data is propagated
  if(prev_time == 0) {
    return current_data;
  }

  rs2_vector increment;
  rs2_vector value_interp;

  if(target_time > current_time) {
    value_interp = current_data;
  } else if (target_time > prev_time) {
    increment.x = current_data.x - prev_data.x;
    increment.y = current_data.y - prev_data.y;
    increment.z = current_data.z - prev_data.z;

    double factor = (target_time - prev_time) / (current_time - prev_time);

    value_interp.x = prev_data.x + increment.x * factor;
    value_interp.y = prev_data.y + increment.y * factor;
    value_interp.z = prev_data.z + increment.z * factor;

    // zero interpolation
    value_interp = current_data;
  } else {
    value_interp = prev_data;
  }

  return value_interp;
}

Eigen::Isometry3d RsHelper::getExtrinsics(
    rs2::pipeline_profile& pipe_profile,
    rs2_stream from_cam_type, rs2_stream to_cam_type,
    int from_cam_index, int to_cam_index,
    bool print) {
  auto from_stream = pipe_profile.get_stream(from_cam_type, from_cam_index);
  auto to_stream = pipe_profile.get_stream(to_cam_type, to_cam_index);
  return getExtrinsics(from_stream, to_stream, print);
}

Eigen::Isometry3d RsHelper::getExtrinsics(
    rs2::stream_profile& from_stream,
    rs2::stream_profile& to_stream, bool print) {

  Eigen::Isometry3f ext_f = Eigen::Isometry3f::Identity();

  float* R = from_stream.get_extrinsics_to(to_stream).rotation;
  float* t = from_stream.get_extrinsics_to(to_stream).translation;
  ext_f.linear() = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R);
  ext_f.translation() = Eigen::Map<Eigen::Matrix<float, 3, 1>>(t);
  Eigen::Isometry3d ext = ext_f.cast<double>();
  if (print) {
    std::cout << "** RsHelper: Extrinsics from '" << from_stream.stream_name()
              << "' to '" << to_stream.stream_name() << "':" << std::endl;
    std::cout << ext.matrix() << std::endl;
    // for(int i = 0; i<3; i++){
    //     for(int j = 0; j<3; j++)
    //         std::cout << R[i*3 + j] << ", ";
    //     std::cout << t[i] << "\n";
    // }
  }

  return ext;
}

rs2_intrinsics RsHelper::getCameraIntrinsics(
    rs2::pipeline_profile& pipe_profile,
    rs2_stream cam_type, int cam_index,
    bool print) {
  auto cam_stream = pipe_profile.get_stream(cam_type, cam_index);
  return getCameraIntrinsics(cam_stream, print);
}

rs2_intrinsics RsHelper::getCameraIntrinsics(rs2::stream_profile& cam_stream, bool print) {
  rs2_intrinsics intrinsics_cam = cam_stream.as<rs2::video_stream_profile>().get_intrinsics();
  if (print) {
    std::cout << "** RsHelper: Intrinsics of '" << cam_stream.stream_name() << "':" << std::endl;
    std::cout << " height = " << intrinsics_cam.height << std::endl;
    std::cout << " width = " << intrinsics_cam.width << std::endl;
    std::cout << " fx = " << intrinsics_cam.fx << std::endl;
    std::cout << " fy = " << intrinsics_cam.fy << std::endl;
    std::cout << " cx = " << intrinsics_cam.ppx << std::endl;
    std::cout << " cy = " << intrinsics_cam.ppy << std::endl;
    std::cout << " DistortionModel = " << intrinsics_cam.model << std::endl;
    std::cout << " Coeff = "
              << intrinsics_cam.coeffs[0] << ", "
              << intrinsics_cam.coeffs[1] << ", "
              << intrinsics_cam.coeffs[2] << ", "
              << intrinsics_cam.coeffs[3] << ", "
              << intrinsics_cam.coeffs[4] << ", "
              << std::endl;
  }
  return intrinsics_cam;
}


} // namespace slam_dataset


#endif // USE_HEAR_SLAM
