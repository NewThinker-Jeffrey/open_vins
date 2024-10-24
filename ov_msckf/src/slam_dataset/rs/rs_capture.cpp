#ifndef USE_HEAR_SLAM

#include "rs_capture.h"
#include "rs_helper.h"

#include <iostream>

namespace slam_dataset {

RsCapture::~RsCapture() {
  stopStreaming();
}

bool RsCapture::initSersors() {
  rs2::context ctx;
  rs2::device_list devices = ctx.query_devices();
  rs2::device selected_device;
  if (devices.size() == 0) {
    std::cerr << "No device connected, please connect a RealSense device" << std::endl;
    return false;
  } else {
    ///TODO(Jeffrey@ Oct 11th, 2023): select device by name.
    selected_device = devices[0];
  }

  std::vector<rs2::sensor> sensors = selected_device.query_sensors();
  int index = 0;
  // We can now iterate the sensors and print their names
  for (rs2::sensor sensor : sensors) {
    if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
      ++index;
      std::cout << " ************************** " << std::endl;
      std::cout << "  " << index << " : " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
      // RsHelper::getSensorOption(sensor);
      if (index == 1) {
        sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
        // sensor.set_option(RS2_OPTION_AUTO_EXPOSURE_LIMIT,5000);  // D455 doesn't support this option.
        if (vsensor_type_ == VisualSensorType::RGBD ||
            vsensor_type_ == VisualSensorType::DEPTH) {
          sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1); // switch on emitter
        } else {
          sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0); // switch off emitter
        }
      } else if (index == 2){
        // RGB camera
        // sensor.set_option(RS2_OPTION_EXPOSURE,100.f);
        sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
      } else if (index == 3){
        sensor.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION,0);
      }
      RsHelper::getSensorOption(sensor);
    }
  }
  std::cout << " ************************** " << std::endl;
  return true;
}


bool RsCapture::startStreaming() {

  if (rs_) {
    // streaming is already started.
    return false;
  }

  if (!initSersors()) {
    return false;
  }

  rs_.reset(new RsHelper());
  RsHelper& rs = *rs_;
  rs2::config default_cofig;
  rs.cfg = default_cofig;

  if (vsensor_type_ == VisualSensorType::MONO ||
      vsensor_type_ == VisualSensorType::STEREO) {
    rs.cfg.enable_stream(
        RS2_STREAM_INFRARED, 1,
        bs_.infra_width, bs_.infra_height,
        RS2_FORMAT_Y8,
        bs_.infra_framerate);
    rs.image_framerate = bs_.infra_framerate;
  }

  if (vsensor_type_ == VisualSensorType::STEREO) {
    rs.cfg.enable_stream(
        RS2_STREAM_INFRARED, 2,
        bs_.infra_width, bs_.infra_height,
        RS2_FORMAT_Y8,
        bs_.infra_framerate);
    rs.image_framerate = bs_.infra_framerate;
  }

  if (vsensor_type_ == VisualSensorType::RGBD ||
      vsensor_type_ == VisualSensorType::DEPTH) {
    rs.cfg.enable_stream(
        RS2_STREAM_DEPTH,
        bs_.depth_width, bs_.depth_height,
        RS2_FORMAT_Z16,
        bs_.depth_framerate);
    rs.image_framerate = bs_.depth_framerate;
  }

  if (vsensor_type_ == VisualSensorType::RGBD) {
    rs.cfg.enable_stream(
        RS2_STREAM_COLOR,
        bs_.color_width, bs_.color_height,
        RS2_FORMAT_RGB8,
        bs_.color_framerate);
    rs.image_framerate = bs_.color_framerate;
  }

  if (capture_imu_) {
    if (bs_.accel_framerate > 0) {
      rs.cfg.enable_stream(
          RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, bs_.accel_framerate);
    } else {
      // use the sensor's default framerate
      rs.cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    }

    if (bs_.gyro_framerate > 0) {
      rs.cfg.enable_stream(
          RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, bs_.gyro_framerate);
    } else {
      // use the sensor's default framerate
      rs.cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
    }
  }

  // // start and stop just to get necessary profile
  // rs.pipe_profile = rs.pipe.start(rs.cfg);
  // rs.pipe.stop();

  // if (vsensor_type_ == VisualSensorType::RGBD) {
  //   // Align depth and RGB frames
  //   //Pipeline could choose a device that does not have a color stream
  //   //If there is no color stream, choose to align depth to another stream
  //   rs.align_to = RsHelper::findStreamToAlign(rs.pipe_profile.get_streams());

  //   // Create a rs2::align object.
  //   // rs2::align allows us to perform alignment of depth frames to others frames
  //   //The "align_to" is the stream type to which we plan to align depth frames.
  //   rs.align = std::make_shared<rs2::align>(rs.align_to);
  // }

  auto publish_imu_sync = [this](double gyro_time, rs2_vector gyro_data) {
    RsHelper& rs = *rs_;
    rs2_vector interp_acc = RsHelper::interpolateMeasure(
        gyro_time,
        rs.acc_list.back().second, rs.acc_list.back().first,
        rs.acc_list.front().second, rs.acc_list.front().first);
    ImuData imu;
    imu.timestamp = gyro_time;
    imu.wm = Eigen::Vector3d(gyro_data.x, gyro_data.y, gyro_data.z);
    imu.am = Eigen::Vector3d(interp_acc.x, interp_acc.y, interp_acc.z);
    runImuCallbacks(rs.imu_count - 1, std::move(imu));
  };

  rs.image_process_thread_stop_request = false;
  rs.image_process_thread = std::make_shared<std::thread> ([this]() {
    pthread_setname_np(pthread_self(), "rs_img_cvt");
    RsHelper& rs = *rs_;
    while(!rs.image_process_thread_stop_request) {
      std::shared_ptr<rs2::frameset> fs;
      rs2_stream align_to;  // only for RGBD
      std::shared_ptr<rs2::align> align;  // only for RGBD
      int image_index = 0;
      {
        std::unique_lock<std::mutex> lk(rs.frame_mutex);
        if(!rs.image_ready) {
          rs.cond_image_rec.wait(lk);
        }
        if (rs.image_process_thread_stop_request) {
          std::cout << "Stop rs.image_process_thread!\n";
          break;
        }

        // std::chrono::steady_clock::time_point time_Start_Process = std::chrono::steady_clock::now();
        // // std::chrono::monotonic_clock::time_point time_Start_Process = std::chrono::monotonic_clock::now();

        fs = rs.fs;
        align_to = rs.align_to;  // only for RGBD
        align = rs.align;  // only for RGBD
        image_index = rs.image_count - 1;

        if(rs.count_im_buffer > 1) {
          std::cout << rs.count_im_buffer -1 << " dropped frs\n";
        }
        rs.count_im_buffer = 0;
        rs.image_ready = false;
      }

      CameraData cam;
      cam.timestamp = fs->get_timestamp()*1e-3;
      if (vsensor_type_ == VisualSensorType::RGBD) {
        // Perform alignment here
        auto start_time = std::chrono::steady_clock::now();
        // auto start_time = std::chrono::monotonic_clock::now();
        auto processed = align->process(*fs);
        // Trying to get both other and aligned depth frames
        rs2::video_frame color_frame = processed.first(align_to);
        rs2::depth_frame depth_frame = processed.get_depth_frame();

        auto end_time = std::chrono::steady_clock::now();
        // auto end_time = std::chrono::monotonic_clock::now();

        // auto duration = end_time - start_time;
        // std::cout << "Aligning depth takes " << duration.count() * 1e-6
        //           << " ms" << std::endl;

        auto duration =
            std::chrono::duration_cast<std::chrono::duration<double>>
            (end_time - start_time);
        std::cout << "Aligning depth takes " << duration.count() * 1e3
                  << " ms" << std::endl;

        cv::Mat color = cv::Mat(cv::Size(bs_.color_width, bs_.color_height),
                                CV_8UC3,
                                (void*)(color_frame.get_data()),
                                cv::Mat::AUTO_STEP).clone();
        cv::Mat depth = cv::Mat(cv::Size(bs_.depth_width, bs_.depth_height),
                                CV_16U,
                                (void*)(depth_frame.get_data()),
                                cv::Mat::AUTO_STEP).clone();

        cam.sensor_ids.push_back(COLOR_CAM_ID);
        cam.sensor_ids.push_back(DEPTH_CAM_ID);

        // mask has max value of 255 (white) if it should be removed
        cam.masks.push_back(cv::Mat::zeros(color.rows, color.cols, CV_8UC1));
        cam.masks.push_back(cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1));

        cam.images.push_back(std::move(color));
        cam.images.push_back(std::move(depth));
      } else if (vsensor_type_ == VisualSensorType::STEREO) {
        cv::Mat left = cv::Mat(cv::Size(bs_.infra_width, bs_.infra_height),
                              CV_8U, 
                              (void*)(fs->get_infrared_frame(1).get_data()),
                              cv::Mat::AUTO_STEP).clone();
        cv::Mat right = cv::Mat(cv::Size(bs_.infra_width, bs_.infra_height),
                                CV_8U,
                                (void*)(fs->get_infrared_frame(2).get_data()),
                                cv::Mat::AUTO_STEP).clone();
        cam.sensor_ids.push_back(LEFT_CAM_ID);
        cam.sensor_ids.push_back(RIGHT_CAM_ID);

        // mask has max value of 255 (white) if it should be removed
        cam.masks.push_back(cv::Mat::zeros(left.rows, left.cols, CV_8UC1));
        cam.masks.push_back(cv::Mat::zeros(right.rows, right.cols, CV_8UC1));

        cam.images.push_back(std::move(left));
        cam.images.push_back(std::move(right));
      } else if (vsensor_type_ == VisualSensorType::DEPTH) {
        rs2::depth_frame depth_frame = fs->get_depth_frame();
        cv::Mat depth = cv::Mat(cv::Size(bs_.depth_width, bs_.depth_height),
                                CV_16U,
                                (void*)(depth_frame.get_data()),
                                cv::Mat::AUTO_STEP).clone();
        cam.sensor_ids.push_back(DEPTH_CAM_ID);
        cam.masks.push_back(cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1));
        cam.images.push_back(std::move(depth));
      } else if (vsensor_type_ == VisualSensorType::MONO) {
        cv::Mat left = cv::Mat(cv::Size(bs_.infra_width, bs_.infra_height),
                               CV_8U,
                               (void*)(fs->get_infrared_frame().get_data()),
                               //(void*)(fs->get_infrared_frame(1).get_data()),
                               cv::Mat::AUTO_STEP).clone();
        cam.sensor_ids.push_back(LEFT_CAM_ID);

        // mask has max value of 255 (white) if it should be removed
        cam.masks.push_back(cv::Mat::zeros(left.rows, left.cols, CV_8UC1));
        cam.images.push_back(std::move(left));
      } else {
        std::cout << "Unsupported vsensor_type_: " << int64_t(vsensor_type_) << std::endl;
        continue;
      }

      runImageCallbacks(image_index, std::move(cam));
    }
  });

  auto frame_callback = [this, publish_imu_sync](const rs2::frame& frame) {
    RsHelper& rs = *rs_;
    std::unique_lock<std::mutex> lock(rs.frame_mutex);
    // if (rs.imu_count + rs.image_count == 0) {
    //   pthread_setname_np(pthread_self(), "rs_frm_cb");
    // }

    if(rs2::frameset fs = frame.as<rs2::frameset>()) {
      double new_timestamp_image = fs.get_timestamp()*1e-3;
      if (rs.first_frame_time < 0) {
        rs.first_frame_time = new_timestamp_image;
        std::cout.setf(std::ios::fixed);
        std::cout << "realsense_first_frame_time = " << std::setprecision(3) << rs.first_frame_time << std::endl;
      }

      if (rs.image_count % (rs.image_framerate * 10) == 0) {
        std::cout << "realsense_image_callback_thread: " << pthread_self()
                  << ", current frame time = " << new_timestamp_image - rs.first_frame_time
                  << ", received image frames = " << rs.image_count
                  << std::endl;
      }
      if (rs.image_count == 0) {
        pthread_setname_np(pthread_self(), "rs_img_cb");
      }

      rs.count_im_buffer++;
      rs.image_count ++;

      if(abs(rs.timestamp_image-new_timestamp_image)<0.001){
        rs.count_im_buffer--;
        return;
      }

      if (vsensor_type_ == VisualSensorType::RGBD) {
        if (!rs.align || RsHelper::profileChanged(rs.pipe.get_active_profile().get_streams(), rs.pipe_profile.get_streams()))
        {
            //If the profile was changed, update the align object, and also get the new device's depth scale
            rs.pipe_profile = rs.pipe.get_active_profile();
            rs.align_to = RsHelper::findStreamToAlign(rs.pipe_profile.get_streams());
            rs.align = std::make_shared<rs2::align>(rs.align_to);
        }
      }

      // Processing images (expecially aligning depth and rgb) takes long time,
      // so move it out of the interruption to avoid losing IMU measurements
      rs.fs = std::make_shared<rs2::frameset>(std::move(fs));

      rs.timestamp_image = rs.fs->get_timestamp()*1e-3;
      rs.image_ready = true;
      rs.cond_image_rec.notify_all();
    } else if (rs2::motion_frame m_frame = frame.as<rs2::motion_frame>()) {
      if (rs.first_frame_time < 0) {
        rs.first_frame_time = m_frame.get_timestamp() * 1e-3;
        std::cout.setf(std::ios::fixed);
        std::cout << "realsense_first_frame_time = " << std::setprecision(3) << rs.first_frame_time << std::endl;
      }

      if (m_frame.get_profile().stream_name() == "Gyro") {
        double gyro_time = m_frame.get_timestamp() * 1e-3;
        if (rs.gyro_count % (bs_.gyro_framerate > 0 ? bs_.gyro_framerate * 10 : 2000) == 0) {
          std::cout << "realsense_gyro_callback_thread: " << pthread_self()
                    << ", current frame time = " << gyro_time - rs.first_frame_time
                    << ", received gyro frames = " << rs.gyro_count
                    << std::endl;
        }
        if (rs.gyro_count == 0) {
          pthread_setname_np(pthread_self(), "rs_gyro_cb");
        }

        rs.imu_count ++;
        rs.gyro_count ++;
        rs2_vector gyro_data = m_frame.get_motion_data();

        if (rs.acc_list.size() < 2 || rs.acc_list.back().first < gyro_time) {
          rs.gyro_wait_list.push_back(std::make_pair(gyro_time, gyro_data));
        } else {
          publish_imu_sync(gyro_time, gyro_data);
        }
      } else if (m_frame.get_profile().stream_name() == "Accel") {
        double acc_time = m_frame.get_timestamp() * 1e-3;
        if (rs.acc_count % (bs_.accel_framerate > 0 ? bs_.accel_framerate * 10 : 625) == 0) {
          std::cout << "realsense_acc_callback_thread: " << pthread_self()
                    << ", current frame time = " << acc_time - rs.first_frame_time
                    << ", received acc frames = " << rs.acc_count
                    << std::endl;
        }
        if (rs.acc_count == 0) {
          pthread_setname_np(pthread_self(), "rs_acc_cb");
        }

        rs.acc_count ++;
        rs2_vector acc_data = m_frame.get_motion_data();
        rs.acc_list.push_back(std::make_pair(acc_time, acc_data));
        while (rs.acc_list.size() > 2) {
          rs.acc_list.pop_front();
        }
        if (rs.acc_list.size() == 2) {
          while (!rs.gyro_wait_list.empty() &&
                  rs.gyro_wait_list.front().first < acc_time) {
            publish_imu_sync(
                rs.gyro_wait_list.front().first, rs.gyro_wait_list.front().second);
            rs.gyro_wait_list.pop_front();
          }
        }
      }
    }
  };

  rs.pipe_profile = rs.pipe.start(rs.cfg, frame_callback);

  {
    //// DEBUG Intrinsics and Extrinsics
    // bool print = false;
    bool print = true;

    if (capture_imu_) {
      RsHelper::getExtrinsics(rs.pipe_profile, RS2_STREAM_ACCEL, RS2_STREAM_GYRO, -1, -1, print);
    }

    if (vsensor_type_ == VisualSensorType::MONO ||
        vsensor_type_ == VisualSensorType::STEREO) {      
      RsHelper::getCameraIntrinsics(rs.pipe_profile, RS2_STREAM_INFRARED, 1, print);
      if (capture_imu_) {
        RsHelper::getExtrinsics(rs.pipe_profile, RS2_STREAM_INFRARED, RS2_STREAM_GYRO, 1, -1, print);
      }
    }

    if (vsensor_type_ == VisualSensorType::STEREO) {
      RsHelper::getCameraIntrinsics(rs.pipe_profile, RS2_STREAM_INFRARED, 2, print);
      RsHelper::getExtrinsics(rs.pipe_profile, RS2_STREAM_INFRARED, RS2_STREAM_INFRARED, 2, 1, print);
      if (capture_imu_) {
        RsHelper::getExtrinsics(rs.pipe_profile, RS2_STREAM_INFRARED, RS2_STREAM_GYRO, 2, -1, print);
      }
    }

    if (vsensor_type_ == VisualSensorType::RGBD ||
        vsensor_type_ == VisualSensorType::DEPTH) {
      RsHelper::getCameraIntrinsics(rs.pipe_profile, RS2_STREAM_DEPTH, -1, print);
      if (capture_imu_) {
        RsHelper::getExtrinsics(rs.pipe_profile, RS2_STREAM_DEPTH, RS2_STREAM_GYRO, -1, -1, print);
      }
    }

    if (vsensor_type_ == VisualSensorType::RGBD) {
      RsHelper::getCameraIntrinsics(rs.pipe_profile, RS2_STREAM_COLOR, -1, print);
      RsHelper::getExtrinsics(rs.pipe_profile, RS2_STREAM_DEPTH, RS2_STREAM_COLOR, -1, -1, print);
      if (capture_imu_) {
        RsHelper::getExtrinsics(rs.pipe_profile, RS2_STREAM_COLOR, RS2_STREAM_GYRO, -1, -1, print);
      }      
    }
  }

  return true;
}


void RsCapture::stopStreaming() {
  if (rs_) {
    RsHelper& rs = *rs_;
    if (rs.image_process_thread) {
      {
        rs.image_process_thread_stop_request = true;
        std::unique_lock<std::mutex> lock(rs.frame_mutex);
        rs.cond_image_rec.notify_all();
      }
      rs.image_process_thread->join();
      rs.image_process_thread.reset();
    }
    rs.pipe.stop();
    rs_.reset();
  }
}

bool RsCapture::isStreaming() const {
  if (rs_) {
    return true;
  } else {
    return false;
  }
}

} // namespace slam_dataset


#endif // USE_HEAR_SLAM
