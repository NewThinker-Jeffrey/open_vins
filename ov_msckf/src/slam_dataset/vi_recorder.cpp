#ifndef USE_HEAR_SLAM

#include "vi_recorder.h"

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <math.h>

namespace {

void runcmd(const std::string& cmd) {
  system(cmd.c_str());
}

std::string getCmdToIndexImageFolder(const std::string& image_folder) {
  return "ls " + image_folder + " | "
         "sort | "
         "awk -F. '{print $1\",\"$0 }' > " + image_folder + ".csv";
}

uint8_t* log_depth_table = nullptr;

cv::Mat cvtToLogDepth(const cv::Mat& depth_img) {
  assert(depth_img.type() == CV_16U);

  if (!log_depth_table) {
    // build log_depth_table
    log_depth_table = new uint8_t[65536];
    const double inv_log2__x__16 = 16.0 / std::log(2.0);
    auto log_value = [](int depth_i) {
      double depth = 0.001 * depth_i;  // in meters
      static const double base_depth = 0.3;
      double delta_depth = (depth - base_depth);
      delta_depth = std::max(delta_depth, 0.0);
      return log(double(1000.0*delta_depth+1.0));
      // return log(double(delta_depth+1.0));
      // return log(double(depth+1.0));
    };
    double min_value = log_value(0);
    double max_value = log_value(65535);
    double ratio = 255.0 / (max_value - min_value);
    for (int i=0; i<65536; i++) {
      int v = (log_value(i) - min_value) * ratio;
      v = std::min(v, 255);
      v = std::max(v, 0);
      log_depth_table[i] = 255 - v;
    }

    // 0 for unavailable.
    log_depth_table[0] = 0;
  }

  cv::Mat log_depth_img(depth_img.size(), CV_8U);
  auto* ptr_log = log_depth_img.ptr<uint8_t>();
  const auto* ptr = depth_img.ptr<uint16_t>();
  for (size_t i=0; i<depth_img.rows*depth_img.cols; i++) {
    ptr_log[i] = log_depth_table[ptr[i]];
  }

  return log_depth_img;
}

}  // namespace

namespace slam_dataset {


struct ViRecorder::Context {
  std::ofstream imu_file;
  std::string save_folder;
  std::string indexing_script;

  //// io thread
  std::shared_ptr<std::thread> io_thread;
  std::mutex io_mutex;
  std::condition_variable io_cond;
  bool stop_request;
  std::deque<CameraData> cam_queue;
  std::deque<ImuData> imu_queue;

  std::atomic<bool> show_image;
};

bool ViRecorder::startRecord(
    ViCapture* capture, const std::string& save_folder,
    VisualSensorType type, bool record_imu,
    const std::string& thread_name) {
  if (c_) {
    // Recording is already started
    return false;
  }

  if (capture) {
    vsensor_type_ = capture->getVisualSensorType();
    record_imu_ = capture->isImuEnabled();
  } else {
    vsensor_type_ = type;
    record_imu_ = record_imu;
  }
  // Issue a crash if vsensor_type has not been set.
  assert(vsensor_type_ != VisualSensorType::NONE);

  c_.reset(new Context());
  c_->show_image = false;
  c_->save_folder = save_folder;
  if (c_->save_folder.empty()) {
    static const std::string DATASET_ROOT = "./slam_datasets";
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);  // NOLINT
    char dt[100];
    sprintf(  // NOLINT
        dt, "%04d-%02d-%02d_%02d-%02d-%02d", tm.tm_year + 1900, tm.tm_mon + 1,
        tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    c_->save_folder = DATASET_ROOT + "/" + std::string(dt);
  }
  
  runcmd("mkdir -p " + c_->save_folder);
  c_->indexing_script = c_->save_folder + "/indexing_images.sh";
  std::ofstream ofs(c_->indexing_script, std::ios::out);
  ofs << "#!/usr/bin/env bash" << std::endl;
  ofs << "dataset=$(cd $(dirname $0) && pwd)" << std::endl;

  if (vsensor_type_ == VisualSensorType::STEREO) {
    runcmd("mkdir -p " + c_->save_folder + "/cam0/data");
    runcmd("mkdir -p " + c_->save_folder + "/cam1/data");
    ofs << getCmdToIndexImageFolder("${dataset}/cam0/data") << std::endl;
    ofs << getCmdToIndexImageFolder("${dataset}/cam1/data") << std::endl;
  } else if (vsensor_type_ == VisualSensorType::MONO) {
    runcmd("mkdir -p " + c_->save_folder + "/cam0/data");
    ofs << getCmdToIndexImageFolder("${dataset}/cam0/data") << std::endl;
  } else if (vsensor_type_ == VisualSensorType::RGBD) {
    runcmd("mkdir -p " + c_->save_folder + "/color/data/");
    runcmd("mkdir -p " + c_->save_folder + "/depth/data/");
    ofs << getCmdToIndexImageFolder("${dataset}/color/data") << std::endl;
    ofs << getCmdToIndexImageFolder("${dataset}/depth/data") << std::endl;
  } else if (vsensor_type_ == VisualSensorType::DEPTH) {
    runcmd("mkdir -p " + c_->save_folder + "/depth/data/");
    ofs << getCmdToIndexImageFolder("${dataset}/depth/data") << std::endl;
  }
  ofs.close();
  runcmd("chmod +x " + c_->indexing_script);

  if (record_imu_) {
    runcmd("mkdir -p " + c_->save_folder + "/imu0/");
    std::string imu_file_name = c_->save_folder + "/imu0/data.csv";
    c_->imu_file = std::ofstream(imu_file_name, std::ios::out);
  }

  c_->stop_request = false;
  c_->io_thread = std::make_shared<std::thread> ([this, thread_name]() {
    std::string truncated_name = thread_name;
    if (truncated_name.length() > 15) {
      truncated_name = truncated_name.substr(0, 15);
    }
    pthread_setname_np(pthread_self(), truncated_name.c_str());
    bool stop = false;
    while(!stop) {
      std::deque<CameraData> cam_queue;
      std::deque<ImuData> imu_queue;
      {
        std::unique_lock<std::mutex> lock(c_->io_mutex);
        if(c_->cam_queue.empty() && c_->imu_queue.empty()) {
          c_->io_cond.wait(lock);
        }
        if (c_->stop_request) {
          stop = true;
        }
        std::swap(c_->cam_queue, cam_queue);
        std::swap(c_->imu_queue, imu_queue);
      }

      if (!imu_queue.empty() && c_->imu_file.is_open()) {
        for (const auto & imu : imu_queue) {
          c_->imu_file
              << int64_t(imu.timestamp * 1e9) << "," 
              << imu.wm[0] << "," << imu.wm[1] << "," << imu.wm[2] << ","
              << imu.am[0] << "," << imu.am[1] << "," << imu.am[2]
              << std::endl;
        }
      }

      if (!cam_queue.empty()) {
        // const std::string img_save_format = ".png";
        const std::string img_save_format = ".jpg";
        for (const auto & cam : cam_queue) {
          if (vsensor_type_ == VisualSensorType::STEREO) {
            cv::imwrite(c_->save_folder + "/cam0/data/"
                        + std::to_string(int64_t(cam.timestamp * 1e9))
                        + img_save_format, cam.images[0]);
            cv::imwrite(c_->save_folder + "/cam1/data/"
                        + std::to_string(int64_t(cam.timestamp * 1e9))
                        + img_save_format, cam.images[1]);
            if (c_->show_image) {
              cv::imshow("left", cam.images[0]);
              cv::imshow("right", cam.images[1]);
              cv::waitKey(1);
            }
          } else if (vsensor_type_ == VisualSensorType::MONO) {
            cv::imwrite(c_->save_folder + "/cam0/data/"
                        + std::to_string(int64_t(cam.timestamp * 1e9))
                        + img_save_format, cam.images[0]);
            if (c_->show_image) {
              cv::imshow("left", cam.images[0]);
              cv::waitKey(1);
            }
          } else if (vsensor_type_ == VisualSensorType::RGBD) {
            // imshow() and imwrite() needs BGR format, while
            // rs'color image is in RGB.
            cv::Mat bgr;
            cv::cvtColor(cam.images[0], bgr, cv::COLOR_RGB2BGR);
            cv::imwrite(c_->save_folder + "/color/data/"
                        + std::to_string(int64_t(cam.timestamp * 1e9))
                        + img_save_format, bgr);
            // depth image will be always saved in .png format 
            cv::imwrite(c_->save_folder + "/depth/data/"
                        + std::to_string(int64_t(cam.timestamp * 1e9))
                        + ".png", cam.images[1]);
            if (c_->show_image) {
              cv::imshow("color", bgr);              
              cv::imshow("depth", cvtToLogDepth(cam.images[1]));
              cv::waitKey(1);
            }
          } else if (vsensor_type_ == VisualSensorType::DEPTH) {
            // depth image will be always saved in .png format 
            cv::imwrite(c_->save_folder + "/depth/data/"
                        + std::to_string(int64_t(cam.timestamp * 1e9))
                        + ".png", cam.images[0]);
            if (c_->show_image) {
              cv::imshow("depth", cvtToLogDepth(cam.images[0]));
              cv::waitKey(1);
            }
          }
        }
      }
    }
    std::cout << "**** ViRecorder:  data stopped! *****" << std::endl;
  });

  if (capture) {
    capture->registerImageCallback([this](int, CameraData msg){
      feedCameraData(std::move(msg));
    });
    capture->registerImuCallback([this](int, ImuData msg){
      feedImuData(std::move(msg));
    });
  }

  return true;
}


void ViRecorder::stopRecord() {
  if (c_) {
    // stop threads?
    {
      std::unique_lock<std::mutex> lock(c_->io_mutex);
      c_->stop_request  = true;
      c_->io_cond.notify_all();
    }

    if (c_->io_thread) {
      c_->io_thread->join();
      c_->io_thread.reset();
    }

    if (c_->imu_file.is_open()) {
      c_->imu_file.close();
    }

    std::cout << "**** ViRecorder:  generating indexing file ! *****" << std::endl;
    runcmd(c_->indexing_script);
    std::cout << "**** ViRecorder:  FINISHED ! *****" << std::endl;    

    c_.reset();
  }
}

void ViRecorder::enableImageWindow() {
  c_->show_image = true;
}

void ViRecorder::disableImageWindow() {
  c_->show_image = false;
}

void ViRecorder::feedCameraData(CameraData cam) {
  if (c_) {
    std::unique_lock<std::mutex> lock(c_->io_mutex);
    if (!c_->stop_request) {
      c_->cam_queue.push_back(std::move(cam));
      c_->io_cond.notify_one();
      // std::cout << "ViRecorder image_queue size = " << c_->cam_queue.size() << std::endl;
    }
  }
}

void ViRecorder::feedImuData(ImuData imu) {
  if (c_) {
    std::unique_lock<std::mutex> lock(c_->io_mutex);
    if (!c_->stop_request) {
      c_->imu_queue.push_back(std::move(imu));
      c_->io_cond.notify_one();
      // std::cout << "ViRecorder imu_queue size = " << c_->imu_queue.size() << std::endl;
    }
  }
}

} // namespace slam_dataset

#endif // USE_HEAR_SLAM
