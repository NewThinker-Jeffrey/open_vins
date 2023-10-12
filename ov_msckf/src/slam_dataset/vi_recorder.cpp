#include "vi_recorder.h"

namespace {

void runcmd(const std::string& cmd) {
  system(cmd.c_str());
}

std::string getCmdToIndexImageFolder(const std::string& image_folder) {
  return "ls " + image_folder + " | "
         "sort | "
         "awk -F. '{print $1\",\"$0 }' > " + image_folder + ".csv";
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
};

bool ViRecorder::startRecord(const std::string& save_folder) {
  if (c_) {
    // Recording is already started
    return false;
  }

  c_ = new Context();
  c_->save_folder = save_folder;

  runcmd("mkdir -p " + save_folder);
  c_->indexing_script = save_folder + "/indexing_images.sh";
  std::ofstream ofs(c_->indexing_script, std::ios::out);
  ofs << "#!/usr/bin/env bash" << std::endl;
  ofs << "dataset=$(cd $(dirname $0) && pwd)" << std::endl;

  if (vsensor_type_ == VisualSensorType::STEREO) {
    runcmd("mkdir -p " + save_folder + "/cam0/data");
    runcmd("mkdir -p " + save_folder + "/cam1/data");
    ofs << getCmdToIndexImageFolder("${dataset}/cam0/data") << std::endl;
    ofs << getCmdToIndexImageFolder("${dataset}/cam1/data") << std::endl;
  } else if (vsensor_type_ == VisualSensorType::MONO) {
    runcmd("mkdir -p " + save_folder + "/cam0/data");
    ofs << getCmdToIndexImageFolder("${dataset}/cam0/data") << std::endl;
  } else if (vsensor_type_ == VisualSensorType::RGBD) {
    runcmd("mkdir -p " + save_folder + "/color/data/");
    runcmd("mkdir -p " + save_folder + "/depth/data/");
    ofs << getCmdToIndexImageFolder("${dataset}/color/data") << std::endl;
    ofs << getCmdToIndexImageFolder("${dataset}/depth/data") << std::endl;
  } else if (vsensor_type_ == VisualSensorType::DEPTH) {
    runcmd("mkdir -p " + save_folder + "/depth/data/");
    ofs << getCmdToIndexImageFolder("${dataset}/depth/data") << std::endl;
  }
  ofs.close();
  runcmd("chmod +x " + c_->indexing_script);

  if (record_imu_) {
    runcmd("mkdir -p " + save_folder + "/imu0/");
    std::string imu_file_name = save_folder + "/imu0/data.csv";
    c_->imu_file = std::ofstream(imu_file_name, std::ios::out);
  }

  
  c_->stop_request = false;
  c_->io_thread = new std::thread([this](){
    bool stop = false;
    while(!stop) {
      std::deque<CameraData> cam_queue;
      std::deque<ImuData> imu_queue;
      {
        std::unique_lock<std::mutex> lock(c_->io_mutex);
        if(c_->cam_queue.empty() && c_->imu_queue.empty()) {
          rs.cond_image_rec.wait(lk);
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
              << imu.timestamp << "," 
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
                        + std::to_string(cam.timestamp)
                        + img_save_format, cam.images[0]);
            cv::imwrite(c_->save_folder + "/cam1/data/"
                        + std::to_string(cam.timestamp)
                        + img_save_format, cam.images[1]);
          } else if (vsensor_type_ == VisualSensorType::MONO) {
            cv::imwrite(c_->save_folder + "/cam0/data/"
                        + std::to_string(cam.timestamp)
                        + img_save_format, cam.images[0]);
          } else if (vsensor_type_ == VisualSensorType::RGBD) {
            cv::imwrite(c_->save_folder + "/color/data/"
                        + std::to_string(cam.timestamp)
                        + img_save_format, cam.images[0]);
            // depth image will be always saved in .png format 
            cv::imwrite(c_->save_folder + "/depth/data/"
                        + std::to_string(cam.timestamp)
                        + ".png", cam.images[1]);
          } else if (vsensor_type_ == VisualSensorType::DEPTH) {
            // depth image will be always saved in .png format 
            cv::imwrite(c_->save_folder + "/depth/data/"
                        + std::to_string(cam.timestamp)
                        + ".png", cam.images[1]);
          }
        }
      }
    }
  });

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

    runcmd(c_->indexing_script);
    
    c_.reset();
  }
}

void ViRecorder::feedCameraData(CameraData cam) {
  if (c_) {
    std::unique_lock<std::mutex> lock(c_->io_mutex);
    if (!c_->stop_request) {
      c_->cam_queue.push_back(std::move(cam));
      c_->io_cond.notify_one();
    }
  }
}

void ViRecorder::feedImuData(ImuData imu) {
  if (c_) {
    std::unique_lock<std::mutex> lock(c_->io_mutex);
    if (!c_->stop_request) {
      c_->imu_queue.push_back(std::move(imu));
      c_->io_cond.notify_one();
    }
  }
}

} // namespace slam_dataset

