#include "vi_player.h"

#include <mutex>
#include <condition_variable>

namespace slam_dataset {

namespace {

bool checkFileExistence(const std::string& path);

void loadIMU(const std::string &imu_path,
             std::vector<ImuData>& imu_data);

void loadImages(const std::vector<std::string>& img_folders,
                const std::string &img_time_path,
                std::vector< std::vector<std::string>* > img_files,
                std::vector<double>* timestamps);

void readImg(const std::string& img_path,
             CameraData& message);

void readStereoImg(const std::string& img_path_l,
                   const std::string& img_path_r,
                   CameraData& message);

void readRGBDImg(const std::string& img_path_rgb,
                 const std::string& img_path_depth,
                 CameraData& message);

void readDepthImg(const std::string& img_path_depth,
                  CameraData& message);

}  // namespace


ViPlayer::~ViPlayer() {
  stopStreaming();
}

bool ViPlayer::startStreaming() {
  std::cout << "ViPlayer::startPlay(): "
            << "dataset='" << dataset_ << "', "
            << "vsensor_type_=" << int64_t(vsensor_type_) << ", "
            << "play_rate=" << play_rate_ << std::endl;
  
  loadDataset();
  stop_request_ = false;
  data_play_thread_.reset(new std::thread([this](){
    pthread_setname_np(pthread_self(), "slam_play");
    dataPlay();
    is_streaming_ = false;
  }));
  is_streaming_ = true;
  return true;
}

void ViPlayer::stopStreaming() {
  stop_request_ = true;
  if (data_play_thread_ && data_play_thread_->joinable()) {
    data_play_thread_->join();
    data_play_thread_.reset();
  }
  is_streaming_ = false;
}

bool ViPlayer::isStreaming() const {
  return is_streaming_;
}

void ViPlayer::loadDataset() {

  loadIMU(dataset_ + "/imu0/data.csv", imu_data_);

  std::vector<std::string> img_folders;
  std::vector< std::vector<std::string>* > img_files;

  if (vsensor_type_ == VisualSensorType::STEREO) {
    img_folders.push_back(dataset_ + "/cam0/data");
    img_folders.push_back(dataset_ + "/cam1/data");
    img_files.push_back(&camname_to_imgfiles_["left"]);
    img_files.push_back(&camname_to_imgfiles_["right"]);
  } else if (vsensor_type_ == VisualSensorType::RGBD) {
    img_folders.push_back(dataset_ + "/color/data");
    img_folders.push_back(dataset_ + "/depth/data");
    img_files.push_back(&camname_to_imgfiles_["color"]);
    img_files.push_back(&camname_to_imgfiles_["depth"]);
  } else if (vsensor_type_ == VisualSensorType::MONO) {
    img_folders.push_back(dataset_ + "/cam0/data");
    img_files.push_back(&camname_to_imgfiles_["left"]);
  } else if (vsensor_type_ == VisualSensorType::DEPTH) {
    img_folders.push_back(dataset_ + "/depth/data");
    img_files.push_back(&camname_to_imgfiles_["depth"]);    
  } else {

  }

  if (!img_folders.empty()) {
    loadImages(img_folders, img_folders[0] + ".csv",
               img_files, &image_times_);
  }

  std::cout << "ViPlayer::loadDataset():  imu frames: "
            << imu_data_.size() << std::endl;
  std::cout << "ViPlayer::loadDataset():  image frames: "
            << image_times_.size() << std::endl;
}

void ViPlayer::dataPlay() {
  std::cout << "**ViPlayer::data_play()**" << std::endl;
  size_t imu_idx = 0;
  size_t image_idx = 0;
  std::map<int, double> camera_last_timestamp;
  double imu_start_time = imu_data_.empty() ? -1 : imu_data_[imu_idx].timestamp;
  double image_start_time = image_times_.empty() ? -1 : image_times_[image_idx];
  if (image_start_time > 0 && imu_start_time > 0) {
    // sensor_start_time_ = imu_start_time;
    // sensor_start_time_ = std::min(imu_start_time, image_start_time);
    sensor_start_time_ = std::max(imu_start_time, image_start_time);    
  } else if (imu_start_time > 0) {
    sensor_start_time_ = imu_start_time;
  } else {
    sensor_start_time_ = imu_start_time;
  }
  play_start_time_ = std::chrono::high_resolution_clock::now();

  std::mutex mutex;
  std::condition_variable cond;

  // std::cout << "image_start_time - imu_start_time = " << image_start_time - imu_start_time << std::endl;
  // for (size_t i=0; i<10; i++) {
  //   std::cout << "imu " << i << ":" << (imu_data_[i].timestamp - imu_data_[0].timestamp) << std::endl;
  // }
  // for (size_t i=0; i<10; i++) {    
  //   std::cout << "image " << i << ":" << (image_times_[i] - image_times_[0]) << std::endl;
  // }

  std::cout << "ViPlayer::dataPlay():  play_rate: " << play_rate_ << std::endl;
  std::cout << "ViPlayer::dataPlay():  imu frames: " << imu_data_.size() << std::endl;
  std::cout << "ViPlayer::dataPlay():  image frames: " << image_times_.size() << std::endl;
  std::cout << "ViPlayer::dataPlay():  imu_start_time: " << imu_start_time << std::endl;
  std::cout << "ViPlayer::dataPlay():  image_start_time: " << image_start_time << std::endl;
  std::cout << "ViPlayer::dataPlay():  sensor_start_time: " << sensor_start_time_ << std::endl;

  double next_sensor_time;
  auto get_next_sensor_time = [&]() {
    double next_imu_time = imu_idx < imu_data_.size() ? imu_data_[imu_idx].timestamp : -1;
    double next_image_time = image_idx < image_times_.size() ? image_times_[image_idx] : -1;
    if (next_imu_time < 0) {
      return next_image_time;
    } else if (next_image_time < 0) {
      return next_imu_time;
    } else {
      return std::min(next_image_time, next_imu_time);
    }
  };

  // std::cout << "sensor_start_time " << sensor_start_time_ << std::endl;

  while(!stop_request_ && (next_sensor_time = get_next_sensor_time()) > 0) {
    double sensor_dt = next_sensor_time - sensor_start_time_;
    int ms_play_dt = int(sensor_dt / play_rate_ * 1000);
    // std::cout << "ViPlayer::dataPlay():  ms_play_dt: " << ms_play_dt << ",  " << sensor_dt / play_rate_ * 1000 << std::endl;
    auto next_play_time = play_start_time_ + std::chrono::milliseconds(int(sensor_dt / play_rate_ * 1000));
    {
      std::unique_lock<std::mutex> locker(mutex);
      while (cond.wait_until(locker, next_play_time) != std::cv_status::timeout);
    }

    if (image_idx < image_times_.size() && image_times_[image_idx] <= next_sensor_time) {

      double timestamp = image_times_[image_idx];
      CameraData cam;
      cam.timestamp = timestamp;

      if (vsensor_type_ == VisualSensorType::STEREO) {
        cam.sensor_ids.push_back(LEFT_CAM_ID);
        cam.sensor_ids.push_back(RIGHT_CAM_ID);
        readStereoImg(camname_to_imgfiles_["left"][image_idx],
                      camname_to_imgfiles_["right"][image_idx],
                      cam);
      } else if (vsensor_type_ == VisualSensorType::RGBD) {
        cam.sensor_ids.push_back(COLOR_CAM_ID);
        cam.sensor_ids.push_back(DEPTH_CAM_ID);
        readRGBDImg(camname_to_imgfiles_["color"][image_idx],
                    camname_to_imgfiles_["depth"][image_idx],
                    cam);
      } else if (vsensor_type_ == VisualSensorType::MONO) {
        cam.sensor_ids.push_back(LEFT_CAM_ID);
        readImg(camname_to_imgfiles_["left"][image_idx], cam);
      } else if (vsensor_type_ == VisualSensorType::DEPTH) {
        cam.sensor_ids.push_back(DEPTH_CAM_ID);
        readDepthImg(camname_to_imgfiles_["depth"][image_idx], cam);
      } else {
      }

      // set masks for images
      for (size_t i = 0; i < cam.images.size(); i ++) {
        // mask has max value of 255 (white) if it should be removed
        int rows = cam.images[i].rows;
        int cols = cam.images[i].cols;
        cam.masks.push_back(cv::Mat::zeros(rows, cols, CV_8UC1));
      }

      runImageCallbacks(image_idx, std::move(cam));
      image_idx ++;
    }

    if (imu_idx < imu_data_.size() && imu_data_[imu_idx].timestamp <= next_sensor_time) {
      auto& imu = imu_data_[imu_idx];
      runImuCallbacks(imu_idx, std::move(imu));
      imu_idx ++;
    }
  }

  std::cout << "**** ViPlayer: play over! *****" << std::endl;
}


namespace {

bool checkFileExistence(const std::string& path) {
  std::ifstream f(path.c_str());
  return f.good();
}

void loadIMU(const std::string &imu_path,
             std::vector<ImuData>& imu_data) {
  std::ifstream imu_file;
  imu_file.open(imu_path.c_str());
  imu_data.reserve(5000);
  while(!imu_file.eof()) {
    std::string s;
    std::getline(imu_file,s);
    if (s[0] == '#')
      continue;
    if(!s.empty())
    {
      std::string item;
      size_t pos = 0;
      double data[7];
      int count = 0;
      while ((pos = s.find(',')) != std::string::npos) {
        item = s.substr(0, pos);
        data[count++] = std::stod(item);
        s.erase(0, pos + 1);
      }
      item = s.substr(0, pos);
      data[6] = std::stod(item);
      ImuData imu;
      imu.timestamp = data[0]/1e9;
      imu.wm = Eigen::Vector3d(data[1],data[2],data[3]);
      imu.am = Eigen::Vector3d(data[4],data[5],data[6]);
      imu_data.push_back(imu);
    }
  }
}

void loadImages(const std::vector<std::string>& img_folders,
                const std::string &img_time_path,
                std::vector< std::vector<std::string>* > img_files,
                std::vector<double>* timestamps) {
  std::ifstream time_file;
  std::vector<std::string> pic_suffix(img_folders.size(), "");
  time_file.open(img_time_path.c_str());
  timestamps->reserve(5000);
  for (auto* p : img_files) {
    p->clear();
    p->reserve(5000);
  }

  while(!time_file.eof()) {
    std::string s;
    std::getline(time_file, s);
    int pos;
    if ((pos = s.find(',')) != std::string::npos) {
      s = s.substr(0, pos);
    }
    if(!s.empty()) {
      std::stringstream ss;
      ss << s;
      for (size_t i = 0; i < img_folders.size(); i++) {
        if (pic_suffix[i].empty()) {
          std::string testfile = img_folders[i] + "/" + ss.str();
          std::vector <std::string> suffix_list = {".png", ".jpg"};
          for (const std::string & suffix : suffix_list) {
            if (checkFileExistence(testfile + suffix)) {
              pic_suffix[i] = suffix;
              break;
            }
          }
        }
        img_files[i]->push_back(img_folders[i] + "/" + ss.str() + pic_suffix[i]);
      }
      double t;
      ss >> t;
      timestamps->push_back(t/1e9);
    }
  }
}

void readImg(const std::string& img_path,
             CameraData& message) {
  cv::Mat img;
  // img = cv::imread(img_path,cv::IMREAD_UNCHANGED);
  img = cv::imread(img_path,cv::IMREAD_GRAYSCALE);
  message.images.push_back(img.clone());
}

void readStereoImg(const std::string& img_path_l,
                   const std::string& img_path_r,
                   CameraData& message) {
  cv::Mat img_left, img_right;

  // img_left = cv::imread(img_path_l,cv::IMREAD_UNCHANGED);
  // img_right = cv::imread(img_path_r,cv::IMREAD_UNCHANGED);

  img_left = cv::imread(img_path_l,cv::IMREAD_GRAYSCALE);
  img_right = cv::imread(img_path_r,cv::IMREAD_GRAYSCALE);

  message.images.push_back(img_left.clone());
  message.images.push_back(img_right.clone());
}

void readRGBDImg(const std::string& img_path_rgb,
                 const std::string& img_path_depth,
                 CameraData& message) {
  cv::Mat img_rgb, img_depth;

  //   img_rgb = cv::imread(img_path_rgb, cv::IMREAD_UNCHANGED);
  //   img_depth = cv::imread(img_path_depth, cv::IMREAD_UNCHANGED);

  img_rgb = cv::imread(img_path_rgb, cv::IMREAD_COLOR);
  img_depth = cv::imread(img_path_depth, cv::IMREAD_GRAYSCALE);

  message.images.push_back(img_rgb.clone());
  message.images.push_back(img_depth.clone());
}

void readDepthImg(const std::string& img_path_depth,
                  CameraData& message) {
  cv::Mat img_depth;

  //   img_depth = cv::imread(img_path_depth, cv::IMREAD_UNCHANGED);

  img_depth = cv::imread(img_path_depth, cv::IMREAD_GRAYSCALE);

  message.images.push_back(img_depth.clone());
}

}  // namespace


}  // namespace slam_dataset

