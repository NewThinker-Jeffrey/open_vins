
#include "VIO.h"

#include <memory>
#include <string>
#include <mutex>
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

// #if ROS_AVAILABLE == 2
// #include <rclcpp/rclcpp.hpp>
// std::weak_ptr<rclcpp::Node> unique_parser_node;
// // extern std::weak_ptr<rclcpp::Node> unique_parser_node;
// #endif


struct ov_interface::VIO::Impl {
  Impl(const std::string& config_file) : config_file_(config_file) {
  }

  std::string config_file_;
  ov_msckf::VioManagerOptions params_;
  std::shared_ptr<ov_msckf::VioManager> internal_;
};

// extern std::shared_ptr<ov_msckf::VioManager> getVioManagerFromVioInterface(ov_interface::VIO*);
std::shared_ptr<ov_msckf::VioManager> getVioManagerFromVioInterface(ov_interface::VIO* vio) {
  // return std::dynamic_pointer_cast<ov_msckf::VioManager>(vio->impl());
  return vio->impl()->internal_;
}


namespace ov_interface {

VIO::VIO(const char* config_file) : impl_(new Impl(config_file)) {
  
}

VIO::~VIO() {
  Shutdown();
  if (impl_) {
    delete impl_;
  }
}

bool VIO::Init() {
  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(impl_->config_file_);

// #if ROS_AVAILABLE == 2
//   if (auto node = unique_parser_node.lock()) {
//     parser->set_node(node);
//   }
// #endif

  // Verbosity
  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  ov_msckf::VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = true;
  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }
  impl_->params_ = params;
  Reset();
  return true;
}

void VIO::ReceiveImu(const IMU_MSG &imu_msg) {
#ifndef USE_INTERNAL_MSG_TYPE
  if (!imu_msg.valid) {
    return;
  }

  const double* a = imu_msg.linear_acceleration;
  const double* w = imu_msg.angle_velocity;
  ov_core::ImuData message;
  message.timestamp = imu_msg.timestamp;
  message.wm = Eigen::Vector3d(w[0], w[1], w[2]);
  message.am = Eigen::Vector3d(a[0], a[1], a[2]);
  impl_->internal_->feed_measurement_imu(message);
#else
  impl_->internal_->feed_measurement_imu(imu_msg);
#endif
}

void VIO::ReceiveCamera(const IMG_MSG &img_msg) {
#ifndef USE_INTERNAL_MSG_TYPE  
  if (!img_msg.valid) {
    return;
  }

  cv::Mat gray;
  if (img_msg.channel == 1) {
    gray = cv::Mat(img_msg.height, img_msg.width, CV_8UC1, const_cast<char*>(img_msg.data)).clone();
  } else if (img_msg.channel == 3) {
    cv::Mat color(img_msg.height, img_msg.width, CV_8UC3, const_cast<char*>(img_msg.data));
    cv::cvtColor(color, gray, cv::COLOR_RGB2GRAY);
    // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  }

  ov_core::CameraData message;
  int cam_id0 = img_msg.cam_id;
  message.timestamp = img_msg.timestamp;
  message.sensor_ids.push_back(cam_id0);
  message.images.push_back(gray);

  if (impl_->internal_->get_params().use_mask) {
    message.masks.push_back(impl_->internal_->get_params().masks.at(cam_id0));
  } else {
    message.masks.push_back(cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1));
  }

  impl_->internal_->feed_measurement_camera(std::move(message));  // todo: run this in another thread?
#else
  impl_->internal_->feed_measurement_camera(img_msg);  // todo: run this in another thread?
#endif
}

void VIO::ReceiveStereoCamera(const STEREO_IMG_MSG &img_msg) {
#ifndef USE_INTERNAL_MSG_TYPE
  if (!img_msg.valid) {
    return;
  }
  cv::Mat gray_l, gray_r;
  if (img_msg.channel == 1) {
    gray_l = cv::Mat(img_msg.height, img_msg.width, CV_8UC1, const_cast<char*>(img_msg.data_l)).clone();
    gray_r = cv::Mat(img_msg.height, img_msg.width, CV_8UC1, const_cast<char*>(img_msg.data_r)).clone();
  } else if (img_msg.channel == 3) {
    cv::Mat color_l(img_msg.height, img_msg.width, CV_8UC3, const_cast<char*>(img_msg.data_l));
    cv::Mat color_r(img_msg.height, img_msg.width, CV_8UC3, const_cast<char*>(img_msg.data_r));
    cv::cvtColor(color_l, gray_l, cv::COLOR_RGB2GRAY);
    cv::cvtColor(color_r, gray_r, cv::COLOR_RGB2GRAY);

    // cv::cvtColor(color_l, gray_l, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(color_r, gray_r, cv::COLOR_BGR2GRAY);
  }

  ov_core::CameraData message;
  int cam_id0 = img_msg.cam_id_left;
  int cam_id1 = img_msg.cam_id_right;
  message.timestamp = img_msg.timestamp;
  message.sensor_ids.push_back(cam_id0);
  message.sensor_ids.push_back(cam_id1);
  message.images.push_back(gray_l);
  message.images.push_back(gray_r);

  if (impl_->internal_->get_params().use_mask) {
    message.masks.push_back(impl_->internal_->get_params().masks.at(cam_id0));
    message.masks.push_back(impl_->internal_->get_params().masks.at(cam_id1));
  } else {
    message.masks.push_back(cv::Mat::zeros(gray_l.rows, gray_l.cols, CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(gray_r.rows, gray_r.cols, CV_8UC1));
  }

  impl_->internal_->feed_measurement_camera(std::move(message));  // todo: run this in another thread?
#else
  impl_->internal_->feed_measurement_camera(img_msg);  // todo: run this in another thread?
#endif
}



LOC_MSG VIO::Localization(bool predict_with_imu) {
  LOC_MSG loc;
  loc.timestamp = -1.0;
  loc.err = 0;
  auto output = impl_->internal_->getLastOutput(true, false);
  if (!output->status.initialized) {
    // vio has not been initilized yet.
    return loc;
  }

  if (!predict_with_imu) {
    // Eigen::Isometry3d imu_pose(output->state_clone->_imu->Rot().inverse());
    // imu_pose.translation() = output->state_clone->_imu->pos();
    // loc.pose = imu_pose;
    auto jpl_q = output->state_clone->_imu->quat();
    auto pos = output->state_clone->_imu->pos();

    // loc.q[0] = -jpl_q[0];
    // loc.q[1] = -jpl_q[1];
    // loc.q[2] = -jpl_q[2];
    loc.q[0] =  jpl_q[0];
    loc.q[1] =  jpl_q[1];
    loc.q[2] =  jpl_q[2];
    loc.q[3] =  jpl_q[3];

    loc.p[0] = pos[0];
    loc.p[1] = pos[1];
    loc.p[2] = pos[2];

    std::vector<std::shared_ptr<ov_type::Type>> variables;
    variables.push_back(output->state_clone->_imu->pose());
    auto cov = ov_msckf::StateHelper::get_marginal_covariance(output->state_clone, variables);

    loc.timestamp = output->status.timestamp;
    memcpy(loc.cov, cov.data(), sizeof(loc.cov));
  } else {
    Eigen::Matrix<double, 13, 1> state_plus;
    Eigen::Matrix<double, 12, 12> covariance;
    double output_time;
    bool propagate_ok = impl_->internal_->get_propagator()->fast_state_propagate(
        output->state_clone, -1.0, state_plus, covariance, &output_time);
    if (propagate_ok) {
      Eigen::Matrix<double, 4, 1> jpl_q = state_plus.block(0, 0, 4, 1);
      Eigen::Vector3d pos = state_plus.block(4, 0, 3, 1);
      // Eigen::Isometry3d imu_pose(ov_core::quat_2_Rot(jpl_q).inverse());
      // imu_pose.translation() = pos;
      // loc.pose = imu_pose;

      // loc.q[0] = -jpl_q[0];
      // loc.q[1] = -jpl_q[1];
      // loc.q[2] = -jpl_q[2];
      loc.q[0] =  jpl_q[0];
      loc.q[1] =  jpl_q[1];
      loc.q[2] =  jpl_q[2];
      loc.q[3] =  jpl_q[3];

      loc.p[0] = pos[0];
      loc.p[1] = pos[1];
      loc.p[2] = pos[2];

      Eigen::Matrix<double, 6, 6> cov = covariance.block(0, 0, 6, 6);
      loc.timestamp = output_time;
      memcpy(loc.cov, cov.data(), sizeof(loc.cov));
    }
  }

  return loc;
}


void VIO::Reset() {
  Shutdown();
  impl_->internal_.reset();
  impl_->internal_ = std::make_shared<ov_msckf::VioManager>(impl_->params_);
}

void VIO::Shutdown() {
  if (impl_) {
    if (impl_->internal_) {
      impl_->internal_->stop_threads();
    }
  }
}

}
