
#include "HeisenbergVIO.h"

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


// extern std::shared_ptr<ov_msckf::VioManager> getVioManagerFromVioInterface(heisenberg_algo::VIO*);
std::shared_ptr<ov_msckf::VioManager> getVioManagerFromVioInterface(heisenberg_algo::VIO* vio) {
  return std::dynamic_pointer_cast<ov_msckf::VioManager>(vio->impl());
}

namespace heisenberg_algo {

class VIO::Impl : public ov_msckf::VioManager {
 public:
  Impl(ov_msckf::VioManagerOptions &params) : ov_msckf::VioManager(params) {}
};


VIO::VIO(const std::string& config_file) : config_file_(config_file), impl_(nullptr) {

}

VIO::~VIO() {
  Shutdown();
}

bool VIO::Init() {
  Reset();
  return true;
}

void VIO::ReceiveImu(const IMU_MSG &imu_msg) {
  if (!imu_msg.valid) {
    return;
  }

  const double* a = imu_msg.linear_acceleration;
  const double* w = imu_msg.angle_velocity;
  ov_core::ImuData message;
  message.timestamp = imu_msg.timestamp;
  message.wm = Eigen::Vector3d(w[0], w[1], w[2]);
  message.am = Eigen::Vector3d(a[0], a[1], a[2]);
  impl_->feed_measurement_imu(message);
}

void VIO::ReceiveWheel(const WHEEL_MSG &wheel_msg) {

}

void VIO::ReceiveGnss(const GNSS_MSG &gnss_msg) {

}

void VIO::ReceiveCamera(const IMG_MSG &img_msg) {
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

  if (impl_->get_params().use_mask) {
    message.masks.push_back(impl_->get_params().masks.at(cam_id0));
  } else {
    message.masks.push_back(cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1));
  }

  impl_->feed_measurement_camera(std::move(message));  // todo: run this in another thread?
}

void VIO::ReceiveStereoCamera(const STEREO_IMG_MSG &img_msg) {
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

  if (impl_->get_params().use_mask) {
    message.masks.push_back(impl_->get_params().masks.at(cam_id0));
    message.masks.push_back(impl_->get_params().masks.at(cam_id1));
  } else {
    message.masks.push_back(cv::Mat::zeros(gray_l.rows, gray_l.cols, CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(gray_r.rows, gray_r.cols, CV_8UC1));
  }

  impl_->feed_measurement_camera(std::move(message));  // todo: run this in another thread?
}

LOC_MSG VIO::Localization(double timestamp) {
  LOC_MSG loc;
  loc.timestamp = -1.0;
  loc.err = 0;
  auto output = impl_->getLastOutput(true, false);
  if (!output->status.initialized) {
    // vio has not been initilized yet.
    return loc;
  }

  bool predict_with_imu = true;

  if (!predict_with_imu || timestamp < 0 || timestamp < output->status.timestamp) {

    // Eigen::Isometry3d imu_pose(output->state_clone->_imu->Rot().inverse());
    // imu_pose.translation() = output->state_clone->_imu->pos();
    // loc.pose = imu_pose;
    auto jpl_q = output->state_clone->_imu->quat();
    auto pos = output->state_clone->_imu->pos();
    loc.q[0] = -jpl_q[0];
    loc.q[1] = -jpl_q[1];
    loc.q[2] = -jpl_q[2];
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
    bool propagate_ok = impl_->get_propagator()->fast_state_propagate(
        output->state_clone, timestamp, state_plus, covariance, &output_time);
    if (propagate_ok) {
      Eigen::Matrix<double, 4, 1> jpl_q = state_plus.block(0, 0, 4, 1);
      Eigen::Vector3d pos = state_plus.block(4, 0, 3, 1);
      // Eigen::Isometry3d imu_pose(ov_core::quat_2_Rot(jpl_q).inverse());
      // imu_pose.translation() = pos;
      // loc.pose = imu_pose;
      loc.q[0] = -jpl_q[0];
      loc.q[1] = -jpl_q[1];
      loc.q[2] = -jpl_q[2];
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
  impl_.reset();


  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_file_);

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
  impl_ = std::make_shared<Impl>(params);

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }
}

void VIO::Shutdown() {
  if (impl_) {
    impl_->stop_threads();
  }
}

}
