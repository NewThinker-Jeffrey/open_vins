#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>

#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>

#include <memory>
#include <rclcpp/rclcpp.hpp>


class HeisenbergExportImu {

public:
  HeisenbergExportImu(std::shared_ptr<rclcpp::Node> node) : _node(node) {
    if (_node->has_parameter("heisenberg_dataset")) {
      std::string heisenberg_dataset;
      _node->get_parameter<std::string>("heisenberg_dataset", heisenberg_dataset);
      std::string imu_file = heisenberg_dataset + "/imu.txt";
      imu_ofs.reset(new std::ofstream(imu_file));
      (*imu_ofs) << "#ts gx gy gz ax ay az" << std::endl;
      setup_subscribers();
      std::cout << "saving imu data to " << imu_file << " ... "<< std::endl;
    }
  }

  /**
   * @brief Will setup ROS subscribers and callbacks
   * @param parser Configuration file parser
   */
  void setup_subscribers() {
    // Create imu subscriber (handle legacy ros param info)
    std::string topic_imu = "/hps/robot/imu";
    sub_imu = _node->create_subscription<sensor_msgs::msg::Imu>(topic_imu, rclcpp::SensorDataQoS(),
                                                                std::bind(&HeisenbergExportImu::callback_inertial, this, std::placeholders::_1));
    std::cout << "subscribing to IMU: " << topic_imu << std::endl;
  }

  /// Callback for inertial information
  void callback_inertial(const sensor_msgs::msg::Imu::SharedPtr msg) {

    int64_t imu_ts = msg->header.stamp.sec;
    imu_ts = imu_ts * 1000000000 + msg->header.stamp.nanosec;
    auto gyr = msg->angular_velocity;
    auto acc = msg->linear_acceleration;
    if (imu_ofs) {
      (*imu_ofs) << imu_ts << " "
                << gyr.x << " " << gyr.y << " " << gyr.z << " "
                << acc.x << " " << acc.y << " " << acc.z << std::endl;
    }
  }


protected:

  /// Global node handler
  std::shared_ptr<rclcpp::Node> _node;

  // Our subscribers and camera synchronizers
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;

  std::shared_ptr<std::ofstream> imu_ofs;
};


// Main function
int main(int argc, char **argv) {

  // Launch our ros node
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("heisenberg_export_imu", options);

  // Load the config
  // node->get_parameter<std::string>("config_path", config_path);
  // auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  // parser->set_node(node);

  // // Verbosity
  // std::string verbosity = "DEBUG";
  // parser->parse_config("verbosity", verbosity);
  // ov_core::Printer::setPrintLevel(verbosity);


  auto app = std::make_shared<HeisenbergExportImu>(node);
  rclcpp::spin(node);
  rclcpp::shutdown();

  // Done!
  return EXIT_SUCCESS;
}
