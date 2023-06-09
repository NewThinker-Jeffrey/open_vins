#ifndef HEISENBERG_VIO_H
#define HEISENBERG_VIO_H

#include "HeisenbergSensor.h"

#include <memory>
#include <string>

namespace heisenberg_algo {

class VIO {

public:
  VIO(const std::string& config_file = "");
  ~VIO();

  bool Initial();
  void ReceiveImu(const IMU_MSG &imu_msg);
  void ReceiveWheel(const WHEEL_MSG &wheel_msg);
  void ReceiveGnss(const GNSS_MSG &gnss_msg);
  void ReceiveCamera(const IMG_MSG &img_msg);
  void ReceiveStereoCamera(const STEREO_IMG_MSG &img_msg);
  LOC_MSG Localization(double timestamp);
  void Reset();
  void Shutdown();

public:
  class Impl;

  // for internal debug
  std::shared_ptr<Impl> impl() {return impl_;}

private:
  std::shared_ptr<Impl> impl_;
  std::string config_file_;
};

} // namespace heisenberg_algo

#endif // HEISENBERG_VIO_H
