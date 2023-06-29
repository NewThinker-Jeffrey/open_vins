#ifndef HEISENBERG_VIO_H
#define HEISENBERG_VIO_H

#include "HeisenbergSensor.h"

namespace heisenberg_algo {

class VIO {

public:
  VIO(const char* config_file = "");
  ~VIO();

  bool Init();
  void ReceiveImu(const IMU_MSG &imu_msg);
  void ReceiveWheel(const WHEEL_MSG &wheel_msg);
  void ReceiveGnss(const GNSS_MSG &gnss_msg);
  void ReceiveCamera(const IMG_MSG &img_msg);
  void ReceiveStereoCamera(const STEREO_IMG_MSG &img_msg);
  LOC_MSG Localization();
  void Reset();
  void Shutdown();

  IMU_MSG GetLatestIMU();

public:
  class Impl;

  // for internal debug
  Impl* impl() {return impl_;}

private:
  Impl* impl_;
};

} // namespace heisenberg_algo

#endif // HEISENBERG_VIO_H
