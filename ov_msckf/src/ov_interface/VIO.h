#ifndef OV_INTERFACE_VIO_H
#define OV_INTERFACE_VIO_H

#include "Sensor.h"

namespace ov_interface {

class VIO {

public:
  VIO(const char* config_file = "");
  ~VIO();

  bool Init();
  void ReceiveImu(const IMU_MSG &imu_msg);
  void ReceiveCamera(const IMG_MSG &img_msg);
  void ReceiveStereoCamera(const STEREO_IMG_MSG &img_msg);
  LOC_MSG Localization();
  void Reset();
  void Shutdown();

public:
  class Impl;

  // for internal debug
  Impl* impl() {return impl_;}

private:
  Impl* impl_;
};

} // namespace ov_interface

#endif // OV_INTERFACE_VIO_H
