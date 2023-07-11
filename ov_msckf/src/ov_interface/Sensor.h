#ifndef OV_INTERFACE_SENSOR_H
#define OV_INTERFACE_SENSOR_H

#include <stdint.h>

namespace ov_interface {

const int kMaxImageSize = 1000 * 1000 * 3;

struct IMU_MSG {
    double timestamp;
    double linear_acceleration[3];
    double angle_velocity[3];
    bool valid;
};

struct IMG_MSG {
    double timestamp;
    int cam_id;
    int width;
    int height;
    int channel;
    int flags;
    int size;
    char data[kMaxImageSize];
    bool valid;
};

struct STEREO_IMG_MSG {
    double timestamp;
    int cam_id_left;
    int cam_id_right;
    int width;
    int height;
    int channel;
    int flags;
    int size;
    char data_l[kMaxImageSize];
    char data_r[kMaxImageSize];
    bool valid;
};

struct LOC_MSG {
  double timestamp;
  double q[4];  // in Hamilton convention. memory order = xyzw (consistent with Eigen::Quaterniond::coeffs()) 
  double p[3];
  double cov[36];
  int64_t err;
};

} // namespace ov_interface

#endif // OV_INTERFACE_SENSOR_H
