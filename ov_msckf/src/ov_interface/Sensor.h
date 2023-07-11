#ifndef OV_INTERFACE_SENSOR_H
#define OV_INTERFACE_SENSOR_H

#include <stdint.h>

namespace ov_interface {

// const int kImageWidth = 1280;
// const int kImageHeight = 720;
// const int kImageChannel = 3;
// const int kImageSize = kImageWidth * kImageHeight * kImageChannel;
// const int kMaxImageSize = kImageSize;
// const int kMaxImageSize = 2000 * 1000 * 3;
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
  double v[3];
  double b_g[3];
  double b_a[3];

  double cov[15*15];  // order:  q, p, v, b_g, b_a
  int64_t err;
};

} // namespace ov_interface

#endif // OV_INTERFACE_SENSOR_H
