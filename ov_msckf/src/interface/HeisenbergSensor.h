#ifndef HEISENBERG_SENSOR_H
#define HEISENBERG_SENSOR_H

#include <stdint.h>
#include <Eigen/Geometry>


namespace heisenberg_algo {

// const int kImageWidth = 1280;
// const int kImageHeight = 720;
// const int kImageChannel = 3;
// const int kImageSize = kImageWidth * kImageHeight * kImageChannel;
// const int kMaxImageSize = kImageSize;
// const int kMaxImageSize = 2000 * 1000 * 3;
const int kMaxImageSize = 1000 * 1000 * 3;

struct GNSS_MSG {
    double timestamp; // us
    double longitude;
    double latitude;
    double altitude;
    double velocity;
    bool valid;
};

struct IMU_MSG {
    double timestamp;
    double linear_acceleration[3];
    double angle_velocity[3];
    bool valid;
};

struct WHEEL_MSG {
    double timestamp;
    double wheel_velocity[4];
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

} // namespace heisenberg_algo

#endif // HEISENBERG_SENSOR_H
