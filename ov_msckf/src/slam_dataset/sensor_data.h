#ifndef SLAM_DATASET_SENSOR_DATA_H
#define SLAM_DATASET_SENSOR_DATA_H

#ifdef USE_HEAR_SLAM

#include "hear_slam/common/datasource/vi_source.h"
#include "hear_slam/common/datasource/vi_recorder.h"
#include "hear_slam/common/datasource/vi_player.h"
#include "hear_slam/common/datasource/rs/rs_capture.h"

namespace slam_dataset {
  using namespace hear_slam;
  using ViCapture = ViDatasource;
}

#else
#include "utils/sensor_data.h"

namespace slam_dataset {

using ImuData = ov_core::ImuData;
using CameraData = ov_core::CameraData;

} // namespace slam_dataset
#endif // USE_HEAR_SLAM

#endif // SLAM_DATASET_SENSOR_DATA_H