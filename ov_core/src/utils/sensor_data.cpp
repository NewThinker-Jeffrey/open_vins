/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2022 Patrick Geneva
 * Copyright (C) 2018-2022 Guoquan Huang
 * Copyright (C) 2018-2022 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include "sensor_data.h"
#include "print.h"
#include "colors.h"

namespace ov_core {

#ifdef USE_HEAR_SLAM

ImuData interpolate_data(const ImuData &imu_1, const ImuData &imu_2, double timestamp) {
  return hear_slam::Imu::interpolate(imu_1, imu_2, timestamp);
}

std::vector<ImuData> fill_imu_data_gaps(const std::vector<ImuData>& in_data, double max_gap) {
  return hear_slam::Imu::fillDataGaps(in_data, max_gap);

}

std::vector<ImuData> select_imu_readings(const std::vector<ImuData> &imu_data, double time0, double time1,
                                                  bool warn) {
  return hear_slam::Imu::selectBetween(imu_data, time0, time1, warn);
}

#else

ImuData interpolate_data(const ImuData &imu_1, const ImuData &imu_2, double timestamp) {
  // time-distance lambda
  double lambda = (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp);
  // PRINT_DEBUG("lambda - %d\n", lambda);
  // interpolate between the two times
  ImuData data;
  data.timestamp = timestamp;
  data.am = (1 - lambda) * imu_1.am + lambda * imu_2.am;
  data.wm = (1 - lambda) * imu_1.wm + lambda * imu_2.wm;
  return data;
}

std::vector<ImuData> fill_imu_data_gaps(const std::vector<ImuData>& in_data, double max_gap) {
  std::vector<ImuData> out_data;
  if (in_data.empty()) {
    return out_data;
  }

  // If "in_data" is of size 1, then so will be the out_data.

  for (size_t i = 0; i < in_data.size() - 1; i++) {
    out_data.push_back(in_data[i]);
    double t0 = in_data[i].timestamp;
    double t1 = in_data[i+1].timestamp;
    double gap = t1 - t0;
    if (gap > max_gap) {
      if (gap > 0.1) {
        PRINT_WARNING(YELLOW "fill_imu_data_gaps(): LARGE_IMU_GAP!! We're filling a large imu gap (%.3f) from %.3f to %.3f!!!\n" RESET, gap, t0, t1);
      }

      int to_fill = gap / max_gap;
      double interval = gap / (to_fill + 1);
      for (int k = 0; k < to_fill; k++) {
        double t = t0 + (k + 1) * interval;
        assert(t < t1 - 0.5 * interval);
        out_data.push_back(interpolate_data(in_data[i], in_data[i+1], t));
      }
    }
  }

  out_data.push_back(in_data.back());
  return out_data;
}

std::vector<ImuData> select_imu_readings(const std::vector<ImuData> &imu_data, double time0, double time1,
                                                  bool warn) {

  // Our vector imu readings
  std::vector<ov_core::ImuData> prop_data;

  // Ensure we have some measurements in the first place!
  if (imu_data.empty()) {
    if (warn)
      PRINT_WARNING(YELLOW "select_imu_readings(): No IMU measurements. IMU-CAMERA are likely messed up!!!\n" RESET);
    return prop_data;
  }

  // Loop through and find all the needed measurements to propagate with
  // Note we split measurements based on the given state time, and the update timestamp
  for (size_t i = 0; i < imu_data.size() - 1; i++) {

    // START OF THE INTEGRATION PERIOD
    // If the next timestamp is greater then our current state time
    // And the current is not greater then it yet...
    // Then we should "split" our current IMU measurement
    if (imu_data.at(i + 1).timestamp > time0 && imu_data.at(i).timestamp < time0) {
      ov_core::ImuData data = interpolate_data(imu_data.at(i), imu_data.at(i + 1), time0);
      prop_data.push_back(data);
      // PRINT_DEBUG("propagation #%d = CASE 1 = %.3f => %.3f\n", (int)i, data.timestamp - prop_data.at(0).timestamp,
      //             time0 - prop_data.at(0).timestamp);
      continue;
    }

    // MIDDLE OF INTEGRATION PERIOD
    // If our imu measurement is right in the middle of our propagation period
    // Then we should just append the whole measurement time to our propagation vector
    if (imu_data.at(i).timestamp >= time0 && imu_data.at(i + 1).timestamp <= time1) {
      prop_data.push_back(imu_data.at(i));
      // PRINT_DEBUG("propagation #%d = CASE 2 = %.3f\n", (int)i, imu_data.at(i).timestamp - prop_data.at(0).timestamp);
      continue;
    }

    // END OF THE INTEGRATION PERIOD
    // If the current timestamp is greater then our update time
    // We should just "split" the NEXT IMU measurement to the update time,
    // NOTE: we add the current time, and then the time at the end of the interval (so we can get a dt)
    // NOTE: we also break out of this loop, as this is the last IMU measurement we need!
    if (imu_data.at(i + 1).timestamp > time1) {
      // If we have a very low frequency IMU then, we could have only recorded the first integration (i.e. case 1) and nothing else
      // In this case, both the current IMU measurement and the next is greater than the desired intepolation, thus we should just cut the
      // current at the desired time Else, we have hit CASE2 and this IMU measurement is not past the desired propagation time, thus add the
      // whole IMU reading
      if (imu_data.at(i).timestamp > time1 && i == 0) {
        // This case can happen if we don't have any imu data that has occured before the startup time
        // This means that either we have dropped IMU data, or we have not gotten enough.
        // In this case we can't propgate forward in time, so there is not that much we can do.
        break;
      } else if (imu_data.at(i).timestamp > time1) {
        ov_core::ImuData data = interpolate_data(imu_data.at(i - 1), imu_data.at(i), time1);
        prop_data.push_back(data);
        // PRINT_DEBUG("propagation #%d = CASE 3.1 = %.3f => %.3f\n",
        // (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
      } else {
        prop_data.push_back(imu_data.at(i));
        // PRINT_DEBUG("propagation #%d = CASE 3.2 = %.3f => %.3f\n",
        // (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
      }
      // If the added IMU message doesn't end exactly at the camera time
      // Then we need to add another one that is right at the ending time
      if (prop_data.at(prop_data.size() - 1).timestamp != time1) {
        ov_core::ImuData data = interpolate_data(imu_data.at(i), imu_data.at(i + 1), time1);
        prop_data.push_back(data);
        // PRINT_DEBUG("propagation #%d = CASE 3.3 = %.3f => %.3f\n", (int)i,data.timestamp-prop_data.at(0).timestamp,data.timestamp-time0);
      }
      break;
    }
  }

  // Check that we have at least one measurement to propagate with
  if (prop_data.empty()) {
    if (warn)
      PRINT_WARNING(
          YELLOW
          "select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET,
          (int)prop_data.size());
    return prop_data;
  }

  // If we did not reach the whole integration period (i.e., the last inertial measurement we have is smaller then the time we want to
  // reach) Then we should just "stretch" the last measurement to be the whole period (case 3 in the above loop)
  // if(time1-imu_data.at(imu_data.size()-1).timestamp > 1e-3) {
  //    PRINT_DEBUG(YELLOW "select_imu_readings(): Missing inertial measurements to propagate with (%.6f sec missing).
  //    IMU-CAMERA are likely messed up!!!\n" RESET, (time1-imu_data.at(imu_data.size()-1).timestamp)); return prop_data;
  //}

  // Loop through and ensure we do not have an zero dt values
  // This would cause the noise covariance to be Infinity
  for (size_t i = 0; i < prop_data.size() - 1; i++) {
    if (std::abs(prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp) < 1e-12) {
      if (warn)
        PRINT_WARNING(YELLOW "select_imu_readings(): Zero DT between IMU reading %d and %d, removing it!\n" RESET, (int)i,
                      (int)(i + 1));
      prop_data.erase(prop_data.begin() + i);
      i--;
    }
  }

  // Check that we have at least one measurement to propagate with
  if (prop_data.size() < 2) {
    if (warn)
      PRINT_WARNING(
          YELLOW
          "select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET,
          (int)prop_data.size());
    return prop_data;
  }

  // Success :D
  return prop_data;
}
#endif  // USE_HEAR_SLAM

}  // namespace ov_core