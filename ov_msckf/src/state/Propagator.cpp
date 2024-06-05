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

#include "Propagator.h"

#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/quat_ops.h"
#include <Eigen/Geometry>

#ifdef USE_HEAR_SLAM
#include "hear_slam/basic/logging.h"
#include "hear_slam/basic/time.h"
#endif

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;


void Propagator::propagate_and_clone(std::shared_ptr<State> state, double timestamp, Eigen::Matrix3d* output_rotation) {

  // If the difference between the current update time and state is zero
  // We should crash, as this means we would have two clones at the same time!!!!
  if (state->_timestamp == timestamp) {
    PRINT_ERROR(RED "Propagator::propagate_and_clone(): Propagation called again at same timestep at last update timestep!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // We should crash if we are trying to propagate backwards
  if (state->_timestamp > timestamp) {
    PRINT_ERROR(RED "Propagator::propagate_and_clone(): Propagation called trying to propagate backwards in time!!!!\n" RESET);
    PRINT_ERROR(RED "Propagator::propagate_and_clone(): desired propagation = %.4f\n" RESET, (timestamp - state->_timestamp));
    std::exit(EXIT_FAILURE);
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Set the last time offset value if we have just started the system up
  if (!have_last_prop_time_offset) {
    last_prop_time_offset = state->_calib_dt_CAMtoIMU->value()(0);
    have_last_prop_time_offset = true;
  }

  // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
  double t_off_new = state->_calib_dt_CAMtoIMU->value()(0);

  // First lets construct an IMU vector of measurements we need
  double time0 = state->_timestamp + last_prop_time_offset;
  double time1 = timestamp + t_off_new;
  std::vector<ov_core::ImuData> prop_data;
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    prop_data = ov_core::select_imu_readings(imu_data, time0, time1);
  }

  prop_data = fill_imu_data_gaps(prop_data);

  if (output_rotation) {
    if (prop_data.size() < 2) {
      PRINT_WARNING(YELLOW "propagate_and_clone::output_rotation: no prop_data!\n" RESET);
      *output_rotation = Eigen::Matrix3d::Identity();
    } else {
      Eigen::Quaterniond q(1,0,0,0);
      for (size_t i=0; i<prop_data.size()-1; i++) {
        const auto & d0 = prop_data[i];
        const auto & d1 = prop_data[i+1];
        double dt = d1.timestamp - d0.timestamp;
        Eigen::Vector3d w_ave = 0.5 * (d0.wm + d1.wm) - state->_imu->bias_g();
        Eigen::Vector3d im = 0.5 * w_ave * dt;    
        Eigen::Quaterniond dq(1, im.x(), im.y(), im.z());
        q = q * dq;
      }
      q.normalize();
      *output_rotation = q.toRotationMatrix();
    }
  }

  // We are going to sum up all the state transition matrices, so we can do a single large multiplication at the end
  // Phi_summed = Phi_i*Phi_summed
  // Q_summed = Phi_i*Q_summed*Phi_i^T + Q_i
  // After summing we can multiple the total phi to get the updated covariance
  // We will then add the noise to the IMU portion of the state
  Eigen::Matrix<double, 15, 15> Phi_summed = Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::Matrix<double, 15, 15> Qd_summed = Eigen::Matrix<double, 15, 15>::Zero();
  double dt_summed = 0;

  // Loop through all IMU messages, and use them to move the state forward in time
  // This uses the zero'th order quat, and then constant acceleration discrete
  if (prop_data.size() > 1) {
    for (size_t i = 0; i < prop_data.size() - 1; i++) {

      // Get the next state Jacobian and noise Jacobian for this IMU reading
      Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Zero();
      Eigen::Matrix<double, 15, 15> Qdi = Eigen::Matrix<double, 15, 15>::Zero();
      predict_and_compute(state, prop_data.at(i), prop_data.at(i + 1), F, Qdi);

      // Next we should propagate our IMU covariance
      // Pii' = F*Pii*F.transpose() + G*Q*G.transpose()
      // Pci' = F*Pci and Pic' = Pic*F.transpose()
      // NOTE: Here we are summing the state transition F so we can do a single mutiplication later
      // NOTE: Phi_summed = Phi_i*Phi_summed
      // NOTE: Q_summed = Phi_i*Q_summed*Phi_i^T + G*Q_i*G^T
      Phi_summed = F * Phi_summed;
      Qd_summed = F * Qd_summed * F.transpose() + Qdi;
      Qd_summed = 0.5 * (Qd_summed + Qd_summed.transpose());
      dt_summed += prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp;
    }
  }

  if (std::abs((time1 - time0) - dt_summed) < 1e-4) {
    std::cout << "time1 - time0 = " << time1 - time0 << ",  dt_summed = " << dt_summed << ", " << "a-b= " << (time1 - time0) - dt_summed << std::endl;
    std::cout << "prop_data.size() = " << prop_data.size() << ",  prop_data_dt = " << prop_data.back().timestamp - prop_data.front().timestamp << std::endl;
    assert(std::abs((time1 - time0) - dt_summed) < 1e-4);
  }


  // Last angular velocity (used for cloning when estimating time offset)
  Eigen::Matrix<double, 3, 1> last_w = Eigen::Matrix<double, 3, 1>::Zero();
  if (prop_data.size() > 1)
    last_w = prop_data.at(prop_data.size() - 2).wm - state->_imu->bias_g();
  else if (!prop_data.empty())
    last_w = prop_data.at(prop_data.size() - 1).wm - state->_imu->bias_g();

  // Do the update to the covariance with our "summed" state transition and IMU noise addition...
  std::vector<std::shared_ptr<Type>> Phi_order;
  Phi_order.push_back(state->_imu);
  StateHelper::EKFPropagation(state, Phi_order, Phi_order, Phi_summed, Qd_summed);

  // Set timestamp data
  state->_timestamp = timestamp;
  last_prop_time_offset = t_off_new;

  // Now perform stochastic cloning
  StateHelper::augment_clone(state, last_w);
}

bool Propagator::fast_state_propagate(std::shared_ptr<State> state, double timestamp, Eigen::Matrix<double, 13, 1> &state_plus,
                                      Eigen::Matrix<double, 12, 12> &covariance, double* output_timestamp) {
  // First we will store the current calibration / estimates of the state
  double state_time = state->_timestamp;
  Eigen::MatrixXd state_est = state->_imu->value();
  Eigen::MatrixXd state_covariance = StateHelper::get_marginal_covariance(state, {state->_imu});
  double t_off = state->_calib_dt_CAMtoIMU->value()(0);

  // First lets construct an IMU vector of measurements we need
  std::vector<ov_core::ImuData> prop_data;
  {
    double time0 = state_time + t_off;
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    double time1;
    if (timestamp < 0) {
      time1 = imu_data.back().timestamp;
    } else {
      time1 = timestamp + t_off;
    }
    prop_data = ov_core::select_imu_readings(imu_data, time0, time1, false);
  }
  if (prop_data.size() < 2)
    return false;

  prop_data = fill_imu_data_gaps(prop_data);
  if (output_timestamp) {
    *output_timestamp = prop_data.back().timestamp - t_off;
  }
  

  // Biases
  Eigen::Vector3d bias_g = state_est.block(10, 0, 3, 1);
  Eigen::Vector3d bias_a = state_est.block(13, 0, 3, 1);

  // Loop through all IMU messages, and use them to move the state forward in time
  // This uses the zero'th order quat, and then constant acceleration discrete
  for (size_t i = 0; i < prop_data.size() - 1; i++) {

    // Corrected imu measurements
    double dt = prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp;
    Eigen::Vector3d w_hat = 0.5 * (prop_data.at(i + 1).wm + prop_data.at(i).wm) - bias_g;
    Eigen::Vector3d a_hat = 0.5 * (prop_data.at(i + 1).am + prop_data.at(i).am) - bias_a;
    Eigen::Matrix3d R_Gtoi = quat_2_Rot(state_est.block(0, 0, 4, 1));
    Eigen::Vector3d v_iinG = state_est.block(7, 0, 3, 1);
    Eigen::Vector3d p_iinG = state_est.block(4, 0, 3, 1);

    // State transition and noise matrix
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Zero();
    Eigen::Matrix<double, 15, 15> Qd = Eigen::Matrix<double, 15, 15>::Zero();
    F.block(0, 0, 3, 3) = exp_so3(-w_hat * dt);
    F.block(0, 9, 3, 3).noalias() = -exp_so3(-w_hat * dt) * Jr_so3(-w_hat * dt) * dt;
    F.block(9, 9, 3, 3).setIdentity();
    F.block(6, 0, 3, 3).noalias() = -R_Gtoi.transpose() * skew_x(a_hat * dt);
    F.block(6, 6, 3, 3).setIdentity();
    F.block(6, 12, 3, 3) = -R_Gtoi.transpose() * dt;
    F.block(12, 12, 3, 3).setIdentity();
    F.block(3, 0, 3, 3).noalias() = -0.5 * R_Gtoi.transpose() * skew_x(a_hat * dt * dt);
    F.block(3, 6, 3, 3) = Eigen::Matrix3d::Identity() * dt;
    F.block(3, 12, 3, 3) = -0.5 * R_Gtoi.transpose() * dt * dt;
    F.block(3, 3, 3, 3).setIdentity();
    Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
    G.block(0, 0, 3, 3) = -exp_so3(-w_hat * dt) * Jr_so3(-w_hat * dt) * dt;
    G.block(6, 3, 3, 3) = -R_Gtoi.transpose() * dt;
    G.block(3, 3, 3, 3) = -0.5 * R_Gtoi.transpose() * dt * dt;
    G.block(9, 6, 3, 3).setIdentity();
    G.block(12, 9, 3, 3).setIdentity();

    // Construct our discrete noise covariance matrix
    // Note that we need to convert our continuous time noises to discrete
    // Equations (129) amd (130) of Trawny tech report
    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    Qc.block(0, 0, 3, 3) = _noises.sigma_w_2 / dt * Eigen::Matrix3d::Identity();
    Qc.block(3, 3, 3, 3) = _noises.sigma_a_2 / dt * Eigen::Matrix3d::Identity();
    Qc.block(6, 6, 3, 3) = _noises.sigma_wb_2 * dt * Eigen::Matrix3d::Identity();
    Qc.block(9, 9, 3, 3) = _noises.sigma_ab_2 * dt * Eigen::Matrix3d::Identity();
    Qd = G * Qc * G.transpose();
    Qd = 0.5 * (Qd + Qd.transpose());
    state_covariance = F * state_covariance * F.transpose() + Qd;

    // Propagate the mean forward
    state_est.block(0, 0, 4, 1) = rot_2_quat(exp_so3(-w_hat * dt) * R_Gtoi);
    state_est.block(4, 0, 3, 1) = p_iinG + v_iinG * dt + 0.5 * R_Gtoi.transpose() * a_hat * dt * dt - 0.5 * _gravity * dt * dt;
    state_est.block(7, 0, 3, 1) = v_iinG + R_Gtoi.transpose() * a_hat * dt - _gravity * dt;
  }

  // Now record what the predicted state should be
  Eigen::Vector4d q_Gtoi = state_est.block(0, 0, 4, 1);
  Eigen::Vector3d v_iinG = state_est.block(7, 0, 3, 1);
  Eigen::Vector3d p_iinG = state_est.block(4, 0, 3, 1);
  state_plus.setZero();
  state_plus.block(0, 0, 4, 1) = q_Gtoi;
  state_plus.block(4, 0, 3, 1) = p_iinG;
  state_plus.block(7, 0, 3, 1) = quat_2_Rot(q_Gtoi) * v_iinG;
  state_plus.block(10, 0, 3, 1) = 0.5 * (prop_data.at(prop_data.size() - 1).wm + prop_data.at(prop_data.size() - 2).wm) - bias_g;

  // Do a covariance propagation for our velocity
  // TODO: more properly do the covariance of the angular velocity here...
  // TODO: it should be dependent on the state bias, thus correlated with the pose
  covariance.setZero();
  Eigen::Matrix<double, 15, 15> Phi = Eigen::Matrix<double, 15, 15>::Identity();
  Phi.block(6, 6, 3, 3) = quat_2_Rot(q_Gtoi);
  state_covariance = Phi * state_covariance * Phi.transpose();
  covariance.block(0, 0, 9, 9) = state_covariance.block(0, 0, 9, 9);
  double dt = prop_data.at(prop_data.size() - 1).timestamp - prop_data.at(prop_data.size() - 2).timestamp;
  covariance.block(9, 9, 3, 3) = _noises.sigma_w_2 / dt * Eigen::Matrix3d::Identity();
  return true;
}

void Propagator::predict_and_compute(std::shared_ptr<State> state, const ov_core::ImuData &data_minus, const ov_core::ImuData &data_plus,
                                     Eigen::Matrix<double, 15, 15> &F, Eigen::Matrix<double, 15, 15> &Qd) {

  // Set them to zero
  F.setZero();
  Qd.setZero();

  // Time elapsed over interval
  double dt = data_plus.timestamp - data_minus.timestamp;
  // assert(data_plus.timestamp>data_minus.timestamp);

  // Corrected imu measurements
  Eigen::Matrix<double, 3, 1> w_hat = data_minus.wm - state->_imu->bias_g();
  Eigen::Matrix<double, 3, 1> a_hat = data_minus.am - state->_imu->bias_a();
  Eigen::Matrix<double, 3, 1> w_hat2 = data_plus.wm - state->_imu->bias_g();
  Eigen::Matrix<double, 3, 1> a_hat2 = data_plus.am - state->_imu->bias_a();

  // Compute the new state mean value
  Eigen::Vector4d new_q;
  Eigen::Vector3d new_v, new_p;
  if (state->_options.use_rk4_integration)
    predict_mean_rk4(state, dt, w_hat, a_hat, w_hat2, a_hat2, new_q, new_v, new_p);
  else
    predict_mean_discrete(state, dt, w_hat, a_hat, w_hat2, a_hat2, new_q, new_v, new_p);

  // Get the locations of each entry of the imu state
  int th_id = state->_imu->q()->id() - state->_imu->id();
  int p_id = state->_imu->p()->id() - state->_imu->id();
  int v_id = state->_imu->v()->id() - state->_imu->id();
  int bg_id = state->_imu->bg()->id() - state->_imu->id();
  int ba_id = state->_imu->ba()->id() - state->_imu->id();

  // Allocate noise Jacobian
  Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();

  // Now compute Jacobian of new state wrt old state and noise
  if (state->_options.do_fej) {

    // This is the change in the orientation from the end of the last prop to the current prop
    // This is needed since we need to include the "k-th" updated orientation information
    Eigen::Matrix<double, 3, 3> Rfej = state->_imu->Rot_fej();
    Eigen::Matrix<double, 3, 3> dR = quat_2_Rot(new_q) * Rfej.transpose();

    Eigen::Matrix<double, 3, 1> v_fej = state->_imu->vel_fej();
    Eigen::Matrix<double, 3, 1> p_fej = state->_imu->pos_fej();

    F.block(th_id, th_id, 3, 3) = dR;
    F.block(th_id, bg_id, 3, 3).noalias() = -dR * Jr_so3(-w_hat * dt) * dt;
    // F.block(th_id, bg_id, 3, 3).noalias() = -dR * Jr_so3(-log_so3(dR)) * dt;
    F.block(bg_id, bg_id, 3, 3).setIdentity();
    F.block(v_id, th_id, 3, 3).noalias() = -skew_x(new_v - v_fej + _gravity * dt) * Rfej.transpose();
    // F.block(v_id, th_id, 3, 3).noalias() = -Rfej.transpose() * skew_x(Rfej*(new_v-v_fej+_gravity*dt));
    F.block(v_id, v_id, 3, 3).setIdentity();
    F.block(v_id, ba_id, 3, 3) = -Rfej.transpose() * dt;
    F.block(ba_id, ba_id, 3, 3).setIdentity();
    F.block(p_id, th_id, 3, 3).noalias() = -skew_x(new_p - p_fej - v_fej * dt + 0.5 * _gravity * dt * dt) * Rfej.transpose();
    // F.block(p_id, th_id, 3, 3).noalias() = -0.5 * Rfej.transpose() * skew_x(2*Rfej*(new_p-p_fej-v_fej*dt+0.5*_gravity*dt*dt));
    F.block(p_id, v_id, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * dt;
    F.block(p_id, ba_id, 3, 3) = -0.5 * Rfej.transpose() * dt * dt;
    F.block(p_id, p_id, 3, 3).setIdentity();

    G.block(th_id, 0, 3, 3) = -dR * Jr_so3(-w_hat * dt) * dt;
    // G.block(th_id, 0, 3, 3) = -dR * Jr_so3(-log_so3(dR)) * dt;
    G.block(v_id, 3, 3, 3) = -Rfej.transpose() * dt;
    G.block(p_id, 3, 3, 3) = -0.5 * Rfej.transpose() * dt * dt;
    G.block(bg_id, 6, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
    G.block(ba_id, 9, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();

  } else {

    Eigen::Matrix<double, 3, 3> R_Gtoi = state->_imu->Rot();

    F.block(th_id, th_id, 3, 3) = exp_so3(-w_hat * dt);
    F.block(th_id, bg_id, 3, 3).noalias() = -exp_so3(-w_hat * dt) * Jr_so3(-w_hat * dt) * dt;
    F.block(bg_id, bg_id, 3, 3).setIdentity();
    F.block(v_id, th_id, 3, 3).noalias() = -R_Gtoi.transpose() * skew_x(a_hat * dt);
    F.block(v_id, v_id, 3, 3).setIdentity();
    F.block(v_id, ba_id, 3, 3) = -R_Gtoi.transpose() * dt;
    F.block(ba_id, ba_id, 3, 3).setIdentity();
    F.block(p_id, th_id, 3, 3).noalias() = -0.5 * R_Gtoi.transpose() * skew_x(a_hat * dt * dt);
    F.block(p_id, v_id, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * dt;
    F.block(p_id, ba_id, 3, 3) = -0.5 * R_Gtoi.transpose() * dt * dt;
    F.block(p_id, p_id, 3, 3).setIdentity();

    G.block(th_id, 0, 3, 3) = -exp_so3(-w_hat * dt) * Jr_so3(-w_hat * dt) * dt;
    G.block(v_id, 3, 3, 3) = -R_Gtoi.transpose() * dt;
    G.block(p_id, 3, 3, 3) = -0.5 * R_Gtoi.transpose() * dt * dt;
    G.block(bg_id, 6, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
    G.block(ba_id, 9, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
  }

  // Construct our discrete noise covariance matrix
  // Note that we need to convert our continuous time noises to discrete
  // Equations (129) amd (130) of Trawny tech report
  Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
  Qc.block(0, 0, 3, 3) = _noises.sigma_w_2 / dt * Eigen::Matrix<double, 3, 3>::Identity();
  Qc.block(3, 3, 3, 3) = _noises.sigma_a_2 / dt * Eigen::Matrix<double, 3, 3>::Identity();
  Qc.block(6, 6, 3, 3) = _noises.sigma_wb_2 * dt * Eigen::Matrix<double, 3, 3>::Identity();
  Qc.block(9, 9, 3, 3) = _noises.sigma_ab_2 * dt * Eigen::Matrix<double, 3, 3>::Identity();

  // Compute the noise injected into the state over the interval
  Qd = G * Qc * G.transpose();
  Qd = 0.5 * (Qd + Qd.transpose());

  // Now replace imu estimate and fej with propagated values
  Eigen::Matrix<double, 16, 1> imu_x = state->_imu->value();
  imu_x.block(0, 0, 4, 1) = new_q;
  imu_x.block(4, 0, 3, 1) = new_p;
  imu_x.block(7, 0, 3, 1) = new_v;
  state->_imu->set_value(imu_x);
  state->_imu->set_fej(imu_x);
}

void Propagator::predict_mean_discrete(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat1,
                                       const Eigen::Vector3d &a_hat1, const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2,
                                       Eigen::Vector4d &new_q, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p) {

  // If we are averaging the IMU, then do so
  Eigen::Vector3d w_hat = w_hat1;
  Eigen::Vector3d a_hat = a_hat1;
  if (state->_options.imu_avg) {
    w_hat = .5 * (w_hat1 + w_hat2);
    a_hat = .5 * (a_hat1 + a_hat2);
  }

  // Pre-compute things
  double w_norm = w_hat.norm();
  Eigen::Matrix<double, 4, 4> I_4x4 = Eigen::Matrix<double, 4, 4>::Identity();
  Eigen::Matrix<double, 3, 3> R_Gtoi = state->_imu->Rot();

  // Orientation: Equation (101) and (103) and of Trawny indirect TR
  Eigen::Matrix<double, 4, 4> bigO;
  if (w_norm > 1e-20) {
    bigO = cos(0.5 * w_norm * dt) * I_4x4 + 1 / w_norm * sin(0.5 * w_norm * dt) * Omega(w_hat);
  } else {
    bigO = I_4x4 + 0.5 * dt * Omega(w_hat);
  }
  new_q = quatnorm(bigO * state->_imu->quat());
  // new_q = rot_2_quat(exp_so3(-w_hat*dt)*R_Gtoi);

  // Velocity: just the acceleration in the local frame, minus global gravity
  new_v = state->_imu->vel() + R_Gtoi.transpose() * a_hat * dt - _gravity * dt;

  // Position: just velocity times dt, with the acceleration integrated twice
  new_p = state->_imu->pos() + state->_imu->vel() * dt + 0.5 * R_Gtoi.transpose() * a_hat * dt * dt - 0.5 * _gravity * dt * dt;
}

void Propagator::predict_mean_rk4(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat1, const Eigen::Vector3d &a_hat1,
                                  const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2, Eigen::Vector4d &new_q,
                                  Eigen::Vector3d &new_v, Eigen::Vector3d &new_p) {

  // Pre-compute things
  Eigen::Vector3d w_hat = w_hat1;
  Eigen::Vector3d a_hat = a_hat1;
  Eigen::Vector3d w_alpha = (w_hat2 - w_hat1) / dt;
  Eigen::Vector3d a_jerk = (a_hat2 - a_hat1) / dt;

  // y0 ================
  Eigen::Vector4d q_0 = state->_imu->quat();
  Eigen::Vector3d p_0 = state->_imu->pos();
  Eigen::Vector3d v_0 = state->_imu->vel();

  // k1 ================
  Eigen::Vector4d dq_0 = {0, 0, 0, 1};
  Eigen::Vector4d q0_dot = 0.5 * Omega(w_hat) * dq_0;
  Eigen::Vector3d p0_dot = v_0;
  Eigen::Matrix3d R_Gto0 = quat_2_Rot(quat_multiply(dq_0, q_0));
  Eigen::Vector3d v0_dot = R_Gto0.transpose() * a_hat - _gravity;

  Eigen::Vector4d k1_q = q0_dot * dt;
  Eigen::Vector3d k1_p = p0_dot * dt;
  Eigen::Vector3d k1_v = v0_dot * dt;

  // k2 ================
  w_hat += 0.5 * w_alpha * dt;
  a_hat += 0.5 * a_jerk * dt;

  Eigen::Vector4d dq_1 = quatnorm(dq_0 + 0.5 * k1_q);
  // Eigen::Vector3d p_1 = p_0+0.5*k1_p;
  Eigen::Vector3d v_1 = v_0 + 0.5 * k1_v;

  Eigen::Vector4d q1_dot = 0.5 * Omega(w_hat) * dq_1;
  Eigen::Vector3d p1_dot = v_1;
  Eigen::Matrix3d R_Gto1 = quat_2_Rot(quat_multiply(dq_1, q_0));
  Eigen::Vector3d v1_dot = R_Gto1.transpose() * a_hat - _gravity;

  Eigen::Vector4d k2_q = q1_dot * dt;
  Eigen::Vector3d k2_p = p1_dot * dt;
  Eigen::Vector3d k2_v = v1_dot * dt;

  // k3 ================
  Eigen::Vector4d dq_2 = quatnorm(dq_0 + 0.5 * k2_q);
  // Eigen::Vector3d p_2 = p_0+0.5*k2_p;
  Eigen::Vector3d v_2 = v_0 + 0.5 * k2_v;

  Eigen::Vector4d q2_dot = 0.5 * Omega(w_hat) * dq_2;
  Eigen::Vector3d p2_dot = v_2;
  Eigen::Matrix3d R_Gto2 = quat_2_Rot(quat_multiply(dq_2, q_0));
  Eigen::Vector3d v2_dot = R_Gto2.transpose() * a_hat - _gravity;

  Eigen::Vector4d k3_q = q2_dot * dt;
  Eigen::Vector3d k3_p = p2_dot * dt;
  Eigen::Vector3d k3_v = v2_dot * dt;

  // k4 ================
  w_hat += 0.5 * w_alpha * dt;
  a_hat += 0.5 * a_jerk * dt;

  Eigen::Vector4d dq_3 = quatnorm(dq_0 + k3_q);
  // Eigen::Vector3d p_3 = p_0+k3_p;
  Eigen::Vector3d v_3 = v_0 + k3_v;

  Eigen::Vector4d q3_dot = 0.5 * Omega(w_hat) * dq_3;
  Eigen::Vector3d p3_dot = v_3;
  Eigen::Matrix3d R_Gto3 = quat_2_Rot(quat_multiply(dq_3, q_0));
  Eigen::Vector3d v3_dot = R_Gto3.transpose() * a_hat - _gravity;

  Eigen::Vector4d k4_q = q3_dot * dt;
  Eigen::Vector3d k4_p = p3_dot * dt;
  Eigen::Vector3d k4_v = v3_dot * dt;

  // y+dt ================
  Eigen::Vector4d dq = quatnorm(dq_0 + (1.0 / 6.0) * k1_q + (1.0 / 3.0) * k2_q + (1.0 / 3.0) * k3_q + (1.0 / 6.0) * k4_q);
  new_q = quat_multiply(dq, q_0);
  new_p = p_0 + (1.0 / 6.0) * k1_p + (1.0 / 3.0) * k2_p + (1.0 / 3.0) * k3_p + (1.0 / 6.0) * k4_p;
  new_v = v_0 + (1.0 / 6.0) * k1_v + (1.0 / 3.0) * k2_v + (1.0 / 3.0) * k3_v + (1.0 / 6.0) * k4_v;
}


////////////////////////////////
// propagate_and_clone_with_stereo_feature
////////////////////////////////


void Propagator::propagate_and_clone_with_stereo_feature(
    std::shared_ptr<State> state, double timestamp,
    const GetStereoFeatureForPropagationFunc& get_stereo_feat_func,
    Eigen::Matrix3d* output_rotation) {
  // If the difference between the current update time and state is zero
  // We should crash, as this means we would have two clones at the same time!!!!
  if (state->_timestamp == timestamp) {
    PRINT_ERROR(RED "Propagator::propagate_and_clone_with_stereo_feature(): Propagation called again at same timestep at last update timestep!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // We should crash if we are trying to propagate backwards
  if (state->_timestamp > timestamp) {
    PRINT_ERROR(RED "Propagator::propagate_and_clone_with_stereo_feature(): Propagation called trying to propagate backwards in time!!!!\n" RESET);
    PRINT_ERROR(RED "Propagator::propagate_and_clone_with_stereo_feature(): desired propagation = %.4f\n" RESET, (timestamp - state->_timestamp));
    std::exit(EXIT_FAILURE);
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Set the last time offset value if we have just started the system up
  if (!have_last_prop_time_offset) {
    last_prop_time_offset = state->_calib_dt_CAMtoIMU->value()(0);
    have_last_prop_time_offset = true;
  }

  // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
  double t_off_new = state->_calib_dt_CAMtoIMU->value()(0);

  // First lets construct an IMU vector of measurements we need
  double time0 = state->_timestamp + last_prop_time_offset;
  double time1 = timestamp + t_off_new;
  std::vector<ov_core::ImuData> prop_data;
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    prop_data = ov_core::select_imu_readings(imu_data, time0, time1);
  }

  prop_data = fill_imu_data_gaps(prop_data);

  //////////////

  size_t n = prop_data.size() - 1;
  std::vector<Eigen::Matrix3d> R;
  std::vector<Eigen::Matrix3d> C;
  std::vector<Eigen::Vector3d> so3vec;

  R.reserve(n + 1);
  C.reserve(n + 1);
  so3vec.reserve(n);
  Eigen::Vector3d bw = state->_imu->bias_g();
  auto qk = state->_imu->quat();
  Eigen::Quaterniond qk_eigen(qk[3], qk[0], qk[1], qk[2]);

  for (size_t i=0; i<n; i++) {
    Eigen::Vector3d wm = 0.5 * (prop_data.at(i).wm + prop_data.at(i+1).wm) - bw;
    // Eigen::Vector3d wm = prop_data.at(i).wm - bw;
    double dt = prop_data.at(i+1).timestamp - prop_data.at(i).timestamp;
    so3vec.push_back(wm * dt);
  }
  ASSERT(so3vec.size() == n);

  Eigen::Matrix3d R0(qk_eigen.toRotationMatrix());
  // Eigen::Matrix3d R0_test = state->_imu->Rot().transpose();
  // std::cout << "DEBUG R0-R0_test: " << R0-R0_test << std::endl;

  R.push_back(R0);
  for (size_t i=0; i<n; i++) {
    R.push_back(R.back() * exp_so3(so3vec[i]));
  }
  ASSERT(R.size() == n+1);
  Eigen::Matrix3d& Rn = R[n];
  Eigen::Matrix3d Rn_inv = Rn.transpose();

  for (size_t i=0; i<n; i++) {
    C.push_back(Rn_inv * R[i]);
  }
  C.push_back(Eigen::Matrix3d::Identity());
  ASSERT(C.size() == n+1);
  Eigen::Matrix3d& Cn = C[n];
  Eigen::Matrix3d& C0 = C[0];

  double prev_image_time = state->_timestamp;
  if (output_rotation) {
    *output_rotation = C0.transpose();
  }

  std::shared_ptr<StereoFeatureForPropagation> stereo_feat =
      get_stereo_feat_func(prev_image_time, C0.transpose());
  // if (!stereo_feat) {
  //   propagate_and_clone(state, timestamp, output_rotation);
  //   return;
  // }
  ASSERT(stereo_feat);

  // Propagate the state forward
  double DT = timestamp - prev_image_time;
  Eigen::Vector3d old_p = state->_imu->pos();
  Eigen::Vector3d new_p = old_p + R0 * stereo_feat->feat_pos_frame0 - Rn * stereo_feat->feat_pos_frame1;
  Eigen::Vector3d new_v = (new_p - old_p) / DT;

  // Now replace imu estimate and fej with propagated values
  Eigen::Matrix<double, 16, 1> imu_x = state->_imu->value();
  Eigen::Quaterniond new_qk_eigen(Rn);
  Eigen::Vector4d new_q = new_qk_eigen.coeffs();
  imu_x.block(0, 0, 4, 1) = new_q;
  imu_x.block(4, 0, 3, 1) = new_p;
  imu_x.block(7, 0, 3, 1) = new_v;
  state->_imu->set_value(imu_x);
  state->_imu->set_fej(imu_x);

  // Compute Q
  Eigen::Matrix<double, 15, 15> Q = Eigen::Matrix<double, 15, 15>::Zero();

  // Compute Q for gyro
  // Eigen::Matrix<double, 9, 9> Q_g = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::Matrix<double, 3, 3> Rn_skew_feat = Rn * skew_x(stereo_feat->feat_pos_frame1);
  Eigen::Matrix<double, 3, 3> ptheta_pbw = Eigen::Matrix<double, 3, 3>::Zero();

// #define DEBUG_EQUIVALENT_FORM
#ifdef DEBUG_EQUIVALENT_FORM
  // For debug another equivalent form.
  Eigen::Matrix<double, 3, 3> ptheta_pbw_debug = Eigen::Matrix<double, 3, 3>::Zero();
#endif

  for (size_t i=0; i<n; i++) {
    double dt = prop_data.at(i+1).timestamp - prop_data.at(i).timestamp;
    Eigen::Matrix<double, 3, 3> dQ = Eigen::Matrix<double, 3, 3>::Zero();
    double var = _noises.sigma_w_2 / dt;

    // consider gyro scale error (0.02)
    const double scale_error = 0.02;
    const double scale_error2 = scale_error * scale_error;
    const Eigen::Vector3d omega = so3vec[i] / dt;

    dQ(0,0) = var + scale_error2 * omega.x() * omega.x();
    dQ(1,1) = var + scale_error2 * omega.y() * omega.y();
    dQ(2,2) = var + scale_error2 * omega.z() * omega.z();

    Eigen::Matrix<double, 3, 3> ptheta_pnwi = - dt * C[i] * Jr_so3(-so3vec[i]);
    ptheta_pbw += ptheta_pnwi;

#ifdef DEBUG_EQUIVALENT_FORM
    // For debug another equivalent form.
    Eigen::Matrix<double, 3, 3> ptheta_pnwi_debug = - dt * C[i+1] * Jr_so3(so3vec[i]);
    ptheta_pbw_debug += ptheta_pnwi_debug;
#endif

    Eigen::Matrix<double, 3, 3> ppos_pnwi = Rn_skew_feat * ptheta_pnwi;
    Eigen::Matrix<double, 3, 3> pvec_pnwi = ppos_pnwi / DT;
    Eigen::Matrix<double, 9, 3> J;
    J << ptheta_pnwi, ppos_pnwi, pvec_pnwi;
    // Q_g += J * dQ * J.transpose();
    Q.block<9,9>(0,0) += J * dQ * J.transpose();
  }

#ifdef DEBUG_EQUIVALENT_FORM
  // debug equivalent form.
  std::cout << "DEBUG_ptheta_pbw_equivalent_form (the diff should be nearly 0):\n" << ptheta_pbw_debug - ptheta_pbw << std::endl;
#endif


  Eigen::Matrix<double, 3, 3> Q_bw = _noises.sigma_wb_2 * DT * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> Q_ba = _noises.sigma_ab_2 * DT * Eigen::Matrix<double, 3, 3>::Identity();
  Q.block<3,3>(9,9) = Q_bw;
  Q.block<3,3>(12,12) = Q_ba;

  {
    Eigen::Matrix<double, 3, 3> ppos_pnf0 = -R0;
    Eigen::Matrix<double, 3, 3> pvec_pnf0 = ppos_pnf0 / DT;
    Eigen::Matrix<double, 6, 3> J;
    J << ppos_pnf0, pvec_pnf0;
    Q.block<6,6>(3,3) += J * stereo_feat->feat_pos_frame0_cov * J.transpose();
  }

  {
    Eigen::Matrix<double, 3, 3> ppos_pnf1 = Rn;
    Eigen::Matrix<double, 3, 3> pvec_pnf1 = ppos_pnf1 / DT;
    Eigen::Matrix<double, 6, 3> J;
    J << ppos_pnf1, pvec_pnf1;
    Q.block<6,6>(3,3) += J * stereo_feat->feat_pos_frame1_cov * J.transpose();
  }
  // std::cout << "DEBUG_Q:\n" << Q << std::endl;

  // compute Phi
  Eigen::Matrix<double, 15, 15> Phi = Eigen::Matrix<double, 15, 15>::Zero();

  Phi.block<3,3>(0,0) = C0;
  Phi.block<3,3>(0,9) = ptheta_pbw;

  Eigen::Matrix<double, 3, 3> ppos_ptheta =
      - R0 * skew_x(stereo_feat->feat_pos_frame0) +
      Rn * skew_x(stereo_feat->feat_pos_frame1) * C0;
  Eigen::Matrix<double, 3, 3> ppos_pbw =
      Rn * skew_x(stereo_feat->feat_pos_frame1) * ptheta_pbw;
  Phi.block<3,3>(3,0) = ppos_ptheta;
  Phi.block<3,3>(3,3) = Eigen::Matrix<double, 3, 3>::Identity();
  Phi.block<3,3>(3,9) = ppos_pbw;
  
  Phi.block<3,3>(6,0) = ppos_ptheta / DT;
  Phi.block<3,3>(6,9) = ppos_pbw / DT;

  Phi.block<3,3>(9,9) = Eigen::Matrix<double, 3, 3>::Identity();
  Phi.block<3,3>(12,12) = Eigen::Matrix<double, 3, 3>::Identity();


  // std::cout << "DEBUG_Phi:\n" << Phi << std::endl;


  //////////////


  // Last angular velocity (used for cloning when estimating time offset)
  Eigen::Matrix<double, 3, 1> last_w = Eigen::Matrix<double, 3, 1>::Zero();
  if (prop_data.size() > 1)
    last_w = prop_data.at(prop_data.size() - 2).wm - state->_imu->bias_g();
  else if (!prop_data.empty())
    last_w = prop_data.at(prop_data.size() - 1).wm - state->_imu->bias_g();

  // Do the update to the covariance with our "summed" state transition and IMU noise addition...
  std::vector<std::shared_ptr<Type>> Phi_order;
  Phi_order.push_back(state->_imu);


  // auto P = StateHelper::get_marginal_covariance(state, Phi_order);
  // std::cout << "DEBUG_P:\n" << P << std::endl;

  // StateHelper::EKFPropagation(state, Phi_order, Phi_order, Phi_summed, Qd_summed);
  StateHelper::EKFPropagation(state, Phi_order, Phi_order, Phi, Q);

  // auto P2 = StateHelper::get_marginal_covariance(state, Phi_order);
  // std::cout << "DEBUG_P-P:\n" << P2 - (Phi * P * Phi.transpose() + Q) << std::endl;



  // Set timestamp data
  state->_timestamp = timestamp;
  last_prop_time_offset = t_off_new;

  // Now perform stochastic cloning
  StateHelper::augment_clone(state, last_w);
}

void Propagator::gravity_update(std::shared_ptr<State> state) {
  // a naive implementation (regardless of bias).

  std::vector<double> cloned_times;
  for (const auto &clone_imu : state->_clones_IMU) {
    cloned_times.push_back(clone_imu.first);
  }

  if (cloned_times.size() < 3) {
    return;
  }

  double target_time = -1;
  const double half_time_window = 0.03;  // 
  for (size_t i=cloned_times.size()-2; i>=0; i--) {
    if (cloned_times[i] < state->_timestamp - half_time_window) {
      target_time = cloned_times[i];
      break;
    }
  }

  if (target_time <= next_gravity_update_time) {
    return;
  }

  // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
  double t_offset = state->_calib_dt_CAMtoIMU->value()(0);

  // const double filter_window = 0.05;
  double time0 = target_time - half_time_window + t_offset;
  double time1 = target_time + half_time_window + t_offset;
  std::vector<ov_core::ImuData> prop_data;
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    if (imu_data[0].timestamp > time0) {
      return;
    }
    prop_data = ov_core::select_imu_readings(imu_data, time0, time1);
  }
  prop_data = fill_imu_data_gaps(prop_data);

  // calc mean acc
  Eigen::Vector3d mean_acc(0, 0, 0);
  for (const auto &imu_data : prop_data) {
    mean_acc += imu_data.am;
  }
  mean_acc /= prop_data.size();
  mean_acc -= state->_imu->bias_a();

  std::shared_ptr<ov_type::PoseJPL> pose = state->_clones_IMU.at(target_time);

  Eigen::Matrix3d Rinv = pose->Rot();

  std::cout << "DEBUG_gravity_update: mean_acc = " << (Rinv.transpose() * mean_acc).transpose() << std::endl;
  // if ((mean_acc - Rinv*Eigen::Vector3d(0, 0, 9.81)).norm() > 2.0) {
  //   return;
  // }

  mean_acc.normalize();


  Eigen::Vector3d Z(0,0,1);
  Eigen::Vector3d ZinI = Rinv * Z;
  Eigen::Vector3d XinI;
  if (fabs(ZinI.z()) < fabs(ZinI.x()) && fabs(ZinI.z()) < fabs(ZinI.y())) {
    XinI = Eigen::Vector3d(0, 0, 1);
  } else if (fabs(ZinI.x()) < fabs(ZinI.y())) {
    XinI = Eigen::Vector3d(1, 0, 0);
  } else {
    XinI = Eigen::Vector3d(0, 1, 0);
  }
  Eigen::Vector3d YinI = ZinI.cross(XinI);
  YinI.normalize();
  XinI = YinI.cross(ZinI);

  Eigen::Matrix<double, 2, 3> A;
  A << XinI.transpose(), YinI.transpose();

  Eigen::Matrix<double, 2, 3> H = A * skew_x(ZinI);

  const double direction_sigma = 0.1;  // rad
  const double direction_sigma2 = direction_sigma * direction_sigma;
  Eigen::Matrix<double, 2, 2> R = direction_sigma2 * Eigen::Matrix<double, 2, 2>::Identity();

  Eigen::Vector2d res = A * (mean_acc - ZinI);
  std::cout << "DEBUG_gravity_update: res = " << (pose->Rot().transpose() * (mean_acc - ZinI)).transpose() << std::endl;

  Eigen::MatrixXd q_cov = StateHelper::get_marginal_covariance(state, {pose->q()});
  double mal_square = res.transpose() * (H * q_cov * H.transpose() + R).inverse() * res;
  std::cout << "DEBUG_gravity_update: mal = " << sqrt(mal_square) << std::endl;

  if (mal_square > 2.0 * 2.0) {
    return;
  }

  std::cout << "DEBUG_gravity_update: ACCEPTED++++++++++++++++++++++++++++++++++++++++++" << std::endl;

  std::vector<std::shared_ptr<Type>> H_order;
  H_order.push_back(pose->q());

  StateHelper::EKFUpdate(state, H_order, H, res, R);

  next_gravity_update_time = target_time + half_time_window;
}
