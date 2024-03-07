#ifndef UGPM_COST_FUNCTIONS_H
#define UGPM_COST_FUNCTIONS_H

#include "math_utils.h"
#include "types.h"
#include <ceres/ceres.h>

namespace ugpm {

using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class GpNormCostFunction : public ceres::CostFunction {
private:
  const MatX KK_inv_;
  const int nb_data_;
  VecX weight_;
  MatX jacobian_;

public:
  GpNormCostFunction(const MatX &KK_inv, const VecX &var)
      : KK_inv_(KK_inv), nb_data_(KK_inv_.cols()) {
    weight_ = (var.array().inverse().sqrt()).matrix();
    set_num_residuals(nb_data_);
    std::vector<int> *block_sizes = mutable_parameter_block_sizes();
    block_sizes->push_back(nb_data_);

    jacobian_ = KK_inv - MatX::Identity(nb_data_, nb_data_);
    for (int i = 0; i < nb_data_; ++i) {
      if (std::isnan(weight_[i]))
        weight_[i] = 1.0;
      jacobian_.col(i) *= weight_[i];
    }
  };

  // Combined cost function for SO3 integration
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const {
    const Eigen::Map<const VecX> s(&(parameters[0][0]), nb_data_, 1);

    const VecX d_rot_d_t = KK_inv_ * s;

    Eigen::Map<VecX> r(residuals, nb_data_, 1);
    r = (((d_rot_d_t - s).array()) * (weight_.array())).matrix();

    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        Eigen::Map<MatX> j_s(&(jacobians[0][0]), nb_data_, nb_data_);
        j_s = jacobian_;
      }
    }
    return true;
  }
};

inline Mat3_6 JacobianRes(const Vec3 &r, const Vec3 &d_r) {
  Mat3_6 output;

  const double r0_sq = r(0) * r(0);
  const double r1_sq = r(1) * r(1);
  const double r2_sq = r(2) * r(2);
  const double temp_r = r.squaredNorm();
  const double norm_r = std::sqrt(temp_r);

  if (norm_r > kExpNormTolerance) {
    const double r0_cu = r(0) * r(0) * r(0);
    const double r1_cu = r(1) * r(1) * r(1);
    const double r2_cu = r(2) * r(2) * r(2);
    const double norm_r_2 = std::pow(temp_r, 2);
    const double norm_r_3 = std::pow(temp_r, 1.5);
    const double norm_r_5 = std::pow(temp_r, 2.5);
    const double s_r = std::sin(norm_r);
    const double c_r = std::cos(norm_r);

    output(0, 0) =
        d_r(1) * ((r(0) * r(2) * s_r) / norm_r_3 -
                  (r(1) * (s_r - norm_r)) / norm_r_3 +
                  (2.0 * r(0) * r(2) * (c_r - 1.0)) / norm_r_2 +
                  (3.0 * r0_sq * r(1) * (s_r - norm_r)) / norm_r_5 +
                  (r(0) * r(1) * (r(0) / norm_r - (r(0) * c_r) / norm_r)) /
                      norm_r_3) -
        d_r(2) * ((r(2) * (s_r - norm_r)) / norm_r_3 +
                  (r(0) * r(1) * s_r) / norm_r_3 +
                  (2.0 * r(0) * r(1) * (c_r - 1.0)) / norm_r_2 -
                  (3.0 * r0_sq * r(2) * (s_r - norm_r)) / norm_r_5 -
                  (r(0) * r(2) * (r(0) / norm_r - (r(0) * c_r) / norm_r)) /
                      norm_r_3) -
        d_r(0) * ((r1_sq * (r(0) / norm_r - (r(0) * c_r) / norm_r)) / norm_r_3 +
                  (r2_sq * (r(0) / norm_r - (r(0) * c_r) / norm_r)) / norm_r_3 +
                  (3.0 * r(0) * r1_sq * (s_r - norm_r)) / norm_r_5 +
                  (3.0 * r(0) * r2_sq * (s_r - norm_r)) / norm_r_5);

    output(0, 1) =
        d_r(2) * ((c_r - 1.0) / temp_r - (r1_sq * s_r) / norm_r_3 -
                  (2.0 * r1_sq * (c_r - 1.0)) / norm_r_2 +
                  (r(0) * r(2) * (r(1) / norm_r - (r(1) * c_r) / norm_r)) /
                      norm_r_3 +
                  (3.0 * r(0) * r(1) * r(2) * (s_r - norm_r)) / norm_r_5) -
        d_r(0) * ((3.0 * r1_cu * (s_r - norm_r)) / norm_r_5 +
                  (r1_sq * (r(1) / norm_r - (r(1) * c_r) / norm_r)) / norm_r_3 +
                  (r2_sq * (r(1) / norm_r - (r(1) * c_r) / norm_r)) / norm_r_3 -
                  (2.0 * r(1) * (s_r - norm_r)) / norm_r_3 +
                  (3.0 * r(1) * r2_sq * (s_r - norm_r)) / norm_r_5) +
        d_r(1) * ((r(1) * r(2) * s_r) / norm_r_3 -
                  (r(0) * (s_r - norm_r)) / norm_r_3 +
                  (2.0 * r(1) * r(2) * (c_r - 1.0)) / norm_r_2 +
                  (3.0 * r(0) * r1_sq * (s_r - norm_r)) / norm_r_5 +
                  (r(0) * r(1) * (r(1) / norm_r - (r(1) * c_r) / norm_r)) /
                      norm_r_3);

    output(0, 2) =
        d_r(1) * ((r2_sq * s_r) / norm_r_3 - (c_r - 1.0) / temp_r +
                  (2.0 * r2_sq * (c_r - 1.0)) / norm_r_2 +
                  (r(0) * r(1) * (r(2) / norm_r - (r(2) * c_r) / norm_r)) /
                      norm_r_3 +
                  (3.0 * r(0) * r(1) * r(2) * (s_r - norm_r)) / norm_r_5) -
        d_r(0) * ((3.0 * r2_cu * (s_r - norm_r)) / norm_r_5 +
                  (r1_sq * (r(2) / norm_r - (r(2) * c_r) / norm_r)) / norm_r_3 +
                  (r2_sq * (r(2) / norm_r - (r(2) * c_r) / norm_r)) / norm_r_3 -
                  (2.0 * r(2) * (s_r - norm_r)) / norm_r_3 +
                  (3.0 * r1_sq * r(2) * (s_r - norm_r)) / norm_r_5) -
        d_r(2) * ((r(0) * (s_r - norm_r)) / norm_r_3 +
                  (r(1) * r(2) * s_r) / norm_r_3 +
                  (2.0 * r(1) * r(2) * (c_r - 1.0)) / norm_r_2 -
                  (3.0 * r(0) * r2_sq * (s_r - norm_r)) / norm_r_5 -
                  (r(0) * r(2) * (r(2) / norm_r - (r(2) * c_r) / norm_r)) /
                      norm_r_3);

    output(0, 3) = (r1_sq * (s_r - norm_r)) / norm_r_3 +
                   (r2_sq * (s_r - norm_r)) / norm_r_3 + 1.0;

    output(0, 4) = -(r(2) * (c_r - 1.0)) / temp_r -
                   (r(0) * r(1) * (s_r - norm_r)) / norm_r_3;

    output(0, 5) = (r(1) * (c_r - 1.0)) / temp_r -
                   (r(0) * r(2) * (s_r - norm_r)) / norm_r_3;

    output(1, 0) =
        d_r(2) * ((r0_sq * s_r) / norm_r_3 - (c_r - 1.0) / temp_r +
                  (2.0 * r0_sq * (c_r - 1.0)) / norm_r_2 +
                  (r(1) * r(2) * (r(0) / norm_r - (r(0) * c_r) / norm_r)) /
                      norm_r_3 +
                  (3.0 * r(0) * r(1) * r(2) * (s_r - norm_r)) / norm_r_5) -
        d_r(1) * ((3.0 * r0_cu * (s_r - norm_r)) / norm_r_5 +
                  (r0_sq * (r(0) / norm_r - (r(0) * c_r) / norm_r)) / norm_r_3 +
                  (r2_sq * (r(0) / norm_r - (r(0) * c_r) / norm_r)) / norm_r_3 -
                  (2.0 * r(0) * (s_r - norm_r)) / norm_r_3 +
                  (3.0 * r(0) * r2_sq * (s_r - norm_r)) / norm_r_5) -
        d_r(0) * ((r(1) * (s_r - norm_r)) / norm_r_3 +
                  (r(0) * r(2) * s_r) / norm_r_3 +
                  (2.0 * r(0) * r(2) * (c_r - 1.0)) / norm_r_2 -
                  (3.0 * r0_sq * r(1) * (s_r - norm_r)) / norm_r_5 -
                  (r(0) * r(1) * (r(0) / norm_r - (r(0) * c_r) / norm_r)) /
                      norm_r_3);

    output(1, 1) =
        d_r(2) * ((r(0) * r(1) * s_r) / norm_r_3 -
                  (r(2) * (s_r - norm_r)) / norm_r_3 +
                  (2.0 * r(0) * r(1) * (c_r - 1.0)) / norm_r_2 +
                  (3.0 * r1_sq * r(2) * (s_r - norm_r)) / norm_r_5 +
                  (r(1) * r(2) * (r(1) / norm_r - (r(1) * c_r) / norm_r)) /
                      norm_r_3) -
        d_r(0) * ((r(0) * (s_r - norm_r)) / norm_r_3 +
                  (r(1) * r(2) * s_r) / norm_r_3 +
                  (2.0 * r(1) * r(2) * (c_r - 1.0)) / norm_r_2 -
                  (3.0 * r(0) * r1_sq * (s_r - norm_r)) / norm_r_5 -
                  (r(0) * r(1) * (r(1) / norm_r - (r(1) * c_r) / norm_r)) /
                      norm_r_3) -
        d_r(1) * ((r0_sq * (r(1) / norm_r - (r(1) * c_r) / norm_r)) / norm_r_3 +
                  (r2_sq * (r(1) / norm_r - (r(1) * c_r) / norm_r)) / norm_r_3 +
                  (3.0 * r0_sq * r(1) * (s_r - norm_r)) / norm_r_5 +
                  (3.0 * r(1) * r2_sq * (s_r - norm_r)) / norm_r_5);

    output(1, 2) =
        d_r(0) * ((c_r - 1.0) / temp_r - (r2_sq * s_r) / norm_r_3 -
                  (2.0 * r2_sq * (c_r - 1.0)) / norm_r_2 +
                  (r(0) * r(1) * (r(2) / norm_r - (r(2) * c_r) / norm_r)) /
                      norm_r_3 +
                  (3.0 * r(0) * r(1) * r(2) * (s_r - norm_r)) / norm_r_5) -
        d_r(1) * ((3.0 * r2_cu * (s_r - norm_r)) / norm_r_5 +
                  (r0_sq * (r(2) / norm_r - (r(2) * c_r) / norm_r)) / norm_r_3 +
                  (r2_sq * (r(2) / norm_r - (r(2) * c_r) / norm_r)) / norm_r_3 -
                  (2.0 * r(2) * (s_r - norm_r)) / norm_r_3 +
                  (3.0 * r0_sq * r(2) * (s_r - norm_r)) / norm_r_5) +
        d_r(2) * ((r(0) * r(2) * s_r) / norm_r_3 -
                  (r(1) * (s_r - norm_r)) / norm_r_3 +
                  (2.0 * r(0) * r(2) * (c_r - 1.0)) / norm_r_2 +
                  (3.0 * r(1) * r2_sq * (s_r - norm_r)) / norm_r_5 +
                  (r(1) * r(2) * (r(2) / norm_r - (r(2) * c_r) / norm_r)) /
                      norm_r_3);

    output(1, 3) = (r(2) * (c_r - 1.0)) / temp_r -
                   (r(0) * r(1) * (s_r - norm_r)) / norm_r_3;

    output(1, 4) = (r0_sq * (s_r - norm_r)) / norm_r_3 +
                   (r2_sq * (s_r - norm_r)) / norm_r_3 + 1.0;

    output(1, 5) = -(r(0) * (c_r - 1.0)) / temp_r -
                   (r(1) * r(2) * (s_r - norm_r)) / norm_r_3;

    output(2, 0) =
        d_r(1) * ((c_r - 1.0) / temp_r - (r0_sq * s_r) / norm_r_3 -
                  (2.0 * r0_sq * (c_r - 1.0)) / norm_r_2 +
                  (r(1) * r(2) * (r(0) / norm_r - (r(0) * c_r) / norm_r)) /
                      norm_r_3 +
                  (3.0 * r(0) * r(1) * r(2) * (s_r - norm_r)) / norm_r_5) -
        d_r(2) * ((3.0 * r0_cu * (s_r - norm_r)) / norm_r_5 +
                  (r0_sq * (r(0) / norm_r - (r(0) * c_r) / norm_r)) / norm_r_3 +
                  (r1_sq * (r(0) / norm_r - (r(0) * c_r) / norm_r)) / norm_r_3 -
                  (2.0 * r(0) * (s_r - norm_r)) / norm_r_3 +
                  (3.0 * r(0) * r1_sq * (s_r - norm_r)) / norm_r_5) +
        d_r(0) * ((r(0) * r(1) * s_r) / norm_r_3 -
                  (r(2) * (s_r - norm_r)) / norm_r_3 +
                  (2.0 * r(0) * r(1) * (c_r - 1.0)) / norm_r_2 +
                  (3.0 * r0_sq * r(2) * (s_r - norm_r)) / norm_r_5 +
                  (r(0) * r(2) * (r(0) / norm_r - (r(0) * c_r) / norm_r)) /
                      norm_r_3);

    output(2, 1) =
        d_r(0) * ((r1_sq * s_r) / norm_r_3 - (c_r - 1.0) / temp_r +
                  (2.0 * r1_sq * (c_r - 1.0)) / norm_r_2 +
                  (r(0) * r(2) * (r(1) / norm_r - (r(1) * c_r) / norm_r)) /
                      norm_r_3 +
                  (3.0 * r(0) * r(1) * r(2) * (s_r - norm_r)) / norm_r_5) -
        d_r(2) * ((3.0 * r1_cu * (s_r - norm_r)) / norm_r_5 +
                  (r0_sq * (r(1) / norm_r - (r(1) * c_r) / norm_r)) / norm_r_3 +
                  (r1_sq * (r(1) / norm_r - (r(1) * c_r) / norm_r)) / norm_r_3 -
                  (2.0 * r(1) * (s_r - norm_r)) / norm_r_3 +
                  (3.0 * r0_sq * r(1) * (s_r - norm_r)) / norm_r_5) -
        d_r(1) * ((r(2) * (s_r - norm_r)) / norm_r_3 +
                  (r(0) * r(1) * s_r) / norm_r_3 +
                  (2.0 * r(0) * r(1) * (c_r - 1.0)) / norm_r_2 -
                  (3.0 * r1_sq * r(2) * (s_r - norm_r)) / norm_r_5 -
                  (r(1) * r(2) * (r(1) / norm_r - (r(1) * c_r) / norm_r)) /
                      norm_r_3);

    output(2, 2) =
        d_r(0) * ((r(1) * r(2) * s_r) / norm_r_3 -
                  (r(0) * (s_r - norm_r)) / norm_r_3 +
                  (2.0 * r(1) * r(2) * (c_r - 1.0)) / norm_r_2 +
                  (3.0 * r(0) * r2_sq * (s_r - norm_r)) / norm_r_5 +
                  (r(0) * r(2) * (r(2) / norm_r - (r(2) * c_r) / norm_r)) /
                      norm_r_3) -
        d_r(1) * ((r(1) * (s_r - norm_r)) / norm_r_3 +
                  (r(0) * r(2) * s_r) / norm_r_3 +
                  (2.0 * r(0) * r(2) * (c_r - 1.0)) / norm_r_2 -
                  (3.0 * r(1) * r2_sq * (s_r - norm_r)) / norm_r_5 -
                  (r(1) * r(2) * (r(2) / norm_r - (r(2) * c_r) / norm_r)) /
                      norm_r_3) -
        d_r(2) * ((r0_sq * (r(2) / norm_r - (r(2) * c_r) / norm_r)) / norm_r_3 +
                  (r1_sq * (r(2) / norm_r - (r(2) * c_r) / norm_r)) / norm_r_3 +
                  (3.0 * r0_sq * r(2) * (s_r - norm_r)) / norm_r_5 +
                  (3.0 * r1_sq * r(2) * (s_r - norm_r)) / norm_r_5);

    output(2, 3) = -(r(1) * (c_r - 1.0)) / temp_r -
                   (r(0) * r(2) * (s_r - norm_r)) / norm_r_3;

    output(2, 4) = (r(0) * (c_r - 1.0)) / temp_r -
                   (r(1) * r(2) * (s_r - norm_r)) / norm_r_3;

    output(2, 5) = (r0_sq * (s_r - norm_r)) / norm_r_3 +
                   (r1_sq * (s_r - norm_r)) / norm_r_3 + 1.0;
  } else {
    output.block<3, 3>(0, 0) = 0.5 * skew(d_r);
    output.block<3, 3>(0, 3) = Mat3::Identity();
  }

  return output;
}

class RotCostFunction : public ceres::CostFunction {
private:
  const MatX *ang_vel_;
  std::vector<MatX> K_s_K_inv_;
  std::vector<MatX> K_s_int_K_inv_;
  Vec3 mean_;
  const int nb_data_;
  const int nb_state_;
  Mat3 weight_;
  const VecX d_time_;

public:
  RotCostFunction(MatX *ang_vel, const VecX &ang_vel_time,
                  const VecX &state_time, const std::vector<MatX> &K_inv,
                  const double start_t, const std::vector<GPSeHyper> &hyper,
                  const double gyr_var)
      : ang_vel_(ang_vel), K_s_K_inv_(3), K_s_int_K_inv_(3),
        nb_data_(ang_vel_time.rows()), nb_state_(state_time.rows()),
        d_time_((ang_vel_time.array() - start_t).matrix()) {
    weight_ = Mat3::Zero();
    weight_(0, 0) = std::sqrt(1.0 / gyr_var);
    weight_(1, 1) = std::sqrt(1.0 / gyr_var);
    weight_(2, 2) = std::sqrt(1.0 / gyr_var);
    for (int i = 0; i < 3; ++i) {
      const double l2 = hyper[i].l2;
      const double sf2 = hyper[i].sf2;
      const MatX ks_int =
          seKernelIntegral(start_t, ang_vel_time, state_time, l2, sf2);
      const MatX ks = seKernel(ang_vel_time, state_time, l2, sf2);
      K_s_K_inv_[i] = ks * K_inv[i];
      K_s_int_K_inv_[i] = ks_int * K_inv[i];
      mean_[i] = hyper[i].mean;
    }

    set_num_residuals(3 * nb_data_);
    std::vector<int> *block_sizes = mutable_parameter_block_sizes();
    block_sizes->push_back(nb_state_);
    block_sizes->push_back(nb_state_);
    block_sizes->push_back(nb_state_);
  };

  // Combined cost function for SO3 integration
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const {
    // Read the state variables from ceres and infer the r and d_r
    MatX d_rot_d_t(nb_data_, 3);
    MatX rot(nb_data_, 3);
    for (int i = 0; i < 3; ++i) {
      Eigen::Map<const VecX> s_col(&(parameters[i][0]), nb_state_, 1);
      d_rot_d_t.col(i) = K_s_K_inv_.at(i) * s_col;
      rot.col(i) = K_s_int_K_inv_.at(i) * s_col;
    }

    MatX temp(nb_data_, 3);
    for (int i = 0; i < nb_data_; ++i) {
      const Vec3 rot_vec = rot.row(i).transpose() + (d_time_(i) * mean_);
      const Vec3 d_rot_vec = d_rot_d_t.row(i).transpose() + mean_;
      temp.row(i) = (jacobianRighthandSO3(rot_vec) * (d_rot_vec)).transpose();

      if (jacobians != NULL) {
        const Mat3_6 d_res_d_rdr = JacobianRes(rot_vec, d_rot_vec);

        for (int axis = 0; axis < 3; ++axis) {
          if (jacobians[axis] != NULL) {
            MatX d_rdr_d_s(2, nb_state_);
            d_rdr_d_s.row(0) = K_s_int_K_inv_.at(axis).row(i);
            d_rdr_d_s.row(1) = K_s_K_inv_.at(axis).row(i);

            Mat3_2 d_res_d_local;
            d_res_d_local.col(0) = d_res_d_rdr.col(axis);
            d_res_d_local.col(1) = d_res_d_rdr.col(axis + 3);

            Eigen::Map<RowMajorMatrix> j_s(
                &(jacobians[axis][i * 3 * nb_state_]), 3, nb_state_);
            j_s = d_res_d_local * d_rdr_d_s;
          }
        }
      }
    }

    // Map the residuals out put to a Eigen matrix
    Eigen::Map<RowMajorMatrix> r(residuals, nb_data_, 3);
    r = temp - ang_vel_->transpose();

    return true;
  }
};

class AccCostFunction : public ceres::CostFunction {
private:
  const MatX *acc_;
  std::vector<MatX> K_acc_K_inv_;
  std::vector<MatX> K_gyr_int_K_inv_;
  const int nb_data_;
  const int nb_state_;
  const Mat3 weight_;
  const VecX d_time_;
  Vec3 mean_acc_;
  Vec3 mean_dr_;

public:
  AccCostFunction(const MatX *acc, const VecX &acc_time, const VecX &state_time,
                  const std::vector<MatX> &K_inv, const double start_t,
                  const std::vector<GPSeHyper> &hyper, const double acc_var)
      : acc_(acc), K_acc_K_inv_(3), K_gyr_int_K_inv_(3),
        nb_data_(acc_time.rows()), nb_state_(state_time.rows()),
        weight_((Vec3::Ones() * std::sqrt(1.0 / acc_var)).asDiagonal()),
        d_time_((acc_time.array() - start_t).matrix()) {

    for (int i = 0; i < 6; ++i) {
      const double l2 = hyper[i].l2;
      const double sf2 = hyper[i].sf2;
      if (i < 3) {
        const MatX ks_int =
            seKernelIntegral(start_t, acc_time, state_time, l2, sf2);
        K_gyr_int_K_inv_[i] = ks_int * K_inv[i];
        mean_dr_[i] = hyper[i].mean;
      } else {
        const MatX ks = seKernel(acc_time, state_time, l2, sf2);
        K_acc_K_inv_[i - 3] = ks * K_inv[i];
        mean_acc_[i - 3] = hyper[i].mean;
      }
    }

    set_num_residuals(3 * nb_data_);
    std::vector<int> *block_sizes = mutable_parameter_block_sizes();
    block_sizes->push_back(nb_state_);
    block_sizes->push_back(nb_state_);
    block_sizes->push_back(nb_state_);
    block_sizes->push_back(nb_state_);
    block_sizes->push_back(nb_state_);
    block_sizes->push_back(nb_state_);
  };

  // Combined cost function for SO3 integration
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const {
    // Read the state variables from ceres and infer the r and d_r
    MatX rot(nb_data_, 3);
    MatX acc(nb_data_, 3);

    for (int i = 0; i < 3; ++i) {
      Eigen::Map<const VecX> dr_col(&(parameters[i][0]), nb_state_);
      rot.col(i) = K_gyr_int_K_inv_.at(i) * dr_col;
    }

    for (int i = 0; i < 3; ++i) {
      Eigen::Map<const VecX> acc_col(&(parameters[i + 3][0]), nb_state_);
      acc.col(i) = K_acc_K_inv_.at(i) * acc_col;
    }

    MatX temp(nb_data_, 3);
    for (int i = 0; i < nb_data_; ++i) {
      const Vec3 rot_vec = rot.row(i).transpose() + (d_time_(i) * mean_dr_);
      const Mat3 R_T = expMap(-rot_vec);
      const Vec3 acc_vec = acc.row(i).transpose() + mean_acc_;
      temp.row(i) = (R_T * acc_vec).transpose();

      if (jacobians == NULL) {
        continue;
      }
      const Mat3 K = skew(temp.row(i).transpose());
      const Mat3 J = jacobianRighthandSO3(rot_vec);
      const Mat3 d_res_d_r = K * J;

      for (int axis = 0; axis < 3; ++axis) {
        if (jacobians[axis] == NULL) {
          continue;
        }
        Eigen::Map<RowMajorMatrix> j_s(&(jacobians[axis][i * 3 * nb_state_]), 3,
                                       nb_state_);
        j_s = weight_ * d_res_d_r.col(axis) * K_gyr_int_K_inv_.at(axis).row(i);
      }

      for (int axis = 0; axis < 3; ++axis) {
        if (jacobians[3 + axis] == NULL) {
          continue;
        }
        Eigen::Map<RowMajorMatrix> j_s(
            &(jacobians[3 + axis][i * 3 * nb_state_]), 3, nb_state_);
        j_s = weight_ * R_T.col(axis) * K_acc_K_inv_.at(axis).row(i);
      }
    }

    // Map the residuals out put to a Eigen matrix
    Eigen::Map<RowMajorMatrix> r(residuals, nb_data_, 3);
    r = (temp - acc_->transpose()) * weight_;

    return true;
  }
};

} // namespace ugpm

#endif
