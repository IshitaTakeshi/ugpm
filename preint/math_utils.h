#ifndef UGPM_MATH_H
#define UGPM_MATH_H

#include "Eigen/Geometry"
#include "Eigen/unsupported/SpecialFunctions"
#include "types.h"

namespace ugpm {

const double kExpNormTolerance = 1e-14;
const double kLogTraceTolerance = 3.0 - kExpNormTolerance;

const double kNumDtJacobianDelta = 0.01;
const double kNumAccBiasJacobianDelta = 0.0001;
const double kNumGyrBiasJacobianDelta = 0.0001;

const double kSqrt2 = std::sqrt(2.0);
const double kSqrtPi = std::sqrt(M_PI);

inline std::vector<double> to_std_vector(const VecX &v) {
  return std::vector<double>(v.data(), v.data() + v.size());
}

inline Mat3 eulToRotMat(double eul_z, double eul_y, double eul_x) {

  Mat3 transform;
  double c1 = std::cos(eul_x);
  double c2 = std::cos(eul_y);
  double c3 = std::cos(eul_z);
  double s1 = std::sin(eul_x);
  double s2 = std::sin(eul_y);
  double s3 = std::sin(eul_z);
  transform << c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2, c2 * s1,
      c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, -s2, c2 * s3, c2 * c3;

  return transform;
}

inline Mat3 eulToRotMat(std::vector<double> eul) {
  if (eul.size() != 3)
    throw std::range_error(
        "Wrong vector size for Euler to Rotation matrix conversion");
  return eulToRotMat(eul[2], eul[1], eul[0]);
}

// SO3 Log mapping
inline Vec3 logMap(const Mat3 &rot_mat) {
  Eigen::AngleAxisd rot_axis(rot_mat);
  return rot_axis.angle() * rot_axis.axis();
}

// SO3 Exp mapping
inline Mat3 expMap(const Vec3 &vec) {
  Eigen::AngleAxisd rot_axis(vec.norm(), vec.normalized());
  return rot_axis.toRotationMatrix();
}

Mat3 skew(const Vec3 &v) {
  Mat3 m;
  // clang-format off
  m <<
    0.0, -v(2), v(1),
    v(2), 0.0, -v(0),
    -v(1), v(0), 0.0;
  // clang-format on
  return m;
}

// Righthand Jacobian of SO3 Exp mapping
inline Mat3 jacobianRighthandSO3(const Vec3 &rot_vec) {
  const double k = rot_vec.norm();
  const Mat3 I = Mat3::Identity();
  if (k > kExpNormTolerance) {
    const Mat3 M = skew(rot_vec);
    const double sin = std::sin(k);
    const double cos = std::cos(k);

    return I + (k - sin) / (k * k * k) * M * M - (1.0 - cos) / (k * k) * M;
  }
  return I;
}

// Inverse Righthand Jacobian of SO3 Exp mapping
inline Mat3 inverseJacobianRighthandSO3(const Vec3 &rot_vec) {
  const Mat3 I = Mat3::Identity();
  const double k = rot_vec.norm();
  if (k > kExpNormTolerance) {
    const Mat3 M = skew(rot_vec);
    const double sin = std::sin(k);
    const double cos = std::cos(k);
    return I + 0.5 * M + (1.0 / (k * k) - (1 + cos) / (2.0 * k * sin)) * M * M;
  }
  return I;
}

inline MatX seKernel(const VecX &x1, const VecX &x2, const double l2,
                     const double sf2) {
  MatX D2(x1.size(), x2.size());
  for (int i = 0; i < x2.size(); i++) {
    D2.col(i) = (x1.array() - x2(i)).square();
  }
  return ((D2 * (-0.5 / l2)).array().exp() * sf2).matrix();
}

inline MatX seKernelIntegral(const double a, const VecX &b, const VecX &x2,
                             const double l2, const double sf2) {
  double sqrt_inv_l2 = std::sqrt(1.0 / l2);
  double alpha = kSqrt2 * sf2 * kSqrtPi / (2.0 * sqrt_inv_l2);

  MatX A(b.size(), x2.size());
  RowX c = (kSqrt2 * (-x2.transpose().array() + a) * sqrt_inv_l2 / 2.0)
               .erf()
               .matrix();
  for (int i = 0; i < x2.size(); i++) {
    A.col(i) = (kSqrt2 * (b.array() - x2(i)) * sqrt_inv_l2 / 2.0)
                   .array()
                   .erf()
                   .matrix();
  }
  return alpha * (A.rowwise() - c);
}

inline MatX seKernelIntegralDt(const double a, const VecX &b, const VecX &x2,
                               const double l2, const double sf2) {
  MatX A(b.size(), x2.size());
  RowX c = sf2 *
           ((x2.transpose().array() - a).square() / (-2.0 * l2)).exp().matrix();
  for (int i = 0; i < x2.size(); i++) {
    A.col(i) = sf2 * ((b.array() - x2(i)).pow(2) / (-2.0 * l2)).exp();
  }
  MatX out(b.size(), x2.size());
  return A.rowwise() - c;
}

inline MatX seKernelIntegral2(const double a, const VecX &b, const VecX &x2,
                              const double l2, const double sf2) {
  double sqrt_inv_l2 = std::sqrt(1.0 / l2);

  RowX a_x2 = (-x2.transpose().array() + a).matrix();
  RowX a_x2_erf = (kSqrt2 * (a_x2)*sqrt_inv_l2 / 2.0).array().erf().matrix();
  RowX c = (kSqrt2 * (a_x2.array().square() / (-2.0 * l2)).exp() /
                (kSqrtPi * sqrt_inv_l2) +
            a_x2_erf.array() * (a_x2.array()))
               .matrix();
  MatX A(b.size(), x2.size());
  for (int i = 0; i < x2.size(); i++) {
    VecX b_x2 = (b.array() - x2(i)).matrix();

    A.col(i) =
        (a_x2_erf(i) * (-b.array() + a) +
         (kSqrt2 * (b_x2)*sqrt_inv_l2 / 2.0).array().erf() * ((b_x2).array()) +
         kSqrt2 * (b_x2.array().square() / (-2.0 * l2)).exp() /
             (kSqrtPi * sqrt_inv_l2))
            .matrix();
  }
  double alpha = kSqrt2 * sf2 * kSqrtPi / (2.0 * sqrt_inv_l2);
  return alpha * (A.matrix().rowwise() - c).matrix();
}

inline MatX seKernelIntegral2Dt(const double a, const VecX &b, const VecX &x2,
                                const double l2, const double sf2) {
  double sqrt_inv_l2 = std::sqrt(1.0 / l2);
  RowX a_x2 = (-x2.transpose().array() + a).matrix();
  RowX a_x2_exp = (a_x2.array().square() / (-2.0 * l2)).exp();
  RowX a_x2_erf = (kSqrt2 * (a_x2)*sqrt_inv_l2 / 2.0).array().erf().matrix();
  MatX A(b.size(), x2.size());
  for (int i = 0; i < x2.size(); i++) {
    A.col(i) = ((((-kSqrt2 * (b.array() - x2(i)) * sqrt_inv_l2 * 0.5).erf() +
                  a_x2_erf(i)) *
                     kSqrt2 * sf2 * kSqrtPi +
                 2.0 * sqrt_inv_l2 * sf2 * (b.array() - a) * a_x2_exp(i)) /
                (-2.0 * sqrt_inv_l2))
                   .matrix();
  }
  return A;
}

inline Row9 mat3ToRow(Mat3 R) {
  Row9 output;
  output = Eigen::Map<Row9>(R.data());
  return output;
}

inline Vec9 jacobianExpMapZeroV(const Vec3 &v) {
  Vec9 output;
  output << 0, v(2), -v(1), -v(2), 0, v(0), v(1), -v(0), 0;
  return output;
}
inline Mat9_3 jacobianExpMapZeroM(const Mat3 &M) {
  Mat9_3 output;
  output << 0, 0, 0, M(2, 0), M(2, 1), M(2, 2), -M(1, 0), -M(1, 1), -M(1, 2),
      -M(2, 0), -M(2, 1), -M(2, 2), 0, 0, 0, M(0, 0), M(0, 1), M(0, 2), M(1, 0),
      M(1, 1), M(1, 2), -M(0, 0), -M(0, 1), -M(0, 2), 0, 0, 0;
  return output;
}

inline Mat3_9 jacobianLogMap(const Mat3 &rot_mat) {
  Mat3_9 output;

  if (rot_mat.trace() < kLogTraceTolerance) {

    // Equation from MATLAB symbolic toolbox (might have a better formualtion,
    // to inspect later)
    output(0, 0) =
        -(rot_mat(1, 2) - rot_mat(2, 1)) /
            (4 * (std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                               rot_mat(2, 2) / 2.0 - 0.5,
                           2) -
                  1)) -
        (std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) *
         (rot_mat(1, 2) - rot_mat(2, 1)) *
         (rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 + rot_mat(2, 2) / 2.0 -
          0.5)) /
            (4 *
             std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      1.5));
    output(0, 1) = 0.0;
    output(0, 2) = 0.0;
    output(0, 3) = 0.0;
    output(0, 4) =
        -(rot_mat(1, 2) - rot_mat(2, 1)) /
            (4 * (std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                               rot_mat(2, 2) / 2.0 - 0.5,
                           2) -
                  1)) -
        (std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) *
         (rot_mat(1, 2) - rot_mat(2, 1)) *
         (rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 + rot_mat(2, 2) / 2.0 -
          0.5)) /
            (4 *
             std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      1.5));
    output(0, 5) =
        std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                  rot_mat(2, 2) / 2.0 - 0.5) /
        (2 * std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      0.5));
    output(0, 6) = 0.0;
    output(0, 7) =
        -std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) /
        (2 * std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      0.5));
    output(0, 8) =
        -(rot_mat(1, 2) - rot_mat(2, 1)) /
            (4 * (std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                               rot_mat(2, 2) / 2.0 - 0.5,
                           2) -
                  1)) -
        (std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) *
         (rot_mat(1, 2) - rot_mat(2, 1)) *
         (rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 + rot_mat(2, 2) / 2.0 -
          0.5)) /
            (4 *
             std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      1.5));
    output(1, 0) =
        (rot_mat(0, 2) - rot_mat(2, 0)) /
            (4 * (std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                               rot_mat(2, 2) / 2.0 - 0.5,
                           2) -
                  1)) +
        (std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) *
         (rot_mat(0, 2) - rot_mat(2, 0)) *
         (rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 + rot_mat(2, 2) / 2.0 -
          0.5)) /
            (4 *
             std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      1.5));
    output(1, 1) = 0.0;
    output(1, 2) =
        -std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) /
        (2 * std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      0.5));
    output(1, 3) = 0.0;
    output(1, 4) =
        (rot_mat(0, 2) - rot_mat(2, 0)) /
            (4 * (std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                               rot_mat(2, 2) / 2.0 - 0.5,
                           2) -
                  1)) +
        (std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) *
         (rot_mat(0, 2) - rot_mat(2, 0)) *
         (rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 + rot_mat(2, 2) / 2.0 -
          0.5)) /
            (4 *
             std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      1.5));
    output(1, 5) = 0.0;
    output(1, 6) =
        std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                  rot_mat(2, 2) / 2.0 - 0.5) /
        (2 * std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      0.5));
    output(1, 7) = 0.0;
    output(1, 8) =
        (rot_mat(0, 2) - rot_mat(2, 0)) /
            (4 * (std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                               rot_mat(2, 2) / 2.0 - 0.5,
                           2) -
                  1)) +
        (std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) *
         (rot_mat(0, 2) - rot_mat(2, 0)) *
         (rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 + rot_mat(2, 2) / 2.0 -
          0.5)) /
            (4 *
             std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      1.5));
    output(2, 0) =
        -(rot_mat(0, 1) - rot_mat(1, 0)) /
            (4 * (std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                               rot_mat(2, 2) / 2.0 - 0.5,
                           2) -
                  1)) -
        (std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) *
         (rot_mat(0, 1) - rot_mat(1, 0)) *
         (rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 + rot_mat(2, 2) / 2.0 -
          0.5)) /
            (4 *
             std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      1.5));
    output(2, 1) =
        std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                  rot_mat(2, 2) / 2.0 - 0.5) /
        (2 * std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      0.5));
    output(2, 2) = 0.0;
    output(2, 3) =
        -std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) /
        (2 * std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      0.5));
    output(2, 4) =
        -(rot_mat(0, 1) - rot_mat(1, 0)) /
            (4 * (std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                               rot_mat(2, 2) / 2.0 - 0.5,
                           2) -
                  1)) -
        (std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) *
         (rot_mat(0, 1) - rot_mat(1, 0)) *
         (rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 + rot_mat(2, 2) / 2.0 -
          0.5)) /
            (4 *
             std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      1.5));
    output(2, 5) = 0.0;
    output(2, 6) = 0.0;
    output(2, 7) = 0.0;
    output(2, 8) =
        -(rot_mat(0, 1) - rot_mat(1, 0)) /
            (4 * (std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                               rot_mat(2, 2) / 2.0 - 0.5,
                           2) -
                  1)) -
        (std::acos(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                   rot_mat(2, 2) / 2.0 - 0.5) *
         (rot_mat(0, 1) - rot_mat(1, 0)) *
         (rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 + rot_mat(2, 2) / 2.0 -
          0.5)) /
            (4 *
             std::pow(1 - std::pow(rot_mat(0, 0) / 2.0 + rot_mat(1, 1) / 2.0 +
                                       rot_mat(2, 2) / 2.0 - 0.5,
                                   2),
                      1.5));
  } else {
    output << 0, 0, 0, 0, 0, 0.5, 0, -0.5, 0, 0, 0, -0.5, 0, 0, 0, 0.5, 0, 0, 0,
        0.5, 0, -0.5, 0, 0, 0, 0, 0;
  }

  return output;
}

inline Mat3_9 jacobianXv(const Vec3 &v) {
  Mat3_9 output;
  output << v[0], 0.0, 0.0, v[1], 0.0, 0.0, v[2], 0.0, 0.0, 0.0, v[0], 0.0, 0.0,
      v[1], 0.0, 0.0, v[2], 0.0, 0.0, 0.0, v[0], 0.0, 0.0, v[1], 0.0, 0.0, v[2];
  return output;
}

inline Mat9 jacobianTranspose() {
  Mat9 output;
  output << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  return output;
}

inline Mat9 jacobianYX(const Mat3 &Y) {
  Mat9 output;
  output = Mat9::Zero();
  output.block<3, 3>(0, 0) = Y;
  output.block<3, 3>(3, 3) = Y;
  output.block<3, 3>(6, 6) = Y;
  return output;
}
inline Mat3_9 jacobianYXv(const Mat3 &Y, const Vec3 &v) {
  Mat3_9 output;
  output << mat3ToRow((v * Y.row(0)).transpose()),
      mat3ToRow((v * Y.row(1)).transpose()),
      mat3ToRow((v * Y.row(2)).transpose());
  return output;
}

inline Mat9 jacobianYXW(const Mat3 &Y, const Mat3 &W) {
  Mat9 output;
  output << mat3ToRow((W.col(0) * Y.row(0)).transpose()),
      mat3ToRow((W.col(0) * Y.row(1)).transpose()),
      mat3ToRow((W.col(0) * Y.row(2)).transpose()),
      mat3ToRow((W.col(1) * Y.row(0)).transpose()),
      mat3ToRow((W.col(1) * Y.row(1)).transpose()),
      mat3ToRow((W.col(1) * Y.row(2)).transpose()),
      mat3ToRow((W.col(2) * Y.row(0)).transpose()),
      mat3ToRow((W.col(2) * Y.row(1)).transpose()),
      mat3ToRow((W.col(2) * Y.row(2)).transpose());
  return output;
}

inline double kssInt(const double a, const double b, const double l2,
                     const double sf2) {
  return 2.0 * l2 * sf2 * std::exp(-std::pow(a - b, 2) / (2.0 * l2)) -
         2.0 * l2 * sf2 +
         (std::sqrt(2.0) * sf2 * std::sqrt(M_PI) *
          std::erf((std::sqrt(2.0) * (a - b) * std::sqrt(1.0 / l2)) / 2.0) *
          (a - b)) /
             std::sqrt(1.0 / l2);
}

inline Vec3 addN2Pi(const Vec3 &r, const int n) {
  double norm_r = r.norm();
  if (norm_r != 0) {
    Vec3 unit_r = r / norm_r;
    return unit_r * (2.0 * M_PI * n + norm_r);
  } else {
    return r;
  }
}

inline std::pair<Vec3, int> getClosest(const Vec3 &t,
                                       const std::vector<Vec3> s) {
  int id_min = 0;
  double dist_min = std::numeric_limits<double>::max();
  for (size_t i = 0; i < s.size(); ++i) {
    if ((t - s[i]).norm() < dist_min) {
      dist_min = (t - s[i]).norm();
      id_min = i;
    }
  }
  return {s[id_min], id_min};
}

inline MatX reprojectAccData(const std::vector<PreintMeas> &preint,
                             const MatX &acc) {
  MatX output(3, acc.cols());

  for (int i = 0; i < acc.cols(); ++i) {
    output.col(i) = preint[i].delta_R * acc.col(i);
  }
  return output;
}

inline MatX reprojectAccData(const std::vector<PreintMeas> &preint,
                             const MatX &acc, const Mat3 &delta_R_dt_start,
                             std::vector<MatX> &d_acc_d_bf,
                             std::vector<MatX> &d_acc_d_bw,
                             std::vector<VecX> &d_acc_d_dt) {
  MatX output(3, acc.cols());
  for (int i = 0; i < 3; ++i) {
    d_acc_d_bf.push_back(MatX(acc.cols(), 3));
    d_acc_d_bw.push_back(MatX(acc.cols(), 3));
    d_acc_d_dt.push_back(VecX(acc.cols()));
  }

  for (int i = 0; i < acc.cols(); ++i) {
    Vec3 temp_acc = acc.col(i);

    d_acc_d_bf[0].row(i) = preint[i].delta_R.row(0);
    d_acc_d_bf[1].row(i) = preint[i].delta_R.row(1);
    d_acc_d_bf[2].row(i) = preint[i].delta_R.row(2);

    Mat9_3 temp_d_R_d_bw = jacobianExpMapZeroM(preint[i].d_delta_R_d_bw);
    Row9 temp_1;
    temp_1 << preint[i].delta_R(0, 0) * temp_acc(0),
        preint[i].delta_R(0, 1) * temp_acc(0),
        preint[i].delta_R(0, 2) * temp_acc(0),
        preint[i].delta_R(0, 0) * temp_acc(1),
        preint[i].delta_R(0, 1) * temp_acc(1),
        preint[i].delta_R(0, 2) * temp_acc(1),
        preint[i].delta_R(0, 0) * temp_acc(2),
        preint[i].delta_R(0, 1) * temp_acc(2),
        preint[i].delta_R(0, 2) * temp_acc(2);
    Row9 temp_2;
    temp_2 << preint[i].delta_R(1, 0) * temp_acc(0),
        preint[i].delta_R(1, 1) * temp_acc(0),
        preint[i].delta_R(1, 2) * temp_acc(0),
        preint[i].delta_R(1, 0) * temp_acc(1),
        preint[i].delta_R(1, 1) * temp_acc(1),
        preint[i].delta_R(1, 2) * temp_acc(1),
        preint[i].delta_R(1, 0) * temp_acc(2),
        preint[i].delta_R(1, 1) * temp_acc(2),
        preint[i].delta_R(1, 2) * temp_acc(2);
    Row9 temp_3;
    temp_3 << preint[i].delta_R(2, 0) * temp_acc(0),
        preint[i].delta_R(2, 1) * temp_acc(0),
        preint[i].delta_R(2, 2) * temp_acc(0),
        preint[i].delta_R(2, 0) * temp_acc(1),
        preint[i].delta_R(2, 1) * temp_acc(1),
        preint[i].delta_R(2, 2) * temp_acc(1),
        preint[i].delta_R(2, 0) * temp_acc(2),
        preint[i].delta_R(2, 1) * temp_acc(2),
        preint[i].delta_R(2, 2) * temp_acc(2);
    d_acc_d_bw[0].row(i) = temp_1 * temp_d_R_d_bw;
    d_acc_d_bw[1].row(i) = temp_2 * temp_d_R_d_bw;
    d_acc_d_bw[2].row(i) = temp_3 * temp_d_R_d_bw;

    temp_acc = preint[i].delta_R * temp_acc;
    Vec3 acc_rot_dt = delta_R_dt_start.transpose() * temp_acc;
    Vec3 d_acc_d_t = (acc_rot_dt - temp_acc) / kNumDtJacobianDelta;
    d_acc_d_dt[0][i] = d_acc_d_t(0);
    d_acc_d_dt[1][i] = d_acc_d_t(1);
    d_acc_d_dt[2][i] = d_acc_d_t(2);

    output.col(i) = temp_acc;
  }
  return output;
}

inline std::pair<VecX, VecX>
linearInterpolation(const VecX &data, const VecX &time, const double var,
                    const SortIndexTracker2<double> &infer_t) {
  VecX out_val(infer_t.size());
  VecX out_var(infer_t.size());

  if (time.rows() < 2) {
    throw std::range_error("InterpolateLinear: this function need at least 2 "
                           "data points to interpolate");
  }

  int ptr = 0;
  double alpha = (data(1) - data(0)) / (time(1) - time(0));
  double beta = data(0) - (alpha * time(0));

  for (int i = 0; i < infer_t.size(); ++i) {
    if (infer_t.get(i) > time(0)) {
      bool loop = true;
      while (loop) {
        if (ptr != (time.rows() - 2)) {
          if ((infer_t.get(i) <= time(ptr + 1)) &&
              (infer_t.get(i) > time(ptr))) {
            loop = false;
          } else {
            ptr++;
            alpha = (data(ptr + 1) - data(ptr)) / (time(ptr + 1) - time(ptr));
            beta = data(ptr) - (alpha * time(ptr));
          }
        } else {
          loop = false;
        }
      }
    }
    out_val(i) = alpha * infer_t.get(i) + beta;
    // Might want more complex policy on variance
    out_var(i) = var;
  }
  return {out_val, out_var};
}

VecX linearInterpolation(const VecX &data, const VecX &time,
                         const SortIndexTracker2<double> &infer_t) {
  auto [val, var] = linearInterpolation(data, time, 0, infer_t);
  return val;
}

inline Vec9 perturbationPropagation(const Vec18 &eps, const PreintMeas &prev,
                                    const PreintMeas &curr) {
  Vec3 eps_r1 = eps.segment<3>(0);
  Vec3 eps_v1 = eps.segment<3>(3);
  Vec3 eps_p1 = eps.segment<3>(6);
  Vec3 eps_r2 = eps.segment<3>(9);
  Vec3 eps_v2 = eps.segment<3>(12);
  Vec3 eps_p2 = eps.segment<3>(15);

  Vec9 output;
  Mat3 exp_eps_r1 = expMap(eps_r1);
  Mat3 R_exp_eps_r1 = prev.delta_R * exp_eps_r1;
  output.segment<3>(0) = logMap(curr.delta_R.transpose() * exp_eps_r1 *
                                curr.delta_R * expMap(eps_r2));
  output.segment<3>(3) = eps_v1 + R_exp_eps_r1 * (curr.delta_v + eps_v2);
  output.segment<3>(6) =
      eps_p1 + R_exp_eps_r1 * (curr.delta_p + eps_p2) + prev.dt * eps_v1;
  return output;
}

// Numerical propagation, not the most efficient/elegant, but works
inline Mat9 propagatePreintCov(const PreintMeas &prev, const PreintMeas &curr) {
  double quantum = 1e-5;
  MatX d_eps_d_eps(9, 18);
  Vec18 eps = Vec18::Zero();
  Vec9 perturbation = perturbationPropagation(eps, prev, curr);
  for (int i = 0; i < 18; ++i) {
    eps(i) = quantum;
    d_eps_d_eps.col(i) =
        (perturbationPropagation(eps, prev, curr) - perturbation) / quantum;
    eps(i) = 0;
  }
  MatX cov = MatX::Zero(18, 18);
  cov.block<9, 9>(0, 0) = prev.cov;
  cov.block<9, 9>(9, 9) = curr.cov;
  return d_eps_d_eps * cov * d_eps_d_eps.transpose();
}

inline Mat3 propagateJacobianRp(const Mat3 &R, const Mat3 &d_r, const Vec3 &p,
                                const Mat3 &d_p) {

  const Mat9_3 d_R = jacobianYX(R) * jacobianExpMapZeroM(d_r);
  const Mat3 D0 = d_R.block<3, 3>(0, 0);
  const Mat3 D3 = d_R.block<3, 3>(3, 0);
  const Mat3 D6 = d_R.block<3, 3>(6, 0);
  return R * d_p + p(0) * D0 + p(1) * D3 + p(2) * D6;
}

inline Vec3 propagateJacobianRp(const Mat3 &R, const Vec3 &d_r, const Vec3 &p,
                                const Vec3 &d_p) {

  const Vec9 d_R = jacobianYX(R) * jacobianExpMapZeroV(d_r);
  const Eigen::Map<const Mat3> DR(d_R.data());
  return R * d_p + DR * p;
}

inline Mat3 propagateJacobianRR(const Mat3 &R1, const Mat3 &d_r1,
                                const Mat3 &R2, const Mat3 &d_r2) {

  const Mat9_3 d_R1 = jacobianYX(R1) * jacobianExpMapZeroM(d_r1);
  const Mat9_3 d_R2 = jacobianYX(R2) * jacobianExpMapZeroM(d_r2);

  const Eigen::Map<const Mat3> DR2(d_R2.data());
  const Mat3 P0 = R1 * d_R2.block<3, 3>(0, 0);
  const Mat3 P3 = R1 * d_R2.block<3, 3>(3, 0);
  const Mat3 P6 = R1 * d_R2.block<3, 3>(6, 0);

  const Mat3 D0 = d_R1.block<3, 3>(0, 0);
  const Mat3 D3 = d_R1.block<3, 3>(3, 0);
  const Mat3 D6 = d_R1.block<3, 3>(6, 0);

  Mat9_3 d_RR;
  d_RR.block<3, 3>(0, 0) = P0 + D0 * R2(0, 0) + D3 * R2(1, 0) + D6 * R2(2, 0);
  d_RR.block<3, 3>(3, 0) = P3 + D0 * R2(0, 1) + D3 * R2(1, 1) + D6 * R2(2, 1);
  d_RR.block<3, 3>(6, 0) = P6 + D0 * R2(0, 2) + D3 * R2(1, 2) + D6 * R2(2, 2);

  return jacobianLogMap(R1 * R2) * d_RR;
}

inline Vec3 propagateJacobianRR(const Mat3 &R1, const Vec3 &d_r1,
                                const Mat3 &R2, const Vec3 &d_r2) {
  const Vec9 d_R1 = jacobianYX(R1) * jacobianExpMapZeroV(d_r1);
  const Vec9 d_R2 = jacobianYX(R2) * jacobianExpMapZeroV(d_r2);

  const Eigen::Map<const Mat3> DR1(d_R1.data());
  const Eigen::Map<const Mat3> DR2(d_R2.data());
  const Mat3 P = DR1 * R2 + R1 * DR2;
  const Eigen::Map<const Vec9> d_RR(P.data());

  return jacobianLogMap(R1 * R2) * d_RR;
}

inline PreintMeas combinePreints(const PreintMeas &prev,
                                 const PreintMeas &preint) {
  if (preint.dt == 0.0)
    return prev;

  PreintMeas temp = preint;
  temp.cov = Mat9::Zero();

  temp.cov = propagatePreintCov(prev, preint);

  // Propagation of acc Jacobians
  temp.d_delta_p_d_bf = prev.d_delta_p_d_bf + (temp.dt * prev.d_delta_v_d_bf) +
                        (prev.delta_R * temp.d_delta_p_d_bf);

  temp.d_delta_v_d_bf =
      prev.d_delta_v_d_bf + (prev.delta_R * temp.d_delta_v_d_bf);

  const Mat3 dpw = propagateJacobianRp(prev.delta_R, prev.d_delta_R_d_bw,
                                       temp.delta_p, temp.d_delta_p_d_bw);
  const Mat3 dvw = propagateJacobianRp(prev.delta_R, prev.d_delta_R_d_bw,
                                       temp.delta_v, temp.d_delta_v_d_bw);

  const Mat3 drw =
      propagateJacobianRR(temp.delta_R.transpose(), prev.d_delta_R_d_bw,
                          temp.delta_R, temp.d_delta_R_d_bw);
  const Vec3 dpt = propagateJacobianRp(prev.delta_R, prev.d_delta_R_d_t,
                                       temp.delta_p, temp.d_delta_p_d_t);

  const Vec3 dvt = propagateJacobianRp(prev.delta_R, prev.d_delta_R_d_t,
                                       temp.delta_v, temp.d_delta_v_d_t);
  const Vec3 drt =
      propagateJacobianRR(temp.delta_R.transpose(), prev.d_delta_R_d_t,
                          temp.delta_R, temp.d_delta_R_d_t);

  // Propagation of gyr Jacobians
  temp.d_delta_p_d_bw =
      prev.d_delta_p_d_bw + (temp.dt * prev.d_delta_v_d_bw) + dpw;
  temp.d_delta_v_d_bw = prev.d_delta_v_d_bw + dvw;
  temp.d_delta_R_d_bw = drw;

  // Propagation of time-shift Jacobians
  temp.d_delta_p_d_t =
      prev.d_delta_p_d_t + (temp.dt * prev.d_delta_v_d_t) + dpt;
  temp.d_delta_v_d_t = prev.d_delta_v_d_t + dvt;
  temp.d_delta_R_d_t = drt;

  // Chunck combination
  temp.delta_p =
      prev.delta_p + prev.delta_v * temp.dt + prev.delta_R * temp.delta_p;
  temp.delta_v = prev.delta_v + prev.delta_R * temp.delta_v;
  temp.delta_R = prev.delta_R * temp.delta_R;

  temp.dt = prev.dt + temp.dt;
  temp.dt_sq_half = 0.5 * temp.dt * temp.dt;

  return temp;
};

} // namespace ugpm
#endif
