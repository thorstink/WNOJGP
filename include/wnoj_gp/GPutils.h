/**
 *  @file  GPutils.h
 *  @brief GP utils, calculation of Qc, Q, Lamda matrices etc for
 *WNOJ-prior.Based on GPMP2's GPutils.h for WNOA-prior
 *  @author Xinyan Yan, Jing Dong, Thomas Horstink
 *  @date Nov 25, 2020
 **/

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>

#include <cmath>

namespace wnoj {

inline gtsam::Matrix3 rightJacobianRot3inv(const gtsam::Vector3 &omega) {
  using std::cos;
  using std::sin;
  double theta2 = omega.dot(omega);
  if (theta2 <= std::numeric_limits<double>::epsilon()) {
    gtsam::Matrix3 ANS = gtsam::Matrix::Identity(3, 3);
    return ANS;
  }
  double theta = std::sqrt(theta2); // rotation angle
  const gtsam::Matrix3 X =
      gtsam::skewSymmetric(omega); // element of Lie algebra so(3): X = omega^
  gtsam::Matrix3 ans =
      (gtsam::Matrix::Identity(3, 3) + 0.5 * X +
       (1 / (theta * theta) - (1 + cos(theta)) / (2 * theta * sin(theta))) * X *
           X);
  return ans;
}

inline gtsam::Matrix3 rightJacobianPose3Q(const gtsam::Vector6 &xi) {
  const gtsam::Vector3 omega = xi.head(3), rho = xi.tail(3);
  const double theta = omega.norm(); // rotation angle
  const gtsam::Matrix3 X = gtsam::skewSymmetric(omega),
                       Y = gtsam::skewSymmetric(rho);

  const gtsam::Matrix3 XY = X * Y, YX = Y * X, XYX = X * YX;
  gtsam::Matrix3 ANS;
  if (fabs(theta) > 1e-5) {
    const double sin_theta = sin(theta), cos_theta = cos(theta);
    const double theta2 = theta * theta, theta3 = theta2 * theta,
                 theta4 = theta3 * theta, theta5 = theta4 * theta;

    ANS << -0.5 * Y + (theta - sin_theta) / theta3 * (XY + YX - XYX) +
               (1.0 - 0.5 * theta2 - cos_theta) / theta4 *
                   (X * XY + YX * X - 3.0 * XYX) -
               0.5 *
                   ((1.0 - 0.5 * theta2 - cos_theta) / theta4 -
                    3.0 * (theta - sin_theta - theta3 / 6.0) / theta5) *
                   (XYX * X + X * XYX);

  } else {
    ANS << -0.5 * Y + 1.0 / 6.0 * (XY + YX - XYX) +
               1.0 / 24.0 * (X * XY + YX * X - 3.0 * XYX) -
               0.5 * (1.0 / 24.0 + 3.0 / 120.0) * (XYX * X + X * XYX);
  }
  return ANS;
}

gtsam::Matrix rightJacobianPose3inv(const gtsam::Vector &xi) {
  const gtsam::Vector3 w = xi.head<3>();
  const gtsam::Matrix3 Jw = rightJacobianRot3inv(w);
  const gtsam::Matrix3 Q = rightJacobianPose3Q(xi);
  gtsam::Matrix Q2 = -Jw * Q * Jw;
  const gtsam::Matrix J =
      (gtsam::Matrix(6, 6) << Jw, gtsam::Matrix::Zero(3, 3), Q2, Jw).finished();

  return J;
}

/// get Qc covariance matrix from noise model
inline gtsam::Matrix getQc(const gtsam::SharedNoiseModel Qc_model) {
  gtsam::noiseModel::Gaussian *Gassian_model =
      dynamic_cast<gtsam::noiseModel::Gaussian *>(Qc_model.get());
  return (Gassian_model->R().transpose() * Gassian_model->R()).inverse();
}

/// calculate Q
inline gtsam::Matrix calcQ(const gtsam::Matrix &Qc, double tau) {
  assert(Qc.rows() == Qc.cols());
  return (gtsam::Matrix(3 * Qc.rows(), 3 * Qc.rows())
              << 1.0 / 20. * pow(tau, 5.0) * Qc,
          1.0 / 8. * pow(tau, 4.0) * Qc, 1.0 / 6. * pow(tau, 3.0) * Qc,
          1.0 / 8. * pow(tau, 4.0) * Qc, 1.0 / 3. * pow(tau, 3.0) * Qc,
          1.0 / 2. * pow(tau, 2.0) * Qc, 1.0 / 6. * pow(tau, 3.0) * Qc,
          1.0 / 2. * pow(tau, 2.0) * Qc, tau * Qc)
      .finished();
}

/// calculate Q_inv
inline gtsam::Matrix calcQ_inv(const gtsam::Matrix &Qc, double tau) {
  assert(Qc.rows() == Qc.cols());
  const gtsam::Matrix Qc_inv = Qc.inverse();
  return (gtsam::Matrix(3 * Qc.rows(), 3 * Qc.rows())
              << 
          720 * pow(tau, -5.0) * Qc_inv, -360 * pow(tau, -4.0) * Qc_inv,
          60 * pow(tau, -3.0) * Qc_inv, -360 * pow(tau, -4.0) * Qc_inv,
          192 * pow(tau, -3.0) * Qc_inv, -36 * pow(tau, -2.0) * Qc_inv,
          60 * pow(tau, -3.0) * Qc_inv, -36 * pow(tau, -2.0) * Qc_inv,
          9. * 1.0 / tau * Qc_inv)
      .finished();
}

// numerical diff
gtsam::Matrix6 jacobianMethodNumercialDiff(boost::function<gtsam::Matrix6(const gtsam::Vector6&)> func,
    const gtsam::Vector6& xi, const gtsam::Vector6& x, double dxi = 1e-6) {
  using namespace gtsam;
  Matrix6 Diff = Matrix6();
  for (size_t i = 0; i < 6; i++) {
    Vector6 xi_dxip = xi, xi_dxin = xi;
    xi_dxip(i) += dxi;
    Matrix6 Jdiffp = func(xi_dxip);
    xi_dxin(i) -= dxi;
    Matrix6 Jdiffn = func(xi_dxin);
    Diff.block<6,1>(0,i) = (Jdiffp - Jdiffn) / (2.0 * dxi) * x;
  }
  return Diff;
}

/// calculate Phi
inline gtsam::Matrix calcPhi(size_t dof, double tau) {
  return (gtsam::Matrix(3 * dof, 3 * dof) << gtsam::Matrix::Identity(dof, dof),
          tau * gtsam::Matrix::Identity(dof, dof),
          0.5 * tau * tau * gtsam::Matrix::Identity(dof, dof),
          gtsam::Matrix::Zero(dof, dof), gtsam::Matrix::Identity(dof, dof),
          tau * gtsam::Matrix::Identity(dof, dof),
          gtsam::Matrix::Zero(dof, 2 * dof), gtsam::Matrix::Identity(dof, dof))
      .finished();
}

/// calculate Lambda
inline gtsam::Matrix calcLambda(const gtsam::Matrix &Qc, double delta_t,
                                const double tau) {
  assert(Qc.rows() == Qc.cols());
  return calcPhi(Qc.rows(), tau) -
         calcQ(Qc, tau) * (calcPhi(Qc.rows(), delta_t - tau).transpose()) *
             calcQ_inv(Qc, delta_t) * calcPhi(Qc.rows(), delta_t);
}

/// calculate Psi
inline gtsam::Matrix calcPsi(const gtsam::Matrix &Qc, double delta_t,
                             double tau) {
  assert(Qc.rows() == Qc.cols());
  return calcQ(Qc, tau) * (calcPhi(Qc.rows(), delta_t - tau).transpose()) *
         calcQ_inv(Qc, delta_t);
}

} // namespace wnoj
