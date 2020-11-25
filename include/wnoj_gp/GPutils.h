/**
 *  @file  GPutils.h
 *  @brief GP utils, calculation of Qc, Q, Lamda matrices etc for WNOJ-prior.Based on GPMP2's GPutils.h for WNOA-prior
 *  @author Xinyan Yan, Jing Dong, Thomas Horstink
 *  @date Nov 25, 2020
 **/

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>

#include <cmath>

namespace gpmp2 {

/// get Qc covariance matrix from noise model
inline gtsam::Matrix getQc(const gtsam::SharedNoiseModel Qc_model) {
  gtsam::noiseModel::Gaussian *Gassian_model =
      dynamic_cast<gtsam::noiseModel::Gaussian*>(Qc_model.get());
  return (Gassian_model->R().transpose() * Gassian_model->R()).inverse();
}


/// calculate Q
inline gtsam::Matrix calcQ(const gtsam::Matrix &Qc, double tau) {
  assert(Qc.rows() == Qc.cols());
  return (gtsam::Matrix(3 * Qc.rows(), 3 * Qc.rows())
              << 1.0 / 2. * pow(tau, 5.0) * Qc,
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
              << 1.0 / 2 * pow(tau, 5.0) * Qc,
          720 * pow(tau, -5.0) * Qc_inv, -360 * pow(tau, -4.0) * Qc_inv,
          60 * pow(tau, -3.0) * Qc_inv, -360 * pow(tau, -4.0) * Qc_inv,
          192 * pow(tau, -3.0) * Qc_inv, -36 * pow(tau, -2.0) * Qc_inv,
          60 * pow(tau, -3.0) * Qc_inv, -36 * pow(tau, -2.0) * Qc_inv,
          -9. * 1.0 / tau * Qc_inv)
      .finished();
}

/// calculate Phi
inline gtsam::Matrix calcPhi(size_t dof, double tau) {
  return (gtsam::Matrix(3 * dof, 3 * dof) << gtsam::Matrix::Identity(dof, dof),
          tau * gtsam::Matrix::Identity(dof, dof),
          0.5 * tau * tau * gtsam::Matrix::Identity(dof, dof),
          gtsam::Matrix::Zero(dof, dof), gtsam::Matrix::Identity(dof, dof),
          tau * gtsam::Matrix::Identity(dof, dof),
          gtsam::Matrix::Zero(dof, dof), gtsam::Matrix::Zero(dof, dof),
          gtsam::Matrix::Identity(dof, dof))
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

} // namespace gpmp2
