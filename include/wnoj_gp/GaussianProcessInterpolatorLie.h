/**
 *  @file GaussianProcessInterpolatorLie.h
 *  @brief Base and utils for Gaussian Process Interpolated measurement factor works on any Lie group
 *  @author Jing Dong
 *  @date Oct 3, 2016
 **/

#pragma once

#include "wnoj_gp/GPutils.h"

#include <gtsam/base/Lie.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>


#include <iostream>


namespace gpmp2 {

/**
 * 
 * 6-way factor for Gaussian Process interpolator, templated Lie group version
 * interpolate pose, velocity and acceleration given consecutive poses, velocities and accelerations
 */
template <typename T>
class GaussianProcessInterpolatorLie {

private:
  BOOST_CONCEPT_ASSERT((gtsam::IsLieGroup<T>));
  typedef GaussianProcessInterpolatorLie<T> This;
  
  size_t dof_;
  double delta_t_;		// t_{i+1} - t_i
  double tau_;			// tau - t_i. we use tau as time interval from t_i instead of from t_0 as in Barfoot papers
  gtsam::Matrix Qc_;
  gtsam::Matrix Lambda_;
  gtsam::Matrix Psi_;

public:

  /// Default constructor: only for serialization
  GaussianProcessInterpolatorLie() {}

  /**
   * Constructor
   * @param Qc noise model of Qc
   * @param delta_t the time between the two states
   * @param tau the time of interval status
   */
  GaussianProcessInterpolatorLie(const gtsam::SharedNoiseModel& Qc_model, double delta_t, double tau) :
    dof_(Qc_model->dim()), delta_t_(delta_t), tau_(tau) {

    // Calcuate Lambda and Psi
    Qc_ = getQc(Qc_model);
    Lambda_ = calcLambda(Qc_, delta_t_, tau_);
    Psi_ = calcPsi(Qc_, delta_t_, tau_);

    // std::cout << Qc_ << std::endl;
    // std::cout << Lambda_ << std::endl;
    // std::cout << Psi_ << std::endl;
  }

  /** Virtual destructor */
  virtual ~GaussianProcessInterpolatorLie() {}


  /// interpolate pose with Jacobians
  T interpolatePose(
      const T& pose1, const gtsam::Vector& vel1, const gtsam::Vector& acc1,
      const T& pose2, const gtsam::Vector& vel2, const gtsam::Vector& acc2,
      gtsam::OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H1 = boost::none,
      gtsam::OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H2 = boost::none,
      gtsam::OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H3 = boost::none,
      gtsam::OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H4 = boost::none,
      gtsam::OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H5 = boost::none,
      gtsam::OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H6 = boost::none) const {

    using namespace gtsam;

    const Vector r1 = (Vector(3*dof_) << Vector::Zero(dof_), vel1, acc1).finished();
    Matrix Hinv, Hcomp11, Hcomp12, Hlogmap;
    Vector r;
    if (H1 || H2 || H3 || H4 || H5 || H6)
      r = traits<T>::Logmap(traits<T>::Compose(traits<T>::Inverse(pose1, Hinv),
          pose2, Hcomp11, Hcomp12), Hlogmap);
    else
      r = traits<T>::Logmap(traits<T>::Compose(traits<T>::Inverse(pose1, Hinv), pose2));
    const Vector r2 = (Vector(3*dof_) << r, vel2, acc1).finished();

    T pose;
    if (H1 || H2 || H3 || H4 || H5 || H6) {
      Matrix Hcomp21, Hcomp22, Hexp;
      pose = traits<T>::Compose(pose1, traits<T>::Expmap(Lambda_.block(0, 0, dof_, 3*dof_) * r1
          + Psi_.block(0, 0, dof_, 3*dof_) * r2, Hexp), Hcomp21, Hcomp22);
      Matrix Hexpr1 = Hcomp22*Hexp;
      if (H1) *H1 = Hcomp21 + Hexpr1*Psi_.block(0, 0, dof_, dof_)*Hlogmap*Hcomp11*Hinv;
      if (H2) *H2 = Hexpr1*Lambda_.block(0, dof_, dof_, dof_);
      if (H3) *H3 = Hexpr1*(Lambda_.block(0, 2*dof_, dof_, dof_)+Psi_.block(0, 2*dof_, dof_, dof_));
      if (H4) *H4 = Hexpr1*Psi_.block(0, 0, dof_, dof_)*Hlogmap*Hcomp12;
      if (H5) *H5 = Hexpr1*Psi_.block(0, dof_, dof_, dof_);
      if (H6) *H6 = Matrix::Zero(dof_,dof_); // WNOA assumes acc1=acc2
    } else {
      pose = traits<T>::Compose(pose1, traits<T>::Expmap(Lambda_.block(0, 0, dof_, 3*dof_) * r1
          + Psi_.block(0, 0, dof_, 3*dof_) * r2));
    }

    return pose;
  }

  /// update jacobian based on interpolated jacobians
  static void updatePoseJacobians(const gtsam::Matrix& Hpose,  const gtsam::Matrix& Hint1,
      const gtsam::Matrix& Hint2, const gtsam::Matrix& Hint3, const gtsam::Matrix& Hint4, 
      const gtsam::Matrix& Hint5, const gtsam::Matrix& Hint6,
      boost::optional<gtsam::Matrix&> H1, boost::optional<gtsam::Matrix&> H2,
      boost::optional<gtsam::Matrix&> H3, boost::optional<gtsam::Matrix&> H4,
      boost::optional<gtsam::Matrix&> H5, boost::optional<gtsam::Matrix&> H6) {
    if (H1) *H1 = Hpose * Hint1;
    if (H2) *H2 = Hpose * Hint2;
    if (H3) *H3 = Hpose * Hint3;
    if (H4) *H4 = Hpose * Hint4;
    if (H5) *H3 = Hpose * Hint5;
    if (H6) *H4 = Hpose * Hint6;
  }

  /**
   * Testables
   */

  /** equals specialized to this factor */
  virtual bool equals(const This& expected, double tol=1e-9) const {
    return fabs(this->delta_t_ - expected.delta_t_) < tol &&
        fabs(this->tau_ - expected.tau_) < tol &&
        gtsam::equal_with_abs_tol(this->Qc_, expected.Qc_, tol) &&
        gtsam::equal_with_abs_tol(this->Lambda_, expected.Lambda_, tol) &&
        gtsam::equal_with_abs_tol(this->Psi_, expected.Psi_, tol);
  }

  /** print contents */
  void print(const std::string& s="") const {
    std::cout << s << "GaussianProcessInterpolatorLie<" << dof_ << ">" << std::endl;
    std::cout << "delta_t = " << delta_t_ << ", tau = " << tau_ << std::endl;
    //std::cout << "Qc = " << Qc_ << std::endl;
  }


private:

  /** Serialization function */
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(dof_);
    ar & BOOST_SERIALIZATION_NVP(delta_t_);
    ar & BOOST_SERIALIZATION_NVP(tau_);
    using namespace boost::serialization;
    ar & make_nvp("Qc", make_array(Qc_.data(), Qc_.size()));
    ar & make_nvp("Lambda", make_array(Lambda_.data(), Lambda_.size()));
    ar & make_nvp("Psi", make_array(Psi_.data(), Psi_.size()));
  }
  
};

} // \ namespace gpmp2


