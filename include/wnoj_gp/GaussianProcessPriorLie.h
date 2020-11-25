/**
 *  @file  WNOJGaussianProcessPriorLie.h
 *  @brief GP prior works on any Lie group
 *  @author Jing Dong
 *  @date Oct 3, 2016
 **/

#pragma once

#include "wnoj_gp/GPutils.h"

#include <gtsam/geometry/concepts.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Testable.h>

#include <boost/lexical_cast.hpp>
#include <boost/serialization/export.hpp>

#include <ostream>


namespace gpmp2 {

/**
 * 6-way factor for Gaussian Process prior factor on any Lie group
 */
template <typename T>
class GaussianProcessPriorLie: public gtsam::NoiseModelFactor6<T, gtsam::Vector, gtsam::Vector,
    T, gtsam::Vector, gtsam::Vector> {

private:
  BOOST_CONCEPT_ASSERT((gtsam::IsLieGroup<T>));
  typedef GaussianProcessPriorLie<T> This;
  typedef gtsam::NoiseModelFactor6<T, gtsam::Vector, gtsam::Vector, T, gtsam::Vector, gtsam::Vector> Base;

  size_t dof_;
  double delta_t_;

public:

  GaussianProcessPriorLie() {}	/* Default constructor only for serialization */

  /// Constructor
  /// @param delta_t is the time between the two states
  GaussianProcessPriorLie(gtsam::Key poseKey1, gtsam::Key velKey1, gtsam::Key accKey1,
      gtsam::Key poseKey2, gtsam::Key velKey2,gtsam::Key accKey2,
      double delta_t, const gtsam::SharedNoiseModel& Qc_model) :
    Base(gtsam::noiseModel::Gaussian::Covariance(calcQ(getQc(Qc_model), delta_t)),
        poseKey1, velKey1, accKey1, poseKey2, velKey2, accKey2), dof_(Qc_model->dim()), delta_t_(delta_t) {}

  virtual ~GaussianProcessPriorLie() {}


  /// @return a deep copy of this factor
  virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this))); } 

  /// factor error function
  gtsam::Vector evaluateError(
      const T& pose1, const gtsam::Vector& vel1, const gtsam::Vector& acc1,
      const T& pose2, const gtsam::Vector& vel2, const gtsam::Vector& acc2,
      boost::optional<gtsam::Matrix&> H1 = boost::none,
      boost::optional<gtsam::Matrix&> H2 = boost::none,
      boost::optional<gtsam::Matrix&> H3 = boost::none,
      boost::optional<gtsam::Matrix&> H4 = boost::none,
      boost::optional<gtsam::Matrix&> H5 = boost::none,
      boost::optional<gtsam::Matrix&> H6 = boost::none) const override {

    using namespace gtsam;

    Matrix Hinv, Hcomp1, Hcomp2, Hlogmap;
    Vector r;
    if (H1 || H2 || H3 || H4 || H5 || H6)
      r = traits<T>::Logmap(traits<T>::Compose(traits<T>::Inverse(pose1, Hinv),
          pose2, Hcomp1, Hcomp2), Hlogmap);
    else
      r = traits<T>::Logmap(traits<T>::Compose(traits<T>::Inverse(pose1, Hinv), pose2));

    // jacobians
    if (H1) *H1 = (Matrix(3*dof_, dof_) << Hlogmap * Hcomp1 * Hinv, Matrix::Zero(2*dof_, dof_)).finished();
    if (H2) *H2 = (Matrix(3*dof_, dof_) << -delta_t_ * Matrix::Identity(dof_, dof_), -Matrix::Identity(dof_, dof_), Matrix::Zero(dof_, dof_)).finished();
    if (H3) *H3 = (Matrix(3*dof_, dof_) << -0.5*delta_t_ * delta_t_* Matrix::Identity(dof_, dof_), -delta_t_ * Matrix::Identity(dof_, dof_), -Matrix::Identity(dof_, dof_)).finished();
    if (H4) *H4 = (Matrix(3*dof_, dof_) << Hlogmap * Hcomp2, Matrix::Zero(2*dof_, dof_)).finished();
    if (H5) *H5 = (Matrix(3*dof_, dof_) << Matrix::Zero(dof_, dof_), Matrix::Identity(dof_, dof_), Matrix::Zero(dof_, dof_)).finished();
    if (H6) *H6 = (Matrix(3*dof_, dof_) << Matrix::Zero(2*dof_, dof_), Matrix::Identity(dof_, dof_)).finished();

    return (Vector(3*dof_) << (r - vel1 * delta_t_ - 0.5*acc1*delta_t_*delta_t_), (vel2 - vel1) - acc1*delta_t_, (acc2-acc1)).finished();
  }

  /** number of variables attached to this factor */
  size_t size() const {
    return 6;
  }

  /** equals specialized to this factor */
  virtual bool equals(const gtsam::NonlinearFactor& expected, double tol=1e-9) const override {
    const This *e =  dynamic_cast<const This*> (&expected);
    return e != NULL && Base::equals(*e, tol) && fabs(this->delta_t_ - e->delta_t_) < tol;
  }

  /** print contents */
  void print(const std::string& s="", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const override {
    std::cout << s << "6-way Gaussian Process Factor on Lie<" << dof_ << ">" << std::endl;
    Base::print("", keyFormatter);
  }

private:

  /** Serialization function */
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Base);
    ar & BOOST_SERIALIZATION_NVP(dof_);
    ar & BOOST_SERIALIZATION_NVP(delta_t_);
  }

}; // GaussianProcessPriorLie

} // namespace gpmp2


