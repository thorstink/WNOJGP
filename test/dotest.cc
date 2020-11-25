/**
 *  @file testGaussianProcessInterpolatorPose2.cpp
 *  @author Jing Dong
 **/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>

#include <gtsam/geometry/Pose2.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include "wnoj_gp/GaussianProcessInterpolatorLie.h"
#include "wnoj_gp/GaussianProcessPriorLie.h"

#include <iostream>

using namespace std;
using namespace gtsam;
using namespace gpmp2;

typedef GaussianProcessInterpolatorLie<gtsam::Pose2>
    GaussianProcessInterpolatorPose2;

/* ************************************************************************** */
TEST(GaussianProcessInterpolatorPose2, interpolatePose) {
  Pose2 p1, p2, expect, actual;
  Vector3 v1, v2, a1, a2;
  Matrix actualH1, actualH2, actualH3, actualH4, actualH5, actualH6;
  Matrix expectH1, expectH2, expectH3, expectH4, expectH5, expectH6;
  Matrix3 Qc = 0.01 * Matrix::Identity(3, 3);
  noiseModel::Gaussian::shared_ptr Qc_model =
      noiseModel::Gaussian::Covariance(Qc);
  double dt = 0.1, tau = 0.03;
  GaussianProcessInterpolatorPose2 base(Qc_model, dt, tau);

  // test at origin
  p1 = Pose2(0, 0, 0);
  p2 = Pose2(0, 0, 0);
  v1 = (Vector3() << 0, 0, 0).finished();
  v2 = (Vector3() << 0, 0, 0).finished();
  a1 = (Vector3() << 0, 0, 0).finished();
  a2 = (Vector3() << 0, 0, 0).finished();

  actual = base.interpolatePose(p1, v1, a1, p2, v2, a2, actualH1, actualH2,
                                actualH3, actualH4, actualH5, actualH6);
  expect = Pose2(0, 0, 0);

  // helper function
  boost::function<Pose2(const Pose2 &, const Vector3 &, const Vector3 &,
                        const Pose2 &, const Vector3 &, const Vector3 &)>
      f1 = [base](const Pose2 &p1, const Vector3 &v1, const Vector3 &a1,
                  const Pose2 &p2, const Vector3 &v2, const Vector3 &a2) {
        return base.interpolatePose(p1, v1, a1, p2, v2, a2);
      };

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-6);

  EXPECT(assert_equal(expect, actual, 1e-6));
  EXPECT(assert_equal(expectH1, actualH1, 1e-8));
  EXPECT(assert_equal(expectH2, actualH2, 1e-8));
  EXPECT(assert_equal(expectH3, actualH3, 1e-8));
  EXPECT(assert_equal(expectH4, actualH4, 1e-8));
  EXPECT(assert_equal(expectH5, actualH5, 1e-8));
  EXPECT(assert_equal(expectH6, actualH6, 1e-8));

  // test forward constant velocity
  p1 = Pose2(0, 0, 0);
  p2 = Pose2(0.1, 0, 0);
  v1 = (Vector3() << 1, 0, 0).finished();
  v2 = (Vector3() << 1, 0, 0).finished();
  a1 = (Vector3() << 0, 0, 0).finished();
  a2 = (Vector3() << 0, 0, 0).finished();

  actual = base.interpolatePose(p1, v1, a1, p2, v2, a2, actualH1, actualH2,
                                actualH3, actualH4, actualH5, actualH6);
  expect = Pose2(0.03, 0, 0);

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-4);

  EXPECT(assert_equal(expect, actual, 1e-6));
  EXPECT(assert_equal(expectH1, actualH1, 1e-6));
  EXPECT(assert_equal(expectH2, actualH2, 1e-6));
  EXPECT(assert_equal(expectH3, actualH3, 1e-6));
  EXPECT(assert_equal(expectH4, actualH4, 1e-6));
  EXPECT(assert_equal(expectH5, actualH5, 1e-6));
  EXPECT(assert_equal(expectH6, actualH6, 1e-6));

  // test forward constant acceleration
  p1 = Pose2(0, 0, 0);
  p2 = Pose2(0.5 * dt * dt, 0, 0);
  v1 = (Vector3() << 0, 0, 0).finished();
  v2 = (Vector3() << dt * 1.0, 0, 0).finished();
  a1 = (Vector3() << 1, 0, 0).finished();
  a2 = (Vector3() << 1, 0, 0).finished();

  actual = base.interpolatePose(p1, v1, a1, p2, v2, a2, actualH1, actualH2,
                                actualH3, actualH4, actualH5, actualH6);
  expect = Pose2(0.5 * tau * tau, 0, 0);

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-4);

  EXPECT(assert_equal(expect, actual, 1e-6));
  EXPECT(assert_equal(expectH1, actualH1, 1e-6));
  EXPECT(assert_equal(expectH2, actualH2, 1e-5));
  EXPECT(assert_equal(expectH3, actualH3, 1e-6));
  EXPECT(assert_equal(expectH4, actualH4, 1e-6));
  EXPECT(assert_equal(expectH5, actualH5, 1e-5));
  EXPECT(assert_equal(expectH6, actualH6, 1e-6));
}

/* ************************************************************************** */
TEST(GaussianProcessPriorLie, Factor) {

  const double delta_t = 0.1;
  const double dt = delta_t;
  Matrix Qc = 0.01 * Matrix::Identity(3,3);
  noiseModel::Gaussian::shared_ptr Qc_model = noiseModel::Gaussian::Covariance(Qc);
  Key key_pose1 = Symbol('x', 1), key_pose2 = Symbol('x', 2);
  Key key_vel1 = Symbol('v', 1), key_vel2 = Symbol('v', 2);
  Key key_acc1 = Symbol('a', 1), key_acc2 = Symbol('a', 2);
  GaussianProcessPriorLie<Pose2> factor(key_pose1, key_vel1, key_acc1, key_pose2, key_vel2, key_acc2, delta_t, Qc_model);
  Pose2 p1, p2;
  Vector3 v1, v2, a1, a2;
  Matrix actualH1, actualH2, actualH3, actualH4, actualH5, actualH6;
  Matrix expectH1, expectH2, expectH3, expectH4, expectH5, expectH6;
  Vector actual, expect;

  // helper function
  boost::function<Vector(const Pose2 &, const Vector3 &, const Vector3 &,
                        const Pose2 &, const Vector3 &, const Vector3 &)>
      f1 = [factor](const Pose2 &p1, const Vector3 &v1, const Vector3 &a1,
                  const Pose2 &p2, const Vector3 &v2, const Vector3 &a2) {
        return factor.evaluateError(p1, v1, a1, p2, v2, a2);
      };

  // test at origin
  p1 = Pose2(0, 0, 0);
  p2 = Pose2(0, 0, 0);
  v1 = (Vector3() << 0, 0, 0).finished();
  v2 = (Vector3() << 0, 0, 0).finished();
  a1 = (Vector3() << 0, 0, 0).finished();
  a2 = (Vector3() << 0, 0, 0).finished();
  actual = factor.evaluateError(p1, v1, a1, p2, v2, a2, actualH1, actualH2, actualH3, actualH4 ,actualH5, actualH6);
  expect = (Vector(9) << 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-6);

  EXPECT(assert_equal(expect, actual, 1e-6));
  EXPECT(assert_equal(expectH1, actualH1, 1e-8));
  EXPECT(assert_equal(expectH2, actualH2, 1e-8));
  EXPECT(assert_equal(expectH3, actualH3, 1e-8));
  EXPECT(assert_equal(expectH4, actualH4, 1e-8));
  EXPECT(assert_equal(expectH5, actualH5, 1e-8));
  EXPECT(assert_equal(expectH6, actualH6, 1e-8));

  // constant velocity
  p1 = Pose2(0, 0, 0);
  p2 = Pose2(0.1, 0, 0);
  v1 = (Vector3() << 1, 0, 0).finished();
  v2 = (Vector3() << 1, 0, 0).finished();
  a1 = (Vector3() << 0, 0, 0).finished();
  a2 = (Vector3() << 0, 0, 0).finished();
  actual = factor.evaluateError(p1, v1, a1, p2, v2, a2, actualH1, actualH2, actualH3, actualH4 ,actualH5, actualH6);
  expect = (Vector(9) << 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-6);

  EXPECT(assert_equal(expect, actual, 1e-6));
  EXPECT(assert_equal(expectH1, actualH1, 1e-5));
  EXPECT(assert_equal(expectH2, actualH2, 1e-6));
  EXPECT(assert_equal(expectH3, actualH3, 1e-6));
  EXPECT(assert_equal(expectH4, actualH4, 1e-5));
  EXPECT(assert_equal(expectH5, actualH5, 1e-6));
  EXPECT(assert_equal(expectH6, actualH6, 1e-6));

  // constant acceleration
  p1 = Pose2(0, 0, 0);
  p2 = Pose2(0.5 * dt * dt, 0, 0);
  v1 = (Vector3() << 0, 0, 0).finished();
  v2 = (Vector3() << dt * 1.0, 0, 0).finished();
  a1 = (Vector3() << 1, 0, 0).finished();
  a2 = (Vector3() << 1, 0, 0).finished();
  actual = factor.evaluateError(p1, v1, a1, p2, v2, a2, actualH1, actualH2, actualH3, actualH4 ,actualH5, actualH6);
  expect = (Vector(9) << 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-6);

  EXPECT(assert_equal(expect, actual, 1e-6));
  EXPECT(assert_equal(expectH1, actualH1, 1e-6));
  EXPECT(assert_equal(expectH2, actualH2, 1e-8));
  EXPECT(assert_equal(expectH3, actualH3, 1e-8));
  EXPECT(assert_equal(expectH4, actualH4, 1e-6));
  EXPECT(assert_equal(expectH5, actualH5, 1e-8));
  EXPECT(assert_equal(expectH6, actualH6, 1e-8));

  // random to checkou non-zero error jacobian.
  p1 = Pose2(-0.1, 1.2, 0.3);
  p2 = Pose2(2.4, -2.5, 3.7);
  v1 = (Vector3() << 5, 4, 9).finished();
  v2 = (Vector3() << 0, 6, 4).finished();
  a1 = (Vector3() << 2, 3, -2).finished();
  a2 = (Vector3() << 1, -2, 0.1).finished();
  factor.evaluateError(p1, v1, a1, p2, v2, a2, actualH1, actualH2, actualH3, actualH4 ,actualH5, actualH6);

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-6);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-6);

  EXPECT(assert_equal(expectH1, actualH1, 1e-8));
  EXPECT(assert_equal(expectH2, actualH2, 1e-8));
  EXPECT(assert_equal(expectH3, actualH3, 1e-8));
  EXPECT(assert_equal(expectH4, actualH4, 1e-8));
  EXPECT(assert_equal(expectH5, actualH5, 1e-8));
  EXPECT(assert_equal(expectH6, actualH6, 1e-8));


}

TEST(GaussianProcessPriorLie, Optimization) {
  /**
   * A simple graph:
   *
   *  p1   p2
   *  |    |
   *  x1   x2
   *   \  /
   * a1-gp-a2
   *   /  \
   *  v1  v2
   *
   * p1 and p2 are pose prior factor to fix the poses, gp is the GP factor
   * that get correct velocity of v2
   */

  noiseModel::Isotropic::shared_ptr model_prior = noiseModel::Isotropic::Sigma(3, 0.001);
  double delta_t = 1;
  Matrix Qc = 0.01 * Matrix::Identity(3,3);
  noiseModel::Gaussian::shared_ptr Qc_model = noiseModel::Gaussian::Covariance(Qc);

  Pose2 pose1(0,0,0), pose2(1,0,0);
  Vector v1 = (Vector(3) << 1, 0, 0).finished();
  Vector v2 = (Vector(3) << 2.0, -0.5, 0.6).finished();   // rnd value
  Vector a1 = (Vector(3) << 0, 0, 0).finished();
  Vector a2 = (Vector(3) << 0, 0, 0).finished();   // rnd value

  NonlinearFactorGraph graph;
  graph.add(PriorFactor<Pose2>(Symbol('x', 1), pose1, model_prior));
  graph.add(PriorFactor<Pose2>(Symbol('x', 2), pose2, model_prior));
  graph.add(PriorFactor<Vector3>(Symbol('v', 1), v1, model_prior));
  // graph.add(PriorFactor<Vector3>(Symbol('a', 1), a1, model_prior));
  graph.add(GaussianProcessPriorLie<Pose2>(Symbol('x', 1), Symbol('v', 1), Symbol('a', 1),
      Symbol('x', 2), Symbol('v', 2), Symbol('a', 2), delta_t, Qc_model));

  Values init_values;
  init_values.insert(Symbol('x', 1), pose1);
  init_values.insert(Symbol('v', 1), v1);
  init_values.insert(Symbol('a', 1), a1);
  init_values.insert(Symbol('x', 2), pose2);
  init_values.insert(Symbol('v', 2), v2);
  init_values.insert(Symbol('a', 2), a2);

  GaussNewtonParams parameters;
  GaussNewtonOptimizer optimizer(graph, init_values, parameters);
  optimizer.optimize();
  Values values = optimizer.values();

  EXPECT_DOUBLES_EQUAL(0, graph.error(values), 1e-6);
  EXPECT(assert_equal(pose1, values.at<Pose2>(Symbol('x', 1)), 1e-6));
  EXPECT(assert_equal(pose2, values.at<Pose2>(Symbol('x', 2)), 1e-6));
  EXPECT(assert_equal(v1, values.at<Vector>(Symbol('v', 1)), 1e-6));
  EXPECT(assert_equal(v1, values.at<Vector>(Symbol('v', 2)), 1e-6));
  EXPECT(assert_equal(a1, values.at<Vector>(Symbol('a', 1)), 1e-6));
  EXPECT(assert_equal(a2, values.at<Vector>(Symbol('a', 2)), 1e-6));
}

TEST(GaussianProcessPriorLie, Optimization2) {
  /**
   * A simple graph:
   *
   *  p1   p2
   *  |    |
   *  x1   x2
   *   \  /
   * a1-gp-a2
   *   /  \
   *  v1  v2
   *
   * p1 and p2 are pose prior factor to fix the poses, gp is the GP factor
   * that get correct velocity of v2
   */

  noiseModel::Isotropic::shared_ptr model_prior = noiseModel::Isotropic::Sigma(3, 0.001);
  double delta_t = 1;
  Matrix Qc = 0.01 * Matrix::Identity(3,3);
  noiseModel::Gaussian::shared_ptr Qc_model = noiseModel::Gaussian::Covariance(Qc);

  Pose2 pose1(0,0,0), pose2(1,1,1);
  Vector v1 = (Vector(3) << 0, 0, 0).finished();
  Vector v2 = (Vector(3) << -3.0, -0.5, 0.6).finished();   // rnd value
  Vector a1 = (Vector(3) << 1, 0, 0).finished();
  Vector a2 = (Vector(3) << -1, 3, 0).finished();   // rnd value

  NonlinearFactorGraph graph;
  graph.add(PriorFactor<Pose2>(Symbol('x', 1), pose1, model_prior));
  graph.add(PriorFactor<Vector3>(Symbol('v', 1), v1, model_prior));
  graph.add(PriorFactor<Vector3>(Symbol('a', 1), a1, model_prior));
  graph.add(GaussianProcessPriorLie<Pose2>(Symbol('x', 1), Symbol('v', 1), Symbol('a', 1),
      Symbol('x', 2), Symbol('v', 2), Symbol('a', 2), delta_t, Qc_model));

  Values init_values;
  init_values.insert(Symbol('x', 1), pose1);
  init_values.insert(Symbol('v', 1), v1);
  init_values.insert(Symbol('a', 1), a1);
  init_values.insert(Symbol('x', 2), pose2);
  init_values.insert(Symbol('v', 2), v2);
  init_values.insert(Symbol('a', 2), a2);

  GaussNewtonParams parameters;
  GaussNewtonOptimizer optimizer(graph, init_values, parameters);
  optimizer.optimize();
  Values values = optimizer.values();

  EXPECT_DOUBLES_EQUAL(0, graph.error(values), 1e-6);
  EXPECT(assert_equal(pose1, values.at<Pose2>(Symbol('x', 1)), 1e-6));
  EXPECT(assert_equal(Pose2(0.5, 0, 0), values.at<Pose2>(Symbol('x', 2)), 1e-6));
  EXPECT(assert_equal(v1, values.at<Vector>(Symbol('v', 1)), 1e-6));
  EXPECT(assert_equal(Vector3(1,0,0), values.at<Vector>(Symbol('v', 2)), 1e-6));
  EXPECT(assert_equal(a1, values.at<Vector>(Symbol('a', 1)), 1e-6));
  EXPECT(assert_equal(a1, values.at<Vector>(Symbol('a', 2)), 1e-6));
}
/* ************************************************************************** */

/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
