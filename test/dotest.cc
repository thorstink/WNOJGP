/**
 *  @file testGaussianProcessInterpolatorPose2.cpp
 *  @author Jing Dong
 **/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include "wnoj_gp/GaussianProcessInterpolatorLie.h"
#include "wnoj_gp/GaussianProcessPriorLie.h"
#include <gtsam/navigation/Scenario.h>

#include <iostream>

using namespace std;
using namespace gtsam;
using namespace wnoj;

typedef GaussianProcessInterpolatorLie<gtsam::Pose2>
    GaussianProcessInterpolatorPose2;

typedef GaussianProcessInterpolatorLie<gtsam::Pose3>
    GaussianProcessInterpolatorPose3;

TEST(GaussianProcessInterpolatorPose3, interpolatePose) {
  Pose3 p1, p2, expect, actual;
  Vector6 v1, v2, a1, a2;
  Matrix actualH1, actualH2, actualH3, actualH4, actualH5, actualH6;
  Matrix expectH1, expectH2, expectH3, expectH4, expectH5, expectH6;
  Matrix6 Qc = 0.1 * Matrix::Identity(6, 6);
  noiseModel::Gaussian::shared_ptr Qc_model =
      noiseModel::Gaussian::Covariance(Qc);
  double dt = 1.0, tau = 0.97;
  GaussianProcessInterpolatorPose3 base(Qc_model, dt, tau);

  // test at origin
  p1 = Pose3(Rot3(), Vector3(0, 0, 0));
  p2 = Pose3(Rot3(), Vector3(0, 0, 0));
  v1 = (Vector6() << 0, 0, 0, 0, 0, 0).finished();
  v2 = (Vector6() << 0, 0, 0, 0, 0, 0).finished();
  a1 = (Vector6() << 0, 0, 0, 0, 0, 0).finished();
  a2 = (Vector6() << 0, 0, 0, 0, 0, 0).finished();

  actual = base.interpolatePose(p1, v1, a1, p2, v2, a2, actualH1, actualH2,
                                actualH3, actualH4, actualH5, actualH6);
  expect = Pose3();

  // helper function
  boost::function<Pose3(const Pose3 &, const Vector6 &, const Vector6 &,
                        const Pose3 &, const Vector6 &, const Vector6 &)>
      f1 = [base](const Pose3 &p1, const Vector6 &v1, const Vector6 &a1,
                  const Pose3 &p2, const Vector6 &v2, const Vector6 &a2) {
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
  EXPECT(assert_equal(expectH1, actualH1, 1e-5));
  EXPECT(assert_equal(expectH2, actualH2, 1e-5));
  EXPECT(assert_equal(expectH3, actualH3, 1e-5));
  EXPECT(assert_equal(expectH4, actualH4, 1e-5));
  EXPECT(assert_equal(expectH5, actualH5, 1e-5));
  EXPECT(assert_equal(expectH6, actualH6, 1e-5));

  // stant velocity
  p1 = Pose3(Rot3::Rz(M_PI_2), Vector3(0, 0, 0));
  Vector3 w, v;
  w = Vector3(0, 0, 0.5);
  v = Vector3(0.5, 0, 0);
  auto s = ConstantTwistScenario(w, v, p1);
  p2 = s.pose(dt);
  v1 = (Vector6() << w, v).finished();
  v2 = (Vector6() << w, v).finished();
  a1 = (Vector6() << 0, 0, 0, 0, 0, 0).finished();
  a2 = (Vector6() << 0, 0, 0, 0, 0, 0).finished();

  actual = base.interpolatePose(p1, v1, a1, p2, v2, a2, actualH1, actualH2,
                                actualH3, actualH4, actualH5, actualH6);

  expect = s.pose(tau);

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-4);

  EXPECT(assert_equal(expect, actual, 1e-6));
  EXPECT(assert_equal(expectH1, actualH1, 1e-5));
  EXPECT(assert_equal(expectH2, actualH2, 1e-5));
  EXPECT(assert_equal(expectH3, actualH3, 1e-5));
  EXPECT(assert_equal(expectH4, actualH4, 1e-5));
  EXPECT(assert_equal(expectH5, actualH5, 1e-5));
  EXPECT(assert_equal(expectH6, actualH6, 1e-5));

  // acc
  // test forward constant acceleration
  a1 = (Vector6() << 0, 0, 0.5, 1, 0, 0).finished();
  a2 = (Vector6() << 0, 0, 0.5, 1, 0, 0).finished();

  actual = base.interpolatePose(p1, v1, a1, p2, v2, a2, actualH1, actualH2,
                                actualH3, actualH4, actualH5, actualH6);

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-4);

  p1.print();
  p2.print();
  expect.print();
  actual.print();

  EXPECT(0);

  EXPECT(assert_equal(expectH1, actualH1, 1e-5));
  EXPECT(assert_equal(expectH2, actualH2, 1e-5));
  EXPECT(assert_equal(expectH3, actualH3, 1e-5));
  EXPECT(assert_equal(expectH4, actualH4, 1e-5));
  EXPECT(assert_equal(expectH5, actualH5, 1e-5));
  EXPECT(assert_equal(expectH6, actualH6, 1e-5));

  a1 = (Vector6() << 0, 0, 0.5, 1, 0, 0).finished();
  a2 = (Vector6() << 0, 0, 0, 0, 0, 0).finished();

  actual = base.interpolatePose(p1, v1, a1, p2, v2, a2, actualH1, actualH2,
                                actualH3, actualH4, actualH5, actualH6);

  // Calculate derivates using numerical approximation
  expectH1 = numericalDerivative61(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH2 = numericalDerivative62(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH3 = numericalDerivative63(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH4 = numericalDerivative64(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH5 = numericalDerivative65(f1, p1, v1, a1, p2, v2, a2, 1e-4);
  expectH6 = numericalDerivative66(f1, p1, v1, a1, p2, v2, a2, 1e-4);

  p1.print();
  p2.print();
  expect.print();
  actual.print();

  EXPECT(0);

  EXPECT(assert_equal(expectH1, actualH1, 1e-5));
  EXPECT(assert_equal(expectH2, actualH2, 1e-5));
  EXPECT(assert_equal(expectH3, actualH3, 1e-5));
  EXPECT(assert_equal(expectH4, actualH4, 1e-5));
  EXPECT(assert_equal(expectH5, actualH5, 1e-5));
  EXPECT(assert_equal(expectH6, actualH6, 1e-5));
}

/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
