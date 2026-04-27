/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Geometry/FiniteDeformationKinematics.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {
namespace test {
namespace {

void expectMatrixNear(const math::Matrix<Real, 3, 3>& a,
                      const math::Matrix<Real, 3, 3>& b,
                      Real tol)
{
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(a(i, j), b(i, j), tol) << "entry (" << i << ", " << j << ")";
        }
    }
}

void expectVectorNear(const math::Vector<Real, 3>& a,
                      const math::Vector<Real, 3>& b,
                      Real tol)
{
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "entry " << i;
    }
}

math::Matrix<Real, 3, 3> plusScaled(const math::Matrix<Real, 3, 3>& a,
                                    const math::Matrix<Real, 3, 3>& b,
                                    Real scale)
{
    math::Matrix<Real, 3, 3> out{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            out(i, j) = a(i, j) + scale * b(i, j);
        }
    }
    return out;
}

} // namespace

TEST(FiniteDeformationKinematics, LargeRotationPatchHasZeroFiniteStrain)
{
    const Real angle = Real(0.5) * Real(3.141592653589793238462643383279502884);
    math::Matrix<Real, 3, 3> R{};
    R(0, 0) = std::cos(angle);
    R(0, 1) = -std::sin(angle);
    R(1, 0) = std::sin(angle);
    R(1, 1) = std::cos(angle);
    R(2, 2) = Real(1);

    const auto kin = finiteDeformationFromGradient(R, 3);

    EXPECT_NEAR(kin.J, Real(1), Real(1e-14));
    expectMatrixNear(kin.C, finiteDeformationIdentity(), Real(1e-14));
    expectMatrixNear(kin.b, finiteDeformationIdentity(), Real(1e-14));
    expectMatrixNear(kin.green_lagrange, math::Matrix<Real, 3, 3>{}, Real(1e-14));
    expectMatrixNear(kin.almansi, math::Matrix<Real, 3, 3>{}, Real(1e-14));
}

TEST(FiniteDeformationKinematics, StretchInflationBenchmarkHasExpectedInvariants)
{
    const Real lambda = Real(1.2);
    math::Matrix<Real, 3, 3> F{};
    F(0, 0) = lambda;
    F(1, 1) = lambda;
    F(2, 2) = Real(1) / (lambda * lambda);

    const auto kin = finiteDeformationFromGradient(F, 3);

    EXPECT_NEAR(kin.J, Real(1), Real(1e-14));
    EXPECT_NEAR(kin.C(0, 0), lambda * lambda, Real(1e-14));
    EXPECT_NEAR(kin.C(1, 1), lambda * lambda, Real(1e-14));
    EXPECT_NEAR(kin.C(2, 2), Real(1) / (lambda * lambda * lambda * lambda),
                Real(1e-14));
    EXPECT_NEAR(kin.green_lagrange(0, 0), Real(0.5) * (lambda * lambda - Real(1)),
                Real(1e-14));
    EXPECT_NEAR(kin.almansi(0, 0), Real(0.5) * (Real(1) - Real(1) / (lambda * lambda)),
                Real(1e-14));
}

TEST(FiniteDeformationKinematics, CantileverLargeDeflectionShearPatchIsQuadratic)
{
    const Real gamma = Real(0.65);
    math::Matrix<Real, 3, 3> grad_u{};
    grad_u(0, 1) = gamma;

    const auto kin = finiteDeformationFromDisplacementGradient(grad_u, 3);

    EXPECT_NEAR(kin.green_lagrange(0, 0), Real(0), Real(1e-14));
    EXPECT_NEAR(kin.green_lagrange(0, 1), Real(0.5) * gamma, Real(1e-14));
    EXPECT_NEAR(kin.green_lagrange(1, 0), Real(0.5) * gamma, Real(1e-14));
    EXPECT_NEAR(kin.green_lagrange(1, 1), Real(0.5) * gamma * gamma,
                Real(1e-14));
}

TEST(FiniteDeformationKinematics, AnalyticKinematicLinearizationMatchesFiniteDifference)
{
    math::Matrix<Real, 3, 3> F{};
    F(0, 0) = Real(1.25);
    F(0, 1) = Real(0.15);
    F(0, 2) = Real(-0.04);
    F(1, 0) = Real(0.07);
    F(1, 1) = Real(0.92);
    F(1, 2) = Real(0.11);
    F(2, 0) = Real(0.03);
    F(2, 1) = Real(-0.08);
    F(2, 2) = Real(1.08);

    math::Matrix<Real, 3, 3> dF{};
    dF(0, 0) = Real(0.09);
    dF(0, 1) = Real(-0.03);
    dF(0, 2) = Real(0.02);
    dF(1, 0) = Real(0.01);
    dF(1, 1) = Real(0.04);
    dF(1, 2) = Real(-0.05);
    dF(2, 0) = Real(0.02);
    dF(2, 1) = Real(0.03);
    dF(2, 2) = Real(-0.07);

    const auto kin = finiteDeformationFromGradient(F, 3);
    const auto inc = linearizeFiniteDeformationKinematics(kin, dF);
    const Real eps = Real(1e-7);
    const auto plus = finiteDeformationFromGradient(plusScaled(F, dF, eps), 3);
    const auto minus = finiteDeformationFromGradient(plusScaled(F, dF, -eps), 3);

    EXPECT_NEAR(inc.dJ, (plus.J - minus.J) / (Real(2) * eps), Real(2e-8));
    expectMatrixNear(inc.dFinv,
                     (plus.Finv - minus.Finv) * (Real(0.5) / eps),
                     Real(2e-8));
    expectMatrixNear(inc.dGreenLagrange,
                     (plus.green_lagrange - minus.green_lagrange) *
                         (Real(0.5) / eps),
                     Real(2e-8));
    expectMatrixNear(inc.dAlmansi,
                     (plus.almansi - minus.almansi) * (Real(0.5) / eps),
                     Real(2e-8));
}

TEST(FiniteDeformationKinematics, FollowerLoadNansonDerivativeMatchesFiniteDifference)
{
    math::Matrix<Real, 3, 3> F{};
    F(0, 0) = Real(1.2);
    F(0, 1) = Real(0.2);
    F(0, 2) = Real(0.1);
    F(1, 0) = Real(0.1);
    F(1, 1) = Real(0.9);
    F(1, 2) = Real(0.3);
    F(2, 0) = Real(0.0);
    F(2, 1) = Real(0.2);
    F(2, 2) = Real(1.1);

    math::Matrix<Real, 3, 3> dF{};
    dF(0, 0) = Real(0.03);
    dF(0, 1) = Real(-0.02);
    dF(0, 2) = Real(0.04);
    dF(1, 0) = Real(0.01);
    dF(1, 1) = Real(0.02);
    dF(1, 2) = Real(-0.01);
    dF(2, 0) = Real(-0.04);
    dF(2, 1) = Real(0.03);
    dF(2, 2) = Real(0.02);

    const math::Vector<Real, 3> reference_normal{Real(0), Real(0), Real(1)};
    const Real reference_measure = Real(1.7);

    const auto kin = finiteDeformationFromGradient(F, 3);
    const auto lin = linearizeNansonSurfaceTransform(reference_normal, reference_measure,
                                                     kin, dF);

    const Real eps = Real(1e-7);
    const auto plus = nansonSurfaceTransform(reference_normal, reference_measure,
                                             finiteDeformationFromGradient(plusScaled(F, dF, eps), 3));
    const auto minus = nansonSurfaceTransform(reference_normal, reference_measure,
                                              finiteDeformationFromGradient(plusScaled(F, dF, -eps), 3));

    expectVectorNear(lin.oriented_measure_vector_derivative,
                     (plus.oriented_measure_vector - minus.oriented_measure_vector) /
                         (Real(2) * eps),
                     Real(2e-8));
    EXPECT_NEAR(lin.measure_derivative,
                (plus.measure - minus.measure) / (Real(2) * eps),
                Real(2e-8));
    expectVectorNear(lin.normal_derivative,
                     (plus.normal - minus.normal) / (Real(2) * eps),
                     Real(2e-8));
}

TEST(FiniteDeformationKinematics, ReferenceAndCurrentGradientOperatorsRoundTrip)
{
    math::Matrix<Real, 3, 3> F{};
    F(0, 0) = Real(2);
    F(1, 1) = Real(3);
    F(2, 2) = Real(4);
    const auto kin = finiteDeformationFromGradient(F, 3);

    const math::Vector<Real, 3> grad_ref{Real(6), Real(9), Real(8)};
    const auto grad_cur = currentGradientFromReferenceGradient(grad_ref, kin);
    EXPECT_NEAR(grad_cur[0], Real(3), Real(1e-14));
    EXPECT_NEAR(grad_cur[1], Real(3), Real(1e-14));
    EXPECT_NEAR(grad_cur[2], Real(2), Real(1e-14));
    expectVectorNear(referenceGradientFromCurrentGradient(grad_cur, kin),
                     grad_ref, Real(1e-14));
}

TEST(FiniteDeformationKinematics, StaticSmallStrainRegressionMatchesSymmetricGradient)
{
    math::Matrix<Real, 3, 3> grad_u{};
    grad_u(0, 0) = Real(1e-8);
    grad_u(0, 1) = Real(2e-8);
    grad_u(1, 0) = Real(-1e-8);
    grad_u(1, 1) = Real(3e-8);
    grad_u(2, 2) = Real(-2e-8);

    const auto kin = finiteDeformationFromDisplacementGradient(grad_u, 3);
    const auto small_strain = (grad_u + grad_u.transpose()) * Real(0.5);

    expectMatrixNear(kin.green_lagrange, small_strain, Real(1e-15));
}

} // namespace test
} // namespace geometry
} // namespace FE
} // namespace svmp
