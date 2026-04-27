/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FiniteDeformationForms.h"
#include "Spaces/SpaceFactory.h"

namespace svmp {
namespace FE {
namespace forms {
namespace test {
namespace {

FormExpr diagonalTensor(Real a, Real b, Real c)
{
    const auto z = FormExpr::constant(Real(0));
    return FormExpr::asTensor({
        {FormExpr::constant(a), z, z},
        {z, FormExpr::constant(b), z},
        {z, z, FormExpr::constant(c)},
    });
}

} // namespace

TEST(FiniteDeformationForms, BuildsFrameExplicitKinematicTerminals)
{
    auto vector_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                            ElementType::Tetra4,
                                            /*order=*/1,
                                            /*components=*/3);
    const auto u = FormExpr::trialFunction(*vector_space, "u");

    const auto kin = finite_deformation::kinematics(u, 3);

    ASSERT_TRUE(kin.F.isValid());
    ASSERT_TRUE(kin.J.isValid());
    ASSERT_TRUE(kin.Finv.isValid());
    ASSERT_TRUE(kin.FinvT.isValid());
    ASSERT_TRUE(kin.C.isValid());
    ASSERT_TRUE(kin.b.isValid());
    ASSERT_TRUE(kin.green_lagrange.isValid());
    ASSERT_TRUE(kin.almansi.isValid());
    EXPECT_EQ(kin.J.node()->type(), FormExprType::Determinant);
    EXPECT_EQ(kin.Finv.node()->type(), FormExprType::Inverse);
    EXPECT_TRUE(kin.F.hasTrial());
}

TEST(FiniteDeformationForms, BuildsConsistentLinearizationExpressionsWithoutFiniteDifferences)
{
    auto vector_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                            ElementType::Tetra4,
                                            /*order=*/1,
                                            /*components=*/3);
    const auto u = FormExpr::trialFunction(*vector_space, "u");
    const auto du = FormExpr::trialFunction(*vector_space, "du");

    const auto lin = finite_deformation::linearizeKinematics(u, du, 3);

    ASSERT_TRUE(lin.dF.isValid());
    ASSERT_TRUE(lin.dJ.isValid());
    ASSERT_TRUE(lin.dFinv.isValid());
    ASSERT_TRUE(lin.dGreenLagrange.isValid());
    ASSERT_TRUE(lin.dAlmansi.isValid());
    EXPECT_EQ(lin.dF.node()->type(), FormExprType::Gradient);
    EXPECT_EQ(lin.dJ.node()->type(), FormExprType::DoubleContraction);
    EXPECT_TRUE(lin.dAlmansi.hasTrial());
}

TEST(FiniteDeformationForms, ProvidesPushPullAndGradientFrameHelpers)
{
    auto vector_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                            ElementType::Tetra4,
                                            /*order=*/1,
                                            /*components=*/3);
    const auto u = FormExpr::trialFunction(*vector_space, "u");
    const auto F = finite_deformation::deformationGradient(u, 3);
    const auto g_ref = FormExpr::asVector({FormExpr::constant(Real(1)),
                                           FormExpr::constant(Real(2)),
                                           FormExpr::constant(Real(3))});

    const auto g_cur = finite_deformation::scalarCurrentGradientFromReferenceGradient(g_ref, F);
    const auto g_back = finite_deformation::scalarReferenceGradientFromCurrentGradient(g_cur, F);
    const auto hdiv = finite_deformation::contravariantPiolaPushForward(g_ref, F);
    const auto hcurl = finite_deformation::covariantPiolaPushForward(g_ref, F);

    EXPECT_TRUE(g_cur.isValid());
    EXPECT_TRUE(g_back.isValid());
    EXPECT_TRUE(hdiv.isValid());
    EXPECT_TRUE(hcurl.isValid());
    EXPECT_TRUE(g_back.hasTrial());
}

TEST(FiniteDeformationForms, ProvidesAssemblyNeutralGeometricSensitivityHooks)
{
    auto vector_space = spaces::VectorSpace(spaces::SpaceType::H1,
                                            ElementType::Tetra4,
                                            /*order=*/1,
                                            /*components=*/3);
    const auto u = FormExpr::trialFunction(*vector_space, "u");
    const auto du = FormExpr::trialFunction(*vector_space, "du");
    const auto v = FormExpr::testFunction(*vector_space, "v");
    const auto F = finite_deformation::deformationGradient(u, 3);
    const auto dF = finite_deformation::deformationGradientVariation(du);
    const auto N = FormExpr::normal();
    const auto stress = diagonalTensor(Real(2), Real(3), Real(4));

    const auto follower_tangent =
        finite_deformation::nansonMeasureVectorVariation(F, dF, N);
    const auto geometric_stiffness =
        finite_deformation::initialStressGeometricStiffnessDensity(stress, du, v);
    const auto pk1_virtual_work =
        finite_deformation::pk1InternalVirtualWorkDensity(stress, v);

    EXPECT_TRUE(follower_tangent.isValid());
    EXPECT_TRUE(geometric_stiffness.isValid());
    EXPECT_TRUE(pk1_virtual_work.isValid());
    EXPECT_TRUE(geometric_stiffness.hasTest());
    EXPECT_TRUE(geometric_stiffness.hasTrial());
    EXPECT_TRUE(pk1_virtual_work.hasTest());
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
