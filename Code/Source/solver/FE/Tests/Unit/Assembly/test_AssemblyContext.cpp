/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_AssemblyContext.cpp
 * @brief Unit tests for AssemblyContext
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyContext.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// AssemblyContext Tests
// ============================================================================

class AssemblyContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_.reserve(10, 10, 3);
    }

    AssemblyContext ctx_;
};

TEST_F(AssemblyContextTest, DefaultConstruction) {
    AssemblyContext ctx;
    EXPECT_EQ(ctx.numTestDofs(), 0);
    EXPECT_EQ(ctx.numTrialDofs(), 0);
    EXPECT_EQ(ctx.numQuadraturePoints(), 0);
}

TEST_F(AssemblyContextTest, SetQuadratureData) {
    std::vector<AssemblyContext::Point3D> points = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.5, 0.5, 0.0}
    };
    std::vector<Real> weights = {0.5, 0.25, 0.25};

    ctx_.setQuadratureData(points, weights);

    EXPECT_EQ(ctx_.numQuadraturePoints(), 3);

    auto pt0 = ctx_.quadraturePoint(0);
    EXPECT_DOUBLE_EQ(pt0[0], 0.0);
    EXPECT_DOUBLE_EQ(pt0[1], 0.0);

    EXPECT_DOUBLE_EQ(ctx_.quadratureWeight(0), 0.5);
    EXPECT_DOUBLE_EQ(ctx_.quadratureWeight(1), 0.25);
}

TEST_F(AssemblyContextTest, SetPhysicalPoints) {
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx_.setQuadratureData(quad_pts, weights);

    std::vector<AssemblyContext::Point3D> phys_pts = {{1.5, 2.5, 3.5}};
    ctx_.setPhysicalPoints(phys_pts);

    auto pt = ctx_.physicalPoint(0);
    EXPECT_DOUBLE_EQ(pt[0], 1.5);
    EXPECT_DOUBLE_EQ(pt[1], 2.5);
    EXPECT_DOUBLE_EQ(pt[2], 3.5);
}

TEST_F(AssemblyContextTest, SetJacobianData) {
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx_.setQuadratureData(quad_pts, weights);

    AssemblyContext::Matrix3x3 J = {{{1.0, 0.0, 0.0},
                                      {0.0, 2.0, 0.0},
                                      {0.0, 0.0, 3.0}}};
    AssemblyContext::Matrix3x3 Jinv = {{{1.0, 0.0, 0.0},
                                         {0.0, 0.5, 0.0},
                                         {0.0, 0.0, 1.0/3.0}}};
    std::vector<AssemblyContext::Matrix3x3> jacs = {J};
    std::vector<AssemblyContext::Matrix3x3> jinvs = {Jinv};
    std::vector<Real> dets = {6.0};

    ctx_.setJacobianData(jacs, jinvs, dets);

    EXPECT_DOUBLE_EQ(ctx_.jacobianDet(0), 6.0);

    auto jac = ctx_.jacobian(0);
    EXPECT_DOUBLE_EQ(jac[1][1], 2.0);

    auto inv = ctx_.inverseJacobian(0);
    EXPECT_DOUBLE_EQ(inv[1][1], 0.5);
}

TEST_F(AssemblyContextTest, SetIntegrationWeights) {
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx_.setQuadratureData(quad_pts, weights);

    std::vector<Real> int_wts = {0.25, 0.125};  // weights * |J|
    ctx_.setIntegrationWeights(int_wts);

    EXPECT_DOUBLE_EQ(ctx_.integrationWeight(0), 0.25);
    EXPECT_DOUBLE_EQ(ctx_.integrationWeight(1), 0.125);
}

TEST_F(AssemblyContextTest, SetBasisData) {
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx_.setQuadratureData(quad_pts, weights);

    LocalIndex n_dofs = 2;
    LocalIndex n_qpts = 2;

    // Values: phi_i(q) stored as [i * n_qpts + q]
    std::vector<Real> values = {
        1.0, 0.5,   // phi_0 at q=0,1
        0.0, 0.5    // phi_1 at q=0,1
    };

    std::vector<AssemblyContext::Vector3D> grads = {
        {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},  // grad phi_0 at q=0,1
        { 1.0, 0.0, 0.0}, { 1.0, 0.0, 0.0}   // grad phi_1 at q=0,1
    };

    ctx_.setTestBasisData(n_dofs, values, grads);

    EXPECT_EQ(ctx_.numTestDofs(), 2);
    EXPECT_DOUBLE_EQ(ctx_.basisValue(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(ctx_.basisValue(0, 1), 0.5);
    EXPECT_DOUBLE_EQ(ctx_.basisValue(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(ctx_.basisValue(1, 1), 0.5);

    auto grad = ctx_.referenceGradient(0, 0);
    EXPECT_DOUBLE_EQ(grad[0], -1.0);
}

TEST_F(AssemblyContextTest, SetPhysicalGradients) {
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx_.setQuadratureData(quad_pts, weights);

    LocalIndex n_dofs = 2;
    std::vector<Real> values = {1.0, 0.0};
    std::vector<AssemblyContext::Vector3D> ref_grads = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    ctx_.setTestBasisData(n_dofs, values, ref_grads);

    // Physical gradients (transformed by J^{-T})
    std::vector<AssemblyContext::Vector3D> phys_grads = {
        {2.0, 0.0, 0.0},
        {0.0, 2.0, 0.0}
    };
    ctx_.setPhysicalGradients(phys_grads, phys_grads);

    auto grad = ctx_.physicalGradient(0, 0);
    EXPECT_DOUBLE_EQ(grad[0], 2.0);
}

TEST_F(AssemblyContextTest, TrialBasisWithSameAsTest) {
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx_.setQuadratureData(quad_pts, weights);

    LocalIndex n_dofs = 2;
    std::vector<Real> values = {1.0, 0.5};
    std::vector<AssemblyContext::Vector3D> grads = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    };
    ctx_.setTestBasisData(n_dofs, values, grads);

    std::vector<AssemblyContext::Vector3D> phys_grads = {
        {2.0, 0.0, 0.0},
        {0.0, 2.0, 0.0}
    };
    ctx_.setPhysicalGradients(phys_grads, phys_grads);

    // Trial should return same as test (trial_is_test_ flag)
    EXPECT_DOUBLE_EQ(ctx_.trialBasisValue(0, 0), ctx_.basisValue(0, 0));

    auto trial_grad = ctx_.trialPhysicalGradient(0, 0);
    auto test_grad = ctx_.physicalGradient(0, 0);
    EXPECT_DOUBLE_EQ(trial_grad[0], test_grad[0]);
}

TEST_F(AssemblyContextTest, SolutionData) {
    // Setup basic context
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx_.setQuadratureData(quad_pts, weights);

    LocalIndex n_dofs = 2;
    std::vector<Real> values = {1.0, 0.5, 0.0, 0.5};  // Linear interpolation
    std::vector<AssemblyContext::Vector3D> grads = {
        {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
        { 1.0, 0.0, 0.0}, { 1.0, 0.0, 0.0}
    };
    ctx_.setTestBasisData(n_dofs, values, grads);

    std::vector<AssemblyContext::Vector3D> phys_grads = grads;  // Unit element
    ctx_.setPhysicalGradients(phys_grads, phys_grads);

    // Set solution coefficients
    std::vector<Real> coeffs = {1.0, 3.0};  // u = 1*phi_0 + 3*phi_1
    ctx_.setSolutionCoefficients(coeffs);

    EXPECT_TRUE(ctx_.hasSolutionData());

    // u(q=0) = 1*1.0 + 3*0.0 = 1.0
    EXPECT_DOUBLE_EQ(ctx_.solutionValue(0), 1.0);

    // u(q=1) = 1*0.5 + 3*0.5 = 2.0
    EXPECT_DOUBLE_EQ(ctx_.solutionValue(1), 2.0);

    // grad u = 1*(-1,0,0) + 3*(1,0,0) = (2,0,0)
    auto sol_grad = ctx_.solutionGradient(0);
    EXPECT_DOUBLE_EQ(sol_grad[0], 2.0);
}

TEST_F(AssemblyContextTest, Clear) {
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx_.setQuadratureData(quad_pts, weights);

    EXPECT_EQ(ctx_.numQuadraturePoints(), 1);

    ctx_.clear();

    EXPECT_EQ(ctx_.numQuadraturePoints(), 0);
    EXPECT_EQ(ctx_.numTestDofs(), 0);
}

TEST_F(AssemblyContextTest, BasisValuesSpan) {
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx_.setQuadratureData(quad_pts, weights);

    LocalIndex n_dofs = 2;
    std::vector<Real> values = {1.0, 0.5, 0.0, 0.5};
    std::vector<AssemblyContext::Vector3D> grads(4);
    ctx_.setTestBasisData(n_dofs, values, grads);

    auto vals = ctx_.basisValues(0);
    EXPECT_EQ(vals.size(), 2u);
    EXPECT_DOUBLE_EQ(vals[0], 1.0);
    EXPECT_DOUBLE_EQ(vals[1], 0.5);
}

// ============================================================================
// AssemblyContextPool Tests
// ============================================================================

TEST(AssemblyContextPoolTest, Construction) {
    AssemblyContextPool pool(4, 10, 10, 3);
    EXPECT_EQ(pool.size(), 4);
}

TEST(AssemblyContextPoolTest, GetContext) {
    AssemblyContextPool pool(2, 5, 5, 2);

    auto& ctx0 = pool.getContext(0);
    auto& ctx1 = pool.getContext(1);

    // Should be different objects
    EXPECT_NE(&ctx0, &ctx1);
}

TEST(AssemblyContextPoolTest, OutOfRangeThrows) {
    AssemblyContextPool pool(2, 5, 5, 2);

    EXPECT_THROW(pool.getContext(-1), std::out_of_range);
    EXPECT_THROW(pool.getContext(2), std::out_of_range);
    EXPECT_THROW(pool.getContext(10), std::out_of_range);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(AssemblyContextErrors, QuadratureOutOfRange) {
    AssemblyContext ctx;
    std::vector<AssemblyContext::Point3D> pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> wts = {1.0};
    ctx.setQuadratureData(pts, wts);

    EXPECT_THROW(ctx.quadraturePoint(5), std::out_of_range);
    EXPECT_THROW(ctx.quadratureWeight(5), std::out_of_range);
    EXPECT_THROW(ctx.integrationWeight(5), std::out_of_range);
}

TEST(AssemblyContextErrors, BasisOutOfRange) {
    AssemblyContext ctx;
    std::vector<AssemblyContext::Point3D> pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> wts = {1.0};
    ctx.setQuadratureData(pts, wts);

    std::vector<Real> vals = {1.0};
    std::vector<AssemblyContext::Vector3D> grads = {{0.0, 0.0, 0.0}};
    ctx.setTestBasisData(1, vals, grads);

    EXPECT_THROW(ctx.basisValue(5, 0), std::out_of_range);
    EXPECT_THROW(ctx.basisValue(0, 5), std::out_of_range);
}

TEST(AssemblyContextErrors, SolutionNotSet) {
    AssemblyContext ctx;
    EXPECT_FALSE(ctx.hasSolutionData());
    EXPECT_THROW(ctx.solutionValue(0), std::logic_error);
    EXPECT_THROW(ctx.solutionGradient(0), std::logic_error);
}

TEST(AssemblyContextErrors, NormalForCellContext) {
    AssemblyContext ctx;
    std::vector<AssemblyContext::Point3D> pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> wts = {1.0};
    ctx.setQuadratureData(pts, wts);

    // Cell context (default) should throw for normal access
    EXPECT_THROW(ctx.normal(0), std::logic_error);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
