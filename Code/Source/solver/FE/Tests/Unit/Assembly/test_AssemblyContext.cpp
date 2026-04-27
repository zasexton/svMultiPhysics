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
#include "Spaces/H1Space.h"
#include "Spaces/HCurlSpace.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

namespace {

static void setupTwoDofTwoQptScalarContext(AssemblyContext& ctx)
{
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx.setQuadratureData(quad_pts, weights);

    const LocalIndex n_dofs = 2;
    const LocalIndex n_qpts = 2;

    std::vector<Real> values = {
        1.0, 0.5,   // phi_0 at q=0,1
        0.0, 0.5    // phi_1 at q=0,1
    };

    std::vector<AssemblyContext::Vector3D> grads = {
        {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
        { 1.0, 0.0, 0.0}, { 1.0, 0.0, 0.0}
    };

    ctx.setTestBasisData(n_dofs, values, grads);
    std::vector<AssemblyContext::Vector3D> phys_grads(static_cast<std::size_t>(n_dofs * n_qpts));
    for (LocalIndex q = 0; q < n_qpts; ++q) {
        for (LocalIndex i = 0; i < n_dofs; ++i) {
            phys_grads[static_cast<std::size_t>(q * n_dofs + i)] =
                grads[static_cast<std::size_t>(i * n_qpts + q)];
        }
    }
    ctx.setPhysicalGradients(phys_grads, phys_grads);

    // Minimal integration weights to avoid accidental use of unset storage.
    std::vector<Real> int_wts = {0.25, 0.25};
    ctx.setIntegrationWeights(int_wts);
}

} // namespace

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

TEST_F(AssemblyContextTest, CopyGeometryDataFromClonesPreparedGeometryState) {
    AssemblyContext src;
    src.reserve(4, 4, 3);

    const std::vector<AssemblyContext::Point3D> quad_pts = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    const std::vector<Real> weights = {0.25, 0.75};
    src.setQuadratureData(quad_pts, weights);

    const std::vector<AssemblyContext::Point3D> phys_pts = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    };
    src.setPhysicalPoints(phys_pts);

    const std::vector<AssemblyContext::Matrix3x3> jacs = {
        {{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}}},
        {{{4.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 6.0}}},
    };
    const std::vector<AssemblyContext::Matrix3x3> jinvs = {
        {{{1.0, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 1.0 / 3.0}}},
        {{{0.25, 0.0, 0.0}, {0.0, 0.2, 0.0}, {0.0, 0.0, 1.0 / 6.0}}},
    };
    const std::vector<Real> dets = {6.0, 120.0};
    src.setJacobianData(jacs, jinvs, dets);
    src.setIntegrationWeights(std::vector<Real>{1.5, 2.5});
    src.setEntityMeasures(7.0, 8.0, 0.0);

    AssemblyContext dst;
    dst.reserve(4, 4, 3);
    dst.copyGeometryDataFrom(src);

    EXPECT_EQ(dst.numQuadraturePoints(), 2);
    EXPECT_DOUBLE_EQ(dst.quadratureWeight(1), 0.75);
    EXPECT_DOUBLE_EQ(dst.physicalPoint(0)[1], 2.0);
    EXPECT_DOUBLE_EQ(dst.jacobianDet(1), 120.0);
    EXPECT_DOUBLE_EQ(dst.integrationWeight(0), 1.5);
    EXPECT_DOUBLE_EQ(dst.cellDiameter(), 7.0);
    EXPECT_DOUBLE_EQ(dst.cellVolume(), 8.0);
}

TEST_F(AssemblyContextTest, MovingDomainDataAccessorsAreFrameExplicitAndFailLoudly) {
    const std::vector<AssemblyContext::Point3D> quad_pts = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    };
    const std::vector<Real> weights = {0.5, 0.5};
    ctx_.setQuadratureData(quad_pts, weights);

    EXPECT_THROW((void)ctx_.meshDisplacement(0), std::logic_error);
    EXPECT_THROW((void)ctx_.meshVelocityJacobian(0), std::logic_error);
    EXPECT_THROW((void)ctx_.referencePhysicalPoint(0), std::logic_error);

    const std::vector<AssemblyContext::Point3D> reference_points = {
        {0.25, 0.25, 0.0},
        {0.75, 0.25, 0.0},
    };
    const std::vector<AssemblyContext::Point3D> current_points = {
        {1.25, 0.25, 0.0},
        {1.75, 0.25, 0.0},
    };

    const AssemblyContext::Matrix3x3 I = {{{1.0, 0.0, 0.0},
                                           {0.0, 1.0, 0.0},
                                           {0.0, 0.0, 1.0}}};
    const AssemblyContext::Matrix3x3 Jcur = {{{2.0, 0.0, 0.0},
                                              {0.0, 3.0, 0.0},
                                              {0.0, 0.0, 1.0}}};
    const std::vector<AssemblyContext::Matrix3x3> reference_jacs = {I, I};
    const std::vector<AssemblyContext::Matrix3x3> current_jacs = {Jcur, Jcur};
    const std::vector<Real> reference_measures = {1.0, 1.0};
    const std::vector<Real> current_measures = {6.0, 6.0};

    ctx_.setReferenceGeometry(reference_points, reference_jacs, reference_jacs, reference_measures);
    ctx_.setCurrentGeometry(current_points, current_jacs, reference_jacs, current_measures);
    ctx_.setConfigurationTransforms(current_jacs);
    ctx_.setSurfaceJacobians(current_jacs);

    const std::vector<AssemblyContext::Vector3D> normals = {
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    ctx_.setReferenceNormals(normals);
    ctx_.setCurrentNormals(normals);

    const std::vector<AssemblyContext::Vector3D> displacements = {
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    };
    const std::vector<AssemblyContext::Vector3D> velocities = {
        {0.0, 2.0, 0.0},
        {0.0, 2.0, 0.0},
    };
    const std::vector<AssemblyContext::Vector3D> accelerations = {
        {0.0, 0.0, 3.0},
        {0.0, 0.0, 3.0},
    };
    ctx_.setMeshDisplacements(displacements);
    ctx_.setMeshVelocities(velocities);
    ctx_.setMeshAccelerations(accelerations);
    ctx_.setPreviousCoordinates(reference_points);

    const AssemblyContext::Matrix3x3 displacement_jac = {{{0.1, 0.2, 0.0},
                                                          {0.0, 0.3, 0.0},
                                                          {0.0, 0.0, 0.0}}};
    const AssemblyContext::Matrix3x3 velocity_jac = {{{1.0, 0.0, 0.0},
                                                      {0.5, 2.0, 0.0},
                                                      {0.0, 0.0, 0.0}}};
    const AssemblyContext::Matrix3x3 acceleration_jac = {{{0.0, 0.0, 3.0},
                                                          {0.0, 0.0, 0.0},
                                                          {0.0, 0.0, 4.0}}};
    const std::vector<AssemblyContext::Matrix3x3> displacement_jacs = {displacement_jac, displacement_jac};
    const std::vector<AssemblyContext::Matrix3x3> velocity_jacs = {velocity_jac, velocity_jac};
    const std::vector<AssemblyContext::Matrix3x3> acceleration_jacs = {acceleration_jac, acceleration_jac};
    ctx_.setMeshDisplacementJacobians(displacement_jacs);
    ctx_.setMeshVelocityJacobians(velocity_jacs);
    ctx_.setMeshAccelerationJacobians(acceleration_jacs);
    ctx_.setPreviousMeshVelocities(velocities);
    ctx_.setPredictedMeshVelocities(velocities);
    ctx_.setPreviousMeshVelocityJacobians(velocity_jacs);
    ctx_.setPredictedMeshVelocityJacobians(velocity_jacs);

    EXPECT_DOUBLE_EQ(ctx_.referencePhysicalPoint(0)[0], 0.25);
    EXPECT_DOUBLE_EQ(ctx_.currentPhysicalPoint(1)[0], 1.75);
    EXPECT_DOUBLE_EQ(ctx_.currentJacobian(0)[1][1], 3.0);
    EXPECT_DOUBLE_EQ(ctx_.currentMeasure(1), 6.0);
    EXPECT_DOUBLE_EQ(ctx_.configurationTransform(0)[0][0], 2.0);
    EXPECT_DOUBLE_EQ(ctx_.surfaceJacobian(0)[1][1], 3.0);
    EXPECT_DOUBLE_EQ(ctx_.referenceNormal(0)[1], 1.0);
    EXPECT_DOUBLE_EQ(ctx_.currentNormal(1)[1], 1.0);
    EXPECT_DOUBLE_EQ(ctx_.meshDisplacement(0)[0], 1.0);
    EXPECT_DOUBLE_EQ(ctx_.meshVelocity(1)[1], 2.0);
    EXPECT_DOUBLE_EQ(ctx_.meshAcceleration(0)[2], 3.0);
    EXPECT_DOUBLE_EQ(ctx_.previousCoordinate(1)[0], 0.75);
    EXPECT_DOUBLE_EQ(ctx_.meshDisplacementJacobian(0)[0][1], 0.2);
    EXPECT_DOUBLE_EQ(ctx_.meshVelocityJacobian(1)[1][0], 0.5);
    EXPECT_DOUBLE_EQ(ctx_.meshAccelerationJacobian(0)[2][2], 4.0);
    EXPECT_DOUBLE_EQ(ctx_.previousMeshVelocityJacobian(0)[1][1], 2.0);
    EXPECT_DOUBLE_EQ(ctx_.predictedMeshVelocityJacobian(1)[0][0], 1.0);

    EXPECT_THROW((void)ctx_.currentPhysicalPoint(2), std::out_of_range);

    spaces::H1Space space(ElementType::Quad4, /*order=*/1);
    ctx_.configure(/*cell_id=*/0, space, space, RequiredData::None);
    EXPECT_TRUE(ctx_.meshDisplacements().empty());
    EXPECT_TRUE(ctx_.meshVelocityJacobians().empty());
    EXPECT_THROW((void)ctx_.meshDisplacement(0), std::logic_error);
    EXPECT_THROW((void)ctx_.meshVelocityJacobian(0), std::logic_error);
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

TEST_F(AssemblyContextTest, VectorBasisJacobiansUseQuadratureMajorStorageAndBuildSolutionJacobian) {
    spaces::HCurlSpace space(ElementType::Tetra4, 0, BasisType::Nedelec);
    ctx_.configure(/*cell_id=*/0,
                   space,
                   space,
                   RequiredData::SolutionValues | RequiredData::SolutionGradients);

    std::vector<AssemblyContext::Point3D> quad_pts = {
        {0.0, 0.0, 0.0},
        {0.25, 0.25, 0.25},
    };
    std::vector<Real> weights = {0.5, 0.5};
    ctx_.setQuadratureData(quad_pts, weights);

    const LocalIndex n_dofs = static_cast<LocalIndex>(space.dofs_per_element());
    const LocalIndex n_qpts = static_cast<LocalIndex>(quad_pts.size());
    std::vector<AssemblyContext::Vector3D> values(static_cast<std::size_t>(n_dofs * n_qpts));
    std::vector<AssemblyContext::Matrix3x3> jacobians(static_cast<std::size_t>(n_dofs * n_qpts));

    for (LocalIndex i = 0; i < n_dofs; ++i) {
        for (LocalIndex q = 0; q < n_qpts; ++q) {
            const auto idx = static_cast<std::size_t>(i * n_qpts + q);
            values[idx] = {Real(i + 1), Real(q + 2), Real(i + q)};
            jacobians[idx][0][0] = Real(10 * i + q);
            jacobians[idx][1][2] = Real(100 + 10 * i + q);
        }
    }

    ctx_.setTestVectorBasisValues(n_dofs, values);
    ctx_.setTestVectorBasisJacobians(n_dofs, jacobians);

    ASSERT_EQ(ctx_.testBasisVectorJacobiansRaw().size(), jacobians.size());
    for (LocalIndex q = 0; q < n_qpts; ++q) {
        for (LocalIndex i = 0; i < n_dofs; ++i) {
            const auto in = jacobians[static_cast<std::size_t>(i * n_qpts + q)];
            const auto stored = ctx_.basisVectorJacobian(i, q);
            EXPECT_DOUBLE_EQ(stored[0][0], in[0][0]);
            EXPECT_DOUBLE_EQ(stored[1][2], in[1][2]);
            EXPECT_DOUBLE_EQ(ctx_.testBasisVectorJacobiansRaw()[static_cast<std::size_t>(q * n_dofs + i)][0][0],
                             in[0][0]);
        }
    }

    std::vector<Real> coefficients(static_cast<std::size_t>(n_dofs), Real(0));
    coefficients[1] = Real(2.0);
    coefficients[3] = Real(-0.5);
    ctx_.setSolutionCoefficients(coefficients);

    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto J = ctx_.solutionVectorJacobian(q);
        const auto J1 = jacobians[static_cast<std::size_t>(1 * n_qpts + q)];
        const auto J3 = jacobians[static_cast<std::size_t>(3 * n_qpts + q)];
        EXPECT_DOUBLE_EQ(J[0][0], Real(2.0) * J1[0][0] - Real(0.5) * J3[0][0]);
        EXPECT_DOUBLE_EQ(J[1][2], Real(2.0) * J1[1][2] - Real(0.5) * J3[1][2]);
    }
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
    setupTwoDofTwoQptScalarContext(ctx_);

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

// ============================================================================
// Historical Solution Access Tests
// ============================================================================

TEST(AssemblyContextHistory, PreviousSolutionValueAndGradient) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    setupTwoDofTwoQptScalarContext(ctx);

    // Previous solution coefficients: u^{n-1} = 2*phi0 + 4*phi1
    std::vector<Real> prev_coeffs = {2.0, 4.0};
    ctx.setPreviousSolutionCoefficients(prev_coeffs);

    EXPECT_DOUBLE_EQ(ctx.previousSolutionValue(0), 2.0);
    EXPECT_DOUBLE_EQ(ctx.previousSolutionValue(1), 3.0);

    // grad u^{n-1} = 2*(-1,0,0) + 4*(1,0,0) = (2,0,0)
    auto grad = ctx.previousSolutionGradient(0);
    EXPECT_DOUBLE_EQ(grad[0], 2.0);
    EXPECT_DOUBLE_EQ(grad[1], 0.0);
}

TEST(AssemblyContextHistory, PreviousSolutionValueThrowsWithoutHistory) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    setupTwoDofTwoQptScalarContext(ctx);

    EXPECT_THROW(ctx.previousSolutionValue(0), std::logic_error);
}

TEST(AssemblyContextHistory, PreviousSolutionValueK1K2) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    setupTwoDofTwoQptScalarContext(ctx);

    std::vector<Real> prev1 = {2.0, 0.0};  // u^{n-1}
    std::vector<Real> prev2 = {0.0, 2.0};  // u^{n-2}
    ctx.setPreviousSolutionCoefficients(prev1);
    ctx.setPreviousSolution2Coefficients(prev2);

    EXPECT_DOUBLE_EQ(ctx.previousSolutionValue(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(ctx.previousSolutionValue(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(ctx.previousSolutionValue(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(ctx.previousSolutionValue(1, 2), 1.0);
}

// ============================================================================
// Multi-Field Access Tests
// ============================================================================

TEST(AssemblyContextMultiField, FieldValueSingleScalarField) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx.setQuadratureData(quad_pts, weights);

    const FieldId temperature = 2;
    const std::vector<Real> values = {300.0, 310.0};
    ctx.setFieldSolutionScalar(temperature, values, /*gradients=*/{}, /*hessians=*/{}, /*laplacians=*/{});

    EXPECT_TRUE(ctx.hasFieldSolutionData(temperature));
    EXPECT_DOUBLE_EQ(ctx.fieldValue(temperature, 0), 300.0);
    EXPECT_DOUBLE_EQ(ctx.fieldValue(temperature, 1), 310.0);
}

TEST(AssemblyContextMultiField, MultipleFieldsIndependentAccess) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx.setQuadratureData(quad_pts, weights);

    const FieldId velocity = 0;
    const FieldId pressure = 1;
    const FieldId temperature = 2;

    const std::vector<AssemblyContext::Vector3D> vel_values = {
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
    };
    const std::vector<AssemblyContext::Matrix3x3> vel_jac = {
        {{{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}},
        {{{0.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 0.0}}},
    };
    ctx.setFieldSolutionVector(velocity, /*value_dimension=*/3, vel_values, vel_jac,
                               /*component_hessians=*/{}, /*component_laplacians=*/{});

    const std::vector<Real> p_values = {100.0, 110.0};
    const std::vector<AssemblyContext::Vector3D> p_grads = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    ctx.setFieldSolutionScalar(pressure, p_values, p_grads, /*hessians=*/{}, /*laplacians=*/{});

    const std::vector<Real> t_values = {300.0, 310.0};
    ctx.setFieldSolutionScalar(temperature, t_values, /*gradients=*/{}, /*hessians=*/{}, /*laplacians=*/{});

    EXPECT_DOUBLE_EQ(ctx.fieldValue(pressure, 1), 110.0);
    auto pgrad = ctx.fieldGradient(pressure, 0);
    EXPECT_DOUBLE_EQ(pgrad[0], 1.0);

    auto v0 = ctx.fieldVectorValue(velocity, 0);
    EXPECT_DOUBLE_EQ(v0[0], 1.0);
    EXPECT_DOUBLE_EQ(ctx.fieldValue(temperature, 0), 300.0);
}

TEST(AssemblyContextMultiField, CopyFieldSolutionDataFromClonesActiveFieldState) {
    AssemblyContext src;
    src.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    src.setQuadratureData(quad_pts, weights);

    const FieldId velocity = 0;
    const FieldId pressure = 1;

    const std::vector<AssemblyContext::Vector3D> vel_values = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    };
    const std::vector<AssemblyContext::Matrix3x3> vel_jac = {
        {{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}}},
        {{{4.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 6.0}}},
    };
    src.setFieldSolutionVector(velocity, /*value_dimension=*/3, vel_values, vel_jac,
                               /*component_hessians=*/{}, /*component_laplacians=*/{});

    const std::vector<Real> pressure_values = {100.0, 110.0};
    const std::vector<AssemblyContext::Vector3D> pressure_grads = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    };
    src.setFieldSolutionScalar(pressure, pressure_values, pressure_grads,
                               /*hessians=*/{}, /*laplacians=*/{});

    AssemblyContext dst;
    dst.reserve(4, 4, 4);
    dst.setQuadratureData(quad_pts, weights);
    dst.copyFieldSolutionDataFrom(src);

    EXPECT_TRUE(dst.hasFieldSolutionData(velocity));
    EXPECT_TRUE(dst.hasFieldSolutionData(pressure));

    const auto copied_velocity = dst.fieldVectorValue(velocity, 1);
    EXPECT_DOUBLE_EQ(copied_velocity[0], 4.0);
    EXPECT_DOUBLE_EQ(copied_velocity[1], 5.0);
    EXPECT_DOUBLE_EQ(copied_velocity[2], 6.0);

    const auto copied_pressure_grad = dst.fieldGradient(pressure, 0);
    EXPECT_DOUBLE_EQ(copied_pressure_grad[0], 1.0);
    EXPECT_DOUBLE_EQ(copied_pressure_grad[1], 2.0);
    EXPECT_DOUBLE_EQ(copied_pressure_grad[2], 3.0);

    const auto jit_fields = dst.jitFieldSolutionTable();
    ASSERT_EQ(jit_fields.size(), 2u);
    EXPECT_EQ(jit_fields[0].field_id, velocity);
    EXPECT_EQ(jit_fields[1].field_id, pressure);
    ASSERT_NE(jit_fields[0].vector_values_xyz, nullptr);
    ASSERT_NE(jit_fields[1].values, nullptr);
    EXPECT_DOUBLE_EQ(jit_fields[1].values[1], 110.0);
}

TEST(AssemblyContextMultiField, PreviousScalarAndVectorFieldHistoryAccess) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx.setQuadratureData(quad_pts, weights);

    const FieldId pressure = 1;
    const FieldId velocity = 2;

    const std::vector<Real> pressure_values = {100.0, 110.0};
    ctx.setFieldSolutionScalar(pressure, pressure_values);
    ctx.setFieldPreviousSolutionScalarK(pressure, /*k=*/1, std::vector<Real>{95.0, 96.0});
    ctx.setFieldPreviousSolutionScalarK(pressure, /*k=*/2, std::vector<Real>{90.0, 91.0});

    const std::vector<AssemblyContext::Vector3D> velocity_values = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    };
    ctx.setFieldSolutionVector(velocity, /*value_dimension=*/3, velocity_values);
    ctx.setFieldPreviousSolutionVectorK(
        velocity, /*k=*/1, /*value_dimension=*/3,
        std::vector<AssemblyContext::Vector3D>{{0.5, 1.5, 2.5}, {3.5, 4.5, 5.5}});

    EXPECT_DOUBLE_EQ(ctx.fieldPreviousValue(pressure, 0, 1), 95.0);
    EXPECT_DOUBLE_EQ(ctx.fieldPreviousValue(pressure, 1, 2), 91.0);

    const auto prev_velocity = ctx.fieldPreviousVectorValue(velocity, 1, 1);
    EXPECT_DOUBLE_EQ(prev_velocity[0], 3.5);
    EXPECT_DOUBLE_EQ(prev_velocity[1], 4.5);
    EXPECT_DOUBLE_EQ(prev_velocity[2], 5.5);

    const auto jit_fields = ctx.jitFieldSolutionTable();
    ASSERT_EQ(jit_fields.size(), 2u);
    ASSERT_NE(jit_fields[0].history_values, nullptr);
    ASSERT_NE(jit_fields[1].history_vector_values_xyz, nullptr);
}

TEST(AssemblyContextMultiField, CopyFieldSolutionDataFromClonesHistoryState) {
    AssemblyContext src;
    src.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    src.setQuadratureData(quad_pts, weights);

    const FieldId pressure = 4;
    const FieldId velocity = 5;

    src.setFieldSolutionScalar(pressure, std::vector<Real>{1.0, 2.0});
    src.setFieldPreviousSolutionScalarK(pressure, /*k=*/1, std::vector<Real>{3.0, 4.0});

    src.setFieldSolutionVector(
        velocity, /*value_dimension=*/3,
        std::vector<AssemblyContext::Vector3D>{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}});
    src.setFieldPreviousSolutionVectorK(
        velocity, /*k=*/1, /*value_dimension=*/3,
        std::vector<AssemblyContext::Vector3D>{{5.0, 6.0, 7.0}, {8.0, 9.0, 10.0}});

    AssemblyContext dst;
    dst.reserve(4, 4, 4);
    dst.setQuadratureData(quad_pts, weights);
    dst.copyFieldSolutionDataFrom(src);

    EXPECT_DOUBLE_EQ(dst.fieldPreviousValue(pressure, 1, 1), 4.0);
    const auto prev_velocity = dst.fieldPreviousVectorValue(velocity, 0, 1);
    EXPECT_DOUBLE_EQ(prev_velocity[0], 5.0);
    EXPECT_DOUBLE_EQ(prev_velocity[1], 6.0);
    EXPECT_DOUBLE_EQ(prev_velocity[2], 7.0);
}

TEST(AssemblyContextMultiField, CopyFieldSolutionDataSubsetFromKeepsRequestedHistoryOnly) {
    AssemblyContext src;
    src.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    src.setQuadratureData(quad_pts, weights);

    const FieldId pressure = 6;
    const FieldId velocity = 7;
    const FieldId temperature = 8;

    src.setFieldSolutionScalar(pressure, std::vector<Real>{1.0, 2.0});
    src.setFieldPreviousSolutionScalarK(pressure, /*k=*/1, std::vector<Real>{3.0, 4.0});
    src.setFieldSolutionVector(
        velocity, /*value_dimension=*/3,
        std::vector<AssemblyContext::Vector3D>{{5.0, 6.0, 7.0}, {8.0, 9.0, 10.0}});
    src.setFieldPreviousSolutionVectorK(
        velocity, /*k=*/1, /*value_dimension=*/3,
        std::vector<AssemblyContext::Vector3D>{{11.0, 12.0, 13.0}, {14.0, 15.0, 16.0}});
    src.setFieldSolutionScalar(temperature, std::vector<Real>{17.0, 18.0});

    AssemblyContext dst;
    dst.reserve(4, 4, 4);
    dst.setQuadratureData(quad_pts, weights);

    const std::array<FieldId, 2> wanted = {pressure, velocity};
    EXPECT_TRUE(dst.copyFieldSolutionDataSubsetFrom(src, wanted));

    EXPECT_TRUE(dst.hasFieldSolutionData(pressure));
    EXPECT_TRUE(dst.hasFieldSolutionData(velocity));
    EXPECT_FALSE(dst.hasFieldSolutionData(temperature));
    EXPECT_DOUBLE_EQ(dst.fieldPreviousValue(pressure, 1, 1), 4.0);
    const auto prev_velocity = dst.fieldPreviousVectorValue(velocity, 0, 1);
    EXPECT_DOUBLE_EQ(prev_velocity[0], 11.0);
    EXPECT_DOUBLE_EQ(prev_velocity[1], 12.0);
    EXPECT_DOUBLE_EQ(prev_velocity[2], 13.0);
}

TEST(AssemblyContextMultiField, FieldGradientOutOfRangeFieldThrows) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx.setQuadratureData(quad_pts, weights);

    EXPECT_THROW(ctx.fieldGradient(/*field=*/123, /*q=*/0), std::logic_error);
}

TEST(AssemblyContextMultiField, FieldJacobianForVectorField) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx.setQuadratureData(quad_pts, weights);

    const FieldId velocity = 0;
    const std::vector<AssemblyContext::Vector3D> values = {{1.0, 2.0, 3.0}};
    const std::vector<AssemblyContext::Matrix3x3> jac = {
        {{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}}},
    };
    ctx.setFieldSolutionVector(velocity, /*value_dimension=*/3, values, jac,
                               /*component_hessians=*/{}, /*component_laplacians=*/{});

    const auto J = ctx.fieldJacobian(velocity, 0);
    EXPECT_DOUBLE_EQ(J[0][0], 1.0);
    EXPECT_DOUBLE_EQ(J[1][1], 2.0);
    EXPECT_DOUBLE_EQ(J[2][2], 3.0);
}

TEST(AssemblyContextMultiField, FieldHessianAndLaplacianForScalarField) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx.setQuadratureData(quad_pts, weights);

    const FieldId pressure = 1;
    const std::vector<Real> values = {42.0};
    const std::vector<AssemblyContext::Vector3D> grads = {{1.0, 0.0, 0.0}};
    const std::vector<AssemblyContext::Matrix3x3> hess = {
        {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}},
    };
    const std::vector<Real> lap = {15.0};
    ctx.setFieldSolutionScalar(pressure, values, grads, hess, lap);

    EXPECT_DOUBLE_EQ(ctx.fieldValue(pressure, 0), 42.0);
    auto H = ctx.fieldHessian(pressure, 0);
    EXPECT_DOUBLE_EQ(H[1][2], 6.0);
    EXPECT_DOUBLE_EQ(ctx.fieldLaplacian(pressure, 0), 15.0);
}

TEST(AssemblyContextMultiField, VectorFieldComponentAccess) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx.setQuadratureData(quad_pts, weights);

    const FieldId velocity = 0;
    const std::vector<AssemblyContext::Vector3D> values = {{1.0, 2.0, 3.0}};
    const std::vector<AssemblyContext::Matrix3x3> jac = {{{{0.0, 0.0, 0.0},
                                                           {0.0, 0.0, 0.0},
                                                           {0.0, 0.0, 0.0}}}};
    ctx.setFieldSolutionVector(velocity, /*value_dimension=*/3, values, jac,
                               /*component_hessians=*/{}, /*component_laplacians=*/{});

    const auto v = ctx.fieldVectorValue(velocity, 0);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
}

// ============================================================================
// Material State Tests
// ============================================================================

TEST(AssemblyContextMaterialState, BasicAccessAndIsolation) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    std::vector<Real> weights = {0.5, 0.5};
    ctx.setQuadratureData(quad_pts, weights);

    constexpr std::size_t bytes_per_qpt = 8;
    constexpr std::size_t stride = 16;  // include padding
    std::vector<std::byte> old_storage(stride * 2);
    std::vector<std::byte> work_storage(stride * 2);

    // Initialize old to 0x11 and work to 0x22
    std::fill(old_storage.begin(), old_storage.end(), std::byte{0x11});
    std::fill(work_storage.begin(), work_storage.end(), std::byte{0x22});

    ctx.setMaterialState(old_storage.data(), work_storage.data(), bytes_per_qpt, stride);

    auto s0 = ctx.materialState(0);
    ASSERT_EQ(s0.size(), bytes_per_qpt);
    s0[0] = std::byte{0xAB};
    EXPECT_EQ(work_storage[0], std::byte{0xAB});
    EXPECT_EQ(old_storage[0], std::byte{0x11});

    auto old0 = ctx.materialStateOld(0);
    EXPECT_EQ(old0[0], std::byte{0x11});

    auto s1 = ctx.materialState(1);
    EXPECT_EQ(s1.data(), work_storage.data() + stride);
}

TEST(AssemblyContextMaterialState, MultipleQuadraturePointsIndependentStorage) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    std::vector<Real> weights = {1.0, 1.0, 1.0};
    ctx.setQuadratureData(quad_pts, weights);

    constexpr std::size_t bytes_per_qpt = 4;
    constexpr std::size_t stride = 4;
    std::vector<std::byte> work_storage(stride * 3);
    ctx.setMaterialState(/*old=*/nullptr, work_storage.data(), bytes_per_qpt, stride);

    ctx.materialState(0)[0] = std::byte{0x01};
    ctx.materialState(1)[0] = std::byte{0x02};
    ctx.materialState(2)[0] = std::byte{0x03};

    EXPECT_EQ(work_storage[0], std::byte{0x01});
    EXPECT_EQ(work_storage[stride], std::byte{0x02});
    EXPECT_EQ(work_storage[2 * stride], std::byte{0x03});
}

TEST(AssemblyContextMaterialState, MultiComponentStateRoundTrip) {
    AssemblyContext ctx;
    ctx.reserve(4, 4, 4);
    std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    std::vector<Real> weights = {1.0};
    ctx.setQuadratureData(quad_pts, weights);

    constexpr std::size_t n_components = 6;
    constexpr std::size_t bytes_per_qpt = n_components * sizeof(double);
    std::vector<std::byte> storage(bytes_per_qpt);
    ctx.setMaterialState(storage.data(), bytes_per_qpt, bytes_per_qpt);

    const double input[n_components] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::memcpy(ctx.materialState(0).data(), input, bytes_per_qpt);

    double output[n_components]{};
    std::memcpy(output, ctx.materialState(0).data(), bytes_per_qpt);

    EXPECT_DOUBLE_EQ(output[0], 1.0);
    EXPECT_DOUBLE_EQ(output[5], 6.0);
}

// ============================================================================
// Geometric Measure + Context Type Tests
// ============================================================================

TEST(AssemblyContextMeasures, CellAndFaceMeasureAccessRules) {
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    AssemblyContext cell_ctx;
    cell_ctx.configure(/*cell_id=*/0, space, space, RequiredData::None);
    cell_ctx.setEntityMeasures(/*diameter=*/2.0, /*volume=*/1.5, /*facet_area=*/0.0);
    EXPECT_DOUBLE_EQ(cell_ctx.cellDiameter(), 2.0);
    EXPECT_DOUBLE_EQ(cell_ctx.cellVolume(), 1.5);
    EXPECT_THROW(cell_ctx.facetArea(), std::logic_error);

    AssemblyContext face_ctx;
    face_ctx.configureFace(/*face_id=*/7, /*cell_id=*/0, /*local_face_id=*/0,
                           space, space, RequiredData::None, ContextType::BoundaryFace);
    face_ctx.setEntityMeasures(/*diameter=*/2.0, /*volume=*/0.0, /*facet_area=*/0.25);
    EXPECT_DOUBLE_EQ(face_ctx.cellDiameter(), 2.0);
    EXPECT_DOUBLE_EQ(face_ctx.facetArea(), 0.25);
    EXPECT_THROW(face_ctx.cellVolume(), std::logic_error);
}

TEST(AssemblyContextTypes, ConfigureSetsContextType) {
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);
    AssemblyContext ctx;
    ctx.configure(/*cell_id=*/0, space, space, RequiredData::None);
    EXPECT_EQ(ctx.contextType(), ContextType::Cell);

    ctx.configureFace(/*face_id=*/1, /*cell_id=*/0, /*local_face_id=*/0,
                      space, space, RequiredData::None, ContextType::BoundaryFace);
    EXPECT_EQ(ctx.contextType(), ContextType::BoundaryFace);

    ctx.configureFace(/*face_id=*/2, /*cell_id=*/0, /*local_face_id=*/1,
                      space, space, RequiredData::None, ContextType::InteriorFace);
    EXPECT_EQ(ctx.contextType(), ContextType::InteriorFace);
}

// ============================================================================
// Petrov-Galerkin Tests
// ============================================================================

TEST(AssemblyContextPetrovGalerkin, NonSquareTestTrialAndTrialBasisDistinct) {
    spaces::H1Space test_space(ElementType::Tetra4, /*order=*/1);
    spaces::H1Space trial_space(ElementType::Tetra4, /*order=*/2);

    AssemblyContext ctx;
    ctx.configure(/*cell_id=*/0, test_space, trial_space, RequiredData::None);
    EXPECT_NE(ctx.numTestDofs(), ctx.numTrialDofs());
    EXPECT_FALSE(ctx.isSquare());

    const std::vector<AssemblyContext::Point3D> quad_pts = {{0.0, 0.0, 0.0}};
    const std::vector<Real> weights = {1.0};
    ctx.setQuadratureData(quad_pts, weights);

    const LocalIndex n_qpts = 1;
    const LocalIndex n_test = ctx.numTestDofs();
    const LocalIndex n_trial = ctx.numTrialDofs();

    std::vector<Real> test_vals(static_cast<std::size_t>(n_test * n_qpts));
    std::vector<AssemblyContext::Vector3D> test_grads(static_cast<std::size_t>(n_test * n_qpts),
                                                      {0.0, 0.0, 0.0});
    for (LocalIndex i = 0; i < n_test; ++i) {
        test_vals[static_cast<std::size_t>(i)] = 1.0 + static_cast<Real>(i);
    }
    ctx.setTestBasisData(n_test, test_vals, test_grads);

    std::vector<Real> trial_vals(static_cast<std::size_t>(n_trial * n_qpts));
    std::vector<AssemblyContext::Vector3D> trial_grads(static_cast<std::size_t>(n_trial * n_qpts),
                                                       {0.0, 0.0, 0.0});
    for (LocalIndex j = 0; j < n_trial; ++j) {
        trial_vals[static_cast<std::size_t>(j)] = 10.0 + static_cast<Real>(j);
    }
    ctx.setTrialBasisData(n_trial, trial_vals, trial_grads);

    ctx.setPhysicalGradients(test_grads, trial_grads);

    EXPECT_DOUBLE_EQ(ctx.basisValue(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(ctx.trialBasisValue(0, 0), 10.0);
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

    auto vals = ctx_.basisValuesAtQpt(0);
    EXPECT_EQ(vals.size(), 2u);       // 2 dofs at qpt 0
    EXPECT_DOUBLE_EQ(vals[0], 1.0);   // dof 0 at qpt 0
    EXPECT_DOUBLE_EQ(vals[1], 0.0);   // dof 1 at qpt 0
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
