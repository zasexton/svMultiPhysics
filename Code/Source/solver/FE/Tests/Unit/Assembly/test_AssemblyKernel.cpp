/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_AssemblyKernel.cpp
 * @brief Unit tests for AssemblyKernel implementations
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyKernel.h"
#include "Assembly/AssemblyContext.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// Mock AssemblyContext for testing
// ============================================================================

/**
 * @brief Create a simple test context for unit element [0,1]^2
 */
class TestContextHelper {
public:
    static void setupUnitSquareP1(AssemblyContext& ctx) {
        // 4 DOFs for bilinear element
        // 4 quadrature points (2x2 Gauss)

        // Quadrature points in reference coordinates
        const Real g = 1.0 / std::sqrt(3.0);
        std::vector<AssemblyContext::Point3D> quad_pts = {
            {-g, -g, 0.0},
            { g, -g, 0.0},
            {-g,  g, 0.0},
            { g,  g, 0.0}
        };

        // Quadrature weights (reference element [-1,1]^2)
        std::vector<Real> quad_wts = {1.0, 1.0, 1.0, 1.0};

        // Physical points (unit square [0,1]^2 mapping)
        // xi in [-1,1] maps to x in [0,1]: x = (1+xi)/2
        std::vector<AssemblyContext::Point3D> phys_pts;
        for (const auto& pt : quad_pts) {
            phys_pts.push_back({(1.0 + pt[0]) / 2.0, (1.0 + pt[1]) / 2.0, 0.0});
        }

        // Jacobian for unit square: dx/dxi = 0.5, dy/deta = 0.5
        // |J| = 0.25
        const Real jac_det = 0.25;
        std::vector<Real> jac_dets(4, jac_det);

        // Integration weights = quad_wt * |J|
        std::vector<Real> int_wts;
        for (Real w : quad_wts) {
            int_wts.push_back(w * jac_det);
        }

        // Jacobian matrix (constant for linear mapping)
        AssemblyContext::Matrix3x3 J = {{{0.5, 0.0, 0.0},
                                          {0.0, 0.5, 0.0},
                                          {0.0, 0.0, 1.0}}};
        AssemblyContext::Matrix3x3 Jinv = {{{2.0, 0.0, 0.0},
                                             {0.0, 2.0, 0.0},
                                             {0.0, 0.0, 1.0}}};
        std::vector<AssemblyContext::Matrix3x3> jacs(4, J);
        std::vector<AssemblyContext::Matrix3x3> jinvs(4, Jinv);

        // Bilinear basis functions on reference element [-1,1]^2:
        // N0 = (1-xi)(1-eta)/4, N1 = (1+xi)(1-eta)/4
        // N2 = (1-xi)(1+eta)/4, N3 = (1+xi)(1+eta)/4

        LocalIndex n_dofs = 4;
        LocalIndex n_qpts = 4;

        std::vector<Real> basis_vals(n_dofs * n_qpts);
        std::vector<AssemblyContext::Vector3D> ref_grads(n_dofs * n_qpts);

        for (LocalIndex q = 0; q < n_qpts; ++q) {
            Real xi = quad_pts[q][0];
            Real eta = quad_pts[q][1];

            // Basis values
            basis_vals[0 * n_qpts + q] = (1 - xi) * (1 - eta) / 4.0;
            basis_vals[1 * n_qpts + q] = (1 + xi) * (1 - eta) / 4.0;
            basis_vals[2 * n_qpts + q] = (1 - xi) * (1 + eta) / 4.0;
            basis_vals[3 * n_qpts + q] = (1 + xi) * (1 + eta) / 4.0;

            // Reference gradients
            ref_grads[0 * n_qpts + q] = {-(1 - eta) / 4.0, -(1 - xi) / 4.0, 0.0};
            ref_grads[1 * n_qpts + q] = { (1 - eta) / 4.0, -(1 + xi) / 4.0, 0.0};
            ref_grads[2 * n_qpts + q] = {-(1 + eta) / 4.0,  (1 - xi) / 4.0, 0.0};
            ref_grads[3 * n_qpts + q] = { (1 + eta) / 4.0,  (1 + xi) / 4.0, 0.0};
        }

        // Physical gradients = Jinv^T * ref_grad
        // For our case: phys_grad = 2 * ref_grad
        std::vector<AssemblyContext::Vector3D> phys_grads(n_dofs * n_qpts);
        for (LocalIndex i = 0; i < n_dofs; ++i) {
            for (LocalIndex q = 0; q < n_qpts; ++q) {
                auto& rg = ref_grads[i * n_qpts + q];
                phys_grads[i * n_qpts + q] = {2.0 * rg[0], 2.0 * rg[1], 0.0};
            }
        }

        // Set data on context
        ctx.setQuadratureData(quad_pts, quad_wts);
        ctx.setPhysicalPoints(phys_pts);
        ctx.setJacobianData(jacs, jinvs, jac_dets);
        ctx.setIntegrationWeights(int_wts);
        ctx.setTestBasisData(n_dofs, basis_vals, ref_grads);
        ctx.setPhysicalGradients(phys_grads, phys_grads);
    }
};

// ============================================================================
// KernelOutput Tests
// ============================================================================

TEST(KernelOutputTest, Reserve) {
    KernelOutput output;
    output.reserve(3, 4, true, true);

    EXPECT_EQ(output.n_test_dofs, 3);
    EXPECT_EQ(output.n_trial_dofs, 4);
    EXPECT_TRUE(output.has_matrix);
    EXPECT_TRUE(output.has_vector);
    EXPECT_EQ(output.local_matrix.size(), 12u);  // 3x4
    EXPECT_EQ(output.local_vector.size(), 3u);
}

TEST(KernelOutputTest, MatrixAccess) {
    KernelOutput output;
    output.reserve(2, 3, true, false);

    output.matrixEntry(0, 0) = 1.0;
    output.matrixEntry(0, 2) = 2.0;
    output.matrixEntry(1, 1) = 3.0;

    EXPECT_DOUBLE_EQ(output.matrixEntry(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(output.matrixEntry(0, 2), 2.0);
    EXPECT_DOUBLE_EQ(output.matrixEntry(1, 1), 3.0);
}

TEST(KernelOutputTest, Clear) {
    KernelOutput output;
    output.reserve(2, 2, true, true);
    output.matrixEntry(0, 0) = 5.0;
    output.vectorEntry(0) = 3.0;

    output.clear();

    EXPECT_DOUBLE_EQ(output.matrixEntry(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(output.vectorEntry(0), 0.0);
}

// ============================================================================
// RequiredData Tests
// ============================================================================

TEST(RequiredDataTest, BitwiseOr) {
    RequiredData data = RequiredData::BasisValues | RequiredData::PhysicalGradients;

    EXPECT_TRUE(hasFlag(data, RequiredData::BasisValues));
    EXPECT_TRUE(hasFlag(data, RequiredData::PhysicalGradients));
    EXPECT_FALSE(hasFlag(data, RequiredData::SolutionValues));
}

TEST(RequiredDataTest, StandardPreset) {
    RequiredData standard = RequiredData::Standard;

    EXPECT_TRUE(hasFlag(standard, RequiredData::PhysicalPoints));
    EXPECT_TRUE(hasFlag(standard, RequiredData::JacobianDets));
    EXPECT_TRUE(hasFlag(standard, RequiredData::BasisValues));
    EXPECT_TRUE(hasFlag(standard, RequiredData::BasisGradients));
}

// ============================================================================
// MassKernel Tests
// ============================================================================

class MassKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_.reserve(10, 10, 2);
        TestContextHelper::setupUnitSquareP1(ctx_);
    }

    AssemblyContext ctx_;
};

TEST_F(MassKernelTest, RequiredData) {
    MassKernel kernel;
    auto required = kernel.getRequiredData();

    EXPECT_TRUE(hasFlag(required, RequiredData::BasisValues));
    EXPECT_TRUE(hasFlag(required, RequiredData::IntegrationWeights));
}

TEST_F(MassKernelTest, Properties) {
    MassKernel kernel;

    EXPECT_TRUE(kernel.isSymmetric());
    EXPECT_TRUE(kernel.isMatrixOnly());
    EXPECT_FALSE(kernel.isVectorOnly());
    EXPECT_EQ(kernel.name(), "MassKernel");
}

TEST_F(MassKernelTest, ComputeCell) {
    MassKernel kernel;
    KernelOutput output;

    kernel.computeCell(ctx_, output);

    EXPECT_TRUE(output.has_matrix);
    EXPECT_FALSE(output.has_vector);
    EXPECT_EQ(output.n_test_dofs, 4);
    EXPECT_EQ(output.n_trial_dofs, 4);

    // Mass matrix should be symmetric
    for (LocalIndex i = 0; i < output.n_test_dofs; ++i) {
        for (LocalIndex j = i + 1; j < output.n_trial_dofs; ++j) {
            EXPECT_NEAR(output.matrixEntry(i, j), output.matrixEntry(j, i), 1e-12);
        }
    }

    // Diagonal entries should be positive
    for (LocalIndex i = 0; i < output.n_test_dofs; ++i) {
        EXPECT_GT(output.matrixEntry(i, i), 0.0);
    }

    // Sum of row should equal integral of basis function
    // For unit square, integral of sum of bilinear basis = 1.0
    Real total = 0.0;
    for (LocalIndex i = 0; i < output.n_test_dofs; ++i) {
        for (LocalIndex j = 0; j < output.n_trial_dofs; ++j) {
            total += output.matrixEntry(i, j);
        }
    }
    // Total mass = area of element = 1.0 for unit square
    EXPECT_NEAR(total, 1.0, 1e-10);
}

TEST_F(MassKernelTest, WithCoefficient) {
    MassKernel kernel_1(1.0);
    MassKernel kernel_2(2.0);
    KernelOutput output_1, output_2;

    kernel_1.computeCell(ctx_, output_1);
    kernel_2.computeCell(ctx_, output_2);

    // Kernel with coefficient 2 should give 2x the values
    for (LocalIndex i = 0; i < output_1.n_test_dofs; ++i) {
        for (LocalIndex j = 0; j < output_1.n_trial_dofs; ++j) {
            EXPECT_NEAR(output_2.matrixEntry(i, j),
                        2.0 * output_1.matrixEntry(i, j), 1e-12);
        }
    }
}

// ============================================================================
// StiffnessKernel Tests
// ============================================================================

class StiffnessKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_.reserve(10, 10, 2);
        TestContextHelper::setupUnitSquareP1(ctx_);
    }

    AssemblyContext ctx_;
};

TEST_F(StiffnessKernelTest, RequiredData) {
    StiffnessKernel kernel;
    auto required = kernel.getRequiredData();

    EXPECT_TRUE(hasFlag(required, RequiredData::PhysicalGradients));
    EXPECT_TRUE(hasFlag(required, RequiredData::IntegrationWeights));
}

TEST_F(StiffnessKernelTest, Properties) {
    StiffnessKernel kernel;

    EXPECT_TRUE(kernel.isSymmetric());
    EXPECT_TRUE(kernel.isMatrixOnly());
    EXPECT_EQ(kernel.name(), "StiffnessKernel");
}

TEST_F(StiffnessKernelTest, ComputeCell) {
    StiffnessKernel kernel;
    KernelOutput output;

    kernel.computeCell(ctx_, output);

    EXPECT_TRUE(output.has_matrix);
    EXPECT_FALSE(output.has_vector);

    // Stiffness matrix should be symmetric
    for (LocalIndex i = 0; i < output.n_test_dofs; ++i) {
        for (LocalIndex j = i + 1; j < output.n_trial_dofs; ++j) {
            EXPECT_NEAR(output.matrixEntry(i, j), output.matrixEntry(j, i), 1e-12);
        }
    }

    // Stiffness matrix has zero row sums (constant functions in kernel)
    for (LocalIndex i = 0; i < output.n_test_dofs; ++i) {
        Real row_sum = 0.0;
        for (LocalIndex j = 0; j < output.n_trial_dofs; ++j) {
            row_sum += output.matrixEntry(i, j);
        }
        EXPECT_NEAR(row_sum, 0.0, 1e-10);
    }
}

// ============================================================================
// SourceKernel Tests
// ============================================================================

class SourceKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_.reserve(10, 10, 2);
        TestContextHelper::setupUnitSquareP1(ctx_);
    }

    AssemblyContext ctx_;
};

TEST_F(SourceKernelTest, ConstantSource) {
    SourceKernel kernel(1.0);  // f(x) = 1
    KernelOutput output;

    kernel.computeCell(ctx_, output);

    EXPECT_FALSE(output.has_matrix);
    EXPECT_TRUE(output.has_vector);
    EXPECT_TRUE(kernel.isVectorOnly());

    // Sum of load vector = integral of f * sum(phi_i) = integral of f = 1.0
    Real total = 0.0;
    for (LocalIndex i = 0; i < output.n_test_dofs; ++i) {
        total += output.vectorEntry(i);
    }
    EXPECT_NEAR(total, 1.0, 1e-10);
}

TEST_F(SourceKernelTest, FunctionSource) {
    // f(x,y,z) = x
    auto source_func = [](Real x, Real /*y*/, Real /*z*/) { return x; };
    SourceKernel kernel(source_func);
    KernelOutput output;

    kernel.computeCell(ctx_, output);

    // Sum = integral of x over [0,1]^2 = 0.5
    Real total = 0.0;
    for (LocalIndex i = 0; i < output.n_test_dofs; ++i) {
        total += output.vectorEntry(i);
    }
    EXPECT_NEAR(total, 0.5, 1e-10);
}

// ============================================================================
// PoissonKernel Tests
// ============================================================================

class PoissonKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_.reserve(10, 10, 2);
        TestContextHelper::setupUnitSquareP1(ctx_);
    }

    AssemblyContext ctx_;
};

TEST_F(PoissonKernelTest, ComputesBoth) {
    PoissonKernel kernel(1.0);  // f(x) = 1
    KernelOutput output;

    kernel.computeCell(ctx_, output);

    EXPECT_TRUE(output.has_matrix);
    EXPECT_TRUE(output.has_vector);
    EXPECT_TRUE(kernel.isSymmetric());
}

TEST_F(PoissonKernelTest, EquivalentToSeparateKernels) {
    // Poisson kernel should give same result as Stiffness + Source
    PoissonKernel poisson_kernel(1.0);
    StiffnessKernel stiffness_kernel;
    SourceKernel source_kernel(1.0);

    KernelOutput poisson_output, stiffness_output, source_output;

    poisson_kernel.computeCell(ctx_, poisson_output);
    stiffness_kernel.computeCell(ctx_, stiffness_output);
    source_kernel.computeCell(ctx_, source_output);

    // Compare matrix
    for (LocalIndex i = 0; i < poisson_output.n_test_dofs; ++i) {
        for (LocalIndex j = 0; j < poisson_output.n_trial_dofs; ++j) {
            EXPECT_NEAR(poisson_output.matrixEntry(i, j),
                        stiffness_output.matrixEntry(i, j), 1e-12);
        }
    }

    // Compare vector
    for (LocalIndex i = 0; i < poisson_output.n_test_dofs; ++i) {
        EXPECT_NEAR(poisson_output.vectorEntry(i),
                    source_output.vectorEntry(i), 1e-12);
    }
}

// ============================================================================
// CompositeKernel Tests
// ============================================================================

TEST(CompositeKernelTest, CombineMassAndStiffness) {
    auto mass = std::make_shared<MassKernel>(1.0);
    auto stiffness = std::make_shared<StiffnessKernel>(1.0);

    CompositeKernel composite;
    composite.addKernel(mass, 1.0);
    composite.addKernel(stiffness, 1.0);

    // Required data should be union
    auto required = composite.getRequiredData();
    EXPECT_TRUE(hasFlag(required, RequiredData::BasisValues));
    EXPECT_TRUE(hasFlag(required, RequiredData::PhysicalGradients));
    EXPECT_TRUE(hasFlag(required, RequiredData::IntegrationWeights));
}

TEST(CompositeKernelTest, ScaledCombination) {
    AssemblyContext ctx;
    ctx.reserve(10, 10, 2);
    TestContextHelper::setupUnitSquareP1(ctx);

    auto mass = std::make_shared<MassKernel>(1.0);

    CompositeKernel composite;
    composite.addKernel(mass, 2.0);

    KernelOutput mass_output, composite_output;
    mass->computeCell(ctx, mass_output);
    composite.computeCell(ctx, composite_output);

    // Composite with scale 2 should give 2x mass matrix
    for (LocalIndex i = 0; i < mass_output.n_test_dofs; ++i) {
        for (LocalIndex j = 0; j < mass_output.n_trial_dofs; ++j) {
            EXPECT_NEAR(composite_output.matrixEntry(i, j),
                        2.0 * mass_output.matrixEntry(i, j), 1e-12);
        }
    }
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
