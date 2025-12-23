/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NonlinearAssemblyDriver.cpp
 * @brief Unit tests for NonlinearAssemblyDriver
 */

#include <gtest/gtest.h>
#include "Assembly/NonlinearAssemblyDriver.h"
#include "Core/Types.h"

#include <vector>
#include <memory>
#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

namespace {

// ============================================================================
// Mock Classes
// ============================================================================

/**
 * @brief Simple nonlinear kernel for testing
 *
 * Implements a simple nonlinear operator: F(u) = u^2 - f
 * Jacobian: J = 2*diag(u)
 */
class MockNonlinearKernel : public INonlinearKernel {
public:
    void computeResidual(AssemblyContext& /*context*/, KernelOutput& output) override {
        // Simple residual: F = u^2 - source
        LocalIndex n = 4;  // Assume 4 DOFs per element
        output.local_vector.resize(static_cast<std::size_t>(n));

        for (LocalIndex i = 0; i < n; ++i) {
            // Placeholder - actual impl would use context.getSolutionValues()
            output.local_vector[static_cast<std::size_t>(i)] = 1.0;  // u^2 - f
        }
    }

    void computeJacobian(AssemblyContext& /*context*/, KernelOutput& output) override {
        // Jacobian: J_ij = 2*u_i * delta_ij (diagonal)
        LocalIndex n = 4;
        output.local_matrix.resize(static_cast<std::size_t>(n * n), 0.0);

        for (LocalIndex i = 0; i < n; ++i) {
            // Diagonal element
            output.local_matrix[static_cast<std::size_t>(i * n + i)] = 2.0;
        }
    }

    void computeBoth(AssemblyContext& context, KernelOutput& output) override {
        computeResidual(context, output);
        computeJacobian(context, output);
    }

    [[nodiscard]] bool hasOptimizedBoth() const noexcept override { return true; }
};

/**
 * @brief Mock mesh access for nonlinear tests
 */
class MockMeshAccess : public IMeshAccess {
public:
    GlobalIndex numCells() const override { return 10; }
    GlobalIndex numOwnedCells() const override { return 10; }
    GlobalIndex numBoundaryFaces() const override { return 0; }
    GlobalIndex numInteriorFaces() const override { return 0; }
    int dimension() const override { return 3; }

    bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    ElementType getCellType(GlobalIndex /*cell_id*/) const override {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        nodes.clear();
        for (int i = 0; i < 4; ++i) {
            nodes.push_back(cell_id * 4 + i);
        }
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return {
            static_cast<Real>(node_id % 10),
            static_cast<Real>((node_id / 10) % 10),
            static_cast<Real>(node_id / 100)
        };
    }

    void getCellCoordinates(GlobalIndex cell_id,
                           std::vector<std::array<Real, 3>>& coords) const override {
        coords.clear();
        for (int i = 0; i < 4; ++i) {
            coords.push_back(getNodeCoordinates(cell_id * 4 + i));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override {
        return 0;
    }

    std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex i = 0; i < numCells(); ++i) {
            callback(i);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(callback);
    }

    void forEachBoundaryFace(int /*marker*/,
        std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override {}

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override {}
};

} // namespace

// ============================================================================
// Test Fixtures
// ============================================================================

class NonlinearAssemblyDriverTest : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = std::make_unique<MockMeshAccess>();
        kernel_ = std::make_unique<MockNonlinearKernel>();
    }

    std::unique_ptr<MockMeshAccess> mesh_;
    std::unique_ptr<MockNonlinearKernel> kernel_;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(NonlinearAssemblyDriverTest, DefaultConstruction) {
    NonlinearAssemblyDriver driver;
    EXPECT_FALSE(driver.isConfigured());
}

TEST_F(NonlinearAssemblyDriverTest, ConstructionWithOptions) {
    NonlinearAssemblyOptions options;
    options.mode = NonlinearAssemblyMode::Both;
    options.jacobian_strategy = JacobianStrategy::Analytic;
    options.fd_perturbation = 1e-8;

    NonlinearAssemblyDriver driver(options);

    EXPECT_EQ(driver.getOptions().mode, NonlinearAssemblyMode::Both);
    EXPECT_EQ(driver.getOptions().jacobian_strategy, JacobianStrategy::Analytic);
    EXPECT_DOUBLE_EQ(driver.getOptions().fd_perturbation, 1e-8);
}

// ============================================================================
// Options Tests
// ============================================================================

TEST_F(NonlinearAssemblyDriverTest, NonlinearAssemblyModeEnum) {
    EXPECT_NE(NonlinearAssemblyMode::ResidualOnly, NonlinearAssemblyMode::JacobianOnly);
    EXPECT_NE(NonlinearAssemblyMode::JacobianOnly, NonlinearAssemblyMode::Both);
}

TEST_F(NonlinearAssemblyDriverTest, JacobianStrategyEnum) {
    EXPECT_NE(JacobianStrategy::Analytic, JacobianStrategy::AD_Forward);
    EXPECT_NE(JacobianStrategy::AD_Forward, JacobianStrategy::FiniteDifference);
}

TEST_F(NonlinearAssemblyDriverTest, OptionsDefaults) {
    NonlinearAssemblyOptions options;

    EXPECT_EQ(options.mode, NonlinearAssemblyMode::Both);
    EXPECT_EQ(options.jacobian_strategy, JacobianStrategy::Analytic);
    EXPECT_GT(options.fd_perturbation, 0.0);
    EXPECT_TRUE(options.fd_relative_perturbation);
    EXPECT_TRUE(options.apply_constraints);
    EXPECT_TRUE(options.zero_before_assembly);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(NonlinearAssemblyDriverTest, SetMesh) {
    NonlinearAssemblyDriver driver;
    driver.setMesh(*mesh_);
    // Still not fully configured without space, dof_map, and kernel
    EXPECT_FALSE(driver.isConfigured());
}

TEST_F(NonlinearAssemblyDriverTest, SetKernel) {
    NonlinearAssemblyDriver driver;
    driver.setKernel(*kernel_);
    EXPECT_FALSE(driver.isConfigured());  // Still needs mesh, space, dof_map
}

// ============================================================================
// Solution Management Tests
// ============================================================================

TEST_F(NonlinearAssemblyDriverTest, SetCurrentSolution) {
    NonlinearAssemblyDriver driver;

    std::vector<Real> solution(40, 1.0);
    driver.setCurrentSolution(solution);

    auto current = driver.getCurrentSolution();
    EXPECT_EQ(current.size(), 40u);
    EXPECT_DOUBLE_EQ(current[0], 1.0);
}

TEST_F(NonlinearAssemblyDriverTest, SetCurrentSolutionDifferentValues) {
    NonlinearAssemblyDriver driver;

    std::vector<Real> solution(40);
    for (std::size_t i = 0; i < 40; ++i) {
        solution[i] = static_cast<Real>(i) * 0.1;
    }
    driver.setCurrentSolution(solution);

    auto current = driver.getCurrentSolution();
    EXPECT_EQ(current.size(), 40u);
    for (std::size_t i = 0; i < 40; ++i) {
        EXPECT_DOUBLE_EQ(current[i], static_cast<Real>(i) * 0.1);
    }
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(NonlinearAssemblyDriverTest, AssemblyStatsDefaults) {
    NonlinearAssemblyStats stats;

    EXPECT_EQ(stats.num_cells, 0);
    EXPECT_EQ(stats.num_boundary_faces, 0);
    EXPECT_EQ(stats.num_interior_faces, 0);
    EXPECT_DOUBLE_EQ(stats.residual_assembly_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.jacobian_assembly_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.total_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.residual_norm, 0.0);
}

// ============================================================================
// Jacobian Verification Result Tests
// ============================================================================

TEST_F(NonlinearAssemblyDriverTest, JacobianVerificationResultDefaults) {
    JacobianVerificationResult result;

    EXPECT_FALSE(result.passed);
    EXPECT_DOUBLE_EQ(result.max_abs_error, 0.0);
    EXPECT_DOUBLE_EQ(result.max_rel_error, 0.0);
    EXPECT_DOUBLE_EQ(result.avg_abs_error, 0.0);
    EXPECT_EQ(result.worst_row, -1);
    EXPECT_EQ(result.worst_col, -1);
    EXPECT_EQ(result.num_entries_checked, 0u);
}

// ============================================================================
// Kernel Interface Tests
// ============================================================================

TEST_F(NonlinearAssemblyDriverTest, MockKernelCapabilities) {
    EXPECT_TRUE(kernel_->canComputeResidual());
    EXPECT_TRUE(kernel_->canComputeJacobian());
    EXPECT_TRUE(kernel_->hasOptimizedBoth());
}

TEST_F(NonlinearAssemblyDriverTest, MockKernelComputeResidual) {
    AssemblyContext context;
    KernelOutput output;

    kernel_->computeResidual(context, output);

    EXPECT_EQ(output.local_vector.size(), 4u);
}

TEST_F(NonlinearAssemblyDriverTest, MockKernelComputeJacobian) {
    AssemblyContext context;
    KernelOutput output;

    kernel_->computeJacobian(context, output);

    EXPECT_EQ(output.local_matrix.size(), 16u);  // 4x4

    // Check diagonal elements
    EXPECT_DOUBLE_EQ(output.local_matrix[0], 2.0);   // (0,0)
    EXPECT_DOUBLE_EQ(output.local_matrix[5], 2.0);   // (1,1)
    EXPECT_DOUBLE_EQ(output.local_matrix[10], 2.0);  // (2,2)
    EXPECT_DOUBLE_EQ(output.local_matrix[15], 2.0);  // (3,3)

    // Check off-diagonal elements are zero
    EXPECT_DOUBLE_EQ(output.local_matrix[1], 0.0);   // (0,1)
    EXPECT_DOUBLE_EQ(output.local_matrix[4], 0.0);   // (1,0)
}

TEST_F(NonlinearAssemblyDriverTest, MockKernelComputeBoth) {
    AssemblyContext context;
    KernelOutput output;

    kernel_->computeBoth(context, output);

    EXPECT_EQ(output.local_vector.size(), 4u);
    EXPECT_EQ(output.local_matrix.size(), 16u);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(NonlinearAssemblyDriverTest, CreateNonlinearAssemblyDriver) {
    auto driver = createNonlinearAssemblyDriver();
    EXPECT_NE(driver, nullptr);
    EXPECT_FALSE(driver->isConfigured());
}

TEST_F(NonlinearAssemblyDriverTest, CreateNonlinearAssemblyDriverWithOptions) {
    NonlinearAssemblyOptions options;
    options.jacobian_strategy = JacobianStrategy::FiniteDifference;

    auto driver = createNonlinearAssemblyDriver(options);
    EXPECT_NE(driver, nullptr);
    EXPECT_EQ(driver->getOptions().jacobian_strategy, JacobianStrategy::FiniteDifference);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp
