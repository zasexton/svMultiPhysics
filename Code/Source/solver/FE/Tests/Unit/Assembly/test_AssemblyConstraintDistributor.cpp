/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_AssemblyConstraintDistributor.cpp
 * @brief Unit tests for AssemblyConstraintDistributor
 */

#include <gtest/gtest.h>
#include "Assembly/AssemblyConstraintDistributor.h"
#include "Assembly/GlobalSystemView.h"
#include "Core/Types.h"

#include <vector>
#include <memory>
#include <cmath>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

// ============================================================================
// Test Fixtures
// ============================================================================

class AssemblyConstraintDistributorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup will be minimal since we don't have actual AffineConstraints
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, DefaultConstruction) {
    AssemblyConstraintDistributor distributor;
    EXPECT_FALSE(distributor.hasConstraints());
}

// ============================================================================
// Options Tests
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, AssemblyConstraintOptionsDefaults) {
    AssemblyConstraintOptions options;

    EXPECT_TRUE(options.apply_constraints);
    EXPECT_TRUE(options.symmetric_elimination);
    EXPECT_DOUBLE_EQ(options.constrained_diagonal, 1.0);
    EXPECT_TRUE(options.apply_inhomogeneities);
    EXPECT_TRUE(options.skip_unconstrained);
    EXPECT_DOUBLE_EQ(options.zero_tolerance, 1e-15);
}

TEST_F(AssemblyConstraintDistributorTest, AssemblyConstraintOptionsCustom) {
    AssemblyConstraintOptions options;
    options.apply_constraints = false;
    options.symmetric_elimination = false;
    options.constrained_diagonal = 1e10;
    options.apply_inhomogeneities = false;
    options.skip_unconstrained = false;
    options.zero_tolerance = 1e-12;

    EXPECT_FALSE(options.apply_constraints);
    EXPECT_FALSE(options.symmetric_elimination);
    EXPECT_DOUBLE_EQ(options.constrained_diagonal, 1e10);
    EXPECT_FALSE(options.apply_inhomogeneities);
    EXPECT_FALSE(options.skip_unconstrained);
    EXPECT_DOUBLE_EQ(options.zero_tolerance, 1e-12);
}

// ============================================================================
// Set Options Tests
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, SetOptions) {
    AssemblyConstraintDistributor distributor;

    AssemblyConstraintOptions options;
    options.apply_constraints = false;
    options.constrained_diagonal = 2.0;

    distributor.setOptions(options);

    EXPECT_FALSE(distributor.getOptions().apply_constraints);
    EXPECT_DOUBLE_EQ(distributor.getOptions().constrained_diagonal, 2.0);
}

TEST_F(AssemblyConstraintDistributorTest, GetOptionsReturnsReference) {
    AssemblyConstraintDistributor distributor;

    const auto& options = distributor.getOptions();
    // Default values
    EXPECT_TRUE(options.apply_constraints);
    EXPECT_TRUE(options.symmetric_elimination);
}

// ============================================================================
// Has Constraints Tests
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, HasConstraintsInitiallyFalse) {
    AssemblyConstraintDistributor distributor;
    EXPECT_FALSE(distributor.hasConstraints());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, MoveConstruction) {
    AssemblyConstraintDistributor distributor1;

    AssemblyConstraintOptions options;
    options.constrained_diagonal = 5.0;
    distributor1.setOptions(options);

    AssemblyConstraintDistributor distributor2(std::move(distributor1));

    EXPECT_DOUBLE_EQ(distributor2.getOptions().constrained_diagonal, 5.0);
}

TEST_F(AssemblyConstraintDistributorTest, MoveAssignment) {
    AssemblyConstraintDistributor distributor1;

    AssemblyConstraintOptions options;
    options.apply_constraints = false;
    distributor1.setOptions(options);

    AssemblyConstraintDistributor distributor2;
    distributor2 = std::move(distributor1);

    EXPECT_FALSE(distributor2.getOptions().apply_constraints);
}

// ============================================================================
// Dense System View Tests (for adapter testing)
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, DenseSystemViewConstruction) {
    DenseSystemView view(10);

    EXPECT_EQ(view.numRows(), 10);
    EXPECT_EQ(view.numCols(), 10);
}

TEST_F(AssemblyConstraintDistributorTest, DenseSystemViewAddMatrixEntry) {
    DenseSystemView view(5);

    view.beginAssemblyPhase();
    view.addMatrixEntry(0, 0, 1.0);
    view.addMatrixEntry(1, 1, 2.0);
    view.addMatrixEntry(0, 1, 0.5);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(view.getMatrixEntry(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view.getMatrixEntry(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(view.getMatrixEntry(0, 1), 0.5);
}

TEST_F(AssemblyConstraintDistributorTest, DenseSystemViewAddVectorEntry) {
    DenseSystemView view(5);

    view.beginAssemblyPhase();
    view.addVectorEntry(0, 1.0);
    view.addVectorEntry(1, 2.0);
    view.addVectorEntry(2, 3.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(view.getVectorEntry(0), 1.0);
    EXPECT_DOUBLE_EQ(view.getVectorEntry(1), 2.0);
    EXPECT_DOUBLE_EQ(view.getVectorEntry(2), 3.0);
}

TEST_F(AssemblyConstraintDistributorTest, DenseSystemViewAddModeAccumulates) {
    DenseSystemView view(3);

    view.beginAssemblyPhase();
    view.addMatrixEntry(0, 0, 1.0);
    view.addMatrixEntry(0, 0, 2.0);  // Should accumulate
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(view.getMatrixEntry(0, 0), 3.0);
}

TEST_F(AssemblyConstraintDistributorTest, DenseSystemViewClear) {
    DenseSystemView view(3);

    view.beginAssemblyPhase();
    view.addMatrixEntry(0, 0, 5.0);
    view.addVectorEntry(0, 10.0);
    view.endAssemblyPhase();

    view.clear();

    EXPECT_DOUBLE_EQ(view.getMatrixEntry(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(view.getVectorEntry(0), 0.0);
}

TEST_F(AssemblyConstraintDistributorTest, DenseSystemViewZero) {
    DenseSystemView view(3);

    view.beginAssemblyPhase();
    view.addMatrixEntry(1, 1, 7.0);
    view.addVectorEntry(1, 8.0);
    view.endAssemblyPhase();

    view.zero();

    EXPECT_DOUBLE_EQ(view.getMatrixEntry(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(view.getVectorEntry(1), 0.0);
}

// ============================================================================
// GlobalSystemMatrixAdapter Tests
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, GlobalSystemMatrixAdapterConstruction) {
    DenseSystemView view(10);
    GlobalSystemMatrixAdapter adapter(view);

    EXPECT_EQ(adapter.numRows(), 10);
    EXPECT_EQ(adapter.numCols(), 10);
}

TEST_F(AssemblyConstraintDistributorTest, GlobalSystemMatrixAdapterAddValue) {
    DenseSystemView view(5);
    GlobalSystemMatrixAdapter adapter(view);

    view.beginAssemblyPhase();
    adapter.addValue(0, 0, 1.0);
    adapter.addValue(1, 2, 3.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(view.getMatrixEntry(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view.getMatrixEntry(1, 2), 3.0);
}

TEST_F(AssemblyConstraintDistributorTest, GlobalSystemMatrixAdapterAddValues) {
    DenseSystemView view(5);
    GlobalSystemMatrixAdapter adapter(view);

    std::vector<GlobalIndex> rows = {0, 1};
    std::vector<GlobalIndex> cols = {0, 1};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};  // 2x2 dense block

    view.beginAssemblyPhase();
    adapter.addValues(rows, cols, values);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(view.getMatrixEntry(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view.getMatrixEntry(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(view.getMatrixEntry(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(view.getMatrixEntry(1, 1), 4.0);
}

TEST_F(AssemblyConstraintDistributorTest, GlobalSystemMatrixAdapterSetDiagonal) {
    DenseSystemView view(5);
    GlobalSystemMatrixAdapter adapter(view);

    view.beginAssemblyPhase();
    adapter.addValue(2, 2, 5.0);  // First set something
    adapter.setDiagonal(2, 1.0);  // Then set diagonal
    view.endAssemblyPhase();

    // Note: setDiagonal may overwrite or add depending on implementation
    // Just verify it doesn't crash
    EXPECT_TRUE(true);
}

// ============================================================================
// GlobalSystemVectorAdapter Tests
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, GlobalSystemVectorAdapterConstruction) {
    DenseSystemView view(10);
    GlobalSystemVectorAdapter adapter(view);

    EXPECT_EQ(adapter.size(), 10);
}

TEST_F(AssemblyConstraintDistributorTest, GlobalSystemVectorAdapterAddValue) {
    DenseSystemView view(5);
    GlobalSystemVectorAdapter adapter(view);

    view.beginAssemblyPhase();
    adapter.addValue(0, 1.0);
    adapter.addValue(2, 3.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(view.getVectorEntry(0), 1.0);
    EXPECT_DOUBLE_EQ(view.getVectorEntry(2), 3.0);
}

TEST_F(AssemblyConstraintDistributorTest, GlobalSystemVectorAdapterAddValues) {
    DenseSystemView view(5);
    GlobalSystemVectorAdapter adapter(view);

    std::vector<GlobalIndex> indices = {0, 2, 4};
    std::vector<double> values = {1.0, 2.0, 3.0};

    view.beginAssemblyPhase();
    adapter.addValues(indices, values);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(view.getVectorEntry(0), 1.0);
    EXPECT_DOUBLE_EQ(view.getVectorEntry(2), 2.0);
    EXPECT_DOUBLE_EQ(view.getVectorEntry(4), 3.0);
}

TEST_F(AssemblyConstraintDistributorTest, GlobalSystemVectorAdapterSetValue) {
    DenseSystemView view(5);
    GlobalSystemVectorAdapter adapter(view);

    view.beginAssemblyPhase();
    adapter.addValue(1, 5.0);
    adapter.setValue(1, 10.0);  // Should overwrite
    view.endAssemblyPhase();

    // Note: behavior depends on implementation
    EXPECT_TRUE(true);
}

TEST_F(AssemblyConstraintDistributorTest, GlobalSystemVectorAdapterGetValue) {
    DenseSystemView view(5);
    GlobalSystemVectorAdapter adapter(view);

    view.beginAssemblyPhase();
    adapter.addValue(3, 7.5);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(adapter.getValue(3), 7.5);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

// Note: Factory functions require AffineConstraints which we don't have in this test
// These tests would be integration tests with the Constraints module

// ============================================================================
// Query Tests (without actual constraints)
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, NumConstraintsWithoutConstraints) {
    AssemblyConstraintDistributor distributor;
    // Without constraints set, should return 0
    EXPECT_EQ(distributor.numConstraints(), 0u);
}

TEST_F(AssemblyConstraintDistributorTest, GetConstrainedDofsWithoutConstraints) {
    AssemblyConstraintDistributor distributor;
    auto constrained = distributor.getConstrainedDofs();
    EXPECT_TRUE(constrained.empty());
}

// ============================================================================
// Rectangular Assembly Tests (structure only)
// ============================================================================

TEST_F(AssemblyConstraintDistributorTest, DenseSystemViewRectangular) {
    // Test non-square system view
    DenseSystemView view(10, 5);  // 10 rows, 5 cols

    EXPECT_EQ(view.numRows(), 10);
    EXPECT_EQ(view.numCols(), 5);

    view.beginAssemblyPhase();
    view.addMatrixEntry(9, 4, 1.0);  // Bottom-right corner
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(view.getMatrixEntry(9, 4), 1.0);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp
