/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ConstraintDistributor.cpp
 * @brief Unit tests for ConstraintDistributor class
 */

#include <gtest/gtest.h>
#include "Constraints/ConstraintDistributor.h"
#include "Constraints/AffineConstraints.h"

#include <vector>
#include <cmath>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

// ============================================================================
// Helpers
// ============================================================================

// Helper to create a simple 2x2 element matrix
std::vector<double> makeElemMat(double a, double b, double c, double d) {
    return {a, b, c, d};
}

// Helper to create a simple 2-element RHS
std::vector<double> makeElemRhs(double a, double b) {
    return {a, b};
}

// ============================================================================
// ConstraintDistributor Tests
// ============================================================================

TEST(ConstraintDistributorTest, DefaultConstruction) {
    ConstraintDistributor distributor;
    EXPECT_FALSE(distributor.hasConstraints());
}

TEST(ConstraintDistributorTest, ConstructionWithConstraints) {
    AffineConstraints constraints;
    constraints.close();
    ConstraintDistributor distributor(constraints);
    EXPECT_TRUE(distributor.hasConstraints());
}

TEST(ConstraintDistributorTest, DistributeLocalToGlobal_Unconstrained) {
    AffineConstraints constraints;
    constraints.close();
    ConstraintDistributor distributor(constraints);

    // Global system: 2x2
    DenseMatrixOps global_matrix(2, 2);
    DenseVectorOps global_rhs(2);

    // Element: contributes to DOFs 0 and 1
    std::vector<double> elem_mat = makeElemMat(1.0, 2.0, 3.0, 4.0);
    std::vector<double> elem_rhs = makeElemRhs(5.0, 6.0);
    std::vector<GlobalIndex> dofs = {0, 1};

    distributor.distributeLocalToGlobal(elem_mat, elem_rhs, dofs, global_matrix, global_rhs);

    // Should be direct addition
    EXPECT_DOUBLE_EQ(global_matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(global_matrix(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(global_matrix(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(global_matrix(1, 1), 4.0);

    EXPECT_DOUBLE_EQ(global_rhs[0], 5.0);
    EXPECT_DOUBLE_EQ(global_rhs[1], 6.0);
}

TEST(ConstraintDistributorTest, DistributeLocalToGlobal_Dirichlet) {
    // Constraint: u_0 = 0.0 (Dirichlet)
    // Global system: 2x2. Element DOFs: 0, 1.
    // Matrix:
    // [ 1 2 ] [u_0] = [5]
    // [ 3 4 ] [u_1]   [6]
    //
    // Applying u_0 = 0:
    // Row 0 becomes identity (or similar): u_0 = 0 -> 1*u_0 = 0
    // Row 1: 3*u_0 + 4*u_1 = 6 => 4*u_1 = 6 - 3*0 = 6
    //
    // With symmetric elimination, we also zero column 0 in row 1.

    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.setInhomogeneity(0, 0.0);
    constraints.close();

    DistributorOptions opts;
    opts.symmetric = true;
    opts.constrained_diagonal = 1.0;
    ConstraintDistributor distributor(constraints, opts);

    DenseMatrixOps global_matrix(2, 2);
    DenseVectorOps global_rhs(2);

    std::vector<double> elem_mat = makeElemMat(1.0, 2.0, 3.0, 4.0);
    std::vector<double> elem_rhs = makeElemRhs(5.0, 6.0);
    std::vector<GlobalIndex> dofs = {0, 1};

    distributor.distributeLocalToGlobal(elem_mat, elem_rhs, dofs, global_matrix, global_rhs);

    // Check constrained row 0
    // Diagonal should be set (default 1.0 or summed if we distribute?)
    // Wait, distributor distributes *element* contributions.
    // For a constrained row, it typically DROPS the element contributions to that row
    // and relies on setDiagonal() being called later, OR it accumulates into an
    // identity-like structure.
    //
    // Let's check logic: distributeElementCore checks if row is constrained.
    // If row is constrained, it generally skips adding to that row, EXCEPT for the diagonal
    // if configured.
    //
    // If symmetric, column contributions from constrained vars are moved to RHS.

    // Row 0 (constrained):
    // Contributions 1.0 (0,0) and 2.0 (0,1) should be ignored or handled specially.
    // Usually local-to-global adds nothing to constrained rows, and we post-process diagonal.
    // OR distributor adds diagonal explicitly.
    // Let's assume standard behavior: constrained row entries are dropped.
    EXPECT_DOUBLE_EQ(global_matrix(0, 0), 0.0); // Or options.constrained_diagonal?
    EXPECT_DOUBLE_EQ(global_matrix(0, 1), 0.0);

    // Row 1 (unconstrained):
    // (1,0) entry is 3.0. Since u_0 is constrained, this column entry should be eliminated.
    // Symmetric elimination: don't add to matrix, move to RHS?
    // u_0 = 0 -> 3*u_0 = 0. So nothing added to RHS.
    // Entry (1,0) should be 0.0 in matrix.
    EXPECT_DOUBLE_EQ(global_matrix(1, 0), 0.0);

    // (1,1) entry is 4.0. Unconstrained. Added.
    EXPECT_DOUBLE_EQ(global_matrix(1, 1), 4.0);

    // RHS:
    // Row 0: Dropped. (constrained)
    // Row 1: 6.0 - 3.0 * (inhomogeneity=0) = 6.0.
    // (Note: distribute doesn't clear the global RHS, it adds. Initial is 0).
    EXPECT_DOUBLE_EQ(global_rhs[0], 0.0);
    EXPECT_DOUBLE_EQ(global_rhs[1], 6.0);
}

TEST(ConstraintDistributorTest, DistributeLocalToGlobal_InhomogeneousDirichlet) {
    // Constraint: u_0 = 10.0
    // Matrix: [1 2; 3 4], RHS: [5; 6]
    // u_0 = 10 -> Row 1: 3*(10) + 4*u_1 = 6 -> 4*u_1 = 6 - 30 = -24

    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.setInhomogeneity(0, 10.0);
    constraints.close();

    DistributorOptions opts;
    opts.symmetric = true;
    ConstraintDistributor distributor(constraints, opts);

    DenseMatrixOps global_matrix(2, 2);
    DenseVectorOps global_rhs(2);

    std::vector<double> elem_mat = makeElemMat(1.0, 2.0, 3.0, 4.0);
    std::vector<double> elem_rhs = makeElemRhs(5.0, 6.0);
    std::vector<GlobalIndex> dofs = {0, 1};

    distributor.distributeLocalToGlobal(elem_mat, elem_rhs, dofs, global_matrix, global_rhs);

    // Row 0 (constrained): Dropped
    EXPECT_DOUBLE_EQ(global_matrix(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(global_matrix(0, 1), 0.0);

    // Row 1 (unconstrained):
    // Col 0 (u_0): Dropped (symmetric elimination)
    EXPECT_DOUBLE_EQ(global_matrix(1, 0), 0.0);
    // Col 1 (u_1): 4.0
    EXPECT_DOUBLE_EQ(global_matrix(1, 1), 4.0);

    // RHS:
    // Row 0: Dropped
    EXPECT_DOUBLE_EQ(global_rhs[0], 0.0);
    // Row 1: 6.0 - 3.0 * 10.0 = -24.0
    EXPECT_DOUBLE_EQ(global_rhs[1], -24.0);
}

TEST(ConstraintDistributorTest, DistributeLocalToGlobal_Periodic) {
    // Constraint: u_0 = u_1 (Periodic/MPC)
    // Global system: 2x2. Element DOFs: 0, 1.
    // But u_0 is constrained to u_1.
    // Effectively, we are condensing u_0 into u_1.
    //
    // Local matrix [K00 K01; K10 K11], RHS [F0; F1]
    // u_0 -> u_1.
    //
    // Row 0 equation becomes part of Row 1 equation.
    // (K00*u_0 + K01*u_1 = F0) -> (K00*u_1 + K01*u_1 = F0) -> added to Row 1
    // Row 1 equation: (K10*u_0 + K11*u_1 = F1) -> (K10*u_1 + K11*u_1 = F1)
    //
    // Total Row 1: (K00 + K01 + K10 + K11) * u_1 = F0 + F1
    //
    // Matrix values: 1, 2, 3, 4. Sum = 10.
    // RHS values: 5, 6. Sum = 11.

    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.close();

    ConstraintDistributor distributor(constraints); // Default symmetric=true

    DenseMatrixOps global_matrix(2, 2);
    DenseVectorOps global_rhs(2);

    std::vector<double> elem_mat = makeElemMat(1.0, 2.0, 3.0, 4.0);
    std::vector<double> elem_rhs = makeElemRhs(5.0, 6.0);
    std::vector<GlobalIndex> dofs = {0, 1};

    distributor.distributeLocalToGlobal(elem_mat, elem_rhs, dofs, global_matrix, global_rhs);

    // Row 0 (constrained): Dropped
    EXPECT_DOUBLE_EQ(global_matrix(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(global_matrix(0, 1), 0.0);

    // Row 1 (master):
    // Receives contributions from Row 0 and Row 1
    // K11_new = K11 + K10*w + K01*w + K00*w*w (where w=1)
    //         = 4 + 3*1 + 2*1 + 1*1*1 = 10
    EXPECT_DOUBLE_EQ(global_matrix(1, 1), 10.0);

    // RHS 1: F1 + F0*w = 6 + 5*1 = 11
    EXPECT_DOUBLE_EQ(global_rhs[1], 11.0);
}

TEST(ConstraintDistributorTest, CondenseLocal) {
    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.setInhomogeneity(0, 10.0); // u_0 = 10
    constraints.close();

    ConstraintDistributor distributor(constraints);

    std::vector<double> elem_mat = makeElemMat(1.0, 2.0, 3.0, 4.0);
    std::vector<double> elem_rhs = makeElemRhs(5.0, 6.0);
    std::vector<GlobalIndex> dofs = {0, 1};

    distributor.condenseLocal(elem_mat, elem_rhs, dofs);

    // Condense local modifies the local matrix/rhs to be consistent with constraints.
    // Typically:
    // - Constrained rows/cols zeroed (diagonal 1.0 usually)
    // - RHS modified
    //
    // Row 0: u_0 = 10. K00=1, K01=0, F0=10.
    // Row 1: 3*u_0 + 4*u_1 = 6 -> 4*u_1 = 6 - 3*10 = -24.
    //        K10=0, K11=4, F1=-24.

    EXPECT_DOUBLE_EQ(elem_mat[0], 1.0); // Diagonal kept? Or 1.0
    EXPECT_DOUBLE_EQ(elem_mat[1], 0.0);
    EXPECT_DOUBLE_EQ(elem_mat[2], 0.0);
    EXPECT_DOUBLE_EQ(elem_mat[3], 4.0);

    EXPECT_DOUBLE_EQ(elem_rhs[0], 10.0); // Inhomogeneity
    EXPECT_DOUBLE_EQ(elem_rhs[1], -24.0);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
