/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_AssemblyConstraintDistributor_Integration.cpp
 * @brief Integration-style tests for AssemblyConstraintDistributor using real AffineConstraints
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyConstraintDistributor.h"
#include "Assembly/GlobalSystemView.h"
#include "Constraints/AffineConstraints.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace testing {

TEST(AssemblyConstraintDistributorIntegrationTest, MPC_SymmetricElimination_DistributesRowsAndColumnsToMasters)
{
    constraints::AffineConstraints c;
    c.addLine(/*slave=*/1);
    c.addEntry(/*slave=*/1, /*master=*/0, /*weight=*/1.0);
    c.close();

    AssemblyConstraintOptions opts;
    opts.symmetric_elimination = true;
    opts.constrained_diagonal = 42.0;
    opts.skip_unconstrained = false;

    AssemblyConstraintDistributor dist(c, opts);

    DenseSystemView sys(/*n_dofs=*/3);
    sys.zero();
    sys.beginAssemblyPhase();

    const std::array<GlobalIndex, 3> dofs = {0, 1, 2};
    const std::array<Real, 9> A = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    const std::array<Real, 3> b = {10.0, 11.0, 12.0};

    dist.distributeLocalToGlobal(A, b, dofs, sys, sys);
    dist.finalizeConstrainedRows(sys);

    sys.endAssemblyPhase();

    // Constraint: u1 = u0. With symmetric elimination, constrained row/col contributions
    // are substituted into the master DOF 0, producing P^T A P on the unconstrained block.
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(0, 0), 1.0 + 2.0 + 4.0 + 5.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(0, 2), 3.0 + 6.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(2, 0), 7.0 + 8.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(2, 2), 9.0);

    // Constrained row/col eliminated; diagonal set during finalization.
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(1, 1), opts.constrained_diagonal);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(1, 2), 0.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(2, 1), 0.0);

    // RHS: unconstrained row + constrained-row distributed to master.
    EXPECT_DOUBLE_EQ(sys.getVectorEntry(0), 10.0 + 11.0);
    EXPECT_DOUBLE_EQ(sys.getVectorEntry(1), 0.0);
    EXPECT_DOUBLE_EQ(sys.getVectorEntry(2), 12.0);
}

TEST(AssemblyConstraintDistributorIntegrationTest, MPC_NonsymmetricElimination_DoesNotDistributeConstrainedColumns)
{
    constraints::AffineConstraints c;
    c.addLine(/*slave=*/1);
    c.addEntry(/*slave=*/1, /*master=*/0, /*weight=*/1.0);
    c.close();

    AssemblyConstraintOptions opts;
    opts.symmetric_elimination = false;
    opts.constrained_diagonal = 7.0;
    opts.skip_unconstrained = false;

    AssemblyConstraintDistributor dist(c, opts);

    DenseSystemView sys(/*n_dofs=*/3);
    sys.zero();
    sys.beginAssemblyPhase();

    const std::array<GlobalIndex, 3> dofs = {0, 1, 2};
    const std::array<Real, 9> A = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    const std::array<Real, 3> b = {10.0, 11.0, 12.0};

    dist.distributeLocalToGlobal(A, b, dofs, sys, sys);
    dist.finalizeConstrainedRows(sys);

    sys.endAssemblyPhase();

    // With symmetric_elimination=false, constrained columns are not substituted to masters.
    // Contributions where the constrained DOF appears in a column are dropped, while
    // constrained rows are still distributed to masters.
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(0, 0), 1.0 + 4.0 + 5.0); // A00 + A10 + A11 (A01 dropped)
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(0, 2), 3.0 + 6.0);       // A02 + A12
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(2, 0), 7.0);             // A20 (A21 dropped)
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(2, 2), 9.0);

    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(1, 1), opts.constrained_diagonal);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(1, 2), 0.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(2, 1), 0.0);

    EXPECT_DOUBLE_EQ(sys.getVectorEntry(0), 10.0 + 11.0);
    EXPECT_DOUBLE_EQ(sys.getVectorEntry(1), 0.0);
    EXPECT_DOUBLE_EQ(sys.getVectorEntry(2), 12.0);
}

} // namespace testing
} // namespace assembly
} // namespace FE
} // namespace svmp

