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

TEST(AssemblyConstraintDistributorIntegrationTest, RetainsScaledContributionsAndAffineLift)
{
    // u0 = u3/2, u1 = g.  The free equations are P^T A P and P^T (b - A g).
    for (const double g : {1.0, 0x1p-80}) {
        constraints::AffineConstraints c;
        c.addLine(0);
        c.addEntry(0, 3, 0.5);
        c.addDirichlet(1, g);
        c.close();
        AssemblyConstraintDistributor dist(c);
        for (const bool rectangular : {false, true}) {
            DenseSystemView sys(4);
            sys.beginAssemblyPhase();
            if (rectangular) {
                const std::array<GlobalIndex, 2> rows{0,2}, cols{1,2};
                dist.distributeLocalToGlobal(std::array{1/g,4*g,2.,2*g}, std::array{0.,7*g},
                                             rows, cols, sys, sys);
            } else {
                const std::array<GlobalIndex, 3> dofs{0,1,2};
                dist.distributeLocalToGlobal(std::array{0.,1/g,4*g, 0.,0.,0., 0.,2.,2*g},
                                             std::array{0.,0.,7*g}, dofs, sys, sys);
            }
            EXPECT_DOUBLE_EQ(sys.getMatrixEntry(3, 2), 2*g);
            EXPECT_DOUBLE_EQ(sys.getMatrixEntry(2, 2), 2*g);
            EXPECT_DOUBLE_EQ(sys.getVectorEntry(3), -0.5);
            EXPECT_DOUBLE_EQ(sys.getVectorEntry(2), 5*g);
        }
    }
}

TEST(AssemblyConstraintDistributorIntegrationTest, ExplicitDropAppliesToFinalContributions)
{
    const double g = 0x1p-80;
    constraints::AffineConstraints c;
    c.addDirichlet(0, g);
    c.close();
    AssemblyConstraintOptions opts;
    opts.zero_tolerance = 1e-12;
    AssemblyConstraintDistributor dist(c, opts);
    const std::array<GlobalIndex,2> dofs{0,1};
    DenseSystemView sys(2);
    sys.beginAssemblyPhase();
    dist.distributeLocalToGlobal(std::array{0.,0.,1/g,g}, std::array{0.,g}, dofs, sys, sys);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(1,1), 0.);
    EXPECT_DOUBLE_EQ(sys.getVectorEntry(1), -1.);
}

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

TEST(AssemblyConstraintDistributorIntegrationTest, MPC_NonsymmetricElimination_StillDistributesConstrainedColumns)
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

    // General affine elimination must substitute constrained trial columns
    // into their masters even for nonsymmetric operators.
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(0, 0), 1.0 + 2.0 + 4.0 + 5.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(0, 2), 3.0 + 6.0);
    EXPECT_DOUBLE_EQ(sys.getMatrixEntry(2, 0), 7.0 + 8.0);
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
