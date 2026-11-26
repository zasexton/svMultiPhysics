/**
 * @file test_ConstrainedAssembly.cpp
 * @brief Unit tests for ConstrainedAssembly
 */

#include <gtest/gtest.h>

#include "FE/Dofs/ConstrainedAssembly.h"
#include "FE/Dofs/DofConstraints.h"

#include <array>
#include <string_view>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::dofs::ConstrainedAssembly;
using svmp::FE::dofs::ConstrainedAssemblyOptions;
using svmp::FE::dofs::DenseMatrixAdapter;
using svmp::FE::dofs::DofConstraints;

TEST(ConstrainedAssembly, NoConstraintsAssemblesDirectly) {
    DenseMatrixAdapter adapter(/*n_rows=*/3, /*n_cols=*/3);
    ConstrainedAssembly assembly; // not initialized => constraints_ == nullptr

    const std::vector<GlobalIndex> dofs = {0, 1};
    const std::vector<double> cell_matrix = {
        1.0, 2.0,
        3.0, 4.0
    };
    const std::vector<double> cell_rhs = {10.0, 20.0};

    assembly.distributeLocalToGlobal(cell_matrix, cell_rhs, dofs, adapter);

    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(1, 1), 4.0);

    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(0), 10.0);
    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(1), 20.0);
}

TEST(ConstrainedAssembly, DirichletEliminatesColumnContributionToRhs) {
    DenseMatrixAdapter adapter(/*n_rows=*/3, /*n_cols=*/3);

    DofConstraints constraints;
    constraints.addDirichletBC(0, 5.0);
    constraints.close();

    ConstrainedAssembly assembly;
    assembly.initialize(constraints, ConstrainedAssemblyOptions{});

    const std::vector<GlobalIndex> dofs = {0, 1};
    const std::vector<double> cell_matrix = {
        1.0, 2.0,
        3.0, 4.0
    };
    const std::vector<double> cell_rhs = {10.0, 20.0};

    assembly.distributeLocalToGlobal(cell_matrix, cell_rhs, dofs, adapter);

    // Row/col associated with constrained DOF 0 is dropped at assembly-time.
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(1, 1), 4.0);

    // RHS for DOF 1 is modified: rhs1 -= A10 * g = 20 - 3*5 = 5.
    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(0), 0.0);
    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(1), 5.0);
}



TEST(ConstrainedAssembly, AffineSlaveContributesToProjectedLoad) {
    DofConstraints constraints;
    constraints.addLinearConstraint(1, std::array<GlobalIndex, 1>{0}, std::array{1.0}, 1.0);
    constraints.close();
    ConstrainedAssembly assembly;
    assembly.initialize(constraints);
    DenseMatrixAdapter adapter(2, 2);
    assembly.distributeLocalToGlobal(std::array{1., 0., 0., 1.}, std::array{0., 0.},
                                     std::array<GlobalIndex, 2>{0, 1}, adapter);

    // x=(y,y+1) in A=I, b=0 gives the reduced equation 2y=-1.
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 0), 2.);
    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(0), -1.);
}

TEST(ConstrainedAssembly, NestedMultipointAndFixedDofsCondenseIntoSparseMasters) {
    DofConstraints constraints;
    constraints.addLinearConstraint(3, std::array<GlobalIndex, 2>{1, 2}, std::array{2., .25}, 4.);
    constraints.addLinearConstraint(1, std::array<GlobalIndex, 2>{0, 4}, std::array{.5, -1.}, 2.);
    constraints.addDirichletBC(2, 3.);
    constraints.close();
    ConstrainedAssembly assembly;
    assembly.initialize(constraints);

    // Local x=P*y+g with rows (1,-2), (1,0), (0,0), (.5,-1)
    // and g=(8.75,0,3,2). Master 4 lies outside this element.
    const std::array<GlobalIndex, 4> dofs{3, 0, 2, 1};
    const std::array matrix{3., -4., 1., 7., -2., 9., 2., 0., 6., 1., 5., 2., 0., 8., -2., 4.};
    const std::array rhs{2., -1., 6., 3.};
    svmp::FE::dofs::CSRMatrixAdapter adapter(std::array<GlobalIndex, 6>{0, 2, 2, 2, 2, 4},
                                             std::array<GlobalIndex, 4>{0, 4, 0, 4});
    assembly.distributeLocalToGlobal(matrix, rhs, dofs, adapter);

    const auto reduced = adapter.getValues();
    EXPECT_DOUBLE_EQ(reduced[0], 14.5);
    EXPECT_DOUBLE_EQ(reduced[1], -11.);
    EXPECT_DOUBLE_EQ(reduced[2], -15.);
    EXPECT_DOUBLE_EQ(reduced[3], 30.);
    const auto load = adapter.getVectorData();
    EXPECT_DOUBLE_EQ(load[0], -30.25);
    EXPECT_DOUBLE_EQ(load[1], 0.);
    EXPECT_DOUBLE_EQ(load[2], 0.);
    EXPECT_DOUBLE_EQ(load[3], 0.);
    EXPECT_DOUBLE_EQ(load[4], 81.5);
}

TEST(ConstrainedAssembly, MatrixAndOffsetScalingPreserveFiniteLift) {
    for (const double scale : {1.e-20, 1.e20}) {
        SCOPED_TRACE(scale);
        DofConstraints constraints;
        constraints.addLinearConstraint(1, std::array<GlobalIndex, 1>{0}, std::array{1.}, 1. / scale);
        constraints.close();
        ConstrainedAssembly assembly;
        assembly.initialize(constraints);
        DenseMatrixAdapter adapter(2, 2);
        assembly.distributeLocalToGlobal(std::array{2.*scale, -scale, 3.*scale, 4.*scale},
                                         std::array{0., 0.}, std::array<GlobalIndex, 2>{0, 1}, adapter);
        EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 0), 8.*scale);
        EXPECT_DOUBLE_EQ(adapter.getVectorEntry(0), -3.);
    }
}

TEST(ConstrainedAssembly, OpenChainRequiresClosureBeforeWritingContributions) {
    DofConstraints constraints;
    constraints.addLinearConstraint(1, std::array<GlobalIndex, 1>{0}, std::array{2.}, 1.);
    constraints.addLinearConstraint(2, std::array<GlobalIndex, 1>{1}, std::array{-.5}, .75);
    ConstrainedAssembly assembly;
    assembly.initialize(constraints);
    DenseMatrixAdapter adapter(3, 3);

    // The flat constraint x1 = 2 x0 + 1 is usable before closure.
    assembly.distributeLocalToGlobal(std::array{3.}, std::array{10.},
                                     std::array<GlobalIndex, 1>{1}, adapter);
    ASSERT_DOUBLE_EQ(adapter.getMatrixEntry(0, 0), 12.);
    ASSERT_DOUBLE_EQ(adapter.getVectorEntry(0), 14.);

    const std::array<GlobalIndex, 2> dofs{0, 2};
    const std::array matrix{2., 3., 5., 7.};
    const std::array rhs{11., 13.};
    try {
        assembly.distributeLocalToGlobal(matrix, rhs, dofs, adapter);
        ADD_FAILURE() << "The active constraint on DOF 2 still refers to constrained DOF 1";
    } catch (const svmp::FE::FEException& error) {
        EXPECT_NE(std::string_view(error.what()).find("DofConstraints::close()"), std::string_view::npos);
    }
    // The first cell of either batch is valid. A retry after closing must add it only once.
    const std::array cell_matrices{2., 7.};
    EXPECT_THROW(assembly.distributeLocalToGlobalBatch(2, cell_matrices, rhs,
                     std::array<GlobalIndex, 3>{0, 1, 2}, dofs, adapter), svmp::FE::FEException);
    EXPECT_THROW(assembly.distributeLocalToGlobalUniform(2, 1, cell_matrices, rhs, dofs, adapter),
                 svmp::FE::FEException);
    for (GlobalIndex row = 0; row < 3; ++row) {
        EXPECT_DOUBLE_EQ(adapter.getVectorEntry(row), row == 0 ? 14. : 0.);
        for (GlobalIndex col = 0; col < 3; ++col) {
            EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(row, col), row == 0 && col == 0 ? 12. : 0.);
        }
    }

    // Closure gives x2 = -x0 + 1/4. This element adds 1 to the matrix and -1 to the load.
    constraints.close();
    assembly.distributeLocalToGlobal(matrix, rhs, dofs, adapter);
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 0), 13.);
    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(0), 13.);
}

TEST(ConstrainedAssembly, SmallMasterWeightsSurviveInputAndTransitiveClosure) {
    const std::array<GlobalIndex, 1> master{0};
    DofConstraints constraints;
    constraints.addLinearConstraint(1, master, std::array{1.e-16});
    constraints.addHangingNodeConstraint(2, master, std::array{1.e-16});
    constraints.addLinearConstraint(3, master, std::array{1.e-8});
    constraints.addLinearConstraint(4, std::array<GlobalIndex, 1>{3}, std::array{1.e-8});
    constraints.close();
    ConstrainedAssembly assembly;
    assembly.initialize(constraints);

    for (const GlobalIndex slave : {1, 2, 4}) {
        SCOPED_TRACE(slave);
        DenseMatrixAdapter adapter(5, 5);
        assembly.distributeLocalToGlobal(std::array{1.e32}, std::array{1.e16},
                                         std::array<GlobalIndex, 1>{slave}, adapter);
        EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 0), 1.);
        EXPECT_DOUBLE_EQ(adapter.getVectorEntry(0), 1.);
    }
}

TEST(ConstrainedAssembly, SeparateAssemblyAccumulatesOnlyItsOwnOutput) {
    DofConstraints constraints;
    constraints.addLinearConstraint(1, std::array<GlobalIndex, 1>{0}, std::array{1.}, 1.);
    constraints.close();
    ConstrainedAssembly assembly;
    assembly.initialize(constraints);
    const std::array<GlobalIndex, 2> dofs{0, 1};
    DenseMatrixAdapter adapter(2, 2);
    adapter.setVectorValues(dofs, std::array{7., 8.});
    assembly.distributeMatrixToGlobal(std::array{1., 2., 3., 4.}, dofs, adapter);
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 0), 10.);
    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(0), 7.);
    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(1), 8.);
    assembly.distributeRhsToGlobal(std::array{5., 6.}, dofs, adapter);
    EXPECT_DOUBLE_EQ(adapter.getMatrixEntry(0, 0), 10.);
    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(0), 18.);
    EXPECT_DOUBLE_EQ(adapter.getVectorEntry(1), 8.);
}

TEST(ConstrainedAssembly, BatchEntryPointsAccumulateCellLifts) {
    DofConstraints constraints;
    constraints.addLinearConstraint(1, std::array<GlobalIndex, 1>{0}, std::array{1.}, 1.);
    constraints.close();
    ConstrainedAssembly assembly;
    assembly.initialize(constraints);
    DenseMatrixAdapter mixed(2, 2), uniform(2, 2);
    assembly.distributeLocalToGlobalBatch(2, std::array{1., 0., 0., 1., 2.}, std::array{0., 0., 0.},
                                          std::array<GlobalIndex, 3>{0, 2, 3},
                                          std::array<GlobalIndex, 3>{0, 1, 1}, mixed);
    assembly.distributeLocalToGlobalUniform(2, 1, std::array{1., 2.}, std::array{0., 0.},
                                            std::array<GlobalIndex, 2>{1, 1}, uniform);
    EXPECT_DOUBLE_EQ(mixed.getMatrixEntry(0, 0), 4.);
    EXPECT_DOUBLE_EQ(mixed.getVectorEntry(0), -3.);
    EXPECT_DOUBLE_EQ(uniform.getMatrixEntry(0, 0), 3.);
    EXPECT_DOUBLE_EQ(uniform.getVectorEntry(0), -3.);
}
