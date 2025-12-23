/**
 * @file test_ConstrainedAssembly.cpp
 * @brief Unit tests for ConstrainedAssembly
 */

#include <gtest/gtest.h>

#include "FE/Dofs/ConstrainedAssembly.h"
#include "FE/Dofs/DofConstraints.h"

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

