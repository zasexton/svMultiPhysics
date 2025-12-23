/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_LagrangeMultiplier.cpp
 * @brief Unit tests for LagrangeMultiplier class
 */

#include <gtest/gtest.h>

#include "Constraints/AffineConstraints.h"
#include "Constraints/LagrangeMultiplier.h"

#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(LagrangeMultiplierTest, InitializeFromAffineConstraints) {
    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.addEntry(0, 1, 2.0);
    constraints.setInhomogeneity(0, 3.0);
    constraints.close();

    LagrangeMultiplier lagrange(constraints);

    EXPECT_TRUE(lagrange.isFinalized());
    EXPECT_EQ(lagrange.numConstraints(), 1);

    const auto& rows = lagrange.getConstraints();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].constrained_dofs.size(), 2u);
    EXPECT_EQ(rows[0].constrained_dofs[0], 0);
    EXPECT_EQ(rows[0].constrained_dofs[1], 1);
    EXPECT_DOUBLE_EQ(rows[0].coefficients[0], 1.0);
    EXPECT_DOUBLE_EQ(rows[0].coefficients[1], -2.0);
    EXPECT_DOUBLE_EQ(rows[0].rhs, 3.0);
}

TEST(LagrangeMultiplierTest, CSRExportAndResidual) {
    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.addEntry(0, 1, 2.0);
    constraints.setInhomogeneity(0, 3.0);
    constraints.close();

    LagrangeMultiplier lagrange(constraints);

    std::vector<GlobalIndex> row_offsets;
    std::vector<GlobalIndex> col_indices;
    std::vector<double> values;
    lagrange.getConstraintMatrixCSR(row_offsets, col_indices, values);

    ASSERT_EQ(row_offsets.size(), 2u);
    EXPECT_EQ(row_offsets[0], 0);
    EXPECT_EQ(row_offsets[1], 2);

    ASSERT_EQ(col_indices.size(), 2u);
    ASSERT_EQ(values.size(), 2u);
    EXPECT_EQ(col_indices[0], 0);
    EXPECT_EQ(col_indices[1], 1);
    EXPECT_DOUBLE_EQ(values[0], 1.0);
    EXPECT_DOUBLE_EQ(values[1], -2.0);

    auto rhs = lagrange.getConstraintRhs();
    ASSERT_EQ(rhs.size(), 1u);
    EXPECT_DOUBLE_EQ(rhs[0], 3.0);

    // Choose u satisfying: u0 - 2*u1 = 3
    std::vector<double> u = {7.0, 2.0};
    EXPECT_TRUE(lagrange.checkSatisfaction(u, 1e-12));

    auto residual = lagrange.computeResidual(u);
    ASSERT_EQ(residual.size(), 1u);
    EXPECT_NEAR(residual[0], 0.0, 1e-12);
}

TEST(LagrangeMultiplierTest, ConstraintForcesEqualTranspose) {
    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.addEntry(0, 1, 2.0);
    constraints.setInhomogeneity(0, 3.0);
    constraints.close();

    LagrangeMultiplier lagrange(constraints);

    std::vector<double> lambda = {5.0};
    auto forces = lagrange.computeConstraintForces(lambda);
    ASSERT_EQ(forces.size(), 2u);
    EXPECT_DOUBLE_EQ(forces[0], 5.0);
    EXPECT_DOUBLE_EQ(forces[1], -10.0);
}

TEST(LagrangeMultiplierTest, SaddlePointOperatorApplication) {
    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.addEntry(0, 1, 2.0);
    constraints.setInhomogeneity(0, 3.0);
    constraints.close();

    LagrangeMultiplier lagrange(constraints);

    // A is identity
    auto A_apply = [](std::span<const double> x, std::span<double> y) {
        ASSERT_EQ(x.size(), y.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            y[i] = x[i];
        }
    };

    std::vector<double> u = {7.0, 2.0};        // satisfies constraint
    std::vector<double> lambda = {5.0};
    std::vector<double> out_u(2, 0.0);
    std::vector<double> out_lambda(1, 0.0);

    lagrange.applySaddlePointOperator(A_apply, u, lambda, out_u, out_lambda);

    // out_u = A*u + B^T*lambda = u + [5, -10]
    EXPECT_DOUBLE_EQ(out_u[0], 12.0);
    EXPECT_DOUBLE_EQ(out_u[1], -8.0);

    // out_lambda = B*u = rhs (since u satisfies); stabilization default 0
    EXPECT_NEAR(out_lambda[0], 3.0, 1e-12);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

