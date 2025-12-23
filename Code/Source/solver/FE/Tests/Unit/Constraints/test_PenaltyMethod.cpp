/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_PenaltyMethod.cpp
 * @brief Unit tests for PenaltyMethod class
 */

#include <gtest/gtest.h>

#include "Constraints/AffineConstraints.h"
#include "Constraints/PenaltyMethod.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(PenaltyMethodTest, InitializeFromAffineConstraints) {
    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.addEntry(0, 1, 2.0);
    constraints.setInhomogeneity(0, 3.0);
    constraints.close();

    PenaltyMethod penalty(constraints);

    EXPECT_EQ(penalty.numConstraints(), 1);
    const auto& pcs = penalty.getConstraints();
    ASSERT_EQ(pcs.size(), 1u);

    EXPECT_EQ(pcs[0].dofs.size(), 2u);
    EXPECT_EQ(pcs[0].dofs[0], 0);
    EXPECT_EQ(pcs[0].dofs[1], 1);
    EXPECT_DOUBLE_EQ(pcs[0].coefficients[0], 1.0);
    EXPECT_DOUBLE_EQ(pcs[0].coefficients[1], -2.0);
    EXPECT_DOUBLE_EQ(pcs[0].rhs, 3.0);
    EXPECT_GT(pcs[0].penalty, 0.0);
}

TEST(PenaltyMethodTest, PenaltyMatrixAndRhsCSR) {
    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.addEntry(0, 1, 2.0);
    constraints.setInhomogeneity(0, 3.0);
    constraints.close();

    PenaltyMethod penalty(constraints);

    std::vector<GlobalIndex> row_offsets;
    std::vector<GlobalIndex> col_indices;
    std::vector<double> values;
    penalty.getPenaltyMatrixCSR(row_offsets, col_indices, values, /*n_dofs=*/2);

    ASSERT_EQ(row_offsets.size(), 3u);
    EXPECT_EQ(row_offsets[0], 0);
    EXPECT_EQ(row_offsets[1], 2);
    EXPECT_EQ(row_offsets[2], 4);

    ASSERT_EQ(col_indices.size(), 4u);
    ASSERT_EQ(values.size(), 4u);
    EXPECT_EQ(col_indices[0], 0);
    EXPECT_EQ(col_indices[1], 1);
    EXPECT_EQ(col_indices[2], 0);
    EXPECT_EQ(col_indices[3], 1);

    const double alpha = penalty.getConstraints()[0].penalty;
    EXPECT_DOUBLE_EQ(values[0], alpha * 1.0 * 1.0);
    EXPECT_DOUBLE_EQ(values[1], alpha * 1.0 * -2.0);
    EXPECT_DOUBLE_EQ(values[2], alpha * -2.0 * 1.0);
    EXPECT_DOUBLE_EQ(values[3], alpha * -2.0 * -2.0);

    auto rhs = penalty.getPenaltyRhs(/*n_dofs=*/2);
    ASSERT_EQ(rhs.size(), 2u);
    EXPECT_DOUBLE_EQ(rhs[0], alpha * 3.0 * 1.0);
    EXPECT_DOUBLE_EQ(rhs[1], alpha * 3.0 * -2.0);
}

TEST(PenaltyMethodTest, ApplyPenaltyOperatorMatchesResidual) {
    PenaltyMethod penalty;
    penalty.addConstraint(std::array<GlobalIndex, 2>{{0, 1}}, std::array<double, 2>{{1.0, -2.0}}, 3.0, 1e6);

    std::vector<double> x = {7.0, 2.0};
    std::vector<double> y(2, 0.0);
    penalty.applyPenaltyOperator(x, y);

    // b^T x = 7 - 4 = 3; alpha*b*(b^T x) = 1e6 * [3, -6]
    EXPECT_DOUBLE_EQ(y[0], 3e6);
    EXPECT_DOUBLE_EQ(y[1], -6e6);

    auto residuals = penalty.computeResiduals(x);
    ASSERT_EQ(residuals.size(), 1u);
    EXPECT_NEAR(residuals[0], 0.0, 1e-12);
    EXPECT_TRUE(penalty.checkSatisfaction(x, 1e-12));
}

TEST(PenaltyMethodTest, UtilityFunctions) {
    EXPECT_DOUBLE_EQ(computeOptimalPenalty(10.0, 1e-2), 1000.0);

    std::array<double, 2> penalties{{10.0, 10.0}};
    std::array<double, 2> residuals{{1e-3, 1e-9}};
    auto adjusted = adaptPenalties(penalties, residuals, 1e-6);
    ASSERT_EQ(adjusted.size(), 2u);
    EXPECT_DOUBLE_EQ(adjusted[0], 10.0 * 1e-3 / 1e-6);
    EXPECT_DOUBLE_EQ(adjusted[1], 10.0);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
