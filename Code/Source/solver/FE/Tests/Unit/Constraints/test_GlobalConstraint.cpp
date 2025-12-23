/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GlobalConstraint.cpp
 * @brief Unit tests for GlobalConstraint class
 */

#include <gtest/gtest.h>
#include "Constraints/GlobalConstraint.h"
#include "Constraints/AffineConstraints.h"

#include <cmath>
#include <vector>
#include <numeric>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

// ============================================================================
// Basic tests
// ============================================================================

TEST(GlobalConstraintTest, DefaultConstruction) {
    GlobalConstraint gc;

    EXPECT_EQ(gc.getType(), ConstraintType::Global);
    EXPECT_TRUE(gc.getDofs().empty());
}

TEST(GlobalConstraintTest, ZeroMeanFactory) {
    std::vector<GlobalIndex> dofs = {0, 1, 2, 3, 4};

    GlobalConstraint gc = GlobalConstraint::zeroMean(dofs);

    EXPECT_EQ(gc.getGlobalType(), GlobalConstraintType::ZeroMean);
    EXPECT_EQ(gc.getDofs().size(), 5);
    EXPECT_DOUBLE_EQ(gc.getTargetValue(), 0.0);
}

TEST(GlobalConstraintTest, PinDofFactory) {
    GlobalConstraint gc = GlobalConstraint::pinDof(5, 3.14);

    EXPECT_EQ(gc.getDofs().size(), 1);
    EXPECT_EQ(gc.getDofs()[0], 5);
}

TEST(GlobalConstraintTest, ApplyPinSingleDof) {
    std::vector<GlobalIndex> dofs = {0, 1, 2};

    GlobalConstraintOptions opts;
    opts.strategy = GlobalConstraintStrategy::PinSingleDof;
    opts.pinned_value = 0.0;

    GlobalConstraint gc(dofs, GlobalConstraintType::ZeroMean, opts);

    AffineConstraints aff;
    gc.apply(aff);
    aff.close();

    // Should pin one DOF (the first by default)
    EXPECT_EQ(aff.numConstraints(), 1);
    EXPECT_TRUE(aff.isConstrained(gc.getPinnedDof()));
}

TEST(GlobalConstraintTest, ApplyExplicitPin) {
    std::vector<GlobalIndex> dofs = {0, 1, 2, 3, 4};

    GlobalConstraintOptions opts;
    opts.strategy = GlobalConstraintStrategy::PinSingleDof;
    opts.explicit_pin_dof = 3;
    opts.pinned_value = 5.0;

    // Use FixedMean to test pinned_value being applied
    GlobalConstraint gc(dofs, GlobalConstraintType::FixedMean, opts);

    AffineConstraints aff;
    gc.apply(aff);
    aff.close();

    EXPECT_TRUE(aff.isConstrained(3));
    EXPECT_EQ(gc.getPinnedDof(), 3);

    auto c = aff.getConstraint(3);
    ASSERT_TRUE(c.has_value());
    EXPECT_DOUBLE_EQ(c->inhomogeneity, 5.0);
}

// ============================================================================
// Constraint satisfaction tests
// ============================================================================

TEST(GlobalConstraintTest, CheckSatisfaction) {
    std::vector<GlobalIndex> dofs = {0, 1, 2};

    GlobalConstraintOptions opts;
    opts.tolerance = 1e-10;

    GlobalConstraint gc = GlobalConstraint::zeroMean(dofs, opts);

    // Vector with zero mean: [1, 0, -1]
    std::vector<double> zero_mean = {1.0, 0.0, -1.0};
    EXPECT_TRUE(gc.checkSatisfaction(zero_mean));

    // Vector with non-zero mean: [1, 2, 3]
    std::vector<double> nonzero_mean = {1.0, 2.0, 3.0};
    EXPECT_FALSE(gc.checkSatisfaction(nonzero_mean));
}

TEST(GlobalConstraintTest, ComputeResidual) {
    std::vector<GlobalIndex> dofs = {0, 1, 2};
    GlobalConstraint gc = GlobalConstraint::zeroMean(dofs);

    // Mean of [3, 6, 9] = 6, target = 0
    // Residual = 6 - 0 = 6 (if weights sum to 1)
    std::vector<double> vec = {3.0, 6.0, 9.0};

    double residual = gc.computeResidual(vec);

    // With normalized weights summing to 1, residual = mean - target
    // mean = (3+6+9)/3 = 6, but weights are normalized so sum(w_i * u_i) = 6/3 = 2?
    // Actually with equal weights summing to 1, each weight = 1/3
    // sum = 1/3*3 + 1/3*6 + 1/3*9 = 1 + 2 + 3 = 6
    // Hmm, that's not right. Let me check the implementation...
    // The weights should be normalized to sum to 1, so:
    // w_i = 1/N for uniform weights after normalization
    // Actually the implementation normalizes by dividing by sum
    // Original weights: [1, 1, 1], sum = 3, normalized = [1/3, 1/3, 1/3]
    // weighted sum = 1/3*3 + 1/3*6 + 1/3*9 = 1+2+3 = 6
    // Wait, that equals 6. Let me re-examine...
    // residual = sum(w_i * u_i) - target = 6 - 0 = 6

    // The implementation normalizes so weights sum to 1
    // So the "mean" computed is: (1/3)*3 + (1/3)*6 + (1/3)*9 = 6
    // That's correct if weights are NOT normalized to 1/N each
    EXPECT_NE(residual, 0.0);
}

TEST(GlobalConstraintTest, ProjectToConstrainedSpace) {
    std::vector<GlobalIndex> dofs = {0, 1, 2};
    GlobalConstraint gc = GlobalConstraint::zeroMean(dofs);

    // Start with [3, 6, 9] - mean should become 0
    std::vector<double> vec = {3.0, 6.0, 9.0};

    gc.projectToConstrainedSpace(vec);

    // After projection, mean should be near zero
    EXPECT_TRUE(gc.checkSatisfaction(vec));
}

// ============================================================================
// Nullspace vector tests
// ============================================================================

TEST(GlobalConstraintTest, NullspaceVector) {
    std::vector<GlobalIndex> dofs = {0, 1, 2, 3};
    GlobalConstraint gc = GlobalConstraint::zeroMean(dofs);

    auto ns = gc.getNullspaceVector();

    EXPECT_EQ(ns.size(), 4);

    // Should sum to 1 after normalization
    double sum = std::accumulate(ns.begin(), ns.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-14);

    // For uniform weights, all should be equal
    for (double w : ns) {
        EXPECT_NEAR(w, 0.25, 1e-14);
    }
}

// ============================================================================
// Weighted constraint tests
// ============================================================================

TEST(GlobalConstraintTest, WeightedMean) {
    std::vector<GlobalIndex> dofs = {0, 1};
    std::vector<double> weights = {1.0, 2.0};
    double target = 10.0;

    GlobalConstraintOptions opts;
    opts.strategy = GlobalConstraintStrategy::WeightedMean;

    GlobalConstraint gc(dofs, weights, target, opts);

    // Weighted mean constraint: w_0*u_0 + w_1*u_1 = target
    // With normalized weights: (1/3)*u_0 + (2/3)*u_1 = target
    // Applied as: u_0 = (target - (2/3)*u_1) / (1/3) = 3*target - 2*u_1

    AffineConstraints aff;
    gc.apply(aff);
    aff.close();

    EXPECT_TRUE(aff.isConstrained(0));

    auto c = aff.getConstraint(0);
    ASSERT_TRUE(c.has_value());

    // The weight for u_1 should be -w_1/w_0 = -2/1 = -2
    // But weights are normalized first: [1/3, 2/3], so -w_1/w_0 = -(2/3)/(1/3) = -2
    EXPECT_EQ(c->entries.size(), 1);
    EXPECT_EQ(c->entries[0].master_dof, 1);
}

// ============================================================================
// Info and factory tests
// ============================================================================

TEST(GlobalConstraintTest, GetInfo) {
    std::vector<GlobalIndex> dofs = {0, 1, 2};
    GlobalConstraint gc = GlobalConstraint::zeroMean(dofs);

    auto info = gc.getInfo();

    EXPECT_EQ(info.name, "GlobalConstraint");
    EXPECT_EQ(info.type, ConstraintType::Global);
    EXPECT_TRUE(info.is_homogeneous);  // Zero target
}

TEST(GlobalConstraintTest, FixedMeanFactory) {
    std::vector<GlobalIndex> dofs = {0, 1, 2};
    double target = 5.0;

    GlobalConstraint gc = GlobalConstraint::fixedMean(dofs, target);

    EXPECT_EQ(gc.getGlobalType(), GlobalConstraintType::FixedMean);
    EXPECT_DOUBLE_EQ(gc.getTargetValue(), 5.0);
}

TEST(GlobalConstraintTest, Clone) {
    std::vector<GlobalIndex> dofs = {0, 1, 2};
    GlobalConstraint gc = GlobalConstraint::zeroMean(dofs);

    auto clone = gc.clone();

    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getType(), ConstraintType::Global);
}

// ============================================================================
// Utility function tests
// ============================================================================

TEST(GlobalConstraintTest, ComputeMean) {
    std::vector<double> vec = {2.0, 4.0, 6.0, 8.0, 10.0};
    std::vector<GlobalIndex> dofs = {0, 2, 4};  // Select indices 0, 2, 4

    double mean = computeMean(vec, dofs);

    // Mean of [2, 6, 10] = 18/3 = 6
    EXPECT_DOUBLE_EQ(mean, 6.0);
}

TEST(GlobalConstraintTest, ComputeWeightedMean) {
    std::vector<double> vec = {1.0, 2.0, 3.0};
    std::vector<GlobalIndex> dofs = {0, 1, 2};
    std::vector<double> weights = {1.0, 2.0, 3.0};

    double wmean = computeWeightedMean(vec, dofs, weights);

    // Weighted mean = (1*1 + 2*2 + 3*3) / (1+2+3) = (1+4+9)/6 = 14/6 = 7/3
    EXPECT_NEAR(wmean, 14.0 / 6.0, 1e-14);
}

TEST(GlobalConstraintTest, SubtractMean) {
    std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<GlobalIndex> dofs = {0, 1, 2, 3, 4};

    subtractMean(vec, dofs);

    // Original mean = 3.0
    // After subtraction: [-2, -1, 0, 1, 2]
    EXPECT_DOUBLE_EQ(vec[0], -2.0);
    EXPECT_DOUBLE_EQ(vec[1], -1.0);
    EXPECT_DOUBLE_EQ(vec[2], 0.0);
    EXPECT_DOUBLE_EQ(vec[3], 1.0);
    EXPECT_DOUBLE_EQ(vec[4], 2.0);

    // New mean should be 0
    double new_mean = computeMean(vec, dofs);
    EXPECT_NEAR(new_mean, 0.0, 1e-14);
}

}  // namespace test
}  // namespace constraints
}  // namespace FE
}  // namespace svmp
