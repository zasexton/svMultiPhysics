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

TEST(GlobalConstraintTest, WeightedMeanUsesRelativeMeasureAtEveryScale)
{
    const std::vector<GlobalIndex> dofs{0, 1};
    const double target = 0x1p-80;
    const auto pin = GlobalConstraint::pinDof(0, target);
    EXPECT_DOUBLE_EQ(pin.getTargetValue(), target);
    EXPECT_FALSE(pin.getInfo().is_homogeneous);
    GlobalConstraintOptions opts;
    opts.strategy = GlobalConstraintStrategy::WeightedMean;
    for (double measure : {1.0, 0x1p-300, 0x1p-1022, 0x1p1022}) {
        const std::vector<double> weights{measure, 3*measure};
        const std::vector<double> values{5., 9.};
        GlobalConstraint mean(dofs, weights, target, opts);
        EXPECT_TRUE(mean.checkSatisfaction(std::vector<double>{target, target}));
        EXPECT_FALSE(mean.checkSatisfaction(std::vector<double>{0., 0.}));
        auto absolute = opts;
        absolute.tolerance = 2*target;
        EXPECT_TRUE(GlobalConstraint(dofs, weights, target, absolute).checkSatisfaction(std::vector<double>{0., 0.}));
        EXPECT_FALSE(mean.getInfo().is_homogeneous);
        EXPECT_DOUBLE_EQ(computeWeightedMean(values, dofs, weights), 8.);
        EXPECT_DOUBLE_EQ(mean.getWeights()[0], 0.25);
        EXPECT_DOUBLE_EQ(mean.getNullspaceVector()[1], 0.75);
        AffineConstraints c;
        ASSERT_NO_THROW(mean.apply(c));
        c.close();
        const auto line = c.getConstraint(0);
        ASSERT_TRUE(line);
        EXPECT_DOUBLE_EQ(line->inhomogeneity, 4*target);
        ASSERT_EQ(line->entries.size(), 1u);
        EXPECT_DOUBLE_EQ(line->entries[0].weight, -3.);
    }
}

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

    const std::vector<double> vec = {3.0, 6.0, 9.0};
    EXPECT_DOUBLE_EQ(gc.computeResidual(vec), 6.);
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

    ASSERT_EQ(c->entries.size(), 1u);
    EXPECT_EQ(c->entries[0].master_dof, 1);
    EXPECT_DOUBLE_EQ(c->entries[0].weight, -2.);
    EXPECT_DOUBLE_EQ(c->inhomogeneity, 30.);
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
