/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_HangingNodeConstraint.cpp
 * @brief Unit tests for HangingNodeConstraint class
 */

#include <gtest/gtest.h>
#include "Constraints/HangingNodeConstraint.h"
#include "Constraints/AffineConstraints.h"

#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

// ============================================================================
// Basic tests
// ============================================================================

TEST(HangingNodeConstraintTest, DefaultConstruction) {
    HangingNodeConstraint constraint;

    EXPECT_EQ(constraint.numHangingNodes(), 0);
    EXPECT_EQ(constraint.getType(), ConstraintType::HangingNode);
}

TEST(HangingNodeConstraintTest, SimpleEdgeMidpoint) {
    // Create a single edge midpoint constraint: u_2 = 0.5*u_0 + 0.5*u_1
    HangingNodeConstraint constraint({2}, {0}, {1});

    EXPECT_EQ(constraint.numHangingNodes(), 1);

    AffineConstraints aff;
    constraint.apply(aff);
    aff.close();

    EXPECT_TRUE(aff.isConstrained(2));
    EXPECT_FALSE(aff.isConstrained(0));
    EXPECT_FALSE(aff.isConstrained(1));

    auto c = aff.getConstraint(2);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->entries.size(), 2);

    // Check weights sum to 1
    double sum = 0.0;
    for (const auto& e : c->entries) {
        sum += e.weight;
        EXPECT_DOUBLE_EQ(e.weight, 0.5);
    }
    EXPECT_DOUBLE_EQ(sum, 1.0);
}

TEST(HangingNodeConstraintTest, MultipleHangingNodes) {
    std::vector<HangingNodeData> nodes;

    // Node 0: hanging at DOF 10, parents 0 and 1
    HangingNodeData n1;
    n1.hanging_dof = 10;
    n1.parent_dofs = {0, 1};
    n1.weights = {0.5, 0.5};
    n1.dimension = 1;
    nodes.push_back(n1);

    // Node 1: hanging at DOF 11, parents 2 and 3
    HangingNodeData n2;
    n2.hanging_dof = 11;
    n2.parent_dofs = {2, 3};
    n2.weights = {0.5, 0.5};
    n2.dimension = 1;
    nodes.push_back(n2);

    HangingNodeConstraint constraint(std::move(nodes));

    EXPECT_EQ(constraint.numHangingNodes(), 2);

    AffineConstraints aff;
    constraint.apply(aff);
    aff.close();

    EXPECT_TRUE(aff.isConstrained(10));
    EXPECT_TRUE(aff.isConstrained(11));
    EXPECT_EQ(aff.numConstraints(), 2);
}

// ============================================================================
// Factory method tests
// ============================================================================

TEST(HangingNodeConstraintTest, P1EdgesFactory) {
    std::vector<GlobalIndex> hanging = {5, 6, 7};
    std::vector<std::array<GlobalIndex, 2>> edges = {
        {{0, 1}},  // Edge 0-1, midpoint at DOF 5
        {{1, 2}},  // Edge 1-2, midpoint at DOF 6
        {{2, 0}}   // Edge 2-0, midpoint at DOF 7
    };

    HangingNodeConstraint constraint = HangingNodeConstraint::forP1Edges(hanging, edges);

    EXPECT_EQ(constraint.numHangingNodes(), 3);

    AffineConstraints aff;
    constraint.apply(aff);
    aff.close();

    // Check constraint for DOF 5: should be 0.5*u_0 + 0.5*u_1
    auto c5 = aff.getConstraint(5);
    ASSERT_TRUE(c5.has_value());

    bool has_0 = false, has_1 = false;
    for (const auto& e : c5->entries) {
        if (e.master_dof == 0) {
            has_0 = true;
            EXPECT_DOUBLE_EQ(e.weight, 0.5);
        }
        if (e.master_dof == 1) {
            has_1 = true;
            EXPECT_DOUBLE_EQ(e.weight, 0.5);
        }
    }
    EXPECT_TRUE(has_0);
    EXPECT_TRUE(has_1);
}

TEST(HangingNodeConstraintTest, AddHangingNode1D) {
    HangingNodeConstraint constraint;

    constraint.addHangingNode1D(10, 0, 1);
    constraint.addHangingNode1D(11, 2, 3);

    EXPECT_EQ(constraint.numHangingNodes(), 2);

    const auto& nodes = constraint.getHangingNodes();
    EXPECT_EQ(nodes[0].hanging_dof, 10);
    EXPECT_EQ(nodes[0].parent_dofs.size(), 2);
    EXPECT_DOUBLE_EQ(nodes[0].weights[0], 0.5);
    EXPECT_DOUBLE_EQ(nodes[0].weights[1], 0.5);
}

// ============================================================================
// Validation tests
// ============================================================================

TEST(HangingNodeConstraintTest, ValidateWeights) {
    std::vector<HangingNodeData> nodes;

    HangingNodeData n;
    n.hanging_dof = 5;
    n.parent_dofs = {0, 1};
    n.weights = {0.5, 0.5};  // Sum = 1, valid
    nodes.push_back(n);

    HangingNodeConstraintOptions opts;
    opts.validate_weights = true;

    HangingNodeConstraint constraint(std::move(nodes), opts);

    std::string error = constraint.validate();
    EXPECT_TRUE(error.empty());
}

TEST(HangingNodeConstraintTest, ValidateWeightsFailure) {
    std::vector<HangingNodeData> nodes;

    HangingNodeData n;
    n.hanging_dof = 5;
    n.parent_dofs = {0, 1};
    n.weights = {0.3, 0.3};  // Sum = 0.6, invalid!
    nodes.push_back(n);

    HangingNodeConstraintOptions opts;
    opts.validate_weights = true;

    HangingNodeConstraint constraint(std::move(nodes), opts);

    std::string error = constraint.validate();
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(error.find("sum") != std::string::npos ||
                error.find("1.0") != std::string::npos);
}

TEST(HangingNodeConstraintTest, ValidateDuplicateHanging) {
    std::vector<HangingNodeData> nodes;

    HangingNodeData n1, n2;
    n1.hanging_dof = 5;
    n1.parent_dofs = {0, 1};
    n1.weights = {0.5, 0.5};

    n2.hanging_dof = 5;  // Duplicate!
    n2.parent_dofs = {2, 3};
    n2.weights = {0.5, 0.5};

    nodes.push_back(n1);
    nodes.push_back(n2);

    HangingNodeConstraint constraint(std::move(nodes));

    std::string error = constraint.validate();
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(error.find("Duplicate") != std::string::npos);
}

// ============================================================================
// Weight computation utility tests
// ============================================================================

TEST(HangingNodeConstraintTest, ComputeP1EdgeWeights) {
    auto weights = computeP1EdgeWeights();
    EXPECT_EQ(weights.size(), 2);
    EXPECT_DOUBLE_EQ(weights[0], 0.5);
    EXPECT_DOUBLE_EQ(weights[1], 0.5);
}

TEST(HangingNodeConstraintTest, ComputeP2EdgeWeights) {
    // At midpoint t=0.5
    auto weights = computeP2EdgeWeights(0.5);
    EXPECT_EQ(weights.size(), 3);

    // Sum should be 1
    double sum = weights[0] + weights[1] + weights[2];
    EXPECT_NEAR(sum, 1.0, 1e-14);

    // At midpoint, the middle node should have value 1, endpoints 0
    // N0(0.5) = 0.5 * 0 = 0
    // N1(0.5) = 4 * 0.5 * 0.5 = 1
    // N2(0.5) = 0.5 * 0 = 0
    EXPECT_NEAR(weights[0], 0.0, 1e-14);
    EXPECT_NEAR(weights[1], 1.0, 1e-14);
    EXPECT_NEAR(weights[2], 0.0, 1e-14);
}

TEST(HangingNodeConstraintTest, ComputeQ1FaceWeights) {
    // At center (0, 0)
    auto weights = computeQ1FaceWeights(0.0, 0.0);
    EXPECT_EQ(weights.size(), 4);

    // All should be equal at center
    for (double w : weights) {
        EXPECT_DOUBLE_EQ(w, 0.25);
    }

    // Sum should be 1
    double sum = 0.0;
    for (double w : weights) sum += w;
    EXPECT_DOUBLE_EQ(sum, 1.0);
}

TEST(HangingNodeConstraintTest, ComputeQ1VolumeWeights) {
    // At center (0, 0, 0)
    auto weights = computeQ1VolumeWeights(0.0, 0.0, 0.0);
    EXPECT_EQ(weights.size(), 8);

    // All should be equal at center
    for (double w : weights) {
        EXPECT_DOUBLE_EQ(w, 0.125);
    }

    // Sum should be 1
    double sum = 0.0;
    for (double w : weights) sum += w;
    EXPECT_DOUBLE_EQ(sum, 1.0);
}

// ============================================================================
// Clone and copy tests
// ============================================================================

TEST(HangingNodeConstraintTest, Clone) {
    HangingNodeConstraint original;
    original.addHangingNode1D(10, 0, 1);

    auto clone = original.clone();

    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getType(), ConstraintType::HangingNode);

    auto* typed_clone = dynamic_cast<HangingNodeConstraint*>(clone.get());
    ASSERT_NE(typed_clone, nullptr);
    EXPECT_EQ(typed_clone->numHangingNodes(), 1);
}

TEST(HangingNodeConstraintTest, Clear) {
    HangingNodeConstraint constraint;
    constraint.addHangingNode1D(10, 0, 1);
    constraint.addHangingNode1D(11, 2, 3);

    EXPECT_EQ(constraint.numHangingNodes(), 2);

    constraint.clear();

    EXPECT_EQ(constraint.numHangingNodes(), 0);
}

}  // namespace test
}  // namespace constraints
}  // namespace FE
}  // namespace svmp
