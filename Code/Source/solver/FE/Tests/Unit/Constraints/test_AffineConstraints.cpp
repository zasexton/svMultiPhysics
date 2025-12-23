/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_AffineConstraints.cpp
 * @brief Unit tests for AffineConstraints class
 */

#include <gtest/gtest.h>
#include "Constraints/AffineConstraints.h"

#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

// ============================================================================
// Basic construction tests
// ============================================================================

TEST(AffineConstraintsTest, DefaultConstruction) {
    AffineConstraints constraints;

    EXPECT_EQ(constraints.numConstraints(), 0);
    EXPECT_FALSE(constraints.isClosed());
    EXPECT_TRUE(constraints.empty());
}

TEST(AffineConstraintsTest, AddSingleConstraint) {
    AffineConstraints constraints;

    // Add constraint: u_5 = 2.0 * u_10 + 0.5
    constraints.addLine(5);
    constraints.addEntry(5, 10, 2.0);
    constraints.setInhomogeneity(5, 0.5);

    // Before close(), can check if constrained
    EXPECT_TRUE(constraints.isConstrained(5));
    EXPECT_FALSE(constraints.isConstrained(10));

    // numConstraints() is accurate after close()
    constraints.close();
    EXPECT_EQ(constraints.numConstraints(), 1);
}

TEST(AffineConstraintsTest, CloseSingleConstraint) {
    AffineConstraints constraints;

    constraints.addLine(5);
    constraints.addEntry(5, 10, 2.0);
    constraints.setInhomogeneity(5, 0.5);
    constraints.close();

    EXPECT_TRUE(constraints.isClosed());
    EXPECT_TRUE(constraints.isConstrained(5));

    auto c = constraints.getConstraint(5);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->entries.size(), 1);
    EXPECT_EQ(c->entries[0].master_dof, 10);
    EXPECT_DOUBLE_EQ(c->entries[0].weight, 2.0);
    EXPECT_DOUBLE_EQ(c->inhomogeneity, 0.5);
}

// ============================================================================
// Transitive closure tests
// ============================================================================

TEST(AffineConstraintsTest, TransitiveClosureSimple) {
    AffineConstraints constraints;

    // Chain: u_0 = u_1, u_1 = u_2
    // After closure: u_0 = u_2, u_1 = u_2
    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);

    constraints.addLine(1);
    constraints.addEntry(1, 2, 1.0);

    constraints.close();

    // u_0 should now depend on u_2
    auto c0 = constraints.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_EQ(c0->entries.size(), 1);
    EXPECT_EQ(c0->entries[0].master_dof, 2);
    EXPECT_DOUBLE_EQ(c0->entries[0].weight, 1.0);

    // u_1 should depend on u_2
    auto c1 = constraints.getConstraint(1);
    ASSERT_TRUE(c1.has_value());
    EXPECT_EQ(c1->entries.size(), 1);
    EXPECT_EQ(c1->entries[0].master_dof, 2);
}

TEST(AffineConstraintsTest, TransitiveClosureWithWeights) {
    AffineConstraints constraints;

    // u_0 = 2 * u_1
    // u_1 = 0.5 * u_2
    // After closure: u_0 = 2 * 0.5 * u_2 = u_2
    constraints.addLine(0);
    constraints.addEntry(0, 1, 2.0);

    constraints.addLine(1);
    constraints.addEntry(1, 2, 0.5);

    constraints.close();

    auto c0 = constraints.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_EQ(c0->entries.size(), 1);
    EXPECT_EQ(c0->entries[0].master_dof, 2);
    EXPECT_DOUBLE_EQ(c0->entries[0].weight, 1.0);  // 2.0 * 0.5
}

TEST(AffineConstraintsTest, TransitiveClosureWithInhomogeneity) {
    AffineConstraints constraints;

    // u_0 = u_1 + 1
    // u_1 = u_2 + 2
    // After closure: u_0 = u_2 + 3
    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.setInhomogeneity(0, 1.0);

    constraints.addLine(1);
    constraints.addEntry(1, 2, 1.0);
    constraints.setInhomogeneity(1, 2.0);

    constraints.close();

    auto c0 = constraints.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_EQ(c0->entries.size(), 1);
    EXPECT_EQ(c0->entries[0].master_dof, 2);
    EXPECT_DOUBLE_EQ(c0->inhomogeneity, 3.0);  // 1.0 + 2.0
}

TEST(AffineConstraintsTest, TransitiveClosureMultipleMasters) {
    AffineConstraints constraints;

    // u_0 = 0.5 * u_1 + 0.5 * u_2  (hanging node)
    // u_1 = u_3
    // After closure: u_0 = 0.5 * u_3 + 0.5 * u_2
    constraints.addLine(0);
    constraints.addEntry(0, 1, 0.5);
    constraints.addEntry(0, 2, 0.5);

    constraints.addLine(1);
    constraints.addEntry(1, 3, 1.0);

    constraints.close();

    auto c0 = constraints.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_EQ(c0->entries.size(), 2);

    // Check that we have u_3 and u_2 as masters
    double sum_weights = 0.0;
    bool has_2 = false, has_3 = false;
    for (const auto& e : c0->entries) {
        sum_weights += e.weight;
        if (e.master_dof == 2) {
            has_2 = true;
            EXPECT_DOUBLE_EQ(e.weight, 0.5);
        }
        if (e.master_dof == 3) {
            has_3 = true;
            EXPECT_DOUBLE_EQ(e.weight, 0.5);
        }
    }
    EXPECT_TRUE(has_2);
    EXPECT_TRUE(has_3);
    EXPECT_DOUBLE_EQ(sum_weights, 1.0);
}

// ============================================================================
// Vector distribution tests
// ============================================================================

TEST(AffineConstraintsTest, DistributeVector) {
    AffineConstraints constraints;

    // u_0 = 0.5 * u_1 + 0.5 * u_2
    constraints.addLine(0);
    constraints.addEntry(0, 1, 0.5);
    constraints.addEntry(0, 2, 0.5);
    constraints.close();

    std::vector<double> vec = {100.0, 2.0, 4.0};  // u_0 will be overwritten
    constraints.distribute(vec.data(), static_cast<GlobalIndex>(vec.size()));

    // u_0 should become 0.5 * 2 + 0.5 * 4 = 3
    EXPECT_DOUBLE_EQ(vec[0], 3.0);
    EXPECT_DOUBLE_EQ(vec[1], 2.0);  // Unchanged
    EXPECT_DOUBLE_EQ(vec[2], 4.0);  // Unchanged
}

TEST(AffineConstraintsTest, DistributeVectorWithInhomogeneity) {
    AffineConstraints constraints;

    // u_0 = u_1 + 10.0 (Dirichlet-like)
    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.setInhomogeneity(0, 10.0);
    constraints.close();

    std::vector<double> vec = {0.0, 5.0};
    constraints.distribute(vec.data(), static_cast<GlobalIndex>(vec.size()));

    EXPECT_DOUBLE_EQ(vec[0], 15.0);  // 5.0 + 10.0
    EXPECT_DOUBLE_EQ(vec[1], 5.0);
}

// ============================================================================
// Clear and reinitialize tests
// ============================================================================

TEST(AffineConstraintsTest, ClearConstraints) {
    AffineConstraints constraints;

    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.close();

    EXPECT_EQ(constraints.numConstraints(), 1);

    constraints.clear();

    EXPECT_EQ(constraints.numConstraints(), 0);
    EXPECT_FALSE(constraints.isClosed());
    EXPECT_FALSE(constraints.isConstrained(0));
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(AffineConstraintsTest, DirichletConstraint) {
    AffineConstraints constraints;

    // Pure Dirichlet: u_0 = 5.0 (no masters)
    constraints.addLine(0);
    constraints.setInhomogeneity(0, 5.0);
    constraints.close();

    auto c = constraints.getConstraint(0);
    ASSERT_TRUE(c.has_value());
    EXPECT_TRUE(c->entries.empty());
    EXPECT_DOUBLE_EQ(c->inhomogeneity, 5.0);
}

TEST(AffineConstraintsTest, MergeToSameMaster) {
    AffineConstraints constraints;

    // u_0 = 0.25*u_1 + 0.25*u_1 + 0.5*u_2
    // Should merge to: u_0 = 0.5*u_1 + 0.5*u_2
    constraints.addLine(0);
    constraints.addEntry(0, 1, 0.25);
    constraints.addEntry(0, 1, 0.25);
    constraints.addEntry(0, 2, 0.5);
    constraints.close();

    auto c = constraints.getConstraint(0);
    ASSERT_TRUE(c.has_value());

    // Find u_1 entry and check weight
    for (const auto& e : c->entries) {
        if (e.master_dof == 1) {
            EXPECT_DOUBLE_EQ(e.weight, 0.5);
        }
    }
}

TEST(AffineConstraintsTest, GetConstrainedDofs) {
    AffineConstraints constraints;

    constraints.addLine(5);
    constraints.addEntry(5, 10, 1.0);

    constraints.addLine(3);
    constraints.addEntry(3, 10, 1.0);

    constraints.close();

    auto dofs = constraints.getConstrainedDofs();
    EXPECT_EQ(dofs.size(), 2);

    // Should be sorted
    EXPECT_EQ(dofs[0], 3);
    EXPECT_EQ(dofs[1], 5);
}

// ============================================================================
// Determinism tests
// ============================================================================

TEST(AffineConstraintsTest, DeterministicOrdering) {
    // Build constraints twice in different orders and verify same result
    auto build1 = []() {
        AffineConstraints c;
        c.addLine(10);
        c.addEntry(10, 20, 0.3);
        c.addEntry(10, 30, 0.7);
        c.addLine(5);
        c.addEntry(5, 15, 0.5);
        c.addEntry(5, 25, 0.5);
        c.close();
        return c;
    };

    auto build2 = []() {
        AffineConstraints c;
        c.addLine(5);
        c.addEntry(5, 25, 0.5);
        c.addEntry(5, 15, 0.5);
        c.addLine(10);
        c.addEntry(10, 30, 0.7);
        c.addEntry(10, 20, 0.3);
        c.close();
        return c;
    };

    auto c1 = build1();
    auto c2 = build2();

    // Same constrained DOFs
    auto dofs1 = c1.getConstrainedDofs();
    auto dofs2 = c2.getConstrainedDofs();
    ASSERT_EQ(dofs1.size(), dofs2.size());

    for (std::size_t i = 0; i < dofs1.size(); ++i) {
        EXPECT_EQ(dofs1[i], dofs2[i]);
    }

    // Same constraint data
    for (GlobalIndex dof : dofs1) {
        auto cn1 = c1.getConstraint(dof);
        auto cn2 = c2.getConstraint(dof);

        ASSERT_TRUE(cn1.has_value());
        ASSERT_TRUE(cn2.has_value());
        EXPECT_DOUBLE_EQ(cn1->inhomogeneity, cn2->inhomogeneity);
        ASSERT_EQ(cn1->entries.size(), cn2->entries.size());

        // Note: entry ordering within a constraint may differ, but
        // the set of (master, weight) pairs should be the same
    }
}

}  // namespace test
}  // namespace constraints
}  // namespace FE
}  // namespace svmp
