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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
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

TEST(AffineConstraintsTest, ConstraintLayoutRevisionTracksStructuralChanges) {
    AffineConstraints constraints;
    const auto initial = constraints.constraintLayoutRevision();

    constraints.addLine(5);
    EXPECT_GT(constraints.constraintLayoutRevision(), initial);
    const auto after_line = constraints.constraintLayoutRevision();

    constraints.addEntry(5, 10, 2.0);
    EXPECT_GT(constraints.constraintLayoutRevision(), after_line);
    const auto after_entry = constraints.constraintLayoutRevision();

    constraints.setInhomogeneity(5, 0.5);
    EXPECT_EQ(constraints.constraintLayoutRevision(), after_entry);

    constraints.close();
    EXPECT_GT(constraints.constraintLayoutRevision(), after_entry);
    const auto after_close = constraints.constraintLayoutRevision();

    constraints.updateInhomogeneity(5, 1.0);
    EXPECT_EQ(constraints.constraintLayoutRevision(), after_close);
}

TEST(AffineConstraintsTest, SemanticFingerprintTracksEveryConstraintValue) {
    const auto make_constraints = [](GlobalIndex slave,
                                     GlobalIndex master,
                                     double weight,
                                     double inhomogeneity) {
        AffineConstraints constraints;
        constraints.addLine(slave);
        constraints.addEntry(slave, master, weight);
        constraints.setInhomogeneity(slave, inhomogeneity);
        constraints.close();
        return constraints;
    };

    const auto reference = make_constraints(5, 10, 2.0, 0.5);
    const auto identical = make_constraints(5, 10, 2.0, 0.5);
    EXPECT_EQ(
        constraintSemanticFingerprint(reference),
        constraintSemanticFingerprint(identical));

    EXPECT_NE(
        constraintSemanticFingerprint(reference),
        constraintSemanticFingerprint(
            make_constraints(6, 10, 2.0, 0.5)));
    EXPECT_NE(
        constraintSemanticFingerprint(reference),
        constraintSemanticFingerprint(
            make_constraints(5, 11, 2.0, 0.5)));
    EXPECT_NE(
        constraintSemanticFingerprint(reference),
        constraintSemanticFingerprint(
            make_constraints(5, 10, 3.0, 0.5)));
    EXPECT_NE(
        constraintSemanticFingerprint(reference),
        constraintSemanticFingerprint(
            make_constraints(5, 10, 2.0, 0.75)));
}

TEST(AffineConstraintsTest, AddDirichletDuplicateSameValueIsIdempotent) {
    AffineConstraints constraints;

    EXPECT_NO_THROW(constraints.addDirichlet(5, 1.0));
    EXPECT_NO_THROW(constraints.addDirichlet(5, 1.0));

    constraints.close();
    auto c = constraints.getConstraint(5);
    ASSERT_TRUE(c.has_value());
    EXPECT_TRUE(c->entries.empty());
    EXPECT_DOUBLE_EQ(c->inhomogeneity, 1.0);
}

TEST(AffineConstraintsTest, AddDirichletDuplicateDifferentValueThrows) {
    AffineConstraints constraints;
    constraints.addDirichlet(5, 1.0);

    EXPECT_THROW(constraints.addDirichlet(5, 2.0), ConstraintException);
}

TEST(AffineConstraintsTest, AddDirichletOverridesMasterBearingLine) {
    AffineConstraints constraints;

    // Create a non-Dirichlet (master-bearing) constraint line first.
    constraints.addLine(5);
    constraints.addEntry(5, 10, 1.0);

    // Strong Dirichlet data wins over derived master-bearing lines
    // (small-cut aggregation, hanging nodes): interface-tracking constraints
    // cannot reliably pre-exclude every strong-BC dof, so the pin replaces
    // the line instead of throwing.
    constraints.addDirichlet(5, 3.0);
    constraints.close();
    const auto line = constraints.getConstraint(5);
    ASSERT_TRUE(line.has_value());
    EXPECT_TRUE(line->isDirichlet());
    EXPECT_DOUBLE_EQ(line->inhomogeneity, 3.0);
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

TEST(AffineConstraintsTest, DistributeVectorHomogeneousIgnoresInhomogeneity) {
    AffineConstraints constraints;

    // u_0 = u_1 + 10.0
    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.setInhomogeneity(0, 10.0);
    constraints.close();

    std::vector<double> vec = {0.0, 5.0};
    constraints.distributeHomogeneous(vec.data(), static_cast<GlobalIndex>(vec.size()));

    EXPECT_DOUBLE_EQ(vec[0], 5.0);
    EXPECT_DOUBLE_EQ(vec[1], 5.0);
}

TEST(AffineConstraintsTest, DistributeVectorHomogeneousDirichletIsZero) {
    AffineConstraints constraints;

    // u_0 = 10.0 (Dirichlet)
    constraints.addLine(0);
    constraints.setInhomogeneity(0, 10.0);
    constraints.close();

    std::vector<double> vec = {123.0};
    constraints.distributeHomogeneous(vec.data(), static_cast<GlobalIndex>(vec.size()));

    EXPECT_DOUBLE_EQ(vec[0], 0.0);
}

TEST(AffineConstraintsTest, HasMasterBearingLines) {
    AffineConstraints empty_constraints;
    empty_constraints.close();
    EXPECT_FALSE(empty_constraints.hasMasterBearingLines());

    AffineConstraints dirichlet_only;
    dirichlet_only.addLine(0);
    dirichlet_only.setInhomogeneity(0, 10.0);
    dirichlet_only.close();
    EXPECT_FALSE(dirichlet_only.hasMasterBearingLines());

    AffineConstraints mixed;
    mixed.addLine(0);
    mixed.setInhomogeneity(0, 10.0);
    mixed.addLine(3);
    mixed.addEntry(3, 1, 0.5);
    mixed.addEntry(3, 2, 0.5);
    mixed.close();
    EXPECT_TRUE(mixed.hasMasterBearingLines());
}

TEST(AffineConstraintsTest, DistributeMasterBearingSkipsDirichletLines) {
    AffineConstraints constraints;

    // u_0 = u_1 + 10.0 (master-bearing MPC with inhomogeneity)
    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.setInhomogeneity(0, 10.0);
    // u_2 = 7.0 (plain Dirichlet, no masters)
    constraints.addLine(2);
    constraints.setInhomogeneity(2, 7.0);
    constraints.close();

    std::vector<double> vec = {0.0, 5.0, 123.0};
    constraints.distributeMasterBearing(vec);

    EXPECT_DOUBLE_EQ(vec[0], 15.0);   // MPC line applied: 5.0 + 10.0
    EXPECT_DOUBLE_EQ(vec[1], 5.0);    // master untouched
    EXPECT_DOUBLE_EQ(vec[2], 123.0);  // Dirichlet line untouched
}

TEST(AffineConstraintsTest, DistributeMasterBearingHomogeneousSkipsDirichletLines) {
    AffineConstraints constraints;

    // uDot_0 = uDot_1 (rate of an MPC slave with time-constant weights;
    // the value-constraint inhomogeneity must NOT leak into the rate)
    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.setInhomogeneity(0, 10.0);
    // uDot_2 carries the finite-difference rate of time-dependent Dirichlet
    // data (g_dot) and must not be zeroed.
    constraints.addLine(2);
    constraints.setInhomogeneity(2, 7.0);
    constraints.close();

    std::vector<double> vec = {0.0, 5.0, 4.5};
    constraints.distributeMasterBearingHomogeneous(vec);

    EXPECT_DOUBLE_EQ(vec[0], 5.0);  // master combination only
    EXPECT_DOUBLE_EQ(vec[1], 5.0);
    EXPECT_DOUBLE_EQ(vec[2], 4.5);  // Dirichlet rate (g_dot) untouched
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

TEST(AffineConstraintsTest, TransitiveClosureLongChain) {
    AffineConstraints constraints;

    // Chain: u_0 = u_1 = u_2 = u_3 = u_4 = u_5 = u_6
    for (GlobalIndex i = 0; i < 6; ++i) {
        constraints.addLine(i);
        constraints.addEntry(i, i + 1, 1.0);
    }

    constraints.close();

    for (GlobalIndex i = 0; i < 6; ++i) {
        auto c = constraints.getConstraint(i);
        ASSERT_TRUE(c.has_value());
        ASSERT_EQ(c->entries.size(), 1u);
        EXPECT_EQ(c->entries[0].master_dof, 6);
        EXPECT_DOUBLE_EQ(c->entries[0].weight, 1.0);
    }
}

TEST(AffineConstraintsTest, TransitiveClosureDiamondPattern) {
    AffineConstraints constraints;

    // u_0 = 0.5*u_1 + 0.5*u_2, u_1 = u_3, u_2 = u_3 -> u_0 = u_3
    constraints.addLine(0);
    constraints.addEntry(0, 1, 0.5);
    constraints.addEntry(0, 2, 0.5);

    constraints.addLine(1);
    constraints.addEntry(1, 3, 1.0);

    constraints.addLine(2);
    constraints.addEntry(2, 3, 1.0);

    constraints.close();

    auto c0 = constraints.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    ASSERT_EQ(c0->entries.size(), 1u);
    EXPECT_EQ(c0->entries[0].master_dof, 3);
    EXPECT_DOUBLE_EQ(c0->entries[0].weight, 1.0);
}

TEST(AffineConstraintsTest, TransitiveClosureWeightAccumulation) {
    AffineConstraints constraints;

    // u_0 = 0.7*u_1, u_1 = 0.3*u_2 + 0.7*u_3 -> u_0 = 0.21*u_2 + 0.49*u_3
    constraints.addLine(0);
    constraints.addEntry(0, 1, 0.7);

    constraints.addLine(1);
    constraints.addEntry(1, 2, 0.3);
    constraints.addEntry(1, 3, 0.7);

    constraints.close();

    auto c0 = constraints.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    ASSERT_EQ(c0->entries.size(), 2u);

    bool has_2 = false;
    bool has_3 = false;
    for (const auto& e : c0->entries) {
        if (e.master_dof == 2) {
            has_2 = true;
            EXPECT_NEAR(e.weight, 0.21, 1e-14);
        }
        if (e.master_dof == 3) {
            has_3 = true;
            EXPECT_NEAR(e.weight, 0.49, 1e-14);
        }
    }
    EXPECT_TRUE(has_2);
    EXPECT_TRUE(has_3);
}

TEST(AffineConstraintsTest, CycleDetectionSelfReference) {
    AffineConstraints constraints;
    constraints.addLine(0);
    EXPECT_THROW(constraints.addEntry(0, 0, 1.0), ConstraintException);
}

TEST(AffineConstraintsTest, CycleDetectionLongCycle) {
    AffineConstraints constraints;

    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.addLine(1);
    constraints.addEntry(1, 2, 1.0);
    constraints.addLine(2);
    constraints.addEntry(2, 3, 1.0);
    constraints.addLine(3);
    constraints.addEntry(3, 0, 1.0);

    EXPECT_THROW(constraints.close(), ConstraintCycleException);
}

TEST(AffineConstraintsTest, NearZeroWeightElimination) {
    {
        AffineConstraints constraints;
        constraints.addLine(0);
        constraints.addEntry(0, 1, 1e-16);
        constraints.close();

        auto c0 = constraints.getConstraint(0);
        ASSERT_TRUE(c0.has_value());
        EXPECT_TRUE(c0->entries.empty());
    }

    {
        AffineConstraints constraints;
        constraints.addLine(0);
        constraints.addEntry(0, 1, 1e-14);
        constraints.close();

        auto c0 = constraints.getConstraint(0);
        ASSERT_TRUE(c0.has_value());
        ASSERT_EQ(c0->entries.size(), 1u);
        EXPECT_EQ(c0->entries[0].master_dof, 1);
        EXPECT_DOUBLE_EQ(c0->entries[0].weight, 1e-14);
    }
}

TEST(AffineConstraintsTest, InhomogeneityOnlyUpdate) {
    AffineConstraints constraints;

    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.setInhomogeneity(0, 1.0);
    constraints.close();

    std::vector<double> vec = {0.0, 2.0};
    constraints.distribute(vec);
    EXPECT_DOUBLE_EQ(vec[0], 3.0);

    constraints.updateInhomogeneity(0, 5.0);

    auto c0 = constraints.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    ASSERT_EQ(c0->entries.size(), 1u);
    EXPECT_EQ(c0->entries[0].master_dof, 1);
    EXPECT_DOUBLE_EQ(c0->entries[0].weight, 1.0);
    EXPECT_DOUBLE_EQ(c0->inhomogeneity, 5.0);

    vec = {0.0, 2.0};
    constraints.distribute(vec);
    EXPECT_DOUBLE_EQ(vec[0], 7.0);
}

TEST(AffineConstraintsTest, CopyAndMoveSemantics) {
    AffineConstraints original;
    original.addLine(0);
    original.addEntry(0, 1, 1.0);
    original.setInhomogeneity(0, 2.0);
    original.addLine(2);
    original.addEntry(2, 3, 0.5);
    original.close();

    AffineConstraints copy = original;

    original.updateInhomogeneity(0, 10.0);
    EXPECT_DOUBLE_EQ(original.getConstraint(0)->inhomogeneity, 10.0);
    EXPECT_DOUBLE_EQ(copy.getConstraint(0)->inhomogeneity, 2.0);

    AffineConstraints moved = std::move(original);
    EXPECT_TRUE(moved.isClosed());
    EXPECT_TRUE(moved.isConstrained(0));
    EXPECT_TRUE(moved.isConstrained(2));

    EXPECT_EQ(original.numConstraints(), 0u);
    EXPECT_FALSE(original.isConstrained(0));

    original.clear();
    original.addDirichlet(5, 1.0);
    original.close();
    EXPECT_TRUE(original.isConstrained(5));
    EXPECT_DOUBLE_EQ(original.getConstraint(5)->inhomogeneity, 1.0);
}

TEST(AffineConstraintsTest, MergeConstraintSets) {
    AffineConstraints a;
    a.addLine(0);
    a.addEntry(0, 1, 1.0);

    AffineConstraints b;
    b.addLine(2);
    b.addEntry(2, 3, 2.0);

    a.merge(b);
    a.close();

    EXPECT_TRUE(a.isConstrained(0));
    EXPECT_TRUE(a.isConstrained(2));

    auto c2 = a.getConstraint(2);
    ASSERT_TRUE(c2.has_value());
    ASSERT_EQ(c2->entries.size(), 1u);
    EXPECT_EQ(c2->entries[0].master_dof, 3);
    EXPECT_DOUBLE_EQ(c2->entries[0].weight, 2.0);
}

TEST(AffineConstraintsTest, MergeConstraintSetsConflict) {
    AffineConstraints a;
    a.addLine(0);
    a.addEntry(0, 1, 1.0);

    AffineConstraints b;
    b.addLine(0);
    b.addEntry(0, 2, 1.0);

    EXPECT_THROW(a.merge(b), ConstraintException);

    a.merge(b, /*overwrite=*/true);
    a.close();

    auto c0 = a.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    ASSERT_EQ(c0->entries.size(), 1u);
    EXPECT_EQ(c0->entries[0].master_dof, 2);
}


// ============================================================================
// Closure reference comparison
// ============================================================================

namespace {

// Straightforward closure kept as a reference: every line is merged (sort,
// sum equal masters in order, drop near-zero weights), then each line is
// expanded through its constrained masters and merged again. close() skips
// lines without constrained masters and merges in place; it must reproduce
// this reference bit for bit.
void referenceMergeEntries(std::vector<ConstraintEntry>& entries)
{
    if (entries.size() < 2) {
        return;
    }
    std::sort(entries.begin(), entries.end());
    std::vector<ConstraintEntry> merged;
    for (const auto& entry : entries) {
        if (!merged.empty() && merged.back().master_dof == entry.master_dof) {
            merged.back().weight += entry.weight;
        } else {
            merged.push_back(entry);
        }
    }
    merged.erase(std::remove_if(merged.begin(), merged.end(),
                                [](const ConstraintEntry& e) {
                                    return std::abs(e.weight) < 1e-15;
                                }),
                 merged.end());
    entries = std::move(merged);
}

struct ReferenceLine {
    std::vector<ConstraintEntry> entries;
    double inhomogeneity{0.0};
    bool closed{false};
};

void referenceCloseLine(std::map<GlobalIndex, ReferenceLine>& lines, GlobalIndex slave)
{
    auto& line = lines.at(slave);
    if (line.closed) {
        return;
    }
    std::vector<ConstraintEntry> expanded;
    double inhomogeneity = line.inhomogeneity;
    for (const auto& entry : line.entries) {
        const auto master = lines.find(entry.master_dof);
        if (master != lines.end()) {
            referenceCloseLine(lines, entry.master_dof);
            for (const auto& master_entry : master->second.entries) {
                expanded.push_back({master_entry.master_dof, entry.weight * master_entry.weight});
            }
            inhomogeneity += entry.weight * master->second.inhomogeneity;
        } else {
            expanded.push_back(entry);
        }
    }
    line.entries = std::move(expanded);
    line.inhomogeneity = inhomogeneity;
    referenceMergeEntries(line.entries);
    line.closed = true;
}

std::uint64_t bits(double value)
{
    std::uint64_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

struct GeneratedLine {
    GlobalIndex slave{-1};
    std::vector<ConstraintEntry> entries;
    double inhomogeneity{0.0};
};

// Deterministic acyclic constraint set (masters have larger indices than
// their slaves) mixing Dirichlet lines, terminal lines, chains through
// constrained masters, repeated masters and exact cancellations.
std::vector<GeneratedLine> generateConstraintLines(std::uint64_t seed)
{
    std::uint64_t state = seed;
    auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 33);
    };
    std::vector<GeneratedLine> lines;
    for (GlobalIndex slave = 0; slave < 400; slave += 1 + static_cast<GlobalIndex>(next() % 3)) {
        GeneratedLine line;
        line.slave = slave;
        line.inhomogeneity = (next() % 4 == 0) ? static_cast<double>(next() % 17) / 7.0 : 0.0;
        const auto n_entries = next() % 7;
        for (std::uint32_t k = 0; k < n_entries; ++k) {
            const GlobalIndex master = slave + 1 + static_cast<GlobalIndex>(next() % 60);
            const double weight = (static_cast<double>(next() % 23) - 11.0) / 9.0;
            line.entries.push_back({master, weight});
            if (next() % 5 == 0) {
                line.entries.push_back({master, -weight});  // cancels exactly
            } else if (next() % 4 == 0) {
                line.entries.push_back({master, weight / 3.0});
            }
        }
        lines.push_back(std::move(line));
    }
    return lines;
}

void expectCloseMatchesReference(bool merge_duplicates)
{
    for (const std::uint64_t seed : {1ULL, 7ULL, 12345ULL}) {
        const auto generated = generateConstraintLines(seed);
        AffineConstraintsOptions options;
        options.merge_duplicates = merge_duplicates;
        AffineConstraints constraints(options);
        std::map<GlobalIndex, ReferenceLine> reference;
        for (const auto& line : generated) {
            constraints.addLine(line.slave);
            auto& ref = reference[line.slave];
            for (const auto& entry : line.entries) {
                constraints.addEntry(line.slave, entry.master_dof, entry.weight);
                if (std::abs(entry.weight) >= options.zero_tolerance) {
                    ref.entries.push_back(entry);
                }
            }
            constraints.setInhomogeneity(line.slave, line.inhomogeneity);
            ref.inhomogeneity = line.inhomogeneity;
        }
        if (merge_duplicates) {
            for (auto& [slave, line] : reference) {
                static_cast<void>(slave);
                referenceMergeEntries(line.entries);
            }
        }
        for (auto& [slave, line] : reference) {
            static_cast<void>(line);
            referenceCloseLine(reference, slave);
        }
        constraints.close();

        ASSERT_EQ(constraints.numConstraints(), reference.size());
        for (const auto& [slave, ref] : reference) {
            const auto view = constraints.getConstraint(slave);
            ASSERT_TRUE(view.has_value()) << "slave " << slave;
            ASSERT_EQ(view->entries.size(), ref.entries.size()) << "slave " << slave;
            EXPECT_EQ(bits(view->inhomogeneity), bits(ref.inhomogeneity)) << "slave " << slave;
            for (std::size_t i = 0; i < ref.entries.size(); ++i) {
                EXPECT_EQ(view->entries[i].master_dof, ref.entries[i].master_dof)
                    << "slave " << slave << " entry " << i;
                EXPECT_EQ(bits(view->entries[i].weight), bits(ref.entries[i].weight))
                    << "slave " << slave << " entry " << i;
            }
        }
    }
}

}  // namespace

TEST(AffineConstraintsTest, CloseMatchesReferenceClosureBitwise) {
    expectCloseMatchesReference(/*merge_duplicates=*/true);
}

TEST(AffineConstraintsTest, CloseWithoutMergeMatchesReferenceClosureBitwise) {
    expectCloseMatchesReference(/*merge_duplicates=*/false);
}

TEST(AffineConstraintsTest, MergeEntriesMatchesReferenceMerge) {
    std::uint64_t state = 99;
    auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 33);
    };
    for (int trial = 0; trial < 200; ++trial) {
        ConstraintLine line;
        const auto n = next() % 9;
        for (std::uint32_t k = 0; k < n; ++k) {
            const double weight = (next() % 6 == 0) ? 0.0
                                                    : (static_cast<double>(next() % 19) - 9.0) / 7.0;
            line.entries.push_back({static_cast<GlobalIndex>(next() % 5), weight});
        }
        auto expected = line.entries;
        referenceMergeEntries(expected);
        line.mergeEntries();
        ASSERT_EQ(line.entries.size(), expected.size()) << "trial " << trial;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(line.entries[i].master_dof, expected[i].master_dof);
            EXPECT_EQ(bits(line.entries[i].weight), bits(expected[i].weight));
        }
    }

    // A single entry is kept as is, even with a zero weight.
    ConstraintLine single;
    single.entries.push_back({3, 0.0});
    single.mergeEntries();
    ASSERT_EQ(single.entries.size(), 1u);
    EXPECT_EQ(single.entries[0].master_dof, 3);
}

TEST(AffineConstraintsTest, CloseStillDetectsCyclesBehindTerminalLines) {
    AffineConstraints constraints;
    // Terminal lines (masters unconstrained) around a two-line cycle.
    constraints.addLine(0);
    constraints.addEntry(0, 100, 1.0);
    constraints.addLine(1);
    constraints.addEntry(1, 2, 1.0);
    constraints.addLine(2);
    constraints.addEntry(2, 1, 1.0);
    constraints.addLine(3);
    constraints.addEntry(3, 101, 0.5);
    constraints.addEntry(3, 102, 0.5);
    EXPECT_THROW(constraints.close(), ConstraintCycleException);
}

}  // namespace test
}  // namespace constraints
}  // namespace FE
}  // namespace svmp
