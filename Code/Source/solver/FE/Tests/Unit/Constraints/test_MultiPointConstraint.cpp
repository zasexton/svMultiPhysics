/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_MultiPointConstraint.cpp
 * @brief Unit tests for MultiPointConstraint class
 */

#include <gtest/gtest.h>
#include "Constraints/MultiPointConstraint.h"
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

TEST(MultiPointConstraintTest, DefaultConstruction) {
    MultiPointConstraint mpc;

    EXPECT_TRUE(mpc.empty());
    EXPECT_EQ(mpc.numConstraints(), 0);
    EXPECT_EQ(mpc.getType(), ConstraintType::MultiPoint);
}

TEST(MultiPointConstraintTest, ExplicitSlavemaster) {
    MultiPointConstraint mpc;

    // u_0 = 0.5*u_1 + 0.5*u_2
    mpc.addConstraint(0, {{1, 0.5}, {2, 0.5}});

    EXPECT_EQ(mpc.numConstraints(), 1);

    AffineConstraints aff;
    mpc.apply(aff);
    aff.close();

    auto c = aff.getConstraint(0);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->entries.size(), 2);

    double sum = 0.0;
    for (const auto& e : c->entries) {
        sum += e.weight;
    }
    EXPECT_DOUBLE_EQ(sum, 1.0);
}

TEST(MultiPointConstraintTest, SimpleEquality) {
    MultiPointConstraint mpc;

    // u_0 = u_1
    mpc.addConstraint(0, 1, 1.0, 0.0);

    AffineConstraints aff;
    mpc.apply(aff);
    aff.close();

    auto c = aff.getConstraint(0);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->entries.size(), 1);
    EXPECT_EQ(c->entries[0].master_dof, 1);
    EXPECT_DOUBLE_EQ(c->entries[0].weight, 1.0);
}

TEST(MultiPointConstraintTest, WithInhomogeneity) {
    MultiPointConstraint mpc;

    // u_0 = u_1 + 5.0
    mpc.addConstraint(0, 1, 1.0, 5.0);

    AffineConstraints aff;
    mpc.apply(aff);
    aff.close();

    auto c = aff.getConstraint(0);
    ASSERT_TRUE(c.has_value());
    EXPECT_DOUBLE_EQ(c->inhomogeneity, 5.0);
}

// ============================================================================
// Equation-based interface tests
// ============================================================================

TEST(MultiPointConstraintTest, EquationForm) {
    MultiPointConstraint mpc;

    // Equation: 2*u_0 - u_1 - u_2 = 0
    // => u_0 = 0.5*u_1 + 0.5*u_2
    MPCEquation eq;
    eq.addTerm(0, 2.0);
    eq.addTerm(1, -1.0);
    eq.addTerm(2, -1.0);
    eq.rhs = 0.0;

    mpc.addEquation(eq);

    AffineConstraints aff;
    mpc.apply(aff);
    aff.close();

    EXPECT_TRUE(aff.isConstrained(0));
    EXPECT_FALSE(aff.isConstrained(1));
    EXPECT_FALSE(aff.isConstrained(2));

    auto c = aff.getConstraint(0);
    ASSERT_TRUE(c.has_value());

    // Should have u_1 and u_2 as masters with weight 0.5 each
    // (since we solve 2*u_0 = u_1 + u_2 => u_0 = 0.5*u_1 + 0.5*u_2)
    EXPECT_EQ(c->entries.size(), 2);
}

TEST(MultiPointConstraintTest, EquationWithRhs) {
    MultiPointConstraint mpc;

    // Equation: u_0 - u_1 = 3.0
    // => u_0 = u_1 + 3.0
    MPCEquation eq;
    eq.addTerm(0, 1.0);
    eq.addTerm(1, -1.0);
    eq.rhs = 3.0;

    mpc.addEquation(eq);

    AffineConstraints aff;
    mpc.apply(aff);
    aff.close();

    auto c = aff.getConstraint(0);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->entries.size(), 1);
    EXPECT_EQ(c->entries[0].master_dof, 1);
    EXPECT_DOUBLE_EQ(c->entries[0].weight, 1.0);
    EXPECT_DOUBLE_EQ(c->inhomogeneity, 3.0);
}

// ============================================================================
// Common pattern tests
// ============================================================================

TEST(MultiPointConstraintTest, RigidLink) {
    MultiPointConstraint mpc;

    // DOFs 1, 2, 3 rigidly linked to DOF 0
    std::vector<GlobalIndex> slaves = {1, 2, 3};
    mpc.addRigidLink(slaves, 0);

    EXPECT_EQ(mpc.numConstraints(), 3);

    AffineConstraints aff;
    mpc.apply(aff);
    aff.close();

    for (GlobalIndex s : slaves) {
        EXPECT_TRUE(aff.isConstrained(s));
        auto c = aff.getConstraint(s);
        ASSERT_TRUE(c.has_value());
        EXPECT_EQ(c->entries.size(), 1);
        EXPECT_EQ(c->entries[0].master_dof, 0);
        EXPECT_DOUBLE_EQ(c->entries[0].weight, 1.0);
    }
}

TEST(MultiPointConstraintTest, Average) {
    MultiPointConstraint mpc;

    // u_0 = average(u_1, u_2, u_3)
    std::vector<GlobalIndex> masters = {1, 2, 3};
    mpc.addAverage(0, masters);

    AffineConstraints aff;
    mpc.apply(aff);
    aff.close();

    auto c = aff.getConstraint(0);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->entries.size(), 3);

    double sum = 0.0;
    for (const auto& e : c->entries) {
        EXPECT_NEAR(e.weight, 1.0 / 3.0, 1e-14);
        sum += e.weight;
    }
    EXPECT_NEAR(sum, 1.0, 1e-14);
}

TEST(MultiPointConstraintTest, WeightedAverage) {
    MultiPointConstraint mpc;

    // u_0 = weighted average of u_1, u_2 with weights 1, 3
    // => u_0 = (1*u_1 + 3*u_2) / 4 = 0.25*u_1 + 0.75*u_2
    std::vector<GlobalIndex> masters = {1, 2};
    std::vector<double> weights = {1.0, 3.0};

    mpc.addWeightedAverage(0, masters, weights);

    AffineConstraints aff;
    mpc.apply(aff);
    aff.close();

    auto c = aff.getConstraint(0);
    ASSERT_TRUE(c.has_value());

    bool found_1 = false, found_2 = false;
    for (const auto& e : c->entries) {
        if (e.master_dof == 1) {
            found_1 = true;
            EXPECT_NEAR(e.weight, 0.25, 1e-14);
        }
        if (e.master_dof == 2) {
            found_2 = true;
            EXPECT_NEAR(e.weight, 0.75, 1e-14);
        }
    }
    EXPECT_TRUE(found_1);
    EXPECT_TRUE(found_2);
}

// ============================================================================
// Factory method tests
// ============================================================================

TEST(MultiPointConstraintTest, RigidLinkFactory) {
    std::vector<GlobalIndex> slaves = {1, 2};
    GlobalIndex master = 0;

    MultiPointConstraint mpc = MultiPointConstraint::rigidLink(slaves, master);

    EXPECT_EQ(mpc.numConstraints(), 2);
}

TEST(MultiPointConstraintTest, AverageFactory) {
    std::vector<GlobalIndex> averaged = {1, 2, 3};

    MultiPointConstraint mpc = MultiPointConstraint::average(0, averaged);

    EXPECT_EQ(mpc.numConstraints(), 1);
}

// ============================================================================
// Validation tests
// ============================================================================

TEST(MultiPointConstraintTest, ValidateValid) {
    MultiPointConstraint mpc;
    mpc.addConstraint(0, 1, 1.0);
    mpc.addConstraint(2, 3, 0.5, 1.0);

    std::string error = mpc.validate();
    EXPECT_TRUE(error.empty());
}

TEST(MultiPointConstraintTest, ValidateDuplicateSlave) {
    MultiPointConstraint mpc;
    mpc.addConstraint(0, 1, 1.0);
    mpc.addConstraint(0, 2, 0.5);  // Duplicate slave DOF 0

    std::string error = mpc.validate();
    EXPECT_FALSE(error.empty());
}

// ============================================================================
// Clear and info tests
// ============================================================================

TEST(MultiPointConstraintTest, Clear) {
    MultiPointConstraint mpc;
    mpc.addConstraint(0, 1, 1.0);
    mpc.addConstraint(2, 3, 1.0);

    EXPECT_EQ(mpc.numConstraints(), 2);

    mpc.clear();

    EXPECT_TRUE(mpc.empty());
    EXPECT_EQ(mpc.numConstraints(), 0);
}

TEST(MultiPointConstraintTest, GetInfo) {
    MultiPointConstraint mpc;
    mpc.addConstraint(0, 1, 1.0);
    mpc.addConstraint(2, 3, 1.0, 5.0);  // With inhomogeneity

    auto info = mpc.getInfo();

    EXPECT_EQ(info.name, "MultiPointConstraint");
    EXPECT_EQ(info.type, ConstraintType::MultiPoint);
    EXPECT_EQ(info.num_constrained_dofs, 2);
    EXPECT_FALSE(info.is_time_dependent);
    EXPECT_FALSE(info.is_homogeneous);  // One has inhomogeneity
}

TEST(MultiPointConstraintTest, Clone) {
    MultiPointConstraint mpc;
    mpc.addConstraint(0, 1, 1.0);

    auto clone = mpc.clone();

    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getType(), ConstraintType::MultiPoint);
}

}  // namespace test
}  // namespace constraints
}  // namespace FE
}  // namespace svmp
