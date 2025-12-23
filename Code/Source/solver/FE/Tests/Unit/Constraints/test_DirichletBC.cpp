/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_DirichletBC.cpp
 * @brief Unit tests for DirichletBC class
 */

#include <gtest/gtest.h>
#include "Constraints/DirichletBC.h"
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

TEST(DirichletBCTest, ConstantValueConstruction) {
    std::vector<GlobalIndex> dofs = {0, 1, 2};
    double value = 1.5;

    DirichletBC bc(dofs, value);

    EXPECT_EQ(bc.getType(), ConstraintType::Dirichlet);

    auto info = bc.getInfo();
    EXPECT_EQ(info.num_constrained_dofs, 3);
    EXPECT_FALSE(info.is_time_dependent);
}

TEST(DirichletBCTest, ApplyToConstraints) {
    std::vector<GlobalIndex> dofs = {5, 10, 15};
    double value = 2.5;

    DirichletBC bc(dofs, value);

    AffineConstraints constraints;
    bc.apply(constraints);
    constraints.close();

    EXPECT_TRUE(constraints.isConstrained(5));
    EXPECT_TRUE(constraints.isConstrained(10));
    EXPECT_TRUE(constraints.isConstrained(15));
    EXPECT_FALSE(constraints.isConstrained(0));

    auto c = constraints.getConstraint(5);
    ASSERT_TRUE(c.has_value());
    EXPECT_TRUE(c->entries.empty());  // Dirichlet has no masters
    EXPECT_DOUBLE_EQ(c->inhomogeneity, 2.5);
}

TEST(DirichletBCTest, FunctionBasedDirichlet) {
    std::vector<GlobalIndex> dofs = {0, 1, 2};
    std::vector<std::array<double, 3>> coords = {
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}}
    };

    // u(x,y,z) = x + 2*y
    auto func = [](double x, double y, double /*z*/) {
        return x + 2.0 * y;
    };

    DirichletBC bc(dofs, coords, func);

    AffineConstraints constraints;
    bc.apply(constraints);
    constraints.close();

    // Check values: (0,0,0)->0, (1,0,0)->1, (0,1,0)->2
    EXPECT_DOUBLE_EQ(constraints.getConstraint(0)->inhomogeneity, 0.0);
    EXPECT_DOUBLE_EQ(constraints.getConstraint(1)->inhomogeneity, 1.0);
    EXPECT_DOUBLE_EQ(constraints.getConstraint(2)->inhomogeneity, 2.0);
}

TEST(DirichletBCTest, TimeDependent) {
    std::vector<GlobalIndex> dofs = {0};
    std::vector<std::array<double, 3>> coords = {{{0.0, 0.0, 0.0}}};

    // u(x,y,z,t) = t
    auto func = [](double /*x*/, double /*y*/, double /*z*/, double t) {
        return t;
    };

    // Construct with time-dependent function, initial_time = 0.0
    DirichletBC bc(dofs, coords, func, 0.0);

    // Apply at t=0
    AffineConstraints c0;
    bc.apply(c0);
    c0.close();
    EXPECT_DOUBLE_EQ(c0.getConstraint(0)->inhomogeneity, 0.0);
    EXPECT_TRUE(bc.isTimeDependent());
}

TEST(DirichletBCTest, Clone) {
    std::vector<GlobalIndex> dofs = {0, 1};
    DirichletBC bc(dofs, 3.14);

    auto clone = bc.clone();

    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getType(), ConstraintType::Dirichlet);

    AffineConstraints c1, c2;
    bc.apply(c1);
    clone->apply(c2);
    c1.close();
    c2.close();

    EXPECT_DOUBLE_EQ(c1.getConstraint(0)->inhomogeneity,
                     c2.getConstraint(0)->inhomogeneity);
}

TEST(DirichletBCTest, SingleDof) {
    // Use single DOF constructor
    DirichletBC bc(10, 5.0);

    AffineConstraints constraints;
    bc.apply(constraints);
    constraints.close();

    EXPECT_TRUE(constraints.isConstrained(10));
    EXPECT_DOUBLE_EQ(constraints.getConstraint(10)->inhomogeneity, 5.0);
}

TEST(DirichletBCTest, HomogeneousFactory) {
    std::vector<GlobalIndex> dofs = {1, 2, 3};
    DirichletBC bc = DirichletBC::homogeneous(dofs);

    AffineConstraints constraints;
    bc.apply(constraints);
    constraints.close();

    for (GlobalIndex d : dofs) {
        EXPECT_TRUE(constraints.isConstrained(d));
        EXPECT_DOUBLE_EQ(constraints.getConstraint(d)->inhomogeneity, 0.0);
    }
}

}  // namespace test
}  // namespace constraints
}  // namespace FE
}  // namespace svmp
