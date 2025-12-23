/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ConstraintTools.cpp
 * @brief Unit tests for ConstraintTools
 */

#include <gtest/gtest.h>
#include "Constraints/ConstraintTools.h"
#include "Constraints/AffineConstraints.h"

#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(ConstraintToolsTest, MakeDirichletConstraints_Constant) {
    std::vector<GlobalIndex> dofs = {0, 5, 10};
    AffineConstraints constraints;
    
    makeDirichletConstraints(dofs, 3.14, constraints);
    constraints.close();

    EXPECT_EQ(constraints.numConstraints(), 3);
    
    auto c0 = constraints.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_DOUBLE_EQ(c0->inhomogeneity, 3.14);
    EXPECT_TRUE(c0->entries.empty());

    auto c5 = constraints.getConstraint(5);
    ASSERT_TRUE(c5.has_value());
    EXPECT_DOUBLE_EQ(c5->inhomogeneity, 3.14);
}

TEST(ConstraintToolsTest, ExtractComponentDofs) {
    // 3 DOFs per node, 2 nodes
    // Node 0: 0(u), 1(v), 2(w)
    // Node 1: 3(u), 4(v), 5(w)
    std::vector<GlobalIndex> all_dofs = {0, 1, 2, 3, 4, 5};
    
    auto u_dofs = extractComponentDofs(all_dofs, 0, 3);
    EXPECT_EQ(u_dofs.size(), 2);
    EXPECT_EQ(u_dofs[0], 0);
    EXPECT_EQ(u_dofs[1], 3);

    auto v_dofs = extractComponentDofs(all_dofs, 1, 3);
    EXPECT_EQ(v_dofs.size(), 2);
    EXPECT_EQ(v_dofs[0], 1);
    EXPECT_EQ(v_dofs[1], 4);
}

TEST(ConstraintToolsTest, MergeConstraints) {
    AffineConstraints c1;
    c1.addLine(0);
    c1.setInhomogeneity(0, 1.0);

    AffineConstraints c2;
    c2.addLine(1);
    c2.setInhomogeneity(1, 2.0);

    AffineConstraints target;
    mergeConstraints(target, c1);
    mergeConstraints(target, c2);
    target.close();

    EXPECT_EQ(target.numConstraints(), 2);
    EXPECT_TRUE(target.isConstrained(0));
    EXPECT_TRUE(target.isConstrained(1));
}

TEST(ConstraintToolsTest, MakePeriodicConstraints_Translation) {
    // Simple 1D periodic: u_0 (x=0) = u_1 (x=1)
    std::vector<GlobalIndex> slave = {0};
    std::vector<std::array<double, 3>> slave_coords = {{{0.0, 0.0, 0.0}}};
    
    std::vector<GlobalIndex> master = {1};
    std::vector<std::array<double, 3>> master_coords = {{{1.0, 0.0, 0.0}}};

    AffineConstraints constraints;
    std::array<double, 3> translation = {{1.0, 0.0, 0.0}}; // slave + trans = master

    makePeriodicConstraintsTranslation(slave, slave_coords, master, master_coords, translation, constraints);
    constraints.close();

    // Check u_0 = u_1
    auto c = constraints.getConstraint(0);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c->entries.size(), 1);
    EXPECT_EQ(c->entries[0].master_dof, 1);
    EXPECT_DOUBLE_EQ(c->entries[0].weight, 1.0);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
