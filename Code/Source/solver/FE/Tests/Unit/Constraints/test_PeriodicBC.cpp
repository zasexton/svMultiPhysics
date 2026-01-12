/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_PeriodicBC.cpp
 * @brief Unit tests for PeriodicBC class
 */

#include <gtest/gtest.h>
#include "Constraints/PeriodicBC.h"
#include "Constraints/AffineConstraints.h"

#include <cmath>
#include <vector>
#include <array>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

// ============================================================================
// Basic tests
// ============================================================================

TEST(PeriodicBCTest, DirectPairConstruction) {
    std::vector<GlobalIndex> slaves = {0, 1, 2};
    std::vector<GlobalIndex> masters = {10, 11, 12};

    PeriodicBC bc(slaves, masters);

    EXPECT_EQ(bc.getType(), ConstraintType::Periodic);
    EXPECT_EQ(bc.numPairs(), 3);

    AffineConstraints aff;
    bc.apply(aff);
    aff.close();

    EXPECT_TRUE(aff.isConstrained(0));
    EXPECT_TRUE(aff.isConstrained(1));
    EXPECT_TRUE(aff.isConstrained(2));

    // Check u_0 = u_10
    auto c0 = aff.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_EQ(c0->entries.size(), 1);
    EXPECT_EQ(c0->entries[0].master_dof, 10);
    EXPECT_DOUBLE_EQ(c0->entries[0].weight, 1.0);
}

TEST(PeriodicBCTest, AntiPeriodic) {
    std::vector<GlobalIndex> slaves = {0, 1};
    std::vector<GlobalIndex> masters = {10, 11};

    PeriodicBCOptions opts;
    opts.anti_periodic = true;

    PeriodicBC bc(slaves, masters, opts);

    AffineConstraints aff;
    bc.apply(aff);
    aff.close();

    // For anti-periodic: u_slave = -u_master
    auto c0 = aff.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_DOUBLE_EQ(c0->entries[0].weight, -1.0);
}

TEST(PeriodicBCTest, CoordinateMatching) {
    // Left boundary: DOFs 0, 1 at x=0
    // Right boundary: DOFs 10, 11 at x=1
    std::vector<GlobalIndex> left_dofs = {0, 1};
    std::vector<std::array<double, 3>> left_coords = {
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.5, 0.0}}
    };

    std::vector<GlobalIndex> right_dofs = {10, 11};
    std::vector<std::array<double, 3>> right_coords = {
        {{1.0, 0.0, 0.0}},
        {{1.0, 0.5, 0.0}}
    };

    // Translation from left to right
    std::array<double, 3> translation = {{1.0, 0.0, 0.0}};

    PeriodicBC bc(left_dofs, left_coords, right_dofs, right_coords, translation);

    // Should match (0,0,0) + (1,0,0) = (1,0,0) -> DOF 10
    // Should match (0,0.5,0) + (1,0,0) = (1,0.5,0) -> DOF 11
    EXPECT_EQ(bc.numPairs(), 2);

    AffineConstraints aff;
    bc.apply(aff);
    aff.close();

    auto c0 = aff.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_EQ(c0->entries[0].master_dof, 10);

    auto c1 = aff.getConstraint(1);
    ASSERT_TRUE(c1.has_value());
    EXPECT_EQ(c1->entries[0].master_dof, 11);
}

TEST(PeriodicBCTest, NonMatchingPeriodicMeshTolerance) {
    std::vector<GlobalIndex> left_dofs = {0, 1};
    std::vector<std::array<double, 3>> left_coords = {
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.5 + 1e-10, 0.0}}
    };

    std::vector<GlobalIndex> right_dofs = {10, 11};
    std::vector<std::array<double, 3>> right_coords = {
        {{1.0, 0.0, 0.0}},
        {{1.0, 0.5, 0.0}}
    };

    std::array<double, 3> translation = {{1.0, 0.0, 0.0}};

    PeriodicBCOptions opts;
    opts.matching_tolerance = 1e-9;

    PeriodicBC bc(left_dofs, left_coords, right_dofs, right_coords, translation, opts);
    EXPECT_EQ(bc.numPairs(), 2);

    AffineConstraints aff;
    bc.apply(aff);
    aff.close();

    auto c1 = aff.getConstraint(1);
    ASSERT_TRUE(c1.has_value());
    EXPECT_EQ(c1->entries[0].master_dof, 11);
}

TEST(PeriodicBCTest, PeriodicConstraintChainResolution) {
    // Chain: u_0 = u_1, u_1 = u_2 -> after closure u_0 = u_2.
    PeriodicBC bc({0, 1}, {1, 2});

    AffineConstraints aff;
    bc.apply(aff);
    aff.close();

    auto c0 = aff.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    ASSERT_EQ(c0->entries.size(), 1u);
    EXPECT_EQ(c0->entries[0].master_dof, 2);
    EXPECT_DOUBLE_EQ(c0->entries[0].weight, 1.0);
}

// ============================================================================
// Factory method tests
// ============================================================================

TEST(PeriodicBCTest, XPeriodicFactory) {
    std::vector<GlobalIndex> left_dofs = {0, 1};
    std::vector<std::array<double, 3>> left_coords = {
        {{0.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}}
    };

    std::vector<GlobalIndex> right_dofs = {10, 11};
    std::vector<std::array<double, 3>> right_coords = {
        {{2.0, 0.0, 0.0}},
        {{2.0, 1.0, 0.0}}
    };

    PeriodicBC bc = PeriodicBC::xPeriodic(
        left_dofs, left_coords,
        right_dofs, right_coords,
        2.0  // domain_length
    );

    EXPECT_EQ(bc.numPairs(), 2);
}

TEST(PeriodicBCTest, YPeriodicFactory) {
    std::vector<GlobalIndex> bottom_dofs = {0};
    std::vector<std::array<double, 3>> bottom_coords = {{{0.5, 0.0, 0.0}}};

    std::vector<GlobalIndex> top_dofs = {10};
    std::vector<std::array<double, 3>> top_coords = {{{0.5, 1.0, 0.0}}};

    PeriodicBC bc = PeriodicBC::yPeriodic(
        bottom_dofs, bottom_coords,
        top_dofs, top_coords,
        1.0
    );

    EXPECT_EQ(bc.numPairs(), 1);
}

// ============================================================================
// Modification tests
// ============================================================================

TEST(PeriodicBCTest, AddPair) {
    PeriodicBC bc;

    bc.addPair(5, 15, 1.0);
    bc.addPair(6, 16, 1.0);

    EXPECT_EQ(bc.numPairs(), 2);

    const auto& pairs = bc.getPairs();
    EXPECT_EQ(pairs[0].slave_dof, 5);
    EXPECT_EQ(pairs[0].master_dof, 15);
}

TEST(PeriodicBCTest, AddPairs) {
    PeriodicBC bc;

    std::vector<GlobalIndex> slaves = {0, 1, 2};
    std::vector<GlobalIndex> masters = {10, 11, 12};

    bc.addPairs(slaves, masters);

    EXPECT_EQ(bc.numPairs(), 3);
}

// ============================================================================
// Clone test
// ============================================================================

TEST(PeriodicBCTest, Clone) {
    std::vector<GlobalIndex> slaves = {0, 1};
    std::vector<GlobalIndex> masters = {10, 11};

    PeriodicBC original(slaves, masters);
    auto clone = original.clone();

    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getType(), ConstraintType::Periodic);

    AffineConstraints c1, c2;
    original.apply(c1);
    clone->apply(c2);
    c1.close();
    c2.close();

    EXPECT_EQ(c1.numConstraints(), c2.numConstraints());
}

// ============================================================================
// Info test
// ============================================================================

TEST(PeriodicBCTest, GetInfo) {
    std::vector<GlobalIndex> slaves = {0, 1, 2};
    std::vector<GlobalIndex> masters = {10, 11, 12};

    PeriodicBC bc(slaves, masters);

    auto info = bc.getInfo();

    EXPECT_EQ(info.name, "PeriodicBC");
    EXPECT_EQ(info.type, ConstraintType::Periodic);
    EXPECT_EQ(info.num_constrained_dofs, 3);
    EXPECT_FALSE(info.is_time_dependent);
    EXPECT_TRUE(info.is_homogeneous);
}

}  // namespace test
}  // namespace constraints
}  // namespace FE
}  // namespace svmp
