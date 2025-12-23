/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NeumannBC.cpp
 * @brief Unit tests for NeumannBC class
 */

#include <gtest/gtest.h>

#include "Constraints/NeumannBC.h"

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(NeumannBCTest, ConstantFluxConstructionAndEvaluate) {
    NeumannBC bc(7, 2.5);

    EXPECT_EQ(bc.getBoundaryId(), 7);
    EXPECT_FALSE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());
    EXPECT_DOUBLE_EQ(bc.getConstantFlux(), 2.5);
    EXPECT_DOUBLE_EQ(bc.evaluateFlux(1.0, 2.0, 3.0), 2.5);
}

TEST(NeumannBCTest, SpatialFluxFunctionEvaluate) {
    NeumannBC bc(3, [](double x, double y, double z) { return x + 2.0 * y + 3.0 * z; });

    EXPECT_EQ(bc.getBoundaryId(), 3);
    EXPECT_FALSE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());
    EXPECT_DOUBLE_EQ(bc.evaluateFlux(1.0, 2.0, 3.0), 1.0 + 4.0 + 9.0);
}

TEST(NeumannBCTest, TimeDependentFluxFunctionEvaluate) {
    NeumannBC bc(3, [](double x, double y, double z, double t) { return x + y + z + t; });

    EXPECT_EQ(bc.getBoundaryId(), 3);
    EXPECT_FALSE(bc.isVectorValued());
    EXPECT_TRUE(bc.isTimeDependent());
    EXPECT_DOUBLE_EQ(bc.evaluateFlux(1.0, 2.0, 3.0, 4.0), 10.0);
}

TEST(NeumannBCTest, ConstantTractionConstructionAndEvaluate) {
    NeumannBC bc(9, std::array<double, 3>{{1.0, 2.0, 3.0}});

    EXPECT_EQ(bc.getBoundaryId(), 9);
    EXPECT_TRUE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());

    auto traction = bc.evaluateTraction(0.0, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(traction[0], 1.0);
    EXPECT_DOUBLE_EQ(traction[1], 2.0);
    EXPECT_DOUBLE_EQ(traction[2], 3.0);
}

TEST(NeumannBCTest, TractionFunctionEvaluate) {
    NeumannBC bc(2, [](double x, double y, double z) {
        return std::array<double, 3>{{x, y, z}};
    });

    EXPECT_TRUE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());

    auto traction = bc.evaluateTraction(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(traction[0], 1.0);
    EXPECT_DOUBLE_EQ(traction[1], 2.0);
    EXPECT_DOUBLE_EQ(traction[2], 3.0);
}

TEST(NeumannBCTest, TimeDependentTractionFunctionEvaluate) {
    NeumannBC bc(2, [](double x, double y, double z, double t) {
        return std::array<double, 3>{{x + t, y + t, z + t}};
    });

    EXPECT_TRUE(bc.isVectorValued());
    EXPECT_TRUE(bc.isTimeDependent());

    auto traction = bc.evaluateTraction(1.0, 2.0, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(traction[0], 5.0);
    EXPECT_DOUBLE_EQ(traction[1], 6.0);
    EXPECT_DOUBLE_EQ(traction[2], 7.0);
}

TEST(NeumannBCTest, SetFluxResetsVectorAndTimeFlags) {
    NeumannBC bc(1, [](double x, double /*y*/, double /*z*/, double t) { return x + t; });
    EXPECT_TRUE(bc.isTimeDependent());

    bc.setFlux(4.2);
    EXPECT_FALSE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());
    EXPECT_DOUBLE_EQ(bc.evaluateFlux(1.0, 0.0, 0.0, 9.0), 4.2);
}

TEST(NeumannBCTest, SetTractionResetsScalarAndTimeFlags) {
    NeumannBC bc(1, 3.14);
    EXPECT_FALSE(bc.isVectorValued());

    bc.setTraction({{1.0, 0.0, -1.0}});
    EXPECT_TRUE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());

    auto traction = bc.evaluateTraction(0.0, 0.0, 0.0, 10.0);
    EXPECT_DOUBLE_EQ(traction[0], 1.0);
    EXPECT_DOUBLE_EQ(traction[1], 0.0);
    EXPECT_DOUBLE_EQ(traction[2], -1.0);
}

TEST(NeumannBCTest, CollectionLookup) {
    NeumannBCCollection bcs;
    EXPECT_TRUE(bcs.empty());

    bcs.add(NeumannBC(1, 1.0));
    bcs.add(NeumannBC(2, std::array<double, 3>{{1.0, 2.0, 3.0}}));

    EXPECT_EQ(bcs.size(), 2u);
    EXPECT_TRUE(bcs.hasBC(1));
    EXPECT_TRUE(bcs.hasBC(2));
    EXPECT_FALSE(bcs.hasBC(3));

    const auto* bc = bcs.find(2);
    ASSERT_NE(bc, nullptr);
    EXPECT_TRUE(bc->isVectorValued());
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

