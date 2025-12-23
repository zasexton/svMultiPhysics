/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_RobinBC.cpp
 * @brief Unit tests for RobinBC class
 */

#include <gtest/gtest.h>

#include "Constraints/RobinBC.h"

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(RobinBCTest, DefaultConstruction) {
    RobinBC bc;
    EXPECT_TRUE(bc.isConstant());

    const auto data = bc.evaluate(0.0, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(data.alpha, 1.0);
    EXPECT_DOUBLE_EQ(data.beta, 1.0);
    EXPECT_DOUBLE_EQ(data.g, 0.0);
    EXPECT_TRUE(data.isValid());
}

TEST(RobinBCTest, ConstantConstructionAndEvaluate) {
    RobinBC bc(2.0, 0.5, 3.0);
    EXPECT_TRUE(bc.isConstant());

    EXPECT_DOUBLE_EQ(bc.evaluateAlpha(1.0, 2.0, 3.0), 2.0);
    EXPECT_DOUBLE_EQ(bc.evaluateG(1.0, 2.0, 3.0), 3.0);

    const auto data = bc.evaluate(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(data.alpha, 2.0);
    EXPECT_DOUBLE_EQ(data.beta, 0.5);
    EXPECT_DOUBLE_EQ(data.g, 3.0);
}

TEST(RobinBCTest, FunctionConstructionAndEvaluate) {
    RobinBC bc([](double x, double y, double z, double t) {
        RobinData data;
        data.alpha = x + t;
        data.beta = y;
        data.g = z - t;
        return data;
    });
    EXPECT_FALSE(bc.isConstant());

    const auto data = bc.evaluate(1.0, 2.0, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(data.alpha, 5.0);
    EXPECT_DOUBLE_EQ(data.beta, 2.0);
    EXPECT_DOUBLE_EQ(data.g, -1.0);
}

TEST(RobinBCTest, EvaluateMultiple) {
    RobinBC bc(2.0, 1.0, 3.0);

    std::vector<std::array<double, 3>> pts = {
        {{0.0, 0.0, 0.0}},
        {{1.0, 2.0, 3.0}}
    };

    auto data = bc.evaluateMultiple(pts, 0.0);
    ASSERT_EQ(data.size(), 2u);
    EXPECT_DOUBLE_EQ(data[0].alpha, 2.0);
    EXPECT_DOUBLE_EQ(data[1].g, 3.0);
}

TEST(RobinBCTest, ComputeMatrixContribution) {
    std::array<double, 2> N{{0.5, 0.5}};
    std::array<double, 4> local{};

    RobinBC::computeMatrixContribution(N, /*alpha=*/2.0, /*weight=*/1.0, local);

    // alpha * N_i * N_j = 2 * 0.25 = 0.5 for all entries
    EXPECT_DOUBLE_EQ(local[0], 0.5);
    EXPECT_DOUBLE_EQ(local[1], 0.5);
    EXPECT_DOUBLE_EQ(local[2], 0.5);
    EXPECT_DOUBLE_EQ(local[3], 0.5);
}

TEST(RobinBCTest, ComputeRhsContribution) {
    std::array<double, 2> N{{0.5, 0.5}};
    std::array<double, 2> rhs{};

    RobinBC::computeRhsContribution(N, /*g=*/4.0, /*weight=*/2.0, rhs);

    // g * weight * N_i = 4 * 2 * 0.5 = 4 for each i
    EXPECT_DOUBLE_EQ(rhs[0], 4.0);
    EXPECT_DOUBLE_EQ(rhs[1], 4.0);
}

TEST(RobinBCTest, ConvectiveFactory) {
    RobinBC bc = RobinBC::convective(/*h=*/10.0, /*T_inf=*/3.0);
    const auto data = bc.evaluate(0.0, 0.0, 0.0);

    EXPECT_DOUBLE_EQ(data.alpha, 10.0);
    EXPECT_DOUBLE_EQ(data.beta, 1.0);
    EXPECT_DOUBLE_EQ(data.g, 30.0);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

