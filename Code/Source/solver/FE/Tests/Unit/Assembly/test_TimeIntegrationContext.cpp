/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_TimeIntegrationContext.cpp
 * @brief Unit tests for TimeIntegrationContext and TimeDerivativeStencil
 */

#include <gtest/gtest.h>

#include "Assembly/TimeIntegrationContext.h"

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// TimeDerivativeStencil Tests
// ============================================================================

TEST(TimeDerivativeStencilTest, ConstructionAndDefaults) {
    TimeDerivativeStencil s;
    EXPECT_EQ(s.order, 0);
    EXPECT_TRUE(s.a.empty());
    EXPECT_EQ(s.requiredHistoryStates(), 0);
}

TEST(TimeDerivativeStencilTest, RequiredHistoryStatesEmptyCoefficients) {
    TimeDerivativeStencil s;
    s.order = 1;
    s.a = {};
    EXPECT_EQ(s.requiredHistoryStates(), 0);
}

TEST(TimeDerivativeStencilTest, RequiredHistoryStatesSingleCoefficient) {
    TimeDerivativeStencil s;
    s.order = 1;
    s.a = {1.0};
    EXPECT_EQ(s.requiredHistoryStates(), 0);
}

TEST(TimeDerivativeStencilTest, RequiredHistoryStatesTwoCoefficients) {
    TimeDerivativeStencil s;
    s.order = 1;
    s.a = {1.5, -0.5};
    EXPECT_EQ(s.requiredHistoryStates(), 1);
}

TEST(TimeDerivativeStencilTest, RequiredHistoryStatesThreeCoefficients) {
    TimeDerivativeStencil s;
    s.order = 2;
    s.a = {1.5, -2.0, 0.5};
    EXPECT_EQ(s.requiredHistoryStates(), 2);
}

TEST(TimeDerivativeStencilTest, RequiredHistoryStatesTrailingZeros) {
    TimeDerivativeStencil s;
    s.order = 1;
    s.a = {1.0, 0.5, 0.0, 0.0};
    EXPECT_EQ(s.requiredHistoryStates(), 1);
}

TEST(TimeDerivativeStencilTest, RequiredHistoryStatesAllZeroCoefficients) {
    TimeDerivativeStencil s;
    s.order = 1;
    s.a = {0.0, 0.0, 0.0};
    EXPECT_EQ(s.requiredHistoryStates(), 0);
}

TEST(TimeDerivativeStencilTest, CoeffValidIndex) {
    TimeDerivativeStencil s;
    s.order = 2;
    s.a = {1.0, -2.0, 1.0};
    EXPECT_DOUBLE_EQ(s.coeff(1), -2.0);
}

TEST(TimeDerivativeStencilTest, CoeffNegativeIndex) {
    TimeDerivativeStencil s;
    s.order = 1;
    s.a = {1.0, 2.0};
    EXPECT_DOUBLE_EQ(s.coeff(-1), 0.0);
}

TEST(TimeDerivativeStencilTest, CoeffOutOfRangeIndex) {
    TimeDerivativeStencil s;
    s.order = 1;
    s.a = {1.0, 2.0};
    EXPECT_DOUBLE_EQ(s.coeff(5), 0.0);
}

TEST(TimeDerivativeStencilTest, CoeffZeroIndex) {
    TimeDerivativeStencil s;
    s.order = 1;
    s.a = {3.14};
    EXPECT_DOUBLE_EQ(s.coeff(0), 3.14);
}

// ============================================================================
// TimeIntegrationContext Tests
// ============================================================================

TEST(TimeIntegrationContextTest, ConstructionAndDefaults) {
    TimeIntegrationContext ctx;

    EXPECT_EQ(ctx.integrator_name, "<unset>");
    EXPECT_FALSE(ctx.dt1.has_value());
    EXPECT_FALSE(ctx.dt2.has_value());

    EXPECT_DOUBLE_EQ(ctx.time_derivative_term_weight, 1.0);
    EXPECT_DOUBLE_EQ(ctx.non_time_derivative_term_weight, 1.0);
    EXPECT_DOUBLE_EQ(ctx.dt1_term_weight, 1.0);
    EXPECT_DOUBLE_EQ(ctx.dt2_term_weight, 1.0);
}

TEST(TimeIntegrationContextTest, StencilOrder1) {
    TimeIntegrationContext ctx;
    ctx.dt1 = TimeDerivativeStencil{1, {1.0, -1.0}};

    const auto* s = ctx.stencil(1);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->order, 1);
    EXPECT_DOUBLE_EQ(s->coeff(1), -1.0);
}

TEST(TimeIntegrationContextTest, StencilOrder2) {
    TimeIntegrationContext ctx;
    ctx.dt2 = TimeDerivativeStencil{2, {1.0, -2.0, 1.0}};

    const auto* s = ctx.stencil(2);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->order, 2);
    EXPECT_EQ(s->requiredHistoryStates(), 2);
}

TEST(TimeIntegrationContextTest, StencilOrder0ReturnsNull) {
    TimeIntegrationContext ctx;
    EXPECT_EQ(ctx.stencil(0), nullptr);
}

TEST(TimeIntegrationContextTest, StencilInvalidOrderReturnsNull) {
    TimeIntegrationContext ctx;
    EXPECT_EQ(ctx.stencil(3), nullptr);
    EXPECT_EQ(ctx.stencil(-1), nullptr);
}

TEST(TimeIntegrationContextTest, StencilUnsetOptionalReturnsNull) {
    TimeIntegrationContext ctx;
    EXPECT_EQ(ctx.stencil(1), nullptr);
}

TEST(TimeIntegrationContextTest, WeightMultipliersStored) {
    TimeIntegrationContext ctx;
    ctx.time_derivative_term_weight = 0.5;
    ctx.non_time_derivative_term_weight = 2.0;

    EXPECT_DOUBLE_EQ(ctx.time_derivative_term_weight, 0.5);
    EXPECT_DOUBLE_EQ(ctx.non_time_derivative_term_weight, 2.0);
}

TEST(TimeIntegrationContextTest, PerDerivativeWeightsStoredIndependently) {
    TimeIntegrationContext ctx;
    ctx.dt1_term_weight = 0.25;
    ctx.dt2_term_weight = 0.75;

    EXPECT_DOUBLE_EQ(ctx.dt1_term_weight, 0.25);
    EXPECT_DOUBLE_EQ(ctx.dt2_term_weight, 0.75);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp

