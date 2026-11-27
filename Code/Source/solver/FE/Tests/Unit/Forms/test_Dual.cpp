/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_Dual.cpp
 * @brief Unit tests for FE/Forms forward-mode Dual scalar utilities
 */

#include <gtest/gtest.h>

#include "Forms/Dual.h"

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(DualTest, BasicArithmeticAndDerivatives)
{
    DualWorkspace ws;
    ws.reset(2);

    Dual u = makeDualConstant(3.0, ws.alloc());
    Dual v = makeDualConstant(4.0, ws.alloc());

    u.deriv[0] = 1.0;  // du/du = 1
    v.deriv[1] = 1.0;  // dv/dv = 1

    // f(u,v) = u*v + u
    Dual uv = mul(u, v, makeDualConstant(0.0, ws.alloc()));
    Dual f = add(uv, u, makeDualConstant(0.0, ws.alloc()));

    EXPECT_NEAR(f.value, 15.0, 0.0);
    EXPECT_NEAR(f.deriv[0], 5.0, 1e-15);  // df/du = v + 1
    EXPECT_NEAR(f.deriv[1], 3.0, 1e-15);  // df/dv = u
}

TEST(DualTest, NegationAndSubtraction)
{
    DualWorkspace ws;
    ws.reset(1);

    Dual u = makeDualConstant(2.0, ws.alloc());
    u.deriv[0] = 1.0;

    Dual nu = neg(u, makeDualConstant(0.0, ws.alloc()));
    EXPECT_NEAR(nu.value, -2.0, 0.0);
    EXPECT_NEAR(nu.deriv[0], -1.0, 0.0);

    Dual z = sub(u, u, makeDualConstant(0.0, ws.alloc()));
    EXPECT_NEAR(z.value, 0.0, 0.0);
    EXPECT_NEAR(z.deriv[0], 0.0, 0.0);
}

TEST(DualTest, PowerKeepsEachIndependentSeed)
{
    DualWorkspace ws;
    ws.reset(2);
    auto base = makeDualConstant(-2.0, ws.alloc());
    auto exponent = makeDualConstant(2.0, ws.alloc());
    base.deriv[0] = 1.0;
    exponent.deriv[1] = 1.0;
    auto out = makeDualConstant(0.0, ws.alloc());
    auto result = pow(base, exponent, out);
    EXPECT_DOUBLE_EQ(result.deriv[0], -4.0);
    EXPECT_TRUE(std::isnan(result.deriv[1]));
    base.value = 2.0;
    result = pow(base, exponent, out);
    EXPECT_DOUBLE_EQ(result.deriv[0], 4.0);
    EXPECT_DOUBLE_EQ(result.deriv[1], 4.0 * std::log(2.0));
    base.value = 0.0;
    exponent.value = 1.0;
    result = pow(base, exponent, out);
    EXPECT_DOUBLE_EQ(result.deriv[0], 1.0);
    EXPECT_DOUBLE_EQ(result.deriv[1], 0.0);
    exponent.value = 0.0;
    EXPECT_DOUBLE_EQ(pow(base, exponent, out).deriv[0], 0.0);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

