/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorCanonicalize.h"

#include "Forms/FormExpr.h"
#include "Forms/Index.h"

#include <gtest/gtest.h>

namespace svmp::FE::forms::tensor {

TEST(TensorCanonicalize, CanonicalStringIgnoresIndexIds)
{
    const auto A = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});
    const auto B = FormExpr::asVector({FormExpr::constant(4.0), FormExpr::constant(5.0), FormExpr::constant(6.0)});

    forms::Index i1("i");
    const auto expr1 = A(i1) * B(i1);

    forms::Index i2("i");
    const auto expr2 = A(i2) * B(i2);

    EXPECT_EQ(toCanonicalString(expr1), toCanonicalString(expr2));
}

TEST(TensorCanonicalize, CanonicalizeAddOrder)
{
    const auto a = FormExpr::parameter("a");
    const auto b = FormExpr::parameter("b");
    const auto c = FormExpr::parameter("c");

    const auto expr1 = (a + b) + c;
    const auto expr2 = c + (b + a);

    EXPECT_EQ(canonicalizeTermOrder(expr1).toString(), canonicalizeTermOrder(expr2).toString());
}

TEST(TensorCanonicalize, CanonicalizeMultiplyScalarOrder)
{
    const auto a = FormExpr::parameter("a");
    const auto b = FormExpr::parameter("b");
    const auto c = FormExpr::parameter("c");

    const auto expr1 = (a * b) * c;
    const auto expr2 = c * (b * a);

    EXPECT_EQ(canonicalizeTermOrder(expr1).toString(), canonicalizeTermOrder(expr2).toString());
}

TEST(TensorCanonicalize, CanonicalizeScalarVectorScaling)
{
    const auto s = FormExpr::parameter("s");
    const auto v = FormExpr::asVector({FormExpr::parameter("v0"), FormExpr::parameter("v1")});

    const auto expr1 = s * v;
    const auto expr2 = v * s;

    EXPECT_EQ(canonicalizeTermOrder(expr1).toString(), canonicalizeTermOrder(expr2).toString());
}

TEST(TensorCanonicalize, DoesNotReorderUnknownMultiply)
{
    const auto v = FormExpr::asVector({FormExpr::parameter("v0"), FormExpr::parameter("v1")});
    const auto w = FormExpr::asVector({FormExpr::parameter("w0"), FormExpr::parameter("w1")});

    const auto expr1 = v * w;
    const auto expr2 = w * v;

    EXPECT_NE(canonicalizeTermOrder(expr1).toString(), canonicalizeTermOrder(expr2).toString());
}

} // namespace svmp::FE::forms::tensor

