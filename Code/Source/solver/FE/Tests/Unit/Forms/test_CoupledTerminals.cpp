/**
 * @file test_CoupledTerminals.cpp
 * @brief Unit tests for symbolic coupled-BC FormExpr terminals
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"

using svmp::FE::Real;

TEST(FormExprCoupledTerminals, BoundaryIntegralSymbol_StoresMarkerNameAndChild)
{
    using namespace svmp::FE::forms;

    const auto one = FormExpr::constant(Real(1.0));
    const auto sym = FormExpr::boundaryIntegral(one, /*boundary_marker=*/7, /*name=*/"Q");

    ASSERT_TRUE(sym.isValid());
    ASSERT_NE(sym.node(), nullptr);
    EXPECT_EQ(sym.node()->type(), FormExprType::BoundaryFunctionalSymbol);

    const auto marker = sym.node()->boundaryMarker();
    ASSERT_TRUE(marker.has_value());
    EXPECT_EQ(*marker, 7);

    const auto name = sym.node()->symbolName();
    ASSERT_TRUE(name.has_value());
    EXPECT_EQ(*name, "Q");

    const auto kids = sym.node()->children();
    ASSERT_EQ(kids.size(), 1u);
    ASSERT_NE(kids[0], nullptr);
    EXPECT_EQ(kids[0]->type(), FormExprType::Constant);
}

TEST(FormExprCoupledTerminals, AuxiliaryStateSymbol_StoresName)
{
    using namespace svmp::FE::forms;

    const auto sym = FormExpr::auxiliaryState("X");
    ASSERT_TRUE(sym.isValid());
    ASSERT_NE(sym.node(), nullptr);
    EXPECT_EQ(sym.node()->type(), FormExprType::AuxiliaryStateSymbol);

    const auto name = sym.node()->symbolName();
    ASSERT_TRUE(name.has_value());
    EXPECT_EQ(*name, "X");

    const auto kids = sym.node()->children();
    EXPECT_TRUE(kids.empty());
}

