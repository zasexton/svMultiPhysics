/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormExpr.cpp
 * @brief Unit tests for FE/Forms expression vocabulary (AST)
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(FormExprTest, DefaultConstructionInvalid)
{
    FormExpr e;
    EXPECT_FALSE(e.isValid());
}

TEST(FormExprTest, TerminalsAndQueries)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto c = FormExpr::constant(2.0);
    const auto I = FormExpr::identity(3);
    const auto n = FormExpr::normal();

    EXPECT_TRUE(u.isValid());
    EXPECT_TRUE(v.isValid());
    EXPECT_TRUE(c.isValid());
    EXPECT_TRUE(I.isValid());
    EXPECT_TRUE(n.isValid());

    EXPECT_FALSE(u.hasTest());
    EXPECT_TRUE(u.hasTrial());
    EXPECT_TRUE(v.hasTest());
    EXPECT_FALSE(v.hasTrial());
    EXPECT_FALSE(c.hasTest());
    EXPECT_FALSE(c.hasTrial());

    EXPECT_EQ(u.node()->type(), FormExprType::TrialFunction);
    EXPECT_EQ(v.node()->type(), FormExprType::TestFunction);
    EXPECT_EQ(c.node()->type(), FormExprType::Constant);
    EXPECT_EQ(I.node()->type(), FormExprType::Identity);
    EXPECT_EQ(n.node()->type(), FormExprType::Normal);
}

TEST(FormExprTest, Coefficients)
{
    const auto f = [](Real x, Real y, Real z) { return x + 2.0 * y + 3.0 * z; };
    const auto g = [](Real /*x*/, Real /*y*/, Real /*z*/) { return std::array<Real, 3>{1.0, 2.0, 3.0}; };

    const auto cf = FormExpr::coefficient("f", f);
    const auto cg = FormExpr::coefficient("g", g);

    EXPECT_TRUE(cf.isValid());
    EXPECT_TRUE(cg.isValid());
    EXPECT_EQ(cf.node()->type(), FormExprType::Coefficient);
    EXPECT_EQ(cg.node()->type(), FormExprType::Coefficient);
}

TEST(FormExprTest, OperatorsAndMeasures)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto gu = grad(u);
    const auto gv = grad(v);
    const auto ju = jump(u);
    const auto av = avg(v);

    EXPECT_EQ(gu.node()->type(), FormExprType::Gradient);
    EXPECT_EQ(gv.node()->type(), FormExprType::Gradient);
    EXPECT_EQ(ju.node()->type(), FormExprType::Jump);
    EXPECT_EQ(av.node()->type(), FormExprType::Average);

    const auto expr = inner(gu, gv) - (u * v);
    EXPECT_TRUE(expr.hasTest());
    EXPECT_TRUE(expr.hasTrial());

    const auto dx_term = expr.dx();
    const auto ds_term = expr.ds(2);
    const auto dS_term = (inner(ju, ju)).dS();

    EXPECT_EQ(dx_term.node()->type(), FormExprType::CellIntegral);
    EXPECT_EQ(ds_term.node()->type(), FormExprType::BoundaryIntegral);
    EXPECT_EQ(dS_term.node()->type(), FormExprType::InteriorFaceIntegral);

    EXPECT_EQ(ds_term.node()->boundaryMarker().value_or(-1), 2);
}

TEST(FormExprTest, TimeDerivativeNode)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto dtu = dt(u, 2);

    ASSERT_TRUE(dtu.isValid());
    EXPECT_EQ(dtu.node()->type(), FormExprType::TimeDerivative);
    EXPECT_EQ(dtu.node()->timeDerivativeOrder().value_or(0), 2);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
