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

#include <optional>
#include <stdexcept>

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

TEST(FormExprTest, ComponentOfPackedAggregateReturnsPackedChild)
{
    const auto a = FormExpr::parameterRef(0);
    const auto b = FormExpr::parameterRef(1);
    const auto c = FormExpr::parameterRef(2);
    const auto d = FormExpr::parameterRef(3);

    const auto vector = FormExpr::asVector({a, b, c, d});
    EXPECT_EQ(vector.component(2).nodeShared(), c.nodeShared());

    const auto tensor = FormExpr::asTensor({{a, b}, {c, d}});
    EXPECT_EQ(tensor.component(1, 0).nodeShared(), c.nodeShared());

    EXPECT_EQ(vector.component(8).node()->type(), FormExprType::Component);
    EXPECT_EQ(tensor.component(2, 0).node()->type(), FormExprType::Component);
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

TEST(FormExprTest, ExteriorBoundaryMeasuresAreExplicitAndDistinct)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto integrand = u * v;

    const auto full =
        ExteriorBoundaryMeasure::fullPhysical(/*physical_boundary_marker=*/2);
    const auto active =
        ExteriorBoundaryMeasure::generatedActiveSubset(
            /*physical_boundary_marker=*/2,
            /*generated_active_boundary_marker=*/17);

    const auto full_term = integrand.dExteriorBoundary(full);
    ASSERT_EQ(full_term.node()->type(), FormExprType::BoundaryIntegral);
    EXPECT_EQ(full_term.node()->boundaryMarker().value_or(-1), 2);
    ASSERT_NE(full_term.node()->exteriorBoundaryMeasure(), nullptr);
    EXPECT_EQ(*full_term.node()->exteriorBoundaryMeasure(), full);
    EXPECT_TRUE(
        full_term.node()->exteriorBoundaryMeasure()->isFullPhysical());
    EXPECT_EQ(
        full_term.node()
            ->exteriorBoundaryMeasure()
            ->physicalBoundaryMarker(),
        2);
    EXPECT_EQ(
        full_term.node()
            ->exteriorBoundaryMeasure()
            ->generatedActiveBoundaryMarker(),
        -1);

    const auto active_term = integrand.dExteriorBoundary(active);
    ASSERT_EQ(active_term.node()->type(), FormExprType::InterfaceIntegral);
    EXPECT_EQ(active_term.node()->interfaceMarker().value_or(-1), 17);
    ASSERT_NE(active_term.node()->exteriorBoundaryMeasure(), nullptr);
    EXPECT_EQ(*active_term.node()->exteriorBoundaryMeasure(), active);
    EXPECT_TRUE(active_term.node()
                    ->exteriorBoundaryMeasure()
                    ->isGeneratedActiveSubset());
    EXPECT_EQ(
        active_term.node()
            ->exteriorBoundaryMeasure()
            ->physicalBoundaryMarker(),
        2);
    EXPECT_EQ(
        active_term.node()
            ->exteriorBoundaryMeasure()
            ->generatedActiveBoundaryMarker(),
        17);

    EXPECT_EQ(integrand.ds(2).node()->exteriorBoundaryMeasure(), nullptr);
    EXPECT_EQ(integrand.dI(17).node()->exteriorBoundaryMeasure(), nullptr);
    EXPECT_NE(full, active);
    EXPECT_NE(full_term.toString(), integrand.ds(2).toString());
    EXPECT_NE(active_term.toString(), integrand.dI(17).toString());
    EXPECT_NE(full_term.toString(), active_term.toString());
}

TEST(FormExprTest, ExteriorBoundaryMeasuresRejectInvalidMarkers)
{
    EXPECT_THROW(
        (void)ExteriorBoundaryMeasure::fullPhysical(-1),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ExteriorBoundaryMeasure::generatedActiveSubset(-1, 4),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ExteriorBoundaryMeasure::generatedActiveSubset(3, -1),
        std::invalid_argument);
}

TEST(FormExprTest, TransformNodesPreservesExteriorBoundaryMeasure)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto integrand = FormExpr::constant(2.0) * u * v;

    const auto replace_constant =
        [](const FormExprNode& node) -> std::optional<FormExpr> {
        if (node.type() == FormExprType::Constant &&
            node.constantValue().value_or(0.0) == 2.0) {
            return FormExpr::constant(3.0);
        }
        return std::nullopt;
    };

    const auto full =
        ExteriorBoundaryMeasure::fullPhysical(/*physical_boundary_marker=*/5);
    const auto full_original = integrand.dExteriorBoundary(full);
    const auto full_transformed =
        full_original.transformNodes(replace_constant);
    EXPECT_NE(full_transformed.nodeShared(), full_original.nodeShared());
    ASSERT_NE(
        full_transformed.node()->exteriorBoundaryMeasure(), nullptr);
    EXPECT_EQ(
        *full_transformed.node()->exteriorBoundaryMeasure(), full);
    EXPECT_EQ(
        full_transformed.node()->boundaryMarker().value_or(-1), 5);

    const auto active =
        ExteriorBoundaryMeasure::generatedActiveSubset(
            /*physical_boundary_marker=*/5,
            /*generated_active_boundary_marker=*/29);
    const auto active_original = integrand.dExteriorBoundary(active);
    const auto active_transformed =
        active_original.transformNodes(replace_constant);
    EXPECT_NE(active_transformed.nodeShared(), active_original.nodeShared());
    ASSERT_NE(
        active_transformed.node()->exteriorBoundaryMeasure(), nullptr);
    EXPECT_EQ(
        *active_transformed.node()->exteriorBoundaryMeasure(), active);
    EXPECT_EQ(
        active_transformed.node()->interfaceMarker().value_or(-1), 29);
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
