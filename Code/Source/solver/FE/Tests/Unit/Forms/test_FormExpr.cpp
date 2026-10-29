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
#include "Forms/BoundaryConditions.h"
#include "Forms/ResidualContributions.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

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

TEST(ResidualContributions, InterleavedFieldsKeepExactLeftFoldAndSelectedObservations)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto v = TestField(17, space, "v");
    const auto q = TestField(3, space, "q");
    const auto seed = (2.0 * v).dx();
    const auto a = (3.0 * v).ds(4);
    const auto b = (5.0 * q).dx();
    const auto grouped = ((7.0 * v) + (11.0 * v)).dI(9);
    FormExpr first = seed;
    FormExpr second;
    FormExpr observed;
    FormExpr seeded_observation = seed;
    FormExpr unselected = b;
    ResidualContributions rows({
        {17, first, {&observed, nullptr, &seeded_observation}},
        {3, second},
    });

    rows.append(17, a);
    const auto first_add = first.nodeShared();
    EXPECT_EQ(first.node()->children(), (std::vector<const FormExprNode*>{seed.node(), a.node()}));
    EXPECT_EQ(observed.nodeShared(), a.nodeShared());
    rows.append(3, b);
    rows.append(17, grouped);

    EXPECT_EQ(first.node()->type(), FormExprType::Add);
    EXPECT_EQ(first.node()->children(), (std::vector<const FormExprNode*>{first_add.get(), grouped.node()}));
    EXPECT_EQ(second.nodeShared(), b.nodeShared());
    EXPECT_EQ(observed.node()->children(), (std::vector<const FormExprNode*>{a.node(), grouped.node()}));
    ASSERT_EQ(seeded_observation.node()->children().size(), 2u);
    EXPECT_EQ(seeded_observation.node()->children()[1], grouped.node());
    EXPECT_EQ(seeded_observation.node()->children()[0]->children(),
              (std::vector<const FormExprNode*>{seed.node(), a.node()}));
    EXPECT_EQ(unselected.nodeShared(), b.nodeShared());
}

TEST(ResidualContributions, CopiesBindingsAndObservationLists)
{
    FormExpr residual;
    FormExpr observation;
    ResidualContributionRow binding{12, residual, {&observation}};
    ResidualContributions rows({binding});
    binding.observations.clear();
    const auto term = FormExpr::constant(2.0);
    rows.append(12, term);
    EXPECT_EQ(residual.nodeShared(), term.nodeShared());
    EXPECT_EQ(observation.nodeShared(), term.nodeShared());
}

TEST(ResidualContributions, CanonicalGeneratedRouteRetainsExactUnscaledAdditiveAnchor)
{
    auto scalar = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace vector(scalar, 3);
    const auto u = StateField(29, vector, "u");
    const auto v = TestField(29, vector, "v");
    const auto zero = FormExpr::asVector({FormExpr::constant(0.0), FormExpr::constant(0.0), FormExpr::constant(0.0)});
    const auto route = bc::buildGeneratedBoundaryNitscheTraceTerms(
        u, v, zero, bc::symmetricGradientForm(FormExpr::constant(2.5)), 4, 19);
    const auto& anchor = route.binding.routeAnchor();
    ASSERT_EQ(route.route_contribution.nodeShared(), anchor.nodeShared());
    const auto seed = inner(u, v).dx();
    const auto tail = inner(u, v).ds(6);
    FormExpr residual = seed;
    FormExpr observed;
    ResidualContributions rows({{29, residual, {&observed}}});
    rows.append(29, route.route_contribution);
    rows.append(29, tail);

    ASSERT_EQ(residual.node()->type(), FormExprType::Add);
    const auto* prefix = residual.node()->children()[0];
    ASSERT_EQ(prefix->type(), FormExprType::Add);
    EXPECT_EQ(prefix->children(), (std::vector<const FormExprNode*>{seed.node(), anchor.node()}));
    EXPECT_EQ(residual.node()->children()[1], tail.node());
    EXPECT_EQ(observed.node()->type(), FormExprType::Add);
    EXPECT_EQ(observed.node()->children(), (std::vector<const FormExprNode*>{anchor.node(), tail.node()}));
    EXPECT_EQ(route.binding.velocityField(), 29);
}

TEST(ResidualContributions, SnapshotsTermsAliasingAnyBorrowedTarget)
{
    const auto seed = FormExpr::constant(2.0);
    FormExpr residual = seed;
    FormExpr observation;
    FormExpr other;
    ResidualContributions rows({{5, residual, {&observation}}, {20, other}});
    rows.append(5, residual);
    EXPECT_EQ(residual.node()->children(), (std::vector<const FormExprNode*>{seed.node(), seed.node()}));
    EXPECT_EQ(observation.nodeShared(), seed.nodeShared());
    const auto observation_before = observation.nodeShared();
    rows.append(5, observation);
    EXPECT_EQ(observation.node()->children(),
              (std::vector<const FormExprNode*>{observation_before.get(), observation_before.get()}));
    rows.append(20, residual);
    EXPECT_EQ(other.nodeShared(), residual.nodeShared());
}

TEST(ResidualContributions, RejectsInvalidDuplicateAndUnknownIdsWithoutUndoingEarlierAppends)
{
    FormExpr first;
    FormExpr second;
    EXPECT_THROW((ResidualContributions{{INVALID_FIELD_ID, first}}), std::invalid_argument);
    EXPECT_THROW((ResidualContributions{{4, first}, {4, second}}), std::invalid_argument);
    ResidualContributions rows({{4, first}, {31, second}});
    const auto term = FormExpr::constant(3.0);
    rows.append(4, term);
    EXPECT_THROW(rows.append(8, term), std::invalid_argument);
    EXPECT_THROW(rows.append(INVALID_FIELD_ID, term), std::invalid_argument);
    EXPECT_EQ(first.nodeShared(), term.nodeShared());
    EXPECT_FALSE(second.isValid());
}

TEST(ResidualContributions, RejectsAllRepeatedTargetsBeforeMutation)
{
    FormExpr first;
    FormExpr second;
    FormExpr observation;
    EXPECT_THROW((ResidualContributions{{4, first, {&first}}}), std::invalid_argument);
    EXPECT_THROW((ResidualContributions{{4, first, {&observation, &observation}}}), std::invalid_argument);
    EXPECT_THROW((ResidualContributions{{4, first}, {5, first}}), std::invalid_argument);
    EXPECT_THROW((ResidualContributions{{4, first, {&second}}, {5, second}}), std::invalid_argument);
    EXPECT_THROW((ResidualContributions{{4, first}, {5, second, {&first}}}), std::invalid_argument);
    EXPECT_THROW((ResidualContributions{{4, first, {&observation}}, {5, second, {&observation}}}),
                 std::invalid_argument);
    EXPECT_FALSE(first.isValid());
    EXPECT_FALSE(second.isValid());
    EXPECT_FALSE(observation.isValid());
}

TEST(ResidualContributions, InvalidTermsRetainSumOfBehaviorAndProductionFirstProgress)
{
    FormExpr residual;
    FormExpr observation;
    const FormExpr invalid;
    ResidualContributions rows({{0, residual, {&observation}}});
    EXPECT_NO_THROW(rows.append(0, invalid));
    EXPECT_FALSE(residual.isValid());
    const auto seed = FormExpr::constant(4.0);
    residual = seed;
    EXPECT_THROW(rows.append(0, invalid), std::invalid_argument);
    EXPECT_EQ(residual.nodeShared(), seed.nodeShared());
    EXPECT_FALSE(observation.isValid());

    // The empty production target accepts the invalid handle before the seeded
    // observation rejects it, matching separate production-first sumOf calls.
    residual = {};
    observation = seed;
    EXPECT_THROW(rows.append(0, invalid), std::invalid_argument);
    EXPECT_FALSE(residual.isValid());
    EXPECT_EQ(observation.nodeShared(), seed.nodeShared());
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
