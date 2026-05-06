/**
 * @file test_BoundaryConditionHelpers.cpp
 * @brief Unit tests for Forms boundary-condition helper APIs
 */

#include <gtest/gtest.h>

#include "Forms/BoundaryConditions.h"
#include "Forms/FormCompiler.h"
#include "Forms/NitscheBC.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/HDivSpace.h"

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>

using svmp::FE::ElementType;

namespace {

struct NeumannBC {
    int boundary_marker{-1};
    svmp::FE::Real flux{0.0};
};

struct RobinBC {
    int boundary_marker{-1};
    svmp::FE::Real alpha{0.0};
    svmp::FE::Real rhs{0.0};
};

struct DirichletBC {
    int boundary_marker{-1};
    svmp::FE::Real value{0.0};
};

struct NeumannBCValue {
    int boundary_marker{-1};
    svmp::FE::forms::bc::ScalarValue flux{};
};

struct RobinBCValue {
    int boundary_marker{-1};
    svmp::FE::forms::bc::ScalarValue alpha{};
    svmp::FE::forms::bc::ScalarValue rhs{};
};

struct DirichletBCValue {
    int boundary_marker{-1};
    svmp::FE::forms::bc::ScalarValue value{};
};

} // namespace

TEST(FormsBoundaryConditions, ToScalarExpr_ConvertsCommonScalarValueTypes)
{
    using svmp::FE::forms::FormExprType;

    const auto c = svmp::FE::forms::bc::toScalarExpr(2.5, "unused");
    ASSERT_TRUE(c.isValid());
    ASSERT_NE(c.node(), nullptr);
    EXPECT_EQ(c.node()->type(), FormExprType::Constant);
    ASSERT_TRUE(c.node()->constantValue().has_value());
    EXPECT_DOUBLE_EQ(*c.node()->constantValue(), 2.5);

    svmp::FE::forms::ScalarCoefficient s =
        [](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z) { return x + y + z; };
    const auto sc = svmp::FE::forms::bc::toScalarExpr(s, "spatial");
    ASSERT_TRUE(sc.isValid());
    ASSERT_NE(sc.node(), nullptr);
    EXPECT_EQ(sc.node()->type(), FormExprType::Coefficient);
    EXPECT_EQ(sc.toString(), "spatial");
    ASSERT_NE(sc.node()->scalarCoefficient(), nullptr);
    EXPECT_EQ((*sc.node()->scalarCoefficient())(1.0, 2.0, 3.0), 6.0);

    svmp::FE::forms::TimeScalarCoefficient t =
        [](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real time) {
            return x + y + z + time;
        };
    const auto tc = svmp::FE::forms::bc::toScalarExpr(t, "time");
    ASSERT_TRUE(tc.isValid());
    ASSERT_NE(tc.node(), nullptr);
    EXPECT_EQ(tc.node()->type(), FormExprType::Coefficient);
    EXPECT_EQ(tc.toString(), "time");
    ASSERT_NE(tc.node()->timeScalarCoefficient(), nullptr);
    EXPECT_EQ((*tc.node()->timeScalarCoefficient())(1.0, 2.0, 3.0, 4.0), 10.0);
}

TEST(FormsBoundaryConditions, ScalarValueConstantHelpersIdentifyConstantZeros)
{
    const svmp::FE::forms::bc::ScalarValue zero{svmp::FE::Real{0.0}};
    const svmp::FE::forms::bc::ScalarValue nonzero{svmp::FE::Real{2.0}};
    const svmp::FE::forms::bc::ScalarValue coefficient{
        svmp::FE::forms::ScalarCoefficient(
            [](svmp::FE::Real, svmp::FE::Real, svmp::FE::Real) { return 0.0; })};

    EXPECT_TRUE(svmp::FE::forms::bc::isConstantScalarValue(zero));
    EXPECT_TRUE(svmp::FE::forms::bc::isZeroConstantScalarValue(zero));
    EXPECT_TRUE(svmp::FE::forms::bc::isConstantScalarValue(nonzero));
    EXPECT_FALSE(svmp::FE::forms::bc::isZeroConstantScalarValue(nonzero));
    EXPECT_FALSE(svmp::FE::forms::bc::isConstantScalarValue(coefficient));
    EXPECT_FALSE(svmp::FE::forms::bc::isZeroConstantScalarValue(coefficient));
}

TEST(FormsBoundaryConditions, ToVectorExpr_GeneratesStableComponentNames)
{
    std::array<svmp::FE::forms::bc::ScalarValue, 3> values{
        svmp::FE::Real{1.0},
        svmp::FE::forms::ScalarCoefficient(
            [](svmp::FE::Real, svmp::FE::Real, svmp::FE::Real) { return 2.0; }),
        svmp::FE::Real{3.0}};

    const auto indexed =
        svmp::FE::forms::bc::toVectorExpr(values, 2, "uD", 7);
    ASSERT_EQ(indexed.size(), 2u);
    EXPECT_DOUBLE_EQ(*indexed[0].node()->constantValue(), 1.0);
    EXPECT_EQ(indexed[1].toString(), "uD_7_1");

    const auto component =
        svmp::FE::forms::bc::toVectorExpr(
            values, 2, "uD", 7, svmp::FE::forms::bc::ComponentValueNameStyle::Component);
    ASSERT_EQ(component.size(), 2u);
    EXPECT_EQ(component[1].toString(), "uD_7_c1");
}

TEST(FormsBoundaryConditions, ComponentEssentialBC_GeneratesSparseStrongConstraints)
{
    std::vector<std::pair<int, svmp::FE::forms::FormExpr>> components;
    components.emplace_back(0, svmp::FE::forms::FormExpr::constant(1.0));
    components.emplace_back(2, svmp::FE::forms::FormExpr::constant(3.0));

    svmp::FE::forms::bc::ComponentEssentialBC bc(
        /*boundary_marker=*/5, std::move(components), "u", "unit");

    const auto strong = bc.getStrongConstraints(/*field_id=*/11);
    ASSERT_EQ(strong.size(), 2u);
    EXPECT_EQ(strong[0].field, 11);
    EXPECT_EQ(strong[0].boundary_marker, 5);
    EXPECT_EQ(strong[0].component, 0);
    EXPECT_DOUBLE_EQ(*strong[0].value.node()->constantValue(), 1.0);
    EXPECT_EQ(strong[1].component, 2);
    EXPECT_DOUBLE_EQ(*strong[1].value.node()->constantValue(), 3.0);

    const auto metadata = bc.analysisMetadata(/*field_id=*/11, nullptr);
    ASSERT_EQ(metadata.size(), 2u);
    EXPECT_EQ(metadata[0].component, 0);
    EXPECT_EQ(metadata[1].component, 2);
}

TEST(FormsBoundaryConditions, ApplyNeumann_AddsBoundaryTermsWithMarkers)
{
    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/1);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = (u * v).dx();

    const std::array<NeumannBC, 2> bcs = {NeumannBC{.boundary_marker = 1, .flux = 2.0},
                                         NeumannBC{.boundary_marker = 3, .flux = 4.0}};

    residual = svmp::FE::forms::bc::applyNeumann(
        residual, v, std::span<const NeumannBC>(bcs),
        [](const NeumannBC& bc, std::size_t) { return svmp::FE::forms::FormExpr::constant(bc.flux); });

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    EXPECT_TRUE(ir.hasCellTerms());
    EXPECT_TRUE(ir.hasBoundaryTerms());

    std::vector<int> markers;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        markers.push_back(term.boundary_marker);
    }

    std::sort(markers.begin(), markers.end());
    EXPECT_EQ(markers.size(), 2u);
    EXPECT_EQ(markers[0], 1);
    EXPECT_EQ(markers[1], 3);
}

TEST(FormsBoundaryConditions, ApplyNeumannValue_AddsBoundaryTermsWithMarkers)
{
    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/1);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = (u * v).dx();

    const std::array<NeumannBCValue, 2> bcs = {
        NeumannBCValue{.boundary_marker = 1, .flux = svmp::FE::Real(2.0)},
        NeumannBCValue{
            .boundary_marker = 3,
            .flux = svmp::FE::forms::ScalarCoefficient(
                [](svmp::FE::Real, svmp::FE::Real, svmp::FE::Real) { return 4.0; }),
        },
    };

    residual = svmp::FE::forms::bc::applyNeumannValue(
        residual, v, std::span<const NeumannBCValue>(bcs), &NeumannBCValue::flux, "neumann");

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    EXPECT_TRUE(ir.hasCellTerms());
    EXPECT_TRUE(ir.hasBoundaryTerms());

    std::vector<int> markers;
    std::vector<std::string> dbg;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        markers.push_back(term.boundary_marker);
        dbg.push_back(term.debug_string);
    }

    std::sort(markers.begin(), markers.end());
    EXPECT_EQ(markers.size(), 2u);
    EXPECT_EQ(markers[0], 1);
    EXPECT_EQ(markers[1], 3);

    // Only the function-valued BC requires an auto-generated coefficient name.
    bool saw_named = false;
    for (const auto& s : dbg) {
        if (s.find("neumann_3_1") != std::string::npos) {
            saw_named = true;
            break;
        }
    }
    EXPECT_TRUE(saw_named);
}

TEST(FormsBoundaryConditions, ApplyRobin_AddsTwoBoundaryTermsPerMarker)
{
    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/1);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = (u * v).dx();

    const std::array<RobinBC, 1> bcs = {RobinBC{.boundary_marker = 5, .alpha = 3.0, .rhs = 7.0}};

    residual = svmp::FE::forms::bc::applyRobin(
        residual, u, v, std::span<const RobinBC>(bcs),
        [](const RobinBC& bc, std::size_t) { return svmp::FE::forms::FormExpr::constant(bc.alpha); },
        [](const RobinBC& bc, std::size_t) { return svmp::FE::forms::FormExpr::constant(bc.rhs); });

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_5 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker == 5) {
            ++count_marker_5;
        }
    }

    EXPECT_EQ(count_marker_5, 2);
}

TEST(FormsBoundaryConditions, ApplyRobinValue_AddsTwoBoundaryTermsPerMarker)
{
    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/1);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = (u * v).dx();

    const std::array<RobinBCValue, 1> bcs = {RobinBCValue{
        .boundary_marker = 5,
        .alpha = svmp::FE::forms::ScalarCoefficient(
            [](svmp::FE::Real, svmp::FE::Real, svmp::FE::Real) { return 3.0; }),
        .rhs = svmp::FE::forms::ScalarCoefficient(
            [](svmp::FE::Real, svmp::FE::Real, svmp::FE::Real) { return 7.0; }),
    }};

    residual = svmp::FE::forms::bc::applyRobinValue(residual,
                                                    u,
                                                    v,
                                                    std::span<const RobinBCValue>(bcs),
                                                    &RobinBCValue::alpha,
                                                    "alpha",
                                                    &RobinBCValue::rhs,
                                                    "rhs");

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_5 = 0;
    bool saw_alpha = false;
    bool saw_rhs = false;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker != 5) continue;
        ++count_marker_5;
        saw_alpha = saw_alpha || (term.debug_string.find("alpha_5_0") != std::string::npos);
        saw_rhs = saw_rhs || (term.debug_string.find("rhs_5_0") != std::string::npos);
    }

    EXPECT_EQ(count_marker_5, 2);
    EXPECT_TRUE(saw_alpha);
    EXPECT_TRUE(saw_rhs);
}

TEST(FormsBoundaryConditions, NormalComponentHelperBuildsScalarBoundaryTrace) {
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto expr = svmp::FE::forms::bc::normalComponent(u);

    ASSERT_TRUE(expr.isValid());
    EXPECT_TRUE(expr.toString().find("n") != std::string::npos);
}

TEST(FormsBoundaryConditions, TraceLoadBC_AddsBoundaryTermOnNormalTrace) {
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = svmp::FE::forms::dot(u, v).dx();
    svmp::FE::forms::bc::TraceLoadBC bc(4, svmp::FE::forms::FormExpr::constant(2.0));
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_4 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker == 4) {
            ++count_marker_4;
        }
    }

    EXPECT_EQ(count_marker_4, 1);
}

TEST(FormsBoundaryConditions, TraceRobinBC_AddsTwoBoundaryTermsOnNormalTrace) {
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = svmp::FE::forms::dot(u, v).dx();
    svmp::FE::forms::bc::TraceRobinBC bc(
        5, svmp::FE::forms::FormExpr::constant(3.0), svmp::FE::forms::FormExpr::constant(7.0));
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_5 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker == 5) {
            ++count_marker_5;
        }
    }

    EXPECT_EQ(count_marker_5, 2);
}

TEST(FormsBoundaryConditions, ApplyTraceInequality_AddsBoundaryTermOnNormalTrace) {
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = svmp::FE::forms::dot(u, v).dx();
    svmp::FE::forms::bc::TraceInequalityOptions opts;
    opts.sense = svmp::FE::forms::bc::TraceInequalitySense::LessEqual;
    residual = svmp::FE::forms::bc::applyTraceInequality(std::move(residual),
                                                         u,
                                                         v,
                                                         6,
                                                         svmp::FE::forms::FormExpr::constant(1.0),
                                                         svmp::FE::forms::FormExpr::constant(3.0),
                                                         opts);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_6 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker == 6) {
            ++count_marker_6;
        }
    }

    EXPECT_EQ(count_marker_6, 1);
}

TEST(FormsBoundaryConditions, TraceInequalityBC_SmoothLinearizationCompiles) {
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = svmp::FE::forms::dot(u, v).dx();
    svmp::FE::forms::bc::TraceInequalityOptions opts;
    opts.linearization = svmp::FE::forms::bc::TraceInequalityLinearization::Smooth;
    opts.smoothing_epsilon = svmp::FE::forms::FormExpr::constant(1e-3);
    svmp::FE::forms::bc::TraceInequalityBC bc(
        7,
        svmp::FE::forms::FormExpr::constant(0.0),
        svmp::FE::forms::FormExpr::constant(2.0),
        opts);
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_7 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker == 7) {
            ++count_marker_7;
        }
    }

    EXPECT_EQ(count_marker_7, 1);
}

TEST(FormsBoundaryConditions, InterfaceNormalComponentHelperBuildsScalarInterfaceTrace) {
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto expr =
        svmp::FE::forms::bc::interfaceNormalComponent(
            u, svmp::FE::forms::bc::InterfaceTraceReduction::Jump);

    ASSERT_TRUE(expr.isValid());
    const auto s = expr.toString();
    EXPECT_TRUE(s.find("n") != std::string::npos);
    EXPECT_TRUE(s.find("+") != std::string::npos || s.find("-") != std::string::npos);
}

TEST(FormsBoundaryConditions, InterfaceTraceLoadBC_AddsInterfaceTermOnNormalTrace) {
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = svmp::FE::forms::dot(u, v).dx();
    svmp::FE::forms::bc::InterfaceTraceLoadBC bc(
        12,
        svmp::FE::forms::FormExpr::constant(2.0),
        svmp::FE::forms::bc::ScalarTraceOperator::NormalComponent,
        svmp::FE::forms::bc::InterfaceTraceReduction::Minus);
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_12 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::InterfaceFace) continue;
        if (term.interface_marker == 12) {
            ++count_marker_12;
        }
    }

    EXPECT_EQ(count_marker_12, 1);
}

TEST(FormsBoundaryConditions, InterfaceTraceRobinBC_AddsTwoInterfaceTermsOnNormalTrace) {
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = svmp::FE::forms::dot(u, v).dx();
    svmp::FE::forms::bc::InterfaceTraceRobinBC bc(
        13, svmp::FE::forms::FormExpr::constant(3.0), svmp::FE::forms::FormExpr::constant(7.0));
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_13 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::InterfaceFace) continue;
        if (term.interface_marker == 13) {
            ++count_marker_13;
        }
    }

    EXPECT_EQ(count_marker_13, 2);
}

TEST(FormsBoundaryConditions, InterfaceTraceJumpPenaltyBC_AddsPenaltyTermsOnInterfaceJump) {
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = svmp::FE::forms::dot(u, v).dx();
    svmp::FE::forms::bc::InterfaceTraceJumpPenaltyBC bc(
        14, svmp::FE::forms::FormExpr::constant(5.0), svmp::FE::forms::FormExpr::constant(0.0));
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_14 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::InterfaceFace) continue;
        if (term.interface_marker == 14) {
            ++count_marker_14;
        }
    }

    EXPECT_EQ(count_marker_14, 2);
}

TEST(FormsBoundaryConditions, ApplyTraceNitsche_AddsThreeBoundaryTermsOnIdentityTrace)
{
    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/2);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");
    const auto n = svmp::FE::forms::FormExpr::normal();

    auto residual = (u * v).dx();
    residual = svmp::FE::forms::bc::applyTraceNitsche(std::move(residual),
                                                      u,
                                                      v,
                                                      /*boundary_marker=*/15,
                                                      svmp::FE::forms::FormExpr::constant(1.0),
                                                      inner(grad(u), n),
                                                      inner(grad(v), n),
                                                      svmp::FE::forms::FormExpr::constant(1.0) /
                                                          svmp::FE::forms::h(),
                                                      svmp::FE::forms::bc::ScalarTraceOperator::Identity);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_15 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker == 15) {
            ++count_marker_15;
        }
    }

    EXPECT_EQ(count_marker_15, 3);
}

TEST(FormsBoundaryConditions, ApplyInterfaceTraceNitsche_AddsThreeInterfaceTermsOnTraceJump)
{
    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/1);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");
    const auto n_minus = svmp::FE::forms::FormExpr::normal().minus();

    auto residual = (u * v).dx();
    residual = svmp::FE::forms::bc::applyInterfaceTraceNitsche(
        std::move(residual),
        u,
        v,
        /*interface_marker=*/16,
        svmp::FE::forms::FormExpr::constant(0.0),
        inner(avg(grad(u)), n_minus),
        inner(avg(grad(v)), n_minus),
        avg(svmp::FE::forms::FormExpr::constant(1.0) / svmp::FE::forms::hNormal()),
        svmp::FE::forms::bc::ScalarTraceOperator::Identity,
        svmp::FE::forms::bc::InterfaceTraceReduction::Jump);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_16 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::InterfaceFace) continue;
        if (term.interface_marker == 16) {
            ++count_marker_16;
        }
    }

    EXPECT_EQ(count_marker_16, 3);
}

TEST(FormsBoundaryConditions, TraceNitscheBC_AddsThreeBoundaryTermsOnNormalTrace)
{
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = svmp::FE::forms::dot(u, v).dx();
    svmp::FE::forms::bc::TraceNitscheBC bc(
        17,
        svmp::FE::forms::FormExpr::constant(0.0),
        svmp::FE::forms::bc::normalComponent(u),
        svmp::FE::forms::bc::normalComponent(v),
        svmp::FE::forms::FormExpr::constant(4.0),
        svmp::FE::forms::bc::ScalarTraceOperator::NormalComponent);
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_17 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker == 17) {
            ++count_marker_17;
        }
    }

    EXPECT_EQ(count_marker_17, 3);
}

TEST(FormsBoundaryConditions, InterfaceTraceNitscheBC_AddsThreeInterfaceTermsOnNormalTraceJump)
{
    auto space = svmp::FE::spaces::HDivSpace(ElementType::Tetra4, /*order=*/0);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");
    const auto n_minus = svmp::FE::forms::FormExpr::normal().minus();

    auto residual = svmp::FE::forms::dot(u, v).dx();
    svmp::FE::forms::bc::InterfaceTraceNitscheBC bc(
        18,
        svmp::FE::forms::FormExpr::constant(0.0),
        inner(avg(u), n_minus),
        inner(avg(v), n_minus),
        svmp::FE::forms::FormExpr::constant(6.0),
        svmp::FE::forms::bc::ScalarTraceOperator::NormalComponent,
        svmp::FE::forms::bc::InterfaceTraceReduction::Jump);
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int count_marker_18 = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::InterfaceFace) continue;
        if (term.interface_marker == 18) {
            ++count_marker_18;
        }
    }

    EXPECT_EQ(count_marker_18, 3);
}

TEST(FormsBoundaryConditions, MakeStrongDirichletList_ThrowsOnInvalidMarker)
{
    const std::array<DirichletBC, 1> bcs = {DirichletBC{.boundary_marker = -1, .value = 1.0}};

    EXPECT_THROW(
        (void)svmp::FE::forms::bc::makeStrongDirichletList(
            /*field=*/0,
            std::span<const DirichletBC>(bcs),
            [](const DirichletBC& bc, std::size_t) { return svmp::FE::forms::FormExpr::constant(bc.value); },
            "u"),
        std::invalid_argument);
}

TEST(FormsBoundaryConditions, MakeStrongDirichletListValue_ThrowsOnInvalidMarker)
{
    const std::array<DirichletBCValue, 1> bcs = {DirichletBCValue{.boundary_marker = -1, .value = 1.0}};

    EXPECT_THROW((void)svmp::FE::forms::bc::makeStrongDirichletListValue(
                     /*field=*/0,
                     std::span<const DirichletBCValue>(bcs),
                     &DirichletBCValue::value,
                     "dirichlet",
                     "u"),
                 std::invalid_argument);
}

TEST(FormsBoundaryConditions, MakeStrongDirichletListValue_GeneratesStableNames)
{
    const std::array<DirichletBCValue, 2> bcs = {
        DirichletBCValue{
            .boundary_marker = 1,
            .value = svmp::FE::forms::ScalarCoefficient(
                [](svmp::FE::Real, svmp::FE::Real, svmp::FE::Real) { return 9.0; }),
        },
        DirichletBCValue{
            .boundary_marker = 2,
            .value = svmp::FE::forms::ScalarCoefficient(
                [](svmp::FE::Real, svmp::FE::Real, svmp::FE::Real) { return 3.0; }),
        },
    };

    const auto out = svmp::FE::forms::bc::makeStrongDirichletListValue(
        /*field=*/0, std::span<const DirichletBCValue>(bcs), &DirichletBCValue::value, "dirichlet", "u");

    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].boundary_marker, 1);
    EXPECT_EQ(out[0].value.toString(), "dirichlet_1_0");
    EXPECT_EQ(out[1].boundary_marker, 2);
    EXPECT_EQ(out[1].value.toString(), "dirichlet_2_1");
}

TEST(FormsBoundaryConditions, ApplyNitscheDirichletPoissonValue_AddsBoundaryTermsAndInfersRequiredData)
{
    using svmp::FE::assembly::RequiredData;

    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/2);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = (u * v).dx();
    const auto k = svmp::FE::forms::FormExpr::constant(1.0);

    const std::array<DirichletBCValue, 1> bcs = {DirichletBCValue{
        .boundary_marker = 4,
        .value = svmp::FE::forms::ScalarCoefficient(
            [](svmp::FE::Real, svmp::FE::Real, svmp::FE::Real) { return 1.0; }),
    }};

    svmp::FE::forms::bc::NitscheDirichletOptions opts;
    opts.gamma = 5.0;
    opts.variant = svmp::FE::forms::bc::NitscheVariant::Symmetric;
    opts.scale_with_p = true;

    residual = svmp::FE::forms::bc::applyNitscheDirichletPoissonValue(
        residual,
        k,
        u,
        v,
        std::span<const DirichletBCValue>(bcs),
        &DirichletBCValue::value,
        "nitsche",
        opts);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    EXPECT_TRUE(ir.hasCellTerms());
    EXPECT_TRUE(ir.hasBoundaryTerms());
    EXPECT_TRUE(svmp::FE::assembly::hasFlag(ir.requiredData(), RequiredData::Normals));
    EXPECT_TRUE(svmp::FE::assembly::hasFlag(ir.requiredData(), RequiredData::EntityMeasures));

    int count_marker_4 = 0;
    bool saw_named_value = false;
    for (const auto& term : ir.terms()) {
        if (term.domain != svmp::FE::forms::IntegralDomain::Boundary) continue;
        if (term.boundary_marker != 4) continue;
        ++count_marker_4;
        saw_named_value = saw_named_value || (term.debug_string.find("nitsche_4_0") != std::string::npos);
    }

    EXPECT_EQ(count_marker_4, 3);
    EXPECT_TRUE(saw_named_value);
}

TEST(FormsBoundaryConditions, ApplyNitscheDirichletPoissonValue_ThrowsOnInvalidMarker)
{
    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/1);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    const auto residual0 = (u * v).dx();
    const auto k = svmp::FE::forms::FormExpr::constant(1.0);

    const std::array<DirichletBCValue, 1> bcs = {DirichletBCValue{.boundary_marker = -1, .value = 1.0}};

    EXPECT_THROW((void)svmp::FE::forms::bc::applyNitscheDirichletPoissonValue(
                     residual0,
                     k,
                     u,
                     v,
                     std::span<const DirichletBCValue>(bcs),
                     &DirichletBCValue::value,
                     "nitsche"),
                 std::invalid_argument);
}
