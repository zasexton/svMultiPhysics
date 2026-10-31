/**
 * @file test_BoundaryConditionHelpers.cpp
 * @brief Unit tests for Forms boundary-condition helper APIs
 */

#include <gtest/gtest.h>

#include "Forms/BoundaryConditions.h"
#include "Forms/FormCompiler.h"
#include "Forms/NitscheBC.h"
#include "Forms/PointEvaluator.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SystemManifest.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

using svmp::FE::ElementType;

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

TEST(FormsBoundaryConditions, NumericValidationKeepsFinitenessSeparateFromSign)
{
    namespace numeric = svmp::FE::numeric;
    using svmp::FE::Real;
    struct Case {
        Real value;
        bool finite;
        bool nonnegative;
        bool positive;
    };
    const Real infinity = std::numeric_limits<Real>::infinity();
    const std::array cases{
        Case{-1.0, true, false, false}, Case{-0.0, true, true, false},
        Case{0.0, true, true, false}, Case{1.0, true, true, true},
        Case{-infinity, false, false, false}, Case{infinity, false, true, true},
        Case{std::numeric_limits<Real>::quiet_NaN(), false, false, false}};
    for (const auto& c : cases) {
        EXPECT_EQ(numeric::isFinite(c.value), c.finite);
        EXPECT_EQ(numeric::isNonnegative(c.value), c.nonnegative);
        EXPECT_EQ(numeric::isPositive(c.value), c.positive);
    }
    EXPECT_NO_THROW(numeric::validate(infinity, numeric::isPositive, "positive required"));
    try {
        numeric::validate(infinity, numeric::isFinite, std::string("owner: finite required"));
        FAIL() << "Nonfinite value must be rejected";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "owner: finite required");
    }
}

TEST(FormsBoundaryConditions, ScalarLiteralValidationDoesNotSampleOrInterpretOtherVariants)
{
    namespace bc = svmp::FE::forms::bc;
    using svmp::FE::Real;
    int callback_calls = 0;
    int predicate_calls = 0;
    const auto predicate = [&](Real value) {
        ++predicate_calls;
        return svmp::FE::numeric::isFinite(value);
    };
    const auto invalid = std::numeric_limits<Real>::infinity();
    const std::array<bc::ScalarValue, 4> deferred{
        svmp::FE::forms::ScalarCoefficient([&](Real, Real, Real) {
            ++callback_calls;
            return invalid;
        }),
        svmp::FE::forms::TimeScalarCoefficient([&](Real, Real, Real, Real) {
            ++callback_calls;
            return invalid;
        }),
        svmp::FE::forms::FormExpr::constant(invalid), svmp::FE::forms::FormExpr{}};
    for (const auto& value : deferred) {
        EXPECT_NO_THROW(bc::validateScalarLiteral(value, predicate, "finite required"));
        try {
            (void)bc::requireScalarLiteral(value, "model: literal required");
            FAIL() << "Only the Real alternative is a scalar literal";
        } catch (const std::invalid_argument& error) {
            EXPECT_STREQ(error.what(), "model: literal required");
        }
    }
    EXPECT_EQ(callback_calls, 0);
    EXPECT_EQ(predicate_calls, 0);
    EXPECT_NO_THROW(bc::validateScalarLiteral(bc::ScalarValue{2.0}, predicate, "finite required"));
    EXPECT_EQ(predicate_calls, 1);
    EXPECT_THROW(bc::validateScalarLiteral(bc::ScalarValue{invalid}, predicate, "finite required"),
                 std::invalid_argument);
    EXPECT_EQ(predicate_calls, 2);
}

TEST(FormsBoundaryConditions, RequiringScalarLiteralPreservesItsValueWithoutNumericValidation)
{
    namespace bc = svmp::FE::forms::bc;
    using svmp::FE::Real;
    const auto infinity = std::numeric_limits<Real>::infinity();
    EXPECT_EQ(bc::requireScalarLiteral(bc::ScalarValue{infinity}, "literal required"), infinity);
    EXPECT_EQ(bc::requireScalarLiteral(bc::ScalarValue{-infinity}, "literal required"), -infinity);
    EXPECT_TRUE(std::isnan(bc::requireScalarLiteral(
        bc::ScalarValue{std::numeric_limits<Real>::quiet_NaN()}, "literal required")));
    EXPECT_TRUE(std::signbit(bc::requireScalarLiteral(bc::ScalarValue{-0.0}, "literal required")));
    EXPECT_DOUBLE_EQ(bc::requireScalarLiteral(bc::ScalarValue{-2.5}, "literal required"), -2.5);
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

TEST(FormsBoundaryConditions, ToVectorExpr_NamesUnmarkedComponentsByPrefix)
{
    std::array<svmp::FE::forms::bc::ScalarValue, 3> values{
        svmp::FE::Real{1.0},
        svmp::FE::forms::ScalarCoefficient(
            [](svmp::FE::Real, svmp::FE::Real, svmp::FE::Real) { return 2.0; }),
        svmp::FE::Real{3.0}};

    const auto named = svmp::FE::forms::bc::toVectorExpr(values, 3, "ns_body_force_");
    ASSERT_EQ(named.size(), 3u);
    EXPECT_DOUBLE_EQ(*named[0].node()->constantValue(), 1.0);
    EXPECT_EQ(named[1].toString(), "ns_body_force_1");
    EXPECT_DOUBLE_EQ(*named[2].node()->constantValue(), 3.0);
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

TEST(FormsBoundaryConditions, VectorBoundaryDataGivesTheSpelledOutConditions)
{
    namespace forms = svmp::FE::forms;
    namespace bc = forms::bc;
    using svmp::FE::Real;

    const forms::ScalarCoefficient g = [](Real x, Real y, Real) { return x - y; };
    bc::BoundaryData<bc::ComponentValues> data{
        .boundary_marker = 6,
        .value = {.values = {bc::ScalarValue{1.5}, bc::ScalarValue{g}, bc::ScalarValue{-2.0}}}};
    const std::array<bc::ScalarValue, 3> raw = data.value.values;

    const auto sameStrong = [](const bc::BoundaryCondition& a, const bc::BoundaryCondition& b) {
        const auto sa = a.getStrongConstraints(/*field_id=*/2);
        const auto sb = b.getStrongConstraints(/*field_id=*/2);
        ASSERT_EQ(sa.size(), sb.size());
        for (std::size_t i = 0; i < sa.size(); ++i) {
            EXPECT_EQ(sa[i].boundary_marker, sb[i].boundary_marker);
            EXPECT_EQ(sa[i].component, sb[i].component);
            EXPECT_EQ(sa[i].symbol, sb[i].symbol);
            EXPECT_EQ(svmp::FE::systems::formExprTreeHash(sa[i].value),
                      svmp::FE::systems::formExprTreeHash(sb[i].value));
            EXPECT_EQ(sa[i].value.toString(), sb[i].value.toString());
        }
        const auto ma = a.analysisMetadata(2, nullptr);
        const auto mb = b.analysisMetadata(2, nullptr);
        ASSERT_EQ(ma.size(), mb.size());
        for (std::size_t i = 0; i < ma.size(); ++i) {
            EXPECT_EQ(ma[i].source, mb[i].source);
        }
    };

    // Every component prescribed: EssentialBC with the named components.
    sameStrong(*bc::essentialBC(data, 3, "disp", "d"),
               bc::EssentialBC(6, bc::toVectorExpr(raw, 3, "disp", 6), "d"));
    sameStrong(*bc::essentialBC(data, 3, "disp", "d", bc::ComponentValueNameStyle::Component),
               bc::EssentialBC(6, bc::toVectorExpr(raw, 3, "disp", 6, bc::ComponentValueNameStyle::Component), "d"));

    // Selected components: ComponentEssentialBC.
    data.value.active = {true, false, true};
    sameStrong(*bc::componentEssentialBC(data, 3, "disp", "d", "Wall"),
               *bc::vectorComponentEssentialBC(raw, data.value.active, 3, 6, "disp", "d", "Wall"));
    EXPECT_THROW((void)bc::essentialBC(data, 3, "disp", "d"), std::invalid_argument);
    EXPECT_THROW((void)bc::naturalBC(data, 3, "traction"), std::invalid_argument);

    // A vector flux: NaturalBC of asVector of the named components.
    data.value.active = {true, true, true};
    auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 3);
    const auto u = forms::FormExpr::trialFunction(*vector, "u");
    const auto v = forms::FormExpr::testFunction(*vector, "v");
    auto typed = (inner(u, v)).dx();
    auto spelled = typed;
    bc::naturalBC(data, 3, "traction")->contributeToResidual(typed, u, v);
    bc::NaturalBC(6, forms::FormExpr::asVector(bc::toVectorExpr(raw, 3, "traction", 6)))
        .contributeToResidual(spelled, u, v);
    EXPECT_EQ(svmp::FE::systems::formExprTreeHash(typed), svmp::FE::systems::formExprTreeHash(spelled));
    EXPECT_EQ(typed.toString(), spelled.toString());

    // The scalar value names its callback after the prefix and the marker.
    const bc::BoundaryData<bc::ScalarValue> scalar_data{.boundary_marker = 9, .value = g};
    EXPECT_EQ(svmp::FE::systems::formExprTreeHash(bc::value(scalar_data, "q")),
              svmp::FE::systems::formExprTreeHash(bc::toScalarExpr(g, bc::markerValueName("q", 9))));
    EXPECT_THROW((void)bc::value(bc::BoundaryData<bc::ScalarValue>{}, "q"), std::invalid_argument);
}

TEST(FormsBoundaryConditions, NormalEssentialBCInstallsTheNormalComponentConstraint)
{
    namespace forms = svmp::FE::forms;
    forms::bc::NormalEssentialBC bc(/*boundary_marker=*/3, forms::FormExpr::constant(0.5));
    EXPECT_EQ(bc.boundaryMarker(), 3);
    EXPECT_FALSE(bc.hasWeakTerms());
    EXPECT_FALSE(bc.allowsMarkerSharing());
    EXPECT_TRUE(bc.getStrongConstraints(/*field_id=*/0).empty());
    const auto metadata = bc.analysisMetadata(/*field_id=*/0, nullptr);
    ASSERT_EQ(metadata.size(), 1u);
    EXPECT_EQ(metadata[0].trace_kind, svmp::FE::analysis::TraceKind::NormalComponent);
    EXPECT_EQ(metadata[0].enforcement_kind, svmp::FE::analysis::EnforcementKind::Strong);
    EXPECT_TRUE(metadata[0].anchors_rigid_body_translation);
    EXPECT_FALSE(metadata[0].anchors_rigid_body_rotation);

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 3);
    svmp::FE::systems::FESystem system(mesh);
    const auto u = system.addField({.name = "u", .space = vector, .components = 3});
    bc.installSystemConstraints(system, u);
    bool installed = false;
    for (const auto& line : svmp::FE::systems::constraintManifestLines(system)) {
        installed = installed || (line.find("NormalComponentConstraint") != std::string::npos &&
                                  line.find("markers=3") != std::string::npos);
    }
    EXPECT_TRUE(installed);
    EXPECT_THROW(forms::bc::NormalEssentialBC(-1), std::invalid_argument);
}

TEST(FormsBoundaryConditions, NaturalBCUsesGeneratedActiveBoundaryWhenProvided)
{
    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/1);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = (u * v).dx();
    svmp::FE::forms::bc::NaturalBC bc(
        /*boundary_marker=*/5,
        svmp::FE::forms::FormExpr::constant(2.0),
        /*generated_active_boundary_marker=*/105);
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);
    int generated_terms = 0;
    int whole_face_terms = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain == svmp::FE::forms::IntegralDomain::InterfaceFace &&
            term.interface_marker == 105) {
            ++generated_terms;
        }
        if (term.domain == svmp::FE::forms::IntegralDomain::Boundary &&
            term.boundary_marker == 5) {
            ++whole_face_terms;
        }
    }

    EXPECT_EQ(generated_terms, 1);
    EXPECT_EQ(whole_face_terms, 0);
}

TEST(FormsBoundaryConditions, RobinBCUsesGeneratedActiveBoundaryWhenProvided)
{
    auto space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/1);
    const auto u = svmp::FE::forms::FormExpr::trialFunction(space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(space, "v");

    auto residual = (u * v).dx();
    svmp::FE::forms::bc::RobinBC bc(
        /*boundary_marker=*/7,
        svmp::FE::forms::FormExpr::constant(3.0),
        svmp::FE::forms::FormExpr::constant(4.0),
        /*generated_active_boundary_marker=*/107);
    bc.contributeToResidual(residual, u, v);

    svmp::FE::forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);
    int generated_terms = 0;
    int whole_face_terms = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain == svmp::FE::forms::IntegralDomain::InterfaceFace &&
            term.interface_marker == 107) {
            ++generated_terms;
        }
        if (term.domain == svmp::FE::forms::IntegralDomain::Boundary &&
            term.boundary_marker == 7) {
            ++whole_face_terms;
        }
    }

    EXPECT_EQ(generated_terms, 2);
    EXPECT_EQ(whole_face_terms, 0);
}

TEST(FormsBoundaryConditions,
     LegacyIntegerBoundaryRoutesRemainUnqualified)
{
    namespace forms = svmp::FE::forms;

    auto space = svmp::FE::spaces::H1Space(
        ElementType::Tetra4, /*order=*/1);
    const auto u = forms::FormExpr::trialFunction(
        space, "u");
    const auto v = forms::FormExpr::testFunction(
        space, "v");
    const auto n = forms::FormExpr::normal();
    constexpr int marker = 12;
    auto legacy = (u * v).dx();
    legacy = legacy - (forms::FormExpr::constant(2.0) * v).ds(marker);
    forms::bc::NaturalBC natural(
        marker, forms::FormExpr::constant(3.0));
    natural.contributeToResidual(legacy, u, v);
    forms::bc::TraceLoadBC trace_load(
        marker,
        forms::FormExpr::constant(4.0),
        forms::bc::ScalarTraceOperator::Identity);
    trace_load.contributeToResidual(legacy, u, v);
    legacy = forms::bc::applyTraceNitsche(
        std::move(legacy),
        u,
        v,
        marker,
        forms::FormExpr::constant(0.0),
        inner(grad(u), n),
        inner(grad(v), n),
        forms::FormExpr::constant(1.0) /
            forms::h(),
        forms::bc::ScalarTraceOperator::Identity);

    const auto legacy_ir =
        forms::FormCompiler{}.compileResidual(legacy);
    int raw_terms = 0;
    for (const auto& term : legacy_ir.terms()) {
        if (term.domain ==
                forms::IntegralDomain::Boundary &&
            term.boundary_marker == marker) {
            ++raw_terms;
            EXPECT_FALSE(
                term.exterior_boundary_measure.has_value());
        }
    }
    EXPECT_EQ(raw_terms, 6);

    const auto full =
        forms::ExteriorBoundaryMeasure::fullPhysical(
            marker);
    auto explicit_full = (u * v).dx();
    forms::bc::NaturalBC full_natural(
        full, forms::FormExpr::constant(3.0));
    full_natural.contributeToResidual(
        explicit_full, u, v);
    forms::bc::TraceLoadBC full_trace_load(
        full,
        forms::FormExpr::constant(4.0),
        forms::bc::ScalarTraceOperator::Identity);
    full_trace_load.contributeToResidual(
        explicit_full, u, v);

    const auto full_ir =
        forms::FormCompiler{}.compileResidual(
            explicit_full);
    int explicit_terms = 0;
    for (const auto& term : full_ir.terms()) {
        if (term.domain !=
                forms::IntegralDomain::Boundary ||
            term.boundary_marker != marker) {
            continue;
        }
        ++explicit_terms;
        ASSERT_TRUE(
            term.exterior_boundary_measure.has_value());
        EXPECT_TRUE(
            term.exterior_boundary_measure
                ->isFullPhysical());
    }
    EXPECT_EQ(explicit_terms, 2);
}

TEST(FormsBoundaryConditions, GeneratedActiveBoundaryMarkerMustBeNonnegative)
{
    EXPECT_THROW(
        svmp::FE::forms::bc::NaturalBC(
            /*boundary_marker=*/3,
            svmp::FE::forms::FormExpr::constant(1.0),
            /*generated_active_boundary_marker=*/-1),
        std::invalid_argument);
    EXPECT_THROW(
        svmp::FE::forms::bc::RobinBC(
            /*boundary_marker=*/3,
            svmp::FE::forms::FormExpr::constant(1.0),
            svmp::FE::forms::FormExpr::constant(0.0),
            /*generated_active_boundary_marker=*/-1),
        std::invalid_argument);
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

TEST(FormsBoundaryConditions,
     TraceBoundaryWrappersUseGeneratedActiveBoundaryMeasures)
{
    namespace forms = svmp::FE::forms;

    auto space = svmp::FE::spaces::HDivSpace(
        ElementType::Tetra4, /*order=*/0);
    const auto u = forms::FormExpr::trialFunction(
        space, "u");
    const auto v = forms::FormExpr::testFunction(
        space, "v");
    constexpr int physical_marker = 9;
    constexpr int active_marker = 109;
    const auto measure =
        forms::ExteriorBoundaryMeasure::
            generatedActiveSubset(
                physical_marker, active_marker);

    auto residual = forms::dot(u, v).dx();
    auto load = forms::bc::makeTraceLoadBC(
        measure, forms::FormExpr::constant(2.0));
    auto robin = forms::bc::makeTraceRobinBC(
        measure,
        forms::FormExpr::constant(3.0),
        forms::FormExpr::constant(4.0));
    auto inequality = forms::bc::makeTraceInequalityBC(
        measure,
        forms::FormExpr::constant(0.0),
        forms::FormExpr::constant(5.0));
    load->contributeToResidual(residual, u, v);
    robin->contributeToResidual(residual, u, v);
    inequality->contributeToResidual(residual, u, v);

    const auto ir = forms::FormCompiler{}.compileResidual(
        residual);
    int active_terms = 0;
    int whole_face_terms = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain ==
                forms::IntegralDomain::InterfaceFace &&
            term.interface_marker == active_marker &&
            term.exterior_boundary_measure.has_value() &&
            term.exterior_boundary_measure
                    ->physicalBoundaryMarker() ==
                physical_marker) {
            ++active_terms;
        }
        if (term.domain ==
                forms::IntegralDomain::Boundary &&
            term.boundary_marker == physical_marker) {
            ++whole_face_terms;
        }
    }

    EXPECT_EQ(active_terms, 4);
    EXPECT_EQ(whole_face_terms, 0);
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

TEST(FormsBoundaryConditions,
     TraceNitschePathsUseGeneratedActiveBoundaryMeasures)
{
    namespace forms = svmp::FE::forms;

    auto space = svmp::FE::spaces::H1Space(
        ElementType::Tetra4, /*order=*/2);
    const auto u = forms::FormExpr::trialFunction(
        space, "u");
    const auto v = forms::FormExpr::testFunction(
        space, "v");
    const auto n = forms::FormExpr::normal();
    constexpr int physical_marker = 19;
    constexpr int active_marker = 119;
    const auto measure =
        forms::ExteriorBoundaryMeasure::
            generatedActiveSubset(
                physical_marker, active_marker);

    auto residual = (u * v).dx();
    residual = forms::bc::applyTraceNitsche(
        std::move(residual),
        u,
        v,
        measure,
        forms::FormExpr::constant(1.0),
        inner(grad(u), n),
        inner(grad(v), n),
        forms::FormExpr::constant(1.0) /
            forms::h(),
        forms::bc::ScalarTraceOperator::Identity);
    forms::bc::ScalarNitscheBC scalar_bc(
        measure,
        forms::FormExpr::constant(0.0),
        forms::FormExpr::constant(1.0),
        /*penalty_gamma=*/10.0,
        /*symmetric=*/false);
    const auto scalar_metadata =
        scalar_bc.analysisMetadata(
            /*field_id=*/0, nullptr);
    ASSERT_EQ(scalar_metadata.size(), 1u);
    ASSERT_TRUE(
        scalar_metadata.front().nitsche.has_value());
    EXPECT_FALSE(
        scalar_metadata.front()
            .nitsche->penalty_scaling_verified);
    scalar_bc.contributeToResidual(
        residual, u, v);
    forms::bc::TraceNitscheBC trace_bc(
        measure,
        forms::FormExpr::constant(0.0),
        inner(grad(u), n),
        inner(grad(v), n),
        forms::FormExpr::constant(1.0) /
            forms::h(),
        forms::bc::ScalarTraceOperator::Identity);
    const auto trace_metadata =
        trace_bc.analysisMetadata(
            /*field_id=*/0, nullptr);
    ASSERT_EQ(trace_metadata.size(), 1u);
    ASSERT_TRUE(
        trace_metadata.front().nitsche.has_value());
    EXPECT_FALSE(
        trace_metadata.front()
            .nitsche->penalty_scaling_verified);
    trace_bc.contributeToResidual(
        residual, u, v);

    const auto ir = forms::FormCompiler{}.compileResidual(
        residual);
    int active_terms = 0;
    int whole_face_terms = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain ==
                forms::IntegralDomain::InterfaceFace &&
            term.interface_marker == active_marker &&
            term.exterior_boundary_measure.has_value() &&
            term.exterior_boundary_measure
                    ->physicalBoundaryMarker() ==
                physical_marker) {
            ++active_terms;
        }
        if (term.domain ==
                forms::IntegralDomain::Boundary &&
            term.boundary_marker == physical_marker) {
            ++whole_face_terms;
        }
    }

    EXPECT_EQ(active_terms, 9);
    EXPECT_EQ(whole_face_terms, 0);
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

TEST(FormsBoundaryConditions, BuildTraceNitschePenaltyScalesWithPolynomialOrder)
{
    auto p1_space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/1);
    auto p3_space = svmp::FE::spaces::H1Space(ElementType::Tetra4, /*order=*/3);
    const auto u_p1 = svmp::FE::forms::FormExpr::trialFunction(p1_space, "u_p1");
    const auto u_p3 = svmp::FE::forms::FormExpr::trialFunction(p3_space, "u_p3");
    const auto weight = svmp::FE::forms::FormExpr::constant(2.0);

    svmp::FE::forms::bc::TraceNitscheOptions opts;
    opts.gamma = 5.0;
    opts.scale_with_p = true;

    const auto p1_penalty =
        svmp::FE::forms::bc::buildTraceNitschePenalty(weight, u_p1, opts);
    const auto p3_penalty =
        svmp::FE::forms::bc::buildTraceNitschePenalty(weight, u_p3, opts);

    const svmp::FE::forms::PointEvalContext ctx;
    EXPECT_DOUBLE_EQ(svmp::FE::forms::evaluateScalarAt(p1_penalty, ctx), 10.0);
    EXPECT_DOUBLE_EQ(svmp::FE::forms::evaluateScalarAt(p3_penalty, ctx), 90.0);

    opts.scale_with_p = false;
    const auto unscaled =
        svmp::FE::forms::bc::buildTraceNitschePenalty(weight, u_p3, opts);
    EXPECT_DOUBLE_EQ(svmp::FE::forms::evaluateScalarAt(unscaled, ctx), 10.0);
}

