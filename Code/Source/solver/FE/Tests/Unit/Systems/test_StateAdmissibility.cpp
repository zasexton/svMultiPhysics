/**
 * @file test_StateAdmissibility.cpp
 * @brief Unit tests for generic state admissibility checks.
 */

#include <gtest/gtest.h>

#include "Spaces/H1Space.h"
#include "Systems/FieldRegistry.h"
#include "Systems/StateAdmissibility.h"

#include <limits>
#include <memory>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::systems::FieldRegistry;
using svmp::FE::systems::FieldShapeKind;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::StateAdmissibilityCheckKind;
using svmp::FE::systems::StateAdmissibilityDescriptor;
using svmp::FE::systems::StateAdmissibilityEvaluator;
using svmp::FE::systems::StateAdmissibilityFailurePolicy;
using svmp::FE::systems::StateAdmissibilitySample;
using svmp::FE::systems::StateAdmissibilityScope;
using svmp::FE::systems::StateAdmissibilityTarget;
using svmp::FE::systems::StateGroupKind;
using svmp::FE::systems::StateGroupSpec;
using svmp::FE::spaces::H1Space;

TEST(StateAdmissibility, DetectsScalarNonnegativityViolations)
{
    const FieldId field = 0;
    StateAdmissibilityDescriptor descriptor;
    descriptor.id = "nonnegative";
    descriptor.target = StateAdmissibilityTarget::fieldTarget(field);
    descriptor.check = StateAdmissibilityCheckKind::LowerBound;
    descriptor.scope = StateAdmissibilityScope::NodalValues;
    descriptor.lower_bound = 0.0;

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {1.0}, .label = "ok"},
        StateAdmissibilitySample{.values = {-0.25}, .label = "bad"}};

    const auto result = StateAdmissibilityEvaluator::evaluate(descriptor, samples);
    EXPECT_FALSE(result.admissible);
    EXPECT_EQ(result.checked_state_count, 2u);
    EXPECT_EQ(result.violation_count, 1u);
    EXPECT_DOUBLE_EQ(result.max_residual, 0.25);
    ASSERT_EQ(result.violations.size(), 1u);
    EXPECT_EQ(result.violations[0].sample_index, 1u);
    EXPECT_EQ(result.violations[0].label, "bad");
}

TEST(StateAdmissibility, ChecksBoundedScalarIntervals)
{
    StateAdmissibilityDescriptor descriptor;
    descriptor.id = "unit_interval";
    descriptor.target = StateAdmissibilityTarget::fieldTarget(0);
    descriptor.check = StateAdmissibilityCheckKind::Interval;
    descriptor.lower_bound = 0.0;
    descriptor.upper_bound = 1.0;

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {0.5}},
        StateAdmissibilitySample{.values = {1.25}}};

    const auto result = StateAdmissibilityEvaluator::evaluate(descriptor, samples);
    EXPECT_FALSE(result.admissible);
    EXPECT_DOUBLE_EQ(result.max_residual, 0.25);
}

TEST(StateAdmissibility, ChecksPhaseLikeSumEquality)
{
    StateAdmissibilityDescriptor descriptor;
    descriptor.id = "sum_one";
    descriptor.target = StateAdmissibilityTarget::stateGroupTarget(0);
    descriptor.check = StateAdmissibilityCheckKind::SumEquality;
    descriptor.scope = StateAdmissibilityScope::QuadratureValues;
    descriptor.sum_target = 1.0;
    descriptor.tolerance = 1.0e-12;

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {0.4, 0.6}},
        StateAdmissibilitySample{.values = {0.5, 0.6}}};

    const auto result = StateAdmissibilityEvaluator::evaluate(descriptor, samples);
    EXPECT_FALSE(result.admissible);
    EXPECT_EQ(result.violation_count, 1u);
    EXPECT_NEAR(result.max_residual, 0.1, 1.0e-12);
}

TEST(StateAdmissibility, SupportsGroupLevelViolationReporting)
{
    FieldRegistry registry;
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    FieldId a = registry.add(FieldSpec{.name = "a", .space = space, .components = 1});
    FieldId b = registry.add(FieldSpec{.name = "b", .space = space, .components = 1});
    auto group = registry.addStateGroup(StateGroupSpec{.name = "fractions",
                                                       .kind = StateGroupKind::VolumeFractions,
                                                       .shape = FieldShapeKind::IndexedScalarSet,
                                                       .fields = {a, b}});

    StateAdmissibilityDescriptor descriptor;
    descriptor.id = "bounded_fractions";
    descriptor.target = StateAdmissibilityTarget::stateGroupTarget(group);
    descriptor.check = StateAdmissibilityCheckKind::Interval;
    descriptor.scope = StateAdmissibilityScope::CellAverages;
    descriptor.failure_policy = StateAdmissibilityFailurePolicy::WarnOnly;
    descriptor.lower_bound = 0.0;
    descriptor.upper_bound = 1.0;

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {0.2, 0.8}, .label = "cell0"},
        StateAdmissibilitySample{.values = {-0.1, 1.1}, .label = "cell1"}};

    const auto result = StateAdmissibilityEvaluator::evaluate(descriptor, samples);
    EXPECT_FALSE(result.admissible);
    EXPECT_EQ(result.violation_count, 1u);
    EXPECT_NEAR(result.max_residual, 0.1, 1.0e-12);
    ASSERT_EQ(result.violations.size(), 1u);
    EXPECT_EQ(result.violations[0].label, "cell1");
}

TEST(StateAdmissibility, EmitsInvariantDomainSummary)
{
    FieldRegistry registry;
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    FieldId a = registry.add(FieldSpec{.name = "a", .space = space, .components = 1});
    FieldId b = registry.add(FieldSpec{.name = "b", .space = space, .components = 1});
    auto group = registry.addStateGroup(StateGroupSpec{.name = "bounded_pair",
                                                       .fields = {a, b}});

    StateAdmissibilityDescriptor descriptor;
    descriptor.id = "bounded_pair_domain";
    descriptor.target = StateAdmissibilityTarget::stateGroupTarget(group);
    descriptor.check = StateAdmissibilityCheckKind::Interval;
    descriptor.lower_bound = 0.0;
    descriptor.upper_bound = 1.0;

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {0.25, 0.75}},
        StateAdmissibilitySample{.values = {0.25, 1.25}}};

    const auto result = StateAdmissibilityEvaluator::evaluate(descriptor, samples);
    const auto summary =
        StateAdmissibilityEvaluator::makeInvariantDomainSummary(descriptor, result, &registry);

    EXPECT_EQ(summary.invariant_set_id, "bounded_pair_domain");
    EXPECT_TRUE(summary.lower_bound_active);
    EXPECT_TRUE(summary.upper_bound_active);
    EXPECT_DOUBLE_EQ(summary.lower_bound, 0.0);
    EXPECT_DOUBLE_EQ(summary.upper_bound, 1.0);
    EXPECT_TRUE(summary.source_admissibility_evidence_present);
    EXPECT_EQ(summary.post_step_violation_count, 1u);
    ASSERT_EQ(summary.variables.size(), 2u);
    EXPECT_EQ(summary.variables[0].field_id, a);
    EXPECT_EQ(summary.variables[1].field_id, b);
}

TEST(StateAdmissibility, SupportsCallbackResidualsAndFiniteChecks)
{
    StateAdmissibilityDescriptor finite;
    finite.id = "finite";
    finite.target = StateAdmissibilityTarget::fieldTarget(0);
    finite.check = StateAdmissibilityCheckKind::Finite;

    std::vector<StateAdmissibilitySample> finite_samples = {
        StateAdmissibilitySample{.values = {1.0}},
        StateAdmissibilitySample{.values = {std::numeric_limits<double>::quiet_NaN()}}};
    EXPECT_FALSE(StateAdmissibilityEvaluator::evaluate(finite, finite_samples).admissible);

    StateAdmissibilityDescriptor callback;
    callback.id = "callback";
    callback.target = StateAdmissibilityTarget::fieldTarget(0);
    callback.check = StateAdmissibilityCheckKind::CallbackResidual;
    callback.tolerance = 0.5;
    callback.residual_callback = [](std::span<const svmp::FE::Real> values) {
        return values.empty() ? 1.0 : values[0];
    };

    std::vector<StateAdmissibilitySample> callback_samples = {
        StateAdmissibilitySample{.values = {0.25}},
        StateAdmissibilitySample{.values = {0.75}}};
    const auto callback_result =
        StateAdmissibilityEvaluator::evaluate(callback, callback_samples);
    EXPECT_FALSE(callback_result.admissible);
    EXPECT_DOUBLE_EQ(callback_result.max_residual, 0.25);
}
