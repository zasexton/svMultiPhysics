/**
 * @file test_BoundedUpdatePolicy.cpp
 * @brief Unit tests for bounded update policy plumbing.
 */

#include <gtest/gtest.h>

#include "Spaces/H1Space.h"
#include "Systems/BoundedUpdatePolicy.h"
#include "Systems/FieldRegistry.h"

#include <memory>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::systems::BoundedUpdateController;
using svmp::FE::systems::BoundedUpdateDecision;
using svmp::FE::systems::BoundedUpdatePolicy;
using svmp::FE::systems::BoundedUpdatePolicyKind;
using svmp::FE::systems::FieldRegistry;
using svmp::FE::systems::FieldShapeKind;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::StateAdmissibilityCheckKind;
using svmp::FE::systems::StateAdmissibilityDescriptor;
using svmp::FE::systems::StateAdmissibilityFailurePolicy;
using svmp::FE::systems::StateAdmissibilitySample;
using svmp::FE::systems::StateAdmissibilityTarget;
using svmp::FE::systems::StateGroupKind;
using svmp::FE::systems::StateGroupSpec;
using svmp::FE::spaces::H1Space;

namespace {

StateAdmissibilityDescriptor unitIntervalDescriptor()
{
    StateAdmissibilityDescriptor descriptor;
    descriptor.id = "unit_interval";
    descriptor.target = StateAdmissibilityTarget::fieldTarget(0);
    descriptor.check = StateAdmissibilityCheckKind::Interval;
    descriptor.lower_bound = 0.0;
    descriptor.upper_bound = 1.0;
    return descriptor;
}

} // namespace

TEST(BoundedUpdatePolicy, NoneLeavesValuesUntouched)
{
    BoundedUpdatePolicy policy;
    policy.kind = BoundedUpdatePolicyKind::None;

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {-2.0, 3.0}, .label = "candidate"}};

    const auto result = BoundedUpdateController::apply(policy, samples);
    EXPECT_EQ(result.decision, BoundedUpdateDecision::Accepted);
    EXPECT_FALSE(result.modified);
    EXPECT_EQ(samples[0].values, std::vector<double>({-2.0, 3.0}));
}

TEST(BoundedUpdatePolicy, RejectsViolatingCandidate)
{
    BoundedUpdatePolicy policy;
    policy.kind = BoundedUpdatePolicyKind::RejectStepOnViolation;
    policy.admissibility = unitIntervalDescriptor();

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {-0.1}, .label = "candidate"}};

    const auto result = BoundedUpdateController::apply(policy, samples);
    EXPECT_EQ(result.decision, BoundedUpdateDecision::Rejected);
    EXPECT_FALSE(result.before.admissible);
    EXPECT_EQ(result.before.violation_count, 1u);
}

TEST(BoundedUpdatePolicy, CheckOnlyAcceptsWithWarningSemantics)
{
    BoundedUpdatePolicy policy;
    policy.kind = BoundedUpdatePolicyKind::CheckOnly;
    policy.admissibility = unitIntervalDescriptor();
    policy.admissibility.failure_policy = StateAdmissibilityFailurePolicy::WarnOnly;

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {1.5}, .label = "candidate"}};

    const auto result = BoundedUpdateController::apply(policy, samples);
    EXPECT_EQ(result.decision, BoundedUpdateDecision::Accepted);
    EXPECT_FALSE(result.before.admissible);
    EXPECT_EQ(result.before.violation_count, 1u);
}

TEST(BoundedUpdatePolicy, ClampCellAverageModifiesBoundedValues)
{
    BoundedUpdatePolicy policy;
    policy.kind = BoundedUpdatePolicyKind::ClampCellAverage;
    policy.admissibility = unitIntervalDescriptor();

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {-0.5, 0.25, 1.5}, .label = "cell0"}};

    const auto result = BoundedUpdateController::apply(policy, samples);
    EXPECT_EQ(result.decision, BoundedUpdateDecision::AcceptedWithModification);
    EXPECT_TRUE(result.modified);
    EXPECT_FALSE(result.before.admissible);
    EXPECT_TRUE(result.after.admissible);
    EXPECT_EQ(samples[0].values, std::vector<double>({0.0, 0.25, 1.0}));
}

TEST(BoundedUpdatePolicy, HandlesGroupedBoundChecks)
{
    FieldRegistry registry;
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    FieldId a = registry.add(FieldSpec{.name = "a", .space = space, .components = 1});
    FieldId b = registry.add(FieldSpec{.name = "b", .space = space, .components = 1});
    auto group = registry.addStateGroup(StateGroupSpec{.name = "fractions",
                                                       .kind = StateGroupKind::VolumeFractions,
                                                       .shape = FieldShapeKind::IndexedScalarSet,
                                                       .fields = {a, b}});

    BoundedUpdatePolicy policy;
    policy.kind = BoundedUpdatePolicyKind::RejectStepOnViolation;
    policy.admissibility.id = "fraction_bounds";
    policy.admissibility.target = StateAdmissibilityTarget::stateGroupTarget(group);
    policy.admissibility.check = StateAdmissibilityCheckKind::Interval;
    policy.admissibility.lower_bound = 0.0;
    policy.admissibility.upper_bound = 1.0;

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {0.4, 0.6}, .label = "cell0"},
        StateAdmissibilitySample{.values = {1.2, -0.2}, .label = "cell1"}};

    const auto result = BoundedUpdateController::apply(policy, samples);
    EXPECT_EQ(result.decision, BoundedUpdateDecision::Rejected);
    EXPECT_EQ(result.before.violation_count, 1u);
}

TEST(BoundedUpdatePolicy, SupportsCallbacksAndCandidateFilters)
{
    BoundedUpdatePolicy policy;
    policy.kind = BoundedUpdatePolicyKind::CheckOnly;
    policy.admissibility = unitIntervalDescriptor();
    bool pre_called = false;
    bool post_called = false;
    policy.pre_update_callback = [&](std::vector<StateAdmissibilitySample>& samples) {
        pre_called = true;
        samples[0].values[0] = 0.25;
    };
    policy.post_update_callback = [&](std::vector<StateAdmissibilitySample>&) {
        post_called = true;
    };
    policy.candidate_state_filter = [](std::span<const StateAdmissibilitySample> samples) {
        return !samples.empty();
    };

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {2.0}, .label = "candidate"}};

    const auto result = BoundedUpdateController::apply(policy, samples);
    EXPECT_EQ(result.decision, BoundedUpdateDecision::Accepted);
    EXPECT_TRUE(pre_called);
    EXPECT_TRUE(post_called);
    EXPECT_TRUE(result.before.admissible);
}

TEST(BoundedUpdatePolicy, AppliesToMutableSpans)
{
    BoundedUpdatePolicy policy;
    policy.kind = BoundedUpdatePolicyKind::ClampCellAverage;
    policy.admissibility = unitIntervalDescriptor();

    std::vector<double> current_state = {-1.0, 0.5, 2.0};
    const auto result = BoundedUpdateController::applyToSpan(
        policy, std::span<double>(current_state.data(), current_state.size()), "current");

    EXPECT_EQ(result.decision, BoundedUpdateDecision::AcceptedWithModification);
    EXPECT_EQ(current_state, std::vector<double>({0.0, 0.5, 1.0}));
}
