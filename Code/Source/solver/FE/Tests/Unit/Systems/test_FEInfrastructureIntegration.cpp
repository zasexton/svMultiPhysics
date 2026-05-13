/**
 * @file test_FEInfrastructureIntegration.cpp
 * @brief Integration-style tests for generic FE infrastructure helpers.
 */

#include <gtest/gtest.h>

#include "Spaces/H1Space.h"
#include "Systems/BoundedUpdatePolicy.h"
#include "Systems/FieldRegistry.h"
#include "Systems/PairedExchange.h"
#include "Systems/StateAdmissibility.h"

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
using svmp::FE::systems::PairedExchange;
using svmp::FE::systems::PairedExchangeDescriptor;
using svmp::FE::systems::PairedExchangeEndpoint;
using svmp::FE::systems::StateAdmissibilityCheckKind;
using svmp::FE::systems::StateAdmissibilityDescriptor;
using svmp::FE::systems::StateAdmissibilityEvaluator;
using svmp::FE::systems::StateAdmissibilitySample;
using svmp::FE::systems::StateAdmissibilityScope;
using svmp::FE::systems::StateAdmissibilityTarget;
using svmp::FE::systems::StateGroupKind;
using svmp::FE::systems::StateGroupSpec;
using svmp::FE::spaces::H1Space;

namespace {

std::shared_ptr<H1Space> scalarSpace()
{
    return std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
}

StateAdmissibilityDescriptor unitInterval(FieldId field)
{
    StateAdmissibilityDescriptor descriptor;
    descriptor.id = "bounded_scalar";
    descriptor.target = StateAdmissibilityTarget::fieldTarget(field);
    descriptor.check = StateAdmissibilityCheckKind::Interval;
    descriptor.scope = StateAdmissibilityScope::CellAverages;
    descriptor.lower_bound = 0.0;
    descriptor.upper_bound = 1.0;
    return descriptor;
}

} // namespace

TEST(FEInfrastructureIntegration, StateGroupFeedsAdmissibilitySummary)
{
    FieldRegistry registry;
    const auto space = scalarSpace();
    const FieldId a = registry.add(FieldSpec{.name = "component_a",
                                             .space = space,
                                             .components = 1});
    const FieldId b = registry.add(FieldSpec{.name = "component_b",
                                             .space = space,
                                             .components = 1});
    const auto group = registry.addStateGroup(StateGroupSpec{
        .name = "component_pair",
        .kind = StateGroupKind::ConservedComponents,
        .shape = FieldShapeKind::IndexedScalarSet,
        .fields = {a, b}});

    StateAdmissibilityDescriptor descriptor;
    descriptor.id = "component_sum";
    descriptor.target = StateAdmissibilityTarget::stateGroupTarget(group);
    descriptor.check = StateAdmissibilityCheckKind::SumEquality;
    descriptor.scope = StateAdmissibilityScope::CellAverages;
    descriptor.sum_target = 1.0;
    descriptor.tolerance = 1.0e-12;

    const std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {0.25, 0.75}, .label = "cell0"},
        StateAdmissibilitySample{.values = {0.4, 0.6}, .label = "cell1"}};
    const auto result = StateAdmissibilityEvaluator::evaluate(descriptor, samples);
    const auto summary =
        StateAdmissibilityEvaluator::makeInvariantDomainSummary(descriptor,
                                                                result,
                                                                &registry);

    EXPECT_TRUE(result.admissible);
    EXPECT_EQ(summary.invariant_set_id, "component_sum");
    EXPECT_EQ(summary.post_step_violation_count, 0u);
    ASSERT_EQ(summary.variables.size(), 2u);
    EXPECT_EQ(summary.variables[0].field_id, a);
    EXPECT_EQ(summary.variables[1].field_id, b);
}

TEST(FEInfrastructureIntegration, BoundedUpdateFeedsInvariantDomainSummary)
{
    BoundedUpdatePolicy policy;
    policy.kind = BoundedUpdatePolicyKind::ClampCellAverage;
    policy.admissibility = unitInterval(0);

    std::vector<StateAdmissibilitySample> samples = {
        StateAdmissibilitySample{.values = {-0.2, 0.4, 1.3}, .label = "cell0"}};

    const auto result = BoundedUpdateController::apply(policy, samples);
    const auto summary =
        StateAdmissibilityEvaluator::makeInvariantDomainSummary(policy.admissibility,
                                                                result.after);

    EXPECT_EQ(result.decision, BoundedUpdateDecision::AcceptedWithModification);
    EXPECT_TRUE(result.modified);
    EXPECT_TRUE(result.after.admissible);
    EXPECT_EQ(samples[0].values, std::vector<double>({0.0, 0.4, 1.0}));
    EXPECT_EQ(summary.post_step_violation_count, 0u);
}

TEST(FEInfrastructureIntegration, PairedExchangeProducesConservationSummary)
{
    PairedExchangeDescriptor descriptor;
    descriptor.id = "exchange";
    descriptor.balance_group = "generic_balance";
    descriptor.exchanged_quantity_name = "generic_quantity";
    descriptor.donor = PairedExchangeEndpoint::fieldEndpoint(0);
    descriptor.receiver = PairedExchangeEndpoint::fieldEndpoint(1);
    descriptor.conservation_tolerance = 1.0e-12;

    const std::vector<double> exchange = {0.2, 0.3};
    std::vector<double> donor = {0.0, 0.0};
    std::vector<double> receiver = {0.0, 0.0};

    PairedExchange::applyEqualAndOpposite(descriptor, exchange, donor, receiver);
    const auto balance = PairedExchange::checkBalance(descriptor, donor, receiver);
    const auto summary = PairedExchange::makeFluxBalanceSummary(descriptor, balance);

    EXPECT_TRUE(balance.conserved);
    EXPECT_DOUBLE_EQ(balance.global_residual_norm, 0.0);
    EXPECT_TRUE(summary.symbolic_balance_evidence_present);
    EXPECT_EQ(summary.symbolic_balance_group, "generic_balance");
    EXPECT_DOUBLE_EQ(summary.global_residual_norm, 0.0);
}
