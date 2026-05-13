/**
 * @file test_PairedExchange.cpp
 * @brief Unit tests for paired exchange balance helpers.
 */

#include <gtest/gtest.h>

#include "Spaces/H1Space.h"
#include "Systems/FieldRegistry.h"
#include "Systems/PairedExchange.h"

#include <memory>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::analysis::BalanceRole;
using svmp::FE::analysis::ContributionRole;
using svmp::FE::analysis::OperatorTraitFlags;
using svmp::FE::analysis::hasFlag;
using svmp::FE::systems::FieldRegistry;
using svmp::FE::systems::FieldShapeKind;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::PairedExchange;
using svmp::FE::systems::PairedExchangeDescriptor;
using svmp::FE::systems::PairedExchangeEndpoint;
using svmp::FE::systems::StateGroupKind;
using svmp::FE::systems::StateGroupSpec;
using svmp::FE::spaces::H1Space;

namespace {

PairedExchangeDescriptor scalarDescriptor()
{
    PairedExchangeDescriptor descriptor;
    descriptor.id = "exchange";
    descriptor.balance_group = "quantity_balance";
    descriptor.exchanged_quantity_name = "quantity";
    descriptor.donor = PairedExchangeEndpoint::fieldEndpoint(0);
    descriptor.receiver = PairedExchangeEndpoint::fieldEndpoint(1);
    descriptor.conservation_tolerance = 1.0e-12;
    return descriptor;
}

} // namespace

TEST(PairedExchange, AppliesEqualAndOppositeScalarExchange)
{
    auto descriptor = scalarDescriptor();
    std::vector<double> exchange = {0.25, 0.5};
    std::vector<double> donor = {0.0, 0.0};
    std::vector<double> receiver = {0.0, 0.0};

    PairedExchange::applyEqualAndOpposite(descriptor, exchange, donor, receiver);

    EXPECT_EQ(donor, std::vector<double>({-0.25, -0.5}));
    EXPECT_EQ(receiver, std::vector<double>({0.25, 0.5}));
    const auto balance = PairedExchange::checkBalance(descriptor, donor, receiver);
    EXPECT_TRUE(balance.conserved);
    EXPECT_DOUBLE_EQ(balance.global_residual_norm, 0.0);
}

TEST(PairedExchange, EmitsContributionMetadataForFieldExchange)
{
    auto descriptor = scalarDescriptor();
    descriptor.source_contribution_id = "source_term";
    const auto contributions = PairedExchange::makeContributions(descriptor);

    ASSERT_EQ(contributions.size(), 2u);
    EXPECT_EQ(contributions[0].contribution_id, "source_term_donor_0");
    EXPECT_EQ(contributions[1].contribution_id, "source_term_receiver_0");
    EXPECT_EQ(contributions[0].role, ContributionRole::SourceVector);
    EXPECT_TRUE(hasFlag(contributions[0].traits, OperatorTraitFlags::SourceLike));
    ASSERT_TRUE(contributions[0].balance.has_value());
    EXPECT_EQ(contributions[0].balance->balance_group, "quantity_balance");
    EXPECT_EQ(contributions[0].balance->role, BalanceRole::ExchangeLike);
    EXPECT_EQ(contributions[0].balance->sign, -1);
    ASSERT_TRUE(contributions[1].balance.has_value());
    EXPECT_EQ(contributions[1].balance->sign, 1);
}

TEST(PairedExchange, ExpandsStateGroupEndpoints)
{
    FieldRegistry registry;
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    FieldId a0 = registry.add(FieldSpec{.name = "a0", .space = space, .components = 1});
    FieldId a1 = registry.add(FieldSpec{.name = "a1", .space = space, .components = 1});
    FieldId b0 = registry.add(FieldSpec{.name = "b0", .space = space, .components = 1});
    auto donor_group = registry.addStateGroup(StateGroupSpec{.name = "donor_group",
                                                             .kind = StateGroupKind::MomentOrBinSet,
                                                             .shape = FieldShapeKind::IndexedScalarSet,
                                                             .fields = {a0, a1}});
    auto receiver_group = registry.addStateGroup(StateGroupSpec{.name = "receiver_group",
                                                                .fields = {b0}});

    PairedExchangeDescriptor descriptor;
    descriptor.id = "group_exchange";
    descriptor.balance_group = "group_balance";
    descriptor.exchanged_quantity_name = "group_quantity";
    descriptor.donor = PairedExchangeEndpoint::stateGroupEndpoint(donor_group);
    descriptor.receiver = PairedExchangeEndpoint::stateGroupEndpoint(receiver_group);

    const auto contributions = PairedExchange::makeContributions(descriptor, &registry);
    ASSERT_EQ(contributions.size(), 3u);
    EXPECT_EQ(contributions[0].related_variables[0].field_id, a0);
    EXPECT_EQ(contributions[1].related_variables[0].field_id, a1);
    EXPECT_EQ(contributions[2].related_variables[0].field_id, b0);
}

TEST(PairedExchange, ReportsFailedConservationTolerance)
{
    auto descriptor = scalarDescriptor();
    descriptor.conservation_tolerance = 1.0e-6;
    std::vector<double> donor = {-0.25, -0.5};
    std::vector<double> receiver = {0.25, 0.25};

    const auto balance = PairedExchange::checkBalance(descriptor, donor, receiver);
    EXPECT_FALSE(balance.conserved);
    EXPECT_EQ(balance.local_violation_count, 1u);
    EXPECT_DOUBLE_EQ(balance.global_residual_norm, 0.25);
}

TEST(PairedExchange, EmitsFluxBalanceSummary)
{
    auto descriptor = scalarDescriptor();
    std::vector<double> donor = {-0.25};
    std::vector<double> receiver = {0.25};
    const auto balance = PairedExchange::checkBalance(descriptor, donor, receiver);
    const auto summary = PairedExchange::makeFluxBalanceSummary(descriptor, balance);

    EXPECT_EQ(summary.balance_group, "quantity_balance");
    EXPECT_DOUBLE_EQ(summary.balance_tolerance, 1.0e-12);
    EXPECT_DOUBLE_EQ(summary.global_residual_norm, 0.0);
    EXPECT_TRUE(summary.symbolic_balance_evidence_present);
    EXPECT_EQ(summary.symbolic_balance_group, "quantity_balance");
    EXPECT_TRUE(summary.source_quadrature_consistency_present);
}

TEST(PairedExchange, DoesNotClaimSymbolicBalanceWithoutGroup)
{
    auto descriptor = scalarDescriptor();
    descriptor.balance_group.clear();
    std::vector<double> donor = {-0.25};
    std::vector<double> receiver = {0.25};
    const auto balance = PairedExchange::checkBalance(descriptor, donor, receiver);
    const auto summary = PairedExchange::makeFluxBalanceSummary(descriptor, balance);

    EXPECT_TRUE(balance.conserved);
    EXPECT_FALSE(summary.symbolic_balance_evidence_present);
    EXPECT_TRUE(summary.symbolic_balance_group.empty());
}

TEST(PairedExchange, SupportsWeightedBalanceChecks)
{
    auto descriptor = scalarDescriptor();
    descriptor.donor_weight = 2.0;
    descriptor.receiver_weight = 1.0;
    std::vector<double> donor = {-0.25};
    std::vector<double> receiver = {0.5};

    const auto balance = PairedExchange::checkBalance(descriptor, donor, receiver);
    EXPECT_TRUE(balance.conserved);
    EXPECT_DOUBLE_EQ(balance.global_residual_norm, 0.0);
}
