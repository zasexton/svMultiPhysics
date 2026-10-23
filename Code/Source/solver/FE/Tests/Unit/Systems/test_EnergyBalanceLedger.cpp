/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/EnergyBalanceLedger.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace systems = svmp::FE::systems;
using systems::EnergyBalanceAttemptMetadata;
using systems::EnergyBalanceCatalogue;
using systems::EnergyBalanceInput;
using systems::EnergyBalanceLedger;
using systems::EnergyBalanceRejectionReason;
using systems::EnergyBalanceSources;
using systems::EnergyBalanceTemporalScheme;
using systems::EnergyBalanceValues;
using systems::EnergyChannelApplicability;
using systems::EnergyChannelDeclaration;
using systems::EnergyChannelIdentity;
using systems::EnergyChannelKind;
using systems::EnergyChannelSource;
using Rule = systems::EnergyChannelIdentity::Rule;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool sameBits(double left, double right)
{
    return std::bit_cast<std::uint64_t>(left) == std::bit_cast<std::uint64_t>(right);
}

/// Two stored energies, one rate, two external works (one an identity), three
/// numerical works (two identities).
EnergyBalanceCatalogue catalogue()
{
    return EnergyBalanceCatalogue(
        "test energy",
        {{
            {{.name = "kinetic", .nonnegative = true}, {.name = "potential"}},
            {{.name = "viscous"}},
            {{.name = "body"},
             {.name = "pressure",
              .identity = {.rule = Rule::StepWork, .stored = {}, .input = "pressure_power"}}},
            {{.name = "transport",
              .identity = {.rule = Rule::StoredTransport,
                           .stored = {"kinetic", "potential"},
                           .input = "transport_power"}},
             {.name = "domain",
              .identity = {.rule = Rule::DomainTransport,
                           .stored = {"kinetic"},
                           .input = "kinetic_on_endpoint_domain"}},
             {.name = "stabilization"}},
        }});
}

EnergyBalanceAttemptMetadata metadata(std::uint64_t transaction_id,
                                      std::uint64_t step = 4u,
                                      double time_before = 0.75)
{
    const auto revision = 10u * step;
    return EnergyBalanceAttemptMetadata{
        .transaction_id = transaction_id,
        .step = step,
        .attempt = 1u,
        .time_before = time_before,
        .time_after = time_before + 0.25,
        .dt = 0.25,
        .temporal_scheme = EnergyBalanceTemporalScheme::BackwardEuler,
        .physical_evaluation_time = time_before + 0.25,
        .physical_evaluation_stage_fraction = 1.0,
        .algebraic_state_revision_before = revision + 1u,
        .physical_endpoint_algebraic_state_revision = revision + 12u,
        .algebraic_state_revision_after = revision + 11u,
        .snapshot_set_revision_before = revision + 2u,
        .physical_endpoint_snapshot_set_revision = revision + 12u,
        .snapshot_set_revision_after = revision + 12u,
        .mesh_topology_set_revision_before = 7u,
        .physical_endpoint_mesh_topology_set_revision = 7u,
        .mesh_topology_set_revision_after = 7u,
        .cut_topology_set_revision_before = 9u,
        .physical_endpoint_cut_topology_set_revision = 9u,
        .cut_topology_set_revision_after = 9u,
    };
}

EnergyBalanceSources produced(const EnergyBalanceCatalogue& declared)
{
    EnergyBalanceSources sources;
    for (const auto kind : systems::kEnergyChannelKinds) {
        for (const auto& channel : declared.channels(kind)) {
            sources[systems::energyChannelKindIndex(kind)].push_back(EnergyChannelSource{
                .applicability = EnergyChannelApplicability::Produced,
                .owner = "owner." + channel.name,
            });
        }
    }
    return sources;
}

EnergyBalanceValues values(std::vector<double> before, std::vector<double> after)
{
    return EnergyBalanceValues{
        .stored_before = std::move(before),
        .stored_physical_endpoint = after,
        .stored_after = after,
        .dissipation_rate = {0.5},
        .external_work = {0.125, kNaN},
        .numerical_work = {kNaN, kNaN, -0.0625},
    };
}

std::vector<EnergyBalanceInput> inputs()
{
    return {{.name = "pressure_power", .value = 0.3},
            {.name = "transport_power", .value = 1.7},
            {.name = "kinetic_on_endpoint_domain", .value = 2.25}};
}

TEST(EnergyBalanceLedger, EvaluatesDeclaredIdentitiesAndSumsEachKindInDeclarationOrder)
{
    EnergyBalanceLedger ledger(catalogue());
    ledger.beginAttempt(metadata(1u));
    // Values where the order of summation changes the rounded result.
    const std::vector<double> before{2.0, 1.0e16};
    const std::vector<double> after{3.0, -1.0e16 + 2.0};
    ledger.stageBalance(values(before, after), produced(ledger.catalogue()), inputs());
    ledger.commitAttempt();
    ASSERT_EQ(ledger.acceptedAttempts().size(), 1u);
    const auto& record = ledger.acceptedAttempts().front();

    const double dt = 0.25;
    // StepWork, StoredTransport and DomainTransport, as the formulas read.
    EXPECT_TRUE(sameBits(record.values.external_work[1], dt * 0.3));
    EXPECT_TRUE(sameBits(record.values.numerical_work[0],
                         (((after[0] + after[1]) - before[0]) - before[1]) - dt * 1.7));
    EXPECT_TRUE(sameBits(record.values.numerical_work[1], 2.25 - before[0]));
    // Totals: each kind left to right, then the balance.
    const double energy_before = before[0] + before[1];
    const double energy_after = after[0] + after[1];
    EXPECT_TRUE(sameBits(record.stored_energy_before, energy_before));
    EXPECT_TRUE(sameBits(record.stored_energy_after, energy_after));
    EXPECT_TRUE(sameBits(record.integrated_physical_dissipation, dt * 0.5));
    const double external = 0.125 + dt * 0.3;
    const double numerical = (record.values.numerical_work[0] + record.values.numerical_work[1]) +
                             -0.0625;
    EXPECT_TRUE(sameBits(record.total_external_work, external));
    EXPECT_TRUE(sameBits(record.total_numerical_work, numerical));
    EXPECT_TRUE(sameBits(record.accepted_balance_residual,
                         (energy_after - energy_before) + dt * 0.5 - external - numerical));
}

TEST(EnergyBalanceLedger, SumsAndIdentityHelpersFoldLeftToRight)
{
    const std::vector<double> terms{1.0e16, 1.0, -1.0e16, 1.0};
    EXPECT_TRUE(sameBits(systems::sumInOrder(terms), ((1.0e16 + 1.0) + -1.0e16) + 1.0));
    EXPECT_TRUE(sameBits(systems::sumInOrder({}), 0.0));
    const std::vector<double> negative_zero{-0.0};
    EXPECT_TRUE(std::signbit(systems::sumInOrder(negative_zero)));
    const std::vector<double> after{0.1, 0.2};
    const std::vector<double> before{0.3};
    EXPECT_TRUE(sameBits(systems::energyChange(after, before), (0.1 + 0.2) - 0.3));
    EXPECT_TRUE(sameBits(systems::transportCouplingWork(0.7, 0.25, 0.9), 0.7 - 0.25 * 0.9));
}

TEST(EnergyBalanceLedger, IdentityChannelsAreEvaluatedOnlyByTheLedger)
{
    EnergyBalanceLedger ledger(catalogue());
    ledger.beginAttempt(metadata(1u));
    auto supplied = values({1.0, 2.0}, {1.0, 2.0});
    supplied.numerical_work[0] = 0.0;
    EXPECT_THROW(ledger.stageBalance(supplied, produced(ledger.catalogue()), inputs()),
                 std::invalid_argument);
    auto missing = inputs();
    missing.pop_back();
    try {
        ledger.stageBalance(values({1.0, 2.0}, {1.0, 2.0}), produced(ledger.catalogue()), missing);
        FAIL() << "a missing identity input must fail";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("kinetic_on_endpoint_domain"), std::string::npos);
    }
    ledger.stageBalance(values({1.0, 2.0}, {1.0, 2.0}), produced(ledger.catalogue()), inputs());
    EXPECT_NE(ledger.trialBalance(), nullptr);
}

TEST(EnergyBalanceLedger, CatalogueRejectsDuplicateNamesAndIncompleteIdentities)
{
    using Channels = std::array<std::vector<EnergyChannelDeclaration>, 4>;
    EXPECT_THROW(EnergyBalanceCatalogue("e", Channels{{{{.name = "a"}, {.name = "a"}}, {}, {}, {}}}),
                 std::invalid_argument);
    EXPECT_THROW(EnergyBalanceCatalogue("", Channels{}), std::invalid_argument);
    EXPECT_THROW(
        EnergyBalanceCatalogue(
            "e",
            Channels{{{{.name = "a",
                        .identity = {.rule = Rule::StepWork, .stored = {}, .input = "p"}}},
                      {},
                      {},
                      {}}}),
        std::invalid_argument);
    EXPECT_THROW(
        EnergyBalanceCatalogue(
            "e",
            Channels{{{{.name = "a"}},
                      {},
                      {},
                      {{.name = "t",
                        .identity = {.rule = Rule::StoredTransport, .stored = {"b"}, .input = "p"}}}}}),
        std::invalid_argument);
    EXPECT_THROW(
        EnergyBalanceCatalogue(
            "e",
            Channels{{{{.name = "a"}, {.name = "b"}},
                      {},
                      {},
                      {{.name = "t",
                        .identity = {.rule = Rule::DomainTransport,
                                     .stored = {"a", "b"},
                                     .input = "p"}}}}}),
        std::invalid_argument);
    const auto declared = catalogue();
    EXPECT_EQ(declared.count(EnergyChannelKind::NumericalWork), 3u);
    EXPECT_EQ(declared.position(EnergyChannelKind::NumericalWork, "stabilization"), 2u);
    EXPECT_FALSE(declared.find(EnergyChannelKind::Stored, "stabilization").has_value());
    EXPECT_EQ(declared.qualifiedName(EnergyChannelKind::DissipationRate, 0u), "dissipation.viscous");
    EXPECT_THROW((void)declared.position(EnergyChannelKind::Stored, "absent"), std::out_of_range);
}

TEST(EnergyBalanceLedger, ChannelsRequireNamedOwnersFiniteValuesAndExactZeroWhenInapplicable)
{
    EnergyBalanceLedger ledger(catalogue());
    ledger.beginAttempt(metadata(1u));
    auto sources = produced(ledger.catalogue());
    sources[0][1].owner = " ";
    EXPECT_THROW(ledger.stageBalance(values({1.0, 2.0}, {1.0, 2.0}), sources, inputs()),
                 std::invalid_argument);
    sources = produced(ledger.catalogue());
    sources[2][0].applicability = EnergyChannelApplicability::NotApplicable;
    try {
        ledger.stageBalance(values({1.0, 2.0}, {1.0, 2.0}), sources, inputs());
        FAIL() << "an inapplicable nonzero channel must fail";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()),
                  "Inapplicable test energy channel 'external.body' must be exactly zero.");
    }
    auto negative = values({1.0, 2.0}, {1.0, 2.0});
    negative.stored_after[0] = -1.0;
    negative.stored_physical_endpoint[0] = -1.0;
    EXPECT_THROW(ledger.stageBalance(negative, produced(ledger.catalogue()), inputs()),
                 std::invalid_argument);
    auto negative_rate = values({1.0, 2.0}, {1.0, 2.0});
    negative_rate.dissipation_rate[0] = -1.0;
    EXPECT_THROW(ledger.stageBalance(negative_rate, produced(ledger.catalogue()), inputs()),
                 std::invalid_argument);
    auto short_values = values({1.0, 2.0}, {1.0, 2.0});
    short_values.numerical_work.pop_back();
    EXPECT_THROW(ledger.stageBalance(short_values, produced(ledger.catalogue()), inputs()),
                 std::invalid_argument);
    ledger.stageBalance(values({1.0, 2.0}, {1.0, 2.0}), produced(ledger.catalogue()), inputs());
    ledger.commitAttempt();

    // Ownership is immutable within one history; the next step continues the
    // accepted endpoint and its stored values.
    ledger.beginAttempt(metadata(2u, 5u, 1.0));
    auto changed = produced(ledger.catalogue());
    changed[3][2].owner = "another.owner";
    EXPECT_THROW(ledger.stageBalance(values({1.0, 2.0}, {1.0, 2.0}), changed, inputs()),
                 std::invalid_argument);
    EXPECT_THROW(
        ledger.stageBalance(values({1.0, 2.5}, {1.0, 2.0}), produced(ledger.catalogue()), inputs()),
        std::invalid_argument);
    ledger.stageBalance(values({1.0, 2.0}, {1.0, 2.0}), produced(ledger.catalogue()), inputs());
    ledger.commitAttempt();
    EXPECT_EQ(ledger.acceptedAttempts().size(), 2u);
}

TEST(EnergyBalanceLedger, UnstagedRejectionKeepsOneUnavailableValuePerChannel)
{
    EnergyBalanceLedger ledger(catalogue());
    auto partial = metadata(1u);
    partial.physical_evaluation_time = kNaN;
    partial.physical_evaluation_stage_fraction = kNaN;
    partial.physical_endpoint_algebraic_state_revision = 0u;
    partial.algebraic_state_revision_after = 0u;
    ledger.beginAttempt(partial);
    ledger.rejectUnstagedAttempt(EnergyBalanceRejectionReason::NonlinearSolveFailure);
    ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
    const auto& rejected = ledger.rejectedAttempts().front();
    EXPECT_FALSE(rejected.balance_staged);
    EXPECT_EQ(rejected.values.stored_before.size(), 2u);
    EXPECT_EQ(rejected.values.numerical_work.size(), 3u);
    EXPECT_TRUE(std::isnan(rejected.values.external_work[0]));
    EXPECT_EQ(rejected.sources[3].size(), 3u);
    EXPECT_EQ(rejected.sources[3][0].applicability, EnergyChannelApplicability::Unspecified);
    EXPECT_TRUE(std::isnan(rejected.trial_balance_residual));
    EXPECT_EQ(rejected.accepted_balance_residual, 0.0);
    EXPECT_STREQ(systems::energyBalanceRejectionReasonName(rejected.rejection_reason),
                 "nonlinear_solve_failure");

    // The retry is the next attempt on the same starting endpoint.
    auto retry = metadata(2u);
    retry.attempt = 2u;
    ledger.beginAttempt(retry);
    ledger.stageBalance(values({1.0, 2.0}, {1.0, 2.0}), produced(ledger.catalogue()), inputs());
    ledger.commitAttempt();
    EXPECT_EQ(ledger.acceptedAttempts().front().metadata.attempt, 2u);
}

TEST(EnergyBalanceLedger, SourceLogEntriesNameApplicabilityAndQuotedOwner)
{
    std::ostringstream stream;
    systems::appendEnergyChannelSource(
        stream,
        "stored_kinetic",
        EnergyChannelSource{.applicability = EnergyChannelApplicability::NotApplicable,
                            .owner = "Owner.with \"quote\""});
    EXPECT_EQ(stream.str(),
              " stored_kinetic_applicability=not_applicable"
              " stored_kinetic_owner=\"Owner.with \\\"quote\\\"\"");
}

TEST(EnergyBalanceLedger, ResidualStepWorkIsMinusDtTimesThePairingInDeclaredOrder)
{
    const std::vector<std::string> channels{"convection", "weak_boundary", "ghost_penalty"};
    const std::vector<systems::ResidualWorkPairing> pairings{
        {.channel = "ghost_penalty", .produced = false, .owner_component = "owner.ghost"},
        {.channel = "convection",
         .produced = true,
         .owner_component = "owner.convection",
         .operator_tag = "residual",
         .residual_state_pairing = 0.3},
        {.channel = "weak_boundary",
         .produced = true,
         .owner_component = "owner.boundary",
         .operator_tag = "residual",
         .residual_state_pairing = -1.1},
    };
    const auto works = systems::evaluateResidualStepWork(pairings, channels, 0.25, "test work");
    ASSERT_EQ(works.size(), 3u);
    EXPECT_TRUE(sameBits(works[0], -0.25 * 0.3));
    EXPECT_TRUE(sameBits(works[1], -0.25 * -1.1));
    EXPECT_TRUE(sameBits(works[2], 0.0));

    auto duplicate = pairings;
    duplicate[0].channel = "convection";
    EXPECT_THROW((void)systems::evaluateResidualStepWork(duplicate, channels, 0.25, "test work"),
                 std::invalid_argument);
    auto unknown = pairings;
    unknown[0].channel = "pruning";
    EXPECT_THROW((void)systems::evaluateResidualStepWork(unknown, channels, 0.25, "test work"),
                 std::invalid_argument);
    auto untagged = pairings;
    untagged[1].operator_tag.clear();
    EXPECT_THROW((void)systems::evaluateResidualStepWork(untagged, channels, 0.25, "test work"),
                 std::invalid_argument);
    auto nonzero = pairings;
    nonzero[0].residual_state_pairing = 1.0;
    EXPECT_THROW((void)systems::evaluateResidualStepWork(nonzero, channels, 0.25, "test work"),
                 std::invalid_argument);
    EXPECT_THROW((void)systems::evaluateResidualStepWork(pairings, channels, 0.0, "test work"),
                 std::invalid_argument);
    EXPECT_THROW(
        (void)systems::evaluateResidualStepWork(
            std::span(pairings).first(2), channels, 0.25, "test work"),
        std::invalid_argument);
}

} // namespace
