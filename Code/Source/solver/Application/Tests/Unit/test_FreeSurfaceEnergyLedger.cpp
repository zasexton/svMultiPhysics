/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Application/Core/FreeSurfaceEnergyLedger.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using application::core::FreeSurfaceEnergyAttemptMetadata;
using application::core::FreeSurfaceEnergyAttemptStatus;
using application::core::FreeSurfaceEnergyChannelApplicability;
using application::core::FreeSurfaceEnergyChannelSource;
using application::core::FreeSurfaceEnergyChannelSources;
using application::core::FreeSurfaceEnergyRejectionReason;
using application::core::FreeSurfaceEnergyTemporalScheme;
using application::core::FreeSurfaceEnergyLedger;
using application::core::FreeSurfaceExternalWork;
using application::core::FreeSurfaceGasEnergyApplicability;
using application::core::FreeSurfaceNumericalWork;
using application::core::FreeSurfacePhysicalDissipationRate;
using application::core::FreeSurfaceStoredEnergy;

FreeSurfaceEnergyAttemptMetadata metadata(
    std::uint64_t transaction_id,
    std::uint64_t cut_topology_before = 31u,
    std::uint64_t cut_topology_after = 31u)
{
  return FreeSurfaceEnergyAttemptMetadata{
      .transaction_id = transaction_id,
      .step = 4u,
      .attempt = 1u,
      .time_before = 0.75,
      .time_after = 1.0,
      .dt = 0.25,
      .temporal_scheme =
          FreeSurfaceEnergyTemporalScheme::BackwardEuler,
      .physical_evaluation_time = 1.0,
      .physical_evaluation_stage_fraction = 1.0,
      .algebraic_state_revision_before = 11u,
      .physical_endpoint_algebraic_state_revision = 12u,
      .algebraic_state_revision_after = 13u,
      .snapshot_set_revision_before = 21u,
      .physical_endpoint_snapshot_set_revision = 22u,
      .snapshot_set_revision_after = 23u,
      .mesh_topology_set_revision_before = 7u,
      .physical_endpoint_mesh_topology_set_revision = 7u,
      .mesh_topology_set_revision_after = 7u,
      .cut_topology_set_revision_before = cut_topology_before,
      .physical_endpoint_cut_topology_set_revision =
          cut_topology_before,
      .cut_topology_set_revision_after = cut_topology_after,
      .extension_map_revision_before = 41u,
      .physical_endpoint_extension_map_revision = 42u,
      .extension_map_revision_after = 43u,
  };
}

FreeSurfaceEnergyAttemptMetadata continuationMetadata(
    std::uint64_t transaction_id)
{
  return FreeSurfaceEnergyAttemptMetadata{
      .transaction_id = transaction_id,
      .step = 5u,
      .attempt = 1u,
      .time_before = 1.0,
      .time_after = 1.25,
      .dt = 0.25,
      .temporal_scheme =
          FreeSurfaceEnergyTemporalScheme::BackwardEuler,
      .physical_evaluation_time = 1.25,
      .physical_evaluation_stage_fraction = 1.0,
      .algebraic_state_revision_before = 13u,
      .physical_endpoint_algebraic_state_revision = 14u,
      .algebraic_state_revision_after = 15u,
      .snapshot_set_revision_before = 23u,
      .physical_endpoint_snapshot_set_revision = 24u,
      .snapshot_set_revision_after = 25u,
      .mesh_topology_set_revision_before = 7u,
      .physical_endpoint_mesh_topology_set_revision = 7u,
      .mesh_topology_set_revision_after = 7u,
      .cut_topology_set_revision_before = 31u,
      .physical_endpoint_cut_topology_set_revision = 31u,
      .cut_topology_set_revision_after = 31u,
      .extension_map_revision_before = 43u,
      .physical_endpoint_extension_map_revision = 44u,
      .extension_map_revision_after = 45u,
  };
}

FreeSurfaceEnergyAttemptMetadata metadataForAttempt(
    std::uint64_t transaction_id,
    std::uint64_t attempt)
{
  auto value = metadata(transaction_id);
  value.attempt = attempt;
  return value;
}

FreeSurfaceEnergyAttemptMetadata partialMetadata(
    std::uint64_t transaction_id)
{
  auto partial = metadata(transaction_id);
  partial.temporal_scheme =
      FreeSurfaceEnergyTemporalScheme::Unspecified;
  partial.physical_evaluation_time =
      std::numeric_limits<double>::quiet_NaN();
  partial.physical_evaluation_stage_fraction =
      std::numeric_limits<double>::quiet_NaN();
  partial.physical_endpoint_algebraic_state_revision = 0u;
  partial.algebraic_state_revision_after = 0u;
  partial.physical_endpoint_snapshot_set_revision = 0u;
  partial.snapshot_set_revision_after = 0u;
  partial.physical_endpoint_mesh_topology_set_revision = 0u;
  partial.mesh_topology_set_revision_after = 0u;
  partial.physical_endpoint_cut_topology_set_revision = 0u;
  partial.cut_topology_set_revision_after = 0u;
  partial.physical_endpoint_extension_map_revision.reset();
  partial.extension_map_revision_after.reset();
  return partial;
}

FreeSurfaceStoredEnergy storedEnergy(
    double kinetic,
    double gravitational,
    double surface,
    double wall)
{
  return FreeSurfaceStoredEnergy{
      .kinetic = kinetic,
      .gravitational = gravitational,
      .liquid_gas_surface = surface,
      .solid_liquid_wall = wall,
      .gas_applicability =
          FreeSurfaceGasEnergyApplicability::NotApplicable,
      .gas_or_compressibility = 0.0,
  };
}

FreeSurfacePhysicalDissipationRate dissipation()
{
  return FreeSurfacePhysicalDissipationRate{
      .bulk_viscous = 0.5,
      .navier_slip = 1.0,
      .line_friction = 0.5,
  };
}

FreeSurfaceExternalWork externalWork()
{
  return FreeSurfaceExternalWork{
      .pressure = 0.4,
      .body_force = 0.1,
      .imposed_traction = 0.0,
      .open_boundary_flux = 0.0,
  };
}

FreeSurfaceNumericalWork numericalWork()
{
  return FreeSurfaceNumericalWork{
      .time_discretization = -0.03,
      .kinetic_domain_transport = 0.04,
      .gravitational_transport_coupling = -0.02,
      .convection = 0.01,
      .pressure_continuity = -0.02,
      .surface_transport_coupling = 0.03,
      .weak_boundary = -0.01,
      .vms_pspg = 0.10,
      .cut_stabilization = 0.02,
      .ghost_penalty = 0.01,
      .aggregation = 0.02,
      .extension = 0.03,
      .pruning = 0.04,
      .limiting = 0.05,
      .redistancing = 0.06,
      .local_reconciliation = 0.07,
      .global_correction = 0.08,
  };
}

FreeSurfaceEnergyChannelSources channelSources()
{
  const auto produced = [](const char* owner) {
    return FreeSurfaceEnergyChannelSource{
        .applicability =
            FreeSurfaceEnergyChannelApplicability::Produced,
        .owner = owner,
    };
  };
  const auto inapplicable = [](const char* owner) {
    return FreeSurfaceEnergyChannelSource{
        .applicability =
            FreeSurfaceEnergyChannelApplicability::NotApplicable,
        .owner = owner,
    };
  };
  return FreeSurfaceEnergyChannelSources{
      .stored =
          {
              .kinetic = produced("test.active_volume"),
              .gravitational = produced("test.active_volume"),
              .liquid_gas_surface = produced("test.snapshot"),
              .solid_liquid_wall = produced("test.snapshot"),
              .gas_or_compressibility =
                  inapplicable("test.one_phase_model"),
          },
      .dissipation =
          {
              .bulk_viscous = produced("test.viscous_form"),
              .navier_slip = produced("test.wall_form"),
              .line_friction = produced("test.contact_form"),
          },
      .external =
          {
              .pressure = produced("test.pressure_form"),
              .body_force = produced("test.body_force_form"),
              .imposed_traction = produced("test.traction_form"),
              .open_boundary_flux =
                  produced("test.open_boundary_form"),
          },
      .numerical =
          {
              .time_discretization =
                  produced("test.backward_euler_form"),
              .kinetic_domain_transport =
                  produced("test.kinetic_domain_transport"),
              .gravitational_transport_coupling =
                  produced("test.gravitational_transport"),
              .convection = produced("test.convection_form"),
              .pressure_continuity =
                  produced("test.pressure_continuity_form"),
              .surface_transport_coupling =
                  produced("test.surface_transport"),
              .weak_boundary =
                  produced("test.weak_boundary_form"),
              .vms_pspg = produced("test.vms_pspg_form"),
              .cut_stabilization =
                  produced("test.cut_stabilization_form"),
              .ghost_penalty =
                  produced("test.ghost_penalty_form"),
              .aggregation = produced("test.aggregation"),
              .extension = produced("test.extension"),
              .pruning = produced("test.pruning"),
              .limiting = produced("test.limiting"),
              .redistancing = produced("test.redistancing"),
              .local_reconciliation =
                  produced("test.local_reconciliation"),
              .global_correction =
                  produced("test.global_correction"),
          },
  };
}

void stageWithoutMaintenance(
    FreeSurfaceEnergyLedger& ledger,
    FreeSurfaceStoredEnergy before,
    FreeSurfaceStoredEnergy after,
    FreeSurfacePhysicalDissipationRate dissipation_rate,
    FreeSurfaceExternalWork external_work,
    FreeSurfaceNumericalWork numerical_work,
    FreeSurfaceEnergyChannelSources channel_sources)
{
  ledger.stageBalance(
      before,
      after,
      after,
      dissipation_rate,
      external_work,
      numerical_work,
      channel_sources);
}

TEST(FreeSurfaceEnergyLedger,
     CommitsOneCompleteBackwardEulerFixedTopologyBalance)
{
  FreeSurfaceEnergyLedger ledger;
  ledger.beginAttempt(metadata(1u));
  ledger.stageBalance(
      storedEnergy(2.0, 3.0, 4.0, -1.0),
      storedEnergy(2.4, 3.1, 4.0, -0.9),
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      dissipation(),
      externalWork(),
      numericalWork(),
      channelSources());

  ASSERT_NE(ledger.trialBalance(), nullptr);
  EXPECT_EQ(
      ledger.trialBalance()->status,
      FreeSurfaceEnergyAttemptStatus::Trial);
  EXPECT_NEAR(ledger.trialBalance()->stored_energy_before, 8.0, 1e-14);
  EXPECT_NEAR(
      ledger.trialBalance()
          ->stored_energy_physical_endpoint_before_maintenance,
      8.6,
      1e-14);
  EXPECT_NEAR(ledger.trialBalance()->stored_energy_after, 8.7, 1e-14);
  EXPECT_NEAR(
      ledger.trialBalance()->physical_stored_energy_change,
      0.6,
      1e-14);
  EXPECT_NEAR(
      ledger.trialBalance()->maintenance_stored_energy_change,
      0.1,
      1e-14);
  EXPECT_NEAR(ledger.trialBalance()->stored_energy_change, 0.7, 1e-14);
  EXPECT_NEAR(
      ledger.trialBalance()->integrated_physical_dissipation,
      0.5,
      1e-14);
  EXPECT_NEAR(ledger.trialBalance()->total_external_work, 0.5, 1e-14);
  EXPECT_NEAR(ledger.trialBalance()->total_numerical_work, 0.48, 1e-14);
  EXPECT_NEAR(ledger.trialBalance()->trial_balance_residual, 0.22, 1e-14);
  EXPECT_DOUBLE_EQ(ledger.trialBalance()->accepted_balance_residual, 0.0);

  ledger.commitAttempt();

  EXPECT_FALSE(ledger.attemptActive());
  EXPECT_EQ(ledger.trialBalance(), nullptr);
  ASSERT_EQ(ledger.acceptedAttempts().size(), 1u);
  EXPECT_EQ(
      ledger.acceptedAttempts().front().status,
      FreeSurfaceEnergyAttemptStatus::Accepted);
  EXPECT_EQ(
      ledger.acceptedAttempts().front().metadata.temporal_scheme,
      FreeSurfaceEnergyTemporalScheme::BackwardEuler);
  EXPECT_DOUBLE_EQ(
      ledger.acceptedAttempts().front()
          .metadata.physical_evaluation_time,
      ledger.acceptedAttempts().front().metadata.time_after);
  EXPECT_DOUBLE_EQ(
      ledger.acceptedAttempts().front()
          .metadata.physical_evaluation_stage_fraction,
      1.0);
  EXPECT_NEAR(
      ledger.acceptedAttempts().front().accepted_balance_residual,
      0.22,
      1e-14);
  EXPECT_NEAR(
      ledger.acceptedAttempts().front().accepted_stored_energy_change,
      0.7,
      1e-14);
  EXPECT_NEAR(
      ledger.acceptedAttempts().front()
          .accepted_physical_stored_energy_change,
      0.6,
      1e-14);
  EXPECT_NEAR(
      ledger.acceptedAttempts().front()
          .accepted_maintenance_stored_energy_change,
      0.1,
      1e-14);
  EXPECT_NEAR(
      ledger.acceptedAttempts().front()
          .accepted_integrated_physical_dissipation,
      0.5,
      1e-14);
  EXPECT_NEAR(
      ledger.acceptedAttempts().front().accepted_external_work,
      0.5,
      1e-14);
  EXPECT_NEAR(
      ledger.acceptedAttempts().front().accepted_numerical_work,
      0.48,
      1e-14);
  EXPECT_TRUE(ledger.rejectedAttempts().empty());
}

TEST(FreeSurfaceEnergyLedger,
     StageBalanceRequiresExplicitBackwardEulerAcceptedEndpointIdentity)
{
  const auto expect_stage_rejection = [](
      const FreeSurfaceEnergyAttemptMetadata& invalid_metadata) {
    FreeSurfaceEnergyLedger ledger;
    ledger.beginAttempt(invalid_metadata);
    EXPECT_THROW(
        stageWithoutMaintenance(ledger,
            storedEnergy(2.0, 3.0, 4.0, -1.0),
            storedEnergy(2.5, 3.2, 3.8, -0.8),
            dissipation(),
            externalWork(),
            numericalWork(),
            channelSources()),
        std::invalid_argument);
    EXPECT_TRUE(ledger.attemptActive());
    EXPECT_EQ(ledger.trialBalance(), nullptr);
    ledger.rejectUnstagedAttempt(
        FreeSurfaceEnergyRejectionReason::PreacceptRejection);
    ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
    EXPECT_FALSE(ledger.rejectedAttempts().front().balance_staged);
    EXPECT_EQ(
        ledger.rejectedAttempts().front().metadata.temporal_scheme,
        invalid_metadata.temporal_scheme);
  };

  auto unidentified = metadata(1u);
  unidentified.temporal_scheme =
      FreeSurfaceEnergyTemporalScheme::Unspecified;
  expect_stage_rejection(unidentified);

  auto generalized_alpha = metadata(1u);
  generalized_alpha.temporal_scheme =
      FreeSurfaceEnergyTemporalScheme::GeneralizedAlpha;
  expect_stage_rejection(generalized_alpha);

  auto nonendpoint_backward_euler = metadata(1u);
  nonendpoint_backward_euler.physical_evaluation_stage_fraction = 0.5;
  expect_stage_rejection(nonendpoint_backward_euler);

  auto mismatched_endpoint_time = metadata(1u);
  mismatched_endpoint_time.physical_evaluation_time =
      std::nextafter(mismatched_endpoint_time.time_after, 0.75);
  expect_stage_rejection(mismatched_endpoint_time);
}

TEST(FreeSurfaceEnergyLedger,
     AcceptedHistoryRequiresContinuousEndpointAndOwnerProvenance)
{
  FreeSurfaceEnergyLedger ledger;
  ledger.beginAttempt(metadata(1u));
  stageWithoutMaintenance(ledger,
      storedEnergy(2.0, 3.0, 4.0, -1.0),
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      dissipation(),
      externalWork(),
      numericalWork(),
      channelSources());
  ledger.commitAttempt();

  auto invalid_chain = continuationMetadata(2u);
  invalid_chain.algebraic_state_revision_before = 99u;
  EXPECT_THROW(
      ledger.beginAttempt(invalid_chain),
      std::invalid_argument);
  EXPECT_FALSE(ledger.attemptActive());

  auto invalid_time_chain = continuationMetadata(2u);
  invalid_time_chain.time_before =
      std::nextafter(invalid_time_chain.time_before, 2.0);
  invalid_time_chain.time_after =
      invalid_time_chain.time_before + invalid_time_chain.dt;
  EXPECT_DOUBLE_EQ(
      invalid_time_chain.time_after - invalid_time_chain.time_before,
      invalid_time_chain.dt);
  EXPECT_THROW(
      ledger.beginAttempt(invalid_time_chain),
      std::invalid_argument);
  EXPECT_FALSE(ledger.attemptActive());

  ledger.beginAttempt(continuationMetadata(2u));
  auto changed_owner = channelSources();
  changed_owner.numerical.extension.owner =
      "test.changed_extension_owner";
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          storedEnergy(2.6, 3.1, 3.7, -0.7),
          dissipation(),
          externalWork(),
          numericalWork(),
          changed_owner),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  auto changed_start = storedEnergy(2.5, 3.2, 3.8, -0.8);
  changed_start.kinetic =
      std::nextafter(changed_start.kinetic, 3.0);
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          changed_start,
          storedEnergy(2.6, 3.1, 3.7, -0.7),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  stageWithoutMaintenance(ledger,
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      storedEnergy(2.6, 3.1, 3.7, -0.7),
      dissipation(),
      externalWork(),
      numericalWork(),
      channelSources());
  ledger.commitAttempt();
  ASSERT_EQ(ledger.acceptedAttempts().size(), 2u);
  EXPECT_EQ(
      ledger.acceptedAttempts().back()
          .metadata.algebraic_state_revision_before,
      ledger.acceptedAttempts().front()
          .metadata.algebraic_state_revision_after);
  EXPECT_DOUBLE_EQ(
      ledger.acceptedAttempts().front()
          .metadata.physical_evaluation_time,
      ledger.acceptedAttempts().back().metadata.time_before);
  EXPECT_EQ(
      ledger.acceptedAttempts().back().metadata.temporal_scheme,
      FreeSurfaceEnergyTemporalScheme::BackwardEuler);
}

TEST(FreeSurfaceEnergyLedger,
     RejectedHistoryAlsoLocksChannelOwnerProvenance)
{
  FreeSurfaceEnergyLedger ledger;
  ledger.beginAttempt(metadata(1u));
  stageWithoutMaintenance(ledger,
      storedEnergy(2.0, 3.0, 4.0, -1.0),
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      dissipation(),
      externalWork(),
      numericalWork(),
      channelSources());
  ledger.rejectAttempt(
      FreeSurfaceEnergyRejectionReason::StepControllerRejection);

  ledger.beginAttempt(metadataForAttempt(2u, 2u));
  ledger.rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason::NonlinearSolveFailure);

  ledger.beginAttempt(metadataForAttempt(3u, 3u));
  auto changed_owner = channelSources();
  changed_owner.numerical.kinetic_domain_transport.owner =
      "test.changed_kinetic_domain_transport_owner";
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.0, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          changed_owner),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.1, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  stageWithoutMaintenance(ledger,
      storedEnergy(2.0, 3.0, 4.0, -1.0),
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      dissipation(),
      externalWork(),
      numericalWork(),
      channelSources());
  ledger.rejectAttempt(
      FreeSurfaceEnergyRejectionReason::StepControllerRejection);
  ASSERT_EQ(ledger.rejectedAttempts().size(), 3u);
  EXPECT_FALSE(ledger.rejectedAttempts()[1].balance_staged);
}

TEST(FreeSurfaceEnergyLedger,
     RejectedRetryMayChangeDurationButMustRestageBackwardEulerEndpoint)
{
  FreeSurfaceEnergyLedger ledger;
  ledger.beginAttempt(metadata(1u));
  stageWithoutMaintenance(ledger,
      storedEnergy(2.0, 3.0, 4.0, -1.0),
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      dissipation(),
      externalWork(),
      numericalWork(),
      channelSources());
  ledger.rejectAttempt(
      FreeSurfaceEnergyRejectionReason::StepControllerRejection);

  auto generalized_alpha_retry = metadataForAttempt(2u, 2u);
  generalized_alpha_retry.time_after = 0.875;
  generalized_alpha_retry.dt = 0.125;
  generalized_alpha_retry.temporal_scheme =
      FreeSurfaceEnergyTemporalScheme::GeneralizedAlpha;
  generalized_alpha_retry.physical_evaluation_time = 0.875;
  generalized_alpha_retry.physical_evaluation_stage_fraction = 1.0;
  ledger.beginAttempt(generalized_alpha_retry);
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.0, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  ledger.rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason::PreacceptRejection);

  auto backward_euler_retry = metadataForAttempt(3u, 3u);
  backward_euler_retry.time_after = 0.8125;
  backward_euler_retry.dt = 0.0625;
  backward_euler_retry.physical_evaluation_time = 0.8125;
  ledger.beginAttempt(backward_euler_retry);
  stageWithoutMaintenance(ledger,
      storedEnergy(2.0, 3.0, 4.0, -1.0),
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      dissipation(),
      externalWork(),
      numericalWork(),
      channelSources());
  ledger.commitAttempt();

  ASSERT_EQ(ledger.rejectedAttempts().size(), 2u);
  EXPECT_EQ(
      ledger.rejectedAttempts().back().metadata.temporal_scheme,
      FreeSurfaceEnergyTemporalScheme::GeneralizedAlpha);
  EXPECT_FALSE(ledger.rejectedAttempts().back().balance_staged);
  ASSERT_EQ(ledger.acceptedAttempts().size(), 1u);
  const auto& accepted = ledger.acceptedAttempts().front().metadata;
  EXPECT_EQ(accepted.attempt, 3u);
  EXPECT_EQ(
      accepted.temporal_scheme,
      FreeSurfaceEnergyTemporalScheme::BackwardEuler);
  EXPECT_DOUBLE_EQ(accepted.dt, 0.0625);
  EXPECT_DOUBLE_EQ(accepted.physical_evaluation_time, accepted.time_after);
  EXPECT_DOUBLE_EQ(accepted.physical_evaluation_stage_fraction, 1.0);
}

TEST(FreeSurfaceEnergyLedger,
     UnstagedRejectionPreservesReasonWithoutInventingBalanceValues)
{
  FreeSurfaceEnergyLedger ledger;
  ledger.beginAttempt(partialMetadata(1u));

  EXPECT_THROW(
      ledger.rejectUnstagedAttempt(
          FreeSurfaceEnergyRejectionReason::None),
      std::invalid_argument);
  EXPECT_TRUE(ledger.attemptActive());
  ledger.rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason::NonlinearSolveFailure);

  EXPECT_FALSE(ledger.attemptActive());
  EXPECT_EQ(ledger.trialBalance(), nullptr);
  EXPECT_TRUE(ledger.acceptedAttempts().empty());
  ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
  const auto& rejected = ledger.rejectedAttempts().front();
  EXPECT_EQ(
      rejected.status,
      FreeSurfaceEnergyAttemptStatus::Rejected);
  EXPECT_EQ(
      rejected.rejection_reason,
      FreeSurfaceEnergyRejectionReason::NonlinearSolveFailure);
  EXPECT_FALSE(rejected.balance_staged);
  EXPECT_TRUE(std::isnan(rejected.stored_energy_before));
  EXPECT_TRUE(std::isnan(rejected.trial_balance_residual));
  EXPECT_DOUBLE_EQ(rejected.accepted_stored_energy_change, 0.0);
  EXPECT_DOUBLE_EQ(
      rejected.accepted_physical_stored_energy_change,
      0.0);
  EXPECT_DOUBLE_EQ(
      rejected.accepted_maintenance_stored_energy_change,
      0.0);
  EXPECT_DOUBLE_EQ(
      rejected.accepted_integrated_physical_dissipation,
      0.0);
  EXPECT_DOUBLE_EQ(rejected.accepted_external_work, 0.0);
  EXPECT_DOUBLE_EQ(rejected.accepted_numerical_work, 0.0);
  EXPECT_DOUBLE_EQ(rejected.accepted_balance_residual, 0.0);

  auto topology_change = metadataForAttempt(2u, 2u);
  topology_change.physical_endpoint_cut_topology_set_revision = 32u;
  ledger.beginAttempt(topology_change);
  EXPECT_THROW(
      ledger.rejectUnstagedAttempt(
          FreeSurfaceEnergyRejectionReason::PreacceptRejection),
      std::invalid_argument);
  ledger.rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason::TopologyChange);
  ASSERT_EQ(ledger.rejectedAttempts().size(), 2u);
  EXPECT_EQ(
      ledger.rejectedAttempts().back().rejection_reason,
      FreeSurfaceEnergyRejectionReason::TopologyChange);
}

TEST(FreeSurfaceEnergyLedger,
     PartialEndpointProvenanceCannotStageOrClaimTopologyChange)
{
  FreeSurfaceEnergyLedger ledger;
  ledger.beginAttempt(partialMetadata(1u));
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.0, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);
  EXPECT_THROW(
      ledger.rejectUnstagedAttempt(
          FreeSurfaceEnergyRejectionReason::TopologyChange),
      std::invalid_argument);
  ledger.rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason::StepControllerRejection);
  ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
  EXPECT_FALSE(ledger.rejectedAttempts().front().balance_staged);

  FreeSurfaceEnergyLedger partial_extension_ledger;
  auto partial_extension = metadata(1u);
  partial_extension.physical_endpoint_extension_map_revision.reset();
  partial_extension.extension_map_revision_after.reset();
  partial_extension_ledger.beginAttempt(partial_extension);
  EXPECT_THROW(
      stageWithoutMaintenance(partial_extension_ledger,
          storedEnergy(2.0, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  partial_extension_ledger.rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason::PreacceptRejection);
}

TEST(FreeSurfaceEnergyLedger,
     TopologyChangeCannotCommitAndRejectedAttemptContributesZero)
{
  // Retain the registered fixture identity while enforcing the stronger
  // pre-staging fixed-topology boundary.
  FreeSurfaceEnergyLedger ledger;
  auto topology_change = metadata(1u);
  topology_change.physical_endpoint_cut_topology_set_revision = 32u;
  ledger.beginAttempt(topology_change);
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.0, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  EXPECT_TRUE(ledger.attemptActive());
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  EXPECT_THROW(
      ledger.rejectUnstagedAttempt(
          FreeSurfaceEnergyRejectionReason::PreacceptRejection),
      std::invalid_argument);
  ledger.rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason::TopologyChange);

  EXPECT_TRUE(ledger.acceptedAttempts().empty());
  ASSERT_EQ(ledger.rejectedAttempts().size(), 1u);
  const auto& rejected = ledger.rejectedAttempts().front();
  EXPECT_EQ(rejected.status, FreeSurfaceEnergyAttemptStatus::Rejected);
  EXPECT_EQ(
      rejected.rejection_reason,
      FreeSurfaceEnergyRejectionReason::TopologyChange);
  EXPECT_FALSE(rejected.balance_staged);
  EXPECT_TRUE(std::isnan(rejected.trial_balance_residual));
  EXPECT_DOUBLE_EQ(rejected.accepted_balance_residual, 0.0);
  EXPECT_DOUBLE_EQ(rejected.accepted_stored_energy_change, 0.0);
  EXPECT_DOUBLE_EQ(
      rejected.accepted_physical_stored_energy_change,
      0.0);
  EXPECT_DOUBLE_EQ(
      rejected.accepted_maintenance_stored_energy_change,
      0.0);
  EXPECT_DOUBLE_EQ(
      rejected.accepted_integrated_physical_dissipation,
      0.0);
  EXPECT_DOUBLE_EQ(rejected.accepted_external_work, 0.0);
  EXPECT_DOUBLE_EQ(rejected.accepted_numerical_work, 0.0);
}

TEST(FreeSurfaceEnergyLedger,
     MissingChannelsAndNegativeDissipationFailClosed)
{
  FreeSurfaceEnergyLedger ledger;
  ledger.beginAttempt(metadata(1u));

  auto invalid_kinetic =
      storedEnergy(-1.0, 3.0, 4.0, -1.0);
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          invalid_kinetic,
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  auto invalid_surface =
      storedEnergy(2.0, 3.0, -1.0, -1.0);
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          invalid_surface,
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  auto invalid_dissipation = dissipation();
  invalid_dissipation.bulk_viscous = -1.0;
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.0, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          invalid_dissipation,
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  auto missing_numerical_channel = numericalWork();
  missing_numerical_channel.vms_pspg =
      std::numeric_limits<double>::quiet_NaN();
  try {
    stageWithoutMaintenance(ledger,
        storedEnergy(2.0, 3.0, 4.0, -1.0),
        storedEnergy(2.5, 3.2, 3.8, -0.8),
        dissipation(),
        externalWork(),
        missing_numerical_channel,
        channelSources());
    FAIL() << "A missing numerical-work channel was accepted.";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(
        std::string(error.what()).find("numerical.vms_pspg"),
        std::string::npos);
  }
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  stageWithoutMaintenance(ledger,
      storedEnergy(2.0, 3.0, 4.0, -1.0),
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      dissipation(),
      externalWork(),
      numericalWork(),
      channelSources());
  EXPECT_THROW(
      ledger.rejectAttempt(FreeSurfaceEnergyRejectionReason::None),
      std::invalid_argument);
  ledger.rejectAttempt(
      FreeSurfaceEnergyRejectionReason::PreacceptRejection);
}

TEST(FreeSurfaceEnergyLedger,
     EveryChannelRequiresOneNamedApplicableOwner)
{
  FreeSurfaceEnergyLedger ledger;
  ledger.beginAttempt(metadata(1u));

  auto missing_owner = channelSources();
  missing_owner.numerical.extension.owner.clear();
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.0, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          missing_owner),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  auto nonzero_inapplicable = channelSources();
  nonzero_inapplicable.numerical.extension.applicability =
      FreeSurfaceEnergyChannelApplicability::NotApplicable;
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.0, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          nonzero_inapplicable),
      std::invalid_argument);
  EXPECT_EQ(ledger.trialBalance(), nullptr);

  auto values = numericalWork();
  values.extension = 0.0;
  stageWithoutMaintenance(ledger,
      storedEnergy(2.0, 3.0, 4.0, -1.0),
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      dissipation(),
      externalWork(),
      values,
      nonzero_inapplicable);
  ASSERT_NE(ledger.trialBalance(), nullptr);
  EXPECT_EQ(
      ledger.trialBalance()
          ->channel_sources.numerical.extension.applicability,
      FreeSurfaceEnergyChannelApplicability::NotApplicable);
  EXPECT_EQ(
      ledger.trialBalance()->channel_sources.numerical.extension.owner,
      "test.extension");
  EXPECT_THROW(
      ledger.rejectAttempt(
          FreeSurfaceEnergyRejectionReason::TopologyChange),
      std::invalid_argument);
  ledger.rejectAttempt(
      FreeSurfaceEnergyRejectionReason::PreacceptRejection);
}

TEST(FreeSurfaceEnergyLedger,
     RequiresOneEndpointIntervalAndIncreasingTransactionIdentifiers)
{
  FreeSurfaceEnergyLedger ledger;
  auto invalid_step = metadata(1u);
  invalid_step.step = 0u;
  EXPECT_THROW(
      ledger.beginAttempt(invalid_step),
      std::invalid_argument);

  auto invalid_interval = metadata(1u);
  invalid_interval.time_after = 0.9;
  EXPECT_THROW(
      ledger.beginAttempt(invalid_interval),
      std::invalid_argument);

  auto invalid_mesh_revision = metadata(1u);
  invalid_mesh_revision.mesh_topology_set_revision_before = 0u;
  EXPECT_THROW(
      ledger.beginAttempt(invalid_mesh_revision),
      std::invalid_argument);

  auto invalid_physical_revision = metadata(1u);
  invalid_physical_revision
      .physical_endpoint_algebraic_state_revision = 0u;
  ledger.beginAttempt(invalid_physical_revision);
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          storedEnergy(2.0, 3.0, 4.0, -1.0),
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);
  ledger.rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason::NonlinearSolveFailure);

  EXPECT_THROW(
      ledger.beginAttempt(metadata(2u)),
      std::invalid_argument);
  auto skipped_step = continuationMetadata(2u);
  EXPECT_THROW(
      ledger.beginAttempt(skipped_step),
      std::invalid_argument);
  auto changed_start = metadataForAttempt(2u, 2u);
  changed_start.algebraic_state_revision_before = 99u;
  EXPECT_THROW(
      ledger.beginAttempt(changed_start),
      std::invalid_argument);

  ledger.beginAttempt(metadataForAttempt(2u, 2u));
  stageWithoutMaintenance(ledger,
      storedEnergy(2.0, 3.0, 4.0, -1.0),
      storedEnergy(2.5, 3.2, 3.8, -0.8),
      dissipation(),
      externalWork(),
      numericalWork(),
      channelSources());
  ledger.commitAttempt();
  ASSERT_EQ(ledger.acceptedAttempts().size(), 1u);
  EXPECT_EQ(ledger.acceptedAttempts().front().metadata.attempt, 2u);

  EXPECT_THROW(
      ledger.beginAttempt(metadataForAttempt(2u, 3u)),
      std::invalid_argument);
  auto invalid_next_attempt = continuationMetadata(3u);
  invalid_next_attempt.attempt = 2u;
  EXPECT_THROW(
      ledger.beginAttempt(invalid_next_attempt),
      std::invalid_argument);
  ledger.beginAttempt(continuationMetadata(3u));
  ledger.rejectUnstagedAttempt(
      FreeSurfaceEnergyRejectionReason::NonlinearSolveFailure);
  ASSERT_EQ(ledger.rejectedAttempts().size(), 2u);
  EXPECT_EQ(ledger.rejectedAttempts().back().metadata.step, 5u);
  EXPECT_EQ(ledger.rejectedAttempts().back().metadata.attempt, 1u);
}

TEST(FreeSurfaceEnergyLedger,
     GasApplicabilityMustBeExplicitAndStableAcrossTheStep)
{
  FreeSurfaceEnergyLedger ledger;
  ledger.beginAttempt(metadata(1u));

  auto before = storedEnergy(2.0, 3.0, 4.0, -1.0);
  before.gas_applicability =
      FreeSurfaceGasEnergyApplicability::Unspecified;
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          before,
          storedEnergy(2.5, 3.2, 3.8, -0.8),
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);

  before = storedEnergy(2.0, 3.0, 4.0, -1.0);
  auto after = storedEnergy(2.5, 3.2, 3.8, -0.8);
  after.gas_applicability =
      FreeSurfaceGasEnergyApplicability::Active;
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          before,
          after,
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);

  before.gas_applicability =
      FreeSurfaceGasEnergyApplicability::Active;
  before.gas_or_compressibility = 0.2;
  after.gas_or_compressibility = 0.3;
  EXPECT_THROW(
      stageWithoutMaintenance(ledger,
          before,
          after,
          dissipation(),
          externalWork(),
          numericalWork(),
          channelSources()),
      std::invalid_argument);

  auto active_gas_sources = channelSources();
  active_gas_sources.stored.gas_or_compressibility.applicability =
      FreeSurfaceEnergyChannelApplicability::Produced;
  stageWithoutMaintenance(ledger,
      before,
      after,
      dissipation(),
      externalWork(),
      numericalWork(),
      active_gas_sources);
  ledger.rejectAttempt(
      FreeSurfaceEnergyRejectionReason::PreacceptRejection);
}

} // namespace
