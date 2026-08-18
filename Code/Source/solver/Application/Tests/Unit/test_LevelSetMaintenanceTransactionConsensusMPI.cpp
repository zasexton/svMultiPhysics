#include <gtest/gtest.h>

#include "Application/Core/ApplicationDriver.h"
#include "Application/Core/LevelSetMaintenanceTransactionConsensus.h"
#include "Mesh/Core/MeshComm.h"

#include <mpi.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace {

using application::core::LevelSetAuthoritativeFunctionalValue;
using application::core::LevelSetMaintenanceDeclaredStage;
using application::core::LevelSetMaintenanceTransactionDecision;
using application::core::LevelSetMaintenanceWorkLedger;
using application::core::LevelSetMaintenanceWorkRow;
using application::core::LevelSetMaintenanceWorkSubstage;
using application::core::LevelSetMaintenanceWorkTransaction;

enum class TransactionDrift : std::uint8_t {
  None,
  FunctionalContent,
  ModeledEnergyContent,
  AlgebraicRevision,
  SnapshotRevision,
  MeshTopologyRevision,
  CutTopologyRevision,
  ExtensionMapRevision,
  MissingActiveTransaction,
  RowCount,
  CurrentContentRevision,
  HistoryContentRevision,
  GeometryTransactionPresence,
  GeometryRevisionContent
};

LevelSetAuthoritativeFunctionalValue makeFunctional(
    double total_potential,
    std::uint64_t snapshot_revision,
    std::uint64_t mesh_topology_revision,
    std::uint64_t cut_topology_revision)
{
  return LevelSetAuthoritativeFunctionalValue{
      .interface_marker = 407,
      .snapshot_revision = snapshot_revision,
      .mesh_topology_revision = mesh_topology_revision,
      .cut_topology_revision = cut_topology_revision,
      .liquid_volume = 1.25,
      .liquid_gas_area = 2.5,
      .wetted_wall_area = 0.75,
      .contact_measure = 0.5,
      .surface_energy = 3.0,
      .young_wall_energy = -0.25,
      .volume_constraint_potential = total_potential - 2.75,
      .total_potential = total_potential,
      .kinetic_energy = 1.0,
      .gravitational_energy = 2.0,
      .gravitational_potential_power = -0.5,
      .surface_wall_potential_power = 0.25,
      .volume_constraint_potential_power = -0.125,
      .bulk_viscous_dissipation_rate = 0.75,
      .external_pressure_power = -0.4,
      .modeled_stored_energy = 5.75,
  };
}

bool rowsExactlyEqual(
    const LevelSetMaintenanceWorkRow& left,
    const LevelSetMaintenanceWorkRow& right)
{
  return left.transaction_id == right.transaction_id &&
         left.status == right.status &&
         left.substage == right.substage &&
         left.step == right.step &&
         left.attempt == right.attempt &&
         left.time == right.time &&
         left.dt == right.dt &&
         left.algebraic_state_revision_before ==
             right.algebraic_state_revision_before &&
         left.algebraic_state_revision_after ==
             right.algebraic_state_revision_after &&
         left.snapshot_set_revision_before ==
             right.snapshot_set_revision_before &&
         left.snapshot_set_revision_after ==
             right.snapshot_set_revision_after &&
         left.mesh_topology_set_revision_before ==
             right.mesh_topology_set_revision_before &&
         left.mesh_topology_set_revision_after ==
             right.mesh_topology_set_revision_after &&
         left.cut_topology_set_revision_before ==
             right.cut_topology_set_revision_before &&
         left.cut_topology_set_revision_after ==
             right.cut_topology_set_revision_after &&
         left.extension_map_revision_before ==
             right.extension_map_revision_before &&
         left.extension_map_revision_after ==
             right.extension_map_revision_after &&
         left.declared_stage == right.declared_stage &&
         left.before == right.before &&
         left.after == right.after &&
         left.numerical_work == right.numerical_work &&
         left.accepted_numerical_work ==
             right.accepted_numerical_work &&
         left.modeled_energy_numerical_work ==
             right.modeled_energy_numerical_work &&
         left.accepted_modeled_energy_numerical_work ==
             right.accepted_modeled_energy_numerical_work;
}

void seedAcceptedLedgerRow(LevelSetMaintenanceWorkLedger& ledger)
{
  ledger.beginTransaction(LevelSetMaintenanceWorkTransaction{
      .transaction_id = 700u,
      .step = 9u,
      .attempt = 1u,
      .time = 0.45,
      .dt = 0.05,
      .declared_stage =
          LevelSetMaintenanceDeclaredStage::AcceptedEndpointPostStep,
      .extension_map_revision = 390u,
  });
  ledger.stageRow(
      LevelSetMaintenanceWorkSubstage::Reinitialization,
      1001u,
      1002u,
      {makeFunctional(2.0, 101u, 201u, 301u)},
      {makeFunctional(2.125, 102u, 202u, 302u)},
      391u);
  ledger.commitTransaction();
}

void beginCandidateTransaction(
    LevelSetMaintenanceWorkLedger& ledger,
    TransactionDrift drift,
    bool fault_rank)
{
  if (fault_rank &&
      drift == TransactionDrift::MissingActiveTransaction) {
    return;
  }
  ledger.beginTransaction(LevelSetMaintenanceWorkTransaction{
      .transaction_id = 701u,
      .step = 10u,
      .attempt = 2u,
      .time = 0.5,
      .dt = 0.05,
      .declared_stage =
          LevelSetMaintenanceDeclaredStage::AcceptedEndpointPostStep,
      .extension_map_revision = 392u,
  });

  auto before = makeFunctional(3.0, 111u, 211u, 311u);
  auto after = makeFunctional(3.25, 112u, 212u, 312u);
  std::uint64_t algebraic_revision_before = 2001u;
  std::uint64_t algebraic_revision_after = 2002u;
  std::optional<std::uint64_t> extension_map_revision_after{393u};
  if (fault_rank) {
    switch (drift) {
      case TransactionDrift::None:
        break;
      case TransactionDrift::FunctionalContent:
        after.liquid_volume += 0.125;
        break;
      case TransactionDrift::ModeledEnergyContent:
        *after.kinetic_energy += 0.25;
        *after.modeled_stored_energy += 0.25;
        break;
      case TransactionDrift::AlgebraicRevision:
        ++algebraic_revision_after;
        break;
      case TransactionDrift::SnapshotRevision:
        ++after.snapshot_revision;
        break;
      case TransactionDrift::MeshTopologyRevision:
        ++after.mesh_topology_revision;
        break;
      case TransactionDrift::CutTopologyRevision:
        ++after.cut_topology_revision;
        break;
      case TransactionDrift::ExtensionMapRevision:
        ++*extension_map_revision_after;
        break;
      case TransactionDrift::MissingActiveTransaction:
      case TransactionDrift::RowCount:
      case TransactionDrift::CurrentContentRevision:
      case TransactionDrift::HistoryContentRevision:
      case TransactionDrift::GeometryTransactionPresence:
      case TransactionDrift::GeometryRevisionContent:
        break;
    }
  }
  ledger.stageRow(
      LevelSetMaintenanceWorkSubstage::GeometryReconciliation,
      algebraic_revision_before,
      algebraic_revision_after,
      {before},
      {after},
      extension_map_revision_after);
  if (fault_rank && drift == TransactionDrift::RowCount) {
    auto second_after =
        makeFunctional(3.5, 113u, 213u, 313u);
    ledger.stageRow(
        LevelSetMaintenanceWorkSubstage::GlobalCorrection,
        algebraic_revision_after,
        2004u,
        {after},
        {second_after},
        394u);
  }
}

void expectSameDecisionOnEveryRank(
    LevelSetMaintenanceTransactionDecision decision)
{
  const int local = static_cast<int>(decision);
  int minimum = 0;
  int maximum = 0;
  MPI_Allreduce(
      &local, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(
      &local, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  EXPECT_EQ(minimum, maximum);
}

void runCommitCase()
{
  LevelSetMaintenanceWorkLedger ledger;
  seedAcceptedLedgerRow(ledger);
  const auto baseline_row = ledger.acceptedRows().front();
  beginCandidateTransaction(
      ledger, TransactionDrift::None, false);

  const std::array<std::uint64_t, 8> commit_state_words{
      1u, 1u, 4001u, 2u, 4002u, 4003u, 8101u, 8201u};
  const auto decision =
      application::core::
          collectiveLevelSetMaintenanceTransactionDecision(
              ledger,
              true,
              svmp::MeshComm::world(),
              commit_state_words);
  expectSameDecisionOnEveryRank(decision);
  ASSERT_EQ(
      decision,
      LevelSetMaintenanceTransactionDecision::Commit);
  ledger.commitTransaction();

  ASSERT_EQ(ledger.acceptedRows().size(), 2u);
  EXPECT_TRUE(
      rowsExactlyEqual(ledger.acceptedRows().front(), baseline_row));
  EXPECT_TRUE(ledger.rejectedRows().empty());
}

void runRejectCase(
    TransactionDrift drift,
    bool reject_invariant,
    int rank,
    int size)
{
  LevelSetMaintenanceWorkLedger ledger;
  seedAcceptedLedgerRow(ledger);
  const auto baseline_row = ledger.acceptedRows().front();
  const auto baseline_accepted_attempts =
      ledger.acceptedAttempts().size();
  const bool fault_rank = rank + 1 == size;
  beginCandidateTransaction(ledger, drift, fault_rank);
  auto commit_state_words =
      std::array<std::uint64_t, 8>{
          1u, 1u, 4001u, 2u, 4002u, 4003u, 8101u, 8201u};
  if (fault_rank) {
    if (drift == TransactionDrift::CurrentContentRevision) {
      ++commit_state_words[2];
    } else if (
        drift == TransactionDrift::HistoryContentRevision) {
      ++commit_state_words[5];
    } else if (
        drift == TransactionDrift::GeometryTransactionPresence) {
      commit_state_words[1] = 0u;
    } else if (
        drift == TransactionDrift::GeometryRevisionContent) {
      // Model a final live-geometry content/revision disagreement after
      // otherwise identical state/history and transaction participation.
      ++commit_state_words[7];
    }
  }
  const bool had_active_transaction =
      ledger.transactionActive();
  const auto local_trial_row_count = ledger.trialRows().size();
  const bool local_invariants_satisfied =
      !(reject_invariant && fault_rank);
  const auto decision =
      application::core::
          collectiveLevelSetMaintenanceTransactionDecision(
              ledger,
              local_invariants_satisfied,
              svmp::MeshComm::world(),
              commit_state_words);
  expectSameDecisionOnEveryRank(decision);
  ASSERT_EQ(
      decision,
      LevelSetMaintenanceTransactionDecision::Reject);

  if (had_active_transaction) {
    ledger.rejectTransaction();
  }

  ASSERT_EQ(ledger.acceptedRows().size(), 1u);
  EXPECT_TRUE(
      rowsExactlyEqual(ledger.acceptedRows().front(), baseline_row));
  EXPECT_EQ(
      ledger.acceptedAttempts().size(),
      baseline_accepted_attempts);
  ASSERT_EQ(
      ledger.rejectedRows().size(),
      had_active_transaction ? local_trial_row_count : 0u);
  for (const auto& row : ledger.rejectedRows()) {
    EXPECT_DOUBLE_EQ(row.accepted_numerical_work, 0.0);
    ASSERT_TRUE(
        row.accepted_modeled_energy_numerical_work.has_value());
    EXPECT_DOUBLE_EQ(
        *row.accepted_modeled_energy_numerical_work, 0.0);
  }
  ASSERT_EQ(
      ledger.rejectedAttempts().size(),
      had_active_transaction ? 1u : 0u);
  if (had_active_transaction) {
    EXPECT_EQ(
        ledger.rejectedAttempts().front().row_count,
        local_trial_row_count);
    EXPECT_DOUBLE_EQ(
        ledger.rejectedAttempts().front().accepted_numerical_work,
        0.0);
    ASSERT_TRUE(
        ledger.rejectedAttempts()
            .front()
            .accepted_modeled_energy_numerical_work.has_value());
    EXPECT_DOUBLE_EQ(
        *ledger.rejectedAttempts()
             .front()
             .accepted_modeled_energy_numerical_work,
        0.0);
  }
}

std::string_view driftName(TransactionDrift drift)
{
  switch (drift) {
    case TransactionDrift::None:
      return "none";
    case TransactionDrift::FunctionalContent:
      return "functional_content";
    case TransactionDrift::ModeledEnergyContent:
      return "modeled_energy_content";
    case TransactionDrift::AlgebraicRevision:
      return "algebraic_revision";
    case TransactionDrift::SnapshotRevision:
      return "snapshot_revision";
    case TransactionDrift::MeshTopologyRevision:
      return "mesh_topology_revision";
    case TransactionDrift::CutTopologyRevision:
      return "cut_topology_revision";
    case TransactionDrift::ExtensionMapRevision:
      return "extension_map_revision";
    case TransactionDrift::MissingActiveTransaction:
      return "missing_active_transaction";
    case TransactionDrift::RowCount:
      return "row_count";
    case TransactionDrift::CurrentContentRevision:
      return "current_content_revision";
    case TransactionDrift::HistoryContentRevision:
      return "history_content_revision";
    case TransactionDrift::GeometryTransactionPresence:
      return "geometry_transaction_presence";
    case TransactionDrift::GeometryRevisionContent:
      return "geometry_revision_content";
  }
  return "unknown";
}

void runCommitRejectAndContentRevisionAgreement(
    int expected_size)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != expected_size) {
    GTEST_SKIP()
        << "This exact consensus fixture requires "
        << expected_size << " ranks.";
  }

  runCommitCase();
  {
    SCOPED_TRACE("last_rank_invariant");
    runRejectCase(
        TransactionDrift::None,
        true,
        rank,
        size);
  }
  constexpr std::array<TransactionDrift, 13> drifts{
      TransactionDrift::FunctionalContent,
      TransactionDrift::ModeledEnergyContent,
      TransactionDrift::AlgebraicRevision,
      TransactionDrift::SnapshotRevision,
      TransactionDrift::MeshTopologyRevision,
      TransactionDrift::CutTopologyRevision,
      TransactionDrift::ExtensionMapRevision,
      TransactionDrift::MissingActiveTransaction,
      TransactionDrift::RowCount,
      TransactionDrift::CurrentContentRevision,
      TransactionDrift::HistoryContentRevision,
      TransactionDrift::GeometryTransactionPresence,
      TransactionDrift::GeometryRevisionContent,
  };
  for (const auto drift : drifts) {
    SCOPED_TRACE(driftName(drift));
    runRejectCase(drift, false, rank, size);
  }
}

TEST(LevelSetMaintenanceTransactionConsensusMPI,
     TwoRankCommitRejectAndContentRevisionAgreement)
{
  runCommitRejectAndContentRevisionAgreement(2);
}

TEST(LevelSetMaintenanceTransactionConsensusMPI,
     FourRankCommitRejectAndContentRevisionAgreement)
{
  runCommitRejectAndContentRevisionAgreement(4);
}

} // namespace
