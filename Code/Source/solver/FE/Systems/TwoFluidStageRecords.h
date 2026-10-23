/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SYSTEMS_TWO_FLUID_STAGE_RECORDS_H
#define SVMP_FE_SYSTEMS_TWO_FLUID_STAGE_RECORDS_H

/**
 * @file
 * @brief The two-fluid accepted-stage diagnostic records of an FE system:
 *        declarations, the pending group of a candidate (stage diagnostics,
 *        momentum reconciliations, solve numerics) and the accepted history.
 *
 * The records are an attachment of the FE system (FESystem::attachment): an
 * accepted-stage record store (AcceptedStageRecords.h) of the two-fluid
 * declarations and records, so they live as long as the system.
 * twoFluidStageRecords(system) is the interface that FESystem had for them;
 * the record data types are in FE/Systems/TwoFluidRecordTypes.h for now.
 * The two-fluid formulation declares them; the two-fluid stage participant
 * stages, discards and publishes them.
 */

#include "Constraints/SmallCutAggregationConstraint.h"
#include "Core/Types.h"
#include "Interfaces/IncompressibleTwoFluidDiagnostics.h"
#include "Systems/AcceptedStageRecords.h"
#include "Systems/FESystem.h"
#include "Systems/TwoFluidRecordTypes.h"

#include <cstdint>
#include <span>
#include <vector>

namespace svmp::FE::systems {

/// The records of one FE system (its attachment): declarations, the pending
/// group of the current attempt and the accepted history.
using TwoFluidStageRecordStore =
    AcceptedStageRecordStore<TwoFluidAcceptedStageDiagnosticDeclaration,
                             TwoFluidAcceptedStageDiagnosticHistoryRecord>;

/**
 * @brief The two-fluid accepted-stage diagnostic records of an FE system.
 *
 * A lightweight handle on the system and its record store.  Declarations
 * are made before setup; a candidate stages one pending group, which is
 * discarded with the candidate or published after acceptance (collective
 * where the FE system is).
 */
class TwoFluidStageRecords {
public:
  explicit TwoFluidStageRecords(svmp::FE::systems::FESystem& system);

  /** Register one deterministic two-fluid accepted-stage diagnostic owner. */
  void declareTwoFluidAcceptedStageDiagnostics(
      svmp::FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration declaration);
  [[nodiscard]] std::span<
      const svmp::FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration>
  twoFluidAcceptedStageDiagnosticDeclarations() const noexcept {
    return two_fluid_accepted_stage_declarations_;
  }
  /** Stage communicator-global diagnostics for a prospective accepted step. */
  void stageTwoFluidAcceptedStageDiagnostics(
      svmp::FE::systems::OperatorStageMeasurementMetadata metadata,
      std::span<const svmp::FE::systems::AcceptedTwoFluidStageDiagnosticState> states);
  /** Bind required post-maintenance momentum records to the pending group. */
  void bindPendingTwoFluidMomentumReconciliations(
      std::span<const svmp::FE::interfaces::
                    IncompressibleTwoFluidMomentumReconciliation>
          reconciliations);
  /** Bind solver and canonical phase-stability evidence to the pending group. */
  void bindPendingTwoFluidAcceptedStageNumerics(
      svmp::FE::systems::TwoFluidAcceptedStageSolveTelemetry solve,
      std::span<const svmp::FE::constraints::SmallCutAggregationRefreshReport>
          aggregation_reports);
  /** Discard a rejected candidate without publishing accepted history. */
  void discardPendingTwoFluidAcceptedStageDiagnostics() noexcept;
  /** Publish the complete pending group after irreversible step acceptance. */
  void commitPendingTwoFluidAcceptedStageDiagnostics(
      std::uint64_t accepted_step,
      svmp::FE::Real accepted_time,
      svmp::FE::Real dt);
  [[nodiscard]] std::span<
      const svmp::FE::systems::TwoFluidAcceptedStageDiagnosticHistoryRecord>
  pendingTwoFluidAcceptedStageDiagnostics() const noexcept {
    return pending_two_fluid_accepted_stage_diagnostics_;
  }
  [[nodiscard]] std::span<
      const svmp::FE::systems::TwoFluidAcceptedStageDiagnosticHistoryRecord>
  twoFluidAcceptedStageDiagnosticHistory() const noexcept {
    return two_fluid_accepted_stage_diagnostic_history_;
  }

private:
  // The FE system state that the record code reads, under the names it had
  // as FESystem members.
  struct FieldRegistryAccess {
    const svmp::FE::systems::FESystem* system;
    [[nodiscard]] const svmp::FE::systems::FieldRecord& get(svmp::FE::FieldId field) const
    {
      return system->fieldRecord(field);
    }
    [[nodiscard]] bool has(svmp::FE::FieldId field) const noexcept
    {
      return system->hasField(field);
    }
  };

  [[nodiscard]] bool isSetup() const noexcept { return system_->isSetup(); }
  [[nodiscard]] const svmp::FE::dofs::DofHandler& dofHandler() const noexcept
  {
    return system_->dofHandler();
  }
  [[nodiscard]] const svmp::FE::assembly::IMeshAccess& meshAccess() const
  {
    return system_->meshAccess();
  }

  svmp::FE::systems::FESystem* system_;
  std::vector<svmp::FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration>&
      two_fluid_accepted_stage_declarations_;
  std::vector<svmp::FE::systems::TwoFluidAcceptedStageDiagnosticHistoryRecord>&
      pending_two_fluid_accepted_stage_diagnostics_;
  std::vector<svmp::FE::systems::TwoFluidAcceptedStageDiagnosticHistoryRecord>&
      two_fluid_accepted_stage_diagnostic_history_;
  bool& two_fluid_accepted_stage_declarations_frozen_;
  bool& two_fluid_accepted_stage_transaction_active_;
  FieldRegistryAccess field_registry_;
};

/// Read-only two-fluid records of an FE system; empty without records.
class ConstTwoFluidStageRecords {
public:
  explicit ConstTwoFluidStageRecords(
      const svmp::FE::systems::FESystem& system) noexcept;

  [[nodiscard]] std::span<
      const svmp::FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration>
  twoFluidAcceptedStageDiagnosticDeclarations() const noexcept;
  [[nodiscard]] std::span<
      const svmp::FE::systems::TwoFluidAcceptedStageDiagnosticHistoryRecord>
  pendingTwoFluidAcceptedStageDiagnostics() const noexcept;
  [[nodiscard]] std::span<
      const svmp::FE::systems::TwoFluidAcceptedStageDiagnosticHistoryRecord>
  twoFluidAcceptedStageDiagnosticHistory() const noexcept;

private:
  const TwoFluidStageRecordStore* store_{nullptr};
};

/// The two-fluid records of @p system (created on first use).
[[nodiscard]] TwoFluidStageRecords twoFluidStageRecords(svmp::FE::systems::FESystem& system);

/// The two-fluid records of @p system, read-only.
[[nodiscard]] ConstTwoFluidStageRecords twoFluidStageRecords(
    const svmp::FE::systems::FESystem& system) noexcept;

} // namespace svmp::FE::systems

#endif // SVMP_FE_SYSTEMS_TWO_FLUID_STAGE_RECORDS_H
