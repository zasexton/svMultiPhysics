/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

/**
 * @file
 * @brief The two-fluid accepted-stage diagnostic records of an FE system:
 *        declarations, the pending group of a candidate (stage diagnostics,
 *        momentum reconciliations, solve numerics) and the accepted history.
 *
 * The records are an attachment of the FE system (FESystem::attachment), so
 * they live as long as the system.  records(system) is the interface that
 * FESystem had for them; the record data types are in
 * FE/Systems/TwoFluidRecordTypes.h for now.
 */

#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Core/Types.h"
#include "FE/Interfaces/IncompressibleTwoFluidDiagnostics.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TwoFluidRecordTypes.h"

#include <cstdint>
#include <span>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::two_fluid {

/// The records of one FE system (its attachment).
struct TwoFluidStageRecordStore {
  std::vector<svmp::FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration>
      declarations{};
  std::vector<svmp::FE::systems::TwoFluidAcceptedStageDiagnosticHistoryRecord>
      pending{};
  std::vector<svmp::FE::systems::TwoFluidAcceptedStageDiagnosticHistoryRecord>
      history{};
  bool declarations_frozen{false};
  bool transaction_active{false};
};

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
[[nodiscard]] TwoFluidStageRecords records(svmp::FE::systems::FESystem& system);

/// The two-fluid records of @p system, read-only.
[[nodiscard]] ConstTwoFluidStageRecords records(
    const svmp::FE::systems::FESystem& system) noexcept;

} // namespace svmp::Physics::formulations::navier_stokes::two_fluid
