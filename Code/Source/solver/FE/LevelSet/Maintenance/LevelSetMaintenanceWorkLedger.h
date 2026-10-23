/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetMaintenanceWorkLedger.h
 * @ingroup fe_level_set
 * @brief The work of level-set maintenance on the declared functionals: the
 *        transactional ledger of maintenance substages, the accepted-step
 *        maintenance account and their logs.
 *
 * The maintenance pipeline (LevelSetMaintenancePipeline.h) reports every
 * substage that changed the level set to its stage observer;
 * stageLevelSetMaintenanceWorkRow records it in the ledger.  Physical
 * Transport is kept apart from the numerical maintenance substages
 * (limiting, reinitialization, geometry reconciliation, global correction).
 *
 * The types, the ledger, the accepted-step aggregation and the log formats
 * were moved from Physics/Formulations/NavierStokes/FreeSurface/
 * FreeSurfaceEnergyAccount with bodies, messages and log text unchanged.
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetMaintenanceSubstage.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace level_set {

enum class LevelSetMaintenanceWorkStatus : std::uint8_t {
  Trial,
  Accepted,
  Rejected
};

using LevelSetMaintenanceWorkSubstage = MaintenanceSubstage;

enum class LevelSetMaintenanceDeclaredStage : std::uint8_t {
  ProspectiveAcceptedEndpoint,
  AcceptedEndpointPostStep
};

/**
 * @brief The functionals of one maintained interface at a maintenance
 *        endpoint, evaluated on the authoritative snapshot of that state.
 *
 * These are the snapshot functionals of FE/Interfaces
 * (FreeSurfaceGeometrySnapshot) and of the accepted functional records of
 * FE/Systems (FreeSurfaceRecordTypes.h): the caller evaluates them and the
 * ledger checks their identities, records their change and sums them.
 */
struct LevelSetAuthoritativeFunctionalValue {
  int interface_marker{-1};
  std::uint64_t snapshot_revision{0};
  // Mesh topology epochs may legitimately start at zero. The cut-topology
  // fingerprint is separately required and is never zero, so an omitted
  // interface-topology provenance value cannot masquerade as an initial mesh.
  std::uint64_t mesh_topology_revision{0};
  std::uint64_t cut_topology_revision{0};
  double liquid_volume{0.0};
  double liquid_gas_area{0.0};
  double wetted_wall_area{0.0};
  double contact_measure{0.0};
  double surface_energy{0.0};
  double young_wall_energy{0.0};
  double volume_constraint_potential{0.0};
  double total_potential{0.0};
  std::optional<double> kinetic_energy{};
  std::optional<double> gravitational_energy{};
  std::optional<double> gravitational_potential_power{};
  std::optional<double> surface_wall_potential_power{};
  std::optional<double> volume_constraint_potential_power{};
  std::optional<double> bulk_viscous_dissipation_rate{};
  std::optional<double> external_pressure_power{};
  /**
   * One-phase modeled stored energy for this record:
   * kinetic + gravitational + liquid-gas surface + Young wall energy.
   *
   * This is absent when the active-volume owner is unavailable. It excludes
   * the volume-constraint potential, which is diagnostic rather than modeled
   * stored energy.
   */
  std::optional<double> modeled_stored_energy{};

  bool operator==(
      const LevelSetAuthoritativeFunctionalValue&) const = default;
};

struct LevelSetMaintenanceWorkTransaction {
  std::uint64_t transaction_id{0};
  std::uint64_t step{0};
  std::uint64_t attempt{0};
  double time{0.0};
  double dt{0.0};
  LevelSetMaintenanceDeclaredStage declared_stage{
      LevelSetMaintenanceDeclaredStage::AcceptedEndpointPostStep};
  std::optional<std::uint64_t> extension_map_revision{};
};

struct LevelSetMaintenanceWorkRow {
  std::uint64_t transaction_id{0};
  LevelSetMaintenanceWorkStatus status{
      LevelSetMaintenanceWorkStatus::Trial};
  LevelSetMaintenanceWorkSubstage substage{
      LevelSetMaintenanceWorkSubstage::Reinitialization};
  std::uint64_t step{0};
  std::uint64_t attempt{0};
  double time{0.0};
  double dt{0.0};
  // Staged vectors have no published backend revision. These two values are
  // deterministic fingerprints of their complete FE-ordered coefficients.
  std::uint64_t algebraic_state_revision_before{0};
  std::uint64_t algebraic_state_revision_after{0};
  std::uint64_t snapshot_set_revision_before{0};
  std::uint64_t snapshot_set_revision_after{0};
  std::uint64_t mesh_topology_set_revision_before{0};
  std::uint64_t mesh_topology_set_revision_after{0};
  std::uint64_t cut_topology_set_revision_before{0};
  std::uint64_t cut_topology_set_revision_after{0};
  std::optional<std::uint64_t> extension_map_revision_before{};
  std::optional<std::uint64_t> extension_map_revision_after{};
  LevelSetMaintenanceDeclaredStage declared_stage{
      LevelSetMaintenanceDeclaredStage::AcceptedEndpointPostStep};
  std::vector<LevelSetAuthoritativeFunctionalValue> before{};
  std::vector<LevelSetAuthoritativeFunctionalValue> after{};
  double numerical_work{0.0};
  double accepted_numerical_work{0.0};
  std::optional<double> modeled_energy_numerical_work{};
  std::optional<double> accepted_modeled_energy_numerical_work{};
};

struct LevelSetMaintenanceModeledEnergySubstage {
  std::size_t row_count{0};
  /**
   * Sum of the modeled stored-energy changes for this substage. This remains
   * absent when the substage has no rows or any row lacks the active-volume
   * modeled-energy owner.
   */
  std::optional<double> modeled_energy_change{};
  /**
   * Accepted-state contribution. Rejected attempts publish exact zero when
   * the diagnostic substage sum is available.
   */
  std::optional<double> accepted_modeled_energy_change{};

  bool operator==(
      const LevelSetMaintenanceModeledEnergySubstage&) const = default;
};

struct LevelSetMaintenanceModeledEnergyBreakdown {
  /** Physical conservative transport; not maintenance numerical work. */
  LevelSetMaintenanceModeledEnergySubstage transport{};
  LevelSetMaintenanceModeledEnergySubstage limiting{};
  /** This maps to the complete ledger's redistancing channel. */
  LevelSetMaintenanceModeledEnergySubstage reinitialization{};
  /** This maps to the complete ledger's local-reconciliation channel. */
  LevelSetMaintenanceModeledEnergySubstage geometry_reconciliation{};
  LevelSetMaintenanceModeledEnergySubstage global_correction{};
  /**
   * Sum over every non-transport substage above. Transport is deliberately
   * excluded so a physical domain-transport change cannot be hidden in the
   * numerical-maintenance account.
   */
  LevelSetMaintenanceModeledEnergySubstage numerical_maintenance_total{};

  bool operator==(
      const LevelSetMaintenanceModeledEnergyBreakdown&) const = default;
};

struct LevelSetMaintenanceWorkAttempt {
  std::uint64_t transaction_id{0};
  LevelSetMaintenanceWorkStatus status{
      LevelSetMaintenanceWorkStatus::Trial};
  std::uint64_t step{0};
  std::uint64_t attempt{0};
  double time{0.0};
  double dt{0.0};
  LevelSetMaintenanceDeclaredStage declared_stage{
      LevelSetMaintenanceDeclaredStage::AcceptedEndpointPostStep};
  std::optional<std::uint64_t> extension_map_revision{};
  std::size_t row_count{0};
  double numerical_work{0.0};
  double accepted_numerical_work{0.0};
  /**
   * Compatibility aggregate over every row, including a physical Transport
   * row. Use modeled_energy_breakdown.numerical_maintenance_total for the
   * numerical-maintenance account.
   */
  std::optional<double> modeled_energy_numerical_work{};
  std::optional<double> accepted_modeled_energy_numerical_work{};
  LevelSetMaintenanceModeledEnergyBreakdown modeled_energy_breakdown{};
};

struct LevelSetMaintenanceAcceptedStepEnergyAccount {
  struct Endpoint {
    std::uint64_t algebraic_state_revision{0};
    std::uint64_t snapshot_set_revision{0};
    std::uint64_t mesh_topology_set_revision{0};
    std::uint64_t cut_topology_set_revision{0};
    std::optional<std::uint64_t> extension_map_revision{};
    std::optional<double> kinetic_energy{};
    std::optional<double> gravitational_energy{};
    std::optional<double> gravitational_potential_power{};
    double liquid_gas_surface_energy{0.0};
    double solid_liquid_wall_energy{0.0};
    std::optional<double> surface_wall_potential_power{};
    std::optional<double> volume_constraint_potential_power{};
    std::optional<double> bulk_viscous_dissipation_rate{};
    std::optional<double> external_pressure_power{};
    std::optional<double> modeled_stored_energy{};
  };

  std::uint64_t step{0};
  std::uint64_t attempt{0};
  double time{0.0};
  double dt{0.0};
  std::size_t transaction_count{0};
  std::size_t row_count{0};
  LevelSetMaintenanceModeledEnergyBreakdown
      modeled_energy_breakdown{};
  /** Absent only when every accepted transaction has zero rows. */
  std::optional<Endpoint> maintenance_start{};
  /** Equals maintenance_start when no Transport row is present. */
  std::optional<Endpoint> post_transport{};
  /** Absent only when every accepted transaction has zero rows. */
  std::optional<Endpoint> maintenance_end{};
  /**
   * Endpoint change minus the corresponding accepted row sum. These remain
   * absent when the row group has no modeled-energy coverage.
   */
  std::optional<double> physical_transport_endpoint_residual{};
  std::optional<double>
      numerical_maintenance_endpoint_residual{};
};

/**
 * Combine every accepted maintenance transaction for one accepted-step
 * attempt. The result keeps physical Transport separate from numerical
 * maintenance and preserves unavailable modeled-energy channels.
 *
 * Empty input means that no maintenance transaction was published and
 * returns no account. Mixed step metadata, rejected attempts, malformed
 * substage coverage, inconsistent attempt totals, incomplete row coverage,
 * or a discontinuous row chain fail closed.
 */
[[nodiscard]] std::optional<
    LevelSetMaintenanceAcceptedStepEnergyAccount>
aggregateLevelSetMaintenanceAcceptedStepEnergy(
    std::span<const LevelSetMaintenanceWorkAttempt> attempts,
    std::span<const LevelSetMaintenanceWorkRow> rows);

/**
 * @brief The sizes of a ledger's published histories, taken before a
 *        transaction so that its outcome can be logged.
 */
struct LevelSetMaintenanceWorkMark {
  std::size_t accepted_rows{0};
  std::size_t rejected_rows{0};
  std::size_t accepted_attempts{0};
  std::size_t rejected_attempts{0};
};

/**
 * @brief Work of level-set maintenance on the declared functionals, by
 *        substage and transaction.
 *
 * A transaction is one maintenance attempt (the prospective endpoint of a
 * step candidate or the post-step maintenance of an accepted step).  Each
 * staged row records one substage: the algebraic, snapshot, topology and
 * extension-map revisions before and after it, the functional values before
 * and after it, the numerical work (change of the total potential) and the
 * change of the modeled stored energy.  Rows must chain exactly.  Commit
 * publishes the rows and the attempt as accepted, rejection as rejected
 * with zero accepted work; both are allocation-free once staged.
 */
class LevelSetMaintenanceWorkLedger {
public:
  void beginTransaction(LevelSetMaintenanceWorkTransaction transaction);

  void stageRow(
      LevelSetMaintenanceWorkSubstage substage,
      std::uint64_t algebraic_state_revision_before,
      std::uint64_t algebraic_state_revision_after,
      std::vector<LevelSetAuthoritativeFunctionalValue> before,
      std::vector<LevelSetAuthoritativeFunctionalValue> after,
      std::optional<std::uint64_t> extension_map_revision_after =
          std::nullopt);

  void commitTransaction();
  void rejectTransaction();

  [[nodiscard]] bool transactionActive() const noexcept;
  [[nodiscard]] const LevelSetMaintenanceWorkTransaction*
  activeTransaction() const noexcept;
  [[nodiscard]] const std::vector<LevelSetMaintenanceWorkRow>&
  trialRows() const noexcept;
  [[nodiscard]] const std::vector<LevelSetMaintenanceWorkRow>&
  acceptedRows() const noexcept;
  [[nodiscard]] const std::vector<LevelSetMaintenanceWorkRow>&
  rejectedRows() const noexcept;
  [[nodiscard]] const std::vector<LevelSetMaintenanceWorkAttempt>&
  acceptedAttempts() const noexcept;
  [[nodiscard]] const std::vector<LevelSetMaintenanceWorkAttempt>&
  rejectedAttempts() const noexcept;
  /** The current sizes of the published histories. */
  [[nodiscard]] LevelSetMaintenanceWorkMark mark() const noexcept;

private:
  std::optional<LevelSetMaintenanceWorkTransaction> active_transaction_{};
  std::uint64_t last_transaction_id_{0};
  std::vector<LevelSetMaintenanceWorkRow> trial_rows_{};
  std::vector<LevelSetMaintenanceWorkRow> accepted_rows_{};
  std::vector<LevelSetMaintenanceWorkRow> rejected_rows_{};
  std::vector<LevelSetMaintenanceWorkAttempt> accepted_attempts_{};
  std::vector<LevelSetMaintenanceWorkAttempt> rejected_attempts_{};
};

/** FNV-1a mixing of one 64-bit word into a maintenance revision. */
void mixMaintenanceRevision(
    std::uint64_t& revision,
    std::uint64_t value) noexcept;

/** Log one row per maintenance work row (diagnostic=level_set_maintenance_work). */
void logLevelSetMaintenanceWorkRows(
    std::span<const LevelSetMaintenanceWorkRow> rows);

/** Log one row per maintenance work attempt. */
void logLevelSetMaintenanceWorkAttempts(
    std::span<const LevelSetMaintenanceWorkAttempt> attempts);

/** Log the accepted-step maintenance energy account. */
void logLevelSetMaintenanceAcceptedStepEnergyAccount(
    const LevelSetMaintenanceAcceptedStepEnergyAccount& account);


/**
 * @brief Log the attempts, then the rows, that @p ledger published with
 *        @p status (Accepted or Rejected) since @p mark.
 */
void logLevelSetMaintenanceWorkPublishedSince(
    const LevelSetMaintenanceWorkLedger& ledger,
    const LevelSetMaintenanceWorkMark& mark,
    LevelSetMaintenanceWorkStatus status);

/// What a ledger row records of a candidate state; supplied by the caller.
struct LevelSetMaintenanceWorkRowSource {
  /// The functional values of a candidate state (collective).
  std::function<std::vector<LevelSetAuthoritativeFunctionalValue>(
      std::span<const Real>)>
      functional_values{};
  /// The algebraic state revision of a candidate state (collective).
  std::function<std::uint64_t(std::span<const Real>)>
      algebraic_state_revision{};
  /// The extension-map revision after the substage (collective).
  std::function<std::optional<std::uint64_t>()> extension_map_revision{};
};

/**
 * @brief Stage the row of one maintenance substage, as reported to the
 *        pipeline's stage observer.
 *
 * Evaluates the functional values of @p after_candidate, then the algebraic
 * revisions of @p before_candidate and @p after_candidate and the
 * extension-map revision, and stages the row from @p current (the values of
 * the state before the substage) to the new values.  @p current is then
 * advanced to the new values.  Collective in the order above.
 */
void stageLevelSetMaintenanceWorkRow(
    LevelSetMaintenanceWorkLedger& ledger,
    const LevelSetMaintenanceWorkRowSource& source,
    std::vector<LevelSetAuthoritativeFunctionalValue>& current,
    MaintenanceSubstage substage,
    std::span<const Real> before_candidate,
    std::span<const Real> after_candidate);

} // namespace level_set
} // namespace FE
} // namespace svmp
