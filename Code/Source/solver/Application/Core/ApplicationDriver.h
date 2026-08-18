#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

class Parameters;

namespace application {
namespace core {

struct SimulationComponents;
struct VtkTimeSeriesCollection;

enum class LevelSetMaintenanceWorkStatus : std::uint8_t {
  Trial,
  Accepted,
  Rejected
};

enum class LevelSetMaintenanceWorkSubstage : std::uint8_t {
  Transport,
  Limiting,
  Reinitialization,
  GeometryReconciliation,
  GlobalCorrection
};

enum class LevelSetMaintenanceDeclaredStage : std::uint8_t {
  ProspectiveAcceptedEndpoint,
  AcceptedEndpointPostStep
};

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

struct LevelSetMaintenancePhysicalEndpointChannels {
  std::optional<double> surface_wall_energy_change{};
  std::optional<double> surface_transport_coupling_work{};
  std::optional<double> gravitational_energy_change{};
  std::optional<double>
      gravitational_transport_coupling_work{};
  std::optional<double> bulk_viscous_dissipation_rate{};
  std::optional<double> external_pressure_work{};
};

/**
 * Pair the post-Transport endpoint with the preceding accepted
 * post-maintenance stored state. Coupling work uses
 *   endpoint stored-energy change - dt * endpoint potential power.
 * Exterior-pressure power is converted to signed step work with dt.
 * Missing producers remain unavailable.
 */
[[nodiscard]] LevelSetMaintenancePhysicalEndpointChannels
evaluateLevelSetMaintenancePhysicalEndpointChannels(
    const LevelSetMaintenanceAcceptedStepEnergyAccount& account,
    std::optional<double>
        preceding_gravitational_energy,
    std::optional<double>
        preceding_surface_wall_energy);

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

private:
  std::optional<LevelSetMaintenanceWorkTransaction> active_transaction_{};
  std::uint64_t last_transaction_id_{0};
  std::vector<LevelSetMaintenanceWorkRow> trial_rows_{};
  std::vector<LevelSetMaintenanceWorkRow> accepted_rows_{};
  std::vector<LevelSetMaintenanceWorkRow> rejected_rows_{};
  std::vector<LevelSetMaintenanceWorkAttempt> accepted_attempts_{};
  std::vector<LevelSetMaintenanceWorkAttempt> rejected_attempts_{};
};

class ApplicationDriver {
public:
  static bool shouldUseNewSolver(const std::string& xml_file);
  static void run(const std::string& xml_file);

private:
  static void runWithParameters(const Parameters& params);
  static void runSteadyState(SimulationComponents& sim, const Parameters& params, VtkTimeSeriesCollection* pvd);
  static void runTransient(SimulationComponents& sim, const Parameters& params, VtkTimeSeriesCollection* pvd);
  static void outputResults(const SimulationComponents& sim, const Parameters& params, int step,
                            double time, VtkTimeSeriesCollection* pvd);
};

} // namespace core
} // namespace application
