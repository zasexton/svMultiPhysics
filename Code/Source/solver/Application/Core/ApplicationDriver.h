#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
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
};

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
