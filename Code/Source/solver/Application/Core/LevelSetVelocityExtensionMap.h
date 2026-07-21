#pragma once

#include "FE/Core/Types.h"
#include "FE/LevelSet/LevelSetVelocityExtensionConstraint.h"
#include "Mesh/Core/MeshComm.h"
#include "Mesh/Core/MeshTypes.h"
#include "Mesh/Mesh.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace application::core {

inline constexpr double kVelocityExtensionMaxRegressionCondition = 1.0e8;
inline constexpr double kVelocityExtensionCoefficientTolerance = 1.0e-12;
inline constexpr double kVelocityExtensionRowTolerance = 1.0e-10;
inline constexpr double kVelocityExtensionMaxWetToDryAmplification = 16.0;

struct WallVelocityExtensionConstraint {
  svmp::label_t boundary_label{svmp::INVALID_LABEL};
  std::array<bool, 3> constrained_components{false, false, false};
  bool project_boundary_normal{false};
};

struct WallCompatibleVelocityExtensionResult {
  std::size_t extended_vertices{0u};
  std::size_t vertices_outside_band{0u};
  std::size_t wall_projected_vertices{0u};
  std::size_t component_collision_vertices{0u};
  std::size_t regression_candidate_rows{0u};
  std::size_t regression_accepted_rows{0u};
  std::size_t bounded_fallback_rows{0u};
  std::size_t condition_rejected_rows{0u};
  std::size_t coefficient_rejected_rows{0u};
  double max_wall_normal_velocity{0.0};
  double max_regression_condition{0.0};
  double max_abs_graph_coefficient{0.0};
  double max_graph_row_l1{0.0};
  double max_graph_row_sum_error{0.0};
  double max_negative_graph_coefficient{0.0};
  double max_constant_reproduction_error{0.0};
  double max_linear_reproduction_error{0.0};
  double max_extrapolation_distance{0.0};
  double max_seed_speed{0.0};
  double max_extended_speed{0.0};
};

enum class VelocityExtensionRowDisposition : std::uint8_t {
  TraceSeed,
  Regression,
  BoundedFallback,
  OutsideBandZero,
};

[[nodiscard]] std::string_view velocityExtensionRowDispositionName(
    VelocityExtensionRowDisposition disposition) noexcept;

struct VelocityExtensionGraphDependency {
  svmp::FE::GlobalIndex local_vertex{svmp::FE::INVALID_GLOBAL_INDEX};
  svmp::gid_t global_vertex{svmp::INVALID_GID};
  double coefficient{0.0};
};

/**
 * Owner-local evidence for one scalar graph row. Vector constraint rows reuse
 * this graph row before applying the declared wall-component projection.
 */
struct VelocityExtensionGraphRowDiagnostic {
  svmp::FE::GlobalIndex local_vertex{svmp::FE::INVALID_GLOBAL_INDEX};
  svmp::gid_t global_vertex{svmp::INVALID_GID};
  VelocityExtensionRowDisposition disposition{
      VelocityExtensionRowDisposition::OutsideBandZero};
  std::int64_t component_assignment{-1};
  std::size_t component_candidates{0u};
  int band_layer{0};
  int reconstruction_dimension{0};
  int numerical_rank{0};
  bool assigned{false};
  bool regression_attempted{false};
  bool regression_accepted{false};
  bool bounded_fallback_used{false};
  bool condition_rejected{false};
  bool coefficient_rejected{false};
  bool wall_projected{false};
  double condition_estimate{0.0};
  double proposed_coefficient_sum{0.0};
  double proposed_coefficient_l1{0.0};
  double proposed_max_abs_coefficient{0.0};
  std::size_t proposed_negative_weight_count{0u};
  double proposed_max_negative_coefficient{0.0};
  double coefficient_sum{0.0};
  double coefficient_l1{0.0};
  double max_abs_coefficient{0.0};
  std::size_t negative_weight_count{0u};
  double max_negative_coefficient{0.0};
  double constant_reproduction_error{0.0};
  double max_tangential_linear_reproduction_error{0.0};
  double extrapolation_distance{0.0};
  double dependency_max_speed{0.0};
  double preview_speed{0.0};
  double preview_amplification{0.0};
  std::vector<VelocityExtensionGraphDependency> dependencies{};
};

struct SymmetricRankConditionEstimate {
  int numerical_rank{0};
  double condition_estimate{0.0};
};

struct VelocityExtensionMapRevision {
  std::uint64_t mesh_geometry{0u};
  std::uint64_t mesh_topology{0u};
  std::uint64_t mesh_ownership{0u};
  std::uint64_t mesh_numbering{0u};
  std::uint64_t free_surface_geometry{0u};
  std::uint64_t level_set_values{0u};
  std::uint64_t active_set{0u};

  [[nodiscard]] std::uint64_t key() const noexcept;
  [[nodiscard]] bool complete() const noexcept;
  bool operator==(const VelocityExtensionMapRevision&) const = default;
};

class VelocityExtensionMapSnapshot final {
public:
  VelocityExtensionMapSnapshot(
      VelocityExtensionMapRevision revision,
      std::size_t components,
      std::vector<double> preview,
      std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows,
      std::vector<std::int64_t> component_assignment,
      std::vector<VelocityExtensionGraphRowDiagnostic> row_diagnostics,
      WallCompatibleVelocityExtensionResult report,
      double wet_to_dry_amplification);

  [[nodiscard]] const VelocityExtensionMapRevision& revision() const noexcept
  {
    return revision_;
  }
  [[nodiscard]] std::size_t components() const noexcept { return components_; }
  [[nodiscard]] std::span<const double> preview() const noexcept
  {
    return preview_;
  }
  [[nodiscard]] std::span<const svmp::FE::level_set::VelocityExtensionConstraintRow>
  rows() const noexcept
  {
    return rows_;
  }
  [[nodiscard]] std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>
  copyRows() const
  {
    return rows_;
  }
  [[nodiscard]] std::span<const std::int64_t> componentAssignment() const noexcept
  {
    return component_assignment_;
  }
  [[nodiscard]] std::span<const VelocityExtensionGraphRowDiagnostic>
  rowDiagnostics() const noexcept
  {
    return row_diagnostics_;
  }
  [[nodiscard]] const WallCompatibleVelocityExtensionResult& report() const noexcept
  {
    return report_;
  }
  [[nodiscard]] double wetToDryAmplification() const noexcept
  {
    return wet_to_dry_amplification_;
  }

private:
  VelocityExtensionMapRevision revision_{};
  std::size_t components_{0u};
  std::vector<double> preview_{};
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows_{};
  std::vector<std::int64_t> component_assignment_{};
  std::vector<VelocityExtensionGraphRowDiagnostic> row_diagnostics_{};
  WallCompatibleVelocityExtensionResult report_{};
  double wet_to_dry_amplification_{0.0};
};

struct VelocityExtensionMapChangeReport {
  bool previous_available{false};
  std::uint64_t previous_revision_key{0u};
  bool revision_changed{false};
  bool mesh_geometry_changed{false};
  bool mesh_topology_changed{false};
  bool mesh_ownership_changed{false};
  bool mesh_numbering_changed{false};
  bool free_surface_geometry_changed{false};
  bool level_set_values_changed{false};
  bool active_set_changed{false};
  std::size_t common_owner_rows{0u};
  std::size_t added_owner_rows{0u};
  std::size_t removed_owner_rows{0u};
  std::size_t changed_owner_rows{0u};
  std::size_t component_assignment_changes{0u};
  std::size_t row_decision_changes{0u};
  std::size_t dependency_row_changes{0u};
  std::size_t preview_values_compared{0u};
  double maximum_coefficient_change{0.0};
  double preview_l2_change{0.0};
  double preview_linf_change{0.0};
};

struct VelocityExtensionMapArtifactContext {
  std::string level_set_field_name{};
  std::string source_velocity_field_name{};
  std::string target_velocity_field_name{};
  std::string geometry_domain_id{};
  std::string operator_tag{};
  std::string extension_method{};
  std::string retained_side{};
  std::uint64_t accepted_step{0u};
  double accepted_time{0.0};
  double time_step{0.0};
  std::uint64_t state_revision{0u};
  double isovalue{0.0};
  int extension_band_layers{0};
  bool enforce_wall_impermeability{false};
  int rank{0};
  int ranks{1};
};

struct VelocityExtensionMapArtifactResult {
  bool success{false};
  std::filesystem::path path{};
  std::uintmax_t bytes{0u};
  std::size_t owner_rows{0u};
  std::size_t constraint_rows{0u};
  std::string diagnostic{};
};

[[nodiscard]] VelocityExtensionMapChangeReport
compareVelocityExtensionMapSnapshots(
    const VelocityExtensionMapSnapshot& current,
    const VelocityExtensionMapSnapshot* previous = nullptr);

/** Atomically publish one accepted rank-local map shard without replacement. */
[[nodiscard]] VelocityExtensionMapArtifactResult
writeVelocityExtensionMapArtifact(
    const std::filesystem::path& output_directory,
    const VelocityExtensionMapArtifactContext& context,
    const VelocityExtensionMapSnapshot& snapshot,
    const VelocityExtensionMapSnapshot* previous = nullptr);

[[nodiscard]] VelocityExtensionMapRevision velocityExtensionMapRevision(
    std::uint64_t mesh_geometry,
    std::uint64_t mesh_topology,
    std::uint64_t mesh_ownership,
    std::uint64_t mesh_numbering,
    std::uint64_t free_surface_geometry,
    std::span<const double> level_set_values,
    std::span<const std::uint8_t> active_set);

// Map construction expects phi relative to the configured isovalue and
// oriented so the retained physical side is nonpositive. This convention
// makes signed graph-path offsets measure distance from either interface side.
[[nodiscard]] std::shared_ptr<const VelocityExtensionMapSnapshot>
buildVelocityExtensionMapSnapshot(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    VelocityExtensionMapRevision revision,
    std::span<const double> phi,
    std::span<const double> source_velocity,
    std::size_t source_components,
    std::span<const std::uint8_t> active,
    std::size_t target_components,
    std::size_t copy_components,
    int band_layers,
    bool enforce_wall_impermeability,
    std::span<const WallVelocityExtensionConstraint> wall_constraints);

[[nodiscard]] double estimateSymmetricConditionNumber(
    const std::array<std::array<double, 4>, 4>& matrix,
    int size);

[[nodiscard]] SymmetricRankConditionEstimate
estimateSymmetricRankAndCondition(
    const std::array<std::array<double, 4>, 4>& matrix,
    int size);

[[nodiscard]] std::size_t globalOwnedVelocityExtensionMaskCount(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const std::uint8_t> mask);

[[nodiscard]] std::size_t globalVelocityExtensionGeometrySampleCount(
    std::size_t local_count,
    const svmp::MeshComm& comm);

[[nodiscard]] std::size_t markVelocityExtensionTraceSupportCells(
    const svmp::Mesh& mesh,
    std::span<const svmp::FE::MeshIndex> cells,
    std::vector<std::uint8_t>& trace_support);

[[nodiscard]] std::vector<svmp::FE::MeshIndex>
nodalVelocityExtensionInterfaceCells(
    const svmp::Mesh& mesh,
    std::span<const double> phi,
    double isovalue);

[[nodiscard]] std::size_t synchronizeVelocityExtensionTraceSupportMask(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::vector<std::uint8_t>& trace_support);

[[nodiscard]] std::vector<std::vector<std::size_t>>
velocityExtensionEdgeAdjacency(const svmp::Mesh& mesh);

// The same wet-negative orientation contract applies to phi here.
[[nodiscard]] WallCompatibleVelocityExtensionResult
extendVelocityInLevelSetNormalBand(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const double> phi,
    std::span<const double> source_velocity,
    std::size_t source_components,
    std::span<const std::uint8_t> active,
    std::size_t target_components,
    std::size_t copy_components,
    int band_layers,
    bool enforce_wall_impermeability,
    std::span<const WallVelocityExtensionConstraint> wall_constraints,
    std::vector<double>& extended,
    std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>*
        constraint_rows = nullptr,
    std::vector<std::int64_t>* component_assignment = nullptr,
    std::vector<VelocityExtensionGraphRowDiagnostic>* row_diagnostics =
        nullptr);

[[nodiscard]] WallCompatibleVelocityExtensionResult
extendVelocityInLevelSetNormalBand(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const double> phi,
    std::span<const double> source_velocity,
    std::size_t source_components,
    std::span<const std::uint8_t> active,
    std::size_t target_components,
    std::size_t copy_components,
    int band_layers,
    bool enforce_wall_impermeability,
    std::span<const svmp::label_t> wall_boundary_labels,
    std::vector<double>& extended);

[[nodiscard]] WallCompatibleVelocityExtensionResult
extendVelocityInLevelSetNormalBand(
    const svmp::Mesh& mesh,
    std::span<const double> phi,
    std::span<const double> source_velocity,
    std::size_t source_components,
    std::span<const std::uint8_t> active,
    std::size_t target_components,
    std::size_t copy_components,
    int band_layers,
    bool enforce_wall_impermeability,
    std::span<const svmp::label_t> wall_boundary_labels,
    std::vector<double>& extended);

} // namespace application::core
