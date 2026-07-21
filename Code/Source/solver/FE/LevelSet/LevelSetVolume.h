#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Cut-cell volume measurement and global-shift correction utilities.
 */

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Dofs/DofHandler.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Systems/FESystem.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

struct LevelSetVolumeOptions {
    Real isovalue{0.0};
    Real tolerance{1.0e-12};
    bool use_generated_interface_quadrature{false};
    std::string level_set_field_name{};
    std::string generated_domain_id{"volume_correction"};
    int requested_interface_marker{-1};
    std::optional<int> quadrature_order{};
    std::optional<int> interface_quadrature_order{};
    std::optional<int> volume_quadrature_order{};
    GeneratedInterfaceGeometryMode geometry_mode{
        GeneratedInterfaceGeometryMode::LinearCorner};
    ImplicitCutQuadratureBackend implicit_cut_quadrature_backend{
        ImplicitCutQuadratureBackend::LinearCorner};
    ImplicitCutFallbackPolicy implicit_cut_fallback_policy{
        ImplicitCutFallbackPolicy::Fail};
    GeometryTangentPolicy geometry_tangent_policy{
        GeometryTangentPolicy::RefreshedFrozenQuadrature};
    Real implicit_cut_root_tolerance{1.0e-10};
    Real implicit_cut_root_coordinate_tolerance{1.0e-12};
    int implicit_cut_root_max_iterations{48};
    int implicit_cut_max_subdivision_depth{16};
    int affected_cell_neighborhood_layers{0};
    bool allow_corner_linearized_geometry{false};
    bool require_production_qualified_implicit_cut_backend{false};
};

struct LevelSetVolumeResult {
    // In an initialized MPI run, success and the cell/physical-measure members
    // through positive_volume are communicator-global values reduced from
    // locally owned cells; ghost cells never contribute physical measure.
    // Backend/cache telemetry remains rank-local work accounting.
    bool success{false};
    std::size_t cells{0};
    std::size_t cut_cells{0};
    std::size_t full_negative_cells{0};
    std::size_t full_positive_cells{0};
    Real total_volume{0.0};
    Real negative_volume{0.0};
    Real positive_volume{0.0};
    std::string diagnostic{};
    std::uint64_t generated_value_revision{0};
    std::size_t generated_cell_cache_hits{0};
    std::size_t generated_cell_cache_misses{0};
    std::size_t generated_cell_cache_unchanged_dof_hits{0};
    std::size_t generated_cell_refresh_candidate_count{0};
    std::size_t generated_directly_affected_cell_count{0};
    std::size_t generated_affected_cell_neighborhood_count{0};
    std::size_t generated_domain_cache_hits{0};
    std::size_t generated_linear_full_cell_fast_path_count{0};
    double generated_backend_elapsed_seconds{0.0};
};

struct LevelSetGlobalShiftCorrectionOptions {
    Real target_negative_volume{0.0};
    Real volume_tolerance{1.0e-10};
    int max_iterations{50};
    // Zero retains the low-level utility's historical always-correct behavior.
    // The application supplies a positive fallback threshold.
    Real minimum_relative_volume_error{0.0};
    // Every individual correction event is bounded.  The low-level default
    // permits at most one minimum mesh-edge length; applications should
    // normally choose a tighter fraction.  This is not a cumulative-in-time
    // displacement bound.  Zero is rejected rather than silently restoring an
    // unbounded global shift.
    Real maximum_interface_displacement_fraction{1.0};
};

struct LevelSetGlobalShiftCorrectionResult {
    // All decision-driving volumes, shifts, and displacement metrics are
    // communicator-global and therefore identical on every participating rank.
    bool success{false};
    bool correction_triggered{false};
    bool correction_applied{false};
    bool target_reached{false};
    // True when the target lies outside the final per-event bracket imposed
    // by either the physical-displacement or topology-stability guard.
    bool limited_by_displacement_bound{false};
    int iterations{0};
    Real applied_shift{0.0};
    Real target_negative_volume{0.0};
    Real initial_negative_volume{0.0};
    Real corrected_negative_volume{0.0};
    Real volume_error{0.0};
    Real trigger_volume_error{0.0};
    Real minimum_edge_length{0.0};
    Real maximum_allowed_interface_displacement{0.0};
    // Certified symmetric coefficient-shift limit that preserves every
    // current simplicial P1 vertex sign with a two-tolerance margin.
    Real maximum_topology_stable_shift{0.0};
    Real max_interface_displacement{0.0};
    // Conservative maximum over all physical-boundary level-set
    // intersections; this bounds the wall-contact-line subset.
    Real max_contact_line_displacement{0.0};
    // Retained compatibility/telemetry name for the wall-tangent contact-line
    // displacement estimate.  It equals max_contact_line_displacement.
    Real contact_line_displacement_bound{0.0};
    LevelSetVolumeResult initial_volume{};
    LevelSetVolumeResult corrected_volume{};
    std::string diagnostic{};
    std::size_t generated_volume_measurement_count{0};
    std::size_t generated_cell_cache_hits{0};
    std::size_t generated_cell_cache_misses{0};
    std::size_t generated_cell_cache_unchanged_dof_hits{0};
    std::size_t generated_cell_refresh_candidate_count{0};
    std::size_t generated_directly_affected_cell_count{0};
    std::size_t generated_affected_cell_neighborhood_count{0};
    std::size_t generated_domain_cache_hits{0};
    std::size_t generated_linear_full_cell_fast_path_count{0};
    double generated_backend_elapsed_seconds{0.0};
};

[[nodiscard]] LevelSetVolumeResult computeLevelSetCutCellVolume(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& options,
    std::span<const Real> coefficients);

[[nodiscard]] LevelSetVolumeResult computeLevelSetCutCellVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const Real> solution);

// The correction and its vertex-based displacement certification are
// qualified only for scalar, nodal, continuous P1 fields on linear triangle
// or tetrahedron meshes.  Other spaces/topologies fail closed before state is
// modified.  A triggered correction also requires a nondegenerate interface
// with a strict edge crossing and every vertex outside a two-tolerance safety
// margin around the isovalue.
[[nodiscard]] LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const Real> coefficients,
    std::vector<Real>& corrected_coefficients);

// System-level overload with the same qualified-space and interface guards.
[[nodiscard]] LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const Real> solution,
    std::vector<Real>& corrected_solution);

} // namespace svmp::FE::level_set
