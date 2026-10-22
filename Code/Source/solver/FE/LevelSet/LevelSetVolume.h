#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Cut-cell volume measurement utilities.
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

} // namespace svmp::FE::level_set
