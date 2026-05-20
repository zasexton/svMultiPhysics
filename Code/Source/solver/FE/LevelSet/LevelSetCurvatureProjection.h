#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Projected curvature recovery utilities for level-set interfaces.
 */

#include "Assembly/Assembler.h"
#include "Core/Types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

struct LevelSetCurvatureProjectionOptions {
    Real isovalue{0.0};
    Real gradient_tolerance{1.0e-10};
    Real normal_equation_tolerance{1.0e-12};
    Real max_normalized_fit_residual{0.0};
    int max_neighbor_rings{2};
    int smoothing_iterations{0};
    Real smoothing_relaxation{0.25};
};

struct LevelSetCurvatureProjectionSample {
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    std::array<Real, 3> coordinate{{0.0, 0.0, 0.0}};
    Real value{0.0};
};

struct LevelSetCurvatureProjectionResult {
    bool success{false};
    std::size_t vertices{0};
    std::size_t supplemental_samples{0};
    std::size_t supplemental_sample_rows{0};
    std::size_t vertices_with_supplemental_samples{0};
    std::size_t fitted_vertices{0};
    std::size_t fallback_vertices{0};
    std::size_t zero_fallback_vertices{0};
    std::size_t insufficient_stencil_vertices{0};
    std::size_t singular_stencil_vertices{0};
    std::size_t small_gradient_vertices{0};
    std::size_t fit_residual_failure_vertices{0};
    std::size_t smoothing_iterations_applied{0};
    Real min_curvature{0.0};
    Real max_curvature{0.0};
    Real max_abs_curvature{0.0};
    Real mean_fit_rms_residual{0.0};
    Real max_fit_rms_residual{0.0};
    Real mean_normalized_fit_residual{0.0};
    Real max_normalized_fit_residual{0.0};
    Real smoothing_mean_abs_update{0.0};
    Real smoothing_max_abs_update{0.0};
    bool reused_vertex_adjacency{false};
    bool reused_sample_adjacency{false};
    std::size_t vertex_adjacency_builds{0};
    std::size_t sample_adjacency_builds{0};
    std::string diagnostic{};
};

struct LevelSetCurvatureProjectionWorkspace {
    bool vertex_adjacency_valid{false};
    bool sample_adjacency_valid{false};
    GlobalIndex mesh_vertices{0};
    GlobalIndex mesh_cells{0};
    int mesh_dimension{0};
    bool mesh_revision_tracking_available{false};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t mesh_ownership_revision{0};
    std::uint64_t mesh_numbering_revision{0};
    std::uint64_t mesh_coordinate_configuration_key{0};
    std::uint64_t sample_signature{0};
    std::size_t vertex_adjacency_builds{0};
    std::size_t sample_adjacency_builds{0};
    std::vector<std::vector<GlobalIndex>> vertex_adjacency{};
    std::vector<std::vector<std::size_t>> sample_adjacency{};

    void clear() noexcept
    {
        vertex_adjacency_valid = false;
        sample_adjacency_valid = false;
        mesh_vertices = 0;
        mesh_cells = 0;
        mesh_dimension = 0;
        mesh_revision_tracking_available = false;
        mesh_geometry_revision = 0;
        mesh_topology_revision = 0;
        mesh_ownership_revision = 0;
        mesh_numbering_revision = 0;
        mesh_coordinate_configuration_key = 0;
        sample_signature = 0;
        vertex_adjacency_builds = 0;
        sample_adjacency_builds = 0;
        vertex_adjacency.clear();
        sample_adjacency.clear();
    }
};

/**
 * Recover a nodal projected mean-curvature field from vertex samples of an
 * implicit level set by fitting a local quadratic patch around each mesh
 * vertex.  The recovered curvature is div(grad(phi)/|grad(phi)|), so a signed
 * distance circle/sphere with outward-positive phi has positive curvature.
 *
 * This is a stabilized data-recovery utility for supplied-curvature
 * free-surface forcing.  It does not differentiate generated cut geometry and
 * it does not replace conservative level-set transport or signed-distance
 * reinitialization.
 */
[[nodiscard]] LevelSetCurvatureProjectionResult
projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values);

[[nodiscard]] LevelSetCurvatureProjectionResult
projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values);

[[nodiscard]] LevelSetCurvatureProjectionResult
projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace& workspace);

} // namespace svmp::FE::level_set
