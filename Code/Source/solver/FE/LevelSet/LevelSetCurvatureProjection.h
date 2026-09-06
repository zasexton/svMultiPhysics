#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Projected curvature recovery utilities for level-set interfaces.
 */

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::FE::systems {
class FESystem;
}

namespace svmp::FE::interfaces {
class FreeSurfaceGeometrySnapshot;
struct FreeSurfaceDiscreteFunctionalParameters;
}

namespace svmp::FE::level_set {

enum class LevelSetCurvatureSmoothingMode : std::uint8_t {
    LocalGraph = 0,
    MassStiffnessOperator = 1,
};

enum class LevelSetCurvatureRecoveryMode : std::uint8_t {
    LevelSetQuadratic = 0,
    GeneratedInterfacePatch = 1,
    KinematicAreaGradient = 2,
};

[[nodiscard]] const char* levelSetCurvatureRecoveryModeName(
    LevelSetCurvatureRecoveryMode mode) noexcept;

[[nodiscard]] LevelSetCurvatureRecoveryMode
parseLevelSetCurvatureRecoveryMode(std::string_view value);

[[nodiscard]] const char* levelSetCurvatureSmoothingModeName(
    LevelSetCurvatureSmoothingMode mode) noexcept;

[[nodiscard]] LevelSetCurvatureSmoothingMode
parseLevelSetCurvatureSmoothingMode(std::string_view value);

struct LevelSetKinematicAreaGradientYoungWall {
    int boundary_marker{-1};
    Real equilibrium_contact_angle_radians{0.0};
};

struct LevelSetCurvatureProjectionOptions {
    Real isovalue{0.0};
    Real gradient_tolerance{1.0e-10};
    // Relative numerical-rank tolerance for the coordinate-scaled,
    // column-pivoted weighted least-squares fit.  The historical member/XML
    // name is retained for input compatibility; normal equations are not
    // formed.
    Real normal_equation_tolerance{1.0e-12};
    Real max_normalized_fit_residual{0.0};
    int max_neighbor_rings{2};
    int max_neighbor_fallback_vertices{-1};
    int max_zero_fallback_vertices{-1};
    Real supplemental_sample_weight{1.0};
    LevelSetCurvatureRecoveryMode recovery_mode{
        LevelSetCurvatureRecoveryMode::LevelSetQuadratic};
    // Dimensionless coefficient c_l for the component-wise Helmholtz radius
    // ell_h = c_l sqrt(h_Gamma R_Gamma), where h_Gamma is the mean active
    // graph-edge length and R_Gamma is the equal-measure circle/sphere radius.
    // It is ignored by the other recovery modes; zero disables regularization.
    Real kinematic_area_gradient_filter_coefficient{1.0};
    bool kinematic_area_gradient_negative_liquid_side{true};
    std::vector<LevelSetKinematicAreaGradientYoungWall>
        kinematic_area_gradient_young_walls{};
    Real narrow_band_width{0.0};
    int smoothing_iterations{0};
    Real smoothing_relaxation{0.25};
    LevelSetCurvatureSmoothingMode smoothing_mode{
        LevelSetCurvatureSmoothingMode::LocalGraph};
};

struct LevelSetCurvatureProjectionSample {
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    std::array<Real, 3> coordinate{{0.0, 0.0, 0.0}};
    Real value{0.0};
    std::uint64_t free_surface_snapshot_revision_key{0};
    std::uint64_t source_value_revision{0};
    std::uint64_t cut_topology_revision{0};
    // Generated interface samples anchor the active band and provide the
    // geometric point cloud for GeneratedInterfacePatch recovery. Field-value
    // samples instead augment the level-set polynomial fit.
    bool generated_interface_geometry{false};
};

/**
 * Call-scoped producer binding for the authoritative energy/volume pair.
 * The producer must supply the unchanged coefficients used to construct the
 * snapshot and an independently obtained source identity. Matching epochs
 * alone do not prove coefficient equality. The snapshot, parameters, input
 * storage and mesh/system must outlive the call and must not be changed,
 * refreshed or rolled back concurrently. Curvature output must not alias
 * input storage. Neither this binding nor its borrowed objects are retained.
 */
struct LevelSetAuthoritativeDerivativeBinding {
    const interfaces::FreeSurfaceGeometrySnapshot& snapshot;
    const interfaces::FreeSurfaceDiscreteFunctionalParameters& functional;
    interfaces::LevelSetInterfaceSource input_source;
};

struct LevelSetCurvatureProjectionResult {
    bool success{false};
    std::uint64_t free_surface_snapshot_revision_key{0};
    std::uint64_t source_value_revision{0};
    std::uint64_t cut_rule_signature{0};
    std::size_t vertices{0};
    std::size_t supplemental_samples{0};
    std::size_t generated_interface_geometry_samples{0};
    std::size_t supplemental_sample_rows{0};
    std::size_t vertices_with_supplemental_samples{0};
    Real supplemental_sample_weight{1.0};
    LevelSetCurvatureRecoveryMode recovery_mode{
        LevelSetCurvatureRecoveryMode::LevelSetQuadratic};
    std::size_t generated_interface_patch_fitted_vertices{0};
    std::size_t generated_interface_patch_expanded_vertices{0};
    std::size_t kinematic_area_gradient_cut_cells{0};
    std::size_t kinematic_area_gradient_operator_vertices{0};
    std::size_t kinematic_area_gradient_operator_nonzeros{0};
    std::size_t kinematic_area_gradient_measure_evaluations{0};
    std::size_t kinematic_area_gradient_tie_break_vertices{0};
    std::size_t kinematic_area_gradient_linear_iterations{0};
    bool kinematic_area_gradient_minimum_norm_solver{false};
    std::size_t kinematic_area_gradient_young_wall_count{0};
    std::size_t kinematic_area_gradient_young_wall_boundary_faces{0};
    std::size_t kinematic_area_gradient_young_wall_cut_faces{0};
    std::size_t kinematic_area_gradient_young_wall_measure_evaluations{0};
    bool kinematic_area_gradient_collective_replication{false};
    int kinematic_area_gradient_parallel_size{1};
    std::size_t kinematic_area_gradient_gathered_owned_cells{0};
    std::size_t kinematic_area_gradient_gathered_owned_boundary_faces{0};
    std::size_t kinematic_area_gradient_components{0};
    Real kinematic_area_gradient_interface_measure{0.0};
    Real kinematic_area_gradient_kinematic_mass{0.0};
    Real kinematic_area_gradient_mass_weighted_mean_curvature{0.0};
    Real kinematic_area_gradient_mass_weighted_rms_deviation{0.0};
    Real kinematic_area_gradient_filter_coefficient{0.0};
    Real kinematic_area_gradient_min_characteristic_radius{0.0};
    Real kinematic_area_gradient_max_characteristic_radius{0.0};
    Real kinematic_area_gradient_min_filter_radius_cells{0.0};
    Real kinematic_area_gradient_max_filter_radius_cells{0.0};
    Real kinematic_area_gradient_min_filter_radius{0.0};
    Real kinematic_area_gradient_max_filter_radius{0.0};
    Real kinematic_area_gradient_max_tie_break_value{0.0};
    int kinematic_area_gradient_tie_break_sign{0};
    Real kinematic_area_gradient_surface_gradient_norm{0.0};
    Real kinematic_area_gradient_young_wall_gradient_norm{0.0};
    Real kinematic_area_gradient_total_energy_gradient_norm{0.0};
    // Derivatives of A_lg - sum cos(theta_w) A_sl,w and liquid volume for the
    // recovery working cut. Unbound success or supplemental-sample provenance
    // does not establish derivatives of an authoritative snapshot's scalar;
    // working coefficients may have been displaced. Only successful explicit
    // binding establishes the joint authoritative pair. Surface tension is
    // applied by the consumer exactly once; volume is unscaled.
    // Mesh-access overloads use visible-vertex order. FESystem overloads use
    // the field's global scalar-DOF order and set the flag below.
    std::vector<Real> kinematic_area_gradient_total_energy_derivative{};
    std::vector<Real> kinematic_area_gradient_liquid_volume_derivative{};
    bool kinematic_area_gradient_derivatives_global_dof_order{false};
    Real kinematic_area_gradient_max_relative_fd_disagreement{0.0};
    Real kinematic_area_gradient_max_regularized_identity_residual{0.0};
    Real kinematic_area_gradient_max_relative_regularized_identity_residual{
        0.0};
    Real kinematic_area_gradient_relative_linear_residual{0.0};
    Real narrow_band_width{0.0};
    std::size_t narrow_band_vertices{0};
    std::size_t skipped_far_vertices{0};
    std::size_t fitted_vertices{0};
    std::size_t fallback_vertices{0};
    std::size_t zero_fallback_vertices{0};
    std::size_t insufficient_stencil_vertices{0};
    std::size_t singular_stencil_vertices{0};
    std::size_t small_gradient_vertices{0};
    std::size_t fit_residual_failure_vertices{0};
    LevelSetCurvatureSmoothingMode smoothing_mode{
        LevelSetCurvatureSmoothingMode::LocalGraph};
    std::size_t smoothing_iterations_applied{0};
    std::size_t smoothing_operator_edges{0};
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
    std::uint64_t free_surface_snapshot_revision_key{0};
    std::uint64_t source_value_revision{0};
    std::uint64_t cut_rule_signature{0};
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
        free_surface_snapshot_revision_key = 0;
        source_value_revision = 0;
        cut_rule_signature = 0;
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
 * reinitialization.  A positive narrow_band_width restricts recovery,
 * fallback, and smoothing to vertices within |phi-isovalue| <= width plus
 * vertices touched by supplemental interface samples.
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

/**
 * Collectively recover curvature on a distributed P1 field. Owned simplex
 * cells and exterior faces are canonicalized by field DOF and global entity
 * identity, the geometric operator is solved identically on every rank, and
 * the result is mapped back to locally visible mesh vertices.
 */
[[nodiscard]] LevelSetCurvatureProjectionResult
projectLevelSetMeanCurvatureToVertices(
    const systems::FESystem& system,
    FieldId level_set_field,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values);

[[nodiscard]] LevelSetCurvatureProjectionResult
projectLevelSetMeanCurvatureToVertices(
    const systems::FESystem& system,
    FieldId level_set_field,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace& workspace);

/**
 * Request the joint authoritative energy/volume derivative pair. Bound
 * success requires validated source, functional, geometry and retained
 * support, and stamps the result with the bound snapshot/source revisions.
 * Failure leaves no derivative arrays. Geometry eligibility is currently
 * unverified, so otherwise valid bindings report source_branch_unverified.
 * Existing unbound overloads retain their standalone recovery contract.
 *
 * This mesh-only overload requires an evaluator source and values in visible
 * mesh-vertex order on a serial mesh. Historical coefficient equality is a
 * trusted producer precondition, not inferred from size or source epochs.
 */
[[nodiscard]] LevelSetCurvatureProjectionResult
projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> producer_vertex_values,
    const LevelSetAuthoritativeDerivativeBinding& binding,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace* workspace = nullptr);

/**
 * Collective field overload of the same bound contract. The producer span
 * is the full solution in global system order, including other fields.
 * Required scalar P1 field values are sampled directly from this span;
 * prescribed storage and point-search fallback are unsupported. Successful
 * derivatives use global scalar field DOF order, as in the unbound overload.
 * All ranks on the system communicator must call this overload together.
 */
[[nodiscard]] LevelSetCurvatureProjectionResult
projectLevelSetMeanCurvatureToVertices(
    const systems::FESystem& system,
    FieldId level_set_field,
    std::span<const Real> producer_solution,
    const LevelSetAuthoritativeDerivativeBinding& binding,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace* workspace = nullptr);

} // namespace svmp::FE::level_set
