#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Conservative P1 phase-state projection from authoritative cut rules.
 */

#include "Core/Types.h"
#include "Geometry/CutQuadrature.h"
#include "LevelSet/LevelSetConservativePhaseOperator.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::systems {
class FESystem;
}

namespace svmp::FE::level_set {

struct LevelSetP1PhaseProjectionOptions {
    int interface_marker{-1};
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    /// Zero selects the same geometry-aware full-cell rule as the graph.
    int quadrature_order{0};
    Real invariant_tolerance{1.0e-12};
};

/**
 * @brief Lumped projection of the retained liquid characteristic function.
 *
 * `liquid_phase_mass[i]` is the assembled moment
 * `integral_liquid N_i dx`. The bounded indicator is that moment divided by
 * the graph control volume. Both vectors use the phase field's global DOF
 * numbering and are replicated after communicator reduction.
 */
struct LevelSetP1PhaseProjectionResult {
    bool success{false};
    bool phase_bounds_satisfied{false};
    bool rule_moment_closure_satisfied{false};
    bool global_measure_closure_satisfied{false};
    bool complement_bounds_satisfied{false};
    int interface_marker{-1};
    geometry::CutIntegrationSide liquid_side{
        geometry::CutIntegrationSide::Negative};
    std::size_t nodes{0u};
    std::size_t owned_rules{0u};
    std::size_t quadrature_points{0u};
    std::uint64_t cut_context_revision{0u};
    std::uint64_t source_value_revision{0u};
    Real retained_liquid_measure{0.0};
    Real projected_liquid_measure{0.0};
    Real measure_closure_residual{0.0};
    Real maximum_rule_moment_closure_residual{0.0};
    Real minimum_liquid_indicator{0.0};
    Real maximum_liquid_indicator{0.0};
    Real maximum_lower_bound_violation{0.0};
    Real maximum_upper_bound_violation{0.0};
    std::vector<Real> liquid_phase_mass{};
    std::vector<Real> liquid_indicator{};
    std::string diagnostic{};
};

/**
 * @brief Project an authoritative retained cut domain into a P1 phase field.
 *
 * Only locally owned rules contribute. The supplied graph must describe the
 * same current mesh and phase-field layout. Partial rules must integrate P1
 * moments; stale rule revisions, duplicate/malformed geometry, nonpositive
 * mapped weights, incomplete phase coverage, and bound or measure failures
 * are rejected before a state is returned.
 */
[[nodiscard]] LevelSetP1PhaseProjectionResult
projectLevelSetP1PhaseIndicatorFromCutContext(
    const systems::FESystem& system,
    FieldId liquid_indicator_field,
    const LevelSetP1PhaseTransportGraph& graph,
    const LevelSetP1PhaseProjectionOptions& options);

struct LevelSetP1PhaseGeometrySensitivityEdge {
    GlobalIndex first_node{-1};
    GlobalIndex second_node{-1};
    Real coefficient{0.0};
};

/**
 * @brief Interface shape derivative of the retained nodal phase moments.
 *
 * The symmetric matrix represented by `diagonal` and `edges` is
 *
 *     S_ij = integral_interface N_i N_j / |grad(phi)| ds.
 *
 * The derivative of a negative-side phase moment is `-S`; the derivative of
 * a positive-side moment is `+S`. The phase and level-set fields are required
 * to have identical scalar P1 cell layouts so the matrix has one canonical
 * replicated nodal numbering in serial and distributed runs.
 */
struct LevelSetP1PhaseGeometrySensitivityResult {
    bool success{false};
    bool field_layouts_identical{false};
    bool level_set_null_space_satisfied{false};
    bool positive_diagonal_satisfied{false};
    int interface_marker{-1};
    int dimension{0};
    std::size_t nodes{0u};
    std::size_t active_nodes{0u};
    std::size_t owned_rules{0u};
    std::size_t quadrature_points{0u};
    std::uint64_t cut_context_revision{0u};
    std::uint64_t source_value_revision{0u};
    Real interface_measure{0.0};
    Real minimum_level_set_gradient{0.0};
    Real minimum_cell_node_distance{0.0};
    Real maximum_level_set_null_residual{0.0};
    std::vector<Real> diagonal{};
    std::vector<LevelSetP1PhaseGeometrySensitivityEdge> edges{};
    std::string diagnostic{};
};

/**
 * @brief Assemble the local phase-moment response to a P1 level-set update.
 *
 * Interface rules and their source revision come from the authoritative cut
 * context. Only owned rules contribute before communicator reduction.
 * `solution` is the complete FE-ordered state used to build that context.
 */
[[nodiscard]] LevelSetP1PhaseGeometrySensitivityResult
buildLevelSetP1PhaseGeometrySensitivity(
    const systems::FESystem& system,
    FieldId level_set_field,
    FieldId liquid_indicator_field,
    const LevelSetP1PhaseTransportGraph& graph,
    const LevelSetP1PhaseProjectionOptions& options,
    std::span<const Real> solution);

struct LevelSetP1PhaseGeometryCorrectionOptions {
    Real invariant_tolerance{1.0e-12};
    Real relative_linear_tolerance{1.0e-10};
    /// Zero selects a size-dependent bounded iteration count.
    int maximum_linear_iterations{0};
};

/**
 * @brief Minimum-norm fixed-topology update for local phase-moment closure.
 *
 * The trace mass matrix has at least one scaling null mode per disconnected
 * interface component because multiplying `phi` by a positive constant does
 * not move its zero set. Tensor-product traces can have additional
 * zero-on-interface modes. The projected, unpreconditioned solve removes the
 * known scaling modes and returns the minimum-norm range-space update across
 * any remaining trace kernel.
 */
struct LevelSetP1PhaseGeometryCorrectionResult {
    bool success{false};
    bool target_compatible{false};
    bool linear_solve_converged{false};
    int iterations{0};
    std::size_t active_nodes{0u};
    std::size_t interface_components{0u};
    Real right_hand_side_norm{0.0};
    Real maximum_null_compatibility_residual{0.0};
    Real linear_residual_norm{0.0};
    Real maximum_predicted_mass_residual{0.0};
    std::vector<Real> level_set_increment{};
    std::vector<Real> predicted_liquid_mass_change{};
    std::string diagnostic{};
};

[[nodiscard]] LevelSetP1PhaseGeometryCorrectionResult
solveLevelSetP1PhaseGeometryCorrection(
    const LevelSetP1PhaseGeometrySensitivityResult& sensitivity,
    geometry::CutIntegrationSide liquid_side,
    std::span<const Real> current_level_set,
    std::span<const Real> current_liquid_phase_mass,
    std::span<const Real> target_liquid_phase_mass,
    const LevelSetP1PhaseGeometryCorrectionOptions& options = {});

} // namespace svmp::FE::level_set
