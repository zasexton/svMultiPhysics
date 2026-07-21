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

} // namespace svmp::FE::level_set
