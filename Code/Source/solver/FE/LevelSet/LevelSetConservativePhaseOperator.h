#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Geometry-aware P1 graph operator for conservative liquid-indicator
 * transport.
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetConservativePhaseTransport.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::systems {
class FESystem;
}

namespace svmp::FE::level_set {

/**
 * @brief One canonical algebraic edge of the assembled gradient matrix.
 *
 * For `first_node=i` and `second_node=j`, the two vector coefficients are
 *
 *     C_ij = integral N_i grad(N_j) dx,
 *     C_ji = integral N_j grad(N_i) dx.
 */
struct LevelSetP1PhaseGradientEdge {
    GlobalIndex first_node{-1};
    GlobalIndex second_node{-1};
    int owner_rank{-1};
    std::array<Real, 3> first_test_second_gradient{};
    std::array<Real, 3> second_test_first_gradient{};
};

struct LevelSetP1PhaseGraphOptions {
    /// Zero selects a geometry-order-aware rule independently on each cell.
    int quadrature_order{0};
    Real invariant_tolerance{1.0e-12};
};

/**
 * @brief P1 control-volume and gradient graph assembled from an FE system.
 *
 * `boundary_column_sum[i] = sum_j C_ji` is the discrete boundary vector
 * `integral_boundary N_i n ds`. `diagonal_gradient[i]` stores `C_ii`; the
 * off-diagonal coefficients are stored exactly once in `edges`.
 */
struct LevelSetP1PhaseTransportGraph {
    bool success{false};
    bool partition_of_unity_satisfied{false};
    bool gradient_partition_satisfied{false};
    bool positive_control_volumes_satisfied{false};
    bool gradient_row_sum_satisfied{false};
    bool measure_closure_satisfied{false};
    bool edge_ownership_satisfied{false};
    bool distributed{false};
    bool replicated_sparse_graph{false};
    int dimension{0};
    int parallel_rank{0};
    int parallel_size{1};
    int maximum_quadrature_order{0};
    std::size_t cells{0u};
    std::size_t local_owned_cells{0u};
    std::size_t nodes{0u};
    std::size_t locally_owned_edges{0u};
    std::uint64_t geometry_revision{0u};
    std::uint64_t topology_revision{0u};
    std::uint64_t ownership_revision{0u};
    std::uint64_t numbering_revision{0u};
    std::uint64_t dof_layout_revision{0u};
    Real physical_measure{0.0};
    Real total_lumped_control_volume{0.0};
    Real minimum_lumped_control_volume{0.0};
    Real minimum_jacobian_determinant{0.0};
    Real maximum_partition_of_unity_residual{0.0};
    Real maximum_gradient_partition_residual{0.0};
    Real maximum_gradient_row_sum_residual{0.0};
    Real measure_closure_residual{0.0};
    std::vector<Real> lumped_control_volume{};
    std::vector<std::array<Real, 3>> diagonal_gradient{};
    std::vector<std::array<Real, 3>> boundary_column_sum{};
    std::vector<LevelSetP1PhaseGradientEdge> edges{};
    std::string diagnostic{};
};

/**
 * @brief Assemble the P1 mass-lumped control volumes and CG gradient graph.
 *
 * In a multi-rank build, only owned cells contribute. Nodal quantities are
 * summed on the field communicator and sparse edge fragments are merged into
 * the same canonical replicated graph on every rank. Every edge has one
 * logical owner: the lower of its two endpoint-owner ranks. Replication
 * matches the globally indexed level-set state contract and does not duplicate
 * an edge inside a stage ledger.
 */
[[nodiscard]] LevelSetP1PhaseTransportGraph
buildLevelSetP1PhaseTransportGraph(
    const systems::FESystem& system,
    FieldId liquid_indicator_field,
    const LevelSetP1PhaseGraphOptions& options = {});

struct LevelSetP1PhaseStageOptions {
    Real invariant_tolerance{1.0e-12};
    Real component_activity_tolerance{1.0e-8};
    Real maximum_courant{1.0};
    bool enforce_courant_limit{true};
    bool require_constant_preservation{true};
};

/**
 * @brief Complete graph-flux construction and correction result for one
 * forward-Euler stage.
 */
struct LevelSetP1PhaseTransportStageResult {
    bool success{false};
    bool courant_satisfied{false};
    bool low_order_coefficients_nonnegative{false};
    bool strong_form_decomposition_satisfied{false};
    Real maximum_courant{0.0};
    Real minimum_low_order_coefficient{0.0};
    Real maximum_strong_form_decomposition_residual{0.0};
    std::vector<Real> nodal_courant{};
    std::vector<Real> physical_boundary_mass_transfer{};
    std::vector<Real> discrete_divergence_mass_source{};
    std::vector<LevelSetPhaseFluxEdge> flux_edges{};
    LevelSetPhaseFluxCorrectionResult correction{};
    std::string diagnostic{};
};

/**
 * @brief Build and limit one conservative P1 phase-indicator stage.
 *
 * Velocity values use the graph's nodal numbering. The low-order graph
 * viscosity is symmetric. Its coefficient on edge `(i,j)` is
 *
 *     d_ij = max(abs(C_ij dot u_j), abs(C_ji dot u_i)).
 *
 * The raw antidiffusive transfer removes this viscosity, recovering the
 * lumped strong-CG advective target. Boundary and discrete-divergence terms
 * are retained separately in the returned ledger.
 */
[[nodiscard]] LevelSetP1PhaseTransportStageResult
advanceLevelSetP1ConservativePhaseStage(
    const LevelSetP1PhaseTransportGraph& graph,
    std::span<const Real> previous_liquid_indicator,
    std::span<const Real> lower_liquid_indicator,
    std::span<const Real> upper_liquid_indicator,
    std::span<const std::array<Real, 3>> nodal_velocity,
    Real time_step,
    const LevelSetP1PhaseStageOptions& options = {});

} // namespace svmp::FE::level_set
