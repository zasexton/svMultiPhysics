#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Conservative algebraic-edge correction for a liquid indicator.
 */

#include "Core/Types.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

/**
 * @brief One canonically oriented interior algebraic edge.
 *
 * An edge is stored exactly once with `first_node < second_node`.  A positive
 * transfer is mass delivered to `first_node` and removed from `second_node`.
 * Both transfers have units of liquid-indicator measure integrated over the
 * complete time stage.
 */
struct LevelSetPhaseFluxEdge {
    GlobalIndex first_node{-1};
    GlobalIndex second_node{-1};
    Real low_order_mass_transfer{0.0};
    Real raw_antidiffusive_mass_transfer{0.0};
};

/**
 * @brief Inputs for one fully discrete conservative correction stage.
 *
 * The transported unknown is the nodal liquid indicator q, with q=1 in the
 * liquid and q=0 outside it.  Its declared discrete phase measure is
 * `sum_i lumped_control_volume[i] * q[i]`.  Per-node lower and upper bounds
 * must lie in [0, 1].  Empty boundary/source spans mean identically zero.
 *
 * The divergence source represents the explicit q div(u) contribution when
 * an advective equation is written in conservative flux form.  It must be
 * zero when the advecting field is discretely divergence compatible.  A
 * caller may waive constant preservation only when physical boundary data
 * intentionally replace an initially constant state.
 */
struct LevelSetPhaseFluxStageView {
    std::span<const Real> lumped_control_volume{};
    std::span<const Real> previous_liquid_indicator{};
    std::span<const Real> lower_liquid_indicator{};
    std::span<const Real> upper_liquid_indicator{};
    std::span<const LevelSetPhaseFluxEdge> interior_edges{};
    std::span<const Real> physical_boundary_mass_transfer{};
    std::span<const Real> discrete_divergence_mass_source{};
    Real invariant_tolerance{1.0e-12};
    Real component_activity_tolerance{1.0e-8};
    bool require_constant_preservation{true};
};

struct LevelSetPhaseFluxEdgeLedger {
    GlobalIndex first_node{-1};
    GlobalIndex second_node{-1};
    Real low_order_mass_transfer{0.0};
    Real raw_antidiffusive_mass_transfer{0.0};
    Real correction_factor{1.0};
    Real limited_antidiffusive_mass_transfer{0.0};
    Real low_order_pair_cancellation_residual{0.0};
    Real raw_pair_cancellation_residual{0.0};
    Real limited_pair_cancellation_residual{0.0};
};

struct LevelSetPhaseFluxNodeLedger {
    GlobalIndex node{-1};
    Real lumped_control_volume{0.0};
    Real previous_liquid_indicator{0.0};
    Real lower_liquid_indicator{0.0};
    Real upper_liquid_indicator{0.0};
    Real physical_boundary_mass_transfer{0.0};
    Real discrete_divergence_mass_source{0.0};
    Real low_order_interior_mass_transfer{0.0};
    Real raw_antidiffusive_mass_transfer{0.0};
    Real limited_antidiffusive_mass_transfer{0.0};
    Real positive_raw_antidiffusive_mass{0.0};
    Real negative_raw_antidiffusive_mass{0.0};
    Real positive_correction_factor{1.0};
    Real negative_correction_factor{1.0};
    Real low_order_liquid_indicator{0.0};
    Real raw_target_liquid_indicator{0.0};
    Real limited_liquid_indicator{0.0};
    Real low_order_local_mass_balance_residual{0.0};
    Real raw_target_local_mass_balance_residual{0.0};
    Real local_mass_balance_residual{0.0};
};

/**
 * @brief Balance of one connected phase-support component during a stage.
 *
 * Components are built from nodes whose previous, low-order, raw, or limited
 * phase indicator, normalized source, or normalized algebraic transfer
 * exceeds the declared component-activity tolerance. Active nodes joined by
 * an algebraic edge share one component. The identifier is the smallest
 * canonical node index in the component, so it is deterministic for
 * replicated distributed graphs. Subthreshold nonzero support is retained in
 * a separate balance record and therefore cannot hide phase measure.
 */
struct LevelSetPhaseFluxComponentLedger {
    GlobalIndex component_id{INVALID_GLOBAL_INDEX};
    std::size_t nodes{0u};
    Real previous_liquid_measure{0.0};
    Real low_order_liquid_measure{0.0};
    Real raw_target_liquid_measure{0.0};
    Real limited_liquid_measure{0.0};
    Real physical_boundary_mass_transfer{0.0};
    Real discrete_divergence_mass_source{0.0};
    Real low_order_interior_mass_transfer{0.0};
    Real raw_antidiffusive_mass_transfer{0.0};
    Real limited_antidiffusive_mass_transfer{0.0};
    Real low_order_balance_residual{0.0};
    Real raw_target_balance_residual{0.0};
    Real limited_balance_residual{0.0};
};

struct LevelSetPhaseFluxCorrectionResult {
    bool success{false};
    bool applied{false};
    bool low_order_bounds_satisfied{false};
    bool limited_bounds_satisfied{false};
    bool interior_cancellation_satisfied{false};
    bool local_balance_satisfied{false};
    bool global_balance_satisfied{false};
    bool component_balance_satisfied{false};
    bool component_measure_closure_satisfied{false};
    bool subthreshold_component_present{false};
    bool constant_state_input{false};
    bool constant_preservation_required{true};
    bool constant_preservation_satisfied{false};
    std::size_t limited_edges{0u};
    Real total_previous_liquid_measure{0.0};
    Real total_low_order_liquid_measure{0.0};
    Real total_raw_target_liquid_measure{0.0};
    Real total_limited_liquid_measure{0.0};
    Real total_physical_boundary_mass_transfer{0.0};
    Real total_discrete_divergence_mass_source{0.0};
    Real low_order_nodal_cancellation_residual{0.0};
    Real raw_nodal_cancellation_residual{0.0};
    Real limited_nodal_cancellation_residual{0.0};
    Real maximum_edge_pair_cancellation_residual{0.0};
    Real maximum_low_order_local_mass_balance_residual{0.0};
    Real maximum_raw_target_local_mass_balance_residual{0.0};
    Real maximum_local_mass_balance_residual{0.0};
    Real maximum_component_balance_residual{0.0};
    Real component_activity_tolerance{0.0};
    Real previous_component_measure_closure_residual{0.0};
    Real low_order_component_measure_closure_residual{0.0};
    Real raw_target_component_measure_closure_residual{0.0};
    Real limited_component_measure_closure_residual{0.0};
    Real boundary_component_transfer_closure_residual{0.0};
    Real divergence_component_source_closure_residual{0.0};
    Real low_order_component_transfer_closure_residual{0.0};
    Real raw_component_transfer_closure_residual{0.0};
    Real limited_component_transfer_closure_residual{0.0};
    Real low_order_global_mass_balance_residual{0.0};
    Real raw_target_global_mass_balance_residual{0.0};
    Real global_mass_balance_residual{0.0};
    Real maximum_constant_preservation_error{0.0};
    Real minimum_low_order_liquid_indicator{0.0};
    Real maximum_low_order_liquid_indicator{0.0};
    Real minimum_raw_target_liquid_indicator{0.0};
    Real maximum_raw_target_liquid_indicator{0.0};
    Real minimum_limited_liquid_indicator{0.0};
    Real maximum_limited_liquid_indicator{0.0};
    std::vector<LevelSetPhaseFluxNodeLedger> nodes{};
    std::vector<LevelSetPhaseFluxEdgeLedger> edges{};
    std::vector<GlobalIndex> node_component_ids{};
    std::vector<LevelSetPhaseFluxComponentLedger> components{};
    LevelSetPhaseFluxComponentLedger subthreshold_component{};
    std::string diagnostic{};
};

/**
 * @brief Apply a conservative invariant-domain edge-flux correction.
 *
 * The low-order predictor is assembled from the previous nodal masses,
 * physical-boundary transfers, declared divergence sources, and antisymmetric
 * low-order edge transfers.  Raw antidiffusive edge transfers are constrained
 * by symmetric pair factors, so every accepted correction is conservative by
 * construction.  The returned ledger retains every quantity needed to verify
 * local nodal balance, deterministic connected-component balance, and global
 * phase-measure balance.
 */
[[nodiscard]] LevelSetPhaseFluxCorrectionResult
applyLevelSetConservativePhaseFluxCorrection(
    const LevelSetPhaseFluxStageView& stage);

} // namespace svmp::FE::level_set
