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
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::systems {
class FESystem;
}

namespace svmp::FE::level_set {

namespace detail {
struct LevelSetP1PhaseCollectiveState;
}

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
    /// Rank-local mesh cache stamps used only to decide when this rank's
    /// graph is stale. Valid partitions need not have equal values.
    std::uint64_t geometry_revision{0u};
    std::uint64_t topology_revision{0u};
    std::uint64_t ownership_revision{0u};
    std::uint64_t numbering_revision{0u};
    /// Communicator-replicated FE field-layout identity.
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
    /**
     * Opaque ownership of the duplicated communicator used by collective
     * stage APIs. In a distributed call every rank must retain the collective
     * state and communicator metadata of its graph from the same collective
     * builder invocation. The stage API has no independent communicator with
     * which it could safely diagnose an asymmetric missing/null handle.
     */
    std::shared_ptr<const detail::LevelSetP1PhaseCollectiveState>
        collective_state{};
    std::string diagnostic{};
};

/**
 * @brief Execution-layout-sensitive identity of one transport graph.
 *
 * `content_revision` hashes the complete replicated graph in canonical node
 * and edge order, including communicator size, distribution mode, and logical
 * edge ownership. It is stable only within one execution layout; serial/MPI
 * numerical equivalence must compare operator or ledger values instead. The
 * four mesh revisions are deliberately retained as local cache stamps: valid
 * distributed partitions need not assign them identical values on every
 * rank, but one rank must observe the same stamps at q^n and at the operator
 * endpoint.
 */
struct LevelSetP1PhaseGraphIdentity {
    int dimension{0};
    std::size_t nodes{0u};
    std::size_t edges{0u};
    std::uint64_t geometry_revision{0u};
    std::uint64_t topology_revision{0u};
    std::uint64_t ownership_revision{0u};
    std::uint64_t numbering_revision{0u};
    std::uint64_t dof_layout_revision{0u};
    std::uint64_t content_revision{0u};
};

[[nodiscard]] LevelSetP1PhaseGraphIdentity
levelSetP1PhaseGraphIdentity(
    const LevelSetP1PhaseTransportGraph& graph) noexcept;

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
    /// True only after every replicated input has passed collective equality.
    bool replicated_stage_inputs_satisfied{false};
    Real maximum_courant{0.0};
    Real minimum_low_order_coefficient{0.0};
    Real maximum_strong_form_decomposition_residual{0.0};
    /// Exact step supplied to the successful stage.
    Real time_step{0.0};
    /// Exact operator-endpoint nodal velocity sampled by this stage.
    std::vector<std::array<Real, 3>> sampled_nodal_velocity{};
    /// Exact policy and thresholds used to execute this stage.
    LevelSetP1PhaseStageOptions executed_options{};
    std::vector<Real> nodal_courant{};
    std::vector<Real> physical_boundary_mass_transfer{};
    std::vector<Real> discrete_divergence_mass_source{};
    std::vector<LevelSetPhaseFluxEdge> flux_edges{};
    LevelSetPhaseFluxCorrectionResult correction{};
    std::string diagnostic{};
};

/**
 * @brief Supported identifier for an explicit-indicator split stage.
 *
 * The Backward-Euler entry advances q^n explicitly over dt while sampling the
 * converged operator velocity u^{n+1}. It does not describe an implicit solve
 * for q^{n+1}. Generalized-alpha is named for fail-closed diagnostics only and
 * is not supported by this contract.
 */
enum class LevelSetP1PhaseSplitScheme : std::uint8_t {
    BackwardEulerExplicitIndicatorEndpointVelocity = 1u,
    GeneralizedAlphaUnsupported = 2u,
};

[[nodiscard]] const char* levelSetP1PhaseSplitSchemeName(
    LevelSetP1PhaseSplitScheme scheme) noexcept;

enum class LevelSetP1PhaseTransportMeshPolicy : std::uint8_t {
    FixedBackground = 1u,
    MovingMeshUnsupported = 2u,
};

/**
 * @brief Bit-exact metadata and content provenance for one BE split stage.
 *
 * A valid record has q_input_time == step_start_time, velocity_state_time ==
 * step_end_time, and step_end_time == step_start_time + time_step by exact
 * `Real` object bits. Both graph identities must agree exactly on the local
 * rank. Operator state, q, velocity, graph content, and ledger provenance use
 * nonzero 64-bit revisions or versioned fingerprints, so content fingerprints
 * retain the usual probabilistic collision risk of a fixed-width digest. The
 * fixed-background policy is mandatory because this operator has no ALE
 * mesh-flux/GCL term.
 */
struct LevelSetP1PhaseSplitStageProvenance {
    LevelSetP1PhaseSplitScheme scheme{
        LevelSetP1PhaseSplitScheme::
            BackwardEulerExplicitIndicatorEndpointVelocity};
    LevelSetP1PhaseTransportMeshPolicy transport_mesh_policy{
        LevelSetP1PhaseTransportMeshPolicy::FixedBackground};
    int temporal_order{0};
    std::uint64_t prospective_step{0u};
    std::uint64_t attempt{0u};
    Real step_start_time{0.0};
    Real step_end_time{0.0};
    Real q_input_time{0.0};
    Real velocity_state_time{0.0};
    Real time_step{0.0};
    std::uint64_t operator_state_revision{0u};
    std::uint64_t previous_q_revision{0u};
    std::uint64_t nodal_velocity_revision{0u};
    LevelSetP1PhaseGraphIdentity previous_graph_identity{};
    LevelSetP1PhaseGraphIdentity operator_graph_identity{};
    std::uint64_t final_flux_ledger_digest{0u};
    /// Exact policy and thresholds claimed for the executed stage.
    LevelSetP1PhaseStageOptions stage_options{};
};

struct LevelSetP1PhaseSplitStageValidationResult {
    bool valid{false};
    LevelSetP1PhaseSplitStageProvenance provenance{};
    LevelSetP1PhaseGraphIdentity actual_operator_graph_identity{};
    std::uint64_t computed_previous_q_revision{0u};
    std::uint64_t computed_nodal_velocity_revision{0u};
    std::uint64_t computed_flux_ledger_digest{0u};
    std::string diagnostic{};
};

[[nodiscard]] std::uint64_t levelSetP1PhaseScalarContentRevision(
    std::span<const Real> values) noexcept;

[[nodiscard]] std::uint64_t levelSetP1PhaseVelocityContentRevision(
    std::span<const std::array<Real, 3>> values) noexcept;

[[nodiscard]] std::uint64_t levelSetP1PhaseFluxLedgerDigest(
    const LevelSetP1PhaseTransportStageResult& stage) noexcept;

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
 * are retained separately in the returned ledger. A distributed graph makes
 * this a collective call: nodal inputs, the time step, and every stage option
 * must be identical on its field communicator. Constant-state validation
 * covers the low-order, raw-target, and limited states. Every distributed
 * rank must retain the collective state and communicator metadata of its
 * graph from the same collective builder call; an asymmetrically removed
 * collective state violates this API's entry precondition and cannot be
 * failure-consensused without an external communicator.
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

/**
 * @brief Validate a fixed-mesh BE split-stage metadata/fingerprint binding.
 *
 * Distributed validation is collective on the graph communicator. Replicated
 * metadata and content digests must match exactly across ranks; rank-local
 * mesh cache stamps are checked against the previous/operator identities only
 * on their owning rank. The validator independently streams the graph and
 * claimed ledger equations using O(nodes) scratch and no duplicate O(edges)
 * result before it accepts the claimed ledger digest. This production
 * contract also re-derives the exact q^n one-ring min/max bounds from the
 * canonical graph; a stage executed with caller-selected generic bounds is
 * not accepted as this production split stage. Every distributed rank must
 * retain the collective state and communicator metadata of its graph from the
 * same collective builder call; the validator has no separate communicator
 * on which to diagnose an asymmetrically missing graph communicator.
 */
[[nodiscard]] LevelSetP1PhaseSplitStageValidationResult
validateLevelSetP1PhaseSplitStage(
    const LevelSetP1PhaseTransportGraph& graph,
    std::span<const Real> previous_liquid_indicator,
    const LevelSetP1PhaseTransportStageResult& stage,
    const LevelSetP1PhaseSplitStageProvenance& provenance);

} // namespace svmp::FE::level_set
