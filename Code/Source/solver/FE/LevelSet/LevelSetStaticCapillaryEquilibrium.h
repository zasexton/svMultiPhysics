#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Transactional fixed-volume minimization for a discrete static cap.
 */

#include "Core/Types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

/**
 * Options for piecewise-smooth minimization of the snapshot-owned surface,
 * Young-wall, and gravitational potential energy at a prescribed liquid
 * volume.
 */
struct LevelSetStaticCapillaryEquilibriumOptions {
    Real target_liquid_volume{0.0};
    Real volume_tolerance{1.0e-10};
    Real projected_gradient_tolerance{1.0e-8};
    Real pressure_representability_max_residual_norm{1.0e-10};
    Real pressure_representability_max_relative_distance{1.0e-8};
    Real physical_equilibrium_max_residual_norm{1.0e-10};
    Real constant_pressure_kkt_max_residual_norm{1.0e-10};
    Real constant_pressure_kkt_max_relative_distance{1.0e-8};

    // Coefficient-valued scales. The relative finite-difference step is
    // multiplied by the larger of this reference scale and the coefficient
    // magnitude. The inverse stiffness has units coefficient^2 / energy.
    Real finite_difference_reference_coefficient_scale{1.0};
    Real finite_difference_relative_step{1.0e-5};
    Real minimum_finite_difference_step{1.0e-12};
    int finite_difference_max_shrinks{12};

    int max_iterations{50};
    int max_line_search_iterations{20};
    // Exact functional derivatives may opt into a bounded sequence of cut
    // topology epochs. Difference gradients always remain in one epoch.
    bool allow_topology_epoch_transitions{false};
    int max_topology_epoch_transitions{8};
    Real projected_gradient_inverse_stiffness{1.0};
    Real tangent_trust_radius{0.05};
    Real maximum_coefficient_update_linf{0.25};
    Real line_search_shrink{0.5};
    Real armijo_fraction{1.0e-4};
    // Limited-memory inverse-Hessian updates accelerate the tangent solve
    // without assembling a dense shape Hessian. A zero history size selects
    // the safeguarded projected-gradient direction alone.
    int limited_memory_history_size{8};
    Real limited_memory_curvature_tolerance{1.0e-12};
    // Energy per unit volume. The iteration also raises this value above the
    // current multiplier magnitude so the l1 volume merit has a descent step.
    Real minimum_volume_merit_penalty{1.0};
};

/**
 * Globally reduced evaluation of one immutable candidate geometry.
 *
 * The energy excludes the volume-multiplier term. The evaluator must build
 * all fields from the supplied coefficients, return identical decision data
 * on every participating rank, and leave published solver state unchanged.
 */
struct LevelSetStaticCapillaryEquilibriumEvaluation {
    bool success{false};
    std::uint64_t snapshot_revision_key{0u};
    // Candidate-stable combinatorial key. It must exclude source-value and
    // snapshot revisions so geometry can move within one topology epoch.
    std::uint64_t cut_topology_key{0u};
    // Deterministic fingerprint of the constrained admissible trace used by
    // the candidate. The evaluator must recompute it from slave/master
    // relations, weights, and prescribed values.
    std::uint64_t constraint_semantics_key{0u};
    Real surface_wall_energy{
        std::numeric_limits<Real>::quiet_NaN()};
    Real gravitational_potential_energy{0.0};
    Real liquid_volume{std::numeric_limits<Real>::quiet_NaN()};
    // Optional exact derivatives in the evaluator's coefficient order. When
    // present, both arrays must cover every supplied coefficient; the
    // minimizer then avoids finite-difference trial geometry entirely.
    bool functional_derivatives_available{false};
    std::vector<Real> physical_potential_derivative{};
    std::vector<Real> liquid_volume_derivative{};
    bool pressure_representability_available{false};
    bool pressure_representability_converged{false};
    bool pressure_representability_breakdown{false};
    Real pressure_representability_residual_norm{
        std::numeric_limits<Real>::quiet_NaN()};
    Real pressure_representability_relative_distance{
        std::numeric_limits<Real>::quiet_NaN()};
    Real production_residual_norm{
        std::numeric_limits<Real>::quiet_NaN()};
    // A free constant pressure trace provides a sharper scalar KKT check for
    // zero-gravity equilibria. Fixed gauges and hydrostatic pressure fields
    // use the general constrained pressure-space certificate instead.
    bool constant_pressure_kkt_required{true};
    bool constant_pressure_kkt_available{false};
    Real constant_pressure_kkt_residual_norm{
        std::numeric_limits<Real>::quiet_NaN()};
    Real constant_pressure_kkt_relative_distance{
        std::numeric_limits<Real>::quiet_NaN()};
    // Trial evaluation must never obtain equilibrium by replacing the
    // production surface/Young load with a pressure-range projection.
    bool production_force_projection_applied{false};
    std::string diagnostic{};
};

enum class LevelSetStaticCapillaryEvaluationPurpose : std::uint8_t {
    FunctionalTrial,
    AcceptanceCertificate,
};

using LevelSetStaticCapillaryEquilibriumEvaluator =
    std::function<LevelSetStaticCapillaryEquilibriumEvaluation(
        std::span<const Real>,
        LevelSetStaticCapillaryEvaluationPurpose)>;

inline constexpr std::size_t
    kLevelSetStaticCapillaryLineSearchTraceCapacity{64u};

enum class LevelSetStaticCapillaryLineSearchPhase : std::uint8_t {
    Trial,
    DeferredReproduction,
};

enum class LevelSetStaticCapillaryLineSearchDisposition : std::uint8_t {
    UnavailableOrForbidden,
    TopologyRejected,
    ConstraintRejected,
    Deferred,
    SameTopologyMeritRejected,
    ArmijoAccepted,
    DerivativeResolutionAccepted,
    ReproductionAccepted,
    ReproductionRejected,
};

/**
 * Scalar evidence for one trial in the latest line-search attempt.
 *
 * At most kLevelSetStaticCapillaryLineSearchTraceCapacity records are
 * retained. Uncomputed scalar measurements remain NaN and are distinguished
 * from computed values by evaluation_available.
 */
struct LevelSetStaticCapillaryLineSearchRecord {
    std::size_t accepted_iteration_index{0u};
    std::size_t line_search_trial_index{0u};
    LevelSetStaticCapillaryLineSearchPhase phase{
        LevelSetStaticCapillaryLineSearchPhase::Trial};
    LevelSetStaticCapillaryLineSearchDisposition disposition{
        LevelSetStaticCapillaryLineSearchDisposition::
            UnavailableOrForbidden};
    bool evaluation_available{false};
    bool used_limited_memory_direction{false};
    bool used_projected_gradient_fallback_direction{false};
    Real step_size{std::numeric_limits<Real>::quiet_NaN()};
    std::uint64_t current_cut_topology_key{0u};
    std::uint64_t trial_cut_topology_key{0u};
    std::uint64_t current_constraint_semantics_key{0u};
    std::uint64_t trial_constraint_semantics_key{0u};
    Real current_merit{std::numeric_limits<Real>::quiet_NaN()};
    Real trial_merit{std::numeric_limits<Real>::quiet_NaN()};
    Real armijo_bound{std::numeric_limits<Real>::quiet_NaN()};
    Real predicted_merit_decrease{
        std::numeric_limits<Real>::quiet_NaN()};
    Real trial_volume_error{std::numeric_limits<Real>::quiet_NaN()};
};

struct LevelSetStaticCapillaryEquilibriumResult {
    bool success{false};
    bool converged{false};
    bool accepted_coefficients_assigned{false};
    int iterations{0};
    std::size_t functional_evaluations{0u};
    std::size_t acceptance_certificate_evaluations{0u};
    std::size_t finite_difference_step_shrinks{0u};
    std::size_t finite_difference_fourth_order_components{0u};
    Real minimum_finite_difference_step_used{0.0};
    Real maximum_finite_difference_step_used{0.0};
    Real maximum_energy_derivative_relative_correction{0.0};
    Real maximum_volume_derivative_relative_correction{0.0};
    std::size_t analytic_derivative_evaluations{0u};
    std::size_t derivative_resolution_step_acceptances{0u};
    std::size_t line_search_rejections{0u};
    std::size_t topology_change_rejections{0u};
    std::size_t topology_epoch_transitions{0u};
    std::size_t constraint_change_rejections{0u};
    std::size_t limited_memory_updates{0u};
    std::size_t limited_memory_resets{0u};
    std::size_t limited_memory_peak_history{0u};
    std::size_t projected_gradient_fallbacks{0u};
    std::size_t line_search_trace_total_attempt_count{0u};
    std::size_t line_search_trace_omitted_count{0u};
    std::vector<LevelSetStaticCapillaryLineSearchRecord>
        line_search_trace{};

    std::uint64_t initial_snapshot_revision_key{0u};
    std::uint64_t final_snapshot_revision_key{0u};
    std::uint64_t cut_topology_key{0u};
    std::uint64_t constraint_semantics_key{0u};
    Real initial_surface_wall_energy{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_surface_wall_energy{
        std::numeric_limits<Real>::quiet_NaN()};
    Real initial_gravitational_potential_energy{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_gravitational_potential_energy{
        std::numeric_limits<Real>::quiet_NaN()};
    Real initial_physical_potential_energy{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_physical_potential_energy{
        std::numeric_limits<Real>::quiet_NaN()};
    Real initial_liquid_volume{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_liquid_volume{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_volume_error{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_volume_multiplier{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_projected_gradient_norm{
        std::numeric_limits<Real>::quiet_NaN()};
    bool final_pressure_representability_available{false};
    bool final_pressure_representability_converged{false};
    bool final_pressure_representability_breakdown{false};
    Real final_pressure_representability_residual_norm{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_pressure_representability_relative_distance{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_production_residual_norm{
        std::numeric_limits<Real>::quiet_NaN()};
    bool final_constant_pressure_kkt_required{true};
    bool final_constant_pressure_kkt_available{false};
    Real final_constant_pressure_kkt_residual_norm{
        std::numeric_limits<Real>::quiet_NaN()};
    Real final_constant_pressure_kkt_relative_distance{
        std::numeric_limits<Real>::quiet_NaN()};
    std::string diagnostic{};
};

/**
 * Minimize a piecewise-smooth discrete capillary functional.
 *
 * Roundoff-balanced fourth-order differences are evaluated only inside the
 * current cut-topology and constrained-trace epoch. With exact derivatives,
 * an explicit option permits strictly merit-decreasing transitions into a
 * bounded number of new epochs. Each transition is reproduced before it is
 * accepted and discards all secant history. Each SQP-like step uses a
 * safeguarded limited-memory tangent inverse Hessian, satisfies the linearized
 * volume constraint, and descends an l1 volume-merit function. Convergence
 * additionally requires the evaluator's unprojected physical-potential load
 * to pass the constrained pressure-space and production-residual gates. A
 * constant-pressure KKT gate is additionally required when the evaluator
 * declares that the pressure constraints preserve the constant mode.
 *
 * `accepted_coefficients` is assigned only after every convergence gate
 * passes. It is left byte-for-byte unchanged on invalid input, evaluator
 * failure, disallowed topology change, line-search failure, or
 * nonconvergence.
 */
[[nodiscard]] LevelSetStaticCapillaryEquilibriumResult
minimizeLevelSetStaticCapillaryEquilibrium(
    const LevelSetStaticCapillaryEquilibriumOptions& options,
    std::span<const Real> input_coefficients,
    std::span<const std::size_t> active_coefficient_indices,
    const LevelSetStaticCapillaryEquilibriumEvaluator& evaluator,
    std::vector<Real>& accepted_coefficients);

} // namespace svmp::FE::level_set
