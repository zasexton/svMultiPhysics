#pragma once

/**
 * @file
 * @brief Transactional fixed-volume minimization of a level-set energy.
 *
 * The minimizer changes level-set coefficients c to minimize an energy E(c),
 * the sum of named components, subject to a prescribed volume V(c) = V0, for
 * example the surface, wall and gravitational energy of a discrete static
 * cap.  A caller-supplied evaluator measures one candidate.  What a converged
 * candidate must also satisfy is data: named acceptance certificates with
 * their limits and diagnostics (LevelSetVolumeConstrainedProblem), measured
 * by the evaluator.
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
 * Options for piecewise-smooth minimization of the energy at a prescribed
 * liquid volume.  The limits of the acceptance certificates are part of the
 * problem (LevelSetVolumeConstrainedAcceptanceCheck).
 */
struct LevelSetVolumeConstrainedMinimizerOptions {
    Real target_liquid_volume{0.0};
    Real volume_tolerance{1.0e-10};
    Real projected_gradient_tolerance{1.0e-8};

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
 * One acceptance certificate of a converged candidate, declared as data.
 *
 * The evaluator measures it on every candidate
 * (LevelSetVolumeConstrainedCheckValue, one value per limit).  A
 * parameter-stationary candidate is accepted only when every required
 * certificate is available, converged without breakdown, and each measured
 * value is finite, nonnegative and at most its limit.  The diagnostics are
 * the texts the minimizer reports.
 */
struct LevelSetVolumeConstrainedAcceptanceCheck {
    std::string name{};
    std::vector<Real> limits{};
    // When nonempty, an evaluation whose available certificate has a
    // nonfinite or negative value is invalid, with this diagnostic.
    std::string invalid_state_diagnostic{};
    // A required certificate is unavailable at the stationary candidate.
    std::string unavailable_diagnostic{};
    // Not converged, broken down, or a value that is not finite and
    // nonnegative or exceeds its limit.
    std::string failed_diagnostic{};
};

/// One certificate as measured on a candidate, in the problem's order.
struct LevelSetVolumeConstrainedCheckValue {
    bool required{true};
    bool available{false};
    bool converged{true};
    bool breakdown{false};
    std::vector<Real> values{};
};

/// What the minimizer minimizes and certifies.
struct LevelSetVolumeConstrainedProblem {
    // The named components of the energy; the energy is their sum, in this
    // order.
    std::vector<std::string> energy_components{};
    std::vector<LevelSetVolumeConstrainedAcceptanceCheck> acceptance_checks{};
    // The diagnostics of an evaluation that applied a forbidden substitution
    // (LevelSetVolumeConstrainedEvaluation::forbidden_substitution_applied):
    // the evaluation's, and the minimizer's.
    std::string forbidden_substitution_evaluation_diagnostic{};
    std::string forbidden_substitution_diagnostic{};
};

/**
 * Globally reduced evaluation of one immutable candidate geometry.
 *
 * The energy excludes the volume-multiplier term. The evaluator must build
 * all fields from the supplied coefficients, return identical decision data
 * on every participating rank, and leave published solver state unchanged.
 */
struct LevelSetVolumeConstrainedEvaluation {
    bool success{false};
    std::uint64_t snapshot_revision_key{0u};
    // Candidate-stable combinatorial key. It must exclude source-value and
    // snapshot revisions so geometry can move within one topology epoch.
    std::uint64_t cut_topology_key{0u};
    // Deterministic fingerprint of the constrained admissible trace used by
    // the candidate. The evaluator must recompute it from slave/master
    // relations, weights, and prescribed values.
    std::uint64_t constraint_semantics_key{0u};
    // The problem's energy components, in its order.
    std::vector<Real> energy_components{};
    Real liquid_volume{std::numeric_limits<Real>::quiet_NaN()};
    // Optional exact derivatives in the evaluator's coefficient order. When
    // present, both arrays must cover every supplied coefficient; the
    // minimizer then avoids finite-difference trial geometry entirely.
    bool functional_derivatives_available{false};
    std::vector<Real> energy_derivative{};
    std::vector<Real> liquid_volume_derivative{};
    // The problem's acceptance certificates, in its order.
    std::vector<LevelSetVolumeConstrainedCheckValue> checks{};
    // Trial evaluation must never obtain the energy's equilibrium by a
    // substitution the problem forbids (for a static cap, replacing the
    // production surface/Young load with a pressure-range projection).
    bool forbidden_substitution_applied{false};
    std::string diagnostic{};
};

enum class LevelSetVolumeConstrainedEvaluationPurpose : std::uint8_t {
    FunctionalTrial,
    AcceptanceCertificate,
};

using LevelSetVolumeConstrainedEvaluator =
    std::function<LevelSetVolumeConstrainedEvaluation(
        std::span<const Real>,
        LevelSetVolumeConstrainedEvaluationPurpose)>;

inline constexpr std::size_t
    kLevelSetVolumeConstrainedLineSearchTraceCapacity{64u};

enum class LevelSetVolumeConstrainedLineSearchPhase : std::uint8_t {
    Trial,
    DeferredReproduction,
};

enum class LevelSetVolumeConstrainedLineSearchDisposition : std::uint8_t {
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
 * At most kLevelSetVolumeConstrainedLineSearchTraceCapacity records are
 * retained. Uncomputed scalar measurements remain NaN and are distinguished
 * from computed values by evaluation_available.
 */
struct LevelSetVolumeConstrainedLineSearchRecord {
    std::size_t accepted_iteration_index{0u};
    std::size_t line_search_trial_index{0u};
    LevelSetVolumeConstrainedLineSearchPhase phase{
        LevelSetVolumeConstrainedLineSearchPhase::Trial};
    LevelSetVolumeConstrainedLineSearchDisposition disposition{
        LevelSetVolumeConstrainedLineSearchDisposition::
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

inline constexpr std::size_t
    kLevelSetVolumeConstrainedFailureStateCoefficientCapacity{65536u};

enum class LevelSetVolumeConstrainedFailureStateStatus : std::uint8_t {
    NotApplicable,
    Available,
    CapacityExceeded,
    RecordingFailed,
};

struct LevelSetVolumeConstrainedFailureState {
    LevelSetVolumeConstrainedFailureStateStatus status{
        LevelSetVolumeConstrainedFailureStateStatus::NotApplicable};
    std::size_t coefficient_count{0u};
    std::size_t active_coefficient_count{0u};
    std::size_t accepted_iteration_index{0u};
    std::uint64_t snapshot_revision_key{0u};
    std::uint64_t cut_topology_key{0u};
    std::uint64_t constraint_semantics_key{0u};
    Real current_merit{std::numeric_limits<Real>::quiet_NaN()};
    Real merit_penalty{std::numeric_limits<Real>::quiet_NaN()};
    // Full raw field-coefficient order, before the evaluator's affine map.
    std::vector<Real> coefficients{};
    // Sorted coefficient indices used by the existing minimizer.
    std::vector<std::size_t> active_coefficient_indices{};
    // All three arrays below use that active-index order, not full-field order.
    std::vector<Real> energy_gradient{};
    std::vector<Real> volume_gradient{};
    std::vector<Real> direction{};
};

struct LevelSetVolumeConstrainedMinimizerResult {
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
    std::vector<LevelSetVolumeConstrainedLineSearchRecord>
        line_search_trace{};
    LevelSetVolumeConstrainedFailureState line_search_failure_state{};

    std::uint64_t initial_snapshot_revision_key{0u};
    std::uint64_t final_snapshot_revision_key{0u};
    std::uint64_t cut_topology_key{0u};
    std::uint64_t constraint_semantics_key{0u};
    // The energy components (empty until the initial candidate is
    // evaluated) and their sum.
    std::vector<Real> initial_energy_components{};
    std::vector<Real> final_energy_components{};
    Real initial_energy{std::numeric_limits<Real>::quiet_NaN()};
    Real final_energy{std::numeric_limits<Real>::quiet_NaN()};
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
    // The acceptance certificates of the final candidate (empty until the
    // initial candidate is evaluated).
    std::vector<LevelSetVolumeConstrainedCheckValue> final_checks{};
    std::string diagnostic{};
};

/**
 * Minimize a piecewise-smooth discrete energy at a prescribed volume.
 *
 * Roundoff-balanced fourth-order differences are evaluated only inside the
 * current cut-topology and constrained-trace epoch. With exact derivatives,
 * an explicit option permits strictly merit-decreasing transitions into a
 * bounded number of new epochs. Each transition is reproduced before it is
 * accepted and discards all secant history. Each SQP-like step uses a
 * safeguarded limited-memory tangent inverse Hessian, satisfies the linearized
 * volume constraint, and descends an l1 volume-merit function. Convergence
 * additionally requires the problem's acceptance certificates, measured by
 * the evaluator, in their declared order (for a static cap: the constrained
 * pressure-space and production-residual gates of the unprojected
 * physical-potential load, and a constant-pressure KKT gate when the
 * evaluator declares it required).
 *
 * `accepted_coefficients` is assigned only after every convergence gate
 * passes. It is left byte-for-byte unchanged on invalid input, evaluator
 * failure, disallowed topology change, line-search failure, or
 * nonconvergence.
 */
[[nodiscard]] LevelSetVolumeConstrainedMinimizerResult
minimizeLevelSetVolumeConstrained(
    const LevelSetVolumeConstrainedMinimizerOptions& options,
    const LevelSetVolumeConstrainedProblem& problem,
    std::span<const Real> input_coefficients,
    std::span<const std::size_t> active_coefficient_indices,
    const LevelSetVolumeConstrainedEvaluator& evaluator,
    std::vector<Real>& accepted_coefficients);

} // namespace svmp::FE::level_set
