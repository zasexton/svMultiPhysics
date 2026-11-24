/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_NEWTON_SOLVER_H
#define SVMP_FE_TIMESTEPPING_NEWTON_SOLVER_H

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Core/Types.h"
#include "Systems/TransientSystem.h"
#include "TimeStepping/TimeHistory.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

enum class JacobianCheckGeometryMode : std::uint8_t {
    FixedGeometry,
    RefreshedGeometry,
    FullGeometryPerturbation
};

struct NewtonJacobianCheckDiagnostic {
    int iteration{0};
    std::size_t sweep_index{0u};
    double step_size{0.0};
    double matrix_action_norm{0.0};
    double full_action_norm{0.0};
    double finite_difference_norm{0.0};
    double error_norm{0.0};
    double relative_error{0.0};
    JacobianCheckGeometryMode geometry_mode{JacobianCheckGeometryMode::FixedGeometry};
    std::string geometry_tangent_policy{};
    std::string geometry_result{};
    std::string component_filter{};
    std::string finite_difference_scheme{};
};

class NewtonObserver;
class NewtonAcceptanceCheck;
class NewtonInitializer;

/**
 * @brief The extensions of a Newton solve, installed by its caller.
 *
 * Each kind has its own authority over the solve.  An observer reads the
 * assembled constrained residual at the residual-assembly points, also on
 * line-search trials that are later rejected; it changes neither the
 * iterate nor any decision.  An acceptance check can veto a converged
 * candidate.  An initializer proposes a correction of named fields of the
 * current guess, which the solver applies as a transaction: it is undone
 * when the solve fails, and its consumption is recorded in the workspace's
 * initializer ledger.  One object may play several of these parts.  Every
 * rank installs the same extensions in the same order, which the solver
 * checks before any extension collective.
 */
struct NewtonExtensions {
    std::vector<std::shared_ptr<NewtonObserver>> observers{};
    std::vector<std::shared_ptr<NewtonAcceptanceCheck>> acceptance_checks{};
    std::vector<std::shared_ptr<NewtonInitializer>> initializers{};
};

struct NewtonOptions {
    /**
     * @brief Points at which residual-defining external state is synchronized.
     *
     * During a line search, AcceptedNonlinearState names the refresh needed to
     * evaluate the exact state that would be accepted; it is still a
     * speculative transaction until FESystem commits the geometric trial.
     * Callbacks at LineSearchTrialResidual and AcceptedNonlinearState must
     * therefore derive reproducible state from the supplied SystemStateView,
     * and every side effect must be reversible by RestoredNonlinearState.
     * Irreversible output, history advancement, or external commits do not
     * belong in these callbacks. In a distributed solve the callback must also
     * execute the same collective sequence on every rank of the active system
     * communicator and convert rank-local failures into a collective failure;
     * Newton cannot repair peers already blocked inside a callback collective.
     *
     * EndpointCandidateState and ProjectedEndpointCandidateState are emitted
     * by TimeLoop only after every adaptive acceptance decision, but before
     * the first irreversible time-step commit.  The first point may construct
     * state-dependent constraints; TimeLoop then applies the endpoint-time
     * constraints to the candidate before emitting the projected point.  A
     * rejected attempt or an exception before commit is paired with
     * RestoredTimeStepState and RestoredProjectedTimeStepState, derived from
     * the last accepted solution.  Endpoint callbacks therefore obey the same
     * reversible generated-state contract as nonlinear trial callbacks.
     */
    enum class StateSynchronizationPoint : std::uint8_t {
        OuterFixedPointState,
        ProjectedOuterFixedPointState,
        EndpointCandidateState,
        ProjectedEndpointCandidateState,
        AcceptedNonlinearState,
        ResidualAssembly,
        JacobianAssembly,
        JacobianAndResidualAssembly,
        LineSearchTrialResidual,
        RestoredNonlinearState,
        RestoredOuterFixedPointState,
        RestoredProjectedOuterFixedPointState,
        RestoredTimeStepState,
        RestoredProjectedTimeStepState,
        FinalResidualAssembly
    };

    systems::OperatorTag residual_op{"residual"};
    systems::OperatorTag jacobian_op{"jacobian"};

    /// The observers, acceptance checks and initializers of the solve.
    NewtonExtensions extensions{};

    /**
     * @brief Additional convergence requirement for one unknown FE field.
     *
     * The Newton solver computes an L2 norm from the field's owned residual
     * entries and reduces it on the FESystem communicator.  A criterion is
     * satisfied when either its enabled absolute or relative tolerance is met.
     * Every configured field criterion must be satisfied in addition to the
     * existing monolithic (and, when present, auxiliary-block) criteria.
     */
    struct FieldResidualCriterion {
        FieldId field{INVALID_FIELD_ID};
        double abs_tolerance{0.0};
        double rel_tolerance{0.0};
    };

    std::vector<FieldResidualCriterion> field_residual_criteria{};

    struct PseudoTransientContinuationOptions {
        bool enabled{false};

        // If true, activates PTC only after a linear-solve failure.
        // If false, uses `gamma_initial` from the first Newton iteration.
        bool activate_on_linear_failure{true};

        // Scaling applied to the lumped dt-only Jacobian diagonal (mass-like term).
        // Larger values add more diagonal dominance.
        double gamma_initial{0.0};
        double gamma_growth{10.0};
        double gamma_max{1e12};

        // If gamma falls below this, treat it as zero (exact Newton).
        double gamma_drop_tolerance{1e-14};

        // Max retries of linear.solve() per Newton iteration when PTC is active.
        int max_linear_retries{8};

        // Switched evolution relaxation update: gamma <- gamma * (||r_k|| / ||r_{k-1}||)
        bool update_from_residual_ratio{true};
    };

    PseudoTransientContinuationOptions pseudo_transient{};

    /**
     * @brief Nested fixed point for residual-defining generated state.
     *
     * Some nonlinear operators depend on state-derived data G(u), such as
     * implicit cut quadrature, projected level-set curvature, active-set
     * constraints, or an algebraic velocity-extension map.  When the assembled
     * Jacobian differentiates R(u,G) with G frozen, refreshing G inside a
     * Newton line search is inconsistent: the residual follows R(u,G(u)) but
     * the matrix omits dG/du.
     *
     * With this option enabled, `synchronize_state` is called first at
     * OuterFixedPointState to construct constraints and then at
     * ProjectedOuterFixedPointState so all remaining generated data see the
     * projected iterate and transient history. A complete inner Newton solve
     * then holds that generated state fixed. The process is
     * repeated until a newly refreshed problem satisfies the acceptance gate
     * before taking any inner Newton update.
     * Relative tolerances are deliberately disabled in the inner solves because
     * each refresh defines a new residual reference; accepting a per-inner
     * relative reduction would not prove convergence of R(u,G(u)).
     * Instead, the gate of every later refresh is fixed once per step from the
     * first fresh residual of that step: max(abs_tolerance, rel_tolerance *
     * ||R(u_0,G(u_0))||), and likewise for each field residual criterion
     * (see scaledExternalStateOuterGate). With zero relative tolerances this
     * is the absolute gate.
     *
     * Callbacks must be reproducible from the supplied SystemStateView and must
     * not commit irreversible side effects.  On failure the algebraic/history,
     * optional rate, auxiliary, and bordered states are restored before a
     * RestoredOuterFixedPointState callback rebuilds the entry generated state.
     * When the failure left a constraint rebuild incomplete
     * (FESystem::constraintRebuildIncomplete() on any rank), that callback runs
     * once more, before the constraints are refreshed, so repeating it must be
     * harmless.
     */
    struct ExternalStateFixedPointOptions {
        /**
         * @brief Safeguarded vector delta-squared relaxation.
         *
         * The first outer update uses `initial_factor`.  Later updates infer a
         * scalar factor from two consecutive raw fixed-point updates.  The
         * estimate is accepted only while the affine-constraint semantics are
         * unchanged, is bounded by the configured interval, and falls back to
         * `initial_factor` when its denominator or candidate is unsafe.  This
         * changes only the next outer iterate: convergence still requires a
         * freshly regenerated zero-update inner solve.
         */
        struct DynamicRelaxationOptions {
            bool enabled{false};
            double initial_factor{1.0};
            double minimum_factor{0.05};
            double maximum_factor{25.0};
            double denominator_relative_tolerance{1.0e-12};
            // Index (0-based) of the first outer refresh whose update is
            // relaxed. The updates of earlier refreshes are taken unchanged;
            // their raw updates only seed the first estimate, so a step that
            // converges before this refresh is bitwise identical to an
            // unrelaxed one. Zero relaxes every update.
            int start_iteration{0};
        };

        bool enabled{false};
        int max_iterations{12};
        // Outer refreshes may cross a nonsmooth generated-state epoch (for
        // example a cut-topology change when a moving interface passes a mesh
        // vertex).  Zero retains the transactional stop-and-restore behavior.
        // A positive value permits this many acknowledged restarts on the
        // newly refreshed frozen problem before stopping, and also lets the
        // initial canonicalization adopt the epoch of the entry (predicted)
        // state, which does not count against the budget: the canonical
        // entry and its rollback fingerprint are then defined in that epoch.
        // The outer iteration limit bounds the work in either case.
        int max_discontinuity_restarts{0};
        // Index (0-based) of the first outer refresh at which a persistent
        // two-state alternation of the affine-constraint semantics ends the
        // attempt on the current frozen problem, as a revisited epoch does
        // (see external_state_epoch_revisited).  The alternation is
        // persistent when the refreshed constraints of the last four
        // refreshes read A, B, A, B on every rank, with A != B on some rank:
        // a discrete generated decision (for example the root of an
        // aggregated small cut next to a vertex whose level-set value
        // crosses zero) flips with the state it was solved in, so no
        // self-consistent refresh exists and the iterates repeat.  The inner
        // solve on the refreshed problem must still converge to the usual
        // tolerances.  A step accepted before this refresh is unaffected.
        // Negative disables.
        int constraint_cycle_exit_start_iteration{-1};
        DynamicRelaxationOptions dynamic_relaxation{};
        // Refreshes whose fresh residual the contraction of the two
        // preceding fresh residuals of the attempt predicts to pass the gate
        // (the zero-update certificate pass) assemble the residual first and
        // the Jacobian only if it is needed (NewtonOptions::
        // initial_residual_first). The iterates and the acceptance decision
        // are unchanged; only the Jacobian assembly of a certificate pass is
        // saved. The environment variable
        // SVMP_DISABLE_PREDICTED_CERTIFICATE_RESIDUAL_FIRST=1 turns it off.
        bool residual_first_on_predicted_certificate{true};
    };

    ExternalStateFixedPointOptions external_state_fixed_point{};

    // When true, the Newton solver may accept an approximate linear solution
    // that did not reach tolerance, provided every rank reports finite,
    // positive-iteration residual reduction and the distributed correction is
    // finite and globally nonzero. Keep this off by default.
    bool accept_inexact_linear_solutions{false};


    int max_iterations{25};
    int min_iterations{0};
    double abs_tolerance{1e-10};
    double rel_tolerance{1e-8};
    double step_tolerance{0.0};

    // Stagnation detector threshold. When enabled, stagnation is only
    // accepted as convergence if the configured nonlinear tolerances are
    // already satisfied; otherwise the solver keeps iterating and reports the
    // true residual history. Set to 0 to disable the detector entirely.
    double stagnation_tolerance{0.99};

    bool assemble_both_when_possible{true};

    /**
     * Assemble and certify only the entry residual, then return without
     * assembling the production Jacobian or applying a nonlinear update.
     *
     * This is an output-only diagnostic path for a caller that needs the
     * observations of its extensions attached to the exact supplied state.
     * It requires `min_iterations == 0`, cannot be combined with the
     * external-state fixed point or an enabled initializer, and reports
     * nonconvergence when the entry residual tolerances are not met.
     */
    bool initial_residual_only_certificate{false};

    /**
     * Assemble the residual alone in the first iteration and the Jacobian
     * only when that residual does not satisfy the tolerances, instead of one
     * combined assembly. Both assemblies produce the same residual and
     * Jacobian as the combined one, so the iterates are unchanged; when the
     * entry state already satisfies the tolerances the Jacobian assembly is
     * saved. Ignored unless the residual and Jacobian operators are the same,
     * combined assembly is allowed, and there are no monolithic auxiliary
     * unknowns. With SVMP_RESIDUAL_FIRST_SELF_CHECK=1 the combined assembly
     * is also performed and its residual compared bit for bit (diagnostic).
     */
    bool initial_residual_first{false};

    // Modified Newton: reuse the Jacobian for multiple nonlinear iterations.
    // `1` => full Newton (assemble every iteration).
    int jacobian_rebuild_period{1};

    // Heuristic update scaling for dt(·) fields (rate-form-like behavior).
    // Scales the Newton update du on time-derivative fields by `dt_increment_scale`
    // (or, when <=0, by 1/dt1_stencil[0] when available).
    bool scale_dt_increments{false};
    double dt_increment_scale{0.0};

    // Backtracking line search on the residual norm (merit function 0.5*||r||^2).
    bool use_line_search{true};
    int line_search_max_iterations{10};
    double line_search_alpha_min{1e-6};
    double line_search_shrink{0.5};
    double line_search_c1{1e-4};
    bool line_search_fail_on_no_reduction{false};

    // Runtime Jacobian-check metadata. The check remains diagnostic-only, but
    // this classifies whether residual finite differences are assembled on a
    // fixed geometry, refreshed geometry, or a future full geometry perturbation
    // path.
    JacobianCheckGeometryMode jacobian_check_geometry_mode{
        JacobianCheckGeometryMode::FixedGeometry};
    std::string jacobian_check_geometry_tangent_policy{};
    double jacobian_check_relative_tolerance{1e-6};
    std::function<void(const NewtonJacobianCheckDiagnostic&)>
        jacobian_check_diagnostic{};

    std::function<void(const systems::SystemStateView&, StateSynchronizationPoint)>
        synchronize_state{};

    /**
     * @brief Detect a nonsmooth generated-state change after synchronization.
     *
     * The external-state fixed point invokes this hook after each complete
     * constraint/projection synchronization and before the corresponding
     * frozen inner Newton solve.  A true result stops the outer transaction
     * before that inner solve and restores its entry state.  Rank-local true
     * results are combined across the active system communicator.
     *
     * Every rank must install and invoke the same callback sequence.  The
     * callback must not publish irreversible state, and any local observation
     * it uses must be available without leaving peers blocked in a collective.
     */
    std::function<bool(StateSynchronizationPoint)>
        external_state_discontinuity{};

    /**
     * @brief Acknowledge a generated-state epoch change within an attempt.
     *
     * This hook runs on every active rank after the collective discontinuity
     * decision and before solving the newly refreshed frozen problem.  It is
     * invoked for a later-outer restart within the restart budget, and for a
     * discontinuity found by the initial canonicalization when the budget is
     * positive (epoch adoption).  It is not invoked once the budget is
     * exhausted.  A generated-state owner can use it to advance only its
     * current nonlinear-attempt epoch; committed state must remain unchanged
     * until the enclosing time-step candidate is committed.
     */
    std::function<void(StateSynchronizationPoint)>
        acknowledge_external_state_discontinuity{};

    /**
     * @brief Detect a return to an epoch already visited in this attempt.
     *
     * Invoked after a collective discontinuity decision at a later outer
     * refresh.  A true result (combined across ranks) means the outer fixed
     * point alternates across a nonsmooth switching surface, for example a
     * level-set value at a mesh vertex whose sign flips with the topology it
     * was solved in.  No self-consistent epoch exists then.  The solver
     * acknowledges the revisited epoch, solves the frozen problem on it to
     * the usual inner tolerances, and accepts that state without a further
     * refresh (NewtonReport::external_state_cycle_frozen_epoch).  Without the
     * hook the fixed point keeps refreshing until the outer iteration limit.
     */
    std::function<bool(StateSynchronizationPoint)>
        external_state_epoch_revisited{};

    /**
     * @brief Notify the owner that the attempt finishes on a frozen epoch.
     *
     * Runs on every active rank after the frozen inner solve converged and
     * before the solver returns, so the owner can accept a candidate whose
     * regenerated state differs from the epoch it was solved in.
     */
    std::function<void()> acknowledge_external_state_frozen_epoch{};

    // Set when synchronize_state can change residual-defining external state
    // (for example generated cut geometry, projected curvature, affine
    // constraints, transient MPC histories, or an extended advection
    // velocity). Newton then reassembles the residual after the prospective
    // AcceptedNonlinearState synchronization before Armijo and convergence
    // tests. Without this gate, a frozen-geometry trial norm can be mistaken
    // for the norm of the refreshed nonlinear problem.
    bool accepted_state_sync_invalidates_residual{false};
};

/// The outcome of one acceptance check on a candidate.
enum class CertificateStatus : std::uint8_t {
    Passed,
    Failed,
    Unavailable   ///< the evidence the check needs is missing
};

/// One decision of an acceptance check, with a stable id and reason.
struct CertificateDecision {
    std::string id{};
    CertificateStatus status{CertificateStatus::Unavailable};
    std::string reason{};
    double value{std::numeric_limits<double>::quiet_NaN()};
    double threshold{std::numeric_limits<double>::quiet_NaN()};
};

struct NewtonReport {
    bool converged{false};
    std::string failure_message{};
    bool external_state_discontinuity{false};
    int external_state_discontinuity_restarts{0};
    // True when the initial canonicalization adopted the epoch of the entry
    // state (see ExternalStateFixedPointOptions::max_discontinuity_restarts).
    bool external_state_initial_epoch_adopted{false};
    // True when the outer fixed point revisited an epoch and the step was
    // solved on that frozen epoch (see external_state_epoch_revisited).
    bool external_state_cycle_frozen_epoch{false};
    // True when that frozen epoch was ended by a persistent two-state
    // alternation of the affine constraints (see ExternalStateFixedPoint
    // Options::constraint_cycle_exit_start_iteration).
    bool external_state_constraint_cycle{false};
    int iterations{0};
    int outer_iterations{0};
    int inner_iterations_total{0};
    // Accepted line-search steps whose trial residual was reused because
    // the accepted reprojection left the constraints and state unchanged.
    int accepted_line_search_refresh_skips{0};
    // External-state fixed point refreshes that assembled the residual first
    // because a certificate pass was predicted, and how many of those were
    // certificate passes (no Newton update, Jacobian never assembled).
    int outer_residual_first_refreshes{0};
    int outer_residual_first_certificates{0};
    // External-state fixed point acceptance gate of this step (see
    // scaledExternalStateOuterGate). NaN/empty when no gate was derived.
    bool outer_gate_scaled{false};
    double outer_gate_reference_residual{
        std::numeric_limits<double>::quiet_NaN()};
    double outer_gate_residual_tolerance{
        std::numeric_limits<double>::quiet_NaN()};
    std::vector<double> outer_gate_field_reference_residuals{};
    std::vector<double> outer_gate_field_tolerances{};
    // Initial residual norm of each configured field criterion, in
    // NewtonOptions::field_residual_criteria order.
    std::vector<double> field_criterion_residual_norm0{};
    double outer_state_change_norm{0.0};
    bool outer_dynamic_relaxation_enabled{false};
    int outer_dynamic_relaxation_updates{0};
    int outer_dynamic_relaxation_safeguards{0};
    int outer_dynamic_relaxation_resets{0};
    double outer_dynamic_relaxation_factor{1.0};
    double outer_relaxed_state_change_norm{0.0};
    double outer_raw_contraction_ratio{
        std::numeric_limits<double>::quiet_NaN()};
    double residual_norm0{0.0};
    double residual_norm{0.0};
    double field_residual_norm0{0.0};
    double field_residual_norm{0.0};
    double auxiliary_residual_norm0{0.0};
    double auxiliary_residual_norm{0.0};
    bool component_residual_convergence{false};
    bool pressure_representability_diagnostic_sampled{false};
    bool pressure_representability_available{false};
    bool pressure_representability_converged{false};
    bool pressure_representability_breakdown{false};
    FieldId pressure_representability_pressure_field{INVALID_FIELD_ID};
    double pressure_representability_residual_norm{
        std::numeric_limits<double>::quiet_NaN()};
    double pressure_representability_relative_distance{
        std::numeric_limits<double>::quiet_NaN()};
    std::string pressure_representability_reason{"not_sampled"};
    /**
     * One-dimensional equilibrium subproblem for a physical unit pressure
     * trace.  This reports the best constant pressure jump and the residual
     * left in the assembled prescribed exterior-pressure plus physical-
     * potential virtual work.  It never replaces or projects the production
     * load.
     */
    bool constant_pressure_kkt_available{false};
    bool constant_pressure_unit_coefficients_represent_constant{false};
    bool constant_pressure_constraints_preserve_constants{false};
    std::string constant_pressure_constraint_scope{"full_field"};
    std::string constant_pressure_constraint_scope_reason{"not_attempted"};
    double constant_pressure_kkt_pressure_jump{
        std::numeric_limits<double>::quiet_NaN()};
    double constant_pressure_kkt_volume_multiplier{
        std::numeric_limits<double>::quiet_NaN()};
    double constant_pressure_kkt_direction_norm{
        std::numeric_limits<double>::quiet_NaN()};
    double constant_pressure_kkt_residual_norm{
        std::numeric_limits<double>::quiet_NaN()};
    double constant_pressure_kkt_relative_distance{
        std::numeric_limits<double>::quiet_NaN()};
    double constant_pressure_kkt_relative_orthogonality{
        std::numeric_limits<double>::quiet_NaN()};
    std::string constant_pressure_kkt_reason{"not_sampled"};
    bool constant_pressure_kkt_distance_gate_applied{false};
    bool constant_pressure_kkt_distance_gate_passed{false};
    double constant_pressure_kkt_max_relative_distance{
        std::numeric_limits<double>::quiet_NaN()};
    bool pressure_representability_distance_gate_applied{false};
    bool pressure_representability_distance_gate_passed{false};
    double pressure_representability_max_relative_distance{
        std::numeric_limits<double>::quiet_NaN()};
    bool static_compatible_pressure_initializer_requested{false};
    bool static_compatible_pressure_initializer_applied{false};
    bool static_compatible_pressure_initializer_passed{false};
    std::string static_compatible_pressure_initializer_reason{
        "not_requested"};
    backends::SolverReport linear{};
    /// The decisions of the acceptance checks at the last acceptance of
    /// the solve, in check order.
    std::vector<CertificateDecision> certificates{};
};

/// Which initializers of the options a workspace lifecycle has consumed,
/// one entry per initializer in NewtonExtensions order.  A consumed
/// initializer proposes nothing more until the ledger is restored: by the
/// solver when its transaction fails or an outer attempt is rolled back, and
/// by TimeLoop when a step attempt or a comparison solve is discarded.
struct NewtonInitializerLedger {
    std::vector<bool> consumed{};

    [[nodiscard]] bool isConsumed(std::size_t index) const noexcept
    {
        return index < consumed.size() && consumed[index];
    }
    void markConsumed(std::size_t index)
    {
        if (consumed.size() <= index) {
            consumed.resize(index + 1u, false);
        }
        consumed[index] = true;
    }

    friend bool operator==(const NewtonInitializerLedger&, const NewtonInitializerLedger&) = default;
};

/// Storage an extension keeps in a NewtonWorkspace (matrices and vectors in
/// a backend layout), created by the extension before the production
/// Jacobian (NewtonExtension::allocate).
class NewtonExtensionStorage {
public:
    virtual ~NewtonExtensionStorage() = default;
};

struct NewtonWorkspace {
    std::unique_ptr<backends::GenericMatrix> jacobian{};
    /// The storage of each extension, in NewtonSolver::extensions() order;
    /// null for an extension that keeps none.
    std::vector<std::unique_ptr<NewtonExtensionStorage>> extension_storage{};
    /// The initializers this workspace lifecycle has consumed.  Construct a
    /// new NewtonWorkspace to begin a new lifecycle.
    NewtonInitializerLedger initializer_ledger{};
    std::unique_ptr<backends::GenericVector> residual{};
    std::unique_ptr<backends::GenericVector> delta{};
    std::unique_ptr<backends::GenericVector> u_backup{};
    std::unique_ptr<backends::GenericVector> residual_scratch{};
    std::unique_ptr<backends::GenericVector> residual_base{};
    std::unique_ptr<backends::GenericVector> residual_minus{};
    std::unique_ptr<backends::GenericVector> ptc_mass_lumped{};
    /// Reusable transactional copies used while a line-search trial may
    /// install state-dependent MPC semantics. Trial residuals must see the
    /// history vectors projected with the same constraints as `u`, while every
    /// rejected alpha must restore the accepted history exactly. Optional rate
    /// state is preserved separately by TimeHistory::RateStateSnapshot.
    std::vector<std::unique_ptr<backends::GenericVector>>
        line_search_history_backup{};
    std::vector<GlobalIndex> dt_field_dofs{};
    /// Factory used for allocation; retained so solveStep can reallocate the
    /// Jacobian when the system re-augments its sparsity patterns mid-run
    /// (interface-tracking constraints changing their slave/master topology).
    /// Must outlive the workspace's use (TimeLoop::run's factory does).
    const backends::BackendFactory* factory{nullptr};
    /// Snapshot of FESystem::sparsityPatternRevision() at last (re)allocation.
    std::uint64_t sparsity_revision{0};
    /// Most recent ratio ||R_k|| / ||R_{k-1}|| of consecutive fresh residuals
    /// of the external-state fixed point at refresh index k (within an attempt
    /// epoch), observed in earlier steps. Used only to predict certificate
    /// passes, which selects the assembly order and never the iterates.
    std::vector<double> outer_fresh_residual_ratios{};

    [[nodiscard]] bool isAllocated() const noexcept
    {
        return jacobian != nullptr && residual != nullptr && delta != nullptr &&
               u_backup != nullptr && residual_scratch != nullptr &&
               residual_base != nullptr && residual_minus != nullptr;
    }
};

/**
 * @brief Acceptance gate for the external-state (generated geometry) fixed point.
 *
 * The first refreshed residual of a time step, before any Newton update,
 * defines the reference. A later refresh is accepted when its fresh residual
 * satisfies
 *
 *   ||R|| <= max(abs_tolerance, rel_tolerance * reference)
 *
 * and each field criterion i satisfies the same rule with its own absolute
 * and relative tolerances and reference. The absolute tolerance is a floor
 * that always applies. A zero or nonfinite relative tolerance or reference
 * leaves the corresponding gate at its absolute tolerance, and component
 * (auxiliary-block) convergence keeps the absolute monolithic gate.
 */
struct ExternalStateOuterGate {
    bool scaled{false};
    double reference_residual{std::numeric_limits<double>::quiet_NaN()};
    double residual_tolerance{0.0};
    std::vector<double> field_reference_residuals{};
    std::vector<double> field_tolerances{};
};

[[nodiscard]] ExternalStateOuterGate scaledExternalStateOuterGate(
    double abs_tolerance,
    double rel_tolerance,
    double reference_residual,
    bool component_residual_convergence,
    std::span<const NewtonOptions::FieldResidualCriterion> field_criteria,
    std::span<const double> field_reference_residuals);

/// The name of @p point in the solver's log lines.
[[nodiscard]] const char* stateSynchronizationPointName(
    NewtonOptions::StateSynchronizationPoint point) noexcept;

/// The system and backend an extension allocates its storage for.
struct NewtonStorageContext {
    const systems::FESystem& system;
    const backends::BackendFactory& factory;
    const systems::OperatorTag& jacobian_op;
};

/**
 * @brief What every Newton extension provides.
 *
 * An extension identifies itself by a stable id, the same on every rank.
 * Its storage is created before the production Jacobian, in increasing
 * storageOrder(), so a backend factory that caches the layout of the last
 * matrix keeps that of the production operator; after a sparsity change it
 * is refreshed before or after the production Jacobian.
 */
class NewtonExtension {
public:
    virtual ~NewtonExtension() = default;

    [[nodiscard]] virtual std::string_view id() const = 0;

    [[nodiscard]] virtual int storageOrder() const noexcept { return 0; }

    /// The storage of this extension, or null when it keeps none.
    [[nodiscard]] virtual std::unique_ptr<NewtonExtensionStorage> allocate(
        const NewtonStorageContext& context) const;

    /// Refreshes @p storage (null when none was allocated) after a sparsity
    /// change, before (@p after_production false) and after (true) the
    /// production Jacobian is refreshed.
    virtual void refreshStorage(const NewtonStorageContext& context,
                                NewtonExtensionStorage* storage,
                                bool after_production) const;

    /// Starts a frozen nonlinear solve: drops what an earlier solve kept.
    virtual void beginSolve();

    /// Throws unless every rank of the active communicator of @p system
    /// holds the same policy for this extension; called before any of its
    /// collectives.
    virtual void validateCommunicatorPolicy(const systems::FESystem& system) const;
};

/// What an observer reads at a residual-assembly point.
struct NewtonObservation {
    systems::TransientSystem& transient;
    const systems::SystemStateView& state;
    const char* phase;   ///< null for an unnamed residual assembly
    NewtonOptions::StateSynchronizationPoint sync_point;
    int iteration;
    double solve_time;
    double dt;
    /// The assembled residual with the residual addition applied and the
    /// constrained rows zeroed; observers must not change it.
    backends::GenericVector& residual;
    /// A vector of the production layout the observer may overwrite.
    backends::GenericVector& scratch;
    std::span<const GlobalIndex> constrained_dofs;
    /// The storage of the observer in the workspace, or null.
    NewtonExtensionStorage* storage;
    NewtonReport& report;
    /// Assembles @p request on the observed state into the given views and
    /// writes the solver's assembly diagnostic line under @p label.
    std::function<assembly::AssemblyResult(std::string_view label,
                                           const systems::AssemblyRequest& request,
                                           assembly::GlobalSystemView* matrix,
                                           assembly::GlobalSystemView* vector)>
        assemble;
};

class NewtonObserver : public virtual NewtonExtension {
public:
    virtual void observe(NewtonObservation& observation) = 0;
};

/// The acceptance events of a solve.  Inner convergence is assessed by the
/// solver that converged, unless an outer fixed point owns the acceptance;
/// the outer fixed point assesses a fresh zero-update candidate and a
/// candidate accepted on a frozen epoch, in that order of the loop.
enum class AcceptanceStage : std::uint8_t {
    InnerConvergence,
    FreshOuterCandidate,
    FrozenCycleAcceptance
};

/// The candidate an acceptance check assesses.
struct NewtonAcceptance {
    const systems::FESystem& system;
    AcceptanceStage stage;
    const char* phase;   ///< the convergence phase of the log lines
    int iteration;       ///< the completed inner iterations of the candidate
    std::optional<int> outer_iteration;   ///< set for an outer event
    NewtonReport& candidate;
};

class NewtonAcceptanceCheck : public virtual NewtonExtension {
public:
    /// The decisions on the candidate, called on every rank; the candidate
    /// is accepted only when every decision of every check passed.
    [[nodiscard]] virtual std::vector<CertificateDecision> assess(NewtonAcceptance& acceptance) = 0;

    /// Whether the candidate must carry an observation of the accepted
    /// state, so convergence by step tolerance assembles its residual.
    [[nodiscard]] virtual bool requiresAcceptedObservation() const = 0;
};

/// What an initializer reads: the synchronized current guess, observed at
/// the accepted-state synchronization point (the initializer names the
/// phase of its own samples), and the guess itself, which it must not change.
struct NewtonInitialization {
    NewtonObservation observation;
    backends::GenericVector& current;
};

class NewtonInitializer : public virtual NewtonExtension {
public:
    /// Whether this initializer proposes a correction (the same on every rank).
    [[nodiscard]] virtual bool enabled() const = 0;

    /// Computes and verifies a correction of named fields of the current
    /// guess; returns false, after reporting why, when there is none.
    [[nodiscard]] virtual bool propose(NewtonInitialization& initialization) = 0;

    /// Adds the proposed correction to the owned rows of @p current, on
    /// every rank of @p system; the solver then updates the ghosts and
    /// synchronizes the corrected guess.
    virtual void apply(const systems::FESystem& system, backends::GenericVector& current) = 0;

    /// Reports the applied correction, after the solver synchronized it.
    virtual void reportApplied(const systems::FESystem& system, NewtonReport& report) = 0;

    /// Reports that an earlier solve of the workspace lifecycle consumed it.
    virtual void reportConsumed(const systems::FESystem& system, NewtonReport& report) = 0;
};

/**
 * @brief Newton-Raphson driver for systems assembled through FE/Systems.
 */
class NewtonSolver {
public:
    explicit NewtonSolver(NewtonOptions options = {});

    [[nodiscard]] const NewtonOptions& options() const noexcept { return options_; }

    void allocateWorkspace(const systems::FESystem& system,
                           const backends::BackendFactory& factory,
                           NewtonWorkspace& workspace) const;

    [[nodiscard]] NewtonReport solveStep(systems::TransientSystem& transient,
                                         backends::LinearSolver& linear,
                                         double solve_time,
                                         TimeHistory& history,
                                         NewtonWorkspace& workspace,
                                         const backends::GenericVector* residual_addition = nullptr) const;

private:
    NewtonSolver(NewtonOptions options,
                 bool defer_acceptance_to_outer_loop);

    [[nodiscard]] NewtonReport solveStepFrozenExternalState(
        systems::TransientSystem& transient,
        backends::LinearSolver& linear,
        double solve_time,
        TimeHistory& history,
        NewtonWorkspace& workspace,
        const backends::GenericVector* residual_addition) const;

    [[nodiscard]] systems::SystemStateView makeStateView(const TimeHistory& history, double solve_time) const;

    /// Reallocate workspace.jacobian (and the diagnostic scratch matrix) when
    /// the system's sparsity pattern revision moved since the last allocation.
    /// Vectors are left untouched so mid-step state (u_backup etc.) survives.
    void maybeReallocateJacobianForSparsity(const systems::FESystem& system,
                                            NewtonWorkspace& workspace) const;

    /// The distinct extensions of the options, in the order their storage
    /// is kept in a workspace: observers, acceptance checks, initializers,
    /// each in order of installation.
    [[nodiscard]] std::vector<NewtonExtension*> extensions() const;

    /// The storage of @p extension in @p workspace, or null.
    [[nodiscard]] NewtonExtensionStorage* extensionStorage(
        const NewtonExtension& extension,
        NewtonWorkspace& workspace) const;

    /// Throws unless every rank installs the same extensions in the same
    /// order with the same policies, defers acceptance alike and holds the
    /// same initializer ledger; called before any extension collective.
    void validateExtensionsAcrossCommunicator(
        const systems::FESystem& system,
        const NewtonWorkspace& workspace) const;

    /// Runs every acceptance check on @p candidate and records their
    /// decisions in it; true when all passed.
    [[nodiscard]] bool assessAcceptance(const systems::FESystem& system,
                                        AcceptanceStage stage,
                                        const char* phase,
                                        int iteration,
                                        std::optional<int> outer_iteration,
                                        NewtonReport& candidate) const;

    /// Whether an acceptance check needs an observation of the accepted state.
    [[nodiscard]] bool acceptanceRequiresObservation() const;

    NewtonOptions options_;
    bool defer_acceptance_to_outer_loop_{false};
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_NEWTON_SOLVER_H
