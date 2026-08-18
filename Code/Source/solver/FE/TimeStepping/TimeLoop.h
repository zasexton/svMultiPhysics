/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_TIME_LOOP_H
#define SVMP_FE_TIMESTEPPING_TIME_LOOP_H

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Systems/TransientSystem.h"
#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/StepController.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

enum class SchemeKind : std::uint8_t {
    BackwardEuler,
    BDF2,
    ThetaMethod,
    TRBDF2,
    GeneralizedAlpha,
    Newmark,
    VSVO_BDF,
    DG0,
    DG1,
    DG,
    CG1,
    CG2,
    CG
};

enum class CollocationSolveStrategy : std::uint8_t {
    // Assemble and solve the full block system (stages * n_dofs unknowns).
    Monolithic,
    // Nonlinear block Gauss–Seidel over stages (avoids the block Jacobian).
    // Supported for temporalOrder()==1 and temporalOrder()==2 collocation.
    StageGaussSeidel
};

/**
 * @brief Generalized-alpha parameters attached to an operator-stage observation.
 *
 * The state is evaluated at @f$t_{n+\alpha_f}@f$ and its first time
 * derivative at @f$t_{n+\alpha_m}@f$.  The metadata is absent for schemes
 * whose state and rate are both evaluated at the endpoint.
 */
struct GeneralizedAlphaStageMetadata {
    double alpha_f{1.0};
    double alpha_m{1.0};
    double gamma{1.0};
};

/** Rank-local mesh identity captured with a converged operator stage. */
struct CandidateStageMeshRevision {
    std::uint64_t geometry_revision{0};
    std::uint64_t topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t numbering_revision{0};
    std::uint64_t field_layout_revision{0};
    std::uint64_t label_revision{0};
    std::uint64_t active_configuration_epoch{0};
    std::uint64_t coordinate_configuration_key{0};

    friend bool operator==(const CandidateStageMeshRevision&,
                           const CandidateStageMeshRevision&) = default;
};

/**
 * @brief Immutable snapshot of a converged time-integration operator stage.
 *
 * The state and rate vectors are copied before any endpoint reconstruction,
 * generated geometry replacement, or history rotation.  The rate vector is
 * obtained by applying the exact first-derivative stencil used by the
 * converged operator; it is physically meaningful only for fields listed in
 * `time_derivative_fields`.  Algebraic-field entries are deliberately not
 * advertised as rates.  The const backend-vector pointers remain valid only
 * for the duration of the callback; consumers may read them through a const
 * local span or pass them to a `SystemStateView`-based evaluator that creates
 * the backend's global-indexed read view.
 */
struct CandidateStageObservation {
    SchemeKind scheme{SchemeKind::BackwardEuler};
    int temporal_order{0};
    int step_index{0};
    int attempt_index{0};

    double step_start_time{0.0};
    double step_end_time{0.0};
    double state_time{0.0};
    double rate_time{0.0};
    double dt{0.0};

    std::optional<GeneralizedAlphaStageMetadata> generalized_alpha{};

    std::vector<FieldId> time_derivative_fields{};
    /**
     * Rank-local mesh revisions captured before `on_nonlinear_done` runs.
     * Consumers that combine the copied state/rate with live mesh geometry
     * must require this value and prove that the live revisions still match.
     */
    std::optional<CandidateStageMeshRevision> mesh_revision{};
    const backends::GenericVector* state_vector{nullptr};
    const backends::GenericVector* rate_vector{nullptr};
};

struct TimeLoopOptions {
    double t0{0.0};
    double t_end{0.0};
    double dt{0.0};
    int max_steps{1000000};

    bool adjust_last_step{true};
    // When adjust_last_step is enabled, absorb a terminal remainder smaller
    // than this fraction of the current step by slightly enlarging the current
    // step to land exactly on t_end. A value <= 0 disables absorption.
    double last_step_absorb_fraction{0.0};

    SchemeKind scheme{SchemeKind::BackwardEuler};
    double theta{1.0};
    double trbdf2_gamma{0.5857864376269049}; // 2 - sqrt(2)

    // Generalized-α parameterization via spectral radius at infinity (ρ∞).
    // - For systems with temporalOrder()==1: Jansen–Whiting–Hulbert (first-order generalized-α).
    // - For systems with temporalOrder()==2: Chung–Hulbert generalized-α for structural dynamics.
    double generalized_alpha_rho_inf{1.0};
    // If uDot storage is missing for a first-order generalized-alpha solve,
    // estimate it from a linear system formed by selectively weighted compiled
    // time-derivative/non-time-derivative terms. This is the exact consistent-rate
    // mass solve for separable M*dt(u)+F(u) forms and a linearized startup estimate
    // for nonlinear terms that nest dt(u). Expensive or structurally singular
    // embedded-domain startup solves can disable it and use the finite-difference
    // history fallback directly.
    bool initialize_first_order_rate_from_pde{true};

    // Newmark-β family parameters (structural dynamics).
    // - For systems with temporalOrder()==2, TimeLoop uses a displacement-only Newmark-β update
    //   and requires `TimeHistory` to store velocity/acceleration (`uDot`, `uDDot`).
    // - For systems with temporalOrder()<=1, SchemeKind::Newmark is treated as an alias of
    //   Crank–Nicolson (θ=0.5).
    double newmark_beta{0.25};
    double newmark_gamma{0.5};

    // cG/dG in time (collocation equivalents for first-order systems).
    // For SchemeKind::DG: degree k => (k+1)-stage Radau IIA (order 2k+1).
    // For SchemeKind::CG: degree k => k-stage Gauss collocation (order 2k).
    int dg_degree{1};
    int cg_degree{2};
    CollocationSolveStrategy collocation_solve{CollocationSolveStrategy::Monolithic};
    int collocation_max_outer_iterations{4};
    double collocation_outer_tolerance{0.0}; // 0 disables convergence-based early exit

    NewtonOptions newton{};

    // Optional adaptive step-size controller. If null, TimeLoop uses a fixed dt
    // (with optional last-step adjustment) and throws on nonlinear failure.
    // In distributed runs every decision and failure returned by the controller
    // must be communicator-identical/collective; TimeLoop cannot reconcile
    // rank-divergent retry, rejection, or acceptance decisions.
    std::shared_ptr<StepController> step_controller{};
};

/**
 * @brief Time-loop callbacks and their distributed execution contract.
 *
 * Every rank in the active FE communicator must install the same callback
 * set. Callbacks that execute communicator operations must enter them in the
 * same order, and callbacks that affect acceptance must return the same
 * decision or fail collectively. In particular,
 * `on_step_candidate_discarded` must not throw, while
 * `on_before_step_accept` and `on_step_commit_ready` must coordinate any
 * failure before returning or throwing. TimeLoop explicitly coordinates
 * rank-local `on_candidate_stage` and `on_step_candidate_ready` exceptions
 * and decisions, but it cannot recover peers already blocked inside an
 * asymmetric user-callback collective.
 */
struct TimeLoopCallbacks {
    std::function<void(const TimeHistory&)> on_step_start{};
    /**
     * @brief Optional hook run after step-start history reset and before the physics solve.
     *
     * This is the documented moving-domain insertion point for prescribed/ALE
     * mesh motion at the accepted time level t + dt. The hook runs before
     * `TransientSystem::beginTimeStep()` and before nonlinear iteration/assembly,
     * so geometry-change notifications can invalidate FE caches before use.
     * Remeshing, checkpoint, and output remain later application responsibilities:
     * they should observe the accepted geometry after the nonlinear solve commits.
     *
     * Return false to reject the attempted step before physics assembly.
     */
    std::function<bool(TimeHistory&, double solve_time, double dt)> on_before_physics_solve{};
    std::function<void(const TimeHistory&, const NewtonReport&)> on_nonlinear_done{};
    /**
     * @brief Optional hook run after a nonlinear solve has produced a converged
     * candidate state in `TimeHistory::u()` and before adaptive acceptance or
     * time-history commit.
     *
     * Return false to reject the converged candidate as `ErrorTooLarge`.
     * Adaptive runs retry through the configured StepController; fixed-step
     * runs throw because there is no retry policy.
     */
    std::function<bool(TimeHistory&, const NewtonReport&)> on_before_step_accept{};
    /**
     * @brief Roll back state staged by @ref on_before_step_accept or
     * @ref on_candidate_stage.
     *
     * The hook runs before accepted-state regeneration whenever a prepared
     * candidate is discarded by an adaptive decision or endpoint failure.
     * It must be safe to call after a partially completed preparation hook.
     */
    std::function<void(TimeHistory&)> on_step_candidate_discarded{};
    /**
     * @brief Commit state staged by @ref on_before_step_accept or
     * @ref on_candidate_stage.
     *
     * This hook runs after every retry decision and endpoint finalization,
     * immediately before the first irreversible system acceptance operation.
     */
    std::function<void(TimeHistory&)> on_step_commit_ready{};
    std::function<void(TimeHistory&)> on_step_accepted{};

    std::function<void(const TimeHistory&, StepRejectReason, const NewtonReport&)> on_step_rejected{};
    std::function<void(double old_dt, double new_dt, int step_index, int attempt_index)> on_dt_updated{};
    /**
     * @brief Observe a converged operator stage before endpoint finalization.
     *
     * This hook is currently supported for temporal-order-one systems under
     * Backward Euler, the explicit `SchemeKind::DG0` route, and first-order
     * generalized-alpha. It runs before @ref on_before_step_accept. The
     * observer must only stage rollback-capable data: clear it from @ref
     * on_step_candidate_discarded, validate it without publishing from @ref
     * on_step_commit_ready, and publish accepted semantic history only from
     * @ref on_step_accepted after system and history acceptance complete.
     * Installing this hook automatically arms the rollback-capable candidate
     * transaction boundaries, even when @ref on_before_step_accept is unset.
     * Rank-local exceptions from this observer are coordinated by TimeLoop
     * before rollback; the discard and commit callbacks remain subject to the
     * distributed callback contract above.
     */
    std::function<void(const CandidateStageObservation&)>
        on_candidate_stage{};
    /**
     * @brief Final reversible acceptance gate for a projected endpoint.
     *
     * The hook runs after endpoint reconstruction, constraint projection, and
     * generated-state synchronization, but before auxiliary accepted-event
     * finalization, before the adaptive controller records acceptance, before
     * @ref on_step_commit_ready, and before any system/history commit. Return
     * no value to accept or a typed reason to reject. TimeLoop coordinates
     * callback presence, rank-local exceptions, and the returned decision on
     * the active FE communicator. A rejection follows the ordinary bounded
     * retry path; fixed-step runs restore the candidate and then fail.
     *
     * The callback may stage only rollback-capable state. Such state must be
     * cleared by @ref on_step_candidate_discarded and may be published only
     * after the gate accepts through the existing commit/accepted hooks.
     */
    std::function<std::optional<StepRejectReason>(
        TimeHistory&, const NewtonReport&)>
        on_step_candidate_ready{};
};

struct TimeLoopReport {
    bool success{true};
    int steps_taken{0};
    double final_time{0.0};
    std::string message{};
};

class TimeLoop {
public:
    explicit TimeLoop(TimeLoopOptions options);

    [[nodiscard]] const TimeLoopOptions& options() const noexcept { return options_; }

    [[nodiscard]] TimeLoopReport run(systems::TransientSystem& transient,
                                     const backends::BackendFactory& factory,
                                     backends::LinearSolver& linear,
                                     TimeHistory& history,
                                     const TimeLoopCallbacks& callbacks = {}) const;

private:
    TimeLoopOptions options_;
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_TIME_LOOP_H
