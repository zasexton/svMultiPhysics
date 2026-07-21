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
#include <string>
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
     * repeated until a newly refreshed problem satisfies the configured
     * absolute residual tolerances before taking any inner Newton update.
     * Relative tolerances are deliberately disabled in the inner solves because
     * each refresh defines a new residual reference; accepting a per-inner
     * relative reduction would not prove convergence of R(u,G(u)).
     *
     * Callbacks must be reproducible from the supplied SystemStateView and must
     * not commit irreversible side effects.  On failure the algebraic/history,
     * optional rate, auxiliary, and bordered states are restored before a
     * RestoredOuterFixedPointState callback rebuilds the entry generated state.
     */
    struct ExternalStateFixedPointOptions {
        bool enabled{false};
        int max_iterations{12};
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

    // Set when synchronize_state can change residual-defining external state
    // (for example generated cut geometry, projected curvature, affine
    // constraints, transient MPC histories, or an extended advection
    // velocity). Newton then reassembles the residual after the prospective
    // AcceptedNonlinearState synchronization before Armijo and convergence
    // tests. Without this gate, a frozen-geometry trial norm can be mistaken
    // for the norm of the refreshed nonlinear problem.
    bool accepted_state_sync_invalidates_residual{false};
};

struct NewtonReport {
    bool converged{false};
    int iterations{0};
    int outer_iterations{0};
    int inner_iterations_total{0};
    double outer_state_change_norm{0.0};
    double residual_norm0{0.0};
    double residual_norm{0.0};
    double field_residual_norm0{0.0};
    double field_residual_norm{0.0};
    double auxiliary_residual_norm0{0.0};
    double auxiliary_residual_norm{0.0};
    bool component_residual_convergence{false};
    backends::SolverReport linear{};
};

struct NewtonWorkspace {
    std::unique_ptr<backends::GenericMatrix> jacobian{};
    std::unique_ptr<backends::GenericMatrix> diagnostic_jacobian_scratch{};
    // Optional free-surface pressure-representability workspace.  The matrix
    // stores the symmetric [0,G;G^T,0] diagnostic pair; the vectors keep LSQR
    // entirely in GenericMatrix/GenericVector operations (no backend casts,
    // normal equations, or globally gathered dense matrices).
    std::unique_ptr<backends::GenericMatrix>
        pressure_representability_pair_matrix{};
    std::unique_ptr<backends::GenericVector>
        pressure_representability_load{};
    std::unique_ptr<backends::GenericVector>
        pressure_representability_solution{};
    std::unique_ptr<backends::GenericVector>
        pressure_representability_left_basis{};
    std::unique_ptr<backends::GenericVector>
        pressure_representability_right_basis{};
    std::unique_ptr<backends::GenericVector>
        pressure_representability_direction{};
    std::unique_ptr<backends::GenericVector>
        pressure_representability_work{};
    std::unique_ptr<backends::GenericVector>
        pressure_representability_residual{};
    std::unique_ptr<backends::GenericVector>
        pressure_representability_normal_residual{};
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

    [[nodiscard]] bool isAllocated() const noexcept
    {
        return jacobian != nullptr && residual != nullptr && delta != nullptr &&
               u_backup != nullptr && residual_scratch != nullptr &&
               residual_base != nullptr && residual_minus != nullptr;
    }
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

    NewtonOptions options_;
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_NEWTON_SOLVER_H
