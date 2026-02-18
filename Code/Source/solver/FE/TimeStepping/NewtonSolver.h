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

#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

struct NewtonOptions {
    systems::OperatorTag residual_op{"residual"};
    systems::OperatorTag jacobian_op{"jacobian"};

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

    // When true, the Newton solver accepts approximate linear solutions even when
    // the linear solver doesn't converge to tolerance. This matches the legacy
    // solver's inexact Newton behavior and avoids unnecessary PTC activation.
    bool accept_inexact_linear_solutions{true};

    int max_iterations{25};
    double abs_tolerance{1e-10};
    double rel_tolerance{1e-8};
    double step_tolerance{0.0};

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
};

struct NewtonReport {
    bool converged{false};
    int iterations{0};
    double residual_norm0{0.0};
    double residual_norm{0.0};
    backends::SolverReport linear{};
};

struct NewtonWorkspace {
    std::unique_ptr<backends::GenericMatrix> jacobian{};
    std::unique_ptr<backends::GenericVector> residual{};
    std::unique_ptr<backends::GenericVector> delta{};
    std::unique_ptr<backends::GenericVector> u_backup{};
    std::unique_ptr<backends::GenericVector> residual_scratch{};
    std::unique_ptr<backends::GenericVector> residual_base{};
    std::unique_ptr<backends::GenericVector> ptc_mass_lumped{};
    std::vector<GlobalIndex> dt_field_dofs{};

    [[nodiscard]] bool isAllocated() const noexcept
    {
        return jacobian != nullptr && residual != nullptr && delta != nullptr &&
               u_backup != nullptr && residual_scratch != nullptr && residual_base != nullptr;
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
    [[nodiscard]] systems::SystemStateView makeStateView(const TimeHistory& history, double solve_time) const;
    NewtonOptions options_;
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_NEWTON_SOLVER_H
