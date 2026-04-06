#ifndef SVMP_FE_AUXILIARY_INITIALIZATION_SOLVER_H
#define SVMP_FE_AUXILIARY_INITIALIZATION_SOLVER_H

/**
 * @file AuxiliaryInitializationSolver.h
 * @brief Consistent-initialization solver for auxiliary DAE systems.
 *
 * For DAE systems with algebraic variables, the initial state must
 * satisfy the algebraic constraints.  This solver finds consistent
 * initial values for algebraic variables given fixed differential
 * variable initial conditions.
 *
 * ## Algorithm
 *
 * Newton iteration on the algebraic subsystem:
 * 1. Fix differential variables at their initial values.
 * 2. Set xdot = 0 for all variables (steady-state initialization).
 * 3. Solve g(x_diff_fixed, z) = 0 for algebraic variables z.
 *
 * ## Index-reduction hooks
 *
 * For higher-index DAEs that cannot be directly initialized, an
 * optional index-reduction callback can transform the system before
 * initialization.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateModel.h"
#include "Auxiliary/AuxiliaryDerivativeProvider.h"
#include "Auxiliary/AuxiliaryDAEAnalyzer.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Result of consistent initialization.
 */
struct InitializationResult {
    bool converged{false};
    int iterations{0};
    Real final_residual_norm{0.0};
    std::vector<std::string> diagnostics{};
};

/**
 * @brief Options for the initialization solver.
 */
struct InitializationOptions {
    int max_iterations{100};
    Real tol_abs{1.0e-12};
    Real tol_rel{1.0e-10};

    /// If true, use model's own initializeAlgebraic() if available.
    bool prefer_model_initialization{true};

    /// Optional scaling from DAEAnalyzer.
    bool use_scaling{false};
    std::vector<Real> row_scales{};
    std::vector<Real> variable_scales{};
};

/**
 * @brief Optional callback for index-reduction before initialization.
 *
 * Arguments: (model, x, analysis) → transformed x.
 * Returns true if reduction was applied.
 */
using IndexReductionHook = std::function<bool(
    const AuxiliaryStateModel& model,
    std::span<Real> x,
    const DAEStructuralAnalysis& analysis)>;

/**
 * @brief Consistent-initialization solver for auxiliary DAE systems.
 */
class AuxiliaryInitializationSolver {
public:
    /**
     * @brief Find consistent initial values for algebraic variables.
     *
     * Differential variables in `x` are held fixed.  Algebraic variables
     * are solved via Newton iteration.
     *
     * @param model   The auxiliary model.
     * @param deriv   Derivative provider (for Jacobians).
     * @param x       State vector [in/out].  Differential entries are
     *                inputs; algebraic entries are solved for.
     * @param inputs  Auxiliary input values.
     * @param params  Parameters.
     * @param time    Initialization time.
     * @param opts    Solver options.
     * @param index_reduction  Optional index-reduction hook.
     */
    [[nodiscard]] static InitializationResult solve(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        std::span<Real> x,
        std::span<const Real> inputs,
        std::span<const Real> params,
        Real time,
        const InitializationOptions& opts = {},
        IndexReductionHook index_reduction = {});
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_INITIALIZATION_SOLVER_H
