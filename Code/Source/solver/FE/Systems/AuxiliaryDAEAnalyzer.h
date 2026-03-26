#ifndef SVMP_FE_SYSTEMS_AUXILIARY_DAE_ANALYZER_H
#define SVMP_FE_SYSTEMS_AUXILIARY_DAE_ANALYZER_H

/**
 * @file AuxiliaryDAEAnalyzer.h
 * @brief Structural DAE analysis and derivative diagnostics for auxiliary models.
 *
 * Provides infrastructure-level analysis of auxiliary DAE models:
 * - Variable classification (differential vs algebraic)
 * - Constraint partitioning
 * - Index estimation
 * - Jacobian quality verification
 * - Row/variable scaling recommendations
 * - Structural singularity detection
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Systems/AuxiliaryStateModel.h"
#include "Systems/AuxiliaryDerivativeProvider.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Analysis results
// ---------------------------------------------------------------------------

/**
 * @brief Result of structural DAE analysis.
 */
struct DAEStructuralAnalysis {
    /// Number of differential variables.
    int n_differential{0};

    /// Number of algebraic variables.
    int n_algebraic{0};

    /// Indices of differential variables.
    std::vector<int> differential_indices{};

    /// Indices of algebraic variables.
    std::vector<int> algebraic_indices{};

    /// Constraint groups (partitions of algebraic indices).
    std::vector<std::vector<int>> constraint_groups{};

    /// Estimated differential index (-1 = unknown, 0 = pure ODE, 1+ = DAE).
    int estimated_index{-1};

    /// Whether the system appears structurally nonsingular.
    bool structurally_nonsingular{true};

    /// Diagnostic messages.
    std::vector<std::string> diagnostics{};
};

/**
 * @brief Result of Jacobian quality verification.
 */
struct JacobianQualityReport {
    /// Whether the analytic Jacobian is consistent with FD.
    bool consistent{true};

    /// Maximum absolute error between analytic and FD Jacobians.
    Real max_abs_error{0.0};

    /// Maximum relative error.
    Real max_rel_error{0.0};

    /// Row and column of the worst entry.
    int worst_row{-1};
    int worst_col{-1};

    /// Per-entry errors (row-major, n×n).
    std::vector<Real> abs_errors{};

    /// Diagnostic messages.
    std::vector<std::string> diagnostics{};
};

/**
 * @brief Scaling recommendations for a DAE system.
 */
struct DAEScalingRecommendation {
    /// Row scaling factors (one per residual row).
    std::vector<Real> row_scales{};

    /// Variable scaling factors (one per state variable).
    std::vector<Real> variable_scales{};

    /// Whether scaling was needed (all scales != 1.0).
    bool scaling_needed{false};
};

// ---------------------------------------------------------------------------
//  Analyzer
// ---------------------------------------------------------------------------

/**
 * @brief Structural DAE analyzer for auxiliary models.
 */
class AuxiliaryDAEAnalyzer {
public:
    /**
     * @brief Analyze the structural properties of a DAE model.
     *
     * Classifies variables, partitions constraints, estimates index.
     */
    [[nodiscard]] static DAEStructuralAnalysis analyze(
        const AuxiliaryStateModel& model);

    /**
     * @brief Verify Jacobian quality by comparing analytic vs FD.
     *
     * @param model    The model to check.
     * @param deriv    Derivative provider (for analytic Jacobians).
     * @param ctx      Evaluation context (state, time, etc.).
     * @param fd_eps   FD perturbation size.
     * @param tol      Tolerance for consistency check.
     */
    [[nodiscard]] static JacobianQualityReport verifyJacobian(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        const AuxiliaryLocalContext& ctx,
        Real fd_eps = 1.0e-7,
        Real tol = 1.0e-4);

    /**
     * @brief Compute scaling recommendations for a DAE system.
     *
     * Uses the Jacobian magnitude to suggest row and variable scales.
     *
     * @param model  The model.
     * @param deriv  Derivative provider.
     * @param ctx    Evaluation context.
     */
    [[nodiscard]] static DAEScalingRecommendation computeScaling(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        const AuxiliaryLocalContext& ctx);

    /**
     * @brief Check if the Jacobian appears structurally singular.
     *
     * Uses a simple row/column norm check.
     */
    [[nodiscard]] static bool isStructurallySingular(
        const AuxiliaryStateModel& model,
        const AuxiliaryDerivativeProvider& deriv,
        const AuxiliaryLocalContext& ctx);
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_DAE_ANALYZER_H
