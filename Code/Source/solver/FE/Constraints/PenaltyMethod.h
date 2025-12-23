/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_PENALTYMETHOD_H
#define SVMP_FE_CONSTRAINTS_PENALTYMETHOD_H

/**
 * @file PenaltyMethod.h
 * @brief Penalty-based constraint enforcement
 *
 * PenaltyMethod provides weak enforcement of constraints using penalty terms.
 * Instead of exactly satisfying constraints, this approach adds penalty terms
 * to the variational formulation:
 *
 *   min_u  (1/2) u^T A u - u^T f + (alpha/2) ||B u - g||^2
 *
 * This leads to the modified system:
 *
 *   (A + alpha B^T B) u = f + alpha B^T g
 *
 * where alpha is the penalty parameter.
 *
 * Advantages:
 * - No additional unknowns (unlike Lagrange multipliers)
 * - Same system structure as original problem
 * - Simple implementation
 * - Works with any solver
 *
 * Disadvantages:
 * - Constraints only approximately satisfied
 * - Large alpha needed for accuracy degrades conditioning
 * - Trade-off between accuracy and conditioning
 *
 * @see AffineConstraints for elimination-based enforcement
 * @see LagrangeMultiplier for exact weak enforcement
 */

#include "AffineConstraints.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>
#include <functional>
#include <optional>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief A single penalty constraint
 */
struct PenaltyConstraint {
    std::vector<GlobalIndex> dofs;     ///< DOFs involved
    std::vector<double> coefficients;  ///< Coefficients for B row
    double rhs{0.0};                   ///< Target value g
    double penalty{1e6};               ///< Penalty parameter for this constraint

    /**
     * @brief Check if constraint is valid
     */
    [[nodiscard]] bool isValid() const {
        return !dofs.empty() && dofs.size() == coefficients.size() && penalty > 0;
    }
};

/**
 * @brief Options for penalty method
 */
struct PenaltyMethodOptions {
    double default_penalty{1e6};              ///< Default penalty parameter
    double penalty_scaling{1.0};              ///< Scale factor applied to all penalties
    bool auto_scale_penalty{false};           ///< Auto-scale based on matrix diagonal
    double condition_warning_threshold{1e12}; ///< Warn if condition number estimate exceeds this
    bool monitor_satisfaction{true};          ///< Track constraint satisfaction
};

/**
 * @brief Statistics about penalty enforcement
 */
struct PenaltyStats {
    GlobalIndex n_constraints{0};             ///< Number of penalty constraints
    double max_penalty{0.0};                  ///< Maximum penalty value
    double min_penalty{0.0};                  ///< Minimum penalty value
    double max_residual{0.0};                 ///< Maximum constraint residual
    double rms_residual{0.0};                 ///< RMS constraint residual
    double condition_estimate{0.0};           ///< Estimated condition number contribution
};

/**
 * @brief Penalty method constraint enforcement
 *
 * PenaltyMethod adds penalty terms to the system matrix and RHS to
 * approximately enforce constraints. The modified system is:
 *
 *   (A + alpha B^T B) u = f + alpha B^T g
 *
 * Usage:
 * @code
 *   PenaltyMethod penalty;
 *
 *   // Add constraint: u_0 = 1.0 with penalty 1e8
 *   penalty.addDirichletPenalty(0, 1.0, 1e8);
 *
 *   // Add constraint: u_1 - u_2 = 0 (equal DOFs)
 *   penalty.addConstraint({1, 2}, {1.0, -1.0}, 0.0);
 *
 *   // Apply to matrix and RHS
 *   penalty.applyToSystem(A, f);
 *
 *   // After solve, check satisfaction
 *   auto stats = penalty.computeStats(solution);
 * @endcode
 */
class PenaltyMethod {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    PenaltyMethod();

    /**
     * @brief Construct with options
     */
    explicit PenaltyMethod(const PenaltyMethodOptions& options);

    /**
     * @brief Construct from AffineConstraints
     *
     * @param constraints Closed AffineConstraints
     * @param options Penalty options
     */
    PenaltyMethod(const AffineConstraints& constraints,
                  const PenaltyMethodOptions& options = {});

    /**
     * @brief Destructor
     */
    ~PenaltyMethod();

    /**
     * @brief Move constructor
     */
    PenaltyMethod(PenaltyMethod&& other) noexcept;

    /**
     * @brief Move assignment
     */
    PenaltyMethod& operator=(PenaltyMethod&& other) noexcept;

    // Non-copyable
    PenaltyMethod(const PenaltyMethod&) = delete;
    PenaltyMethod& operator=(const PenaltyMethod&) = delete;

    // =========================================================================
    // Setup
    // =========================================================================

    /**
     * @brief Initialize from AffineConstraints
     *
     * @param constraints Closed AffineConstraints
     */
    void initialize(const AffineConstraints& constraints);

    /**
     * @brief Add a general constraint: sum(c_i * u_i) = g
     *
     * @param dofs DOF indices
     * @param coefficients Coefficients
     * @param rhs Target value
     * @param penalty Penalty parameter (0 = use default)
     */
    void addConstraint(std::span<const GlobalIndex> dofs,
                       std::span<const double> coefficients,
                       double rhs = 0.0,
                       double penalty = 0.0);

    /**
     * @brief Add constraint directly
     */
    void addConstraint(const PenaltyConstraint& constraint);

    /**
     * @brief Add Dirichlet-style penalty: u_dof = value
     *
     * @param dof DOF index
     * @param value Prescribed value
     * @param penalty Penalty parameter (0 = use default)
     */
    void addDirichletPenalty(GlobalIndex dof, double value, double penalty = 0.0);

    /**
     * @brief Add equality constraint: u_i = u_j
     */
    void addEqualityPenalty(GlobalIndex dof1, GlobalIndex dof2, double penalty = 0.0);

    /**
     * @brief Clear all constraints
     */
    void clear();

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get number of constraints
     */
    [[nodiscard]] GlobalIndex numConstraints() const noexcept {
        return static_cast<GlobalIndex>(constraints_.size());
    }

    /**
     * @brief Get all constraints
     */
    [[nodiscard]] const std::vector<PenaltyConstraint>& getConstraints() const noexcept {
        return constraints_;
    }

    /**
     * @brief Get options
     */
    [[nodiscard]] const PenaltyMethodOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Set options
     */
    void setOptions(const PenaltyMethodOptions& options) {
        options_ = options;
    }

    /**
     * @brief Get default penalty parameter
     */
    [[nodiscard]] double getDefaultPenalty() const noexcept {
        return options_.default_penalty * options_.penalty_scaling;
    }

    // =========================================================================
    // System modification
    // =========================================================================

    /**
     * @brief Get penalty matrix contribution B^T * diag(alpha) * B
     *
     * Returns the matrix that should be added to A.
     *
     * @param row_offsets Output CSR row offsets
     * @param col_indices Output CSR column indices
     * @param values Output CSR values
     * @param n_dofs Total number of DOFs (determines matrix size)
     */
    void getPenaltyMatrixCSR(std::vector<GlobalIndex>& row_offsets,
                              std::vector<GlobalIndex>& col_indices,
                              std::vector<double>& values,
                              GlobalIndex n_dofs) const;

    /**
     * @brief Get penalty RHS contribution: alpha * B^T * g
     *
     * @param n_dofs Total number of DOFs
     * @return RHS contribution vector
     */
    [[nodiscard]] std::vector<double> getPenaltyRhs(GlobalIndex n_dofs) const;

    /**
     * @brief Apply penalty to a matrix (add B^T * alpha * B)
     *
     * Interface for abstract matrix operations.
     *
     * @param add_entry Function to add value to matrix: add_entry(i, j, value)
     */
    void applyPenaltyToMatrix(
        const std::function<void(GlobalIndex, GlobalIndex, double)>& add_entry) const;

    /**
     * @brief Apply penalty to RHS vector
     *
     * @param rhs RHS vector (modified in place)
     */
    void applyPenaltyToRhs(std::span<double> rhs) const;

    /**
     * @brief Apply penalty modification to a matrix-free operator
     *
     * Creates a new operator that applies (A + alpha B^T B).
     *
     * @param A_apply Original operator
     * @param n_dofs Number of DOFs
     * @return Modified operator
     */
    [[nodiscard]] std::function<void(std::span<const double>, std::span<double>)>
    createPenalizedOperator(
        std::function<void(std::span<const double>, std::span<double>)> A_apply,
        GlobalIndex n_dofs) const;

    /**
     * @brief Apply penalty operator: y += alpha * B^T * B * x
     *
     * @param x Input vector
     * @param y Output vector (penalty contribution added)
     */
    void applyPenaltyOperator(std::span<const double> x, std::span<double> y) const;

    // =========================================================================
    // Monitoring and statistics
    // =========================================================================

    /**
     * @brief Compute constraint residuals
     *
     * @param solution Solution vector
     * @return Residual for each constraint (B*u - g)
     */
    [[nodiscard]] std::vector<double> computeResiduals(
        std::span<const double> solution) const;

    /**
     * @brief Check if constraints are satisfied within tolerance
     *
     * @param solution Solution vector
     * @param tolerance Tolerance for satisfaction
     * @return True if all constraints satisfied
     */
    [[nodiscard]] bool checkSatisfaction(std::span<const double> solution,
                                          double tolerance = 1e-6) const;

    /**
     * @brief Get statistics about penalty enforcement
     *
     * @param solution Solution vector (optional, for residual computation)
     * @return Penalty statistics
     */
    [[nodiscard]] PenaltyStats computeStats(
        std::optional<std::span<const double>> solution = std::nullopt) const;

    /**
     * @brief Estimate condition number contribution from penalties
     *
     * Large penalties can significantly increase condition number.
     *
     * @param matrix_diagonal_estimate Estimate of original matrix diagonal magnitude
     * @return Estimated condition number factor
     */
    [[nodiscard]] double estimateConditionContribution(
        double matrix_diagonal_estimate = 1.0) const;

    // =========================================================================
    // Penalty scaling
    // =========================================================================

    /**
     * @brief Auto-scale penalty based on matrix diagonal
     *
     * Sets penalty = scale_factor * max(diagonal)
     *
     * @param diagonal_values Matrix diagonal values
     * @param scale_factor Scaling factor (default 1e6)
     */
    void autoScalePenalty(std::span<const double> diagonal_values,
                          double scale_factor = 1e6);

    /**
     * @brief Scale all penalties by a factor
     */
    void scalePenalties(double factor);

    /**
     * @brief Set penalty for all constraints
     */
    void setUniformPenalty(double penalty);

private:
    std::vector<PenaltyConstraint> constraints_;
    PenaltyMethodOptions options_;

    // Cached
    mutable std::vector<double> work_vector_;

    double getEffectivePenalty(double specified_penalty) const;
};

// ============================================================================
// Utility functions
// ============================================================================

/**
 * @brief Compute optimal penalty parameter
 *
 * Based on typical matrix scaling, estimates a penalty that balances
 * accuracy and conditioning.
 *
 * @param stiffness_estimate Estimate of matrix stiffness (e.g., max eigenvalue)
 * @param target_accuracy Target constraint satisfaction accuracy
 * @return Recommended penalty parameter
 */
[[nodiscard]] double computeOptimalPenalty(double stiffness_estimate,
                                            double target_accuracy = 1e-6);

/**
 * @brief Adaptive penalty selection
 *
 * Given residuals from previous solve, compute adjusted penalties.
 *
 * @param current_penalties Current penalty values
 * @param residuals Constraint residuals
 * @param target_residual Target residual magnitude
 * @return Adjusted penalties
 */
[[nodiscard]] std::vector<double> adaptPenalties(
    std::span<const double> current_penalties,
    std::span<const double> residuals,
    double target_residual = 1e-8);

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_PENALTYMETHOD_H
