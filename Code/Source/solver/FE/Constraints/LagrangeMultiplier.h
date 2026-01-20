/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_LAGRANGEMULTIPLIER_H
#define SVMP_FE_CONSTRAINTS_LAGRANGEMULTIPLIER_H

/**
 * @file LagrangeMultiplier.h
 * @brief Lagrange multiplier enforcement for constraints
 *
 * LagrangeMultiplier provides weak enforcement of constraints using Lagrange
 * multipliers. Instead of eliminating constrained DOFs, this approach adds
 * extra DOFs (the multipliers) and forms a saddle-point system:
 *
 *   [ A   B^T ] [ u ]   [ f ]
 *   [ B   0   ] [ l ] = [ g ]
 *
 * where:
 * - A is the original system matrix
 * - B encodes the constraints: B * u = g
 * - l are the Lagrange multipliers (additional unknowns)
 *
 * Advantages:
 * - Exact constraint satisfaction
 * - No modification of original system structure
 * - Multipliers have physical meaning (constraint forces)
 * - Works well for interface coupling
 *
 * Disadvantages:
 * - Larger system to solve
 * - Saddle-point structure requires special solvers
 * - Zero diagonal block can cause issues
 *
 * @see AffineConstraints for elimination-based enforcement
 * @see PenaltyMethod for penalty-based enforcement
 */

#include "AffineConstraints.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>
#include <functional>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief A single Lagrange multiplier constraint
 *
 * Represents one row of the constraint matrix B in B*u = g
 */
struct LagrangeConstraint {
    GlobalIndex multiplier_dof{-1};                ///< Index of the multiplier DOF
    std::vector<GlobalIndex> constrained_dofs;     ///< DOFs involved in constraint
    std::vector<double> coefficients;              ///< Coefficients for each DOF
    double rhs{0.0};                               ///< Right-hand side value g

    /**
     * @brief Check if constraint is valid
     */
    [[nodiscard]] bool isValid() const {
        return multiplier_dof >= 0 &&
               !constrained_dofs.empty() &&
               constrained_dofs.size() == coefficients.size();
    }
};

/**
 * @brief Options for Lagrange multiplier enforcement
 */
struct LagrangeMultiplierOptions {
    GlobalIndex multiplier_dof_offset{0};     ///< Starting DOF index for multipliers
    bool auto_assign_multiplier_dofs{true};   ///< Automatically assign multiplier DOF indices
    double stabilization_param{0.0};          ///< Stabilization parameter (0 = none)
    bool symmetric_formulation{true};         ///< Use symmetric saddle-point structure
};

/**
 * @brief Statistics about Lagrange multiplier system
 */
struct LagrangeStats {
    GlobalIndex n_constraints{0};             ///< Number of constraints
    GlobalIndex n_multiplier_dofs{0};         ///< Number of multiplier DOFs
    GlobalIndex constraint_matrix_nnz{0};     ///< Non-zeros in B matrix
};

/**
 * @brief Lagrange multiplier enforcement strategy
 *
 * LagrangeMultiplier manages the setup and assembly of saddle-point systems
 * for constraint enforcement via Lagrange multipliers.
 *
 * Usage:
 * @code
 *   // Define constraints
 *   LagrangeMultiplier lagrange;
 *
 *   // Add constraint: u_0 + u_1 - 2*u_2 = 0
 *   lagrange.addConstraint({0, 1, 2}, {1.0, 1.0, -2.0}, 0.0);
 *
 *   // Get augmented system structure
 *   // Original system has n DOFs, augmented has n + n_multipliers
 *   GlobalIndex n_multipliers = lagrange.numMultipliers();
 *
 *   // During assembly, add constraint contributions
 *   lagrange.assembleConstraintBlock(B_matrix);
 *   lagrange.assembleRhs(rhs_vector);
 * @endcode
 *
 * After solving, the Lagrange multipliers can be extracted and interpreted
 * as constraint reaction forces.
 */
class LagrangeMultiplier {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    LagrangeMultiplier();

    /**
     * @brief Construct with options
     */
    explicit LagrangeMultiplier(const LagrangeMultiplierOptions& options);

    /**
     * @brief Construct from AffineConstraints
     *
     * Converts elimination-style constraints to Lagrange multiplier form.
     *
     * @param constraints Closed AffineConstraints
     * @param options Lagrange multiplier options
     */
    LagrangeMultiplier(const AffineConstraints& constraints,
                       const LagrangeMultiplierOptions& options = {});

    /**
     * @brief Destructor
     */
    ~LagrangeMultiplier();

    /**
     * @brief Move constructor
     */
    LagrangeMultiplier(LagrangeMultiplier&& other) noexcept;

    /**
     * @brief Move assignment
     */
    LagrangeMultiplier& operator=(LagrangeMultiplier&& other) noexcept;

    // Non-copyable (large data structures)
    LagrangeMultiplier(const LagrangeMultiplier&) = delete;
    LagrangeMultiplier& operator=(const LagrangeMultiplier&) = delete;

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
     * @brief Add a constraint: sum(c_i * u_i) = g
     *
     * @param constrained_dofs DOF indices involved
     * @param coefficients Coefficients for each DOF
     * @param rhs Right-hand side value
     */
    void addConstraint(std::span<const GlobalIndex> constrained_dofs,
                       std::span<const double> coefficients,
                       double rhs = 0.0);

    /**
     * @brief Add a constraint directly
     */
    void addConstraint(const LagrangeConstraint& constraint);

    /**
     * @brief Finalize constraint setup
     *
     * Assigns multiplier DOF indices if auto_assign is enabled.
     */
    void finalize();

    /**
     * @brief Check if finalized
     */
    [[nodiscard]] bool isFinalized() const noexcept { return finalized_; }

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get number of constraints (= number of multipliers)
     */
    [[nodiscard]] GlobalIndex numConstraints() const noexcept {
        return static_cast<GlobalIndex>(constraints_.size());
    }

    /**
     * @brief Get number of multiplier DOFs
     */
    [[nodiscard]] GlobalIndex numMultipliers() const noexcept {
        return numConstraints();
    }

    /**
     * @brief Get all constraints
     */
    [[nodiscard]] const std::vector<LagrangeConstraint>& getConstraints() const noexcept {
        return constraints_;
    }

    /**
     * @brief Get multiplier DOF range
     *
     * @return Pair of (first_multiplier_dof, one_past_last)
     */
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getMultiplierDofRange() const noexcept {
        GlobalIndex start = options_.multiplier_dof_offset;
        return {start, start + numMultipliers()};
    }

    /**
     * @brief Get options
     */
    [[nodiscard]] const LagrangeMultiplierOptions& getOptions() const noexcept {
        return options_;
    }

    /**
     * @brief Get statistics
     */
    [[nodiscard]] LagrangeStats getStats() const;

    // =========================================================================
    // Assembly support
    // =========================================================================

    /**
     * @brief Get constraint matrix B in CSR format
     *
     * B is n_constraints x n_dofs, where B_ij = coefficient of u_j in constraint i
     *
     * @param row_offsets Output CSR row offsets
     * @param col_indices Output CSR column indices
     * @param values Output CSR values
     */
    void getConstraintMatrixCSR(std::vector<GlobalIndex>& row_offsets,
                                 std::vector<GlobalIndex>& col_indices,
                                 std::vector<double>& values) const;

    /**
     * @brief Get constraint RHS vector
     *
     * @return Vector of RHS values (size = n_constraints)
     */
    [[nodiscard]] std::vector<double> getConstraintRhs() const;

    /**
     * @brief Apply B^T to a vector: result = B^T * lambda
     *
     * @param lambda Multiplier vector (size = n_constraints)
     * @param result Output vector (size = n_primal_dofs)
     */
    void applyTranspose(std::span<const double> lambda,
                        std::span<double> result) const;

    /**
     * @brief Apply B to a vector: result = B * u
     *
     * @param u Primal vector (size = n_primal_dofs)
     * @param result Output vector (size = n_constraints)
     */
    void applyConstraintMatrix(std::span<const double> u,
                               std::span<double> result) const;

    /**
     * @brief Compute constraint residual: r = B*u - g
     *
     * @param u Solution vector
     * @return Residual vector (size = n_constraints)
     */
    [[nodiscard]] std::vector<double> computeResidual(std::span<const double> u) const;

    /**
     * @brief Check if constraints are satisfied
     *
     * @param u Solution vector
     * @param tolerance Tolerance for satisfaction
     * @return True if all constraints satisfied within tolerance
     */
    [[nodiscard]] bool checkSatisfaction(std::span<const double> u,
                                          double tolerance = 1e-10) const;

    // =========================================================================
    // Saddle-point system support
    // =========================================================================

    /**
     * @brief Apply the full saddle-point operator
     *
     * [ A   B^T ] [ u ]   [ A*u + B^T*lambda ]
     * [ B   -s  ] [ l ] = [ B*u - s*lambda   ]
     *
     * where s is the stabilization parameter.
     *
     * @param A_apply Function that applies A: y = A*x
     * @param u Primal unknowns
     * @param lambda Multipliers
     * @param result_u Output for primal equations
     * @param result_lambda Output for constraint equations
     */
    void applySaddlePointOperator(
        const std::function<void(std::span<const double>, std::span<double>)>& A_apply,
        std::span<const double> u,
        std::span<const double> lambda,
        std::span<double> result_u,
        std::span<double> result_lambda) const;

    /**
     * @brief Create a function that applies the full saddle-point operator
     *
     * For use with iterative solvers. The combined vector is [u; lambda].
     *
     * @param A_apply Function that applies A
     * @param n_primal_dofs Number of primal DOFs
     * @return Function that applies saddle-point operator
     */
    [[nodiscard]] std::function<void(std::span<const double>, std::span<double>)>
    createSaddlePointOperator(
        std::function<void(std::span<const double>, std::span<double>)> A_apply,
        GlobalIndex n_primal_dofs) const;

    // =========================================================================
    // Multiplier interpretation
    // =========================================================================

    /**
     * @brief Extract multipliers from combined solution
     *
     * @param combined Combined solution [u; lambda]
     * @param n_primal_dofs Number of primal DOFs
     * @return Multiplier values
     */
    [[nodiscard]] std::vector<double> extractMultipliers(
        std::span<const double> combined,
        GlobalIndex n_primal_dofs) const;

    /**
     * @brief Extract primal solution from combined
     *
     * @param combined Combined solution [u; lambda]
     * @param n_primal_dofs Number of primal DOFs
     * @return Primal solution
     */
    [[nodiscard]] std::vector<double> extractPrimal(
        std::span<const double> combined,
        GlobalIndex n_primal_dofs) const;

    /**
     * @brief Compute constraint forces from multipliers
     *
     * The constraint forces are the reactions needed to enforce constraints.
     * For constraint i: force contribution to DOF j = B_ij * lambda_i
     *
     * @param lambda Multiplier values
     * @return Force vector (size = max DOF index + 1)
     */
    [[nodiscard]] std::vector<double> computeConstraintForces(
        std::span<const double> lambda) const;

private:
    std::vector<LagrangeConstraint> constraints_;
    LagrangeMultiplierOptions options_;
    bool finalized_{false};

    // Cached for efficient operations
    GlobalIndex max_dof_index_{0};

    void assignMultiplierDofs();
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_LAGRANGEMULTIPLIER_H
