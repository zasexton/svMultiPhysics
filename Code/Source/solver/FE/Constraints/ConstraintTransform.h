/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_CONSTRAINTTRANSFORM_H
#define SVMP_FE_CONSTRAINTS_CONSTRAINTTRANSFORM_H

/**
 * @file ConstraintTransform.h
 * @brief Strong enforcement policies and reduced operators
 *
 * ConstraintTransform provides transformation-based constraint enforcement
 * through projection operators. Instead of eliminating constraints during
 * assembly, this approach transforms the full system to a reduced system
 * of unconstrained DOFs:
 *
 *   u = P * z + c
 *
 * where:
 * - u is the full solution vector (all DOFs)
 * - z is the reduced solution vector (unconstrained DOFs only)
 * - P is the projection matrix (maps reduced to full)
 * - c is the inhomogeneity vector
 *
 * The reduced system is: P^T A P z = P^T (b - A c)
 *
 * Advantages of this approach:
 * - Smaller system to solve
 * - Exact satisfaction of constraints (no iterative penalty)
 * - Works with any solver (direct or iterative)
 * - Supports matrix-free application
 *
 * @see AffineConstraints for constraint storage
 * @see ConstraintDistributor for assembly-time enforcement
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
 * @brief Sparse row of the projection matrix P
 */
struct ProjectionRow {
    GlobalIndex full_index;                   ///< Index in full system
    std::vector<GlobalIndex> reduced_indices; ///< Indices in reduced system
    std::vector<double> weights;              ///< Coefficients
    double inhomogeneity{0.0};                ///< Constant term (c_i)

    /**
     * @brief Check if this row is an identity (unconstrained DOF)
     */
    [[nodiscard]] bool isIdentity() const {
        return reduced_indices.size() == 1 && std::abs(weights[0] - 1.0) < 1e-15;
    }
};

/**
 * @brief Statistics about the reduced system
 */
struct ReductionStats {
    GlobalIndex n_full_dofs{0};         ///< Number of DOFs in full system
    GlobalIndex n_reduced_dofs{0};      ///< Number of DOFs in reduced system
    GlobalIndex n_constrained{0};       ///< Number of constrained DOFs eliminated
    GlobalIndex projection_nnz{0};      ///< Non-zeros in projection matrix
    double reduction_ratio{1.0};        ///< n_reduced / n_full
};

/**
 * @brief Options for constraint transformation
 */
struct TransformOptions {
    double tolerance{1e-15};            ///< Tolerance for near-zero weights
    bool compute_transpose{true};       ///< Pre-compute P^T for efficiency
    bool optimize_identity_rows{true};  ///< Skip computation for unconstrained DOFs
};

/**
 * @brief Constraint transformation via projection operators
 *
 * ConstraintTransform builds and applies the projection operator P that
 * maps between full and reduced DOF spaces according to constraints.
 *
 * For unconstrained DOF i:
 *   u_i = z_{map[i]}  (identity, P_ij = delta_{ij})
 *
 * For constrained DOF i with constraint u_i = sum_k a_k u_{m_k} + b:
 *   u_i = sum_k a_k z_{map[m_k]} + b  (P_i,map[m_k] = a_k, c_i = b)
 *
 * Usage:
 * @code
 *   AffineConstraints constraints;
 *   // ... add constraints ...
 *   constraints.close();
 *
 *   ConstraintTransform transform(constraints, n_total_dofs);
 *
 *   // Build reduced system
 *   // A_reduced = P^T * A * P
 *   // b_reduced = P^T * (b - A * c)
 *
 *   // Solve reduced system
 *   solve(A_reduced, z, b_reduced);
 *
 *   // Recover full solution
 *   transform.expandSolution(z, u);
 * @endcode
 */
class ConstraintTransform {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    ConstraintTransform();

    /**
     * @brief Construct from constraints
     *
     * @param constraints Closed AffineConstraints object
     * @param n_full_dofs Total number of DOFs in full system
     * @param options Transformation options
     */
    ConstraintTransform(const AffineConstraints& constraints,
                        GlobalIndex n_full_dofs,
                        const TransformOptions& options = {});

    /**
     * @brief Destructor
     */
    ~ConstraintTransform();

    /**
     * @brief Move constructor
     */
    ConstraintTransform(ConstraintTransform&& other) noexcept;

    /**
     * @brief Move assignment
     */
    ConstraintTransform& operator=(ConstraintTransform&& other) noexcept;

    // Non-copyable (large data structures)
    ConstraintTransform(const ConstraintTransform&) = delete;
    ConstraintTransform& operator=(const ConstraintTransform&) = delete;

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * @brief Initialize from constraints
     *
     * @param constraints Closed AffineConstraints object
     * @param n_full_dofs Total number of DOFs in full system
     */
    void initialize(const AffineConstraints& constraints, GlobalIndex n_full_dofs);

    /**
     * @brief Check if initialized
     */
    [[nodiscard]] bool isInitialized() const noexcept { return initialized_; }

    // =========================================================================
    // Mappings
    // =========================================================================

    /**
     * @brief Get mapping from full to reduced indices
     *
     * @return Vector where result[full_idx] = reduced_idx, or -1 if constrained
     */
    [[nodiscard]] const std::vector<GlobalIndex>& getFullToReduced() const noexcept {
        return full_to_reduced_;
    }

    /**
     * @brief Get mapping from reduced to full indices
     *
     * @return Vector where result[reduced_idx] = full_idx
     */
    [[nodiscard]] const std::vector<GlobalIndex>& getReducedToFull() const noexcept {
        return reduced_to_full_;
    }

    /**
     * @brief Get number of DOFs in full system
     */
    [[nodiscard]] GlobalIndex numFullDofs() const noexcept { return n_full_; }

    /**
     * @brief Get number of DOFs in reduced system
     */
    [[nodiscard]] GlobalIndex numReducedDofs() const noexcept { return n_reduced_; }

    /**
     * @brief Get reduction statistics
     */
    [[nodiscard]] ReductionStats getStats() const;

    // =========================================================================
    // Projection operations: P and P^T
    // =========================================================================

    /**
     * @brief Apply P: z (reduced) -> u (full)
     *
     * Computes u = P * z + c
     *
     * @param z_reduced Reduced solution vector (size = n_reduced)
     * @param u_full Output full solution vector (size = n_full)
     */
    void applyProjection(std::span<const double> z_reduced,
                         std::span<double> u_full) const;

    /**
     * @brief Apply P^T: f (full) -> g (reduced)
     *
     * Computes g = P^T * f
     *
     * @param f_full Full vector (size = n_full)
     * @param g_reduced Output reduced vector (size = n_reduced)
     */
    void applyTranspose(std::span<const double> f_full,
                        std::span<double> g_reduced) const;

    /**
     * @brief Get inhomogeneity vector c
     *
     * @return Reference to inhomogeneity vector (size = n_full)
     */
    [[nodiscard]] const std::vector<double>& getInhomogeneity() const noexcept {
        return inhomogeneity_;
    }

    // =========================================================================
    // Solution expansion/restriction
    // =========================================================================

    /**
     * @brief Expand reduced solution to full
     *
     * Same as applyProjection but with std::vector interface.
     *
     * @param z_reduced Reduced solution
     * @return Full solution
     */
    [[nodiscard]] std::vector<double> expandSolution(
        std::span<const double> z_reduced) const;

    /**
     * @brief Restrict full vector to reduced
     *
     * Extracts unconstrained DOF values.
     *
     * @param u_full Full vector
     * @return Reduced vector
     */
    [[nodiscard]] std::vector<double> restrictVector(
        std::span<const double> u_full) const;

    // =========================================================================
    // Matrix-free operators
    // =========================================================================

    /**
     * @brief Create reduced operator from full operator
     *
     * Creates a function that applies P^T A P to a reduced vector.
     *
     * @param A_full Function that applies full operator: y = A * x
     * @return Function that applies reduced operator
     */
    [[nodiscard]] std::function<void(std::span<const double>, std::span<double>)>
    createReducedOperator(
        std::function<void(std::span<const double>, std::span<double>)> A_full) const;

    /**
     * @brief Apply reduced operator: g = P^T A P z
     *
     * @param A_full Function that applies A
     * @param z_reduced Input reduced vector
     * @param g_reduced Output reduced vector
     */
    void applyReducedOperator(
        const std::function<void(std::span<const double>, std::span<double>)>& A_full,
        std::span<const double> z_reduced,
        std::span<double> g_reduced) const;

    /**
     * @brief Compute reduced RHS: g = P^T (b - A c)
     *
     * @param A_full Function that applies A
     * @param b_full Full RHS vector
     * @param g_reduced Output reduced RHS
     */
    void computeReducedRhs(
        const std::function<void(std::span<const double>, std::span<double>)>& A_full,
        std::span<const double> b_full,
        std::span<double> g_reduced) const;

    // =========================================================================
    // CSR matrix access (for explicit reduced system construction)
    // =========================================================================

    /**
     * @brief Get projection matrix P in CSR format
     *
     * @param row_offsets Output CSR row offsets
     * @param col_indices Output CSR column indices
     * @param values Output CSR values
     */
    void getProjectionCSR(std::vector<GlobalIndex>& row_offsets,
                          std::vector<GlobalIndex>& col_indices,
                          std::vector<double>& values) const;

private:
    // State
    bool initialized_{false};
    TransformOptions options_;

    // Dimensions
    GlobalIndex n_full_{0};
    GlobalIndex n_reduced_{0};

    // Mappings
    std::vector<GlobalIndex> full_to_reduced_;   // Size n_full, -1 if constrained
    std::vector<GlobalIndex> reduced_to_full_;   // Size n_reduced

    // Projection data (CSR-like for rows)
    std::vector<ProjectionRow> projection_rows_; // Size n_full

    // Inhomogeneity vector
    std::vector<double> inhomogeneity_;          // Size n_full

    // Scratch space for operations
    mutable std::vector<double> work_full_;
    mutable std::vector<double> work_full2_;

    // Build projection from constraints
    void buildProjection(const AffineConstraints& constraints);
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_CONSTRAINTTRANSFORM_H
