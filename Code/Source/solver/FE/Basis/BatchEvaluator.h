/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_BATCHEVALUATOR_H
#define SVMP_FE_BASIS_BATCHEVALUATOR_H

/**
 * @file BatchEvaluator.h
 * @brief SIMD-optimized batch evaluation of basis functions
 *
 * Provides high-performance evaluation of basis functions at multiple
 * quadrature points simultaneously. Uses SIMD intrinsics when available
 * to vectorize across quadrature points, significantly improving throughput
 * for element-level computations in FE assembly.
 */

#include "BasisFunction.h"
#include "Quadrature/QuadratureRule.h"
#include "Math/SIMD.h"
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {

/**
 * @brief Storage layout for batched basis evaluation results
 *
 * Data is stored in Structure-of-Arrays (SoA) format for optimal
 * SIMD memory access patterns. For N basis functions evaluated at
 * Q quadrature points:
 *
 * values: [N_0(q_0), N_0(q_1), ..., N_0(q_{Q-1}), N_1(q_0), ...]
 *
 * This layout enables vectorization across quadrature points.
 */
struct BatchedBasisData {
    /// Basis function values at all quadrature points
    /// Layout: [num_basis][num_quad_points]
    std::vector<Real> values;

    /// Gradients at all quadrature points
    /// Layout: [num_basis][3][num_quad_points] (interleaved by dimension)
    std::vector<Real> gradients;

    /// Hessians at all quadrature points (optional)
    /// Layout: [num_basis][9][num_quad_points]
    std::vector<Real> hessians;

    /// Number of basis functions
    std::size_t num_basis = 0;

    /// Number of quadrature points
    std::size_t num_quad_points = 0;

    /// Whether gradients are populated
    bool has_gradients = false;

    /// Whether Hessians are populated
    bool has_hessians = false;

    /**
     * @brief Access value N_i(q_j)
     */
    Real value(std::size_t i, std::size_t j) const {
        return values[i * num_quad_points + j];
    }

    /**
     * @brief Access gradient dN_i/d(xi_d) at q_j
     * @param i Basis function index
     * @param d Dimension (0, 1, or 2)
     * @param j Quadrature point index
     */
    Real gradient(std::size_t i, std::size_t d, std::size_t j) const {
        return gradients[(i * 3 + d) * num_quad_points + j];
    }

    /**
     * @brief Access Hessian d^2N_i/(d xi_d1 d xi_d2) at q_j
     * @param i Basis function index
     * @param d1 First dimension
     * @param d2 Second dimension
     * @param j Quadrature point index
     */
    Real hessian(std::size_t i, std::size_t d1, std::size_t d2, std::size_t j) const {
        return hessians[(i * 9 + d1 * 3 + d2) * num_quad_points + j];
    }

    /**
     * @brief Get pointer to contiguous values for basis function i
     */
    const Real* values_for_basis(std::size_t i) const {
        return values.data() + i * num_quad_points;
    }

    /**
     * @brief Get pointer to contiguous gradient component for basis function i
     */
    const Real* gradients_for_basis(std::size_t i, std::size_t d) const {
        return gradients.data() + (i * 3 + d) * num_quad_points;
    }
};

/**
 * @brief High-performance batch evaluator for basis functions
 *
 * Evaluates all basis functions at all quadrature points in a single
 * call, utilizing SIMD vectorization for improved throughput. The
 * results are stored in SoA format for optimal cache utilization
 * during subsequent computations (e.g., stiffness matrix assembly).
 */
class BatchEvaluator {
public:
    /**
     * @brief Create a batch evaluator for the given basis and quadrature
     * @param basis The basis function to evaluate
     * @param quad The quadrature rule providing evaluation points
     * @param compute_gradients Whether to compute gradients
     * @param compute_hessians Whether to compute Hessians
     */
    BatchEvaluator(const BasisFunction& basis,
                   const quadrature::QuadratureRule& quad,
                   bool compute_gradients = true,
                   bool compute_hessians = false);

    /**
     * @brief Get the precomputed batched data
     */
    const BatchedBasisData& data() const { return data_; }

    /**
     * @brief Number of basis functions
     */
    std::size_t num_basis() const { return data_.num_basis; }

    /**
     * @brief Number of quadrature points
     */
    std::size_t num_quad_points() const { return data_.num_quad_points; }

    /**
     * @brief Compute weighted sum of basis values at all quad points
     *
     * result[j] = sum_i coeffs[i] * N_i(q_j) * weights[j]
     *
     * This is a common operation in FE assembly. Uses SIMD when available.
     *
     * @param coeffs Coefficients for each basis function (size = num_basis)
     * @param weights Quadrature weights (size = num_quad_points)
     * @param result Output array (size = num_quad_points)
     */
    void weighted_sum(const Real* coeffs,
                      const Real* weights,
                      Real* result) const;

    /**
     * @brief Compute weighted sum of gradients at all quad points
     *
     * result[d][j] = sum_i coeffs[i] * dN_i/d(xi_d)(q_j) * weights[j]
     *
     * @param coeffs Coefficients for each basis function
     * @param weights Quadrature weights
     * @param result Output array [3][num_quad_points]
     */
    void weighted_gradient_sum(const Real* coeffs,
                               const Real* weights,
                               Real* result) const;

    /**
     * @brief Perform batched matrix-vector product for element stiffness
     *
     * For each quadrature point, computes:
     *   K_ij += w_q * dN_i . (D . dN_j)
     *
     * where D is a material tensor. Optimized for small dense matrices.
     *
     * @param D Material matrix (dimension x dimension)
     * @param weights Quadrature weights scaled by Jacobian determinant
     * @param K Output stiffness matrix (num_basis x num_basis)
     */
    void assemble_stiffness_contribution(const Real* D,
                                         const Real* weights,
                                         Real* K) const;

private:
    BatchedBasisData data_;
    int dimension_;

    // Precomputed SIMD-aligned copies for hot loops
    std::vector<Real, math::simd::AlignedAllocator<Real>> aligned_values_;
    std::vector<Real, math::simd::AlignedAllocator<Real>> aligned_gradients_;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BATCHEVALUATOR_H
