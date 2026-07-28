/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_MATH_DENSELINEARALGEBRA_H
#define SVMP_FE_MATH_DENSELINEARALGEBRA_H

#include "Core/Types.h"

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace math {

// Dense solve, inverse, rank, and pseudo-inverse support for FE construction
// utilities. Matrices are row-major: matrix[row * cols + col].
[[nodiscard]] Real dense_matrix_max_abs(std::span<const Real> matrix) noexcept;

[[nodiscard]] Real dense_matrix_pivot_tolerance(std::size_t rows,
                                                std::size_t cols,
                                                Real max_abs,
                                                Real multiplier = Real(64)) noexcept;

[[nodiscard]] Real dense_matrix_singular_value_tolerance(std::size_t rows,
                                                         std::size_t cols,
                                                         Real largest_singular_value,
                                                         Real multiplier = Real(64)) noexcept;

struct DensePseudoInverseResult {
    std::vector<Real> inverse;
    std::size_t rank{0};
    Real tolerance{0};
    Real largest_singular_value{0};
    Real smallest_retained_singular_value{0};
};

struct DenseMatrixDiagnostics {
    std::size_t rank{0};
    Real tolerance{0};
    Real largest_singular_value{0};
    Real smallest_retained_singular_value{0};
    Real condition_estimate{std::numeric_limits<Real>::infinity()};
};

struct DenseSymmetricEigenvalueBounds {
    Real smallest_eigenvalue{0};
    Real largest_eigenvalue{0};
    Real maximum_off_diagonal{0};
    Real tolerance{0};
    std::size_t sweeps{0};
    bool converged{false};
};

/**
 * Diagnostics for a symmetric positive-semidefinite generalized eigenproblem
 *
 * The finite eigenvalues of
 *
 *     numerator * x = lambda * denominator * x
 *
 * are evaluated on a Gershgorin-certified positive spectral subspace of
 * `denominator`.  A null mode is accepted only when it is structurally zero
 * after diagonalization and `numerator` annihilates its computed basis vector
 * exactly in the long-double accumulator.  Ambiguous modes fail closed.  The
 * reported compatibility tolerance is diagnostic only.
 *
 * `conservative_upper_bound` is an outward-rounded Gershgorin bound of the
 * computed quotient with floating-point padding.  Callers that need the
 * padded coefficient should not use the raw Jacobi diagonal estimate.
 */
struct DensePsdGeneralizedEigenvalueBound {
    std::size_t dimension{0};
    std::size_t positive_rank{0};
    std::size_t nullity{0};
    Real denominator_scale{0};
    Real numerator_scale{0};
    Real denominator_eigenvalue_tolerance{0};
    Real nullspace_compatibility_tolerance{0};
    Real maximum_nullspace_residual{0};
    Real smallest_positive_denominator_eigenvalue{0};
    Real largest_denominator_eigenvalue{0};
    Real smallest_quotient_eigenvalue{0};
    Real largest_quotient_eigenvalue{0};
    Real conservative_upper_bound{0};
    Real quotient_maximum_off_diagonal{0};
    Real quotient_tolerance{0};
    std::size_t denominator_sweeps{0};
    std::size_t quotient_sweeps{0};
    bool denominator_converged{false};
    bool quotient_converged{false};
};

struct DenseInverseResult {
    std::vector<Real> inverse;
    DenseMatrixDiagnostics diagnostics;
    bool used_svd_fallback{false};
};

[[nodiscard]] Real dense_matrix_condition_fallback_threshold() noexcept;
[[nodiscard]] Real dense_matrix_condition_error_threshold() noexcept;

struct DenseLUSolver {
    std::size_t n{0};
    std::vector<Real> lu;
    std::vector<std::size_t> pivots;
    DenseMatrixDiagnostics diagnostics;
    Real pivot_tolerance{0};
    std::string label;

    [[nodiscard]] bool empty() const noexcept { return n == 0; }

    void solve_in_place(std::span<Real> rhs) const;
    void solve_in_place(std::span<Real> rhs, std::size_t rhs_count) const;
    [[nodiscard]] std::vector<Real> solve(std::span<const Real> rhs) const;
};

// Inverses and pseudo-inverses keep the same row-major convention for their
// returned dimensions.
[[nodiscard]] DenseMatrixDiagnostics dense_matrix_diagnostics(
    std::span<const Real> matrix,
    std::size_t rows,
    std::size_t cols,
    std::string_view label = "dense matrix");

/** Deterministic cyclic-Jacobi eigenvalue bounds for a real symmetric matrix.
 * This path is available independently of optional dense-linear-algebra
 * backends and is intended for small certification matrices. */
[[nodiscard]] DenseSymmetricEigenvalueBounds
dense_symmetric_eigenvalue_bounds(
    std::span<const Real> matrix,
    std::size_t n,
    std::string_view label = "dense symmetric matrix");

/**
 * Compute a finite upper generalized-eigenvalue bound for a PSD pencil.
 *
 * This backend-independent path is intended for small certification
 * problems with a compatible physical nullspace (for example rigid modes in
 * a symmetric-gradient energy).  It rejects indefinite or unresolved
 * matrices, inputs that are not exactly symmetric, and any computed numerator
 * action on the denominator nullspace.
 */
[[nodiscard]] DensePsdGeneralizedEigenvalueBound
dense_psd_generalized_eigenvalue_bound(
    std::span<const Real> numerator,
    std::span<const Real> denominator,
    std::size_t n,
    std::string_view label = "dense PSD generalized eigenproblem");

[[nodiscard]] DenseLUSolver factor_dense_matrix(std::vector<Real> matrix,
                                                std::size_t n,
                                                std::string_view label = "dense matrix");

[[nodiscard]] std::vector<Real> invert_dense_matrix(std::vector<Real> matrix,
                                                    std::size_t n,
                                                    std::string_view label = "dense matrix");

[[nodiscard]] DenseInverseResult invert_dense_matrix_with_diagnostics(
    std::vector<Real> matrix,
    std::size_t n,
    std::string_view label = "dense matrix");

void validate_dense_inverse_diagnostics(
    const DenseInverseResult& result,
    std::size_t expected_rank,
    std::string_view label = "dense matrix",
    Real max_condition = dense_matrix_condition_error_threshold());

[[nodiscard]] std::size_t dense_matrix_rank(std::vector<Real> matrix,
                                            std::size_t rows,
                                            std::size_t cols);

[[nodiscard]] DensePseudoInverseResult rank_revealing_pseudo_inverse(
    std::span<const Real> matrix,
    std::size_t rows,
    std::size_t cols,
    std::string_view label = "dense matrix");

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_DENSELINEARALGEBRA_H
