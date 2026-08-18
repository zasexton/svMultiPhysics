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
 * Exact certificate for a binary64 symmetric generalized quotient.
 *
 * The implementation interprets every finite `Real` entry as its exact
 * dyadic rational value.  It proves that the denominator is positive
 * definite, the numerator is positive semidefinite, and
 *
 *     directly_proven_upper_bound * denominator - numerator
 *
 * is positive semidefinite using fraction-free integer congruence.  No
 * tolerance, diagonal shift, or small-mode deletion participates in these
 * predicates.  The search is over finite nonnegative binary64 values, so the
 * returned coefficient is itself one of the values proved directly.
 */
struct DenseExactDyadicSpdGeneralizedUpperBound {
    bool applied{false};
    bool denominator_positive_definite_proven{false};
    bool numerator_positive_semidefinite_proven{false};
    bool upper_inequality_proven{false};
    std::size_t dimension{0};
    std::size_t denominator_rank{0};
    std::size_t numerator_rank{0};
    // False when q=0 is already a passing bound and no failing neighbor
    // exists; otherwise largest_failing_lower_bound was tested exactly.
    bool failing_lower_bound_proven{false};
    Real largest_failing_lower_bound{0};
    Real directly_proven_upper_bound{0};
    std::size_t psd_oracle_calls{0};
    std::size_t binary64_search_steps{0};
    std::size_t exact_update_count{0};
    std::size_t maximum_integer_bits{0};
};

/**
 * Diagnostics for an explicitly supplied structural common nullspace.
 *
 * The nullspace basis is interpreted as a caller-owned mathematical
 * contract. The dense helper normalizes its columns, selects deterministic
 * coordinate anchors, and checks the full numerator and denominator actions
 * against scale-relative roundoff guards before forming a principal
 * coordinate quotient. Accepted nonzero actions are therefore diagnostics
 * for the canonical structural quotient; they do not certify a genuinely
 * incompatible raw pencil.
 */
struct DenseExplicitNullspaceDiagnostics {
    bool applied{false};
    std::size_t supplied_nullity{0};
    std::size_t reduced_dimension{0};
    Real original_denominator_scale{0};
    Real original_numerator_scale{0};
    Real basis_rank_tolerance{0};
    Real smallest_selected_row_residual{0};
    Real denominator_action_tolerance{0};
    Real maximum_denominator_action{0};
    Real numerator_action_tolerance{0};
    Real maximum_numerator_action{0};
    bool exact_binary64_actions_proven{false};
    bool exact_binary64_anchor_rank_proven{false};
    std::vector<std::size_t> eliminated_coordinates{};
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
    DenseExplicitNullspaceDiagnostics explicit_nullspace{};
    DenseExactDyadicSpdGeneralizedUpperBound exact_dyadic{};
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
 * Prove a generalized upper bound on an already-formed SPD quotient.
 *
 * This deliberately does not discover or project a nullspace.  Callers must
 * first form a mathematically valid coordinate quotient of any common
 * kernel.  The exact backend has fixed dimension, integer-bit, modeled-work,
 * and arithmetic-update caps and fails closed when any cap is exceeded.
 */
[[nodiscard]] DenseExactDyadicSpdGeneralizedUpperBound
dense_exact_dyadic_spd_generalized_upper_bound(
    std::span<const Real> numerator,
    std::span<const Real> denominator,
    std::size_t n,
    std::string_view label =
        "exact dyadic SPD generalized quotient");

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

/**
 * Compute the PSD generalized bound after quotienting a known common kernel.
 *
 * `common_nullspace` stores an `n x explicit_nullity` row-major basis. The
 * helper deterministically selects a nonsingular set of coordinate rows,
 * removes those coordinates, and applies
 * `dense_psd_generalized_eigenvalue_bound()` to the remaining principal
 * pencil. For a true common kernel this coordinate gauge preserves every
 * finite generalized eigenvalue without introducing a rounded orthogonal
 * projection. The returned `dimension` is the original dimension and
 * `nullity` includes both supplied modes and any additional compatible modes
 * found in the reduced pencil.
 *
 * Full input validation and full-action nullspace checks occur before any
 * coordinate is removed. No diagonal shift, clipping, or small-positive-mode
 * deletion is performed.
 */
[[nodiscard]] DensePsdGeneralizedEigenvalueBound
dense_psd_generalized_eigenvalue_bound_with_explicit_nullspace(
    std::span<const Real> numerator,
    std::span<const Real> denominator,
    std::size_t n,
    std::span<const Real> common_nullspace,
    std::size_t explicit_nullity,
    std::string_view label =
        "dense PSD generalized eigenproblem with explicit nullspace");

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
