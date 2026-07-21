/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "DenseLinearAlgebra.h"

#include "Core/FEException.h"

#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
#include <Eigen/Dense>
#endif

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <utility>

#define DENSE_LINALG_CHECK(condition, message) \
    FE_THROW_IF(!(condition), FEException, message)

namespace svmp {
namespace FE {
namespace math {

namespace {

constexpr std::size_t kDenseSolveRhsBlock = 32u;

std::vector<Real> jacobi_singular_values(
    std::span<const Real> matrix,
    std::size_t rows,
    std::size_t cols,
    std::string_view label) {
    const std::size_t work_rows = std::max(rows, cols);
    const std::size_t work_cols = std::min(rows, cols);
    std::vector<Real> work(work_rows * work_cols, Real(0));
    const Real input_scale = dense_matrix_max_abs(matrix);
    if (!(input_scale > Real(0))) {
        return std::vector<Real>(work_cols, Real(0));
    }

    if (rows >= cols) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < cols; ++column) {
                work[row * work_cols + column] =
                    matrix[row * cols + column] / input_scale;
            }
        }
    } else {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < cols; ++column) {
                work[column * work_cols + row] =
                    matrix[row * cols + column] / input_scale;
            }
        }
    }

    const long double correlation_tolerance =
        64.0L * static_cast<long double>(std::numeric_limits<Real>::epsilon()) *
        static_cast<long double>(std::max(work_rows, work_cols));
    constexpr std::size_t maximum_sweeps = 64u;
    bool converged = work_cols < 2u;
    for (std::size_t sweep = 0; sweep < maximum_sweeps && !converged; ++sweep) {
        bool rotated = false;
        for (std::size_t p = 0; p < work_cols; ++p) {
            for (std::size_t q = p + 1u; q < work_cols; ++q) {
                long double alpha = 0.0L;
                long double beta = 0.0L;
                long double gamma = 0.0L;
                for (std::size_t row = 0; row < work_rows; ++row) {
                    const long double value_p = static_cast<long double>(
                        work[row * work_cols + p]);
                    const long double value_q = static_cast<long double>(
                        work[row * work_cols + q]);
                    alpha += value_p * value_p;
                    beta += value_q * value_q;
                    gamma += value_p * value_q;
                }
                if (!(alpha > 0.0L) || !(beta > 0.0L)) {
                    continue;
                }
                const long double correlation =
                    std::abs(gamma) / std::sqrt(alpha * beta);
                if (correlation <= correlation_tolerance) {
                    continue;
                }

                const long double tau = (beta - alpha) / (2.0L * gamma);
                const long double tangent = std::copysign(
                    1.0L / (std::abs(tau) + std::hypot(1.0L, tau)), tau);
                const long double cosine = 1.0L / std::hypot(1.0L, tangent);
                const long double sine = tangent * cosine;
                for (std::size_t row = 0; row < work_rows; ++row) {
                    const long double value_p = static_cast<long double>(
                        work[row * work_cols + p]);
                    const long double value_q = static_cast<long double>(
                        work[row * work_cols + q]);
                    work[row * work_cols + p] = static_cast<Real>(
                        cosine * value_p - sine * value_q);
                    work[row * work_cols + q] = static_cast<Real>(
                        sine * value_p + cosine * value_q);
                }
                rotated = true;
            }
        }
        converged = !rotated;
    }
    DENSE_LINALG_CHECK(
        converged,
        std::string(label) +
            ": one-sided Jacobi singular-value iteration did not converge");

    std::vector<Real> singular_values(work_cols, Real(0));
    for (std::size_t column = 0; column < work_cols; ++column) {
        long double norm_squared = 0.0L;
        for (std::size_t row = 0; row < work_rows; ++row) {
            const long double value = static_cast<long double>(
                work[row * work_cols + column]);
            norm_squared += value * value;
        }
        singular_values[column] =
            input_scale * static_cast<Real>(std::sqrt(norm_squared));
    }
    std::sort(singular_values.begin(), singular_values.end(), std::greater<>());
    return singular_values;
}

void materialize_inverse_from_solver(const DenseLUSolver& solver,
                                     std::vector<Real>& inverse) {
    const std::size_t n = solver.n;
    inverse.assign(n * n, Real(0));
    for (std::size_t diag = 0; diag < n; ++diag) {
        inverse[diag * n + diag] = Real(1);
    }
    solver.solve_in_place(std::span<Real>(inverse.data(), inverse.size()), n);
}

} // namespace

Real dense_matrix_max_abs(std::span<const Real> matrix) noexcept {
    Real max_abs = Real(0);
    for (const Real value : matrix) {
        max_abs = std::max(max_abs, std::abs(value));
    }
    return max_abs;
}

Real dense_matrix_pivot_tolerance(std::size_t rows,
                                  std::size_t cols,
                                  Real max_abs,
                                  Real multiplier) noexcept {
    const Real size_scale = static_cast<Real>(std::max<std::size_t>(rows, cols));
    const Real value_scale = std::max(Real(1), max_abs);
    return multiplier * std::numeric_limits<Real>::epsilon() *
           std::max(Real(1), size_scale) * value_scale;
}

Real dense_matrix_singular_value_tolerance(std::size_t rows,
                                           std::size_t cols,
                                           Real largest_singular_value,
                                           Real multiplier) noexcept {
    const Real size_scale = static_cast<Real>(std::max<std::size_t>(rows, cols));
    return multiplier * std::numeric_limits<Real>::epsilon() *
           std::max(Real(1), size_scale) *
           std::max(Real(1), largest_singular_value);
}

Real dense_matrix_condition_fallback_threshold() noexcept {
    return Real(1.0e12);
}

Real dense_matrix_condition_error_threshold() noexcept {
    return Real(1.0e14);
}

void DenseLUSolver::solve_in_place(std::span<Real> rhs) const {
    solve_in_place(rhs, 1u);
}

void DenseLUSolver::solve_in_place(std::span<Real> rhs,
                                   std::size_t rhs_count) const {
    DENSE_LINALG_CHECK(rhs_count > 0,
                             label + ": dense solve requires at least one right-hand side");
    DENSE_LINALG_CHECK(rhs.size() == n * rhs_count,
                             label + ": dense multi-RHS solve size mismatch");
    DENSE_LINALG_CHECK(lu.size() == n * n && pivots.size() == n,
                             label + ": dense solver is not factorized");

    for (std::size_t k = 0; k < n; ++k) {
        if (pivots[k] != k) {
            for (std::size_t block = 0; block < rhs_count; block += kDenseSolveRhsBlock) {
                const std::size_t end =
                    std::min(rhs_count, block + kDenseSolveRhsBlock);
                for (std::size_t r = block; r < end; ++r) {
                    std::swap(rhs[k * rhs_count + r],
                              rhs[pivots[k] * rhs_count + r]);
                }
            }
        }
    }

    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < row; ++col) {
            const Real factor = lu[row * n + col];
            for (std::size_t block = 0; block < rhs_count; block += kDenseSolveRhsBlock) {
                const std::size_t end =
                    std::min(rhs_count, block + kDenseSolveRhsBlock);
                for (std::size_t r = block; r < end; ++r) {
                    rhs[row * rhs_count + r] -= factor * rhs[col * rhs_count + r];
                }
            }
        }
    }

    for (std::size_t rev = 0; rev < n; ++rev) {
        const std::size_t row = n - 1u - rev;
        for (std::size_t col = row + 1u; col < n; ++col) {
            const Real factor = lu[row * n + col];
            for (std::size_t block = 0; block < rhs_count; block += kDenseSolveRhsBlock) {
                const std::size_t end =
                    std::min(rhs_count, block + kDenseSolveRhsBlock);
                for (std::size_t r = block; r < end; ++r) {
                    rhs[row * rhs_count + r] -= factor * rhs[col * rhs_count + r];
                }
            }
        }
        const Real pivot = lu[row * n + row];
        DENSE_LINALG_CHECK(
            std::abs(pivot) > pivot_tolerance,
            label + ": zero pivot during dense solve");
        for (std::size_t block = 0; block < rhs_count; block += kDenseSolveRhsBlock) {
            const std::size_t end =
                std::min(rhs_count, block + kDenseSolveRhsBlock);
            for (std::size_t r = block; r < end; ++r) {
                rhs[row * rhs_count + r] /= pivot;
            }
        }
    }
}

std::vector<Real> DenseLUSolver::solve(std::span<const Real> rhs) const {
    std::vector<Real> x(rhs.begin(), rhs.end());
    solve_in_place(std::span<Real>(x.data(), x.size()));
    return x;
}

DenseMatrixDiagnostics dense_matrix_diagnostics(
    std::span<const Real> matrix,
    std::size_t rows,
    std::size_t cols,
    std::string_view label) {
    DENSE_LINALG_CHECK(matrix.size() == rows * cols,
                             std::string(label) + ": diagnostic size mismatch");
    DENSE_LINALG_CHECK(rows > 0 && cols > 0,
                             std::string(label) + ": diagnostics require a nonempty matrix");

#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    using RowMajorMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    const Eigen::Map<const RowMajorMatrix> A(matrix.data(),
                                             static_cast<Eigen::Index>(rows),
                                             static_cast<Eigen::Index>(cols));
    const Matrix dense = A;
    Eigen::JacobiSVD<Matrix> svd(dense);

    DenseMatrixDiagnostics diagnostics;
    const auto& singular_values = svd.singularValues();
    diagnostics.largest_singular_value =
        (singular_values.size() > 0) ? singular_values[0] : Real(0);
    diagnostics.tolerance =
        dense_matrix_singular_value_tolerance(rows, cols,
                                              diagnostics.largest_singular_value);

    for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
        const Real sigma = singular_values[i];
        if (sigma <= diagnostics.tolerance) {
            continue;
        }
        ++diagnostics.rank;
        diagnostics.smallest_retained_singular_value = sigma;
    }

    const std::size_t full_rank = std::min(rows, cols);
    if (diagnostics.rank == full_rank &&
        diagnostics.smallest_retained_singular_value > Real(0)) {
        diagnostics.condition_estimate =
            diagnostics.largest_singular_value /
            diagnostics.smallest_retained_singular_value;
    }
    return diagnostics;
#else
    DenseMatrixDiagnostics diagnostics;
    const auto singular_values =
        jacobi_singular_values(matrix, rows, cols, label);
    diagnostics.largest_singular_value =
        singular_values.empty() ? Real(0) : singular_values.front();
    diagnostics.tolerance =
        dense_matrix_singular_value_tolerance(
            rows, cols, diagnostics.largest_singular_value);
    for (const Real sigma : singular_values) {
        if (sigma <= diagnostics.tolerance) {
            continue;
        }
        ++diagnostics.rank;
        diagnostics.smallest_retained_singular_value = sigma;
    }
    const std::size_t full_rank = std::min(rows, cols);
    if (diagnostics.rank == full_rank &&
        diagnostics.smallest_retained_singular_value > Real(0)) {
        diagnostics.condition_estimate =
            diagnostics.largest_singular_value /
            diagnostics.smallest_retained_singular_value;
    }
    return diagnostics;
#endif
}

DenseSymmetricEigenvalueBounds dense_symmetric_eigenvalue_bounds(
    std::span<const Real> matrix,
    std::size_t n,
    std::string_view label) {
    DENSE_LINALG_CHECK(matrix.size() == n * n,
                       std::string(label) + ": eigenvalue size mismatch");
    DENSE_LINALG_CHECK(n > 0,
                       std::string(label) + ": eigenvalues require a nonempty matrix");

    const Real scale = dense_matrix_max_abs(matrix);
    const Real tolerance = Real(64) * std::numeric_limits<Real>::epsilon() *
                           static_cast<Real>(std::max<std::size_t>(n, 1u)) *
                           std::max(Real(1), scale);
    Real maximum_skew = Real(0);
    std::vector<Real> work(matrix.begin(), matrix.end());
    for (std::size_t row = 0; row < n; ++row) {
        DENSE_LINALG_CHECK(std::isfinite(work[row * n + row]),
                           std::string(label) + ": nonfinite diagonal entry");
        for (std::size_t column = row + 1u; column < n; ++column) {
            const Real upper = work[row * n + column];
            const Real lower = work[column * n + row];
            DENSE_LINALG_CHECK(std::isfinite(upper) && std::isfinite(lower),
                               std::string(label) + ": nonfinite off-diagonal entry");
            maximum_skew = std::max(maximum_skew, std::abs(upper - lower));
            const Real average = Real(0.5) * (upper + lower);
            work[row * n + column] = average;
            work[column * n + row] = average;
        }
    }
    DENSE_LINALG_CHECK(maximum_skew <= Real(4) * tolerance,
                       std::string(label) + ": matrix is not numerically symmetric");

    DenseSymmetricEigenvalueBounds result;
    result.tolerance = tolerance;
    const std::size_t maximum_sweeps =
        std::max<std::size_t>(16u, 12u * n * n);
    for (std::size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
        Real maximum_off_diagonal = Real(0);
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1u; q < n; ++q) {
                maximum_off_diagonal = std::max(
                    maximum_off_diagonal, std::abs(work[p * n + q]));
            }
        }
        result.maximum_off_diagonal = maximum_off_diagonal;
        result.sweeps = sweep;
        if (maximum_off_diagonal <= tolerance) {
            result.converged = true;
            break;
        }

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1u; q < n; ++q) {
                const Real off_diagonal = work[p * n + q];
                if (std::abs(off_diagonal) <= tolerance) {
                    continue;
                }
                const Real diagonal_p = work[p * n + p];
                const Real diagonal_q = work[q * n + q];
                const Real tau =
                    (diagonal_q - diagonal_p) / (Real(2) * off_diagonal);
                const Real tangent = std::copysign(
                    Real(1) /
                        (std::abs(tau) + std::hypot(Real(1), tau)),
                    tau);
                const Real cosine =
                    Real(1) / std::hypot(Real(1), tangent);
                const Real sine = tangent * cosine;

                work[p * n + p] = diagonal_p - tangent * off_diagonal;
                work[q * n + q] = diagonal_q + tangent * off_diagonal;
                work[p * n + q] = Real(0);
                work[q * n + p] = Real(0);
                for (std::size_t k = 0; k < n; ++k) {
                    if (k == p || k == q) {
                        continue;
                    }
                    const Real value_p = work[k * n + p];
                    const Real value_q = work[k * n + q];
                    const Real rotated_p = cosine * value_p - sine * value_q;
                    const Real rotated_q = sine * value_p + cosine * value_q;
                    work[k * n + p] = rotated_p;
                    work[p * n + k] = rotated_p;
                    work[k * n + q] = rotated_q;
                    work[q * n + k] = rotated_q;
                }
            }
        }
        result.sweeps = sweep + 1u;
    }
    DENSE_LINALG_CHECK(result.converged,
                       std::string(label) + ": cyclic Jacobi iteration did not converge");

    result.smallest_eigenvalue = work.front();
    result.largest_eigenvalue = work.front();
    for (std::size_t diagonal = 1; diagonal < n; ++diagonal) {
        const Real value = work[diagonal * n + diagonal];
        result.smallest_eigenvalue =
            std::min(result.smallest_eigenvalue, value);
        result.largest_eigenvalue =
            std::max(result.largest_eigenvalue, value);
    }
    return result;
}

DenseLUSolver factor_dense_matrix(std::vector<Real> matrix,
                                  std::size_t n,
                                  std::string_view label) {
    DENSE_LINALG_CHECK(matrix.size() == n * n,
                             std::string(label) + ": dense factorization size mismatch");

    DenseLUSolver solver;
    solver.n = n;
    solver.lu = std::move(matrix);
    solver.pivots.resize(n);
    const Real max_abs = dense_matrix_max_abs(solver.lu);
    solver.pivot_tolerance =
        dense_matrix_pivot_tolerance(n, n, max_abs);
    solver.label = std::string(label);

    Real max_pivot_abs = Real(0);
    Real min_pivot_abs = std::numeric_limits<Real>::infinity();
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot_row = col;
        Real pivot_abs = std::abs(solver.lu[col * n + col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const Real candidate = std::abs(solver.lu[row * n + col]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        DENSE_LINALG_CHECK(
            pivot_abs > solver.pivot_tolerance,
            solver.label + ": rank-deficient matrix (rank " +
                std::to_string(col) + " of " + std::to_string(n) +
                ", pivot below scale-aware tolerance " +
                std::to_string(solver.pivot_tolerance) + ")");

        solver.pivots[col] = pivot_row;
        if (pivot_row != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(solver.lu[col * n + j], solver.lu[pivot_row * n + j]);
            }
        }

        const Real pivot = solver.lu[col * n + col];
        DENSE_LINALG_CHECK(
            std::abs(pivot) > solver.pivot_tolerance,
            solver.label + ": zero pivot after row exchange");
        const Real pivot_magnitude = std::abs(pivot);
        max_pivot_abs = std::max(max_pivot_abs, pivot_magnitude);
        min_pivot_abs = std::min(min_pivot_abs, pivot_magnitude);

        for (std::size_t row = col + 1; row < n; ++row) {
            const Real factor = solver.lu[row * n + col] / pivot;
            solver.lu[row * n + col] = factor;
            for (std::size_t j = col + 1; j < n; ++j) {
                solver.lu[row * n + j] -= factor * solver.lu[col * n + j];
            }
        }
    }

    solver.diagnostics.rank = n;
    solver.diagnostics.tolerance = solver.pivot_tolerance;
    solver.diagnostics.largest_singular_value = max_abs;
    solver.diagnostics.smallest_retained_singular_value =
        std::isfinite(min_pivot_abs) ? min_pivot_abs : Real(0);
    if (solver.diagnostics.smallest_retained_singular_value > Real(0)) {
        solver.diagnostics.condition_estimate =
            max_pivot_abs / solver.diagnostics.smallest_retained_singular_value;
    }
    return solver;
}

DenseInverseResult invert_dense_matrix_with_diagnostics(
    std::vector<Real> matrix,
    std::size_t n,
    std::string_view label) {
    DENSE_LINALG_CHECK(matrix.size() == n * n,
                             std::string(label) + ": dense inverse size mismatch");
    std::vector<Real> matrix_for_lu = matrix;
    const DenseLUSolver solver =
        factor_dense_matrix(std::move(matrix_for_lu), n, label);

    DenseInverseResult result;
    result.diagnostics =
        dense_matrix_diagnostics(std::span<const Real>(matrix.data(), matrix.size()),
                                 n, n, label);

#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    if (std::isfinite(solver.diagnostics.condition_estimate) &&
        std::isfinite(result.diagnostics.condition_estimate) &&
        result.diagnostics.condition_estimate > dense_matrix_condition_fallback_threshold()) {
        using RowMajorMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
        const Eigen::Map<const RowMajorMatrix> A(matrix.data(),
                                                 static_cast<Eigen::Index>(n),
                                                 static_cast<Eigen::Index>(n));
        const Matrix dense = A;
        Eigen::JacobiSVD<Matrix> svd(dense,
                                     Eigen::ComputeFullU | Eigen::ComputeFullV);
        Matrix sigma_inverse = Matrix::Zero(static_cast<Eigen::Index>(n),
                                            static_cast<Eigen::Index>(n));
        const auto& singular_values = svd.singularValues();
        for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
            DENSE_LINALG_CHECK(
                singular_values[i] > solver.diagnostics.tolerance,
                std::string(label) + ": high-condition SVD fallback encountered a dropped singular value");
            sigma_inverse(i, i) = Real(1) / singular_values[i];
        }
        const Matrix inverse = svd.matrixV() * sigma_inverse * svd.matrixU().transpose();
        result.inverse.assign(n * n, Real(0));
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                result.inverse[row * n + col] =
                    inverse(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
            }
        }
        result.used_svd_fallback = true;
        return result;
    }
#endif

    materialize_inverse_from_solver(solver, result.inverse);
    return result;
}

void validate_dense_inverse_diagnostics(
    const DenseInverseResult& result,
    std::size_t expected_rank,
    std::string_view label,
    Real max_condition) {
    DENSE_LINALG_CHECK(
        result.diagnostics.rank == expected_rank,
        std::string(label) + ": rank-deficient matrix (rank " +
            std::to_string(result.diagnostics.rank) + " of " +
            std::to_string(expected_rank) + ")");

    if (!std::isfinite(result.diagnostics.condition_estimate)) {
        return;
    }

    DENSE_LINALG_CHECK(
        result.diagnostics.condition_estimate <= max_condition,
        std::string(label) + ": condition estimate " +
            std::to_string(result.diagnostics.condition_estimate) +
            " exceeds supported threshold " + std::to_string(max_condition));
}

std::vector<Real> invert_dense_matrix(std::vector<Real> matrix,
                                      std::size_t n,
                                      std::string_view label) {
    const DenseLUSolver solver = factor_dense_matrix(std::move(matrix), n, label);
    std::vector<Real> inverse;
    materialize_inverse_from_solver(solver, inverse);
    return inverse;
}

std::size_t dense_matrix_rank(std::vector<Real> matrix,
                              std::size_t rows,
                              std::size_t cols) {
    DENSE_LINALG_CHECK(matrix.size() == rows * cols,
                             "dense_matrix_rank: size mismatch");
    const Real tolerance =
        dense_matrix_pivot_tolerance(rows, cols, dense_matrix_max_abs(matrix));

    std::size_t rank = 0;
    std::size_t pivot_row = 0;
    for (std::size_t col = 0; col < cols && pivot_row < rows; ++col) {
        std::size_t best_row = pivot_row;
        Real best_abs = std::abs(matrix[pivot_row * cols + col]);
        for (std::size_t row = pivot_row + 1; row < rows; ++row) {
            const Real candidate = std::abs(matrix[row * cols + col]);
            if (candidate > best_abs) {
                best_abs = candidate;
                best_row = row;
            }
        }
        if (best_abs <= tolerance) {
            continue;
        }

        if (best_row != pivot_row) {
            for (std::size_t c = col; c < cols; ++c) {
                std::swap(matrix[pivot_row * cols + c], matrix[best_row * cols + c]);
            }
        }

        const Real pivot = matrix[pivot_row * cols + col];
        for (std::size_t row = pivot_row + 1; row < rows; ++row) {
            const Real factor = matrix[row * cols + col] / pivot;
            if (std::abs(factor) <= tolerance) {
                matrix[row * cols + col] = Real(0);
                continue;
            }
            matrix[row * cols + col] = Real(0);
            for (std::size_t c = col + 1; c < cols; ++c) {
                matrix[row * cols + c] -= factor * matrix[pivot_row * cols + c];
            }
        }

        ++rank;
        ++pivot_row;
    }
    return rank;
}

DensePseudoInverseResult rank_revealing_pseudo_inverse(
    std::span<const Real> matrix,
    std::size_t rows,
    std::size_t cols,
    std::string_view label) {
    DENSE_LINALG_CHECK(matrix.size() == rows * cols,
                             std::string(label) + ": pseudo-inverse size mismatch");
    DENSE_LINALG_CHECK(rows > 0 && cols > 0,
                             std::string(label) + ": pseudo-inverse requires a nonempty matrix");

#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    using RowMajorMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    const Eigen::Map<const RowMajorMatrix> A(matrix.data(),
                                             static_cast<Eigen::Index>(rows),
                                             static_cast<Eigen::Index>(cols));
    const Matrix dense = A;
    Eigen::JacobiSVD<Matrix> svd(dense, Eigen::ComputeFullU | Eigen::ComputeFullV);

    DensePseudoInverseResult result;
    result.inverse.assign(cols * rows, Real(0));

    const auto& singular_values = svd.singularValues();
    result.largest_singular_value =
        (singular_values.size() > 0) ? singular_values[0] : Real(0);
    result.tolerance =
        dense_matrix_singular_value_tolerance(rows, cols, result.largest_singular_value);

    Matrix sigma_inverse = Matrix::Zero(static_cast<Eigen::Index>(cols),
                                        static_cast<Eigen::Index>(rows));
    for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
        const Real sigma = singular_values[i];
        if (sigma <= result.tolerance) {
            continue;
        }
        sigma_inverse(i, i) = Real(1) / sigma;
        ++result.rank;
        result.smallest_retained_singular_value = sigma;
    }

    const Matrix pseudo_inverse =
        svd.matrixV() * sigma_inverse * svd.matrixU().transpose();
    for (std::size_t r = 0; r < cols; ++r) {
        for (std::size_t c = 0; c < rows; ++c) {
            result.inverse[r * rows + c] =
                pseudo_inverse(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
        }
    }
    return result;
#else
    DENSE_LINALG_CHECK(
        false,
        std::string(label) +
            ": rank-revealing pseudo-inverse requires FE_ENABLE_EIGEN");
    return {};
#endif
}

} // namespace math
} // namespace FE
} // namespace svmp

#undef DENSE_LINALG_CHECK
