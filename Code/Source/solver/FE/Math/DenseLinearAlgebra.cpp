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

struct SymmetricJacobiDecomposition {
    std::vector<Real> diagonalized{};
    std::vector<Real> eigenvectors{};
    Real tolerance{0};
    Real maximum_off_diagonal{0};
    std::size_t sweeps{0};
    bool converged{false};
};

[[nodiscard]] Real normalized_relative_tolerance(
    std::size_t n,
    Real multiplier) noexcept
{
    return multiplier * std::numeric_limits<Real>::epsilon() *
           static_cast<Real>(std::max<std::size_t>(n, 1u));
}

[[nodiscard]] Real round_nonnegative_up(
    long double value,
    std::string_view label)
{
    DENSE_LINALG_CHECK(
        value >= 0.0L &&
            value <=
                static_cast<long double>(
                    std::numeric_limits<Real>::max()) &&
            std::isfinite(value),
        std::string(label) +
            ": nonnegative value is outside the representable range");
    if (value == 0.0L) {
        return Real(0);
    }

    Real rounded = static_cast<Real>(value);
    if (static_cast<long double>(rounded) < value) {
        rounded = std::nextafter(
            rounded, std::numeric_limits<Real>::infinity());
    }
    DENSE_LINALG_CHECK(
        rounded > Real(0) && std::isfinite(rounded) &&
            static_cast<long double>(rounded) >= value,
        std::string(label) +
            ": nonnegative value cannot be rounded upward safely");
    return rounded;
}

[[nodiscard]] Real round_positive_down(
    long double value,
    std::string_view label)
{
    DENSE_LINALG_CHECK(
        value > 0.0L && std::isfinite(value),
        std::string(label) +
            ": positive value is not finite");
    Real rounded = static_cast<Real>(value);
    if (static_cast<long double>(rounded) > value) {
        rounded = std::nextafter(rounded, Real(0));
    }
    DENSE_LINALG_CHECK(
        rounded > Real(0) && std::isfinite(rounded) &&
            static_cast<long double>(rounded) <= value,
        std::string(label) +
            ": positive value is below the representable range");
    return rounded;
}

[[nodiscard]] std::vector<long double> right_multiply_basis(
    std::span<const long double> matrix,
    std::span<const Real> basis,
    std::size_t n)
{
    DENSE_LINALG_CHECK(
        matrix.size() == n * n && basis.size() == n * n,
        "dense basis multiplication size mismatch");
    std::vector<long double> product(n * n, 0.0L);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t basis_column = 0;
             basis_column < n;
             ++basis_column) {
            long double value = 0.0L;
            for (std::size_t column = 0; column < n; ++column) {
                value +=
                    matrix[row * n + column] *
                    static_cast<long double>(
                        basis[column * n + basis_column]);
            }
            product[row * n + basis_column] = value;
        }
    }
    return product;
}

[[nodiscard]] std::vector<Real> validated_symmetric_copy(
    std::span<const Real> matrix,
    std::size_t n,
    Real tolerance,
    std::string_view label)
{
    DENSE_LINALG_CHECK(
        matrix.size() == n * n,
        std::string(label) + ": symmetric matrix size mismatch");
    DENSE_LINALG_CHECK(
        n > 0,
        std::string(label) + ": symmetric matrix must be nonempty");

    Real maximum_skew = Real(0);
    std::vector<Real> work(matrix.begin(), matrix.end());
    for (std::size_t row = 0; row < n; ++row) {
        DENSE_LINALG_CHECK(
            std::isfinite(work[row * n + row]),
            std::string(label) + ": nonfinite diagonal entry");
        for (std::size_t column = row + 1u; column < n; ++column) {
            const Real upper = work[row * n + column];
            const Real lower = work[column * n + row];
            DENSE_LINALG_CHECK(
                std::isfinite(upper) && std::isfinite(lower),
                std::string(label) + ": nonfinite off-diagonal entry");
            maximum_skew =
                std::max(maximum_skew, std::abs(upper - lower));
            const Real average = Real(0.5) * (upper + lower);
            work[row * n + column] = average;
            work[column * n + row] = average;
        }
    }
    DENSE_LINALG_CHECK(
        maximum_skew <= Real(4) * tolerance,
        std::string(label) + ": matrix is not numerically symmetric");
    return work;
}

[[nodiscard]] SymmetricJacobiDecomposition diagonalize_symmetric(
    std::vector<Real> work,
    std::size_t n,
    Real tolerance,
    std::string_view label)
{
    DENSE_LINALG_CHECK(
        work.size() == n * n,
        std::string(label) + ": Jacobi matrix size mismatch");
    DENSE_LINALG_CHECK(
        n > 0,
        std::string(label) + ": Jacobi matrix must be nonempty");
    DENSE_LINALG_CHECK(
        std::isfinite(tolerance) && tolerance >= Real(0),
        std::string(label) + ": invalid Jacobi tolerance");

    SymmetricJacobiDecomposition result;
    result.tolerance = tolerance;
    result.eigenvectors.assign(n * n, Real(0));
    for (std::size_t diagonal = 0; diagonal < n; ++diagonal) {
        result.eigenvectors[diagonal * n + diagonal] = Real(1);
    }

    const std::size_t maximum_sweeps =
        std::max<std::size_t>(16u, 12u * n * n);
    for (std::size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
        Real maximum_off_diagonal = Real(0);
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1u; q < n; ++q) {
                maximum_off_diagonal = std::max(
                    maximum_off_diagonal,
                    std::abs(work[p * n + q]));
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
                const long double tau =
                    (static_cast<long double>(diagonal_q) -
                     static_cast<long double>(diagonal_p)) /
                    (2.0L * static_cast<long double>(off_diagonal));
                const long double tangent_long = std::copysign(
                    1.0L /
                        (std::abs(tau) + std::hypot(1.0L, tau)),
                    tau);
                const Real tangent =
                    static_cast<Real>(tangent_long);
                const Real cosine =
                    Real(1) / std::hypot(Real(1), tangent);
                const Real sine = tangent * cosine;

                work[p * n + p] =
                    diagonal_p - tangent * off_diagonal;
                work[q * n + q] =
                    diagonal_q + tangent * off_diagonal;
                work[p * n + q] = Real(0);
                work[q * n + p] = Real(0);
                for (std::size_t k = 0; k < n; ++k) {
                    if (k == p || k == q) {
                        continue;
                    }
                    const Real value_p = work[k * n + p];
                    const Real value_q = work[k * n + q];
                    const Real rotated_p =
                        cosine * value_p - sine * value_q;
                    const Real rotated_q =
                        sine * value_p + cosine * value_q;
                    work[k * n + p] = rotated_p;
                    work[p * n + k] = rotated_p;
                    work[k * n + q] = rotated_q;
                    work[q * n + k] = rotated_q;
                }
                for (std::size_t row = 0; row < n; ++row) {
                    const Real value_p =
                        result.eigenvectors[row * n + p];
                    const Real value_q =
                        result.eigenvectors[row * n + q];
                    result.eigenvectors[row * n + p] =
                        cosine * value_p - sine * value_q;
                    result.eigenvectors[row * n + q] =
                        sine * value_p + cosine * value_q;
                }
            }
        }
        result.sweeps = sweep + 1u;
    }
    DENSE_LINALG_CHECK(
        result.converged,
        std::string(label) +
            ": cyclic Jacobi eigendecomposition did not converge");
    result.diagonalized = std::move(work);
    return result;
}

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
    auto decomposition = diagonalize_symmetric(
        validated_symmetric_copy(matrix, n, tolerance, label),
        n,
        tolerance,
        label);

    DenseSymmetricEigenvalueBounds result;
    result.tolerance = tolerance;
    result.maximum_off_diagonal =
        decomposition.maximum_off_diagonal;
    result.sweeps = decomposition.sweeps;
    result.converged = decomposition.converged;
    result.smallest_eigenvalue =
        decomposition.diagonalized.front();
    result.largest_eigenvalue =
        decomposition.diagonalized.front();
    for (std::size_t diagonal = 1; diagonal < n; ++diagonal) {
        const Real value =
            decomposition.diagonalized[diagonal * n + diagonal];
        result.smallest_eigenvalue =
            std::min(result.smallest_eigenvalue, value);
        result.largest_eigenvalue =
            std::max(result.largest_eigenvalue, value);
    }
    return result;
}

DensePsdGeneralizedEigenvalueBound
dense_psd_generalized_eigenvalue_bound(
    std::span<const Real> numerator,
    std::span<const Real> denominator,
    std::size_t n,
    std::string_view label)
{
    const std::string label_text(label);
    DENSE_LINALG_CHECK(
        numerator.size() == n * n,
        label_text + ": numerator size mismatch");
    DENSE_LINALG_CHECK(
        denominator.size() == n * n,
        label_text + ": denominator size mismatch");
    DENSE_LINALG_CHECK(
        n > 0,
        label_text + ": generalized eigenproblem must be nonempty");
    for (const Real value : numerator) {
        DENSE_LINALG_CHECK(
            std::isfinite(value),
            label_text + ": numerator contains a nonfinite entry");
    }
    for (const Real value : denominator) {
        DENSE_LINALG_CHECK(
            std::isfinite(value),
            label_text + ": denominator contains a nonfinite entry");
    }
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = row + 1u; column < n; ++column) {
            DENSE_LINALG_CHECK(
                numerator[row * n + column] ==
                    numerator[column * n + row],
                label_text + ": numerator is not exactly symmetric");
            DENSE_LINALG_CHECK(
                denominator[row * n + column] ==
                    denominator[column * n + row],
                label_text + ": denominator is not exactly symmetric");
        }
    }

    DensePsdGeneralizedEigenvalueBound result;
    result.dimension = n;
    result.denominator_scale = dense_matrix_max_abs(denominator);
    result.numerator_scale = dense_matrix_max_abs(numerator);

    const Real symmetry_tolerance =
        normalized_relative_tolerance(n, Real(64));
    std::vector<Real> normalized_denominator(n * n, Real(0));
    if (result.denominator_scale > Real(0)) {
        for (std::size_t i = 0; i < denominator.size(); ++i) {
            normalized_denominator[i] =
                denominator[i] / result.denominator_scale;
            DENSE_LINALG_CHECK(
                denominator[i] == Real(0) ||
                    normalized_denominator[i] != Real(0),
                label_text +
                    ": denominator normalization loses a nonzero entry");
        }
    }
    normalized_denominator = validated_symmetric_copy(
        normalized_denominator,
        n,
        symmetry_tolerance,
        label_text + " denominator");

    std::vector<long double> denominator_wide(n * n, 0.0L);
    std::vector<long double> numerator_wide(n * n, 0.0L);
    for (std::size_t i = 0; i < n * n; ++i) {
        denominator_wide[i] =
            static_cast<long double>(denominator[i]);
        numerator_wide[i] =
            static_cast<long double>(numerator[i]);
    }

    auto denominator_decomposition =
        diagonalize_symmetric(
            normalized_denominator,
            n,
            symmetry_tolerance,
            label_text + " denominator");
    result.denominator_converged =
        denominator_decomposition.converged;
    result.denominator_sweeps =
        denominator_decomposition.sweeps;

    const auto denominator_times_basis =
        right_multiply_basis(
            denominator_wide,
            denominator_decomposition.eigenvectors,
            n);
    const auto numerator_times_basis =
        right_multiply_basis(
            numerator_wide,
            denominator_decomposition.eigenvectors,
            n);
    std::vector<long double> denominator_in_basis(
        n * n, 0.0L);
    long double transformed_maximum_off_diagonal = 0.0L;
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a; b < n; ++b) {
            long double projected_denominator = 0.0L;
            for (std::size_t row = 0; row < n; ++row) {
                const long double q_row =
                    static_cast<long double>(
                        denominator_decomposition
                            .eigenvectors[row * n + a]);
                if (q_row == 0.0L) {
                    continue;
                }
                projected_denominator +=
                    q_row *
                    denominator_times_basis[row * n + b];
            }
            denominator_in_basis[a * n + b] =
                projected_denominator;
            denominator_in_basis[b * n + a] =
                projected_denominator;
            if (a != b) {
                transformed_maximum_off_diagonal =
                    std::max(
                        transformed_maximum_off_diagonal,
                        std::abs(projected_denominator));
            }
        }
    }

    const long double denominator_spectral_tolerance =
        static_cast<long double>(
            normalized_relative_tolerance(n, Real(256))) *
            static_cast<long double>(result.denominator_scale) +
        static_cast<long double>(n > 0 ? n - 1u : 0u) *
            transformed_maximum_off_diagonal;
    result.denominator_eigenvalue_tolerance =
        round_nonnegative_up(
            denominator_spectral_tolerance,
            label_text + " denominator spectral tolerance");

    std::vector<std::size_t> positive_indices;
    std::vector<std::size_t> null_indices;
    std::vector<long double> safe_denominator_eigenvalues;
    positive_indices.reserve(n);
    null_indices.reserve(n);
    safe_denominator_eigenvalues.reserve(n);
    for (std::size_t index = 0; index < n; ++index) {
        const long double diagonal =
            denominator_in_basis[index * n + index];
        long double row_radius = 0.0L;
        for (std::size_t column = 0; column < n; ++column) {
            if (column == index) {
                continue;
            }
            row_radius += std::abs(
                denominator_in_basis[index * n + column]);
        }
        if (diagonal > row_radius) {
            positive_indices.push_back(index);
            safe_denominator_eigenvalues.push_back(
                diagonal - row_radius);
        } else if (diagonal == 0.0L && row_radius == 0.0L) {
            null_indices.push_back(index);
        } else {
            DENSE_LINALG_CHECK(
                false,
                label_text +
                    ": denominator has an indefinite or numerically unresolved spectral mode");
        }
    }
    result.positive_rank = positive_indices.size();
    result.nullity = null_indices.size();

    if (!positive_indices.empty()) {
        long double smallest =
            safe_denominator_eigenvalues.front();
        long double largest =
            denominator_in_basis[
                positive_indices.front() * n +
                positive_indices.front()];
        for (std::size_t a = 0; a < positive_indices.size(); ++a) {
            const auto index = positive_indices[a];
            const long double eigenvalue =
                denominator_in_basis[index * n + index];
            smallest = std::min(
                smallest, safe_denominator_eigenvalues[a]);
            largest = std::max(largest, eigenvalue);
        }
        result.smallest_positive_denominator_eigenvalue =
            round_positive_down(
                smallest,
                label_text +
                    " smallest positive denominator eigenvalue");
        result.largest_denominator_eigenvalue = round_nonnegative_up(
            largest,
            label_text + " largest denominator eigenvalue");
    }

    const Real compatibility_tolerance_normalized =
        normalized_relative_tolerance(n, Real(512));
    result.nullspace_compatibility_tolerance =
        compatibility_tolerance_normalized *
        result.numerator_scale;
    long double maximum_nullspace_residual = 0.0L;
    if (result.numerator_scale > Real(0)) {
        for (const auto null_index : null_indices) {
            for (std::size_t row = 0; row < n; ++row) {
                const long double value =
                    numerator_times_basis[row * n + null_index];
                maximum_nullspace_residual =
                    std::max(
                        maximum_nullspace_residual,
                        std::abs(value));
            }
        }
    }
    result.maximum_nullspace_residual = round_nonnegative_up(
        maximum_nullspace_residual,
        label_text + " nullspace residual");
    DENSE_LINALG_CHECK(
        maximum_nullspace_residual == 0.0L,
        label_text +
            ": numerator does not annihilate the denominator nullspace");

    if (positive_indices.empty() ||
        !(result.numerator_scale > Real(0))) {
        result.quotient_converged = true;
        return result;
    }

    const std::size_t rank = positive_indices.size();
    std::vector<long double> quotient_wide(
        rank * rank, 0.0L);
    std::vector<long double> quotient_row_upper(
        rank, 0.0L);
    for (std::size_t a = 0; a < rank; ++a) {
        const auto index_a = positive_indices[a];
        for (std::size_t b = a; b < rank; ++b) {
            const auto index_b = positive_indices[b];
            long double projected_numerator = 0.0L;
            for (std::size_t row = 0; row < n; ++row) {
                const long double q_row =
                    static_cast<long double>(
                        denominator_decomposition
                            .eigenvectors[row * n + index_a]);
                if (q_row == 0.0L) {
                    continue;
                }
                projected_numerator +=
                    q_row *
                    numerator_times_basis[row * n + index_b];
            }
            const long double denominator_factor =
                std::sqrt(
                    static_cast<long double>(
                        safe_denominator_eigenvalues[a])) *
                std::sqrt(
                    static_cast<long double>(
                        safe_denominator_eigenvalues[b]));
            DENSE_LINALG_CHECK(
                denominator_factor > 0.0L &&
                    std::isfinite(denominator_factor),
                label_text +
                    ": invalid quotient denominator factor");
            const long double value_long =
                projected_numerator / denominator_factor;
            DENSE_LINALG_CHECK(
                std::isfinite(value_long),
                label_text +
                    ": projected quotient entry is not finite");
            quotient_wide[a * rank + b] = value_long;
            quotient_wide[b * rank + a] = value_long;
            if (a == b) {
                quotient_row_upper[a] += value_long;
            } else {
                const long double magnitude =
                    std::abs(value_long);
                quotient_row_upper[a] += magnitude;
                quotient_row_upper[b] += magnitude;
            }
        }
    }

    long double quotient_scale_long = 0.0L;
    for (const long double value : quotient_wide) {
        quotient_scale_long =
            std::max(quotient_scale_long, std::abs(value));
    }
    if (!(quotient_scale_long > 0.0L)) {
        result.quotient_converged = true;
        return result;
    }

    DENSE_LINALG_CHECK(
        quotient_scale_long <=
                static_cast<long double>(
                    std::numeric_limits<Real>::max()) &&
            std::isfinite(quotient_scale_long),
        label_text + ": generalized eigenvalue scale is not finite");
    std::vector<Real> quotient_base(
        rank * rank, Real(0));
    for (std::size_t i = 0; i < quotient_wide.size(); ++i) {
        const long double normalized =
            quotient_wide[i] / quotient_scale_long;
        quotient_base[i] = static_cast<Real>(normalized);
        DENSE_LINALG_CHECK(
            quotient_wide[i] == 0.0L ||
                quotient_base[i] != Real(0),
            label_text +
                ": quotient normalization loses a nonzero entry");
    }

    const Real quotient_jacobi_tolerance =
        normalized_relative_tolerance(rank, Real(64));
    auto quotient_decomposition =
        diagonalize_symmetric(
            validated_symmetric_copy(
                quotient_base,
                rank,
                quotient_jacobi_tolerance,
                label_text + " quotient"),
            rank,
            quotient_jacobi_tolerance,
            label_text + " quotient");
    result.quotient_converged =
        quotient_decomposition.converged;
    result.quotient_sweeps =
        quotient_decomposition.sweeps;
    result.quotient_maximum_off_diagonal =
        round_nonnegative_up(
            static_cast<long double>(
                quotient_decomposition
                    .maximum_off_diagonal) *
                quotient_scale_long,
            label_text + " quotient off-diagonal residual");
    result.quotient_tolerance =
        round_nonnegative_up(
            static_cast<long double>(
                quotient_decomposition.tolerance) *
                quotient_scale_long,
            label_text + " quotient tolerance");

    Real smallest_quotient =
        quotient_decomposition.diagonalized.front();
    Real largest_quotient = smallest_quotient;
    for (std::size_t diagonal = 0; diagonal < rank; ++diagonal) {
        const Real value =
            quotient_decomposition.diagonalized[
                diagonal * rank + diagonal];
        long double row_radius = 0.0L;
        for (std::size_t column = 0; column < rank; ++column) {
            if (column == diagonal) {
                continue;
            }
            row_radius += std::abs(
                static_cast<long double>(
                    quotient_decomposition.diagonalized[
                        diagonal * rank + column]));
        }
        DENSE_LINALG_CHECK(
            static_cast<long double>(value) > row_radius ||
                (value == Real(0) && row_radius == 0.0L),
            label_text +
                ": numerator quotient has an indefinite or numerically unresolved spectral mode");
        smallest_quotient =
            std::min(smallest_quotient, value);
        largest_quotient =
            std::max(largest_quotient, value);
    }
    result.smallest_quotient_eigenvalue = round_nonnegative_up(
        static_cast<long double>(
            std::max(Real(0), smallest_quotient)) *
            quotient_scale_long,
        label_text + " smallest quotient eigenvalue");
    result.largest_quotient_eigenvalue = round_nonnegative_up(
        static_cast<long double>(
            std::max(Real(0), largest_quotient)) *
            quotient_scale_long,
        label_text + " largest quotient eigenvalue");

    const long double quotient_base_upper =
        std::max(
            0.0L,
            *std::max_element(
                quotient_row_upper.begin(),
                quotient_row_upper.end()));
    const long double quotient_base_padding =
        1024.0L *
        static_cast<long double>(
            std::numeric_limits<Real>::epsilon()) *
        static_cast<long double>(rank) *
        static_cast<long double>(rank) *
        quotient_scale_long;
    const long double conservative_upper_long =
        quotient_base_upper + quotient_base_padding;
    result.conservative_upper_bound = round_nonnegative_up(
        conservative_upper_long,
        label_text + " conservative generalized eigenvalue bound");
    DENSE_LINALG_CHECK(
        std::isfinite(result.conservative_upper_bound) &&
            result.conservative_upper_bound >=
                result.largest_quotient_eigenvalue,
        label_text +
            ": conservative generalized eigenvalue bound is invalid");
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
