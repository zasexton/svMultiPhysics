#ifndef SVMP_FE_MATH_LU_H
#define SVMP_FE_MATH_LU_H

/**
 * @file LU.h
 * @brief LU decomposition and direct solvers for small linear systems
 *
 * This header provides efficient solvers for small linear systems commonly
 * encountered in FE element assembly. Includes analytical solutions for 2x2
 * and 3x3 systems (using Cramer's rule), and Gauss elimination with partial
 * pivoting for 4x4 and larger systems.
 */

#include "Vector.h"
#include "Matrix.h"
#include "MathConstants.h"
#include "MathUtils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace svmp {
namespace FE {
namespace math {

/**
 * @brief Solve 2x2 linear system using Cramer's rule (analytical)
 * @tparam T Floating-point type
 * @param A 2x2 coefficient matrix
 * @param b Right-hand side vector
 * @return Solution vector x such that Ax = b
 * @throw std::runtime_error if matrix is singular
 */
template<typename T>
inline Vector<T, 2> solve_2x2(const Matrix<T, 2, 2>& A, const Vector<T, 2>& b) {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

    // Compute determinant
    T det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

    // Check for singularity with very small threshold (allow near-singular matrices)
    // Use machine epsilon directly to only reject truly singular matrices
    using std::abs;
    if (abs(det) < std::numeric_limits<T>::epsilon()) {
        throw std::runtime_error("solve_2x2: Matrix is singular");
    }

    // Cramer's rule
    Vector<T, 2> x;
    x[0] = (b[0] * A(1, 1) - b[1] * A(0, 1)) / det;
    x[1] = (A(0, 0) * b[1] - A(1, 0) * b[0]) / det;

    return x;
}

/**
 * @brief Solve 3x3 linear system using Cramer's rule (analytical)
 * @tparam T Floating-point type
 * @param A 3x3 coefficient matrix
 * @param b Right-hand side vector
 * @return Solution vector x such that Ax = b
 * @throw std::runtime_error if matrix is singular
 */
template<typename T>
inline Vector<T, 3> solve_3x3(const Matrix<T, 3, 3>& A, const Vector<T, 3>& b) {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

    // Compute determinant using rule of Sarrus
    T det = A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
          - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
          + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

    // Check for singularity with very small threshold (allow near-singular matrices)
    // Use machine epsilon directly to only reject truly singular matrices
    using std::abs;
    if (abs(det) < std::numeric_limits<T>::epsilon()) {
        throw std::runtime_error("solve_3x3: Matrix is singular");
    }

    // Cramer's rule - compute determinants of modified matrices
    Vector<T, 3> x;

    // x[0] = det(A0) / det(A), where A0 has first column replaced by b
    x[0] = (b[0] * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
          - A(0, 1) * (b[1] * A(2, 2) - A(1, 2) * b[2])
          + A(0, 2) * (b[1] * A(2, 1) - A(1, 1) * b[2])) / det;

    // x[1] = det(A1) / det(A), where A1 has second column replaced by b
    x[1] = (A(0, 0) * (b[1] * A(2, 2) - A(1, 2) * b[2])
          - b[0] * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
          + A(0, 2) * (A(1, 0) * b[2] - b[1] * A(2, 0))) / det;

    // x[2] = det(A2) / det(A), where A2 has third column replaced by b
    x[2] = (A(0, 0) * (A(1, 1) * b[2] - b[1] * A(2, 1))
          - A(0, 1) * (A(1, 0) * b[2] - b[1] * A(2, 0))
          + b[0] * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0))) / det;

    return x;
}

/**
 * @brief Solve 4x4 linear system using Gauss elimination with partial pivoting
 * @tparam T Floating-point type
 * @param A 4x4 coefficient matrix
 * @param b Right-hand side vector
 * @return Solution vector x such that Ax = b
 * @throw std::runtime_error if matrix is singular
 */
template<typename T>
inline Vector<T, 4> solve_4x4(const Matrix<T, 4, 4>& A, const Vector<T, 4>& b) {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

    // Create augmented matrix [A|b]
    Matrix<T, 4, 4> work_A = A;
    Vector<T, 4> work_b = b;

    // Forward elimination with partial pivoting
    for (std::size_t k = 0; k < 3; ++k) {
        // Find pivot
        std::size_t pivot_row = k;
        T pivot_val = abs(work_A(k, k));

        for (std::size_t i = k + 1; i < 4; ++i) {
            T val = abs(work_A(i, k));
            if (val > pivot_val) {
                pivot_val = val;
                pivot_row = i;
            }
        }

        // Check for singularity
        if (pivot_val < tolerance<T>) {
            throw std::runtime_error("solve_4x4: Matrix is singular");
        }

        // Swap rows if needed
        if (pivot_row != k) {
            for (std::size_t j = k; j < 4; ++j) {
                std::swap(work_A(k, j), work_A(pivot_row, j));
            }
            std::swap(work_b[k], work_b[pivot_row]);
        }

        // Eliminate column below pivot
        for (std::size_t i = k + 1; i < 4; ++i) {
            T factor = work_A(i, k) / work_A(k, k);
            for (std::size_t j = k + 1; j < 4; ++j) {
                work_A(i, j) -= factor * work_A(k, j);
            }
            work_b[i] -= factor * work_b[k];
            work_A(i, k) = T(0);  // Explicitly zero out for clarity
        }
    }

    // Check last pivot
    if (abs(work_A(3, 3)) < tolerance<T>) {
        throw std::runtime_error("solve_4x4: Matrix is singular");
    }

    // Back substitution
    Vector<T, 4> x;
    x[3] = work_b[3] / work_A(3, 3);
    x[2] = (work_b[2] - work_A(2, 3) * x[3]) / work_A(2, 2);
    x[1] = (work_b[1] - work_A(1, 2) * x[2] - work_A(1, 3) * x[3]) / work_A(1, 1);
    x[0] = (work_b[0] - work_A(0, 1) * x[1] - work_A(0, 2) * x[2] - work_A(0, 3) * x[3]) / work_A(0, 0);

    return x;
}

/**
 * @brief LU decomposition with partial pivoting for general NxN matrices
 * @tparam T Floating-point type
 * @tparam N Matrix dimension
 *
 * Performs LU decomposition of a matrix A into L and U factors such that
 * PA = LU, where P is a permutation matrix. Uses partial pivoting for
 * numerical stability.
 */
template<typename T, std::size_t N>
class LUDecomposition {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");
    static_assert(N > 0, "Matrix dimension must be positive");

private:
    Matrix<T, N, N> LU_;           // Combined L and U factors
    std::array<std::size_t, N> pivot_;  // Pivot indices
    int sign_;                     // Sign of permutation (+1 or -1)
    bool singular_;                // Singularity flag

    /**
     * @brief Perform LU decomposition with partial pivoting
     * @param A Input matrix
     */
    void decompose(const Matrix<T, N, N>& A) {
        // Initialize
        LU_ = A;
        for (std::size_t i = 0; i < N; ++i) {
            pivot_[i] = i;
        }
        sign_ = 1;
        singular_ = false;

        // Gaussian elimination with partial pivoting
        for (std::size_t k = 0; k < N - 1; ++k) {
            // Find pivot
            std::size_t pivot_row = k;
            T pivot_val = abs(LU_(k, k));

            for (std::size_t i = k + 1; i < N; ++i) {
                T val = abs(LU_(i, k));
                if (val > pivot_val) {
                    pivot_val = val;
                    pivot_row = i;
                }
            }

            // Check for singularity
            if (pivot_val < tolerance<T>) {
                singular_ = true;
                return;
            }

            // Swap rows if needed
            if (pivot_row != k) {
                for (std::size_t j = 0; j < N; ++j) {
                    std::swap(LU_(k, j), LU_(pivot_row, j));
                }
                std::swap(pivot_[k], pivot_[pivot_row]);
                sign_ = -sign_;
            }

            // Compute multipliers and eliminate
            for (std::size_t i = k + 1; i < N; ++i) {
                LU_(i, k) /= LU_(k, k);  // Store multiplier in L
                for (std::size_t j = k + 1; j < N; ++j) {
                    LU_(i, j) -= LU_(i, k) * LU_(k, j);
                }
            }
        }

        // Check last diagonal element
        if (abs(LU_(N - 1, N - 1)) < tolerance<T>) {
            singular_ = true;
        }
    }

public:
    /**
     * @brief Constructor - performs decomposition
     * @param A Matrix to decompose
     */
    explicit LUDecomposition(const Matrix<T, N, N>& A) {
        decompose(A);
    }

    /**
     * @brief Check if matrix is singular
     * @return True if matrix is singular
     */
    bool is_singular() const { return singular_; }

    /**
     * @brief Solve linear system Ax = b
     * @param b Right-hand side vector
     * @return Solution vector x
     * @throw std::runtime_error if matrix is singular
     */
    Vector<T, N> solve(const Vector<T, N>& b) const {
        if (singular_) {
            throw std::runtime_error("LUDecomposition::solve: Matrix is singular");
        }

        // Apply the final row permutation: x = P*b where P(i, pivot_[i]) = 1.
        // Note: pivot_ stores the original row index that ended up at row i.
        Vector<T, N> x;
        for (std::size_t i = 0; i < N; ++i) {
            x[i] = b[pivot_[i]];
        }

        // Forward substitution (Ly = Pb)
        for (std::size_t i = 1; i < N; ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                x[i] -= LU_(i, j) * x[j];
            }
        }

        // Back substitution (Ux = y)
        for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
            std::size_t idx = static_cast<std::size_t>(i);
            for (std::size_t j = idx + 1; j < N; ++j) {
                x[idx] -= LU_(idx, j) * x[j];
            }
            x[idx] /= LU_(idx, idx);
        }

        return x;
    }

    /**
     * @brief Solve multiple linear systems with same matrix
     * @param B Matrix of right-hand side vectors
     * @return Matrix of solution vectors
     * @throw std::runtime_error if matrix is singular
     */
    template<std::size_t M>
    Matrix<T, N, M> solve(const Matrix<T, N, M>& B) const {
        Matrix<T, N, M> X;
        for (std::size_t j = 0; j < M; ++j) {
            Vector<T, N> b_col = B.column(j);
            Vector<T, N> x_col = solve(b_col);
            for (std::size_t i = 0; i < N; ++i) {
                X(i, j) = x_col[i];
            }
        }
        return X;
    }

    /**
     * @brief Compute determinant
     * @return Determinant of original matrix
     */
    T determinant() const {
        if (singular_) return T(0);

        T det = T(sign_);
        for (std::size_t i = 0; i < N; ++i) {
            det *= LU_(i, i);
        }
        return det;
    }

    /**
     * @brief Compute matrix inverse
     * @return Inverse of original matrix
     * @throw std::runtime_error if matrix is singular
     */
    Matrix<T, N, N> inverse() const {
        if (singular_) {
            throw std::runtime_error("LUDecomposition::inverse: Matrix is singular");
        }

        // Solve AX = I for X = A^(-1)
        Matrix<T, N, N> inv;
        for (std::size_t j = 0; j < N; ++j) {
            Vector<T, N> e;  // Unit vector
            e[j] = T(1);
            Vector<T, N> col = solve(e);
            for (std::size_t i = 0; i < N; ++i) {
                inv(i, j) = col[i];
            }
        }
        return inv;
    }

    /**
     * @brief Get L factor (lower triangular with unit diagonal)
     * @return L matrix
     */
    Matrix<T, N, N> get_L() const {
        Matrix<T, N, N> L;
        for (std::size_t i = 0; i < N; ++i) {
            L(i, i) = T(1);  // Unit diagonal
            for (std::size_t j = 0; j < i; ++j) {
                L(i, j) = LU_(i, j);
            }
        }
        return L;
    }

    /**
     * @brief Get U factor (upper triangular)
     * @return U matrix
     */
    Matrix<T, N, N> get_U() const {
        Matrix<T, N, N> U;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i; j < N; ++j) {
                U(i, j) = LU_(i, j);
            }
        }
        return U;
    }

    /**
     * @brief Get permutation matrix P
     * @return Permutation matrix
     */
    Matrix<T, N, N> get_P() const {
        Matrix<T, N, N> P;
        for (std::size_t i = 0; i < N; ++i) {
            P(i, pivot_[i]) = T(1);
        }
        return P;
    }
};

/**
 * @brief Generic solver that selects optimal method based on size
 * @tparam T Floating-point type
 * @tparam N Matrix dimension
 * @param A Coefficient matrix
 * @param b Right-hand side vector
 * @return Solution vector x such that Ax = b
 * @throw std::runtime_error if matrix is singular
 */
template<typename T, std::size_t N>
inline Vector<T, N> solve(const Matrix<T, N, N>& A, const Vector<T, N>& b) {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

    if constexpr (N == 2) {
        return solve_2x2(A, b);
    } else if constexpr (N == 3) {
        return solve_3x3(A, b);
    } else if constexpr (N == 4) {
        return solve_4x4(A, b);
    } else {
        LUDecomposition<T, N> lu(A);
        return lu.solve(b);
    }
}

/**
 * @brief Perform LU factorization and return L, U, P matrices
 * @tparam T Floating-point type
 * @tparam N Matrix dimension
 * @param A Matrix to factorize
 * @return Tuple of (L, U, P) matrices such that PA = LU
 * @throw std::runtime_error if matrix is singular
 */
template<typename T, std::size_t N>
inline std::tuple<Matrix<T, N, N>, Matrix<T, N, N>, Matrix<T, N, N>>
lu_factorize(const Matrix<T, N, N>& A) {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

    LUDecomposition<T, N> lu(A);

    if (lu.is_singular()) {
        throw std::runtime_error("lu_factorize: Matrix is singular");
    }

    return std::make_tuple(lu.get_L(), lu.get_U(), lu.get_P());
}

/**
 * @brief Solve linear system using pre-computed LU factorization
 * @tparam T Floating-point type
 * @tparam N Matrix dimension
 * @param L Lower triangular matrix with unit diagonal
 * @param U Upper triangular matrix
 * @param P Permutation matrix
 * @param b Right-hand side vector
 * @return Solution vector x such that PA = LU and Ax = b
 */
template<typename T, std::size_t N>
inline Vector<T, N> lu_solve(const Matrix<T, N, N>& L, const Matrix<T, N, N>& U,
                              const Matrix<T, N, N>& P, const Vector<T, N>& b) {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

    // Apply permutation to b: Pb
    Vector<T, N> Pb = P * b;

    // Forward substitution: solve Ly = Pb
    Vector<T, N> y;
    for (std::size_t i = 0; i < N; ++i) {
        T sum = Pb[i];
        for (std::size_t j = 0; j < i; ++j) {
            sum -= L(i, j) * y[j];
        }
        y[i] = sum / L(i, i);  // L has unit diagonal, so this is just sum
    }

    // Back substitution: solve Ux = y
    Vector<T, N> x;
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        std::size_t idx = static_cast<std::size_t>(i);
        T sum = y[idx];
        for (std::size_t j = idx + 1; j < N; ++j) {
            sum -= U(idx, j) * x[j];
        }
        x[idx] = sum / U(idx, idx);
    }

    return x;
}

// Note: inverse() and determinant() free functions are defined in Matrix.h
// to avoid redefinition errors. Matrix.h provides both member functions and
// free function wrappers that delegate to the member functions.

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_LU_H
