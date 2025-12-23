#ifndef SVMP_FE_MATH_EIGENSOLVERS_H
#define SVMP_FE_MATH_EIGENSOLVERS_H

/**
 * @file Eigensolvers.h
 * @brief Analytical eigensolvers for symmetric 2x2 and 3x3 matrices
 *
 * This header provides specialized eigenvalue/eigenvector computations for
 * symmetric matrices commonly encountered in continuum mechanics, particularly
 * for computing principal stresses and strains. Only analytical solutions are
 * implemented for maximum performance and numerical stability.
 */

#include "Matrix.h"
#include "Vector.h"
#include "MathConstants.h"
#include "MathUtils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace svmp {
namespace FE {
namespace math {

/**
 * @brief Eigendecomposition for symmetric 3x3 matrices
 * @tparam T Scalar type (float, double)
 *
 * Used for material characterization, computing principal stresses/strains,
 * and tensor decomposition. Uses analytical solution via characteristic polynomial
 * for guaranteed convergence and optimal performance.
 */
template<typename T>
class SymmetricEigen3x3 {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

private:
    Matrix3x3<T> eigenvectors_;  // Column-wise eigenvectors
    Vector3<T> eigenvalues_;     // Eigenvalues in ascending order

public:
    /**
     * @brief Compute eigendecomposition of symmetric 3x3 matrix
     * @param A Symmetric 3x3 matrix
     *
     * Uses analytical solution of characteristic polynomial:
     * det(A - λI) = 0
     * λ³ - I₁λ² + I₂λ - I₃ = 0
     * where I₁, I₂, I₃ are the invariants of A
     */
    explicit SymmetricEigen3x3(const Matrix3x3<T>& A) {
        compute(A);
    }

    /**
     * @brief Get computed eigenvalues (sorted in ascending order)
     * @return Vector of eigenvalues [λ₁ ≤ λ₂ ≤ λ₃]
     */
    const Vector3<T>& eigenvalues() const { return eigenvalues_; }

    /**
     * @brief Get computed eigenvectors (as columns of matrix)
     * @return Matrix with eigenvectors as columns
     */
    const Matrix3x3<T>& eigenvectors() const { return eigenvectors_; }

     /**
     * @brief Get principal values (alias for eigenvalues)
     * @return Principal values sorted in ascending order
     */
    const Vector3<T>& principal_values() const { return eigenvalues_; }

    /**
     * @brief Get principal directions (alias for eigenvectors)
     * @return Principal directions as column vectors
     */
    const Matrix3x3<T>& principal_directions() const { return eigenvectors_; }

    /**
     * @brief Reconstruct original matrix: A = Q * Λ * Q^T
     * @return Reconstructed matrix
     */
    Matrix3x3<T> reconstruct() const {
        Matrix3x3<T> Lambda = Matrix3x3<T>::zeros();
        Lambda(0, 0) = eigenvalues_[0];
        Lambda(1, 1) = eigenvalues_[1];
        Lambda(2, 2) = eigenvalues_[2];

        return eigenvectors_ * Lambda * eigenvectors_.transpose();
    }

private:
    void compute(const Matrix3x3<T>& A) {
        // Ensure symmetry (use average of A and A^T)
        Matrix3x3<T> S;
        for (std::size_t i = 0; i < 3; ++i) {
            S(i, i) = A(i, i);
            for (std::size_t j = i + 1; j < 3; ++j) {
                T avg = (A(i, j) + A(j, i)) / T(2);
                S(i, j) = avg;
                S(j, i) = avg;
            }
        }

        // Compute characteristic polynomial coefficients
        // char poly: λ³ - I₁λ² + I₂λ - I₃ = 0
        T I1 = S(0, 0) + S(1, 1) + S(2, 2);  // Trace

        T I2 = S(0, 0) * S(1, 1) + S(1, 1) * S(2, 2) + S(2, 2) * S(0, 0)
             - S(0, 1) * S(0, 1) - S(1, 2) * S(1, 2) - S(0, 2) * S(0, 2);

        T I3 = S.determinant();

        // Solve characteristic polynomial using Cardano's method
        // The characteristic polynomial is: λ³ - I₁λ² + I₂λ - I₃ = 0
        // Transform to depressed cubic: t³ + pt + q = 0
        // where λ = t + I₁/3
        // Substituting: (t + I₁/3)³ - I₁(t + I₁/3)² + I₂(t + I₁/3) - I₃ = 0
        // Expanding and simplifying: t³ + pt + q = 0
        T p = I2 - I1 * I1 / T(3);
        T q = -T(2) * I1 * I1 * I1 / T(27) + I1 * I2 / T(3) - I3;

        // For symmetric matrices, all roots are real
        // Use trigonometric solution
        if (std::abs(p) < epsilon<T>) {
            // Special case: p ≈ 0
            eigenvalues_[0] = eigenvalues_[1] = eigenvalues_[2] = I1 / T(3);
        } else if (p < 0) {
            // For symmetric matrices with p < 0, use trigonometric solution
            // This gives three real roots using Vieta's substitution
            T r = std::sqrt(-p * p * p / T(27));
            T arg = -q / (T(2) * r);
            // Clamp argument to valid range for acos
            arg = std::max(T(-1), std::min(T(1), arg));
            T theta = std::acos(arg);
            T two_cbrt_r = T(2) * std::cbrt(r);

            eigenvalues_[0] = two_cbrt_r * std::cos(theta / T(3)) + I1 / T(3);
            eigenvalues_[1] = two_cbrt_r * std::cos((theta + T(2) * pi<T>) / T(3)) + I1 / T(3);
            eigenvalues_[2] = two_cbrt_r * std::cos((theta + T(4) * pi<T>) / T(3)) + I1 / T(3);
        } else {
            // p > 0 shouldn't happen for symmetric matrices, but handle it anyway
            // This would give one real root and two complex roots
            // For now, just return diagonal approximation
            eigenvalues_[0] = eigenvalues_[1] = eigenvalues_[2] = I1 / T(3);
        }

        // Compute eigenvectors for each eigenvalue (before sorting)
        for (std::size_t k = 0; k < 3; ++k) {
            T lambda = eigenvalues_[k];

            // Solve (A - λI)v = 0
            Matrix3x3<T> B = S;
            B(0, 0) -= lambda;
            B(1, 1) -= lambda;
            B(2, 2) -= lambda;

            // Find the eigenvector using cross products of rows
            Vector3<T> row0{B(0, 0), B(0, 1), B(0, 2)};
            Vector3<T> row1{B(1, 0), B(1, 1), B(1, 2)};
            Vector3<T> row2{B(2, 0), B(2, 1), B(2, 2)};

            Vector3<T> v1 = cross(row0, row1);
            Vector3<T> v2 = cross(row1, row2);
            Vector3<T> v3 = cross(row0, row2);

            // Choose the cross product with largest magnitude
            T norm1 = v1.norm_squared();
            T norm2 = v2.norm_squared();
            T norm3 = v3.norm_squared();

            Vector3<T> eigenvector;
            if (norm1 >= norm2 && norm1 >= norm3) {
                eigenvector = v1.normalized();
            } else if (norm2 >= norm3) {
                eigenvector = v2.normalized();
            } else {
                eigenvector = v3.normalized();
            }

            // Handle degenerate cases
            if (eigenvector.norm_squared() < epsilon<T>) {
                // Use a default vector orthogonal to previous eigenvectors
                if (k == 0) {
                    eigenvector = Vector3<T>{T(1), T(0), T(0)};
                } else if (k == 1) {
                    Vector3<T> v0;
                    v0[0] = eigenvectors_(0, 0);
                    v0[1] = eigenvectors_(1, 0);
                    v0[2] = eigenvectors_(2, 0);
                    eigenvector = cross(v0, Vector3<T>{T(0), T(1), T(0)});
                    if (eigenvector.norm_squared() < epsilon<T>) {
                        eigenvector = cross(v0, Vector3<T>{T(0), T(0), T(1)});
                    }
                    eigenvector = eigenvector.normalized();
                } else { // k == 2
                    Vector3<T> v0{eigenvectors_(0, 0), eigenvectors_(1, 0), eigenvectors_(2, 0)};
                    Vector3<T> v1_prev{eigenvectors_(0, 1), eigenvectors_(1, 1), eigenvectors_(2, 1)};
                    eigenvector = cross(v0, v1_prev).normalized();
                }
            }

            // Store eigenvector as column
            eigenvectors_(0, k) = eigenvector[0];
            eigenvectors_(1, k) = eigenvector[1];
            eigenvectors_(2, k) = eigenvector[2];
        }

        // Ensure eigenvectors are orthonormal (Gram-Schmidt if needed)
        orthonormalize_eigenvectors();

        // Sort eigenvalues and eigenvectors in ascending order (smallest to largest)
        // Use bubble sort to keep eigenvector columns synchronized with eigenvalues
        auto swap_columns = [this](std::size_t i, std::size_t j) {
            std::swap(eigenvalues_[i], eigenvalues_[j]);
            // Swap eigenvector columns
            for (std::size_t row = 0; row < 3; ++row) {
                std::swap(eigenvectors_(row, i), eigenvectors_(row, j));
            }
        };

        if (eigenvalues_[0] > eigenvalues_[1]) swap_columns(0, 1);
        if (eigenvalues_[1] > eigenvalues_[2]) swap_columns(1, 2);
        if (eigenvalues_[0] > eigenvalues_[1]) swap_columns(0, 1);
    }

    void orthonormalize_eigenvectors() {
        // Apply Gram-Schmidt orthogonalization to ensure orthonormality
        for (std::size_t i = 0; i < 3; ++i) {
            Vector3<T> vi{eigenvectors_(0, i), eigenvectors_(1, i), eigenvectors_(2, i)};

            // Orthogonalize against previous vectors
            for (std::size_t j = 0; j < i; ++j) {
                Vector3<T> vj{eigenvectors_(0, j), eigenvectors_(1, j), eigenvectors_(2, j)};
                T proj = vi.dot(vj);
                vi = vi - proj * vj;
            }

            // Normalize
            vi = vi.normalized();

            // Store back
            eigenvectors_(0, i) = vi[0];
            eigenvectors_(1, i) = vi[1];
            eigenvectors_(2, i) = vi[2];
        }
    }
};

/**
 * @brief Analytical eigendecomposition for symmetric 2x2 matrix
 * @tparam T Scalar type (float, double)
 * @param A Symmetric 2x2 matrix
 * @return Pair of (eigenvalues, eigenvectors) where eigenvalues are sorted descending
 *
 * Uses closed-form solution:
 * λ = (trace ± √(trace² - 4·det)) / 2
 */
template<typename T>
std::pair<Vector2<T>, Matrix2x2<T>> eigen_2x2_symmetric(const Matrix2x2<T>& A) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

    // Ensure symmetry
    T a00 = A(0, 0);
    T a11 = A(1, 1);
    T a01 = (A(0, 1) + A(1, 0)) / T(2);

    // Compute eigenvalues using trace and determinant
    T trace = a00 + a11;
    T det = a00 * a11 - a01 * a01;
    T discriminant = trace * trace - T(4) * det;

    Vector2<T> eigenvalues;
    Matrix2x2<T> eigenvectors;

    if (discriminant < T(0)) {
        discriminant = T(0);  // Clamp negative values due to rounding
    }

    T sqrt_disc = std::sqrt(discriminant);
    eigenvalues[0] = (trace + sqrt_disc) / T(2);  // First eigenvalue
    eigenvalues[1] = (trace - sqrt_disc) / T(2);  // Second eigenvalue

    // Ensure eigenvalues are sorted in descending order
    if (eigenvalues[0] < eigenvalues[1]) {
        std::swap(eigenvalues[0], eigenvalues[1]);
    }

    // Compute eigenvectors
    if (std::abs(a01) > epsilon<T>) {
        // Non-diagonal matrix
        // For eigenvalue λ, eigenvector is [λ - a11, a01] or [a01, λ - a00]
        // We use the more numerically stable choice
        Vector2<T> v1, v2;

        // For first eigenvalue
        if (std::abs(eigenvalues[0] - a00) > std::abs(eigenvalues[0] - a11)) {
            v1 = Vector2<T>{a01, eigenvalues[0] - a00};
        } else {
            v1 = Vector2<T>{eigenvalues[0] - a11, a01};
        }

        // For second eigenvalue
        if (std::abs(eigenvalues[1] - a00) > std::abs(eigenvalues[1] - a11)) {
            v2 = Vector2<T>{a01, eigenvalues[1] - a00};
        } else {
            v2 = Vector2<T>{eigenvalues[1] - a11, a01};
        }

        // Normalize
        v1 = v1.normalized();
        v2 = v2.normalized();

        eigenvectors(0, 0) = v1[0];
        eigenvectors(1, 0) = v1[1];
        eigenvectors(0, 1) = v2[0];
        eigenvectors(1, 1) = v2[1];
    } else if (std::abs(a00 - a11) > epsilon<T>) {
        // Diagonal matrix with different eigenvalues
        if (eigenvalues[0] == a00) {
            // First eigenvalue corresponds to first diagonal
            eigenvectors(0, 0) = T(1);
            eigenvectors(1, 0) = T(0);
            eigenvectors(0, 1) = T(0);
            eigenvectors(1, 1) = T(1);
        } else {
            // First eigenvalue corresponds to second diagonal
            eigenvectors(0, 0) = T(0);
            eigenvectors(1, 0) = T(1);
            eigenvectors(0, 1) = T(1);
            eigenvectors(1, 1) = T(0);
        }
    } else {
        // Multiple eigenvalue (spherical tensor)
        eigenvectors = Matrix2x2<T>::identity();
    }

    return {eigenvalues, eigenvectors};
}

/**
 * @brief Analytical eigendecomposition for symmetric 3x3 matrix (function interface)
 * @tparam T Scalar type (float, double)
 * @param A Symmetric 3x3 matrix
 * @return Pair of (eigenvalues, eigenvectors) where eigenvalues are sorted ascending
 *
 * Wrapper function for SymmetricEigen3x3 class providing a simpler interface.
 */
template<typename T>
std::pair<Vector3<T>, Matrix3x3<T>> eigen_3x3_symmetric(const Matrix3x3<T>& A) {
    SymmetricEigen3x3<T> decomp(A);
    return {decomp.eigenvalues(), decomp.eigenvectors()};
}

/**
 * @brief Compute principal stresses from stress tensor
 * @tparam T Scalar type (float, double)
 * @param stress Symmetric 3x3 stress tensor
 * @return Tuple of (principal stresses, principal directions, von Mises stress)
 *
 * Principal stresses are eigenvalues sorted as σ₁ ≤ σ₂ ≤ σ₃
 * Von Mises stress: σᵥₘ = √(0.5 * [(σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²])
 */
template<typename T>
std::tuple<Vector3<T>, Matrix3x3<T>, T> compute_principal_stresses(const Matrix3x3<T>& stress) {
    SymmetricEigen3x3<T> decomp(stress);
    Vector3<T> principal = decomp.eigenvalues();
    Matrix3x3<T> directions = decomp.eigenvectors();

    // Compute von Mises stress
    T s1_s2 = principal[0] - principal[1];
    T s2_s3 = principal[1] - principal[2];
    T s3_s1 = principal[2] - principal[0];
    T von_mises = std::sqrt(T(0.5) * (s1_s2 * s1_s2 + s2_s3 * s2_s3 + s3_s1 * s3_s1));

    return {principal, directions, von_mises};
}

/**
 * @brief Compute principal strains from strain tensor
 * @tparam T Scalar type (float, double)
 * @param strain Symmetric 3x3 strain tensor
 * @return Tuple of (principal strains, principal directions, equivalent strain)
 *
 * Principal strains are eigenvalues sorted as ε₁ ≤ ε₂ ≤ ε₃
 * Equivalent strain uses von Mises criterion
 */
template<typename T>
std::tuple<Vector3<T>, Matrix3x3<T>, T> compute_principal_strains(const Matrix3x3<T>& strain) {
    SymmetricEigen3x3<T> decomp(strain);
    Vector3<T> principal = decomp.eigenvalues();
    Matrix3x3<T> directions = decomp.eigenvectors();

    // Compute equivalent strain (von Mises criterion)
    T e1_e2 = principal[0] - principal[1];
    T e2_e3 = principal[1] - principal[2];
    T e3_e1 = principal[2] - principal[0];
    T equiv = std::sqrt(T(2.0/3.0) * (e1_e2 * e1_e2 + e2_e3 * e2_e3 + e3_e1 * e3_e1));

    return {principal, directions, equiv};
}

namespace detail {

template<typename T, std::size_t N>
inline Matrix<T, N, N> symmetrize(const Matrix<T, N, N>& A) {
    Matrix<T, N, N> S;
    for (std::size_t i = 0; i < N; ++i) {
        S(i, i) = A(i, i);
        for (std::size_t j = i + 1; j < N; ++j) {
            T avg = (A(i, j) + A(j, i)) / T(2);
            S(i, j) = avg;
            S(j, i) = avg;
        }
    }
    return S;
}

template<typename T, std::size_t N, std::size_t K>
inline Matrix<T, K, K> leading_principal_submatrix(const Matrix<T, N, N>& A) {
    Matrix<T, K, K> sub;
    for (std::size_t i = 0; i < K; ++i) {
        for (std::size_t j = 0; j < K; ++j) {
            sub(i, j) = A(i, j);
        }
    }
    return sub;
}

template<typename T, std::size_t N, std::size_t K>
struct SylvesterPositiveDefinite {
    static bool check(const Matrix<T, N, N>& A, T tol) {
        if (!SylvesterPositiveDefinite<T, N, K - 1>::check(A, tol)) {
            return false;
        }
        const auto sub = leading_principal_submatrix<T, N, K>(A);
        return sub.determinant() > tol;
    }
};

template<typename T, std::size_t N>
struct SylvesterPositiveDefinite<T, N, 1> {
    static bool check(const Matrix<T, N, N>& A, T tol) {
        return A(0, 0) > tol;
    }
};

} // namespace detail

/**
 * @brief Check if a symmetric matrix is positive definite
 * @tparam T Scalar type
 * @tparam N Matrix dimension
 * @param A Matrix to test (symmetrized internally)
 * @param tol Tolerance for positivity checks
 * @return True if matrix is positive definite
 */
template<typename T, std::size_t N>
bool is_positive_definite(const Matrix<T, N, N>& A, T tol = epsilon<T>) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

    const auto S = detail::symmetrize(A);

    if constexpr (N == 2) {
        auto [eigenvals, eigenvecs] = eigen_2x2_symmetric(S);
        return eigenvals[0] > tol && eigenvals[1] > tol;
    } else if constexpr (N == 3) {
        SymmetricEigen3x3<T> decomp(S);
        const auto& eigenvals = decomp.eigenvalues();
        return eigenvals[0] > tol && eigenvals[1] > tol && eigenvals[2] > tol;
    } else {
        // Sylvester's criterion: symmetric A is SPD iff all leading principal minors are positive.
        return detail::SylvesterPositiveDefinite<T, N, N>::check(S, tol);
    }
}

// =============================================================================
// Convenience wrapper functions for common use cases
// =============================================================================

/**
 * @brief Compute eigenvalues and eigenvectors of a symmetric 2x2 matrix
 * @param A Symmetric 2x2 matrix
 * @return Tuple of (eigenvalues, eigenvectors) where eigenvalues are sorted descending
 */
template<typename T>
inline std::tuple<Vector2<T>, Matrix2x2<T>> eigen_symmetric_2x2(const Matrix2x2<T>& A) {
    return eigen_2x2_symmetric(A);
}

/**
 * @brief Compute eigenvalues and eigenvectors of a symmetric 3x3 matrix
 * @param A Symmetric 3x3 matrix
 * @return Tuple of (eigenvalues, eigenvectors) where eigenvalues are sorted ascending
 */
template<typename T>
inline std::tuple<Vector3<T>, Matrix3x3<T>> eigen_symmetric_3x3(const Matrix3x3<T>& A) {
    SymmetricEigen3x3<T> decomp(A);
    return {decomp.eigenvalues(), decomp.eigenvectors()};
}

/**
 * @brief Compute principal stresses and their directions
 * @param stress Stress tensor (symmetric 3x3)
 * @return Tuple of (principal stresses, principal directions)
 */
template<typename T>
inline std::tuple<Vector3<T>, Matrix3x3<T>> principal_stresses(const Matrix3x3<T>& stress) {
    auto [principal, directions, equiv] = compute_principal_stresses(stress);
    return {principal, directions};
}

/**
 * @brief Compute principal strains and their directions
 * @param strain Strain tensor (symmetric 3x3)
 * @return Tuple of (principal strains, principal directions)
 */
template<typename T>
inline std::tuple<Vector3<T>, Matrix3x3<T>> principal_strains(const Matrix3x3<T>& strain) {
    auto [principal, directions, equiv] = compute_principal_strains(strain);
    return {principal, directions};
}

/**
 * @brief Compute matrix power using eigendecomposition
 * @param A Symmetric matrix
 * @param p Power to raise matrix to
 * @return A^p computed via eigendecomposition
 */
template<typename T>
inline Matrix3x3<T> matrix_power(const Matrix3x3<T>& A, T p) {
    SymmetricEigen3x3<T> decomp(A);
    const auto& eigenvals = decomp.eigenvalues();
    const auto& eigenvecs = decomp.eigenvectors();

    // Compute Λ^p
    Matrix3x3<T> Lambda_p = Matrix3x3<T>::zeros();
    Lambda_p(0, 0) = std::pow(eigenvals[0], p);
    Lambda_p(1, 1) = std::pow(eigenvals[1], p);
    Lambda_p(2, 2) = std::pow(eigenvals[2], p);

    // A^p = Q * Λ^p * Q^T
    return eigenvecs * Lambda_p * eigenvecs.transpose();
}

/**
 * @brief Compute matrix square root using eigendecomposition
 * @param A Symmetric positive semi-definite matrix
 * @return A^(1/2) computed via eigendecomposition
 */
template<typename T>
inline Matrix3x3<T> matrix_sqrt(const Matrix3x3<T>& A) {
    return matrix_power(A, T(0.5));
}

/**
 * @brief Compute stress tensor invariants
 * @param stress Stress tensor (3x3)
 * @return Tuple of (I1, I2, I3) invariants
 */
template<typename T>
inline std::tuple<T, T, T> stress_invariants(const Matrix3x3<T>& stress) {
    // First invariant: trace
    T I1 = stress(0, 0) + stress(1, 1) + stress(2, 2);

    // Second invariant
    T I2 = stress(0, 0) * stress(1, 1) + stress(1, 1) * stress(2, 2) +
           stress(2, 2) * stress(0, 0) - stress(0, 1) * stress(0, 1) -
           stress(1, 2) * stress(1, 2) - stress(0, 2) * stress(0, 2);

    // Third invariant: determinant
    T I3 = stress.determinant();

    return {I1, I2, I3};
}

/**
 * @brief Compute deviatoric stress tensor
 * @param stress Stress tensor (3x3)
 * @return Deviatoric stress tensor (zero trace)
 */
template<typename T>
inline Matrix3x3<T> deviatoric_stress(const Matrix3x3<T>& stress) {
    T trace = stress(0, 0) + stress(1, 1) + stress(2, 2);
    T mean_stress = trace / T(3);

    Matrix3x3<T> dev = stress;
    dev(0, 0) -= mean_stress;
    dev(1, 1) -= mean_stress;
    dev(2, 2) -= mean_stress;

    return dev;
}

/**
 * @brief Compute von Mises stress from a stress tensor (overload for Matrix)
 * @param stress Stress tensor as Matrix3x3
 * @return von Mises equivalent stress
 */
template<typename T>
inline T von_mises_stress(const Matrix3x3<T>& stress) {
    // Compute deviatoric stress
    Matrix3x3<T> dev = deviatoric_stress(stress);

    // von Mises stress = sqrt(3/2 * dev:dev)
    T dev_double_contract = T(0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            dev_double_contract += dev(i, j) * dev(i, j);
        }
    }
    return std::sqrt(T(3.0/2.0) * dev_double_contract);
}

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_EIGENSOLVERS_H
