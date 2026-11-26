#ifndef SVMP_FE_MATH_EIGENSOLVERS_H
#define SVMP_FE_MATH_EIGENSOLVERS_H

/**
 * @file Eigensolvers.h
 * @brief Fixed-size eigensolvers for symmetric 2x2 and 3x3 matrices
 *
 * This header provides specialized eigenvalue/eigenvector computations for
 * symmetric matrices commonly encountered in continuum mechanics, particularly
 * for computing principal stresses and strains.
 */

#include "Matrix.h"
#include "Vector.h"
#include "MathConstants.h"
#include "MathUtils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

namespace svmp {
namespace FE {
namespace math {

/// Remove a scalar multiple of I before computing spectral gaps. Using the
/// diagonal entry nearest zero (or zero for mixed signs) avoids overflow in
/// the subtraction and preserves the small differences of a shifted spectrum.
template<typename T, std::size_t N>
T remove_diagonal_shift(Matrix<T, N, N>& matrix)
{
    T low = matrix(0, 0), high = low;
    for (std::size_t i = 1; i < N; ++i) {
        low = std::min(low, matrix(i, i));
        high = std::max(high, matrix(i, i));
    }
    const T shift = std::clamp(T(0), low, high);
    for (std::size_t i = 0; i < N; ++i) matrix(i, i) -= shift;
    return shift;
}

/// Normalize extreme spectra with an exact power of two, leaving headroom
/// for the sum of up to N entries during orthogonal transformations.
template<typename T, std::size_t N>
int scale_symmetric_spectrum(Matrix<T, N, N>& matrix)
{
    T magnitude = 0;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; j < N; ++j)
            magnitude = std::max(magnitude, std::abs(matrix(i, j)));
    int exponent = 0;
    if (magnitude > std::numeric_limits<T>::max() / T(N))
        exponent = N == 2 ? 1 : 2;
    else if (magnitude > 0 && magnitude < std::numeric_limits<T>::min() / std::numeric_limits<T>::epsilon())
        std::frexp(magnitude, &exponent);
    if (exponent != 0)
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                matrix(i, j) = std::scalbn(matrix(i, j), -exponent);
    return exponent;
}

namespace detail {

// A Jacobi rotation diagonalizes a 2x2 pivot exactly. Scaling its half-gap
// and off-diagonal together keeps the angle independent of the units.
template<typename T, std::size_t N>
std::pair<Vector<T, N>, Matrix<T, N, N>> small_symmetric_eigenpairs(Matrix<T, N, N> matrix)
{
    static_assert((N == 2 || N == 3) && std::is_floating_point_v<T>);
    auto vectors = Matrix<T, N, N>::identity();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            matrix(i, j) = matrix(j, i) = std::midpoint(matrix(i, j), matrix(j, i));
        }
    }
    const T shift = remove_diagonal_shift(matrix);
    const int exponent = scale_symmetric_spectrum(matrix);
    constexpr int max_rotations = N == 2 ? 1 : 24;
    for (int step = 0; step < max_rotations; ++step) {
        std::size_t p = 0, q = 1;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                if (std::abs(matrix(i, j)) > std::abs(matrix(p, q))) {
                    p = i;
                    q = j;
                }
            }
        }
        const T off = matrix(p, q);
        if (off == T(0)) break;
        const T difference = matrix(q, q) - matrix(p, p);
        // Scale before halving a tiny pivot difference. For opposite,
        // near-limit diagonals their half-difference is representable.
        const bool full_gap = std::isfinite(difference);
        const T gap = full_gap ? difference : std::midpoint(matrix(q, q), -matrix(p, p));
        const T scale = std::max(std::abs(gap), std::abs(off));
        const T d = (gap / scale) * (full_gap ? T(0.5) : T(1)), b = off / scale;
        const T tangent = b / (d + std::copysign(std::hypot(d, b), d));
        const T cosine = T(1) / std::sqrt(T(1) + tangent * tangent);
        const T sine = tangent * cosine;
        matrix(p, p) -= tangent * off;
        matrix(q, q) += tangent * off;
        matrix(p, q) = matrix(q, p) = T(0);
        for (std::size_t k = 0; k < N; ++k) {
            if (k != p && k != q) {
                const T a = matrix(k, p), c = matrix(k, q);
                matrix(k, p) = matrix(p, k) = cosine * a - sine * c;
                matrix(k, q) = matrix(q, k) = sine * a + cosine * c;
            }
            const T a = vectors(k, p), c = vectors(k, q);
            vectors(k, p) = cosine * a - sine * c;
            vectors(k, q) = sine * a + cosine * c;
        }
    }
    Vector<T, N> values;
    for (std::size_t i = 0; i < N; ++i) values[i] = matrix(i, i);
    // Preserve the public ordering conventions, including when adding the
    // shift makes two distinct, centered eigenvalues round to the same value.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            if ((N == 2) ? values[j] > values[i] : values[j] < values[i]) {
                std::swap(values[i], values[j]);
                for (std::size_t k = 0; k < N; ++k) std::swap(vectors(k, i), vectors(k, j));
            }
        }
        // A centered root may exceed the finite range even though translating
        // it back gives a representable eigenvalue.
        if (exponent > 0 && std::abs(values[i]) > std::scalbn(std::numeric_limits<T>::max(), -exponent))
            values[i] = std::scalbn(values[i] + std::scalbn(shift, -exponent), exponent);
        else
            values[i] = (exponent ? std::scalbn(values[i], exponent) : values[i]) + shift;
    }
    return {values, vectors};
}

} // namespace detail

/**
 * @brief Eigendecomposition for symmetric 3x3 matrices
 * @tparam T Scalar type (float, double)
 *
 * Used for material characterization, computing principal stresses/strains,
 * and tensor decomposition.
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
        std::tie(eigenvalues_, eigenvectors_) = detail::small_symmetric_eigenpairs(A);
    }
};

/**
 * @brief Analytical eigendecomposition for symmetric 2x2 matrix
 * @tparam T Scalar type (float, double)
 * @param A Symmetric 2x2 matrix
 * @return Pair of (eigenvalues, eigenvectors) where eigenvalues are sorted descending
 *
 */
template<typename T>
std::pair<Vector2<T>, Matrix2x2<T>> eigen_2x2_symmetric(const Matrix2x2<T>& A) {
    return detail::small_symmetric_eigenpairs(A);
}

/**
 * @brief Eigendecomposition for symmetric 3x3 matrix (function interface)
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
