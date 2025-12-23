#ifndef SVMP_FE_MATH_TENSOR_H
#define SVMP_FE_MATH_TENSOR_H

/**
 * @file Tensor.h
 * @brief Tensor operations for continuum mechanics and FE computations
 *
 * This header provides rank-2 and rank-4 tensor operations essential for
 * stress/strain calculations, material constitutive models, and continuum
 * mechanics. Includes specialized operations for symmetric tensors, invariants,
 * principal values, and tensor contractions.
 */

#include "Vector.h"
#include "Matrix.h"
#include "Eigensolvers.h"
#include "MathUtils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <type_traits>

namespace svmp {
namespace FE {
namespace math {

// Forward declarations
template<typename T, std::size_t Dim> class Tensor2;
template<typename T, std::size_t Dim> class Tensor4;

/**
 * @brief Base class for general tensors of arbitrary rank
 * @tparam T Scalar type (float, double)
 * @tparam Rank Tensor rank (order)
 * @tparam Dim Spatial dimension (2 or 3)
 *
 * This class provides the foundation for tensor storage and basic operations.
 * Specialized classes Tensor2 and Tensor4 provide rank-specific operations.
 */
template<typename T, std::size_t Rank, std::size_t Dim>
class TensorBase {
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");
    static_assert(Dim == 2 || Dim == 3, "Dimension must be 2 or 3");
    static_assert(Rank >= 2, "Tensor rank must be at least 2");

protected:
    // Compile-time power computation for tensor size
    static constexpr std::size_t compute_size() {
        std::size_t result = 1;
        for (std::size_t i = 0; i < Rank; ++i) {
            result *= Dim;
        }
        return result;
    }
    static constexpr std::size_t size_ = compute_size();
    alignas(32) T data_[size_];  // Aligned for SIMD

    // Helper to compute linear index from multi-index
    template<typename... Indices>
    static constexpr std::size_t compute_index(Indices... idx) {
        static_assert(sizeof...(idx) == Rank, "Number of indices must match tensor rank");
        std::size_t index = 0;
        // Process indices in reverse order for row-major storage
        ((index = static_cast<std::size_t>(idx) + index * Dim), ...);
        return index;
    }

public:
    // Type definitions
    using value_type = T;
    using size_type = std::size_t;

    /**
     * @brief Default constructor - zero initializes all components
     */
    constexpr TensorBase() : data_{} {}

    /**
     * @brief Fill constructor - initializes all components with same value
     * @param value Value to fill tensor with
     */
    constexpr explicit TensorBase(T value) {
        for (size_type i = 0; i < size_; ++i) {
            data_[i] = value;
        }
    }

    /**
     * @brief Access element with multi-index
     * @tparam Indices Index types
     * @param idx Indices for each dimension
     * @return Reference to element
     */
    template<typename... Indices>
    T& operator()(Indices... idx) {
        return data_[compute_index(idx...)];
    }

    /**
     * @brief Access element with multi-index (const)
     * @tparam Indices Index types
     * @param idx Indices for each dimension
     * @return Const reference to element
     */
    template<typename... Indices>
    const T& operator()(Indices... idx) const {
        return data_[compute_index(idx...)];
    }

    /**
     * @brief Get raw data pointer
     * @return Pointer to internal data array
     */
    T* data() { return data_; }
    const T* data() const { return data_; }

    /**
     * @brief Get tensor size (total number of components)
     * @return Number of components
     */
    static constexpr size_type size() { return size_; }

    /**
     * @brief Get tensor rank
     * @return Tensor rank
     */
    static constexpr size_type rank() { return Rank; }

    /**
     * @brief Get spatial dimension
     * @return Spatial dimension
     */
    static constexpr size_type dim() { return Dim; }
};

/**
 * @brief Rank-2 tensor (matrix-like) for stress, strain, and deformation
 * @tparam T Scalar type (float, double)
 * @tparam Dim Spatial dimension (2 or 3)
 *
 * Provides specialized operations for second-order tensors including
 * symmetric/antisymmetric decomposition, invariants, eigenvalues, and
 * principal values commonly used in continuum mechanics.
 */
template<typename T, std::size_t Dim>
class Tensor2 : public TensorBase<T, 2, Dim> {
    using Base = TensorBase<T, 2, Dim>;
    using Base::data_;

public:
    using Base::Base;  // Inherit constructors

    /**
     * @brief Construct from Matrix
     * @param mat Input matrix
     */
    explicit Tensor2(const Matrix<T, Dim, Dim>& mat) {
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                (*this)(i, j) = mat(i, j);
            }
        }
    }

    /**
     * @brief Create identity tensor
     * @return Identity tensor (Kronecker delta)
     */
    static Tensor2 identity() {
        Tensor2 result;
        for (std::size_t i = 0; i < Dim; ++i) {
            result(i, i) = T(1);
        }
        return result;
    }

    /**
     * @brief Create zero tensor
     * @return Zero tensor
     */
    static Tensor2 zero() {
        return Tensor2{};
    }

    /**
     * @brief Create tensor from matrix
     * @param mat Input matrix
     * @return Corresponding tensor
     */
    static Tensor2 from_matrix(const Matrix<T, Dim, Dim>& mat) {
        Tensor2 result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                result(i, j) = mat(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Convert tensor to matrix
     * @return Matrix representation
     */
    Matrix<T, Dim, Dim> to_matrix() const {
        Matrix<T, Dim, Dim> result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                result(i, j) = (*this)(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Compute transpose
     * @return Transposed tensor
     */
    Tensor2 transpose() const {
        Tensor2 result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                result(i, j) = (*this)(j, i);
            }
        }
        return result;
    }

    /**
     * @brief Compute symmetric part: (A + A^T)/2
     * @return Symmetric part of tensor
     */
    Tensor2 symmetric_part() const {
        Tensor2 result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                result(i, j) = ((*this)(i, j) + (*this)(j, i)) / T(2);
            }
        }
        return result;
    }

    /**
     * @brief Compute antisymmetric part: (A - A^T)/2
     * @return Antisymmetric (skew-symmetric) part of tensor
     */
    Tensor2 antisymmetric_part() const {
        Tensor2 result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                result(i, j) = ((*this)(i, j) - (*this)(j, i)) / T(2);
            }
        }
        return result;
    }

    /**
     * @brief Compute deviatoric part (traceless tensor)
     * @return Deviatoric tensor (A - (1/dim)tr(A)I)
     */
    Tensor2 deviatoric() const {
        T trace_val = trace();
        Tensor2 result = *this;
        for (std::size_t i = 0; i < Dim; ++i) {
            result(i, i) -= trace_val / T(Dim);
        }
        return result;
    }

    /**
     * @brief Compute trace (first invariant I1)
     * @return Sum of diagonal elements
     */
    T trace() const {
        T result = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            result += (*this)(i, i);
        }
        return result;
    }

    /**
     * @brief Compute determinant (related to third invariant I3)
     * @return Determinant of tensor
     */
    T determinant() const {
        if constexpr (Dim == 2) {
            return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        } else { // Dim == 3
            return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1))
                 - (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0))
                 + (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
        }
    }

    /**
     * @brief Compute first invariant I1 = tr(A)
     * @return First invariant
     */
    T first_invariant() const {
        return trace();
    }

    /**
     * @brief Compute second invariant I2
     * @return Second invariant (for 3D: (I1^2 - tr(A^2))/2)
     */
    T second_invariant() const {
        if constexpr (Dim == 2) {
            return determinant();
        } else { // Dim == 3
            T I1 = trace();
            T tr_A2 = T(0);
            for (std::size_t i = 0; i < Dim; ++i) {
                for (std::size_t j = 0; j < Dim; ++j) {
                    tr_A2 += (*this)(i, j) * (*this)(j, i);
                }
            }
            return (I1 * I1 - tr_A2) / T(2);
        }
    }

    /**
     * @brief Compute third invariant I3 = det(A)
     * @return Third invariant
     */
    T third_invariant() const {
        return determinant();
    }

    /**
     * @brief Compute von Mises stress/strain (for deviatoric tensors)
     * @return Von Mises equivalent value
     */
    T von_mises() const {
        Tensor2 dev = deviatoric();
        T J2 = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                J2 += dev(i, j) * dev(i, j);
            }
        }
        J2 /= T(2);
        return std::sqrt(T(3) * J2);
    }

    /**
     * @brief Compute Frobenius norm
     * @return sqrt(A:A) = sqrt(sum of squared components)
     */
    T frobenius_norm() const {
        T sum = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                sum += square((*this)(i, j));
            }
        }
        return std::sqrt(sum);
    }

    /**
     * @brief Compute inverse (for non-singular tensors)
     * @return Inverse tensor
     */
    Tensor2 inverse() const {
        Matrix<T, Dim, Dim> mat = to_matrix();
        Matrix<T, Dim, Dim> inv = mat.inverse();
        return Tensor2::from_matrix(inv);
    }

    /**
     * @brief Double contraction with another rank-2 tensor: A:B
     * @param other Other tensor
     * @return Scalar result of A_ij * B_ij
     */
    T double_contract(const Tensor2& other) const {
        T result = T(0);
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                result += (*this)(i, j) * other(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Single contraction with vector: A·v
     * @param v Input vector
     * @return Result vector
     */
    Vector<T, Dim> contract(const Vector<T, Dim>& v) const {
        Vector<T, Dim> result;
        for (std::size_t i = 0; i < Dim; ++i) {
            T sum = T(0);
            for (std::size_t j = 0; j < Dim; ++j) {
                sum += (*this)(i, j) * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    /**
     * @brief Tensor product (dyadic product) of two vectors
     * @param u First vector
     * @param v Second vector
     * @return Rank-2 tensor u⊗v
     */
    static Tensor2 dyad(const Vector<T, Dim>& u, const Vector<T, Dim>& v) {
        Tensor2 result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                result(i, j) = u[i] * v[j];
            }
        }
        return result;
    }

    /**
     * @brief Compute tensor invariants
     * @return Array of invariants [I1, I2, I3] (I3 only for 3D)
     */
    std::array<T, Dim> invariants() const {
        std::array<T, Dim> inv;

        if constexpr (Dim == 2) {
            // First invariant: trace
            inv[0] = (*this)(0, 0) + (*this)(1, 1);
            // Second invariant: determinant
            inv[1] = (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        } else { // Dim == 3
            // First invariant: trace
            inv[0] = (*this)(0, 0) + (*this)(1, 1) + (*this)(2, 2);
            // Second invariant
            inv[1] = (*this)(0, 0) * (*this)(1, 1) + (*this)(1, 1) * (*this)(2, 2) +
                    (*this)(2, 2) * (*this)(0, 0) - (*this)(0, 1) * (*this)(0, 1) -
                    (*this)(1, 2) * (*this)(1, 2) - (*this)(0, 2) * (*this)(0, 2);
            // Third invariant: determinant
            inv[2] = (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
                    (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
                    (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
        }

        return inv;
    }

    /**
     * @brief Compute eigenvalues for symmetric tensor (analytical for 2D/3D)
     * @return Array of eigenvalues in descending order
     */
    std::array<T, Dim> eigenvalues() const {
        // Ensure tensor is symmetric
        std::array<T, Dim> eigvals;

        if constexpr (Dim == 2) {
            // Analytical solution for 2x2 symmetric matrix
            T a = (*this)(0, 0);
            T b = (*this)(0, 1);
            T c = (*this)(1, 1);

            T trace = a + c;
            T det = a * c - b * b;
            T discriminant = trace * trace - T(4) * det;

            if (discriminant < T(0)) discriminant = T(0);  // Handle numerical errors
            T sqrt_disc = std::sqrt(discriminant);

            eigvals[0] = (trace + sqrt_disc) / T(2);
            eigvals[1] = (trace - sqrt_disc) / T(2);
        } else { // Dim == 3
            // Analytical solution for 3x3 symmetric matrix using characteristic equation
            T a11 = (*this)(0, 0), a12 = (*this)(0, 1), a13 = (*this)(0, 2);
            T a22 = (*this)(1, 1), a23 = (*this)(1, 2);
            T a33 = (*this)(2, 2);

            // Coefficients of characteristic polynomial: -λ³ + I1λ² - I2λ + I3 = 0
            T I1 = a11 + a22 + a33;  // Trace
            T I2 = a11*a22 + a22*a33 + a33*a11 - a12*a12 - a23*a23 - a13*a13;
            T I3 = a11*(a22*a33 - a23*a23) - a12*(a12*a33 - a13*a23) + a13*(a12*a23 - a13*a22);

            // Use Cardano's method for cubic equation
            T p = I2 - I1*I1/T(3);
            T q = I1*I2/T(3) - I3 - T(2)*I1*I1*I1/T(27);

            T sqrt_val = q*q/T(4) + p*p*p/T(27);

            if (abs(sqrt_val) < epsilon<T>) {
                // Three real eigenvalues (repeated root case)
                // With discriminant Δ≈0, u=v=cbrt(-q/2) and roots are {2u, -u, -u}.
                T m = std::cbrt(-q / T(2));
                eigvals[0] = I1/T(3) + T(2)*m;
                eigvals[1] = I1/T(3) - m;
                eigvals[2] = I1/T(3) - m;
            } else if (sqrt_val > T(0)) {
                // One real root case (shouldn't happen for symmetric matrices)
                T sqrt_D = std::sqrt(sqrt_val);
                T u = std::cbrt(-q/T(2) + sqrt_D);
                T v = std::cbrt(-q/T(2) - sqrt_D);
                eigvals[0] = eigvals[1] = eigvals[2] = I1/T(3) + u + v;
            } else {
                // Three distinct real eigenvalues
                T rho = std::sqrt(-p*p*p/T(27));
                T theta = std::acos(-q/(T(2)*rho));
                T rho_cbrt = std::cbrt(rho);

                eigvals[0] = I1/T(3) + T(2)*rho_cbrt*std::cos(theta/T(3));
                eigvals[1] = I1/T(3) + T(2)*rho_cbrt*std::cos((theta + T(2)*Constants<T>::pi)/T(3));
                eigvals[2] = I1/T(3) + T(2)*rho_cbrt*std::cos((theta + T(4)*Constants<T>::pi)/T(3));
            }

            // Sort eigenvalues in descending order
            std::sort(eigvals.begin(), eigvals.end(), std::greater<T>());
        }

        return eigvals;
    }

    /**
     * @brief Compute principal values (eigenvalues) and directions (eigenvectors)
     * @return Pair of eigenvalue array and eigenvector matrix
     */
    std::pair<std::array<T, Dim>, Matrix<T, Dim, Dim>> eigen_decomposition() const {
        // Eigendecomposition is defined for symmetric tensors. We defensively
        // symmetrize the tensor before calling the specialized eigensolvers.
        Matrix<T, Dim, Dim> A;
        for (std::size_t i = 0; i < Dim; ++i) {
            A(i, i) = (*this)(i, i);
            for (std::size_t j = i + 1; j < Dim; ++j) {
                T avg = ((*this)(i, j) + (*this)(j, i)) / T(2);
                A(i, j) = avg;
                A(j, i) = avg;
            }
        }

        if constexpr (Dim == 2) {
            const auto [vals, vecs] = eigen_2x2_symmetric(A);
            return {{vals[0], vals[1]}, vecs};
        } else { // Dim == 3
            SymmetricEigen3x3<T> decomp(A);
            const auto& vals = decomp.eigenvalues();
            const auto& vecs = decomp.eigenvectors();
            // Return in descending order to match Tensor2::eigenvalues().
            Matrix<T, 3, 3> eigvecs;
            for (std::size_t i = 0; i < 3; ++i) {
                eigvecs(i, 0) = vecs(i, 2);
                eigvecs(i, 1) = vecs(i, 1);
                eigvecs(i, 2) = vecs(i, 0);
            }
            return {{vals[2], vals[1], vals[0]}, eigvecs};
        }
    }

    // Arithmetic operators
    Tensor2 operator+(const Tensor2& other) const {
        Tensor2 result;
        for (std::size_t i = 0; i < Base::size(); ++i) {
            result.data()[i] = data_[i] + other.data()[i];
        }
        return result;
    }

    Tensor2 operator-(const Tensor2& other) const {
        Tensor2 result;
        for (std::size_t i = 0; i < Base::size(); ++i) {
            result.data()[i] = data_[i] - other.data()[i];
        }
        return result;
    }

    Tensor2 operator*(T scalar) const {
        Tensor2 result;
        for (std::size_t i = 0; i < Base::size(); ++i) {
            result.data()[i] = data_[i] * scalar;
        }
        return result;
    }

    Tensor2 operator/(T scalar) const {
        Tensor2 result;
        for (std::size_t i = 0; i < Base::size(); ++i) {
            result.data()[i] = data_[i] / scalar;
        }
        return result;
    }

    Tensor2& operator+=(const Tensor2& other) {
        for (std::size_t i = 0; i < Base::size(); ++i) {
            data_[i] += other.data()[i];
        }
        return *this;
    }

    Tensor2& operator-=(const Tensor2& other) {
        for (std::size_t i = 0; i < Base::size(); ++i) {
            data_[i] -= other.data()[i];
        }
        return *this;
    }

    Tensor2& operator*=(T scalar) {
        for (std::size_t i = 0; i < Base::size(); ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    Tensor2& operator/=(T scalar) {
        for (std::size_t i = 0; i < Base::size(); ++i) {
            data_[i] /= scalar;
        }
        return *this;
    }
};

/**
 * @brief Rank-4 tensor for elasticity and material stiffness
 * @tparam T Scalar type (float, double)
 * @tparam Dim Spatial dimension (2 or 3)
 *
 * Provides operations for fourth-order tensors used in constitutive models,
 * particularly the elasticity tensor C_ijkl that relates stress and strain.
 */
template<typename T, std::size_t Dim>
class Tensor4 : public TensorBase<T, 4, Dim> {
    using Base = TensorBase<T, 4, Dim>;
    using Base::data_;

public:
    using Base::Base;  // Inherit constructors

    /**
     * @brief Create identity tensor (fourth-order identity)
     * @return Identity tensor I_ijkl = 0.5(δ_ik*δ_jl + δ_il*δ_jk)
     */
    static Tensor4 identity() {
        Tensor4 result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                for (std::size_t k = 0; k < Dim; ++k) {
                    for (std::size_t l = 0; l < Dim; ++l) {
                        if ((i == k && j == l) || (i == l && j == k)) {
                            result(i, j, k, l) = T(0.5);
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * @brief Create zero tensor
     * @return Zero tensor
     */
    static Tensor4 zero() {
        return Tensor4{};
    }

    /**
     * @brief Create symmetric identity tensor (for elasticity)
     * @return Symmetric identity I_ijkl = 0.5 * (δ_ik*δ_jl + δ_il*δ_jk)
     */
    static Tensor4 symmetric_identity() {
        Tensor4 result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                for (std::size_t k = 0; k < Dim; ++k) {
                    for (std::size_t l = 0; l < Dim; ++l) {
                        T delta_ik = (i == k) ? T(1) : T(0);
                        T delta_jl = (j == l) ? T(1) : T(0);
                        T delta_il = (i == l) ? T(1) : T(0);
                        T delta_jk = (j == k) ? T(1) : T(0);
                        result(i, j, k, l) = T(0.5) * (delta_ik * delta_jl + delta_il * delta_jk);
                    }
                }
            }
        }
        return result;
    }

    /**
     * @brief Double contraction with rank-2 tensor: C:ε
     * @param strain Strain tensor
     * @return Stress tensor σ = C:ε
     */
    Tensor2<T, Dim> double_contract(const Tensor2<T, Dim>& strain) const {
        Tensor2<T, Dim> result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                T sum = T(0);
                for (std::size_t k = 0; k < Dim; ++k) {
                    for (std::size_t l = 0; l < Dim; ++l) {
                        sum += (*this)(i, j, k, l) * strain(k, l);
                    }
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    /**
     * @brief Apply minor symmetries: C_ijkl = C_jikl = C_ijlk
     */
    void apply_minor_symmetries() {
        // Create a copy for reading original values
        Tensor4 original = *this;

        // Apply symmetrization by averaging all minor symmetric components
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                for (std::size_t k = 0; k < Dim; ++k) {
                    for (std::size_t l = 0; l < Dim; ++l) {
                        // Average all four minor symmetric components
                        T value = (original(i, j, k, l) + original(j, i, k, l) +
                                  original(i, j, l, k) + original(j, i, l, k)) / T(4);
                        (*this)(i, j, k, l) = value;
                    }
                }
            }
        }
    }

    /**
     * @brief Apply major symmetry: C_ijkl = C_klij
     */
    void apply_major_symmetry() {
        impose_major_symmetry();
    }

    /**
     * @brief Impose major symmetry: C_ijkl = C_klij
     */
    void impose_major_symmetry() {
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                for (std::size_t k = i; k < Dim; ++k) {
                    for (std::size_t l = (k == i ? j + 1 : 0); l < Dim; ++l) {
                        T avg = ((*this)(i, j, k, l) + (*this)(k, l, i, j)) / T(2);
                        (*this)(i, j, k, l) = avg;
                        (*this)(k, l, i, j) = avg;
                    }
                }
            }
        }
    }

    /**
     * @brief Impose minor symmetry: C_ijkl = C_jikl = C_ijlk
     */
    void impose_minor_symmetry() {
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                for (std::size_t k = 0; k < Dim; ++k) {
                    for (std::size_t l = 0; l < Dim; ++l) {
                        T avg = ((*this)(i, j, k, l) + (*this)(j, i, k, l) +
                                (*this)(i, j, l, k) + (*this)(j, i, l, k)) / T(4);
                        (*this)(i, j, k, l) = avg;
                        (*this)(j, i, k, l) = avg;
                        (*this)(i, j, l, k) = avg;
                        (*this)(j, i, l, k) = avg;
                    }
                }
            }
        }
    }

    /**
     * @brief Create isotropic elasticity tensor from Lamé parameters
     * @param lambda First Lamé parameter
     * @param mu Second Lamé parameter (shear modulus)
     * @return Isotropic elasticity tensor
     */
    static Tensor4 isotropic(T lambda, T mu) {
        Tensor4 result;
        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim; ++j) {
                for (std::size_t k = 0; k < Dim; ++k) {
                    for (std::size_t l = 0; l < Dim; ++l) {
                        T delta_ij = (i == j) ? T(1) : T(0);
                        T delta_kl = (k == l) ? T(1) : T(0);
                        T delta_ik = (i == k) ? T(1) : T(0);
                        T delta_jl = (j == l) ? T(1) : T(0);
                        T delta_il = (i == l) ? T(1) : T(0);
                        T delta_jk = (j == k) ? T(1) : T(0);

                        result(i, j, k, l) = lambda * delta_ij * delta_kl +
                                           mu * (delta_ik * delta_jl + delta_il * delta_jk);
                    }
                }
            }
        }
        return result;
    }

    /**
     * @brief Create transversely isotropic tensor (5 independent constants)
     * @param C11 Stiffness along fiber direction
     * @param C12 In-plane Poisson effect
     * @param C13 Out-of-plane Poisson effect
     * @param C33 Transverse stiffness
     * @param C44 Transverse shear modulus
     * @return Transversely isotropic elasticity tensor
     */
    static Tensor4 transversely_isotropic(T C11, T C12, T C13, T C33, T C44) {
        Tensor4 result;
        if constexpr (Dim == 3) {
            // Assuming z-axis (index 2) is the axis of symmetry
            T C66 = (C11 - C12) / T(2);  // In-plane shear modulus

            // Fill non-zero components
            result(0, 0, 0, 0) = C11;
            result(1, 1, 1, 1) = C11;
            result(2, 2, 2, 2) = C33;

            result(0, 0, 1, 1) = result(1, 1, 0, 0) = C12;
            result(0, 0, 2, 2) = result(2, 2, 0, 0) = C13;
            result(1, 1, 2, 2) = result(2, 2, 1, 1) = C13;

            result(0, 2, 0, 2) = result(0, 2, 2, 0) =
            result(2, 0, 0, 2) = result(2, 0, 2, 0) = C44;

            result(1, 2, 1, 2) = result(1, 2, 2, 1) =
            result(2, 1, 1, 2) = result(2, 1, 2, 1) = C44;

            result(0, 1, 0, 1) = result(0, 1, 1, 0) =
            result(1, 0, 0, 1) = result(1, 0, 1, 0) = C66;
        }
        return result;
    }

    // Arithmetic operators
    Tensor4 operator+(const Tensor4& other) const {
        Tensor4 result;
        for (std::size_t i = 0; i < Base::size(); ++i) {
            result.data()[i] = data_[i] + other.data()[i];
        }
        return result;
    }

    Tensor4 operator-(const Tensor4& other) const {
        Tensor4 result;
        for (std::size_t i = 0; i < Base::size(); ++i) {
            result.data()[i] = data_[i] - other.data()[i];
        }
        return result;
    }

    Tensor4 operator*(T scalar) const {
        Tensor4 result;
        for (std::size_t i = 0; i < Base::size(); ++i) {
            result.data()[i] = data_[i] * scalar;
        }
        return result;
    }

    Tensor4 operator/(T scalar) const {
        Tensor4 result;
        for (std::size_t i = 0; i < Base::size(); ++i) {
            result.data()[i] = data_[i] / scalar;
        }
        return result;
    }
};

// Non-member operators
template<typename T, std::size_t Dim>
Tensor2<T, Dim> operator*(T scalar, const Tensor2<T, Dim>& tensor) {
    return tensor * scalar;
}

template<typename T, std::size_t Dim>
Tensor4<T, Dim> operator*(T scalar, const Tensor4<T, Dim>& tensor) {
    return tensor * scalar;
}

// Type aliases for common dimensions
template<typename T> using Tensor2D = Tensor2<T, 2>;
template<typename T> using Tensor3D = Tensor2<T, 3>;
template<typename T> using Tensor4_2D = Tensor4<T, 2>;
template<typename T> using Tensor4_3D = Tensor4<T, 3>;

// =============================================================================
// Helper Functions for Tensor Operations
// =============================================================================

/**
 * @brief Create tensor product (dyadic product) of two vectors
 * @param u First vector
 * @param v Second vector
 * @return Rank-2 tensor u⊗v
 */
template<typename T, std::size_t Dim>
inline Tensor2<T, Dim> tensor_product(const Vector<T, Dim>& u, const Vector<T, Dim>& v) {
    return Tensor2<T, Dim>::dyad(u, v);
}

/**
 * @brief Compute symmetric part of a tensor: (A + A^T)/2
 * @param A Input tensor
 * @return Symmetric part of A
 */
template<typename T, std::size_t Dim>
inline Tensor2<T, Dim> symmetric_part(const Tensor2<T, Dim>& A) {
    Tensor2<T, Dim> result;
    for (std::size_t i = 0; i < Dim; ++i) {
        for (std::size_t j = 0; j < Dim; ++j) {
            result(i, j) = (A(i, j) + A(j, i)) / T(2);
        }
    }
    return result;
}

/**
 * @brief Compute skew-symmetric (antisymmetric) part of a tensor: (A - A^T)/2
 * @param A Input tensor
 * @return Skew-symmetric part of A
 */
template<typename T, std::size_t Dim>
inline Tensor2<T, Dim> skew_symmetric_part(const Tensor2<T, Dim>& A) {
    Tensor2<T, Dim> result;
    for (std::size_t i = 0; i < Dim; ++i) {
        for (std::size_t j = 0; j < Dim; ++j) {
            result(i, j) = (A(i, j) - A(j, i)) / T(2);
        }
    }
    return result;
}

/**
 * @brief Compute deviatoric part of a tensor: A - (tr(A)/dim)I
 * @param A Input tensor
 * @return Deviatoric part of A (zero trace)
 */
template<typename T, std::size_t Dim>
inline Tensor2<T, Dim> deviator(const Tensor2<T, Dim>& A) {
    T trace = A.trace();
    T spherical_value = trace / T(Dim);

    Tensor2<T, Dim> result = A;
    for (std::size_t i = 0; i < Dim; ++i) {
        result(i, i) -= spherical_value;
    }
    return result;
}

/**
 * @brief Compute spherical (hydrostatic) part of a tensor: (tr(A)/dim)I
 * @param A Input tensor
 * @return Spherical part of A
 */
template<typename T, std::size_t Dim>
inline Tensor2<T, Dim> spherical_part(const Tensor2<T, Dim>& A) {
    T trace = A.trace();
    T spherical_value = trace / T(Dim);

    Tensor2<T, Dim> result = Tensor2<T, Dim>::zeros();
    for (std::size_t i = 0; i < Dim; ++i) {
        result(i, i) = spherical_value;
    }
    return result;
}

/**
 * @brief Compute Frobenius norm of a tensor: sqrt(A:A)
 * @param A Input tensor
 * @return Frobenius norm ||A||_F
 */
template<typename T, std::size_t Dim>
inline T frobenius_norm(const Tensor2<T, Dim>& A) {
    T sum = T(0);
    for (std::size_t i = 0; i < Dim; ++i) {
        for (std::size_t j = 0; j < Dim; ++j) {
            sum += A(i, j) * A(i, j);
        }
    }
    return std::sqrt(sum);
}

/**
 * @brief Compute von Mises stress from a stress tensor
 * @param stress Stress tensor (3x3)
 * @return von Mises equivalent stress
 */
template<typename T>
inline T von_mises_stress(const Tensor2<T, 3>& stress) {
    // Compute deviatoric stress
    Tensor2<T, 3> dev = deviator(stress);

    // von Mises stress = sqrt(3/2 * dev:dev)
    T dev_double_contract = dev.double_contract(dev);
    return std::sqrt(T(3.0/2.0) * dev_double_contract);
}

/**
 * @brief Compute von Mises stress from principal stresses
 * @param s1 First principal stress
 * @param s2 Second principal stress
 * @param s3 Third principal stress
 * @return von Mises equivalent stress
 */
template<typename T>
inline T von_mises_stress(T s1, T s2, T s3) {
    T s12 = s1 - s2;
    T s23 = s2 - s3;
    T s31 = s3 - s1;
    return std::sqrt(T(0.5) * (s12*s12 + s23*s23 + s31*s31));
}

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_TENSOR_H
