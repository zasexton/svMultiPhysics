#ifndef SVMP_FE_MATH_VOIGT_NOTATION_H
#define SVMP_FE_MATH_VOIGT_NOTATION_H

/**
 * @file VoigtNotation.h
 * @brief Voigt notation conversions for stress/strain tensors in FE computations
 *
 * This header provides conversion utilities between tensor notation and Voigt
 * (vector) notation commonly used in computational mechanics. Supports both
 * 2D (3-component) and 3D (6-component) representations with proper handling
 * of engineering vs tensor strain conventions.
 */

#include "Vector.h"
#include "Matrix.h"
#include "Tensor.h"
#include <array>
#include <cmath>
#include <type_traits>

namespace svmp {
namespace FE {
namespace math {

/**
 * @brief Voigt notation for 3D symmetric rank-2 tensors (6 components)
 * @tparam T Scalar type (float, double)
 *
 * Converts between symmetric 3x3 tensors and 6-component Voigt vectors.
 * Standard ordering: [σ11, σ22, σ33, σ12, σ23, σ13] or [σxx, σyy, σzz, σxy, σyz, σxz]
 * Handles engineering strain convention (γ = 2ε for shear strains).
 */
template<typename T>
class VoigtVector6 {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

private:
    Vector<T, 6> data_;  // [σ11, σ22, σ33, σ12, σ23, σ13]

    // Index mapping from tensor indices to Voigt index
    static constexpr std::size_t voigt_index(std::size_t i, std::size_t j) {
        // Map symmetric tensor indices to Voigt notation
        // Standard mechanics convention: (0,0)->0, (1,1)->1, (2,2)->2, (0,1)->3, (1,2)->4, (0,2)->5
        if (i == j) {
            return i;  // Diagonal terms
        }
        // Off-diagonal terms (ensure i < j for consistency)
        std::size_t ii = (i < j) ? i : j;
        std::size_t jj = (i < j) ? j : i;
        if (ii == 0 && jj == 1) return 3;
        if (ii == 1 && jj == 2) return 4;
        if (ii == 0 && jj == 2) return 5;
        return 0;  // Should not reach here
    }

public:
    /**
     * @brief Default constructor - zero initializes
     */
    constexpr VoigtVector6() : data_{} {}

    /**
     * @brief Constructor from 6 components
     * @param s11 Normal stress/strain in x-direction
     * @param s22 Normal stress/strain in y-direction
     * @param s33 Normal stress/strain in z-direction
     * @param s12 Shear stress/strain in xy-plane
     * @param s23 Shear stress/strain in yz-plane
     * @param s13 Shear stress/strain in xz-plane
     */
    constexpr VoigtVector6(T s11, T s22, T s33, T s12, T s23, T s13)
        : data_{s11, s22, s33, s12, s23, s13} {}

    /**
     * @brief Constructor from vector
     * @param vec 6-component vector in Voigt ordering
     */
    explicit VoigtVector6(const Vector<T, 6>& vec) : data_(vec) {}

    /**
     * @brief Constructor from symmetric 3x3 tensor (stress notation)
     * @param tensor Symmetric rank-2 tensor
     * @param is_strain If true, applies engineering strain factor to shear components
     */
    explicit VoigtVector6(const Tensor2<T, 3>& tensor, bool is_strain = false) {
        // Normal components
        data_[0] = tensor(0, 0);  // σ11
        data_[1] = tensor(1, 1);  // σ22
        data_[2] = tensor(2, 2);  // σ33

        // Shear components - Standard Voigt ordering: [σ11, σ22, σ33, σ12, σ23, σ13]
        T factor = is_strain ? T(2) : T(1);  // Engineering strain convention
        data_[3] = tensor(0, 1) * factor;  // σ12 (or 2*ε12 for strain)
        data_[4] = tensor(1, 2) * factor;  // σ23 (or 2*ε23 for strain)
        data_[5] = tensor(0, 2) * factor;  // σ13 (or 2*ε13 for strain)
    }

    /**
     * @brief Constructor from symmetric 3x3 matrix (stress notation)
     * @param matrix Symmetric 3x3 matrix
     * @param is_strain If true, applies engineering strain factor to shear components
     */
    explicit VoigtVector6(const Matrix<T, 3, 3>& matrix, bool is_strain = false) {
        // Normal components
        data_[0] = matrix(0, 0);  // σ11
        data_[1] = matrix(1, 1);  // σ22
        data_[2] = matrix(2, 2);  // σ33

        // Shear components - Standard Voigt ordering: [σ11, σ22, σ33, σ12, σ23, σ13]
        T factor = is_strain ? T(2) : T(1);  // Engineering strain convention
        data_[3] = matrix(0, 1) * factor;  // σ12 (or 2*ε12 for strain)
        data_[4] = matrix(1, 2) * factor;  // σ23 (or 2*ε23 for strain)
        data_[5] = matrix(0, 2) * factor;  // σ13 (or 2*ε13 for strain)
    }

    /**
     * @brief Convert to symmetric 3x3 tensor
     * @param is_strain If true, divides shear components by 2 (engineering to tensor strain)
     * @return Symmetric rank-2 tensor
     */
    Tensor2<T, 3> to_tensor(bool is_strain = false) const {
        Tensor2<T, 3> result;

        // Normal components
        result(0, 0) = data_[0];  // σ11
        result(1, 1) = data_[1];  // σ22
        result(2, 2) = data_[2];  // σ33

        // Shear components (symmetric) - Standard Voigt ordering: [σ11, σ22, σ33, σ12, σ23, σ13]
        T factor = is_strain ? T(0.5) : T(1);  // Convert engineering to tensor strain
        result(0, 1) = result(1, 0) = data_[3] * factor;  // σ12
        result(1, 2) = result(2, 1) = data_[4] * factor;  // σ23
        result(0, 2) = result(2, 0) = data_[5] * factor;  // σ13

        return result;
    }

    /**
     * @brief Convert to symmetric 3x3 matrix
     * @param is_strain If true, divides shear components by 2 (engineering to tensor strain)
     * @return Symmetric 3x3 matrix
     */
    Matrix<T, 3, 3> to_matrix(bool is_strain = false) const {
        Matrix<T, 3, 3> result;

        // Normal components
        result(0, 0) = data_[0];  // σ11
        result(1, 1) = data_[1];  // σ22
        result(2, 2) = data_[2];  // σ33

        // Shear components (symmetric) - Standard Voigt ordering: [σ11, σ22, σ33, σ12, σ23, σ13]
        T factor = is_strain ? T(0.5) : T(1);  // Convert engineering to tensor strain
        result(0, 1) = result(1, 0) = data_[3] * factor;  // σ12
        result(1, 2) = result(2, 1) = data_[4] * factor;  // σ23
        result(0, 2) = result(2, 0) = data_[5] * factor;  // σ13

        return result;
    }

    /**
     * @brief Apply engineering strain factor (multiply shear components by 2)
     */
    void apply_engineering_strain_factor() {
        data_[3] *= T(2);  // γ12 = 2*ε12
        data_[4] *= T(2);  // γ23 = 2*ε23
        data_[5] *= T(2);  // γ13 = 2*ε13
    }

    /**
     * @brief Remove engineering strain factor (divide shear components by 2)
     */
    void remove_engineering_strain_factor() {
        data_[3] /= T(2);  // ε12 = γ12/2
        data_[4] /= T(2);  // ε23 = γ23/2
        data_[5] /= T(2);  // ε13 = γ13/2
    }

    /**
     * @brief Access component by index
     * @param i Index (0-5)
     * @return Reference to component
     */
    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    /**
     * @brief Get underlying vector
     * @return 6-component vector
     */
    Vector<T, 6>& vector() { return data_; }
    const Vector<T, 6>& vector() const { return data_; }

    /**
     * @brief Compute von Mises equivalent stress/strain
     * @param is_strain If true, accounts for engineering strain convention
     * @return Von Mises equivalent value
     */
    T von_mises(bool is_strain = false) const {
        T s11 = data_[0], s22 = data_[1], s33 = data_[2];
        T s12 = data_[3], s23 = data_[4], s13 = data_[5];

        // For engineering strains, convert to tensor strains for calculation
        if (is_strain) {
            s12 /= T(2);
            s23 /= T(2);
            s13 /= T(2);
        }

        // Von Mises stress: sqrt(3/2 * sij:sij) where sij is deviatoric part
        T mean = (s11 + s22 + s33) / T(3);
        T d11 = s11 - mean;
        T d22 = s22 - mean;
        T d33 = s33 - mean;

        T vm_squared = d11*d11 + d22*d22 + d33*d33 + T(2)*(s12*s12 + s23*s23 + s13*s13);
        return std::sqrt(T(3)/T(2) * vm_squared);
    }

    /**
     * @brief Compute hydrostatic (mean) stress/strain
     * @return Mean of normal components
     */
    T hydrostatic() const {
        return (data_[0] + data_[1] + data_[2]) / T(3);
    }

    /**
     * @brief Compute deviatoric part
     * @return Voigt vector with hydrostatic part removed
     */
    VoigtVector6 deviatoric() const {
        T mean = hydrostatic();
        VoigtVector6 result = *this;
        result[0] -= mean;
        result[1] -= mean;
        result[2] -= mean;
        return result;
    }

    // Arithmetic operators
    VoigtVector6 operator+(const VoigtVector6& other) const {
        return VoigtVector6(data_ + other.data_);
    }

    VoigtVector6 operator-(const VoigtVector6& other) const {
        return VoigtVector6(data_ - other.data_);
    }

    VoigtVector6 operator*(T scalar) const {
        return VoigtVector6(data_ * scalar);
    }

    VoigtVector6 operator/(T scalar) const {
        return VoigtVector6(data_ / scalar);
    }

    VoigtVector6& operator+=(const VoigtVector6& other) {
        data_ += other.data_;
        return *this;
    }

    VoigtVector6& operator-=(const VoigtVector6& other) {
        data_ -= other.data_;
        return *this;
    }

    VoigtVector6& operator*=(T scalar) {
        data_ *= scalar;
        return *this;
    }

    VoigtVector6& operator/=(T scalar) {
        data_ /= scalar;
        return *this;
    }
};

/**
 * @brief Voigt notation for 3D rank-4 tensors (6x6 matrix)
 * @tparam T Scalar type (float, double)
 *
 * Converts between rank-4 elasticity tensors and 6x6 Voigt matrices.
 * Used for constitutive relations: σ = C:ε in Voigt form becomes {σ} = [C]{ε}
 */
template<typename T>
class VoigtMatrix6x6 {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

private:
    Matrix<T, 6, 6> data_;

    // Map tensor indices (i,j,k,l) to Voigt matrix indices (I,J)
    static constexpr std::size_t voigt_index(std::size_t i, std::size_t j) {
        if (i == j) return i;
        if ((i == 0 && j == 1) || (i == 1 && j == 0)) return 3;
        if ((i == 1 && j == 2) || (i == 2 && j == 1)) return 4;
        if ((i == 0 && j == 2) || (i == 2 && j == 0)) return 5;
        return 0;
    }

public:
    /**
     * @brief Default constructor - zero initializes
     */
    constexpr VoigtMatrix6x6() : data_{} {}

    /**
     * @brief Constructor from 6x6 matrix
     * @param mat 6x6 stiffness/compliance matrix
     */
    explicit VoigtMatrix6x6(const Matrix<T, 6, 6>& mat) : data_(mat) {}

    /**
     * @brief Constructor from rank-4 tensor
     * @param tensor Rank-4 elasticity tensor
     * @param is_strain If true, applies engineering strain factors
     */
    explicit VoigtMatrix6x6(const Tensor4<T, 3>& tensor, bool is_strain = false) {
        // Fill Voigt matrix from tensor components
        // For stiffness tensor: σ = C:ε (tensor form) → σ_voigt = D * γ_voigt (engineering strain)
        // Engineering shear strain: γ_ij = 2*ε_ij, so we need to DIVIDE by factors
        T factor_off = is_strain ? T(0.5) : T(1);      // Divide by 2 for one shear index
        T factor_shear = is_strain ? T(0.25) : T(1);   // Divide by 4 for two shear indices

        // Normal-normal terms
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                data_(i, j) = tensor(i, i, j, j);
            }
        }

        // Normal-shear terms (Standard Voigt: indices 3=xy, 4=yz, 5=xz)
        for (std::size_t i = 0; i < 3; ++i) {
            data_(i, 3) = tensor(i, i, 0, 1) * factor_off;  // σii from γ12 (divide by 2)
            data_(i, 4) = tensor(i, i, 1, 2) * factor_off;  // σii from γ23 (divide by 2)
            data_(i, 5) = tensor(i, i, 0, 2) * factor_off;  // σii from γ13 (divide by 2)

            data_(3, i) = tensor(0, 1, i, i) * factor_off;  // σ12 from εii (divide by 2)
            data_(4, i) = tensor(1, 2, i, i) * factor_off;  // σ23 from εii (divide by 2)
            data_(5, i) = tensor(0, 2, i, i) * factor_off;  // σ13 from εii (divide by 2)
        }

        // Shear-shear terms (Standard Voigt: indices 3=xy, 4=yz, 5=xz)
        data_(3, 3) = tensor(0, 1, 0, 1) * factor_shear;  // σ12 from γ12 (divide by 4)
        data_(3, 4) = tensor(0, 1, 1, 2) * factor_shear;  // σ12 from γ23 (divide by 4)
        data_(3, 5) = tensor(0, 1, 0, 2) * factor_shear;  // σ12 from γ13 (divide by 4)

        data_(4, 3) = tensor(1, 2, 0, 1) * factor_shear;  // σ23 from γ12 (divide by 4)
        data_(4, 4) = tensor(1, 2, 1, 2) * factor_shear;  // σ23 from γ23 (divide by 4)
        data_(4, 5) = tensor(1, 2, 0, 2) * factor_shear;  // σ23 from γ13 (divide by 4)

        data_(5, 3) = tensor(0, 2, 0, 1) * factor_shear;  // σ13 from γ12 (divide by 4)
        data_(5, 4) = tensor(0, 2, 1, 2) * factor_shear;  // σ13 from γ23 (divide by 4)
        data_(5, 5) = tensor(0, 2, 0, 2) * factor_shear;  // σ13 from γ13 (divide by 4)
    }

    /**
     * @brief Convert to rank-4 tensor
     * @param is_strain If true, accounts for engineering strain factors
     * @return Rank-4 elasticity tensor
     */
    Tensor4<T, 3> to_tensor(bool is_strain = false) const {
        Tensor4<T, 3> result;

        // Map Voigt matrix to tensor with proper symmetries
        // Standard Voigt: indices 0=xx, 1=yy, 2=zz, 3=xy, 4=yz, 5=xz
        std::array<std::pair<std::size_t, std::size_t>, 6> voigt_map = {
            {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {0, 2}}
        };

        for (std::size_t I = 0; I < 6; ++I) {
            for (std::size_t J = 0; J < 6; ++J) {
                auto [i, j] = voigt_map[I];
                auto [k, l] = voigt_map[J];

                T value = data_(I, J);

                // Apply factors for engineering strain convention (reverse of constructor)
                // In constructor we divided, so here we multiply to get back tensor components
                if (is_strain) {
                    if (I >= 3 && J < 3) value *= T(2);
                    else if (I < 3 && J >= 3) value *= T(2);
                    else if (I >= 3 && J >= 3) value *= T(4);
                }

                // Fill with symmetries
                result(i, j, k, l) = value;
                result(i, j, l, k) = value;
                result(j, i, k, l) = value;
                result(j, i, l, k) = value;
            }
        }

        return result;
    }

    /**
     * @brief Matrix-vector multiplication: stress = C * strain
     * @param strain Strain in Voigt notation
     * @return Stress in Voigt notation
     */
    VoigtVector6<T> operator*(const VoigtVector6<T>& strain) const {
        Vector<T, 6> result = data_ * strain.vector();
        return VoigtVector6<T>(result);
    }

    /**
     * @brief Access matrix element
     * @param i Row index (0-5)
     * @param j Column index (0-5)
     * @return Reference to element
     */
    T& operator()(std::size_t i, std::size_t j) { return data_(i, j); }
    const T& operator()(std::size_t i, std::size_t j) const { return data_(i, j); }

    /**
     * @brief Get underlying matrix
     * @return 6x6 matrix
     */
    Matrix<T, 6, 6>& matrix() { return data_; }
    const Matrix<T, 6, 6>& matrix() const { return data_; }

    /**
     * @brief Create isotropic stiffness matrix from elastic constants
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @return Isotropic stiffness matrix in Voigt form
     */
    static VoigtMatrix6x6 isotropic(T E, T nu) {
        VoigtMatrix6x6 result;
        T factor = E / ((T(1) + nu) * (T(1) - T(2)*nu));
        T lambda = factor * nu;
        T mu = E / (T(2) * (T(1) + nu));

        // Fill diagonal terms
        result(0, 0) = result(1, 1) = result(2, 2) = lambda + T(2)*mu;
        result(3, 3) = result(4, 4) = result(5, 5) = mu;

        // Fill off-diagonal terms
        result(0, 1) = result(0, 2) = result(1, 0) = lambda;
        result(1, 2) = result(2, 0) = result(2, 1) = lambda;

        return result;
    }

    /**
     * @brief Create orthotropic stiffness matrix (9 independent constants)
     * @param E1 Young's modulus in direction 1
     * @param E2 Young's modulus in direction 2
     * @param E3 Young's modulus in direction 3
     * @param nu12 Poisson's ratio 12
     * @param nu13 Poisson's ratio 13
     * @param nu23 Poisson's ratio 23
     * @param G12 Shear modulus 12
     * @param G13 Shear modulus 13
     * @param G23 Shear modulus 23
     * @return Orthotropic stiffness matrix
     */
    static VoigtMatrix6x6 orthotropic(T E1, T E2, T E3,
                                      T nu12, T nu13, T nu23,
                                      T G12, T G13, T G23) {
        VoigtMatrix6x6 result;

        // Compute compliance matrix first
        T nu21 = nu12 * E2 / E1;
        T nu31 = nu13 * E3 / E1;
        T nu32 = nu23 * E3 / E2;

        T delta = T(1) - nu12*nu21 - nu23*nu32 - nu31*nu13 - T(2)*nu21*nu32*nu13;
        T factor = T(1) / delta;

        result(0, 0) = E1 * (T(1) - nu23*nu32) * factor;
        result(1, 1) = E2 * (T(1) - nu13*nu31) * factor;
        result(2, 2) = E3 * (T(1) - nu12*nu21) * factor;

        result(0, 1) = result(1, 0) = E1 * (nu21 + nu31*nu23) * factor;
        result(0, 2) = result(2, 0) = E1 * (nu31 + nu21*nu32) * factor;
        result(1, 2) = result(2, 1) = E2 * (nu32 + nu12*nu31) * factor;

        // Shear terms follow the VoigtVector6 ordering: [12, 23, 13]
        result(3, 3) = G12;
        result(4, 4) = G23;
        result(5, 5) = G13;

        return result;
    }
};

/**
 * @brief Voigt notation for 2D symmetric rank-2 tensors (3 components)
 * @tparam T Scalar type (float, double)
 *
 * Converts between symmetric 2x2 tensors and 3-component Voigt vectors.
 * Standard ordering: [σ11, σ22, σ12] or [σxx, σyy, σxy]
 * Used for plane stress/plane strain problems.
 */
template<typename T>
class VoigtVector3 {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

private:
    Vector<T, 3> data_;  // [σ11, σ22, σ12]

public:
    /**
     * @brief Default constructor - zero initializes
     */
    constexpr VoigtVector3() : data_{} {}

    /**
     * @brief Constructor from 3 components
     * @param s11 Normal stress/strain in x-direction
     * @param s22 Normal stress/strain in y-direction
     * @param s12 Shear stress/strain in xy-plane
     */
    constexpr VoigtVector3(T s11, T s22, T s12)
        : data_{s11, s22, s12} {}

    /**
     * @brief Constructor from vector
     * @param vec 3-component vector in Voigt ordering
     */
    explicit VoigtVector3(const Vector<T, 3>& vec) : data_(vec) {}

    /**
     * @brief Constructor from symmetric 2x2 tensor
     * @param tensor Symmetric rank-2 tensor
     * @param is_strain If true, applies engineering strain factor to shear component
     */
    explicit VoigtVector3(const Tensor2<T, 2>& tensor, bool is_strain = false) {
        data_[0] = tensor(0, 0);  // σ11
        data_[1] = tensor(1, 1);  // σ22
        T factor = is_strain ? T(2) : T(1);
        data_[2] = tensor(0, 1) * factor;  // σ12 (or 2*ε12 for strain)
    }

    /**
     * @brief Constructor from symmetric 2x2 matrix
     * @param matrix Symmetric 2x2 matrix
     * @param is_strain If true, applies engineering strain factor to shear component
     */
    explicit VoigtVector3(const Matrix<T, 2, 2>& matrix, bool is_strain = false) {
        data_[0] = matrix(0, 0);  // σ11
        data_[1] = matrix(1, 1);  // σ22
        T factor = is_strain ? T(2) : T(1);
        data_[2] = matrix(0, 1) * factor;  // σ12 (or 2*ε12 for strain)
    }

    /**
     * @brief Convert to symmetric 2x2 tensor
     * @param is_strain If true, divides shear component by 2 (engineering to tensor strain)
     * @return Symmetric rank-2 tensor
     */
    Tensor2<T, 2> to_tensor(bool is_strain = false) const {
        Tensor2<T, 2> result;
        result(0, 0) = data_[0];  // σ11
        result(1, 1) = data_[1];  // σ22
        T factor = is_strain ? T(0.5) : T(1);
        result(0, 1) = result(1, 0) = data_[2] * factor;  // σ12
        return result;
    }

    /**
     * @brief Convert to symmetric 2x2 matrix
     * @param is_strain If true, divides shear component by 2 (engineering to tensor strain)
     * @return Symmetric 2x2 matrix
     */
    Matrix<T, 2, 2> to_matrix(bool is_strain = false) const {
        Matrix<T, 2, 2> result;
        result(0, 0) = data_[0];  // σ11
        result(1, 1) = data_[1];  // σ22
        T factor = is_strain ? T(0.5) : T(1);
        result(0, 1) = result(1, 0) = data_[2] * factor;  // σ12
        return result;
    }

    /**
     * @brief Apply engineering strain factor (multiply shear component by 2)
     */
    void apply_engineering_strain_factor() {
        data_[2] *= T(2);  // γ12 = 2*ε12
    }

    /**
     * @brief Remove engineering strain factor (divide shear component by 2)
     */
    void remove_engineering_strain_factor() {
        data_[2] /= T(2);  // ε12 = γ12/2
    }

    /**
     * @brief Access component by index
     * @param i Index (0-2)
     * @return Reference to component
     */
    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    /**
     * @brief Get underlying vector
     * @return 3-component vector
     */
    Vector<T, 3>& vector() { return data_; }
    const Vector<T, 3>& vector() const { return data_; }

    /**
     * @brief Compute von Mises equivalent stress/strain
     * @param is_strain If true, accounts for engineering strain convention
     * @return Von Mises equivalent value
     */
    T von_mises(bool is_strain = false) const {
        T s11 = data_[0], s22 = data_[1], s12 = data_[2];

        if (is_strain) {
            s12 /= T(2);  // Convert engineering to tensor strain
        }

        // For 2D: von Mises = sqrt(s11^2 - s11*s22 + s22^2 + 3*s12^2)
        return std::sqrt(s11*s11 - s11*s22 + s22*s22 + T(3)*s12*s12);
    }

    /**
     * @brief Compute hydrostatic (mean) stress/strain
     * @return Mean of normal components
     */
    T hydrostatic() const {
        return (data_[0] + data_[1]) / T(2);
    }

    /**
     * @brief Compute deviatoric part
     * @return Voigt vector with hydrostatic part removed
     */
    VoigtVector3 deviatoric() const {
        T mean = hydrostatic();
        VoigtVector3 result = *this;
        result[0] -= mean;
        result[1] -= mean;
        return result;
    }

    // Arithmetic operators
    VoigtVector3 operator+(const VoigtVector3& other) const {
        return VoigtVector3(data_ + other.data_);
    }

    VoigtVector3 operator-(const VoigtVector3& other) const {
        return VoigtVector3(data_ - other.data_);
    }

    VoigtVector3 operator*(T scalar) const {
        return VoigtVector3(data_ * scalar);
    }

    VoigtVector3 operator/(T scalar) const {
        return VoigtVector3(data_ / scalar);
    }
};

/**
 * @brief Voigt notation for 2D rank-4 tensors (3x3 matrix)
 * @tparam T Scalar type (float, double)
 *
 * Converts between rank-4 elasticity tensors and 3x3 Voigt matrices
 * for plane stress/plane strain problems.
 */
template<typename T>
class VoigtMatrix3x3 {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

private:
    Matrix<T, 3, 3> data_;

public:
    /**
     * @brief Default constructor - zero initializes
     */
    constexpr VoigtMatrix3x3() : data_{} {}

    /**
     * @brief Constructor from 3x3 matrix
     * @param mat 3x3 stiffness/compliance matrix
     */
    explicit VoigtMatrix3x3(const Matrix<T, 3, 3>& mat) : data_(mat) {}

    /**
     * @brief Matrix-vector multiplication: stress = C * strain
     * @param strain Strain in Voigt notation
     * @return Stress in Voigt notation
     */
    VoigtVector3<T> operator*(const VoigtVector3<T>& strain) const {
        Vector<T, 3> result = data_ * strain.vector();
        return VoigtVector3<T>(result);
    }

    /**
     * @brief Access matrix element
     * @param i Row index (0-2)
     * @param j Column index (0-2)
     * @return Reference to element
     */
    T& operator()(std::size_t i, std::size_t j) { return data_(i, j); }
    const T& operator()(std::size_t i, std::size_t j) const { return data_(i, j); }

    /**
     * @brief Get underlying matrix
     * @return 3x3 matrix
     */
    Matrix<T, 3, 3>& matrix() { return data_; }
    const Matrix<T, 3, 3>& matrix() const { return data_; }

    /**
     * @brief Create plane stress stiffness matrix
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @return Plane stress stiffness matrix
     */
    static VoigtMatrix3x3 plane_stress(T E, T nu) {
        VoigtMatrix3x3 result;
        T factor = E / (T(1) - nu*nu);

        result(0, 0) = factor;
        result(1, 1) = factor;
        result(0, 1) = result(1, 0) = factor * nu;
        result(2, 2) = factor * (T(1) - nu) / T(2);

        return result;
    }

    /**
     * @brief Create plane strain stiffness matrix
     * @param E Young's modulus
     * @param nu Poisson's ratio
     * @return Plane strain stiffness matrix
     */
    static VoigtMatrix3x3 plane_strain(T E, T nu) {
        VoigtMatrix3x3 result;
        T factor = E / ((T(1) + nu) * (T(1) - T(2)*nu));

        result(0, 0) = result(1, 1) = factor * (T(1) - nu);
        result(0, 1) = result(1, 0) = factor * nu;
        result(2, 2) = factor * (T(1) - T(2)*nu) / T(2);

        return result;
    }
};

// Non-member operators
template<typename T>
VoigtVector6<T> operator*(T scalar, const VoigtVector6<T>& vec) {
    return vec * scalar;
}

template<typename T>
VoigtVector3<T> operator*(T scalar, const VoigtVector3<T>& vec) {
    return vec * scalar;
}

// Free function wrappers for convenient tensor-Voigt conversions

/**
 * @brief Convert 3D tensor to Voigt stress vector
 * @tparam T Scalar type
 * @param tensor Symmetric rank-2 tensor
 * @return 6-component Voigt vector (no engineering strain factor)
 */
template<typename T>
inline Vector<T, 6> tensor_to_voigt_stress(const Tensor2<T, 3>& tensor) {
    VoigtVector6<T> voigt(tensor, false);  // false = stress (no factor of 2)
    return voigt.vector();
}

/**
 * @brief Convert 3D tensor to Voigt strain vector (engineering convention)
 * @tparam T Scalar type
 * @param tensor Symmetric rank-2 tensor
 * @return 6-component Voigt vector (with engineering strain factor γ = 2ε)
 */
template<typename T>
inline Vector<T, 6> tensor_to_voigt_strain(const Tensor2<T, 3>& tensor) {
    VoigtVector6<T> voigt(tensor, true);  // true = strain (factor of 2)
    return voigt.vector();
}

/**
 * @brief Convert Voigt vector to 3D stress tensor
 * @tparam T Scalar type
 * @param voigt_vec 6-component Voigt vector
 * @return Symmetric rank-2 tensor
 */
template<typename T>
inline Tensor2<T, 3> voigt_to_tensor_stress(const Vector<T, 6>& voigt_vec) {
    VoigtVector6<T> voigt(voigt_vec);
    return voigt.to_tensor(false);
}

/**
 * @brief Convert Voigt vector to 3D strain tensor (engineering to tensor)
 * @tparam T Scalar type
 * @param voigt_vec 6-component Voigt vector (engineering strain)
 * @return Symmetric rank-2 tensor (tensor strain ε = γ/2)
 */
template<typename T>
inline Tensor2<T, 3> voigt_to_tensor_strain(const Vector<T, 6>& voigt_vec) {
    VoigtVector6<T> voigt(voigt_vec);
    return voigt.to_tensor(true);
}

/**
 * @brief Convert 2D tensor to Voigt stress vector
 * @tparam T Scalar type
 * @param tensor Symmetric rank-2 tensor (2D)
 * @return 3-component Voigt vector
 */
template<typename T>
inline Vector<T, 3> tensor_to_voigt_stress_2d(const Tensor2<T, 2>& tensor) {
    VoigtVector3<T> voigt(tensor, false);
    return voigt.vector();
}

/**
 * @brief Convert rank-4 tensor to Voigt stiffness matrix
 * @tparam T Scalar type
 * @param tensor Rank-4 elasticity tensor
 * @return 6x6 Voigt matrix
 */
template<typename T>
inline Matrix<T, 6, 6> tensor4_to_voigt_stiffness(const Tensor4<T, 3>& tensor) {
    VoigtMatrix6x6<T> voigt(tensor, false);
    return voigt.matrix();
}

/**
 * @brief Convert Voigt stiffness matrix to rank-4 tensor
 * @tparam T Scalar type
 * @param matrix 6x6 stiffness matrix
 * @return Rank-4 elasticity tensor
 */
template<typename T>
inline Tensor4<T, 3> voigt_to_tensor4_stiffness(const Matrix<T, 6, 6>& matrix) {
    VoigtMatrix6x6<T> voigt(matrix);
    return voigt.to_tensor(false);
}

/**
 * @brief Create isotropic stiffness matrix from elastic constants
 * @tparam T Scalar type
 * @param E Young's modulus
 * @param nu Poisson's ratio
 * @return 6x6 isotropic stiffness matrix
 */
template<typename T>
inline Matrix<T, 6, 6> isotropic_stiffness_matrix(T E, T nu) {
    return VoigtMatrix6x6<T>::isotropic(E, nu).matrix();
}

/**
 * @brief Create plane stress stiffness matrix
 * @tparam T Scalar type
 * @param E Young's modulus
 * @param nu Poisson's ratio
 * @return 3x3 plane stress stiffness matrix
 */
template<typename T>
inline Matrix<T, 3, 3> plane_stress_stiffness(T E, T nu) {
    return VoigtMatrix3x3<T>::plane_stress(E, nu).matrix();
}

/**
 * @brief Create plane strain stiffness matrix
 * @tparam T Scalar type
 * @param E Young's modulus
 * @param nu Poisson's ratio
 * @return 3x3 plane strain stiffness matrix
 */
template<typename T>
inline Matrix<T, 3, 3> plane_strain_stiffness(T E, T nu) {
    return VoigtMatrix3x3<T>::plane_strain(E, nu).matrix();
}

/**
 * @brief Map Voigt index to tensor indices
 * @param voigt_index Voigt index (0-5)
 * @return Pair of tensor indices (i, j)
 *
 * Mapping: 0→(0,0), 1→(1,1), 2→(2,2), 3→(0,1), 4→(1,2), 5→(0,2)
 */
inline std::pair<std::size_t, std::size_t> voigt_to_tensor_index(std::size_t voigt_index) {
    static const std::pair<std::size_t, std::size_t> map[6] = {
        {0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {0, 2}
    };
    return map[voigt_index];
}

/**
 * @brief Map tensor indices to Voigt index
 * @param i First tensor index
 * @param j Second tensor index
 * @return Voigt index (0-5)
 *
 * Mapping: (0,0)→0, (1,1)→1, (2,2)→2, (0,1)→3, (1,2)→4, (0,2)→5
 */
inline std::size_t tensor_to_voigt_index(std::size_t i, std::size_t j) {
    if (i == j) return i;  // Diagonal: 0, 1, 2
    if (i > j) std::swap(i, j);  // Ensure i < j
    if (i == 0 && j == 1) return 3;
    if (i == 1 && j == 2) return 4;
    if (i == 0 && j == 2) return 5;
    return 0;  // Should not reach
}

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_VOIGT_NOTATION_H
