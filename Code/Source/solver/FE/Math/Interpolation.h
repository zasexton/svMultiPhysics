#ifndef SVMP_FE_MATH_INTERPOLATION_H
#define SVMP_FE_MATH_INTERPOLATION_H

/**
 * @file Interpolation.h
 * @brief Interpolation utilities for finite element computations
 *
 * This header provides interpolation operations essential for FE shape function
 * evaluation, including barycentric coordinates, linear/bilinear/trilinear
 * interpolation, and utilities for mapping between reference and physical elements.
 */

#include "Vector.h"
#include "Matrix.h"
#include "MathConstants.h"
#include "MathUtils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace math {

/**
 * @brief Barycentric coordinate computations for simplicial elements
 *
 * Barycentric coordinates express a point as a weighted combination of
 * vertices, essential for interpolation in triangular and tetrahedral elements.
 */
template<typename T>
class BarycentricCoords {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

public:
    /**
     * @brief Compute barycentric coordinates for a point in 2D triangle
     * @param p Query point
     * @param v0 First vertex of triangle
     * @param v1 Second vertex of triangle
     * @param v2 Third vertex of triangle
     * @return Vector3 containing barycentric coordinates [λ₀, λ₁, λ₂]
     *
     * Barycentric coordinates satisfy:
     * - p = λ₀*v0 + λ₁*v1 + λ₂*v2
     * - λ₀ + λ₁ + λ₂ = 1
     * - Point is inside triangle if all λᵢ ∈ [0,1]
     */
    static Vector3<T> compute_2d(const Vector2<T>& p,
                                  const Vector2<T>& v0,
                                  const Vector2<T>& v1,
                                  const Vector2<T>& v2) {
        // Compute vectors from v0
        Vector2<T> v0v1 = v1 - v0;
        Vector2<T> v0v2 = v2 - v0;
        Vector2<T> v0p = p - v0;

        // Compute dot products
        T d00 = v0v1.dot(v0v1);
        T d01 = v0v1.dot(v0v2);
        T d11 = v0v2.dot(v0v2);
        T d20 = v0p.dot(v0v1);
        T d21 = v0p.dot(v0v2);

        // Compute barycentric coordinates
        T denom = d00 * d11 - d01 * d01;

        Vector3<T> bary;
        if (std::abs(denom) < epsilon<T>) {
            // Degenerate triangle
            bary[0] = T(1.0/3.0);
            bary[1] = T(1.0/3.0);
            bary[2] = T(1.0/3.0);
        } else {
            T inv_denom = T(1) / denom;
            bary[1] = (d11 * d20 - d01 * d21) * inv_denom;
            bary[2] = (d00 * d21 - d01 * d20) * inv_denom;
            bary[0] = T(1) - bary[1] - bary[2];
        }

        return bary;
    }

    /**
     * @brief Compute barycentric coordinates for a point in 3D tetrahedron
     * @param p Query point
     * @param v0 First vertex of tetrahedron
     * @param v1 Second vertex of tetrahedron
     * @param v2 Third vertex of tetrahedron
     * @param v3 Fourth vertex of tetrahedron
     * @return Vector4 containing barycentric coordinates [λ₀, λ₁, λ₂, λ₃]
     *
     * Uses volume ratios to compute coordinates:
     * λᵢ = Volume(tetrahedron with p replacing vertex i) / Volume(original tetrahedron)
     */
    static Vector4<T> compute_3d(const Vector3<T>& p,
                                  const Vector3<T>& v0,
                                  const Vector3<T>& v1,
                                  const Vector3<T>& v2,
                                  const Vector3<T>& v3) {
        // Build the transformation matrix from barycentric to Cartesian
        Matrix<T, 3, 3> M;
        M(0, 0) = v1[0] - v0[0];  M(0, 1) = v2[0] - v0[0];  M(0, 2) = v3[0] - v0[0];
        M(1, 0) = v1[1] - v0[1];  M(1, 1) = v2[1] - v0[1];  M(1, 2) = v3[1] - v0[1];
        M(2, 0) = v1[2] - v0[2];  M(2, 1) = v2[2] - v0[2];  M(2, 2) = v3[2] - v0[2];

        T det = M.determinant();
        Vector4<T> bary;

        if (std::abs(det) < epsilon<T>) {
            // Degenerate tetrahedron
            bary[0] = T(0.25);
            bary[1] = T(0.25);
            bary[2] = T(0.25);
            bary[3] = T(0.25);
        } else {
            // Solve for barycentric coordinates
            Vector3<T> rhs = p - v0;
            Matrix<T, 3, 3> M_inv = M.inverse();
            Vector3<T> lambda = M_inv * rhs;

            bary[0] = T(1) - lambda[0] - lambda[1] - lambda[2];
            bary[1] = lambda[0];
            bary[2] = lambda[1];
            bary[3] = lambda[2];
        }

        return bary;
    }

    /**
     * @brief Check if point is inside simplex based on barycentric coordinates
     * @tparam N Dimension of barycentric coordinate vector
     * @param bary Barycentric coordinates
     * @param tolerance Tolerance for boundary inclusion
     * @return True if point is inside or on boundary of simplex
     */
    template<std::size_t N>
    static bool is_inside(const Vector<T, N>& bary, T tolerance = T(0)) {
        T sum = T(0);
        for (std::size_t i = 0; i < N; ++i) {
            if (bary[i] < -tolerance || bary[i] > T(1) + tolerance) {
                return false;
            }
            sum += bary[i];
        }
        return std::abs(sum - T(1)) <= tolerance;
    }

    /**
     * @brief Interpolate values at vertices using barycentric coordinates
     * @tparam ValueType Type of values to interpolate
     * @tparam N Number of vertices
     * @param bary Barycentric coordinates
     * @param values Array of values at vertices
     * @return Interpolated value
     */
    template<typename ValueType, std::size_t N>
    static ValueType interpolate(const Vector<T, N>& bary,
                                 const std::array<ValueType, N>& values) {
        ValueType result = bary[0] * values[0];
        for (std::size_t i = 1; i < N; ++i) {
            result = result + bary[i] * values[i];
        }
        return result;
    }
};

// Note: lerp is defined in MathUtils.h with signature lerp(a, b, t)
// Using that version for consistency

/**
 * @brief Smooth interpolation using cubic Hermite curve
 * @tparam T Interpolation parameter type
 * @param t Parameter in [0, 1]
 * @return Smoothed parameter value
 *
 * Uses smoothstep function: 3t² - 2t³
 * Has zero derivative at t=0 and t=1
 */
template<typename T>
inline constexpr T smoothstep(T t) {
    t = clamp(t, T(0), T(1));
    const T endpoint_tol = sqrt_epsilon<T>;
    if (t <= endpoint_tol) {
        return T(0);
    }
    if (t >= T(1) - endpoint_tol) {
        return T(1);
    }
    return t * t * (T(3) - T(2) * t);
}

/**
 * @brief Smoother interpolation using quintic curve
 * @tparam T Interpolation parameter type
 * @param t Parameter in [0, 1]
 * @return Smoothed parameter value
 *
 * Uses smootherstep function: 6t⁵ - 15t⁴ + 10t³
 * Has zero first and second derivatives at endpoints
 */
template<typename T>
inline constexpr T smootherstep(T t) {
    t = clamp(t, T(0), T(1));
    const T endpoint_tol = sqrt_epsilon<T>;
    if (t <= endpoint_tol) {
        return T(0);
    }
    if (t >= T(1) - endpoint_tol) {
        return T(1);
    }
    return t * t * t * (t * (t * T(6) - T(15)) + T(10));
}

/**
 * @brief Bilinear interpolation on a 2D grid
 * @tparam T Scalar type
 * @param x X coordinate in [0, 1]
 * @param y Y coordinate in [0, 1]
 * @param f00 Value at (0, 0)
 * @param f10 Value at (1, 0)
 * @param f01 Value at (0, 1)
 * @param f11 Value at (1, 1)
 * @return Interpolated value
 *
 * Performs linear interpolation in x, then in y:
 * f(x,y) = (1-x)(1-y)f₀₀ + x(1-y)f₁₀ + (1-x)y f₀₁ + xy f₁₁
 */
template<typename T>
inline constexpr T bilinear(T x, T y, T f00, T f10, T f01, T f11) {
    T fx0 = lerp(f00, f10, x);  // Interpolate along x at y=0
    T fx1 = lerp(f01, f11, x);  // Interpolate along x at y=1
    return lerp(fx0, fx1, y);   // Interpolate along y
}

/**
 * @brief Trilinear interpolation on a 3D grid
 * @tparam T Scalar type
 * @param x X coordinate in [0, 1]
 * @param y Y coordinate in [0, 1]
 * @param z Z coordinate in [0, 1]
 * @param f000 Value at (0, 0, 0)
 * @param f100 Value at (1, 0, 0)
 * @param f010 Value at (0, 1, 0)
 * @param f110 Value at (1, 1, 0)
 * @param f001 Value at (0, 0, 1)
 * @param f101 Value at (1, 0, 1)
 * @param f011 Value at (0, 1, 1)
 * @param f111 Value at (1, 1, 1)
 * @return Interpolated value
 *
 * Performs bilinear interpolation in xy-planes, then linear in z
 */
template<typename T>
inline constexpr T trilinear(T x, T y, T z,
                             T f000, T f100, T f010, T f110,
                             T f001, T f101, T f011, T f111) {
    T fz0 = bilinear(x, y, f000, f100, f010, f110);  // z=0 plane
    T fz1 = bilinear(x, y, f001, f101, f011, f111);  // z=1 plane
    return lerp(fz0, fz1, z);                        // Interpolate along z
}

/**
 * @brief Shape function evaluation helpers for common element types
 */
template<typename T>
class ShapeFunctionHelpers {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

public:
    /**
     * @brief Evaluate 1D linear shape functions
     * @param xi Local coordinate in [-1, 1]
     * @return Vector2 of shape function values [N₀, N₁]
     */
    static Vector2<T> linear_1d(T xi) {
        return Vector2<T>{
            (T(1) - xi) / T(2),  // N₀ = (1 - ξ)/2
            (T(1) + xi) / T(2)   // N₁ = (1 + ξ)/2
        };
    }

    /**
     * @brief Evaluate 1D quadratic shape functions
     * @param xi Local coordinate in [-1, 1]
     * @return Vector3 of shape function values [N₀, N₁, N₂]
     */
    static Vector3<T> quadratic_1d(T xi) {
        return Vector3<T>{
            xi * (xi - T(1)) / T(2),    // N₀ = ξ(ξ-1)/2
            T(1) - xi * xi,              // N₁ = 1 - ξ²
            xi * (xi + T(1)) / T(2)     // N₂ = ξ(ξ+1)/2
        };
    }

    /**
     * @brief Evaluate 2D bilinear shape functions on quad
     * @param xi First local coordinate in [-1, 1]
     * @param eta Second local coordinate in [-1, 1]
     * @return Vector4 of shape function values
     */
    static Vector4<T> bilinear_2d(T xi, T eta) {
        return Vector4<T>{
            (T(1) - xi) * (T(1) - eta) / T(4),  // N₀ = (1-ξ)(1-η)/4
            (T(1) + xi) * (T(1) - eta) / T(4),  // N₁ = (1+ξ)(1-η)/4
            (T(1) + xi) * (T(1) + eta) / T(4),  // N₂ = (1+ξ)(1+η)/4
            (T(1) - xi) * (T(1) + eta) / T(4)   // N₃ = (1-ξ)(1+η)/4
        };
    }

    /**
     * @brief Evaluate 2D linear shape functions on triangle
     * @param xi First local coordinate
     * @param eta Second local coordinate
     * @return Vector3 of shape function values
     *
     * Uses area coordinates: ξ, η, ζ = 1-ξ-η
     */
    static Vector3<T> linear_triangle(T xi, T eta) {
        return Vector3<T>{
            T(1) - xi - eta,  // N₀ = 1 - ξ - η
            xi,               // N₁ = ξ
            eta               // N₂ = η
        };
    }

    /**
     * @brief Evaluate 3D trilinear shape functions on hex
     * @param xi First local coordinate in [-1, 1]
     * @param eta Second local coordinate in [-1, 1]
     * @param zeta Third local coordinate in [-1, 1]
     * @return Vector of 8 shape function values
     */
    static Vector<T, 8> trilinear_3d(T xi, T eta, T zeta) {
        Vector<T, 8> N;
        T xi_m = T(1) - xi;
        T xi_p = T(1) + xi;
        T eta_m = T(1) - eta;
        T eta_p = T(1) + eta;
        T zeta_m = T(1) - zeta;
        T zeta_p = T(1) + zeta;

        N[0] = xi_m * eta_m * zeta_m / T(8);
        N[1] = xi_p * eta_m * zeta_m / T(8);
        N[2] = xi_p * eta_p * zeta_m / T(8);
        N[3] = xi_m * eta_p * zeta_m / T(8);
        N[4] = xi_m * eta_m * zeta_p / T(8);
        N[5] = xi_p * eta_m * zeta_p / T(8);
        N[6] = xi_p * eta_p * zeta_p / T(8);
        N[7] = xi_m * eta_p * zeta_p / T(8);

        return N;
    }

    /**
     * @brief Evaluate 3D linear shape functions on tetrahedron
     * @param xi First local coordinate
     * @param eta Second local coordinate
     * @param zeta Third local coordinate
     * @return Vector4 of shape function values
     *
     * Uses volume coordinates: L₀ = 1-ξ-η-ζ, L₁ = ξ, L₂ = η, L₃ = ζ
     */
    static Vector4<T> linear_tetrahedron(T xi, T eta, T zeta) {
        return Vector4<T>{
            T(1) - xi - eta - zeta,  // N₀ = 1 - ξ - η - ζ
            xi,                      // N₁ = ξ
            eta,                     // N₂ = η
            zeta                     // N₃ = ζ
        };
    }
};

/**
 * @brief Lagrange polynomial interpolation
 * @tparam T Scalar type
 */
template<typename T>
class LagrangeInterpolation {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

public:
    /**
     * @brief Evaluate 1D Lagrange basis function
     * @param i Index of basis function
     * @param x Evaluation point
     * @param nodes Array of interpolation nodes
     * @param n Number of nodes
     * @return Value of i-th Lagrange basis at x
     */
    static T basis_1d(std::size_t i, T x, const T* nodes, std::size_t n) {
        T result = T(1);
        T xi = nodes[i];

        for (std::size_t j = 0; j < n; ++j) {
            if (j != i) {
                result *= (x - nodes[j]) / (xi - nodes[j]);
            }
        }

        return result;
    }

    /**
     * @brief Interpolate function values using Lagrange polynomials
     * @param x Evaluation point
     * @param nodes Array of interpolation nodes
     * @param values Array of function values at nodes
     * @param n Number of nodes
     * @return Interpolated value at x
     */
    static T interpolate_1d(T x, const T* nodes, const T* values, std::size_t n) {
        T result = T(0);

        for (std::size_t i = 0; i < n; ++i) {
            result += values[i] * basis_1d(i, x, nodes, n);
        }

        return result;
    }

    /**
     * @brief Compute derivative of Lagrange interpolation
     * @param x Evaluation point
     * @param nodes Array of interpolation nodes
     * @param values Array of function values at nodes
     * @param n Number of nodes
     * @return Derivative of interpolated function at x
     */
    static T derivative_1d(T x, const T* nodes, const T* values, std::size_t n) {
        T result = T(0);

        for (std::size_t i = 0; i < n; ++i) {
            T deriv_basis = T(0);
            T xi = nodes[i];

            // Compute derivative of i-th basis function
            for (std::size_t j = 0; j < n; ++j) {
                if (j != i) {
                    T prod = T(1) / (xi - nodes[j]);
                    for (std::size_t k = 0; k < n; ++k) {
                        if (k != i && k != j) {
                            prod *= (x - nodes[k]) / (xi - nodes[k]);
                        }
                    }
                    deriv_basis += prod;
                }
            }

            result += values[i] * deriv_basis;
        }

        return result;
    }
};

/**
 * @brief Hermite interpolation for smooth curves
 * @tparam T Scalar type
 */
template<typename T>
class HermiteInterpolation {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

public:
    /**
     * @brief Cubic Hermite interpolation
     * @param t Parameter in [0, 1]
     * @param p0 Value at t=0
     * @param p1 Value at t=1
     * @param m0 Derivative at t=0
     * @param m1 Derivative at t=1
     * @return Interpolated value
     *
     * Uses cubic Hermite basis functions:
     * h₀₀(t) = 2t³ - 3t² + 1
     * h₁₀(t) = t³ - 2t² + t
     * h₀₁(t) = -2t³ + 3t²
     * h₁₁(t) = t³ - t²
     */
    template<typename ValueType>
    static ValueType cubic(T t, const ValueType& p0, const ValueType& p1,
                           const ValueType& m0, const ValueType& m1) {
        T t2 = t * t;
        T t3 = t2 * t;

        T h00 = T(2) * t3 - T(3) * t2 + T(1);
        T h10 = t3 - T(2) * t2 + t;
        T h01 = -T(2) * t3 + T(3) * t2;
        T h11 = t3 - t2;

        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
    }

    /**
     * @brief Catmull-Rom spline interpolation
     * @param t Parameter in [0, 1] between p1 and p2
     * @param p0 Point before p1
     * @param p1 Start point
     * @param p2 End point
     * @param p3 Point after p2
     * @return Interpolated value
     *
     * Special case of cubic Hermite with:
     * m₀ = (p₂ - p₀) / 2
     * m₁ = (p₃ - p₁) / 2
     */
    template<typename ValueType>
    static ValueType catmull_rom(T t, const ValueType& p0, const ValueType& p1,
                                 const ValueType& p2, const ValueType& p3) {
        ValueType m0 = (p2 - p0) * T(0.5);
        ValueType m1 = (p3 - p1) * T(0.5);
        return cubic(t, p1, p2, m0, m1);
    }
};

/**
 * @brief Inverse distance weighting interpolation
 * @tparam T Scalar type
 * @tparam Dim Spatial dimension
 */
template<typename T, std::size_t Dim>
class IDWInterpolation {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

public:
    /**
     * @brief Interpolate using inverse distance weighting
     * @param query Query point
     * @param points Array of data points
     * @param values Array of values at data points
     * @param n Number of data points
     * @param power Power parameter (typically 2)
     * @return Interpolated value at query point
     *
     * Weight for point i: wᵢ = 1 / distance(query, pointᵢ)^power
     */
    template<typename ValueType>
    static ValueType interpolate(const Vector<T, Dim>& query,
                                 const Vector<T, Dim>* points,
                                 const ValueType* values,
                                 std::size_t n,
                                 T power = T(2)) {
        T total_weight = T(0);
        ValueType result = ValueType{} * T(0);  // Zero-initialized

        for (std::size_t i = 0; i < n; ++i) {
            Vector<T, Dim> diff = query - points[i];
            T dist = diff.norm();

            if (dist < epsilon<T>) {
                // Query point coincides with data point
                return values[i];
            }

            T weight = T(1) / std::pow(dist, power);
            total_weight += weight;
            result = result + weight * values[i];
        }

        if (total_weight > epsilon<T>) {
            result = result * (T(1) / total_weight);
        }

        return result;
    }
};

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_INTERPOLATION_H
