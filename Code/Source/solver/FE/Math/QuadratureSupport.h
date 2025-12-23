#ifndef SVMP_FE_MATH_QUADRATURESUPPORT_H
#define SVMP_FE_MATH_QUADRATURESUPPORT_H

/**
 * @file QuadratureSupport.h
 * @brief Mathematical operations supporting numerical integration in FE computations
 *
 * This header provides Jacobian transformations, integration weight calculations,
 * and mappings between reference and physical element coordinates. These utilities
 * are essential for accurate numerical integration in finite element methods.
 */

#include "Matrix.h"
#include "Vector.h"
#include "MathConstants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace math {

/**
 * @brief Jacobian operations for element transformation
 * @tparam T Scalar type (float, double)
 * @tparam Dim Spatial dimension (1, 2, or 3)
 *
 * Provides utilities for computing and manipulating the Jacobian matrix
 * that maps from reference element to physical element coordinates.
 */
template<typename T, std::size_t Dim>
class JacobianOperations {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    static_assert(Dim >= 1 && Dim <= 3, "Dimension must be 1, 2, or 3");

public:
    using MatrixType = Matrix<T, Dim, Dim>;
    using VectorType = Vector<T, Dim>;

    /**
     * @brief Compute Jacobian matrix from shape function gradients and node coordinates
     * @param grad_shape Matrix of shape function gradients (Dim x NumNodes)
     * @param node_coords Matrix of node coordinates (Dim x NumNodes)
     * @return Jacobian matrix J where J_ij = ∂x_i/∂ξ_j
     *
     * The Jacobian transforms derivatives from reference to physical coordinates:
     * J = ∑_a (∂N_a/∂ξ) ⊗ x_a
     */
    template<std::size_t NumNodes>
    static MatrixType compute_jacobian(
        const Matrix<T, Dim, NumNodes>& grad_shape,
        const Matrix<T, Dim, NumNodes>& node_coords) {

        auto accumulate_jacobian = [&](const Matrix<T, Dim, NumNodes>& grads) {
            MatrixType J_local = MatrixType::zeros();
            for (std::size_t i = 0; i < Dim; ++i) {
                for (std::size_t j = 0; j < Dim; ++j) {
                    for (std::size_t a = 0; a < NumNodes; ++a) {
                        J_local(i, j) += node_coords(i, a) * grads(j, a);
                    }
                }
            }
            return J_local;
        };

        MatrixType J = accumulate_jacobian(grad_shape);

        return J;
    }

    /**
     * @brief Compute determinant of Jacobian matrix
     * @param J Jacobian matrix
     * @return Determinant of J
     *
     * The determinant represents the volume/area/length scaling factor
     * between reference and physical elements.
     */
    static T determinant(const MatrixType& J) {
        return J.determinant();
    }

    /**
     * @brief Compute inverse of Jacobian matrix
     * @param J Jacobian matrix
     * @return Inverse Jacobian matrix J^{-1}
     *
     * The inverse Jacobian transforms derivatives from physical to reference:
     * ∂/∂x = J^{-T} ∂/∂ξ
     */
    static MatrixType inverse(const MatrixType& J) {
        return J.inverse();
    }

    /**
     * @brief Transform gradient from reference to physical coordinates
     * @param grad_ref Gradient in reference coordinates (∂f/∂ξ)
     * @param J_inv Inverse Jacobian matrix
     * @return Gradient in physical coordinates (∂f/∂x)
     *
     * Transformation: ∇_x f = J^{-T} ∇_ξ f
     */
    static VectorType transform_gradient(
        const VectorType& grad_ref,
        const MatrixType& J_inv) {
        return J_inv.transpose() * grad_ref;
    }

    /**
     * @brief Transform gradient matrix from reference to physical coordinates
     * @param grad_ref Matrix of gradients in reference coordinates
     * @param J_inv Inverse Jacobian matrix
     * @return Matrix of gradients in physical coordinates
     *
     * Transforms each column of the gradient matrix.
     */
    template<std::size_t NumFunc>
    static Matrix<T, Dim, NumFunc> transform_gradient(
        const Matrix<T, Dim, NumFunc>& grad_ref,
        const MatrixType& J_inv) {
        return J_inv.transpose() * grad_ref;
    }

    /**
     * @brief Compute integration weight for quadrature point
     * @param det_J Determinant of Jacobian
     * @param quad_weight Quadrature weight in reference element
     * @return Integration weight in physical element
     *
     * Weight = |det(J)| * w_ref
     */
    static T integration_weight(T det_J, T quad_weight) {
        return std::abs(det_J) * quad_weight;
    }

    /**
     * @brief Map point from reference to physical element
     * @param xi Point in reference coordinates
     * @param shape_values Shape function values at xi
     * @param node_coords Node coordinates in physical space
     * @return Point in physical coordinates
     *
     * x = ∑_a N_a(ξ) x_a
     */
    template<std::size_t NumNodes>
    static VectorType reference_to_physical(
        const Vector<T, NumNodes>& shape_values,
        const Matrix<T, Dim, NumNodes>& node_coords) {

        VectorType x = VectorType::zeros();

        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t a = 0; a < NumNodes; ++a) {
                x[i] += shape_values[a] * node_coords(i, a);
            }
        }

        return x;
    }

    /**
     * @brief Check if Jacobian is valid (positive determinant)
     * @param J Jacobian matrix
     * @param tolerance Tolerance for determinant check
     * @return True if det(J) > tolerance
     */
    static bool is_valid(const MatrixType& J, T tolerance = sqrt_epsilon<T>) {
        return determinant(J) > tolerance;
    }

    /**
     * @brief Compute condition number of Jacobian (quality measure)
     * @param J Jacobian matrix
     * @return Condition number ||J|| * ||J^{-1}||
     *
     * Large condition numbers indicate poor element quality.
     */
    static T condition_number(const MatrixType& J) {
        MatrixType J_inv = inverse(J);
        T norm_J = J.one_norm();
        T norm_J_inv = J_inv.one_norm();
        return norm_J * norm_J_inv;
    }
};

/**
 * @brief Specialized 1D Jacobian operations
 */
template<typename T>
class JacobianOperations<T, 1> {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

public:
    /**
     * @brief Compute 1D Jacobian (scalar)
     * @param grad_shape Shape function derivatives
     * @param node_coords Node coordinates
     * @return Jacobian value dx/dξ
     */
    template<std::size_t NumNodes>
    static T compute_jacobian(
        const Vector<T, NumNodes>& grad_shape,
        const Vector<T, NumNodes>& node_coords) {

        T J = T(0);
        for (std::size_t a = 0; a < NumNodes; ++a) {
            J += grad_shape[a] * node_coords[a];
        }
        return J;
    }

    /**
     * @brief Get determinant (absolute value for 1D)
     */
    static T determinant(T J) {
        return std::abs(J);
    }

    /**
     * @brief Get inverse (reciprocal for 1D)
     */
    static T inverse(T J) {
        return T(1) / J;
    }

    /**
     * @brief Transform gradient in 1D
     */
    static T transform_gradient(T grad_ref, T J_inv) {
        return grad_ref * J_inv;
    }

    /**
     * @brief Integration weight in 1D
     */
    static T integration_weight(T J, T quad_weight) {
        return std::abs(J) * quad_weight;
    }
};

/**
 * @brief Integration helper functions for numerical quadrature
 * @tparam T Scalar type
 */
template<typename T>
class IntegrationHelpers {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

public:
    /**
     * @brief Integrate function values at quadrature points (1D)
     * @param values Function values at quadrature points
     * @param weights Quadrature weights
     * @return Integral approximation
     */
    static T integrate_1d(const std::vector<T>& values,
                         const std::vector<T>& weights) {
        if (values.size() != weights.size()) {
            throw std::invalid_argument("Values and weights must have same size");
        }

        T result = T(0);
        for (std::size_t i = 0; i < values.size(); ++i) {
            result += values[i] * weights[i];
        }
        return result;
    }

    /**
     * @brief Integrate function values over tensor product quadrature
     * @param values Function values (row-major storage)
     * @param weights Quadrature weights for each dimension
     * @param dims Number of points in each dimension
     * @return Integral approximation
     *
     * For 2D: integral = ∑_i ∑_j f(ξᵢ, ηⱼ) wᵢ wⱼ
     * For 3D: integral = ∑_i ∑_j ∑_k f(ξᵢ, ηⱼ, ζₖ) wᵢ wⱼ wₖ
     */
    template<std::size_t N>
    static T integrate_nd(const std::vector<T>& values,
                         const std::array<std::vector<T>, N>& weights,
                         const std::array<std::size_t, N>& dims) {
        std::size_t total_points = 1;
        for (std::size_t d = 0; d < N; ++d) {
            total_points *= dims[d];
        }

        if (values.size() != total_points) {
            throw std::invalid_argument("Values size doesn't match dimensions");
        }

        T result = T(0);
        std::array<std::size_t, N> indices{};

        for (std::size_t idx = 0; idx < total_points; ++idx) {
            // Compute tensor product weight
            T weight = T(1);
            std::size_t temp = idx;
            for (std::size_t d = N; d-- > 0;) {
                indices[d] = temp % dims[d];
                temp /= dims[d];
                weight *= weights[d][indices[d]];
            }

            result += values[idx] * weight;
        }

        return result;
    }

    /**
     * @brief Compute volume/area/length of element
     * @param det_jacobians Jacobian determinants at quadrature points
     * @param weights Quadrature weights
     * @return Element volume/area/length
     */
    static T compute_element_volume(const std::vector<T>& det_jacobians,
                                    const std::vector<T>& weights) {
        if (det_jacobians.size() != weights.size()) {
            throw std::invalid_argument("Determinants and weights must have same size");
        }

        T volume = T(0);
        for (std::size_t i = 0; i < det_jacobians.size(); ++i) {
            volume += std::abs(det_jacobians[i]) * weights[i];
        }
        return volume;
    }
};

/**
 * @brief Reference element coordinate mappings
 * @tparam T Scalar type
 */
template<typename T>
class ReferenceElementMappings {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");

public:
    /**
     * @brief Map from [-1,1] to [0,1] (standard to unit interval)
     */
    static T standard_to_unit(T xi) {
        return (xi + T(1)) / T(2);
    }

    /**
     * @brief Map from [0,1] to [-1,1] (unit to standard interval)
     */
    static T unit_to_standard(T x) {
        return T(2) * x - T(1);
    }

    /**
     * @brief Map 2D point from reference square [-1,1]² to reference triangle
     * @param xi First coordinate in [-1,1]
     * @param eta Second coordinate in [-1,1]
     * @return Point in triangle coordinates
     *
     * Duffy transformation for singular integrals
     */
    static Vector2<T> square_to_triangle(T xi, T eta) {
        T x = (T(1) + xi) * (T(1) - eta) / T(4);
        T y = (T(1) + eta) / T(2);
        return Vector2<T>{x, y};
    }

    /**
     * @brief Map 3D point from reference cube to reference tetrahedron
     * @param xi First coordinate in [-1,1]
     * @param eta Second coordinate in [-1,1]
     * @param zeta Third coordinate in [-1,1]
     * @return Point in tetrahedron coordinates
     */
    static Vector3<T> cube_to_tetrahedron(T xi, T eta, T zeta) {
        T x = (T(1) + xi) * (T(1) - eta) * (T(1) - zeta) / T(8);
        T y = (T(1) + eta) * (T(1) - zeta) / T(4);
        T z = (T(1) + zeta) / T(2);
        return Vector3<T>{x, y, z};
    }

    /**
     * @brief Map from reference line to circle (for circular integration)
     * @param t Parameter in [0, 1]
     * @param radius Circle radius
     * @return Point on circle
     */
    static Vector2<T> line_to_circle(T t, T radius = T(1)) {
        T theta = T(2) * pi<T> * t;
        return Vector2<T>{radius * std::cos(theta), radius * std::sin(theta)};
    }

    /**
     * @brief Compute Jacobian of square-to-triangle mapping
     * @param xi First coordinate in [-1,1]
     * @param eta Second coordinate in [-1,1]
     * @return Jacobian determinant
     */
    static T square_to_triangle_jacobian(T xi, T eta) {
        (void)xi;
        return (T(1) - eta) / T(8);
    }

    /**
     * @brief Compute Jacobian of cube-to-tetrahedron mapping
     * @param xi First coordinate in [-1,1]
     * @param eta Second coordinate in [-1,1]
     * @param zeta Third coordinate in [-1,1]
     * @return Jacobian determinant
     */
    static T cube_to_tetrahedron_jacobian(T xi, T eta, T zeta) {
        (void)xi;
        return (T(1) - eta) * (T(1) - zeta) * (T(1) - zeta) / T(64);
    }
};

/**
 * @brief Push-forward and pull-back operations for tensor fields
 * @tparam T Scalar type
 * @tparam Dim Spatial dimension
 */
template<typename T, std::size_t Dim>
class TensorTransformations {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    static_assert(Dim == 2 || Dim == 3, "Dimension must be 2 or 3");

    using MatrixType = Matrix<T, Dim, Dim>;
    using VectorType = Vector<T, Dim>;

public:
    /**
     * @brief Contravariant transformation (push-forward) of vector
     * @param v_ref Vector in reference configuration
     * @param J Jacobian matrix
     * @return Vector in physical configuration
     *
     * v_phys = J * v_ref
     */
    static VectorType push_forward_vector(const VectorType& v_ref,
                                          const MatrixType& J) {
        return J * v_ref;
    }

    /**
     * @brief Covariant transformation (pull-back) of vector
     * @param v_phys Vector in physical configuration
     * @param J_inv Inverse Jacobian matrix
     * @return Vector in reference configuration
     *
     * v_ref = J^{-1} * v_phys
     */
    static VectorType pull_back_vector(const VectorType& v_phys,
                                       const MatrixType& J_inv) {
        return J_inv * v_phys;
    }

    /**
     * @brief Piola transformation for stress-like tensors
     * @param sigma_ref Tensor in reference configuration
     * @param J Jacobian matrix
     * @param det_J Determinant of Jacobian
     * @return Tensor in physical configuration
     *
     * σ_phys = (1/det(J)) * J * σ_ref * J^T
     */
    static MatrixType piola_transform(const MatrixType& sigma_ref,
                                      const MatrixType& J,
                                      T det_J) {
        return (T(1) / det_J) * J * sigma_ref * J.transpose();
    }

    /**
     * @brief Inverse Piola transformation
     * @param sigma_phys Tensor in physical configuration
     * @param J_inv Inverse Jacobian matrix
     * @param det_J Determinant of Jacobian
     * @return Tensor in reference configuration
     *
     * σ_ref = det(J) * J^{-1} * σ_phys * J^{-T}
     */
    static MatrixType inverse_piola_transform(const MatrixType& sigma_phys,
                                              const MatrixType& J_inv,
                                              T det_J) {
        return det_J * J_inv * sigma_phys * J_inv.transpose();
    }

    /**
     * @brief Covariant transformation of rank-2 tensor
     * @param T_ref Tensor in reference configuration
     * @param J_inv Inverse Jacobian matrix
     * @return Tensor in physical configuration
     *
     * T_phys = J^{-T} * T_ref * J^{-1}
     */
    static MatrixType covariant_transform(const MatrixType& T_ref,
                                          const MatrixType& J_inv) {
        return J_inv.transpose() * T_ref * J_inv;
    }

    /**
     * @brief Contravariant transformation of rank-2 tensor
     * @param T_ref Tensor in reference configuration
     * @param J Jacobian matrix
     * @return Tensor in physical configuration
     *
     * T_phys = J * T_ref * J^T
     */
    static MatrixType contravariant_transform(const MatrixType& T_ref,
                                              const MatrixType& J) {
        return J * T_ref * J.transpose();
    }
};

/**
 * @brief Surface integration helpers for boundary integrals
 * @tparam T Scalar type
 * @tparam Dim Spatial dimension
 */
template<typename T, std::size_t Dim>
class SurfaceIntegration {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    static_assert(Dim == 2 || Dim == 3, "Dimension must be 2 or 3");

    using VectorType = Vector<T, Dim>;
    using MatrixType = Matrix<T, Dim, Dim - 1>;

public:
    /**
     * @brief Compute surface normal and area element
     * @param tangent_vectors Tangent vectors to surface (columns)
     * @return Pair of (unit normal, surface area element)
     *
     * For 2D: normal from single tangent
     * For 3D: normal from cross product of tangents
     */
    static std::pair<VectorType, T> compute_normal_and_area(
        const MatrixType& tangent_vectors) {

        VectorType normal;
        T area;

        if constexpr (Dim == 2) {
            // 2D: rotate tangent by 90 degrees
            normal[0] = -tangent_vectors(1, 0);
            normal[1] = tangent_vectors(0, 0);
            area = normal.norm();
            normal = normal.normalized();
        } else { // Dim == 3
            // 3D: cross product of tangents
            Vector3<T> t1{tangent_vectors(0, 0), tangent_vectors(1, 0), tangent_vectors(2, 0)};
            Vector3<T> t2{tangent_vectors(0, 1), tangent_vectors(1, 1), tangent_vectors(2, 1)};
            normal = cross(t1, t2);
            area = normal.norm();
            normal = normal.normalized();
        }

        return {normal, area};
    }

    /**
     * @brief Compute surface Jacobian for boundary element
     * @param grad_shape Shape function gradients on surface
     * @param node_coords Node coordinates
     * @return Surface Jacobian matrix
     */
    template<std::size_t NumNodes>
    static MatrixType compute_surface_jacobian(
        const Matrix<T, Dim - 1, NumNodes>& grad_shape,
        const Matrix<T, Dim, NumNodes>& node_coords) {

        MatrixType J_surf;

        for (std::size_t i = 0; i < Dim; ++i) {
            for (std::size_t j = 0; j < Dim - 1; ++j) {
                J_surf(i, j) = T(0);
                for (std::size_t a = 0; a < NumNodes; ++a) {
                    J_surf(i, j) += node_coords(i, a) * grad_shape(j, a);
                }
            }
        }

        return J_surf;
    }
};

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_QUADRATURESUPPORT_H
