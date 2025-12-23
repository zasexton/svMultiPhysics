/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_CURVILINEAR_EVAL_H
#define SVMP_CURVILINEAR_EVAL_H

#include "../Core/MeshTypes.h"
#include "../Topology/CellShape.h"
#include <array>
#include <vector>
#include <functional>
#include <limits>

namespace svmp {

// Forward declarations
class MeshBase;

/**
 * @brief Parametric point in reference element coordinates
 *
 * Reference coordinates (ξ, η, ζ) for standard element types:
 * - Line:     ξ ∈ [-1, 1]
 * - Triangle: ξ, η ∈ [0, 1], ξ + η ≤ 1
 * - Quad:     ξ, η ∈ [-1, 1]
 * - Tet:      ξ, η, ζ ∈ [0, 1], ξ + η + ζ ≤ 1
 * - Hex:      ξ, η, ζ ∈ [-1, 1]
 * - Wedge:    ξ, η ∈ [0, 1], ξ + η ≤ 1, ζ ∈ [-1, 1]
 * - Pyramid:  ξ, η ∈ [-1, 1], ζ ∈ [0, 1], |ξ|, |η| ≤ 1 - ζ
 */
using ParametricPoint = std::array<real_t, 3>;

/**
 * @brief Jacobian matrix of coordinate transformation
 *
 * J[i][j] = ∂x_i / ∂ξ_j where x = (x, y, z), ξ = (ξ, η, ζ)
 *
 * For 2D elements embedded in 3D:
 * - J is 3×2 (columns correspond to ξ, η)
 * - det(J^T J) gives area scaling
 *
 * For 3D elements:
 * - J is 3×3
 * - det(J) gives volume scaling
 */
struct Jacobian {
    std::array<std::array<real_t, 3>, 3> matrix;  // J[physical_dim][parametric_dim]
    int parametric_dim{3};  // 1, 2, or 3
    int physical_dim{3};    // Always 3 for embedded geometries

    /// Compute determinant (for 3D elements only)
    real_t determinant() const;

    /// Compute inverse Jacobian (for 3D elements)
    Jacobian inverse() const;

    /// Compute metric tensor G = J^T J (for surface/edge elements)
    std::array<std::array<real_t, 3>, 3> metric_tensor() const;

    /// Compute sqrt(det(J^T J)) (area/length scaling factor)
    real_t metric_determinant() const;
};

/**
 * @brief Shape function values and derivatives at a parametric point
 *
 * Contains basis function values N_i(ξ) and derivatives dN_i/dξ_j
 * for all nodes in the element.
 */
struct ShapeFunctionValues {
    std::vector<real_t> N;                          // Shape function values [n_nodes]
    std::vector<std::array<real_t, 3>> dN_dxi;      // Derivatives w.r.t. parametric coords [n_nodes]
    std::vector<std::array<real_t, 3>> dN_dx;       // Derivatives w.r.t. physical coords [n_nodes]
    int polynomial_order{1};                         // Order of shape functions
};

/**
 * @brief Physical coordinates and geometric quantities at a parametric point
 */
struct GeometryEvaluation {
    std::array<real_t, 3> coordinates;  // Physical position x(ξ)
    Jacobian jacobian;                  // dx/dξ
    real_t det_jacobian{0};             // det(J) or sqrt(det(J^T J))
    bool is_valid{true};                // False if Jacobian is singular
};

/**
 * @brief Curvilinear/high-order geometry evaluator
 *
 * Evaluates geometric quantities at arbitrary parametric points within
 * elements using high-order shape functions. Supports all CellShape types
 * with polynomial orders p >= 1.
 *
 * **Supported Cell Types (all polynomial orders):**
 * - Vertex (point)
 * - Line (edge): Lagrange basis on [-1, 1]
 * - Triangle: Lagrange basis on reference triangle
 * - Quad: Tensor-product Lagrange or Serendipity
 * - Tetrahedron: Lagrange basis on reference tet
 * - Hexahedron: Tensor-product Lagrange or Serendipity
 * - Wedge: Mixed tensor-product basis
 * - Pyramid: Pyramid-specific basis
 *
 * **Key Operations:**
 * - Map parametric point ξ to physical coordinates x(ξ)
 * - Compute Jacobian J = dx/dξ and det(J)
 * - Evaluate shape functions N_i(ξ) and gradients ∇N_i
 * - Support isoparametric (geometry = field basis) and sub/super-parametric
 *
 * **Design Philosophy:**
 * - Dimension-agnostic: works for 1D/2D/3D elements embedded in 3D
 * - Order-agnostic: automatically determines order from element vertex count
 * - Topology-aware: uses CellTopology for canonical reference coordinates
 */
class CurvilinearEvaluator {
public:
    /**
     * @brief Evaluate geometry at a parametric point within a cell
     * @param mesh The mesh
     * @param cell Cell index
     * @param xi Parametric coordinates
     * @param cfg Reference or current configuration
     * @return Geometric evaluation (coordinates, Jacobian, det(J))
     */
    static GeometryEvaluation evaluate_geometry(
        const MeshBase& mesh,
        index_t cell,
        const ParametricPoint& xi,
        Configuration cfg = Configuration::Reference);

    /**
     * @brief Evaluate shape functions at a parametric point
     * @param shape Cell shape type
     * @param order Polynomial order
     * @param xi Parametric coordinates
     * @return Shape function values and derivatives
     */
    static ShapeFunctionValues evaluate_shape_functions(
        const CellShape& shape,
        size_t n_nodes,
        const ParametricPoint& xi);

    /**
     * @brief Compute Jacobian at a parametric point
     * @param mesh The mesh
     * @param cell Cell index
     * @param xi Parametric coordinates
     * @param cfg Reference or current configuration
     * @return Jacobian matrix
     */
    static Jacobian compute_jacobian(
        const MeshBase& mesh,
        index_t cell,
        const ParametricPoint& xi,
        Configuration cfg = Configuration::Reference);

    /**
     * @brief Map physical point to parametric coordinates (inverse mapping)
     * @param mesh The mesh
     * @param cell Cell index
     * @param physical_point Physical coordinates
     * @param cfg Reference or current configuration
     * @param tolerance Convergence tolerance for Newton iteration
     * @param max_iterations Maximum Newton iterations
     * @return Parametric coordinates (or invalid if not converged)
     *
     * Uses Newton-Raphson iteration to solve x(ξ) = x_target for ξ.
     */
    static std::pair<ParametricPoint, bool> inverse_map(
        const MeshBase& mesh,
        index_t cell,
        const std::array<real_t, 3>& physical_point,
        Configuration cfg = Configuration::Reference,
        real_t tolerance = 1e-10,
        int max_iterations = 50);

    /**
     * @brief Check if parametric point is inside reference element
     * @param shape Cell shape
     * @param xi Parametric coordinates
     * @param tolerance Tolerance for boundary tests
     * @return True if inside or on boundary
     */
    static bool is_inside_reference_element(
        const CellShape& shape,
        const ParametricPoint& xi,
        real_t tolerance = 1e-10);

    /**
     * @brief Get reference element center in parametric coordinates
     * @param shape Cell shape
     * @return Centroid in parametric space
     */
    static ParametricPoint reference_element_center(const CellShape& shape);

    /**
     * @brief Determine polynomial order from vertex count
     * @param shape Cell shape
     * @param n_vertices Number of vertices in element
     * @return Polynomial order (1 for linear, 2 for quadratic, etc.)
     */
    static int deduce_order(const CellShape& shape, size_t n_nodes);

private:
    // ---- Shape function evaluation (per cell type) ----

    static ShapeFunctionValues eval_line_shape_functions(
        int order, const ParametricPoint& xi);

    static ShapeFunctionValues eval_triangle_shape_functions(
        int order, const ParametricPoint& xi);

    static ShapeFunctionValues eval_quad_shape_functions(
        int order, size_t n_nodes, const ParametricPoint& xi);

    static ShapeFunctionValues eval_tet_shape_functions(
        int order, const ParametricPoint& xi);

    static ShapeFunctionValues eval_hex_shape_functions(
        int order, size_t n_nodes, const ParametricPoint& xi);

    static ShapeFunctionValues eval_wedge_shape_functions(
        int order, size_t n_nodes, const ParametricPoint& xi);

    static ShapeFunctionValues eval_pyramid_shape_functions(
        int order, size_t n_nodes, const ParametricPoint& xi);

    // ---- Lagrange basis functions ----

    /// Lagrange polynomial and derivative on [-1, 1]
    static void lagrange_1d(
        int order,
        real_t xi,
        std::vector<real_t>& N,
        std::vector<real_t>& dN);

    /// Lagrange basis for triangle in (ξ, η) with ξ + η ≤ 1
    static void lagrange_triangle(
        int order,
        real_t xi,
        real_t eta,
        std::vector<real_t>& N,
        std::vector<real_t>& dN_dxi,
        std::vector<real_t>& dN_deta);

    /// Tensor-product Lagrange for quad/hex
    static void lagrange_tensor_product(
        int order,
        int dim,
        const ParametricPoint& xi,
        std::vector<real_t>& N,
        std::vector<std::array<real_t, 3>>& dN);

    // ---- Reference element bounds ----

    struct ReferenceBounds {
        std::array<real_t, 3> min;
        std::array<real_t, 3> max;
        std::function<bool(const ParametricPoint&)> inside_test;
    };

    static ReferenceBounds get_reference_bounds(CellShape shape);
};

/**
 * @brief Quadrature rule for integration over reference elements
 *
 * Provides Gauss quadrature points and weights for accurate integration
 * of polynomial functions over reference elements.
 */
class QuadratureRule {
public:
    struct QuadraturePoint {
        ParametricPoint xi;  // Parametric coordinates
        real_t weight;       // Quadrature weight
    };

    /**
     * @brief Get Gauss quadrature rule for a cell shape
     * @param shape Cell shape
     * @param order Desired polynomial exactness (integrates polynomials up to degree `order`)
     * @return Quadrature points and weights
     */
    static std::vector<QuadraturePoint> get_quadrature_rule(
        const CellShape& shape,
        int order);

    /**
     * @brief Get Gauss-Legendre points and weights for 1D [-1, 1]
     * @param n_points Number of quadrature points
     * @return Points and weights
     */
    static std::vector<QuadraturePoint> gauss_legendre_1d(int n_points);

    /**
     * @brief Get quadrature rule for triangle
     * @param order Polynomial exactness
     * @return Points and weights
     */
    static std::vector<QuadraturePoint> triangle_quadrature(int order);

    /**
     * @brief Get quadrature rule for tetrahedron
     * @param order Polynomial exactness
     * @return Points and weights
     */
    static std::vector<QuadraturePoint> tet_quadrature(int order);

    /**
     * @brief Tensor-product quadrature for quad/hex
     * @param shape Quad or Hex
     * @param order Polynomial exactness
     * @return Points and weights
     */
    static std::vector<QuadraturePoint> tensor_product_quadrature(
        const CellShape& shape,
        int order);

private:
    /// Compute Gauss-Legendre abscissas and weights
    static void gauss_legendre_nodes(
        int n,
        std::vector<real_t>& nodes,
        std::vector<real_t>& weights);
};

/**
 * @brief High-order mesh distortion detector
 *
 * Detects invalid (negative Jacobian) or highly distorted elements in
 * high-order meshes by sampling Jacobian at multiple points.
 */
class DistortionDetector {
public:
    struct DistortionMetrics {
        real_t min_jacobian{std::numeric_limits<real_t>::max()};
        real_t max_jacobian{std::numeric_limits<real_t>::lowest()};
        real_t jacobian_ratio{0};  // max/min (condition number proxy)
        bool has_negative_jacobian{false};
        bool is_highly_distorted{false};
        std::vector<ParametricPoint> negative_jacobian_points;
    };

    /**
     * @brief Detect distortion in a cell by sampling Jacobian
     * @param mesh The mesh
     * @param cell Cell index
     * @param cfg Configuration
     * @param sampling_order Order of quadrature for sampling (higher = more samples)
     * @return Distortion metrics
     */
    static DistortionMetrics detect_distortion(
        const MeshBase& mesh,
        index_t cell,
        Configuration cfg = Configuration::Reference,
        int sampling_order = 3);

    /**
     * @brief Find all inverted cells in mesh
     * @param mesh The mesh
     * @param cfg Configuration
     * @return List of cell indices with negative Jacobian
     */
    static std::vector<index_t> find_inverted_cells(
        const MeshBase& mesh,
        Configuration cfg = Configuration::Reference);
};

} // namespace svmp

#endif // SVMP_CURVILINEAR_EVAL_H
