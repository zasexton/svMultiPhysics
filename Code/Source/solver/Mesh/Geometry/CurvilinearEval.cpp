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

#include "CurvilinearEval.h"
#include "../Core/MeshBase.h"
#include "GeometryConfig.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace svmp {

//=============================================================================
// Jacobian Implementation
//=============================================================================

real_t Jacobian::determinant() const {
    if (parametric_dim != 3) {
        throw std::runtime_error("Determinant only defined for 3D Jacobians");
    }

    const auto& J = matrix;
    return J[0][0] * (J[1][1]*J[2][2] - J[1][2]*J[2][1])
         - J[0][1] * (J[1][0]*J[2][2] - J[1][2]*J[2][0])
         + J[0][2] * (J[1][0]*J[2][1] - J[1][1]*J[2][0]);
}

Jacobian Jacobian::inverse() const {
    if (parametric_dim != 3) {
        throw std::runtime_error("Inverse only defined for 3×3 Jacobians");
    }

    real_t det = determinant();
    if (std::abs(det) < GeometryConfig::volume_epsilon()) {
        throw std::runtime_error("Singular Jacobian");
    }

    Jacobian inv;
    inv.parametric_dim = 3;
    inv.physical_dim = 3;

    const auto& J = matrix;
    real_t inv_det = 1.0 / det;

    inv.matrix[0][0] = (J[1][1]*J[2][2] - J[1][2]*J[2][1]) * inv_det;
    inv.matrix[0][1] = (J[0][2]*J[2][1] - J[0][1]*J[2][2]) * inv_det;
    inv.matrix[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1]) * inv_det;
    inv.matrix[1][0] = (J[1][2]*J[2][0] - J[1][0]*J[2][2]) * inv_det;
    inv.matrix[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0]) * inv_det;
    inv.matrix[1][2] = (J[0][2]*J[1][0] - J[0][0]*J[1][2]) * inv_det;
    inv.matrix[2][0] = (J[1][0]*J[2][1] - J[1][1]*J[2][0]) * inv_det;
    inv.matrix[2][1] = (J[0][1]*J[2][0] - J[0][0]*J[2][1]) * inv_det;
    inv.matrix[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0]) * inv_det;

    return inv;
}

std::array<std::array<real_t, 3>, 3> Jacobian::metric_tensor() const {
    // G = J^T J
    std::array<std::array<real_t, 3>, 3> G{};

    for (int i = 0; i < parametric_dim; ++i) {
        for (int j = 0; j < parametric_dim; ++j) {
            G[i][j] = 0.0;
            for (int k = 0; k < physical_dim; ++k) {
                G[i][j] += matrix[k][i] * matrix[k][j];
            }
        }
    }

    return G;
}

real_t Jacobian::metric_determinant() const {
    if (parametric_dim == 3) {
        return std::abs(determinant());
    } else if (parametric_dim == 2) {
        // det(J^T J) for 3×2 matrix
        auto G = metric_tensor();
        real_t det_G = G[0][0]*G[1][1] - G[0][1]*G[1][0];
        return std::sqrt(std::max(0.0, det_G));
    } else if (parametric_dim == 1) {
        // ||J|| for 3×1 matrix
        real_t norm_sq = 0.0;
        for (int k = 0; k < physical_dim; ++k) {
            norm_sq += matrix[k][0] * matrix[k][0];
        }
        return std::sqrt(norm_sq);
    }

    return 0.0;
}

//=============================================================================
// CurvilinearEvaluator Implementation
//=============================================================================

GeometryEvaluation CurvilinearEvaluator::evaluate_geometry(
    const MeshBase& mesh,
    index_t cell,
    const ParametricPoint& xi,
    Configuration cfg) {

    GeometryEvaluation eval;

    // Get cell shape and vertices
    CellShape shape = mesh.cell_shape(cell);
    auto vertices = mesh.cell_vertices(cell);
    int order = deduce_order(shape, vertices.size());

    // Evaluate shape functions
    auto shape_fns = evaluate_shape_functions(shape, order, xi);

    // Get vertex coordinates
    auto get_coords = [&](index_t vid) -> std::array<real_t, 3> {
        if (cfg == Configuration::Current || cfg == Configuration::Deformed) {
            return mesh.vertex_current_coordinates(vid);
        } else {
            return mesh.vertex_reference_coordinates(vid);
        }
    };

    // Compute physical coordinates: x = Σ N_i x_i
    eval.coordinates = {0, 0, 0};
    for (size_t i = 0; i < vertices.size(); ++i) {
        auto coords = get_coords(vertices[i]);
        eval.coordinates[0] += shape_fns.N[i] * coords[0];
        eval.coordinates[1] += shape_fns.N[i] * coords[1];
        eval.coordinates[2] += shape_fns.N[i] * coords[2];
    }

    // Compute Jacobian: J[k][l] = Σ (dN_i/dξ_l) x_i_k
    int param_dim = CellShapeTraits::parametric_dimension(shape);
    eval.jacobian.parametric_dim = param_dim;
    eval.jacobian.physical_dim = 3;

    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < param_dim; ++l) {
            eval.jacobian.matrix[k][l] = 0.0;
            for (size_t i = 0; i < vertices.size(); ++i) {
                auto coords = get_coords(vertices[i]);
                eval.jacobian.matrix[k][l] += shape_fns.dN_dxi[i][l] * coords[k];
            }
        }
    }

    // Compute determinant
    eval.det_jacobian = eval.jacobian.metric_determinant();
    eval.is_valid = eval.det_jacobian > GeometryConfig::volume_epsilon();

    return eval;
}

ShapeFunctionValues CurvilinearEvaluator::evaluate_shape_functions(
    CellShape shape,
    int order,
    const ParametricPoint& xi) {

    switch (shape) {
        case CellShape::Vertex:
            return ShapeFunctionValues{{1.0}, {{0, 0, 0}}, {{0, 0, 0}}, 0};

        case CellShape::Line:
            return eval_line_shape_functions(order, xi);

        case CellShape::Triangle:
            return eval_triangle_shape_functions(order, xi);

        case CellShape::Quad:
            return eval_quad_shape_functions(order, xi);

        case CellShape::Tetrahedron:
            return eval_tet_shape_functions(order, xi);

        case CellShape::Hexahedron:
            return eval_hex_shape_functions(order, xi);

        case CellShape::Wedge:
            return eval_wedge_shape_functions(order, xi);

        case CellShape::Pyramid:
            return eval_pyramid_shape_functions(order, xi);

        default:
            throw std::runtime_error("Unsupported cell shape");
    }
}

Jacobian CurvilinearEvaluator::compute_jacobian(
    const MeshBase& mesh,
    index_t cell,
    const ParametricPoint& xi,
    Configuration cfg) {

    return evaluate_geometry(mesh, cell, xi, cfg).jacobian;
}

std::pair<ParametricPoint, bool> CurvilinearEvaluator::inverse_map(
    const MeshBase& mesh,
    index_t cell,
    const std::array<real_t, 3>& physical_point,
    Configuration cfg,
    real_t tolerance,
    int max_iterations) {

    CellShape shape = mesh.cell_shape(cell);

    // Initial guess: reference element center
    ParametricPoint xi = reference_element_center(shape);

    // Newton-Raphson iteration: solve x(ξ) = x_target
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto eval = evaluate_geometry(mesh, cell, xi, cfg);

        // Compute residual: r = x_target - x(ξ)
        std::array<real_t, 3> residual = {
            physical_point[0] - eval.coordinates[0],
            physical_point[1] - eval.coordinates[1],
            physical_point[2] - eval.coordinates[2]
        };

        real_t residual_norm = std::sqrt(residual[0]*residual[0] +
                                        residual[1]*residual[1] +
                                        residual[2]*residual[2]);

        if (residual_norm < tolerance) {
            return {xi, true}; // Converged
        }

        // Solve J Δξ = r for Δξ
        try {
            auto J_inv = eval.jacobian.inverse();

            // Δξ = J^{-1} r
            std::array<real_t, 3> delta_xi = {0, 0, 0};
            for (int i = 0; i < eval.jacobian.parametric_dim; ++i) {
                for (int k = 0; k < 3; ++k) {
                    delta_xi[i] += J_inv.matrix[i][k] * residual[k];
                }
            }

            // Update: ξ := ξ + Δξ
            xi[0] += delta_xi[0];
            xi[1] += delta_xi[1];
            xi[2] += delta_xi[2];

        } catch (...) {
            return {xi, false}; // Singular Jacobian
        }
    }

    return {xi, false}; // Did not converge
}

bool CurvilinearEvaluator::is_inside_reference_element(
    CellShape shape,
    const ParametricPoint& xi,
    real_t tolerance) {

    auto bounds = get_reference_bounds(shape);
    return bounds.inside_test(xi);
}

ParametricPoint CurvilinearEvaluator::reference_element_center(CellShape shape) {
    switch (shape) {
        case CellShape::Vertex:
            return {0, 0, 0};
        case CellShape::Line:
            return {0, 0, 0};
        case CellShape::Triangle:
            return {1.0/3.0, 1.0/3.0, 0};
        case CellShape::Quad:
            return {0, 0, 0};
        case CellShape::Tetrahedron:
            return {0.25, 0.25, 0.25};
        case CellShape::Hexahedron:
            return {0, 0, 0};
        case CellShape::Wedge:
            return {1.0/3.0, 1.0/3.0, 0};
        case CellShape::Pyramid:
            return {0, 0, 0.25};
        default:
            return {0, 0, 0};
    }
}

int CurvilinearEvaluator::deduce_order(CellShape shape, int n_vertices) {
    // Determine order from vertex count based on Lagrange node counts
    switch (shape) {
        case CellShape::Line:
            return n_vertices - 1; // p+1 nodes

        case CellShape::Triangle:
            // (p+1)(p+2)/2 nodes → solve for p
            for (int p = 1; p <= 10; ++p) {
                if ((p+1)*(p+2)/2 == n_vertices) return p;
            }
            return 1;

        case CellShape::Quad:
            // (p+1)^2 nodes for tensor-product
            for (int p = 1; p <= 10; ++p) {
                if ((p+1)*(p+1) == n_vertices) return p;
            }
            return 1;

        case CellShape::Tetrahedron:
            // (p+1)(p+2)(p+3)/6 nodes
            for (int p = 1; p <= 10; ++p) {
                if ((p+1)*(p+2)*(p+3)/6 == n_vertices) return p;
            }
            return 1;

        case CellShape::Hexahedron:
            // (p+1)^3 nodes for tensor-product
            for (int p = 1; p <= 10; ++p) {
                if ((p+1)*(p+1)*(p+1) == n_vertices) return p;
            }
            return 1;

        case CellShape::Wedge:
            // (p+1)^2 * (p+2)/2 nodes
            for (int p = 1; p <= 10; ++p) {
                if ((p+1)*(p+1)*(p+2)/2 == n_vertices) return p;
            }
            return 1;

        case CellShape::Pyramid:
            // (p+1)(p+2)(2p+3)/6 nodes
            for (int p = 1; p <= 10; ++p) {
                if ((p+1)*(p+2)*(2*p+3)/6 == n_vertices) return p;
            }
            return 1;

        default:
            return 1;
    }
}

//=============================================================================
// Shape Function Evaluation (Per Cell Type)
//=============================================================================

ShapeFunctionValues CurvilinearEvaluator::eval_line_shape_functions(
    int order,
    const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    lagrange_1d(order, xi[0], vals.N, vals.dN_dxi.emplace_back());

    // Convert to 3D derivatives
    for (size_t i = 0; i < vals.N.size(); ++i) {
        vals.dN_dxi[i] = {vals.dN_dxi[0][i], 0, 0};
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_triangle_shape_functions(
    int order,
    const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    std::vector<real_t> dN_dxi, dN_deta;
    lagrange_triangle(order, xi[0], xi[1], vals.N, dN_dxi, dN_deta);

    vals.dN_dxi.resize(vals.N.size());
    for (size_t i = 0; i < vals.N.size(); ++i) {
        vals.dN_dxi[i] = {dN_dxi[i], dN_deta[i], 0};
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_quad_shape_functions(
    int order,
    const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    lagrange_tensor_product(order, 2, xi, vals.N, vals.dN_dxi);

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_tet_shape_functions(
    int order,
    const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    if (order == 1) {
        // Linear tet: barycentric coordinates
        real_t lambda0 = 1.0 - xi[0] - xi[1] - xi[2];
        real_t lambda1 = xi[0];
        real_t lambda2 = xi[1];
        real_t lambda3 = xi[2];

        vals.N = {lambda0, lambda1, lambda2, lambda3};
        vals.dN_dxi = {
            {-1, -1, -1},
            { 1,  0,  0},
            { 0,  1,  0},
            { 0,  0,  1}
        };
    } else {
        // Higher-order: would require full Lagrange basis on tet
        // Simplified placeholder
        vals.N.resize((order+1)*(order+2)*(order+3)/6, 0.0);
        vals.dN_dxi.resize(vals.N.size(), {0, 0, 0});
        // TODO: Implement full high-order tet basis
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_hex_shape_functions(
    int order,
    const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    lagrange_tensor_product(order, 3, xi, vals.N, vals.dN_dxi);

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_wedge_shape_functions(
    int order,
    const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    if (order == 1) {
        // Linear wedge: triangle × line
        real_t lambda0 = 1.0 - xi[0] - xi[1];
        real_t lambda1 = xi[0];
        real_t lambda2 = xi[1];
        real_t zeta = xi[2];

        real_t N0 = lambda0 * (1.0 - zeta) * 0.5;
        real_t N1 = lambda1 * (1.0 - zeta) * 0.5;
        real_t N2 = lambda2 * (1.0 - zeta) * 0.5;
        real_t N3 = lambda0 * (1.0 + zeta) * 0.5;
        real_t N4 = lambda1 * (1.0 + zeta) * 0.5;
        real_t N5 = lambda2 * (1.0 + zeta) * 0.5;

        vals.N = {N0, N1, N2, N3, N4, N5};
        vals.dN_dxi = {
            {-0.5*(1-zeta), -0.5*(1-zeta), -0.5*lambda0},
            { 0.5*(1-zeta),  0.0,          -0.5*lambda1},
            { 0.0,           0.5*(1-zeta), -0.5*lambda2},
            {-0.5*(1+zeta), -0.5*(1+zeta),  0.5*lambda0},
            { 0.5*(1+zeta),  0.0,           0.5*lambda1},
            { 0.0,           0.5*(1+zeta),  0.5*lambda2}
        };
    } else {
        // Higher-order wedge
        int n_nodes = (order+1)*(order+1)*(order+2)/2;
        vals.N.resize(n_nodes, 0.0);
        vals.dN_dxi.resize(n_nodes, {0, 0, 0});
        // TODO: Implement full high-order wedge basis
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_pyramid_shape_functions(
    int order,
    const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    if (order == 1) {
        // Linear pyramid (5 nodes)
        real_t xi_ = xi[0];
        real_t eta = xi[1];
        real_t zeta = xi[2];

        if (std::abs(1.0 - zeta) < 1e-12) {
            // Apex singularity handling
            vals.N = {0, 0, 0, 0, 1.0};
            vals.dN_dxi.resize(5, {0, 0, 0});
        } else {
            real_t denom = 1.0 - zeta;
            vals.N = {
                0.25 * (1 - xi_) * (1 - eta) * (1 - zeta),
                0.25 * (1 + xi_) * (1 - eta) * (1 - zeta),
                0.25 * (1 + xi_) * (1 + eta) * (1 - zeta),
                0.25 * (1 - xi_) * (1 + eta) * (1 - zeta),
                zeta
            };

            vals.dN_dxi = {
                {-0.25*(1-eta)*(1-zeta), -0.25*(1-xi_)*(1-zeta), -0.25*(1-xi_)*(1-eta)},
                { 0.25*(1-eta)*(1-zeta), -0.25*(1+xi_)*(1-zeta), -0.25*(1+xi_)*(1-eta)},
                { 0.25*(1+eta)*(1-zeta),  0.25*(1+xi_)*(1-zeta), -0.25*(1+xi_)*(1+eta)},
                {-0.25*(1+eta)*(1-zeta),  0.25*(1-xi_)*(1-zeta), -0.25*(1-xi_)*(1+eta)},
                { 0, 0, 1}
            };
        }
    } else {
        // Higher-order pyramid
        int n_nodes = (order+1)*(order+2)*(2*order+3)/6;
        vals.N.resize(n_nodes, 0.0);
        vals.dN_dxi.resize(n_nodes, {0, 0, 0});
        // TODO: Implement full high-order pyramid basis
    }

    return vals;
}

//=============================================================================
// Lagrange Basis Functions
//=============================================================================

void CurvilinearEvaluator::lagrange_1d(
    int order,
    real_t xi,
    std::vector<real_t>& N,
    std::vector<real_t>& dN) {

    int n_nodes = order + 1;
    N.resize(n_nodes);
    dN.resize(n_nodes);

    // Evenly-spaced Lagrange nodes on [-1, 1]
    std::vector<real_t> nodes(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        nodes[i] = -1.0 + 2.0 * i / order;
    }

    // Lagrange basis: L_i(xi) = Π_{j≠i} (xi - xi_j) / (xi_i - xi_j)
    for (int i = 0; i < n_nodes; ++i) {
        N[i] = 1.0;
        dN[i] = 0.0;

        for (int j = 0; j < n_nodes; ++j) {
            if (i == j) continue;

            real_t factor = (xi - nodes[j]) / (nodes[i] - nodes[j]);
            N[i] *= factor;

            // Derivative via product rule
            real_t deriv = 1.0;
            for (int k = 0; k < n_nodes; ++k) {
                if (k == i || k == j) continue;
                deriv *= (xi - nodes[k]) / (nodes[i] - nodes[k]);
            }
            dN[i] += deriv / (nodes[i] - nodes[j]);
        }
    }
}

void CurvilinearEvaluator::lagrange_triangle(
    int order,
    real_t xi,
    real_t eta,
    std::vector<real_t>& N,
    std::vector<real_t>& dN_dxi,
    std::vector<real_t>& dN_deta) {

    int n_nodes = (order + 1) * (order + 2) / 2;
    N.resize(n_nodes);
    dN_dxi.resize(n_nodes);
    dN_deta.resize(n_nodes);

    if (order == 1) {
        // Linear triangle
        real_t lambda0 = 1.0 - xi - eta;
        N[0] = lambda0;
        N[1] = xi;
        N[2] = eta;

        dN_dxi[0] = -1; dN_deta[0] = -1;
        dN_dxi[1] =  1; dN_deta[1] =  0;
        dN_dxi[2] =  0; dN_deta[2] =  1;
    } else {
        // Higher-order: simplified placeholder
        std::fill(N.begin(), N.end(), 0.0);
        std::fill(dN_dxi.begin(), dN_dxi.end(), 0.0);
        std::fill(dN_deta.begin(), dN_deta.end(), 0.0);
        // TODO: Full Lagrange basis for triangles
    }
}

void CurvilinearEvaluator::lagrange_tensor_product(
    int order,
    int dim,
    const ParametricPoint& xi,
    std::vector<real_t>& N,
    std::vector<std::array<real_t, 3>>& dN) {

    // Compute 1D basis in each direction
    std::vector<real_t> N_1d, dN_1d;
    std::array<std::vector<real_t>, 3> N_dir, dN_dir;

    for (int d = 0; d < dim; ++d) {
        lagrange_1d(order, xi[d], N_1d, dN_1d);
        N_dir[d] = N_1d;
        dN_dir[d] = dN_1d;
    }

    // Tensor product
    int n_nodes_per_dim = order + 1;
    int n_nodes = (dim == 2) ? n_nodes_per_dim * n_nodes_per_dim
                             : n_nodes_per_dim * n_nodes_per_dim * n_nodes_per_dim;

    N.resize(n_nodes);
    dN.resize(n_nodes, {0, 0, 0});

    if (dim == 2) {
        for (int j = 0; j < n_nodes_per_dim; ++j) {
            for (int i = 0; i < n_nodes_per_dim; ++i) {
                int idx = j * n_nodes_per_dim + i;
                N[idx] = N_dir[0][i] * N_dir[1][j];
                dN[idx][0] = dN_dir[0][i] * N_dir[1][j];
                dN[idx][1] = N_dir[0][i] * dN_dir[1][j];
            }
        }
    } else if (dim == 3) {
        for (int k = 0; k < n_nodes_per_dim; ++k) {
            for (int j = 0; j < n_nodes_per_dim; ++j) {
                for (int i = 0; i < n_nodes_per_dim; ++i) {
                    int idx = k * n_nodes_per_dim * n_nodes_per_dim + j * n_nodes_per_dim + i;
                    N[idx] = N_dir[0][i] * N_dir[1][j] * N_dir[2][k];
                    dN[idx][0] = dN_dir[0][i] * N_dir[1][j] * N_dir[2][k];
                    dN[idx][1] = N_dir[0][i] * dN_dir[1][j] * N_dir[2][k];
                    dN[idx][2] = N_dir[0][i] * N_dir[1][j] * dN_dir[2][k];
                }
            }
        }
    }
}

//=============================================================================
// Reference Element Bounds
//=============================================================================

CurvilinearEvaluator::ReferenceBounds
CurvilinearEvaluator::get_reference_bounds(CellShape shape) {
    ReferenceBounds bounds;

    switch (shape) {
        case CellShape::Line:
            bounds.min = {-1, 0, 0};
            bounds.max = {1, 0, 0};
            bounds.inside_test = [](const ParametricPoint& xi) {
                return xi[0] >= -1 && xi[0] <= 1;
            };
            break;

        case CellShape::Triangle:
            bounds.min = {0, 0, 0};
            bounds.max = {1, 1, 0};
            bounds.inside_test = [](const ParametricPoint& xi) {
                return xi[0] >= 0 && xi[1] >= 0 && (xi[0] + xi[1]) <= 1;
            };
            break;

        case CellShape::Quad:
            bounds.min = {-1, -1, 0};
            bounds.max = {1, 1, 0};
            bounds.inside_test = [](const ParametricPoint& xi) {
                return xi[0] >= -1 && xi[0] <= 1 && xi[1] >= -1 && xi[1] <= 1;
            };
            break;

        case CellShape::Tetrahedron:
            bounds.min = {0, 0, 0};
            bounds.max = {1, 1, 1};
            bounds.inside_test = [](const ParametricPoint& xi) {
                return xi[0] >= 0 && xi[1] >= 0 && xi[2] >= 0 &&
                       (xi[0] + xi[1] + xi[2]) <= 1;
            };
            break;

        case CellShape::Hexahedron:
            bounds.min = {-1, -1, -1};
            bounds.max = {1, 1, 1};
            bounds.inside_test = [](const ParametricPoint& xi) {
                return xi[0] >= -1 && xi[0] <= 1 &&
                       xi[1] >= -1 && xi[1] <= 1 &&
                       xi[2] >= -1 && xi[2] <= 1;
            };
            break;

        case CellShape::Wedge:
            bounds.min = {0, 0, -1};
            bounds.max = {1, 1, 1};
            bounds.inside_test = [](const ParametricPoint& xi) {
                return xi[0] >= 0 && xi[1] >= 0 && (xi[0] + xi[1]) <= 1 &&
                       xi[2] >= -1 && xi[2] <= 1;
            };
            break;

        case CellShape::Pyramid:
            bounds.min = {-1, -1, 0};
            bounds.max = {1, 1, 1};
            bounds.inside_test = [](const ParametricPoint& xi) {
                return std::abs(xi[0]) <= (1 - xi[2]) &&
                       std::abs(xi[1]) <= (1 - xi[2]) &&
                       xi[2] >= 0 && xi[2] <= 1;
            };
            break;

        default:
            bounds.min = {0, 0, 0};
            bounds.max = {0, 0, 0};
            bounds.inside_test = [](const ParametricPoint&) { return false; };
    }

    return bounds;
}

//=============================================================================
// QuadratureRule Implementation (simplified stubs)
//=============================================================================

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::get_quadrature_rule(CellShape shape, int order) {
    // Simplified: return minimal rule for now
    // Full implementation would use optimized Gauss rules
    std::vector<QuadraturePoint> points;

    switch (shape) {
        case CellShape::Line:
            return gauss_legendre_1d((order + 1) / 2 + 1);
        case CellShape::Triangle:
            return triangle_quadrature(order);
        case CellShape::Quad:
            return tensor_product_quadrature(shape, order);
        case CellShape::Hexahedron:
            return tensor_product_quadrature(shape, order);
        default:
            // Single-point rule at centroid
            points.push_back({CurvilinearEvaluator::reference_element_center(shape), 1.0});
            return points;
    }
}

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::gauss_legendre_1d(int n_points) {
    std::vector<real_t> nodes, weights;
    gauss_legendre_nodes(n_points, nodes, weights);

    std::vector<QuadraturePoint> points(n_points);
    for (int i = 0; i < n_points; ++i) {
        points[i].xi = {nodes[i], 0, 0};
        points[i].weight = weights[i];
    }

    return points;
}

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::triangle_quadrature(int order) {
    // Simplified: 1-point rule at centroid
    std::vector<QuadraturePoint> points;
    points.push_back({{1.0/3.0, 1.0/3.0, 0}, 0.5});
    return points;
}

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::tet_quadrature(int order) {
    // Simplified: 1-point rule at centroid
    std::vector<QuadraturePoint> points;
    points.push_back({{0.25, 0.25, 0.25}, 1.0/6.0});
    return points;
}

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::tensor_product_quadrature(CellShape shape, int order) {
    int n_1d = (order + 1) / 2 + 1;
    auto rule_1d = gauss_legendre_1d(n_1d);

    std::vector<QuadraturePoint> points;

    if (shape == CellShape::Quad) {
        for (const auto& pi : rule_1d) {
            for (const auto& pj : rule_1d) {
                points.push_back({{pi.xi[0], pj.xi[0], 0}, pi.weight * pj.weight});
            }
        }
    } else if (shape == CellShape::Hexahedron) {
        for (const auto& pi : rule_1d) {
            for (const auto& pj : rule_1d) {
                for (const auto& pk : rule_1d) {
                    points.push_back({{pi.xi[0], pj.xi[0], pk.xi[0]},
                                     pi.weight * pj.weight * pk.weight});
                }
            }
        }
    }

    return points;
}

void QuadratureRule::gauss_legendre_nodes(
    int n,
    std::vector<real_t>& nodes,
    std::vector<real_t>& weights) {

    // Simplified: hardcoded for common orders
    nodes.resize(n);
    weights.resize(n);

    if (n == 1) {
        nodes[0] = 0.0;
        weights[0] = 2.0;
    } else if (n == 2) {
        nodes[0] = -1.0 / std::sqrt(3.0);
        nodes[1] =  1.0 / std::sqrt(3.0);
        weights[0] = weights[1] = 1.0;
    } else if (n == 3) {
        nodes[0] = -std::sqrt(3.0/5.0);
        nodes[1] = 0.0;
        nodes[2] =  std::sqrt(3.0/5.0);
        weights[0] = weights[2] = 5.0/9.0;
        weights[1] = 8.0/9.0;
    } else {
        // Fallback: uniform spacing (not Gauss!)
        for (int i = 0; i < n; ++i) {
            nodes[i] = -1.0 + 2.0 * i / (n - 1);
            weights[i] = 2.0 / n;
        }
    }
}

//=============================================================================
// DistortionDetector Implementation
//=============================================================================

DistortionDetector::DistortionMetrics
DistortionDetector::detect_distortion(
    const MeshBase& mesh,
    index_t cell,
    Configuration cfg,
    int sampling_order) {

    DistortionMetrics metrics;

    CellShape shape = mesh.cell_shape(cell);
    auto quad_rule = QuadratureRule::get_quadrature_rule(shape, sampling_order);

    for (const auto& qp : quad_rule) {
        auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, qp.xi, cfg);

        if (eval.det_jacobian < 0) {
            metrics.has_negative_jacobian = true;
            metrics.negative_jacobian_points.push_back(qp.xi);
        }

        metrics.min_jacobian = std::min(metrics.min_jacobian, eval.det_jacobian);
        metrics.max_jacobian = std::max(metrics.max_jacobian, eval.det_jacobian);
    }

    if (metrics.min_jacobian > 0) {
        metrics.jacobian_ratio = metrics.max_jacobian / metrics.min_jacobian;
        metrics.is_highly_distorted = metrics.jacobian_ratio > 100.0;
    } else {
        metrics.jacobian_ratio = std::numeric_limits<real_t>::infinity();
        metrics.is_highly_distorted = true;
    }

    return metrics;
}

std::vector<index_t> DistortionDetector::find_inverted_cells(
    const MeshBase& mesh,
    Configuration cfg) {

    std::vector<index_t> inverted;

    for (index_t cell = 0; cell < mesh.n_cells(); ++cell) {
        auto metrics = detect_distortion(mesh, cell, cfg);
        if (metrics.has_negative_jacobian) {
            inverted.push_back(cell);
        }
    }

    return inverted;
}

} // namespace svmp
