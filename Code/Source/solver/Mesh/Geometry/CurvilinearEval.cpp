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
#include "../Topology/CellTopology.h"
#include "GeometryConfig.h"
#include "PyramidBasis.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {

namespace {

constexpr real_t kEps = 1e-12;

inline real_t clamp(real_t x, real_t lo, real_t hi) {
    return std::max(lo, std::min(hi, x));
}

inline int parametric_dimension(CellFamily family) {
    switch (family) {
        case CellFamily::Point:   return 0;
        case CellFamily::Line:    return 1;
        case CellFamily::Triangle:
        case CellFamily::Quad:
        case CellFamily::Polygon: return 2;
        case CellFamily::Tetra:
        case CellFamily::Hex:
        case CellFamily::Wedge:
        case CellFamily::Pyramid:
        case CellFamily::Polyhedron:
            return 3;
    }
    return 3;
}

inline bool is_serendipity_quadratic(CellFamily family, int order, size_t n_nodes) {
    if (order != 2) return false;
    switch (family) {
        case CellFamily::Quad:    return n_nodes == 8;
        case CellFamily::Hex:     return n_nodes == 20;
        case CellFamily::Wedge:   return n_nodes == 15;
        case CellFamily::Pyramid: return n_nodes == 13;
        default:                  return false;
    }
}

inline size_t lagrange_node_count(CellFamily family, int p) {
    if (p < 1) return 0;
    switch (family) {
        case CellFamily::Point:   return 1;
        case CellFamily::Line:    return static_cast<size_t>(p + 1);
        case CellFamily::Triangle: return static_cast<size_t>((p + 1) * (p + 2) / 2);
        case CellFamily::Quad:     return static_cast<size_t>((p + 1) * (p + 1));
        case CellFamily::Tetra:    return static_cast<size_t>((p + 1) * (p + 2) * (p + 3) / 6);
        case CellFamily::Hex:      return static_cast<size_t>((p + 1) * (p + 1) * (p + 1));
        case CellFamily::Wedge:    return static_cast<size_t>((p + 1) * (p + 1) * (p + 2) / 2);
        case CellFamily::Pyramid:  return static_cast<size_t>((p + 1) * (p + 2) * (2 * p + 3) / 6);
        default:
            return 0;
    }
}

// -------------------------
// Lagrange basis: 1D [-1,1]
// -------------------------

static void lagrange_1d_equispaced(int order, real_t xi, std::vector<real_t>& N, std::vector<real_t>& dN) {
    if (order < 1) {
        throw std::invalid_argument("lagrange_1d_equispaced: order must be >= 1");
    }
    const int n = order + 1;
    N.assign(static_cast<size_t>(n), real_t(0));
    dN.assign(static_cast<size_t>(n), real_t(0));

    std::vector<real_t> nodes(static_cast<size_t>(n), real_t(0));
    for (int i = 0; i < n; ++i) {
        nodes[static_cast<size_t>(i)] = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(order);
    }

    for (int i = 0; i < n; ++i) {
        real_t Li = 1.0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            Li *= (xi - nodes[static_cast<size_t>(j)]) /
                  (nodes[static_cast<size_t>(i)] - nodes[static_cast<size_t>(j)]);
        }
        N[static_cast<size_t>(i)] = Li;

        real_t dLi = 0.0;
        for (int m = 0; m < n; ++m) {
            if (m == i) continue;
            real_t prod = 1.0;
            for (int j = 0; j < n; ++j) {
                if (j == i || j == m) continue;
                prod *= (xi - nodes[static_cast<size_t>(j)]) /
                        (nodes[static_cast<size_t>(i)] - nodes[static_cast<size_t>(j)]);
            }
            prod /= (nodes[static_cast<size_t>(i)] - nodes[static_cast<size_t>(m)]);
            dLi += prod;
        }
        dN[static_cast<size_t>(i)] = dLi;
    }
}

// ------------------------------------------
// Lagrange basis: simplex factors in lambda
// ------------------------------------------

static void simplex_lagrange_factor_sequence(int p,
                                             real_t lambda,
                                             std::vector<real_t>& phi,
                                             std::vector<real_t>& dphi_dlambda) {
    if (p < 1) {
        throw std::invalid_argument("simplex_lagrange_factor_sequence: order must be >= 1");
    }
    const size_t n = static_cast<size_t>(p + 1);
    phi.assign(n, real_t(0));
    dphi_dlambda.assign(n, real_t(0));

    const real_t t = static_cast<real_t>(p) * lambda;
    const real_t dt_dlambda = static_cast<real_t>(p);

    phi[0] = 1.0;
    real_t dphi_dt_prev = 0.0;
    dphi_dlambda[0] = 0.0;

    for (int a = 1; a <= p; ++a) {
        const real_t inv_a = 1.0 / static_cast<real_t>(a);
        const real_t s = (t - static_cast<real_t>(a - 1)) * inv_a;
        const real_t phi_prev = phi[static_cast<size_t>(a - 1)];

        phi[static_cast<size_t>(a)] = s * phi_prev;

        const real_t dphi_dt = inv_a * phi_prev + s * dphi_dt_prev;
        dphi_dlambda[static_cast<size_t>(a)] = dt_dlambda * dphi_dt;
        dphi_dt_prev = dphi_dt;
    }
}

static std::vector<std::array<int, 3>> triangle_exponents_vtk(int p) {
    std::vector<std::array<int, 3>> exps;
    exps.reserve(lagrange_node_count(CellFamily::Triangle, p));

    // corners: (p,0,0), (0,p,0), (0,0,p)
    exps.push_back({p, 0, 0});
    exps.push_back({0, p, 0});
    exps.push_back({0, 0, p});

    // edges in CellTopology edge order
    const auto eview = CellTopology::get_edges_view(CellFamily::Triangle);
    const int steps = std::max(0, p - 1);
    for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        for (int k = 1; k <= steps; ++k) {
            std::array<int, 3> e = {0, 0, 0};
            e[static_cast<size_t>(a)] = p - k;
            e[static_cast<size_t>(b)] = k;
            exps.push_back(e);
        }
    }

    // interior nodes (p>2)
    for (int i = 1; i <= p - 2; ++i) {
        for (int j = 1; j <= p - 1 - i; ++j) {
            exps.push_back({p - i - j, i, j});
        }
    }

    return exps;
}

static std::vector<std::array<int, 4>> tetra_exponents_vtk(int p) {
    std::vector<std::array<int, 4>> exps;
    exps.reserve(lagrange_node_count(CellFamily::Tetra, p));

    // corners: (p,0,0,0), (0,p,0,0), (0,0,p,0), (0,0,0,p)
    exps.push_back({p, 0, 0, 0});
    exps.push_back({0, p, 0, 0});
    exps.push_back({0, 0, p, 0});
    exps.push_back({0, 0, 0, p});

    // edges
    const auto eview = CellTopology::get_edges_view(CellFamily::Tetra);
    const int steps = std::max(0, p - 1);
    for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        for (int k = 1; k <= steps; ++k) {
            std::array<int, 4> e = {0, 0, 0, 0};
            e[static_cast<size_t>(a)] = p - k;
            e[static_cast<size_t>(b)] = k;
            exps.push_back(e);
        }
    }

    // face interior nodes in oriented face order
    const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Tetra);
    for (int fi = 0; fi < fview.face_count; ++fi) {
        const int b = fview.offsets[fi];
        const int e = fview.offsets[fi + 1];
        const int fv = e - b;
        if (fv != 3) continue;
        const int v0 = fview.indices[b + 0];
        const int v1 = fview.indices[b + 1];
        const int v2 = fview.indices[b + 2];
        for (int i = 1; i <= p - 2; ++i) {
            for (int j = 1; j <= p - 1 - i; ++j) {
                std::array<int, 4> f = {0, 0, 0, 0};
                f[static_cast<size_t>(v0)] = p - i - j;
                f[static_cast<size_t>(v1)] = i;
                f[static_cast<size_t>(v2)] = j;
                exps.push_back(f);
            }
        }
    }

    // volume interior nodes
    for (int i = 1; i <= p - 3; ++i) {
        for (int j = 1; j <= p - 2 - i; ++j) {
            for (int k = 1; k <= p - 1 - i - j; ++k) {
                exps.push_back({p - i - j - k, i, j, k});
            }
        }
    }

    return exps;
}

static std::vector<std::array<int, 2>> quad_lagrange_indices_vtk(int p) {
    std::vector<std::array<int, 2>> idx;
    idx.reserve(lagrange_node_count(CellFamily::Quad, p));

    // corners (VTK Quad4 order)
    idx.push_back({0, 0});
    idx.push_back({p, 0});
    idx.push_back({p, p});
    idx.push_back({0, p});

    // edges in CellTopology edge order
    const auto eview = CellTopology::get_edges_view(CellFamily::Quad);
    const int steps = std::max(0, p - 1);
    const std::array<std::array<int, 2>, 4> corner_grid = {{
        {0, 0}, {p, 0}, {p, p}, {0, p}
    }};
    for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        const auto A = corner_grid[static_cast<size_t>(a)];
        const auto B = corner_grid[static_cast<size_t>(b)];
        for (int k = 1; k <= steps; ++k) {
            const real_t t = static_cast<real_t>(k) / static_cast<real_t>(p);
            const int ii = static_cast<int>(std::lround((1.0 - t) * A[0] + t * B[0]));
            const int jj = static_cast<int>(std::lround((1.0 - t) * A[1] + t * B[1]));
            idx.push_back({ii, jj});
        }
    }

    // interior nodes (row-major in j, outer i) to match CellTopology::high_order_pattern
    for (int i = 1; i <= p - 1; ++i) {
        for (int j = 1; j <= p - 1; ++j) {
            idx.push_back({i, j});
        }
    }

    return idx;
}

static std::vector<std::array<int, 3>> hex_lagrange_indices_vtk(int p) {
    std::vector<std::array<int, 3>> idx;
    idx.reserve(lagrange_node_count(CellFamily::Hex, p));

    const std::array<std::array<int, 3>, 8> corner_grid = {{
        {0, 0, 0}, {p, 0, 0}, {p, p, 0}, {0, p, 0},
        {0, 0, p}, {p, 0, p}, {p, p, p}, {0, p, p}
    }};

    // corners
    for (const auto& c : corner_grid) idx.push_back(c);

    // edges
    const auto eview = CellTopology::get_edges_view(CellFamily::Hex);
    const int steps = std::max(0, p - 1);
    for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        const auto A = corner_grid[static_cast<size_t>(a)];
        const auto B = corner_grid[static_cast<size_t>(b)];
        for (int k = 1; k <= steps; ++k) {
            const real_t t = static_cast<real_t>(k) / static_cast<real_t>(p);
            const int ii = static_cast<int>(std::lround((1.0 - t) * A[0] + t * B[0]));
            const int jj = static_cast<int>(std::lround((1.0 - t) * A[1] + t * B[1]));
            const int kk = static_cast<int>(std::lround((1.0 - t) * A[2] + t * B[2]));
            idx.push_back({ii, jj, kk});
        }
    }

    // faces (quad faces, oriented face order) with bilinear interpolation in grid space
    const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Hex);
    for (int fi = 0; fi < fview.face_count; ++fi) {
        const int b = fview.offsets[fi];
        const int e = fview.offsets[fi + 1];
        const int fv = e - b;
        if (fv != 4) continue;
        const int v0 = fview.indices[b + 0];
        const int v1 = fview.indices[b + 1];
        const int v2 = fview.indices[b + 2];
        const int v3 = fview.indices[b + 3];
        const auto A = corner_grid[static_cast<size_t>(v0)];
        const auto B = corner_grid[static_cast<size_t>(v1)];
        const auto C = corner_grid[static_cast<size_t>(v2)];
        const auto D = corner_grid[static_cast<size_t>(v3)];

        for (int i = 1; i <= p - 1; ++i) {
            for (int j = 1; j <= p - 1; ++j) {
                const real_t u = static_cast<real_t>(i) / static_cast<real_t>(p);
                const real_t v = static_cast<real_t>(j) / static_cast<real_t>(p);
                std::array<int, 3> g = {0, 0, 0};
                for (int d = 0; d < 3; ++d) {
                    const real_t x =
                        (1.0 - u) * (1.0 - v) * A[d] +
                        u * (1.0 - v) * B[d] +
                        u * v * C[d] +
                        (1.0 - u) * v * D[d];
                    g[static_cast<size_t>(d)] = static_cast<int>(std::lround(x));
                }
                idx.push_back(g);
            }
        }
    }

    // volume interior nodes
    for (int i = 1; i <= p - 1; ++i) {
        for (int j = 1; j <= p - 1; ++j) {
            for (int k = 1; k <= p - 1; ++k) {
                idx.push_back({i, j, k});
            }
        }
    }

    return idx;
}

struct WedgeLagrangeLabel {
    std::array<int, 3> tri_exp; // (a0,a1,a2) sum p
    int kz = 0;                 // 0..p line index
};

static std::array<real_t, 3> wedge_corner_coords(int local_corner) {
    switch (local_corner) {
        case 0: return {0.0, 0.0, -1.0};
        case 1: return {1.0, 0.0, -1.0};
        case 2: return {0.0, 1.0, -1.0};
        case 3: return {0.0, 0.0,  1.0};
        case 4: return {1.0, 0.0,  1.0};
        case 5: return {0.0, 1.0,  1.0};
        default: return {0.0, 0.0, 0.0};
    }
}

static WedgeLagrangeLabel wedge_label_from_param(int p, const std::array<real_t, 3>& xi) {
    const real_t x = xi[0];
    const real_t y = xi[1];
    const real_t z = xi[2];
    int a1 = static_cast<int>(std::lround(static_cast<real_t>(p) * x));
    int a2 = static_cast<int>(std::lround(static_cast<real_t>(p) * y));
    a1 = std::clamp(a1, 0, p);
    a2 = std::clamp(a2, 0, p);
    int a0 = p - a1 - a2;
    if (a0 < 0) a0 = 0;
    int kz = static_cast<int>(std::lround((z + 1.0) * 0.5 * static_cast<real_t>(p)));
    kz = std::clamp(kz, 0, p);
    return {{a0, a1, a2}, kz};
}

static std::vector<WedgeLagrangeLabel> wedge_lagrange_labels_vtk(int p) {
    std::vector<WedgeLagrangeLabel> labels;
    labels.reserve(lagrange_node_count(CellFamily::Wedge, p));

    // corners
    for (int c = 0; c < 6; ++c) {
        labels.push_back(wedge_label_from_param(p, wedge_corner_coords(c)));
    }

    // edges
    const auto eview = CellTopology::get_edges_view(CellFamily::Wedge);
    const int steps = std::max(0, p - 1);
    for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        const auto A = wedge_corner_coords(a);
        const auto B = wedge_corner_coords(b);
        for (int k = 1; k <= steps; ++k) {
            const real_t t = static_cast<real_t>(k) / static_cast<real_t>(p);
            std::array<real_t, 3> xi = {
                (1.0 - t) * A[0] + t * B[0],
                (1.0 - t) * A[1] + t * B[1],
                (1.0 - t) * A[2] + t * B[2]
            };
            labels.push_back(wedge_label_from_param(p, xi));
        }
    }

    // faces (oriented face order)
    const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Wedge);
    for (int fi = 0; fi < fview.face_count; ++fi) {
        const int b = fview.offsets[fi];
        const int e = fview.offsets[fi + 1];
        const int fv = e - b;
        if (fv == 3) {
            const int v0 = fview.indices[b + 0];
            const int v1 = fview.indices[b + 1];
            const int v2 = fview.indices[b + 2];
            const auto A = wedge_corner_coords(v0);
            const auto B = wedge_corner_coords(v1);
            const auto C = wedge_corner_coords(v2);
            for (int i = 1; i <= p - 2; ++i) {
                for (int j = 1; j <= p - 1 - i; ++j) {
                    const real_t w1 = static_cast<real_t>(i) / static_cast<real_t>(p);
                    const real_t w2 = static_cast<real_t>(j) / static_cast<real_t>(p);
                    const real_t w0 = 1.0 - w1 - w2;
                    std::array<real_t, 3> xi = {
                        w0 * A[0] + w1 * B[0] + w2 * C[0],
                        w0 * A[1] + w1 * B[1] + w2 * C[1],
                        w0 * A[2] + w1 * B[2] + w2 * C[2]
                    };
                    labels.push_back(wedge_label_from_param(p, xi));
                }
            }
        } else if (fv == 4) {
            const int v0 = fview.indices[b + 0];
            const int v1 = fview.indices[b + 1];
            const int v2 = fview.indices[b + 2];
            const int v3 = fview.indices[b + 3];
            const auto A = wedge_corner_coords(v0);
            const auto B = wedge_corner_coords(v1);
            const auto C = wedge_corner_coords(v2);
            const auto D = wedge_corner_coords(v3);
            for (int i = 1; i <= p - 1; ++i) {
                for (int j = 1; j <= p - 1; ++j) {
                    const real_t u = static_cast<real_t>(i) / static_cast<real_t>(p);
                    const real_t v = static_cast<real_t>(j) / static_cast<real_t>(p);
                    std::array<real_t, 3> xi = {
                        (1.0 - u) * (1.0 - v) * A[0] + u * (1.0 - v) * B[0] + u * v * C[0] + (1.0 - u) * v * D[0],
                        (1.0 - u) * (1.0 - v) * A[1] + u * (1.0 - v) * B[1] + u * v * C[1] + (1.0 - u) * v * D[1],
                        (1.0 - u) * (1.0 - v) * A[2] + u * (1.0 - v) * B[2] + u * v * C[2] + (1.0 - u) * v * D[2]
                    };
                    labels.push_back(wedge_label_from_param(p, xi));
                }
            }
        }
    }

    // volume nodes
    for (int i = 1; i <= p - 2; ++i) {
        for (int j = 1; j <= p - 1 - i; ++j) {
            for (int k = 1; k <= p - 1; ++k) {
                labels.push_back({{p - i - j, i, j}, k});
            }
        }
    }

    return labels;
}

// -------------------------
// Serendipity Quad8 / Hex20
// -------------------------

static void eval_quad8(real_t r, real_t s,
                       std::vector<real_t>& N,
                       std::vector<std::array<real_t, 3>>& dN) {
    N.assign(8, real_t(0));
    dN.assign(8, {0, 0, 0});

    // values
    N[0] = 0.25 * (1 - r) * (1 - s) * (-r - s - 1);
    N[1] = 0.25 * (1 + r) * (1 - s) * ( r - s - 1);
    N[2] = 0.25 * (1 + r) * (1 + s) * ( r + s - 1);
    N[3] = 0.25 * (1 - r) * (1 + s) * (-r + s - 1);
    N[4] = 0.5  * (1 - r * r) * (1 - s);
    N[5] = 0.5  * (1 + r) * (1 - s * s);
    N[6] = 0.5  * (1 - r * r) * (1 + s);
    N[7] = 0.5  * (1 - r) * (1 - s * s);

    // derivatives
    dN[0][0] = 0.25 * (-(1 - s) * (-r - s - 1) + (1 - r) * (1 - s) * (-1));
    dN[0][1] = 0.25 * (-(1 - r) * (-r - s - 1) + (1 - r) * (1 - s) * (-1));

    dN[1][0] = 0.25 * ((1 - s) * ( r - s - 1) + (1 + r) * (1 - s) * (1));
    dN[1][1] = 0.25 * (-(1 + r) * ( r - s - 1) + (1 + r) * (1 - s) * (-1));

    dN[2][0] = 0.25 * ((1 + s) * ( r + s - 1) + (1 + r) * (1 + s) * (1));
    dN[2][1] = 0.25 * ((1 + r) * ( r + s - 1) + (1 + r) * (1 + s) * (1));

    dN[3][0] = 0.25 * (-(1 + s) * (-r + s - 1) + (1 - r) * (1 + s) * (-1));
    dN[3][1] = 0.25 * ((1 - r) * (-r + s - 1) + (1 - r) * (1 + s) * (1));

    dN[4][0] = -r * (1 - s);
    dN[4][1] = -0.5 * (1 - r * r);

    dN[5][0] = 0.5 * (1 - s * s);
    dN[5][1] = (1 + r) * (-s);

    dN[6][0] = -r * (1 + s);
    dN[6][1] = 0.5 * (1 - r * r);

    dN[7][0] = -0.5 * (1 - s * s);
    dN[7][1] = (1 - r) * (-s);
}

static void eval_hex20(real_t r, real_t s, real_t t,
                       std::vector<real_t>& N,
                       std::vector<std::array<real_t, 3>>& dN) {
    N.assign(20, real_t(0));
    dN.assign(20, {0, 0, 0});

    // Corner signs (r,s,t) for nodes 0..7
    static constexpr std::array<std::array<real_t, 3>, 8> cs = {{
        {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
        {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}
    }};

    // Corner nodes
    for (int i = 0; i < 8; ++i) {
        const real_t ri = cs[static_cast<size_t>(i)][0];
        const real_t si = cs[static_cast<size_t>(i)][1];
        const real_t ti = cs[static_cast<size_t>(i)][2];
        const real_t A = 1 + r * ri;
        const real_t B = 1 + s * si;
        const real_t C = 1 + t * ti;
        const real_t D = r * ri + s * si + t * ti - 2;
        N[static_cast<size_t>(i)] = 0.125 * A * B * C * D;
        dN[static_cast<size_t>(i)][0] = (ri / 8.0) * B * C * (D + A);
        dN[static_cast<size_t>(i)][1] = (si / 8.0) * A * C * (D + B);
        dN[static_cast<size_t>(i)][2] = (ti / 8.0) * A * B * (D + C);
    }

    // Edge nodes 8..19 in VTK/CellTopology edge order:
    // 8:(0,1)  9:(1,2) 10:(2,3) 11:(3,0)
    // 12:(4,5) 13:(5,6) 14:(6,7) 15:(7,4)
    // 16:(0,4) 17:(1,5) 18:(2,6) 19:(3,7)
    // Each edge midpoint has one coordinate 0 and the other two ±1.

    // Along r (s,t fixed)
    const auto edge_r = [&](int idx, real_t sgn_s, real_t sgn_t) {
        const real_t A = (1 - r * r);
        const real_t B = (1 + s * sgn_s);
        const real_t C = (1 + t * sgn_t);
        N[static_cast<size_t>(idx)] = 0.25 * A * B * C;
        dN[static_cast<size_t>(idx)][0] = -0.5 * r * B * C;
        dN[static_cast<size_t>(idx)][1] = 0.25 * A * sgn_s * C;
        dN[static_cast<size_t>(idx)][2] = 0.25 * A * B * sgn_t;
    };
    edge_r(8,  -1, -1);
    edge_r(10, +1, -1);
    edge_r(12, -1, +1);
    edge_r(14, +1, +1);

    // Along s (r,t fixed)
    const auto edge_s = [&](int idx, real_t sgn_r, real_t sgn_t) {
        const real_t A = (1 + r * sgn_r);
        const real_t B = (1 - s * s);
        const real_t C = (1 + t * sgn_t);
        N[static_cast<size_t>(idx)] = 0.25 * A * B * C;
        dN[static_cast<size_t>(idx)][0] = 0.25 * sgn_r * B * C;
        dN[static_cast<size_t>(idx)][1] = -0.5 * s * A * C;
        dN[static_cast<size_t>(idx)][2] = 0.25 * A * B * sgn_t;
    };
    edge_s(9,  +1, -1);
    edge_s(11, -1, -1);
    edge_s(13, +1, +1);
    edge_s(15, -1, +1);

    // Along t (r,s fixed)
    const auto edge_t = [&](int idx, real_t sgn_r, real_t sgn_s) {
        const real_t A = (1 + r * sgn_r);
        const real_t B = (1 + s * sgn_s);
        const real_t C = (1 - t * t);
        N[static_cast<size_t>(idx)] = 0.25 * A * B * C;
        dN[static_cast<size_t>(idx)][0] = 0.25 * sgn_r * B * C;
        dN[static_cast<size_t>(idx)][1] = 0.25 * A * sgn_s * C;
        dN[static_cast<size_t>(idx)][2] = -0.5 * t * A * B;
    };
    edge_t(16, -1, -1);
    edge_t(17, +1, -1);
    edge_t(18, +1, +1);
    edge_t(19, -1, +1);
}

// ------------------------------------------
// Serendipity Wedge15 (quadratic) via monoms
// ------------------------------------------

static void eval_wedge15(real_t r, real_t s, real_t t,
                         std::vector<real_t>& N,
                         std::vector<std::array<real_t, 3>>& dN) {
    N.assign(15, real_t(0));
    dN.assign(15, {0, 0, 0});

    static const int wedge15_monomial_exponents[15][3] = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 0, 2},
        {0, 1, 0},
        {0, 1, 1},
        {0, 1, 2},
        {0, 2, 0},
        {0, 2, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 0, 2},
        {1, 1, 0},
        {1, 1, 1},
        {2, 0, 0},
        {2, 0, 1}
    };

    static const real_t wedge15_coeffs[15][15] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {-0.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0.5, -0, -0, 0.5, -0, -0, -0, -0, -0, -0, -0, -0, -1, -0, -0},
        {-1, 0, -1, -1, 0, -1, 0, 0, 2, 0, 0, 2, -1, 0, 1},
        {1.5, 0, 0.5, -1.5, 0, -0.5, 0, 0, -2, 0, 0, 2, 0, 0, 0},
        {-0.5, -0, 0.5, -0.5, -0, 0.5, -0, -0, -0, -0, -0, -0, 1, -0, -1},
        {1, 0, 1, 1, 0, 1, 0, 0, -2, 0, 0, -2, 0, 0, 0},
        {-1, 0, -1, 1, 0, 1, 0, 0, 2, 0, 0, -2, 0, 0, 0},
        {-1, -1, 0, -1, -1, 0, 2, 0, 0, 2, 0, 0, -1, 1, 0},
        {1.5, 0.5, 0, -1.5, -0.5, 0, -2, 0, 0, 2, 0, 0, 0, 0, 0},
        {-0.5, 0.5, -0, -0.5, 0.5, -0, -0, -0, -0, -0, -0, -0, 1, -1, -0},
        {2, 0, -0, 2, 0, -0, -2, 2, -2, -2, 2, -2, -0, -0, -0},
        {-2, 0, 0, 2, 0, 0, 2, -2, 2, -2, 2, -2, 0, 0, 0},
        {1, 1, -0, 1, 1, -0, -2, -0, -0, -2, -0, -0, -0, -0, -0},
        {-1, -1, -0, 1, 1, -0, 2, -0, -0, -2, -0, -0, -0, -0, -0}
    };

    real_t phi[15];
    real_t dphi_dr[15], dphi_ds[15], dphi_dt[15];

    for (int j = 0; j < 15; ++j) {
        const int a = wedge15_monomial_exponents[j][0];
        const int b = wedge15_monomial_exponents[j][1];
        const int c = wedge15_monomial_exponents[j][2];

        auto powi = [](real_t x, int e) {
            real_t v = 1.0;
            for (int k = 0; k < e; ++k) v *= x;
            return v;
        };

        const real_t ra = powi(r, a);
        const real_t sb = powi(s, b);
        const real_t tc = powi(t, c);
        phi[j] = ra * sb * tc;

        dphi_dr[j] = (a > 0) ? static_cast<real_t>(a) * powi(r, a - 1) * sb * tc : 0.0;
        dphi_ds[j] = (b > 0) ? ra * static_cast<real_t>(b) * powi(s, b - 1) * tc : 0.0;
        dphi_dt[j] = (c > 0) ? ra * sb * static_cast<real_t>(c) * powi(t, c - 1) : 0.0;
    }

    for (int i = 0; i < 15; ++i) {
        real_t v = 0.0;
        real_t gr = 0.0, gs = 0.0, gt = 0.0;
        for (int j = 0; j < 15; ++j) {
            v  += wedge15_coeffs[j][i] * phi[j];
            gr += wedge15_coeffs[j][i] * dphi_dr[j];
            gs += wedge15_coeffs[j][i] * dphi_ds[j];
            gt += wedge15_coeffs[j][i] * dphi_dt[j];
        }
        N[static_cast<size_t>(i)] = v;
        dN[static_cast<size_t>(i)][0] = gr;
        dN[static_cast<size_t>(i)][1] = gs;
        dN[static_cast<size_t>(i)][2] = gt;
    }
}

// ---------------------------
// Serendipity Pyramid13 via collapsed Hex20
// ---------------------------

static void eval_pyramid13(real_t x, real_t y, real_t z,
                           std::vector<real_t>& N,
                           std::vector<std::array<real_t, 3>>& dN) {
    N.assign(13, real_t(0));
    dN.assign(13, {0, 0, 0});

    const real_t one_minus_z = 1.0 - z;
    if (std::abs(one_minus_z) < 1e-12) {
        N[4] = 1.0;
        return;
    }

    const real_t w = 2.0 * z - 1.0;
    const real_t inv_omz = 1.0 / one_minus_z;
    const real_t u = x * inv_omz;
    const real_t v = y * inv_omz;

    std::vector<real_t> hexN;
    std::vector<std::array<real_t, 3>> hexd;
    eval_hex20(u, v, w, hexN, hexd);

    // chain rule to pyramid coordinates (x,y,z)
    std::vector<std::array<real_t, 3>> hex_grad_xyz(20, {0, 0, 0});
    for (int i = 0; i < 20; ++i) {
        const real_t dN_du = hexd[static_cast<size_t>(i)][0];
        const real_t dN_dv = hexd[static_cast<size_t>(i)][1];
        const real_t dN_dw = hexd[static_cast<size_t>(i)][2];
        hex_grad_xyz[static_cast<size_t>(i)][0] = dN_du * inv_omz;
        hex_grad_xyz[static_cast<size_t>(i)][1] = dN_dv * inv_omz;
        hex_grad_xyz[static_cast<size_t>(i)][2] = dN_du * u * inv_omz + dN_dv * v * inv_omz + dN_dw * 2.0;
    }

    // Map Hex20 nodes to Pyramid13 nodes:
    // Pyramid13 ordering: corners 0-3, apex 4, base edges 5-8, rising edges 9-12
    N[0] = hexN[0]; N[1] = hexN[1]; N[2] = hexN[2]; N[3] = hexN[3];
    dN[0] = hex_grad_xyz[0]; dN[1] = hex_grad_xyz[1]; dN[2] = hex_grad_xyz[2]; dN[3] = hex_grad_xyz[3];

    // Apex: sum top-face corners (4..7) and top edges (12..15)
    for (int hi : {4,5,6,7,12,13,14,15}) {
        N[4] += hexN[static_cast<size_t>(hi)];
        dN[4][0] += hex_grad_xyz[static_cast<size_t>(hi)][0];
        dN[4][1] += hex_grad_xyz[static_cast<size_t>(hi)][1];
        dN[4][2] += hex_grad_xyz[static_cast<size_t>(hi)][2];
    }

    // Base edges
    N[5] = hexN[8];  dN[5] = hex_grad_xyz[8];
    N[6] = hexN[9];  dN[6] = hex_grad_xyz[9];
    N[7] = hexN[10]; dN[7] = hex_grad_xyz[10];
    N[8] = hexN[11]; dN[8] = hex_grad_xyz[11];

    // Rising edges
    N[9]  = hexN[16]; dN[9]  = hex_grad_xyz[16];
    N[10] = hexN[17]; dN[10] = hex_grad_xyz[17];
    N[11] = hexN[18]; dN[11] = hex_grad_xyz[18];
    N[12] = hexN[19]; dN[12] = hex_grad_xyz[19];
}

// ------------------------------------------
// Lagrange Pyramid (stable basis)
// ------------------------------------------
static void eval_pyramid_lagrange(int p, const ParametricPoint& xi,
                                  std::vector<real_t>& N,
                                  std::vector<std::array<real_t, 3>>& dN) {
    PyramidBasis::eval_lagrange(p, xi, N, dN);
}

} // namespace

//=============================================================================
// Jacobian Implementation
//=============================================================================

real_t Jacobian::determinant() const {
    if (parametric_dim != 3) {
        throw std::runtime_error("Jacobian::determinant only defined for 3D Jacobians");
    }

    const auto& J = matrix;
    return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
           J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
           J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
}

Jacobian Jacobian::inverse() const {
    if (parametric_dim != 3) {
        throw std::runtime_error("Jacobian::inverse only defined for 3×3 Jacobians");
    }

    const real_t det = determinant();
    if (std::abs(det) < GeometryConfig::volume_epsilon()) {
        throw std::runtime_error("Jacobian::inverse singular Jacobian");
    }

    Jacobian inv;
    inv.parametric_dim = 3;
    inv.physical_dim = 3;

    const auto& J = matrix;
    const real_t inv_det = 1.0 / det;

    inv.matrix[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * inv_det;
    inv.matrix[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * inv_det;
    inv.matrix[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * inv_det;
    inv.matrix[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * inv_det;
    inv.matrix[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * inv_det;
    inv.matrix[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * inv_det;
    inv.matrix[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * inv_det;
    inv.matrix[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * inv_det;
    inv.matrix[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * inv_det;

    return inv;
}

std::array<std::array<real_t, 3>, 3> Jacobian::metric_tensor() const {
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
    }
    if (parametric_dim == 2) {
        auto G = metric_tensor();
        const real_t det_G = G[0][0] * G[1][1] - G[0][1] * G[1][0];
        return std::sqrt(std::max(real_t(0), det_G));
    }
    if (parametric_dim == 1) {
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

    const CellShape shape = mesh.cell_shape(cell);
    const auto [verts, n_nodes] = mesh.cell_vertices_span(cell);
    const int order = deduce_order(shape, n_nodes);

    const auto sf = evaluate_shape_functions(shape, n_nodes, xi);

    const std::vector<real_t>& X = ((cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords())
                                     ? mesh.X_cur() : mesh.X_ref();
    const int dim = mesh.dim();

    // Compute physical coordinates
    eval.coordinates = {0, 0, 0};
    for (size_t i = 0; i < n_nodes; ++i) {
        const index_t v = verts[i];
        std::array<real_t, 3> p = {0, 0, 0};
        for (int d = 0; d < dim; ++d) {
            p[static_cast<size_t>(d)] = X[static_cast<size_t>(v * dim + d)];
        }
        eval.coordinates[0] += sf.N[i] * p[0];
        eval.coordinates[1] += sf.N[i] * p[1];
        eval.coordinates[2] += sf.N[i] * p[2];
    }

    // Compute Jacobian
    const int pdim = parametric_dimension(shape.family);
    eval.jacobian.parametric_dim = pdim;
    eval.jacobian.physical_dim = 3;

    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < pdim; ++l) {
            real_t acc = 0.0;
            for (size_t i = 0; i < n_nodes; ++i) {
                const index_t v = verts[i];
                const real_t xk = (k < dim) ? X[static_cast<size_t>(v * dim + k)] : 0.0;
                acc += sf.dN_dxi[i][static_cast<size_t>(l)] * xk;
            }
            eval.jacobian.matrix[static_cast<size_t>(k)][static_cast<size_t>(l)] = acc;
        }
    }

    if (pdim == 3) {
        eval.det_jacobian = eval.jacobian.determinant();
        eval.is_valid = eval.det_jacobian > GeometryConfig::volume_epsilon();
    } else {
        eval.det_jacobian = eval.jacobian.metric_determinant();
        eval.is_valid = eval.det_jacobian > GeometryConfig::area_epsilon();
    }

    (void)order; // order currently tracked only in ShapeFunctionValues
    return eval;
}

ShapeFunctionValues CurvilinearEvaluator::evaluate_shape_functions(
    const CellShape& shape,
    size_t n_nodes,
    const ParametricPoint& xi) {

    const int order = deduce_order(shape, n_nodes);

    switch (shape.family) {
        case CellFamily::Point:
            return ShapeFunctionValues{{1.0}, {}, {}, 0};
        case CellFamily::Line:
            return eval_line_shape_functions(order, xi);
        case CellFamily::Triangle:
            return eval_triangle_shape_functions(order, xi);
        case CellFamily::Quad:
            return eval_quad_shape_functions(order, n_nodes, xi);
        case CellFamily::Tetra:
            return eval_tet_shape_functions(order, xi);
        case CellFamily::Hex:
            return eval_hex_shape_functions(order, n_nodes, xi);
        case CellFamily::Wedge:
            return eval_wedge_shape_functions(order, n_nodes, xi);
        case CellFamily::Pyramid:
            return eval_pyramid_shape_functions(order, n_nodes, xi);
        default:
            throw std::runtime_error("CurvilinearEvaluator::evaluate_shape_functions: unsupported cell family");
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

    const CellShape shape = mesh.cell_shape(cell);
    ParametricPoint xi = reference_element_center(shape);

    const int pdim = parametric_dimension(shape.family);
    if (pdim == 0) return {xi, true};

    for (int iter = 0; iter < max_iterations; ++iter) {
        const auto eval = evaluate_geometry(mesh, cell, xi, cfg);
        std::array<real_t, 3> r = {
            physical_point[0] - eval.coordinates[0],
            physical_point[1] - eval.coordinates[1],
            physical_point[2] - eval.coordinates[2]
        };

        const real_t rnorm = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        if (rnorm < tolerance) {
            return {xi, true};
        }

        std::array<real_t, 3> dxi = {0, 0, 0};

        if (pdim == 1) {
            // least squares along a 3-vector
            std::array<real_t, 3> J0 = {eval.jacobian.matrix[0][0], eval.jacobian.matrix[1][0], eval.jacobian.matrix[2][0]};
            const real_t denom = J0[0] * J0[0] + J0[1] * J0[1] + J0[2] * J0[2];
            if (denom < GeometryConfig::length_epsilon()) return {xi, false};
            const real_t num = J0[0] * r[0] + J0[1] * r[1] + J0[2] * r[2];
            dxi[0] = num / denom;
        } else if (pdim == 2) {
            // Solve (J^T J) dxi = J^T r
            const real_t J00 = eval.jacobian.matrix[0][0], J01 = eval.jacobian.matrix[0][1];
            const real_t J10 = eval.jacobian.matrix[1][0], J11 = eval.jacobian.matrix[1][1];
            const real_t J20 = eval.jacobian.matrix[2][0], J21 = eval.jacobian.matrix[2][1];

            const real_t G00 = J00 * J00 + J10 * J10 + J20 * J20;
            const real_t G01 = J00 * J01 + J10 * J11 + J20 * J21;
            const real_t G11 = J01 * J01 + J11 * J11 + J21 * J21;
            const real_t detG = G00 * G11 - G01 * G01;
            if (std::abs(detG) < GeometryConfig::area_epsilon()) return {xi, false};

            const real_t rhs0 = J00 * r[0] + J10 * r[1] + J20 * r[2];
            const real_t rhs1 = J01 * r[0] + J11 * r[1] + J21 * r[2];

            dxi[0] = ( G11 * rhs0 - G01 * rhs1) / detG;
            dxi[1] = (-G01 * rhs0 + G00 * rhs1) / detG;
        } else {
            // 3D: solve J dxi = r via explicit inverse
            const auto inv = eval.jacobian.inverse();
            for (int i = 0; i < 3; ++i) {
                dxi[static_cast<size_t>(i)] =
                    inv.matrix[static_cast<size_t>(i)][0] * r[0] +
                    inv.matrix[static_cast<size_t>(i)][1] * r[1] +
                    inv.matrix[static_cast<size_t>(i)][2] * r[2];
            }
        }

        xi[0] += dxi[0];
        xi[1] += dxi[1];
        xi[2] += dxi[2];
    }

    return {xi, false};
}

bool CurvilinearEvaluator::is_inside_reference_element(
    const CellShape& shape,
    const ParametricPoint& xi,
    real_t tolerance) {

    const real_t x = xi[0];
    const real_t y = xi[1];
    const real_t z = xi[2];
    const real_t tol = std::abs(tolerance);

    switch (shape.family) {
        case CellFamily::Point:
            return true;
        case CellFamily::Line:
            return x >= -1.0 - tol && x <= 1.0 + tol;
        case CellFamily::Triangle:
            return x >= -tol && y >= -tol && (x + y) <= 1.0 + tol;
        case CellFamily::Quad:
            return x >= -1.0 - tol && x <= 1.0 + tol && y >= -1.0 - tol && y <= 1.0 + tol;
        case CellFamily::Tetra:
            return x >= -tol && y >= -tol && z >= -tol && (x + y + z) <= 1.0 + tol;
        case CellFamily::Hex:
            return x >= -1.0 - tol && x <= 1.0 + tol &&
                   y >= -1.0 - tol && y <= 1.0 + tol &&
                   z >= -1.0 - tol && z <= 1.0 + tol;
        case CellFamily::Wedge:
            return x >= -tol && y >= -tol && (x + y) <= 1.0 + tol &&
                   z >= -1.0 - tol && z <= 1.0 + tol;
        case CellFamily::Pyramid:
            return z >= -tol && z <= 1.0 + tol &&
                   std::abs(x) <= (1.0 - z) + tol &&
                   std::abs(y) <= (1.0 - z) + tol;
        default:
            return false;
    }
}

ParametricPoint CurvilinearEvaluator::reference_element_center(const CellShape& shape) {
    switch (shape.family) {
        case CellFamily::Point:
        case CellFamily::Line:
            return {0, 0, 0};
        case CellFamily::Triangle:
            return {1.0 / 3.0, 1.0 / 3.0, 0};
        case CellFamily::Quad:
            return {0, 0, 0};
        case CellFamily::Tetra:
            return {0.25, 0.25, 0.25};
        case CellFamily::Hex:
            return {0, 0, 0};
        case CellFamily::Wedge:
            return {1.0 / 3.0, 1.0 / 3.0, 0};
        case CellFamily::Pyramid:
            return {0, 0, 0.25};
        default:
            return {0, 0, 0};
    }
}

int CurvilinearEvaluator::deduce_order(const CellShape& shape, size_t n_nodes) {
    // Prefer explicit order if present.
    if (shape.order > 1) {
        return shape.order;
    }

    // Infer from node count for known families when order is unset/linear.
    int p = CellTopology::infer_lagrange_order(shape.family, n_nodes);
    if (p > 0) return p;

    // Common quadratic serendipity fallbacks.
    if (shape.family == CellFamily::Quad && n_nodes == 8) return 2;
    if (shape.family == CellFamily::Hex && n_nodes == 20) return 2;
    if (shape.family == CellFamily::Wedge && n_nodes == 15) return 2;
    if (shape.family == CellFamily::Pyramid && (n_nodes == 13 || n_nodes == 14)) return 2;

    p = CellTopology::infer_serendipity_order(shape.family, n_nodes);
    if (p > 0) return p;

    return 1;
}

//=============================================================================
// Shape Function Evaluation
//=============================================================================

ShapeFunctionValues CurvilinearEvaluator::eval_line_shape_functions(
    int order, const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    std::vector<real_t> N1, dN1;
    lagrange_1d_equispaced(order, xi[0], N1, dN1);

    vals.N = std::move(N1);
    vals.dN_dxi.assign(vals.N.size(), {0, 0, 0});
    for (size_t i = 0; i < vals.N.size(); ++i) {
        vals.dN_dxi[i][0] = dN1[i];
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_triangle_shape_functions(
    int order, const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    const real_t l1 = xi[0];
    const real_t l2 = xi[1];
    const real_t l0 = 1.0 - l1 - l2;

    std::vector<real_t> phi0, dphi0, phi1, dphi1, phi2, dphi2;
    simplex_lagrange_factor_sequence(order, l0, phi0, dphi0);
    simplex_lagrange_factor_sequence(order, l1, phi1, dphi1);
    simplex_lagrange_factor_sequence(order, l2, phi2, dphi2);

    const auto exps = triangle_exponents_vtk(order);
    vals.N.assign(exps.size(), 0.0);
    vals.dN_dxi.assign(exps.size(), {0, 0, 0});

    for (size_t n = 0; n < exps.size(); ++n) {
        const int a0 = exps[n][0];
        const int a1 = exps[n][1];
        const int a2 = exps[n][2];

        const real_t f0 = phi0[static_cast<size_t>(a0)];
        const real_t f1 = phi1[static_cast<size_t>(a1)];
        const real_t f2 = phi2[static_cast<size_t>(a2)];

        vals.N[n] = f0 * f1 * f2;

        const real_t d0 = dphi0[static_cast<size_t>(a0)] * f1 * f2;
        const real_t d1 = f0 * dphi1[static_cast<size_t>(a1)] * f2;
        const real_t d2 = f0 * f1 * dphi2[static_cast<size_t>(a2)];

        // d/dxi = -d/dl0 + d/dl1, d/deta = -d/dl0 + d/dl2
        vals.dN_dxi[n][0] = -d0 + d1;
        vals.dN_dxi[n][1] = -d0 + d2;
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_quad_shape_functions(
    int order, size_t n_nodes, const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    if (is_serendipity_quadratic(CellFamily::Quad, order, n_nodes)) {
        eval_quad8(xi[0], xi[1], vals.N, vals.dN_dxi);
        return vals;
    }

    const size_t expected = lagrange_node_count(CellFamily::Quad, order);
    if (n_nodes != expected) {
        throw std::runtime_error("CurvilinearEval: unsupported quad node layout");
    }

    std::vector<real_t> Nx, dNx, Ny, dNy;
    lagrange_1d_equispaced(order, xi[0], Nx, dNx);
    lagrange_1d_equispaced(order, xi[1], Ny, dNy);

    const auto idx = quad_lagrange_indices_vtk(order);
    vals.N.assign(idx.size(), 0.0);
    vals.dN_dxi.assign(idx.size(), {0, 0, 0});

    for (size_t n = 0; n < idx.size(); ++n) {
        const int i = idx[n][0];
        const int j = idx[n][1];
        vals.N[n] = Nx[static_cast<size_t>(i)] * Ny[static_cast<size_t>(j)];
        vals.dN_dxi[n][0] = dNx[static_cast<size_t>(i)] * Ny[static_cast<size_t>(j)];
        vals.dN_dxi[n][1] = Nx[static_cast<size_t>(i)] * dNy[static_cast<size_t>(j)];
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_tet_shape_functions(
    int order, const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    const real_t l1 = xi[0];
    const real_t l2 = xi[1];
    const real_t l3 = xi[2];
    const real_t l0 = 1.0 - l1 - l2 - l3;

    std::vector<real_t> phi0, dphi0, phi1, dphi1, phi2, dphi2, phi3, dphi3;
    simplex_lagrange_factor_sequence(order, l0, phi0, dphi0);
    simplex_lagrange_factor_sequence(order, l1, phi1, dphi1);
    simplex_lagrange_factor_sequence(order, l2, phi2, dphi2);
    simplex_lagrange_factor_sequence(order, l3, phi3, dphi3);

    const auto exps = tetra_exponents_vtk(order);
    vals.N.assign(exps.size(), 0.0);
    vals.dN_dxi.assign(exps.size(), {0, 0, 0});

    for (size_t n = 0; n < exps.size(); ++n) {
        const int a0 = exps[n][0];
        const int a1 = exps[n][1];
        const int a2 = exps[n][2];
        const int a3 = exps[n][3];

        const real_t f0 = phi0[static_cast<size_t>(a0)];
        const real_t f1 = phi1[static_cast<size_t>(a1)];
        const real_t f2 = phi2[static_cast<size_t>(a2)];
        const real_t f3 = phi3[static_cast<size_t>(a3)];

        vals.N[n] = f0 * f1 * f2 * f3;

        const real_t d0 = dphi0[static_cast<size_t>(a0)] * f1 * f2 * f3;
        const real_t d1 = f0 * dphi1[static_cast<size_t>(a1)] * f2 * f3;
        const real_t d2 = f0 * f1 * dphi2[static_cast<size_t>(a2)] * f3;
        const real_t d3 = f0 * f1 * f2 * dphi3[static_cast<size_t>(a3)];

        vals.dN_dxi[n][0] = -d0 + d1;
        vals.dN_dxi[n][1] = -d0 + d2;
        vals.dN_dxi[n][2] = -d0 + d3;
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_hex_shape_functions(
    int order, size_t n_nodes, const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    if (is_serendipity_quadratic(CellFamily::Hex, order, n_nodes)) {
        eval_hex20(xi[0], xi[1], xi[2], vals.N, vals.dN_dxi);
        return vals;
    }

    const size_t expected = lagrange_node_count(CellFamily::Hex, order);
    if (n_nodes != expected) {
        throw std::runtime_error("CurvilinearEval: unsupported hex node layout");
    }

    std::vector<real_t> Nx, dNx, Ny, dNy, Nz, dNz;
    lagrange_1d_equispaced(order, xi[0], Nx, dNx);
    lagrange_1d_equispaced(order, xi[1], Ny, dNy);
    lagrange_1d_equispaced(order, xi[2], Nz, dNz);

    const auto idx = hex_lagrange_indices_vtk(order);
    vals.N.assign(idx.size(), 0.0);
    vals.dN_dxi.assign(idx.size(), {0, 0, 0});

    for (size_t n = 0; n < idx.size(); ++n) {
        const int i = idx[n][0];
        const int j = idx[n][1];
        const int k = idx[n][2];
        const real_t ax = Nx[static_cast<size_t>(i)];
        const real_t ay = Ny[static_cast<size_t>(j)];
        const real_t az = Nz[static_cast<size_t>(k)];
        vals.N[n] = ax * ay * az;
        vals.dN_dxi[n][0] = dNx[static_cast<size_t>(i)] * ay * az;
        vals.dN_dxi[n][1] = ax * dNy[static_cast<size_t>(j)] * az;
        vals.dN_dxi[n][2] = ax * ay * dNz[static_cast<size_t>(k)];
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_wedge_shape_functions(
    int order, size_t n_nodes, const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    if (is_serendipity_quadratic(CellFamily::Wedge, order, n_nodes)) {
        eval_wedge15(xi[0], xi[1], xi[2], vals.N, vals.dN_dxi);
        return vals;
    }

    const size_t expected = lagrange_node_count(CellFamily::Wedge, order);
    if (n_nodes != expected) {
        throw std::runtime_error("CurvilinearEval: unsupported wedge node layout");
    }

    // Triangle part in barycentric coords
    const real_t l1 = xi[0];
    const real_t l2 = xi[1];
    const real_t l0 = 1.0 - l1 - l2;
    std::vector<real_t> phi0, dphi0, phi1, dphi1, phi2, dphi2;
    simplex_lagrange_factor_sequence(order, l0, phi0, dphi0);
    simplex_lagrange_factor_sequence(order, l1, phi1, dphi1);
    simplex_lagrange_factor_sequence(order, l2, phi2, dphi2);

    // Line part along zeta in [-1,1]
    std::vector<real_t> Lz, dLz;
    lagrange_1d_equispaced(order, xi[2], Lz, dLz);

    const auto labels = wedge_lagrange_labels_vtk(order);
    vals.N.assign(labels.size(), 0.0);
    vals.dN_dxi.assign(labels.size(), {0, 0, 0});

    for (size_t n = 0; n < labels.size(); ++n) {
        const int a0 = labels[n].tri_exp[0];
        const int a1 = labels[n].tri_exp[1];
        const int a2 = labels[n].tri_exp[2];
        const int kz = labels[n].kz;

        const real_t f0 = phi0[static_cast<size_t>(a0)];
        const real_t f1 = phi1[static_cast<size_t>(a1)];
        const real_t f2 = phi2[static_cast<size_t>(a2)];
        const real_t triN = f0 * f1 * f2;

        const real_t d0 = dphi0[static_cast<size_t>(a0)] * f1 * f2;
        const real_t d1 = f0 * dphi1[static_cast<size_t>(a1)] * f2;
        const real_t d2 = f0 * f1 * dphi2[static_cast<size_t>(a2)];
        const real_t dtri_dxi  = -d0 + d1;
        const real_t dtri_deta = -d0 + d2;

        vals.N[n] = triN * Lz[static_cast<size_t>(kz)];
        vals.dN_dxi[n][0] = dtri_dxi * Lz[static_cast<size_t>(kz)];
        vals.dN_dxi[n][1] = dtri_deta * Lz[static_cast<size_t>(kz)];
        vals.dN_dxi[n][2] = triN * dLz[static_cast<size_t>(kz)];
    }

    return vals;
}

ShapeFunctionValues CurvilinearEvaluator::eval_pyramid_shape_functions(
    int order, size_t n_nodes, const ParametricPoint& xi) {

    ShapeFunctionValues vals;
    vals.polynomial_order = order;

    if (is_serendipity_quadratic(CellFamily::Pyramid, order, n_nodes)) {
        eval_pyramid13(xi[0], xi[1], xi[2], vals.N, vals.dN_dxi);
        return vals;
    }

    const size_t expected = lagrange_node_count(CellFamily::Pyramid, order);
    if (n_nodes != expected) {
        throw std::runtime_error("CurvilinearEval: unsupported pyramid node layout");
    }

    eval_pyramid_lagrange(order, xi, vals.N, vals.dN_dxi);
    return vals;
}

//=============================================================================
// QuadratureRule (minimal implementations)
//=============================================================================

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::get_quadrature_rule(const CellShape& shape, int order) {
    // Minimal sampling rules; weights are not optimized for exactness.
    if (order < 1) order = 1;

    if (shape.family == CellFamily::Line) {
        return gauss_legendre_1d(std::min(5, std::max(1, (order + 1) / 2)));
    }

    if (shape.family == CellFamily::Quad || shape.family == CellFamily::Hex) {
        return tensor_product_quadrature(shape, order);
    }

    // Simplex rules: use centroid-only as a safe default.
    std::vector<QuadraturePoint> pts;
    QuadraturePoint qp;
    qp.weight = 1.0;
    qp.xi = CurvilinearEvaluator::reference_element_center(shape);
    pts.push_back(qp);
    return pts;
}

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::gauss_legendre_1d(int n_points) {
    std::vector<QuadraturePoint> q;
    if (n_points <= 1) {
        q.push_back({{0,0,0}, 2.0});
        return q;
    }
    if (n_points == 2) {
        const real_t a = std::sqrt(1.0 / 3.0);
        q.push_back({{-a,0,0}, 1.0});
        q.push_back({{ a,0,0}, 1.0});
        return q;
    }
    if (n_points == 3) {
        const real_t a = std::sqrt(3.0 / 5.0);
        q.push_back({{-a,0,0}, 5.0/9.0});
        q.push_back({{ 0,0,0}, 8.0/9.0});
        q.push_back({{ a,0,0}, 5.0/9.0});
        return q;
    }
    if (n_points == 4) {
        const real_t a1 = 0.3399810435848563;
        const real_t a2 = 0.8611363115940526;
        const real_t w1 = 0.6521451548625461;
        const real_t w2 = 0.3478548451374539;
        q.push_back({{-a2,0,0}, w2});
        q.push_back({{-a1,0,0}, w1});
        q.push_back({{ a1,0,0}, w1});
        q.push_back({{ a2,0,0}, w2});
        return q;
    }
    // n_points == 5
    const real_t a1 = 0.5384693101056831;
    const real_t a2 = 0.9061798459386640;
    const real_t w0 = 0.5688888888888889;
    const real_t w1 = 0.4786286704993665;
    const real_t w2 = 0.2369268850561891;
    q.push_back({{-a2,0,0}, w2});
    q.push_back({{-a1,0,0}, w1});
    q.push_back({{ 0,0,0}, w0});
    q.push_back({{ a1,0,0}, w1});
    q.push_back({{ a2,0,0}, w2});
    return q;
}

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::triangle_quadrature(int order) {
    (void)order;
    // Minimal centroid rule on reference triangle area = 0.5
    return {{{{1.0/3.0, 1.0/3.0, 0.0}}, 0.5}};
}

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::tet_quadrature(int order) {
    (void)order;
    // Minimal centroid rule on reference tet volume = 1/6
    return {{{{0.25, 0.25, 0.25}}, 1.0/6.0}};
}

std::vector<QuadratureRule::QuadraturePoint>
QuadratureRule::tensor_product_quadrature(const CellShape& shape, int order) {
    const int dim = (shape.family == CellFamily::Hex) ? 3 : 2;
    auto q1 = gauss_legendre_1d(std::min(5, std::max(1, (order + 1) / 2)));

    std::vector<QuadraturePoint> out;
    if (dim == 2) {
        for (const auto& a : q1) {
            for (const auto& b : q1) {
                out.push_back({{a.xi[0], b.xi[0], 0.0}, a.weight * b.weight});
            }
        }
    } else {
        for (const auto& a : q1) {
            for (const auto& b : q1) {
                for (const auto& c : q1) {
                    out.push_back({{a.xi[0], b.xi[0], c.xi[0]}, a.weight * b.weight * c.weight});
                }
            }
        }
    }
    return out;
}

void QuadratureRule::gauss_legendre_nodes(
    int n,
    std::vector<real_t>& nodes,
    std::vector<real_t>& weights) {
    // Deprecated by fixed tables above; keep for API completeness.
    (void)n;
    nodes.clear();
    weights.clear();
}

//=============================================================================
// DistortionDetector
//=============================================================================

DistortionDetector::DistortionMetrics
DistortionDetector::detect_distortion(
    const MeshBase& mesh,
    index_t cell,
    Configuration cfg,
    int sampling_order) {

    DistortionMetrics m;
    const CellShape shape = mesh.cell_shape(cell);
    const auto q = QuadratureRule::get_quadrature_rule(shape, sampling_order);

    for (const auto& qp : q) {
        const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, qp.xi, cfg);
        const real_t det = eval.det_jacobian;
        m.min_jacobian = std::min(m.min_jacobian, det);
        m.max_jacobian = std::max(m.max_jacobian, det);
        if (det < 0.0) {
            m.has_negative_jacobian = true;
            m.negative_jacobian_points.push_back(qp.xi);
        }
    }

    if (m.min_jacobian > 0.0) {
        m.jacobian_ratio = m.max_jacobian / m.min_jacobian;
        m.is_highly_distorted = (m.jacobian_ratio > 10.0);
    } else {
        m.jacobian_ratio = 0.0;
        m.is_highly_distorted = true;
    }

    return m;
}

std::vector<index_t>
DistortionDetector::find_inverted_cells(const MeshBase& mesh, Configuration cfg) {
    std::vector<index_t> bad;
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
        auto m = detect_distortion(mesh, c, cfg, 2);
        if (m.has_negative_jacobian) bad.push_back(c);
    }
    return bad;
}

} // namespace svmp
