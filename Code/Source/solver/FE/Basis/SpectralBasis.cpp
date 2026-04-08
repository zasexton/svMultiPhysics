/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SpectralBasis.h"
#include "OrthogonalPolynomials.h"
#include "Quadrature/GaussLobattoQuadrature.h"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace svmp {
namespace FE {
namespace basis {

namespace {

bool is_line(ElementType t) {
    return t == ElementType::Line2 || t == ElementType::Line3;
}

bool is_quadrilateral(ElementType t) {
    return t == ElementType::Quad4 || t == ElementType::Quad8 || t == ElementType::Quad9;
}

bool is_hexahedron(ElementType t) {
    return t == ElementType::Hex8 || t == ElementType::Hex20 || t == ElementType::Hex27;
}

bool is_triangle(ElementType t) {
    return t == ElementType::Triangle3 || t == ElementType::Triangle6;
}

bool is_tetrahedron(ElementType t) {
    return t == ElementType::Tetra4 || t == ElementType::Tetra10;
}

bool is_wedge(ElementType t) {
    return t == ElementType::Wedge6 || t == ElementType::Wedge15 || t == ElementType::Wedge18;
}

bool is_pyramid(ElementType t) {
    return t == ElementType::Pyramid5 || t == ElementType::Pyramid13 || t == ElementType::Pyramid14;
}

// -----------------------------------------------------------------------
// Warp & Blend helper: 1D warp function
// Given equispaced nodes on [-1,1] and GLL nodes, compute the displacement
// that maps equispaced -> GLL at each equispaced location.
// -----------------------------------------------------------------------
std::vector<Real> compute_warp_1d(int order) {
    const int n = order + 1;
    // Equispaced nodes on [-1, 1]
    std::vector<Real> equi(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        equi[static_cast<std::size_t>(i)] = Real(-1) + Real(2) * Real(i) / Real(order);
    }
    // GLL nodes
    auto gll_raw = quadrature::GaussLobattoQuadrature1D::generate_raw(n);
    const auto& gll = gll_raw.first;

    // Warp = GLL - equispaced
    std::vector<Real> warp(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        warp[static_cast<std::size_t>(i)] = gll[static_cast<std::size_t>(i)] - equi[static_cast<std::size_t>(i)];
    }
    return warp;
}

// Evaluate the 1D warp at an arbitrary point r in [-1,1] via Lagrange interpolation
// from the equispaced warp values.
Real eval_warp(const std::vector<Real>& warp, int order, Real r) {
    const int n = order + 1;
    // Equispaced nodes
    Real result = Real(0);
    for (int i = 0; i < n; ++i) {
        Real xi = Real(-1) + Real(2) * Real(i) / Real(order);
        Real li = Real(1);
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            Real xj = Real(-1) + Real(2) * Real(j) / Real(order);
            li *= (r - xj) / (xi - xj);
        }
        result += warp[static_cast<std::size_t>(i)] * li;
    }
    return result;
}

// Gauss-Jordan inversion of a row-major n x n matrix in-place.
// Returns false if singular.
bool invert_matrix(std::vector<Real>& A, std::size_t n) {
    std::vector<Real> inv(n * n, Real(0));
    for (std::size_t i = 0; i < n; ++i) inv[i * n + i] = Real(1);

    for (std::size_t col = 0; col < n; ++col) {
        // Partial pivoting
        std::size_t pivot = col;
        Real max_val = std::abs(A[col * n + col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            Real v = std::abs(A[row * n + col]);
            if (v > max_val) { max_val = v; pivot = row; }
        }
        if (max_val < Real(1e-15)) return false;
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(A[col * n + j], A[pivot * n + j]);
                std::swap(inv[col * n + j], inv[pivot * n + j]);
            }
        }
        Real diag = A[col * n + col];
        for (std::size_t j = 0; j < n; ++j) {
            A[col * n + j] /= diag;
            inv[col * n + j] /= diag;
        }
        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) continue;
            Real factor = A[row * n + col];
            for (std::size_t j = 0; j < n; ++j) {
                A[row * n + j] -= factor * A[col * n + j];
                inv[row * n + j] -= factor * inv[col * n + j];
            }
        }
    }
    A = std::move(inv);
    return true;
}

} // namespace

SpectralBasis::SpectralBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order), size_(0) {
    if (order_ < 1) {
        order_ = 1; // spectral elements require at least quadratic to expose GLL structure
    }

    if (is_line(element_type_)) {
        dimension_ = 1;
        size_ = static_cast<std::size_t>(order_ + 1);
        build_nodes();
    } else if (is_quadrilateral(element_type_)) {
        dimension_ = 2;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1));
        build_nodes();
    } else if (is_hexahedron(element_type_)) {
        dimension_ = 3;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 1) * (order_ + 1));
        build_nodes();
    } else if (is_triangle(element_type_)) {
        dimension_ = 2;
        is_simplex_ = true;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 2) / 2);
        build_simplex_nodes_triangle();
        build_inverse_vandermonde();
    } else if (is_tetrahedron(element_type_)) {
        dimension_ = 3;
        is_simplex_ = true;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 2) * (order_ + 3) / 6);
        build_simplex_nodes_tetrahedron();
        build_inverse_vandermonde();
    } else if (is_wedge(element_type_) || is_pyramid(element_type_)) {
        throw NotImplementedException("SpectralBasis currently supports only line/quad/hex/triangle/tet elements",
                                      __FILE__, __LINE__, __func__);
    } else {
        throw BasisElementCompatibilityException("SpectralBasis: unsupported element type",
                                                 __FILE__, __LINE__, __func__);
    }
}

void SpectralBasis::build_nodes() {
    auto raw = quadrature::GaussLobattoQuadrature1D::generate_raw(order_ + 1);
    nodes_1d_ = raw.first;
    barycentric_weights_.assign(nodes_1d_.size(), Real(1));
    for (std::size_t i = 0; i < nodes_1d_.size(); ++i) {
        Real prod = Real(1);
        for (std::size_t j = 0; j < nodes_1d_.size(); ++j) {
            if (i == j) continue;
            prod *= nodes_1d_[i] - nodes_1d_[j];
        }
        barycentric_weights_[i] = Real(1) / prod;
    }
}

std::vector<Real> SpectralBasis::eval_1d(Real x) const {
    std::vector<Real> vals(nodes_1d_.size(), Real(0));
    const std::size_t n = nodes_1d_.size();
    for (std::size_t i = 0; i < n; ++i) {
        Real v = Real(1);
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            v *= (x - nodes_1d_[j]) / (nodes_1d_[i] - nodes_1d_[j]);
        }
        vals[i] = v;
    }
    return vals;
}

std::vector<Real> SpectralBasis::eval_1d_derivative(Real x) const {
    std::vector<Real> ders(nodes_1d_.size(), Real(0));
    const std::size_t n = nodes_1d_.size();
    for (std::size_t i = 0; i < n; ++i) {
        Real sum = Real(0);
        for (std::size_t m = 0; m < n; ++m) {
            if (m == i) continue;
            Real prod = Real(1);
            for (std::size_t j = 0; j < n; ++j) {
                if (j == i || j == m) continue;
                prod *= (x - nodes_1d_[j]) / (nodes_1d_[i] - nodes_1d_[j]);
            }
            prod /= (nodes_1d_[i] - nodes_1d_[m]);
            sum += prod;
        }
        ders[i] = sum;
    }
    return ders;
}

// -----------------------------------------------------------------------
// Warp & Blend triangle node placement (Warburton 2006)
// Reference triangle: (0,0), (1,0), (0,1)
// -----------------------------------------------------------------------
void SpectralBasis::build_simplex_nodes_triangle() {
    const int p = order_;
    simplex_nodes_.clear();
    simplex_nodes_.reserve(size_);

    if (p == 1) {
        // Linear: just the vertices
        simplex_nodes_.push_back({Real(0), Real(0), Real(0)});
        simplex_nodes_.push_back({Real(1), Real(0), Real(0)});
        simplex_nodes_.push_back({Real(0), Real(1), Real(0)});
        return;
    }

    // 1. Generate equispaced barycentric nodes
    std::vector<Real> L1_eq, L2_eq, L3_eq;
    for (int j = 0; j <= p; ++j) {
        for (int i = 0; i <= p - j; ++i) {
            const int k = p - i - j;
            L1_eq.push_back(Real(i) / Real(p));
            L2_eq.push_back(Real(j) / Real(p));
            L3_eq.push_back(Real(k) / Real(p));
        }
    }
    const std::size_t n = L1_eq.size();
    assert(n == size_);

    // 2. Compute 1D warp displacements
    auto warp = compute_warp_1d(p);

    // Blending parameter (Warburton's alpha, empirically chosen)
    const Real alpha_tab[] = {
        Real(0),     // p=0 (unused)
        Real(0),     // p=1
        Real(1.0),   // p=2
        Real(1.0),   // p=3  -- Warburton Table 1 optimal alpha
        Real(1.0),   // p=4
        Real(1.25),  // p=5
        Real(1.25),  // p=6
        Real(1.5),   // p=7
        Real(1.5),   // p=8
        Real(1.5),   // p=9
        Real(1.5),   // p=10
    };
    const Real alpha = (p <= 10) ? alpha_tab[p] : Real(1.5);

    // 3. Apply Warp & Blend to each node
    std::vector<Real> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Map barycentric to Cartesian on reference triangle
        x[i] = L1_eq[i];
        y[i] = L2_eq[i];
    }

    // For each edge of the triangle, compute and apply warp
    // Edge 1: L3=0 (bottom), parametrized by L1-L2 in [-1,1]
    // Edge 2: L1=0 (left),   parametrized by L2-L3 in [-1,1]
    // Edge 3: L2=0 (hyp),    parametrized by L3-L1 in [-1,1]
    for (std::size_t i = 0; i < n; ++i) {
        const Real l1 = L1_eq[i], l2 = L2_eq[i], l3 = L3_eq[i];

        // Edge 1 (L3=0): warp in direction (1,-1)/sqrt(2), param = L1-L2
        {
            Real r = l1 - l2;  // in [-1, 1] on edge
            Real w = eval_warp(warp, p, r);
            // Blending: use L3 to decay away from edge
            Real blend = Real(4) * l2 * l3;
            if (std::abs(Real(1) - l1) > Real(1e-14)) {
                blend *= (Real(1) + std::pow(alpha * l1, 2)) / (Real(1) - l1);
                // Correct for boundary: blend should be 0 on other edges
                if (l1 < Real(1e-14)) blend = Real(0);
            }
            x[i] += blend * w * Real(0.5);
            y[i] -= blend * w * Real(0.5);
        }

        // Edge 2 (L1=0): warp in direction (0,1)-(0.5,0.5), param = L2-L3
        {
            Real r = l2 - l3;
            Real w = eval_warp(warp, p, r);
            Real blend = Real(4) * l1 * l3;
            if (std::abs(Real(1) - l2) > Real(1e-14)) {
                blend *= (Real(1) + std::pow(alpha * l2, 2)) / (Real(1) - l2);
                if (l2 < Real(1e-14)) blend = Real(0);
            }
            // Direction for edge 2: tangent along (−0.5, 0.5) -> shifts y up, x down
            y[i] += blend * w * Real(0.5);
            // no x shift for this edge on the reference (0,0)-(0,1) edge
        }

        // Edge 3 (L2=0): warp in direction along hypotenuse, param = L3-L1
        {
            Real r = l3 - l1;
            Real w = eval_warp(warp, p, r);
            Real blend = Real(4) * l1 * l2;
            if (std::abs(Real(1) - l3) > Real(1e-14)) {
                blend *= (Real(1) + std::pow(alpha * l3, 2)) / (Real(1) - l3);
                if (l3 < Real(1e-14)) blend = Real(0);
            }
            x[i] -= blend * w * Real(0.5);
        }

        // Clamp to valid triangle region
        x[i] = std::max(Real(0), std::min(Real(1), x[i]));
        y[i] = std::max(Real(0), std::min(Real(1) - x[i], y[i]));
    }

    for (std::size_t i = 0; i < n; ++i) {
        simplex_nodes_.push_back({x[i], y[i], Real(0)});
    }
}

// -----------------------------------------------------------------------
// Warp & Blend tetrahedron node placement
// Reference tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
// For simplicity, use equispaced nodes for p<=2 (where warp is small)
// and apply face-based warping for higher orders.
// -----------------------------------------------------------------------
void SpectralBasis::build_simplex_nodes_tetrahedron() {
    const int p = order_;
    simplex_nodes_.clear();
    simplex_nodes_.reserve(size_);

    // Generate equispaced barycentric nodes on the reference tet
    std::vector<Real> L1_eq, L2_eq, L3_eq, L4_eq;
    for (int k = 0; k <= p; ++k) {
        for (int j = 0; j <= p - k; ++j) {
            for (int i = 0; i <= p - j - k; ++i) {
                const int l = p - i - j - k;
                L1_eq.push_back(Real(i) / Real(p));
                L2_eq.push_back(Real(j) / Real(p));
                L3_eq.push_back(Real(k) / Real(p));
                L4_eq.push_back(Real(l) / Real(p));
            }
        }
    }
    const std::size_t n = L1_eq.size();
    assert(n == size_);

    if (p <= 2) {
        // For low orders, equispaced nodes are adequate
        for (std::size_t i = 0; i < n; ++i) {
            simplex_nodes_.push_back({L1_eq[i], L2_eq[i], L3_eq[i]});
        }
        return;
    }

    // For p >= 3, apply face-based Warp & Blend
    // Each face of the tet is a triangle; apply the 2D warp from each face
    // and blend into the volume.
    auto warp = compute_warp_1d(p);
    const Real alpha = Real(1.5);

    std::vector<Real> x(n), y(n), z(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = L1_eq[i];
        y[i] = L2_eq[i];
        z[i] = L3_eq[i];
    }

    // Apply 1D warp along each of the 6 edges, blended by opposite face barycentrics
    // Edges connect pairs of vertices; each edge has a parameter in [-1,1]
    // Edge (v_a, v_b): param = La - Lb, blend = product of other two L's
    struct EdgeInfo {
        int a, b;  // indices into L (0=L1, 1=L2, 2=L3, 3=L4)
    };
    const EdgeInfo edges[6] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };

    auto get_L = [&](std::size_t idx, int v) -> Real {
        switch (v) {
            case 0: return L1_eq[idx];
            case 1: return L2_eq[idx];
            case 2: return L3_eq[idx];
            case 3: return L4_eq[idx];
            default: return Real(0);
        }
    };

    // Direction vectors for each barycentric coordinate
    // L1 -> (1,0,0), L2 -> (0,1,0), L3 -> (0,0,1), L4 -> (-1,-1,-1)+offset
    // The warp shifts nodes along the edge direction e_a - e_b
    const Real dirs[4][3] = {
        { Real(1),  Real(0),  Real(0)},  // grad L1
        { Real(0),  Real(1),  Real(0)},  // grad L2
        { Real(0),  Real(0),  Real(1)},  // grad L3
        {Real(-1), Real(-1), Real(-1)},  // grad L4 (= -grad(L1+L2+L3))
    };

    for (const auto& edge : edges) {
        const int a = edge.a, b = edge.b;
        // Find the two vertices NOT on this edge
        int others[2];
        int oc = 0;
        for (int v = 0; v < 4; ++v) {
            if (v != a && v != b) others[oc++] = v;
        }

        for (std::size_t i = 0; i < n; ++i) {
            const Real La = get_L(i, a);
            const Real Lb = get_L(i, b);
            const Real Lc = get_L(i, others[0]);
            const Real Ld = get_L(i, others[1]);

            Real r = La - Lb;  // edge parameter
            Real w = eval_warp(warp, p, r);

            // Blend: product of the two barycentrics not on this edge
            Real blend = Real(4) * Lc * Ld;
            const Real La_plus_Lb = La + Lb;
            if (La_plus_Lb > Real(1e-14) && La_plus_Lb < Real(1) - Real(1e-14)) {
                blend *= (Real(1) + std::pow(alpha * (Real(1) - La_plus_Lb), 2));
            }

            // Direction: normalize edge direction
            Real dx = dirs[a][0] - dirs[b][0];
            Real dy = dirs[a][1] - dirs[b][1];
            Real dz = dirs[a][2] - dirs[b][2];
            Real len = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (len > Real(1e-14)) {
                dx /= len; dy /= len; dz /= len;
            }

            x[i] += blend * w * dx * Real(0.5);
            y[i] += blend * w * dy * Real(0.5);
            z[i] += blend * w * dz * Real(0.5);
        }
    }

    // Clamp to valid tet region
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::max(Real(0), x[i]);
        y[i] = std::max(Real(0), y[i]);
        z[i] = std::max(Real(0), z[i]);
        Real sum = x[i] + y[i] + z[i];
        if (sum > Real(1)) {
            x[i] /= sum; y[i] /= sum; z[i] /= sum;
        }
        simplex_nodes_.push_back({x[i], y[i], z[i]});
    }
}

// -----------------------------------------------------------------------
// Build inverse Vandermonde for simplex evaluation
// V_{ij} = phi_j(node_i), where phi_j is the j-th modal basis function
// (Dubiner for 2D, Proriol for 3D)
// -----------------------------------------------------------------------
void SpectralBasis::build_inverse_vandermonde() {
    const std::size_t n = size_;
    inv_vandermonde_.resize(n * n);

    if (dimension_ == 2) {
        // Enumerate modal indices (p,q) with p+q <= order_
        struct PQ { int p, q; };
        std::vector<PQ> modes;
        for (int total = 0; total <= order_; ++total) {
            for (int p = 0; p <= total; ++p) {
                modes.push_back({p, total - p});
            }
        }
        assert(modes.size() == n);

        // Build Vandermonde: V(i, j) = dubiner(modes[j].p, modes[j].q, xi_i, eta_i)
        std::vector<Real> V(n * n);
        for (std::size_t i = 0; i < n; ++i) {
            const Real xi = simplex_nodes_[i][0];
            const Real eta = simplex_nodes_[i][1];
            for (std::size_t j = 0; j < n; ++j) {
                V[i * n + j] = orthopoly::dubiner(modes[j].p, modes[j].q, xi, eta);
            }
        }

        // Invert
        if (!invert_matrix(V, n)) {
            throw BasisConstructionException("SpectralBasis: Vandermonde matrix is singular for triangle",
                                             __FILE__, __LINE__, __func__);
        }
        inv_vandermonde_ = std::move(V);

    } else {
        // 3D: Proriol
        struct PQR { int p, q, r; };
        std::vector<PQR> modes;
        for (int total = 0; total <= order_; ++total) {
            for (int p = 0; p <= total; ++p) {
                for (int q = 0; q <= total - p; ++q) {
                    modes.push_back({p, q, total - p - q});
                }
            }
        }
        assert(modes.size() == n);

        std::vector<Real> V(n * n);
        for (std::size_t i = 0; i < n; ++i) {
            const Real xi = simplex_nodes_[i][0];
            const Real eta = simplex_nodes_[i][1];
            const Real zeta = simplex_nodes_[i][2];
            for (std::size_t j = 0; j < n; ++j) {
                V[i * n + j] = orthopoly::proriol(modes[j].p, modes[j].q, modes[j].r,
                                                   xi, eta, zeta);
            }
        }

        if (!invert_matrix(V, n)) {
            throw BasisConstructionException("SpectralBasis: Vandermonde matrix is singular for tetrahedron",
                                             __FILE__, __LINE__, __func__);
        }
        inv_vandermonde_ = std::move(V);
    }
}

void SpectralBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                    std::vector<Real>& values) const {
    if (!is_simplex_) {
        // Original tensor-product path
        values.assign(size_, Real(0));

        if (dimension_ == 1) {
            values = eval_1d(xi[0]);
            return;
        }

        if (dimension_ == 2) {
            auto lx = eval_1d(xi[0]);
            auto ly = eval_1d(xi[1]);
            std::size_t idx = 0;
            for (std::size_t j = 0; j < ly.size(); ++j) {
                for (std::size_t i = 0; i < lx.size(); ++i) {
                    values[idx++] = lx[i] * ly[j];
                }
            }
            return;
        }

        auto lx = eval_1d(xi[0]);
        auto ly = eval_1d(xi[1]);
        auto lz = eval_1d(xi[2]);
        std::size_t idx = 0;
        for (std::size_t k = 0; k < lz.size(); ++k) {
            for (std::size_t j = 0; j < ly.size(); ++j) {
                for (std::size_t i = 0; i < lx.size(); ++i) {
                    values[idx++] = lx[i] * ly[j] * lz[k];
                }
            }
        }
        return;
    }

    // Simplex path: N_i(xi) = sum_j V^{-1}_{ji} * phi_j(xi)
    // where phi_j is the modal basis (Dubiner/Proriol)
    values.assign(size_, Real(0));
    const std::size_t n = size_;

    // Evaluate all modal basis functions at xi
    std::vector<Real> modal(n);
    if (dimension_ == 2) {
        std::size_t idx = 0;
        for (int total = 0; total <= order_; ++total) {
            for (int p = 0; p <= total; ++p) {
                int q = total - p;
                modal[idx++] = orthopoly::dubiner(p, q, xi[0], xi[1]);
            }
        }
    } else {
        std::size_t idx = 0;
        for (int total = 0; total <= order_; ++total) {
            for (int p = 0; p <= total; ++p) {
                for (int q = 0; q <= total - p; ++q) {
                    int r = total - p - q;
                    modal[idx++] = orthopoly::proriol(p, q, r, xi[0], xi[1], xi[2]);
                }
            }
        }
    }

    // N_i = sum_j inv_V[j][i] * modal[j]  (inv_V is stored row-major as inv_V[row * n + col])
    // inv_V[j * n + i] = j-th row, i-th column
    for (std::size_t i = 0; i < n; ++i) {
        Real sum = Real(0);
        for (std::size_t j = 0; j < n; ++j) {
            sum += inv_vandermonde_[j * n + i] * modal[j];
        }
        values[i] = sum;
    }
}

void SpectralBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                       std::vector<Gradient>& gradients) const {
    if (!is_simplex_) {
        // Original tensor-product path
        gradients.assign(size_, Gradient{});

        if (dimension_ == 1) {
            auto d = eval_1d_derivative(xi[0]);
            for (std::size_t i = 0; i < d.size(); ++i) {
                gradients[i][0] = d[i];
            }
            return;
        }

        if (dimension_ == 2) {
            auto lx = eval_1d(xi[0]);
            auto ly = eval_1d(xi[1]);
            auto dx = eval_1d_derivative(xi[0]);
            auto dy = eval_1d_derivative(xi[1]);
            std::size_t idx = 0;
            for (std::size_t j = 0; j < ly.size(); ++j) {
                for (std::size_t i = 0; i < lx.size(); ++i) {
                    gradients[idx][0] = dx[i] * ly[j];
                    gradients[idx][1] = lx[i] * dy[j];
                    ++idx;
                }
            }
            return;
        }

        auto lx = eval_1d(xi[0]);
        auto ly = eval_1d(xi[1]);
        auto lz = eval_1d(xi[2]);
        auto dx = eval_1d_derivative(xi[0]);
        auto dy = eval_1d_derivative(xi[1]);
        auto dz = eval_1d_derivative(xi[2]);

        std::size_t idx = 0;
        for (std::size_t k = 0; k < lz.size(); ++k) {
            for (std::size_t j = 0; j < ly.size(); ++j) {
                for (std::size_t i = 0; i < lx.size(); ++i) {
                    gradients[idx][0] = dx[i] * ly[j] * lz[k];
                    gradients[idx][1] = lx[i] * dy[j] * lz[k];
                    gradients[idx][2] = lx[i] * ly[j] * dz[k];
                    ++idx;
                }
            }
        }
        return;
    }

    // Simplex path: dN_i/dxi_d = sum_j V^{-1}_{ji} * dphi_j/dxi_d
    gradients.assign(size_, Gradient{});
    const std::size_t n = size_;

    // Evaluate modal gradients
    std::vector<Gradient> modal_grads(n);
    if (dimension_ == 2) {
        std::size_t idx = 0;
        for (int total = 0; total <= order_; ++total) {
            for (int p = 0; p <= total; ++p) {
                int q = total - p;
                auto [val, dxi, deta] = orthopoly::dubiner_with_derivatives(p, q, xi[0], xi[1]);
                (void)val;
                modal_grads[idx][0] = dxi;
                modal_grads[idx][1] = deta;
                ++idx;
            }
        }
    } else {
        std::size_t idx = 0;
        for (int total = 0; total <= order_; ++total) {
            for (int p = 0; p <= total; ++p) {
                for (int q = 0; q <= total - p; ++q) {
                    int r = total - p - q;
                    auto [val, dxi, deta, dzeta] =
                        orthopoly::proriol_with_derivatives(p, q, r, xi[0], xi[1], xi[2]);
                    (void)val;
                    modal_grads[idx][0] = dxi;
                    modal_grads[idx][1] = deta;
                    modal_grads[idx][2] = dzeta;
                    ++idx;
                }
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        Gradient g{};
        for (std::size_t j = 0; j < n; ++j) {
            const Real coeff = inv_vandermonde_[j * n + i];
            for (int d = 0; d < dimension_; ++d) {
                g[static_cast<std::size_t>(d)] += coeff * modal_grads[j][static_cast<std::size_t>(d)];
            }
        }
        gradients[i] = g;
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
