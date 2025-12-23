/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "LagrangeBasis.h"
#include "OrthogonalPolynomials.h"
#include "NodeOrderingConventions.h"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace svmp {
namespace FE {
namespace basis {

namespace {

bool is_line(ElementType type) {
    return type == ElementType::Line2 || type == ElementType::Line3;
}

bool is_triangle(ElementType type) {
    return type == ElementType::Triangle3 || type == ElementType::Triangle6;
}

bool is_quadrilateral(ElementType type) {
    return type == ElementType::Quad4 || type == ElementType::Quad8 || type == ElementType::Quad9;
}

bool is_tetrahedron(ElementType type) {
    return type == ElementType::Tetra4 || type == ElementType::Tetra10;
}

bool is_hexahedron(ElementType type) {
    return type == ElementType::Hex8 || type == ElementType::Hex20 || type == ElementType::Hex27;
}

bool is_wedge(ElementType type) {
    return type == ElementType::Wedge6 || type == ElementType::Wedge15 || type == ElementType::Wedge18;
}

bool is_pyramid(ElementType type) {
    return type == ElementType::Pyramid5 || type == ElementType::Pyramid13 || type == ElementType::Pyramid14;
}

ElementType node_ordering_equivalent(ElementType canonical_type, int order) {
    // Map (canonical topology, polynomial order) to a VTK-style element type
    // for which NodeOrderingConventions provides an explicit ordering.
    switch (canonical_type) {
        case ElementType::Line2:
            if (order == 1) return ElementType::Line2;
            if (order == 2) return ElementType::Line3;
            return ElementType::Unknown;
        case ElementType::Triangle3:
            if (order == 1) return ElementType::Triangle3;
            if (order == 2) return ElementType::Triangle6;
            return ElementType::Unknown;
        case ElementType::Quad4:
            if (order == 1) return ElementType::Quad4;
            if (order == 2) return ElementType::Quad9;
            return ElementType::Unknown;
        case ElementType::Tetra4:
            if (order == 1) return ElementType::Tetra4;
            if (order == 2) return ElementType::Tetra10;
            return ElementType::Unknown;
        case ElementType::Hex8:
            if (order == 1) return ElementType::Hex8;
            if (order == 2) return ElementType::Hex27;
            return ElementType::Unknown;
        case ElementType::Wedge6:
            if (order == 1) return ElementType::Wedge6;
            if (order == 2) return ElementType::Wedge18;
            return ElementType::Unknown;
        case ElementType::Pyramid5:
            if (order == 1) return ElementType::Pyramid5;
            // Quadratic polynomial pyramids are not supported by this basis;
            // Pyramid14 is handled separately.
            return ElementType::Unknown;
        case ElementType::Pyramid14:
            if (order == 2) return ElementType::Pyramid14;
            return ElementType::Unknown;
        default:
            return ElementType::Unknown;
    }
}

inline bool coords_close(const math::Vector<Real, 3>& a,
                         const math::Vector<Real, 3>& b,
                         Real tol = Real(1e-12)) {
    return (std::abs(a[0] - b[0]) <= tol) &&
           (std::abs(a[1] - b[1]) <= tol) &&
           (std::abs(a[2] - b[2]) <= tol);
}

// Falling-factorial (equispaced barycentric) Lagrange factors for simplex nodes.
//
// For a fixed polynomial order p and barycentric coordinate λ ∈ [0,1], define
//   φ_a(λ) = Π_{m=0}^{a-1} (p λ - m) / (a - m),   a = 0..p
// Then for a multi-index (i0,i1,...,id) with Σ i_k = p, the simplex Lagrange
// basis function is Π_k φ_{i_k}(λ_k), nodal on the barycentric grid.
inline void simplex_lagrange_factor_sequence(int p,
                                             Real lambda,
                                             std::vector<Real>& phi,
                                             std::vector<Real>& dphi,
                                             std::vector<Real>& d2phi) {
    const std::size_t n = static_cast<std::size_t>(p + 1);
    phi.assign(n, Real(0));
    dphi.assign(n, Real(0));
    d2phi.assign(n, Real(0));

    phi[0] = Real(1);
    if (p == 0) {
        return;
    }

    const Real t = static_cast<Real>(p) * lambda;
    const Real dt_dlambda = static_cast<Real>(p);

    std::vector<Real> dphi_dt(n, Real(0));
    std::vector<Real> d2phi_dt2(n, Real(0));
    dphi_dt[0] = Real(0);
    d2phi_dt2[0] = Real(0);

    for (int a = 1; a <= p; ++a) {
        const std::size_t au = static_cast<std::size_t>(a);
        const Real inv_a = Real(1) / static_cast<Real>(a);
        const Real s = (t - static_cast<Real>(a - 1)) * inv_a;

        phi[au] = s * phi[au - 1];

        // Derivatives w.r.t. t
        dphi_dt[au] = inv_a * phi[au - 1] + s * dphi_dt[au - 1];
        d2phi_dt2[au] = Real(2) * inv_a * dphi_dt[au - 1] + s * d2phi_dt2[au - 1];

        // Convert to derivatives w.r.t. lambda (t = p*lambda)
        dphi[au] = dt_dlambda * dphi_dt[au];
        d2phi[au] = dt_dlambda * dt_dlambda * d2phi_dt2[au];
    }
}

// -------------------------------------------------------------------------
// Pyramid14 rational modal basis and Vandermonde transform
// -------------------------------------------------------------------------

struct Pyramid14RationalData {
    static constexpr int dof = 14;

    struct Mode { int i; int j; int k; };

    // k = 0..2, i,j = 0..k
    static const std::array<Mode, dof>& modes() {
        static const std::array<Mode, dof> m = {{
            // k = 0 (1 mode)
            {0, 0, 0},
            // k = 1 (4 modes)
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
            // k = 2 (9 modes)
            {0, 0, 2}, {1, 0, 2}, {2, 0, 2},
            {0, 1, 2}, {1, 1, 2}, {2, 1, 2},
            {0, 2, 2}, {1, 2, 2}, {2, 2, 2}
        }};
        return m;
    }

    using Matrix14 = std::array<std::array<Real, dof>, dof>;

    Matrix14 modal_to_nodal{};
    bool initialized{false};

    static Pyramid14RationalData& instance() {
        static Pyramid14RationalData data;
        return data;
    }

    static void rst_to_abc(const math::Vector<Real, 3>& xi,
                           Real& a, Real& b, Real& c) {
        const Real r = xi[0];
        const Real s = xi[1];
        const Real z = xi[2];

        // Collapse z in [0,1] to c in [-1,1]
        c = Real(2) * z - Real(1);
        const Real one_minus_c = Real(1) - c;
        const Real eps = Real(1e-12);
        if (std::abs(one_minus_c) < eps) {
            a = Real(0);
            b = Real(0);
        } else {
            const Real inv = Real(2) / one_minus_c;
            a = inv * r;
            b = inv * s;
        }
    }

    static Real gl_node(int k, int i) {
        // Gauss-Legendre nodes on [-1,1] for k+1 points, k in {0,1,2}
        if (k == 0) {
            return Real(0);
        }
        if (k == 1) {
            const Real s = std::sqrt(Real(1) / Real(3));
            return (i == 0) ? -s : s;
        }
        if (k == 2) {
            const Real s = std::sqrt(Real(3) / Real(5));
            if (i == 0) return -s;
            if (i == 1) return Real(0);
            return s;
        }
        // Should never be reached for Pyramid14 (N=2)
        return Real(0);
    }

    static Real lagrange_1d_gl(int k, int i, Real x) {
        // Lagrange basis at Gauss-Legendre nodes of order k (k+1 points)
        if (k == 0) {
            return Real(1);
        }
        Real li = Real(1);
        for (int j = 0; j <= k; ++j) {
            if (j == i) continue;
            const Real xj = gl_node(k, j);
            const Real xi = gl_node(k, i);
            li *= (x - xj) / (xi - xj);
        }
        return li;
    }

    static void eval_modal(const math::Vector<Real, 3>& xi,
                           std::array<Real, dof>& psi) {
        Real a{}, b{}, c{};
        rst_to_abc(xi, a, b, c);

        constexpr int N = 2;
        const auto& m = modes();

        for (int idx = 0; idx < dof; ++idx) {
            const int i = m[static_cast<std::size_t>(idx)].i;
            const int j = m[static_cast<std::size_t>(idx)].j;
            const int k = m[static_cast<std::size_t>(idx)].k;

            const Real La = lagrange_1d_gl(k, i, a);
            const Real Lb = lagrange_1d_gl(k, j, b);

            const Real tmp = (Real(1) - c) * Real(0.5);
            Real pow_factor = Real(1);
            if (k == 1) {
                pow_factor = tmp;
            } else if (k == 2) {
                pow_factor = tmp * tmp;
            }

            const int n_jac = N - k;
            const Real alpha = static_cast<Real>(2 * k + 3);
            const Real beta = Real(0);
            const Real Jc = orthopoly::jacobi(n_jac, alpha, beta, c);

            psi[static_cast<std::size_t>(idx)] = La * Lb * pow_factor * Jc;
        }
    }

    void ensure_initialized() {
        if (initialized) {
            return;
        }

        Matrix14 V{};
        Matrix14 inv{};

        // Initialize inv as identity
        for (int i = 0; i < dof; ++i) {
            for (int j = 0; j < dof; ++j) {
                inv[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    (i == j) ? Real(1) : Real(0);
            }
        }

        // Build Vandermonde matrix: V(n, m) = psi_m(x_n)
        for (int n = 0; n < dof; ++n) {
            const auto xi =
                NodeOrdering::get_node_coords(ElementType::Pyramid14,
                                              static_cast<std::size_t>(n));
            std::array<Real, dof> psi{};
            eval_modal(xi, psi);
            for (int m = 0; m < dof; ++m) {
                V[static_cast<std::size_t>(n)][static_cast<std::size_t>(m)] =
                    psi[static_cast<std::size_t>(m)];
            }
        }

        // Gauss-Jordan elimination with partial pivoting
        for (int col = 0; col < dof; ++col) {
            int pivot = col;
            Real max_abs = std::abs(
                V[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)]);
            for (int row = col + 1; row < dof; ++row) {
                const Real val = std::abs(
                    V[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
                if (val > max_abs) {
                    max_abs = val;
                    pivot = row;
                }
            }

            if (max_abs == Real(0)) {
                throw FEException(
                    "Pyramid14 rational basis: singular Vandermonde matrix",
                    __FILE__, __LINE__, __func__, FEStatus::SingularMapping);
            }

            if (pivot != col) {
                std::swap(V[static_cast<std::size_t>(pivot)],
                          V[static_cast<std::size_t>(col)]);
                std::swap(inv[static_cast<std::size_t>(pivot)],
                          inv[static_cast<std::size_t>(col)]);
            }

            const Real diag =
                V[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
            const Real inv_diag = Real(1) / diag;

            for (int j = 0; j < dof; ++j) {
                V[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)] *=
                    inv_diag;
                inv[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)] *=
                    inv_diag;
            }

            for (int row = 0; row < dof; ++row) {
                if (row == col) continue;
                const Real factor =
                    V[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
                if (factor == Real(0)) continue;
                for (int j = 0; j < dof; ++j) {
                    V[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] -=
                        factor *
                        V[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
                    inv[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] -=
                        factor *
                        inv[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
                }
            }
        }

        // modal_to_nodal = (V^{-1})^T
        for (int i = 0; i < dof; ++i) {
            for (int m = 0; m < dof; ++m) {
                modal_to_nodal[static_cast<std::size_t>(i)]
                              [static_cast<std::size_t>(m)] =
                    inv[static_cast<std::size_t>(m)]
                       [static_cast<std::size_t>(i)];
            }
        }

        initialized = true;
    }
};

} // namespace

LagrangeBasis::LagrangeBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order) {
    // Normalize element-specific orders / unsupported types
    switch (element_type_) {
        case ElementType::Line3:
            order_ = std::max(order_, 2);
            element_type_ = ElementType::Line2;
            break;
        case ElementType::Triangle6:
            order_ = std::max(order_, 2);
            element_type_ = ElementType::Triangle3;
            break;
        case ElementType::Quad9:
            order_ = std::max(order_, 2);
            element_type_ = ElementType::Quad4;
            break;
        case ElementType::Quad8:
            throw FEException("Quad8 serendipity Lagrange basis not implemented",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        case ElementType::Tetra10:
            order_ = std::max(order_, 2);
            element_type_ = ElementType::Tetra4;
            break;
        case ElementType::Hex27:
            order_ = std::max(order_, 2);
            element_type_ = ElementType::Hex8;
            break;
        case ElementType::Hex20:
            throw FEException("Hex20 serendipity Lagrange basis not implemented",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        case ElementType::Wedge18:
            order_ = std::max(order_, 2);
            element_type_ = ElementType::Wedge6;
            break;
        case ElementType::Wedge15:
            throw FEException("Wedge15 serendipity Lagrange basis not implemented",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        case ElementType::Pyramid5:
            if (order_ != 1) {
                throw FEException(
                    "Pyramid5 Lagrange basis is implemented only for linear order 1",
                    __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            order_ = 1;
            break;
        case ElementType::Pyramid13:
            throw FEException(
                "Pyramid13 Lagrange basis is not implemented; use SerendipityBasis (Pyramid13) or LagrangeBasis (Pyramid14) where appropriate",
                __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        case ElementType::Pyramid14:
            if (order_ != 2) {
                throw FEException(
                    "Pyramid14 Lagrange basis implemented only for quadratic order 2",
                    __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            order_ = 2;
            break;
        default:
            break;
    }
    if (order_ < 0) {
        throw FEException("LagrangeBasis requires non-negative polynomial order",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    if (is_line(element_type_)) {
        dimension_ = 1;
    } else if (is_triangle(element_type_) || is_quadrilateral(element_type_)) {
        dimension_ = 2;
    } else if (is_tetrahedron(element_type_) || is_hexahedron(element_type_) || is_wedge(element_type_) || is_pyramid(element_type_)) {
        dimension_ = 3;
    } else if (element_type_ == ElementType::Point1) {
        dimension_ = 0;
    } else {
        throw FEException("Unsupported element type for LagrangeBasis",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    init_nodes();
}

void LagrangeBasis::init_nodes() {
    nodes_.clear();
    nodes_1d_.clear();
    denominators_1d_.clear();
    simplex_exponents_.clear();
    simplex_coefficients_.clear();
    external_to_internal_.clear();

    auto finalize_node_ordering = [&]() {
        external_to_internal_.clear();
        const ElementType ordering_type = node_ordering_equivalent(element_type_, order_);
        if (ordering_type == ElementType::Unknown || nodes_.empty()) {
            return;
        }

        const std::size_t n = nodes_.size();
        if (NodeOrdering::num_nodes(ordering_type) != n) {
            return;
        }

        const auto internal_nodes = nodes_;
        std::vector<math::Vector<Real, 3>> ordered_nodes;
        ordered_nodes.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            ordered_nodes.push_back(NodeOrdering::get_node_coords(ordering_type, i));
        }

        std::vector<std::size_t> perm(n, std::size_t(0));
        std::vector<bool> used(n, false);
        for (std::size_t ext = 0; ext < n; ++ext) {
            const auto& target = ordered_nodes[ext];
            bool found = false;
            for (std::size_t in = 0; in < n; ++in) {
                if (used[in]) {
                    continue;
                }
                if (coords_close(internal_nodes[in], target)) {
                    perm[ext] = in;
                    used[in] = true;
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw FEException("LagrangeBasis: failed to match internal nodes to NodeOrderingConventions",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
        }

        bool identity = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (perm[i] != i) {
                identity = false;
                break;
            }
        }

        nodes_ = std::move(ordered_nodes);
        if (!identity) {
            external_to_internal_ = std::move(perm);
        }
    };

    if (element_type_ == ElementType::Point1) {
        nodes_.push_back(math::Vector<Real, 3>{Real(0), Real(0), Real(0)});
        nodes_1d_.push_back(Real(0));
        denominators_1d_.push_back(Real(1));
        finalize_node_ordering();
        return;
    }

    if (order_ == 0 && !is_triangle(element_type_) && !is_tetrahedron(element_type_) && !is_wedge(element_type_)) {
        nodes_1d_.push_back(Real(0));
        nodes_.push_back(math::Vector<Real, 3>{Real(0), Real(0), Real(0)});
        denominators_1d_.push_back(Real(1));
        finalize_node_ordering();
        return;
    }

    const int n1d = order_ + 1;

    if (is_line(element_type_) || is_quadrilateral(element_type_) || is_hexahedron(element_type_) || is_wedge(element_type_)) {
        if (order_ == 0) {
            nodes_1d_.push_back(Real(0));
        } else {
            const Real step = Real(2) / static_cast<Real>(order_);
            for (int i = 0; i < n1d; ++i) {
                nodes_1d_.push_back(Real(-1) + step * static_cast<Real>(i));
            }
        }

        denominators_1d_.resize(nodes_1d_.size(), Real(1));
        for (std::size_t i = 0; i < nodes_1d_.size(); ++i) {
            Real denom = Real(1);
            for (std::size_t j = 0; j < nodes_1d_.size(); ++j) {
                if (i == j) continue;
                denom *= nodes_1d_[i] - nodes_1d_[j];
            }
            denominators_1d_[i] = denom;
        }

        if (is_line(element_type_)) {
            for (Real x : nodes_1d_) {
                nodes_.push_back(math::Vector<Real, 3>{x, Real(0), Real(0)});
            }
        } else if (is_quadrilateral(element_type_)) {
            for (Real y : nodes_1d_) {
                for (Real x : nodes_1d_) {
                    nodes_.push_back(math::Vector<Real, 3>{x, y, Real(0)});
                }
            }
        } else if (is_hexahedron(element_type_)) {
            for (Real z : nodes_1d_) {
                for (Real y : nodes_1d_) {
                    for (Real x : nodes_1d_) {
                        nodes_.push_back(math::Vector<Real, 3>{x, y, z});
                    }
                }
            }
        } else if (is_wedge(element_type_)) {
            // Triangle in (x,y) with barycentric grid, tensor with 1D line.
            // Reuse triangle simplex tables for the wedge base.
            ElementType original = element_type_;
            element_type_ = ElementType::Triangle3;
            init_simplex_table();
            element_type_ = original;

            if (order_ == 0) {
                // Constant wedge basis: one node at triangle barycenter and mid-plane z=0.
                nodes_.clear();
                nodes_.push_back(math::Vector<Real, 3>{Real(1) / Real(3), Real(1) / Real(3), nodes_1d_.front()});
            } else {
                const std::size_t tri_count = simplex_exponents_.size();
                const std::size_t line_count = nodes_1d_.size();
                nodes_.clear();
                nodes_.reserve(tri_count * line_count);
                for (Real z : nodes_1d_) {
                    for (const auto& e : simplex_exponents_) {
                        const Real l1 = static_cast<Real>(e[1]) / static_cast<Real>(order_);
                        const Real l2 = static_cast<Real>(e[2]) / static_cast<Real>(order_);
                        nodes_.push_back(math::Vector<Real, 3>{l1, l2, z});
                    }
                }
            }
        }
        finalize_node_ordering();
        return;
    }

    if (is_triangle(element_type_) || is_tetrahedron(element_type_)) {
        init_simplex_table();
        finalize_node_ordering();
        return;
    }

    if (element_type_ == ElementType::Pyramid5) {
        // Linear pyramid (Pyramid5) support only. Reference pyramid:
        // base z=0 square (-1,-1,0) .. (1,1,0); apex at (0,0,1).
        if (order_ != 1) {
            throw FEException(
                "Pyramid5 Lagrange basis is implemented only for linear order 1",
                __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        }
        nodes_.push_back(math::Vector<Real,3>{Real(-1), Real(-1), Real(0)});
        nodes_.push_back(math::Vector<Real,3>{Real(1),  Real(-1), Real(0)});
        nodes_.push_back(math::Vector<Real,3>{Real(1),  Real(1),  Real(0)});
        nodes_.push_back(math::Vector<Real,3>{Real(-1), Real(1),  Real(0)});
        nodes_.push_back(math::Vector<Real,3>{Real(0),  Real(0),  Real(1)});
        finalize_node_ordering();
        return;
    }

    if (element_type_ == ElementType::Pyramid14) {
        const std::size_t nn =
            NodeOrdering::num_nodes(ElementType::Pyramid14);
        nodes_.reserve(nn);
        for (std::size_t i = 0; i < nn; ++i) {
            nodes_.push_back(
                NodeOrdering::get_node_coords(ElementType::Pyramid14, i));
        }
        finalize_node_ordering();
        return;
    }

    if (is_pyramid(element_type_)) {
        throw FEException("Pyramid higher-order basis not implemented",
                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
    }
}

void LagrangeBasis::init_simplex_table() {
    if (order_ == 0) {
        simplex_exponents_.push_back({0, 0, 0, 0});
        if (is_triangle(element_type_)) {
            nodes_.push_back(math::Vector<Real, 3>{Real(1) / Real(3),
                                                   Real(1) / Real(3),
                                                   Real(0)});
        } else {
            nodes_.push_back(math::Vector<Real, 3>{Real(0.25),
                                                   Real(0.25),
                                                   Real(0.25)});
        }
        return;
    }

    if (is_triangle(element_type_)) {
        for (int i = 0; i <= order_; ++i) {
            for (int j = 0; j <= order_ - i; ++j) {
                int k = order_ - i - j;
                simplex_exponents_.push_back({i, j, k, 0});

                Real x = static_cast<Real>(j) / static_cast<Real>(order_);
                Real y = static_cast<Real>(k) / static_cast<Real>(order_);
                nodes_.push_back(math::Vector<Real, 3>{x, y, Real(0)});
            }
        }
    } else if (is_tetrahedron(element_type_)) {
        for (int i = 0; i <= order_; ++i) {
            for (int j = 0; j <= order_ - i; ++j) {
                for (int k = 0; k <= order_ - i - j; ++k) {
                    int l = order_ - i - j - k;
                    simplex_exponents_.push_back({i, j, k, l});

                    Real x = static_cast<Real>(j) / static_cast<Real>(order_);
                    Real y = static_cast<Real>(k) / static_cast<Real>(order_);
                    Real z = static_cast<Real>(l) / static_cast<Real>(order_);
                    nodes_.push_back(math::Vector<Real, 3>{x, y, z});
                }
            }
        }
    }
}

std::vector<Real> LagrangeBasis::evaluate_1d(Real xi) const {
    std::vector<Real> vals(nodes_1d_.size(), Real(0));
    if (nodes_1d_.size() == 1) {
        vals[0] = Real(1);
        return vals;
    }

    for (std::size_t i = 0; i < nodes_1d_.size(); ++i) {
        Real v = Real(1);
        for (std::size_t j = 0; j < nodes_1d_.size(); ++j) {
            if (i == j) continue;
            v *= (xi - nodes_1d_[j]) / (nodes_1d_[i] - nodes_1d_[j]);
        }
        vals[i] = v;
    }
    return vals;
}

std::vector<Real> LagrangeBasis::evaluate_1d_derivative(Real xi) const {
    std::vector<Real> ders(nodes_1d_.size(), Real(0));
    if (nodes_1d_.size() == 1) {
        return ders;
    }

    for (std::size_t i = 0; i < nodes_1d_.size(); ++i) {
        Real sum = Real(0);
        for (std::size_t m = 0; m < nodes_1d_.size(); ++m) {
            if (m == i) continue;
            Real prod = Real(1);
            for (std::size_t j = 0; j < nodes_1d_.size(); ++j) {
                if (j == i || j == m) continue;
                prod *= (xi - nodes_1d_[j]) / (nodes_1d_[i] - nodes_1d_[j]);
            }
            prod /= (nodes_1d_[i] - nodes_1d_[m]);
            sum += prod;
        }
        ders[i] = sum;
    }
    return ders;
}

std::vector<Real> LagrangeBasis::evaluate_1d_second_derivative(Real xi) const {
    std::vector<Real> d2(nodes_1d_.size(), Real(0));
    if (nodes_1d_.size() <= 2) {
        // Linear or constant: second derivative is zero
        return d2;
    }

    // L_i''(x) = sum_{m1 != i} sum_{m2 != i, m2 != m1}
    //            [ prod_{j != i, j != m1, j != m2} (x - x_j) / (x_i - x_j) ]
    //            / ((x_i - x_m1) * (x_i - x_m2))
    for (std::size_t i = 0; i < nodes_1d_.size(); ++i) {
        Real sum = Real(0);
        for (std::size_t m1 = 0; m1 < nodes_1d_.size(); ++m1) {
            if (m1 == i) continue;
            for (std::size_t m2 = m1 + 1; m2 < nodes_1d_.size(); ++m2) {
                if (m2 == i) continue;
                Real prod = Real(1);
                for (std::size_t j = 0; j < nodes_1d_.size(); ++j) {
                    if (j == i || j == m1 || j == m2) continue;
                    prod *= (xi - nodes_1d_[j]) / (nodes_1d_[i] - nodes_1d_[j]);
                }
                prod /= (nodes_1d_[i] - nodes_1d_[m1]);
                prod /= (nodes_1d_[i] - nodes_1d_[m2]);
                sum += prod;
            }
        }
        // Factor of 2 because we only summed over m1 < m2
        d2[i] = Real(2) * sum;
    }
    return d2;
}

void LagrangeBasis::evaluate_values(const math::Vector<Real, 3>& xi,
	                                    std::vector<Real>& values) const {
	    values.assign(size(), Real(0));

        auto apply_node_ordering = [&]() {
            if (external_to_internal_.empty()) {
                return;
            }
            std::vector<Real> tmp = values;
            values.assign(external_to_internal_.size(), Real(0));
            for (std::size_t ext = 0; ext < external_to_internal_.size(); ++ext) {
                values[ext] = tmp[external_to_internal_[ext]];
            }
        };

	    if (element_type_ == ElementType::Point1) {
	        values[0] = Real(1);
            apply_node_ordering();
	        return;
	    }

    if (element_type_ == ElementType::Pyramid14) {
        Pyramid14RationalData& data = Pyramid14RationalData::instance();
        data.ensure_initialized();

        std::array<Real, Pyramid14RationalData::dof> psi{};
        Pyramid14RationalData::eval_modal(xi, psi);

        const std::size_t n = size();
	        for (std::size_t i = 0; i < n; ++i) {
	            Real sum = Real(0);
	            for (int m = 0; m < Pyramid14RationalData::dof; ++m) {
	                sum += data.modal_to_nodal[i][static_cast<std::size_t>(m)] *
	                       psi[static_cast<std::size_t>(m)];
	            }
	            values[i] = sum;
	        }
            apply_node_ordering();
	        return;
	    }

	    if (is_line(element_type_)) {
	        values = evaluate_1d(xi[0]);
            apply_node_ordering();
	        return;
	    }

    if (is_quadrilateral(element_type_)) {
        auto lx = evaluate_1d(xi[0]);
        auto ly = evaluate_1d(xi[1]);
        const std::size_t n1d = nodes_1d_.size();
        std::size_t idx = 0;
	        for (std::size_t j = 0; j < n1d; ++j) {
	            for (std::size_t i = 0; i < n1d; ++i) {
	                values[idx++] = lx[i] * ly[j];
	            }
	        }
            apply_node_ordering();
	        return;
	    }

    if (is_hexahedron(element_type_)) {
        auto lx = evaluate_1d(xi[0]);
        auto ly = evaluate_1d(xi[1]);
        auto lz = evaluate_1d(xi[2]);
        const std::size_t n1d = nodes_1d_.size();
        std::size_t idx = 0;
	        for (std::size_t k = 0; k < n1d; ++k) {
	            for (std::size_t j = 0; j < n1d; ++j) {
	                for (std::size_t i = 0; i < n1d; ++i) {
	                    values[idx++] = lx[i] * ly[j] * lz[k];
	                }
	            }
	        }
            apply_node_ordering();
	        return;
	    }

	    if (is_wedge(element_type_)) {
	        auto lz = evaluate_1d(xi[2]);
	        // triangle part
            const int p = order_;
	        const Real l1 = xi[0];
	        const Real l2 = xi[1];
	        const Real l0 = Real(1) - l1 - l2;

            std::vector<Real> phi0, dphi0, d2phi0;
            std::vector<Real> phi1, dphi1, d2phi1;
            std::vector<Real> phi2, dphi2, d2phi2;
            simplex_lagrange_factor_sequence(p, l0, phi0, dphi0, d2phi0);
            simplex_lagrange_factor_sequence(p, l1, phi1, dphi1, d2phi1);
            simplex_lagrange_factor_sequence(p, l2, phi2, dphi2, d2phi2);

            std::vector<Real> tri_vals(simplex_exponents_.size(), Real(0));
            for (std::size_t n = 0; n < simplex_exponents_.size(); ++n) {
                const auto& e = simplex_exponents_[n];
                tri_vals[n] =
                    phi0[static_cast<std::size_t>(e[0])] *
                    phi1[static_cast<std::size_t>(e[1])] *
                    phi2[static_cast<std::size_t>(e[2])];
            }

	        std::size_t idx = 0;
	        for (std::size_t kz = 0; kz < lz.size(); ++kz) {
	            for (std::size_t n = 0; n < tri_vals.size(); ++n) {
	                values[idx++] = tri_vals[n] * lz[kz];
	            }
	        }
            apply_node_ordering();
	        return;
	    }

    if (element_type_ == ElementType::Pyramid5) {
        // Linear pyramid (Pyramid5) with canonical identity mapping:
        // reference coordinates xi = (x,y,z) coincide with geometry coordinates.
        // Shape functions expressed in (x,y,z) corresponding to base corners
        // (-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0) and apex (0,0,1).
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        const Real one_minus_z = Real(1) - z;

        // Handle apex separately to avoid division by zero.
        const Real eps = Real(1e-12);
	        if (std::abs(one_minus_z) < eps) {
	            values[0] = Real(0);
	            values[1] = Real(0);
	            values[2] = Real(0);
	            values[3] = Real(0);
	            values[4] = Real(1);
                apply_node_ordering();
	            return;
	        }

        const Real denom = Real(4) * one_minus_z;
        const Real xm = one_minus_z - x;
        const Real xp = one_minus_z + x;
        const Real ym = one_minus_z - y;
        const Real yp = one_minus_z + y;

        values[0] = (xm * ym) / denom; // (-1,-1,0)
	        values[1] = (xp * ym) / denom; // ( 1,-1,0)
	        values[2] = (xp * yp) / denom; // ( 1, 1,0)
	        values[3] = (xm * yp) / denom; // (-1, 1,0)
	        values[4] = z;                 // apex
            apply_node_ordering();
	        return;
	    }

	    if (is_triangle(element_type_)) {
            const int p = order_;
	        const Real l1 = xi[0];
	        const Real l2 = xi[1];
	        const Real l0 = Real(1) - l1 - l2;

            std::vector<Real> phi0, dphi0, d2phi0;
            std::vector<Real> phi1, dphi1, d2phi1;
            std::vector<Real> phi2, dphi2, d2phi2;
            simplex_lagrange_factor_sequence(p, l0, phi0, dphi0, d2phi0);
            simplex_lagrange_factor_sequence(p, l1, phi1, dphi1, d2phi1);
            simplex_lagrange_factor_sequence(p, l2, phi2, dphi2, d2phi2);

	        for (std::size_t n = 0; n < simplex_exponents_.size(); ++n) {
	            const auto& e = simplex_exponents_[n];
                values[n] =
                    phi0[static_cast<std::size_t>(e[0])] *
                    phi1[static_cast<std::size_t>(e[1])] *
                    phi2[static_cast<std::size_t>(e[2])];
	        }
            apply_node_ordering();
	        return;
	    }

	    if (is_tetrahedron(element_type_)) {
            const int p = order_;
	        const Real l1 = xi[0];
	        const Real l2 = xi[1];
	        const Real l3 = xi[2];
	        const Real l0 = Real(1) - l1 - l2 - l3;

            std::vector<Real> phi0, dphi0, d2phi0;
            std::vector<Real> phi1, dphi1, d2phi1;
            std::vector<Real> phi2, dphi2, d2phi2;
            std::vector<Real> phi3, dphi3, d2phi3;
            simplex_lagrange_factor_sequence(p, l0, phi0, dphi0, d2phi0);
            simplex_lagrange_factor_sequence(p, l1, phi1, dphi1, d2phi1);
            simplex_lagrange_factor_sequence(p, l2, phi2, dphi2, d2phi2);
            simplex_lagrange_factor_sequence(p, l3, phi3, dphi3, d2phi3);

	        for (std::size_t n = 0; n < simplex_exponents_.size(); ++n) {
	            const auto& e = simplex_exponents_[n];
                values[n] =
                    phi0[static_cast<std::size_t>(e[0])] *
                    phi1[static_cast<std::size_t>(e[1])] *
                    phi2[static_cast<std::size_t>(e[2])] *
                    phi3[static_cast<std::size_t>(e[3])];
	        }
            apply_node_ordering();
	        return;
	    }

    throw FEException("Unsupported element in evaluate_values",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

void LagrangeBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
	                                       std::vector<Gradient>& gradients) const {
	    gradients.assign(size(), Gradient{});

        auto apply_node_ordering = [&]() {
            if (external_to_internal_.empty()) {
                return;
            }
            std::vector<Gradient> tmp = gradients;
            gradients.assign(external_to_internal_.size(), Gradient{});
            for (std::size_t ext = 0; ext < external_to_internal_.size(); ++ext) {
                gradients[ext] = tmp[external_to_internal_[ext]];
            }
        };

	    if (element_type_ == ElementType::Point1) {
            apply_node_ordering();
	        return;
	    }

    if (is_line(element_type_)) {
        auto d = evaluate_1d_derivative(xi[0]);
	        for (std::size_t i = 0; i < d.size(); ++i) {
	            gradients[i][0] = d[i];
	        }
            apply_node_ordering();
	        return;
	    }

    if (is_quadrilateral(element_type_)) {
        auto lx = evaluate_1d(xi[0]);
        auto ly = evaluate_1d(xi[1]);
        auto dx = evaluate_1d_derivative(xi[0]);
        auto dy = evaluate_1d_derivative(xi[1]);

        const std::size_t n1d = nodes_1d_.size();
        std::size_t idx = 0;
	        for (std::size_t j = 0; j < n1d; ++j) {
	            for (std::size_t i = 0; i < n1d; ++i) {
	                gradients[idx][0] = dx[i] * ly[j];
	                gradients[idx][1] = lx[i] * dy[j];
	                ++idx;
	            }
	        }
            apply_node_ordering();
	        return;
	    }

    if (is_hexahedron(element_type_)) {
        auto lx = evaluate_1d(xi[0]);
        auto ly = evaluate_1d(xi[1]);
        auto lz = evaluate_1d(xi[2]);
        auto dx = evaluate_1d_derivative(xi[0]);
        auto dy = evaluate_1d_derivative(xi[1]);
        auto dz = evaluate_1d_derivative(xi[2]);

        const std::size_t n1d = nodes_1d_.size();
        std::size_t idx = 0;
	        for (std::size_t k = 0; k < n1d; ++k) {
	            for (std::size_t j = 0; j < n1d; ++j) {
	                for (std::size_t i = 0; i < n1d; ++i) {
	                    gradients[idx][0] = dx[i] * ly[j] * lz[k];
	                    gradients[idx][1] = lx[i] * dy[j] * lz[k];
	                    gradients[idx][2] = lx[i] * ly[j] * dz[k];
	                    ++idx;
	                }
	            }
	        }
            apply_node_ordering();
	        return;
	    }

    if (is_wedge(element_type_)) {
        auto lz = evaluate_1d(xi[2]);
        auto dz = evaluate_1d_derivative(xi[2]);

        const int p = order_;
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;

        std::vector<Real> phi0, dphi0, d2phi0;
        std::vector<Real> phi1, dphi1, d2phi1;
        std::vector<Real> phi2, dphi2, d2phi2;
        simplex_lagrange_factor_sequence(p, l0, phi0, dphi0, d2phi0);
        simplex_lagrange_factor_sequence(p, l1, phi1, dphi1, d2phi1);
        simplex_lagrange_factor_sequence(p, l2, phi2, dphi2, d2phi2);

        std::vector<Real> tri_vals(simplex_exponents_.size(), Real(0));
        std::vector<Gradient> tri_grads(simplex_exponents_.size(), Gradient{});
        for (std::size_t n = 0; n < simplex_exponents_.size(); ++n) {
            const auto& e = simplex_exponents_[n];
            const std::size_t i0 = static_cast<std::size_t>(e[0]);
            const std::size_t i1 = static_cast<std::size_t>(e[1]);
            const std::size_t i2 = static_cast<std::size_t>(e[2]);

            const Real v0 = phi0[i0];
            const Real v1 = phi1[i1];
            const Real v2 = phi2[i2];
            const Real d0 = dphi0[i0];
            const Real d1 = dphi1[i1];
            const Real d2 = dphi2[i2];

            // Triangle basis in physical reference coordinates (x=l1, y=l2):
            // d/dx = d/dl1 - d/dl0, d/dy = d/dl2 - d/dl0.
            const Real dl0 = d0 * v1 * v2;
            const Real dl1 = v0 * d1 * v2;
            const Real dl2 = v0 * v1 * d2;

            tri_vals[n] = v0 * v1 * v2;
            tri_grads[n][0] = dl1 - dl0;
            tri_grads[n][1] = dl2 - dl0;
            tri_grads[n][2] = Real(0);
        }

        std::size_t idx = 0;
        for (std::size_t kz = 0; kz < lz.size(); ++kz) {
            for (std::size_t n = 0; n < tri_vals.size(); ++n) {
                gradients[idx][0] = tri_grads[n][0] * lz[kz];
                gradients[idx][1] = tri_grads[n][1] * lz[kz];
                gradients[idx][2] = tri_vals[n] * dz[kz];
                ++idx;
            }
        }
        apply_node_ordering();
        return;
    }

    if (element_type_ == ElementType::Pyramid5) {
        // Analytic gradients for the rational Pyramid5 shape functions.
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        const Real one_minus_z = Real(1) - z;

        const Real eps = Real(1e-12);
        if (std::abs(one_minus_z) < eps) {
            // The pyramid basis is continuous but not differentiable at the apex.
            // Return the symmetric limit along the centerline (x=y=0).
            gradients[0] = Gradient{Real(-0.25), Real(-0.25), Real(-0.25)};
            gradients[1] = Gradient{Real(0.25),  Real(-0.25), Real(-0.25)};
            gradients[2] = Gradient{Real(0.25),  Real(0.25),  Real(-0.25)};
            gradients[3] = Gradient{Real(-0.25), Real(0.25),  Real(-0.25)};
            gradients[4] = Gradient{Real(0),     Real(0),     Real(1)};
            apply_node_ordering();
            return;
        }

        const Real inv4t = Real(1) / (Real(4) * one_minus_z);
        const Real inv_t2 = Real(1) / (one_minus_z * one_minus_z);
        const Real xy_over_t2 = x * y * inv_t2;

        // Corner nodes: (-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0)
        gradients[0][0] = (y - one_minus_z) * inv4t;
        gradients[0][1] = (x - one_minus_z) * inv4t;
        gradients[0][2] = (xy_over_t2 - Real(1)) * Real(0.25);

        gradients[1][0] = (one_minus_z - y) * inv4t;
        gradients[1][1] = -(one_minus_z + x) * inv4t;
        gradients[1][2] = (-xy_over_t2 - Real(1)) * Real(0.25);

        gradients[2][0] = (one_minus_z + y) * inv4t;
        gradients[2][1] = (one_minus_z + x) * inv4t;
        gradients[2][2] = (xy_over_t2 - Real(1)) * Real(0.25);

        gradients[3][0] = -(one_minus_z + y) * inv4t;
        gradients[3][1] = (one_minus_z - x) * inv4t;
        gradients[3][2] = (-xy_over_t2 - Real(1)) * Real(0.25);

        // Apex node: (0,0,1)
        gradients[4] = Gradient{Real(0), Real(0), Real(1)};

        apply_node_ordering();
        return;
    }

    if (is_pyramid(element_type_)) {
        // Use numerical differentiation for higher-order pyramids (e.g. Pyramid14).
        numerical_gradient(xi, gradients);
        apply_node_ordering();
        return;
    }

    if (is_triangle(element_type_)) {
        const int p = order_;
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;

        std::vector<Real> phi0, dphi0, d2phi0;
        std::vector<Real> phi1, dphi1, d2phi1;
        std::vector<Real> phi2, dphi2, d2phi2;
        simplex_lagrange_factor_sequence(p, l0, phi0, dphi0, d2phi0);
        simplex_lagrange_factor_sequence(p, l1, phi1, dphi1, d2phi1);
        simplex_lagrange_factor_sequence(p, l2, phi2, dphi2, d2phi2);

        for (std::size_t n = 0; n < simplex_exponents_.size(); ++n) {
            const auto& e = simplex_exponents_[n];
            const std::size_t i0 = static_cast<std::size_t>(e[0]);
            const std::size_t i1 = static_cast<std::size_t>(e[1]);
            const std::size_t i2 = static_cast<std::size_t>(e[2]);

            const Real v0 = phi0[i0];
            const Real v1 = phi1[i1];
            const Real v2 = phi2[i2];
            const Real d0 = dphi0[i0];
            const Real d1 = dphi1[i1];
            const Real d2 = dphi2[i2];

            const Real dl0 = d0 * v1 * v2;
            const Real dl1 = v0 * d1 * v2;
            const Real dl2 = v0 * v1 * d2;

            gradients[n][0] = dl1 - dl0;
            gradients[n][1] = dl2 - dl0;
            gradients[n][2] = Real(0);
        }
        apply_node_ordering();
        return;
    }

    if (is_tetrahedron(element_type_)) {
        const int p = order_;
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l3 = xi[2];
        const Real l0 = Real(1) - l1 - l2 - l3;

        std::vector<Real> phi0, dphi0, d2phi0;
        std::vector<Real> phi1, dphi1, d2phi1;
        std::vector<Real> phi2, dphi2, d2phi2;
        std::vector<Real> phi3, dphi3, d2phi3;
        simplex_lagrange_factor_sequence(p, l0, phi0, dphi0, d2phi0);
        simplex_lagrange_factor_sequence(p, l1, phi1, dphi1, d2phi1);
        simplex_lagrange_factor_sequence(p, l2, phi2, dphi2, d2phi2);
        simplex_lagrange_factor_sequence(p, l3, phi3, dphi3, d2phi3);

        for (std::size_t n = 0; n < simplex_exponents_.size(); ++n) {
            const auto& e = simplex_exponents_[n];
            const std::size_t i0 = static_cast<std::size_t>(e[0]);
            const std::size_t i1 = static_cast<std::size_t>(e[1]);
            const std::size_t i2 = static_cast<std::size_t>(e[2]);
            const std::size_t i3 = static_cast<std::size_t>(e[3]);

            const Real v0 = phi0[i0];
            const Real v1 = phi1[i1];
            const Real v2 = phi2[i2];
            const Real v3 = phi3[i3];
            const Real d0 = dphi0[i0];
            const Real d1 = dphi1[i1];
            const Real d2 = dphi2[i2];
            const Real d3 = dphi3[i3];

            const Real dl0 = d0 * v1 * v2 * v3;
            const Real dl1 = v0 * d1 * v2 * v3;
            const Real dl2 = v0 * v1 * d2 * v3;
            const Real dl3 = v0 * v1 * v2 * d3;

            gradients[n][0] = dl1 - dl0;
            gradients[n][1] = dl2 - dl0;
            gradients[n][2] = dl3 - dl0;
        }
        apply_node_ordering();
        return;
    }

    throw FEException("Unsupported element in evaluate_gradients",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

void LagrangeBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                      std::vector<Hessian>& hessians) const {
    hessians.assign(size(), Hessian{});

    auto apply_node_ordering = [&]() {
        if (external_to_internal_.empty()) {
            return;
        }
        std::vector<Hessian> tmp = hessians;
        hessians.assign(external_to_internal_.size(), Hessian{});
        for (std::size_t ext = 0; ext < external_to_internal_.size(); ++ext) {
            hessians[ext] = tmp[external_to_internal_[ext]];
        }
    };

    if (element_type_ == ElementType::Point1) {
        apply_node_ordering();
        return;
    }

    if (is_line(element_type_)) {
        auto d2 = evaluate_1d_second_derivative(xi[0]);
        for (std::size_t i = 0; i < d2.size(); ++i) {
            hessians[i](0, 0) = d2[i];
        }
        apply_node_ordering();
        return;
    }

    if (is_quadrilateral(element_type_)) {
        auto lx = evaluate_1d(xi[0]);
        auto ly = evaluate_1d(xi[1]);
        auto dx = evaluate_1d_derivative(xi[0]);
        auto dy = evaluate_1d_derivative(xi[1]);
        auto d2x = evaluate_1d_second_derivative(xi[0]);
        auto d2y = evaluate_1d_second_derivative(xi[1]);

        const std::size_t n1d = nodes_1d_.size();
        std::size_t idx = 0;
        for (std::size_t j = 0; j < n1d; ++j) {
            for (std::size_t i = 0; i < n1d; ++i) {
                hessians[idx](0, 0) = d2x[i] * ly[j];   // ∂²N/∂x²
                hessians[idx](1, 1) = lx[i] * d2y[j];   // ∂²N/∂y²
                hessians[idx](0, 1) = dx[i] * dy[j];    // ∂²N/∂x∂y
                hessians[idx](1, 0) = hessians[idx](0, 1); // symmetric
                ++idx;
            }
        }
        apply_node_ordering();
        return;
    }

    if (is_hexahedron(element_type_)) {
        auto lx = evaluate_1d(xi[0]);
        auto ly = evaluate_1d(xi[1]);
        auto lz = evaluate_1d(xi[2]);
        auto dx = evaluate_1d_derivative(xi[0]);
        auto dy = evaluate_1d_derivative(xi[1]);
        auto dz = evaluate_1d_derivative(xi[2]);
        auto d2x = evaluate_1d_second_derivative(xi[0]);
        auto d2y = evaluate_1d_second_derivative(xi[1]);
        auto d2z = evaluate_1d_second_derivative(xi[2]);

        const std::size_t n1d = nodes_1d_.size();
        std::size_t idx = 0;
        for (std::size_t k = 0; k < n1d; ++k) {
            for (std::size_t j = 0; j < n1d; ++j) {
                for (std::size_t i = 0; i < n1d; ++i) {
                    // Diagonal terms
                    hessians[idx](0, 0) = d2x[i] * ly[j] * lz[k];
                    hessians[idx](1, 1) = lx[i] * d2y[j] * lz[k];
                    hessians[idx](2, 2) = lx[i] * ly[j] * d2z[k];
                    // Off-diagonal terms
                    hessians[idx](0, 1) = dx[i] * dy[j] * lz[k];
                    hessians[idx](1, 0) = hessians[idx](0, 1);
                    hessians[idx](0, 2) = dx[i] * ly[j] * dz[k];
                    hessians[idx](2, 0) = hessians[idx](0, 2);
                    hessians[idx](1, 2) = lx[i] * dy[j] * dz[k];
                    hessians[idx](2, 1) = hessians[idx](1, 2);
                    ++idx;
                }
            }
        }
        apply_node_ordering();
        return;
    }

    if (is_triangle(element_type_)) {
        const int p = order_;
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;

        std::vector<Real> phi0, dphi0, d2phi0;
        std::vector<Real> phi1, dphi1, d2phi1;
        std::vector<Real> phi2, dphi2, d2phi2;
        simplex_lagrange_factor_sequence(p, l0, phi0, dphi0, d2phi0);
        simplex_lagrange_factor_sequence(p, l1, phi1, dphi1, d2phi1);
        simplex_lagrange_factor_sequence(p, l2, phi2, dphi2, d2phi2);

        for (std::size_t n = 0; n < simplex_exponents_.size(); ++n) {
            const auto& e = simplex_exponents_[n];
            const std::size_t i0 = static_cast<std::size_t>(e[0]);
            const std::size_t i1 = static_cast<std::size_t>(e[1]);
            const std::size_t i2 = static_cast<std::size_t>(e[2]);

            const Real v0 = phi0[i0];
            const Real v1 = phi1[i1];
            const Real v2 = phi2[i2];
            const Real D0 = dphi0[i0];
            const Real D1 = dphi1[i1];
            const Real D2 = dphi2[i2];
            const Real DD0 = d2phi0[i0];
            const Real DD1 = d2phi1[i1];
            const Real DD2 = d2phi2[i2];

            // Hessian in barycentric coordinates (l0,l1,l2)
            const Real H00 = DD0 * v1 * v2;
            const Real H11 = v0 * DD1 * v2;
            const Real H22 = v0 * v1 * DD2;
            const Real H01 = D0 * D1 * v2;
            const Real H02 = D0 * v1 * D2;
            const Real H12 = v0 * D1 * D2;

            // Map to (x,y) with x=l1, y=l2, l0=1-x-y:
            // d/dx = d/dl1 - d/dl0, d/dy = d/dl2 - d/dl0.
            const Real H_xx = H00 - Real(2) * H01 + H11;
            const Real H_yy = H00 - Real(2) * H02 + H22;
            const Real H_xy = H00 - H01 - H02 + H12;

            Hessian H{};
            H(0, 0) = H_xx;
            H(1, 1) = H_yy;
            H(0, 1) = H_xy;
            H(1, 0) = H_xy;
            hessians[n] = H;
        }
        apply_node_ordering();
        return;
    }

    if (is_tetrahedron(element_type_)) {
        const int p = order_;
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l3 = xi[2];
        const Real l0 = Real(1) - l1 - l2 - l3;

        std::vector<Real> phi0, dphi0, d2phi0;
        std::vector<Real> phi1, dphi1, d2phi1;
        std::vector<Real> phi2, dphi2, d2phi2;
        std::vector<Real> phi3, dphi3, d2phi3;
        simplex_lagrange_factor_sequence(p, l0, phi0, dphi0, d2phi0);
        simplex_lagrange_factor_sequence(p, l1, phi1, dphi1, d2phi1);
        simplex_lagrange_factor_sequence(p, l2, phi2, dphi2, d2phi2);
        simplex_lagrange_factor_sequence(p, l3, phi3, dphi3, d2phi3);

        for (std::size_t n = 0; n < simplex_exponents_.size(); ++n) {
            const auto& e = simplex_exponents_[n];
            const std::size_t i0 = static_cast<std::size_t>(e[0]);
            const std::size_t i1 = static_cast<std::size_t>(e[1]);
            const std::size_t i2 = static_cast<std::size_t>(e[2]);
            const std::size_t i3 = static_cast<std::size_t>(e[3]);

            const Real v0 = phi0[i0];
            const Real v1 = phi1[i1];
            const Real v2 = phi2[i2];
            const Real v3 = phi3[i3];
            const Real D0 = dphi0[i0];
            const Real D1 = dphi1[i1];
            const Real D2 = dphi2[i2];
            const Real D3 = dphi3[i3];
            const Real DD0 = d2phi0[i0];
            const Real DD1 = d2phi1[i1];
            const Real DD2 = d2phi2[i2];
            const Real DD3 = d2phi3[i3];

            // Hessian in barycentric coordinates (l0,l1,l2,l3)
            const Real H00 = DD0 * v1 * v2 * v3;
            const Real H11 = v0 * DD1 * v2 * v3;
            const Real H22 = v0 * v1 * DD2 * v3;
            const Real H33 = v0 * v1 * v2 * DD3;

            const Real H01 = D0 * D1 * v2 * v3;
            const Real H02 = D0 * v1 * D2 * v3;
            const Real H03 = D0 * v1 * v2 * D3;
            const Real H12 = v0 * D1 * D2 * v3;
            const Real H13 = v0 * D1 * v2 * D3;
            const Real H23 = v0 * v1 * D2 * D3;

            // Map to (x,y,z) with x=l1, y=l2, z=l3, l0=1-x-y-z:
            // d/dx = d/dl1 - d/dl0, etc.
            const Real H_xx = H00 - Real(2) * H01 + H11;
            const Real H_yy = H00 - Real(2) * H02 + H22;
            const Real H_zz = H00 - Real(2) * H03 + H33;
            const Real H_xy = H00 - H01 - H02 + H12;
            const Real H_xz = H00 - H01 - H03 + H13;
            const Real H_yz = H00 - H02 - H03 + H23;

            Hessian H{};
            H(0, 0) = H_xx;
            H(1, 1) = H_yy;
            H(2, 2) = H_zz;
            H(0, 1) = H_xy;
            H(1, 0) = H_xy;
            H(0, 2) = H_xz;
            H(2, 0) = H_xz;
            H(1, 2) = H_yz;
            H(2, 1) = H_yz;
            hessians[n] = H;
        }
        apply_node_ordering();
        return;
    }

    if (is_wedge(element_type_)) {
        auto lz = evaluate_1d(xi[2]);
        auto dz = evaluate_1d_derivative(xi[2]);
        auto d2z = evaluate_1d_second_derivative(xi[2]);

        const int p = order_;
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;

        std::vector<Real> phi0, dphi0, d2phi0;
        std::vector<Real> phi1, dphi1, d2phi1;
        std::vector<Real> phi2, dphi2, d2phi2;
        simplex_lagrange_factor_sequence(p, l0, phi0, dphi0, d2phi0);
        simplex_lagrange_factor_sequence(p, l1, phi1, dphi1, d2phi1);
        simplex_lagrange_factor_sequence(p, l2, phi2, dphi2, d2phi2);

        std::vector<Real> tri_val(simplex_exponents_.size(), Real(0));
        std::vector<Real> tri_dx(simplex_exponents_.size(), Real(0));
        std::vector<Real> tri_dy(simplex_exponents_.size(), Real(0));
        std::vector<Real> tri_hxx(simplex_exponents_.size(), Real(0));
        std::vector<Real> tri_hyy(simplex_exponents_.size(), Real(0));
        std::vector<Real> tri_hxy(simplex_exponents_.size(), Real(0));

        for (std::size_t n = 0; n < simplex_exponents_.size(); ++n) {
            const auto& e = simplex_exponents_[n];
            const std::size_t i0 = static_cast<std::size_t>(e[0]);
            const std::size_t i1 = static_cast<std::size_t>(e[1]);
            const std::size_t i2 = static_cast<std::size_t>(e[2]);

            const Real v0 = phi0[i0];
            const Real v1 = phi1[i1];
            const Real v2 = phi2[i2];
            const Real D0 = dphi0[i0];
            const Real D1 = dphi1[i1];
            const Real D2 = dphi2[i2];
            const Real DD0 = d2phi0[i0];
            const Real DD1 = d2phi1[i1];
            const Real DD2 = d2phi2[i2];

            tri_val[n] = v0 * v1 * v2;

            // First derivatives in (x,y) with x=l1, y=l2, l0=1-x-y.
            const Real dl0 = D0 * v1 * v2;
            const Real dl1 = v0 * D1 * v2;
            const Real dl2 = v0 * v1 * D2;
            tri_dx[n] = dl1 - dl0;
            tri_dy[n] = dl2 - dl0;

            // Second derivatives in barycentric coordinates (l0,l1,l2)
            const Real H00 = DD0 * v1 * v2;
            const Real H11 = v0 * DD1 * v2;
            const Real H22 = v0 * v1 * DD2;
            const Real H01 = D0 * D1 * v2;
            const Real H02 = D0 * v1 * D2;
            const Real H12 = v0 * D1 * D2;

            tri_hxx[n] = H00 - Real(2) * H01 + H11;
            tri_hyy[n] = H00 - Real(2) * H02 + H22;
            tri_hxy[n] = H00 - H01 - H02 + H12;
        }

        std::size_t idx = 0;
        for (std::size_t kz = 0; kz < lz.size(); ++kz) {
            for (std::size_t n = 0; n < tri_val.size(); ++n) {
                Hessian H{};
                H(0, 0) = tri_hxx[n] * lz[kz];
                H(1, 1) = tri_hyy[n] * lz[kz];
                H(0, 1) = tri_hxy[n] * lz[kz];
                H(1, 0) = H(0, 1);

                H(2, 2) = tri_val[n] * d2z[kz];

                H(0, 2) = tri_dx[n] * dz[kz];
                H(2, 0) = H(0, 2);
                H(1, 2) = tri_dy[n] * dz[kz];
                H(2, 1) = H(1, 2);

                hessians[idx] = H;
                ++idx;
            }
        }
        apply_node_ordering();
        return;
    }

    if (element_type_ == ElementType::Pyramid5) {
        // Analytic Hessians for the rational Pyramid5 shape functions.
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        const Real one_minus_z = Real(1) - z;

        const Real eps = Real(1e-12);
        if (std::abs(one_minus_z) < eps) {
            // Hessians are not well-defined at the apex; return zeros.
            apply_node_ordering();
            return;
        }

        const Real inv4t = Real(1) / (Real(4) * one_minus_z);
        const Real inv_t2 = Real(1) / (one_minus_z * one_minus_z);
        const Real inv_t3 = inv_t2 / one_minus_z;
        const Real xy = x * y;

        auto fill_corner = [&](std::size_t i, Real sign_xy) {
            const Real H_xy = sign_xy * inv4t;
            const Real H_xz = sign_xy * y * inv_t2 * Real(0.25);
            const Real H_yz = sign_xy * x * inv_t2 * Real(0.25);
            const Real H_zz = sign_xy * xy * inv_t3 * Real(0.5);

            Hessian H{};
            H(0, 1) = H_xy;
            H(1, 0) = H_xy;
            H(0, 2) = H_xz;
            H(2, 0) = H_xz;
            H(1, 2) = H_yz;
            H(2, 1) = H_yz;
            H(2, 2) = H_zz;
            hessians[i] = H;
        };

        // Corner nodes alternate the sign s_x*s_y = ±1.
        fill_corner(0, Real(1));
        fill_corner(1, Real(-1));
        fill_corner(2, Real(1));
        fill_corner(3, Real(-1));

        // Apex node has zero second derivatives.
        hessians[4] = Hessian{};

        apply_node_ordering();
        return;
    }

    if (is_pyramid(element_type_)) {
        // Use numerical differentiation for higher-order pyramids (e.g. Pyramid14).
        numerical_hessian(xi, hessians);
        apply_node_ordering();
        return;
    }

    throw FEException("Unsupported element in evaluate_hessians",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

} // namespace basis
} // namespace FE
} // namespace svmp
