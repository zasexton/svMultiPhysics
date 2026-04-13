/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SerendipityBasis.h"
#include "LagrangeBasis.h"
#include "Core/FEException.h"
#include "NodeOrderingConventions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <string>

namespace svmp {
namespace FE {
namespace basis {

namespace {
using Vec3 = math::Vector<Real, 3>;

Real powi(Real x, int exponent) {
    Real value = Real(1);
    for (int i = 0; i < exponent; ++i) {
        value *= x;
    }
    return value;
}

int quad_serendipity_superlinear_degree(int ax, int ay) {
    return (ax > 1 ? ax : 0) + (ay > 1 ? ay : 0);
}

std::vector<std::array<int, 2>> quad_serendipity_exponents(int order) {
    std::vector<std::array<int, 2>> exponents;
    for (int ay = 0; ay <= order; ++ay) {
        for (int ax = 0; ax <= order; ++ax) {
            if (quad_serendipity_superlinear_degree(ax, ay) <= order) {
                exponents.push_back({ax, ay});
            }
        }
    }
    return exponents;
}

std::vector<Vec3> quad_serendipity_nodes(int order, std::size_t total_size) {
    std::vector<Vec3> nodes;
    if (order <= 0) {
        return nodes;
    }

    const Real inv_order = Real(1) / Real(order);

    nodes.push_back(Vec3{Real(-1), Real(-1), Real(0)});
    nodes.push_back(Vec3{Real(1),  Real(-1), Real(0)});
    nodes.push_back(Vec3{Real(1),  Real(1),  Real(0)});
    nodes.push_back(Vec3{Real(-1), Real(1),  Real(0)});

    for (int i = 1; i < order; ++i) {
        nodes.push_back(Vec3{Real(-1) + Real(2 * i) * inv_order, Real(-1), Real(0)});
    }
    for (int i = 1; i < order; ++i) {
        nodes.push_back(Vec3{Real(1), Real(-1) + Real(2 * i) * inv_order, Real(0)});
    }
    for (int i = 1; i < order; ++i) {
        nodes.push_back(Vec3{Real(1) - Real(2 * i) * inv_order, Real(1), Real(0)});
    }
    for (int i = 1; i < order; ++i) {
        nodes.push_back(Vec3{Real(-1), Real(1) - Real(2 * i) * inv_order, Real(0)});
    }

    if (nodes.size() > total_size) {
        throw FEException("SerendipityBasis: quadrilateral serendipity boundary nodes exceed requested size",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    const std::size_t interior_count = total_size - nodes.size();
    if (interior_count == 0u) {
        return nodes;
    }

    std::vector<Vec3> interior_candidates;
    interior_candidates.reserve(static_cast<std::size_t>((order - 1) * (order - 1)));
    for (int j = 1; j < order; ++j) {
        for (int i = 1; i < order; ++i) {
            interior_candidates.push_back(
                Vec3{Real(-1) + Real(2 * i) * inv_order,
                     Real(-1) + Real(2 * j) * inv_order,
                     Real(0)});
        }
    }

    std::sort(interior_candidates.begin(), interior_candidates.end(),
              [](const Vec3& a, const Vec3& b) {
                  const Real a_linf = std::max(std::abs(a[0]), std::abs(a[1]));
                  const Real b_linf = std::max(std::abs(b[0]), std::abs(b[1]));
                  if (a_linf != b_linf) {
                      return a_linf < b_linf;
                  }

                  const Real a_l1 = std::abs(a[0]) + std::abs(a[1]);
                  const Real b_l1 = std::abs(b[0]) + std::abs(b[1]);
                  if (a_l1 != b_l1) {
                      return a_l1 < b_l1;
                  }

                  if (a[1] != b[1]) {
                      return a[1] < b[1];
                  }
                  return a[0] < b[0];
              });

    if (interior_count > interior_candidates.size()) {
        throw FEException("SerendipityBasis: insufficient quadrilateral interior nodes for requested serendipity order",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    nodes.insert(nodes.end(),
                 interior_candidates.begin(),
                 interior_candidates.begin() + static_cast<std::ptrdiff_t>(interior_count));
    return nodes;
}

std::vector<Real> invert_dense_matrix(std::vector<Real> matrix, int n, const char* label) {
    std::vector<Real> inverse(static_cast<std::size_t>(n * n), Real(0));
    for (int i = 0; i < n; ++i) {
        inverse[static_cast<std::size_t>(i * n + i)] = Real(1);
    }

    auto idx = [n](int row, int col) -> std::size_t {
        return static_cast<std::size_t>(row * n + col);
    };

    for (int col = 0; col < n; ++col) {
        int pivot_row = col;
        Real pivot_abs = std::abs(matrix[idx(col, col)]);
        for (int row = col + 1; row < n; ++row) {
            const Real cand_abs = std::abs(matrix[idx(row, col)]);
            if (cand_abs > pivot_abs) {
                pivot_abs = cand_abs;
                pivot_row = row;
            }
        }

        if (pivot_abs <= Real(1e-14)) {
            throw FEException(std::string("SerendipityBasis: singular interpolation matrix for ") + label,
                              __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
        }

        if (pivot_row != col) {
            for (int j = 0; j < n; ++j) {
                std::swap(matrix[idx(col, j)], matrix[idx(pivot_row, j)]);
                std::swap(inverse[idx(col, j)], inverse[idx(pivot_row, j)]);
            }
        }

        const Real pivot = matrix[idx(col, col)];
        for (int j = 0; j < n; ++j) {
            matrix[idx(col, j)] /= pivot;
            inverse[idx(col, j)] /= pivot;
        }

        for (int row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const Real factor = matrix[idx(row, col)];
            if (std::abs(factor) <= Real(0)) {
                continue;
            }
            for (int j = 0; j < n; ++j) {
                matrix[idx(row, j)] -= factor * matrix[idx(col, j)];
                inverse[idx(row, j)] -= factor * inverse[idx(col, j)];
            }
        }
    }

    return inverse;
}

std::vector<Real> quad_serendipity_inverse_vandermonde(
    std::span<const Vec3> nodes,
    std::span<const std::array<int, 2>> exponents,
    int order) {
    const int n = static_cast<int>(nodes.size());
    if (n == 0 || exponents.size() != nodes.size()) {
        throw FEException("SerendipityBasis: invalid quadrilateral serendipity interpolation setup",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    std::vector<Real> vandermonde(static_cast<std::size_t>(n * n), Real(0));
    auto idx = [n](int row, int col) -> std::size_t {
        return static_cast<std::size_t>(row * n + col);
    };

    for (int row = 0; row < n; ++row) {
        const Real x = nodes[static_cast<std::size_t>(row)][0];
        const Real y = nodes[static_cast<std::size_t>(row)][1];
        for (int col = 0; col < n; ++col) {
            const auto [ax, ay] = exponents[static_cast<std::size_t>(col)];
            vandermonde[idx(row, col)] = powi(x, ax) * powi(y, ay);
        }
    }

    const std::string label = "Quad order " + std::to_string(order);
    return invert_dense_matrix(std::move(vandermonde), n, label.c_str());
}
// Mesh use a conventional Hex20 node ordering: corners first, then edge
// midpoints in the order {bottom, top, vertical}. The original polynomial
// generator for the Hex20 field basis used an axis-grouped edge ordering.
//
// This permutation maps Mesh/NodeOrdering index -> internal axis-group index so
// that SerendipityBasis outputs are consistent with NodeOrderingConventions.
constexpr std::array<std::size_t, 20> kHex20NodeOrderingToInternal = {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 13, 10, 12,
    9, 15, 11, 14,
    16, 17, 19, 18
};

constexpr std::array<Real, 13> kPyramid13CenterRedistribution = {
    Real(-0.25), Real(-0.25), Real(-0.25), Real(-0.25),
    Real(0),
    Real(0.5), Real(0.5), Real(0.5), Real(0.5),
    Real(0), Real(0), Real(0), Real(0)
};

static const int hex20_monomial_exponents[20][3] = {
    {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1},
    {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {1, 0, 0}, {1, 0, 1},
    {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0},
    {1, 2, 1}, {2, 0, 0}, {2, 0, 1}, {2, 1, 0}, {2, 1, 1}
};

static const Real hex20_coeffs[20][20] = {
    {-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
    {0.125, 0.125, 0.125, 0.125, -0.125, -0.125, -0.125, -0.125, -0.25, 0.25, -0.25, 0.25, -0.25, -0.25, 0.25, 0.25, 0, 0, 0, 0},
    {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0, 0, 0, 0, 0, 0, 0, 0, -0.25, -0.25, -0.25, -0.25},
    {0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.125, -0.125, -0.25, -0.25, 0.25, 0.25, 0, 0, 0, 0, -0.25, -0.25, 0.25, 0.25},
    {0, 0, 0, 0, 0, 0, 0, 0, 0.25, -0.25, -0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0},
    {-0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.25, -0.25, -0.25},
    {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0, 0, 0, 0, -0.25, -0.25, -0.25, -0.25, 0, 0, 0, 0},
    {-0.125, -0.125, -0.125, -0.125, 0.125, 0.125, 0.125, 0.125, 0, 0, 0, 0, 0.25, 0.25, -0.25, -0.25, 0, 0, 0, 0},
    {0.125, -0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0, 0, 0, 0, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, -0.25, -0.25, 0.25, 0, 0, 0, 0},
    {-0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, -0.25, 0.25, -0.25},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, -0.25, -0.25, 0.25},
    {-0.125, 0.125, -0.125, 0.125, 0.125, -0.125, 0.125, -0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0.125, -0.125, 0.125, -0.125, 0.125, -0.125, 0.125, -0.125, 0, 0, 0, 0, 0, 0, 0, 0, -0.25, 0.25, 0.25, -0.25},
    {-0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.125, 0, 0, 0, 0, 0.25, -0.25, 0.25, -0.25, 0, 0, 0, 0},
    {0.125, -0.125, -0.125, 0.125, -0.125, 0.125, 0.125, -0.125, 0, 0, 0, 0, -0.25, 0.25, 0.25, -0.25, 0, 0, 0, 0},
    {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, -0.25, -0.25, -0.25, -0.25, 0, 0, 0, 0, 0, 0, 0, 0},
    {-0.125, -0.125, -0.125, -0.125, 0.125, 0.125, 0.125, 0.125, 0.25, -0.25, 0.25, -0.25, 0, 0, 0, 0, 0, 0, 0, 0},
    {-0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, 0.25, 0.25, -0.25, -0.25, 0, 0, 0, 0, 0, 0, 0, 0},
    {0.125, 0.125, -0.125, -0.125, -0.125, -0.125, 0.125, 0.125, -0.25, 0.25, 0.25, -0.25, 0, 0, 0, 0, 0, 0, 0, 0}
};

void eval_hex20_internal(Real r, Real s, Real t, Real* internal_vals) {
    Real phi[20];
    for (int j = 0; j < 20; ++j) {
        const int a = hex20_monomial_exponents[j][0];
        const int b = hex20_monomial_exponents[j][1];
        const int c = hex20_monomial_exponents[j][2];
        Real val = Real(1);
        for (int k = 0; k < a; ++k) val *= r;
        for (int k = 0; k < b; ++k) val *= s;
        for (int k = 0; k < c; ++k) val *= t;
        phi[j] = val;
    }
    for (int i = 0; i < 20; ++i) {
        Real v = Real(0);
        for (int j = 0; j < 20; ++j) {
            v += hex20_coeffs[j][i] * phi[j];
        }
        internal_vals[i] = v;
    }
}

void eval_hex20_grad_internal(Real r, Real s, Real t, Gradient* internal_grads) {
    Real dphi_dr[20], dphi_ds[20], dphi_dt[20];
    for (int j = 0; j < 20; ++j) {
        const int a = hex20_monomial_exponents[j][0];
        const int b = hex20_monomial_exponents[j][1];
        const int c = hex20_monomial_exponents[j][2];

        Real ra = Real(1), sb = Real(1), tc = Real(1);
        for (int k = 0; k < a; ++k) ra *= r;
        for (int k = 0; k < b; ++k) sb *= s;
        for (int k = 0; k < c; ++k) tc *= t;

        dphi_dr[j] = (a > 0) ? Real(a) * (a > 1 ? std::pow(r, a - 1) : Real(1)) * sb * tc : Real(0);
        dphi_ds[j] = (b > 0) ? ra * Real(b) * (b > 1 ? std::pow(s, b - 1) : Real(1)) * tc : Real(0);
        dphi_dt[j] = (c > 0) ? ra * sb * Real(c) * (c > 1 ? std::pow(t, c - 1) : Real(1)) : Real(0);
    }

    for (int i = 0; i < 20; ++i) {
        Real gr = Real(0), gs = Real(0), gt = Real(0);
        for (int j = 0; j < 20; ++j) {
            gr += hex20_coeffs[j][i] * dphi_dr[j];
            gs += hex20_coeffs[j][i] * dphi_ds[j];
            gt += hex20_coeffs[j][i] * dphi_dt[j];
        }
        internal_grads[i][0] = gr;
        internal_grads[i][1] = gs;
        internal_grads[i][2] = gt;
    }
}

void eval_hex20_hess_internal(Real r, Real s, Real t, Hessian* internal_hessians) {
    Real d2phi_drr[20], d2phi_dss[20], d2phi_dtt[20];
    Real d2phi_drs[20], d2phi_drt[20], d2phi_dst[20];
    for (int j = 0; j < 20; ++j) {
        const int a = hex20_monomial_exponents[j][0];
        const int b = hex20_monomial_exponents[j][1];
        const int c = hex20_monomial_exponents[j][2];

        const Real ra = (a > 0) ? std::pow(r, static_cast<Real>(a)) : Real(1);
        const Real sb = (b > 0) ? std::pow(s, static_cast<Real>(b)) : Real(1);
        const Real tc = (c > 0) ? std::pow(t, static_cast<Real>(c)) : Real(1);

        d2phi_drr[j] = (a > 1) ? Real(a * (a - 1)) * std::pow(r, static_cast<Real>(a - 2)) * sb * tc : Real(0);
        d2phi_dss[j] = (b > 1) ? ra * Real(b * (b - 1)) * std::pow(s, static_cast<Real>(b - 2)) * tc : Real(0);
        d2phi_dtt[j] = (c > 1) ? ra * sb * Real(c * (c - 1)) * std::pow(t, static_cast<Real>(c - 2)) : Real(0);
        d2phi_drs[j] = (a > 0 && b > 0) ? Real(a * b) *
            std::pow(r, static_cast<Real>(a - 1)) * std::pow(s, static_cast<Real>(b - 1)) * tc : Real(0);
        d2phi_drt[j] = (a > 0 && c > 0) ? Real(a * c) *
            std::pow(r, static_cast<Real>(a - 1)) * sb * std::pow(t, static_cast<Real>(c - 1)) : Real(0);
        d2phi_dst[j] = (b > 0 && c > 0) ? ra * Real(b * c) *
            std::pow(s, static_cast<Real>(b - 1)) * std::pow(t, static_cast<Real>(c - 1)) : Real(0);
    }

    for (int i = 0; i < 20; ++i) {
        Hessian H{};
        for (int j = 0; j < 20; ++j) {
            H(0, 0) += hex20_coeffs[j][i] * d2phi_drr[j];
            H(1, 1) += hex20_coeffs[j][i] * d2phi_dss[j];
            H(2, 2) += hex20_coeffs[j][i] * d2phi_dtt[j];
            H(0, 1) += hex20_coeffs[j][i] * d2phi_drs[j];
            H(0, 2) += hex20_coeffs[j][i] * d2phi_drt[j];
            H(1, 2) += hex20_coeffs[j][i] * d2phi_dst[j];
        }
        H(1, 0) = H(0, 1);
        H(2, 0) = H(0, 2);
        H(2, 1) = H(1, 2);
        internal_hessians[i] = H;
    }
}

} // namespace

SerendipityBasis::SerendipityBasis(ElementType type, int order, bool geometry_mode)
    : element_type_(type), dimension_(0), order_(order), size_(0), geometry_mode_(geometry_mode) {
    if (type == ElementType::Quad4 || type == ElementType::Quad8) {
        dimension_ = 2;
        if (order_ < 1) {
            order_ = 1;
        }
        if (type == ElementType::Quad8 && order_ != 2) {
            throw NotImplementedException("SerendipityBasis: Quad8 is only valid for quadratic order 2; use Quad4 for higher-order quadrilateral serendipity",
                                          __FILE__, __LINE__, __func__);
        }
        quad_monomial_exponents_ = quad_serendipity_exponents(order_);
        size_ = quad_monomial_exponents_.size();
        nodes_ = quad_serendipity_nodes(order_, size_);
        if (nodes_.size() != size_) {
            throw FEException("SerendipityBasis: quadrilateral serendipity setup produced inconsistent sizes",
                              __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
        }
        quad_inv_vandermonde_ = quad_serendipity_inverse_vandermonde(nodes_, quad_monomial_exponents_, order_);
    } else if (type == ElementType::Hex8 || type == ElementType::Hex20) {
        dimension_ = 3;
        if (order_ < 1) order_ = 1;
        if (order_ == 1) {
            size_ = 8;
        } else if (order_ == 2) {
            size_ = 20;
        } else {
            throw NotImplementedException("SerendipityBasis supports up to quadratic on hexahedra",
                                          __FILE__, __LINE__, __func__);
        }
    } else if (type == ElementType::Wedge15) {
        dimension_ = 3;
        if (order_ < 2) {
            order_ = 2;
        }
        if (order_ == 2) {
            size_ = 15;
        } else {
            throw NotImplementedException("SerendipityBasis supports up to quadratic on wedge15",
                                          __FILE__, __LINE__, __func__);
        }
    } else if (type == ElementType::Pyramid13) {
        dimension_ = 3;
        if (order_ < 2) {
            order_ = 2;
        }
        if (order_ == 2) {
            size_ = 13;
        } else {
            throw NotImplementedException("SerendipityBasis supports up to quadratic on pyramid13",
                                          __FILE__, __LINE__, __func__);
        }
    } else {
        throw BasisElementCompatibilityException("SerendipityBasis supports Quad4/Quad8, Hex8/Hex20, Wedge15, and Pyramid13 elements",
                                                 __FILE__, __LINE__, __func__);
    }

    if (nodes_.empty()) {
        nodes_.reserve(size_);
        for (std::size_t i = 0; i < size_; ++i) {
            nodes_.push_back(NodeOrdering::get_node_coords(element_type_, i));
        }
    }
}

void SerendipityBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                       std::vector<Real>& values) const {
    values.assign(size_, Real(0));
    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    if (dimension_ == 2) {
        if (quad_monomial_exponents_.size() != size_ ||
            quad_inv_vandermonde_.size() != size_ * size_) {
            throw FEException("SerendipityBasis: quadrilateral interpolation tables are not initialized",
                              __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
        }

        std::vector<Real> monomials(size_, Real(0));
        for (std::size_t j = 0; j < size_; ++j) {
            const auto [ax, ay] = quad_monomial_exponents_[j];
            monomials[j] = powi(x, ax) * powi(y, ay);
        }

        for (std::size_t i = 0; i < size_; ++i) {
            Real value = Real(0);
            for (std::size_t j = 0; j < size_; ++j) {
                value += monomials[j] * quad_inv_vandermonde_[j * size_ + i];
            }
            values[i] = value;
        }
        return;
    }

    if (dimension_ == 2 && order_ == 1) {
        values[0] = Real(0.25) * (Real(1) - x) * (Real(1) - y); // bottom-left
        values[1] = Real(0.25) * (Real(1) + x) * (Real(1) - y); // bottom-right
        values[2] = Real(0.25) * (Real(1) + x) * (Real(1) + y); // top-right
        values[3] = Real(0.25) * (Real(1) - x) * (Real(1) + y); // top-left
        return;
    }

    if (dimension_ == 3 && order_ == 1) {
        // Hex8 trilinear shape functions
        const Real r = x;
        const Real s = y;
        const Real t = z;
        values[0] = Real(0.125) * (Real(1) - r) * (Real(1) - s) * (Real(1) - t);
        values[1] = Real(0.125) * (Real(1) + r) * (Real(1) - s) * (Real(1) - t);
        values[2] = Real(0.125) * (Real(1) + r) * (Real(1) + s) * (Real(1) - t);
        values[3] = Real(0.125) * (Real(1) - r) * (Real(1) + s) * (Real(1) - t);
        values[4] = Real(0.125) * (Real(1) - r) * (Real(1) - s) * (Real(1) + t);
        values[5] = Real(0.125) * (Real(1) + r) * (Real(1) - s) * (Real(1) + t);
        values[6] = Real(0.125) * (Real(1) + r) * (Real(1) + s) * (Real(1) + t);
        values[7] = Real(0.125) * (Real(1) - r) * (Real(1) + s) * (Real(1) + t);
        return;
    }

    if (dimension_ == 2) {
        // Quadratic serendipity (8-node) on [-1,1]^2
        values[0] = Real(0.25) * (Real(1) - x) * (Real(1) - y) * (-x - y - Real(1));
        values[1] = Real(0.25) * (Real(1) + x) * (Real(1) - y) * ( x - y - Real(1));
        values[2] = Real(0.25) * (Real(1) + x) * (Real(1) + y) * ( x + y - Real(1));
        values[3] = Real(0.25) * (Real(1) - x) * (Real(1) + y) * (-x + y - Real(1));
        values[4] = Real(0.5)  * (Real(1) - x * x) * (Real(1) - y);       // mid-bottom
        values[5] = Real(0.5)  * (Real(1) + x) * (Real(1) - y * y);       // mid-right
        values[6] = Real(0.5)  * (Real(1) - x * x) * (Real(1) + y);       // mid-top
        values[7] = Real(0.5)  * (Real(1) - x) * (Real(1) - y * y);       // mid-left
        return;
    }

    const Real r = x;
    const Real s = y;
    const Real t = z;

    if (geometry_mode_ && element_type_ == ElementType::Hex20) {
        // Hex20 geometry mode: use trilinear Hex8 shape functions on corners, edges zero.
        values[0] = Real(0.125) * (Real(1) - r) * (Real(1) - s) * (Real(1) - t);
        values[1] = Real(0.125) * (Real(1) + r) * (Real(1) - s) * (Real(1) - t);
        values[2] = Real(0.125) * (Real(1) + r) * (Real(1) + s) * (Real(1) - t);
        values[3] = Real(0.125) * (Real(1) - r) * (Real(1) + s) * (Real(1) - t);
        values[4] = Real(0.125) * (Real(1) - r) * (Real(1) - s) * (Real(1) + t);
        values[5] = Real(0.125) * (Real(1) + r) * (Real(1) - s) * (Real(1) + t);
        values[6] = Real(0.125) * (Real(1) + r) * (Real(1) + s) * (Real(1) + t);
        values[7] = Real(0.125) * (Real(1) - r) * (Real(1) + s) * (Real(1) + t);
        for (std::size_t i = 8; i < 20; ++i) {
            values[i] = Real(0);
        }
        return;
    }

    if (element_type_ == ElementType::Hex20) {
        Real internal_vals[20];
        eval_hex20_internal(r, s, t, internal_vals);
        for (std::size_t i = 0; i < 20; ++i) {
            values[i] = internal_vals[kHex20NodeOrderingToInternal[i]];
        }
        return;
    }

    if (element_type_ == ElementType::Wedge15) {
        // Wedge15 serendipity-style field basis: polynomial expansion in monomials r^a s^b t^c
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

        static const Real wedge15_coeffs[15][15] = {
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

        Real phi[15];
        for (int j = 0; j < 15; ++j) {
            const int a = wedge15_monomial_exponents[j][0];
            const int b = wedge15_monomial_exponents[j][1];
            const int c = wedge15_monomial_exponents[j][2];
            Real val = Real(1);
            for (int k = 0; k < a; ++k) val *= r;
            for (int k = 0; k < b; ++k) val *= s;
            for (int k = 0; k < c; ++k) val *= t;
            phi[j] = val;
        }

        for (int i = 0; i < 15; ++i) {
            Real v = Real(0);
            for (int j = 0; j < 15; ++j) {
                v += wedge15_coeffs[j][i] * phi[j];
            }
            values[static_cast<std::size_t>(i)] = v;
        }
        return;
    }

    if (element_type_ == ElementType::Pyramid13) {
        static const LagrangeBasis parent(ElementType::Pyramid14, 2);
        std::vector<Real> parent_values;
        parent.evaluate_values(xi, parent_values);
        if (parent_values.size() != 14u) {
            throw FEException("SerendipityBasis: Pyramid14 parent evaluation size mismatch",
                              __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
        }
        for (std::size_t i = 0; i < 13; ++i) {
            values[i] = parent_values[i] + kPyramid13CenterRedistribution[i] * parent_values[13];
        }
        return;
    }
}

void SerendipityBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                          std::vector<Gradient>& gradients) const {
    gradients.assign(size_, Gradient{});

    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    if (dimension_ == 2) {
        if (quad_monomial_exponents_.size() != size_ ||
            quad_inv_vandermonde_.size() != size_ * size_) {
            throw FEException("SerendipityBasis: quadrilateral interpolation tables are not initialized",
                              __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
        }

        std::vector<Real> dmon_dx(size_, Real(0));
        std::vector<Real> dmon_dy(size_, Real(0));
        for (std::size_t j = 0; j < size_; ++j) {
            const auto [ax, ay] = quad_monomial_exponents_[j];
            dmon_dx[j] = (ax > 0) ? Real(ax) * powi(x, ax - 1) * powi(y, ay) : Real(0);
            dmon_dy[j] = (ay > 0) ? powi(x, ax) * Real(ay) * powi(y, ay - 1) : Real(0);
        }

        for (std::size_t i = 0; i < size_; ++i) {
            Real gx = Real(0);
            Real gy = Real(0);
            for (std::size_t j = 0; j < size_; ++j) {
                const Real coeff = quad_inv_vandermonde_[j * size_ + i];
                gx += dmon_dx[j] * coeff;
                gy += dmon_dy[j] * coeff;
            }
            gradients[i][0] = gx;
            gradients[i][1] = gy;
        }
        return;
    }

    // 2D linear quad (Quad4)
    if (dimension_ == 2 && order_ == 1) {
        // dN/dx
        gradients[0][0] = -Real(0.25) * (Real(1) - y);
        gradients[1][0] =  Real(0.25) * (Real(1) - y);
        gradients[2][0] =  Real(0.25) * (Real(1) + y);
        gradients[3][0] = -Real(0.25) * (Real(1) + y);
        // dN/dy
        gradients[0][1] = -Real(0.25) * (Real(1) - x);
        gradients[1][1] = -Real(0.25) * (Real(1) + x);
        gradients[2][1] =  Real(0.25) * (Real(1) + x);
        gradients[3][1] =  Real(0.25) * (Real(1) - x);
        return;
    }

    // 2D quadratic serendipity (Quad8)
    if (dimension_ == 2 && order_ == 2) {
        // Corner nodes
        gradients[0][0] = Real(0.25) * (Real(2) * x + y) * (Real(1) - y);
        gradients[0][1] = Real(0.25) * (Real(1) - x) * (Real(2) * y + x);

        gradients[1][0] = Real(0.25) * (Real(2) * x - y) * (Real(1) - y);
        gradients[1][1] = Real(0.25) * (Real(1) + x) * (Real(2) * y - x);

        gradients[2][0] = Real(0.25) * (Real(2) * x + y) * (Real(1) + y);
        gradients[2][1] = Real(0.25) * (Real(1) + x) * (Real(2) * y + x);

        gradients[3][0] = Real(0.25) * (Real(2) * x - y) * (Real(1) + y);
        gradients[3][1] = Real(0.25) * (Real(1) - x) * (Real(2) * y - x);

        // Mid-edge nodes
        gradients[4][0] = -x * (Real(1) - y);
        gradients[4][1] = -Real(0.5) * (Real(1) - x * x);

        gradients[5][0] =  Real(0.5) * (Real(1) - y * y);
        gradients[5][1] = -y * (Real(1) + x);

        gradients[6][0] = -x * (Real(1) + y);
        gradients[6][1] =  Real(0.5) * (Real(1) - x * x);

        gradients[7][0] = -Real(0.5) * (Real(1) - y * y);
        gradients[7][1] = -y * (Real(1) - x);
        return;
    }

    // 3D linear hex (Hex8)
    if (dimension_ == 3 && order_ == 1) {
        const Real r = x, s = y, t = z;
        gradients[0][0] = -Real(0.125) * (Real(1) - s) * (Real(1) - t);
        gradients[0][1] = -Real(0.125) * (Real(1) - r) * (Real(1) - t);
        gradients[0][2] = -Real(0.125) * (Real(1) - r) * (Real(1) - s);

        gradients[1][0] =  Real(0.125) * (Real(1) - s) * (Real(1) - t);
        gradients[1][1] = -Real(0.125) * (Real(1) + r) * (Real(1) - t);
        gradients[1][2] = -Real(0.125) * (Real(1) + r) * (Real(1) - s);

        gradients[2][0] =  Real(0.125) * (Real(1) + s) * (Real(1) - t);
        gradients[2][1] =  Real(0.125) * (Real(1) + r) * (Real(1) - t);
        gradients[2][2] = -Real(0.125) * (Real(1) + r) * (Real(1) + s);

        gradients[3][0] = -Real(0.125) * (Real(1) + s) * (Real(1) - t);
        gradients[3][1] =  Real(0.125) * (Real(1) - r) * (Real(1) - t);
        gradients[3][2] = -Real(0.125) * (Real(1) - r) * (Real(1) + s);

        gradients[4][0] = -Real(0.125) * (Real(1) - s) * (Real(1) + t);
        gradients[4][1] = -Real(0.125) * (Real(1) - r) * (Real(1) + t);
        gradients[4][2] =  Real(0.125) * (Real(1) - r) * (Real(1) - s);

        gradients[5][0] =  Real(0.125) * (Real(1) - s) * (Real(1) + t);
        gradients[5][1] = -Real(0.125) * (Real(1) + r) * (Real(1) + t);
        gradients[5][2] =  Real(0.125) * (Real(1) + r) * (Real(1) - s);

        gradients[6][0] =  Real(0.125) * (Real(1) + s) * (Real(1) + t);
        gradients[6][1] =  Real(0.125) * (Real(1) + r) * (Real(1) + t);
        gradients[6][2] =  Real(0.125) * (Real(1) + r) * (Real(1) + s);

        gradients[7][0] = -Real(0.125) * (Real(1) + s) * (Real(1) + t);
        gradients[7][1] =  Real(0.125) * (Real(1) - r) * (Real(1) + t);
        gradients[7][2] =  Real(0.125) * (Real(1) - r) * (Real(1) + s);
        return;
    }

    // Hex20 geometry mode: use Hex8 gradients
    if (dimension_ == 3 && order_ == 2 && geometry_mode_ &&
        (element_type_ == ElementType::Hex20 || element_type_ == ElementType::Quad8)) {
        const Real r = x, s = y, t = z;
        gradients[0][0] = -Real(0.125) * (Real(1) - s) * (Real(1) - t);
        gradients[0][1] = -Real(0.125) * (Real(1) - r) * (Real(1) - t);
        gradients[0][2] = -Real(0.125) * (Real(1) - r) * (Real(1) - s);

        gradients[1][0] =  Real(0.125) * (Real(1) - s) * (Real(1) - t);
        gradients[1][1] = -Real(0.125) * (Real(1) + r) * (Real(1) - t);
        gradients[1][2] = -Real(0.125) * (Real(1) + r) * (Real(1) - s);

        gradients[2][0] =  Real(0.125) * (Real(1) + s) * (Real(1) - t);
        gradients[2][1] =  Real(0.125) * (Real(1) + r) * (Real(1) - t);
        gradients[2][2] = -Real(0.125) * (Real(1) + r) * (Real(1) + s);

        gradients[3][0] = -Real(0.125) * (Real(1) + s) * (Real(1) - t);
        gradients[3][1] =  Real(0.125) * (Real(1) - r) * (Real(1) - t);
        gradients[3][2] = -Real(0.125) * (Real(1) - r) * (Real(1) + s);

        gradients[4][0] = -Real(0.125) * (Real(1) - s) * (Real(1) + t);
        gradients[4][1] = -Real(0.125) * (Real(1) - r) * (Real(1) + t);
        gradients[4][2] =  Real(0.125) * (Real(1) - r) * (Real(1) - s);

        gradients[5][0] =  Real(0.125) * (Real(1) - s) * (Real(1) + t);
        gradients[5][1] = -Real(0.125) * (Real(1) + r) * (Real(1) + t);
        gradients[5][2] =  Real(0.125) * (Real(1) + r) * (Real(1) - s);

        gradients[6][0] =  Real(0.125) * (Real(1) + s) * (Real(1) + t);
        gradients[6][1] =  Real(0.125) * (Real(1) + r) * (Real(1) + t);
        gradients[6][2] =  Real(0.125) * (Real(1) + r) * (Real(1) + s);

        gradients[7][0] = -Real(0.125) * (Real(1) + s) * (Real(1) + t);
        gradients[7][1] =  Real(0.125) * (Real(1) - r) * (Real(1) + t);
        gradients[7][2] =  Real(0.125) * (Real(1) - r) * (Real(1) + s);
        // Edge-node gradients remain zero
        return;
    }

    // Hex20 analytical gradients using monomial differentiation
    if (element_type_ == ElementType::Hex20 && order_ == 2) {
        const Real r = x, s = y, t = z;
        Gradient internal_grads[20];
        eval_hex20_grad_internal(r, s, t, internal_grads);
        for (std::size_t i = 0; i < 20; ++i) {
            gradients[i] = internal_grads[kHex20NodeOrderingToInternal[i]];
        }
        return;
    }

    // Wedge15 analytical gradients using monomial differentiation
    if (element_type_ == ElementType::Wedge15 && order_ == 2) {
        const Real r = x, s = y, t = z;

        // Monomial exponents for Wedge15 polynomial basis
        static const int wedge15_monomial_exponents[15][3] = {
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1},
            {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {1, 0, 0}, {1, 0, 1},
            {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {2, 0, 0}, {2, 0, 1}
        };

        // Coefficients matrix (same as in evaluate_values)
        static const Real wedge15_coeffs[15][15] = {
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

        // Compute monomial derivatives
        Real dphi_dr[15], dphi_ds[15], dphi_dt[15];
        for (int j = 0; j < 15; ++j) {
            const int a = wedge15_monomial_exponents[j][0];
            const int b = wedge15_monomial_exponents[j][1];
            const int c = wedge15_monomial_exponents[j][2];

            // Compute r^a, s^b, t^c
            Real ra = Real(1), sb = Real(1), tc = Real(1);
            for (int k = 0; k < a; ++k) ra *= r;
            for (int k = 0; k < b; ++k) sb *= s;
            for (int k = 0; k < c; ++k) tc *= t;

            // Derivatives
            dphi_dr[j] = (a > 0) ? Real(a) * (a > 1 ? std::pow(r, a - 1) : Real(1)) * sb * tc : Real(0);
            dphi_ds[j] = (b > 0) ? ra * Real(b) * (b > 1 ? std::pow(s, b - 1) : Real(1)) * tc : Real(0);
            dphi_dt[j] = (c > 0) ? ra * sb * Real(c) * (c > 1 ? std::pow(t, c - 1) : Real(1)) : Real(0);
        }

        // Compute gradients as linear combinations
        for (int i = 0; i < 15; ++i) {
            Real gr = Real(0), gs = Real(0), gt = Real(0);
            for (int j = 0; j < 15; ++j) {
                gr += wedge15_coeffs[j][i] * dphi_dr[j];
                gs += wedge15_coeffs[j][i] * dphi_ds[j];
                gt += wedge15_coeffs[j][i] * dphi_dt[j];
            }
            gradients[static_cast<std::size_t>(i)][0] = gr;
            gradients[static_cast<std::size_t>(i)][1] = gs;
            gradients[static_cast<std::size_t>(i)][2] = gt;
        }
        return;
    }

    if (element_type_ == ElementType::Pyramid13) {
        static const LagrangeBasis parent(ElementType::Pyramid14, 2);
        std::vector<Gradient> parent_gradients;
        // Pyramid13 inherits the complete-family pyramid apex contract from the
        // parent basis rather than introducing a separate regularized path.
        parent.evaluate_gradients(xi, parent_gradients);
        if (parent_gradients.size() != 14u) {
            throw FEException("SerendipityBasis: Pyramid14 parent gradient size mismatch",
                              __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
        }
        for (std::size_t i = 0; i < 13; ++i) {
            gradients[i] = parent_gradients[i];
            gradients[i][0] += kPyramid13CenterRedistribution[i] * parent_gradients[13][0];
            gradients[i][1] += kPyramid13CenterRedistribution[i] * parent_gradients[13][1];
            gradients[i][2] += kPyramid13CenterRedistribution[i] * parent_gradients[13][2];
        }
        return;
    }

    throw BasisEvaluationException("SerendipityBasis::evaluate_gradients: unsupported serendipity configuration",
                                   __FILE__, __LINE__, __func__);
}

void SerendipityBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                         std::vector<Hessian>& hessians) const {
    hessians.assign(size_, Hessian{});
    const Real x = xi[0];
    const Real y = xi[1];
    const Real z = xi[2];

    if (dimension_ == 2) {
        if (quad_monomial_exponents_.size() != size_ ||
            quad_inv_vandermonde_.size() != size_ * size_) {
            throw FEException("SerendipityBasis: quadrilateral interpolation tables are not initialized",
                              __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
        }

        std::vector<Real> dxx(size_, Real(0));
        std::vector<Real> dxy(size_, Real(0));
        std::vector<Real> dyy(size_, Real(0));
        for (std::size_t j = 0; j < size_; ++j) {
            const auto [ax, ay] = quad_monomial_exponents_[j];
            dxx[j] = (ax > 1) ? Real(ax * (ax - 1)) * powi(x, ax - 2) * powi(y, ay) : Real(0);
            dxy[j] = (ax > 0 && ay > 0) ? Real(ax * ay) * powi(x, ax - 1) * powi(y, ay - 1) : Real(0);
            dyy[j] = (ay > 1) ? Real(ay * (ay - 1)) * powi(x, ax) * powi(y, ay - 2) : Real(0);
        }

        for (std::size_t i = 0; i < size_; ++i) {
            for (std::size_t j = 0; j < size_; ++j) {
                const Real coeff = quad_inv_vandermonde_[j * size_ + i];
                hessians[i](0, 0) += dxx[j] * coeff;
                hessians[i](0, 1) += dxy[j] * coeff;
                hessians[i](1, 1) += dyy[j] * coeff;
            }
            hessians[i](1, 0) = hessians[i](0, 1);
        }
        return;
    }

    if (element_type_ == ElementType::Hex8 && order_ == 1) {
        static const LagrangeBasis parent(ElementType::Hex8, 1);
        parent.evaluate_hessians(xi, hessians);
        return;
    }

    if (geometry_mode_ && element_type_ == ElementType::Hex20) {
        static const LagrangeBasis parent(ElementType::Hex8, 1);
        std::vector<Hessian> parent_hessians;
        parent.evaluate_hessians(xi, parent_hessians);
        for (std::size_t i = 0; i < 8; ++i) {
            hessians[i] = parent_hessians[i];
        }
        return;
    }

    if (element_type_ == ElementType::Hex20 && order_ == 2) {
        Hessian internal_hessians[20];
        eval_hex20_hess_internal(x, y, z, internal_hessians);
        for (std::size_t i = 0; i < 20; ++i) {
            hessians[i] = internal_hessians[kHex20NodeOrderingToInternal[i]];
        }
        return;
    }

    if (element_type_ == ElementType::Wedge15 && order_ == 2) {
        static const int wedge15_monomial_exponents[15][3] = {
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1},
            {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {1, 0, 0}, {1, 0, 1},
            {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {2, 0, 0}, {2, 0, 1}
        };
        static const Real wedge15_coeffs[15][15] = {
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

        Real drr[15], dss[15], dtt[15], drs[15], drt[15], dst[15];
        for (int j = 0; j < 15; ++j) {
            const int a = wedge15_monomial_exponents[j][0];
            const int b = wedge15_monomial_exponents[j][1];
            const int c = wedge15_monomial_exponents[j][2];
            drr[j] = (a > 1) ? Real(a * (a - 1)) * std::pow(x, static_cast<Real>(a - 2)) * std::pow(y, static_cast<Real>(b)) * std::pow(z, static_cast<Real>(c)) : Real(0);
            dss[j] = (b > 1) ? std::pow(x, static_cast<Real>(a)) * Real(b * (b - 1)) * std::pow(y, static_cast<Real>(b - 2)) * std::pow(z, static_cast<Real>(c)) : Real(0);
            dtt[j] = (c > 1) ? std::pow(x, static_cast<Real>(a)) * std::pow(y, static_cast<Real>(b)) * Real(c * (c - 1)) * std::pow(z, static_cast<Real>(c - 2)) : Real(0);
            drs[j] = (a > 0 && b > 0) ? Real(a * b) * std::pow(x, static_cast<Real>(a - 1)) * std::pow(y, static_cast<Real>(b - 1)) * std::pow(z, static_cast<Real>(c)) : Real(0);
            drt[j] = (a > 0 && c > 0) ? Real(a * c) * std::pow(x, static_cast<Real>(a - 1)) * std::pow(y, static_cast<Real>(b)) * std::pow(z, static_cast<Real>(c - 1)) : Real(0);
            dst[j] = (b > 0 && c > 0) ? std::pow(x, static_cast<Real>(a)) * Real(b * c) * std::pow(y, static_cast<Real>(b - 1)) * std::pow(z, static_cast<Real>(c - 1)) : Real(0);
        }

        for (int i = 0; i < 15; ++i) {
            Hessian H{};
            for (int j = 0; j < 15; ++j) {
                H(0, 0) += wedge15_coeffs[j][i] * drr[j];
                H(1, 1) += wedge15_coeffs[j][i] * dss[j];
                H(2, 2) += wedge15_coeffs[j][i] * dtt[j];
                H(0, 1) += wedge15_coeffs[j][i] * drs[j];
                H(0, 2) += wedge15_coeffs[j][i] * drt[j];
                H(1, 2) += wedge15_coeffs[j][i] * dst[j];
            }
            H(1, 0) = H(0, 1);
            H(2, 0) = H(0, 2);
            H(2, 1) = H(1, 2);
            hessians[static_cast<std::size_t>(i)] = H;
        }
        return;
    }

    if (element_type_ == ElementType::Pyramid13) {
        static const LagrangeBasis parent(ElementType::Pyramid14, 2);
        std::vector<Hessian> parent_hessians;
        // Pyramid13 inherits the complete-family pyramid apex contract from the
        // parent basis rather than introducing a separate regularized path.
        parent.evaluate_hessians(xi, parent_hessians);
        if (parent_hessians.size() != 14u) {
            throw FEException("SerendipityBasis: Pyramid14 parent Hessian size mismatch",
                              __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
        }
        for (std::size_t i = 0; i < 13; ++i) {
            hessians[i] = parent_hessians[i];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    const std::size_t sr = static_cast<std::size_t>(r);
                    const std::size_t sc = static_cast<std::size_t>(c);
                    hessians[i](sr, sc) +=
                        kPyramid13CenterRedistribution[i] * parent_hessians[13](sr, sc);
                }
            }
        }
        return;
    }

    throw BasisEvaluationException("SerendipityBasis::evaluate_hessians: unsupported serendipity configuration",
                                   __FILE__, __LINE__, __func__);
}

} // namespace basis
} // namespace FE
} // namespace svmp
