/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SerendipityBasis.h"
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
        // Collapsed-coordinate serendipity basis (conforming).
        // Treats Pyramid13 as a Hex20 with the top face collapsed to the apex.
        // Map pyramid (x,y,z) in [-1,1]x[-1,1]x[0,1] to hex (u,v,w) in [-1,1]^3.
        const Real eps = Real(1e-12);
        const Real one_minus_z = Real(1) - z;

        if (std::abs(one_minus_z) < eps) {
            // Apex node (4) is 1, all others 0
            for (std::size_t i = 0; i < 13; ++i) values[i] = Real(0);
            values[4] = Real(1);
            return;
        }

        // Coordinate transformation
        const Real w = Real(2) * z - Real(1);
        const Real inv_omz = Real(1) / one_minus_z;
        const Real u = x * inv_omz;
        const Real v = y * inv_omz;

        Real internal_vals[20];
        eval_hex20_internal(u, v, w, internal_vals);

        // Get Hex20 values in NodeOrdering
        Real hex_vals[20];
        for (std::size_t i = 0; i < 20; ++i) {
            hex_vals[i] = internal_vals[kHex20NodeOrderingToInternal[i]];
        }

        // Map Hex20 nodes to Pyramid13 nodes (using collapsed Hex20 is an appropriate and deliberate choice)
        // Hex20 NodeOrdering:
        // 0-3: Bottom corners
        // 4-7: Top corners -> Apex
        // 8-11: Bottom edges
        // 12-15: Top edges -> Apex
        // 16-19: Vertical edges

        // Pyramid13 NodeOrdering:
        // 0-3: Base corners
        // 4: Apex
        // 5-8: Base edges
        // 9-12: Rising edges

        // Base corners
        values[0] = hex_vals[0];
        values[1] = hex_vals[1];
        values[2] = hex_vals[2];
        values[3] = hex_vals[3];

        // Apex: sum of all top face nodes
        values[4] = hex_vals[4] + hex_vals[5] + hex_vals[6] + hex_vals[7] +
                    hex_vals[12] + hex_vals[13] + hex_vals[14] + hex_vals[15];

        // Base edges
        values[5] = hex_vals[8];
        values[6] = hex_vals[9];
        values[7] = hex_vals[10];
        values[8] = hex_vals[11];

        // Rising edges
        values[9]  = hex_vals[16];
        values[10] = hex_vals[17];
        values[11] = hex_vals[18];
        values[12] = hex_vals[19];

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
        // Collapsed-coordinate serendipity basis (conforming).
        // Treats Pyramid13 as a Hex20 with the top face collapsed to the apex.
        // Map pyramid (x,y,z) in [-1,1]x[-1,1]x[0,1] to hex (u,v,w) in [-1,1]^3.
        const Real eps = Real(1e-12);
        const Real one_minus_z = Real(1) - z;

        if (std::abs(one_minus_z) < eps) {
            // Apex node has undefined gradient via this map, use numerical fallback
            BasisFunction::numerical_gradient(xi, gradients);
            return;
        }

        // Coordinate transformation
        const Real w = Real(2) * z - Real(1);
        const Real inv_omz = Real(1) / one_minus_z;
        const Real u = x * inv_omz;
        const Real v = y * inv_omz;

        Gradient internal_grads[20];
        eval_hex20_grad_internal(u, v, w, internal_grads);

        // Chain rule
        // du/dx = inv_omz
        // du/dz = x * inv_omz^2 = u * inv_omz
        // dv/dy = inv_omz
        // dv/dz = y * inv_omz^2 = v * inv_omz
        // dw/dz = 2

        Gradient hex_physical_grads[20];
        for(int i = 0; i < 20; ++i) {
            Real dN_du = internal_grads[i][0];
            Real dN_dv = internal_grads[i][1];
            Real dN_dw = internal_grads[i][2];

            hex_physical_grads[i][0] = dN_du * inv_omz;
            hex_physical_grads[i][1] = dN_dv * inv_omz;
            hex_physical_grads[i][2] = dN_du * u * inv_omz + dN_dv * v * inv_omz + dN_dw * Real(2);
        }

        // Reorder Hex20 internal to NodeOrdering
        Gradient hex_vtk[20];
        for (std::size_t i = 0; i < 20; ++i) {
            hex_vtk[i] = hex_physical_grads[kHex20NodeOrderingToInternal[i]];
        }

        // Map to Pyramid13 nodes (using collapsed Hex20 is an appropriate and deliberate choice)
        // Base corners
        gradients[0] = hex_vtk[0];
        gradients[1] = hex_vtk[1];
        gradients[2] = hex_vtk[2];
        gradients[3] = hex_vtk[3];

        // Apex (sum of top nodes)
        gradients[4][0] = hex_vtk[4][0] + hex_vtk[5][0] + hex_vtk[6][0] + hex_vtk[7][0] +
                          hex_vtk[12][0] + hex_vtk[13][0] + hex_vtk[14][0] + hex_vtk[15][0];
        gradients[4][1] = hex_vtk[4][1] + hex_vtk[5][1] + hex_vtk[6][1] + hex_vtk[7][1] +
                          hex_vtk[12][1] + hex_vtk[13][1] + hex_vtk[14][1] + hex_vtk[15][1];
        gradients[4][2] = hex_vtk[4][2] + hex_vtk[5][2] + hex_vtk[6][2] + hex_vtk[7][2] +
                          hex_vtk[12][2] + hex_vtk[13][2] + hex_vtk[14][2] + hex_vtk[15][2];

        // Base edges
        gradients[5] = hex_vtk[8];
        gradients[6] = hex_vtk[9];
        gradients[7] = hex_vtk[10];
        gradients[8] = hex_vtk[11];

        // Rising edges
        gradients[9]  = hex_vtk[16];
        gradients[10] = hex_vtk[17];
        gradients[11] = hex_vtk[18];
        gradients[12] = hex_vtk[19];

        return;
    }

    // Fall back to numerical gradients for other cases
    BasisFunction::numerical_gradient(xi, gradients);
}

} // namespace basis
} // namespace FE
} // namespace svmp
