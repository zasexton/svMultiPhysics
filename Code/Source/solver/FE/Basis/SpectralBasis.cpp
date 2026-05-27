/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SpectralBasis.h"
#include "BasisTraits.h"
#include "BasisTolerance.h"
#include "NodeOrderingConventions.h"
#include "OrthogonalPolynomials.h"
#include "PyramidModalBasis.h"
#include "Math/DenseLinearAlgebra.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace basis {

namespace detail {

struct SpectralModalMatrixData {
    // Row-major [basis, modal] transform used by independent derivative calls.
    std::vector<Real> modal_to_nodal_by_basis;
    // Row-major [modal, basis] transform used by fused accumulation.
    std::vector<Real> modal_to_nodal_by_modal;
};

} // namespace detail

SpectralBasis::~SpectralBasis() = default;
SpectralBasis::SpectralBasis(const SpectralBasis&) = default;
SpectralBasis::SpectralBasis(SpectralBasis&&) = default;
SpectralBasis& SpectralBasis::operator=(const SpectralBasis&) = default;
SpectralBasis& SpectralBasis::operator=(SpectralBasis&&) = default;

namespace {

struct SpectralModalMatrixCacheKey {
    ElementType element_type{ElementType::Unknown};
    int order{0};

    bool operator==(const SpectralModalMatrixCacheKey& other) const noexcept {
        return element_type == other.element_type && order == other.order;
    }
};

struct SpectralModalMatrixCacheKeyHash {
    std::size_t operator()(const SpectralModalMatrixCacheKey& key) const noexcept {
        std::size_t seed = static_cast<std::size_t>(key.element_type);
        seed ^= static_cast<std::size_t>(key.order) + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
        return seed;
    }
};

struct SpectralScratch {
    std::vector<Real> lx;
    std::vector<Real> dx;
    std::vector<Real> ddx;
    std::vector<Real> ly;
    std::vector<Real> dy;
    std::vector<Real> ddy;
    std::vector<Real> lz;
    std::vector<Real> dz;
    std::vector<Real> ddz;
    std::vector<Real> modal_values;
    std::vector<Real> modal_gradient_components;
    std::vector<Real> modal_hessian_components;
    std::vector<Real> nodal_gradient_components;
    std::vector<Real> nodal_hessian_components;
    std::vector<Gradient> modal_gradients;
    std::vector<Hessian> modal_hessians;
    std::vector<Real> face_values;
    std::vector<Real> face_gradients;
    std::vector<Real> face_hessians;
    std::vector<Real> axis_values;
    std::vector<Real> axis_gradients;
    std::vector<Real> axis_hessians;
    std::vector<Real> axis_temp_values;
    std::vector<Real> axis_temp_gradients;
    std::vector<Real> axis_temp_hessians;
    std::vector<math::Vector<Real, 3>> face_points;
    std::vector<math::Vector<Real, 3>> axis_points;
    pyramid_modal::EvaluationPoint pyramid_point;
};

SpectralScratch& spectral_scratch() {
    static thread_local SpectralScratch scratch;
    return scratch;
}

bool points_close(const math::Vector<Real, 3>& a,
                  const math::Vector<Real, 3>& b,
                  Real tol = Real(1e-8)) {
    return std::abs(a[0] - b[0]) <= tol &&
           std::abs(a[1] - b[1]) <= tol &&
           std::abs(a[2] - b[2]) <= tol;
}

bool is_on_pyramid_boundary(const math::Vector<Real, 3>& xi,
                            Real tol = detail::basis_scaled_tolerance()) {
    if (std::abs(xi[2]) <= tol || std::abs(xi[2] - Real(1)) <= tol) {
        return true;
    }
    const Real scale = Real(1) - xi[2];
    return std::abs(std::abs(xi[0]) - scale) <= tol ||
           std::abs(std::abs(xi[1]) - scale) <= tol;
}

Real simplex_node_tolerance(int order) {
    return std::max(Real(1e-10),
                    Real(1024) * std::numeric_limits<Real>::epsilon() *
                        static_cast<Real>((order + 1) * (order + 1)));
}

void validate_triangle_node(const math::Vector<Real, 3>& node,
                            int order,
                            std::size_t index) {
    const Real tol = simplex_node_tolerance(order);
    const Real sum = node[0] + node[1];
    if (node[0] < -tol || node[1] < -tol || std::abs(node[2]) > tol ||
        sum > Real(1) + tol) {
        throw BasisConstructionException(
            "SpectralBasis: generated triangle node " + std::to_string(index) +
            " leaves the reference triangle for order " + std::to_string(order));
    }
}

void validate_tetrahedron_node(const math::Vector<Real, 3>& node,
                               int order,
                               std::size_t index) {
    const Real tol = simplex_node_tolerance(order);
    const Real sum = node[0] + node[1] + node[2];
    if (node[0] < -tol || node[1] < -tol || node[2] < -tol ||
        sum > Real(1) + tol) {
        throw BasisConstructionException(
            "SpectralBasis: generated tetrahedron node " + std::to_string(index) +
            " leaves the reference tetrahedron for order " + std::to_string(order));
    }
}

std::vector<std::vector<Real>> build_gll_zero_one_table(int max_order) {
    std::vector<std::vector<Real>> table(static_cast<std::size_t>(max_order + 1));
    table[0] = {Real(0)};
    for (int order = 1; order <= max_order; ++order) {
        const auto gll = orthopoly::gll_nodes(order + 1);
        auto& nodes = table[static_cast<std::size_t>(order)];
        nodes.resize(static_cast<std::size_t>(order + 1));
        for (int i = 0; i <= order; ++i) {
            nodes[static_cast<std::size_t>(i)] =
                Real(0.5) * (gll[static_cast<std::size_t>(i)] + Real(1));
        }
        nodes.front() = Real(0);
        nodes.back() = Real(1);
    }
    return table;
}

math::Vector<Real, 3> map_triangle_to_pyramid_front(Real a, Real b) {
    return {b - a, -a - b, Real(1) - a - b};
}

math::Vector<Real, 3> map_triangle_to_pyramid_right(Real a, Real b) {
    return {a + b, b - a, Real(1) - a - b};
}

math::Vector<Real, 3> map_triangle_to_pyramid_back(Real a, Real b) {
    return {a - b, a + b, Real(1) - a - b};
}

math::Vector<Real, 3> map_triangle_to_pyramid_left(Real a, Real b) {
    return {-a - b, a - b, Real(1) - a - b};
}

void append_unique_point(std::vector<math::Vector<Real, 3>>& points,
                         const math::Vector<Real, 3>& candidate,
                         Real tol = Real(1e-8)) {
    const auto it = std::find_if(points.begin(), points.end(),
                                 [&](const math::Vector<Real, 3>& existing) {
                                     return points_close(existing, candidate, tol);
                                 });
    if (it == points.end()) {
        points.push_back(candidate);
    }
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
    } else if (is_wedge(element_type_)) {
        dimension_ = 3;
        is_wedge_tensor_product_ = true;
        face_basis_ = std::make_shared<SpectralBasis>(ElementType::Triangle3, order_);
        axis_basis_ = std::make_shared<SpectralBasis>(ElementType::Line2, order_);
        nodes_1d_ = axis_basis_->nodes_1d();
        size_ = face_basis_->size() * axis_basis_->size();
        simplex_nodes_.reserve(size_);
        for (Real z : nodes_1d_) {
            for (const auto& face_node : face_basis_->simplex_nodes_) {
                simplex_nodes_.push_back({face_node[0], face_node[1], z});
            }
        }
    } else if (is_pyramid(element_type_)) {
        dimension_ = 3;
        is_pyramid_modal_ = true;
        size_ = static_cast<std::size_t>((order_ + 1) * (order_ + 2) * (2 * order_ + 3) / 6);
        build_pyramid_nodes();
        build_inverse_vandermonde();
    } else {
        throw BasisElementCompatibilityException("SpectralBasis: unsupported element type",
                                                 __FILE__, __LINE__, __func__);
    }
}

void SpectralBasis::build_nodes() {
    nodes_1d_ = orthopoly::gll_nodes(order_ + 1);
    barycentric_weights_.assign(nodes_1d_.size(), Real(1));
    for (std::size_t i = 0; i < nodes_1d_.size(); ++i) {
        Real prod = Real(1);
        for (std::size_t j = 0; j < nodes_1d_.size(); ++j) {
            if (i == j) continue;
            prod *= nodes_1d_[i] - nodes_1d_[j];
        }
        barycentric_weights_[i] = Real(1) / prod;
    }
    build_1d_derivative_matrices();
}

void SpectralBasis::build_1d_derivative_matrices() {
    const std::size_t n = nodes_1d_.size();
    derivative_matrix_1d_.assign(n * n, Real(0));
    second_derivative_matrix_1d_.assign(n * n, Real(0));

    for (std::size_t row = 0; row < n; ++row) {
        Real diagonal = Real(0);
        for (std::size_t col = 0; col < n; ++col) {
            if (row == col) {
                continue;
            }
            const Real entry =
                barycentric_weights_[col] /
                (barycentric_weights_[row] * (nodes_1d_[row] - nodes_1d_[col]));
            derivative_matrix_1d_[row * n + col] = entry;
            diagonal -= entry;
        }
        derivative_matrix_1d_[row * n + row] = diagonal;
    }

    for (std::size_t row = 0; row < n; ++row) {
        Real diagonal = Real(0);
        const Real first_diagonal = derivative_matrix_1d_[row * n + row];
        for (std::size_t col = 0; col < n; ++col) {
            if (row == col) {
                continue;
            }
            const Real inv_delta = Real(1) / (nodes_1d_[row] - nodes_1d_[col]);
            const Real entry =
                Real(2) * derivative_matrix_1d_[row * n + col] *
                (first_diagonal - inv_delta);
            second_derivative_matrix_1d_[row * n + col] = entry;
            diagonal -= entry;
        }
        second_derivative_matrix_1d_[row * n + row] = diagonal;
    }
}

void SpectralBasis::eval_1d_to(Real x, std::span<Real> values) const {
    const std::size_t n = nodes_1d_.size();
    BASIS_CHECK_EVAL(values.size() == n, "SpectralBasis::eval_1d_to size mismatch");
    std::fill(values.begin(), values.end(), Real(0));

    const Real tol = std::numeric_limits<Real>::epsilon() * Real(128) *
                     std::max(Real(1), std::abs(x));

    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(x - nodes_1d_[i]) <= tol) [[unlikely]] {
            values[i] = Real(1);
            return;
        }
    }

    Real denom = Real(0);
    for (std::size_t i = 0; i < n; ++i) {
        const Real term = barycentric_weights_[i] / (x - nodes_1d_[i]);
        values[i] = term;
        denom += term;
    }

    const Real inv_denom = Real(1) / denom;
    for (Real& value : values) {
        value *= inv_denom;
    }
}

void SpectralBasis::eval_1d_derivative_to(Real x, std::span<Real> derivatives) const {
    const std::size_t n = nodes_1d_.size();
    BASIS_CHECK_EVAL(derivatives.size() == n,
                     "SpectralBasis::eval_1d_derivative_to size mismatch");
    std::fill(derivatives.begin(), derivatives.end(), Real(0));

    const Real tol = std::numeric_limits<Real>::epsilon() * Real(128) *
                     std::max(Real(1), std::abs(x));

    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(x - nodes_1d_[i]) <= tol) [[unlikely]] {
            for (std::size_t j = 0; j < n; ++j) {
                derivatives[j] = derivative_matrix_1d_[i * n + j];
            }
            return;
        }
    }

    Real S0 = Real(0);
    Real S1 = Real(0);
    for (std::size_t i = 0; i < n; ++i) {
        const Real dx = x - nodes_1d_[i];
        const Real a = barycentric_weights_[i] / dx;
        derivatives[i] = a;
        S0 += a;
        S1 -= a / dx;
    }

    const Real inv_S0 = Real(1) / S0;
    const Real S1_over_S0 = S1 * inv_S0;
    for (std::size_t i = 0; i < n; ++i) {
        const Real dx = x - nodes_1d_[i];
        const Real value = derivatives[i] * inv_S0;
        derivatives[i] = value * (-Real(1) / dx - S1_over_S0);
    }
}

void SpectralBasis::eval_1d_second_derivative_to(Real x,
                                                 std::span<Real> second_derivatives) const {
    const std::size_t n = nodes_1d_.size();
    BASIS_CHECK_EVAL(second_derivatives.size() == n,
                     "SpectralBasis::eval_1d_second_derivative_to size mismatch");
    std::fill(second_derivatives.begin(), second_derivatives.end(), Real(0));

    const Real tol = std::numeric_limits<Real>::epsilon() * Real(128) *
                     std::max(Real(1), std::abs(x));

    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(x - nodes_1d_[i]) <= tol) [[unlikely]] {
            for (std::size_t j = 0; j < n; ++j) {
                second_derivatives[j] = second_derivative_matrix_1d_[i * n + j];
            }
            return;
        }
    }

    Real S0 = Real(0);
    Real S1 = Real(0);
    Real S2 = Real(0);
    for (std::size_t i = 0; i < n; ++i) {
        const Real dx = x - nodes_1d_[i];
        const Real a = barycentric_weights_[i] / dx;
        second_derivatives[i] = a;
        S0 += a;
        S1 -= a / dx;
        S2 += Real(2) * a / (dx * dx);
    }

    const Real inv_S0 = Real(1) / S0;
    const Real S1_over_S0 = S1 * inv_S0;
    const Real denom_log_second = (S2 * S0 - S1 * S1) * inv_S0 * inv_S0;
    for (std::size_t i = 0; i < n; ++i) {
        const Real dx = x - nodes_1d_[i];
        const Real value = second_derivatives[i] * inv_S0;
        const Real basis_log_derivative = -Real(1) / dx - S1_over_S0;
        const Real basis_log_second = Real(1) / (dx * dx) - denom_log_second;
        second_derivatives[i] =
            value * (basis_log_derivative * basis_log_derivative + basis_log_second);
    }
}

void SpectralBasis::eval_1d_all_to(Real x,
                                   std::span<Real> values,
                                   std::span<Real> derivatives,
                                   std::span<Real> second_derivatives) const {
    const std::size_t n = nodes_1d_.size();
    BASIS_CHECK_EVAL(values.size() == n && derivatives.size() == n &&
                         second_derivatives.size() == n,
                     "SpectralBasis::eval_1d_all_to size mismatch");
    std::fill(values.begin(), values.end(), Real(0));
    std::fill(derivatives.begin(), derivatives.end(), Real(0));
    std::fill(second_derivatives.begin(), second_derivatives.end(), Real(0));

    const Real tol = std::numeric_limits<Real>::epsilon() * Real(128) *
                     std::max(Real(1), std::abs(x));

    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(x - nodes_1d_[i]) <= tol) [[unlikely]] {
            values[i] = Real(1);
            for (std::size_t j = 0; j < n; ++j) {
                derivatives[j] = derivative_matrix_1d_[i * n + j];
                second_derivatives[j] = second_derivative_matrix_1d_[i * n + j];
            }
            return;
        }
    }

    Real S0 = Real(0);
    Real S1 = Real(0);
    Real S2 = Real(0);
    for (std::size_t i = 0; i < n; ++i) {
        const Real dx = x - nodes_1d_[i];
        const Real a = barycentric_weights_[i] / dx;
        values[i] = a;
        S0 += a;
        S1 -= a / dx;
        S2 += Real(2) * a / (dx * dx);
    }

    const Real inv_S0 = Real(1) / S0;
    const Real S1_over_S0 = S1 * inv_S0;
    const Real denom_log_second = (S2 * S0 - S1 * S1) * inv_S0 * inv_S0;
    for (std::size_t i = 0; i < n; ++i) {
        const Real dx = x - nodes_1d_[i];
        const Real value = values[i] * inv_S0;
        const Real basis_log_derivative = -Real(1) / dx - S1_over_S0;
        const Real basis_log_second = Real(1) / (dx * dx) - denom_log_second;
        values[i] = value;
        derivatives[i] = value * basis_log_derivative;
        second_derivatives[i] =
            value * (basis_log_derivative * basis_log_derivative + basis_log_second);
    }
}

std::vector<Real> SpectralBasis::eval_1d(Real x) const {
    std::vector<Real> values(nodes_1d_.size(), Real(0));
    eval_1d_to(x, std::span<Real>(values.data(), values.size()));
    return values;
}

std::vector<Real> SpectralBasis::eval_1d_derivative(Real x) const {
    std::vector<Real> derivatives(nodes_1d_.size(), Real(0));
    eval_1d_derivative_to(x, std::span<Real>(derivatives.data(), derivatives.size()));
    return derivatives;
}

std::vector<Real> SpectralBasis::eval_1d_second_derivative(Real x) const {
    std::vector<Real> second_derivatives(nodes_1d_.size(), Real(0));
    eval_1d_second_derivative_to(
        x, std::span<Real>(second_derivatives.data(), second_derivatives.size()));
    return second_derivatives;
}

void SpectralBasis::eval_1d_all(Real x,
                                std::vector<Real>& values,
                                std::vector<Real>& derivatives,
                                std::vector<Real>& second_derivatives) const {
    const std::size_t n = nodes_1d_.size();
    values.resize(n);
    derivatives.resize(n);
    second_derivatives.resize(n);
    eval_1d_all_to(x,
                   std::span<Real>(values.data(), values.size()),
                   std::span<Real>(derivatives.data(), derivatives.size()),
                   std::span<Real>(second_derivatives.data(), second_derivatives.size()));
}

// -----------------------------------------------------------------------
// Collapsed GLL triangle node placement.
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

    const auto gll01 = build_gll_zero_one_table(p);
    for (int j = 0; j <= p; ++j) {
        const Real y =
            gll01[static_cast<std::size_t>(p)][static_cast<std::size_t>(j)];
        const Real x_scale = Real(1) - y;
        const int x_order = p - j;
        for (int i = 0; i <= p - j; ++i) {
            const Real x_fraction =
                gll01[static_cast<std::size_t>(x_order)][static_cast<std::size_t>(i)];
            const math::Vector<Real, 3> node{x_scale * x_fraction, y, Real(0)};
            validate_triangle_node(node, p, simplex_nodes_.size());
            simplex_nodes_.push_back(node);
        }
    }

    if (simplex_nodes_.size() != size_) {
        throw BasisConstructionException("SpectralBasis: triangle nodal set size mismatch",
                                         __FILE__, __LINE__, __func__);
    }
}

// -----------------------------------------------------------------------
// Collapsed GLL tetrahedron node placement
// Reference tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
// -----------------------------------------------------------------------
void SpectralBasis::build_simplex_nodes_tetrahedron() {
    const int p = order_;
    simplex_nodes_.clear();
    simplex_nodes_.reserve(size_);

    const auto gll01 = build_gll_zero_one_table(p);
    for (int k = 0; k <= p; ++k) {
        const Real z = gll01[static_cast<std::size_t>(p)][static_cast<std::size_t>(k)];
        const Real xy_scale = Real(1) - z;
        const int y_order = p - k;
        for (int j = 0; j <= p - k; ++j) {
            const Real y_fraction =
                gll01[static_cast<std::size_t>(y_order)][static_cast<std::size_t>(j)];
            const Real y = xy_scale * y_fraction;
            const Real x_scale = xy_scale - y;
            const int x_order = p - j - k;
            for (int i = 0; i <= p - j - k; ++i) {
                const Real x_fraction =
                    gll01[static_cast<std::size_t>(x_order)][static_cast<std::size_t>(i)];
                const math::Vector<Real, 3> node{x_scale * x_fraction, y, z};
                validate_tetrahedron_node(node, p, simplex_nodes_.size());
                simplex_nodes_.push_back(node);
            }
        }
    }

    if (simplex_nodes_.size() != size_) {
        throw BasisConstructionException("SpectralBasis: tetrahedron nodal set size mismatch",
                                         __FILE__, __LINE__, __func__);
    }
}

void SpectralBasis::build_pyramid_nodes() {
    simplex_nodes_.clear();
    simplex_nodes_.reserve(size_);
    pyramid_modal_terms_.clear();
    pyramid_modal_terms_.reserve(size_);

    // Base face: tensor-product GLL nodes.
    SpectralBasis line_basis(ElementType::Line2, order_);
    nodes_1d_ = line_basis.nodes_1d_;
    for (Real y : nodes_1d_) {
        for (Real x : nodes_1d_) {
            append_unique_point(simplex_nodes_, {x, y, Real(0)});
        }
    }

    append_unique_point(simplex_nodes_, {Real(0), Real(0), Real(1)});

    // Rising edges: use the same 1D GLL parameter along each apex edge.
    for (std::size_t i = 1; i + 1 < nodes_1d_.size(); ++i) {
        const Real z = Real(0.5) * (nodes_1d_[i] + Real(1));
        const Real scale = Real(1) - z;
        append_unique_point(simplex_nodes_, {-scale, -scale, z});
        append_unique_point(simplex_nodes_, {scale, -scale, z});
        append_unique_point(simplex_nodes_, {scale, scale, z});
        append_unique_point(simplex_nodes_, {-scale, scale, z});
    }

    // Triangular side-face interiors: reuse the existing triangle spectral
    // nodal set, but only add strictly interior face nodes here. Edge nodes are
    // already represented explicitly above so shared-entity ownership is
    // deterministic.
    SpectralBasis triangle_basis(ElementType::Triangle3, order_);
    for (const auto& node : triangle_basis.simplex_nodes_) {
        const Real a = node[0];
        const Real b = node[1];
        if (a <= Real(1e-10) || b <= Real(1e-10) || a + b >= Real(1) - Real(1e-10)) {
            continue;
        }
        append_unique_point(simplex_nodes_, map_triangle_to_pyramid_front(node[0], node[1]));
        append_unique_point(simplex_nodes_, map_triangle_to_pyramid_right(node[0], node[1]));
        append_unique_point(simplex_nodes_, map_triangle_to_pyramid_back(node[0], node[1]));
        append_unique_point(simplex_nodes_, map_triangle_to_pyramid_left(node[0], node[1]));
    }

    // Interior nodes: retain the standard complete-family pyramid interior
    // locations so the rational modal space remains well-conditioned away from
    // the boundary while the traces stay spectral-face compatible.
    const auto lagrange_nodes = ReferenceNodeLayout::get_lagrange_node_coords(ElementType::Pyramid5, order_);
    for (const auto& node : lagrange_nodes) {
        if (!is_on_pyramid_boundary(node)) {
            append_unique_point(simplex_nodes_, node);
        }
    }

    if (simplex_nodes_.size() != size_) {
        throw BasisConstructionException("SpectralBasis: pyramid nodal set size mismatch (got " +
                                             std::to_string(simplex_nodes_.size()) +
                                             ", expected " + std::to_string(size_) + ")",
                                         __FILE__, __LINE__, __func__);
    }

    pyramid_modal_terms_ = pyramid_modal::build_terms(order_);
    if (pyramid_modal_terms_.size() != size_) {
        throw BasisConstructionException("SpectralBasis: pyramid modal-space size mismatch",
                                         __FILE__, __LINE__, __func__);
    }
}

void SpectralBasis::build_inverse_vandermonde() {
    const std::size_t n = size_;
    const SpectralModalMatrixCacheKey key{element_type_, order_};

    static std::mutex cache_mutex;
    static std::unordered_map<SpectralModalMatrixCacheKey,
                              std::shared_ptr<const detail::SpectralModalMatrixData>,
                              SpectralModalMatrixCacheKeyHash> cache;

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto found = cache.find(key);
        if (found != cache.end()) {
            modal_matrices_ = found->second;
            return;
        }
    }

    std::vector<Real> vandermonde(n * n, Real(0));

    if (is_pyramid_modal_) {
        pyramid_modal::EvaluationPoint pyramid_point;
        for (std::size_t row = 0; row < n; ++row) {
            pyramid_modal::prepare_evaluation_point(
                pyramid_modal_terms_, simplex_nodes_[row], pyramid_point);
            for (std::size_t col = 0; col < n; ++col) {
                Real value = Real(0);
                pyramid_modal::evaluate_term(pyramid_modal_terms_[col], pyramid_point, value);
                vandermonde[row * n + col] = value;
            }
        }
    } else if (dimension_ == 2) {
        // Enumerate modal indices (p,q) with p+q <= order_
        struct PQ { int p, q; };
        std::vector<PQ> modes;
        for (int total = 0; total <= order_; ++total) {
            for (int p = 0; p <= total; ++p) {
                modes.push_back({p, total - p});
            }
        }
        assert(modes.size() == n);

        for (std::size_t row = 0; row < n; ++row) {
            const Real xi = simplex_nodes_[row][0];
            const Real eta = simplex_nodes_[row][1];
            for (std::size_t col = 0; col < n; ++col) {
                vandermonde[row * n + col] =
                    orthopoly::dubiner(modes[col].p, modes[col].q, xi, eta);
            }
        }
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

        for (std::size_t row = 0; row < n; ++row) {
            const Real xi = simplex_nodes_[row][0];
            const Real eta = simplex_nodes_[row][1];
            const Real zeta = simplex_nodes_[row][2];
            for (std::size_t col = 0; col < n; ++col) {
                vandermonde[row * n + col] =
                    orthopoly::proriol(modes[col].p, modes[col].q, modes[col].r, xi, eta, zeta);
            }
        }
    }

    const char* label = is_pyramid_modal_
                            ? "SpectralBasis pyramid Vandermonde"
                            : (dimension_ == 2 ? "SpectralBasis triangle Vandermonde"
                                               : "SpectralBasis tetrahedron Vandermonde");
    auto inverse_result =
        math::invert_dense_matrix_with_diagnostics(std::move(vandermonde), n, label);
    math::validate_dense_inverse_diagnostics(inverse_result, n, label);

    auto matrices = std::make_shared<detail::SpectralModalMatrixData>();
    matrices->modal_to_nodal_by_modal = std::move(inverse_result.inverse);
    const std::vector<Real>& inverse = matrices->modal_to_nodal_by_modal;
    matrices->modal_to_nodal_by_basis.assign(n * n, Real(0));
    for (std::size_t basis_i = 0; basis_i < n; ++basis_i) {
        for (std::size_t modal_j = 0; modal_j < n; ++modal_j) {
            matrices->modal_to_nodal_by_basis[basis_i * n + modal_j] =
                inverse[modal_j * n + basis_i];
        }
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [iter, inserted] = cache.emplace(key, matrices);
        (void)inserted;
        modal_matrices_ = iter->second;
    }
}

// Keep modal transform helpers free of forced-inline attributes unless
// compiler-versioned benchmarks and LLVM IR checks show a stable benefit.
const std::vector<Real>& SpectralBasis::modal_to_nodal_by_basis() const noexcept {
    return modal_matrices_->modal_to_nodal_by_basis;
}

const std::vector<Real>& SpectralBasis::modal_to_nodal_by_modal() const noexcept {
    return modal_matrices_->modal_to_nodal_by_modal;
}

void SpectralBasis::apply_modal_to_nodal(std::span<const Real> modal,
                                         std::span<Real> nodal) const {
    if (modal.size() != size_ || nodal.size() != size_) {
        throw BasisEvaluationException("SpectralBasis modal-to-nodal transform size mismatch",
                                       __FILE__, __LINE__, __func__);
    }

    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        const Real* row = modal_to_nodal_by_basis().data() + basis_i * size_;
        Real value = Real(0);
        for (std::size_t modal_j = 0; modal_j < size_; ++modal_j) {
            value += row[modal_j] * modal[modal_j];
        }
        nodal[basis_i] = value;
    }
}

void SpectralBasis::apply_modal_gradients_to_nodal(
    std::span<const Gradient> modal_gradients,
    std::span<Gradient> nodal_gradients,
    int components) const {
    if (modal_gradients.size() != size_ || nodal_gradients.size() != size_) {
        throw BasisEvaluationException("SpectralBasis modal-gradient transform size mismatch",
                                       __FILE__, __LINE__, __func__);
    }

    std::fill(nodal_gradients.begin(), nodal_gradients.end(), Gradient{});
    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        const Real* row = modal_to_nodal_by_basis().data() + basis_i * size_;
        Gradient gradient{};
        for (std::size_t modal_j = 0; modal_j < size_; ++modal_j) {
            const Real coeff = row[modal_j];
            for (int component = 0; component < components; ++component) {
                const auto c = static_cast<std::size_t>(component);
                gradient[c] += coeff * modal_gradients[modal_j][c];
            }
        }
        nodal_gradients[basis_i] = gradient;
    }
}

void SpectralBasis::apply_modal_hessians_to_nodal(
    std::span<const Hessian> modal_hessians,
    std::span<Hessian> nodal_hessians,
    int components) const {
    if (modal_hessians.size() != size_ || nodal_hessians.size() != size_) {
        throw BasisEvaluationException("SpectralBasis modal-Hessian transform size mismatch",
                                       __FILE__, __LINE__, __func__);
    }

    std::fill(nodal_hessians.begin(), nodal_hessians.end(), Hessian{});
    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        const Real* matrix_row = modal_to_nodal_by_basis().data() + basis_i * size_;
        Hessian hessian{};
        for (std::size_t modal_j = 0; modal_j < size_; ++modal_j) {
            const Real coeff = matrix_row[modal_j];
            for (int row = 0; row < components; ++row) {
                const auto r = static_cast<std::size_t>(row);
                for (int col = 0; col < components; ++col) {
                    const auto c = static_cast<std::size_t>(col);
                    hessian(r, c) += coeff * modal_hessians[modal_j](r, c);
                }
            }
        }
        nodal_hessians[basis_i] = hessian;
    }
}

void SpectralBasis::apply_modal_values_to_nodal_strided(std::span<const Real> modal,
                                                        Real* SVMP_RESTRICT values_out,
                                                        std::size_t output_stride,
                                                        std::size_t q) const {
    if (modal.size() != size_) {
        throw BasisEvaluationException("SpectralBasis modal-to-nodal transform size mismatch",
                                       __FILE__, __LINE__, __func__);
    }

    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        const Real* row = modal_to_nodal_by_basis().data() + basis_i * size_;
        Real value = Real(0);
        for (std::size_t modal_j = 0; modal_j < size_; ++modal_j) {
            value += row[modal_j] * modal[modal_j];
        }
        values_out[basis_i * output_stride + q] = value;
    }
}

void SpectralBasis::apply_modal_gradients_to_nodal_strided(
    std::span<const Gradient> modal_gradients,
    Real* SVMP_RESTRICT gradients_out,
    int components,
    std::size_t output_stride,
    std::size_t q) const {
    if (modal_gradients.size() != size_) {
        throw BasisEvaluationException("SpectralBasis modal-gradient transform size mismatch",
                                       __FILE__, __LINE__, __func__);
    }

    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        const Real* row = modal_to_nodal_by_basis().data() + basis_i * size_;
        Real gradient[3] = {Real(0), Real(0), Real(0)};
        for (std::size_t modal_j = 0; modal_j < size_; ++modal_j) {
            const Real coeff = row[modal_j];
            for (int component = 0; component < components; ++component) {
                const auto c = static_cast<std::size_t>(component);
                gradient[c] += coeff * modal_gradients[modal_j][c];
            }
        }
        for (std::size_t c = 0; c < 3u; ++c) {
            gradients_out[(basis_i * 3u + c) * output_stride + q] = gradient[c];
        }
    }
}

void SpectralBasis::apply_modal_hessians_to_nodal_strided(
    std::span<const Hessian> modal_hessians,
    Real* SVMP_RESTRICT hessians_out,
    int components,
    std::size_t output_stride,
    std::size_t q) const {
    if (modal_hessians.size() != size_) {
        throw BasisEvaluationException("SpectralBasis modal-Hessian transform size mismatch",
                                       __FILE__, __LINE__, __func__);
    }

    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        const Real* row = modal_to_nodal_by_basis().data() + basis_i * size_;
        Real hessian[9] = {};
        for (std::size_t modal_j = 0; modal_j < size_; ++modal_j) {
            const Real coeff = row[modal_j];
            for (int r = 0; r < components; ++r) {
                for (int c = 0; c < components; ++c) {
                    const auto rr = static_cast<std::size_t>(r);
                    const auto cc = static_cast<std::size_t>(c);
                    hessian[rr * 3u + cc] += coeff * modal_hessians[modal_j](rr, cc);
                }
            }
        }
        for (std::size_t rc = 0; rc < 9u; ++rc) {
            hessians_out[(basis_i * 9u + rc) * output_stride + q] = hessian[rc];
        }
    }
}

void SpectralBasis::evaluate_spectral_flat(const math::Vector<Real, 3>& xi,
                                           Real* SVMP_RESTRICT values_out,
                                           Real* SVMP_RESTRICT gradients_out,
                                           Real* SVMP_RESTRICT hessians_out,
                                           FlatOutputLayout layout) const {
    const bool need_values = values_out != nullptr;
    const bool need_gradients = gradients_out != nullptr;
    const bool need_hessians = hessians_out != nullptr;

    if (is_wedge_tensor_product_) {
        auto& scratch = spectral_scratch();
        const std::size_t face_size = face_basis_->size();
        const std::size_t axis_size = axis_basis_->size();
        const bool need_sub_gradients = need_gradients || need_hessians;

        scratch.face_values.resize(face_size);
        scratch.axis_values.resize(axis_size);
        if (need_sub_gradients) {
            scratch.face_gradients.resize(face_size * 3u);
            scratch.axis_gradients.resize(axis_size * 3u);
        }
        if (need_hessians) {
            scratch.face_hessians.resize(face_size * 9u);
            scratch.axis_hessians.resize(axis_size * 9u);
        }

        face_basis_->evaluate_spectral_flat(
            {xi[0], xi[1], Real(0)},
            scratch.face_values.data(),
            need_sub_gradients ? scratch.face_gradients.data() : nullptr,
            need_hessians ? scratch.face_hessians.data() : nullptr,
            FlatOutputLayout{1u, 0u});
        axis_basis_->evaluate_spectral_flat(
            {xi[2], Real(0), Real(0)},
            scratch.axis_values.data(),
            need_sub_gradients ? scratch.axis_gradients.data() : nullptr,
            need_hessians ? scratch.axis_hessians.data() : nullptr,
            FlatOutputLayout{1u, 0u});

        std::size_t idx = 0;
        for (std::size_t k = 0; k < axis_size; ++k) {
            for (std::size_t j = 0; j < face_size; ++j) {
                const Real face_value = scratch.face_values[j];
                const Real axis_value = scratch.axis_values[k];
                if (need_values) {
                    values_out[idx * layout.output_stride + layout.q] =
                        face_value * axis_value;
                }
                if (need_gradients) {
                    const Real* face_gradient = scratch.face_gradients.data() + j * 3u;
                    const Real axis_gradient = scratch.axis_gradients[k * 3u + 0u];
                    gradients_out[(idx * 3u + 0u) * layout.output_stride + layout.q] =
                        face_gradient[0] * axis_value;
                    gradients_out[(idx * 3u + 1u) * layout.output_stride + layout.q] =
                        face_gradient[1] * axis_value;
                    gradients_out[(idx * 3u + 2u) * layout.output_stride + layout.q] =
                        face_value * axis_gradient;
                }
                if (need_hessians) {
                    const Real* face_gradient = scratch.face_gradients.data() + j * 3u;
                    const Real* face_hessian = scratch.face_hessians.data() + j * 9u;
                    const Real axis_gradient = scratch.axis_gradients[k * 3u + 0u];
                    const Real axis_hessian = scratch.axis_hessians[k * 9u + 0u];
                    const Real hxy = face_hessian[1] * axis_value;
                    const Real hxz = face_gradient[0] * axis_gradient;
                    const Real hyz = face_gradient[1] * axis_gradient;
                    const Real hessian[9] = {
                        face_hessian[0] * axis_value, hxy, hxz,
                        hxy, face_hessian[4] * axis_value, hyz,
                        hxz, hyz, face_value * axis_hessian};
                    for (std::size_t rc = 0; rc < 9u; ++rc) {
                        hessians_out[(idx * 9u + rc) * layout.output_stride + layout.q] =
                            hessian[rc];
                    }
                }
                ++idx;
            }
        }
        return;
    }

    if (is_pyramid_modal_) {
        if (need_values) {
            for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
                values_out[basis_i * layout.output_stride + layout.q] = Real(0);
            }
        }
        if (need_gradients) {
            for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
                for (std::size_t c = 0; c < 3u; ++c) {
                    gradients_out[(basis_i * 3u + c) * layout.output_stride + layout.q] =
                        Real(0);
                }
            }
        }
        if (need_hessians) {
            for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
                for (std::size_t rc = 0; rc < 9u; ++rc) {
                    hessians_out[(basis_i * 9u + rc) * layout.output_stride + layout.q] =
                        Real(0);
                }
            }
        }

        auto& pyramid_point = spectral_scratch().pyramid_point;
        pyramid_modal::prepare_evaluation_point(pyramid_modal_terms_, xi, pyramid_point);
        for (std::size_t modal_j = 0; modal_j < size_; ++modal_j) {
            Real modal_value = Real(0);
            Gradient modal_gradient{};
            Hessian modal_hessian{};
            pyramid_modal::evaluate_term(
                pyramid_modal_terms_[modal_j],
                pyramid_point,
                modal_value,
                (need_gradients || need_hessians) ? &modal_gradient : nullptr,
                need_hessians ? &modal_hessian : nullptr);

            const Real* coeffs = modal_to_nodal_by_modal().data() + modal_j * size_;
            const Real* modal_gradient_data = modal_gradient.data();
            const Real* modal_hessian_data = modal_hessian.data();
            for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
                const Real coeff = coeffs[basis_i];
                if (need_values) {
                    values_out[basis_i * layout.output_stride + layout.q] +=
                        coeff * modal_value;
                }
                if (need_gradients) {
                    for (std::size_t c = 0; c < 3u; ++c) {
                        gradients_out[(basis_i * 3u + c) * layout.output_stride + layout.q] +=
                            coeff * modal_gradient_data[c];
                    }
                }
                if (need_hessians) {
                    for (std::size_t rc = 0; rc < 9u; ++rc) {
                        hessians_out[(basis_i * 9u + rc) * layout.output_stride + layout.q] +=
                            coeff * modal_hessian_data[rc];
                    }
                }
            }
        }
        return;
    }

    if (!is_simplex_) {
        auto& scratch = spectral_scratch();
        const std::size_t n1d = nodes_1d_.size();
        const bool need_first = need_gradients || need_hessians;
        const bool need_second = need_hessians;

        auto fill_axis = [&](Real coordinate,
                             std::vector<Real>& values,
                             std::vector<Real>& first,
                             std::vector<Real>& second) {
            values.resize(n1d);
            if (need_first) {
                first.resize(n1d);
            }
            if (need_second) {
                second.resize(n1d);
                eval_1d_all_to(coordinate,
                               std::span<Real>(values.data(), values.size()),
                               std::span<Real>(first.data(), first.size()),
                               std::span<Real>(second.data(), second.size()));
            } else {
                eval_1d_to(coordinate, std::span<Real>(values.data(), values.size()));
                if (need_first) {
                    eval_1d_derivative_to(coordinate,
                                          std::span<Real>(first.data(), first.size()));
                }
            }
        };

        fill_axis(xi[0], scratch.lx, scratch.dx, scratch.ddx);
        if (dimension_ >= 2) {
            fill_axis(xi[1], scratch.ly, scratch.dy, scratch.ddy);
        }
        if (dimension_ == 3) {
            fill_axis(xi[2], scratch.lz, scratch.dz, scratch.ddz);
        }

        for (std::size_t idx = 0; idx < size_; ++idx) {
            const std::size_t i = idx % n1d;
            const std::size_t j = (dimension_ >= 2) ? (idx / n1d) % n1d : 0u;
            const std::size_t k = (dimension_ == 3) ? idx / (n1d * n1d) : 0u;

            const Real lx = scratch.lx[i];
            const Real ly = (dimension_ >= 2) ? scratch.ly[j] : Real(1);
            const Real lz = (dimension_ == 3) ? scratch.lz[k] : Real(1);
            if (need_values) {
                values_out[idx * layout.output_stride + layout.q] = lx * ly * lz;
            }

            if (need_gradients) {
                const Real gx = scratch.dx[i] * ly * lz;
                const Real gy = (dimension_ >= 2) ? lx * scratch.dy[j] * lz : Real(0);
                const Real gz = (dimension_ == 3) ? lx * ly * scratch.dz[k] : Real(0);
                gradients_out[(idx * 3u + 0u) * layout.output_stride + layout.q] = gx;
                gradients_out[(idx * 3u + 1u) * layout.output_stride + layout.q] = gy;
                gradients_out[(idx * 3u + 2u) * layout.output_stride + layout.q] = gz;
            }

            if (need_hessians) {
                const Real dx = scratch.dx[i];
                const Real dy = (dimension_ >= 2) ? scratch.dy[j] : Real(0);
                const Real dz = (dimension_ == 3) ? scratch.dz[k] : Real(0);
                const Real ddx = scratch.ddx[i];
                const Real ddy = (dimension_ >= 2) ? scratch.ddy[j] : Real(0);
                const Real ddz = (dimension_ == 3) ? scratch.ddz[k] : Real(0);
                const Real hxy = (dimension_ >= 2) ? dx * dy * lz : Real(0);
                const Real hxz = (dimension_ == 3) ? dx * ly * dz : Real(0);
                const Real hyz = (dimension_ == 3) ? lx * dy * dz : Real(0);
                const Real hessian[9] = {
                    ddx * ly * lz, hxy, hxz,
                    hxy, lx * ddy * lz, hyz,
                    hxz, hyz, lx * ly * ddz};
                for (std::size_t rc = 0; rc < 9u; ++rc) {
                    hessians_out[(idx * 9u + rc) * layout.output_stride + layout.q] =
                        hessian[rc];
                }
            }
        }
        return;
    }

    auto& simplex_scratch = spectral_scratch();
    if (need_values) {
        simplex_scratch.modal_values.resize(size_);
    }
    if (need_gradients) {
        simplex_scratch.modal_gradient_components.resize(size_ * 3u);
    }
    if (need_hessians) {
        simplex_scratch.modal_hessian_components.resize(size_ * 9u);
    }

    auto store_modal = [&](std::size_t modal_j,
                           Real modal_value,
                           const Real* modal_gradient,
                           const Real* modal_hessian) {
        if (need_values) {
            simplex_scratch.modal_values[modal_j] = modal_value;
        }
        if (need_gradients) {
            Real* gradient = simplex_scratch.modal_gradient_components.data() + modal_j * 3u;
            gradient[0] = modal_gradient[0];
            gradient[1] = modal_gradient[1];
            gradient[2] = modal_gradient[2];
        }
        if (need_hessians) {
            Real* hessian = simplex_scratch.modal_hessian_components.data() + modal_j * 9u;
            for (std::size_t rc = 0; rc < 9u; ++rc) {
                hessian[rc] = modal_hessian[rc];
            }
        }
    };

    std::size_t modal_j = 0;
    if (dimension_ == 2) {
        for (int total = 0; total <= order_; ++total) {
            for (int p = 0; p <= total; ++p) {
                const int q_mode = total - p;
                Real modal_value = Real(0);
                Real modal_gradient[3] = {};
                Real modal_hessian[9] = {};
                if (need_hessians) {
                    const auto modal =
                        orthopoly::dubiner_with_second_derivatives(p, q_mode, xi[0], xi[1]);
                    modal_value = modal.value;
                    modal_gradient[0] = modal.dxi;
                    modal_gradient[1] = modal.deta;
                    modal_hessian[0] = modal.dxx;
                    modal_hessian[1] = modal.dxy;
                    modal_hessian[3] = modal.dxy;
                    modal_hessian[4] = modal.dyy;
                } else if (need_gradients) {
                    const auto modal = orthopoly::dubiner_derivatives(p, q_mode, xi[0], xi[1]);
                    modal_value = modal.value;
                    modal_gradient[0] = modal.dxi;
                    modal_gradient[1] = modal.deta;
                } else {
                    modal_value = orthopoly::dubiner(p, q_mode, xi[0], xi[1]);
                }
                store_modal(modal_j++, modal_value, modal_gradient, modal_hessian);
            }
        }
    } else {
        for (int total = 0; total <= order_; ++total) {
            for (int p = 0; p <= total; ++p) {
                for (int q_mode = 0; q_mode <= total - p; ++q_mode) {
                    const int r_mode = total - p - q_mode;
                    Real modal_value = Real(0);
                    Real modal_gradient[3] = {};
                    Real modal_hessian[9] = {};
                    if (need_hessians) {
                        const auto modal = orthopoly::proriol_with_second_derivatives(
                            p, q_mode, r_mode, xi[0], xi[1], xi[2]);
                        modal_value = modal.value;
                        const Real* gradient = modal.gradient.data();
                        const Real* hessian = modal.hessian.data();
                        for (std::size_t c = 0; c < 3u; ++c) {
                            modal_gradient[c] = gradient[c];
                        }
                        for (std::size_t rc = 0; rc < 9u; ++rc) {
                            modal_hessian[rc] = hessian[rc];
                        }
                    } else if (need_gradients) {
                        const auto modal =
                            orthopoly::proriol_derivatives(p, q_mode, r_mode, xi[0], xi[1], xi[2]);
                        modal_value = modal.value;
                        const Real* gradient = modal.gradient.data();
                        for (std::size_t c = 0; c < 3u; ++c) {
                            modal_gradient[c] = gradient[c];
                        }
                    } else {
                        modal_value = orthopoly::proriol(p, q_mode, r_mode, xi[0], xi[1], xi[2]);
                    }
                    store_modal(modal_j++, modal_value, modal_gradient, modal_hessian);
                }
            }
        }
    }

    if (!need_values && need_gradients && !need_hessians) {
        for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
            const Real* row = modal_to_nodal_by_basis().data() + basis_i * size_;
            Real g0 = Real(0);
            Real g1 = Real(0);
            Real g2 = Real(0);
            if (dimension_ == 2) {
                for (std::size_t modal_index = 0; modal_index < size_; ++modal_index) {
                    const Real coeff = row[modal_index];
                    const Real* modal_gradient =
                        simplex_scratch.modal_gradient_components.data() + modal_index * 3u;
                    g0 += coeff * modal_gradient[0];
                    g1 += coeff * modal_gradient[1];
                }
            } else {
                for (std::size_t modal_index = 0; modal_index < size_; ++modal_index) {
                    const Real coeff = row[modal_index];
                    const Real* modal_gradient =
                        simplex_scratch.modal_gradient_components.data() + modal_index * 3u;
                    g0 += coeff * modal_gradient[0];
                    g1 += coeff * modal_gradient[1];
                    g2 += coeff * modal_gradient[2];
                }
            }
            gradients_out[(basis_i * 3u + 0u) * layout.output_stride + layout.q] = g0;
            gradients_out[(basis_i * 3u + 1u) * layout.output_stride + layout.q] = g1;
            gradients_out[(basis_i * 3u + 2u) * layout.output_stride + layout.q] = g2;
        }
        return;
    }

    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        const Real* row = modal_to_nodal_by_basis().data() + basis_i * size_;
        Real value = Real(0);
        Real gradient[3] = {};
        Real hessian[9] = {};
        for (std::size_t modal_index = 0; modal_index < size_; ++modal_index) {
            const Real coeff = row[modal_index];
            if (need_values) {
                value += coeff * simplex_scratch.modal_values[modal_index];
            }
            if (need_gradients) {
                const Real* modal_gradient =
                    simplex_scratch.modal_gradient_components.data() + modal_index * 3u;
                for (int c = 0; c < dimension_; ++c) {
                    const auto component = static_cast<std::size_t>(c);
                    gradient[component] += coeff * modal_gradient[component];
                }
            }
            if (need_hessians) {
                const Real* modal_hessian =
                    simplex_scratch.modal_hessian_components.data() + modal_index * 9u;
                for (int r = 0; r < dimension_; ++r) {
                    const auto row_component = static_cast<std::size_t>(r);
                    for (int c = 0; c < dimension_; ++c) {
                        const auto col_component = static_cast<std::size_t>(c);
                        const auto component = row_component * 3u + col_component;
                        hessian[component] += coeff * modal_hessian[component];
                    }
                }
            }
        }
        if (need_values) {
            values_out[basis_i * layout.output_stride + layout.q] = value;
        }
        if (need_gradients) {
            for (std::size_t c = 0; c < 3u; ++c) {
                gradients_out[(basis_i * 3u + c) * layout.output_stride + layout.q] =
                    gradient[c];
            }
        }
        if (need_hessians) {
            for (std::size_t rc = 0; rc < 9u; ++rc) {
                hessians_out[(basis_i * 9u + rc) * layout.output_stride + layout.q] =
                    hessian[rc];
            }
        }
    }
}

void SpectralBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                    std::vector<Real>& values) const {
    values.resize(size_);
    evaluate_spectral_flat(xi, values.data(), nullptr, nullptr, FlatOutputLayout{1u, 0u});
}

void SpectralBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                       std::vector<Gradient>& gradients) const {
    gradients.resize(size_);
    auto& flat = spectral_scratch().nodal_gradient_components;
    flat.resize(size_ * 3u);
    evaluate_spectral_flat(xi, nullptr, flat.data(), nullptr, FlatOutputLayout{1u, 0u});
    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        Gradient gradient{};
        const Real* source = flat.data() + basis_i * 3u;
        gradient[0] = source[0];
        gradient[1] = source[1];
        gradient[2] = source[2];
        gradients[basis_i] = gradient;
    }
}

void SpectralBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                      std::vector<Hessian>& hessians) const {
    hessians.resize(size_);
    auto& flat = spectral_scratch().nodal_hessian_components;
    flat.resize(size_ * 9u);
    evaluate_spectral_flat(xi, nullptr, nullptr, flat.data(), FlatOutputLayout{1u, 0u});
    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        Hessian hessian{};
        const Real* source = flat.data() + basis_i * 9u;
        for (std::size_t rc = 0; rc < 9u; ++rc) {
            hessian.data()[rc] = source[rc];
        }
        hessians[basis_i] = hessian;
    }
}

void SpectralBasis::evaluate_all(const math::Vector<Real, 3>& xi,
                                 std::vector<Real>& values,
                                 std::vector<Gradient>& gradients,
                                 std::vector<Hessian>& hessians) const {
    values.resize(size_);
    gradients.resize(size_);
    hessians.resize(size_);
    auto& scratch = spectral_scratch();
    auto& flat_gradients = scratch.nodal_gradient_components;
    auto& flat_hessians = scratch.nodal_hessian_components;
    flat_gradients.resize(size_ * 3u);
    flat_hessians.resize(size_ * 9u);
    evaluate_spectral_flat(xi,
                           values.data(),
                           flat_gradients.data(),
                           flat_hessians.data(),
                           FlatOutputLayout{1u, 0u});
    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        Gradient gradient{};
        const Real* gradient_source = flat_gradients.data() + basis_i * 3u;
        gradient[0] = gradient_source[0];
        gradient[1] = gradient_source[1];
        gradient[2] = gradient_source[2];
        gradients[basis_i] = gradient;

        Hessian hessian{};
        const Real* hessian_source = flat_hessians.data() + basis_i * 9u;
        for (std::size_t rc = 0; rc < 9u; ++rc) {
            hessian.data()[rc] = hessian_source[rc];
        }
        hessians[basis_i] = hessian;
    }
}

void SpectralBasis::write_values_strided(const math::Vector<Real, 3>& xi,
                                         Real* SVMP_RESTRICT values_out,
                                         std::size_t output_stride,
                                         std::size_t q) const {
    evaluate_spectral_flat(xi, values_out, nullptr, nullptr, FlatOutputLayout{output_stride, q});
}

void SpectralBasis::write_gradients_strided(const math::Vector<Real, 3>& xi,
                                            Real* SVMP_RESTRICT gradients_out,
                                            std::size_t output_stride,
                                            std::size_t q) const {
    evaluate_spectral_flat(xi, nullptr, gradients_out, nullptr, FlatOutputLayout{output_stride, q});
}

void SpectralBasis::write_hessians_strided(const math::Vector<Real, 3>& xi,
                                           Real* SVMP_RESTRICT hessians_out,
                                           std::size_t output_stride,
                                           std::size_t q) const {
    evaluate_spectral_flat(xi, nullptr, nullptr, hessians_out, FlatOutputLayout{output_stride, q});
}

void SpectralBasis::write_all_strided(const math::Vector<Real, 3>& xi,
                                      Real* SVMP_RESTRICT values_out,
                                      Real* SVMP_RESTRICT gradients_out,
                                      Real* SVMP_RESTRICT hessians_out,
                                      std::size_t output_stride,
                                      std::size_t q) const {
    evaluate_spectral_flat(xi,
                           values_out,
                           gradients_out,
                           hessians_out,
                           FlatOutputLayout{output_stride, q});
}

void SpectralBasis::write_tensor_product_batch_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    const std::size_t num_qpts = points.size();
    if (num_qpts == 0u ||
        (values_out == nullptr && gradients_out == nullptr && hessians_out == nullptr)) {
        return;
    }

    const bool need_first = gradients_out != nullptr || hessians_out != nullptr;
    const bool need_second = hessians_out != nullptr;
    const std::size_t n1d = nodes_1d_.size();
    auto& scratch = spectral_scratch();

    auto fill_axis = [&](std::size_t component,
                         std::vector<Real>& values,
                         std::vector<Real>& first,
                         std::vector<Real>& second) {
        values.resize(n1d * num_qpts);
        if (need_first) {
            first.resize(n1d * num_qpts);
        }
        if (need_second) {
            second.resize(n1d * num_qpts);
        }
        scratch.axis_temp_values.resize(n1d);
        if (need_first) {
            scratch.axis_temp_gradients.resize(n1d);
        }
        if (need_second) {
            scratch.axis_temp_hessians.resize(n1d);
        }

        for (std::size_t q = 0; q < num_qpts; ++q) {
            Real* v = scratch.axis_temp_values.data();
            Real* d = need_first ? scratch.axis_temp_gradients.data() : nullptr;
            Real* d2 = need_second ? scratch.axis_temp_hessians.data() : nullptr;
            if (need_second) {
                eval_1d_all_to(points[q][component],
                               std::span<Real>(v, n1d),
                               std::span<Real>(d, n1d),
                               std::span<Real>(d2, n1d));
            } else {
                eval_1d_to(points[q][component], std::span<Real>(v, n1d));
                if (need_first) {
                    eval_1d_derivative_to(points[q][component], std::span<Real>(d, n1d));
                }
            }
            for (std::size_t node = 0; node < n1d; ++node) {
                values[node * num_qpts + q] = v[node];
                if (need_first) {
                    first[node * num_qpts + q] = d[node];
                }
                if (need_second) {
                    second[node * num_qpts + q] = d2[node];
                }
            }
        }
    };

    fill_axis(0u, scratch.lx, scratch.dx, scratch.ddx);
    if (dimension_ >= 2) {
        fill_axis(1u, scratch.ly, scratch.dy, scratch.ddy);
    }
    if (dimension_ == 3) {
        fill_axis(2u, scratch.lz, scratch.dz, scratch.ddz);
    }

    auto axis_value = [&](const std::vector<Real>& values,
                          std::size_t node,
                          std::size_t q) {
        return values[node * num_qpts + q];
    };

    if (dimension_ == 3) {
        for (std::size_t idx = 0; idx < size_; ++idx) {
            const std::size_t i = idx % n1d;
            const std::size_t j = (idx / n1d) % n1d;
            const std::size_t k = idx / (n1d * n1d);

            Real* value_row = values_out ? values_out + idx * output_stride : nullptr;
            Real* gradient_row =
                gradients_out ? gradients_out + idx * 3u * output_stride : nullptr;
            Real* hessian_row =
                hessians_out ? hessians_out + idx * 9u * output_stride : nullptr;

            for (std::size_t q = 0; q < num_qpts; ++q) {
                const Real lx = scratch.lx[i * num_qpts + q];
                const Real ly = scratch.ly[j * num_qpts + q];
                const Real lz = scratch.lz[k * num_qpts + q];

                if (value_row != nullptr) {
                    value_row[q] = lx * ly * lz;
                }

                if (gradient_row != nullptr) {
                    const Real dx = scratch.dx[i * num_qpts + q];
                    const Real dy = scratch.dy[j * num_qpts + q];
                    const Real dz = scratch.dz[k * num_qpts + q];
                    gradient_row[0u * output_stride + q] = dx * ly * lz;
                    gradient_row[1u * output_stride + q] = lx * dy * lz;
                    gradient_row[2u * output_stride + q] = lx * ly * dz;
                }

                if (hessian_row != nullptr) {
                    const Real dx = scratch.dx[i * num_qpts + q];
                    const Real dy = scratch.dy[j * num_qpts + q];
                    const Real dz = scratch.dz[k * num_qpts + q];
                    const Real ddx = scratch.ddx[i * num_qpts + q];
                    const Real ddy = scratch.ddy[j * num_qpts + q];
                    const Real ddz = scratch.ddz[k * num_qpts + q];
                    const Real hxy = dx * dy * lz;
                    const Real hxz = dx * ly * dz;
                    const Real hyz = lx * dy * dz;

                    hessian_row[0u * output_stride + q] = ddx * ly * lz;
                    hessian_row[1u * output_stride + q] = hxy;
                    hessian_row[2u * output_stride + q] = hxz;
                    hessian_row[3u * output_stride + q] = hxy;
                    hessian_row[4u * output_stride + q] = lx * ddy * lz;
                    hessian_row[5u * output_stride + q] = hyz;
                    hessian_row[6u * output_stride + q] = hxz;
                    hessian_row[7u * output_stride + q] = hyz;
                    hessian_row[8u * output_stride + q] = lx * ly * ddz;
                }
            }
        }
        return;
    }

    for (std::size_t idx = 0; idx < size_; ++idx) {
        const std::size_t i = idx % n1d;
        const std::size_t j = (dimension_ >= 2) ? (idx / n1d) % n1d : 0u;
        const std::size_t k = (dimension_ == 3) ? idx / (n1d * n1d) : 0u;

        Real* value_row = values_out ? values_out + idx * output_stride : nullptr;
        Real* gradient_row = gradients_out ? gradients_out + idx * 3u * output_stride : nullptr;
        Real* hessian_row = hessians_out ? hessians_out + idx * 9u * output_stride : nullptr;

        for (std::size_t q = 0; q < num_qpts; ++q) {
            const Real lx = axis_value(scratch.lx, i, q);
            const Real ly = (dimension_ >= 2) ? axis_value(scratch.ly, j, q) : Real(1);
            const Real lz = (dimension_ == 3) ? axis_value(scratch.lz, k, q) : Real(1);

            if (value_row != nullptr) {
                value_row[q] = lx * ly * lz;
            }

            if (gradient_row != nullptr) {
                const Real dx = axis_value(scratch.dx, i, q);
                gradient_row[0u * output_stride + q] = dx * ly * lz;
                gradient_row[1u * output_stride + q] = (dimension_ >= 2)
                    ? lx * axis_value(scratch.dy, j, q) * lz
                    : Real(0);
                gradient_row[2u * output_stride + q] = (dimension_ == 3)
                    ? lx * ly * axis_value(scratch.dz, k, q)
                    : Real(0);
            }

            if (hessian_row != nullptr) {
                const Real dx = axis_value(scratch.dx, i, q);
                const Real dy = (dimension_ >= 2) ? axis_value(scratch.dy, j, q) : Real(0);
                const Real dz = (dimension_ == 3) ? axis_value(scratch.dz, k, q) : Real(0);
                const Real ddx = axis_value(scratch.ddx, i, q);
                const Real ddy = (dimension_ >= 2) ? axis_value(scratch.ddy, j, q) : Real(0);
                const Real ddz = (dimension_ == 3) ? axis_value(scratch.ddz, k, q) : Real(0);
                const Real hxy = dx * dy * lz;
                const Real hxz = dx * ly * dz;
                const Real hyz = lx * dy * dz;

                hessian_row[0u * output_stride + q] = ddx * ly * lz;
                hessian_row[1u * output_stride + q] = hxy;
                hessian_row[2u * output_stride + q] = hxz;
                hessian_row[3u * output_stride + q] = hxy;
                hessian_row[4u * output_stride + q] = lx * ddy * lz;
                hessian_row[5u * output_stride + q] = hyz;
                hessian_row[6u * output_stride + q] = hxz;
                hessian_row[7u * output_stride + q] = hyz;
                hessian_row[8u * output_stride + q] = lx * ly * ddz;
            }
        }
    }
}

void SpectralBasis::write_wedge_batch_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    const std::size_t num_qpts = points.size();
    if (num_qpts == 0u ||
        (values_out == nullptr && gradients_out == nullptr && hessians_out == nullptr)) {
        return;
    }

    const bool need_gradients = gradients_out != nullptr || hessians_out != nullptr;
    const bool need_hessians = hessians_out != nullptr;
    const std::size_t face_size = face_basis_->size();
    const std::size_t axis_size = axis_basis_->size();
    auto& scratch = spectral_scratch();

    scratch.face_points.resize(num_qpts);
    scratch.axis_points.resize(num_qpts);
    for (std::size_t q = 0; q < num_qpts; ++q) {
        scratch.face_points[q] = {points[q][0], points[q][1], Real(0)};
        scratch.axis_points[q] = {points[q][2], Real(0), Real(0)};
    }

    scratch.face_values.resize(face_size * num_qpts);
    scratch.axis_values.resize(axis_size * num_qpts);
    Real* face_gradients = nullptr;
    Real* axis_gradients = nullptr;
    Real* face_hessians = nullptr;
    Real* axis_hessians = nullptr;
    if (need_gradients) {
        scratch.face_gradients.resize(face_size * 3u * num_qpts);
        scratch.axis_gradients.resize(axis_size * 3u * num_qpts);
        face_gradients = scratch.face_gradients.data();
        axis_gradients = scratch.axis_gradients.data();
    } else {
        scratch.face_gradients.clear();
        scratch.axis_gradients.clear();
    }
    if (need_hessians) {
        scratch.face_hessians.resize(face_size * 9u * num_qpts);
        scratch.axis_hessians.resize(axis_size * 9u * num_qpts);
        face_hessians = scratch.face_hessians.data();
        axis_hessians = scratch.axis_hessians.data();
    } else {
        scratch.face_hessians.clear();
        scratch.axis_hessians.clear();
    }

    face_basis_->evaluate_at_quadrature_points_strided(
        scratch.face_points,
        num_qpts,
        scratch.face_values.data(),
        face_gradients,
        face_hessians);
    axis_basis_->evaluate_at_quadrature_points_strided(
        scratch.axis_points,
        num_qpts,
        scratch.axis_values.data(),
        axis_gradients,
        axis_hessians);

    for (std::size_t k = 0; k < axis_size; ++k) {
        for (std::size_t j = 0; j < face_size; ++j) {
            const std::size_t idx = k * face_size + j;
            Real* value_row = values_out ? values_out + idx * output_stride : nullptr;
            Real* gradient_row = gradients_out ? gradients_out + idx * 3u * output_stride : nullptr;
            Real* hessian_row = hessians_out ? hessians_out + idx * 9u * output_stride : nullptr;

            for (std::size_t q = 0; q < num_qpts; ++q) {
                const Real face_value = scratch.face_values[j * num_qpts + q];
                const Real axis_value = scratch.axis_values[k * num_qpts + q];

                if (value_row != nullptr) {
                    value_row[q] = face_value * axis_value;
                }

                const Real* face_gradient = need_gradients
                    ? scratch.face_gradients.data() + j * 3u * num_qpts
                    : nullptr;
                const Real axis_gradient = need_gradients
                    ? scratch.axis_gradients[(k * 3u + 0u) * num_qpts + q]
                    : Real(0);

                if (gradient_row != nullptr) {
                    gradient_row[0u * output_stride + q] =
                        face_gradient[0u * num_qpts + q] * axis_value;
                    gradient_row[1u * output_stride + q] =
                        face_gradient[1u * num_qpts + q] * axis_value;
                    gradient_row[2u * output_stride + q] = face_value * axis_gradient;
                }

                if (hessian_row != nullptr) {
                    const Real* face_hessian = scratch.face_hessians.data() + j * 9u * num_qpts;
                    const Real axis_hessian =
                        scratch.axis_hessians[(k * 9u + 0u) * num_qpts + q];
                    const Real hxz = face_gradient[0u * num_qpts + q] * axis_gradient;
                    const Real hyz = face_gradient[1u * num_qpts + q] * axis_gradient;
                    hessian_row[0u * output_stride + q] =
                        face_hessian[0u * num_qpts + q] * axis_value;
                    hessian_row[1u * output_stride + q] =
                        face_hessian[1u * num_qpts + q] * axis_value;
                    hessian_row[2u * output_stride + q] = hxz;
                    hessian_row[3u * output_stride + q] =
                        face_hessian[3u * num_qpts + q] * axis_value;
                    hessian_row[4u * output_stride + q] =
                        face_hessian[4u * num_qpts + q] * axis_value;
                    hessian_row[5u * output_stride + q] = hyz;
                    hessian_row[6u * output_stride + q] = hxz;
                    hessian_row[7u * output_stride + q] = hyz;
                    hessian_row[8u * output_stride + q] = face_value * axis_hessian;
                }
            }
        }
    }
}

void SpectralBasis::write_modal_batch_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    const std::size_t num_qpts = points.size();
    if (num_qpts == 0u ||
        (values_out == nullptr && gradients_out == nullptr && hessians_out == nullptr)) {
        return;
    }

    const bool need_values = values_out != nullptr;
    const bool need_gradients = gradients_out != nullptr;
    const bool need_hessians = hessians_out != nullptr;
    const int components = is_pyramid_modal_ ? 3 : dimension_;
    auto& scratch = spectral_scratch();

    if (need_values) {
        scratch.modal_values.resize(size_ * num_qpts);
    } else {
        scratch.modal_values.clear();
    }
    if (need_gradients) {
        scratch.modal_gradient_components.resize(size_ * 3u * num_qpts);
    } else {
        scratch.modal_gradient_components.clear();
    }
    if (need_hessians) {
        scratch.modal_hessian_components.resize(size_ * 9u * num_qpts);
    } else {
        scratch.modal_hessian_components.clear();
    }

    auto store_modal = [&](std::size_t modal_j,
                           std::size_t point_index,
                           Real modal_value,
                           const Gradient& modal_gradient,
                           const Hessian& modal_hessian) {
        if (need_values) {
            scratch.modal_values[modal_j * num_qpts + point_index] = modal_value;
        }
        if (need_gradients) {
            for (int component = 0; component < components; ++component) {
                const auto c = static_cast<std::size_t>(component);
                scratch.modal_gradient_components[(modal_j * 3u + c) * num_qpts + point_index] =
                    modal_gradient[c];
            }
        }
        if (need_hessians) {
            for (int r = 0; r < components; ++r) {
                const auto sr = static_cast<std::size_t>(r);
                for (int c = 0; c < components; ++c) {
                    const auto sc = static_cast<std::size_t>(c);
                    scratch.modal_hessian_components[(modal_j * 9u + sr * 3u + sc) * num_qpts + point_index] =
                        modal_hessian(sr, sc);
                }
            }
        }
    };

    for (std::size_t point_index = 0; point_index < num_qpts; ++point_index) {
        const auto& xi = points[point_index];
        if (is_pyramid_modal_) {
            pyramid_modal::prepare_evaluation_point(pyramid_modal_terms_, xi, scratch.pyramid_point);
            for (std::size_t modal_j = 0; modal_j < size_; ++modal_j) {
                Real modal_value = Real(0);
                Gradient modal_gradient{};
                Hessian modal_hessian{};
                pyramid_modal::evaluate_term(
                    pyramid_modal_terms_[modal_j],
                    scratch.pyramid_point,
                    modal_value,
                    need_gradients ? &modal_gradient : nullptr,
                    need_hessians ? &modal_hessian : nullptr);
                store_modal(modal_j, point_index, modal_value, modal_gradient, modal_hessian);
            }
            continue;
        }

        std::size_t modal_j = 0;
        if (dimension_ == 2) {
            for (int total = 0; total <= order_; ++total) {
                for (int p = 0; p <= total; ++p) {
                    const int q_mode = total - p;
                    Real modal_value = Real(0);
                    Gradient modal_gradient{};
                    Hessian modal_hessian{};
                    if (need_hessians) {
                        const auto modal =
                            orthopoly::dubiner_with_second_derivatives(p, q_mode, xi[0], xi[1]);
                        modal_value = modal.value;
                        modal_gradient[0] = modal.dxi;
                        modal_gradient[1] = modal.deta;
                        modal_hessian(0, 0) = modal.dxx;
                        modal_hessian(0, 1) = modal.dxy;
                        modal_hessian(1, 0) = modal.dxy;
                        modal_hessian(1, 1) = modal.dyy;
                    } else if (need_gradients) {
                        const auto modal =
                            orthopoly::dubiner_derivatives(p, q_mode, xi[0], xi[1]);
                        modal_value = modal.value;
                        modal_gradient[0] = modal.dxi;
                        modal_gradient[1] = modal.deta;
                    } else {
                        modal_value = orthopoly::dubiner(p, q_mode, xi[0], xi[1]);
                    }
                    store_modal(modal_j++, point_index, modal_value, modal_gradient, modal_hessian);
                }
            }
        } else {
            for (int total = 0; total <= order_; ++total) {
                for (int p = 0; p <= total; ++p) {
                    for (int q_mode = 0; q_mode <= total - p; ++q_mode) {
                        const int r = total - p - q_mode;
                        Real modal_value = Real(0);
                        Gradient modal_gradient{};
                        Hessian modal_hessian{};
                        if (need_hessians) {
                            const auto modal =
                                orthopoly::proriol_with_second_derivatives(
                                    p, q_mode, r, xi[0], xi[1], xi[2]);
                            modal_value = modal.value;
                            modal_gradient = modal.gradient;
                            modal_hessian = modal.hessian;
                        } else if (need_gradients) {
                            const auto modal =
                                orthopoly::proriol_derivatives(
                                    p, q_mode, r, xi[0], xi[1], xi[2]);
                            modal_value = modal.value;
                            modal_gradient = modal.gradient;
                        } else {
                            modal_value = orthopoly::proriol(p, q_mode, r, xi[0], xi[1], xi[2]);
                        }
                        store_modal(modal_j++, point_index, modal_value, modal_gradient, modal_hessian);
                    }
                }
            }
        }
    }

    for (std::size_t basis_i = 0; basis_i < size_; ++basis_i) {
        Real* value_row = need_values ? values_out + basis_i * output_stride : nullptr;
        Real* gradient_rows[3] = {};
        Real* hessian_rows[9] = {};

        if (need_values) {
            std::fill_n(value_row, num_qpts, Real(0));
        }
        if (need_gradients) {
            for (std::size_t c = 0; c < 3u; ++c) {
                gradient_rows[c] = gradients_out + (basis_i * 3u + c) * output_stride;
                std::fill_n(gradient_rows[c], num_qpts, Real(0));
            }
        }
        if (need_hessians) {
            for (std::size_t rc = 0; rc < 9u; ++rc) {
                hessian_rows[rc] = hessians_out + (basis_i * 9u + rc) * output_stride;
                std::fill_n(hessian_rows[rc], num_qpts, Real(0));
            }
        }

        const Real* coeffs = modal_to_nodal_by_basis().data() + basis_i * size_;
        for (std::size_t modal_j = 0; modal_j < size_; ++modal_j) {
            const Real coeff = coeffs[modal_j];
            if (need_values) {
                const Real* modal_values = scratch.modal_values.data() + modal_j * num_qpts;
                for (std::size_t q = 0; q < num_qpts; ++q) {
                    value_row[q] += coeff * modal_values[q];
                }
            }
            if (need_gradients) {
                for (int component = 0; component < components; ++component) {
                    const auto c = static_cast<std::size_t>(component);
                    const Real* modal_gradient_row =
                        scratch.modal_gradient_components.data() + (modal_j * 3u + c) * num_qpts;
                    Real* gradient_row = gradient_rows[c];
                    for (std::size_t q = 0; q < num_qpts; ++q) {
                        gradient_row[q] += coeff * modal_gradient_row[q];
                    }
                }
            }
            if (need_hessians) {
                for (int r = 0; r < components; ++r) {
                    const auto sr = static_cast<std::size_t>(r);
                    for (int c = 0; c < components; ++c) {
                        const auto sc = static_cast<std::size_t>(c);
                        const auto component = sr * 3u + sc;
                        const Real* modal_hessian_row =
                            scratch.modal_hessian_components.data() + (modal_j * 9u + component) * num_qpts;
                        Real* hessian_row = hessian_rows[component];
                        for (std::size_t q = 0; q < num_qpts; ++q) {
                            hessian_row[q] += coeff * modal_hessian_row[q];
                        }
                    }
                }
            }
        }
    }
}

void SpectralBasis::evaluate_values_to(const math::Vector<Real, 3>& xi,
                                       Real* SVMP_RESTRICT values_out) const {
    write_values_strided(xi, values_out, 1u, 0u);
}

void SpectralBasis::evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                                          Real* SVMP_RESTRICT gradients_out) const {
    write_gradients_strided(xi, gradients_out, 1u, 0u);
}

void SpectralBasis::evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                                         Real* SVMP_RESTRICT hessians_out) const {
    write_hessians_strided(xi, hessians_out, 1u, 0u);
}

void SpectralBasis::evaluate_at_quadrature_points(
    const std::vector<math::Vector<Real, 3>>& points,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    evaluate_at_quadrature_points_strided(points,
                                          points.size(),
                                          values_out,
                                          gradients_out,
                                          hessians_out);
}

void SpectralBasis::evaluate_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) const {
    const std::size_t num_qpts = points.size();
    if (output_stride < num_qpts) {
        throw BasisConfigurationException(
            "SpectralBasis strided evaluation requires output_stride >= points.size()",
            __FILE__, __LINE__, __func__);
    }

    if (is_wedge_tensor_product_) {
        write_wedge_batch_strided(points, output_stride, values_out, gradients_out, hessians_out);
        return;
    }

    if (!is_simplex_ && !is_pyramid_modal_) {
        write_tensor_product_batch_strided(
            points, output_stride, values_out, gradients_out, hessians_out);
        return;
    }

    write_modal_batch_strided(points, output_stride, values_out, gradients_out, hessians_out);
}

} // namespace basis
} // namespace FE
} // namespace svmp
