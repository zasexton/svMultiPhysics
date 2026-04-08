/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "LagrangeBasis.h"
#include "NodeOrderingConventions.h"
#include "detail/LagrangeBasisPyramidDetail.h"
#include "detail/LagrangeBasisSimplexDetail.h"
#include "detail/LagrangeBasisUtilityDetail.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace svmp {
namespace FE {
namespace basis {

namespace {

enum class LagrangeTopology {
    Point,
    Line,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
    Wedge,
    Pyramid,
};

struct LagrangeTopologyTraits {
    LagrangeTopology topology;
    int dimension;
};

constexpr Real kNodeCoordTolerance = Real(1e-12);

LagrangeTopologyTraits lagrange_topology_traits(ElementType type) {
    switch (type) {
        case ElementType::Point1:
            return {LagrangeTopology::Point, 0};
        case ElementType::Line2:
        case ElementType::Line3:
            return {LagrangeTopology::Line, 1};
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return {LagrangeTopology::Triangle, 2};
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
            return {LagrangeTopology::Quadrilateral, 2};
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return {LagrangeTopology::Tetrahedron, 3};
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            return {LagrangeTopology::Hexahedron, 3};
        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
            return {LagrangeTopology::Wedge, 3};
        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return {LagrangeTopology::Pyramid, 3};
    }

    throw FEException("Unsupported element type for LagrangeBasis",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

std::size_t lattice_index_pm_one(Real coord, int order, const char* context) {
    if (order <= 0) {
        if (std::abs(coord) > kNodeCoordTolerance) {
            throw FEException(context, __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        return 0;
    }

    const Real scaled = (coord + Real(1)) * static_cast<Real>(order) / Real(2);
    const long idx = std::lround(scaled);
    if (idx < 0 || idx > order ||
        std::abs(coord - detail::equispaced_pm_one_coord(static_cast<int>(idx), order)) > kNodeCoordTolerance) {
        throw FEException(context, __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return static_cast<std::size_t>(idx);
}

int simplex_lattice_index(Real coord, int order, const char* context) {
    if (order <= 0) {
        if (std::abs(coord - Real(0)) > kNodeCoordTolerance &&
            std::abs(coord - Real(0.25)) > kNodeCoordTolerance &&
            std::abs(coord - Real(1) / Real(3)) > kNodeCoordTolerance) {
            throw FEException(context, __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        return 0;
    }

    const Real scaled = coord * static_cast<Real>(order);
    const long idx = std::lround(scaled);
    const Real reconstructed = static_cast<Real>(idx) / static_cast<Real>(order);
    if (idx < 0 || idx > order || std::abs(coord - reconstructed) > kNodeCoordTolerance) {
        throw FEException(context, __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return static_cast<int>(idx);
}

std::array<int, 4> triangle_exponents_from_public_node(const math::Vector<Real, 3>& node,
                                                       int order) {
    if (order == 0) {
        return {0, 0, 0, 0};
    }

    const int j = simplex_lattice_index(node[0], order,
                                        "LagrangeBasis: invalid triangle node coordinate for public ordering");
    const int k = simplex_lattice_index(node[1], order,
                                        "LagrangeBasis: invalid triangle node coordinate for public ordering");
    const int i = order - j - k;
    if (i < 0) {
        throw FEException("LagrangeBasis: invalid triangle barycentric coordinates for public ordering",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return {i, j, k, 0};
}

std::array<int, 4> tetrahedron_exponents_from_public_node(const math::Vector<Real, 3>& node,
                                                          int order) {
    if (order == 0) {
        return {0, 0, 0, 0};
    }

    const int j = simplex_lattice_index(node[0], order,
                                        "LagrangeBasis: invalid tetrahedron node x-coordinate for public ordering");
    const int k = simplex_lattice_index(node[1], order,
                                        "LagrangeBasis: invalid tetrahedron node y-coordinate for public ordering");
    const int l = simplex_lattice_index(node[2], order,
                                        "LagrangeBasis: invalid tetrahedron node z-coordinate for public ordering");
    const int i = order - j - k - l;
    if (i < 0) {
        throw FEException("LagrangeBasis: invalid tetrahedron barycentric coordinates for public ordering",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return {i, j, k, l};
}

struct NormalizedLagrangeRequest {
    ElementType element_type;
    int order;
};

struct AxisBasisEvaluations {
    std::vector<Real> values;
    std::vector<Real> first;
    std::vector<Real> second;
};

AxisBasisEvaluations constant_axis_basis() {
    return AxisBasisEvaluations{{Real(1)}, {Real(0)}, {Real(0)}};
}

AxisBasisEvaluations evaluate_1d_basis(const std::vector<Real>& nodes_1d, Real xi) {
    AxisBasisEvaluations basis;
    basis.values.assign(nodes_1d.size(), Real(0));
    basis.first.assign(nodes_1d.size(), Real(0));
    basis.second.assign(nodes_1d.size(), Real(0));

    if (nodes_1d.size() == 1) {
        basis.values[0] = Real(1);
        return basis;
    }

    for (std::size_t i = 0; i < nodes_1d.size(); ++i) {
        Real value = Real(1);
        for (std::size_t j = 0; j < nodes_1d.size(); ++j) {
            if (i == j) {
                continue;
            }
            value *= (xi - nodes_1d[j]) / (nodes_1d[i] - nodes_1d[j]);
        }
        basis.values[i] = value;

        Real first = Real(0);
        for (std::size_t m = 0; m < nodes_1d.size(); ++m) {
            if (m == i) {
                continue;
            }
            Real prod = Real(1);
            for (std::size_t j = 0; j < nodes_1d.size(); ++j) {
                if (j == i || j == m) {
                    continue;
                }
                prod *= (xi - nodes_1d[j]) / (nodes_1d[i] - nodes_1d[j]);
            }
            prod /= (nodes_1d[i] - nodes_1d[m]);
            first += prod;
        }
        basis.first[i] = first;
    }

    if (nodes_1d.size() <= 2) {
        return basis;
    }

    for (std::size_t i = 0; i < nodes_1d.size(); ++i) {
        Real second = Real(0);
        for (std::size_t m1 = 0; m1 < nodes_1d.size(); ++m1) {
            if (m1 == i) {
                continue;
            }
            for (std::size_t m2 = m1 + 1; m2 < nodes_1d.size(); ++m2) {
                if (m2 == i) {
                    continue;
                }
                Real prod = Real(1);
                for (std::size_t j = 0; j < nodes_1d.size(); ++j) {
                    if (j == i || j == m1 || j == m2) {
                        continue;
                    }
                    prod *= (xi - nodes_1d[j]) / (nodes_1d[i] - nodes_1d[j]);
                }
                prod /= (nodes_1d[i] - nodes_1d[m1]);
                prod /= (nodes_1d[i] - nodes_1d[m2]);
                second += prod;
            }
        }
        basis.second[i] = Real(2) * second;
    }

    return basis;
}

void evaluate_tensor_product_values(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    std::vector<Real>& values) {
    values.assign(tensor_indices.size(), Real(0));
    for (std::size_t n = 0; n < tensor_indices.size(); ++n) {
        const auto& index = tensor_indices[n];
        values[n] = x_axis.values[index[0]] *
                    y_axis.values[index[1]] *
                    z_axis.values[index[2]];
    }
}

void evaluate_tensor_product_gradients(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    std::vector<Gradient>& gradients) {
    gradients.assign(tensor_indices.size(), Gradient{});
    for (std::size_t n = 0; n < tensor_indices.size(); ++n) {
        const auto& index = tensor_indices[n];
        gradients[n][0] = x_axis.first[index[0]] * y_axis.values[index[1]] * z_axis.values[index[2]];
        gradients[n][1] = x_axis.values[index[0]] * y_axis.first[index[1]] * z_axis.values[index[2]];
        gradients[n][2] = x_axis.values[index[0]] * y_axis.values[index[1]] * z_axis.first[index[2]];
    }
}

void evaluate_tensor_product_hessians(
    const std::vector<std::array<std::size_t, 3>>& tensor_indices,
    const AxisBasisEvaluations& x_axis,
    const AxisBasisEvaluations& y_axis,
    const AxisBasisEvaluations& z_axis,
    std::vector<Hessian>& hessians) {
    hessians.assign(tensor_indices.size(), Hessian{});
    for (std::size_t n = 0; n < tensor_indices.size(); ++n) {
        const auto& index = tensor_indices[n];
        Hessian H{};
        H(0, 0) = x_axis.second[index[0]] * y_axis.values[index[1]] * z_axis.values[index[2]];
        H(1, 1) = x_axis.values[index[0]] * y_axis.second[index[1]] * z_axis.values[index[2]];
        H(2, 2) = x_axis.values[index[0]] * y_axis.values[index[1]] * z_axis.second[index[2]];

        H(0, 1) = x_axis.first[index[0]] * y_axis.first[index[1]] * z_axis.values[index[2]];
        H(1, 0) = H(0, 1);
        H(0, 2) = x_axis.first[index[0]] * y_axis.values[index[1]] * z_axis.first[index[2]];
        H(2, 0) = H(0, 2);
        H(1, 2) = x_axis.values[index[0]] * y_axis.first[index[1]] * z_axis.first[index[2]];
        H(2, 1) = H(1, 2);
        hessians[n] = H;
    }
}

NormalizedLagrangeRequest normalize_lagrange_request(ElementType element_type, int order) {
    switch (element_type) {
        case ElementType::Line3:
            return {ElementType::Line2, std::max(order, 2)};
        case ElementType::Triangle6:
            return {ElementType::Triangle3, std::max(order, 2)};
        case ElementType::Quad9:
            return {ElementType::Quad4, std::max(order, 2)};
        case ElementType::Quad8:
            throw FEException("Quad8 serendipity Lagrange basis not implemented",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        case ElementType::Tetra10:
            return {ElementType::Tetra4, std::max(order, 2)};
        case ElementType::Hex27:
            return {ElementType::Hex8, std::max(order, 2)};
        case ElementType::Hex20:
            throw FEException("Hex20 serendipity Lagrange basis not implemented",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        case ElementType::Wedge18:
            return {ElementType::Wedge6, std::max(order, 2)};
        case ElementType::Wedge15:
            throw FEException("Wedge15 serendipity Lagrange basis not implemented",
                              __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        case ElementType::Pyramid13:
            throw FEException(
                "Pyramid13 is a serendipity variant; use SerendipityBasis (Pyramid13) or the complete-family Lagrange path via LagrangeBasis (Pyramid5, order >= 2)",
                __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
        case ElementType::Pyramid14:
            return {ElementType::Pyramid5, std::max(order, 2)};
        default:
            return {element_type, order};
    }
}

} // namespace

LagrangeBasis::LagrangeBasis(ElementType type, int order)
    : element_type_(type), dimension_(0), order_(order) {
    const NormalizedLagrangeRequest normalized = normalize_lagrange_request(element_type_, order_);
    element_type_ = normalized.element_type;
    order_ = normalized.order;

    if (order_ < 0) {
        throw FEException("LagrangeBasis requires non-negative polynomial order",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    dimension_ = lagrange_topology_traits(element_type_).dimension;

    init_nodes();
}

void LagrangeBasis::init_nodes() {
    nodes_.clear();
    nodes_1d_.clear();
    tensor_indices_.clear();
    simplex_exponents_.clear();
    wedge_indices_.clear();
    const auto topology = lagrange_topology_traits(element_type_).topology;
    switch (topology) {
        case LagrangeTopology::Point:
            build_point_nodes();
            return;
        case LagrangeTopology::Line:
            build_tensor_product_nodes(1);
            return;
        case LagrangeTopology::Quadrilateral:
            build_tensor_product_nodes(2);
            return;
        case LagrangeTopology::Hexahedron:
            build_tensor_product_nodes(3);
            return;
        case LagrangeTopology::Triangle:
        case LagrangeTopology::Tetrahedron:
            build_simplex_nodes();
            return;
        case LagrangeTopology::Wedge:
            build_wedge_nodes();
            return;
        case LagrangeTopology::Pyramid:
            build_pyramid_nodes();
            return;
    }

    throw FEException("Unsupported element type in LagrangeBasis::init_nodes",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

void LagrangeBasis::build_point_nodes() {
    nodes_.push_back(math::Vector<Real, 3>{Real(0), Real(0), Real(0)});
}

void LagrangeBasis::init_equispaced_1d_nodes() {
    nodes_1d_.clear();
    for (int i = 0; i <= std::max(order_, 0); ++i) {
        nodes_1d_.push_back(detail::equispaced_pm_one_coord(i, order_));
    }
}

void LagrangeBasis::build_tensor_product_nodes(int dimensions) {
    init_equispaced_1d_nodes();

    if (dimensions < 1 || dimensions > 3) {
        throw FEException("LagrangeBasis::build_tensor_product_nodes requires dimension 1, 2, or 3",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    nodes_ = NodeOrdering::get_lagrange_node_coords(element_type_, order_);
    tensor_indices_.resize(nodes_.size(), TensorNodeIndex{0u, 0u, 0u});
    for (std::size_t n = 0; n < nodes_.size(); ++n) {
        tensor_indices_[n][0] = lattice_index_pm_one(
            nodes_[n][0], order_,
            "LagrangeBasis: invalid tensor-product x-coordinate in public node ordering");
        if (dimensions >= 2) {
            tensor_indices_[n][1] = lattice_index_pm_one(
                nodes_[n][1], order_,
                "LagrangeBasis: invalid tensor-product y-coordinate in public node ordering");
        }
        if (dimensions == 3) {
            tensor_indices_[n][2] = lattice_index_pm_one(
                nodes_[n][2], order_,
                "LagrangeBasis: invalid tensor-product z-coordinate in public node ordering");
        }
    }
}

void LagrangeBasis::build_simplex_nodes() {
    nodes_ = NodeOrdering::get_lagrange_node_coords(element_type_, order_);
    const LagrangeTopology topology = lagrange_topology_traits(element_type_).topology;
    simplex_exponents_.clear();
    simplex_exponents_.reserve(nodes_.size());
    for (const auto& node : nodes_) {
        switch (topology) {
            case LagrangeTopology::Triangle:
                simplex_exponents_.push_back(triangle_exponents_from_public_node(node, order_));
                break;
            case LagrangeTopology::Tetrahedron:
                simplex_exponents_.push_back(tetrahedron_exponents_from_public_node(node, order_));
                break;
            default:
                throw FEException("LagrangeBasis::build_simplex_nodes requires simplex topology",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
    }
}

void LagrangeBasis::build_wedge_nodes() {
    init_equispaced_1d_nodes();
    const auto triangle_nodes = NodeOrdering::get_lagrange_node_coords(ElementType::Triangle3, order_);
    simplex_exponents_.clear();
    simplex_exponents_.reserve(triangle_nodes.size());
    std::map<std::array<int, 4>, std::size_t> triangle_descriptor_to_index;
    for (std::size_t tri = 0; tri < triangle_nodes.size(); ++tri) {
        const auto exponents = triangle_exponents_from_public_node(triangle_nodes[tri], order_);
        simplex_exponents_.push_back(exponents);
        triangle_descriptor_to_index.emplace(exponents, tri);
    }

    nodes_ = NodeOrdering::get_lagrange_node_coords(element_type_, order_);
    wedge_indices_.clear();
    wedge_indices_.reserve(nodes_.size());
    for (const auto& node : nodes_) {
        const auto exponents = triangle_exponents_from_public_node(node, order_);
        const auto found = triangle_descriptor_to_index.find(exponents);
        if (found == triangle_descriptor_to_index.end()) {
            throw FEException("LagrangeBasis: failed to resolve wedge triangle descriptor in public ordering",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        wedge_indices_.push_back(WedgeNodeIndex{
            found->second,
            lattice_index_pm_one(node[2], order_,
                                 "LagrangeBasis: invalid wedge z-coordinate in public node ordering")
        });
    }
}

void LagrangeBasis::build_pyramid_nodes() {
    nodes_ = detail::PyramidLagrangeCache::get(order_).nodes;
}

void LagrangeBasis::evaluate_values(const math::Vector<Real, 3>& xi,
                                    std::vector<Real>& values) const {
    values.assign(size(), Real(0));
    const LagrangeTopology topology = lagrange_topology_traits(element_type_).topology;
    switch (topology) {
        case LagrangeTopology::Point:
            values[0] = Real(1);
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            const AxisBasisEvaluations x_axis = evaluate_1d_basis(nodes_1d_, xi[0]);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();

            if (topology != LagrangeTopology::Line) {
                y_axis = evaluate_1d_basis(nodes_1d_, xi[1]);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = evaluate_1d_basis(nodes_1d_, xi[2]);
            }

            evaluate_tensor_product_values(tensor_indices_, x_axis, y_axis, z_axis, values);
            return;
        }
        case LagrangeTopology::Wedge: {
            const AxisBasisEvaluations z_axis = evaluate_1d_basis(nodes_1d_, xi[2]);
            std::vector<Real> tri_values;
            detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi, &tri_values, nullptr, nullptr);

            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                values[n] = tri_values[index[0]] * z_axis.values[index[1]];
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            const auto& data = detail::PyramidLagrangeCache::get(order_);
            detail::PyramidLagrangeCache::evaluate_values(data, xi, values);
            return;
        }
        case LagrangeTopology::Triangle:
            detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi, &values, nullptr, nullptr);
            return;
        case LagrangeTopology::Tetrahedron:
            detail::evaluate_tetrahedron_simplex_basis(simplex_exponents_, order_, xi, &values, nullptr, nullptr);
            return;
    }

    throw FEException("Unsupported element in evaluate_values",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

void LagrangeBasis::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                       std::vector<Gradient>& gradients) const {
    gradients.assign(size(), Gradient{});
    const LagrangeTopology topology = lagrange_topology_traits(element_type_).topology;
    switch (topology) {
        case LagrangeTopology::Point:
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            const AxisBasisEvaluations x_axis = evaluate_1d_basis(nodes_1d_, xi[0]);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();

            if (topology != LagrangeTopology::Line) {
                y_axis = evaluate_1d_basis(nodes_1d_, xi[1]);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = evaluate_1d_basis(nodes_1d_, xi[2]);
            }

            evaluate_tensor_product_gradients(tensor_indices_, x_axis, y_axis, z_axis, gradients);
            return;
        }
        case LagrangeTopology::Wedge: {
            const AxisBasisEvaluations z_axis = evaluate_1d_basis(nodes_1d_, xi[2]);
            std::vector<Real> tri_values;
            std::vector<Gradient> tri_gradients;
            detail::evaluate_triangle_simplex_basis(
                simplex_exponents_, order_, xi, &tri_values, &tri_gradients, nullptr);

            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                gradients[n][0] = tri_gradients[index[0]][0] * z_axis.values[index[1]];
                gradients[n][1] = tri_gradients[index[0]][1] * z_axis.values[index[1]];
                gradients[n][2] = tri_values[index[0]] * z_axis.first[index[1]];
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            const auto& data = detail::PyramidLagrangeCache::get(order_);
            detail::PyramidLagrangeCache::evaluate_gradients(data, xi, gradients);
            return;
        }
        case LagrangeTopology::Triangle:
            detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi, nullptr, &gradients, nullptr);
            return;
        case LagrangeTopology::Tetrahedron:
            detail::evaluate_tetrahedron_simplex_basis(simplex_exponents_, order_, xi, nullptr, &gradients, nullptr);
            return;
    }

    throw FEException("Unsupported element in evaluate_gradients",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

void LagrangeBasis::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                      std::vector<Hessian>& hessians) const {
    hessians.assign(size(), Hessian{});
    const LagrangeTopology topology = lagrange_topology_traits(element_type_).topology;
    switch (topology) {
        case LagrangeTopology::Point:
            return;
        case LagrangeTopology::Line:
        case LagrangeTopology::Quadrilateral:
        case LagrangeTopology::Hexahedron: {
            const AxisBasisEvaluations x_axis = evaluate_1d_basis(nodes_1d_, xi[0]);
            AxisBasisEvaluations y_axis = constant_axis_basis();
            AxisBasisEvaluations z_axis = constant_axis_basis();

            if (topology != LagrangeTopology::Line) {
                y_axis = evaluate_1d_basis(nodes_1d_, xi[1]);
            }
            if (topology == LagrangeTopology::Hexahedron) {
                z_axis = evaluate_1d_basis(nodes_1d_, xi[2]);
            }

            evaluate_tensor_product_hessians(tensor_indices_, x_axis, y_axis, z_axis, hessians);
            return;
        }
        case LagrangeTopology::Triangle:
            detail::evaluate_triangle_simplex_basis(simplex_exponents_, order_, xi, nullptr, nullptr, &hessians);
            return;
        case LagrangeTopology::Tetrahedron:
            detail::evaluate_tetrahedron_simplex_basis(simplex_exponents_, order_, xi, nullptr, nullptr, &hessians);
            return;
        case LagrangeTopology::Wedge: {
            const AxisBasisEvaluations z_axis = evaluate_1d_basis(nodes_1d_, xi[2]);
            std::vector<Real> tri_values;
            std::vector<Gradient> tri_gradients;
            std::vector<Hessian> tri_hessians;
            detail::evaluate_triangle_simplex_basis(
                simplex_exponents_, order_, xi, &tri_values, &tri_gradients, &tri_hessians);

            for (std::size_t n = 0; n < wedge_indices_.size(); ++n) {
                const auto& index = wedge_indices_[n];
                Hessian H{};
                H(0, 0) = tri_hessians[index[0]](0, 0) * z_axis.values[index[1]];
                H(1, 1) = tri_hessians[index[0]](1, 1) * z_axis.values[index[1]];
                H(0, 1) = tri_hessians[index[0]](0, 1) * z_axis.values[index[1]];
                H(1, 0) = H(0, 1);

                H(2, 2) = tri_values[index[0]] * z_axis.second[index[1]];

                H(0, 2) = tri_gradients[index[0]][0] * z_axis.first[index[1]];
                H(2, 0) = H(0, 2);
                H(1, 2) = tri_gradients[index[0]][1] * z_axis.first[index[1]];
                H(2, 1) = H(1, 2);

                hessians[n] = H;
            }
            return;
        }
        case LagrangeTopology::Pyramid: {
            const auto& data = detail::PyramidLagrangeCache::get(order_);
            detail::PyramidLagrangeCache::evaluate_hessians(data, xi, hessians);
            return;
        }
    }

    throw FEException("Unsupported element in evaluate_hessians",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

} // namespace basis
} // namespace FE
} // namespace svmp
