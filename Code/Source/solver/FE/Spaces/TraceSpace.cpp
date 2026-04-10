/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/TraceSpace.h"

#include "Basis/BSplineBasis.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/NURBSTensorBasis.h"
#include "Basis/SerendipityBasis.h"
#include "Basis/TensorBasis.h"
#include "Basis/VectorBasis.h"
#include "Basis/NodeOrderingConventions.h"
#include "Core/FEException.h"
#include "Elements/ElementTransform.h"
#include "Elements/GeneralBasisElement.h"
#include "Elements/LagrangeElement.h"
#include "Elements/ReferenceElement.h"
#include "Geometry/IsoparametricMapping.h"
#include "Quadrature/QuadratureFactory.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>

namespace svmp {
namespace FE {
namespace spaces {

namespace {

using Vec3 = math::Vector<Real, 3>;
using EmbedFaceFn = std::function<Vec3(const Vec3&)>;

constexpr Real kNodeTol = Real(1e-12);

std::size_t quad_serendipity_dof_count(int order) {
    return static_cast<std::size_t>((order * order + 3 * order + 6) / 2);
}

int dominant_axis(const Vec3& delta, int dimension) {
    int axis = -1;
    Real best = Real(0);
    for (int d = 0; d < dimension; ++d) {
        const Real cand = std::abs(delta[static_cast<std::size_t>(d)]);
        if (cand > best) {
            best = cand;
            axis = d;
        }
    }
    return axis;
}

bool extract_tensor_spline_extents(const basis::BasisFunction& basis,
                                   std::vector<int>& extents) {
    if (const auto* spline =
            dynamic_cast<const basis::TensorProductBasis<basis::BSplineBasis>*>(&basis)) {
        extents = spline->tensor_extents();
        return true;
    }
    if (const auto* nurbs = dynamic_cast<const basis::NURBSTensorBasis*>(&basis)) {
        extents = nurbs->tensor_extents();
        return true;
    }
    return false;
}

int tensor_linear_index(const std::vector<int>& extents,
                        const std::vector<int>& coords) {
    FE_CHECK_ARG(extents.size() == coords.size(),
                 "TraceSpace: tensor index rank mismatch");
    if (extents.size() == 1u) {
        return coords[0];
    }
    if (extents.size() == 2u) {
        return coords[1] * extents[0] + coords[0];
    }
    return (coords[2] * extents[1] + coords[1]) * extents[0] + coords[0];
}

int scalar_trace_quadrature_order(BasisType basis_type, int order) {
    switch (basis_type) {
        case BasisType::BSpline:
            return std::max(2, 2 * order + 1);
        case BasisType::NURBS:
            return std::max(2, 2 * order + 3);
        default:
            return quadrature::QuadratureFactory::recommended_order(order, /*is_mass_matrix=*/true);
    }
}

Vec3 normalized_or_throw(Vec3 v, const char* message) {
    FE_CHECK_ARG(v.norm() > Real(0), message);
    v.normalize();
    return v;
}

std::vector<int> incident_face_edges(const elements::ReferenceElement& ref,
                                     const std::vector<int>& face_vertices) {
    std::vector<int> edge_ids;
    if (face_vertices.size() < 2u) {
        return edge_ids;
    }

    auto matches_edge = [&](const std::vector<LocalIndex>& edge_nodes,
                            int a,
                            int b) {
        FE_CHECK_ARG(edge_nodes.size() == 2u,
                     "TraceSpace: reference edge must have exactly two vertices");
        return (static_cast<int>(edge_nodes[0]) == a && static_cast<int>(edge_nodes[1]) == b) ||
               (static_cast<int>(edge_nodes[0]) == b && static_cast<int>(edge_nodes[1]) == a);
    };

    const std::size_t num_face_edges = (face_vertices.size() == 2u) ? 1u : face_vertices.size();
    edge_ids.reserve(num_face_edges);

    for (std::size_t i = 0; i < num_face_edges; ++i) {
        const int a = face_vertices[i];
        const int b = (face_vertices.size() == 2u)
            ? face_vertices[1]
            : face_vertices[(i + 1u) % face_vertices.size()];

        int matched = -1;
        for (std::size_t e = 0; e < ref.num_edges(); ++e) {
            if (matches_edge(ref.edge_nodes(e), a, b)) {
                matched = static_cast<int>(e);
                break;
            }
        }

        FE_CHECK_ARG(matched >= 0,
                     "TraceSpace: failed to identify a volume edge on the selected face");
        if (std::find(edge_ids.begin(), edge_ids.end(), matched) == edge_ids.end()) {
            edge_ids.push_back(matched);
        }
    }

    return edge_ids;
}

std::vector<int> collect_vector_trace_dofs(const basis::VectorBasisFunction& volume_basis,
                                           const elements::ReferenceElement& ref,
                                           int face_id,
                                           const std::vector<int>& face_vertices,
                                           Continuity continuity,
                                           int volume_dim) {
    const auto associations = volume_basis.dof_associations();
    FE_CHECK_ARG(associations.size() == volume_basis.size(),
                 "TraceSpace: vector-basis DOF association metadata size mismatch");

    const auto face_edges = incident_face_edges(ref, face_vertices);
    auto on_face_edge = [&](int entity_id) {
        return std::find(face_edges.begin(), face_edges.end(), entity_id) != face_edges.end();
    };

    std::vector<int> dof_indices;
    dof_indices.reserve(associations.size());

    for (std::size_t i = 0; i < associations.size(); ++i) {
        const auto& assoc = associations[i];
        bool use_dof = false;

        if (continuity == Continuity::H_div) {
            if (volume_dim == 2) {
                use_dof = assoc.entity_type == basis::DofEntity::Edge && on_face_edge(assoc.entity_id);
            } else {
                use_dof = assoc.entity_type == basis::DofEntity::Face && assoc.entity_id == face_id;
            }
        } else if (continuity == Continuity::H_curl) {
            if (volume_dim == 2) {
                use_dof = assoc.entity_type == basis::DofEntity::Edge && on_face_edge(assoc.entity_id);
            } else {
                use_dof = (assoc.entity_type == basis::DofEntity::Edge && on_face_edge(assoc.entity_id)) ||
                          (assoc.entity_type == basis::DofEntity::Face && assoc.entity_id == face_id);
            }
        }

        if (use_dof) {
            dof_indices.push_back(static_cast<int>(i));
        }
    }

    return dof_indices;
}

Vec3 reference_face_normal(ElementType volume_element_type,
                           int face_id) {
    return normalized_or_throw(
        elements::ElementTransform::reference_facet_normal(volume_element_type, face_id),
        "TraceSpace: reference face normal is degenerate");
}

Vec3 reference_edge_tangent(ElementType volume_element_type,
                            const std::vector<int>& face_vertices) {
    FE_CHECK_ARG(face_vertices.size() == 2u,
                 "TraceSpace: reference edge tangent requires an edge face");

    const Vec3 x0 =
        basis::NodeOrdering::get_node_coords(volume_element_type, static_cast<std::size_t>(face_vertices[0]));
    const Vec3 x1 =
        basis::NodeOrdering::get_node_coords(volume_element_type, static_cast<std::size_t>(face_vertices[1]));
    return normalized_or_throw(x1 - x0,
                               "TraceSpace: reference edge tangent is degenerate");
}

class ScalarComponentTraceBasis final : public basis::BasisFunction {
public:
    ScalarComponentTraceBasis(ElementType face_element_type,
                              int face_dimension,
                              int order,
                              std::shared_ptr<const basis::BasisFunction> volume_basis,
                              std::vector<int> trace_dofs,
                              Vec3 direction,
                              std::string semantic_label,
                              EmbedFaceFn embed_face)
        : face_element_type_(face_element_type)
        , face_dimension_(face_dimension)
        , order_(order)
        , volume_basis_(std::move(volume_basis))
        , trace_dofs_(std::move(trace_dofs))
        , direction_(normalized_or_throw(direction,
                                         "TraceSpace: scalar trace direction is degenerate"))
        , semantic_label_(std::move(semantic_label))
        , embed_face_(std::move(embed_face)) {
        FE_CHECK_NOT_NULL(volume_basis_.get(), "TraceSpace scalar trace basis volume_basis");
        FE_CHECK_ARG(volume_basis_->is_vector_valued(),
                     "TraceSpace scalar trace basis requires a vector-valued volume basis");
        FE_CHECK_ARG(!trace_dofs_.empty(),
                     "TraceSpace scalar trace basis requires at least one traced DOF");
    }

    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return face_element_type_; }
    int dimension() const noexcept override { return face_dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return trace_dofs_.size(); }

    std::string cache_identity() const override {
        std::ostringstream oss;
        oss << "TraceSpace::" << semantic_label_
            << "|face=" << static_cast<int>(face_element_type_)
            << "|order=" << order_
            << "|volume=" << volume_basis_->cache_identity()
            << "|dofs=";
        for (int dof : trace_dofs_) {
            oss << dof << ',';
        }
        return oss.str();
    }

    void evaluate_values(const Vec3& xi,
                         std::vector<Real>& values) const override {
        const Vec3 xi_volume = embed_face_(xi);
        std::vector<Vec3> volume_values(volume_basis_->size());
        volume_basis_->evaluate_vector_values(xi_volume, volume_values);

        values.assign(trace_dofs_.size(), Real(0));
        for (std::size_t i = 0; i < trace_dofs_.size(); ++i) {
            values[i] = volume_values[static_cast<std::size_t>(trace_dofs_[i])].dot(direction_);
        }
    }

private:
    ElementType face_element_type_;
    int face_dimension_;
    int order_;
    std::shared_ptr<const basis::BasisFunction> volume_basis_;
    std::vector<int> trace_dofs_;
    Vec3 direction_;
    std::string semantic_label_;
    EmbedFaceFn embed_face_;
};

class VectorTangentialTraceBasis final : public basis::VectorBasisFunction {
public:
    VectorTangentialTraceBasis(ElementType face_element_type,
                               int face_dimension,
                               int order,
                               std::shared_ptr<const basis::BasisFunction> volume_basis,
                               std::vector<int> trace_dofs,
                               Vec3 normal,
                               EmbedFaceFn embed_face)
        : face_element_type_(face_element_type)
        , face_dimension_(face_dimension)
        , order_(order)
        , volume_basis_(std::move(volume_basis))
        , trace_dofs_(std::move(trace_dofs))
        , normal_(normalized_or_throw(normal,
                                      "TraceSpace: tangential trace normal is degenerate"))
        , embed_face_(std::move(embed_face)) {
        FE_CHECK_NOT_NULL(volume_basis_.get(), "TraceSpace vector trace basis volume_basis");
        FE_CHECK_ARG(volume_basis_->is_vector_valued(),
                     "TraceSpace vector trace basis requires a vector-valued volume basis");
        FE_CHECK_ARG(!trace_dofs_.empty(),
                     "TraceSpace vector trace basis requires at least one traced DOF");
    }

    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return face_element_type_; }
    int dimension() const noexcept override { return face_dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return trace_dofs_.size(); }

    std::string cache_identity() const override {
        std::ostringstream oss;
        oss << "TraceSpace::TangentialVector"
            << "|face=" << static_cast<int>(face_element_type_)
            << "|order=" << order_
            << "|volume=" << volume_basis_->cache_identity()
            << "|dofs=";
        for (int dof : trace_dofs_) {
            oss << dof << ',';
        }
        return oss.str();
    }

    void evaluate_vector_values(const Vec3& xi,
                                std::vector<Vec3>& values) const override {
        const Vec3 xi_volume = embed_face_(xi);
        std::vector<Vec3> volume_values(volume_basis_->size());
        volume_basis_->evaluate_vector_values(xi_volume, volume_values);

        values.assign(trace_dofs_.size(), Vec3{});
        for (std::size_t i = 0; i < trace_dofs_.size(); ++i) {
            const Vec3& v = volume_values[static_cast<std::size_t>(trace_dofs_[i])];
            values[i] = v - normal_ * v.dot(normal_);
        }
    }

private:
    ElementType face_element_type_;
    int face_dimension_;
    int order_;
    std::shared_ptr<const basis::BasisFunction> volume_basis_;
    std::vector<int> trace_dofs_;
    Vec3 normal_;
    EmbedFaceFn embed_face_;
};

struct TensorFaceSlice {
    std::vector<int> local_to_volume_axes;
    std::vector<int> local_axis_signs;
    int fixed_axis{-1};
    int fixed_index{-1};
};

TensorFaceSlice infer_tensor_face_slice(ElementType volume_element_type,
                                        const std::vector<int>& face_vertices,
                                        const std::vector<int>& extents) {
    FE_CHECK_ARG(!face_vertices.empty(),
                 "TraceSpace: tensor face slice requires at least one face vertex");
    FE_CHECK_ARG(extents.size() >= 2u && extents.size() <= 3u,
                 "TraceSpace: tensor face slice requires 2D or 3D tensor extents");

    const int volume_dim = static_cast<int>(extents.size());
    std::vector<Vec3> coords;
    coords.reserve(face_vertices.size());
    for (int v : face_vertices) {
        coords.push_back(
            basis::NodeOrdering::get_node_coords(volume_element_type, static_cast<std::size_t>(v)));
    }

    TensorFaceSlice slice;
    for (int axis = 0; axis < volume_dim; ++axis) {
        bool fixed = true;
        const Real first = coords.front()[static_cast<std::size_t>(axis)];
        for (const auto& x : coords) {
            if (std::abs(x[static_cast<std::size_t>(axis)] - first) > kNodeTol) {
                fixed = false;
                break;
            }
        }
        if (fixed) {
            slice.fixed_axis = axis;
            slice.fixed_index = (first > Real(0))
                ? (extents[static_cast<std::size_t>(axis)] - 1)
                : 0;
            break;
        }
    }

    FE_CHECK_ARG(slice.fixed_axis >= 0,
                 "TraceSpace: failed to infer fixed tensor axis for face");

    if (face_vertices.size() == 2u) {
        const Vec3 delta = coords[1] - coords[0];
        const int axis = dominant_axis(delta, volume_dim);
        FE_CHECK_ARG(axis >= 0 && axis != slice.fixed_axis,
                     "TraceSpace: failed to infer varying tensor axis for edge trace");
        slice.local_to_volume_axes = {axis};
        slice.local_axis_signs = {delta[static_cast<std::size_t>(axis)] >= Real(0) ? 1 : -1};
        return slice;
    }

    FE_CHECK_ARG(face_vertices.size() == 4u,
                 "TraceSpace: tensor-product spline traces currently support line and quadrilateral faces only");

    const Vec3 du = coords[1] - coords[0];
    const Vec3 dv = coords[3] - coords[0];
    const int axis_u = dominant_axis(du, volume_dim);
    const int axis_v = dominant_axis(dv, volume_dim);
    FE_CHECK_ARG(axis_u >= 0 && axis_v >= 0 && axis_u != axis_v,
                 "TraceSpace: failed to infer local tensor axes for quadrilateral face");
    FE_CHECK_ARG(axis_u != slice.fixed_axis && axis_v != slice.fixed_axis,
                 "TraceSpace: quadrilateral face local axes conflict with fixed axis");

    slice.local_to_volume_axes = {axis_u, axis_v};
    slice.local_axis_signs = {
        du[static_cast<std::size_t>(axis_u)] >= Real(0) ? 1 : -1,
        dv[static_cast<std::size_t>(axis_v)] >= Real(0) ? 1 : -1
    };
    return slice;
}

std::vector<int> oriented_tensor_face_dofs(const std::vector<int>& extents,
                                           const TensorFaceSlice& slice) {
    FE_CHECK_ARG(slice.fixed_axis >= 0,
                 "TraceSpace: tensor face slice is missing a fixed axis");

    std::vector<int> coords(extents.size(), 0);
    coords[static_cast<std::size_t>(slice.fixed_axis)] = slice.fixed_index;

    std::vector<int> dofs;
    if (slice.local_to_volume_axes.size() == 1u) {
        const int axis = slice.local_to_volume_axes[0];
        const int sign = slice.local_axis_signs[0];
        const int n = extents[static_cast<std::size_t>(axis)];
        dofs.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            coords[static_cast<std::size_t>(axis)] = (sign > 0) ? i : (n - 1 - i);
            dofs.push_back(tensor_linear_index(extents, coords));
        }
        return dofs;
    }

    FE_CHECK_ARG(slice.local_to_volume_axes.size() == 2u,
                 "TraceSpace: unsupported tensor face rank");
    const int axis_u = slice.local_to_volume_axes[0];
    const int axis_v = slice.local_to_volume_axes[1];
    const int sign_u = slice.local_axis_signs[0];
    const int sign_v = slice.local_axis_signs[1];
    const int nu = extents[static_cast<std::size_t>(axis_u)];
    const int nv = extents[static_cast<std::size_t>(axis_v)];
    dofs.reserve(static_cast<std::size_t>(nu * nv));
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            coords[static_cast<std::size_t>(axis_u)] = (sign_u > 0) ? i : (nu - 1 - i);
            coords[static_cast<std::size_t>(axis_v)] = (sign_v > 0) ? j : (nv - 1 - j);
            dofs.push_back(tensor_linear_index(extents, coords));
        }
    }
    return dofs;
}

std::vector<int> local_face_extents(const std::vector<int>& extents,
                                    const TensorFaceSlice& slice) {
    std::vector<int> face_extents;
    face_extents.reserve(slice.local_to_volume_axes.size());
    for (int axis : slice.local_to_volume_axes) {
        face_extents.push_back(extents[static_cast<std::size_t>(axis)]);
    }
    return face_extents;
}

std::vector<Real> extract_nurbs_face_weights(const basis::NURBSTensorBasis& basis,
                                             const std::vector<int>& extents,
                                             const TensorFaceSlice& slice) {
    std::vector<int> coords(extents.size(), 0);
    coords[static_cast<std::size_t>(slice.fixed_axis)] = slice.fixed_index;

    const auto& weights = basis.weights();
    std::vector<Real> face_weights;
    if (slice.local_to_volume_axes.size() == 1u) {
        const int axis = slice.local_to_volume_axes[0];
        const int n = extents[static_cast<std::size_t>(axis)];
        face_weights.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            coords[static_cast<std::size_t>(axis)] = i;
            face_weights.push_back(weights[static_cast<std::size_t>(tensor_linear_index(extents, coords))]);
        }
        return face_weights;
    }

    const int axis_u = slice.local_to_volume_axes[0];
    const int axis_v = slice.local_to_volume_axes[1];
    const int nu = extents[static_cast<std::size_t>(axis_u)];
    const int nv = extents[static_cast<std::size_t>(axis_v)];
    face_weights.reserve(static_cast<std::size_t>(nu * nv));
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            coords[static_cast<std::size_t>(axis_u)] = i;
            coords[static_cast<std::size_t>(axis_v)] = j;
            face_weights.push_back(weights[static_cast<std::size_t>(tensor_linear_index(extents, coords))]);
        }
    }
    return face_weights;
}

bool try_build_tensor_spline_trace(const FunctionSpace& volume_space,
                                   const std::vector<int>& face_vertices,
                                   ElementType face_shape,
                                   std::shared_ptr<elements::Element>& face_element,
                                   std::vector<int>& face_to_volume_dof) {
    const auto& volume_basis = volume_space.element().basis();
    std::vector<int> extents;
    if (!extract_tensor_spline_extents(volume_basis, extents)) {
        return false;
    }

    const TensorFaceSlice slice =
        infer_tensor_face_slice(volume_space.element_type(), face_vertices, extents);
    face_to_volume_dof = oriented_tensor_face_dofs(extents, slice);

    std::shared_ptr<basis::BasisFunction> face_basis;
    if (const auto* spline =
            dynamic_cast<const basis::TensorProductBasis<basis::BSplineBasis>*>(&volume_basis)) {
        if (slice.local_to_volume_axes.size() == 1u) {
            face_basis = std::make_shared<basis::BSplineBasis>(
                spline->axis_basis(slice.local_to_volume_axes[0]));
        } else {
            face_basis = std::make_shared<basis::TensorProductBasis<basis::BSplineBasis>>(
                spline->axis_basis(slice.local_to_volume_axes[0]),
                spline->axis_basis(slice.local_to_volume_axes[1]));
        }
    } else if (const auto* nurbs =
                   dynamic_cast<const basis::NURBSTensorBasis*>(&volume_basis)) {
        const auto face_extents = local_face_extents(extents, slice);
        const auto face_weights = extract_nurbs_face_weights(*nurbs, extents, slice);
        if (slice.local_to_volume_axes.size() == 1u) {
            const auto& axis = nurbs->axis_basis(slice.local_to_volume_axes[0]);
            face_basis = std::make_shared<basis::BSplineBasis>(
                axis.order(), axis.knots(), face_weights);
        } else {
            face_basis = std::make_shared<basis::NURBSTensorBasis>(
                nurbs->axis_basis(slice.local_to_volume_axes[0]),
                nurbs->axis_basis(slice.local_to_volume_axes[1]),
                face_weights,
                face_extents);
        }
    } else {
        return false;
    }

    FE_CHECK_NOT_NULL(face_basis.get(),
                      "TraceSpace: failed to create tensor-product spline face basis");
    const int qord = scalar_trace_quadrature_order(face_basis->basis_type(), face_basis->order());
    auto face_quad = quadrature::QuadratureFactory::create(
        face_shape, qord, QuadratureType::GaussLegendre, true);
    face_element = std::make_shared<elements::GeneralBasisElement>(
        std::move(face_basis), std::move(face_quad), FieldType::Scalar, volume_space.continuity());
    return true;
}

bool try_build_vector_trace(const FunctionSpace& volume_space,
                            int face_id,
                            const std::vector<int>& face_vertices,
                            ElementType face_shape,
                            const std::shared_ptr<const geometry::GeometryMapping>& face_embedding,
                            std::shared_ptr<elements::Element>& face_element,
                            std::vector<int>& face_to_volume_dof,
                            TraceKind& trace_kind,
                            FieldType& trace_field_type,
                            int& trace_value_dimension,
                            Continuity& trace_continuity,
                            Vec3& face_normal,
                            Vec3& face_tangent1) {
    if (volume_space.field_type() != FieldType::Vector) {
        return false;
    }

    if (volume_space.continuity() != Continuity::H_div &&
        volume_space.continuity() != Continuity::H_curl) {
        return false;
    }

    FE_CHECK_NOT_NULL(face_embedding.get(),
                      "TraceSpace: vector trace requires a valid face embedding");

    auto vector_basis = std::dynamic_pointer_cast<const basis::VectorBasisFunction>(
        volume_space.element_ptr()->basis_ptr());
    FE_CHECK_NOT_NULL(vector_basis.get(),
                      "TraceSpace: vector volume space must expose a vector basis");

    const elements::ReferenceElement ref =
        elements::ReferenceElement::create(volume_space.element_type());
    const int volume_dim = volume_space.topological_dimension();
    face_to_volume_dof = collect_vector_trace_dofs(*vector_basis,
                                                   ref,
                                                   face_id,
                                                   face_vertices,
                                                   volume_space.continuity(),
                                                   volume_dim);
    FE_CHECK_ARG(!face_to_volume_dof.empty(),
                 "TraceSpace: selected vector face has no trace DOFs");

    const Vec3 normal = reference_face_normal(volume_space.element_type(), face_id);
    const EmbedFaceFn embed_face = [face_embedding](const Vec3& xi_face) {
        return face_embedding->map_to_physical(xi_face);
    };

    const int qord =
        quadrature::QuadratureFactory::recommended_order(volume_space.polynomial_order(),
                                                         /*is_mass_matrix=*/true);
    auto face_quad = quadrature::QuadratureFactory::create(
        face_shape, qord, QuadratureType::GaussLegendre, true);

    if (volume_space.continuity() == Continuity::H_div) {
        auto face_basis = std::make_shared<ScalarComponentTraceBasis>(
            face_shape,
            volume_dim - 1,
            volume_space.polynomial_order(),
            vector_basis,
            face_to_volume_dof,
            normal,
            "NormalScalar",
            embed_face);
        face_element = std::make_shared<elements::GeneralBasisElement>(
            std::move(face_basis), std::move(face_quad), FieldType::Scalar, Continuity::C0);
        trace_kind = TraceKind::Normal;
        trace_field_type = FieldType::Scalar;
        trace_value_dimension = 1;
        trace_continuity = Continuity::C0;
        face_normal = normal;
        face_tangent1 = Vec3{};
        return true;
    }

    FE_CHECK_ARG(volume_space.continuity() == Continuity::H_curl,
                 "TraceSpace: unsupported vector trace continuity");

    if (volume_dim == 2) {
        const Vec3 tangent = reference_edge_tangent(volume_space.element_type(), face_vertices);
        auto face_basis = std::make_shared<ScalarComponentTraceBasis>(
            face_shape,
            1,
            volume_space.polynomial_order(),
            vector_basis,
            face_to_volume_dof,
            tangent,
            "TangentialScalar",
            embed_face);
        face_element = std::make_shared<elements::GeneralBasisElement>(
            std::move(face_basis), std::move(face_quad), FieldType::Scalar, Continuity::C0);
        trace_kind = TraceKind::Tangential;
        trace_field_type = FieldType::Scalar;
        trace_value_dimension = 1;
        trace_continuity = Continuity::C0;
        face_normal = normal;
        face_tangent1 = tangent;
        return true;
    }

    auto face_basis = std::make_shared<VectorTangentialTraceBasis>(
        face_shape,
        volume_dim - 1,
        volume_space.polynomial_order(),
        vector_basis,
        face_to_volume_dof,
        normal,
        embed_face);
    face_element = std::make_shared<elements::GeneralBasisElement>(
        std::move(face_basis), std::move(face_quad), FieldType::Vector, Continuity::H_curl);
    trace_kind = TraceKind::Tangential;
    trace_field_type = FieldType::Vector;
    trace_value_dimension = 3;
    trace_continuity = Continuity::H_curl;
    face_normal = normal;
    face_tangent1 = Vec3{};
    return true;
}

struct FacePrototypeSpec {
    ElementType element_type{ElementType::Unknown};
    int order{0};
    BasisType basis_type{BasisType::Lagrange};
};

FacePrototypeSpec infer_face_prototype(ElementType face_shape,
                                       std::size_t num_face_dofs,
                                       int volume_order) {
    FacePrototypeSpec spec;

    if (face_shape == ElementType::Point1) {
        spec.element_type = ElementType::Point1;
        spec.order = 0;
        spec.basis_type = BasisType::Lagrange;
        return spec;
    }

    if (volume_order < 0) {
        throw FEException("TraceSpace: negative polynomial order is not allowed",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    if (face_shape == ElementType::Line2) {
        const std::size_t expected = static_cast<std::size_t>(volume_order + 1);
        if (num_face_dofs != expected) {
            throw FEException("TraceSpace: face DOF count does not match expected Line p+1",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        spec.order = volume_order;
        spec.element_type = (volume_order == 2) ? ElementType::Line3 : ElementType::Line2;
        spec.basis_type = BasisType::Lagrange;
        return spec;
    }

    if (face_shape == ElementType::Triangle3) {
        const std::size_t expected =
            static_cast<std::size_t>((volume_order + 1) * (volume_order + 2) / 2);
        if (num_face_dofs != expected) {
            throw FEException("TraceSpace: face DOF count does not match expected Triangle size",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        spec.order = volume_order;
        spec.element_type = (volume_order == 2) ? ElementType::Triangle6 : ElementType::Triangle3;
        spec.basis_type = BasisType::Lagrange;
        return spec;
    }

    if (face_shape == ElementType::Quad4) {
        const std::size_t expected_lagrange =
            static_cast<std::size_t>((volume_order + 1) * (volume_order + 1));
        const std::size_t expected_serendipity =
            volume_order >= 1 ? quad_serendipity_dof_count(volume_order) : 0u;

        if (num_face_dofs == expected_lagrange) {
            spec.order = volume_order;
            spec.element_type = (volume_order == 2) ? ElementType::Quad9 : ElementType::Quad4;
            spec.basis_type = BasisType::Lagrange;
            return spec;
        }

        if (volume_order >= 2 && num_face_dofs == expected_serendipity) {
            spec.order = volume_order;
            spec.element_type = (volume_order == 2) ? ElementType::Quad8 : ElementType::Quad4;
            spec.basis_type = BasisType::Serendipity;
            return spec;
        }

        throw FEException("TraceSpace: face DOF count does not match expected Quad size",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    throw FEException("TraceSpace: unsupported face shape for prototype element",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
}

ElementType embedding_element_type_for_vertices(std::size_t nverts) {
    switch (nverts) {
        case 1: return ElementType::Point1;
        case 2: return ElementType::Line2;
        case 3: return ElementType::Triangle3;
        case 4: return ElementType::Quad4;
        default: return ElementType::Unknown;
    }
}

std::vector<Vec3> face_reference_nodes(const FacePrototypeSpec& spec) {
    std::vector<Vec3> nodes;

    if (spec.basis_type == BasisType::Serendipity) {
        basis::SerendipityBasis basis(spec.element_type, spec.order, false);
        const auto& bnodes = basis.nodes();
        nodes.assign(bnodes.begin(), bnodes.end());
        return nodes;
    }

    basis::LagrangeBasis basis(spec.element_type, spec.order);
    const auto& bnodes = basis.nodes();
    nodes.assign(bnodes.begin(), bnodes.end());
    return nodes;
}

int find_matching_dof(const Vec3& x,
                      const std::vector<Vec3>& volume_nodes,
                      const std::vector<int>& candidates) {
    int match = -1;
    for (int idx : candidates) {
        if (idx < 0) {
            continue;
        }
        const auto& xn = volume_nodes[static_cast<std::size_t>(idx)];
        if (xn.approx_equal(x, kNodeTol)) {
            if (match != -1) {
                throw FEException("TraceSpace: face node matched multiple volume DOFs",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            match = idx;
        }
    }
    return match;
}

} // namespace

TraceSpace::TraceSpace(std::shared_ptr<FunctionSpace> volume_space,
                       int face_id)
    : volume_space_(std::move(volume_space)),
      face_id_(face_id) {
    FE_CHECK_NOT_NULL(volume_space_.get(), "TraceSpace volume_space");
    FE_CHECK_ARG(volume_space_->topological_dimension() > 0,
                 "TraceSpace requires a volume space with positive topological dimension");

    const elements::ReferenceElement ref =
        elements::ReferenceElement::create(volume_space_->element_type());
    FE_CHECK_ARG(face_id >= 0 && face_id < static_cast<int>(ref.num_faces()),
                 "Invalid face_id for this element type");

    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(face_id_));
    std::vector<int> fverts;
    fverts.reserve(face_nodes.size());
    for (auto v : face_nodes) {
        fverts.push_back(static_cast<int>(v));
    }

    const ElementType face_shape = embedding_element_type_for_vertices(fverts.size());
    FE_CHECK_ARG(face_shape != ElementType::Unknown,
                 "TraceSpace: unsupported face vertex count");

    // Build reference-face embedding mapping into the volume reference element.
    std::shared_ptr<basis::BasisFunction> embed_basis;
    int embed_order = (face_shape == ElementType::Point1) ? 0 : 1;
    embed_basis = std::make_shared<basis::LagrangeBasis>(face_shape, embed_order);

    std::vector<Vec3> embed_nodes;
    embed_nodes.reserve(fverts.size());
    for (int v : fverts) {
        embed_nodes.push_back(
            basis::NodeOrdering::get_node_coords(volume_space_->element_type(),
                                                 static_cast<std::size_t>(v)));
    }
    face_embedding_ = std::make_shared<geometry::IsoparametricMapping>(std::move(embed_basis),
                                                                       std::move(embed_nodes));

    if (try_build_vector_trace(*volume_space_,
                               face_id_,
                               fverts,
                               face_shape,
                               face_embedding_,
                               face_element_,
                               face_to_volume_dof_,
                               trace_kind_,
                               trace_field_type_,
                               trace_value_dimension_,
                               trace_continuity_,
                               face_normal_,
                               face_tangent1_)) {
        face_element_type_ = face_shape;
        return;
    }

    FE_CHECK_ARG(volume_space_->field_type() == FieldType::Scalar,
                 "TraceSpace currently supports scalar spaces and vector H(div)/H(curl) spaces only");

    trace_kind_ = TraceKind::Value;
    trace_field_type_ = FieldType::Scalar;
    trace_value_dimension_ = 1;
    trace_continuity_ = volume_space_->continuity();

    // Initialize face restriction using factory (cached) for scalar trace spaces.
    restriction_ = FaceRestrictionFactory::get(*volume_space_);

    FE_CHECK_ARG(face_id_ < static_cast<int>(restriction_->topology().num_faces),
                 "TraceSpace: scalar face restriction topology mismatch");

    const auto num_face_dofs = restriction_->num_face_dofs(face_id_);
    const bool tensor_spline_trace =
        try_build_tensor_spline_trace(*volume_space_, fverts, face_shape, face_element_, face_to_volume_dof_);

    FacePrototypeSpec spec;
    if (tensor_spline_trace) {
        face_element_type_ = face_shape;
    } else {
        spec = infer_face_prototype(face_shape, num_face_dofs, volume_space_->polynomial_order());
        face_element_type_ = spec.element_type;

        if (spec.basis_type == BasisType::Serendipity) {
            auto face_basis =
                std::make_shared<basis::SerendipityBasis>(spec.element_type, spec.order, false);
            const int qord =
                quadrature::QuadratureFactory::recommended_order(spec.order, /*is_mass_matrix=*/true);
            auto face_quad = quadrature::QuadratureFactory::create(
                spec.element_type, qord, QuadratureType::GaussLegendre, true);
            face_element_ = std::make_shared<elements::GeneralBasisElement>(
                std::move(face_basis), std::move(face_quad), FieldType::Scalar, trace_continuity_);
        } else {
            face_element_ = std::make_shared<elements::LagrangeElement>(
                spec.element_type, spec.order, FieldType::Scalar, trace_continuity_);
        }
    }

    FE_CHECK_NOT_NULL(face_element_.get(), "TraceSpace face_element");
    FE_CHECK_ARG(face_element_->num_dofs() == num_face_dofs,
                 "TraceSpace: face element DOF count does not match FaceRestriction");

    if (tensor_spline_trace) {
        return;
    }

    // Build mapping from face basis DOFs to volume DOF indices.
    const auto face_reference_dofs = face_reference_nodes(spec);
    const auto& vol_nodes = restriction_->dof_nodes();
    const auto candidates = restriction_->face_dofs(face_id_);

    face_to_volume_dof_.assign(face_reference_dofs.size(), -1);
    for (std::size_t i = 0; i < face_reference_dofs.size(); ++i) {
        const Vec3 xi_face = face_reference_dofs[i];
        const Vec3 xi_vol = embed_face_point(xi_face);
        const int match = find_matching_dof(xi_vol, vol_nodes, candidates);
        if (match == -1) {
            throw FEException("TraceSpace: failed to match face DOF to a volume DOF",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        face_to_volume_dof_[i] = match;
    }
}

std::size_t TraceSpace::dofs_per_element() const noexcept {
    return face_to_volume_dof_.size();
}

std::vector<int> TraceSpace::face_dof_indices() const {
    return face_to_volume_dof_;
}

std::vector<Real> TraceSpace::restrict(const std::vector<Real>& element_values) const {
    FE_CHECK_ARG(element_values.size() >= volume_space_->dofs_per_element(),
                 "TraceSpace::restrict: element_values size mismatch");

    std::vector<Real> out(dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < out.size(); ++i) {
        const int vidx = face_to_volume_dof_[i];
        out[i] = element_values[static_cast<std::size_t>(vidx)];
    }
    return out;
}

FunctionSpace::Value TraceSpace::evaluate(const Value& xi,
                                          const std::vector<Real>& coefficients) const {
    return FunctionSpace::evaluate(xi, coefficients);
}

void TraceSpace::interpolate(const ValueFunction& function,
                             std::vector<Real>& coefficients) const {
    FE_CHECK_NOT_NULL(face_element_.get(), "TraceSpace face_element");

    const auto& basis = face_element_->basis();
    const std::size_t ndofs = face_element_->num_dofs();

    if (ndofs == 0) {
        coefficients.clear();
        return;
    }

    if (basis.is_vector_valued()) {
        // Generic fallback: L² projection on the face prototype element.
        FunctionSpace::interpolate(function, coefficients);
        return;
    }

    std::vector<Vec3> nodes;
    const auto btype = basis.basis_type();

    if (btype == BasisType::Lagrange) {
        const auto* lag = dynamic_cast<const basis::LagrangeBasis*>(&basis);
        if (!lag) {
            FunctionSpace::interpolate(function, coefficients);
            return;
        }
        const auto& bnodes = lag->nodes();
        nodes.assign(bnodes.begin(), bnodes.end());
    } else if (btype == BasisType::Serendipity) {
        const auto* ser = dynamic_cast<const basis::SerendipityBasis*>(&basis);
        if (!ser) {
            FunctionSpace::interpolate(function, coefficients);
            return;
        }
        const auto& bnodes = ser->nodes();
        nodes.assign(bnodes.begin(), bnodes.end());
    } else {
        FunctionSpace::interpolate(function, coefficients);
        return;
    }

    FE_CHECK_ARG(nodes.size() == ndofs,
                 "TraceSpace::interpolate: nodal coordinate count mismatch");

    coefficients.resize(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        coefficients[i] = function(nodes[i])[0];
    }
}

void TraceSpace::scatter(const std::vector<Real>& face_values,
                         std::vector<Real>& element_values) const {
    FE_CHECK_ARG(face_values.size() == dofs_per_element(),
                 "TraceSpace::scatter: face_values size mismatch");
    FE_CHECK_ARG(element_values.size() >= volume_space_->dofs_per_element(),
                 "TraceSpace::scatter: element_values size mismatch");

    for (std::size_t i = 0; i < face_values.size(); ++i) {
        const int vidx = face_to_volume_dof_[i];
        element_values[static_cast<std::size_t>(vidx)] += face_values[i];
    }
}

std::vector<Real> TraceSpace::lift(const std::vector<Real>& face_coefficients) const {
    FE_CHECK_ARG(face_coefficients.size() == dofs_per_element(),
                 "TraceSpace::lift: face coefficient size mismatch");

    std::vector<Real> element_coeffs(volume_space_->dofs_per_element(), Real(0));
    scatter(face_coefficients, element_coeffs);
    return element_coeffs;
}

FunctionSpace::Value TraceSpace::embed_face_point(const Value& xi_face) const {
    FE_CHECK_NOT_NULL(face_embedding_.get(), "TraceSpace face_embedding");
    return face_embedding_->map_to_physical(xi_face);
}

FunctionSpace::Value TraceSpace::evaluate_from_face(
    const Value& xi_volume,
    const std::vector<Real>& face_coefficients) const {

    const auto element_coeffs = lift(face_coefficients);
    const auto volume_value = volume_space_->evaluate(xi_volume, element_coeffs);

    if (trace_kind_ == TraceKind::Value) {
        return volume_value;
    }

    if (trace_kind_ == TraceKind::Normal) {
        Value result{};
        result[0] = volume_value.dot(face_normal_);
        return result;
    }

    if (trace_field_type_ == FieldType::Scalar) {
        Value result{};
        result[0] = volume_value.dot(face_tangent1_);
        return result;
    }

    const Real vn = volume_value.dot(face_normal_);
    return volume_value - vn * face_normal_;
}

const FaceRestriction& TraceSpace::face_restriction() const {
    FE_CHECK_NOT_NULL(restriction_.get(),
                      "TraceSpace::face_restriction is only available for scalar trace spaces");
    return *restriction_;
}

} // namespace spaces
} // namespace FE
} // namespace svmp
