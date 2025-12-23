/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/TraceSpace.h"

#include "Basis/LagrangeBasis.h"
#include "Basis/SerendipityBasis.h"
#include "Basis/NodeOrderingConventions.h"
#include "Core/FEException.h"
#include "Elements/IsogeometricElement.h"
#include "Elements/LagrangeElement.h"
#include "Geometry/IsoparametricMapping.h"
#include "Quadrature/QuadratureFactory.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace spaces {

namespace {

using Vec3 = math::Vector<Real, 3>;

constexpr Real kNodeTol = Real(1e-12);

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

        if (num_face_dofs == expected_lagrange) {
            spec.order = volume_order;
            spec.element_type = (volume_order == 2) ? ElementType::Quad9 : ElementType::Quad4;
            spec.basis_type = BasisType::Lagrange;
            return spec;
        }

        // Serendipity quadratic face (Quad8)
        if (volume_order == 2 && num_face_dofs == 8u) {
            spec.order = 2;
            spec.element_type = ElementType::Quad8;
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
        const std::size_t nn = basis::NodeOrdering::num_nodes(spec.element_type);
        nodes.reserve(nn);
        for (std::size_t i = 0; i < nn; ++i) {
            nodes.push_back(basis::NodeOrdering::get_node_coords(spec.element_type, i));
        }
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
    FE_CHECK_ARG(volume_space_->field_type() == FieldType::Scalar,
                 "TraceSpace currently supports scalar volume spaces only");

    // Initialize face restriction using factory (cached)
    restriction_ = FaceRestrictionFactory::get(*volume_space_);

    FE_CHECK_ARG(face_id >= 0 &&
                 face_id < static_cast<int>(restriction_->topology().num_faces),
        "Invalid face_id for this element type");

    const auto& topo = restriction_->topology();
    const auto face_shape = topo.face_types[static_cast<std::size_t>(face_id_)];
    const std::size_t num_face_dofs = restriction_->num_face_dofs(face_id_);

    const FacePrototypeSpec spec =
        infer_face_prototype(face_shape, num_face_dofs, volume_space_->polynomial_order());

    face_element_type_ = spec.element_type;

    if (spec.basis_type == BasisType::Serendipity) {
        auto face_basis = std::make_shared<basis::SerendipityBasis>(spec.element_type, spec.order, false);
        const int qord = quadrature::QuadratureFactory::recommended_order(spec.order, /*is_mass_matrix=*/true);
        auto face_quad = quadrature::QuadratureFactory::create(spec.element_type, qord, QuadratureType::GaussLegendre, true);
        face_element_ = std::make_shared<elements::IsogeometricElement>(
            std::move(face_basis), std::move(face_quad), FieldType::Scalar, volume_space_->continuity());
    } else {
        face_element_ = std::make_shared<elements::LagrangeElement>(
            spec.element_type, spec.order, FieldType::Scalar, volume_space_->continuity());
    }

    FE_CHECK_NOT_NULL(face_element_.get(), "TraceSpace face_element");
    FE_CHECK_ARG(face_element_->num_dofs() == num_face_dofs,
                 "TraceSpace: face element DOF count does not match FaceRestriction");

    // Build reference-face embedding mapping into the volume reference element.
    const auto& fverts = topo.face_vertices[static_cast<std::size_t>(face_id_)];
    const ElementType embed_type = embedding_element_type_for_vertices(fverts.size());
    FE_CHECK_ARG(embed_type != ElementType::Unknown,
                 "TraceSpace: unsupported face vertex count for embedding");

    std::shared_ptr<basis::BasisFunction> embed_basis;
    int embed_order = 1;
    if (embed_type == ElementType::Point1) {
        embed_order = 0;
    }
    embed_basis = std::make_shared<basis::LagrangeBasis>(embed_type, embed_order);

    std::vector<Vec3> embed_nodes;
    embed_nodes.reserve(fverts.size());
    for (int v : fverts) {
        embed_nodes.push_back(basis::NodeOrdering::get_node_coords(volume_space_->element_type(),
                                                                  static_cast<std::size_t>(v)));
    }
    face_embedding_ = std::make_shared<geometry::IsoparametricMapping>(std::move(embed_basis),
                                                                       std::move(embed_nodes));

    // Build mapping from face basis DOFs to volume DOF indices.
    const auto face_nodes = face_reference_nodes(spec);
    const auto& vol_nodes = restriction_->dof_nodes();
    const auto candidates = restriction_->face_dofs(face_id_);

    face_to_volume_dof_.assign(face_nodes.size(), -1);
    for (std::size_t i = 0; i < face_nodes.size(); ++i) {
        const Vec3 xi_face = face_nodes[i];
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
        const std::size_t nn = basis::NodeOrdering::num_nodes(face_element_type_);
        FE_CHECK_ARG(nn == ndofs,
                     "TraceSpace::interpolate: serendipity node count mismatch");
        nodes.reserve(nn);
        for (std::size_t i = 0; i < nn; ++i) {
            nodes.push_back(basis::NodeOrdering::get_node_coords(face_element_type_, i));
        }
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

    return volume_space_->evaluate(xi_volume, element_coeffs);
}

const FaceRestriction& TraceSpace::face_restriction() const {
    return *restriction_;
}

} // namespace spaces
} // namespace FE
} // namespace svmp
