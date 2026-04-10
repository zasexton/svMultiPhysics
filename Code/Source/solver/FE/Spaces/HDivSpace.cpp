/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/HDivSpace.h"
#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace spaces {

HDivSpace::HDivSpace(ElementType element_type,
                     int order)
    : HDivSpace(element_type, order, BasisType::Lagrange) {}

HDivSpace::HDivSpace(ElementType element_type,
                     int order,
                     BasisType basis_type)
    : element_type_(element_type),
      order_(order) {
    FE_CHECK_ARG(order_ >= 0, "HDivSpace requires non-negative polynomial order");

    dimension_ = element_dimension(element_type_);
    FE_CHECK_ARG(dimension_ > 0, "HDivSpace: invalid element dimension");

    if (basis_type == BasisType::Lagrange ||
        basis_type == BasisType::RaviartThomas ||
        basis_type == BasisType::BDM) {
        element_ = std::make_shared<elements::VectorElement>(
            element_type_, order_, Continuity::H_div, basis_type);
    } else {
        elements::ElementRequest req;
        req.element_type = element_type_;
        req.basis_type = basis_type;
        req.field_type = FieldType::Vector;
        req.continuity = Continuity::H_div;
        req.order = order_;
        element_ = elements::ElementFactory::create(req);
    }
}

HDivSpace::HDivSpace(const elements::ElementRequest& request) {
    FE_CHECK_ARG(request.field_type == FieldType::Vector,
                 "HDivSpace request requires FieldType::Vector");
    FE_CHECK_ARG(request.continuity == Continuity::H_div || request.continuity == Continuity::C0,
                 "HDivSpace request requires Continuity::H_div or inference-compatible C0");

    element_ = elements::ElementFactory::create(request);
    FE_CHECK_NOT_NULL(element_.get(), "HDivSpace request element");
    FE_CHECK_ARG(element_->field_type() == FieldType::Vector,
                 "HDivSpace request did not create a vector element");
    FE_CHECK_ARG(element_->continuity() == Continuity::H_div,
                 "HDivSpace request did not create an H(div) element");

    element_type_ = element_->element_type();
    order_ = element_->polynomial_order();
    dimension_ = element_dimension(element_type_);
    FE_CHECK_ARG(dimension_ > 0, "HDivSpace: invalid request element dimension");
}

std::vector<Real> HDivSpace::normal_trace(
    const std::vector<Real>& dof_values,
    const std::vector<Vec3>& eval_points,
    const Vec3& face_normal) const {

    // Evaluate the vector field at each point using the underlying element
    std::vector<Vec3> field_values(eval_points.size());

    for (std::size_t i = 0; i < eval_points.size(); ++i) {
        Value val = evaluate(eval_points[i], dof_values);
        field_values[i] = Vec3{val[0], val[1], val[2]};
    }

    // Compute normal trace using VectorComponentExtractor
    return VectorComponentExtractor::hdiv_normal_trace(field_values, face_normal);
}

std::vector<Real> HDivSpace::apply_face_orientation(
    ElementType face_type,
    const std::vector<Real>& face_dofs,
    const OrientationManager::FaceOrientation& orientation,
    int face_poly_order) {

    switch (face_type) {
        case ElementType::Triangle3:
            return OrientationManager::orient_triangle_face_dofs(face_dofs, orientation, face_poly_order);
        case ElementType::Quad4:
            return OrientationManager::orient_quad_face_dofs(face_dofs, orientation, face_poly_order);
        default:
            throw FEException("HDivSpace::apply_face_orientation: unsupported face type",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
}

} // namespace spaces
} // namespace FE
} // namespace svmp
