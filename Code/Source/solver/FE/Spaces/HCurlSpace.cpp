/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/HCurlSpace.h"
#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace spaces {

HCurlSpace::HCurlSpace(ElementType element_type,
                       int order)
    : HCurlSpace(element_type, order, BasisType::Lagrange) {}

HCurlSpace::HCurlSpace(ElementType element_type,
                       int order,
                       BasisType basis_type)
    : element_type_(element_type),
      order_(order) {
    FE_CHECK_ARG(order_ >= 0, "HCurlSpace requires non-negative polynomial order");

    dimension_ = element_dimension(element_type_);
    FE_CHECK_ARG(dimension_ > 0, "HCurlSpace: invalid element dimension");

    if (basis_type == BasisType::Lagrange || basis_type == BasisType::Nedelec) {
        element_ = std::make_shared<elements::VectorElement>(
            element_type_, order_, Continuity::H_curl, basis_type);
    } else {
        elements::ElementRequest req;
        req.element_type = element_type_;
        req.basis_type = basis_type;
        req.field_type = FieldType::Vector;
        req.continuity = Continuity::H_curl;
        req.order = order_;
        element_ = elements::ElementFactory::create(req);
    }
}

HCurlSpace::HCurlSpace(const elements::ElementRequest& request) {
    FE_CHECK_ARG(request.field_type == FieldType::Vector,
                 "HCurlSpace request requires FieldType::Vector");
    FE_CHECK_ARG(request.continuity == Continuity::H_curl || request.continuity == Continuity::C0,
                 "HCurlSpace request requires Continuity::H_curl or inference-compatible C0");

    element_ = elements::ElementFactory::create(request);
    FE_CHECK_NOT_NULL(element_.get(), "HCurlSpace request element");
    FE_CHECK_ARG(element_->field_type() == FieldType::Vector,
                 "HCurlSpace request did not create a vector element");
    FE_CHECK_ARG(element_->continuity() == Continuity::H_curl,
                 "HCurlSpace request did not create an H(curl) element");

    element_type_ = element_->element_type();
    order_ = element_->polynomial_order();
    dimension_ = element_dimension(element_type_);
    FE_CHECK_ARG(dimension_ > 0, "HCurlSpace: invalid request element dimension");
}

std::vector<HCurlSpace::Vec3> HCurlSpace::tangential_trace(
    const std::vector<Real>& dof_values,
    const std::vector<Vec3>& eval_points,
    const Vec3& face_normal) const {

    // Evaluate the vector field at each point
    std::vector<Vec3> field_values(eval_points.size());

    for (std::size_t i = 0; i < eval_points.size(); ++i) {
        Value val = evaluate(eval_points[i], dof_values);
        field_values[i] = Vec3{val[0], val[1], val[2]};
    }

    // Compute tangential trace using VectorComponentExtractor
    return VectorComponentExtractor::hcurl_tangential_trace(field_values, face_normal);
}

std::vector<Real> HCurlSpace::apply_edge_orientation(
    const std::vector<Real>& edge_dofs,
    OrientationManager::Sign orientation) {

    return OrientationManager::orient_hcurl_edge_dofs(edge_dofs, orientation);
}

std::vector<Real> HCurlSpace::apply_face_orientation(
    ElementType face_type,
    const std::vector<Real>& face_dofs,
    const OrientationManager::FaceOrientation& orientation,
    int face_poly_order) {

    switch (face_type) {
        case ElementType::Quad4:
            return OrientationManager::orient_hcurl_quad_face_dofs(face_dofs, orientation, face_poly_order);
        case ElementType::Triangle3:
            return OrientationManager::orient_hcurl_triangle_face_dofs(face_dofs, orientation, face_poly_order);
        default:
            throw FEException("HCurlSpace::apply_face_orientation: unsupported face type",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
}

} // namespace spaces
} // namespace FE
} // namespace svmp
