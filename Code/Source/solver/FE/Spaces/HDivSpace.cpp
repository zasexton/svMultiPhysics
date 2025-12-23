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
    : element_type_(element_type),
      order_(order) {
    FE_CHECK_ARG(order_ >= 0, "HDivSpace requires non-negative polynomial order");

    dimension_ = element_dimension(element_type_);
    FE_CHECK_ARG(dimension_ > 0, "HDivSpace: invalid element dimension");

    element_ = std::make_shared<elements::VectorElement>(
        element_type_, order_, Continuity::H_div);
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
