/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/SpaceInterpolation.h"

namespace svmp {
namespace FE {
namespace spaces {

void SpaceInterpolation::l2_projection(const FunctionSpace& src_space,
                                       const std::vector<Real>& src_coeffs,
                                       const FunctionSpace& dst_space,
                                       std::vector<Real>& dst_coeffs) {
    // Define reference-space function via source evaluation
    FunctionSpace::ValueFunction f = [&src_space, &src_coeffs](const FunctionSpace::Value& xi) {
        return src_space.evaluate(xi, src_coeffs);
    };
    dst_space.interpolate(f, dst_coeffs);
}

void SpaceInterpolation::nodal_interpolation(const FunctionSpace& src_space,
                                             const std::vector<Real>& src_coeffs,
                                             const FunctionSpace& dst_space,
                                             std::vector<Real>& dst_coeffs) {
    FE_CHECK_ARG(src_space.field_type() == FieldType::Scalar,
                 "SpaceInterpolation::nodal_interpolation: source must be scalar");
    FE_CHECK_ARG(dst_space.field_type() == FieldType::Scalar,
                 "SpaceInterpolation::nodal_interpolation: destination must be scalar");

    // Attempt to access LagrangeBasis for both spaces
    const auto& src_basis = src_space.element().basis();
    const auto& dst_basis = dst_space.element().basis();

    auto src_lagrange = dynamic_cast<const basis::LagrangeBasis*>(&src_basis);
    auto dst_lagrange = dynamic_cast<const basis::LagrangeBasis*>(&dst_basis);
    FE_CHECK_ARG(src_lagrange != nullptr && dst_lagrange != nullptr,
                 "SpaceInterpolation::nodal_interpolation: requires LagrangeBasis");

    const auto& dst_nodes = dst_lagrange->nodes();
    dst_coeffs.resize(dst_nodes.size());

    // For scalar Lagrange spaces, DOFs are nodal values
    for (std::size_t i = 0; i < dst_nodes.size(); ++i) {
        const auto& xi = dst_nodes[i];
        const auto val = src_space.evaluate(xi, src_coeffs);
        dst_coeffs[i] = val[0];
    }
}

} // namespace spaces
} // namespace FE
} // namespace svmp

