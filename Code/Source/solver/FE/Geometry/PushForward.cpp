/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "PushForward.h"

namespace svmp {
namespace FE {
namespace geometry {

math::Vector<Real, 3> PushForward::gradient(const GeometryMapping& mapping,
                                            const math::Vector<Real, 3>& grad_ref,
                                            const math::Vector<Real, 3>& xi) {
    return mapping.transform_gradient(grad_ref, xi);
}

math::Vector<Real, 3> PushForward::hdiv_vector(const GeometryMapping& mapping,
                                               const math::Vector<Real, 3>& v_ref,
                                               const math::Vector<Real, 3>& xi) {
    auto J = mapping.jacobian(xi);
    const Real det = mapping.jacobian_determinant(xi);
    math::Vector<Real, 3> v{};
    const std::size_t dim = static_cast<std::size_t>(mapping.dimension());
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            v[i] += J(i, j) * v_ref[j];
        }
        v[i] /= det;
    }
    return v;
}

math::Vector<Real, 3> PushForward::hcurl_vector(const GeometryMapping& mapping,
                                               const math::Vector<Real, 3>& v_ref,
                                               const math::Vector<Real, 3>& xi) {
    auto JinvT = mapping.jacobian_inverse(xi).transpose();
    math::Vector<Real, 3> v{};
    const std::size_t dim = static_cast<std::size_t>(mapping.dimension());
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            v[i] += JinvT(i, j) * v_ref[j];
        }
    }
    return v;
}

} // namespace geometry
} // namespace FE
} // namespace svmp
