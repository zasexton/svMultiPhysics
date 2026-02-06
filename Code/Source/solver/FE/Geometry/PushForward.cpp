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

math::Vector<Real, 3> PushForward::gradient(const GeometryMapping& mapping,
                                            const math::Vector<Real, 3>& grad_ref,
                                            const math::Matrix<Real, 3, 3>& jacobian_inverse) {
    return mapping.transform_gradient(grad_ref, jacobian_inverse);
}

math::Vector<Real, 3> PushForward::hdiv_vector(const GeometryMapping& mapping,
                                               const math::Vector<Real, 3>& v_ref,
                                               const math::Vector<Real, 3>& xi) {
    auto J = mapping.jacobian(xi);
    const Real det = J.determinant();
    return hdiv_vector(mapping, v_ref, J, det);
}

math::Vector<Real, 3> PushForward::hdiv_vector(const GeometryMapping& mapping,
                                               const math::Vector<Real, 3>& v_ref,
                                               const math::Matrix<Real, 3, 3>& jacobian,
                                               Real det_jacobian) {
    math::Vector<Real, 3> v{};
    const std::size_t dim = static_cast<std::size_t>(mapping.dimension());
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            v[i] += jacobian(i, j) * v_ref[j];
        }
        v[i] /= det_jacobian;
    }
    return v;
}

math::Vector<Real, 3> PushForward::hcurl_vector(const GeometryMapping& mapping,
                                               const math::Vector<Real, 3>& v_ref,
                                               const math::Vector<Real, 3>& xi) {
    return hcurl_vector(mapping, v_ref, mapping.jacobian_inverse(xi));
}

math::Vector<Real, 3> PushForward::hcurl_vector(const GeometryMapping& mapping,
                                                const math::Vector<Real, 3>& v_ref,
                                                const math::Matrix<Real, 3, 3>& jacobian_inverse) {
    auto JinvT = jacobian_inverse.transpose();
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
