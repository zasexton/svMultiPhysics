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

math::Matrix<Real, 3, 3> PushForward::vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Vector<Real, 3>& xi)
{
    return vector_jacobian(mapping, jac_ref, mapping.jacobian_inverse(xi));
}

math::Matrix<Real, 3, 3> PushForward::vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Matrix<Real, 3, 3>& jacobian_inverse)
{
    const std::size_t dim = static_cast<std::size_t>(mapping.dimension());
    math::Matrix<Real, 3, 3> out{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            Real sum = Real(0);
            for (std::size_t a = 0; a < dim; ++a) {
                sum += jac_ref(r, a) * jacobian_inverse(a, c);
            }
            out(r, c) = sum;
        }
    }
    return out;
}

math::Matrix<Real, 3, 3> PushForward::hdiv_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Vector<Real, 3>& xi)
{
    return hdiv_vector_jacobian(mapping,
                                jac_ref,
                                mapping.jacobian(xi),
                                mapping.jacobian_inverse(xi),
                                mapping.jacobian_determinant(xi));
}

math::Matrix<Real, 3, 3> PushForward::hdiv_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Matrix<Real, 3, 3>& jacobian,
    const math::Matrix<Real, 3, 3>& jacobian_inverse,
    Real det_jacobian)
{
    if (!mapping.isAffine()) {
        FE_THROW(FEException,
                 "H(div) vector-basis Jacobians require an affine geometry mapping; "
                 "curved Piola derivative terms are not implemented");
    }
    return (jacobian * jac_ref * jacobian_inverse) * (Real(1) / det_jacobian);
}

math::Matrix<Real, 3, 3> PushForward::hcurl_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Vector<Real, 3>& xi)
{
    return hcurl_vector_jacobian(mapping, jac_ref, mapping.jacobian_inverse(xi));
}

math::Matrix<Real, 3, 3> PushForward::hcurl_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Matrix<Real, 3, 3>& jacobian_inverse)
{
    if (!mapping.isAffine()) {
        FE_THROW(FEException,
                 "H(curl) vector-basis Jacobians require an affine geometry mapping; "
                 "curved Piola derivative terms are not implemented");
    }
    return jacobian_inverse.transpose() * jac_ref * jacobian_inverse;
}

} // namespace geometry
} // namespace FE
} // namespace svmp
