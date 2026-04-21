/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_PUSHFORWARD_H
#define SVMP_FE_GEOMETRY_PUSHFORWARD_H

/**
 * @file PushForward.h
 * @brief Transform scalar/vector/tensor quantities between reference and physical spaces
 */

#include "GeometryMapping.h"

namespace svmp {
namespace FE {
namespace geometry {

class PushForward {
public:
    /// Transform reference gradient to physical gradient (J^{-T} * grad_ref)
    static math::Vector<Real, 3> gradient(const GeometryMapping& mapping,
                                          const math::Vector<Real, 3>& grad_ref,
                                          const math::Vector<Real, 3>& xi);

    /// Transform reference gradient using a precomputed inverse Jacobian.
    static math::Vector<Real, 3> gradient(const GeometryMapping& mapping,
                                          const math::Vector<Real, 3>& grad_ref,
                                          const math::Matrix<Real, 3, 3>& jacobian_inverse);

    /// H(div) Piola transform (contravariant): v_phys = (1/detJ) * J * v_ref
    static math::Vector<Real, 3> hdiv_vector(const GeometryMapping& mapping,
                                             const math::Vector<Real, 3>& v_ref,
                                             const math::Vector<Real, 3>& xi);

    /// H(div) Piola transform using a precomputed Jacobian and determinant.
    static math::Vector<Real, 3> hdiv_vector(const GeometryMapping& mapping,
                                             const math::Vector<Real, 3>& v_ref,
                                             const math::Matrix<Real, 3, 3>& jacobian,
                                             Real det_jacobian);

    /// H(curl) Piola transform (covariant): v_phys = J^{-T} * v_ref
    static math::Vector<Real, 3> hcurl_vector(const GeometryMapping& mapping,
                                              const math::Vector<Real, 3>& v_ref,
                                              const math::Vector<Real, 3>& xi);

    /// H(curl) Piola transform using a precomputed inverse Jacobian.
    static math::Vector<Real, 3> hcurl_vector(const GeometryMapping& mapping,
                                              const math::Vector<Real, 3>& v_ref,
                                              const math::Matrix<Real, 3, 3>& jacobian_inverse);

    /// Ordinary vector-field Jacobian transform: grad_x v = grad_xi(v) * J^{-1}.
    static math::Matrix<Real, 3, 3> vector_jacobian(
        const GeometryMapping& mapping,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const math::Vector<Real, 3>& xi);

    /// Ordinary vector-field Jacobian transform with a precomputed inverse Jacobian.
    static math::Matrix<Real, 3, 3> vector_jacobian(
        const GeometryMapping& mapping,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const math::Matrix<Real, 3, 3>& jacobian_inverse);

    /// Affine H(div) Piola vector-Jacobian transform:
    /// grad_x v = (1/detJ) * J * grad_xi(v_hat) * J^{-1}.
    static math::Matrix<Real, 3, 3> hdiv_vector_jacobian(
        const GeometryMapping& mapping,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const math::Vector<Real, 3>& xi);

    /// Affine H(div) Piola vector-Jacobian transform with precomputed geometry.
    static math::Matrix<Real, 3, 3> hdiv_vector_jacobian(
        const GeometryMapping& mapping,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const math::Matrix<Real, 3, 3>& jacobian,
        const math::Matrix<Real, 3, 3>& jacobian_inverse,
        Real det_jacobian);

    /// Affine H(curl) Piola vector-Jacobian transform:
    /// grad_x v = J^{-T} * grad_xi(v_hat) * J^{-1}.
    static math::Matrix<Real, 3, 3> hcurl_vector_jacobian(
        const GeometryMapping& mapping,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const math::Vector<Real, 3>& xi);

    /// Affine H(curl) Piola vector-Jacobian transform with a precomputed inverse Jacobian.
    static math::Matrix<Real, 3, 3> hcurl_vector_jacobian(
        const GeometryMapping& mapping,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const math::Matrix<Real, 3, 3>& jacobian_inverse);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_PUSHFORWARD_H
