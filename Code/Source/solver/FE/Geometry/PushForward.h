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
    struct PiolaVectorGradientGeometryData {
        math::Matrix<Real, 3, 3> jacobian{};
        math::Matrix<Real, 3, 3> jacobian_inverse{};
        GeometryMapping::MappingHessian mapping_hessian{};
        std::array<math::Matrix<Real, 3, 3>, 3> jacobian_derivatives_x{};
        std::array<math::Matrix<Real, 3, 3>, 3> inverse_jacobian_derivatives_x{};
        std::array<math::Matrix<Real, 3, 3>, 3> inverse_transpose_jacobian_derivatives_x{};
        math::Vector<Real, 3> determinant_derivatives_x{};
        Real determinant{Real(1)};
        int dimension{3};
        bool affine{true};
    };

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

    /// Build reusable geometry data for curved Piola vector-gradient transforms.
    static PiolaVectorGradientGeometryData piola_vector_gradient_geometry_data(
        const GeometryMapping& mapping,
        const math::Vector<Real, 3>& xi);

    /// Build reusable geometry data for curved Piola vector-gradient transforms
    /// from geometry already evaluated by assembly.
    static PiolaVectorGradientGeometryData piola_vector_gradient_geometry_data(
        int dimension,
        const math::Matrix<Real, 3, 3>& jacobian,
        const math::Matrix<Real, 3, 3>& jacobian_inverse,
        Real det_jacobian,
        const GeometryMapping::MappingHessian& mapping_hessian,
        bool affine_mapping);

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

    /// H(div) Piola vector-Jacobian transform on affine or curved mappings:
    /// grad_x v = d_x[(1/detJ) * J * v_hat(xi(x))].
    static math::Matrix<Real, 3, 3> hdiv_vector_jacobian(
        const GeometryMapping& mapping,
        const math::Vector<Real, 3>& v_ref,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const math::Vector<Real, 3>& xi);

    /// H(div) Piola vector-Jacobian transform using reusable curved geometry data.
    static math::Matrix<Real, 3, 3> hdiv_vector_jacobian(
        const math::Vector<Real, 3>& v_ref,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const PiolaVectorGradientGeometryData& geometry);

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

    /// H(curl) Piola vector-Jacobian transform on affine or curved mappings:
    /// grad_x v = d_x[J^{-T} * v_hat(xi(x))].
    static math::Matrix<Real, 3, 3> hcurl_vector_jacobian(
        const GeometryMapping& mapping,
        const math::Vector<Real, 3>& v_ref,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const math::Vector<Real, 3>& xi);

    /// H(curl) Piola vector-Jacobian transform using reusable curved geometry data.
    static math::Matrix<Real, 3, 3> hcurl_vector_jacobian(
        const math::Vector<Real, 3>& v_ref,
        const math::Matrix<Real, 3, 3>& jac_ref,
        const PiolaVectorGradientGeometryData& geometry);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_PUSHFORWARD_H
