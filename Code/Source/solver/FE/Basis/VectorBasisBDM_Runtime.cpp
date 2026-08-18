/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasis.h"
#include "Basis/BasisTraits.h"

#include <cstddef>
#include <vector>

#ifdef FE_CHECK_ARG
#undef FE_CHECK_ARG
#endif
#define FE_CHECK_ARG(condition, message) BASIS_CHECK_CONSTRUCTION((condition), (message))

#include "VectorBasisEvaluationHelpers.h"

namespace svmp {
namespace FE {
namespace basis {

using namespace detail::vector_common;

void BDMBasis::evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                      std::vector<math::Vector<Real, 3>>& values) const {
    if (element_type_ == ElementType::Triangle3 && !nodal_generated_) {
        const Real x = xi[0];
        const Real y = xi[1];
        values.resize(6);
        values[0] = math::Vector<Real, 3>{x, y - Real(1), Real(0)};
        values[1] = math::Vector<Real, 3>{Real(3) * x,
                                          Real(3) - Real(6) * x - Real(3) * y,
                                          Real(0)};
        values[2] = math::Vector<Real, 3>{x, y, Real(0)};
        values[3] = math::Vector<Real, 3>{-Real(3) * x, Real(3) * y, Real(0)};
        values[4] = math::Vector<Real, 3>{x - Real(1), y, Real(0)};
        values[5] = math::Vector<Real, 3>{Real(3) - Real(3) * x - Real(6) * y,
                                          Real(3) * y,
                                          Real(0)};
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_vector_values_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, values);
        return;
    }

    const Real x = xi[0];
    const Real y = xi[1];

    // BDM1 on the reference quadrilateral [-1,1]^2:
    // two edge moments per edge (constant + linear), 8 basis functions total.
    values.resize(8);
    const Real half = Real(0.5);
    const Real one = Real(1);
    // Edge 0-1 (bottom, y=-1): outward normal -y
    values[0] = math::Vector<Real, 3>{Real(0), half * (y - one), Real(0)};                  // flux = 1
    values[1] = math::Vector<Real, 3>{Real(0), half * x * (y - one), Real(0)};              // flux = x
    // Edge 1-2 (right, x=+1): outward normal +x
    values[2] = math::Vector<Real, 3>{half * (one + x), Real(0), Real(0)};                  // flux = 1
    values[3] = math::Vector<Real, 3>{half * y * (one + x), Real(0), Real(0)};              // flux = y
    // Edge 2-3 (top, y=+1): outward normal +y
    values[4] = math::Vector<Real, 3>{Real(0), half * (one + y), Real(0)};                  // flux = 1
    values[5] = math::Vector<Real, 3>{Real(0), half * x * (one + y), Real(0)};              // flux = x
    // Edge 3-0 (left, x=-1): outward normal -x
    values[6] = math::Vector<Real, 3>{half * (x - one), Real(0), Real(0)};                  // flux = 1
    values[7] = math::Vector<Real, 3>{half * y * (x - one), Real(0), Real(0)};              // flux = y
}

void BDMBasis::evaluate_vector_jacobians(const math::Vector<Real, 3>& xi,
                                         std::vector<VectorJacobian>& jacobians) const {
    if (element_type_ == ElementType::Triangle3 && !nodal_generated_) {
        jacobians.assign(6, VectorJacobian{});
        jacobians[0](0, 0) = Real(1);
        jacobians[0](1, 1) = Real(1);
        jacobians[1](0, 0) = Real(3);
        jacobians[1](1, 0) = -Real(6);
        jacobians[1](1, 1) = -Real(3);
        jacobians[2](0, 0) = Real(1);
        jacobians[2](1, 1) = Real(1);
        jacobians[3](0, 0) = -Real(3);
        jacobians[3](1, 1) = Real(3);
        jacobians[4](0, 0) = Real(1);
        jacobians[4](1, 1) = Real(1);
        jacobians[5](0, 0) = -Real(3);
        jacobians[5](0, 1) = -Real(6);
        jacobians[5](1, 1) = Real(3);
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_vector_jacobians_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, jacobians);
        return;
    }

    const Real x = xi[0];
    const Real y = xi[1];
    const Real half = Real(0.5);
    const Real one = Real(1);

    jacobians.assign(8, VectorJacobian{});
    jacobians[0](1, 1) = half;
    jacobians[1](1, 0) = half * (y - one);
    jacobians[1](1, 1) = half * x;
    jacobians[2](0, 0) = half;
    jacobians[3](0, 0) = half * y;
    jacobians[3](0, 1) = half * (one + x);
    jacobians[4](1, 1) = half;
    jacobians[5](1, 0) = half * (one + y);
    jacobians[5](1, 1) = half * x;
    jacobians[6](0, 0) = half;
    jacobians[7](0, 0) = half * y;
    jacobians[7](0, 1) = half * (x - one);
}

void BDMBasis::evaluate_divergence(const math::Vector<Real, 3>& xi,
                                   std::vector<Real>& divergence) const {
    if (element_type_ == ElementType::Triangle3 && !nodal_generated_) {
        (void)xi;
        divergence.resize(6);
        divergence[0] = Real(2);
        divergence[1] = Real(0);
        divergence[2] = Real(2);
        divergence[3] = Real(0);
        divergence[4] = Real(2);
        divergence[5] = Real(0);
        return;
    }

    if (nodal_generated_) {
        evaluate_nodal_modal_divergence_with_limits(
            monomials_, modal_sparse_coeffs_, size_, xi, modal_power_limits_, divergence);
        return;
    }

    const Real x = xi[0];
    const Real y = xi[1];

    divergence.resize(8);
    const Real half = Real(0.5);
    divergence[0] = half;
    divergence[1] = half * x;
    divergence[2] = half;
    divergence[3] = half * y;
    divergence[4] = half;
    divergence[5] = half * x;
    divergence[6] = half;
    divergence[7] = half * y;
}

void BDMBasis::evaluate_vector_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT jacobians_out,
    Real* SVMP_RESTRICT curls_out,
    Real* SVMP_RESTRICT divergence_out) const {
    if (nodal_generated_) {
        evaluate_nodal_modal_vector_strided_with_limits(
            monomials_,
            modal_sparse_coeffs_,
            size_,
            points,
            output_stride,
            modal_power_limits_,
            values_out,
            jacobians_out,
            curls_out,
            divergence_out,
            "BDMBasis");
        return;
    }

    evaluate_vector_public_api_strided(*this,
                                       points,
                                       output_stride,
                                       values_out,
                                       jacobians_out,
                                       curls_out,
                                       divergence_out,
                                       false,
                                       true,
                                       "BDMBasis");
}

} // namespace basis
} // namespace FE
} // namespace svmp
