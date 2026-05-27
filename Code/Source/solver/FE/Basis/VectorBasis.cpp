/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "VectorBasis.h"
#include "VectorBasisEvaluationHelpers.h"

namespace svmp {
namespace FE {
namespace basis {

void VectorBasisFunction::evaluate_vector_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT jacobians_out,
    Real* SVMP_RESTRICT curls_out,
    Real* SVMP_RESTRICT divergence_out) const {
    detail::vector_common::evaluate_vector_public_api_strided(
        *this,
        points,
        output_stride,
        values_out,
        jacobians_out,
        curls_out,
        divergence_out,
        false,
        false,
        "VectorBasisFunction");
}

} // namespace basis
} // namespace FE
} // namespace svmp
