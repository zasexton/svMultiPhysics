/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_VECTORBASISDIRECTSEEDS_H
#define SVMP_FE_BASIS_VECTORBASISDIRECTSEEDS_H

#include "VectorBasis.h"

#include <vector>

namespace svmp {
namespace FE {
namespace basis {
namespace detail {
namespace vector_direct {

using Vec3 = math::Vector<Real, 3>;

void eval_pyramid_rt1_values(const Vec3& xi, std::vector<Vec3>& values);
void eval_pyramid_rt1_divergence(const Vec3& xi, std::vector<Real>& divergence);
void eval_pyramid_rt1_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians);

void eval_pyramid_rt2_values(const Vec3& xi, std::vector<Vec3>& values);
void eval_pyramid_rt2_divergence(const Vec3& xi, std::vector<Real>& divergence);
void eval_pyramid_rt2_jacobians(const Vec3& xi, std::vector<VectorJacobian>& jacobians);

} // namespace vector_direct
} // namespace detail
} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_VECTORBASISDIRECTSEEDS_H
