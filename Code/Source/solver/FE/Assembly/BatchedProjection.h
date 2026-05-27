/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ASSEMBLY_BATCHEDPROJECTION_H
#define SVMP_FE_ASSEMBLY_BATCHEDPROJECTION_H

#include "Basis/BatchEvaluator.h"

namespace svmp {
namespace FE {
namespace assembly {

/**
 * @brief Compute field values at each quadrature point from batched basis data.
 *
 * Computes result[q] = sum_i coeffs[i] * N_i(q) * weights[q].
 */
void weighted_sum(const basis::BatchEvaluator& batch,
                  const Real* coeffs,
                  const Real* weights,
                  Real* result);

/**
 * @brief Compute field gradients at each quadrature point from batched basis data.
 *
 * Computes result[d][q] = sum_i coeffs[i] * dN_i/dxi_d(q) * weights[q].
 * The result layout is component-major with three rows of num_quad_points.
 */
void weighted_gradient_sum(const basis::BatchEvaluator& batch,
                           const Real* coeffs,
                           const Real* weights,
                           Real* result);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_BATCHEDPROJECTION_H
