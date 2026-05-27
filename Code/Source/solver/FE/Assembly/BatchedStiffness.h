/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ASSEMBLY_BATCHEDSTIFFNESS_H
#define SVMP_FE_ASSEMBLY_BATCHEDSTIFFNESS_H

#include "Basis/BatchEvaluator.h"

namespace svmp {
namespace FE {
namespace assembly {

/**
 * @brief Assemble a reference-space scalar stiffness contribution from batched gradients.
 *
 * Computes K_ij = sum_q w_q * grad(N_i)_q . D . grad(N_j)_q. The material or
 * operator tensor D is dense, dimension x dimension, and may be nonsymmetric.
 */
void assemble_stiffness_contribution(const basis::BatchEvaluator& batch,
                                     const Real* D,
                                     const Real* weights,
                                     Real* K);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_BATCHEDSTIFFNESS_H
