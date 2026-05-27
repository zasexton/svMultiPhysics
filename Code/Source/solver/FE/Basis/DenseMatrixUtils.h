/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_DENSEMATRIXUTILS_H
#define SVMP_FE_BASIS_DENSEMATRIXUTILS_H

#include "Math/DenseLinearAlgebra.h"

namespace svmp {
namespace FE {
namespace basis {
namespace detail {

// Compatibility facade for existing Basis construction code. The implementation
// and canonical declarations live in FE/Math/DenseLinearAlgebra.
using math::DenseInverseResult;
using math::DenseLUSolver;
using math::DenseMatrixDiagnostics;
using math::DensePseudoInverseResult;
using math::dense_matrix_condition_error_threshold;
using math::dense_matrix_condition_fallback_threshold;
using math::dense_matrix_diagnostics;
using math::dense_matrix_max_abs;
using math::dense_matrix_pivot_tolerance;
using math::dense_matrix_rank;
using math::dense_matrix_singular_value_tolerance;
using math::factor_dense_matrix;
using math::invert_dense_matrix;
using math::invert_dense_matrix_with_diagnostics;
using math::rank_revealing_pseudo_inverse;
using math::validate_dense_inverse_diagnostics;

} // namespace detail
} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_DENSEMATRIXUTILS_H
