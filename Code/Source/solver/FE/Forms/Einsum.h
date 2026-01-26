#ifndef SVMP_FE_FORMS_EINSUM_H
#define SVMP_FE_FORMS_EINSUM_H

/**
 * @file Einsum.h
 * @brief UFL-like Einstein summation lowering for FE/Forms
 *
 * `einsum(expr)` lowers indexed expressions like `A(i,j) * B(i,j)` into an
 * explicit sum of component-level expressions using `component(·,i[,j])`.
 *
 * The resulting expression contains no index objects and can be compiled and
 * assembled normally.
 */

#include "Forms/FormExpr.h"

namespace svmp {
namespace FE {
namespace forms {

/**
 * @brief Lower indexed expressions via Einstein summation
 *
 * Current limitations (intentional, to keep lowering unambiguous):
 * - Each index id may appear exactly twice (bound / summed) or exactly once (free).
 * - Output rank is limited to <= 2 free indices:
 *     - 0 free indices: scalar output
 *     - 1 free index: vector output (`AsVector`)
 *     - 2 free indices: matrix output (`AsTensor`)
 * - Index ranges are finite (from IndexSet extent; default 3).
 * - Only rank-1 and rank-2 `IndexedAccess` nodes are supported.
 *
 * @throws std::invalid_argument if the expression contains unsupported indexed patterns.
 */
[[nodiscard]] FormExpr einsum(const FormExpr& expr);

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_EINSUM_H
