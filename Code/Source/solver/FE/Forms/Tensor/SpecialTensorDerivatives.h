#ifndef SVMP_FE_FORMS_TENSOR_SPECIAL_TENSOR_DERIVATIVES_H
#define SVMP_FE_FORMS_TENSOR_SPECIAL_TENSOR_DERIVATIVES_H

/**
 * @file SpecialTensorDerivatives.h
 * @brief Derivative helpers for special tensors (delta, epsilon, metric)
 *
 * The current FE/Forms vocabulary does not yet expose Kronecker/Levi-Civita as
 * first-class FormExpr nodes. This header provides a place for those rules as
 * the tensor-calculus roadmap evolves.
 */

#include "Forms/FormExpr.h"
#include "Forms/Tensor/TensorDifferentiation.h"

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

// Placeholder API: when special tensor FormExpr nodes exist, these helpers can
// construct correct symbolic derivatives. For now, they serve as extension points.

[[nodiscard]] inline FormExpr differentiateDelta(const TensorDiffContext&) { return FormExpr::constant(0.0); }
[[nodiscard]] inline FormExpr differentiateLeviCivita(const TensorDiffContext&) { return FormExpr::constant(0.0); }

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_SPECIAL_TENSOR_DERIVATIVES_H

