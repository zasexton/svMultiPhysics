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

/**
 * @brief Derivative of Kronecker delta / identity tensor (0)
 */
[[nodiscard]] FormExpr differentiateDelta(const TensorDiffContext& ctx);

/**
 * @brief Derivative of Levi-Civita tensor (0)
 */
[[nodiscard]] FormExpr differentiateLeviCivita(const TensorDiffContext& ctx);

/**
 * @brief Derivative of metric tensor g_ij (default identity metric => 0)
 */
[[nodiscard]] FormExpr differentiateMetricTensor(const TensorDiffContext& ctx);

/**
 * @brief Derivative of inverse metric tensor g^ij (default identity metric => 0)
 */
[[nodiscard]] FormExpr differentiateInverseMetricTensor(const TensorDiffContext& ctx);

struct DeformationGradientDiffResult {
    FormExpr F{};
    FormExpr dF{};
};

/**
 * @brief Construct and differentiate deformation gradient F = I + grad(u)
 *
 * This helper is intended for hyperelasticity chains. It returns:
 * - `F`: the deformation gradient (primal)
 * - `dF`: its directional derivative w.r.t. `ctx` (e.g., grad(δu))
 */
[[nodiscard]] DeformationGradientDiffResult differentiateDeformationGradient(const FormExpr& displacement,
                                                                            const TensorDiffContext& ctx = {});

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_SPECIAL_TENSOR_DERIVATIVES_H
