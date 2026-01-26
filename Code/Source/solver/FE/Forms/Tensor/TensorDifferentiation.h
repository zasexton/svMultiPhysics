#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_DIFFERENTIATION_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_DIFFERENTIATION_H

/**
 * @file TensorDifferentiation.h
 * @brief Tensor-aware symbolic differentiation entry points
 *
 * This module complements `Forms/SymbolicDifferentiation.*` by providing an API
 * surface for tensor-calculus/index-notation workflows (e.g., expressions that
 * contain `FormExprType::IndexedAccess`).
 */

#include "Forms/FormExpr.h"

#include <optional>
#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

struct TensorDiffIssue {
    FormExprType type{FormExprType::Constant};
    std::string message{};
    std::string subexpr{};
};

struct TensorDiffResult {
    bool ok{true};
    std::optional<TensorDiffIssue> first_issue{};
};

struct TensorDiffContext {
    // Differentiate w.r.t the TrialFunction by default.
    // For multi-field support, set either `wrt_terminal` or `wrt_field`.
    std::optional<FormExpr> wrt_terminal{};
    std::optional<FieldId> wrt_field{};

    // Controls how TrialFunction occurrences in the primal expression are rewritten
    // (see `forms::differentiateResidual` docs).
    FieldId trial_state_field{INVALID_FIELD_ID};
};

[[nodiscard]] TensorDiffResult checkTensorDifferentiability(const FormExpr& expr);

/**
 * @brief Differentiate a residual form in a tensor/index-notation-friendly manner
 *
 * The returned expression is the tangent bilinear form in the directional derivative
 * sense: a(δu, v) = dR/du[δu, v].
 */
[[nodiscard]] FormExpr differentiateTensorResidual(const FormExpr& residual_form,
                                                   const TensorDiffContext& ctx = {});

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_DIFFERENTIATION_H

