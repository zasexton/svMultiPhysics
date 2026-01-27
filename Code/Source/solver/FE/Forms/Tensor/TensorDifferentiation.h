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
#include "Forms/Tensor/TensorIndex.h"

#include <optional>
#include <string>
#include <vector>

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

    // Optional index structure for the differentiated variable (for tensor-valued
    // unknowns and future index-form derivative rules).
    std::vector<TensorIndex> wrt_indices{};

    // Controls how TrialFunction occurrences in the primal expression are rewritten
    // (see `forms::differentiateResidual` docs).
    FieldId trial_state_field{INVALID_FIELD_ID};
};

struct TensorDiffOptions {
    // Ensure generated expressions remain compatible with the current einsum
    // contract (each index id appears at most twice globally) by renaming
    // indices across additive terms produced by product/chain rules.
    bool disambiguate_indices_in_sums{true};

    // Apply tensor-aware simplification (delta/symmetry/epsilon rules) after
    // differentiation. This is conservative and terminating.
    bool simplify_tensor_expr{true};

    int simplify_max_passes{12};
    bool simplify_canonicalize_terms{true};
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

/**
 * @brief Hessian-vector product for a residual form (tensor/index workflow)
 *
 * This returns a bilinear form:
 *   H(u)[w](δu, v) = d/du ( dR/du[δu, v] )[w]
 *
 * where `w` is provided by `direction`. The differentiation target is selected
 * using `ctx` in the same way as `differentiateTensorResidual`.
 *
 * Practical note: this represents the second derivative as a Hessian-vector
 * product so it remains assemblable as a standard matrix (bilinear form).
 */
[[nodiscard]] FormExpr differentiateTensorResidualHessianVector(const FormExpr& residual_form,
                                                                const FormExpr& direction,
                                                                const TensorDiffContext& ctx = {});

/**
 * @brief Post-process a differentiated expression for tensor/index workflows
 *
 * This currently:
 * - renames index ids across additive branches to keep `einsum()`-compatible,
 * - optionally runs `TensorSimplify` passes.
 */
[[nodiscard]] FormExpr postprocessTensorDerivative(const FormExpr& expr,
                                                   const TensorDiffOptions& options = {});

/**
 * @brief Check for presence of tensor/index-notation nodes (`IndexedAccess`)
 */
[[nodiscard]] bool containsTensorCalculusNodes(const FormExpr& expr);

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_DIFFERENTIATION_H
