/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorDifferentiation.h"

#include "Forms/SymbolicDifferentiation.h"

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

TensorDiffResult checkTensorDifferentiability(const FormExpr& expr)
{
    const auto r = checkSymbolicDifferentiability(expr);
    TensorDiffResult out;
    out.ok = r.ok;
    if (r.first_issue) {
        out.first_issue = TensorDiffIssue{
            .type = r.first_issue->type,
            .message = r.first_issue->message,
            .subexpr = r.first_issue->subexpr,
        };
    }
    return out;
}

FormExpr differentiateTensorResidual(const FormExpr& residual_form, const TensorDiffContext& ctx)
{
    if (ctx.wrt_terminal.has_value()) {
        return differentiateResidual(residual_form, *ctx.wrt_terminal, ctx.trial_state_field);
    }
    if (ctx.wrt_field.has_value()) {
        return differentiateResidual(residual_form, *ctx.wrt_field, ctx.trial_state_field);
    }
    return differentiateResidual(residual_form);
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

