#ifndef SVMP_FE_FORMS_SYMBOLIC_DIFFERENTIATION_H
#define SVMP_FE_FORMS_SYMBOLIC_DIFFERENTIATION_H

/**
 * @file SymbolicDifferentiation.h
 * @brief Symbolic differentiation utilities for FE/Forms FormExpr
 *
 * This module differentiates residual FormExpr trees w.r.t. the active TrialFunction
 * to produce a bilinear tangent form suitable for scalar (non-Dual) Jacobian assembly.
 */

#include "Forms/FormExpr.h"

#include <optional>
#include <string>

namespace svmp {
namespace FE {
namespace forms {

struct SymbolicDiffIssue {
    FormExprType type{FormExprType::Constant};
    std::string message{};
    std::string subexpr{};
};

struct SymbolicDiffResult {
    bool ok{true};
    std::optional<SymbolicDiffIssue> first_issue{};
};

/**
 * @brief Check if an expression can be symbolically differentiated w.r.t TrialFunction
 *
 * This is a structural check. It does not compile the form.
 */
[[nodiscard]] SymbolicDiffResult checkSymbolicDifferentiability(const FormExpr& expr);

[[nodiscard]] inline bool canDifferentiateSymbolically(const FormExpr& expr)
{
    return checkSymbolicDifferentiability(expr).ok;
}

/**
 * @brief Differentiate a residual form w.r.t. the TrialFunction
 *
 * The output is a bilinear tangent form in the directional-derivative sense:
 *   a(δu, v) = dR/du[δu, v]
 *
 * Important semantic detail:
 * - Undifferentiated TrialFunction occurrences are rewritten to StateField(INVALID_FIELD_ID),
 *   representing the current solution state u.
 * - Differentiated TrialFunction occurrences become TrialFunction nodes representing δu.
 */
[[nodiscard]] FormExpr differentiateResidual(const FormExpr& residual_form);

/**
 * @brief Differentiate a residual form w.r.t. a specific terminal (multi-field support)
 *
 * Supported targets:
 * - `TrialFunction`: differentiate only that TrialFunction (others, if present, are treated as constants)
 * - `StateField` / `DiscreteField`: differentiate w.r.t. the target's `FieldId`
 *
 * The `trial_state_field` parameter controls how *all* TrialFunction occurrences in the residual are
 * rewritten in the primal expression:
 * - `trial_state_field == INVALID_FIELD_ID` (default): use StateField(INVALID_FIELD_ID) sentinel, i.e. current solution
 * - otherwise: rewrite TrialFunction to `StateField(trial_state_field, ...)` so tangents can be assembled in block/multi-field
 *   contexts where the active trial space differs from the residual's primary unknown.
 */
[[nodiscard]] FormExpr differentiateResidual(const FormExpr& residual_form,
                                            const FormExpr& wrt_terminal,
                                            FieldId trial_state_field = INVALID_FIELD_ID);

/**
 * @brief Differentiate a residual form w.r.t. a specific FieldId (multi-field support)
 *
 * This treats any `StateField(field)` / `DiscreteField(field)` occurrence as the differentiated variable and produces
 * a bilinear tangent with a TrialFunction terminal representing the variation `δ(field)`.
 *
 * The `trial_state_field` parameter controls how TrialFunction occurrences in the residual are rewritten in the primal
 * expression (see `differentiateResidual(residual_form, wrt_terminal, trial_state_field)`).
 */
[[nodiscard]] FormExpr differentiateResidual(const FormExpr& residual_form,
                                            FieldId field,
                                            FieldId trial_state_field = INVALID_FIELD_ID);

/**
 * @brief Simplify a FormExpr with lightweight algebraic rewrites and constant folding
 */
[[nodiscard]] FormExpr simplify(const FormExpr& expr);

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_SYMBOLIC_DIFFERENTIATION_H
