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
 * - Undifferentiated TrialFunction occurrences are rewritten to StateField(CURRENT_SOLUTION_FIELD_ID),
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
 * - `trial_state_field == CURRENT_SOLUTION_FIELD_ID` (default): use StateField(CURRENT_SOLUTION_FIELD_ID) sentinel, i.e. current solution
 * - otherwise: rewrite TrialFunction to `StateField(trial_state_field, ...)` so tangents can be assembled in block/multi-field
 *   contexts where the active trial space differs from the residual's primary unknown.
 */
[[nodiscard]] FormExpr differentiateResidual(const FormExpr& residual_form,
                                            const FormExpr& wrt_terminal,
                                            FieldId trial_state_field = CURRENT_SOLUTION_FIELD_ID);

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
                                            FieldId trial_state_field = CURRENT_SOLUTION_FIELD_ID);

/**
 * @brief Directional derivative of an expression w.r.t. a FieldId
 *
 * This computes: d(expr)/d(field)[direction], treating all TrialFunction/TestFunction
 * terminals as constants. Only `StateField(field)` / `DiscreteField(field)` terminals
 * participate in the differentiation.
 *
 * This is intended as a building block for higher-order derivatives such as
 * Hessian-vector products.
 */
[[nodiscard]] FormExpr directionalDerivativeWrtField(const FormExpr& expr,
                                                     FieldId field,
                                                     const FormExpr& direction);

/**
 * @brief Hessian-vector product of a residual form w.r.t. the active TrialFunction
 *
 * This returns a bilinear form representing:
 *   H(u)[w](δu, v) = d/du ( dR/du[δu, v] )[w]
 *
 * where `w` is provided by `direction` and is expected to be compatible with the
 * residual's unknown (same value shape). Common choices include:
 * - a `StateField`/`DiscreteField` terminal backed by an externally-provided direction vector
 * - `PreviousSolutionRef(k)` when reusing transient history slots in time-integration contexts
 *
 * The returned expression is suitable for standard bilinear assembly (matrix).
 */
[[nodiscard]] FormExpr differentiateResidualHessianVector(const FormExpr& residual_form,
                                                         const FormExpr& direction);

/**
 * @brief Hessian-vector product of a residual form w.r.t. a FieldId (multi-field)
 *
 * This differentiates the residual w.r.t. `field` to form a tangent bilinear form,
 * then takes a directional derivative of that tangent w.r.t. the same field in the
 * provided `direction`.
 */
[[nodiscard]] FormExpr differentiateResidualHessianVector(const FormExpr& residual_form,
                                                         FieldId field,
                                                         const FormExpr& direction,
                                                         FieldId trial_state_field = INVALID_FIELD_ID);

/**
 * @brief Differentiate a form w.r.t. a specific AuxiliaryOutputRef slot.
 *
 * This treats the AuxiliaryOutputRef(slot) terminal as the active variable
 * (derivative = 1) and all other terminals as constants.  The result is a
 * scalar-valued expression suitable for evaluating dR/d(output_k) at each
 * quadrature point.
 *
 * Used for the transpose Jacobian block dR_PDE/dx_aux when PDE forms
 * reference auxiliary outputs.  Works for any form location — NaturalBCs,
 * source terms, material parameters, etc.
 */
[[nodiscard]] FormExpr differentiateWrtAuxiliaryOutput(
    const FormExpr& form, std::uint32_t output_slot);

/**
 * @brief Simplify a FormExpr with lightweight algebraic rewrites and constant folding
 */
[[nodiscard]] FormExpr simplify(const FormExpr& expr);

/**
 * @brief Extract additive sub-terms that reference a specific node type + slot.
 *
 * Decomposes the expression into additive terms (splitting on Add/Subtract)
 * and returns only those terms whose subtree contains a node matching the
 * given type and slot index.  Terms that don't reference the target are
 * dropped entirely (not replaced with zero).
 *
 * This is used before symbolic differentiation to avoid differentiating
 * large sub-expressions that will produce zero.
 */
[[nodiscard]] FormExpr extractTermsReferencing(
    const FormExpr& form, FormExprType target_type, std::uint32_t target_slot);

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_SYMBOLIC_DIFFERENTIATION_H
