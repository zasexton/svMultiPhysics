#include "Auxiliary/AuxiliaryDerivativeProvider.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Forms/PointEvaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace svmp {
namespace FE {
namespace systems {

namespace {

// ---------------------------------------------------------------------------
//  Expression simplification helpers — keep derivative trees compact
// ---------------------------------------------------------------------------

/// Check if a FormExpr is structurally zero (Constant(0) or TypedZero).
bool isStructurallyZero(const forms::FormExpr& e)
{
    if (!e.isValid() || !e.node()) return false;
    const auto& n = *e.node();
    if (n.type() == forms::FormExprType::TypedZero) return true;
    if (n.type() == forms::FormExprType::Constant) {
        auto v = n.constantValue();
        return v.has_value() && *v == 0.0;
    }
    return false;
}

/// Try to extract a constant scalar value.
bool getConstantValue(const forms::FormExpr& e, Real& out)
{
    if (!e.isValid() || !e.node()) return false;
    if (e.node()->type() == forms::FormExprType::Constant) {
        auto v = e.node()->constantValue();
        if (v.has_value()) { out = *v; return true; }
    }
    return false;
}

/// 0 + x = x, x + 0 = x.
forms::FormExpr smartAdd(const forms::FormExpr& a, const forms::FormExpr& b)
{
    if (isStructurallyZero(a)) return b;
    if (isStructurallyZero(b)) return a;
    return a + b;
}

/// x - 0 = x, 0 - x = -x.
forms::FormExpr smartSub(const forms::FormExpr& a, const forms::FormExpr& b)
{
    if (isStructurallyZero(b)) return a;
    if (isStructurallyZero(a)) return -b;
    return a - b;
}

/// 0 * x = 0, 1 * x = x, x * 1 = x, x * 0 = 0.
forms::FormExpr smartMul(const forms::FormExpr& a, const forms::FormExpr& b)
{
    if (isStructurallyZero(a) || isStructurallyZero(b))
        return forms::FormExpr::constant(0.0);
    Real v;
    if (getConstantValue(a, v) && v == 1.0) return b;
    if (getConstantValue(b, v) && v == 1.0) return a;
    return a * b;
}

/// 0 / x = 0.
forms::FormExpr smartDiv(const forms::FormExpr& a, const forms::FormExpr& b)
{
    if (isStructurallyZero(a)) return forms::FormExpr::constant(0.0);
    return a / b;
}

/// -0 = 0.
forms::FormExpr smartNeg(const forms::FormExpr& a)
{
    if (isStructurallyZero(a)) return forms::FormExpr::constant(0.0);
    return -a;
}

// ---------------------------------------------------------------------------
//  Symbolic differentiation of FormExpr w.r.t. AuxiliaryStateRef(target_slot)
// ---------------------------------------------------------------------------

/// Which terminal type is the differentiation variable.
enum class DiffTarget { State, Input, Field };

/// Symbolically differentiate a FormExpr with respect to a terminal slot.
///
/// When target == State:  AuxiliaryStateRef(target_slot) is the variable.
/// When target == Input:  AuxiliaryInputRef(target_slot) is the variable.
/// When target == Field:  DiscreteField/StateField with FieldId == target_slot is the variable.
///   For multi-component fields, target_component selects which component
///   (0-indexed).  -1 means scalar field (legacy behavior: whole field = 1 variable).
/// All other terminals are treated as constants.
///
/// Supports all scalar operations relevant to auxiliary state models.
/// Throws std::invalid_argument for FE-specific or tensor operations.
forms::FormExpr diffWrt(const forms::FormExpr& expr,
                        DiffTarget target, std::uint32_t target_slot,
                        int target_component = -1)
{
    if (!expr.isValid() || !expr.node()) return forms::FormExpr::constant(0.0);

    const auto& node = *expr.node();
    const auto type = node.type();
    using FT = forms::FormExprType;

    // Helper to get the i-th child as FormExpr.
    auto child = [&](std::size_t i) -> forms::FormExpr {
        auto kids = node.childrenShared();
        if (i < kids.size() && kids[i])
            return forms::FormExpr(kids[i]);
        return forms::FormExpr::constant(0.0);
    };

    // Recursive shorthand.
    auto D = [&](const forms::FormExpr& e) {
        return diffWrt(e, target, target_slot, target_component);
    };

    // Check if a node is a DiscreteField/StateField matching the target.
    auto isTargetField = [&](const forms::FormExprNode& nd) -> bool {
        if (target != DiffTarget::Field) return false;
        if (nd.type() != FT::DiscreteField && nd.type() != FT::StateField) return false;
        auto fid = nd.fieldId();
        return fid.has_value() && static_cast<std::uint32_t>(*fid) == target_slot;
    };

    switch (type) {

    // =================================================================
    //  Variable terminals
    // =================================================================
    case FT::AuxiliaryStateRef: {
        if (target != DiffTarget::State) return forms::FormExpr::constant(0.0);
        auto slot = node.slotIndex();
        return forms::FormExpr::constant(
            (slot.has_value() && *slot == target_slot) ? 1.0 : 0.0);
    }
    case FT::AuxiliaryInputRef: {
        if (target != DiffTarget::Input) return forms::FormExpr::constant(0.0);
        auto slot = node.slotIndex();
        return forms::FormExpr::constant(
            (slot.has_value() && *slot == target_slot) ? 1.0 : 0.0);
    }

    // DiscreteField / StateField: variable when target == Field and FieldId matches.
    case FT::DiscreteField:
    case FT::StateField: {
        if (target != DiffTarget::Field) return forms::FormExpr::constant(0.0);
        auto fid = node.fieldId();
        if (!fid.has_value() || static_cast<std::uint32_t>(*fid) != target_slot)
            return forms::FormExpr::constant(0.0);
        if (target_component >= 0) {
            // Multi-component mode: a raw vector DiscreteField can't produce
            // a scalar derivative without component extraction.  Return 0 —
            // the derivative flows through ComponentAccess or InnerProduct.
            return forms::FormExpr::constant(0.0);
        }
        // Scalar field (target_component == -1): whole field is the variable.
        return forms::FormExpr::constant(1.0);
    }

    // =================================================================
    //  Constant terminals (d/d(target) = 0)
    // =================================================================
    case FT::AuxiliaryOutputRef:
    case FT::ParameterRef:
    case FT::Constant:
    case FT::TypedZero:
    case FT::Time:
    case FT::TimeStep:
    case FT::EffectiveTimeStep:
    case FT::BoundaryIntegralRef:
    case FT::Coordinate:
    case FT::ReferenceCoordinate:
    case FT::CellDiameter:
    case FT::CellVolume:
    case FT::FacetArea:
    case FT::CellDomainId:
    case FT::MaterialStateOldRef:
    case FT::MaterialStateWorkRef:
    case FT::PreviousSolutionRef:
    case FT::Identity:
    case FT::Coefficient:
        return forms::FormExpr::constant(0.0);

    // =================================================================
    //  Arithmetic — existing rules
    // =================================================================
    case FT::Negate: {
        auto da = D(child(0));
        return smartNeg(da);
    }
    case FT::Add: {
        auto da = D(child(0));
        auto db = D(child(1));
        return smartAdd(da, db);
    }
    case FT::Subtract: {
        auto da = D(child(0));
        auto db = D(child(1));
        return smartSub(da, db);
    }
    case FT::Multiply: {
        // Product rule: d(a*b) = da*b + a*db
        auto a = child(0);
        auto b = child(1);
        auto da = D(a);
        auto db = D(b);
        return smartAdd(smartMul(da, b), smartMul(a, db));
    }
    case FT::Divide: {
        // Quotient rule: d(a/b) = (da*b - a*db) / (b*b)
        auto a = child(0);
        auto b = child(1);
        auto da = D(a);
        auto db = D(b);
        if (isStructurallyZero(db)) {
            // b is constant w.r.t. target → d(a/b) = da/b
            return smartDiv(da, b);
        }
        return (da * b - a * db) / (b * b);
    }

    // =================================================================
    //  Transcendental functions
    // =================================================================
    case FT::Power: {
        // d(a^b)/dx:
        //   If b is constant c:  c * a^(c-1) * da
        //   General:  a^b * (db * ln(a) + b * da / a)
        auto a = child(0);
        auto b = child(1);
        auto da = D(a);
        auto db = D(b);

        Real exp_val = 0.0;
        if (getConstantValue(b, exp_val)) {
            // Constant exponent.
            if (exp_val == 0.0) {
                // d(a^0)/dx = 0
                return forms::FormExpr::constant(0.0);
            }
            // d(a^c)/dx = c * a^(c-1) * da
            return smartMul(
                forms::FormExpr::constant(exp_val) *
                    a.pow(forms::FormExpr::constant(exp_val - 1.0)),
                da);
        }

        // General case: both a and b may depend on target.
        //
        // d(a^b)/dx = b * da * a^(b-1)          [da term]
        //           + a^b * db * ln(a)           [db term]
        //
        // The da term uses a^(b-1) instead of a^b/a to avoid the 1/a
        // singularity at a=0.  This correctly evaluates at a=0 for
        // b >= 1 (e.g., d(x^1)/dx = 1 at x=0).
        //
        // The db term uses ln(a), NOT ln(|a|).  For a < 0, ln(a) is
        // NaN, making the db term NaN.  This is correct: a^b is only
        // real-valued at isolated integer b when a < 0, so ∂/∂b does
        // not exist (no real-valued neighborhood).  Using ln(a) rather
        // than ln(|a|) ensures the symbolic derivative matches the
        // real-valued domain contract: NaN for a < 0 when db != 0.
        //
        // When db == 0 (b is state-independent), only the da term
        // survives.  That term uses a^(b-1) which evaluates correctly
        // for negative a with integer b (e.g., (-2)^2 = 4).
        //
        // At a=0, the db term is a^b * ln(a) = 0 * (-∞) → 0 for
        // b > 0 (L'Hôpital).  We guard with a conditional on a==0.
        if (isStructurallyZero(da) && isStructurallyZero(db))
            return forms::FormExpr::constant(0.0);

        auto a_is_zero = forms::eq(a, forms::FormExpr::constant(0.0));

        // da term: b * da * a^(b-1)
        auto da_term = smartMul(b, smartMul(da,
            a.pow(b - forms::FormExpr::constant(1.0))));

        if (isStructurallyZero(db))
            return da_term;

        // db term: a^b * db * ln(a)
        // Uses ln(a) (not ln(|a|)) so that a < 0 produces NaN,
        // correctly signaling that ∂/∂b is undefined there.
        //
        // At a=0: 0^b * ln(0).  For b > 0, 0^b = 0 and 0 * (-∞) → 0
        // (L'Hôpital).  For b <= 0, 0^b is undefined (∞ or 0^0 = 1
        // by convention), so the derivative is undefined too.  We guard
        // with a==0 && b>0 to return 0 only in the well-defined case;
        // otherwise let NaN propagate from log(0) or 0^b.
        auto b_positive = forms::gt(b, forms::FormExpr::constant(0.0));
        auto safe_to_zero = a_is_zero * b_positive; // 1 iff a==0 and b>0
        auto safe_a = a + safe_to_zero; // shift to avoid log(0) only when safe
        auto db_log = a.pow(b) * smartMul(db, safe_a.log());
        auto db_term = forms::eq(safe_to_zero, forms::FormExpr::constant(1.0))
                           .conditional(forms::FormExpr::constant(0.0), db_log);

        return smartAdd(da_term, db_term);
    }

    case FT::Sqrt: {
        // d(sqrt(a))/dx = da / (2*sqrt(a))
        // Guard: when sqrt(a)==0, use denom=1 to avoid division by zero.
        auto a = child(0);
        auto da = D(a);
        if (isStructurallyZero(da))
            return forms::FormExpr::constant(0.0);
        auto sq = a.sqrt();
        auto is_zero = forms::eq(sq, forms::FormExpr::constant(0.0));
        auto denom = (forms::FormExpr::constant(2.0) * sq) + is_zero;
        return da / denom;
    }

    case FT::Exp: {
        // d(exp(a))/dx = exp(a) * da
        auto a = child(0);
        auto da = D(a);
        return smartMul(a.exp(), da);
    }

    case FT::Log: {
        // d(ln(a))/dx = da / a
        auto a = child(0);
        auto da = D(a);
        return smartDiv(da, a);
    }

    // =================================================================
    //  Piecewise / comparison operations
    // =================================================================
    case FT::AbsoluteValue: {
        // d(|a|)/dx = sign(a) * da
        // sign(a) = 2*(a >= 0) - 1
        auto a = child(0);
        auto da = D(a);
        if (isStructurallyZero(da))
            return forms::FormExpr::constant(0.0);
        auto is_nonneg = forms::ge(a, forms::FormExpr::constant(0.0));
        auto sgn = forms::FormExpr::constant(2.0) * is_nonneg -
                   forms::FormExpr::constant(1.0);
        return sgn * da;
    }

    case FT::Sign: {
        // d(sign(a))/dx = 0 (piecewise constant)
        return forms::FormExpr::constant(0.0);
    }

    case FT::Minimum: {
        // d(min(a,b))/dx = (a <= b) ? da : db
        auto a = child(0);
        auto b = child(1);
        auto da = D(a);
        auto db = D(b);
        auto cond = forms::le(a, b);
        return cond.conditional(da, db);
    }

    case FT::Maximum: {
        // d(max(a,b))/dx = (a >= b) ? da : db
        auto a = child(0);
        auto b = child(1);
        auto da = D(a);
        auto db = D(b);
        auto cond = forms::ge(a, b);
        return cond.conditional(da, db);
    }

    // Comparisons return 0 or 1 — derivative is 0.
    case FT::Less:
    case FT::LessEqual:
    case FT::Greater:
    case FT::GreaterEqual:
    case FT::Equal:
    case FT::NotEqual:
        return forms::FormExpr::constant(0.0);

    case FT::Conditional: {
        // d(cond ? t : f)/dx = cond ? dt : df
        // (condition is non-differentiable; we differentiate each branch)
        auto kids = node.childrenShared();
        if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2])
            return forms::FormExpr::constant(0.0);
        auto c = forms::FormExpr(kids[0]);
        auto dt = D(forms::FormExpr(kids[1]));
        auto df = D(forms::FormExpr(kids[2]));
        return c.conditional(dt, df);
    }

    // =================================================================
    //  Smooth approximation functions
    // =================================================================
    case FT::SmoothAbsoluteValue: {
        // smoothAbs(x, eps) = sqrt(x^2 + eps^2)
        // d/dx = (x*dx + eps*deps) / smoothAbs(x, eps)
        auto x = child(0);
        auto eps = child(1);
        auto dx = D(x);
        auto deps = D(eps);
        auto sa = x.smoothAbs(eps);
        auto num = smartAdd(smartMul(x, dx), smartMul(eps, deps));
        if (isStructurallyZero(num))
            return forms::FormExpr::constant(0.0);
        auto is_zero = forms::eq(sa, forms::FormExpr::constant(0.0));
        return is_zero.conditional(forms::FormExpr::constant(0.0), num / sa);
    }

    case FT::SmoothSign:
    case FT::SmoothHeaviside: {
        // smoothSign(x, eps) = x / smoothAbs(x, eps)
        // smoothHeaviside(x, eps) = 0.5 + 0.5 * smoothSign(x, eps)
        // d(smoothSign)/dx = (dx * sa - x * d_sa) / sa^2
        // d(smoothHeaviside)/dx = 0.5 * d(smoothSign)/dx
        auto x = child(0);
        auto eps = child(1);
        auto dx = D(x);
        auto deps = D(eps);
        auto sa = x.smoothAbs(eps);
        // d(smoothAbs)/dx = (x*dx + eps*deps) / sa
        auto num_sa = smartAdd(smartMul(x, dx), smartMul(eps, deps));
        auto d_sa = forms::eq(sa, forms::FormExpr::constant(0.0))
                        .conditional(forms::FormExpr::constant(0.0), num_sa / sa);
        // d(smoothSign)/dx = (dx * sa - x * d_sa) / sa^2
        auto d_sign = (dx * sa - x * d_sa) / (sa * sa);
        auto result = forms::eq(sa, forms::FormExpr::constant(0.0))
                          .conditional(forms::FormExpr::constant(0.0), d_sign);
        if (type == FT::SmoothHeaviside)
            result = forms::FormExpr::constant(0.5) * result;
        return result;
    }

    case FT::SmoothMin:
    case FT::SmoothMax: {
        // smoothMin(a, b, eps) = 0.5 * (a + b - smoothAbs(a-b, eps))
        // smoothMax(a, b, eps) = 0.5 * (a + b + smoothAbs(a-b, eps))
        // d/dx = 0.5 * (da + db) ∓ 0.5 * d(smoothAbs(a-b, eps))/dx
        auto kids = node.childrenShared();
        if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2])
            return forms::FormExpr::constant(0.0);
        auto a = forms::FormExpr(kids[0]);
        auto b = forms::FormExpr(kids[1]);
        auto eps = forms::FormExpr(kids[2]);
        auto da = D(a);
        auto db = D(b);
        auto deps = D(eps);

        bool is_min = (type == FT::SmoothMin);

        auto d = a - b;
        auto dd = smartSub(da, db);
        auto sa = d.smoothAbs(eps);
        // d(smoothAbs(d, eps))/dx = (d*dd + eps*deps) / sa
        auto num = smartAdd(smartMul(d, dd), smartMul(eps, deps));
        auto d_sa = forms::eq(sa, forms::FormExpr::constant(0.0))
                        .conditional(forms::FormExpr::constant(0.0), num / sa);

        auto half_sum_deriv = forms::FormExpr::constant(0.5) * smartAdd(da, db);
        auto half_sa_deriv = forms::FormExpr::constant(0.5) * d_sa;
        return is_min ? smartSub(half_sum_deriv, half_sa_deriv)
                      : smartAdd(half_sum_deriv, half_sa_deriv);
    }

    // =================================================================
    //  Component extraction (from vector → scalar)
    // =================================================================
    case FT::Component: {
        auto base = child(0);
        const auto comp_idx = node.componentIndex0().value_or(0);

        // Special case: component(DiscreteField/StateField, i) with per-component
        // field differentiation.  d(u_i)/d(u_k) = δ_{ik}.
        if (target == DiffTarget::Field && target_component >= 0 && base.node()) {
            if (isTargetField(*base.node())) {
                return forms::FormExpr::constant(
                    comp_idx == target_component ? 1.0 : 0.0);
            }
        }

        // General case: d(component(f, i))/dx.  For scalar differentiation
        // targets (State, Input, scalar Field), if the base doesn't depend
        // on the target, derivative is 0.
        auto d_base = D(base);
        if (isStructurallyZero(d_base))
            return forms::FormExpr::constant(0.0);

        // If d_base is a scalar (the base depends on the target through
        // a scalar path), the component extraction collapses:
        // d(component(scalar_expr * vector, i))/dx propagates through.
        // But for vector-valued d_base we can't reduce to scalar here.
        // For the common case where the base is a DiscreteField that
        // was already handled above, this is unreachable.
        throw std::invalid_argument(
            "diffWrtAuxSlot: Component extraction on a state-dependent "
            "vector is not supported in auxiliary symbolic differentiation. "
            "Use component(field, i) for direct field access or provide "
            "an analytic Jacobian for models with vector intermediates.");
    }

    // =================================================================
    //  Tensor operations — scalar-specialized rules
    //
    //  SHAPE ENFORCEMENT: These rules apply scalar formulas (e.g.,
    //  inner(a,b) = a*b, inv(a) = 1/a) that are INCORRECT for
    //  non-scalar operands.  Correctness is enforced by the recursive
    //  structure of diffWrtAuxSlot:
    //
    //  1. AuxiliaryStateRef terminals are always scalar.
    //  2. All supported arithmetic/transcendental/piecewise ops
    //     produce scalar output from scalar input.
    //  3. Non-scalar constructors (AsVector, AsTensor) and FE ops
    //     (Gradient, Divergence, etc.) hit the default case and THROW.
    //
    //  Therefore, if diffWrt(child, slot) returns WITHOUT
    //  throwing, the child expression is guaranteed scalar.  Any
    //  non-scalar child would have thrown during the recursive
    //  descent before reaching these cases.
    //
    //  The zero-derivative short-circuit handles the case where a
    //  tensor op is applied to a non-state-dependent operand (e.g.,
    //  inner(const_vector, const_vector)) — the derivative is zero
    //  regardless of shape.
    // =================================================================

    // For scalar operands, inner/double-contraction/outer = multiply.
    // For vector operands with per-component field differentiation:
    //   d(inner(u, v))/du_k = v_k + (if v depends on u_k: ...).
    //   When one operand is a raw DiscreteField matching the target,
    //   d(u)/du_k = e_k, so inner(e_k, v) = component(v, k).
    case FT::InnerProduct:
    case FT::DoubleContraction:
    case FT::OuterProduct: {
        auto a = child(0);
        auto b = child(1);

        // Per-component field differentiation with vector DiscreteField operands.
        if (target == DiffTarget::Field && target_component >= 0 &&
            a.node() && b.node()) {
            const bool a_is_target = isTargetField(*a.node());
            const bool b_is_target = isTargetField(*b.node());

            if (a_is_target && b_is_target) {
                // inner(u, u): d/du_k = 2 * component(u, k)
                return smartMul(forms::FormExpr::constant(2.0),
                                a.component(target_component));
            }
            if (a_is_target) {
                // inner(u, v): d/du_k = component(v, k) + inner(u, dv/du_k)
                auto db = D(b);
                auto term1 = b.component(target_component);
                if (isStructurallyZero(db)) return term1;
                // dv/du_k is scalar if v depends on u through scalar paths.
                // inner(u, scalar) doesn't make shape sense — fall back to term1.
                return term1;
            }
            if (b_is_target) {
                // inner(a, u): d/du_k = component(a, k) + inner(da/du_k, u)
                auto da = D(a);
                auto term1 = a.component(target_component);
                if (isStructurallyZero(da)) return term1;
                return term1;
            }
        }

        auto da = D(a);
        auto db = D(b);
        if (isStructurallyZero(da) && isStructurallyZero(db))
            return forms::FormExpr::constant(0.0);
        // Children successfully differentiated → both are scalar.
        return smartAdd(smartMul(da, b), smartMul(a, db));
    }

    // For scalars: trace/transpose/det are identity.
    case FT::Transpose:
    case FT::Trace:
    case FT::Determinant: {
        auto da = D(child(0));
        if (isStructurallyZero(da))
            return forms::FormExpr::constant(0.0);
        return da;
    }

    // For scalar a: inv(a) = 1/a, d(1/a)/dx = -da/a².
    case FT::Inverse: {
        auto a = child(0);
        auto da = D(a);
        if (isStructurallyZero(da))
            return forms::FormExpr::constant(0.0);
        return smartNeg(da / (a * a));
    }

    // For scalar a: norm(a) = |a|, d|a|/dx = sign(a) * da.
    case FT::Norm: {
        auto a = child(0);
        auto da = D(a);
        if (isStructurallyZero(da))
            return forms::FormExpr::constant(0.0);
        auto is_nonneg = forms::ge(a, forms::FormExpr::constant(0.0));
        auto sgn = forms::FormExpr::constant(2.0) * is_nonneg -
                   forms::FormExpr::constant(1.0);
        return sgn * da;
    }

    // For scalar a: normalize(a) = sign(a), piecewise constant.
    case FT::Normalize: {
        (void)D(child(0)); // enforce scalar check
        return forms::FormExpr::constant(0.0);
    }

    // For scalar a: sym(a) = a (identity).
    case FT::SymmetricPart: {
        auto da = D(child(0));
        if (isStructurallyZero(da))
            return forms::FormExpr::constant(0.0);
        return da;
    }

    // =================================================================
    //  History operators — constant w.r.t. current state
    // =================================================================
    case FT::HistoryWeightedSum:
    case FT::HistoryConvolution:
        return forms::FormExpr::constant(0.0);

    // =================================================================
    //  Unresolved symbols — should have been resolved by the builder
    // =================================================================
    case FT::AuxiliaryStateSymbol:
    case FT::AuxiliaryInputSymbol:
    case FT::AuxiliaryOutputSymbol:
    case FT::ParameterSymbol:
    case FT::BoundaryFunctionalSymbol:
    case FT::BoundaryIntegralSymbol:
        throw std::invalid_argument(
            "diffWrtAuxSlot: unresolved symbol '" + node.toString() +
            "' in expression.  All symbols must be resolved to slot "
            "references by the model builder before differentiation.");

    // =================================================================
    //  Everything else: matrix/spectral/FE/DG/constitutive operations
    // =================================================================
    default:
        throw std::invalid_argument(
            "diffWrtAuxSlot: unsupported FormExprType " +
            std::to_string(static_cast<int>(type)) +
            " in expression '" + node.toString() +
            "'.  Symbolic differentiation for auxiliary state models "
            "supports arithmetic, transcendental (pow, sqrt, exp, log), "
            "piecewise (abs, sign, min, max, conditional), smooth "
            "approximations, scalar tensor ops (inner, trace, det, inv, "
            "norm), and history operators.  Use FiniteDifference or "
            "provide analytic Jacobians for matrix/spectral/FE ops.");
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  Setup: resolve derivative source
// ---------------------------------------------------------------------------

void AuxiliaryDerivativeProvider::setup(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativePolicy& policy)
{
    // Reset all mutable state so the provider is safe to reuse.
    is_setup_ = false;
    use_analytic_ = false;
    resolved_source_ = AuxiliaryDerivativeSource::Symbolic;
    artifact_ = AuxiliaryDerivativeArtifact{};
    fd_scratch_residual_.clear();
    fd_scratch_perturbed_.clear();
    fd_scratch_x_.clear();

    policy_ = policy;
    const int n = model.dimension();

    // Step 1: Check analytic override.
    if (policy.analytic_override_enabled && model.hasAnalyticJacobian()) {
        use_analytic_ = true;
        resolved_source_ = AuxiliaryDerivativeSource::Analytic;
        artifact_.model_name = model.modelName();
        artifact_.n = n;
        artifact_.valid = true;
        artifact_.source = AuxiliaryDerivativeSource::Analytic;
        is_setup_ = true;
        return;
    }
    use_analytic_ = false;

    // Step 2: Try requested source.
    resolved_source_ = policy.jacobian_source;

    switch (resolved_source_) {
        case AuxiliaryDerivativeSource::Symbolic: {
            if (model.hasResidualExpressions()) {
                try {
                    auto residual_exprs = model.residualExpressions();
                    artifact_.model_name = model.modelName();
                    artifact_.n = n;
                    artifact_.dF_dx_exprs.resize(static_cast<std::size_t>(n * n));

                    auto meta = model.structuralMetadata();
                    for (int i = 0; i < n; ++i) {
                        const bool is_ode =
                            (i < static_cast<int>(meta.variable_kinds.size()) &&
                             meta.variable_kinds[static_cast<std::size_t>(i)] ==
                                 AuxiliaryVariableKind::Differential);
                        for (int j = 0; j < n; ++j) {
                            auto d = diffWrt(
                                residual_exprs[static_cast<std::size_t>(i)],
                                DiffTarget::State,
                                static_cast<std::uint32_t>(j));
                            artifact_.dF_dx_exprs[static_cast<std::size_t>(i * n + j)] =
                                is_ode ? smartNeg(d) : d;
                        }
                    }

                    // Generate dF/d(inputs) if the model has input-dependent
                    // expressions (AuxiliaryInputRef terminals).
                    // Determine input count: built models use signature, custom
                    // models use declaredInputNames().
                    {
                        int n_inp = 0;
                        if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(&model)) {
                            for (const auto& inp : built->signature().inputs) {
                                n_inp += inp.size;
                            }
                        } else {
                            for (const auto& raw : model.declaredInputNames()) {
                                auto colon = raw.find(':');
                                if (colon == std::string::npos) {
                                    ++n_inp;
                                } else {
                                    try {
                                        n_inp += std::stoi(raw.substr(colon + 1));
                                    } catch (const std::exception&) {
                                        ++n_inp;
                                    }
                                }
                            }
                        }
                        if (n_inp > 0) {
                            artifact_.dF_dinputs_exprs.resize(
                                static_cast<std::size_t>(n * n_inp));
                            artifact_.n_inputs = n_inp;
                            for (int i = 0; i < n; ++i) {
                                const bool is_ode_row =
                                    (i < static_cast<int>(meta.variable_kinds.size()) &&
                                     meta.variable_kinds[static_cast<std::size_t>(i)] ==
                                         AuxiliaryVariableKind::Differential);
                                for (int k = 0; k < n_inp; ++k) {
                                    auto d = diffWrt(
                                        residual_exprs[static_cast<std::size_t>(i)],
                                        DiffTarget::Input,
                                        static_cast<std::uint32_t>(k));
                                    artifact_.dF_dinputs_exprs[
                                        static_cast<std::size_t>(i * n_inp + k)] =
                                        is_ode_row ? smartNeg(d) : d;
                                }
                            }
                        }
                    }

                    artifact_.variable_kinds = meta.variable_kinds;
                    artifact_.source = AuxiliaryDerivativeSource::Symbolic;
                    artifact_.valid = true;
                    resolved_source_ = AuxiliaryDerivativeSource::Symbolic;

                    // Generate d(output)/d(state) for the transpose Jacobian block.
                    if (auto* built_model = dynamic_cast<const BuiltAuxiliaryModel*>(&model)) {
                        const auto& out_exprs = built_model->outputExpressions();
                        const int n_out = static_cast<int>(out_exprs.size());
                        if (n_out > 0) {
                            artifact_.n_outputs = n_out;
                            artifact_.dOutput_dx_exprs.resize(
                                static_cast<std::size_t>(n_out * n));
                            for (int k = 0; k < n_out; ++k) {
                                for (int j = 0; j < n; ++j) {
                                    artifact_.dOutput_dx_exprs[
                                        static_cast<std::size_t>(k * n + j)] =
                                        diffWrt(out_exprs[static_cast<std::size_t>(k)].second,
                                                DiffTarget::State,
                                                static_cast<std::uint32_t>(j));
                                }
                            }
                        }
                    }

                    // Generate d(output)/d(input) for the direct coupling term.
                    if (auto* built_model2 = dynamic_cast<const BuiltAuxiliaryModel*>(&model)) {
                        const auto& out_exprs2 = built_model2->outputExpressions();
                        const int n_out2 = static_cast<int>(out_exprs2.size());
                        const auto& sig2 = built_model2->signature();
                        int total_inp = 0;
                        for (const auto& inp : sig2.inputs) total_inp += inp.size;
                        if (n_out2 > 0 && total_inp > 0) {
                            artifact_.dOutput_dInputs_exprs.resize(
                                static_cast<std::size_t>(n_out2 * total_inp));
                            int inp_offset = 0;
                            for (const auto& inp : sig2.inputs) {
                                for (int ic = 0; ic < inp.size; ++ic) {
                                    for (int k = 0; k < n_out2; ++k) {
                                        artifact_.dOutput_dInputs_exprs[
                                            static_cast<std::size_t>(k * total_inp + inp_offset + ic)] =
                                            diffWrt(out_exprs2[static_cast<std::size_t>(k)].second,
                                                    DiffTarget::Input,
                                                    static_cast<std::uint32_t>(inp_offset + ic));
                                    }
                                }
                                inp_offset += inp.size;
                            }
                        }
                    }

                    // Generate dF/d(field_comp) for FE field terminals.
                    // Scan residual expressions for DiscreteField/StateField nodes,
                    // capturing FieldId and value_dimension (number of components).
                    {
                        std::unordered_map<FieldId, int> found_fields;  // fid → n_comp
                        for (const auto& rexpr : residual_exprs) {
                            if (!rexpr.isValid() || !rexpr.node()) continue;
                            std::function<void(const forms::FormExprNode&)> scan =
                                [&](const forms::FormExprNode& nd) {
                                    if (nd.type() == forms::FormExprType::DiscreteField ||
                                        nd.type() == forms::FormExprType::StateField) {
                                        auto fid = nd.fieldId();
                                        if (fid) {
                                            const auto* sig = nd.spaceSignature();
                                            const int nc = (sig && sig->value_dimension > 1)
                                                ? sig->value_dimension : 1;
                                            auto it = found_fields.find(*fid);
                                            if (it == found_fields.end())
                                                found_fields[*fid] = nc;
                                            else
                                                it->second = std::max(it->second, nc);
                                        }
                                    }
                                    for (const auto* c : nd.children()) {
                                        if (c) scan(*c);
                                    }
                                };
                            scan(*rexpr.node());
                        }

                        artifact_.referenced_fields.clear();
                        for (const auto& [fid, nc] : found_fields)
                            artifact_.referenced_fields.push_back(fid);
                        std::sort(artifact_.referenced_fields.begin(),
                                  artifact_.referenced_fields.end());

                        for (const auto fid : artifact_.referenced_fields) {
                            const int nc = found_fields[fid];
                            artifact_.dF_dfield_ncomp[fid] = nc;
                            auto& dvec = artifact_.dF_dfield_exprs[fid];
                            dvec.resize(static_cast<std::size_t>(n * nc));
                            for (int i = 0; i < n; ++i) {
                                const bool is_ode =
                                    (i < static_cast<int>(meta.variable_kinds.size()) &&
                                     meta.variable_kinds[static_cast<std::size_t>(i)] ==
                                         AuxiliaryVariableKind::Differential);
                                for (int c = 0; c < nc; ++c) {
                                    // target_component: -1 for scalar, 0..nc-1 for vector
                                    const int tc = (nc > 1) ? c : -1;
                                    auto d = diffWrt(
                                        residual_exprs[static_cast<std::size_t>(i)],
                                        DiffTarget::Field,
                                        static_cast<std::uint32_t>(fid), tc);
                                    dvec[static_cast<std::size_t>(i * nc + c)] =
                                        is_ode ? smartNeg(d) : d;
                                }
                            }
                        }
                    }
                } catch (const std::invalid_argument& ex) {
                    // Expressions contain unsupported ops — fall back to FD.
                    artifact_.valid = false;
                    artifact_.source = AuxiliaryDerivativeSource::FiniteDifference;
                    artifact_.dF_dx_exprs.clear();
                    artifact_.fallback_reason = ex.what();
                    resolved_source_ = AuxiliaryDerivativeSource::FiniteDifference;

                    std::fprintf(stderr,
                        "[AuxiliaryDerivativeProvider] model '%s': symbolic "
                        "differentiation failed, falling back to finite "
                        "differences. Reason: %s\n",
                        model.modelName().c_str(), ex.what());
                }
            } else {
                // No expressions — fall back to FD.
                artifact_.source = AuxiliaryDerivativeSource::FiniteDifference;
                resolved_source_ = AuxiliaryDerivativeSource::FiniteDifference;
            }
            break;
        }

        case AuxiliaryDerivativeSource::FiniteDifference:
            artifact_.source = AuxiliaryDerivativeSource::FiniteDifference;
            break;
    }

    // Pre-allocate FD scratch buffers when FD is the resolved source.
    if (resolved_source_ == AuxiliaryDerivativeSource::FiniteDifference) {
        fd_scratch_residual_.resize(static_cast<std::size_t>(n), 0.0);
        fd_scratch_perturbed_.resize(static_cast<std::size_t>(n), 0.0);
        fd_scratch_x_.resize(static_cast<std::size_t>(n), 0.0);
    }

    artifact_.model_name = model.modelName();
    artifact_.n = n;
    is_setup_ = true;
}

// ---------------------------------------------------------------------------
//  Jacobian evaluation
// ---------------------------------------------------------------------------

void AuxiliaryDerivativeProvider::evaluateJacobian(
    const AuxiliaryStateModel& model,
    const AuxiliaryLocalContext& ctx,
    AuxiliaryJacobianRequest& request) const
{
    FE_THROW_IF(!is_setup_, InvalidStateException,
                "AuxiliaryDerivativeProvider::evaluateJacobian: not set up");

    if (use_analytic_) {
        model.evaluateJacobian(ctx, request);
        return;
    }

    // Symbolic: evaluate cached derivative expressions.
    if (resolved_source_ == AuxiliaryDerivativeSource::Symbolic && artifact_.valid) {
        const int n = request.n;
        forms::PointEvalContext pctx;
        pctx.time = ctx.time;
        pctx.dt = (ctx.effective_dt > 0.0) ? ctx.effective_dt : ctx.dt;
        pctx.coupled_aux = ctx.x;
        pctx.auxiliary_inputs = ctx.inputs;
        pctx.jit_constants = ctx.params;

        if (!request.dF_dx.empty()) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    const auto idx = static_cast<std::size_t>(i * n + j);
                    request.dF_dx[idx] =
                        forms::evaluateScalarAt(artifact_.dF_dx_exprs[idx], pctx);
                }
            }
        }

        // dF/d(inputs): evaluate symbolic input sensitivity expressions.
        if (!request.dF_dinputs.empty() && !artifact_.dF_dinputs_exprs.empty()) {
            const int n_inp = artifact_.n_inputs;
            for (int i = 0; i < n; ++i) {
                for (int k = 0; k < n_inp; ++k) {
                    const auto idx = static_cast<std::size_t>(i * n_inp + k);
                    if (idx < artifact_.dF_dinputs_exprs.size()) {
                        request.dF_dinputs[idx] =
                            forms::evaluateScalarAt(artifact_.dF_dinputs_exprs[idx], pctx);
                    }
                }
            }
        }

        // dF/dxdot: identity for ODE rows (F = xdot - rhs), zero for algebraic rows.
        if (request.want_dF_dxdot && !request.dF_dxdot.empty()) {
            std::fill(request.dF_dxdot.begin(), request.dF_dxdot.end(), 0.0);
            for (int i = 0; i < n; ++i) {
                const bool is_ode =
                    (i < static_cast<int>(artifact_.variable_kinds.size()) &&
                     artifact_.variable_kinds[static_cast<std::size_t>(i)] ==
                         AuxiliaryVariableKind::Differential);
                if (is_ode) {
                    request.dF_dxdot[static_cast<std::size_t>(i * n + i)] = 1.0;
                }
            }
        }
        return;
    }

    // Fallback: finite differences.
    evaluateJacobianFD(model, ctx, request);
}

// ---------------------------------------------------------------------------
//  Finite-difference Jacobian
// ---------------------------------------------------------------------------

void AuxiliaryDerivativeProvider::evaluateJacobianFD(
    const AuxiliaryStateModel& model,
    const AuxiliaryLocalContext& ctx,
    AuxiliaryJacobianRequest& request) const
{
    const int n = request.n;
    const double eps = policy_.fd_epsilon;

    // Evaluate base residual.
    AuxiliaryResidualRequest base_req;
    base_req.residual = fd_scratch_residual_;
    const_cast<AuxiliaryStateModel&>(model).evaluateResidual(ctx, base_req);

    // Copy current x for perturbation.
    std::copy(ctx.x.begin(), ctx.x.end(), fd_scratch_x_.begin());

    // dF/dx by forward finite differences.
    if (!request.dF_dx.empty()) {
        for (int j = 0; j < n; ++j) {
            const Real x_j_orig = fd_scratch_x_[static_cast<std::size_t>(j)];
            const Real h = eps * (1.0 + std::abs(x_j_orig));

            fd_scratch_x_[static_cast<std::size_t>(j)] = x_j_orig + h;

            // Build perturbed context.
            AuxiliaryLocalContext pert_ctx = ctx;
            pert_ctx.x = fd_scratch_x_;

            AuxiliaryResidualRequest pert_req;
            pert_req.residual = fd_scratch_perturbed_;
            const_cast<AuxiliaryStateModel&>(model).evaluateResidual(pert_ctx, pert_req);

            for (int i = 0; i < n; ++i) {
                const auto idx = static_cast<std::size_t>(i * n + j);
                request.dF_dx[idx] =
                    (fd_scratch_perturbed_[static_cast<std::size_t>(i)] -
                     fd_scratch_residual_[static_cast<std::size_t>(i)]) / h;
            }

            fd_scratch_x_[static_cast<std::size_t>(j)] = x_j_orig;
        }
    }

    // dF/d(xdot) by forward finite differences.
    if (request.want_dF_dxdot && !request.dF_dxdot.empty()) {
        std::vector<Real> xdot_scratch(ctx.xdot.begin(), ctx.xdot.end());

        for (int j = 0; j < n; ++j) {
            const Real xdot_j_orig = xdot_scratch[static_cast<std::size_t>(j)];
            const Real h = eps * (1.0 + std::abs(xdot_j_orig));

            xdot_scratch[static_cast<std::size_t>(j)] = xdot_j_orig + h;

            AuxiliaryLocalContext pert_ctx = ctx;
            pert_ctx.xdot = xdot_scratch;

            AuxiliaryResidualRequest pert_req;
            pert_req.residual = fd_scratch_perturbed_;
            const_cast<AuxiliaryStateModel&>(model).evaluateResidual(pert_ctx, pert_req);

            for (int i = 0; i < n; ++i) {
                const auto idx = static_cast<std::size_t>(i * n + j);
                request.dF_dxdot[idx] =
                    (fd_scratch_perturbed_[static_cast<std::size_t>(i)] -
                     fd_scratch_residual_[static_cast<std::size_t>(i)]) / h;
            }

            xdot_scratch[static_cast<std::size_t>(j)] = xdot_j_orig;
        }
    }

    // dF/d(inputs) by forward finite differences.
    if (!request.dF_dinputs.empty() && request.n_inputs > 0 && !ctx.inputs.empty()) {
        std::vector<Real> inp_scratch(ctx.inputs.begin(), ctx.inputs.end());

        for (int k = 0; k < request.n_inputs; ++k) {
            const Real inp_k_orig = inp_scratch[static_cast<std::size_t>(k)];
            const Real h = eps * (1.0 + std::abs(inp_k_orig));

            inp_scratch[static_cast<std::size_t>(k)] = inp_k_orig + h;

            AuxiliaryLocalContext pert_ctx = ctx;
            pert_ctx.inputs = inp_scratch;

            AuxiliaryResidualRequest pert_req;
            pert_req.residual = fd_scratch_perturbed_;
            const_cast<AuxiliaryStateModel&>(model).evaluateResidual(pert_ctx, pert_req);

            for (int i = 0; i < n; ++i) {
                const auto idx = static_cast<std::size_t>(i * request.n_inputs + k);
                request.dF_dinputs[idx] =
                    (fd_scratch_perturbed_[static_cast<std::size_t>(i)] -
                     fd_scratch_residual_[static_cast<std::size_t>(i)]) / h;
            }

            inp_scratch[static_cast<std::size_t>(k)] = inp_k_orig;
        }
    }
}

// ---------------------------------------------------------------------------
//  Hessian evaluation
// ---------------------------------------------------------------------------

std::vector<Real> AuxiliaryDerivativeProvider::evaluateFieldDerivative(
    FieldId field,
    const AuxiliaryLocalContext& ctx) const
{
    if (!is_setup_ || !artifact_.valid) return {};

    auto it = artifact_.dF_dfield_exprs.find(field);
    if (it == artifact_.dF_dfield_exprs.end()) return {};

    const auto& exprs = it->second;
    const int n = artifact_.n;

    // Determine number of components for this field.
    auto nc_it = artifact_.dF_dfield_ncomp.find(field);
    const int nc = (nc_it != artifact_.dF_dfield_ncomp.end()) ? nc_it->second : 1;

    forms::PointEvalContext pctx;
    pctx.time = ctx.time;
    pctx.dt = (ctx.effective_dt > 0.0) ? ctx.effective_dt : ctx.dt;
    pctx.coupled_aux = ctx.x;
    pctx.auxiliary_inputs = ctx.inputs;
    pctx.jit_constants = ctx.params;
    pctx.field_values = ctx.field_values;

    // Result: n_rows * n_components, row-major: [row * nc + comp].
    std::vector<Real> result(static_cast<std::size_t>(n * nc));
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < nc; ++c) {
            const auto idx = static_cast<std::size_t>(i * nc + c);
            if (idx < exprs.size()) {
                result[idx] = forms::evaluateScalarAt(exprs[idx], pctx);
            }
        }
    }
    return result;
}

void AuxiliaryDerivativeProvider::evaluateHessian(
    const AuxiliaryStateModel& model,
    const AuxiliaryLocalContext& ctx,
    AuxiliaryHessianRequest& request) const
{
    FE_THROW_IF(!is_setup_, InvalidStateException,
                "AuxiliaryDerivativeProvider::evaluateHessian: not set up");

    if (model.hasAnalyticHessian()) {
        model.evaluateHessian(ctx, request);
        return;
    }

    // Symbolic Hessian: differentiate the Jacobian expressions again.
    if (resolved_source_ == AuxiliaryDerivativeSource::Symbolic && artifact_.valid &&
        !artifact_.dF_dx_exprs.empty()) {

        const int n = request.n;

        // Generate Hessian expressions on first request (lazy).
        if (!artifact_.hessian_generated) {
            artifact_.d2F_dx2_exprs.resize(static_cast<std::size_t>(n * n * n));
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    const auto& dFi_dxj = artifact_.dF_dx_exprs[
                        static_cast<std::size_t>(i * n + j)];
                    for (int k = 0; k < n; ++k) {
                        artifact_.d2F_dx2_exprs[
                            static_cast<std::size_t>(i * n * n + j * n + k)] =
                            diffWrt(dFi_dxj, DiffTarget::State,
                                    static_cast<std::uint32_t>(k));
                    }
                }
            }
            artifact_.hessian_generated = true;
        }

        forms::PointEvalContext pctx;
        pctx.time = ctx.time;
        pctx.dt = (ctx.effective_dt > 0.0) ? ctx.effective_dt : ctx.dt;
        pctx.coupled_aux = ctx.x;
        pctx.auxiliary_inputs = ctx.inputs;
        pctx.jit_constants = ctx.params;

        switch (request.mode) {
            case AuxiliarySecondDerivativeMode::Hessian: {
                // Full Hessian: d²F_i/(dx_j dx_k) for all i,j,k.
                FE_THROW_IF(request.hessian.size() < static_cast<std::size_t>(n * n * n),
                            InvalidArgumentException,
                            "evaluateHessian: hessian buffer too small");
                for (std::size_t idx = 0; idx < static_cast<std::size_t>(n * n * n); ++idx) {
                    request.hessian[idx] = forms::evaluateScalarAt(
                        artifact_.d2F_dx2_exprs[idx], pctx);
                }
                break;
            }
            case AuxiliarySecondDerivativeMode::HessianVectorProduct: {
                // HVP: Σ_k d²F_i/(dx_j dx_k) * v_k → n×n matrix.
                FE_THROW_IF(request.hvp.size() < static_cast<std::size_t>(n * n),
                            InvalidArgumentException,
                            "evaluateHessian: hvp buffer too small");
                FE_THROW_IF(request.direction.size() < static_cast<std::size_t>(n),
                            InvalidArgumentException,
                            "evaluateHessian: direction vector too small");

                std::fill(request.hvp.begin(),
                          request.hvp.begin() + static_cast<std::ptrdiff_t>(n * n), 0.0);

                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        Real sum = 0.0;
                        for (int k = 0; k < n; ++k) {
                            const auto val = forms::evaluateScalarAt(
                                artifact_.d2F_dx2_exprs[
                                    static_cast<std::size_t>(i * n * n + j * n + k)],
                                pctx);
                            sum += val * request.direction[static_cast<std::size_t>(k)];
                        }
                        request.hvp[static_cast<std::size_t>(i * n + j)] = sum;
                    }
                }
                break;
            }
            default:
                break;
        }
        return;
    }

    // FD fallback for Hessian: double-perturbation.
    if (request.mode == AuxiliarySecondDerivativeMode::Hessian ||
        request.mode == AuxiliarySecondDerivativeMode::HessianVectorProduct) {

        const int n = request.n;
        const Real eps = 1e-6;

        // Compute base Jacobian.
        std::vector<Real> jac_base(static_cast<std::size_t>(n * n));
        AuxiliaryJacobianRequest jreq;
        jreq.dF_dx = jac_base;
        jreq.n = n;
        evaluateJacobian(model, ctx, jreq);

        if (request.mode == AuxiliarySecondDerivativeMode::Hessian) {
            FE_THROW_IF(request.hessian.size() < static_cast<std::size_t>(n * n * n),
                        InvalidArgumentException,
                        "evaluateHessian: hessian buffer too small");

            std::vector<Real> x_pert(ctx.x.begin(), ctx.x.end());
            AuxiliaryLocalContext ctx_pert = ctx;
            ctx_pert.x = x_pert;

            for (int k = 0; k < n; ++k) {
                const Real orig = x_pert[static_cast<std::size_t>(k)];
                x_pert[static_cast<std::size_t>(k)] = orig + eps;

                std::vector<Real> jac_pert(static_cast<std::size_t>(n * n));
                AuxiliaryJacobianRequest jreq_p;
                jreq_p.dF_dx = jac_pert;
                jreq_p.n = n;
                evaluateJacobian(model, ctx_pert, jreq_p);

                x_pert[static_cast<std::size_t>(k)] = orig;

                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        request.hessian[static_cast<std::size_t>(i * n * n + j * n + k)] =
                            (jac_pert[static_cast<std::size_t>(i * n + j)] -
                             jac_base[static_cast<std::size_t>(i * n + j)]) / eps;
                    }
                }
            }
        } else {
            // HVP via FD: perturb in direction v, compute (J(x+eps*v) - J(x))/eps.
            FE_THROW_IF(request.hvp.size() < static_cast<std::size_t>(n * n),
                        InvalidArgumentException,
                        "evaluateHessian: hvp buffer too small");
            FE_THROW_IF(request.direction.size() < static_cast<std::size_t>(n),
                        InvalidArgumentException,
                        "evaluateHessian: direction vector too small");

            std::vector<Real> x_pert(ctx.x.begin(), ctx.x.end());
            for (int k = 0; k < n; ++k) {
                x_pert[static_cast<std::size_t>(k)] +=
                    eps * request.direction[static_cast<std::size_t>(k)];
            }

            AuxiliaryLocalContext ctx_pert = ctx;
            ctx_pert.x = x_pert;

            std::vector<Real> jac_pert(static_cast<std::size_t>(n * n));
            AuxiliaryJacobianRequest jreq_p;
            jreq_p.dF_dx = jac_pert;
            jreq_p.n = n;
            evaluateJacobian(model, ctx_pert, jreq_p);

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    request.hvp[static_cast<std::size_t>(i * n + j)] =
                        (jac_pert[static_cast<std::size_t>(i * n + j)] -
                         jac_base[static_cast<std::size_t>(i * n + j)]) / eps;
                }
            }
        }
        return;
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
