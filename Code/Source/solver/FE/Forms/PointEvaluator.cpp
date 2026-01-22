/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/PointEvaluator.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace svmp {
namespace FE {
namespace forms {

namespace {

struct Value {
    enum class Kind { Scalar, Vector } kind{Kind::Scalar};
    Real s{0.0};
    std::array<Real, 3> v{0.0, 0.0, 0.0};
};

struct DualValue {
    enum class Kind { Scalar, Vector } kind{Kind::Scalar};
    Dual s{};
    std::array<Real, 3> v{0.0, 0.0, 0.0};
};

[[nodiscard]] Value scalar(Real s)
{
    Value out;
    out.kind = Value::Kind::Scalar;
    out.s = s;
    return out;
}

[[nodiscard]] DualValue scalar(Dual s)
{
    DualValue out;
    out.kind = DualValue::Kind::Scalar;
    out.s = s;
    return out;
}

[[nodiscard]] Value vector(const std::array<Real, 3>& v)
{
    Value out;
    out.kind = Value::Kind::Vector;
    out.v = v;
    return out;
}

[[nodiscard]] DualValue vectorDual(const std::array<Real, 3>& v)
{
    DualValue out;
    out.kind = DualValue::Kind::Vector;
    out.v = v;
    return out;
}

[[nodiscard]] Real requireScalar(const Value& v, std::string_view where)
{
    if (v.kind != Value::Kind::Scalar) {
        throw std::invalid_argument(std::string(where) + ": expected scalar expression");
    }
    return v.s;
}

[[nodiscard]] const Dual& requireScalar(const DualValue& v, std::string_view where)
{
    if (v.kind != DualValue::Kind::Scalar) {
        throw std::invalid_argument(std::string(where) + ": expected scalar expression");
    }
    return v.s;
}

[[nodiscard]] Value eval(const FormExprNode& node, const PointEvalContext& ctx);

[[nodiscard]] DualValue evalDual(const FormExprNode& node,
                                 const PointEvalContext& ctx,
                                 DualWorkspace& ws,
                                 const PointDualSeedContext& seeds);

[[nodiscard]] Value evalUnaryScalar(const FormExprNode& node,
                                    const PointEvalContext& ctx,
                                    const std::function<Real(Real)>& f)
{
    const auto kids = node.childrenShared();
    if (kids.size() != 1u || !kids[0]) {
        throw std::invalid_argument("PointEvaluator: unary node must have exactly 1 child");
    }
    const auto a = requireScalar(eval(*kids[0], ctx), "PointEvaluator");
    return scalar(f(a));
}

[[nodiscard]] DualValue evalUnaryScalarDual(const FormExprNode& node,
                                            const PointEvalContext& ctx,
                                            DualWorkspace& ws,
                                            const PointDualSeedContext& seeds,
                                            const std::function<Dual(const Dual&, Dual)>& f)
{
    const auto kids = node.childrenShared();
    if (kids.size() != 1u || !kids[0]) {
        throw std::invalid_argument("PointEvaluator: unary node must have exactly 1 child");
    }
    const auto& a = requireScalar(evalDual(*kids[0], ctx, ws, seeds), "PointEvaluatorDual");
    Dual out = makeDualConstant(0.0, ws.alloc());
    return scalar(f(a, out));
}

[[nodiscard]] Value evalBinaryScalar(const FormExprNode& node,
                                     const PointEvalContext& ctx,
                                     const std::function<Real(Real, Real)>& f)
{
    const auto kids = node.childrenShared();
    if (kids.size() != 2u || !kids[0] || !kids[1]) {
        throw std::invalid_argument("PointEvaluator: binary node must have exactly 2 children");
    }
    const auto a = requireScalar(eval(*kids[0], ctx), "PointEvaluator");
    const auto b = requireScalar(eval(*kids[1], ctx), "PointEvaluator");
    return scalar(f(a, b));
}

[[nodiscard]] DualValue evalBinaryScalarDual(const FormExprNode& node,
                                             const PointEvalContext& ctx,
                                             DualWorkspace& ws,
                                             const PointDualSeedContext& seeds,
                                             const std::function<Dual(const Dual&, const Dual&, Dual)>& f)
{
    const auto kids = node.childrenShared();
    if (kids.size() != 2u || !kids[0] || !kids[1]) {
        throw std::invalid_argument("PointEvaluator: binary node must have exactly 2 children");
    }
    const auto& a = requireScalar(evalDual(*kids[0], ctx, ws, seeds), "PointEvaluatorDual");
    const auto& b = requireScalar(evalDual(*kids[1], ctx, ws, seeds), "PointEvaluatorDual");
    Dual out = makeDualConstant(0.0, ws.alloc());
    return scalar(f(a, b, out));
}

[[nodiscard]] Value eval(const FormExprNode& node, const PointEvalContext& ctx)
{
    switch (node.type()) {
        case FormExprType::Constant:
            return scalar(node.constantValue().value_or(0.0));
        case FormExprType::Time:
            return scalar(ctx.time);
        case FormExprType::TimeStep:
            return scalar(ctx.dt);
        case FormExprType::EffectiveTimeStep:
            // PointEvalContext does not carry a TimeIntegrationContext; fall back to the physical step size.
            return scalar(ctx.dt);
        case FormExprType::Coordinate:
            return vector(ctx.x);
        case FormExprType::ReferenceCoordinate:
            throw std::invalid_argument("PointEvaluator: ReferenceCoordinate is not available in PointEvalContext");
        case FormExprType::ParameterRef: {
            const auto slot = node.slotIndex();
            if (!slot) {
                throw std::invalid_argument("PointEvaluator: ParameterRef node missing slot");
            }
            if (ctx.jit_constants.empty()) {
                throw std::invalid_argument("PointEvaluator: ParameterRef requires PointEvalContext::jit_constants");
            }
            if (*slot >= ctx.jit_constants.size()) {
                throw std::out_of_range("PointEvaluator: ParameterRef slot out of range");
            }
            return scalar(ctx.jit_constants[*slot]);
        }
        case FormExprType::BoundaryIntegralRef: {
            const auto slot = node.slotIndex();
            if (!slot) {
                throw std::invalid_argument("PointEvaluator: BoundaryIntegralRef node missing slot");
            }
            if (ctx.coupled_integrals.empty()) {
                throw std::invalid_argument("PointEvaluator: BoundaryIntegralRef requires PointEvalContext::coupled_integrals");
            }
            if (*slot >= ctx.coupled_integrals.size()) {
                throw std::out_of_range("PointEvaluator: BoundaryIntegralRef slot out of range");
            }
            return scalar(ctx.coupled_integrals[*slot]);
        }
        case FormExprType::AuxiliaryStateRef: {
            const auto slot = node.slotIndex();
            if (!slot) {
                throw std::invalid_argument("PointEvaluator: AuxiliaryStateRef node missing slot");
            }
            if (ctx.coupled_aux.empty()) {
                throw std::invalid_argument("PointEvaluator: AuxiliaryStateRef requires PointEvalContext::coupled_aux");
            }
            if (*slot >= ctx.coupled_aux.size()) {
                throw std::out_of_range("PointEvaluator: AuxiliaryStateRef slot out of range");
            }
            return scalar(ctx.coupled_aux[*slot]);
        }
        case FormExprType::Coefficient: {
            if (const auto* f = node.timeScalarCoefficient(); f) {
                return scalar((*f)(ctx.x[0], ctx.x[1], ctx.x[2], ctx.time));
            }
            if (const auto* f = node.scalarCoefficient(); f) {
                return scalar((*f)(ctx.x[0], ctx.x[1], ctx.x[2]));
            }
            throw std::invalid_argument("PointEvaluator: only scalar coefficients are supported");
        }

        case FormExprType::Negate:
            return evalUnaryScalar(node, ctx, [](Real a) { return -a; });
        case FormExprType::Add:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return a + b; });
        case FormExprType::Subtract:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return a - b; });
        case FormExprType::Multiply:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return a * b; });
        case FormExprType::Divide:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return a / b; });
        case FormExprType::Power:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return std::pow(a, b); });
        case FormExprType::Minimum:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return std::min(a, b); });
        case FormExprType::Maximum:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return std::max(a, b); });

        case FormExprType::AbsoluteValue:
            return evalUnaryScalar(node, ctx, [](Real a) { return std::abs(a); });
        case FormExprType::Sign:
            return evalUnaryScalar(node, ctx, [](Real a) { return (a > 0) - (a < 0); });
        case FormExprType::Sqrt:
            return evalUnaryScalar(node, ctx, [](Real a) { return std::sqrt(a); });
        case FormExprType::Exp:
            return evalUnaryScalar(node, ctx, [](Real a) { return std::exp(a); });
        case FormExprType::Log:
            return evalUnaryScalar(node, ctx, [](Real a) { return std::log(a); });

        case FormExprType::Less:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return (a < b) ? 1.0 : 0.0; });
        case FormExprType::LessEqual:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return (a <= b) ? 1.0 : 0.0; });
        case FormExprType::Greater:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return (a > b) ? 1.0 : 0.0; });
        case FormExprType::GreaterEqual:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return (a >= b) ? 1.0 : 0.0; });
        case FormExprType::Equal:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return (a == b) ? 1.0 : 0.0; });
        case FormExprType::NotEqual:
            return evalBinaryScalar(node, ctx, [](Real a, Real b) { return (a != b) ? 1.0 : 0.0; });

        case FormExprType::Conditional: {
            const auto kids = node.childrenShared();
            if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                throw std::invalid_argument("PointEvaluator: conditional expects 3 operands");
            }
            const auto c = requireScalar(eval(*kids[0], ctx), "PointEvaluator");
            return (c != 0.0) ? eval(*kids[1], ctx) : eval(*kids[2], ctx);
        }

        case FormExprType::Component: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) {
                throw std::invalid_argument("PointEvaluator: component expects 1 operand");
            }
            const auto a = eval(*kids[0], ctx);
            const auto i = node.componentIndex0().value_or(0);
            const auto j = node.componentIndex1().value_or(-1);

            if (a.kind == Value::Kind::Vector) {
                if (j >= 0) {
                    throw std::invalid_argument("PointEvaluator: vector component expects 1 index");
                }
                if (i < 0 || i >= 3) {
                    throw std::out_of_range("PointEvaluator: vector component index out of range");
                }
                return scalar(a.v[static_cast<std::size_t>(i)]);
            }

            throw std::invalid_argument("PointEvaluator: component access not supported for this operand type");
        }

        // Unsupported in point evaluation:
        case FormExprType::ParameterSymbol:
        case FormExprType::BoundaryFunctionalSymbol:
        case FormExprType::BoundaryIntegralSymbol:
        case FormExprType::AuxiliaryStateSymbol:
        case FormExprType::PreviousSolutionRef:
        case FormExprType::TestFunction:
        case FormExprType::TrialFunction:
        case FormExprType::DiscreteField:
        case FormExprType::StateField:
        case FormExprType::Identity:
        case FormExprType::Jacobian:
        case FormExprType::JacobianInverse:
        case FormExprType::JacobianDeterminant:
        case FormExprType::Normal:
        case FormExprType::CellDiameter:
        case FormExprType::CellVolume:
        case FormExprType::FacetArea:
        case FormExprType::Gradient:
        case FormExprType::Divergence:
        case FormExprType::Curl:
        case FormExprType::Hessian:
        case FormExprType::TimeDerivative:
        case FormExprType::RestrictMinus:
        case FormExprType::RestrictPlus:
        case FormExprType::Jump:
        case FormExprType::Average:
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::OuterProduct:
        case FormExprType::CrossProduct:
        case FormExprType::AsVector:
        case FormExprType::AsTensor:
        case FormExprType::IndexedAccess:
        case FormExprType::Transpose:
        case FormExprType::Trace:
        case FormExprType::Determinant:
        case FormExprType::Inverse:
        case FormExprType::Cofactor:
        case FormExprType::Deviator:
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart:
        case FormExprType::Norm:
        case FormExprType::Normalize:
        case FormExprType::Constitutive:
        case FormExprType::ConstitutiveOutput:
        case FormExprType::CellIntegral:
        case FormExprType::BoundaryIntegral:
        case FormExprType::InteriorFaceIntegral:
        case FormExprType::InterfaceIntegral:
            break;
    }

    throw std::invalid_argument("PointEvaluator: expression node type not supported");
}

[[nodiscard]] DualValue evalDual(const FormExprNode& node,
                                 const PointEvalContext& ctx,
                                 DualWorkspace& ws,
                                 const PointDualSeedContext& seeds)
{
    switch (node.type()) {
        case FormExprType::Constant: {
            const Real v = node.constantValue().value_or(0.0);
            return scalar(makeDualConstant(v, ws.alloc()));
        }
        case FormExprType::Time:
            return scalar(makeDualConstant(ctx.time, ws.alloc()));
        case FormExprType::TimeStep:
            return scalar(makeDualConstant(ctx.dt, ws.alloc()));
        case FormExprType::EffectiveTimeStep:
            // PointEvalContext does not carry a TimeIntegrationContext; fall back to the physical step size.
            return scalar(makeDualConstant(ctx.dt, ws.alloc()));
        case FormExprType::Coordinate:
            return vectorDual(ctx.x);
        case FormExprType::ReferenceCoordinate:
            throw std::invalid_argument("PointEvaluatorDual: ReferenceCoordinate is not available in PointEvalContext");
        case FormExprType::ParameterRef: {
            const auto slot = node.slotIndex();
            if (!slot) {
                throw std::invalid_argument("PointEvaluatorDual: ParameterRef node missing slot");
            }
            if (ctx.jit_constants.empty()) {
                throw std::invalid_argument("PointEvaluatorDual: ParameterRef requires PointEvalContext::jit_constants");
            }
            if (*slot >= ctx.jit_constants.size()) {
                throw std::out_of_range("PointEvaluatorDual: ParameterRef slot out of range");
            }
            return scalar(makeDualConstant(ctx.jit_constants[*slot], ws.alloc()));
        }
        case FormExprType::BoundaryIntegralRef: {
            const auto slot = node.slotIndex();
            if (!slot) {
                throw std::invalid_argument("PointEvaluatorDual: BoundaryIntegralRef node missing slot");
            }
            if (ctx.coupled_integrals.empty()) {
                throw std::invalid_argument("PointEvaluatorDual: BoundaryIntegralRef requires PointEvalContext::coupled_integrals");
            }
            if (*slot >= ctx.coupled_integrals.size()) {
                throw std::out_of_range("PointEvaluatorDual: BoundaryIntegralRef slot out of range");
            }
            Dual out = makeDualConstant(ctx.coupled_integrals[*slot], ws.alloc());
            if (*slot >= out.deriv.size()) {
                throw std::out_of_range("PointEvaluatorDual: BoundaryIntegralRef derivative slot out of range");
            }
            out.deriv[*slot] = 1.0;
            return scalar(out);
        }
        case FormExprType::AuxiliaryStateRef: {
            const auto slot = node.slotIndex();
            if (!slot) {
                throw std::invalid_argument("PointEvaluatorDual: AuxiliaryStateRef node missing slot");
            }

            const auto use_override =
                seeds.aux_override.has_value() && seeds.aux_override->slot == *slot;
            if (use_override) {
                const auto& o = *seeds.aux_override;
                Dual out = makeDualConstant(o.value, ws.alloc());
                if (o.deriv.size() != out.deriv.size()) {
                    throw std::invalid_argument("PointEvaluatorDual: aux_override derivative size mismatch");
                }
                std::copy(o.deriv.begin(), o.deriv.end(), out.deriv.begin());
                return scalar(out);
            }

            if (ctx.coupled_aux.empty()) {
                throw std::invalid_argument("PointEvaluatorDual: AuxiliaryStateRef requires PointEvalContext::coupled_aux");
            }
            if (*slot >= ctx.coupled_aux.size()) {
                throw std::out_of_range("PointEvaluatorDual: AuxiliaryStateRef slot out of range");
            }

            Dual out = makeDualConstant(ctx.coupled_aux[*slot], ws.alloc());
            if (!seeds.aux_dseed.empty()) {
                const std::size_t d = out.deriv.size();
                const std::size_t need = (static_cast<std::size_t>(*slot) + 1u) * d;
                if (seeds.aux_dseed.size() < need) {
                    throw std::out_of_range("PointEvaluatorDual: aux_dseed buffer is too small");
                }
                const auto row = seeds.aux_dseed.subspan(static_cast<std::size_t>(*slot) * d, d);
                std::copy(row.begin(), row.end(), out.deriv.begin());
            }
            return scalar(out);
        }
        case FormExprType::Coefficient: {
            if (const auto* f = node.timeScalarCoefficient(); f) {
                return scalar(makeDualConstant((*f)(ctx.x[0], ctx.x[1], ctx.x[2], ctx.time), ws.alloc()));
            }
            if (const auto* f = node.scalarCoefficient(); f) {
                return scalar(makeDualConstant((*f)(ctx.x[0], ctx.x[1], ctx.x[2]), ws.alloc()));
            }
            throw std::invalid_argument("PointEvaluatorDual: only scalar coefficients are supported");
        }

        case FormExprType::Negate:
            return evalUnaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, Dual out) { return neg(a, out); });
        case FormExprType::Add:
            return evalBinaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, const Dual& b, Dual out) { return add(a, b, out); });
        case FormExprType::Subtract:
            return evalBinaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, const Dual& b, Dual out) { return sub(a, b, out); });
        case FormExprType::Multiply:
            return evalBinaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, const Dual& b, Dual out) { return mul(a, b, out); });
        case FormExprType::Divide:
            return evalBinaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, const Dual& b, Dual out) { return div(a, b, out); });
        case FormExprType::Power: {
            // Prefer constant exponent if RHS has no derivatives.
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) {
                throw std::invalid_argument("PointEvaluatorDual: power node must have exactly 2 children");
            }
            const auto& a = requireScalar(evalDual(*kids[0], ctx, ws, seeds), "PointEvaluatorDual");
            const auto& b = requireScalar(evalDual(*kids[1], ctx, ws, seeds), "PointEvaluatorDual");
            Dual out = makeDualConstant(0.0, ws.alloc());
            const bool b_is_constant = std::all_of(b.deriv.begin(), b.deriv.end(),
                                                  [](Real v) { return v == 0.0; });
            if (b_is_constant) {
                return scalar(pow(a, b.value, out));
            }
            return scalar(pow(a, b, out));
        }
        case FormExprType::Minimum: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) {
                throw std::invalid_argument("PointEvaluatorDual: minimum node must have exactly 2 children");
            }
            const auto& a = requireScalar(evalDual(*kids[0], ctx, ws, seeds), "PointEvaluatorDual");
            const auto& b = requireScalar(evalDual(*kids[1], ctx, ws, seeds), "PointEvaluatorDual");
            Dual out = makeDualConstant(0.0, ws.alloc());
            return scalar((a.value <= b.value) ? copy(a, out) : copy(b, out));
        }
        case FormExprType::Maximum: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) {
                throw std::invalid_argument("PointEvaluatorDual: maximum node must have exactly 2 children");
            }
            const auto& a = requireScalar(evalDual(*kids[0], ctx, ws, seeds), "PointEvaluatorDual");
            const auto& b = requireScalar(evalDual(*kids[1], ctx, ws, seeds), "PointEvaluatorDual");
            Dual out = makeDualConstant(0.0, ws.alloc());
            return scalar((a.value >= b.value) ? copy(a, out) : copy(b, out));
        }

        case FormExprType::AbsoluteValue:
            return evalUnaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, Dual out) { return abs(a, out); });
        case FormExprType::Sign:
            return evalUnaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, Dual out) { return sign(a, out); });
        case FormExprType::Sqrt:
            return evalUnaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, Dual out) { return sqrt(a, out); });
        case FormExprType::Exp:
            return evalUnaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, Dual out) { return exp(a, out); });
        case FormExprType::Log:
            return evalUnaryScalarDual(node, ctx, ws, seeds, [](const Dual& a, Dual out) { return log(a, out); });

        case FormExprType::Less:
        case FormExprType::LessEqual:
        case FormExprType::Greater:
        case FormExprType::GreaterEqual:
        case FormExprType::Equal:
        case FormExprType::NotEqual: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) {
                throw std::invalid_argument("PointEvaluatorDual: comparison expects 2 operands");
            }
            const auto& a = requireScalar(evalDual(*kids[0], ctx, ws, seeds), "PointEvaluatorDual");
            const auto& b = requireScalar(evalDual(*kids[1], ctx, ws, seeds), "PointEvaluatorDual");
            bool pred = false;
            switch (node.type()) {
                case FormExprType::Less:
                    pred = a.value < b.value;
                    break;
                case FormExprType::LessEqual:
                    pred = a.value <= b.value;
                    break;
                case FormExprType::Greater:
                    pred = a.value > b.value;
                    break;
                case FormExprType::GreaterEqual:
                    pred = a.value >= b.value;
                    break;
                case FormExprType::Equal:
                    pred = a.value == b.value;
                    break;
                case FormExprType::NotEqual:
                    pred = a.value != b.value;
                    break;
                default:
                    break;
            }
            return scalar(makeDualConstant(pred ? 1.0 : 0.0, ws.alloc()));
        }

        case FormExprType::Conditional: {
            const auto kids = node.childrenShared();
            if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                throw std::invalid_argument("PointEvaluatorDual: conditional expects 3 operands");
            }
            const auto& c = requireScalar(evalDual(*kids[0], ctx, ws, seeds), "PointEvaluatorDual");
            return (c.value != 0.0) ? evalDual(*kids[1], ctx, ws, seeds)
                                    : evalDual(*kids[2], ctx, ws, seeds);
        }

        case FormExprType::Component: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) {
                throw std::invalid_argument("PointEvaluatorDual: component expects 1 operand");
            }
            const auto a = evalDual(*kids[0], ctx, ws, seeds);
            const auto i = node.componentIndex0().value_or(0);
            const auto j = node.componentIndex1().value_or(-1);
            if (a.kind == DualValue::Kind::Vector) {
                if (j >= 0) {
                    throw std::invalid_argument("PointEvaluatorDual: vector component expects 1 index");
                }
                if (i < 0 || i >= 3) {
                    throw std::out_of_range("PointEvaluatorDual: vector component index out of range");
                }
                return scalar(makeDualConstant(a.v[static_cast<std::size_t>(i)], ws.alloc()));
            }
            throw std::invalid_argument("PointEvaluatorDual: component access not supported for this operand type");
        }

        // Unsupported in point evaluation:
        case FormExprType::ParameterSymbol:
        case FormExprType::BoundaryFunctionalSymbol:
        case FormExprType::BoundaryIntegralSymbol:
        case FormExprType::AuxiliaryStateSymbol:
        case FormExprType::PreviousSolutionRef:
        case FormExprType::TestFunction:
        case FormExprType::TrialFunction:
        case FormExprType::DiscreteField:
        case FormExprType::StateField:
        case FormExprType::Identity:
        case FormExprType::Jacobian:
        case FormExprType::JacobianInverse:
        case FormExprType::JacobianDeterminant:
        case FormExprType::Normal:
        case FormExprType::CellDiameter:
        case FormExprType::CellVolume:
        case FormExprType::FacetArea:
        case FormExprType::Gradient:
        case FormExprType::Divergence:
        case FormExprType::Curl:
        case FormExprType::Hessian:
        case FormExprType::TimeDerivative:
        case FormExprType::RestrictMinus:
        case FormExprType::RestrictPlus:
        case FormExprType::Jump:
        case FormExprType::Average:
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::OuterProduct:
        case FormExprType::CrossProduct:
        case FormExprType::AsVector:
        case FormExprType::AsTensor:
        case FormExprType::IndexedAccess:
        case FormExprType::Transpose:
        case FormExprType::Trace:
        case FormExprType::Determinant:
        case FormExprType::Inverse:
        case FormExprType::Cofactor:
        case FormExprType::Deviator:
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart:
        case FormExprType::Norm:
        case FormExprType::Normalize:
        case FormExprType::Constitutive:
        case FormExprType::ConstitutiveOutput:
        case FormExprType::CellIntegral:
        case FormExprType::BoundaryIntegral:
        case FormExprType::InteriorFaceIntegral:
        case FormExprType::InterfaceIntegral:
            break;
    }

    throw std::invalid_argument("PointEvaluatorDual: expression node type not supported");
}

[[nodiscard]] bool containsTime(const FormExprNode& node)
{
    bool found = false;
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::Time || n.type() == FormExprType::TimeStep ||
            n.type() == FormExprType::EffectiveTimeStep || n.timeScalarCoefficient()) {
            found = true;
            return;
        }
        for (const auto& child : n.childrenShared()) {
            if (child && !found) self(self, *child);
        }
    };
    visit(visit, node);
    return found;
}

} // namespace

Real evaluateScalarAt(const FormExpr& expr, const PointEvalContext& ctx)
{
    if (!expr.isValid()) {
        throw std::invalid_argument("evaluateScalarAt: invalid expression");
    }
    const auto& n = *expr.node();
    if (n.hasTest() || n.hasTrial()) {
        throw std::invalid_argument("evaluateScalarAt: expression must not contain test/trial functions");
    }
    return requireScalar(eval(n, ctx), "evaluateScalarAt");
}

Dual evaluateScalarAtDual(const FormExpr& expr,
                          const PointEvalContext& ctx,
                          DualWorkspace& workspace,
                          const PointDualSeedContext& seeds)
{
    if (!expr.isValid()) {
        throw std::invalid_argument("evaluateScalarAtDual: invalid expression");
    }
    const auto& n = *expr.node();
    if (n.hasTest() || n.hasTrial()) {
        throw std::invalid_argument("evaluateScalarAtDual: expression must not contain test/trial functions");
    }

    workspace.reset(seeds.deriv_dim);

    if (seeds.aux_override.has_value()) {
        if (seeds.aux_override->deriv.size() != seeds.deriv_dim) {
            throw std::invalid_argument("evaluateScalarAtDual: aux_override derivative size mismatch");
        }
    }

    const auto v = evalDual(n, ctx, workspace, seeds);
    return requireScalar(v, "evaluateScalarAtDual");
}

bool isTimeDependent(const FormExpr& expr)
{
    if (!expr.isValid()) {
        return false;
    }
    return containsTime(*expr.node());
}

} // namespace forms
} // namespace FE
} // namespace svmp
