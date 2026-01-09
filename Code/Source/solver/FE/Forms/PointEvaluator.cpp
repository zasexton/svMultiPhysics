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

[[nodiscard]] Value scalar(Real s)
{
    Value out;
    out.kind = Value::Kind::Scalar;
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

[[nodiscard]] Real requireScalar(const Value& v, std::string_view where)
{
    if (v.kind != Value::Kind::Scalar) {
        throw std::invalid_argument(std::string(where) + ": expected scalar expression");
    }
    return v.s;
}

[[nodiscard]] Value eval(const FormExprNode& node, const PointEvalContext& ctx);

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

[[nodiscard]] Value eval(const FormExprNode& node, const PointEvalContext& ctx)
{
    switch (node.type()) {
        case FormExprType::Constant:
            return scalar(node.constantValue().value_or(0.0));
        case FormExprType::Time:
            return scalar(ctx.time);
        case FormExprType::TimeStep:
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
            break;
    }

    throw std::invalid_argument("PointEvaluator: expression node type not supported");
}

[[nodiscard]] bool containsTime(const FormExprNode& node)
{
    bool found = false;
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::Time || n.type() == FormExprType::TimeStep || n.timeScalarCoefficient()) {
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
