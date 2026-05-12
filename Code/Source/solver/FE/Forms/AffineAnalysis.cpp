/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/AffineAnalysis.h"

#include "Forms/FormCompiler.h"

#include <algorithm>
#include <cstddef>

namespace svmp {
namespace FE {
namespace forms {

namespace {

FormExpr wrapIntegral(const FormExpr& integrand, IntegralDomain domain, int boundary_marker)
{
    switch (domain) {
        case IntegralDomain::Cell:
            return integrand.dx();
        case IntegralDomain::Boundary:
            return integrand.ds(boundary_marker);
        case IntegralDomain::InteriorFace:
            return integrand.dS();
        case IntegralDomain::InterfaceFace:
            return integrand.dI(boundary_marker);
    }
    return {};
}

struct Decomp {
    int degree{0}; // 0: no TrialFunction, 1: linear in TrialFunction, 2: nonlinear/unknown
    FormExpr constant{};
    FormExpr linear{};
};

inline FormExpr zeroExpr()
{
    return FormExpr::constant(0.0);
}

bool isZeroConstant(const FormExpr& expr)
{
    const auto* node = expr.node();
    if (!node) {
        return false;
    }
    const auto value = node->constantValue();
    return value.has_value() && *value == 0.0;
}

bool decomposeAffineInTrial(const FormExpr& expr, Decomp& out, std::string* reason);

bool decomposeAffineUnaryLinear(const FormExpr& expr,
                                const FormExpr& child_expr,
                                const Decomp& child,
                                Decomp& out,
                                std::string* /*reason*/,
                                const std::function<FormExpr(const FormExpr&)>& op)
{
    out.degree = child.degree;
    out.constant = isZeroConstant(child.constant) ? zeroExpr() : op(child.constant);
    out.linear = isZeroConstant(child.linear) ? zeroExpr() : op(child.linear);
    (void)expr;
    (void)child_expr;
    return child.degree <= 1;
}

bool decomposeAffineInTrial(const FormExpr& expr, Decomp& out, std::string* reason)
{
    const auto* node = expr.node();
    if (!node) {
        if (reason) *reason = "invalid FormExpr";
        return false;
    }

    switch (node->type()) {
        // ---- Terminals ----
        case FormExprType::TrialFunction:
            out.degree = 1;
            out.constant = zeroExpr();
            out.linear = expr;
            return true;
        case FormExprType::TestFunction:
        case FormExprType::DiscreteField:
        case FormExprType::Coefficient:
        case FormExprType::TypedZero:
        case FormExprType::Constant:
        case FormExprType::Coordinate:
        case FormExprType::ReferenceCoordinate:
        case FormExprType::MeshDisplacement:
        case FormExprType::MeshVelocity:
        case FormExprType::MeshAcceleration:
        case FormExprType::CurrentCoordinate:
        case FormExprType::PreviousCoordinate:
        case FormExprType::ReferencePhysicalCoordinate:
        case FormExprType::PreviousMeshVelocity:
        case FormExprType::PredictedMeshVelocity:
        case FormExprType::CurrentJacobian:
        case FormExprType::ReferenceJacobian:
        case FormExprType::CurrentJacobianDeterminant:
        case FormExprType::ReferenceJacobianDeterminant:
        case FormExprType::CurrentNormal:
        case FormExprType::CurrentMeanCurvature:
        case FormExprType::ReferenceNormal:
        case FormExprType::CurrentMeasure:
        case FormExprType::ReferenceMeasure:
        case FormExprType::SurfaceJacobian:
        case FormExprType::Time:
        case FormExprType::TimeStep:
        case FormExprType::EffectiveTimeStep:
        case FormExprType::Identity:
        case FormExprType::Jacobian:
        case FormExprType::JacobianInverse:
        case FormExprType::JacobianDeterminant:
        case FormExprType::Normal:
        case FormExprType::CellDiameter:
        case FormExprType::CellVolume:
        case FormExprType::FacetArea:
        case FormExprType::CellDomainId:
            out.degree = 0;
            out.constant = expr;
            out.linear = zeroExpr();
            return true;

        case FormExprType::StateField:
            if (reason) *reason = "StateField is not supported by affine residual splitting";
            return false;

        // ---- Differential operators (linear) ----
        case FormExprType::Gradient:
        case FormExprType::Divergence:
        case FormExprType::Curl:
        case FormExprType::Hessian: {
            const auto kids = node->childrenShared();
            if (kids.size() != 1 || !kids[0]) {
                if (reason) *reason = "unary differential operator missing child";
                return false;
            }
            const FormExpr child(kids[0]);
            Decomp dc;
            if (!decomposeAffineInTrial(child, dc, reason)) {
                return false;
            }

            const auto op = [&](const FormExpr& e) {
                switch (node->type()) {
                    case FormExprType::Gradient: return e.grad();
                    case FormExprType::Divergence: return e.div();
                    case FormExprType::Curl: return e.curl();
                    case FormExprType::Hessian: return e.hessian();
                    default: return FormExpr{};
                }
            };

            return decomposeAffineUnaryLinear(expr, child, dc, out, reason, op);
        }

        case FormExprType::TimeDerivative:
            if (reason) *reason = "dt(...) terms are not supported by affine residual splitting";
            return false;

        // ---- Restrictions / DG ops (linear) ----
        case FormExprType::RestrictMinus:
        case FormExprType::RestrictPlus:
        case FormExprType::Jump:
        case FormExprType::Average:
        case FormExprType::Negate:
        case FormExprType::Transpose:
        case FormExprType::Trace:
        case FormExprType::Deviator:
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart:
        case FormExprType::Component:
        case FormExprType::Pullback:
        case FormExprType::Pushforward:
        {
            const auto kids = node->childrenShared();
            if (kids.size() != 1 || !kids[0]) {
                if (reason) *reason = "unary operator missing child";
                return false;
            }
            const FormExpr child(kids[0]);
            Decomp dc;
            if (!decomposeAffineInTrial(child, dc, reason)) {
                return false;
            }

            const auto op = [&](const FormExpr& e) {
                switch (node->type()) {
                    case FormExprType::RestrictMinus: return e.minus();
                    case FormExprType::RestrictPlus: return e.plus();
                    case FormExprType::Jump: return e.jump();
                    case FormExprType::Average: return e.avg();
                    case FormExprType::Negate: return -e;
                    case FormExprType::Transpose: return e.transpose();
                    case FormExprType::Trace: return e.trace();
                    case FormExprType::Deviator: return e.dev();
                    case FormExprType::SymmetricPart: return e.sym();
                    case FormExprType::SkewPart: return e.skew();
                    case FormExprType::Component: {
                        const auto c0 = node->componentIndex0().value_or(-1);
                        const auto c1 = node->componentIndex1().value_or(-1);
                        return e.component(c0, c1);
                    }
                    case FormExprType::Pullback:
                        return pullback(e,
                                        node->fromConfiguration().value_or(GeometryConfiguration::Reference),
                                        node->toConfiguration().value_or(GeometryConfiguration::Current));
                    case FormExprType::Pushforward:
                        return pushforward(e,
                                           node->fromConfiguration().value_or(GeometryConfiguration::Reference),
                                           node->toConfiguration().value_or(GeometryConfiguration::Current));
                    default:
                        return FormExpr{};
                }
            };

            out.degree = dc.degree;
            out.constant = isZeroConstant(dc.constant) ? zeroExpr() : op(dc.constant);
            out.linear = isZeroConstant(dc.linear) ? zeroExpr() : op(dc.linear);
            return dc.degree <= 1;
        }

        // ---- Binary algebra ----
        case FormExprType::Add:
        case FormExprType::Subtract:
        case FormExprType::Multiply:
        case FormExprType::Divide:
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::OuterProduct:
        case FormExprType::CrossProduct: {
            const auto kids = node->childrenShared();
            if (kids.size() != 2 || !kids[0] || !kids[1]) {
                if (reason) *reason = "binary operator missing child";
                return false;
            }
            const FormExpr a(kids[0]);
            const FormExpr b(kids[1]);

            Decomp da, db;
            if (!decomposeAffineInTrial(a, da, reason)) return false;
            if (!decomposeAffineInTrial(b, db, reason)) return false;

            const auto addOpOrZero = [&](const FormExpr& x, const FormExpr& y) {
                const bool x_zero = isZeroConstant(x);
                const bool y_zero = isZeroConstant(y);
                if (node->type() == FormExprType::Add) {
                    if (x_zero && y_zero) return zeroExpr();
                    if (x_zero) return y;
                    if (y_zero) return x;
                    return x + y;
                }
                if (x_zero && y_zero) return zeroExpr();
                if (x_zero) return -y;
                if (y_zero) return x;
                return x - y;
            };

            const auto multLike = [&](const FormExpr& x, const FormExpr& y) {
                switch (node->type()) {
                    case FormExprType::Multiply: return x * y;
                    case FormExprType::InnerProduct: return inner(x, y);
                    case FormExprType::DoubleContraction: return x.doubleContraction(y);
                    case FormExprType::OuterProduct: return x.outer(y);
                    case FormExprType::CrossProduct: return x.cross(y);
                    default: return FormExpr{};
                }
            };
            const auto multLikeOrZero = [&](const FormExpr& x, const FormExpr& y) {
                if (isZeroConstant(x) || isZeroConstant(y)) {
                    return zeroExpr();
                }
                return multLike(x, y);
            };

            if (node->type() == FormExprType::Add || node->type() == FormExprType::Subtract) {
                out.degree = std::max(da.degree, db.degree);
                if (out.degree > 1) {
                    if (reason) *reason = "non-affine add/sub term";
                    return false;
                }
                out.constant = addOpOrZero(da.constant, db.constant);
                out.linear = addOpOrZero(da.linear, db.linear);
                return true;
            }

            if (node->type() == FormExprType::Divide) {
                if (db.degree > 0) {
                    if (reason) *reason = "division by Trial-dependent expression is non-affine";
                    return false;
                }
                out.degree = da.degree;
                if (out.degree > 1) {
                    if (reason) *reason = "non-affine division numerator";
                    return false;
                }
                out.constant = da.constant / db.constant;
                out.linear = da.linear / db.constant;
                return true;
            }

            // Multiply-like ops: (c1 + l1) ⊗ (c2 + l2) is affine iff not (l1 and l2 both nonzero).
            if (da.degree == 1 && db.degree == 1) {
                if (reason) *reason = "product of Trial-dependent expressions is non-affine";
                return false;
            }
            if (da.degree > 1 || db.degree > 1) {
                if (reason) *reason = "non-affine multiply-like term";
                return false;
            }

            out.degree = da.degree + db.degree;
            out.constant = multLikeOrZero(da.constant, db.constant);
            if (da.degree == 1) {
                out.linear = multLikeOrZero(da.linear, db.constant);
            } else if (db.degree == 1) {
                out.linear = multLikeOrZero(da.constant, db.linear);
            } else {
                out.linear = zeroExpr();
            }
            return true;
        }

        // ---- Nonlinear or unsupported constructs ----
        case FormExprType::Power:
        case FormExprType::Minimum:
        case FormExprType::Maximum:
        case FormExprType::Less:
        case FormExprType::LessEqual:
        case FormExprType::Greater:
        case FormExprType::GreaterEqual:
        case FormExprType::Equal:
        case FormExprType::NotEqual:
        case FormExprType::Conditional:
        case FormExprType::Determinant:
        case FormExprType::Inverse:
        case FormExprType::Cofactor:
        case FormExprType::Norm:
        case FormExprType::Normalize:
        case FormExprType::AbsoluteValue:
        case FormExprType::Sign:
        case FormExprType::Sqrt:
        case FormExprType::Exp:
        case FormExprType::Log:
        case FormExprType::Constitutive:
        case FormExprType::ConstitutiveOutput:
        case FormExprType::CellIntegral:
        case FormExprType::BoundaryIntegral:
        case FormExprType::InteriorFaceIntegral:
        case FormExprType::InterfaceIntegral:
        case FormExprType::AsVector:
        case FormExprType::AsTensor:
        default:
            if (reason) *reason = "unsupported/nonlinear construct for affine residual splitting: " + node->toString();
            return false;
    }
}

} // namespace

std::optional<AffineResidualSplit> trySplitAffineResidual(
    const FormExpr& residual_form,
    const AffineResidualOptions& options,
    std::string* reason_out)
{
    if (!residual_form.isValid()) {
        if (reason_out) *reason_out = "invalid residual form";
        return std::nullopt;
    }
    if (!residual_form.hasTest() || !residual_form.hasTrial()) {
        if (reason_out) *reason_out = "residual form must contain both test and trial functions";
        return std::nullopt;
    }

    FormCompiler compiler;
    const auto residual_ir = compiler.compileResidual(residual_form);

    if (!options.allow_time_derivatives && residual_ir.maxTimeDerivativeOrder() > 0) {
        if (reason_out) *reason_out = "residual contains dt(...) terms (not yet supported for affine optimization)";
        return std::nullopt;
    }
    if (!options.allow_interior_face_terms && residual_ir.hasInteriorFaceTerms()) {
        if (reason_out) *reason_out = "residual contains interior-face terms dS (not yet supported for affine optimization)";
        return std::nullopt;
    }

    FormExpr bilinear_sum{};
    FormExpr linear_sum{};
    bool have_bilinear = false;
    bool have_linear = false;

    for (const auto& term : residual_ir.terms()) {
        if (!options.allow_time_derivatives && term.time_derivative_order > 0) {
            if (reason_out) *reason_out = "dt(...) term encountered (not yet supported for affine optimization)";
            return std::nullopt;
        }

        Decomp d;
        std::string local_reason;
        std::string* rptr = reason_out ? &local_reason : nullptr;
        if (!decomposeAffineInTrial(term.integrand, d, rptr)) {
            if (reason_out) {
                *reason_out = local_reason.empty() ? "residual not affine in TrialFunction" : local_reason;
            }
            return std::nullopt;
        }

        if (d.degree > 1) {
            if (reason_out) *reason_out = "residual not affine in TrialFunction";
            return std::nullopt;
        }

        // Bilinear portion: must contain both test+trial functions.
        if (d.linear.isValid() && d.linear.hasTest() && d.linear.hasTrial()) {
            const auto wrapped = wrapIntegral(d.linear, term.domain, term.boundary_marker);
            bilinear_sum = have_bilinear ? (bilinear_sum + wrapped) : wrapped;
            have_bilinear = true;
        }

        // Linear portion: test-only.
        if (d.constant.isValid() && d.constant.hasTest() && !d.constant.hasTrial()) {
            const auto wrapped = wrapIntegral(d.constant, term.domain, term.boundary_marker);
            linear_sum = have_linear ? (linear_sum + wrapped) : wrapped;
            have_linear = true;
        }
    }

    if (!have_bilinear || !bilinear_sum.hasTest() || !bilinear_sum.hasTrial()) {
        if (reason_out) *reason_out = "no bilinear (Trial-dependent) portion detected";
        return std::nullopt;
    }

    AffineResidualSplit split;
    split.bilinear = bilinear_sum;
    split.linear = linear_sum; // may be invalid if have_linear==false
    return split;
}

} // namespace forms
} // namespace FE
} // namespace svmp
