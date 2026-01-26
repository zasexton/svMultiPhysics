/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/SymbolicDifferentiation.h"

#include "Core/Types.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

namespace {

[[nodiscard]] SymbolicDiffIssue issue(const FormExprNode& node, std::string message)
{
    SymbolicDiffIssue out;
    out.type = node.type();
    out.message = std::move(message);
    out.subexpr = node.toString();
    return out;
}

[[nodiscard]] bool isScalarConstantValue(const FormExprNode& node, Real value)
{
    if (node.type() != FormExprType::Constant) return false;
    const auto v = node.constantValue();
    if (!v.has_value()) return false;
    return *v == value;
}

[[nodiscard]] bool isScalarZero(const FormExprNode& node) { return isScalarConstantValue(node, Real(0.0)); }
[[nodiscard]] bool isScalarOne(const FormExprNode& node) { return isScalarConstantValue(node, Real(1.0)); }

[[nodiscard]] bool isZeroLike(const FormExprNode& node)
{
    if (isScalarZero(node)) return true;
    if (node.type() == FormExprType::Multiply) {
        const auto kids = node.childrenShared();
        if (kids.size() != 2u || !kids[0] || !kids[1]) return false;
        return (isScalarZero(*kids[0]) || isScalarZero(*kids[1]));
    }
    if (node.type() == FormExprType::Negate) {
        const auto kids = node.childrenShared();
        if (kids.size() != 1u || !kids[0]) return false;
        return isZeroLike(*kids[0]);
    }
    return false;
}

[[nodiscard]] bool isConstantScalar(const FormExprNode& node, Real& out_value)
{
    if (node.type() != FormExprType::Constant) return false;
    const auto v = node.constantValue();
    if (!v) return false;
    out_value = *v;
    return true;
}

[[nodiscard]] bool isDefinitelyScalarNode(const FormExprNode& node)
{
    switch (node.type()) {
        case FormExprType::Constant:
        case FormExprType::ParameterSymbol:
        case FormExprType::ParameterRef:
        case FormExprType::BoundaryIntegralSymbol:
        case FormExprType::BoundaryIntegralRef:
        case FormExprType::AuxiliaryStateSymbol:
        case FormExprType::AuxiliaryStateRef:
        case FormExprType::MaterialStateOldRef:
        case FormExprType::MaterialStateWorkRef:
        case FormExprType::PreviousSolutionRef:
        case FormExprType::Time:
        case FormExprType::TimeStep:
        case FormExprType::EffectiveTimeStep:
        case FormExprType::JacobianDeterminant:
        case FormExprType::CellDiameter:
        case FormExprType::CellVolume:
        case FormExprType::FacetArea:
        case FormExprType::CellDomainId:
        case FormExprType::Less:
        case FormExprType::LessEqual:
        case FormExprType::Greater:
        case FormExprType::GreaterEqual:
        case FormExprType::Equal:
        case FormExprType::NotEqual:
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::Trace:
        case FormExprType::Determinant:
        case FormExprType::Norm:
        case FormExprType::Component:
            return true;

        case FormExprType::TestFunction:
        case FormExprType::TrialFunction:
        case FormExprType::DiscreteField:
        case FormExprType::StateField: {
            const auto* sig = node.spaceSignature();
            return sig && sig->field_type == FieldType::Scalar;
        }

        case FormExprType::Coefficient:
            return node.scalarCoefficient() != nullptr || node.timeScalarCoefficient() != nullptr;

        case FormExprType::Negate:
        case FormExprType::Sqrt:
        case FormExprType::Exp:
        case FormExprType::Log:
        case FormExprType::AbsoluteValue:
        case FormExprType::Sign:
        case FormExprType::TimeDerivative: {
            const auto kids = node.childrenShared();
            if (kids.size() != 1u || !kids[0]) return false;
            return isDefinitelyScalarNode(*kids[0]);
        }

        case FormExprType::Minimum:
        case FormExprType::Maximum:
        case FormExprType::Power:
        case FormExprType::Add:
        case FormExprType::Subtract:
        case FormExprType::Multiply:
        case FormExprType::Divide: {
            const auto kids = node.childrenShared();
            if (kids.size() != 2u || !kids[0] || !kids[1]) return false;
            return isDefinitelyScalarNode(*kids[0]) && isDefinitelyScalarNode(*kids[1]);
        }

        case FormExprType::Conditional: {
            const auto kids = node.childrenShared();
            if (kids.size() != 3u || !kids[1] || !kids[2]) return false;
            return isDefinitelyScalarNode(*kids[1]) && isDefinitelyScalarNode(*kids[2]);
        }

        default:
            return false;
    }
}

[[nodiscard]] std::string makeDeltaName(std::string_view base)
{
    if (base.empty()) return "du";
    if (base == "u") return "du";
    if (base == "v") return "dv";
    return "d" + std::string(base);
}

struct DiffPair {
    FormExpr primal{};
    FormExpr deriv{};
};

enum class DiffWrtKind : std::uint8_t {
    ActiveTrialFunction,
    SpecificTrialFunction,
    FieldId
};

struct DiffTargetConfig {
    DiffWrtKind kind{DiffWrtKind::ActiveTrialFunction};

    // How to rewrite residual TrialFunction occurrences in the primal expression.
    FieldId trial_state_field{INVALID_FIELD_ID};

    // SpecificTrialFunction target.
    std::optional<FormExprNode::SpaceSignature> trial_sig{};
    std::optional<std::string> trial_name{};

    // FieldId target.
    std::optional<FieldId> field_id{};
    std::optional<FormExprNode::SpaceSignature> field_sig{};
    std::optional<std::string> field_name{};
};

[[nodiscard]] bool spaceSignatureEqual(const FormExprNode::SpaceSignature& a,
                                       const FormExprNode::SpaceSignature& b) noexcept
{
    return a.space_type == b.space_type &&
           a.field_type == b.field_type &&
           a.continuity == b.continuity &&
           a.value_dimension == b.value_dimension &&
           a.topological_dimension == b.topological_dimension &&
           a.polynomial_order == b.polynomial_order &&
           a.element_type == b.element_type;
}

} // namespace

SymbolicDiffResult checkSymbolicDifferentiability(const FormExpr& expr)
{
    SymbolicDiffResult out;
    out.ok = true;

    if (!expr.isValid() || expr.node() == nullptr) {
        out.ok = false;
        out.first_issue = SymbolicDiffIssue{
            .type = FormExprType::Constant,
            .message = "SymbolicDiff: invalid expression",
            .subexpr = {},
        };
        return out;
    }

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (!out.ok) return;

        switch (n.type()) {
            case FormExprType::Constitutive:
            case FormExprType::ConstitutiveOutput:
                out.ok = false;
                out.first_issue = issue(n, "SymbolicDiff: Constitutive nodes must be inlined before differentiation");
                return;
            default:
                break;
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, *expr.node());
    return out;
}

FormExpr simplify(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        return {};
    }

    FormExpr current = expr;

    // Fixed-point simplification using pre-order transforms (children are simplified
    // within each pass; parent-level rewrites become visible on subsequent passes).
    constexpr int kMaxPasses = 8;
    for (int pass = 0; pass < kMaxPasses; ++pass) {
        bool changed = false;

        current = current.transformNodes([&](const FormExprNode& n) -> std::optional<FormExpr> {
            const auto kids = n.childrenShared();

            auto asExpr = [&](const std::shared_ptr<FormExprNode>& p) -> FormExpr {
                return p ? FormExpr(p) : FormExpr{};
            };

            // ---- Unary ----
            if (n.type() == FormExprType::Negate) {
                if (kids.size() != 1u || !kids[0]) return std::nullopt;
                if (kids[0]->type() == FormExprType::Negate) {
                    const auto gkids = kids[0]->childrenShared();
                    if (gkids.size() == 1u && gkids[0]) {
                        changed = true;
                        return FormExpr(gkids[0]);
                    }
                }

                Real v = 0.0;
                if (isConstantScalar(*kids[0], v)) {
                    changed = true;
                    return FormExpr::constant(-v);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Sqrt || n.type() == FormExprType::Exp || n.type() == FormExprType::Log) {
                if (kids.size() != 1u || !kids[0]) return std::nullopt;
                Real v = 0.0;
                if (isConstantScalar(*kids[0], v)) {
                    const Real out = (n.type() == FormExprType::Sqrt) ? std::sqrt(v)
                                   : (n.type() == FormExprType::Exp)  ? std::exp(v)
                                                                     : std::log(v);
                    changed = true;
                    return FormExpr::constant(out);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::AbsoluteValue || n.type() == FormExprType::Sign) {
                if (kids.size() != 1u || !kids[0]) return std::nullopt;
                Real v = 0.0;
                if (isConstantScalar(*kids[0], v)) {
                    const Real out = (n.type() == FormExprType::AbsoluteValue)
                        ? std::abs(v)
                        : (v > 0.0) ? 1.0 : ((v < 0.0) ? -1.0 : 0.0);
                    changed = true;
                    return FormExpr::constant(out);
                }
                return std::nullopt;
            }

            // ---- Binary ----
            if (n.type() == FormExprType::Add) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                if (isZeroLike(*kids[0])) {
                    changed = true;
                    return asExpr(kids[1]);
                }
                if (isZeroLike(*kids[1])) {
                    changed = true;
                    return asExpr(kids[0]);
                }

                if (kids[1]->type() == FormExprType::Negate) {
                    const auto nkids = kids[1]->childrenShared();
                    if (nkids.size() == 1u && nkids[0] && nkids[0].get() == kids[0].get()) {
                        changed = true;
                        return isDefinitelyScalarNode(*kids[0]) ? FormExpr::constant(0.0)
                                                                : (FormExpr::constant(0.0) * asExpr(kids[0]));
                    }
                }
                if (kids[0]->type() == FormExprType::Negate) {
                    const auto nkids = kids[0]->childrenShared();
                    if (nkids.size() == 1u && nkids[0] && nkids[0].get() == kids[1].get()) {
                        changed = true;
                        return isDefinitelyScalarNode(*kids[1]) ? FormExpr::constant(0.0)
                                                                : (FormExpr::constant(0.0) * asExpr(kids[1]));
                    }
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(a + b);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Subtract) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                if (isZeroLike(*kids[1])) {
                    changed = true;
                    return asExpr(kids[0]);
                }
                if (isZeroLike(*kids[0])) {
                    changed = true;
                    return -asExpr(kids[1]);
                }

                if (kids[0].get() == kids[1].get()) {
                    changed = true;
                    return isDefinitelyScalarNode(*kids[0]) ? FormExpr::constant(0.0)
                                                            : (FormExpr::constant(0.0) * asExpr(kids[0]));
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(a - b);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Multiply) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                if (isScalarZero(*kids[0])) {
                    changed = true;
                    return isDefinitelyScalarNode(*kids[1]) ? FormExpr::constant(0.0)
                                                            : (FormExpr::constant(0.0) * asExpr(kids[1]));
                }
                if (isScalarZero(*kids[1])) {
                    changed = true;
                    return isDefinitelyScalarNode(*kids[0]) ? FormExpr::constant(0.0)
                                                            : (FormExpr::constant(0.0) * asExpr(kids[0]));
                }

                if (isScalarOne(*kids[0])) {
                    changed = true;
                    return asExpr(kids[1]);
                }
                if (isScalarOne(*kids[1])) {
                    changed = true;
                    return asExpr(kids[0]);
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(a * b);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Divide) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                if (isZeroLike(*kids[0])) {
                    changed = true;
                    return asExpr(kids[0]);
                }
                if (isScalarOne(*kids[1])) {
                    changed = true;
                    return asExpr(kids[0]);
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(a / b);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Power) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;

                Real exp = 0.0;
                if (isConstantScalar(*kids[1], exp)) {
                    if (exp == 0.0) {
                        changed = true;
                        return FormExpr::constant(1.0);
                    }
                    if (exp == 1.0) {
                        changed = true;
                        return asExpr(kids[0]);
                    }
                    if (exp == 2.0) {
                        const auto base = asExpr(kids[0]);
                        changed = true;
                        return base * base;
                    }
                }

                Real a = 0.0;
                Real b = 0.0;
                if (isConstantScalar(*kids[0], a) && isConstantScalar(*kids[1], b)) {
                    changed = true;
                    return FormExpr::constant(std::pow(a, b));
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Conditional) {
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) return std::nullopt;
                Real c = 0.0;
                if (isConstantScalar(*kids[0], c)) {
                    changed = true;
                    return (c != 0.0) ? asExpr(kids[1]) : asExpr(kids[2]);
                }
                return std::nullopt;
            }

            if (n.type() == FormExprType::Less || n.type() == FormExprType::LessEqual ||
                n.type() == FormExprType::Greater || n.type() == FormExprType::GreaterEqual ||
                n.type() == FormExprType::Equal || n.type() == FormExprType::NotEqual ||
                n.type() == FormExprType::Minimum || n.type() == FormExprType::Maximum) {
                if (kids.size() != 2u || !kids[0] || !kids[1]) return std::nullopt;
                Real a = 0.0;
                Real b = 0.0;
                if (!isConstantScalar(*kids[0], a) || !isConstantScalar(*kids[1], b)) {
                    return std::nullopt;
                }

                if (n.type() == FormExprType::Minimum) {
                    changed = true;
                    return FormExpr::constant(std::min(a, b));
                }
                if (n.type() == FormExprType::Maximum) {
                    changed = true;
                    return FormExpr::constant(std::max(a, b));
                }

                bool truth = false;
                switch (n.type()) {
                    case FormExprType::Less: truth = (a < b); break;
                    case FormExprType::LessEqual: truth = (a <= b); break;
                    case FormExprType::Greater: truth = (a > b); break;
                    case FormExprType::GreaterEqual: truth = (a >= b); break;
                    case FormExprType::Equal: truth = (a == b); break;
                    case FormExprType::NotEqual: truth = (a != b); break;
                    default: break;
                }
                changed = true;
                return FormExpr::constant(truth ? 1.0 : 0.0);
            }

            return std::nullopt;
        });

        if (!changed) {
            break;
        }
    }

    return current;
}

namespace {

FormExpr differentiateResidualImpl(const FormExpr& residual_form,
                                  const DiffTargetConfig& cfg)
{
    if (!residual_form.isValid() || residual_form.node() == nullptr) {
        throw std::invalid_argument("differentiateResidual: invalid residual expression");
    }

    const auto check = checkSymbolicDifferentiability(residual_form);
    if (!check.ok) {
        const auto msg = check.first_issue ? check.first_issue->message : std::string("SymbolicDiff: unsupported expression");
        throw std::invalid_argument("differentiateResidual: " + msg);
    }

    std::unordered_map<const FormExprNode*, DiffPair> memo;

    std::optional<FormExprNode::SpaceSignature> wrt_field_sig = cfg.field_sig;
    std::optional<std::string> wrt_field_name = cfg.field_name;
    std::optional<std::string> wrt_field_delta_name =
        wrt_field_name ? std::optional<std::string>(makeDeltaName(*wrt_field_name)) : std::nullopt;

    const auto diff = [&](const auto& self, const std::shared_ptr<FormExprNode>& node) -> DiffPair {
        if (!node) {
            throw std::invalid_argument("differentiateResidual: encountered null node");
        }
        if (auto it = memo.find(node.get()); it != memo.end()) {
            return it->second;
        }

        DiffPair out;

        const auto kids = node->childrenShared();

        auto diff1 = [&](std::size_t k) -> DiffPair {
            if (kids.size() <= k || !kids[k]) {
                throw std::invalid_argument("differentiateResidual: missing child");
            }
            return self(self, kids[k]);
        };

        auto zeroOf = [&](const FormExpr& ref) -> FormExpr {
            return FormExpr::constant(0.0) * ref;
        };

        const auto isWrtTrial = [&](const FormExprNode::SpaceSignature& sig, const std::string& name) -> bool {
            switch (cfg.kind) {
                case DiffWrtKind::ActiveTrialFunction:
                    return true;
                case DiffWrtKind::SpecificTrialFunction:
                    return cfg.trial_sig.has_value() && cfg.trial_name.has_value() &&
                           spaceSignatureEqual(sig, *cfg.trial_sig) && name == *cfg.trial_name;
                case DiffWrtKind::FieldId:
                    return cfg.field_id.has_value() && (cfg.trial_state_field == *cfg.field_id);
                default:
                    return false;
            }
        };

        const auto isWrtField = [&](const FormExprNode::SpaceSignature& sig, FieldId fid, const std::string& name) -> bool {
            if (cfg.kind != DiffWrtKind::FieldId || !cfg.field_id.has_value() || fid != *cfg.field_id) {
                return false;
            }

            if (wrt_field_sig.has_value()) {
                if (!spaceSignatureEqual(sig, *wrt_field_sig)) {
                    throw std::invalid_argument("differentiateResidual: target FieldId appears with inconsistent SpaceSignature");
                }
            } else {
                wrt_field_sig = sig;
            }

            if (wrt_field_name.has_value()) {
                if (name != *wrt_field_name) {
                    throw std::invalid_argument("differentiateResidual: target FieldId appears with inconsistent symbol name");
                }
            } else {
                wrt_field_name = name;
                wrt_field_delta_name = makeDeltaName(name);
            }

            if (!wrt_field_delta_name.has_value()) {
                throw std::logic_error("differentiateResidual: missing delta name for FieldId target");
            }
            return true;
        };

        switch (node->type()) {
            // ---- Terminals (treated as constants) ----
            case FormExprType::Constant:
            case FormExprType::Coefficient:
            case FormExprType::ParameterSymbol:
            case FormExprType::ParameterRef:
            case FormExprType::BoundaryIntegralSymbol:
            case FormExprType::BoundaryIntegralRef:
            case FormExprType::AuxiliaryStateSymbol:
            case FormExprType::AuxiliaryStateRef:
            case FormExprType::MaterialStateOldRef:
            case FormExprType::MaterialStateWorkRef:
            case FormExprType::PreviousSolutionRef:
            case FormExprType::Coordinate:
            case FormExprType::ReferenceCoordinate:
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
                out.primal = FormExpr(node);
                out.deriv = zeroOf(out.primal);
                break;

            case FormExprType::BoundaryFunctionalSymbol: {
                if (kids.size() != 1u || !kids[0]) {
                    throw std::invalid_argument("differentiateResidual: BoundaryFunctionalSymbol missing integrand");
                }
                const int marker = node->boundaryMarker().value_or(-1);
                const auto name = node->symbolName();
                if (marker < 0 || !name || name->empty()) {
                    throw std::invalid_argument("differentiateResidual: BoundaryFunctionalSymbol missing metadata");
                }
                const auto a = diff1(0);
                out.primal = FormExpr::boundaryIntegral(a.primal, marker, std::string(*name));
                out.deriv = zeroOf(out.primal);
                break;
            }

            case FormExprType::TestFunction:
                out.primal = FormExpr::testFunction(*node->spaceSignature(), node->toString());
                out.deriv = zeroOf(out.primal);
                break;

            case FormExprType::TrialFunction: {
                const auto* sig = node->spaceSignature();
                if (!sig) {
                    throw std::invalid_argument("differentiateResidual: TrialFunction missing SpaceSignature");
                }
                const auto name = node->toString();
                out.primal = FormExpr::stateField(cfg.trial_state_field, *sig, name);
                if (isWrtTrial(*sig, name)) {
                    out.deriv = FormExpr::trialFunction(*sig, makeDeltaName(name));
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;
            }

            case FormExprType::DiscreteField: {
                const auto fid = node->fieldId();
                const auto* sig = node->spaceSignature();
                if (!fid || !sig) {
                    throw std::invalid_argument("differentiateResidual: DiscreteField missing metadata");
                }
                out.primal = FormExpr::discreteField(*fid, *sig, node->toString());
                if (isWrtField(*sig, *fid, node->toString())) {
                    out.deriv = FormExpr::trialFunction(*sig, *wrt_field_delta_name);
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;
            }

            case FormExprType::StateField: {
                const auto fid = node->fieldId();
                const auto* sig = node->spaceSignature();
                if (!fid || !sig) {
                    throw std::invalid_argument("differentiateResidual: StateField missing metadata");
                }
                out.primal = FormExpr::stateField(*fid, *sig, node->toString());
                if (isWrtField(*sig, *fid, node->toString())) {
                    out.deriv = FormExpr::trialFunction(*sig, *wrt_field_delta_name);
                } else {
                    out.deriv = zeroOf(out.primal);
                }
                break;
            }

            // ---- Unary ops ----
            case FormExprType::Negate: {
                const auto a = diff1(0);
                out.primal = -a.primal;
                out.deriv = -a.deriv;
                break;
            }
            case FormExprType::Gradient: {
                const auto a = diff1(0);
                out.primal = a.primal.grad();
                out.deriv = a.deriv.grad();
                break;
            }
            case FormExprType::Divergence: {
                const auto a = diff1(0);
                out.primal = a.primal.div();
                out.deriv = a.deriv.div();
                break;
            }
            case FormExprType::Curl: {
                const auto a = diff1(0);
                out.primal = a.primal.curl();
                out.deriv = a.deriv.curl();
                break;
            }
            case FormExprType::Hessian: {
                const auto a = diff1(0);
                out.primal = a.primal.hessian();
                out.deriv = a.deriv.hessian();
                break;
            }
            case FormExprType::TimeDerivative: {
                const int order = node->timeDerivativeOrder().value_or(1);
                const auto a = diff1(0);
                out.primal = a.primal.dt(order);
                if (a.deriv.isValid() && a.deriv.node() != nullptr && isZeroLike(*a.deriv.node())) {
                    out.deriv = a.deriv;
                } else {
                    out.deriv = a.deriv.dt(order);
                }
                break;
            }
            case FormExprType::RestrictMinus: {
                const auto a = diff1(0);
                out.primal = a.primal.minus();
                out.deriv = a.deriv.minus();
                break;
            }
            case FormExprType::RestrictPlus: {
                const auto a = diff1(0);
                out.primal = a.primal.plus();
                out.deriv = a.deriv.plus();
                break;
            }
            case FormExprType::Jump: {
                const auto a = diff1(0);
                out.primal = a.primal.jump();
                out.deriv = a.deriv.jump();
                break;
            }
            case FormExprType::Average: {
                const auto a = diff1(0);
                out.primal = a.primal.avg();
                out.deriv = a.deriv.avg();
                break;
            }

            // ---- Constructors / indexing ----
            case FormExprType::AsVector: {
                std::vector<FormExpr> pv;
                std::vector<FormExpr> dv;
                pv.reserve(kids.size());
                dv.reserve(kids.size());
                for (const auto& k : kids) {
                    const auto r = self(self, k);
                    pv.push_back(r.primal);
                    dv.push_back(r.deriv);
                }
                out.primal = FormExpr::asVector(std::move(pv));
                out.deriv = FormExpr::asVector(std::move(dv));
                break;
            }
            case FormExprType::AsTensor: {
                const int rows = node->tensorRows().value_or(0);
                const int cols = node->tensorCols().value_or(0);
                if (rows <= 0 || cols <= 0) {
                    throw std::invalid_argument("differentiateResidual: AsTensor missing (rows,cols)");
                }
                if (kids.size() != static_cast<std::size_t>(rows * cols)) {
                    throw std::invalid_argument("differentiateResidual: AsTensor children size mismatch");
                }

                std::vector<std::vector<FormExpr>> prow(static_cast<std::size_t>(rows));
                std::vector<std::vector<FormExpr>> drow(static_cast<std::size_t>(rows));
                for (int r = 0; r < rows; ++r) {
                    prow[static_cast<std::size_t>(r)].reserve(static_cast<std::size_t>(cols));
                    drow[static_cast<std::size_t>(r)].reserve(static_cast<std::size_t>(cols));
                    for (int c = 0; c < cols; ++c) {
                        const auto idx = static_cast<std::size_t>(r * cols + c);
                        const auto rr = self(self, kids[idx]);
                        prow[static_cast<std::size_t>(r)].push_back(rr.primal);
                        drow[static_cast<std::size_t>(r)].push_back(rr.deriv);
                    }
                }
                out.primal = FormExpr::asTensor(std::move(prow));
                out.deriv = FormExpr::asTensor(std::move(drow));
                break;
            }
            case FormExprType::Component: {
                const int i = node->componentIndex0().value_or(0);
                const int j = node->componentIndex1().value_or(-1);
                const auto a = diff1(0);
                out.primal = a.primal.component(i, j);
                out.deriv = a.deriv.component(i, j);
                break;
            }
            case FormExprType::IndexedAccess: {
                const int rank = node->indexRank().value_or(0);
                const auto ids_opt = node->indexIds();
                const auto ext_opt = node->indexExtents();
                if (rank <= 0 || !ids_opt || !ext_opt) {
                    throw std::invalid_argument("differentiateResidual: IndexedAccess missing index metadata");
                }
                const auto a = diff1(0);
                out.primal = FormExpr::indexedAccessRaw(a.primal, rank, *ids_opt, *ext_opt);
                out.deriv = FormExpr::indexedAccessRaw(a.deriv, rank, *ids_opt, *ext_opt);
                break;
            }

            // ---- Algebra ----
            case FormExprType::Add: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal + b.primal;
                out.deriv = a.deriv + b.deriv;
                break;
            }
            case FormExprType::Subtract: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal - b.primal;
                out.deriv = a.deriv - b.deriv;
                break;
            }
            case FormExprType::Multiply: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal * b.primal;
                out.deriv = (a.deriv * b.primal) + (a.primal * b.deriv);
                break;
            }
            case FormExprType::Divide: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal / b.primal;
                out.deriv = ((a.deriv * b.primal) - (a.primal * b.deriv)) / (b.primal * b.primal);
                break;
            }
            case FormExprType::InnerProduct: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = inner(a.primal, b.primal);
                out.deriv = inner(a.deriv, b.primal) + inner(a.primal, b.deriv);
                break;
            }
            case FormExprType::DoubleContraction: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = a.primal.doubleContraction(b.primal);
                out.deriv = a.deriv.doubleContraction(b.primal) + a.primal.doubleContraction(b.deriv);
                break;
            }
            case FormExprType::OuterProduct: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = outer(a.primal, b.primal);
                out.deriv = outer(a.deriv, b.primal) + outer(a.primal, b.deriv);
                break;
            }
            case FormExprType::CrossProduct: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = cross(a.primal, b.primal);
                out.deriv = cross(a.deriv, b.primal) + cross(a.primal, b.deriv);
                break;
            }
            case FormExprType::Power: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = pow(a.primal, b.primal);

                Real exp = 0.0;
                if (b.primal.isValid() && b.primal.node() && isConstantScalar(*b.primal.node(), exp)) {
                    if (exp == 0.0) {
                        out.deriv = zeroOf(out.primal);
                    } else {
                        out.deriv = FormExpr::constant(exp) *
                                    pow(a.primal, FormExpr::constant(exp - 1.0)) *
                                    a.deriv;
                    }
                } else {
                    // Match Dual::pow behavior: when pow(a,b) evaluates to 0, AD returns a zero derivative
                    // without evaluating log(a) or 1/a (which would produce NaNs).
                    const auto is_zero = eq(out.primal, FormExpr::constant(0.0)); // 0/1
                    const auto safe_a = a.primal + is_zero; // shift to avoid log(0) and 1/0 in the masked case
                    out.deriv = out.primal * (b.deriv * log(safe_a) + b.primal * (a.deriv / safe_a));
                }
                break;
            }
            case FormExprType::Minimum: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = min(a.primal, b.primal);
                const auto cond = le(a.primal, b.primal);
                out.deriv = cond.conditional(a.deriv, b.deriv);
                break;
            }
            case FormExprType::Maximum: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                out.primal = max(a.primal, b.primal);
                const auto cond = ge(a.primal, b.primal);
                out.deriv = cond.conditional(a.deriv, b.deriv);
                break;
            }

            // ---- Predicates ----
            case FormExprType::Less:
            case FormExprType::LessEqual:
            case FormExprType::Greater:
            case FormExprType::GreaterEqual:
            case FormExprType::Equal:
            case FormExprType::NotEqual: {
                const auto a = diff1(0);
                const auto b = diff1(1);
                switch (node->type()) {
                    case FormExprType::Less: out.primal = lt(a.primal, b.primal); break;
                    case FormExprType::LessEqual: out.primal = le(a.primal, b.primal); break;
                    case FormExprType::Greater: out.primal = gt(a.primal, b.primal); break;
                    case FormExprType::GreaterEqual: out.primal = ge(a.primal, b.primal); break;
                    case FormExprType::Equal: out.primal = eq(a.primal, b.primal); break;
                    case FormExprType::NotEqual: out.primal = ne(a.primal, b.primal); break;
                    default: break;
                }
                out.deriv = zeroOf(out.primal);
                break;
            }
            case FormExprType::Conditional: {
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                    throw std::invalid_argument("differentiateResidual: Conditional expects 3 children");
                }
                const auto c = self(self, kids[0]);
                const auto t = self(self, kids[1]);
                const auto f = self(self, kids[2]);
                out.primal = c.primal.conditional(t.primal, f.primal);
                out.deriv = c.primal.conditional(t.deriv, f.deriv);
                break;
            }

            // ---- Tensor ops / scalar functions ----
            case FormExprType::Transpose: {
                const auto a = diff1(0);
                out.primal = a.primal.transpose();
                out.deriv = a.deriv.transpose();
                break;
            }
            case FormExprType::Trace: {
                const auto a = diff1(0);
                out.primal = a.primal.trace();
                out.deriv = a.deriv.trace();
                break;
            }
            case FormExprType::Determinant: {
                const auto a = diff1(0);
                out.primal = det(a.primal);
                // Use cofactor-based rule to avoid singular-matrix inversion at det(A)=0:
                //   d(det(A)) = cofactor(A) : dA
                out.deriv = a.primal.cofactor().doubleContraction(a.deriv);
                break;
            }
            case FormExprType::Inverse: {
                const auto a = diff1(0);
                out.primal = inv(a.primal);
                out.deriv = -(out.primal * a.deriv * out.primal);
                break;
            }
            case FormExprType::Cofactor: {
                const auto a = diff1(0);
                const auto detA = det(a.primal);
                const auto invA = inv(a.primal);
                const auto invAT = transpose(invA);
                out.primal = a.primal.cofactor();
                const auto ddet = detA * inner(inv(transpose(a.primal)), a.deriv);
                const auto dinv = -(invA * a.deriv * invA);
                out.deriv = ddet * invAT + detA * transpose(dinv);
                break;
            }
            case FormExprType::Deviator: {
                const auto a = diff1(0);
                out.primal = a.primal.dev();
                out.deriv = a.deriv.dev();
                break;
            }
            case FormExprType::SymmetricPart: {
                const auto a = diff1(0);
                out.primal = a.primal.sym();
                out.deriv = a.deriv.sym();
                break;
            }
            case FormExprType::SkewPart: {
                const auto a = diff1(0);
                out.primal = a.primal.skew();
                out.deriv = a.deriv.skew();
                break;
            }
            case FormExprType::Norm: {
                const auto a = diff1(0);
                out.primal = a.primal.norm();
                const auto nrm = out.primal;
                const auto d = inner(a.primal, a.deriv) / nrm;
                out.deriv = eq(nrm, FormExpr::constant(0.0)).conditional(FormExpr::constant(0.0), d);
                break;
            }
            case FormExprType::Normalize: {
                const auto a = diff1(0);
                out.primal = a.primal.normalize();
                const auto nrm = a.primal.norm();
                const auto dnrm = inner(a.primal, a.deriv) / nrm;
                const auto d = (a.deriv * nrm - a.primal * dnrm) / (nrm * nrm);
                out.deriv = eq(nrm, FormExpr::constant(0.0)).conditional(zeroOf(out.primal), d);
                break;
            }
            case FormExprType::AbsoluteValue: {
                const auto a = diff1(0);
                out.primal = a.primal.abs();
                // Match Dual::abs behavior: pick +da when a >= 0, otherwise -da (at a==0, Dual picks +).
                const auto is_nonneg = ge(a.primal, FormExpr::constant(0.0)); // 0/1
                const auto sign = (FormExpr::constant(2.0) * is_nonneg) - FormExpr::constant(1.0); // +1 or -1
                out.deriv = sign * a.deriv;
                break;
            }
            case FormExprType::Sign: {
                const auto a = diff1(0);
                out.primal = a.primal.sign();
                out.deriv = zeroOf(out.primal);
                break;
            }
            case FormExprType::Sqrt: {
                const auto a = diff1(0);
                out.primal = a.primal.sqrt();
                // Match Dual::sqrt behavior: avoid division by zero by using denom=1 when sqrt(..)==0.
                const auto is_zero = eq(out.primal, FormExpr::constant(0.0)); // 0/1
                const auto denom = (FormExpr::constant(2.0) * out.primal) + is_zero;
                out.deriv = a.deriv / denom;
                break;
            }
            case FormExprType::Exp: {
                const auto a = diff1(0);
                out.primal = a.primal.exp();
                out.deriv = out.primal * a.deriv;
                break;
            }
            case FormExprType::Log: {
                const auto a = diff1(0);
                out.primal = a.primal.log();
                out.deriv = a.deriv / a.primal;
                break;
            }

            // ---- Constitutive hooks (must be inlined) ----
            case FormExprType::Constitutive:
            case FormExprType::ConstitutiveOutput:
                throw std::invalid_argument("differentiateResidual: Constitutive nodes must be inlined before differentiation");

            // ---- Measure wrappers ----
            case FormExprType::CellIntegral: {
                const auto a = diff1(0);
                out.primal = a.primal.dx();
                out.deriv = a.deriv.dx();
                break;
            }
            case FormExprType::BoundaryIntegral: {
                const int marker = node->boundaryMarker().value_or(-1);
                const auto a = diff1(0);
                out.primal = a.primal.ds(marker);
                out.deriv = a.deriv.ds(marker);
                break;
            }
            case FormExprType::InteriorFaceIntegral: {
                const auto a = diff1(0);
                out.primal = a.primal.dS();
                out.deriv = a.deriv.dS();
                break;
            }
            case FormExprType::InterfaceIntegral: {
                const int marker = node->interfaceMarker().value_or(-1);
                const auto a = diff1(0);
                out.primal = a.primal.dI(marker);
                out.deriv = a.deriv.dI(marker);
                break;
            }

            default:
                throw std::invalid_argument("differentiateResidual: unsupported FormExprType " +
                                            std::to_string(static_cast<std::uint16_t>(node->type())));
        }

        memo.emplace(node.get(), out);
        return out;
    };

    const auto pair = diff(diff, residual_form.nodeShared());
    return simplify(pair.deriv);
}

} // namespace

FormExpr differentiateResidual(const FormExpr& residual_form)
{
    DiffTargetConfig cfg;
    cfg.kind = DiffWrtKind::ActiveTrialFunction;
    cfg.trial_state_field = INVALID_FIELD_ID;
    return differentiateResidualImpl(residual_form, cfg);
}

FormExpr differentiateResidual(const FormExpr& residual_form,
                               const FormExpr& wrt_terminal,
                               FieldId trial_state_field)
{
    if (!wrt_terminal.isValid() || wrt_terminal.node() == nullptr) {
        throw std::invalid_argument("differentiateResidual: invalid wrt_terminal");
    }

    DiffTargetConfig cfg;
    cfg.trial_state_field = trial_state_field;

    switch (wrt_terminal.node()->type()) {
        case FormExprType::TrialFunction: {
            const auto* sig = wrt_terminal.node()->spaceSignature();
            if (!sig) {
                throw std::invalid_argument("differentiateResidual: wrt TrialFunction missing SpaceSignature");
            }
            cfg.kind = DiffWrtKind::SpecificTrialFunction;
            cfg.trial_sig = *sig;
            cfg.trial_name = wrt_terminal.node()->toString();
            break;
        }
        case FormExprType::DiscreteField:
        case FormExprType::StateField: {
            const auto fid = wrt_terminal.node()->fieldId();
            const auto* sig = wrt_terminal.node()->spaceSignature();
            if (!fid || !sig) {
                throw std::invalid_argument("differentiateResidual: wrt field terminal missing metadata");
            }
            cfg.kind = DiffWrtKind::FieldId;
            cfg.field_id = *fid;
            cfg.field_sig = *sig;
            cfg.field_name = wrt_terminal.node()->toString();
            break;
        }
        default:
            throw std::invalid_argument("differentiateResidual: wrt_terminal must be TrialFunction/StateField/DiscreteField");
    }

    return differentiateResidualImpl(residual_form, cfg);
}

FormExpr differentiateResidual(const FormExpr& residual_form,
                               FieldId field,
                               FieldId trial_state_field)
{
    DiffTargetConfig cfg;
    cfg.kind = DiffWrtKind::FieldId;
    cfg.field_id = field;
    cfg.trial_state_field = trial_state_field;
    return differentiateResidualImpl(residual_form, cfg);
}

} // namespace forms
} // namespace FE
} // namespace svmp
