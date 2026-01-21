/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/FormCompiler.h"

#include "Core/FEException.h"
#include "Forms/BlockForm.h"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace forms {

struct FormCompiler::Impl {
    SymbolicOptions options{};
};

FormCompiler::FormCompiler()
    : impl_(std::make_unique<Impl>())
{
}

FormCompiler::FormCompiler(SymbolicOptions options)
    : impl_(std::make_unique<Impl>())
{
    impl_->options = std::move(options);
}

FormCompiler::~FormCompiler() = default;

FormCompiler::FormCompiler(FormCompiler&&) noexcept = default;
FormCompiler& FormCompiler::operator=(FormCompiler&&) noexcept = default;

void FormCompiler::setOptions(SymbolicOptions options)
{
    impl_->options = std::move(options);
}

const SymbolicOptions& FormCompiler::options() const noexcept
{
    return impl_->options;
}

namespace detail {

FormExpr makeExprFromNode(const std::shared_ptr<FormExprNode>& node)
{
    return FormExpr(node);
}

void requireNoIndexedAccess(const FormExprNode& node)
{
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::IndexedAccess) {
            throw std::invalid_argument(
                "FormCompiler: detected indexed access (Einstein notation). "
                "Call forms::einsum(expr) to lower it before compilation.");
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    visit(visit, node);
}

void requireNoCoupledPlaceholders(const FormExprNode& node)
{
    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::BoundaryFunctionalSymbol ||
            n.type() == FormExprType::BoundaryIntegralSymbol ||
            n.type() == FormExprType::AuxiliaryStateSymbol) {
            throw std::invalid_argument(
                "FormCompiler: detected coupled placeholder terminal. "
                "Resolve coupled expressions via FE/Systems coupled-BC helpers "
                "(e.g., systems::bc::applyCoupledNeumann/applyCoupledRobin) before compilation.");
        }
        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };
    visit(visit, node);
}

bool spaceSignatureEqual(const FormExprNode::SpaceSignature& a,
                         const FormExprNode::SpaceSignature& b) noexcept
{
    return a.space_type == b.space_type && a.field_type == b.field_type && a.continuity == b.continuity &&
           a.value_dimension == b.value_dimension && a.topological_dimension == b.topological_dimension &&
           a.polynomial_order == b.polynomial_order && a.element_type == b.element_type;
}

struct BoundArgumentInfo {
    std::optional<FormExprNode::SpaceSignature> test_space{};
    std::optional<FormExprNode::SpaceSignature> trial_space{};
    std::optional<std::string> test_name{};
    std::optional<std::string> trial_name{};
};

BoundArgumentInfo analyzeBoundArguments(const FormExprNode& node)
{
    BoundArgumentInfo info;

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::TestFunction) {
            const auto* sig = n.spaceSignature();
            if (!sig) {
                throw std::invalid_argument("FormCompiler: TestFunction must be bound to a FunctionSpace");
            }
            if (!info.test_space) {
                info.test_space = *sig;
                info.test_name = n.toString();
            } else {
                if (!spaceSignatureEqual(*info.test_space, *sig)) {
                    throw std::invalid_argument(
                        "FormCompiler: multiple TestFunction spaces detected (mixed/multi-field not implemented)");
                }
                const auto nm = n.toString();
                if (info.test_name && nm != *info.test_name) {
                    throw std::invalid_argument(
                        "FormCompiler: multiple TestFunction symbols detected (mixed/multi-field not implemented)");
                }
            }
        }

        if (n.type() == FormExprType::TrialFunction) {
            const auto* sig = n.spaceSignature();
            if (!sig) {
                throw std::invalid_argument("FormCompiler: TrialFunction must be bound to a FunctionSpace");
            }
            if (!info.trial_space) {
                info.trial_space = *sig;
                info.trial_name = n.toString();
            } else {
                if (!spaceSignatureEqual(*info.trial_space, *sig)) {
                    throw std::invalid_argument(
                        "FormCompiler: multiple TrialFunction spaces detected (mixed/multi-field not implemented)");
                }
                const auto nm = n.toString();
                if (info.trial_name && nm != *info.trial_name) {
                    throw std::invalid_argument(
                        "FormCompiler: multiple TrialFunction symbols detected (mixed/multi-field not implemented)");
                }
            }
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, node);
    return info;
}

assembly::RequiredData analyzeRequiredData(const FormExprNode& node, FormKind kind)
{
    using assembly::RequiredData;

    RequiredData required = RequiredData::None;

    const auto visit = [&](const auto& self, const FormExprNode& n, int order) -> void {
        order = std::clamp(order, 0, 2);

        switch (n.type()) {
            case FormExprType::TestFunction:
                required |= RequiredData::BasisValues;
                if (order >= 1) required |= RequiredData::PhysicalGradients;
                if (order >= 2) required |= RequiredData::BasisHessians;
                break;
            case FormExprType::TrialFunction:
                if (kind == FormKind::Residual) {
                    required |= RequiredData::SolutionValues;
                    required |= RequiredData::BasisValues; // Needed for AD seeding via trialBasisValue().
                    if (order >= 1) {
                        required |= RequiredData::SolutionGradients;
                        required |= RequiredData::PhysicalGradients; // Needed for AD seeding via trialPhysicalGradient().
                    }
                    if (order >= 2) {
                        required |= RequiredData::SolutionHessians;
                        required |= RequiredData::BasisHessians; // Needed for AD seeding via trialPhysicalHessian().
                    }
                } else {
                    required |= RequiredData::BasisValues;
                    if (order >= 1) required |= RequiredData::PhysicalGradients;
                    if (order >= 2) required |= RequiredData::BasisHessians;
                }
                break;
            case FormExprType::PreviousSolutionRef:
                required |= RequiredData::BasisValues;
                if (order >= 1) required |= RequiredData::PhysicalGradients;
                if (order >= 2) required |= RequiredData::BasisHessians;
                break;
            case FormExprType::Coefficient:
            case FormExprType::Coordinate:
                required |= RequiredData::PhysicalPoints;
                break;
            case FormExprType::ReferenceCoordinate:
                required |= RequiredData::QuadraturePoints;
                break;
            case FormExprType::Jacobian:
                required |= RequiredData::Jacobians;
                break;
            case FormExprType::JacobianInverse:
                required |= RequiredData::InverseJacobians;
                break;
            case FormExprType::JacobianDeterminant:
                required |= RequiredData::JacobianDets;
                break;
            case FormExprType::Normal:
                required |= RequiredData::Normals;
                break;
            case FormExprType::CellDiameter:
            case FormExprType::CellVolume:
            case FormExprType::FacetArea:
                required |= RequiredData::EntityMeasures;
                break;
            case FormExprType::MaterialStateOldRef:
            case FormExprType::MaterialStateWorkRef:
                required |= RequiredData::MaterialState;
                break;
            case FormExprType::Gradient:
            {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0]) self(self, *kids[0], order + 1);
                return;
            }
            case FormExprType::Divergence:
            case FormExprType::Curl: {
                const auto kids = n.childrenShared();
                if (kids.empty() || !kids[0]) {
                    return;
                }

                // H(curl)/H(div) vector bases provide curl/div directly via basis evaluation; do not
                // require PhysicalGradients/SolutionGradients for these operators.
                if (order == 0) {
                    if (const auto* sig = kids[0]->spaceSignature()) {
                        const bool is_vector_basis_hcurl =
                            (n.type() == FormExprType::Curl &&
                             sig->field_type == FieldType::Vector &&
                             sig->continuity == Continuity::H_curl);
                        const bool is_vector_basis_hdiv =
                            (n.type() == FormExprType::Divergence &&
                             sig->field_type == FieldType::Vector &&
                             sig->continuity == Continuity::H_div);

                        if (is_vector_basis_hcurl) {
                            required |= RequiredData::BasisCurls;
                            self(self, *kids[0], 0);
                            return;
                        }
                        if (is_vector_basis_hdiv) {
                            required |= RequiredData::BasisDivergences;
                            self(self, *kids[0], 0);
                            return;
                        }
                    }
                }

                // Fallback: treat as a first-order spatial operator (component-wise vectors).
                self(self, *kids[0], order + 1);
                return;
            }
            case FormExprType::Hessian: {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0]) self(self, *kids[0], order + 2);
                return;
            }
            case FormExprType::Conditional: {
                const auto kids = n.childrenShared();
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                    throw std::invalid_argument("FormCompiler: conditional expects 3 operands");
                }
                self(self, *kids[0], 0);
                self(self, *kids[1], order);
                self(self, *kids[2], order);
                return;
            }
            case FormExprType::Less:
            case FormExprType::LessEqual:
            case FormExprType::Greater:
            case FormExprType::GreaterEqual:
            case FormExprType::Equal:
            case FormExprType::NotEqual: {
                const auto kids = n.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    throw std::invalid_argument("FormCompiler: comparison expects 2 operands");
                }
                self(self, *kids[0], 0);
                self(self, *kids[1], 0);
                return;
            }
            default:
                break;
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child, order);
        }
    };

    visit(visit, node, 0);

    // Always need quadrature/integration weights for any integral evaluation.
    required |= RequiredData::IntegrationWeights;
    return required;
}

std::vector<assembly::FieldRequirement> analyzeFieldRequirements(const FormExprNode& node)
{
    using assembly::FieldRequirement;
    using assembly::RequiredData;

    std::unordered_map<FieldId, RequiredData> req_by_field;

    const auto add = [&](FieldId id, RequiredData bits) {
        if (id == INVALID_FIELD_ID) {
            throw std::invalid_argument("FormCompiler: DiscreteField node has invalid FieldId");
        }
        req_by_field[id] |= bits;
    };

    const auto visit = [&](const auto& self, const FormExprNode& n, int order) -> void {
        order = std::clamp(order, 0, 2);
        switch (n.type()) {
            case FormExprType::DiscreteField:
            case FormExprType::StateField: {
                const auto fid = n.fieldId();
                if (!fid) {
                    throw std::logic_error("FormCompiler: DiscreteField/StateField node missing fieldId()");
                }
                RequiredData bits = RequiredData::SolutionValues;
                if (order >= 1) bits |= RequiredData::SolutionGradients;
                if (order >= 2) bits |= RequiredData::SolutionHessians;
                add(*fid, bits);
                break;
            }
            case FormExprType::Gradient:
            case FormExprType::Divergence:
            case FormExprType::Curl: {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0]) self(self, *kids[0], order + 1);
                return;
            }
            case FormExprType::Hessian: {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0]) self(self, *kids[0], order + 2);
                return;
            }
            case FormExprType::Conditional: {
                const auto kids = n.childrenShared();
                if (kids.size() != 3u || !kids[0] || !kids[1] || !kids[2]) {
                    throw std::invalid_argument("FormCompiler: conditional expects 3 operands");
                }
                self(self, *kids[0], 0);
                self(self, *kids[1], order);
                self(self, *kids[2], order);
                return;
            }
            case FormExprType::Less:
            case FormExprType::LessEqual:
            case FormExprType::Greater:
            case FormExprType::GreaterEqual:
            case FormExprType::Equal:
            case FormExprType::NotEqual: {
                const auto kids = n.childrenShared();
                if (kids.size() != 2u || !kids[0] || !kids[1]) {
                    throw std::invalid_argument("FormCompiler: comparison expects 2 operands");
                }
                self(self, *kids[0], 0);
                self(self, *kids[1], 0);
                return;
            }
            default:
                break;
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child, order);
        }
    };

    visit(visit, node, 0);

    std::vector<FieldRequirement> out;
    out.reserve(req_by_field.size());
    for (const auto& kv : req_by_field) {
        out.push_back(FieldRequirement{kv.first, kv.second});
    }
    std::sort(out.begin(), out.end(),
              [](const FieldRequirement& a, const FieldRequirement& b) { return a.field < b.field; });
    return out;
}

int analyzeTimeDerivativeOrder(const FormExprNode& node, FormKind kind)
{
    int max_order = 0;
    int dt_count = 0;
    std::optional<int> dt_order{};

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        if (n.type() == FormExprType::TimeDerivative) {
            ++dt_count;
            const int order = n.timeDerivativeOrder().value_or(1);
            if (order <= 0) {
                throw std::invalid_argument("FormCompiler: dt(·,k) requires k >= 1");
            }

            if (!dt_order) {
                dt_order = order;
            } else if (*dt_order != order) {
                throw std::invalid_argument("FormCompiler: multiple dt() orders in one term are not supported");
            }
            max_order = std::max(max_order, order);

            const auto kids = n.childrenShared();
            if (kids.size() != 1 || !kids[0]) {
                throw std::invalid_argument("FormCompiler: dt(·,k) must have exactly 1 operand");
            }
            const auto operand_type = kids[0]->type();
            const bool ok =
                (operand_type == FormExprType::TrialFunction) ||
                (kind == FormKind::Residual &&
                 (operand_type == FormExprType::DiscreteField || operand_type == FormExprType::StateField));
            if (!ok) {
                if (kind == FormKind::Residual) {
                    throw std::invalid_argument(
                        "FormCompiler: dt(·,k) currently supports TrialFunction and DiscreteField/StateField operands only");
                }
                throw std::invalid_argument(
                    "FormCompiler: dt(·,k) currently supports TrialFunction operands only");
            }
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, node);

    // For linear/bilinear forms, we restrict each integral term to at most one dt()
    // operator so time-dependent contributions remain affine in the time derivative.
    //
    // Residual forms may be nonlinear in dt(u) (e.g., stabilization terms that reuse
    // a strong residual containing dt(u)), so allow multiple dt() nodes there.
    if (dt_count > 1 && kind != FormKind::Residual) {
        throw std::invalid_argument("FormCompiler: multiple dt() factors in one integral term are not supported");
    }

    return max_order;
}

void collectIntegralTerms(
    const FormExpr& expr,
    int sign,
    std::vector<IntegralTerm>& out_terms)
{
    if (!expr.isValid()) {
        throw std::invalid_argument("collectIntegralTerms: invalid expression");
    }

    const auto& n = *expr.node();
    const auto children = n.childrenShared();

    const auto collect_integrand_terms = [&](const auto& self,
                                             const FormExpr& integrand_expr,
                                             int integrand_sign,
                                             IntegralDomain domain,
                                             int boundary_marker,
                                             int interface_marker) -> void {
        const auto& in = *integrand_expr.node();
        const auto kids = in.childrenShared();

        switch (in.type()) {
            case FormExprType::Add: {
                if (kids.size() != 2) throw std::logic_error("Add node must have 2 children");
                self(self, makeExprFromNode(kids[0]), integrand_sign, domain, boundary_marker, interface_marker);
                self(self, makeExprFromNode(kids[1]), integrand_sign, domain, boundary_marker, interface_marker);
                return;
            }
            case FormExprType::Subtract: {
                if (kids.size() != 2) throw std::logic_error("Subtract node must have 2 children");
                self(self, makeExprFromNode(kids[0]), integrand_sign, domain, boundary_marker, interface_marker);
                self(self, makeExprFromNode(kids[1]), -integrand_sign, domain, boundary_marker, interface_marker);
                return;
            }
            case FormExprType::Negate: {
                if (kids.size() != 1) throw std::logic_error("Negate node must have 1 child");
                self(self, makeExprFromNode(kids[0]), -integrand_sign, domain, boundary_marker, interface_marker);
                return;
            }
            default:
                break;
        }

        FormExpr integrand = integrand_expr;
        if (integrand_sign < 0) {
            integrand = FormExpr::constant(-1.0) * integrand;
        }

        IntegralTerm term;
        term.domain = domain;
        term.boundary_marker = boundary_marker;
        term.interface_marker = interface_marker;
        term.integrand = std::move(integrand);
        term.debug_string = term.integrand.toString();
        out_terms.push_back(std::move(term));
    };

    switch (n.type()) {
        case FormExprType::Add: {
            if (children.size() != 2) throw std::logic_error("Add node must have 2 children");
            collectIntegralTerms(makeExprFromNode(children[0]), sign, out_terms);
            collectIntegralTerms(makeExprFromNode(children[1]), sign, out_terms);
            return;
        }
        case FormExprType::Subtract: {
            if (children.size() != 2) throw std::logic_error("Subtract node must have 2 children");
            collectIntegralTerms(makeExprFromNode(children[0]), sign, out_terms);
            collectIntegralTerms(makeExprFromNode(children[1]), -sign, out_terms);
            return;
        }
        case FormExprType::Negate: {
            if (children.size() != 1) throw std::logic_error("Negate node must have 1 child");
            collectIntegralTerms(makeExprFromNode(children[0]), -sign, out_terms);
            return;
        }
        case FormExprType::CellIntegral: {
            if (children.size() != 1) throw std::logic_error("CellIntegral node must have 1 child");
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::Cell,
                                    /*boundary_marker=*/-1,
                                    /*interface_marker=*/-1);
            return;
        }
        case FormExprType::BoundaryIntegral: {
            if (children.size() != 1) throw std::logic_error("BoundaryIntegral node must have 1 child");
            const int marker = n.boundaryMarker().value_or(-1);
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::Boundary,
                                    /*boundary_marker=*/marker,
                                    /*interface_marker=*/-1);
            return;
        }
        case FormExprType::InteriorFaceIntegral: {
            if (children.size() != 1) throw std::logic_error("InteriorFaceIntegral node must have 1 child");
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::InteriorFace,
                                    /*boundary_marker=*/-1,
                                    /*interface_marker=*/-1);
            return;
        }
        case FormExprType::InterfaceIntegral: {
            if (children.size() != 1) throw std::logic_error("InterfaceIntegral node must have 1 child");
            const int marker = n.interfaceMarker().value_or(-1);
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::InterfaceFace,
                                    /*boundary_marker=*/-1,
                                    /*interface_marker=*/marker);
            return;
        }
        default:
            break;
    }

    throw std::invalid_argument(
        "FormCompiler: top-level expression must be a sum of integrals; got: " + expr.toString());
}

} // namespace detail

FormIR FormCompiler::compileImpl(const FormExpr& form, FormKind kind)
{
    if (!form.isValid()) {
        throw std::invalid_argument("FormCompiler: cannot compile invalid form");
    }

    detail::requireNoIndexedAccess(*form.node());
    detail::requireNoCoupledPlaceholders(*form.node());

    FormIR ir;
    ir.setCompiled(false);
    ir.setKind(kind);

    const auto args = detail::analyzeBoundArguments(*form.node());
    ir.setTestSpace(args.test_space);
    ir.setTrialSpace(args.trial_space);

    std::vector<IntegralTerm> terms;
    detail::collectIntegralTerms(form, /*sign=*/+1, terms);

    assembly::RequiredData required = assembly::RequiredData::None;
    std::unordered_map<FieldId, assembly::RequiredData> field_required{};
    int max_time_order = 0;
    for (auto& t : terms) {
        t.time_derivative_order = detail::analyzeTimeDerivativeOrder(*t.integrand.node(), kind);
        max_time_order = std::max(max_time_order, t.time_derivative_order);

        t.required_data = detail::analyzeRequiredData(*t.integrand.node(), kind);
        for (const auto& fr : detail::analyzeFieldRequirements(*t.integrand.node())) {
            field_required[fr.field] |= fr.required;
        }

        // Face terms require face geometry context (surface measure, normals).
        if (t.domain == IntegralDomain::Boundary ||
            t.domain == IntegralDomain::InteriorFace ||
            t.domain == IntegralDomain::InterfaceFace) {
            t.required_data |= assembly::RequiredData::Normals;
        }

        // Interior-face terms require plus-side (neighbor) context; include
        // DG-oriented flags so assemblers can prepare the correct data.
        if (t.domain == IntegralDomain::InteriorFace || t.domain == IntegralDomain::InterfaceFace) {
            t.required_data |= assembly::RequiredData::NeighborData;
            t.required_data |= assembly::RequiredData::FaceOrientations;
        }
        required |= t.required_data;
    }

    std::vector<assembly::FieldRequirement> field_requirements;
    field_requirements.reserve(field_required.size());
    for (const auto& kv : field_required) {
        field_requirements.push_back(assembly::FieldRequirement{kv.first, kv.second});
    }
    std::sort(field_requirements.begin(), field_requirements.end(),
              [](const assembly::FieldRequirement& a, const assembly::FieldRequirement& b) { return a.field < b.field; });

    ir.setTerms(std::move(terms));
    ir.setRequiredData(required);
    ir.setFieldRequirements(std::move(field_requirements));
    ir.setMaxTimeDerivativeOrder(max_time_order);
    ir.setCompiled(true);

    std::ostringstream oss;
    oss << "FormIR\n";
    oss << "  kind: ";
    switch (kind) {
        case FormKind::Linear: oss << "linear\n"; break;
        case FormKind::Bilinear: oss << "bilinear\n"; break;
        case FormKind::Residual: oss << "residual\n"; break;
    }
    oss << "  terms: " << ir.terms().size() << "\n";
    for (const auto& t : ir.terms()) {
        oss << "    - ";
        switch (t.domain) {
            case IntegralDomain::Cell: oss << "dx"; break;
            case IntegralDomain::Boundary: oss << "ds(" << t.boundary_marker << ")"; break;
            case IntegralDomain::InteriorFace: oss << "dS"; break;
            case IntegralDomain::InterfaceFace: oss << "dI(" << t.interface_marker << ")"; break;
        }
        if (t.time_derivative_order > 0) {
            oss << " [dt^" << t.time_derivative_order << "]";
        }
        oss << " : " << t.debug_string << "\n";
    }
    ir.setDump(oss.str());

    return ir;
}

FormIR FormCompiler::compileLinear(const FormExpr& form)
{
    if (!form.hasTest()) {
        throw std::invalid_argument("FormCompiler::compileLinear: form has no test function");
    }
    if (form.hasTrial()) {
        throw std::invalid_argument("FormCompiler::compileLinear: form contains TrialFunction");
    }
    return compileImpl(form, FormKind::Linear);
}

FormIR FormCompiler::compileBilinear(const FormExpr& form)
{
    if (!form.hasTest() || !form.hasTrial()) {
        throw std::invalid_argument("FormCompiler::compileBilinear: form must contain both test and trial functions");
    }
    return compileImpl(form, FormKind::Bilinear);
}

FormIR FormCompiler::compileResidual(const FormExpr& residual_form)
{
    if (!residual_form.hasTest()) {
        throw std::invalid_argument("FormCompiler::compileResidual: residual form has no test function");
    }
    if (!residual_form.hasTrial()) {
        throw std::invalid_argument("FormCompiler::compileResidual: residual form has no TrialFunction (unknown)");
    }
    return compileImpl(residual_form, FormKind::Residual);
}

std::vector<std::optional<FormIR>> FormCompiler::compileLinear(const BlockLinearForm& blocks)
{
    std::vector<std::optional<FormIR>> out(blocks.numTestFields());
    for (std::size_t i = 0; i < blocks.numTestFields(); ++i) {
        if (!blocks.hasBlock(i)) {
            out[i] = std::nullopt;
            continue;
        }
        out[i] = compileLinear(blocks.block(i));
    }
    return out;
}

std::vector<std::vector<std::optional<FormIR>>> FormCompiler::compileBilinear(const BlockBilinearForm& blocks)
{
    std::vector<std::vector<std::optional<FormIR>>> out;
    out.resize(blocks.numTestFields());
    for (auto& row : out) {
        row.resize(blocks.numTrialFields());
    }

    for (std::size_t i = 0; i < blocks.numTestFields(); ++i) {
        for (std::size_t j = 0; j < blocks.numTrialFields(); ++j) {
            if (!blocks.hasBlock(i, j)) {
                continue;
            }
            out[i][j] = compileBilinear(blocks.block(i, j));
        }
    }
    return out;
}

std::vector<std::vector<std::optional<FormIR>>> FormCompiler::compileResidual(const BlockBilinearForm& blocks)
{
    std::vector<std::vector<std::optional<FormIR>>> out;
    out.resize(blocks.numTestFields());
    for (auto& row : out) {
        row.resize(blocks.numTrialFields());
    }

    for (std::size_t i = 0; i < blocks.numTestFields(); ++i) {
        for (std::size_t j = 0; j < blocks.numTrialFields(); ++j) {
            if (!blocks.hasBlock(i, j)) {
                continue;
            }
            out[i][j] = compileResidual(blocks.block(i, j));
        }
    }
    return out;
}

} // namespace forms
} // namespace FE
} // namespace svmp
