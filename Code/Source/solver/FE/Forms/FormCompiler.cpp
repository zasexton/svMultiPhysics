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

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        switch (n.type()) {
            case FormExprType::TestFunction:
                required |= RequiredData::BasisValues;
                break;
            case FormExprType::TrialFunction:
                if (kind == FormKind::Residual) {
                    required |= RequiredData::SolutionValues;
                } else {
                    required |= RequiredData::BasisValues;
                }
                break;
            case FormExprType::Coefficient:
                required |= RequiredData::PhysicalPoints;
                break;
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
            case FormExprType::Gradient:
                required |= RequiredData::PhysicalGradients;
                if (kind == FormKind::Residual) {
                    const auto kids = n.childrenShared();
                    if (!kids.empty() && kids[0] && kids[0]->type() == FormExprType::TrialFunction) {
                        required |= RequiredData::SolutionGradients;
                    }
                }
                break;
            case FormExprType::Divergence:
            case FormExprType::Curl:
                // div/curl require physical gradients of basis functions; for residuals involving
                // TrialFunction operands, we also need solution gradients (scalar) / Jacobians (vector).
                required |= RequiredData::PhysicalGradients;
                if (kind == FormKind::Residual) {
                    const auto kids = n.childrenShared();
                    if (!kids.empty() && kids[0] && kids[0]->type() == FormExprType::TrialFunction) {
                        required |= RequiredData::SolutionGradients;
                    }
                }
                break;
            case FormExprType::Hessian:
                required |= RequiredData::BasisHessians;
                if (kind == FormKind::Residual) {
                    const auto kids = n.childrenShared();
                    if (!kids.empty() && kids[0]) {
                        const auto* child = kids[0].get();
                        if (child->type() == FormExprType::TrialFunction) {
                            required |= RequiredData::SolutionHessians;
                        } else if (child->type() == FormExprType::Component) {
                            const auto gkids = child->childrenShared();
                            if (!gkids.empty() && gkids[0] && gkids[0]->type() == FormExprType::TrialFunction) {
                                required |= RequiredData::SolutionHessians;
                            }
                        }
                    }
                }
                break;
            case FormExprType::BoundaryIntegral:
                required |= RequiredData::Normals;
                break;
            default:
                break;
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, node);

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

    const auto visit = [&](const auto& self, const FormExprNode& n) -> void {
        switch (n.type()) {
            case FormExprType::DiscreteField: {
                const auto fid = n.fieldId();
                if (!fid) {
                    throw std::logic_error("FormCompiler: DiscreteField node missing fieldId()");
                }
                add(*fid, RequiredData::SolutionValues);
                break;
            }
            case FormExprType::StateField: {
                const auto fid = n.fieldId();
                if (!fid) {
                    throw std::logic_error("FormCompiler: StateField node missing fieldId()");
                }
                add(*fid, RequiredData::SolutionValues);
                break;
            }
            case FormExprType::Gradient:
            case FormExprType::Divergence:
            case FormExprType::Curl: {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0] &&
                    (kids[0]->type() == FormExprType::DiscreteField || kids[0]->type() == FormExprType::StateField)) {
                    const auto fid = kids[0]->fieldId();
                    if (!fid) {
                        throw std::logic_error("FormCompiler: DiscreteField node missing fieldId()");
                    }
                    add(*fid, RequiredData::SolutionValues | RequiredData::SolutionGradients);
                }
                break;
            }
            case FormExprType::Hessian: {
                const auto kids = n.childrenShared();
                if (!kids.empty() && kids[0]) {
                    const auto& child = *kids[0];
                    if (child.type() == FormExprType::DiscreteField || child.type() == FormExprType::StateField) {
                        const auto fid = child.fieldId();
                        if (!fid) {
                            throw std::logic_error("FormCompiler: DiscreteField node missing fieldId()");
                        }
                        add(*fid, RequiredData::SolutionValues | RequiredData::SolutionHessians);
                    } else if (child.type() == FormExprType::Component) {
                        const auto ckids = child.childrenShared();
                        if (!ckids.empty() && ckids[0] &&
                            (ckids[0]->type() == FormExprType::DiscreteField || ckids[0]->type() == FormExprType::StateField)) {
                            const auto fid = ckids[0]->fieldId();
                            if (!fid) {
                                throw std::logic_error("FormCompiler: DiscreteField node missing fieldId()");
                            }
                            add(*fid, RequiredData::SolutionValues | RequiredData::SolutionHessians);
                        }
                    }
                }
                break;
            }
            default:
                break;
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, node);

    std::vector<FieldRequirement> out;
    out.reserve(req_by_field.size());
    for (const auto& kv : req_by_field) {
        out.push_back(FieldRequirement{kv.first, kv.second});
    }
    std::sort(out.begin(), out.end(),
              [](const FieldRequirement& a, const FieldRequirement& b) { return a.field < b.field; });
    return out;
}

int analyzeTimeDerivativeOrder(const FormExprNode& node)
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
            if (kids[0]->type() != FormExprType::TrialFunction) {
                throw std::invalid_argument("FormCompiler: dt(·,k) currently supports TrialFunction operands only");
            }
        }

        for (const auto& child : n.childrenShared()) {
            if (child) self(self, *child);
        }
    };

    visit(visit, node);

    if (dt_count > 1) {
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
                                             int marker) -> void {
        const auto& in = *integrand_expr.node();
        const auto kids = in.childrenShared();

        switch (in.type()) {
            case FormExprType::Add: {
                if (kids.size() != 2) throw std::logic_error("Add node must have 2 children");
                self(self, makeExprFromNode(kids[0]), integrand_sign, domain, marker);
                self(self, makeExprFromNode(kids[1]), integrand_sign, domain, marker);
                return;
            }
            case FormExprType::Subtract: {
                if (kids.size() != 2) throw std::logic_error("Subtract node must have 2 children");
                self(self, makeExprFromNode(kids[0]), integrand_sign, domain, marker);
                self(self, makeExprFromNode(kids[1]), -integrand_sign, domain, marker);
                return;
            }
            case FormExprType::Negate: {
                if (kids.size() != 1) throw std::logic_error("Negate node must have 1 child");
                self(self, makeExprFromNode(kids[0]), -integrand_sign, domain, marker);
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
        term.boundary_marker = marker;
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
                                    /*marker=*/-1);
            return;
        }
        case FormExprType::BoundaryIntegral: {
            if (children.size() != 1) throw std::logic_error("BoundaryIntegral node must have 1 child");
            const int marker = n.boundaryMarker().value_or(-1);
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::Boundary,
                                    marker);
            return;
        }
        case FormExprType::InteriorFaceIntegral: {
            if (children.size() != 1) throw std::logic_error("InteriorFaceIntegral node must have 1 child");
            collect_integrand_terms(collect_integrand_terms,
                                    makeExprFromNode(children[0]),
                                    sign,
                                    IntegralDomain::InteriorFace,
                                    /*marker=*/-1);
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
        t.time_derivative_order = detail::analyzeTimeDerivativeOrder(*t.integrand.node());
        max_time_order = std::max(max_time_order, t.time_derivative_order);

        t.required_data = detail::analyzeRequiredData(*t.integrand.node(), kind);
        for (const auto& fr : detail::analyzeFieldRequirements(*t.integrand.node())) {
            field_required[fr.field] |= fr.required;
        }

        // Face terms require face geometry context (surface measure, normals).
        if (t.domain == IntegralDomain::Boundary || t.domain == IntegralDomain::InteriorFace) {
            t.required_data |= assembly::RequiredData::Normals;
        }

        // Interior-face terms require plus-side (neighbor) context; include
        // DG-oriented flags so assemblers can prepare the correct data.
        if (t.domain == IntegralDomain::InteriorFace) {
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
