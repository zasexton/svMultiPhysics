/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Geometry/CutQuadrature.h"
#include "Forms/FormIR.h"

#include <algorithm>
#include <map>
#include <sstream>

namespace svmp {
namespace FE {
namespace forms {

namespace {
void collectCutSurfaceFields(const FormExprNode& node, std::vector<FieldId>& fields)
{
    if (node.type() == FormExprType::CutSurfaceVariation) {
        const auto field = node.fieldId().value();
        auto it = std::lower_bound(fields.begin(), fields.end(), field);
        if (it == fields.end() || *it != field) fields.insert(it, field);
        return;
    }
    for (const auto& child : node.childrenShared()) collectCutSurfaceFields(*child, fields);
}

} // namespace

struct FormIR::Impl {
    bool compiled{false};
    FormKind kind{FormKind::Linear};
    assembly::RequiredData required{assembly::RequiredData::None};
    std::vector<assembly::FieldRequirement> field_requirements{};
    std::optional<FormExprNode::SpaceSignature> test_space{};
    std::optional<FormExprNode::SpaceSignature> trial_space{};
    int max_time_derivative_order{0};
    bool has_explicit_time_dependency{false};
    GeometrySensitivityOptions geometry_sensitivity{};
    bool has_geometry_sensitivity_terminals{false};
    std::vector<IntegralTerm> terms{};
    std::map<std::pair<int, std::optional<bool>>, std::vector<FieldId>> cut_surface_fields{};

    void indexCutSurfaceFields()
    {
        cut_surface_fields.clear();
        for (const auto& term : terms) {
            if (term.domain != IntegralDomain::InterfaceFace ||
                !assembly::hasFlag(term.required_data, assembly::RequiredData::CutSurfaceVariations)) continue;
            collectCutSurfaceFields(*term.integrand.node(), cut_surface_fields[{term.interface_marker, std::nullopt}]);
            collectCutSurfaceFields(*term.integrand.node(),
                                    cut_surface_fields[{term.interface_marker, interfaceTermIsTwoSided(term)}]);
        }
    }
    std::string dump{};
};

FormIR::FormIR() : impl_(std::make_unique<Impl>()) {}
FormIR::~FormIR() = default;

FormIR::FormIR(FormIR&&) noexcept = default;
FormIR& FormIR::operator=(FormIR&&) noexcept = default;

bool FormIR::isCompiled() const noexcept { return impl_->compiled; }
FormKind FormIR::kind() const noexcept { return impl_->kind; }
assembly::RequiredData FormIR::requiredData() const noexcept { return impl_->required; }
const std::vector<assembly::FieldRequirement>& FormIR::fieldRequirements() const noexcept { return impl_->field_requirements; }
const std::optional<FormExprNode::SpaceSignature>& FormIR::testSpace() const noexcept { return impl_->test_space; }
const std::optional<FormExprNode::SpaceSignature>& FormIR::trialSpace() const noexcept { return impl_->trial_space; }
int FormIR::maxTimeDerivativeOrder() const noexcept { return impl_->max_time_derivative_order; }
bool FormIR::hasExplicitTimeDependency() const noexcept { return impl_->has_explicit_time_dependency; }
const GeometrySensitivityOptions& FormIR::geometrySensitivityOptions() const noexcept { return impl_->geometry_sensitivity; }
bool FormIR::hasGeometrySensitivityTerminals() const noexcept { return impl_->has_geometry_sensitivity_terminals; }
bool FormIR::geometrySensitivityActive() const noexcept
{
    if (impl_->geometry_sensitivity.mode == GeometrySensitivityMode::MeshMotionUnknowns) {
        // A complete shape derivative varies every integral with the mesh.
        return impl_->has_geometry_sensitivity_terminals ||
               impl_->geometry_sensitivity.shape_derivative == ShapeDerivative::Complete;
    }
    if (impl_->geometry_sensitivity.mode ==
        GeometrySensitivityMode::LevelSetCutDomainUnknowns) {
        return !impl_->geometry_sensitivity.level_set_cut_domains.empty();
    }
    return false;
}

bool FormIR::hasCellTerms() const noexcept
{
    for (const auto& t : impl_->terms) {
        if (t.domain == IntegralDomain::Cell) return true;
    }
    return false;
}

bool FormIR::hasBoundaryTerms() const noexcept
{
    for (const auto& t : impl_->terms) {
        if (t.domain == IntegralDomain::Boundary) return true;
    }
    return false;
}

bool FormIR::hasInteriorFaceTerms() const noexcept
{
    for (const auto& t : impl_->terms) {
        if (t.domain == IntegralDomain::InteriorFace) return true;
    }
    return false;
}

bool FormIR::hasInterfaceFaceTerms() const noexcept
{
    for (const auto& t : impl_->terms) {
        if (t.domain == IntegralDomain::InterfaceFace) return true;
    }
    return false;
}

bool holdsTwoSidedInterfaceOperator(const FormExprNode* node) noexcept
{
    if (node == nullptr) {
        return false;
    }
    switch (node->type()) {
        case FormExprType::RestrictPlus:
        case FormExprType::Jump:
        case FormExprType::Average:
            return true;
        default:
            break;
    }
    for (const auto* child : node->children()) {
        if (holdsTwoSidedInterfaceOperator(child)) {
            return true;
        }
    }
    return false;
}

namespace {

/// Distribute sidedness through linear operations and products. A plus trace,
/// jump or average applies to its whole operand. A nonlinear function that
/// reads either trace is evaluated intact.
InterfaceSummands splitInterfaceOperands(const FormExpr& expr)
{
    const auto& node = *expr.node();
    const auto kids = node.childrenShared();

    const auto sum = [](const FormExpr& a, const FormExpr& b, bool subtract = false) {
        if (!b.isValid()) return a;
        if (!a.isValid()) return subtract ? -b : b;
        return subtract ? a - b : a + b;
    };
    switch (node.type()) {
        case FormExprType::Add:
        case FormExprType::Subtract: {
            const auto a = splitInterfaceOperands(FormExpr(kids[0]));
            const auto b = splitInterfaceOperands(FormExpr(kids[1]));
            if (!a.two_sided.isValid() && !b.two_sided.isValid()) return {expr, {}};
            if (!a.one_sided.isValid() && !b.one_sided.isValid()) return {{}, expr};
            const bool subtract = node.type() == FormExprType::Subtract;
            return {sum(a.one_sided, b.one_sided, subtract), sum(a.two_sided, b.two_sided, subtract)};
        }
        case FormExprType::Multiply:
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::OuterProduct:
        case FormExprType::CrossProduct: {
            const auto a = splitInterfaceOperands(FormExpr(kids[0]));
            const auto b = splitInterfaceOperands(FormExpr(kids[1]));
            if (!a.two_sided.isValid() && !b.two_sided.isValid()) return {expr, {}};
            if (!a.one_sided.isValid() || !b.one_sided.isValid()) return {{}, expr};
            // A*B = A1*B1 + A2*B + A1*B2. Keep B factored in the second term.
            const auto left_two_sided = a.two_sided.isValid() ? expr.withChildren({a.two_sided, FormExpr(kids[1])}) : FormExpr{};
            const auto right_two_sided = b.two_sided.isValid() ? expr.withChildren({a.one_sided, b.two_sided}) : FormExpr{};
            return {expr.withChildren({a.one_sided, b.one_sided}), sum(left_two_sided, right_two_sided)};
        }
        case FormExprType::Divide: {
            if (holdsTwoSidedInterfaceOperator(kids[1].get())) return {{}, expr};
            const auto a = splitInterfaceOperands(FormExpr(kids[0]));
            if (!a.two_sided.isValid()) return {expr, {}};
            if (!a.one_sided.isValid()) return {{}, expr};
            return {a.one_sided / FormExpr(kids[1]), a.two_sided / FormExpr(kids[1])};
        }
        case FormExprType::Negate:
        case FormExprType::Gradient:
        case FormExprType::Divergence:
        case FormExprType::Curl:
        case FormExprType::Hessian:
        case FormExprType::TimeDerivative:
        case FormExprType::RestrictMinus:
        case FormExprType::Component:
        case FormExprType::IndexedAccess:
        case FormExprType::Transpose:
        case FormExprType::Trace:
        case FormExprType::Deviator:
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart:
        case FormExprType::Pullback:
        case FormExprType::Pushforward: {
            const auto a = splitInterfaceOperands(FormExpr(kids[0]));
            if (!a.two_sided.isValid()) return {expr, {}};
            if (!a.one_sided.isValid()) return {{}, expr};
            return {expr.withChildren({a.one_sided}), expr.withChildren({a.two_sided})};
        }
        case FormExprType::AsVector:
        case FormExprType::AsTensor: {
            std::vector<FormExpr> one, two;
            const auto zero = FormExpr::constant(0);
            bool has_one = false, has_two = false;
            for (const auto& child : kids) {
                const auto part = splitInterfaceOperands(FormExpr(child));
                has_one |= part.one_sided.isValid();
                has_two |= part.two_sided.isValid();
                one.push_back(part.one_sided.isValid() ? part.one_sided : zero);
                two.push_back(part.two_sided.isValid() ? part.two_sided : zero);
            }
            if (!has_two) return {expr, {}};
            if (!has_one) return {{}, expr};
            return {expr.withChildren(std::move(one)), expr.withChildren(std::move(two))};
        }
        default:
            return holdsTwoSidedInterfaceOperator(&node) ? InterfaceSummands{{}, expr} : InterfaceSummands{expr, {}};
    }
}

} // namespace

InterfaceSummands splitInterfaceSummands(const FormExpr& expr)
{
    if (!holdsTwoSidedInterfaceOperator(expr.node())) return {expr, {}};
    return splitInterfaceOperands(expr);
}

[[nodiscard]] FormExpr wrapIntegralTermMeasure(
    const FormExpr& integrand,
    const IntegralTerm& term)
{
    if (term.exterior_boundary_measure.has_value()) {
        return integrand.dExteriorBoundary(*term.exterior_boundary_measure);
    }
    switch (term.domain) {
        case IntegralDomain::Cell:
            return integrand.dx();
        case IntegralDomain::Boundary:
            return integrand.ds(term.boundary_marker);
        case IntegralDomain::InteriorFace:
            return term.interior_facet_side.has_value()
                       ? integrand.dS(term.interface_marker,
                                      *term.interior_facet_side)
                       : integrand.dS(term.interface_marker);
        case IntegralDomain::InterfaceFace: {
            const auto measure = integrand.dI(term.interface_marker);
            return term.interface_two_sided ? measure.withTwoSidedTerms() : measure;
        }
        case IntegralDomain::CutVolume:
            return integrand.dCutVolume(
                term.interface_marker, term.cut_volume_side);
    }
    throw std::invalid_argument(
        "unsupported integral domain");
}

bool interfaceTermIsTwoSided(const IntegralTerm& term) noexcept
{
    return term.interface_two_sided || holdsTwoSidedInterfaceOperator(term.integrand.node());
}

bool interfaceTermsMixSidesOnMarker(const FormIR& ir, int interface_marker) noexcept
{
    bool one_sided = false;
    bool two_sided = false;
    for (const auto& term : ir.terms()) {
        if (term.domain != IntegralDomain::InterfaceFace) {
            continue;
        }
        if (term.interface_marker >= 0 && term.interface_marker != interface_marker) {
            continue;
        }
        (interfaceTermIsTwoSided(term) ? two_sided : one_sided) = true;
    }
    return one_sided && two_sided;
}

bool interfaceTermsMixSides(const FormIR& ir) noexcept
{
    // Every stated marker, and one no term states (the terms on every marker).
    constexpr int unstated = -2;
    for (const auto& term : ir.terms()) {
        if (term.domain != IntegralDomain::InterfaceFace || term.interface_marker < 0) {
            continue;
        }
        if (interfaceTermsMixSidesOnMarker(ir, term.interface_marker)) {
            return true;
        }
    }
    return interfaceTermsMixSidesOnMarker(ir, unstated);
}

bool FormIR::hasCutVolumeTerms() const noexcept
{
    for (const auto& t : impl_->terms) {
        if (t.domain == IntegralDomain::CutVolume) return true;
    }
    return false;
}

const std::vector<IntegralTerm>& FormIR::terms() const noexcept { return impl_->terms; }
std::string FormIR::dump() const { return impl_->dump; }

FormIR FormIR::clone() const
{
    FormIR out;
    *out.impl_ = *impl_;
    return out;
}

void FormIR::transformIntegrands(const FormExpr::NodeTransform& transform)
{
    if (!transform) {
        return;
    }
    for (auto& term : impl_->terms) {
        if (!term.integrand.isValid()) {
            continue;
        }
        term.integrand = term.integrand.transformNodes(transform);
        term.debug_string = term.integrand.toString();
    }
    // Any cached dump is now stale (term strings may have changed).
    impl_->indexCutSurfaceFields();
    impl_->dump.clear();
}

void FormIR::transformIntegrands(const TermTransform& transform)
{
    if (!transform) {
        return;
    }
    for (auto& term : impl_->terms) {
        if (!term.integrand.isValid()) {
            continue;
        }
        const auto per_term = [&](const FormExprNode& n) -> std::optional<FormExpr> {
            return transform(n, term);
        };
        term.integrand = term.integrand.transformNodes(per_term);
        term.debug_string = term.integrand.toString();
    }
    impl_->indexCutSurfaceFields();
    impl_->dump.clear();
}

void FormIR::setCompiled(bool compiled) { impl_->compiled = compiled; }
void FormIR::setKind(FormKind kind) { impl_->kind = kind; }
void FormIR::setRequiredData(assembly::RequiredData required) { impl_->required = required; }
void FormIR::setFieldRequirements(std::vector<assembly::FieldRequirement> reqs) { impl_->field_requirements = std::move(reqs); }
void FormIR::setTestSpace(std::optional<FormExprNode::SpaceSignature> sig) { impl_->test_space = std::move(sig); }
void FormIR::setTrialSpace(std::optional<FormExprNode::SpaceSignature> sig) { impl_->trial_space = std::move(sig); }
void FormIR::setMaxTimeDerivativeOrder(int order) { impl_->max_time_derivative_order = order; }
void FormIR::setHasExplicitTimeDependency(bool has_time_dependency) { impl_->has_explicit_time_dependency = has_time_dependency; }
void FormIR::setGeometrySensitivityOptions(GeometrySensitivityOptions options) { impl_->geometry_sensitivity = options; }
void FormIR::setHasGeometrySensitivityTerminals(bool has_terminals) { impl_->has_geometry_sensitivity_terminals = has_terminals; }
void FormIR::setTerms(std::vector<IntegralTerm> terms)
{
    impl_->terms = std::move(terms);
    impl_->indexCutSurfaceFields();
}

assembly::RequiredData FormIR::cutDomainRequiredData(int marker, geometry::CutIntegrationSide side) const noexcept
{
    auto result = assembly::RequiredData::None;
    for (const auto& term : impl_->terms) {
        if (term.interface_marker >= 0 && marker >= 0 && term.interface_marker != marker) continue;
        if (side == geometry::CutIntegrationSide::Interface) {
            if (term.domain == IntegralDomain::InterfaceFace) result |= term.required_data;
        } else {
            const auto phase = side == geometry::CutIntegrationSide::Negative ? CutVolumeSide::Negative : CutVolumeSide::Positive;
            if (term.domain == IntegralDomain::Cell || (term.domain == IntegralDomain::CutVolume && term.cut_volume_side == phase))
                result |= term.required_data;
        }
    }
    return result;
}

std::span<const FieldId> FormIR::cutSurfaceFields(int marker, std::optional<bool> two_sided) const noexcept
{
    const auto it = impl_->cut_surface_fields.find({marker, two_sided});
    return it == impl_->cut_surface_fields.end() ? std::span<const FieldId>{} : it->second;
}
void FormIR::setDump(std::string dump) { impl_->dump = std::move(dump); }

} // namespace forms
} // namespace FE
} // namespace svmp
