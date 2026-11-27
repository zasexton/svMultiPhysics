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

namespace {

[[nodiscard]] bool holdsTwoSidedInterfaceOperator(const FormExprNode* node) noexcept
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

} // namespace

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
