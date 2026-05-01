/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingFormBuilder.h"

#include "Core/FEException.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <unordered_set>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

bool declarationMatchesNodeTerminal(
    const CouplingFormTerminalProvenanceDeclaration& declaration,
    const forms::FormExprNode& node)
{
    using forms::FormExprType;

    switch (node.type()) {
    case FormExprType::PreviousSolutionRef:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::PreviousSolution &&
               declaration.temporal_quantity ==
                   CouplingTemporalQuantity::FieldHistoryValue &&
               node.historyIndex().value_or(0) == declaration.history_index;
    case FormExprType::MeshVelocity:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::MeshTemporal &&
               declaration.temporal_quantity ==
                   CouplingTemporalQuantity::MeshVelocity;
    case FormExprType::MeshAcceleration:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::MeshTemporal &&
               declaration.temporal_quantity ==
                   CouplingTemporalQuantity::MeshAcceleration;
    case FormExprType::PreviousMeshVelocity:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::MeshTemporal &&
               declaration.temporal_quantity ==
                   CouplingTemporalQuantity::PreviousMeshVelocity;
    case FormExprType::PredictedMeshVelocity:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::MeshTemporal &&
               declaration.temporal_quantity ==
                   CouplingTemporalQuantity::PredictedMeshVelocity;
    case FormExprType::MeshDisplacement:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::MeshDisplacement;
    case FormExprType::Coordinate:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::Coordinate;
    case FormExprType::ReferenceCoordinate:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::ReferenceCoordinate;
    case FormExprType::CurrentCoordinate:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::CurrentCoordinate;
    case FormExprType::PreviousCoordinate:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::PreviousCoordinate;
    case FormExprType::ReferencePhysicalCoordinate:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::ReferencePhysicalCoordinate;
    case FormExprType::Jacobian:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::Jacobian;
    case FormExprType::JacobianInverse:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::JacobianInverse;
    case FormExprType::JacobianDeterminant:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::JacobianDeterminant;
    case FormExprType::CurrentJacobian:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::CurrentJacobian;
    case FormExprType::ReferenceJacobian:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::ReferenceJacobian;
    case FormExprType::CurrentJacobianDeterminant:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::CurrentJacobianDeterminant;
    case FormExprType::ReferenceJacobianDeterminant:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::ReferenceJacobianDeterminant;
    case FormExprType::Determinant:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               (declaration.geometry_quantity ==
                    CouplingGeometryTerminalQuantity::CurrentJacobianDeterminant ||
                declaration.geometry_quantity ==
                    CouplingGeometryTerminalQuantity::ReferenceJacobianDeterminant);
    case FormExprType::Normal:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::Normal;
    case FormExprType::CurrentNormal:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::CurrentNormal;
    case FormExprType::ReferenceNormal:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::ReferenceNormal;
    case FormExprType::CurrentMeasure:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::CurrentMeasure;
    case FormExprType::ReferenceMeasure:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::ReferenceMeasure;
    case FormExprType::SurfaceJacobian:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::SurfaceJacobian;
    case FormExprType::CellDiameter:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::CellDiameter;
    case FormExprType::CellVolume:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::CellVolume;
    case FormExprType::FacetArea:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::FacetArea;
    case FormExprType::CellDomainId:
        return declaration.kind ==
                   CouplingFormTerminalProvenanceKind::GeometryTerminal &&
               declaration.geometry_quantity ==
                   CouplingGeometryTerminalQuantity::CellDomainId;
    default:
        return false;
    }
}

bool sameFieldUse(const CouplingFieldUse& lhs,
                  const CouplingFieldUse& rhs) noexcept
{
    return lhs.participant_name == rhs.participant_name &&
           lhs.field_name == rhs.field_name;
}

void appendUniqueFieldUse(std::vector<CouplingFieldUse>& fields,
                          CouplingFieldUse field)
{
    const auto exists = std::find_if(
        fields.begin(),
        fields.end(),
        [&](const CouplingFieldUse& existing) {
            return sameFieldUse(existing, field);
        });
    if (exists == fields.end()) {
        fields.push_back(std::move(field));
    }
}

CouplingFieldUse fieldUseForFieldId(const CouplingContext& context,
                                    FieldId field_id)
{
    const CouplingFieldRef* matched = nullptr;
    for (const auto& field : context.fields()) {
        if (field.field_id != field_id) {
            continue;
        }
        if (matched != nullptr) {
            FE_THROW(InvalidArgumentException,
                     "coupling equation field-use inference found an ambiguous field id");
        }
        matched = &field;
    }
    FE_THROW_IF(matched == nullptr, InvalidArgumentException,
                "coupling equation field-use inference found an unknown field id");
    return fieldUse(matched->participant_name, matched->field_name);
}

void appendFieldUseForNode(std::vector<CouplingFieldUse>& fields,
                           const CouplingContext& context,
                           const forms::FormExprNode& node)
{
    const auto field_id = node.fieldId();
    if (!field_id.has_value() || *field_id == INVALID_FIELD_ID) {
        return;
    }
    appendUniqueFieldUse(fields, fieldUseForFieldId(context, *field_id));
}

void inferEquationFieldUses(const CouplingContext& context,
                            const forms::FormExprNode& node,
                            std::vector<CouplingFieldUse>& residual_fields,
                            std::vector<CouplingFieldUse>& trial_fields)
{
    switch (node.type()) {
    case forms::FormExprType::TestFunction:
        appendFieldUseForNode(residual_fields, context, node);
        break;
    case forms::FormExprType::DiscreteField:
    case forms::FormExprType::StateField:
        appendFieldUseForNode(trial_fields, context, node);
        break;
    default:
        break;
    }

    for (const auto* child : node.children()) {
        if (child != nullptr) {
            inferEquationFieldUses(context,
                                   *child,
                                   residual_fields,
                                   trial_fields);
        }
    }
}

void removeResidualFieldOverlaps(std::vector<CouplingFieldUse>& trial_fields,
                                 const std::vector<CouplingFieldUse>&
                                     residual_fields)
{
    trial_fields.erase(
        std::remove_if(
            trial_fields.begin(),
            trial_fields.end(),
            [&](const CouplingFieldUse& trial_field) {
                return std::any_of(
                    residual_fields.begin(),
                    residual_fields.end(),
                    [&](const CouplingFieldUse& residual_field) {
                        return sameFieldUse(trial_field, residual_field);
                    });
            }),
        trial_fields.end());
}

forms::FormExpr restrictToInterfaceSide(const forms::FormExpr& expr,
                                        CouplingInterfaceSide side)
{
    switch (side) {
    case CouplingInterfaceSide::Minus:
        return expr.minus();
    case CouplingInterfaceSide::Plus:
        return expr.plus();
    case CouplingInterfaceSide::None:
        break;
    }
    FE_THROW(InvalidArgumentException,
             "coupling interface view requires an explicit interface side");
    return forms::FormExpr{};
}

forms::FormExpr restrictToRegionSideIfNeeded(const forms::FormExpr& expr,
                                             const CouplingRegionRef& region)
{
    if (region.kind != CouplingRegionKind::InterfaceFace) {
        return expr;
    }
    return restrictToInterfaceSide(expr, region.side);
}

} // namespace

CouplingFormBuilder::CouplingFormBuilder(const CouplingContext& context)
    : context_(&context)
{
}

const CouplingContext& CouplingFormBuilder::context() const noexcept
{
    return *context_;
}

forms::FormExpr CouplingFormBuilder::state(std::string_view participant_name,
                                           std::string_view field_name,
                                           std::string symbol) const
{
    const auto ref = context().field(participant_name, field_name);
    FE_THROW_IF(ref.space == nullptr, InvalidArgumentException,
                "coupling field has no function space");
    return forms::StateField(ref.field_id, *ref.space, std::move(symbol));
}

forms::FormExpr CouplingFormBuilder::requiredState(
    std::string_view participant_name,
    const std::optional<std::string>& field_name,
    std::string symbol) const
{
    const auto required_field = requiredFieldName(field_name);
    return state(participant_name,
                 std::string_view(required_field),
                 std::move(symbol));
}

forms::FormExpr CouplingFormBuilder::test(std::string_view participant_name,
                                          std::string_view field_name,
                                          std::string symbol) const
{
    const auto ref = context().field(participant_name, field_name);
    FE_THROW_IF(ref.space == nullptr, InvalidArgumentException,
                "coupling field has no function space");
    return forms::TestField(ref.field_id, *ref.space, std::move(symbol));
}

forms::FormExpr CouplingFormBuilder::requiredTest(
    std::string_view participant_name,
    const std::optional<std::string>& field_name,
    std::string symbol) const
{
    const auto required_field = requiredFieldName(field_name);
    return test(participant_name,
                std::string_view(required_field),
                std::move(symbol));
}

forms::FormExpr CouplingFormBuilder::timeDerivative(std::string_view participant_name,
                                                    std::string_view field_name,
                                                    std::string symbol,
                                                    int order) const
{
    FE_THROW_IF(order <= 0, InvalidArgumentException,
                "coupling time derivative order must be positive");
    return state(participant_name, field_name, std::move(symbol)).dt(order);
}

forms::FormExpr CouplingFormBuilder::requiredTimeDerivative(
    std::string_view participant_name,
    const std::optional<std::string>& field_name,
    std::string symbol,
    int order) const
{
    const auto required_field = requiredFieldName(field_name);
    return timeDerivative(participant_name,
                          std::string_view(required_field),
                          std::move(symbol),
                          order);
}

forms::FormExpr CouplingFormBuilder::previousSolution(std::string_view participant_name,
                                                      std::string_view field_name,
                                                      int steps_back) const
{
    static_cast<void>(field(participant_name, field_name));
    FE_THROW_IF(steps_back <= 0, InvalidArgumentException,
                "previous solution history index must be positive");
    CouplingFormTerminalProvenanceDeclaration declaration;
    declaration.kind = CouplingFormTerminalProvenanceKind::PreviousSolution;
    declaration.field = CouplingFieldUse{
        .participant_name = std::string(participant_name),
        .field_name = std::string(field_name),
    };
    declaration.temporal_quantity = CouplingTemporalQuantity::FieldHistoryValue;
    declaration.history_index = steps_back;
    return recordTerminal(forms::FormExpr::previousSolution(steps_back),
                          std::move(declaration));
}

forms::FormExpr CouplingFormBuilder::requiredPreviousSolution(
    std::string_view participant_name,
    const std::optional<std::string>& field_name,
    int steps_back) const
{
    const auto required_field = requiredFieldName(field_name);
    return previousSolution(participant_name,
                            std::string_view(required_field),
                            steps_back);
}

forms::FormExpr CouplingFormBuilder::time() const
{
    return forms::t();
}

forms::FormExpr CouplingFormBuilder::timeStep() const
{
    return forms::deltat();
}

forms::FormExpr CouplingFormBuilder::effectiveTimeStep() const
{
    return forms::deltat_eff();
}

forms::FormExpr CouplingFormBuilder::meshDisplacement(
    const CouplingGeometryTerminalScope& scope) const
{
    CouplingFormTerminalProvenanceDeclaration declaration;
    declaration.kind = CouplingFormTerminalProvenanceKind::GeometryTerminal;
    declaration.scope = scope;
    declaration.geometry_quantity =
        CouplingGeometryTerminalQuantity::MeshDisplacement;
    declaration.mesh_motion_role = systems::MeshMotionFieldRole::Displacement;
    return recordTerminal(forms::meshDisplacement(), std::move(declaration));
}

forms::FormExpr CouplingFormBuilder::meshVelocity(
    const CouplingGeometryTerminalScope& scope) const
{
    CouplingFormTerminalProvenanceDeclaration declaration;
    declaration.kind = CouplingFormTerminalProvenanceKind::MeshTemporal;
    declaration.scope = scope;
    declaration.temporal_quantity = CouplingTemporalQuantity::MeshVelocity;
    declaration.mesh_motion_role = systems::MeshMotionFieldRole::Velocity;
    return recordTerminal(forms::meshVelocity(), std::move(declaration));
}

forms::FormExpr CouplingFormBuilder::meshAcceleration(
    const CouplingGeometryTerminalScope& scope) const
{
    CouplingFormTerminalProvenanceDeclaration declaration;
    declaration.kind = CouplingFormTerminalProvenanceKind::MeshTemporal;
    declaration.scope = scope;
    declaration.temporal_quantity = CouplingTemporalQuantity::MeshAcceleration;
    declaration.mesh_motion_role = systems::MeshMotionFieldRole::Acceleration;
    return recordTerminal(forms::meshAcceleration(), std::move(declaration));
}

forms::FormExpr CouplingFormBuilder::previousMeshVelocity(
    const CouplingGeometryTerminalScope& scope) const
{
    CouplingFormTerminalProvenanceDeclaration declaration;
    declaration.kind = CouplingFormTerminalProvenanceKind::MeshTemporal;
    declaration.scope = scope;
    declaration.temporal_quantity =
        CouplingTemporalQuantity::PreviousMeshVelocity;
    declaration.mesh_motion_role = systems::MeshMotionFieldRole::PreviousVelocity;
    return recordTerminal(forms::previousMeshVelocity(), std::move(declaration));
}

forms::FormExpr CouplingFormBuilder::predictedMeshVelocity(
    const CouplingGeometryTerminalScope& scope) const
{
    CouplingFormTerminalProvenanceDeclaration declaration;
    declaration.kind = CouplingFormTerminalProvenanceKind::MeshTemporal;
    declaration.scope = scope;
    declaration.temporal_quantity =
        CouplingTemporalQuantity::PredictedMeshVelocity;
    declaration.mesh_motion_role = systems::MeshMotionFieldRole::PredictedVelocity;
    return recordTerminal(forms::predictedMeshVelocity(), std::move(declaration));
}

forms::FormExpr CouplingFormBuilder::geometryTerminal(
    CouplingGeometryTerminalQuantity quantity,
    const CouplingGeometryTerminalScope& scope) const
{
    CouplingFormTerminalProvenanceDeclaration declaration;
    declaration.kind = CouplingFormTerminalProvenanceKind::GeometryTerminal;
    declaration.scope = scope;
    declaration.geometry_quantity = quantity;

    switch (quantity) {
    case CouplingGeometryTerminalQuantity::MeshDisplacement:
        declaration.mesh_motion_role = systems::MeshMotionFieldRole::Displacement;
        return recordTerminal(forms::meshDisplacement(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::Coordinate:
        return recordTerminal(forms::x(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::ReferenceCoordinate:
        return recordTerminal(forms::X(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::CurrentCoordinate:
        return recordTerminal(forms::currentCoordinate(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::PreviousCoordinate:
        return recordTerminal(forms::previousCoordinate(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::ReferencePhysicalCoordinate:
        return recordTerminal(forms::referenceCoordinatePhysical(),
                              std::move(declaration));
    case CouplingGeometryTerminalQuantity::Jacobian:
        return recordTerminal(forms::J(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::JacobianInverse:
        return recordTerminal(forms::Jinv(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::JacobianDeterminant:
        return recordTerminal(forms::detJ(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::CurrentJacobian:
        return recordTerminal(forms::currentJacobian(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::ReferenceJacobian:
        return recordTerminal(forms::referenceJacobian(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::CurrentJacobianDeterminant:
        return recordTerminal(forms::currentJacobianDeterminant(),
                              std::move(declaration));
    case CouplingGeometryTerminalQuantity::ReferenceJacobianDeterminant:
        return recordTerminal(forms::referenceJacobianDeterminant(),
                              std::move(declaration));
    case CouplingGeometryTerminalQuantity::Normal:
        return recordTerminal(forms::FormExpr::normal(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::CurrentNormal:
        return recordTerminal(forms::currentNormal(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::ReferenceNormal:
        return recordTerminal(forms::referenceNormal(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::CurrentMeasure:
        return recordTerminal(forms::currentMeasure(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::ReferenceMeasure:
        return recordTerminal(forms::referenceMeasure(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::SurfaceJacobian:
        return recordTerminal(forms::surfaceJacobian(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::CellDiameter:
        return recordTerminal(forms::h(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::CellVolume:
        return recordTerminal(forms::vol(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::FacetArea:
        return recordTerminal(forms::area(), std::move(declaration));
    case CouplingGeometryTerminalQuantity::CellDomainId:
        return recordTerminal(forms::domainId(), std::move(declaration));
    }
    FE_THROW(InvalidArgumentException,
             "unsupported coupling geometry terminal quantity");
}

std::vector<CouplingFormTerminalProvenanceDeclaration>
CouplingFormBuilder::terminalProvenanceFor(
    const forms::FormExpr& residual) const
{
    std::vector<CouplingFormTerminalProvenanceDeclaration> matched;
    if (!residual.isValid()) {
        return matched;
    }

    std::unordered_set<const forms::FormExprNode*> visited;
    const auto append_match = [&](const forms::FormExprNode* node) {
        for (const auto& recorded : recorded_terminals_) {
            const auto recorded_node = recorded.node.lock();
            if (!recorded_node || recorded_node.get() != node) {
                continue;
            }
            auto declaration = recorded.declaration;
            declaration.terminal_sequence = matched.size();
            matched.push_back(std::move(declaration));
            return true;
        }
        return false;
    };

    const auto has_lost_provenance = [&](const forms::FormExprNode& node) {
        for (const auto& recorded : recorded_terminals_) {
            if (declarationMatchesNodeTerminal(recorded.declaration, node)) {
                return true;
            }
        }
        return false;
    };

    const std::function<void(const forms::FormExprNode*)> visit =
        [&](const forms::FormExprNode* node) {
            if (node == nullptr || !visited.insert(node).second) {
                return;
            }
            if (!append_match(node) && has_lost_provenance(*node)) {
                FE_THROW(InvalidArgumentException,
                         "coupling terminal provenance identity was lost before attachment");
            }
            for (const auto* child : node->children()) {
                visit(child);
            }
        };

    visit(residual.node());
    return matched;
}

CouplingFormContribution CouplingFormBuilder::attachTerminalProvenance(
    CouplingFormContribution contribution) const
{
    auto matched = terminalProvenanceFor(contribution.residual);
    contribution.terminal_provenance.insert(
        contribution.terminal_provenance.end(),
        std::make_move_iterator(matched.begin()),
        std::make_move_iterator(matched.end()));
    return contribution;
}

CouplingFormContribution CouplingFormBuilder::equationContribution(
    CouplingEquationContributionRequest request) const
{
    CouplingFormContribution contribution;
    contribution.contribution_name = std::move(request.contribution_name);
    contribution.origin = std::move(request.origin);
    contribution.operator_name = std::move(request.operator_name);
    contribution.field_uses = std::move(request.residual_field_uses);
    contribution.extra_trial_field_uses = std::move(request.trial_field_uses);
    contribution.install_options_declaration =
        std::move(request.install_options_declaration);
    if (request.geometry_sensitivity.has_value()) {
        const auto mesh_motion_field =
            request.geometry_sensitivity->mesh_motion_field;
        contribution.install_options_declaration.geometry_sensitivity =
            std::move(request.geometry_sensitivity);
        if (request.include_geometry_sensitivity_field_as_trial &&
            mesh_motion_field.has_value()) {
            appendUniqueFieldUse(contribution.extra_trial_field_uses,
                                 *mesh_motion_field);
        }
    }
    contribution.install_options = std::move(request.install_options);
    contribution.residual = std::move(request.residual);
    return attachTerminalProvenance(std::move(contribution));
}

CouplingEquationSetBuilder CouplingFormBuilder::equationSet(
    CouplingEquationSetRequest request) const
{
    return CouplingEquationSetBuilder(*this, std::move(request));
}

forms::FormExpr CouplingFormBuilder::integrate(const forms::FormExpr& integrand,
                                               const CouplingRegionRef& region) const
{
    switch (region.kind) {
        case CouplingRegionKind::Domain:
            return integrand.dx();
        case CouplingRegionKind::Boundary:
            return integrand.ds(region.marker);
        case CouplingRegionKind::InteriorFace:
            return integrand.dS();
        case CouplingRegionKind::InterfaceFace:
            return integrand.dI(region.marker);
        case CouplingRegionKind::UserDefined:
        case CouplingRegionKind::Curve:
        case CouplingRegionKind::Point:
        case CouplingRegionKind::CutInterface:
            break;
    }
    FE_THROW(InvalidArgumentException,
             "user-defined coupling region requires a concrete Forms integration kind");
}

forms::FormExpr CouplingFormBuilder::integrate(const forms::FormExpr& integrand,
                                               std::string_view participant_name,
                                               std::string_view region_name) const
{
    return integrate(integrand, region(participant_name, region_name));
}

forms::FormExpr CouplingFormBuilder::integrateShared(
    const forms::FormExpr& integrand,
    std::string_view shared_region_name,
    std::string_view participant_name) const
{
    return integrate(integrand, sharedRegion(shared_region_name, participant_name));
}

CouplingSharedInterfaceView CouplingFormBuilder::sharedInterface(
    std::string_view name) const
{
    return CouplingSharedInterfaceView(*this, std::string(name));
}

CouplingRegionRelationView CouplingFormBuilder::regionRelation(
    CouplingRegionRelationRequirement requirement) const
{
    return CouplingRegionRelationView(*this, std::move(requirement));
}

CouplingFieldRef CouplingFormBuilder::field(std::string_view participant_name,
                                            std::string_view field_name) const
{
    return context().field(participant_name, field_name);
}

CouplingRegionRef CouplingFormBuilder::region(std::string_view participant_name,
                                              std::string_view region_name) const
{
    return context().region(participant_name, region_name);
}

CouplingRegionRef CouplingFormBuilder::sharedRegion(std::string_view name,
                                                    std::string_view participant_name) const
{
    return context().sharedRegion(name, participant_name);
}

SharedRegionRef CouplingFormBuilder::sharedRegionGroup(std::string_view name) const
{
    return context().sharedRegionGroup(name);
}

CouplingEquationSetBuilder::CouplingEquationSetBuilder(
    const CouplingFormBuilder& builder,
    CouplingEquationSetRequest request)
    : builder_(&builder)
    , request_(std::move(request))
{
}

CouplingFormContribution CouplingEquationSetBuilder::equation(
    CouplingNamedEquationRequest request) const
{
    FE_CHECK_NOT_NULL(builder_, "coupling equation set form builder");
    auto geometry_sensitivity = std::move(request.geometry_sensitivity);
    if (!geometry_sensitivity.has_value()) {
        geometry_sensitivity = request_.geometry_sensitivity;
    }
    return builder_->equationContribution(CouplingEquationContributionRequest{
        .contribution_name =
            makeCouplingGeneratedName(CouplingGeneratedNameRequest{
                .contract_name = request_.contract_name,
                .relation_name = request_.relation_name,
                .local_name = std::move(request.local_name),
                .explicit_name = std::move(request.explicit_name),
            }),
        .origin = request_.origin,
        .operator_name = request_.operator_name,
        .residual_field_uses = std::move(request.residual_field_uses),
        .trial_field_uses = std::move(request.trial_field_uses),
        .geometry_sensitivity = std::move(geometry_sensitivity),
        .include_geometry_sensitivity_field_as_trial =
            request.include_geometry_sensitivity_field_as_trial,
        .install_options = std::move(request.install_options),
        .residual = std::move(request.residual),
    });
}

CouplingFormContribution CouplingEquationSetBuilder::inferredEquation(
    CouplingInferredEquationRequest request) const
{
    FE_CHECK_NOT_NULL(builder_, "coupling equation set form builder");
    FE_THROW_IF(!request.residual.isValid(), InvalidArgumentException,
                "coupling inferred equation requires a residual form");

    std::vector<CouplingFieldUse> residual_fields;
    std::vector<CouplingFieldUse> trial_fields;
    inferEquationFieldUses(builder_->context(),
                           *request.residual.node(),
                           residual_fields,
                           trial_fields);

    for (const auto& terminal :
         builder_->terminalProvenanceFor(request.residual)) {
        if (terminal.field.has_value()) {
            appendUniqueFieldUse(trial_fields, *terminal.field);
        }
    }

    FE_THROW_IF(residual_fields.empty(), InvalidArgumentException,
                "coupling inferred equation requires a field-bound test function");
    removeResidualFieldOverlaps(trial_fields, residual_fields);

    return equation(CouplingNamedEquationRequest{
        .local_name = std::move(request.local_name),
        .explicit_name = std::move(request.explicit_name),
        .residual_field_uses = std::move(residual_fields),
        .trial_field_uses = std::move(trial_fields),
        .geometry_sensitivity = std::move(request.geometry_sensitivity),
        .include_geometry_sensitivity_field_as_trial =
            request.include_geometry_sensitivity_field_as_trial,
        .install_options = std::move(request.install_options),
        .residual = std::move(request.residual),
    });
}

std::vector<CouplingFormContribution> CouplingEquationSetBuilder::infer(
    std::vector<CouplingInferredEquationRequest> requests) const
{
    std::vector<CouplingFormContribution> contributions;
    contributions.reserve(requests.size());
    for (auto& request : requests) {
        contributions.push_back(inferredEquation(std::move(request)));
    }
    return contributions;
}

CouplingInterfaceSideView::CouplingInterfaceSideView(
    const CouplingFormBuilder& builder,
    std::string shared_region_name,
    CouplingRegionRef region)
    : builder_(&builder)
    , shared_region_name_(std::move(shared_region_name))
    , region_(std::move(region))
{
}

std::string_view CouplingInterfaceSideView::sharedRegionName() const noexcept
{
    return shared_region_name_;
}

std::string_view CouplingInterfaceSideView::participantName() const noexcept
{
    return region_.participant_name;
}

const CouplingRegionRef& CouplingInterfaceSideView::region() const noexcept
{
    return region_;
}

forms::FormExpr CouplingInterfaceSideView::state(std::string_view field_name,
                                                 std::string symbol) const
{
    return restrictToInterfaceSide(
        builder_->state(region_.participant_name, field_name, std::move(symbol)),
        region_.side);
}

forms::FormExpr CouplingInterfaceSideView::requiredState(
    const std::optional<std::string>& field_name,
    std::string symbol) const
{
    const auto required_field = requiredFieldName(field_name);
    return state(std::string_view(required_field), std::move(symbol));
}

forms::FormExpr CouplingInterfaceSideView::test(std::string_view field_name,
                                                std::string symbol) const
{
    return restrictToInterfaceSide(
        builder_->test(region_.participant_name, field_name, std::move(symbol)),
        region_.side);
}

forms::FormExpr CouplingInterfaceSideView::requiredTest(
    const std::optional<std::string>& field_name,
    std::string symbol) const
{
    const auto required_field = requiredFieldName(field_name);
    return test(std::string_view(required_field), std::move(symbol));
}

forms::FormExpr CouplingInterfaceSideView::dt(std::string_view field_name,
                                              std::string symbol,
                                              int order) const
{
    return restrictToInterfaceSide(
        builder_->timeDerivative(region_.participant_name,
                                 field_name,
                                 std::move(symbol),
                                 order),
        region_.side);
}

forms::FormExpr CouplingInterfaceSideView::requiredDt(
    const std::optional<std::string>& field_name,
    std::string symbol,
    int order) const
{
    const auto required_field = requiredFieldName(field_name);
    return dt(std::string_view(required_field), std::move(symbol), order);
}

forms::FormExpr CouplingInterfaceSideView::geometryTerminal(
    CouplingGeometryTerminalQuantity quantity) const
{
    const CouplingGeometryTerminalScope scope{
        .participant_name = region_.participant_name,
        .region = CouplingRegionEndpointDeclaration{
            .participant_name = region_.participant_name,
            .region_name = region_.region_name,
            .shared_region_name = shared_region_name_,
        },
        .location = CouplingGeometryTerminalLocationDeclaration{
            .region_kind = region_.kind,
            .shared_region_name = shared_region_name_,
            .side = region_.side,
        },
    };
    return restrictToInterfaceSide(builder_->geometryTerminal(quantity, scope),
                                   region_.side);
}

forms::FormExpr CouplingInterfaceSideView::normal() const
{
    return geometryTerminal(CouplingGeometryTerminalQuantity::Normal);
}

forms::FormExpr CouplingInterfaceSideView::normalComponent(
    const forms::FormExpr& value) const
{
    return forms::inner(value, normal());
}

forms::FormExpr CouplingInterfaceSideView::normalProjection(
    const forms::FormExpr& value) const
{
    const auto n = normal();
    return forms::inner(value, n) * n;
}

forms::FormExpr CouplingInterfaceSideView::tangentialProjection(
    const forms::FormExpr& value) const
{
    return value - normalProjection(value);
}

CouplingSharedInterfaceView::CouplingSharedInterfaceView(
    const CouplingFormBuilder& builder,
    std::string shared_region_name)
    : builder_(&builder)
    , shared_region_name_(std::move(shared_region_name))
{
}

std::string_view CouplingSharedInterfaceView::name() const noexcept
{
    return shared_region_name_;
}

SharedRegionRef CouplingSharedInterfaceView::group() const
{
    return builder_->sharedRegionGroup(shared_region_name_);
}

CouplingInterfaceSideView CouplingSharedInterfaceView::side(
    std::string_view participant_name) const
{
    return CouplingInterfaceSideView(
        *builder_,
        shared_region_name_,
        builder_->sharedRegion(shared_region_name_, participant_name));
}

forms::FormExpr CouplingSharedInterfaceView::integral(
    const forms::FormExpr& integrand,
    std::string_view integration_participant) const
{
    return builder_->integrateShared(integrand,
                                     shared_region_name_,
                                     integration_participant);
}

CouplingRegionEndpointView::CouplingRegionEndpointView(
    const CouplingFormBuilder& builder,
    std::string relation_name,
    CouplingRegionEndpointDeclaration endpoint,
    CouplingRegionRef region)
    : builder_(&builder)
    , relation_name_(std::move(relation_name))
    , endpoint_(std::move(endpoint))
    , region_(std::move(region))
{
}

std::string_view CouplingRegionEndpointView::relationName() const noexcept
{
    return relation_name_;
}

const CouplingRegionEndpointDeclaration&
CouplingRegionEndpointView::endpoint() const noexcept
{
    return endpoint_;
}

const CouplingRegionRef& CouplingRegionEndpointView::region() const noexcept
{
    return region_;
}

forms::FormExpr CouplingRegionEndpointView::state(std::string_view field_name,
                                                  std::string symbol) const
{
    return restrictToRegionSideIfNeeded(
        builder_->state(region_.participant_name, field_name, std::move(symbol)),
        region_);
}

forms::FormExpr CouplingRegionEndpointView::requiredState(
    const std::optional<std::string>& field_name,
    std::string symbol) const
{
    const auto required_field = requiredFieldName(field_name);
    return state(std::string_view(required_field), std::move(symbol));
}

forms::FormExpr CouplingRegionEndpointView::test(std::string_view field_name,
                                                 std::string symbol) const
{
    return restrictToRegionSideIfNeeded(
        builder_->test(region_.participant_name, field_name, std::move(symbol)),
        region_);
}

forms::FormExpr CouplingRegionEndpointView::requiredTest(
    const std::optional<std::string>& field_name,
    std::string symbol) const
{
    const auto required_field = requiredFieldName(field_name);
    return test(std::string_view(required_field), std::move(symbol));
}

forms::FormExpr CouplingRegionEndpointView::dt(std::string_view field_name,
                                               std::string symbol,
                                               int order) const
{
    return restrictToRegionSideIfNeeded(
        builder_->timeDerivative(region_.participant_name,
                                 field_name,
                                 std::move(symbol),
                                 order),
        region_);
}

forms::FormExpr CouplingRegionEndpointView::requiredDt(
    const std::optional<std::string>& field_name,
    std::string symbol,
    int order) const
{
    const auto required_field = requiredFieldName(field_name);
    return dt(std::string_view(required_field), std::move(symbol), order);
}

forms::FormExpr CouplingRegionEndpointView::geometryTerminal(
    CouplingGeometryTerminalQuantity quantity) const
{
    const CouplingGeometryTerminalScope scope{
        .participant_name = region_.participant_name,
        .region = endpoint_,
        .location = CouplingGeometryTerminalLocationDeclaration{
            .region_kind = region_.kind,
            .shared_region_name = endpoint_.shared_region_name,
            .side = region_.side,
        },
    };
    return restrictToRegionSideIfNeeded(builder_->geometryTerminal(quantity, scope),
                                        region_);
}

forms::FormExpr CouplingRegionEndpointView::normal() const
{
    return geometryTerminal(CouplingGeometryTerminalQuantity::Normal);
}

forms::FormExpr CouplingRegionEndpointView::normalComponent(
    const forms::FormExpr& value) const
{
    return forms::inner(value, normal());
}

forms::FormExpr CouplingRegionEndpointView::normalProjection(
    const forms::FormExpr& value) const
{
    const auto n = normal();
    return forms::inner(value, n) * n;
}

forms::FormExpr CouplingRegionEndpointView::tangentialProjection(
    const forms::FormExpr& value) const
{
    return value - normalProjection(value);
}

forms::FormExpr CouplingRegionEndpointView::integral(
    const forms::FormExpr& integrand) const
{
    return builder_->integrate(integrand, region_);
}

CouplingRegionRelationView::CouplingRegionRelationView(
    const CouplingFormBuilder& builder,
    CouplingRegionRelationRequirement requirement)
    : builder_(&builder)
    , requirement_(std::move(requirement))
{
}

std::string_view CouplingRegionRelationView::name() const noexcept
{
    return requirement_.relation_name;
}

const CouplingRegionRelationRequirement&
CouplingRegionRelationView::requirement() const noexcept
{
    return requirement_;
}

CouplingRegionEndpointView CouplingRegionRelationView::endpoint(
    std::string_view participant_name,
    std::string_view region_name) const
{
    const auto it = std::find_if(
        requirement_.endpoints.begin(),
        requirement_.endpoints.end(),
        [&](const CouplingRegionEndpointDeclaration& endpoint) {
            return endpoint.participant_name == participant_name &&
                   (region_name.empty() || endpoint.region_name == region_name);
        });
    FE_THROW_IF(it == requirement_.endpoints.end(), InvalidArgumentException,
                "coupling relation endpoint is not declared");

    const auto resolved_region =
        it->region_name.empty() && it->shared_region_name.has_value()
            ? builder_->sharedRegion(*it->shared_region_name,
                                     it->participant_name)
            : builder_->region(it->participant_name, it->region_name);

    return CouplingRegionEndpointView(*builder_,
                                      requirement_.relation_name,
                                      *it,
                                      resolved_region);
}

forms::FormExpr CouplingRegionRelationView::integral(
    const forms::FormExpr& integrand,
    std::string_view participant_name,
    std::string_view region_name) const
{
    return endpoint(participant_name, region_name).integral(integrand);
}

forms::FormExpr CouplingRegionRelationView::sum(
    std::span<const forms::FormExpr> endpoint_terms) const
{
    FE_THROW_IF(endpoint_terms.empty(), InvalidArgumentException,
                "coupling relation sum requires at least one endpoint term");
    auto total = endpoint_terms.front();
    for (std::size_t i = 1u; i < endpoint_terms.size(); ++i) {
        total = total + endpoint_terms[i];
    }
    return total;
}

forms::FormExpr CouplingFormBuilder::recordTerminal(
    forms::FormExpr expr,
    CouplingFormTerminalProvenanceDeclaration declaration) const
{
    if (auto node = expr.nodeShared()) {
        recorded_terminals_.push_back(RecordedTerminalProvenance{
            .node = std::move(node),
            .declaration = std::move(declaration),
        });
    }
    return expr;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
