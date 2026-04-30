/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingFormBuilder.h"

#include "Core/FEException.h"
#include "Systems/FESystem.h"

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

forms::FormExpr CouplingFormBuilder::test(std::string_view participant_name,
                                          std::string_view field_name,
                                          std::string symbol) const
{
    const auto ref = context().field(participant_name, field_name);
    FE_THROW_IF(ref.space == nullptr, InvalidArgumentException,
                "coupling field has no function space");
    return forms::TestField(ref.field_id, *ref.space, std::move(symbol));
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

forms::FormExpr CouplingInterfaceSideView::test(std::string_view field_name,
                                                std::string symbol) const
{
    return restrictToInterfaceSide(
        builder_->test(region_.participant_name, field_name, std::move(symbol)),
        region_.side);
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
