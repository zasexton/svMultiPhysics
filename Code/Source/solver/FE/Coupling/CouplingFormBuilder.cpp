/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingFormBuilder.h"

#include "Core/FEException.h"

#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

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
    return forms::FormExpr::previousSolution(steps_back);
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
    static_cast<void>(scope);
    return forms::meshDisplacement();
}

forms::FormExpr CouplingFormBuilder::meshVelocity(
    const CouplingGeometryTerminalScope& scope) const
{
    static_cast<void>(scope);
    return forms::meshVelocity();
}

forms::FormExpr CouplingFormBuilder::meshAcceleration(
    const CouplingGeometryTerminalScope& scope) const
{
    static_cast<void>(scope);
    return forms::meshAcceleration();
}

forms::FormExpr CouplingFormBuilder::previousMeshVelocity(
    const CouplingGeometryTerminalScope& scope) const
{
    static_cast<void>(scope);
    return forms::previousMeshVelocity();
}

forms::FormExpr CouplingFormBuilder::predictedMeshVelocity(
    const CouplingGeometryTerminalScope& scope) const
{
    static_cast<void>(scope);
    return forms::predictedMeshVelocity();
}

forms::FormExpr CouplingFormBuilder::geometryTerminal(
    CouplingGeometryTerminalQuantity quantity,
    const CouplingGeometryTerminalScope& scope) const
{
    switch (quantity) {
    case CouplingGeometryTerminalQuantity::MeshDisplacement:
        return meshDisplacement(scope);
    case CouplingGeometryTerminalQuantity::Coordinate:
        return forms::x();
    case CouplingGeometryTerminalQuantity::ReferenceCoordinate:
        return forms::X();
    case CouplingGeometryTerminalQuantity::CurrentCoordinate:
        return forms::currentCoordinate();
    case CouplingGeometryTerminalQuantity::PreviousCoordinate:
        return forms::previousCoordinate();
    case CouplingGeometryTerminalQuantity::ReferencePhysicalCoordinate:
        return forms::referenceCoordinatePhysical();
    case CouplingGeometryTerminalQuantity::Jacobian:
        return forms::J();
    case CouplingGeometryTerminalQuantity::JacobianInverse:
        return forms::Jinv();
    case CouplingGeometryTerminalQuantity::JacobianDeterminant:
        return forms::detJ();
    case CouplingGeometryTerminalQuantity::CurrentJacobian:
        return forms::currentJacobian();
    case CouplingGeometryTerminalQuantity::ReferenceJacobian:
        return forms::referenceJacobian();
    case CouplingGeometryTerminalQuantity::CurrentJacobianDeterminant:
        return forms::currentJacobianDeterminant();
    case CouplingGeometryTerminalQuantity::ReferenceJacobianDeterminant:
        return forms::referenceJacobianDeterminant();
    case CouplingGeometryTerminalQuantity::Normal:
        return forms::FormExpr::normal();
    case CouplingGeometryTerminalQuantity::CurrentNormal:
        return forms::currentNormal();
    case CouplingGeometryTerminalQuantity::ReferenceNormal:
        return forms::referenceNormal();
    case CouplingGeometryTerminalQuantity::CurrentMeasure:
        return forms::currentMeasure();
    case CouplingGeometryTerminalQuantity::ReferenceMeasure:
        return forms::referenceMeasure();
    case CouplingGeometryTerminalQuantity::SurfaceJacobian:
        return forms::surfaceJacobian();
    case CouplingGeometryTerminalQuantity::CellDiameter:
        return forms::h();
    case CouplingGeometryTerminalQuantity::CellVolume:
        return forms::vol();
    case CouplingGeometryTerminalQuantity::FacetArea:
        return forms::area();
    case CouplingGeometryTerminalQuantity::CellDomainId:
        return forms::domainId();
    }
    FE_THROW(InvalidArgumentException,
             "unsupported coupling geometry terminal quantity");
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

} // namespace coupling
} // namespace FE
} // namespace svmp
