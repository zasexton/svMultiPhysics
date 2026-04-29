/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGGEOMETRYREQUIREMENTS_H
#define SVMP_FE_COUPLING_COUPLINGGEOMETRYREQUIREMENTS_H

/**
 * @file CouplingGeometryRequirements.h
 * @brief Coupling-owned declarations for geometry terminal requirements.
 */

#include "Analysis/ProblemAnalysisTypes.h"
#include "Coupling/CouplingContext.h"
#include "Forms/FormExpr.h"

#include <optional>
#include <string>

namespace svmp {
namespace FE {
namespace coupling {

enum class CouplingGeometryTerminalQuantity : std::uint8_t {
    MeshDisplacement,
    Coordinate,
    ReferenceCoordinate,
    CurrentCoordinate,
    PreviousCoordinate,
    ReferencePhysicalCoordinate,
    Jacobian,
    JacobianInverse,
    JacobianDeterminant,
    CurrentJacobian,
    ReferenceJacobian,
    CurrentJacobianDeterminant,
    ReferenceJacobianDeterminant,
    Normal,
    CurrentNormal,
    ReferenceNormal,
    CurrentMeasure,
    ReferenceMeasure,
    SurfaceJacobian,
    CellDiameter,
    CellVolume,
    FacetArea,
    CellDomainId,
};

struct CouplingGeometryTerminalLocationDeclaration {
    CouplingRegionKind region_kind{CouplingRegionKind::Domain};
    std::optional<std::string> shared_region_name;
    CouplingInterfaceSide side{CouplingInterfaceSide::None};
    forms::GeometryConfiguration coordinate_configuration{
        forms::GeometryConfiguration::Reference};
    std::optional<forms::GeometryConfiguration> transform_from_configuration;
    std::optional<forms::GeometryConfiguration> transform_to_configuration;
};

struct CouplingGeometryTerminalLocationProvenance {
    CouplingRegionKind region_kind{CouplingRegionKind::Domain};
    std::optional<std::string> shared_region_name;
    int marker{-1};
    CouplingInterfaceSide side{CouplingInterfaceSide::None};
    forms::GeometryConfiguration coordinate_configuration{
        forms::GeometryConfiguration::Reference};
    std::optional<forms::GeometryConfiguration> transform_from_configuration;
    std::optional<forms::GeometryConfiguration> transform_to_configuration;
    std::uint64_t geometry_revision{0};
    std::uint64_t quadrature_policy_key{0};
};

struct CouplingGeometryTerminalScope {
    std::optional<std::string> participant_name;
    std::optional<CouplingRegionEndpointDeclaration> region;
    std::optional<CouplingGeometryTerminalLocationDeclaration> location;
};

struct CouplingGeometryTerminalRequirement {
    CouplingGeometryTerminalQuantity quantity{CouplingGeometryTerminalQuantity::MeshDisplacement};
    CouplingGeometryTerminalScope scope;
    std::optional<CouplingFieldUse> mesh_motion_field;
    CouplingRequirement requirement{CouplingRequirement::Required};
};

struct CouplingGeometryTerminalOwnerProvenance {
    std::string participant_name;
    std::string system_name;
    std::optional<std::string> region_name;
    std::optional<std::string> shared_region_name;
};

[[nodiscard]] const char* toString(CouplingGeometryTerminalQuantity quantity) noexcept;

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGGEOMETRYREQUIREMENTS_H
