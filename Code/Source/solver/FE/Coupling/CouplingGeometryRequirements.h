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
#include <span>
#include <string>
#include <vector>

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
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::optional<svmp::search::LogicalInterfaceRegionId> logical_region;
#endif
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

struct CouplingGeometryTerminalAvailability {
    std::vector<CouplingGeometryTerminalQuantity> supported_quantities;
    std::vector<analysis::DomainKind> supported_domains;
    bool supports_reference_configuration{true};
    bool supports_current_configuration{true};
};

struct CouplingGeometryTerminalRequirementSummary {
    std::vector<CouplingGeometryTerminalQuantity> quantities;
    std::vector<analysis::DomainKind> domains;
    bool requires_reference_configuration{false};
    bool requires_current_configuration{false};
};

[[nodiscard]] const char* toString(CouplingGeometryTerminalQuantity quantity) noexcept;
[[nodiscard]] std::optional<analysis::DomainKind> toAnalysisDomainKind(
    CouplingRegionKind kind) noexcept;
[[nodiscard]] CouplingGeometryTerminalRequirementSummary
summarizeGeometryTerminalRequirements(
    const CouplingContext& context,
    std::span<const CouplingGeometryTerminalRequirement> requirements);
[[nodiscard]] CouplingValidationResult validateGeometryTerminalRequirements(
    const CouplingContext& context,
    std::span<const CouplingGeometryTerminalRequirement> requirements,
    const CouplingGeometryTerminalAvailability& availability);

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGGEOMETRYREQUIREMENTS_H
