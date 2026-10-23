/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"

#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/FESystem.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <functional>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

namespace {

[[nodiscard]] int generatedInterfaceMarkerFor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || bc.interface_marker >= 0) {
        return bc.interface_marker;
    }
    if (bc.generated_interface_domain_id.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated unfitted free surface requires a non-empty generated_interface_domain_id");
    }

    return freeSurfaceLevelSetGeometry(bc, system).interfaceMarker();
}

[[nodiscard]] FE::interfaces::GeneratedActiveBoundaryMarkerKey
generatedActiveBoundaryMarkerKey(
    const FreeSurfaceBoundary& bc,
    int physical_boundary_marker,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || bc.interface_marker < 0 ||
        physical_boundary_marker < 0 ||
        bc.active_domain == FreeSurfaceActiveDomain::None) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: sharp active-boundary marker requires a resolved unfitted active domain and physical boundary marker");
    }
    return freeSurfaceLevelSetGeometry(bc, system)
        .side(activeDomainSide(bc.active_domain))
        .boundaryKey(physical_boundary_marker);
}

} // namespace

[[nodiscard]] std::optional<ActivePressureDomain>
activePressureDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
{
    std::optional<ActivePressureDomain> active_domain;
    for (const auto& bc : free_surfaces) {
        if (bc.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        if (bc.implementation != FreeSurfaceImplementation::UnfittedLevelSet) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain pressure operations are only valid for unfitted level-set free surfaces");
        }
        if (bc.level_set_field_name.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain pressure operations require a non-empty level_set_field_name");
        }
        if (active_domain.has_value()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: at most one active-domain free surface may restrict pressure operations");
        }
        active_domain = ActivePressureDomain{&bc, bc.active_domain};
    }
    return active_domain;
}

/**
 * Normal from the same geometric rule that supplies dI/dInterfaceBoundary.
 * Generated interface normals point from phi<0 to phi>0.  Flip only where a
 * directed outward-liquid normal is required; the surface projector itself
 * is orientation independent.
 */
[[nodiscard]] FE::forms::FormExpr generatedInterfaceOutwardNormal(
    const FreeSurfaceBoundary& bc)
{
    if (isUnfittedLevelSet(bc)) {
        return FE::level_set::outwardInterfaceNormal(
            activeDomainSide(bc.active_domain));
    }
    return FE::forms::FormExpr::normal();
}

[[nodiscard]] FE::FieldId resolveLevelSetFieldId(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    const auto phi_id = system.findFieldByName(bc.level_set_field_name);
    if (phi_id == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unfitted free surface references unknown level-set field '" +
            bc.level_set_field_name + "'");
    }

    const auto& rec = system.fieldRecord(phi_id);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: level-set field '" +
            bc.level_set_field_name + "' must be scalar");
    }
    return phi_id;
}

/// The level-set geometry of an unfitted free surface: its field, generated
/// domain id, isovalue and (resolved or requested) interface marker.
[[nodiscard]] FE::level_set::LevelSetGeometry freeSurfaceLevelSetGeometry(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    return FE::level_set::LevelSetGeometry::bind(
        system,
        {.level_set_field = resolveLevelSetFieldId(bc, system),
         .domain_id = bc.generated_interface_domain_id,
         .isovalue = bc.level_set_isovalue,
         .interface_marker = bc.interface_marker});
}

[[nodiscard]] FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey
generatedUnfittedContactLineMarkerKey(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine&
        contact_line,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated contact marker key requires an unfitted free surface");
    }
    if (bc.interface_marker < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated unfitted contact-line marker requires resolved interface_marker >= 0");
    }
    if (bc.generated_interface_domain_id.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated unfitted contact-line marker requires a non-empty generated_interface_domain_id");
    }
    if (contactLineWallBoundaryMarker(contact_line) < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angle requires Contact_line_wall_marker or Contact_line_wall_face");
    }

    return freeSurfaceLevelSetGeometry(bc, system).contactLineKey(
        contactLineWallBoundaryMarker(contact_line),
        contactLineMarker(contact_line));
}

[[nodiscard]] int generatedUnfittedContactLineMarker(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine&
        contact_line,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        return contactLineConstraintMarker(contact_line);
    }
    return FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
        generatedUnfittedContactLineMarkerKey(contact_line, bc, system));
}

[[nodiscard]] int generatedActiveBoundaryMarker(
    const FreeSurfaceBoundary& bc,
    int physical_boundary_marker,
    const FE::systems::FESystem& system)
{
    return FE::interfaces::stableGeneratedActiveBoundaryMarker(
        generatedActiveBoundaryMarkerKey(
            bc, physical_boundary_marker, system));
}

[[nodiscard]] std::optional<int> sharpActiveBoundaryMarkerFor(
    int physical_boundary_marker,
    std::span<const FreeSurfaceBoundary> boundaries,
    const FE::systems::FESystem& system)
{
    std::optional<int> marker;
    std::string owner_key;
    for (const auto& bc : boundaries) {
        if (!isUnfittedLevelSet(bc) ||
            bc.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        const auto key = generatedActiveBoundaryMarkerKey(
            bc, physical_boundary_marker, system);
        const int candidate =
            FE::interfaces::stableGeneratedActiveBoundaryMarker(key);
        if (!marker.has_value()) {
            marker = candidate;
            owner_key = key.stableKey();
            continue;
        }
        if (*marker != candidate || owner_key != key.stableKey()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: physical boundary marker " +
                std::to_string(physical_boundary_marker) +
                " is covered by multiple ambiguous unfitted active domains; select one explicit active-domain owner");
        }
    }
    return marker;
}

[[nodiscard]] FreeSurfaceBoundary withResolvedInterfaceMarker(
    FreeSurfaceBoundary bc,
    const FE::systems::FESystem& system)
{
    if (isUnfittedLevelSet(bc) && bc.interface_marker < 0) {
        bc.interface_marker = generatedInterfaceMarkerFor(bc, system);
    }
    return bc;
}

[[nodiscard]] std::string freeSurfaceCurvatureTangentPolicyName(
    const FreeSurfaceBoundary& bc)
{
    if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
        return "not_applicable";
    }
    // SurfaceStress, the only form (decision D48).
    return isUnfittedLevelSet(bc) ? "generated_geometry_refreshed_frozen"
                                  : "fitted_geometry_surface_variation";
}

[[nodiscard]] std::optional<ActiveVolumeDomain> activeVolumeDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
{
    std::optional<ActiveVolumeDomain> active_domain;
    for (const auto& bc : free_surfaces) {
        if (bc.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        if (active_domain.has_value()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: at most one active-domain free surface may restrict Navier-Stokes volume integration");
        }

        const FE::forms::CutVolumeSide side = activeDomainSide(bc.active_domain);
        active_domain = ActiveVolumeDomain{
            bc.interface_marker,
            side,
            bc.active_domain_method};

        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: active-domain free surface "
            << "marker=" << bc.interface_marker
            << " level_set_field='" << bc.level_set_field_name << "'"
            << " isovalue=" << bc.level_set_isovalue
            << " generated_interface_domain_id='"
            << bc.generated_interface_domain_id << "'"
            << " Active_domain=" << activeDomainName(bc.active_domain)
            << " Active_domain_method="
            << activeDomainMethodName(bc.active_domain_method)
            << " side=" << cutVolumeSideName(side);
        FE_LOG_INFO(oss.str());
    }
    return active_domain;
}

[[nodiscard]] FE::forms::FormExpr integrateOnActiveVolume(
    const FE::forms::FormExpr& integrand,
    const std::optional<ActiveVolumeDomain>& active_domain)
{
    if (!active_domain.has_value()) {
        return integrand.dx(FE::forms::VolumeDomain::full());
    }
    return integrand.dx(FE::forms::VolumeDomain::cut(
        active_domain->interface_marker, active_domain->side));
}

// u_ref of the semi-implicit capillary term: prescribed data in the velocity
// space, registered by registerOn and refreshed by the application.
[[nodiscard]] FE::forms::FormExpr freeSurfaceSemiImplicitReferenceVelocity(
    const FE::systems::FESystem& system)
{
    const std::string name(kFreeSurfaceSemiImplicitReferenceVelocityFieldName);
    const auto field = system.findFieldByName(name);
    if (field == FE::INVALID_FIELD_ID) {
        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule: the semi-implicit "
            "reference velocity field '" + name + "' is not registered");
    }
    const auto& rec = system.fieldRecord(field);
    if (rec.source_kind != FE::systems::FieldSourceKind::PrescribedData ||
        system.fieldParticipatesInUnknownVector(field) || !rec.space) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: the semi-implicit "
            "reference velocity field '" + name +
            "' must be prescribed data, not an unknown");
    }
    return FE::forms::FormExpr::discreteField(field, *rec.space, name);
}

FE::forms::FormExpr freeSurfaceScalar(
    const FE::forms::bc::ScalarValue& value,
    std::string_view name,
    const FreeSurfaceBoundary& bc)
{
    return FE::forms::bc::toScalarExpr(value, freeSurfaceValueName(name, bc));
}

std::function<std::optional<int>(int)> wettedBoundaryResolver(
    FE::systems::FESystem& system,
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
{
    return [&system, &free_surfaces](int marker) -> std::optional<int> {
        const auto generated = sharpActiveBoundaryMarkerFor(marker, free_surfaces, system);
        if (generated.has_value()) {
            system.registerGeneratedEmbeddedInterfaceMarker(*generated);
        }
        return generated;
    };
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
