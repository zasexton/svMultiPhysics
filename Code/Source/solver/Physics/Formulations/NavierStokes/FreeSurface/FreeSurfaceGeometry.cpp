/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Constraints/ContactPointVelocityConstraint.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/SymbolicDifferentiation.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/LevelSet/LevelSetCellEvaluator.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
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

[[nodiscard]] FE::forms::FormExpr freeSurfaceCurvatureField(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (bc.curvature_field_name.empty()) {
        return FE::forms::FormExpr{};
    }

    const auto kappa_id = system.findFieldByName(bc.curvature_field_name);
    if (kappa_id == FE::INVALID_FIELD_ID || !system.hasField(bc.curvature_field_name)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free surface references unknown curvature field '" +
            bc.curvature_field_name + "'");
    }

    const auto& rec = system.fieldRecord(kappa_id);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: curvature field '" +
            bc.curvature_field_name + "' must be scalar");
    }

    if (system.fieldParticipatesInUnknownVector(kappa_id)) {
        return FE::forms::StateField(kappa_id, *rec.space, bc.curvature_field_name);
    }
    return FE::forms::FormExpr::discreteField(
        kappa_id, *rec.space, bc.curvature_field_name);
}

[[nodiscard]] bool appendUniqueExtraTrialField(
    FE::systems::FormInstallOptions& install,
    FE::FieldId field)
{
    if (field == FE::INVALID_FIELD_ID) {
        return false;
    }
    if (std::find(install.extra_trial_fields.begin(),
                  install.extra_trial_fields.end(),
                  field) != install.extra_trial_fields.end()) {
        return false;
    }
    install.extra_trial_fields.push_back(field);
    return true;
}

[[nodiscard]] bool unfittedFreeSurfaceNeedsLevelSetTrialFieldForNavierStokes(
    const FreeSurfaceBoundary& bc) noexcept
{
    if (!isUnfittedLevelSet(bc)) {
        return false;
    }
    // The legacy curvature-traction contact law depends explicitly on
    // n(phi).  Generated-geometry forms instead consume refreshed-frozen
    // interface, contact, and sharp wall geometry from the authoritative cut
    // snapshot; they must not advertise a nonexistent inner-Newton tangent.
    const bool has_dynamic_contact = std::any_of(
            bc.contact_lines.begin(),
            bc.contact_lines.end(),
            [](const FreeSurfaceContactLine& contact_line) {
                return contactLineKind(contact_line) ==
                       ContactLineKind::DynamicRenE;
            });
    return has_dynamic_contact && !usesGeneratedInterfaceNormal(bc);
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
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
        return "not_applicable";
    }
    if (usesSurfaceStress(bc)) {
        return isUnfittedLevelSet(bc)
                   ? "generated_geometry_refreshed_frozen"
                   : "fitted_geometry_surface_variation";
    }
    if (usesKinematicAreaGradientTraction(bc)) {
        return "kinematic_area_gradient_field_picard_generated_geometry_frozen";
    }
    if (!bc.curvature_field_name.empty()) {
        const auto kappa_id = system.findFieldByName(bc.curvature_field_name);
        if (kappa_id != FE::INVALID_FIELD_ID &&
            system.fieldParticipatesInUnknownVector(kappa_id)) {
            return "curvature_field_trial_coupled_geometry_curvature_frozen";
        }
        return "curvature_field_picard_frozen";
    }
    if (isUnfittedLevelSet(bc)) {
        return "picard_frozen_or_guarded";
    }
    return "supplied_scalar_frozen";
}

[[nodiscard]] FE::forms::FormExpr unfittedInterfaceNormal(
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& phi)
{
    auto n = FE::forms::unitNormalFromLevelSet(phi);
    if (bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive) {
        return -n;
    }
    return n;
}

[[nodiscard]] FE::forms::FormExpr integrateOnFreeSurface(
    const FE::forms::FormExpr& integrand,
    const FreeSurfaceBoundary& bc,
    bool ale_enabled)
{
    if (isUnfittedLevelSet(bc)) {
        return integrand.dI(FE::forms::InterfaceDomain(bc.interface_marker));
    }
    // Fitted boundaries integrate over the assembly (active) frame.  The
    // boundary quadrature weights already carry the active surface measure,
    // so the current measure must not be applied a second time.  With ALE
    // the active frame is the current configuration: the coupled
    // mesh-displacement transaction (FE ALEBinding) moves the active
    // coordinates at every trial state, and fitted SurfaceStress and
    // MeshNitsche require that coupling.  The current normal used by the
    // fitted terms is then the normal of the integration frame as well.
    (void)ale_enabled;
    return integrand.ds(
        FE::forms::BoundaryDomain::fullPhysical(bc.boundary_marker));
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

[[nodiscard]] FE::forms::FormExpr freeSurfaceLevelSet(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        return FE::forms::FormExpr{};
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    const auto& rec = system.fieldRecord(phi_id);
    if (system.fieldParticipatesInUnknownVector(phi_id)) {
        return FE::forms::StateField(phi_id, *rec.space, bc.level_set_field_name);
    }
    return FE::forms::FormExpr::discreteField(phi_id, *rec.space, bc.level_set_field_name);
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

[[nodiscard]] FE::forms::FormExpr unfittedTractionCurvature(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    const FE::forms::FormExpr& phi)
{
    FE::forms::FormExpr curvature;
    if (!bc.curvature_field_name.empty()) {
        curvature = freeSurfaceCurvatureField(bc, system);
    } else if (bc.use_level_set_curvature) {
        curvature = FE::forms::meanCurvatureFromLevelSet(phi);
    } else {
        curvature = FE::forms::bc::toScalarExpr(
            bc.curvature,
            freeSurfaceValueName("ns_free_surface_curvature", bc));
    }

    if (bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive) {
        return FE::forms::FormExpr::constant(FE::Real{-1.0}) * curvature;
    }
    return curvature;
}

void appendUnfittedFreeSurfaceLevelSetTrialFields(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    const FE::systems::FESystem& system,
    FE::systems::FormInstallOptions& install)
{
    for (const auto& bc : free_surfaces) {
        if (!unfittedFreeSurfaceNeedsLevelSetTrialFieldForNavierStokes(bc)) {
            if (isUnfittedLevelSet(bc)) {
                std::ostringstream oss;
                oss << "IncompressibleNavierStokesVMSModule: not adding "
                    << "unfitted free-surface level-set field as Navier-Stokes "
                    << "residual extra trial "
                    << "marker=" << bc.interface_marker
                    << " level_set_field='" << bc.level_set_field_name << "'"
                    << " Active_domain=" << activeDomainName(bc.active_domain)
                    << " Active_domain_method="
                    << activeDomainMethodName(bc.active_domain_method)
                    << " diagnostic=unfitted_free_surface_phi_extra_trial_omitted"
                    << " reason=no_explicit_residual_level_set_dependence";
                FE_LOG_INFO(oss.str());
            }
            continue;
        }
        const auto phi_id = resolveLevelSetFieldId(bc, system);
        const auto& rec = system.fieldRecord(phi_id);
        if (!system.fieldParticipatesInUnknownVector(phi_id)) {
            continue;
        }
        if (!appendUniqueExtraTrialField(install, phi_id)) {
            continue;
        }

        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: adding unfitted free-surface "
            << "level-set field as Navier-Stokes extra trial "
            << "marker=" << bc.interface_marker
            << " level_set_field='" << rec.name << "'"
            << " level_set_source_kind=" << fieldSourceKindName(rec.source_kind)
            << " Active_domain=" << activeDomainName(bc.active_domain)
            << " Active_domain_method="
            << activeDomainMethodName(bc.active_domain_method)
            << " diagnostic=unfitted_free_surface_phi_extra_trial";
        FE_LOG_INFO(oss.str());
    }
}

void appendUnfittedFreeSurfaceCurvatureTrialFields(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    const FE::systems::FESystem& system,
    FE::systems::FormInstallOptions& install)
{
    for (const auto& bc : free_surfaces) {
        if (!isUnfittedLevelSet(bc) ||
            usesSurfaceStress(bc) ||
            bc.curvature_field_name.empty() ||
            FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
            continue;
        }

        const auto kappa_id = system.findFieldByName(bc.curvature_field_name);
        if (kappa_id == FE::INVALID_FIELD_ID ||
            !system.fieldParticipatesInUnknownVector(kappa_id)) {
            continue;
        }

        const auto& rec = system.fieldRecord(kappa_id);
        if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: curvature field '" +
                bc.curvature_field_name + "' must be scalar");
        }

        const bool appended = appendUniqueExtraTrialField(install, kappa_id);
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: "
            << (appended ? "added" : "retained")
            << " unfitted free-surface curvature field as Navier-Stokes trial field"
            << " marker=" << bc.interface_marker
            << " curvature_field='" << bc.curvature_field_name << "'"
            << " field_id=" << kappa_id
            << " source_kind=" << fieldSourceKindName(rec.source_kind)
            << " diagnostic=unfitted_free_surface_curvature_trial_field";
        FE_LOG_INFO(oss.str());
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
