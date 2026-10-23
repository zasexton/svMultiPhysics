/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Options/NavierStokesOptionRules.h"
#include "Physics/Options/NavierStokesOptionSchema.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/Interfaces/ContactLineGeometryCertificate.h"
#include "FE/Systems/SystemCollectives.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

namespace {

constexpr std::string_view kOwner = "IncompressibleNavierStokesVMSModule";

/// The geometry check of an unfitted contact line on its wall.
[[nodiscard]] FE::interfaces::ContactLineGeometryCheck contactLineCheck(
    const FreeSurfaceContactLine& contact_line,
    int contact_marker)
{
    return FE::interfaces::ContactLineGeometryCheck{
        .contact_marker = contact_marker,
        .wall_marker = contactLineWallBoundaryMarker(contact_line),
        .wall_normal = normalizedWallNormal(contact_line),
        .owner = std::string(kOwner),
    };
}

[[nodiscard]] int axisAlignedWallNormalAxis(
    const FreeSurfaceContactLine& contact_line,
    int dim)
{
    const auto normal = normalizedWallNormal(contact_line);
    const auto law = contactWallLawName(contact_line);
    constexpr FE::Real tolerance = FE::Real{1.0e-12};
    int axis = -1;
    for (int component = 0; component < 3; ++component) {
        const auto magnitude =
            std::abs(normal[static_cast<std::size_t>(component)]);
        if (component >= dim) {
            if (magnitude > tolerance) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " wall_normal must lie in the velocity space dimension");
            }
            continue;
        }
        if (std::abs(magnitude - FE::Real{1.0}) <= tolerance) {
            if (axis >= 0) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " wall_normal must have exactly one axis-aligned component");
            }
            axis = component;
        } else if (magnitude > tolerance) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " + law +
                " currently requires an axis-aligned wall_normal because general linear-combination normal constraints are unavailable");
        }
    }
    if (axis < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently requires an axis-aligned wall_normal");
    }
    return axis;
}

} // namespace

void validateDynamicContactPlanarWallMarker(
    const FE::systems::FESystem& system,
    const FreeSurfaceContactLine& contact_line)
{
    constexpr FE::Real alignment_tolerance = FE::interfaces::kContactLineAlignmentTolerance;
    const auto configured = normalizedWallNormal(contact_line);
    const auto evidence = FE::interfaces::planarWallEvidence(
        system,
        contactLineWallBoundaryMarker(contact_line),
        configured);

    if (evidence.face_count == 0u || evidence.invalid_face_count != 0u) {
        std::ostringstream message;
        message
            << "IncompressibleNavierStokesVMSModule: "
            << contactWallLawName(contact_line)
            << " requires every face on Contact_line_wall_marker to be a "
               "first-order planar face with the configured physical "
               "outward wall_normal"
            << " wall_boundary_marker="
            << contactLineWallBoundaryMarker(contact_line)
            << " global_owned_faces=" << evidence.face_count
            << " global_invalid_faces=" << evidence.invalid_face_count
            << " minimum_alignment=" << evidence.minimum_alignment
            << " required_alignment>="
            << FE::Real{1.0} - alignment_tolerance
            << " configured_normal=(" << configured[0] << ','
            << configured[1] << ',' << configured[2] << ')';
        if (!evidence.first_failure.empty()) {
            message << " local_first_failure='" << evidence.first_failure
                    << '\'';
        }
        throw std::invalid_argument(message.str());
    }

    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: validated complete DynamicContactAngle planar wall marker") +
        " wall_boundary_marker=" +
        std::to_string(contactLineWallBoundaryMarker(contact_line)) +
        " global_owned_faces=" + std::to_string(evidence.face_count) +
        " minimum_alignment=" +
        std::to_string(evidence.minimum_alignment) +
        " alignment_tolerance=1e-8" +
        " geometry_contract=first_order_planar_current_physical" +
        " diagnostic=dynamic_contact_complete_wall_marker_validation");
}

void validateFreeSurfaceContactGeometryPreflight(
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system)
{
    const auto* context = system.cutIntegrationContext();
    if (context == nullptr) {
        return;
    }
    for (const auto& boundary : free_surfaces) {
        if (!isUnfittedLevelSet(boundary)) {
            continue;
        }
        for (const auto& contact_line : boundary.contact_lines) {
            const auto kind = contactLineKind(contact_line);
            if (kind != ContactLineKind::PrescribedAngle &&
                kind != ContactLineKind::DynamicRenE) {
                continue;
            }
            static_cast<void>(
                FE::interfaces::certifyContactLineRulesCollectively(
                    system,
                    context,
                    contactLineCheck(
                        contact_line,
                        generatedUnfittedContactLineMarker(
                            contact_line, boundary, system)),
                    true,
                    "free_surface_contact_geometry_preflight"));
        }
    }
}

void validateDynamicContactWallEssentialBC(
    const IncompressibleNavierStokesVMSOptions& options,
    const FreeSurfaceContactLine& contact_line,
    int dim)
{
    const int normal_axis = axisAlignedWallNormalAxis(contact_line, dim);
    const auto law = contactWallLawName(contact_line);
    bool normal_is_constrained = false;
    bool saw_wall_bc = false;
    for (const auto& dirichlet : options.velocity_dirichlet) {
        if (dirichlet.boundary_marker != contactLineWallBoundaryMarker(contact_line)) {
            continue;
        }
        saw_wall_bc = true;
        for (int component = 0; component < dim; ++component) {
            if (!dirichlet.active_components[
                    static_cast<std::size_t>(component)]) {
                continue;
            }
            if (!FE::forms::bc::isZeroConstantScalarValue(
                    dirichlet.value[static_cast<std::size_t>(component)])) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " requires a stationary zero-valued normal wall essential condition");
            }
            if (component == normal_axis) {
                normal_is_constrained = true;
            } else {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " rejects tangential/full no-slip Dirichlet constraints on its Navier-slip wall");
            }
        }
    }
    for (const auto& dirichlet : options.velocity_dirichlet_weak) {
        if (dirichlet.boundary_marker == contactLineWallBoundaryMarker(contact_line)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " + law +
                " requires a strong zero normal-only wall essential condition; weak velocity Dirichlet data on the wall are unsupported");
        }
    }
    if (!saw_wall_bc || !normal_is_constrained) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + law +
            " requires an axis-aligned zero normal-only velocity Dirichlet condition on Contact_line_wall_marker");
    }
}

void registerFreeSurfacePrescribedAngleGeometry(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc)
{
    for (const auto& contact_line : bc.contact_lines) {
        if (contactLineKind(contact_line) != ContactLineKind::PrescribedAngle) {
            continue;
        }
        if (isUnfittedLevelSet(bc)) {
            const auto phi_id = system.findFieldByName(bc.level_set_field_name);
            if (phi_id == FE::INVALID_FIELD_ID) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angle references unknown level-set field '" +
                    bc.level_set_field_name + "'");
            }
            const auto& rec = system.fieldRecord(phi_id);
            if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: level-set field '" +
                    bc.level_set_field_name + "' must be scalar");
            }
            if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require the level-set field to be an unknown");
            }

            const int contact_marker =
                generatedUnfittedContactLineMarker(
                    contact_line,
                    bc,
                    system);
            system.registerGeneratedEmbeddedInterfaceMarker(contact_marker);
            const auto check = contactLineCheck(contact_line, contact_marker);
            system.addCutIntegrationContextUpdateCallback(
                FE::systems::CutIntegrationContextUpdateCallback{
                    .name = "navier_stokes_prescribed_contact_geometry:" +
                            std::to_string(contact_marker),
                    .callback =
                        [&system, check](
                            const FE::assembly::CutIntegrationContext*
                                cut_context) {
                            static_cast<void>(
                                FE::interfaces::certifyContactLineRulesCollectively(
                                    system,
                                    cut_context,
                                    check,
                                    cut_context != nullptr,
                                    "prescribed_contact_geometry"));
                        }});
            const auto* current_cut_context =
                system.cutIntegrationContext();
            static_cast<void>(FE::interfaces::certifyContactLineRulesCollectively(
                system,
                current_cut_context,
                check,
                true,
                "prescribed_contact_geometry_initial"));
            FE_LOG_INFO(
                std::string("IncompressibleNavierStokesVMSModule: registered prescribed-angle contact geometry without a level-set residual") +
                " interface_marker=" +
                std::to_string(bc.interface_marker) +
                " contact_marker=" + std::to_string(contact_marker) +
                " level_set_geometry_owner=none_angle_preserving_wall_maintenance"
                " momentum_owner=young_wall_energy"
                " literal_codimension_two_level_set_residual=retired"
                " diagnostic=navier_stokes_prescribed_contact_geometry");
            continue;
        }

        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule: fitted prescribed contact-angle validation did not fail closed");
    }
}

void registerDynamicContactWallNormalCheck(
    FE::systems::FESystem& system,
    const FreeSurfaceContactLine& contact_line,
    int contact_marker)
{
    const auto check = contactLineCheck(contact_line, contact_marker);
    system.addCutIntegrationContextUpdateCallback(
        FE::systems::CutIntegrationContextUpdateCallback{
            .name = "navier_stokes_dynamic_contact_wall_normal:" +
                    std::to_string(contact_marker),
            .callback =
                [&system, check](
                    const FE::assembly::CutIntegrationContext*
                        cut_context) {
                    static_cast<void>(
                        FE::interfaces::certifyContactLineRulesCollectively(
                            system,
                            cut_context,
                            check,
                            cut_context != nullptr &&
                                system.isSetup(),
                            "dynamic_contact_operator_angle"));
                }});
    const auto* current_cut_context =
        system.cutIntegrationContext();
    static_cast<void>(FE::interfaces::certifyContactLineRulesCollectively(
        system,
        current_cut_context,
        check,
        current_cut_context != nullptr && system.isSetup(),
        "dynamic_contact_operator_angle_initial"));
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
