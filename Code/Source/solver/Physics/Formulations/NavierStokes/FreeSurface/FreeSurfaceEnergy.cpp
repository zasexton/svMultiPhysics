/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceCollectives.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceValidation.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesInitialConditions.h"

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

[[nodiscard]] bool shouldDeclareFreeSurfaceDiscreteFunctional(
    const FreeSurfaceBoundary& boundary)
{
    return isExteriorOnePhaseBoundary(boundary) &&
           isUnfittedLevelSet(boundary) &&
           boundary.active_domain != FreeSurfaceActiveDomain::None &&
           (usesSurfaceStress(boundary) ||
            usesKinematicAreaGradientTraction(boundary) ||
            hasWallAwareContactLaw(boundary));
}

} // namespace

void validateFreeSurfaceDiscreteFunctionalPreflight(
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system)
{
    const bool has_pending_declaration = std::any_of(
        free_surfaces.begin(),
        free_surfaces.end(),
        shouldDeclareFreeSurfaceDiscreteFunctional);
    if (!has_pending_declaration) {
        return;
    }
    if (!system.freeSurfaceDiscreteFunctionalHistory().empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free-surface discrete-functional declarations cannot be installed after accepted history has begun");
    }

    const auto existing =
        system.freeSurfaceDiscreteFunctionalDeclarations();
    for (const auto& boundary : free_surfaces) {
        if (!shouldDeclareFreeSurfaceDiscreteFunctional(boundary)) {
            continue;
        }
        const auto conflict = std::find_if(
            existing.begin(),
            existing.end(),
            [&](const auto& declaration) {
                return declaration.interface_marker ==
                       boundary.interface_marker;
            });
        if (conflict != existing.end()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: free-surface interface marker " +
                std::to_string(boundary.interface_marker) +
                " already has discrete-functional owner '" +
                conflict->owner_component + "'");
        }
    }
}

void declareFreeSurfaceDiscreteFunctionals(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    FE::systems::FESystem& system,
    FE::FieldId velocity_field,
    FE::Real density,
    const std::array<FE::Real, 3>& gravitational_acceleration,
    bool static_conservative_body_force_complete,
    FE::Real dynamic_viscosity,
    bool has_constant_dynamic_viscosity)
{
    for (const auto& bc : free_surfaces) {
        if (!shouldDeclareFreeSurfaceDiscreteFunctional(bc)) {
            continue;
        }
        FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
        parameters.liquid_side =
            bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive
                ? FE::geometry::CutIntegrationSide::Positive
                : FE::geometry::CutIntegrationSide::Negative;
        parameters.surface_tension = constantScalarValueOrThrow(
            bc.surface_tension,
            "free-surface discrete-functional surface tension");
        // The pressure unknown is spatially varying and is not the scalar
        // volume multiplier in the capillary functional.  Preserve V_h as a
        // separately reported measure until a scalar multiplier is declared.
        parameters.volume_multiplier = FE::Real{0.0};
        for (const auto& contact_line : bc.contact_lines) {
            const auto kind = contactLineKind(contact_line);
            if (kind != ContactLineKind::PrescribedAngle &&
                kind != ContactLineKind::DynamicRenE) {
                continue;
            }
            parameters.young_wall_coefficients.push_back(
                FE::interfaces::FreeSurfaceYoungWallCoefficient{
                    .boundary_marker =
                        contactLineWallBoundaryMarker(contact_line),
                    .equilibrium_contact_angle_radians =
                        constantScalarValueOrThrow(
                            contactLineAngleRadians(contact_line),
                            "free-surface discrete-functional equilibrium "
                            "contact angle"),
                });
            if (kind == ContactLineKind::DynamicRenE ||
                contactLineHasNavierSlip(contact_line)) {
                const bool dynamic_law =
                    kind == ContactLineKind::DynamicRenE;
                parameters.dynamic_contact_coefficients.push_back(
                    FE::interfaces::FreeSurfaceDynamicContactCoefficient{
                        .boundary_marker =
                            contactLineWallBoundaryMarker(contact_line),
                        .equilibrium_contact_angle_radians =
                            constantScalarValueOrThrow(
                                contactLineAngleRadians(contact_line),
                                "free-surface discrete-functional dynamic "
                                "contact equilibrium angle"),
                        .law = dynamic_law
                            ? FE::interfaces::FreeSurfaceContactLaw::
                                  DynamicRenE
                            : FE::interfaces::FreeSurfaceContactLaw::
                                  PrescribedAngle,
                        .mobility = dynamic_law
                            ? constantScalarValueOrThrow(
                                  contactLineMobility(contact_line),
                                  "free-surface discrete-functional dynamic "
                                  "contact mobility")
                            : FE::Real{0.0},
                        .slip_length = constantScalarValueOrThrow(
                            contactLineSlipLength(contact_line),
                            "free-surface discrete-functional Navier slip "
                            "length"),
                        .dynamic_viscosity = dynamic_viscosity,
                    });
            }
        }
        std::optional<
            FE::interfaces::FreeSurfaceActiveVolumeDissipationParameters>
            active_volume_dissipation_parameters;
        if (has_constant_dynamic_viscosity) {
            active_volume_dissipation_parameters =
                FE::interfaces::
                    FreeSurfaceActiveVolumeDissipationParameters{
                        .liquid_side = parameters.liquid_side,
                        .dynamic_viscosity = dynamic_viscosity,
                    };
        }
        std::optional<
            FE::interfaces::FreeSurfaceExternalPressurePowerParameters>
            external_pressure_power_parameters;
        if (const auto* external_pressure =
                std::get_if<FE::Real>(&bc.external_pressure);
            external_pressure != nullptr) {
            external_pressure_power_parameters =
                FE::interfaces::
                    FreeSurfaceExternalPressurePowerParameters{
                        .liquid_side = parameters.liquid_side,
                        .external_pressure = *external_pressure,
                    };
        }
        FE::FieldId curvature_field = FE::INVALID_FIELD_ID;
        if (usesKinematicAreaGradientTraction(bc)) {
            curvature_field =
                system.findFieldByName(bc.curvature_field_name);
            if (curvature_field == FE::INVALID_FIELD_ID) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: KinematicAreaGradientTraction could not resolve curvature field '" +
                    bc.curvature_field_name + "'");
            }
        }
        system.declareFreeSurfaceDiscreteFunctional(
            FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
                .interface_marker = bc.interface_marker,
                .level_set_field = resolveLevelSetFieldId(bc, system),
                .curvature_field = curvature_field,
                .velocity_field = velocity_field,
                .geometry_domain_id = bc.generated_interface_domain_id,
                .parameters = std::move(parameters),
                .active_volume_energy_parameters =
                    FE::interfaces::
                        FreeSurfaceActiveVolumeEnergyParameters{
                            .liquid_side =
                                bc.active_domain ==
                                        FreeSurfaceActiveDomain::
                                            LevelSetPositive
                                    ? FE::geometry::CutIntegrationSide::
                                          Positive
                                    : FE::geometry::CutIntegrationSide::
                                          Negative,
                            .density = density,
                            .gravitational_acceleration =
                                gravitational_acceleration,
                            .gravitational_reference_point =
                                {{FE::Real{0.0},
                                  FE::Real{0.0},
                                  FE::Real{0.0}}},
                        },
                .static_conservative_body_force_complete =
                    static_conservative_body_force_complete,
                .active_volume_dissipation_parameters =
                    active_volume_dissipation_parameters,
                .external_pressure_power_parameters =
                    external_pressure_power_parameters,
                .endpoint_functional_power_enabled =
                    usesSurfaceStress(bc) ||
                    usesKinematicAreaGradientTraction(bc),
                .capillary_balance_method =
                    usesSurfaceStress(bc)
                        ? FE::systems::FreeSurfaceCapillaryBalanceMethod::
                              DiscreteEnergyVolumeStationarity
                        : usesKinematicAreaGradientTraction(bc)
                        ? FE::systems::FreeSurfaceCapillaryBalanceMethod::
                              KinematicAreaGradientEnergyTraction
                        : FE::systems::FreeSurfaceCapillaryBalanceMethod::
                              Unselected,
                .capillary_balance_qualification =
                    (usesSurfaceStress(bc) ||
                     usesKinematicAreaGradientTraction(bc))
                        ? FE::systems::
                              FreeSurfaceCapillaryBalanceQualification::
                                  PrerequisiteOnly
                        : FE::systems::
                              FreeSurfaceCapillaryBalanceQualification::
                                  Unselected,
                .owner_component =
                    "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary",
            });
    }
}

void declareFreeSurfaceExternalBoundaryEnergy(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options)
{
    const auto velocity_data_are_homogeneous = [](const auto& bcs) {
        return std::all_of(
            bcs.begin(), bcs.end(), [](const auto& bc) {
                for (std::size_t component = 0u;
                     component < bc.active_components.size();
                     ++component) {
                    if (bc.active_components[component] &&
                        !FE::forms::bc::isZeroConstantScalarValue(
                            bc.value[component])) {
                        return false;
                    }
                }
                return true;
            });
    };
    const bool homogeneous_velocity_data =
        velocity_data_are_homogeneous(
            options.velocity_dirichlet) &&
        velocity_data_are_homogeneous(
            options.velocity_dirichlet_weak);
    const bool imposed_traction_absent =
        options.traction_neumann.empty() &&
        options.traction_robin.empty();
    const bool open_boundary_flux_absent =
        options.pressure_outflow.empty() &&
        options.coupled_outflow_rcr.empty() &&
        options.coupled_outflow_rcrcr.empty();
    system.declareFreeSurfaceExternalBoundaryEnergy(
        FE::systems::
            FreeSurfaceExternalBoundaryEnergyDeclaration{
                .imposed_traction =
                    imposed_traction_absent
                        ? FE::systems::
                              FreeSurfaceExternalBoundaryEnergyApplicability::
                                  NotApplicable
                        : FE::systems::
                              FreeSurfaceExternalBoundaryEnergyApplicability::
                                  Unresolved,
                .open_boundary_flux =
                    open_boundary_flux_absent
                        ? FE::systems::
                              FreeSurfaceExternalBoundaryEnergyApplicability::
                                  NotApplicable
                        : FE::systems::
                              FreeSurfaceExternalBoundaryEnergyApplicability::
                                  Unresolved,
                .homogeneous_velocity_data =
                    homogeneous_velocity_data,
                .pressure_dirichlet_absent =
                    options.pressure_dirichlet.empty(),
                .owner_component =
                    "IncompressibleNavierStokesVMSModule.boundary_energy_coverage",
            });
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
