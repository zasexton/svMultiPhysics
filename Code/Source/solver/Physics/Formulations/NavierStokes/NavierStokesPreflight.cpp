/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/NavierStokesPreflight.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
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

const char* pspgPressureGradientFormName(PspgPressureGradientForm form)
{
    switch (form) {
    case PspgPressureGradientForm::Absolute:
        return "absolute";
    case PspgPressureGradientForm::Incremental:
        return "incremental";
    }
    return "unknown";
}

NavierStokesPreflight preflightNavierStokes(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& velocity_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& pressure_space)
{
    if (options.mesh_velocity_source !=
            ALEMeshVelocitySource::PrescribedData &&
        options.mesh_velocity_source !=
            ALEMeshVelocitySource::CoupledDisplacement) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "unsupported ALE mesh-velocity source");
    }
    if (freeSurfacePhysicalModelName(options.free_surface_physical_model) ==
        nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "unsupported_free_surface_physical_model");
    }
    const auto current_schema =
        IncompressibleNavierStokesVMSOptions::
            current_configuration_schema_version;
    if (options.input_configuration_schema_version != current_schema &&
        options.input_configuration_schema_version != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: unsupported free-surface configuration schema version " +
            std::to_string(options.input_configuration_schema_version));
    }
    if (options.input_configuration_schema_version == 1 &&
        !options.explicit_legacy_configuration) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: schema version 1 requires explicit_legacy_configuration=true and cannot inherit the current capability label");
    }
    if (options.input_configuration_schema_version == current_schema &&
        options.explicit_legacy_configuration) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: explicit legacy behavior requires input configuration schema version 1");
    }
    if (!velocity_space) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null velocity_space");
    }
    if (!pressure_space) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null pressure_space");
    }
    if (options.operator_tag.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: operator_tag must be non-empty");
    }

    const int dim = velocity_space->value_dimension();
    if (dim < 1 || dim > 3) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: velocity space must have 1..3 components");
    }
    if (options.rotating_frame_coriolis_enabled && dim != 3) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: rotating-frame Coriolis forcing requires a 3D velocity space");
    }
    if (pressure_space->value_dimension() != 1) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: pressure space must be scalar");
    }
    if (!(options.density > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: density must be > 0");
    }
    if (options.viscosity_model == nullptr && !(options.viscosity > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: viscosity must be > 0 when viscosity_model is not provided");
    }
    const bool enable_vms = options.enable_vms;
    // The area-gradient traction publishes a compatible stationary pressure
    // state and history. Stabilizing its absolute gradient would recreate a
    // pressure residual after the discrete capillary balance has closed.
    const bool kinematic_area_gradient_pressure_reference =
        std::any_of(
            options.free_surface.begin(),
            options.free_surface.end(),
            usesKinematicAreaGradientTraction);
    const PspgPressureGradientForm pspg_pressure_gradient_form =
        kinematic_area_gradient_pressure_reference
            ? PspgPressureGradientForm::Incremental
            : PspgPressureGradientForm::Absolute;
    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pspg_pressure_gradient_form") +
        " source=" +
        (kinematic_area_gradient_pressure_reference
             ? "kinematic_area_gradient_stationary_pressure"
             : "absolute_pressure_default") +
        " form=" +
        pspgPressureGradientFormName(pspg_pressure_gradient_form));
    if (enable_vms && !(options.stabilization_epsilon > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: stabilization_epsilon must be > 0 when VMS is enabled");
    }

    FE::systems::FieldSpec u_spec;
    u_spec.name = options.velocity_field_name;
    u_spec.space = velocity_space;
    u_spec.components = dim;

    FE::systems::FieldSpec p_spec;
    p_spec.name = options.pressure_field_name;
    p_spec.space = pressure_space;
    p_spec.components = 1;

    std::optional<FE::systems::FieldSpec> body_force_spec;
    if (!options.body_force_field_name.empty()) {
        FE::systems::FieldSpec source_spec;
        source_spec.name = options.body_force_field_name;
        source_spec.space = velocity_space;
        source_spec.components = dim;
        source_spec.source_kind =
            FE::systems::FieldSourceKind::PrescribedData;
        body_force_spec = std::move(source_spec);
    }

    const FE::systems::ALEBindingOptions ale_options{
        .enabled = options.enable_ale,
        .dimension = dim,
        .mesh_velocity_source =
            options.mesh_velocity_source ==
                    ALEMeshVelocitySource::CoupledDisplacement
                ? FE::systems::ALEMeshVelocitySource::CoupledDisplacement
                : FE::systems::ALEMeshVelocitySource::PrescribedData,
        .geometry_tangent_path = options.moving_mesh_tangent_path,
        .mesh_velocity_field_name = options.mesh_velocity_field_name,
        .mesh_displacement_field_name =
            options.mesh_displacement_field_name,
        .mesh_velocity_space = options.mesh_velocity_space
            ? options.mesh_velocity_space
            : velocity_space,
        .mesh_displacement_space = velocity_space,
        .auto_register_mesh_velocity_field =
            options.auto_register_mesh_velocity_field,
        .auto_register_mesh_displacement_field =
            options.auto_register_mesh_displacement_field,
    };

    // Preflight every free-surface field, policy, ownership relation, and
    // generated-marker request before adding Navier--Stokes fields, forms, or
    // constraints.  A rejected configuration must leave the system definition
    // unchanged.
    std::vector<FreeSurfaceBoundary> effective_free_surfaces;
    effective_free_surfaces.reserve(options.free_surface.size());
    std::set<int> prospective_fitted_normal_markers;
    for (const auto& bc : options.free_surface) {
        auto effective_bc = withResolvedInterfaceMarker(bc, system);
        validateFreeSurfaceBoundary(effective_bc,
                                    options,
                                    options.enable_ale,
                                    system,
                                    dim);
        if (effective_bc.implementation ==
                FreeSurfaceImplementation::FittedALE &&
            options.enable_ale &&
            options.mesh_velocity_source ==
                ALEMeshVelocitySource::CoupledDisplacement) {
            if (!system.meshTangentialBoundaryPolicyHistory().empty()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface tangential policies cannot be installed "
                    "after accepted tangential-policy history has begun");
            }
            auto displacement_field = system.meshMotionField(
                FE::systems::MeshMotionFieldRole::Displacement);
            if (!displacement_field.has_value()) {
                const auto named = system.findFieldByName(
                    options.mesh_displacement_field_name);
                if (named != FE::INVALID_FIELD_ID) {
                    displacement_field = named;
                }
            }
            const bool installs_normal_relation =
                effective_bc.kinematic_enforcement !=
                    FreeSurfaceKinematicEnforcement::None &&
                effective_bc.normal_kinematic_policy ==
                    FreeSurfaceNormalKinematicPolicy::
                        MatchFluidNormalVelocity;
            if (installs_normal_relation &&
                !system.meshNormalBoundaryConstraintHistory().empty()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface normal relations cannot be installed "
                    "after accepted normal-constraint history has begun");
            }
            if (installs_normal_relation &&
                !prospective_fitted_normal_markers
                     .insert(effective_bc.boundary_marker)
                     .second) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: duplicate fitted "
                    "free-surface normal relation on marker " +
                    std::to_string(effective_bc.boundary_marker));
            }
            if (installs_normal_relation &&
                effective_bc.kinematic_enforcement ==
                    FreeSurfaceKinematicEnforcement::MeshNitsche &&
                displacement_field.has_value()) {
                // The mesh-motion module adds the Nitsche consistency term of
                // its own operator on the boundaries this relation declares,
                // so it must register after the declaration exists.
                const bool mesh_operator_installed = std::any_of(
                    system.formulationRecords().begin(),
                    system.formulationRecords().end(),
                    [&](const auto& record) {
                        const bool volume = std::find(
                            record.active_domains.begin(),
                            record.active_domains.end(),
                            FE::analysis::DomainKind::Cell) !=
                            record.active_domains.end();
                        const bool mesh_rows = std::any_of(
                            record.block_couplings.begin(),
                            record.block_couplings.end(),
                            [&](const auto& coupling) {
                                return coupling.first == *displacement_field &&
                                       coupling.second == *displacement_field;
                            });
                        return volume && mesh_rows;
                    });
                if (mesh_operator_installed) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: MeshNitsche "
                        "free-surface kinematics on marker " +
                        std::to_string(effective_bc.boundary_marker) +
                        " require the fluid equation to be registered before "
                        "the mesh-motion equation, which adds the Nitsche "
                        "consistency term of its operator on the declared "
                        "boundary");
                }
            }
            if (installs_normal_relation &&
                displacement_field.has_value()) {
                const auto& displacement_record =
                    system.fieldRecord(*displacement_field);
                if (displacement_record.space == nullptr ||
                    velocity_space->polynomial_order() >
                        displacement_record.space->polynomial_order()) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: fitted-ALE "
                        "normal measurement quadrature requires velocity "
                        "order " +
                        std::to_string(
                            velocity_space->polynomial_order()) +
                        " <= displacement-primary order " +
                        (displacement_record.space != nullptr
                             ? std::to_string(
                                   displacement_record.space
                                       ->polynomial_order())
                             : std::string("unavailable")) +
                        " on marker " +
                        std::to_string(
                            effective_bc.boundary_marker));
                }
                const auto normal_conflict = std::find_if(
                    system.meshNormalBoundaryConstraints().begin(),
                    system.meshNormalBoundaryConstraints().end(),
                    [&](const auto& declaration) {
                        return declaration.boundary_marker ==
                                   effective_bc.boundary_marker &&
                               declaration.mesh_displacement_field ==
                                   *displacement_field;
                    });
                if (normal_conflict !=
                    system.meshNormalBoundaryConstraints().end()) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: fitted "
                        "free-surface normal relation on marker " +
                        std::to_string(effective_bc.boundary_marker) +
                        " conflicts with existing mesh-motion owner '" +
                        normal_conflict->owner_component + "'");
                }
            }
            const auto conflict = std::find_if(
                system.meshTangentialBoundaryPolicies().begin(),
                system.meshTangentialBoundaryPolicies().end(),
                [&](const auto& declaration) {
                    return declaration.boundary_marker ==
                               effective_bc.boundary_marker &&
                           (!displacement_field.has_value() ||
                            declaration.mesh_displacement_field ==
                                *displacement_field);
                });
            if (conflict !=
                system.meshTangentialBoundaryPolicies().end()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface tangential policy on marker " +
                    std::to_string(effective_bc.boundary_marker) +
                    " conflicts with existing mesh-motion owner '" +
                    conflict->owner_component + "'");
            }
        }
        effective_free_surfaces.push_back(std::move(effective_bc));
    }
    const bool has_kinematic_area_gradient_traction =
        std::any_of(
            effective_free_surfaces.begin(),
            effective_free_surfaces.end(),
            usesKinematicAreaGradientTraction);
    if (has_kinematic_area_gradient_traction &&
        !isSupportedGeneratedBoundaryTraceSpace(*velocity_space, dim)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: KinematicAreaGradientTraction requires an affine P1 Product H1 velocity space on Triangle3 or Tetra4 cells");
    }
    std::optional<FE::systems::FieldSpec> semi_implicit_reference_spec;
    if (std::any_of(effective_free_surfaces.begin(),
                    effective_free_surfaces.end(),
                    usesSemiImplicitNormalIncrement)) {
        if (!isSupportedGeneratedBoundaryTraceSpace(*velocity_space, dim)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: Surface_tension_semi_implicit=NormalIncrement requires an affine P1 Product H1 velocity space on Triangle3 or Tetra4 cells");
        }
        FE::systems::FieldSpec reference_spec;
        reference_spec.name =
            std::string(kFreeSurfaceSemiImplicitReferenceVelocityFieldName);
        reference_spec.space = velocity_space;
        reference_spec.components = dim;
        reference_spec.source_kind =
            FE::systems::FieldSourceKind::PrescribedData;
        FE::systems::FieldBinding::check(
            system,
            reference_spec,
            FE::systems::FieldSourceKind::PrescribedData,
            /*allow_missing=*/true,
            "IncompressibleNavierStokesVMSModule::registerOn semi-implicit reference velocity");
        semi_implicit_reference_spec = std::move(reference_spec);
    }
    validateGeneratedFreeSurfaceMarkerUniqueness(
        effective_free_surfaces, system);
    validateActiveDomainPressureConstraints(
        system,
        options,
        effective_free_surfaces);
    const bool has_exterior_free_surface = std::any_of(
        effective_free_surfaces.begin(),
        effective_free_surfaces.end(),
        isExteriorOnePhaseBoundary);
    const auto active_pressure_domain =
        activePressureDomainFor(effective_free_surfaces);
    if (!(options
              .generated_boundary_nitsche_minimum_energy_ratio >
          FE::Real{0.0}) ||
        !(options
              .generated_boundary_nitsche_minimum_energy_ratio <
          FE::Real{1.0}) ||
        !std::isfinite(
            options
                .generated_boundary_nitsche_minimum_energy_ratio)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "generated_boundary_nitsche_minimum_energy_ratio must be "
            "finite and strictly between zero and one");
    }
    if (!options.velocity_dirichlet_weak.empty() &&
        (!(options.nitsche_gamma > FE::Real{0.0}) ||
         !std::isfinite(options.nitsche_gamma))) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "nitsche_gamma must be finite and > 0 when weak velocity "
            "Dirichlet conditions are configured");
    }
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const bool native_mesh_available_for_trace_certificate =
        system.mesh() != nullptr;
#else
    const bool native_mesh_available_for_trace_certificate =
        false;
#endif
    const bool has_generated_boundary_nitsche_route =
        std::any_of(
            options.velocity_dirichlet_weak.begin(),
            options.velocity_dirichlet_weak.end(),
            [&](const auto& bc) {
                return sharpActiveBoundaryMarkerFor(
                           FE::forms::bc::detail::
                               boundaryMarkerOrThrow(
                                   bc,
                                   "IncompressibleNavierStokesVMSModule::"
                                   "registerOn generated-boundary "
                                   "Nitsche trace preflight"),
                           effective_free_surfaces,
                           system)
                    .has_value();
            });

    FE::systems::FieldBinding::requireDistinctNames(
        "IncompressibleNavierStokesVMSModule::registerOn", {
        {"velocity", u_spec.name},
        {"pressure", p_spec.name},
        {"body-force data",
         body_force_spec.has_value() ? std::string_view(body_force_spec->name)
                                     : std::string_view{}},
        {"mesh velocity",
         options.enable_ale
             ? std::string_view(ale_options.mesh_velocity_field_name)
             : std::string_view{}},
        {"mesh displacement",
         options.enable_ale &&
                 ale_options.mesh_velocity_source ==
                     FE::systems::ALEMeshVelocitySource::CoupledDisplacement
             ? std::string_view(ale_options.mesh_displacement_field_name)
             : std::string_view{}},
        {"semi-implicit reference velocity",
         semi_implicit_reference_spec.has_value()
             ? std::string_view(semi_implicit_reference_spec->name)
             : std::string_view{}},
    });
    FE::systems::FieldBinding::check(
        system,
        u_spec,
        FE::systems::FieldSourceKind::Unknown,
        /*allow_missing=*/true,
        "IncompressibleNavierStokesVMSModule::registerOn velocity");
    FE::systems::FieldBinding::check(
        system,
        p_spec,
        FE::systems::FieldSourceKind::Unknown,
        /*allow_missing=*/true,
        "IncompressibleNavierStokesVMSModule::registerOn pressure");
    if (body_force_spec.has_value()) {
        FE::systems::FieldBinding::check(
            system,
            *body_force_spec,
            FE::systems::FieldSourceKind::PrescribedData,
            options.auto_register_body_force_field,
            "IncompressibleNavierStokesVMSModule::registerOn momentum source");
    }
    FE::systems::validateALEBinding(system, ale_options);
    if (ale_options.mesh_velocity_source !=
        FE::systems::ALEMeshVelocitySource::CoupledDisplacement) {
        for (const auto& bc : effective_free_surfaces) {
            for (const auto& contact_line : bc.contact_lines) {
                if (contactLineKind(contact_line) == ContactLineKind::Pinned) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: pinned fitted contact lines require ALE mesh velocity to be derived from a coupled mesh displacement unknown");
                }
            }
        }
    }
    validateNavierStokesBoundaryConfiguration(
        options,
        effective_free_surfaces,
        system,
        *velocity_space,
        *pressure_space,
        dim);
    if (has_generated_boundary_nitsche_route) {
        if (!active_pressure_domain.has_value() ||
            active_pressure_domain->boundary
                    ->active_domain_method !=
                FreeSurfaceActiveDomainMethod::CutVolume ||
            !active_pressure_domain->boundary
                 ->small_cut_aggregation) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "generated-boundary Nitsche routes require CutVolume "
                "small-cut aggregation and aggregate-trace certification");
        }
        if (!native_mesh_available_for_trace_certificate) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "generated-boundary Nitsche routes require a native mesh "
                "for aggregate-trace certification");
        }
        if (options.viscosity_model != nullptr ||
            !(options.viscosity > FE::Real{0.0}) ||
            !std::isfinite(options.viscosity)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "certified generated-boundary Nitsche routes require "
                "finite positive constant Newtonian viscosity");
        }
        const bool supported_trace_space =
            isSupportedGeneratedBoundaryTraceSpace(
                *velocity_space, dim);
        const auto existing_velocity =
            system.findFieldByName(
                options.velocity_field_name);
        bool supported_existing_velocity = true;
        if (existing_velocity != FE::INVALID_FIELD_ID) {
            const auto& record =
                system.fieldRecord(existing_velocity);
            supported_existing_velocity =
                record.space &&
                record.components == dim &&
                isSupportedGeneratedBoundaryTraceSpace(
                    *record.space, dim);
        }
        if (!supported_trace_space) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "certified generated-boundary Nitsche routes require "
                "an affine P1 Product H1 velocity space on Triangle3 "
                "or Tetra4 cells");
        }
        if (!supported_existing_velocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "the existing velocity field is outside the affine P1 "
                "Product H1 Triangle3/Tetra4 generated-boundary trace "
                "certificate envelope");
        }
    }
    validateFreeSurfaceContactGeometryPreflight(
        effective_free_surfaces, system);
    validateFreeSurfaceDiscreteFunctionalPreflight(
        effective_free_surfaces, system);

    return NavierStokesPreflight{
        .dim = dim,
        .enable_vms = enable_vms,
        .pspg_pressure_gradient_form = pspg_pressure_gradient_form,
        .u_spec = std::move(u_spec),
        .p_spec = std::move(p_spec),
        .body_force_spec = std::move(body_force_spec),
        .semi_implicit_reference_spec = std::move(semi_implicit_reference_spec),
        .ale_options = ale_options,
        .effective_free_surfaces = std::move(effective_free_surfaces),
        .has_exterior_free_surface = has_exterior_free_surface,
    };
}

bool hasNativeMeshOrWarn(const FE::systems::FESystem& system)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const bool native_mesh_available = system.mesh() != nullptr;
#else
    (void)system;
    const bool native_mesh_available = false;
#endif
    if (!native_mesh_available) {
        FE_LOG_WARNING(
            "IncompressibleNavierStokesVMSModule: skipping active-domain "
            "inactive velocity/pressure constraints because native mesh support "
            "is unavailable in this FESystem");
    }
    return native_mesh_available;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
