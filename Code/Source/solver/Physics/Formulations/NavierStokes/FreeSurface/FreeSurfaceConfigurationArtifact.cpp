/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceConfigurationArtifact.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceBoundarySchema.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceCollectives.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnosticOperators.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceResidualWork.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceValidation.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesGeneratedNitsche.h"
#include "Physics/Formulations/NavierStokes/NavierStokesInitialConditions.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPreflight.h"

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

[[nodiscard]] std::string jsonString(std::string_view value)
{
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() + 2u);
    out.push_back('"');
    for (const unsigned char c : value) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20u) {
                out += "\\u00";
                out.push_back(hex[(c >> 4u) & 0x0fu]);
                out.push_back(hex[c & 0x0fu]);
            } else {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    out.push_back('"');
    return out;
}

[[nodiscard]] std::string jsonReal(FE::Real value)
{
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: effective configuration contains a non-finite scalar");
    }
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<FE::Real>::max_digits10)
           << value;
    return stream.str();
}

[[nodiscard]] constexpr const char* jsonBool(bool value) noexcept
{
    return value ? "true" : "false";
}

[[nodiscard]] std::string jsonScalarValue(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value)
{
    if (const auto* literal = std::get_if<FE::Real>(&value)) {
        return jsonReal(*literal);
    }
    if (std::holds_alternative<FE::forms::ScalarCoefficient>(value)) {
        return R"({"kind":"spatial_coefficient"})";
    }
    if (std::holds_alternative<FE::forms::TimeScalarCoefficient>(value)) {
        return R"({"kind":"time_coefficient"})";
    }
    return R"({"kind":"form_expression"})";
}

[[nodiscard]] std::string contactConfigurationJson(
    const FreeSurfaceContactLine& contact_line)
{
    const auto kind = contactLineKind(contact_line);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"model\":" << jsonString(contactLineKindName(kind));
    if (kind != ContactLineKind::None) {
        out << ",\"wall_boundary_marker\":"
            << contactLineWallBoundaryMarker(contact_line)
            << ",\"contact_line_marker\":"
            << contactLineMarker(contact_line);
    }
    if (kind == ContactLineKind::PrescribedAngle ||
        kind == ContactLineKind::DynamicRenE) {
        const auto wall_normal = normalizedWallNormal(contact_line);
        out << ",\"angle_convention\":\"radians_measured_through_liquid\""
            << ",\"angle_unit\":\"radian\""
            << ",\"equilibrium_angle_radians\":"
            << jsonScalarValue(contactLineAngleRadians(contact_line))
            << ",\"wall_normal_convention\":\"outward_from_liquid_into_solid\""
            << ",\"wall_normal\":[" << jsonReal(wall_normal[0]) << ','
            << jsonReal(wall_normal[1]) << ',' << jsonReal(wall_normal[2])
            << ']';
    }
    if (kind == ContactLineKind::PrescribedAngle) {
        // Decision D4: the Young term in momentum is the only angle
        // mechanism; level-set wall maintenance only rescales contact cells.
        out << ",\"level_set_geometry_owner\":"
            << "\"none_angle_preserving_wall_maintenance\""
            << ",\"prescribed_angle_operator\":\"momentum_young_term_only\"";
        if (contactLineHasNavierSlip(contact_line)) {
            out << ",\"wall_slip_model\":\"Navier\""
                << ",\"slip_length\":"
                << jsonScalarValue(contactLineSlipLength(contact_line))
                << ",\"slip_length_unit\":\"length\""
                << ",\"line_friction\":\"absent\"";
        }
    }
    if (kind == ContactLineKind::DynamicRenE) {
        const auto mobility = constantScalarValueOrThrow(
            contactLineMobility(contact_line),
            "effective contact-line mobility");
        out << ",\"law\":\"xi_V_equals_gamma_cos_equilibrium_minus_cos_dynamic\""
            << ",\"mobility\":" << jsonReal(mobility)
            << ",\"mobility_unit\":\"solver_consistent\""
            << ",\"line_friction\":" << jsonReal(FE::Real{1.0} / mobility)
            << ",\"line_friction_unit\":\"reciprocal_mobility\""
            << ",\"slip_length\":"
            << jsonScalarValue(contactLineSlipLength(contact_line))
            << ",\"slip_length_unit\":\"length\"";
    }
    out << '}';
    return out.str();
}

[[nodiscard]] bool freeSurfaceConfigurationLess(
    const FreeSurfaceBoundary& lhs,
    const FreeSurfaceBoundary& rhs)
{
    return std::tie(lhs.role,
                    lhs.implementation,
                    lhs.boundary_marker,
                    lhs.interface_marker,
                    lhs.level_set_field_name,
                    lhs.generated_interface_domain_id) <
           std::tie(rhs.role,
                    rhs.implementation,
                    rhs.boundary_marker,
                    rhs.interface_marker,
                    rhs.level_set_field_name,
                    rhs.generated_interface_domain_id);
}

[[nodiscard]] bool contactConfigurationLess(
    const FreeSurfaceContactLine& lhs,
    const FreeSurfaceContactLine& rhs)
{
    return std::tuple{contactLineKind(lhs),
                      contactLineWallBoundaryMarker(lhs),
                      contactLineMarker(lhs)} <
           std::tuple{contactLineKind(rhs),
                      contactLineWallBoundaryMarker(rhs),
                      contactLineMarker(rhs)};
}

struct TangentialPolicyProvenance {
    const FE::systems::MeshTangentialBoundaryPolicyDeclaration*
        declaration{nullptr};
    const FE::analysis::BoundaryConditionDescriptor* consumed_operator{
        nullptr};
};

[[nodiscard]] TangentialPolicyProvenance
tangentialPolicyProvenance(
    const FreeSurfaceBoundary& boundary,
    const FE::systems::FESystem& system)
{
    if (boundary.implementation !=
        FreeSurfaceImplementation::FittedALE) {
        return {};
    }
    const auto displacement = system.meshMotionField(
        FE::systems::MeshMotionFieldRole::Displacement);
    if (!displacement.has_value()) {
        return {};
    }
    const auto declarations =
        system.meshTangentialBoundaryPolicies();
    const auto declaration = std::find_if(
        declarations.begin(),
        declarations.end(),
        [&](const auto& candidate) {
            return candidate.mesh_displacement_field == *displacement &&
                   candidate.boundary_marker ==
                       boundary.boundary_marker &&
                   candidate.policy == systemTangentialMeshPolicy(
                                           boundary.tangential_mesh_policy);
        });
    if (declaration == declarations.end()) {
        return {};
    }
    if (!declaration->consumer_bound ||
        !system.hasOperator(declaration->consumer_operator_tag) ||
        declaration->consumer_source != fittedTangentialOperatorSource(
            boundary.boundary_marker,
            boundary.tangential_mesh_policy)) {
        return TangentialPolicyProvenance{
            .declaration = &*declaration,
        };
    }

    const auto& descriptors =
        system.boundaryConditionDescriptors();
    const auto matches_consumer = [&](const auto& candidate) {
        return candidate.primary_variable ==
                   FE::analysis::VariableKey::field(
                       declaration->mesh_displacement_field) &&
               candidate.boundary_marker ==
                   declaration->boundary_marker &&
               candidate.trace_kind ==
                   FE::analysis::TraceKind::TangentialComponent &&
               candidate.enforcement_kind ==
                   (boundary.tangential_mesh_policy ==
                            FreeSurfaceTangentialMeshPolicy::Prescribed
                        ? FE::analysis::EnforcementKind::WeakPenalty
                        : FE::analysis::EnforcementKind::
                              WeakConsistent) &&
               candidate.source == declaration->consumer_source;
    };
    const auto consumed = std::find_if(
        descriptors.begin(),
        descriptors.end(),
        matches_consumer);
    const auto consumed_count = std::count_if(
        descriptors.begin(),
        descriptors.end(),
        matches_consumer);
    return TangentialPolicyProvenance{
        .declaration = &*declaration,
        .consumed_operator =
            consumed_count == 1u ? &*consumed : nullptr,
    };
}

} // namespace

[[nodiscard]] EffectiveConfigurationArtifact makeEffectiveConfigurationArtifact(
    const IncompressibleNavierStokesVMSOptions& options,
    std::vector<FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system,
    int dimension,
    bool effective_enable_vms,
    PspgPressureGradientForm pspg_pressure_gradient_form)
{
    std::sort(free_surfaces.begin(), free_surfaces.end(),
              freeSurfaceConfigurationLess);
    const bool has_internal_material_interface_volume = std::any_of(
        free_surfaces.begin(),
        free_surfaces.end(),
        isInternalMaterialInterfaceVolume);
    const bool has_exterior_one_phase_boundary = std::any_of(
        free_surfaces.begin(),
        free_surfaces.end(),
        isExteriorOnePhaseBoundary);

    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"artifact_schema_version\":4"
        << ",\"component\":\"incompressible_navier_stokes_free_surface\""
        << ",\"configuration_schema\":{\"input_version\":"
        << options.input_configuration_schema_version
        << ",\"effective_version\":"
        << IncompressibleNavierStokesVMSOptions::
               current_configuration_schema_version
        << ",\"migration_mode\":"
        << jsonString(options.explicit_legacy_configuration
                          ? "explicit_legacy"
                          : "current")
        << '}'
        << ",\"capability_label\":"
        << jsonString(options.explicit_legacy_configuration
                          ? "legacy_diagnostic"
                          : has_internal_material_interface_volume &&
                                    !has_exterior_one_phase_boundary
                                ? "internal_material_interface_phase_volume"
                                : "one_phase_liquid_sharp_interface")
        << ",\"physical_model\":";
    if (options.explicit_legacy_configuration) {
        out << "null";
    } else if (has_internal_material_interface_volume &&
               !has_exterior_one_phase_boundary) {
        out << "{\"name\":\"internal_material_interface_phase_volume\""
            << ",\"interface_traction_owner\":\"coupled_parent\""
            << ",\"pressure_gauge_owner\":\"coupled_parent\""
            << ",\"external_boundary_model\":false}";
    } else {
        out << "{\"name\":"
            << jsonString(freeSurfacePhysicalModelName(
                   options.free_surface_physical_model))
            << ",\"liquid_phase_count\":1"
            << ",\"liquid_velocity_field_count\":1"
            << ",\"liquid_pressure_field_count\":1"
            << ",\"material_density_state_count\":1"
            << ",\"material_viscosity_state_count\":1"
            << ",\"exterior_pressure_mode\":"
               "\"prescribed_scalar_traction_reference\""
            << ",\"exterior_momentum_solved\":false"
            << ",\"exterior_pressure_field_solved\":false"
            << ",\"incompressible_two_fluid_implemented\":false"
            << ",\"gas_dynamics_implemented\":false}";
    }
    out
        << ",\"units\":{\"system\":\"consistent_solver_units\",\"angle\":\"radian\",\"length\":\"solver_length\",\"pressure\":\"solver_pressure\",\"surface_tension\":\"force_per_length\"}"
        << ",\"fields\":{\"velocity\":"
        << jsonString(options.velocity_field_name)
        << ",\"pressure\":" << jsonString(options.pressure_field_name)
        << ",\"operator\":" << jsonString(options.operator_tag)
        << ",\"dimension\":" << dimension << '}'
        << ",\"ale\":{\"enabled\":" << jsonBool(options.enable_ale)
        << ",\"mesh_velocity_source\":"
        << jsonString(options.mesh_velocity_source ==
                              ALEMeshVelocitySource::CoupledDisplacement
                          ? "CoupledDisplacement"
                          : "PrescribedData")
        << ",\"mesh_velocity_field\":"
        << jsonString(options.mesh_velocity_field_name)
        << ",\"mesh_displacement_field\":"
        << jsonString(options.mesh_displacement_field_name)
        << ",\"geometry_tangent_path\":"
        << jsonString(geometryTangentPathName(options.moving_mesh_tangent_path))
        << '}'
        << ",\"generic_velocity_nitsche\":{\"gamma\":"
        << jsonReal(options.nitsche_gamma)
        << ",\"symmetric\":" << jsonBool(options.nitsche_symmetric)
        << ",\"scale_with_polynomial_order\":"
        << jsonBool(options.nitsche_scale_with_p)
        << ",\"generated_active_boundary_minimum_energy_ratio\":"
        << jsonReal(
               options
                   .generated_boundary_nitsche_minimum_energy_ratio)
        << '}'
        << ",\"stabilization\":{\"vms_enabled\":"
        << jsonBool(effective_enable_vms)
        << ",\"ct_m\":" << jsonReal(options.ct_m)
        << ",\"ct_c\":" << jsonReal(options.ct_c)
        << ",\"epsilon\":" << jsonReal(options.stabilization_epsilon)
        << ",\"pspg_pressure_gradient_form\":"
        << jsonString(
               pspgPressureGradientFormName(
                   pspg_pressure_gradient_form))
        << '}'
        << ",\"maintenance_policy\":{\"owner_component\":\"level_set_transport\",\"coupling\":\"one_way_velocity_to_extension_to_level_set\"}"
        << ",\"extension_guards\":{\"physical_momentum_dry_extension_allowed\":false,\"auxiliary_extension_owner\":\"level_set_transport\",\"external_owner_required\":true}"
        << ",\"free_surfaces\":[";

    for (std::size_t surface_index = 0;
         surface_index < free_surfaces.size(); ++surface_index) {
        if (surface_index != 0u) {
            out << ',';
        }
        const auto& boundary = free_surfaces[surface_index];
        const auto tangential_provenance =
            tangentialPolicyProvenance(boundary, system);
        auto contacts = boundary.contact_lines;
        std::sort(contacts.begin(), contacts.end(), contactConfigurationLess);

        out << "{\"role\":"
            << jsonString(freeSurfaceBoundaryRoleName(boundary.role))
            << ",\"implementation\":"
            << jsonString(freeSurfaceImplementationName(boundary.implementation))
            << ",\"boundary_marker\":" << boundary.boundary_marker
            << ",\"interface_marker\":" << boundary.interface_marker
            << ",\"level_set_field\":"
            << jsonString(boundary.level_set_field_name)
            << ",\"generated_interface_domain\":"
            << jsonString(boundary.generated_interface_domain_id)
            << ",\"generated_interface_geometry\":"
            << jsonString(boundary.generated_interface_geometry)
            << ",\"geometry_tangent_policy\":"
            << jsonString(boundary.geometry_tangent_policy)
            << ",\"level_set_isovalue\":"
            << jsonReal(boundary.level_set_isovalue)
            << ",\"active_domain\":"
            << jsonString(activeDomainName(boundary.active_domain))
            << ",\"active_phase_sign\":"
            << jsonString(activePhaseSignName(boundary.active_domain))
            << ",\"active_domain_method\":"
            << jsonString(activeDomainMethodName(boundary.active_domain_method))
            << ",\"active_domain_smoothing_width\":"
            << jsonReal(boundary.active_domain_smoothing_width)
            << ",\"smoothing_width_unit\":\"length\""
            << ",\"allow_full_domain_unfitted_free_surface\":"
            << jsonBool(boundary.allow_full_domain_unfitted_free_surface)
            << ",\"external_pressure\":"
            << jsonScalarValue(boundary.external_pressure)
            << ",\"surface_tension\":"
            << jsonScalarValue(boundary.surface_tension)
            << ",\"surface_tension_form_requested\":";
        switch (boundary.surface_tension_form) {
        case FreeSurfaceSurfaceTensionForm::Automatic:
            out << "\"Automatic\"";
            break;
        case FreeSurfaceSurfaceTensionForm::KinematicAreaGradientTraction:
            out << "\"KinematicAreaGradientTraction\"";
            break;
        case FreeSurfaceSurfaceTensionForm::SurfaceStress:
            out << "\"SurfaceStress\"";
            break;
        }
        out << ",\"surface_tension_form_effective\":"
            << jsonString(
                   isInternalMaterialInterfaceVolume(boundary)
                       ? "NotApplicableInternalVolume"
                       : surfaceTensionFormName(boundary));
        if (usesSemiImplicitNormalIncrement(boundary)) {
            out << ",\"surface_tension_semi_implicit\":"
                << jsonString(freeSurfaceSurfaceTensionSemiImplicitName(
                       boundary.surface_tension_semi_implicit))
                << ",\"surface_tension_semi_implicit_reference_velocity\":"
                << jsonString(std::string(
                       kFreeSurfaceSemiImplicitReferenceVelocityFieldName));
        }
        if (isExteriorOnePhaseBoundary(boundary) &&
            usesKinematicAreaGradientTraction(boundary)) {
            out << ",\"surface_tension_form_qualification\":"
                << "\"prerequisite_only\""
                << ",\"surface_tension_normal_source\":"
                << "\"generated_interface_rule_geometry\""
                << ",\"surface_tension_curvature_source\":"
                << "\"kinematic_area_gradient_total_surface_young_energy\""
                << ",\"equilibrium_young_force_owner\":"
                << "\"total_energy_gradient_traction\"";
        }
        out << ",\"curvature_policy\":"
            << jsonString(freeSurfaceCurvaturePolicyName(boundary))
            << ",\"curvature_tangent_policy\":"
            << jsonString(
                   freeSurfaceCurvatureTangentPolicyName(boundary))
            << ",\"kinematic\":{\"normal_policy\":"
            << jsonString(normalKinematicPolicyName(
                   boundary.normal_kinematic_policy))
            << ",\"tangential_mesh_policy\":"
            << jsonString(tangentialMeshPolicyName(
                   boundary.tangential_mesh_policy))
            << ",\"prescribed_tangential_mesh_velocity\":["
            << jsonScalarValue(
                   boundary.prescribed_tangential_mesh_velocity[0])
            << ','
            << jsonScalarValue(
                   boundary.prescribed_tangential_mesh_velocity[1])
            << ','
            << jsonScalarValue(
                   boundary.prescribed_tangential_mesh_velocity[2])
            << ']'
            << ",\"tangential_mesh_penalty\":"
            << jsonScalarValue(boundary.tangential_mesh_penalty)
            << ",\"tangential_mesh_owner\":";
        if (tangential_provenance.declaration != nullptr) {
            out << jsonString(
                tangential_provenance.declaration->owner_component);
        } else {
            out << "null";
        }
        out << ",\"policy_consumed\":"
            << jsonBool(
                   tangential_provenance.consumed_operator != nullptr)
            << ",\"operator_tag\":";
        if (tangential_provenance.consumed_operator != nullptr) {
            out << jsonString(
                tangential_provenance.declaration
                    ->consumer_operator_tag);
        } else {
            out << "null";
        }
        out << ",\"operator_source\":";
        if (tangential_provenance.consumed_operator != nullptr) {
            out << jsonString(
                tangential_provenance.consumed_operator->source);
        } else {
            out << "null";
        }
        out << ",\"policy_qualification\":"
            << jsonString(
                   boundary.implementation !=
                           FreeSurfaceImplementation::FittedALE
                       ? "not_applicable"
                       : options.explicit_legacy_configuration
                             ? "unqualified_explicit_legacy"
                             : "supported_configuration_envelope")
            << ",\"enforcement\":"
            << jsonString(kinematicEnforcementName(
                   boundary.kinematic_enforcement))
            << ",\"nitsche\":{\"gamma\":"
            << jsonReal(boundary.kinematic_nitsche_gamma) << "}}"
            << ",\"stabilization\":{\"enabled\":"
            << jsonBool(boundary.cut_cell_stabilization.enabled)
            << ",\"small_cut_aggregation\":"
            << jsonBool(boundary.small_cut_aggregation)
            << ",\"pressure_policy\":"
            << jsonString(pressureStabilizationPolicyName(
                   boundary.cut_cell_stabilization.pressure_policy))
            << ",\"pressure_gradient_penalty\":"
            << jsonScalarValue(
                   boundary.cut_cell_stabilization.pressure_gradient_penalty)
            << ",\"use_cut_metadata_scale\":"
            << jsonBool(
                   boundary.cut_cell_stabilization.use_cut_metadata_scale)
            << ",\"cut_metadata_scale_cap\":";
        if (boundary.cut_cell_stabilization.cut_metadata_scale_cap.has_value()) {
            out << jsonReal(
                *boundary.cut_cell_stabilization.cut_metadata_scale_cap);
        } else {
            out << "null";
        }
        out << ",\"aggregation_guards\":{\"maximum_root_path_length\":"
            << boundary.small_cut_aggregation_guards
                   .maximum_root_path_length
            << ",\"maximum_reference_extrapolation_distance\":"
            << jsonReal(
                   boundary.small_cut_aggregation_guards
                       .maximum_reference_extrapolation_distance)
            << ",\"maximum_absolute_coefficient\":"
            << jsonReal(
                   boundary.small_cut_aggregation_guards
                       .maximum_absolute_coefficient)
            << ",\"maximum_row_l1_norm\":"
            << jsonReal(
                   boundary.small_cut_aggregation_guards
                       .maximum_row_l1_norm)
            << ",\"rootless_fallback\":"
            << jsonBool(
                   boundary.small_cut_aggregation_guards.rootless_fallback)
            << '}';
        out << '}'
            << ",\"pruning\":{\"decision_owner\":\"authoritative_geometry_snapshot\",\"fallback_to_whole_face\":false}"
            << ",\"contact_lines\":[";
        for (std::size_t contact_index = 0;
             contact_index < contacts.size(); ++contact_index) {
            if (contact_index != 0u) {
                out << ',';
            }
            out << contactConfigurationJson(contacts[contact_index]);
        }
        out << "]}";
    }
    out << "]";
    // Every input key of each boundary with its resolved value and source
    // (decision D47); the keys above are unchanged.
    out << ",\"resolved\":" << resolvedFreeSurfacesJson(free_surfaces);
    out << "}";

    return EffectiveConfigurationArtifact{
        .component = "incompressible_navier_stokes_free_surface",
        .json = out.str(),
    };
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
