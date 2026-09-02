/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidModule.h"

#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidInterface.h"

#include "FE/Constraints/CoupledFieldGaugeConstraint.h"
#include "FE/Constraints/GaugeRegistry.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Forms/Vocabulary.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

namespace {

[[nodiscard]] bool spacesCompatible(
    const FE::spaces::FunctionSpace& lhs,
    const FE::spaces::FunctionSpace& rhs) noexcept
{
    return lhs.element_type() == rhs.element_type() &&
           lhs.polynomial_order() == rhs.polynomial_order() &&
           lhs.continuity() == rhs.continuity() &&
           lhs.value_dimension() == rhs.value_dimension();
}

void requireFinitePositive(FE::Real value, std::string_view name)
{
    if (!std::isfinite(value) || !(value > FE::Real{0.0})) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: " + std::string(name) +
            " must be finite and positive");
    }
}

enum class VelocityBoundaryDataPolicy {
    HomogeneousPhaseLocal,
    SharedExternalData,
};

void validateSharedVelocityBoundaryValue(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view scope)
{
    if (const auto* literal = std::get_if<FE::Real>(&value)) {
        if (!std::isfinite(*literal)) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_nonfinite_shared_velocity_boundary:" +
                std::string(scope));
        }
        return;
    }
    if (const auto* coefficient =
            std::get_if<FE::forms::ScalarCoefficient>(&value)) {
        if (!*coefficient) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_empty_shared_velocity_coefficient:" +
                std::string(scope));
        }
        return;
    }
    if (const auto* coefficient =
            std::get_if<FE::forms::TimeScalarCoefficient>(&value)) {
        if (!*coefficient) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_empty_shared_velocity_coefficient:" +
                std::string(scope));
        }
        return;
    }
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] "
        "unsupported_two_fluid_shared_velocity_form_expression:" +
        std::string(scope));
}

void validateVelocityBoundaryData(
    std::span<const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC>
        boundaries,
    std::string_view scope,
    VelocityBoundaryDataPolicy policy)
{
    std::set<int> markers;
    for (const auto& boundary : boundaries) {
        if (boundary.boundary_marker < 0) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_negative_velocity_boundary_marker:" +
                std::string(scope));
        }
        if (!markers.insert(boundary.boundary_marker).second) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_duplicate_velocity_boundary_marker:" +
                std::string(scope));
        }
        for (std::size_t component = 0u;
             component < boundary.active_components.size(); ++component) {
            if (!boundary.active_components[component]) {
                continue;
            }
            if (policy == VelocityBoundaryDataPolicy::SharedExternalData) {
                validateSharedVelocityBoundaryValue(
                    boundary.value[component], scope);
            } else if (!FE::forms::bc::isZeroConstantScalarValue(
                           boundary.value[component])) {
                throw std::invalid_argument(
                    "[svMultiPhysics::Physics] "
                    "unsupported_two_fluid_nonhomogeneous_velocity_boundary:" +
                    std::string(scope));
            }
        }
    }
}

void validateNoSharedVelocityBoundaryOverlap(
    std::span<const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC>
        shared,
    std::span<const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC>
        phase_local,
    std::string_view phase)
{
    std::set<int> shared_markers;
    for (const auto& boundary : shared) {
        shared_markers.insert(boundary.boundary_marker);
    }
    for (const auto& boundary : phase_local) {
        if (shared_markers.find(boundary.boundary_marker) !=
            shared_markers.end()) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_overlapping_shared_velocity_boundary_marker:" +
                std::string(phase));
        }
    }
}

void validateVelocitySpace(
    const std::shared_ptr<const FE::spaces::FunctionSpace>& space,
    std::string_view phase)
{
    if (!space) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: null " + std::string(phase) +
            " velocity space");
    }
    const auto product = std::dynamic_pointer_cast<
        const FE::spaces::ProductSpace>(space);
    const auto dim = space->value_dimension();
    const bool supported_cell =
        (dim == 2 && space->element_type() == FE::ElementType::Triangle3) ||
        (dim == 3 && space->element_type() == FE::ElementType::Tetra4);
    if (!product || !supported_cell || space->polynomial_order() != 1 ||
        space->continuity() != FE::Continuity::C0) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: phase velocity spaces must be affine C0 P1 Product H1 spaces on Triangle3 or Tetra4 cells");
    }
}

void validatePressureSpace(
    const std::shared_ptr<const FE::spaces::FunctionSpace>& space,
    const FE::spaces::FunctionSpace& velocity_space,
    std::string_view phase)
{
    if (!space || space->value_dimension() != 1 ||
        space->space_type() != FE::spaces::SpaceType::H1 ||
        space->element_type() != velocity_space.element_type() ||
        space->polynomial_order() != 1 ||
        space->continuity() != FE::Continuity::C0) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: " + std::string(phase) +
            " pressure space must be a matching scalar affine C0 P1 H1 space");
    }
}

void validateExistingUnknownField(
    const FE::systems::FESystem& system,
    std::string_view name,
    const FE::spaces::FunctionSpace& space,
    int components)
{
    const auto field = system.findFieldByName(name);
    if (field == FE::INVALID_FIELD_ID) {
        return;
    }
    const auto& record = system.fieldRecord(field);
    if (record.source_kind != FE::systems::FieldSourceKind::Unknown ||
        record.components != components || !record.space ||
        !spacesCompatible(*record.space, space)) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: existing field '" +
            std::string(name) +
            "' is incompatible with the requested phase unknown");
    }
}

[[nodiscard]] std::string jsonString(std::string_view value)
{
    std::string out{"\""};
    for (const unsigned char c : value) {
        switch (c) {
        case '\"':
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
                std::ostringstream escaped;
                escaped << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<unsigned int>(c);
                out += escaped.str();
            } else {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    out.push_back('\"');
    return out;
}

[[nodiscard]] std::string jsonReal(FE::Real value)
{
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: effective configuration contains a non-finite scalar");
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<FE::Real>::max_digits10)
        << value;
    return out.str();
}

[[nodiscard]] constexpr const char* jsonBool(bool value) noexcept
{
    return value ? "true" : "false";
}

[[nodiscard]] std::string sharedVelocityValueArtifact(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value)
{
    if (const auto* literal = std::get_if<FE::Real>(&value)) {
        return "{\"kind\":\"literal\",\"value\":" +
               jsonReal(*literal) + '}';
    }
    if (std::holds_alternative<FE::forms::ScalarCoefficient>(value)) {
        return R"({"kind":"spatial_coefficient"})";
    }
    if (std::holds_alternative<FE::forms::TimeScalarCoefficient>(value)) {
        return R"({"kind":"time_coefficient"})";
    }
    return R"({"kind":"form_expression"})";
}

void appendVelocityBoundaryArtifact(
    std::ostringstream& out,
    const IncompressibleTwoFluidOptions& options,
    int dimension)
{
    out << ",\"boundary_conditions\":{"
        << "\"shared_velocity_dirichlet_count\":"
        << options.shared_velocity_dirichlet.size()
        << ",\"shared_velocity_dirichlet_policy\":"
        << jsonString(
               "identical_external_data_on_both_phase_restrictions")
        << ",\"shared_velocity_dirichlet\":[";
    for (std::size_t boundary_index = 0u;
         boundary_index < options.shared_velocity_dirichlet.size();
         ++boundary_index) {
        if (boundary_index != 0u) {
            out << ',';
        }
        const auto& boundary =
            options.shared_velocity_dirichlet[boundary_index];
        out << "{\"marker\":" << boundary.boundary_marker
            << ",\"active_components\":[";
        for (int component = 0; component < dimension; ++component) {
            if (component != 0) {
                out << ',';
            }
            out << jsonBool(
                boundary.active_components[static_cast<std::size_t>(
                    component)]);
        }
        out << "],\"values\":[";
        for (int component = 0; component < dimension; ++component) {
            if (component != 0) {
                out << ',';
            }
            out << sharedVelocityValueArtifact(
                boundary.value[static_cast<std::size_t>(component)]);
        }
        out << "]}";
    }
    out << "]"
        << ",\"negative_phase_local_velocity_dirichlet_count\":"
        << options.negative_phase.velocity_dirichlet.size()
        << ",\"positive_phase_local_velocity_dirichlet_count\":"
        << options.positive_phase.velocity_dirichlet.size() << '}';
}

[[nodiscard]] int resolvedInterfaceMarker(
    const IncompressibleTwoFluidOptions& options,
    FE::FieldId level_set_field)
{
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source =
        FE::interfaces::LevelSetInterfaceSource::fromField(level_set_field);
    key.domain_id = options.generated_interface_domain_id;
    key.isovalue = options.level_set_isovalue;
    key.requested_marker = options.interface_marker;
    return FE::interfaces::stableGeneratedInterfaceMarker(key);
}

void addSharedGaugeEvidence(FE::systems::FESystem& system,
                            FE::FieldId negative_pressure,
                            FE::FieldId positive_pressure)
{
    constexpr std::string_view source =
        "Coupled two-fluid shared pressure gauge constraint";
    for (const auto field : {negative_pressure, positive_pressure}) {
        system.gaugeRegistry().addAnchoring(FE::gauge::AnchoringEvidence{
            .field = field,
            .component = -1,
            .region = -1,
            .family = FE::gauge::NullspaceModeFamily::ScalarConstant,
            .verdict = FE::gauge::AnchoringVerdict::Anchored,
            .source = std::string(source),
        });
    }
}

[[nodiscard]] IncompressibleNavierStokesVMSOptions makePhaseOptions(
    const IncompressibleTwoFluidOptions& owner,
    const IncompressibleTwoFluidPhaseOptions& phase,
    FreeSurfaceActiveDomain active_domain,
    int interface_marker)
{
    IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = phase.velocity_field_name;
    options.pressure_field_name = phase.pressure_field_name;
    options.operator_tag = owner.operator_tag;
    options.density = phase.density;
    options.viscosity = phase.viscosity;
    options.body_force = owner.body_force;
    options.enable_convection = owner.enable_convection;
    options.enable_vms = owner.enable_vms;
    options.ct_m = owner.ct_m;
    options.ct_c = owner.ct_c;
    options.stabilization_epsilon = owner.stabilization_epsilon;
    options.jit_policy = owner.jit_policy;
    options.velocity_dirichlet = phase.velocity_dirichlet;
    options.velocity_dirichlet.insert(
        options.velocity_dirichlet.end(),
        owner.shared_velocity_dirichlet.begin(),
        owner.shared_velocity_dirichlet.end());

    IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary interface;
    interface.role =
        FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume;
    interface.implementation =
        FreeSurfaceImplementation::UnfittedLevelSet;
    interface.boundary_marker = -1;
    interface.interface_marker = interface_marker;
    interface.level_set_field_name = owner.level_set_field_name;
    interface.generated_interface_domain_id =
        owner.generated_interface_domain_id;
    interface.generated_interface_geometry =
        owner.generated_interface_geometry;
    interface.geometry_tangent_policy = owner.geometry_tangent_policy;
    interface.level_set_isovalue = owner.level_set_isovalue;
    interface.active_domain = active_domain;
    interface.active_domain_method =
        FreeSurfaceActiveDomainMethod::CutVolume;
    interface.active_domain_smoothing_width = FE::Real{0.0};
    interface.allow_full_domain_unfitted_free_surface = false;
    interface.external_pressure = FE::Real{0.0};
    interface.surface_tension = FE::Real{0.0};
    interface.surface_tension_form =
        FreeSurfaceSurfaceTensionForm::Automatic;
    interface.curvature = FE::Real{0.0};
    interface.curvature_field_name.clear();
    interface.use_current_geometry_curvature = false;
    interface.use_level_set_curvature = false;
    interface.normal_kinematic_policy =
        FreeSurfaceNormalKinematicPolicy::None;
    interface.tangential_mesh_policy =
        FreeSurfaceTangentialMeshPolicy::Free;
    interface.kinematic_enforcement =
        FreeSurfaceKinematicEnforcement::None;
    interface.cut_cell_stabilization.enabled = true;
    interface.cut_cell_stabilization.pressure_gradient_penalty =
        owner.pressure_gradient_penalty;
    interface.cut_cell_stabilization.pressure_policy =
        FreeSurfacePressureStabilizationPolicy::Enabled;
    interface.cut_cell_stabilization.use_cut_metadata_scale =
        owner.use_cut_metadata_scale;
    interface.cut_cell_stabilization.cut_metadata_scale_cap =
        owner.cut_metadata_scale_cap;
    interface.velocity_extension.enabled = false;
    interface.contact_lines.clear();
    interface.small_cut_aggregation = true;
    interface.small_cut_aggregation_guards =
        owner.small_cut_aggregation_guards;
    options.free_surface.push_back(std::move(interface));
    return options;
}

[[nodiscard]] EffectiveConfigurationArtifact makeArtifact(
    const IncompressibleTwoFluidOptions& options,
    int dimension,
    int interface_marker,
    const IncompressibleTwoFluidInterfaceWeights& weights)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"artifact_schema_version\":3"
        << ",\"component\":\"incompressible_two_fluid\""
        << ",\"capability_label\":\"incompressible_two_phase_sharp_interface_initial_envelope\""
        << ",\"momentum_operator\":"
        << jsonString(options.enable_convection ? "navier_stokes" : "stokes")
        << ",\"fields\":{\"negative_velocity\":"
        << jsonString(options.negative_phase.velocity_field_name)
        << ",\"negative_pressure\":"
        << jsonString(options.negative_phase.pressure_field_name)
        << ",\"positive_velocity\":"
        << jsonString(options.positive_phase.velocity_field_name)
        << ",\"positive_pressure\":"
        << jsonString(options.positive_phase.pressure_field_name)
        << ",\"level_set\":" << jsonString(options.level_set_field_name)
        << ",\"operator\":" << jsonString(options.operator_tag)
        << ",\"dimension\":" << dimension << '}'
        << ",\"material\":{\"negative_density\":"
        << jsonReal(options.negative_phase.density)
        << ",\"negative_viscosity\":"
        << jsonReal(options.negative_phase.viscosity)
        << ",\"positive_density\":"
        << jsonReal(options.positive_phase.density)
        << ",\"positive_viscosity\":"
        << jsonReal(options.positive_phase.viscosity) << '}'
        << ",\"body_force\":[";
    for (int component = 0; component < dimension; ++component) {
        if (component != 0) {
            out << ',';
        }
        out << jsonReal(
            options.body_force[static_cast<std::size_t>(component)]);
    }
    out << ']'
        << ",\"hydrostatic_balance_diagnostic\":\"phasewise_integrated_pressure_gradient_minus_density_body_force\""
        << ",\"interface\":{\"marker\":" << interface_marker
        << ",\"domain\":"
        << jsonString(options.generated_interface_domain_id)
        << ",\"isovalue\":" << jsonReal(options.level_set_isovalue)
        << ",\"geometry\":"
        << jsonString(options.generated_interface_geometry)
        << ",\"geometry_tangent_policy\":"
        << jsonString(options.geometry_tangent_policy)
        << ",\"surface_tension\":" << jsonReal(options.surface_tension)
        << ",\"prescribed_pressure_jump_mode\":\"manufactured_normal_traction_and_diagnostic_target\""
        << ",\"prescribed_pressure_jump_applicable\":"
        << jsonBool(options.prescribed_pressure_jump.has_value());
    if (options.prescribed_pressure_jump.has_value()) {
        out << ",\"prescribed_pressure_jump\":"
            << jsonReal(*options.prescribed_pressure_jump);
    }
    out << ",\"prescribed_viscous_traction_jump_mode\":\"manufactured_global_viscous_traction_and_composed_stress_target\""
        << ",\"prescribed_viscous_traction_jump_applicable\":"
        << jsonBool(options.prescribed_viscous_traction_jump.has_value());
    if (options.prescribed_viscous_traction_jump.has_value()) {
        out << ",\"prescribed_viscous_traction_jump\":[";
        for (int component = 0; component < dimension; ++component) {
            if (component != 0) {
                out << ',';
            }
            out << jsonReal(
                (*options.prescribed_viscous_traction_jump)
                    [static_cast<std::size_t>(component)]);
        }
        out << ']';
    }
    out
        << ",\"nitsche_gamma\":"
        << jsonReal(options.interface_nitsche_gamma)
        << ",\"transient_penalty\":"
        << jsonBool(options.include_transient_interface_penalty)
        << ",\"negative_traction_weight\":"
        << jsonReal(weights.negative_traction)
        << ",\"positive_traction_weight\":"
        << jsonReal(weights.positive_traction)
        << ",\"harmonic_viscosity\":"
        << jsonReal(weights.harmonic_viscosity)
        << ",\"harmonic_density\":"
        << jsonReal(weights.harmonic_density) << '}'
        << ",\"stabilization\":{\"phasewise_vms_pspg\":"
        << jsonBool(options.enable_vms)
        << ",\"phasewise_pressure_ghost_penalty\":true"
        << ",\"phasewise_small_cut_aggregation\":true"
        << ",\"pressure_gradient_penalty\":"
        << jsonReal(options.pressure_gradient_penalty)
        << ",\"use_cut_metadata_scale\":"
        << jsonBool(options.use_cut_metadata_scale)
        << ",\"cut_metadata_scale_cap\":";
    if (options.cut_metadata_scale_cap.has_value()) {
        out << jsonReal(*options.cut_metadata_scale_cap);
    } else {
        out << "null";
    }
    out << '}';
    appendVelocityBoundaryArtifact(out, options, dimension);
    out << ",\"pressure_space\":{\"representation\":\"separate_phase_fields\",\"shared_gauge_count\":1";
    if (options.shared_pressure_gauge.has_value()) {
        out << ",\"shared_gauge_policy\":\"explicit_global_vertex_gid\""
            << ",\"shared_gauge_field\":"
            << jsonString(options.negative_phase.pressure_field_name)
            << ",\"shared_gauge_id_type\":\"Global_vertex_gid\""
            << ",\"shared_gauge_vertex_gid\":"
            << options.shared_pressure_gauge->vertex_gid
            << ",\"shared_gauge_value\":"
            << jsonReal(options.shared_pressure_gauge->pressure);
    } else {
        out << ",\"shared_gauge_policy\":\"automatic_first_unconstrained_pressure_dof\""
            << ",\"shared_gauge_field\":null"
            << ",\"shared_gauge_id_type\":null"
            << ",\"shared_gauge_vertex_gid\":null"
            << ",\"shared_gauge_value\":0";
    }
    out << '}'
        << ",\"solver_contract\":{\"backend\":\"FSILS\",\"method\":\"BlockSchur\",\"unknown_role_order\":[\"material_interface_level_set\",\"conservative_phase_indicator\",\"negative_velocity\",\"positive_velocity\",\"negative_pressure\",\"positive_pressure\"],\"blocks\":[{\"name\":\"TwoFluidMaterialInterfaceComputationalPrimary\",\"role\":\"primary\"},{\"name\":\"TwoFluidPressureConstraints\",\"role\":\"constraint\"}],\"generic_fallback_allowed\":false}"
        << ",\"phase_transport_coupling\":{\"owner\":\"declared_material_interface_phase_pair\",\"weak_bulk_velocity\":\"sharp_phase_local\",\"weak_interface_trace\":\"complementary_weighted\",\"conservative_graph_velocity\":\"complementary_weighted_every_node\",\"momentum_reconciliation_required\":"
        << jsonBool(
               options.require_conservative_phase_momentum_reconciliation)
        << ",\"momentum_reconciliation_policy\":\"phasewise_fail_closed_no_hidden_velocity_update\"}"
        << ",\"accepted_stage_evidence\":{\"required\":true,\"operator_stage_interface_mass_momentum_flux\":true,\"canonical_phase_aggregation\":true,\"pressure_stabilization_configuration\":true,\"shared_coupled_nonlinear_linear_report\":true,\"timing_values_retained\":false,\"rank_local_aggregation_lineage_retained\":false,\"phase_resolved_iterations\":\"unavailable_from_shared_backend\",\"pressure_stabilization_work\":\"not_separately_resolved_by_coupled_operator\"}"
        << ",\"exclusions\":[\"compressible_gas\",\"trapped_gas_pressure\",\"air_cushioning\",\"phase_change\",\"contact\",\"moving_mesh\",\"variable_material_laws\",\"turbulence\"]}";
    return EffectiveConfigurationArtifact{
        .component = "incompressible_two_fluid",
        .json = out.str(),
    };
}

} // namespace

void validateIncompressibleTwoFluidConfigurationSemantics(
    const IncompressibleTwoFluidOptions& options)
{
    for (const auto acceleration : options.body_force) {
        if (!std::isfinite(acceleration)) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_nonfinite_body_force");
        }
    }
    if (options.prescribed_pressure_jump.has_value() &&
        !std::isfinite(*options.prescribed_pressure_jump)) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_nonfinite_prescribed_pressure_jump");
    }
    if (options.prescribed_viscous_traction_jump.has_value() &&
        !std::all_of(
            options.prescribed_viscous_traction_jump->begin(),
            options.prescribed_viscous_traction_jump->end(),
            [](FE::Real value) { return std::isfinite(value); })) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_nonfinite_prescribed_viscous_traction_jump");
    }
    if (options.shared_pressure_gauge.has_value() &&
        options.shared_pressure_gauge->vertex_gid < 0) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_negative_shared_pressure_gauge_vertex_gid");
    }
    if (options.shared_pressure_gauge.has_value() &&
        !std::isfinite(options.shared_pressure_gauge->pressure)) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_nonfinite_shared_pressure_gauge_value");
    }
    validateVelocityBoundaryData(
        options.negative_phase.velocity_dirichlet,
        "negative-phase",
        VelocityBoundaryDataPolicy::HomogeneousPhaseLocal);
    validateVelocityBoundaryData(
        options.positive_phase.velocity_dirichlet,
        "positive-phase",
        VelocityBoundaryDataPolicy::HomogeneousPhaseLocal);
    validateVelocityBoundaryData(
        options.shared_velocity_dirichlet,
        "shared",
        VelocityBoundaryDataPolicy::SharedExternalData);
    validateNoSharedVelocityBoundaryOverlap(
        options.shared_velocity_dirichlet,
        options.negative_phase.velocity_dirichlet,
        "negative-phase");
    validateNoSharedVelocityBoundaryOverlap(
        options.shared_velocity_dirichlet,
        options.positive_phase.velocity_dirichlet,
        "positive-phase");
}

IncompressibleTwoFluidModule::IncompressibleTwoFluidModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> negative_velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> negative_pressure_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> positive_velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> positive_pressure_space,
    IncompressibleTwoFluidOptions options)
    : negative_velocity_space_(std::move(negative_velocity_space)),
      negative_pressure_space_(std::move(negative_pressure_space)),
      positive_velocity_space_(std::move(positive_velocity_space)),
      positive_pressure_space_(std::move(positive_pressure_space)),
      options_(std::move(options))
{
}

void IncompressibleTwoFluidModule::registerOn(
    FE::systems::FESystem& system) const
{
    effective_configuration_artifact_.reset();
    validateVelocitySpace(negative_velocity_space_, "negative-phase");
    validateVelocitySpace(positive_velocity_space_, "positive-phase");
    validatePressureSpace(
        negative_pressure_space_, *negative_velocity_space_, "negative-phase");
    validatePressureSpace(
        positive_pressure_space_, *positive_velocity_space_, "positive-phase");
    const auto dimension = negative_velocity_space_->value_dimension();
    if (positive_velocity_space_->value_dimension() != dimension ||
        !spacesCompatible(*negative_velocity_space_,
                          *positive_velocity_space_) ||
        !spacesCompatible(*negative_pressure_space_,
                          *positive_pressure_space_)) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: both phases must use matching velocity and pressure spaces");
    }
    if (dimension == 2 && options_.body_force[2] != FE::Real{0.0}) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_out_of_plane_body_force");
    }
    if (options_.prescribed_viscous_traction_jump.has_value() &&
        dimension == 2 &&
        (*options_.prescribed_viscous_traction_jump)[2] != FE::Real{0.0}) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_out_of_plane_prescribed_viscous_traction_jump");
    }

    const std::array<std::string_view, 5> field_names{
        options_.negative_phase.velocity_field_name,
        options_.negative_phase.pressure_field_name,
        options_.positive_phase.velocity_field_name,
        options_.positive_phase.pressure_field_name,
        options_.level_set_field_name,
    };
    for (const auto name : field_names) {
        if (name.empty()) {
            throw std::invalid_argument(
                "IncompressibleTwoFluidModule: field names must be nonempty");
        }
    }
    for (std::size_t i = 0; i < field_names.size(); ++i) {
        for (std::size_t j = i + 1u; j < field_names.size(); ++j) {
            if (field_names[i] == field_names[j]) {
                throw std::invalid_argument(
                    "IncompressibleTwoFluidModule: all phase and level-set field names must be distinct");
            }
        }
    }
    if (options_.operator_tag.empty() ||
        options_.generated_interface_domain_id.empty()) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: operator tag and generated-interface domain must be nonempty");
    }
    if (options_.generated_interface_geometry != "LinearCorner" ||
        options_.geometry_tangent_policy !=
            "RefreshedFrozenQuadrature") {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: the initial envelope requires LinearCorner generated geometry with RefreshedFrozenQuadrature tangents");
    }
    if (!std::isfinite(options_.level_set_isovalue) ||
        !std::isfinite(options_.surface_tension) ||
        options_.surface_tension < FE::Real{0.0}) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: level-set isovalue and nonnegative surface tension must be finite");
    }
    if (!options_.enable_vms) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: the affine equal-order initial envelope requires phasewise VMS/PSPG stabilization");
    }
    requireFinitePositive(
        options_.negative_phase.density, "negative density");
    requireFinitePositive(
        options_.positive_phase.density, "positive density");
    requireFinitePositive(
        options_.negative_phase.viscosity, "negative viscosity");
    requireFinitePositive(
        options_.positive_phase.viscosity, "positive viscosity");
    requireFinitePositive(
        options_.interface_nitsche_gamma, "interface Nitsche gamma");
    requireFinitePositive(options_.ct_m, "VMS ct_m");
    requireFinitePositive(options_.ct_c, "VMS ct_c");
    requireFinitePositive(
        options_.stabilization_epsilon, "stabilization epsilon");
    requireFinitePositive(
        options_.pressure_gradient_penalty,
        "pressure-gradient ghost penalty");
    if (options_.cut_metadata_scale_cap.has_value() &&
        (!std::isfinite(*options_.cut_metadata_scale_cap) ||
         *options_.cut_metadata_scale_cap < FE::Real{1.0})) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: cut metadata scale cap must be finite and at least one");
    }
    const auto& guards = options_.small_cut_aggregation_guards;
    if (guards.maximum_root_path_length == 0u ||
        !std::isfinite(guards.maximum_reference_extrapolation_distance) ||
        guards.maximum_reference_extrapolation_distance < FE::Real{0.0} ||
        !std::isfinite(guards.maximum_absolute_coefficient) ||
        guards.maximum_absolute_coefficient < FE::Real{1.0} ||
        !std::isfinite(guards.maximum_row_l1_norm) ||
        guards.maximum_row_l1_norm < guards.maximum_absolute_coefficient) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: invalid small-cut aggregation guards");
    }
    validateIncompressibleTwoFluidConfigurationSemantics(options_);

    const auto level_set =
        system.findFieldByName(options_.level_set_field_name);
    if (level_set == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: level-set field '" +
            options_.level_set_field_name +
            "' must be registered before the coupled fluid owner");
    }
    const auto& level_set_record = system.fieldRecord(level_set);
    const bool supported_level_set_source =
        level_set_record.source_kind ==
            FE::systems::FieldSourceKind::Unknown ||
        level_set_record.source_kind ==
            FE::systems::FieldSourceKind::PrescribedData;
    if (!supported_level_set_source || level_set_record.components != 1 ||
        !level_set_record.space ||
        level_set_record.space->space_type() !=
            FE::spaces::SpaceType::H1 ||
        !spacesCompatible(*level_set_record.space,
                          *negative_pressure_space_)) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: level-set field must be a prescribed or unknown scalar field in the matching affine C0 P1 space");
    }

    validateExistingUnknownField(
        system,
        options_.negative_phase.velocity_field_name,
        *negative_velocity_space_,
        dimension);
    validateExistingUnknownField(
        system,
        options_.negative_phase.pressure_field_name,
        *negative_pressure_space_,
        1);
    validateExistingUnknownField(
        system,
        options_.positive_phase.velocity_field_name,
        *positive_velocity_space_,
        dimension);
    validateExistingUnknownField(
        system,
        options_.positive_phase.pressure_field_name,
        *positive_pressure_space_,
        1);

    const int interface_marker =
        resolvedInterfaceMarker(options_, level_set);
    const IncompressibleTwoFluidInterfaceParameters interface_parameters{
        .dimension = dimension,
        .interface_marker = interface_marker,
        .negative_density = options_.negative_phase.density,
        .positive_density = options_.positive_phase.density,
        .negative_viscosity = options_.negative_phase.viscosity,
        .positive_viscosity = options_.positive_phase.viscosity,
        .nitsche_gamma = options_.interface_nitsche_gamma,
        .surface_tension = options_.surface_tension,
        .include_transient_penalty =
            options_.include_transient_interface_penalty,
        .prescribed_pressure_jump = options_.prescribed_pressure_jump,
        .prescribed_viscous_traction_jump =
            options_.prescribed_viscous_traction_jump,
    };
    const auto interface_weights =
        incompressibleTwoFluidInterfaceWeights(interface_parameters);

    auto negative_options = makePhaseOptions(
        options_,
        options_.negative_phase,
        FreeSurfaceActiveDomain::LevelSetNegative,
        interface_marker);
    auto positive_options = makePhaseOptions(
        options_,
        options_.positive_phase,
        FreeSurfaceActiveDomain::LevelSetPositive,
        interface_marker);

    IncompressibleNavierStokesVMSModule negative_module(
        negative_velocity_space_,
        negative_pressure_space_,
        std::move(negative_options));
    IncompressibleNavierStokesVMSModule positive_module(
        positive_velocity_space_,
        positive_pressure_space_,
        std::move(positive_options));
    negative_module.registerOn(system);
    positive_module.registerOn(system);

    const auto u_negative = system.findFieldByName(
        options_.negative_phase.velocity_field_name);
    const auto p_negative = system.findFieldByName(
        options_.negative_phase.pressure_field_name);
    const auto u_positive = system.findFieldByName(
        options_.positive_phase.velocity_field_name);
    const auto p_positive = system.findFieldByName(
        options_.positive_phase.pressure_field_name);
    if (u_negative == FE::INVALID_FIELD_ID ||
        p_negative == FE::INVALID_FIELD_ID ||
        u_positive == FE::INVALID_FIELD_ID ||
        p_positive == FE::INVALID_FIELD_ID) {
        throw std::logic_error(
            "IncompressibleTwoFluidModule: phase registration did not create all four unknown fields");
    }

    using namespace FE::forms;
    const auto forms = buildIncompressibleTwoFluidInterfaceForms(
        StateField(u_negative,
                   *negative_velocity_space_,
                   options_.negative_phase.velocity_field_name),
        StateField(p_negative,
                   *negative_pressure_space_,
                   options_.negative_phase.pressure_field_name),
        TestField(u_negative, *negative_velocity_space_, "v_negative"),
        TestField(p_negative, *negative_pressure_space_, "q_negative"),
        StateField(u_positive,
                   *positive_velocity_space_,
                   options_.positive_phase.velocity_field_name),
        StateField(p_positive,
                   *positive_pressure_space_,
                   options_.positive_phase.pressure_field_name),
        TestField(u_positive, *positive_velocity_space_, "v_positive"),
        TestField(p_positive, *positive_pressure_space_, "q_positive"),
        interface_parameters);

    auto install = physicsInstallOptions(options_.jit_policy);
    install.compiler_options.use_symbolic_tangent = true;
    install.source_component_tag =
        options_.operator_tag + "_two_fluid_material_interface";
    install.recordDynamicViscosity(
        u_negative,
        options_.negative_phase.viscosity,
        {},
        "negative_dynamic_viscosity");
    install.recordDynamicViscosity(
        u_positive,
        options_.positive_phase.viscosity,
        {},
        "positive_dynamic_viscosity");
    const std::array<FE::FieldId, 4> fields{
        u_negative, p_negative, u_positive, p_positive};
    (void)FE::systems::installFormulation(
        system,
        options_.operator_tag,
        std::span<const FE::FieldId>(fields),
        forms.residual,
        install);

    if (options_.shared_pressure_gauge.has_value()) {
        const auto& gauge = *options_.shared_pressure_gauge;
        std::vector<FE::constraints::VertexDirichletValue> values{
            FE::constraints::VertexDirichletValue{
                .vertex_id = gauge.vertex_gid,
                .value = gauge.pressure,
            }};
        system.addSystemConstraint(
            std::make_unique<FE::constraints::VertexDirichletConstraint>(
                p_negative,
                std::move(values),
                FE::constraints::VertexIdMode::GlobalVertexGid));
    } else {
        system.addSystemConstraint(
            std::make_unique<FE::constraints::CoupledFieldGaugeConstraint>(
                p_negative, p_positive));
    }
    addSharedGaugeEvidence(system, p_negative, p_positive);

    system.declareMaterialInterfaceTransportVelocity(
        FE::interfaces::MaterialInterfaceTransportVelocityDeclaration{
            .dimension = dimension,
            .interface_marker = interface_marker,
            .level_set_field = level_set,
            .negative_velocity_field = u_negative,
            .positive_velocity_field = u_positive,
            .level_set_isovalue = options_.level_set_isovalue,
            .negative_trace_weight =
                interface_weights.negative_complement,
            .positive_trace_weight =
                interface_weights.positive_complement,
            .geometry_domain_id = options_.generated_interface_domain_id,
            .owner_component = "incompressible_two_fluid",
        });

    system.declareTwoFluidAcceptedStageDiagnostics(
        FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration{
            .interface_marker = interface_marker,
            .level_set_field = level_set,
            .negative_velocity_field = u_negative,
            .negative_pressure_field = p_negative,
            .positive_velocity_field = u_positive,
            .positive_pressure_field = p_positive,
            .operator_tag = options_.operator_tag,
            .geometry_domain_id = options_.generated_interface_domain_id,
            .level_set_isovalue = options_.level_set_isovalue,
            .require_conservative_phase_momentum_reconciliation =
                options_
                    .require_conservative_phase_momentum_reconciliation,
            .require_accepted_stage_numerics = true,
            .pressure_stabilization_coefficient =
                options_.pressure_gradient_penalty,
            .pressure_stabilization_use_cut_metadata_scale =
                options_.use_cut_metadata_scale,
            .pressure_stabilization_cut_metadata_scale_cap =
                options_.cut_metadata_scale_cap,
            .parameters =
                FE::interfaces::IncompressibleTwoFluidDiagnosticParameters{
                    .dimension = dimension,
                    .interface_marker = interface_marker,
                    .negative_density = options_.negative_phase.density,
                    .positive_density = options_.positive_phase.density,
                    .negative_viscosity = options_.negative_phase.viscosity,
                    .positive_viscosity = options_.positive_phase.viscosity,
                    .nitsche_gamma = options_.interface_nitsche_gamma,
                    .surface_tension = options_.surface_tension,
                    .include_transient_penalty =
                        options_.include_transient_interface_penalty,
                    .prescribed_pressure_jump =
                        options_.prescribed_pressure_jump,
                    .prescribed_viscous_traction_jump =
                        options_.prescribed_viscous_traction_jump,
                    .body_force = options_.body_force,
                },
            .owner_component = "incompressible_two_fluid",
        });

    effective_configuration_artifact_ = makeArtifact(
        options_, dimension, interface_marker, interface_weights);
}

void IncompressibleTwoFluidModule::applyInitialConditions(
    const FE::systems::FESystem& system,
    FE::backends::GenericVector& u0) const
{
    const auto level_set =
        system.findFieldByName(options_.level_set_field_name);
    if (level_set == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: initial conditions require the registered level-set field '" +
            options_.level_set_field_name + "'");
    }
    const auto interface_marker =
        resolvedInterfaceMarker(options_, level_set);
    auto negative_options = makePhaseOptions(
        options_,
        options_.negative_phase,
        FreeSurfaceActiveDomain::LevelSetNegative,
        interface_marker);
    auto positive_options = makePhaseOptions(
        options_,
        options_.positive_phase,
        FreeSurfaceActiveDomain::LevelSetPositive,
        interface_marker);

    IncompressibleNavierStokesVMSModule negative_module(
        negative_velocity_space_,
        negative_pressure_space_,
        std::move(negative_options));
    IncompressibleNavierStokesVMSModule positive_module(
        positive_velocity_space_,
        positive_pressure_space_,
        std::move(positive_options));
    negative_module.applyInitialConditions(system, u0);
    positive_module.applyInitialConditions(system, u0);
}

std::optional<EffectiveConfigurationArtifact>
IncompressibleTwoFluidModule::effectiveConfigurationArtifact() const
{
    return effective_configuration_artifact_;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
