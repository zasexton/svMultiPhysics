/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#include "Physics/Options/NavierStokesOptionSchema.h"

#include "Physics/Options/FreeSurfaceProfiles.h"
#include "Physics/Options/InputReaders.h"
#include "Physics/Options/NavierStokesRegistration.h"

#include "FE/Systems/FESystem.h"
#include "Physics/Options/NavierStokesOptionRules.h"
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
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/SystemCollectives.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
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

namespace json = FE::options::json;

namespace {

namespace opt = svmp::FE::options;
using ScalarValue = FE::forms::bc::ScalarValue;
using Guards = FE::constraints::SmallCutAggregationGuardOptions;

constexpr double kPi = 3.14159265358979323846;

/// A real key of the register: parsed with std::stod and the whole text
/// consumed; non-finite values pass unless the register checks them.
opt::RealCodec anyReal()
{
    return opt::RealCodec{.require_finite = false};
}

ScalarValueCodec anyScalar()
{
    return ScalarValueCodec{.real = anyReal()};
}

/// The register reads every boolean of the boundary with parse_bool_relaxed.
opt::BoolCodec relaxedBool()
{
    return opt::BoolCodec{opt::BoolLexicon::Relaxed};
}

template <class E>
opt::EnumToken<E> token(E value, std::string canonical, std::vector<std::string> synonyms = {})
{
    return opt::EnumToken<E>{value, std::move(canonical), std::move(synonyms)};
}

opt::KeySettings doc(std::string text)
{
    return opt::KeySettings{.doc = std::move(text)};
}

/// An optional real that the input can only set: "none" is not accepted and
/// an empty value is an error, as when the register read it with std::stod.
struct SetOptionalRealCodec {
    using value_type = std::optional<FE::Real>;
    static constexpr opt::EmptyPolicy empty_policy = opt::EmptyPolicy::Parse;
    opt::RealCodec real{};

    [[nodiscard]] value_type parse(std::string_view text, std::string& error) const
    {
        return real.parse(text, error);
    }
    [[nodiscard]] std::string format(const value_type& value) const
    {
        return value ? real.format(*value) : std::string("none");
    }
    [[nodiscard]] std::string describe() const { return "optional " + real.describe(); }
};

opt::OptionSchema<FreeSurfaceBoundary> makeFreeSurfaceBoundarySchema()
{
    using B = FreeSurfaceBoundary;
    opt::OptionSchema<B> schema("Free_surface");
    schema
        .key("Implementation", &B::implementation,
             opt::EnumCodec<FreeSurfaceImplementation>{{
                 token(FreeSurfaceImplementation::FittedALE, "FittedALE", {"fitted", "ale"}),
                 token(FreeSurfaceImplementation::UnfittedLevelSet, "UnfittedLevelSet",
                       {"unfitted", "levelset", "embeddedlevelset"}),
             }},
             doc("fitted ALE boundary or unfitted level-set interface"))
        .key("Active_domain", &B::active_domain,
             opt::EnumCodec<FreeSurfaceActiveDomain>{{
                 token(FreeSurfaceActiveDomain::None, "None", {"disabled", "off", "inactive"}),
                 token(FreeSurfaceActiveDomain::LevelSetNegative, "LevelSetNegative",
                       {"negative", "negativelevelset", "phinegative"}),
                 token(FreeSurfaceActiveDomain::LevelSetPositive, "LevelSetPositive",
                       {"positive", "positivelevelset", "phipositive"}),
             }},
             doc("liquid side of the level set"))
        .key("Interface_marker", &B::interface_marker, opt::IntCodec{})
        .key("Level_set_field_name", &B::level_set_field_name, opt::StringCodec{})
        .key("Generated_interface_domain_id", &B::generated_interface_domain_id, opt::StringCodec{})
        // The spellings of the application's cut request (LevelSetCutConfiguration).
        .key("Generated_interface_geometry", &B::generated_interface_geometry,
             opt::EnumCodec<FreeSurfaceGeneratedInterfaceGeometry>{{
                 token(FreeSurfaceGeneratedInterfaceGeometry::LinearCorner, "LinearCorner",
                       {"cornerlinear", "linear", "legacy"}),
                 token(FreeSurfaceGeneratedInterfaceGeometry::HighOrderImplicit, "HighOrderImplicit",
                       {"highorder", "curvedimplicit", "implicitcurved"}),
             }})
        .key("Geometry_tangent_policy", &B::geometry_tangent_policy,
             opt::EnumCodec<FreeSurfaceGeometryTangentPolicy>{{
                 token(FreeSurfaceGeometryTangentPolicy::RefreshedFrozenQuadrature,
                       "RefreshedFrozenQuadrature",
                       {"refreshedfrozen", "frozenquadrature", "quasinewton", "quasinewtongeometry"}),
                 token(FreeSurfaceGeometryTangentPolicy::DifferentiatedQuadrature,
                       "DifferentiatedQuadrature",
                       {"differentiated", "exactgeometrytangent", "exactsensitivities",
                        "shapederivative"}),
             }})
        .key("Level_set_isovalue", &B::level_set_isovalue, anyReal())
        .key("External_pressure", &B::external_pressure, anyScalar(),
             doc("prescribed exterior pressure"))
        .key("Surface_tension", &B::surface_tension, anyScalar(), doc("surface tension gamma"))
        .key("Surface_tension_semi_implicit", &B::surface_tension_semi_implicit,
             opt::EnumCodec<FreeSurfaceSurfaceTensionSemiImplicit>{{
                 token(FreeSurfaceSurfaceTensionSemiImplicit::None, "None", {"off", "disabled"}),
                 token(FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement, "NormalIncrement",
                       {"laggednormalincrement"}),
             }})
        .key("Tangential_mesh_policy", &B::tangential_mesh_policy,
             opt::EnumCodec<FreeSurfaceTangentialMeshPolicy>{{
                 token(FreeSurfaceTangentialMeshPolicy::Free, "Free", {"freetangential", "unconstrained"}),
                 token(FreeSurfaceTangentialMeshPolicy::SmoothingOnly, "SmoothingOnly",
                       {"smoothing", "meshsmoothing", "smooth"}),
                 token(FreeSurfaceTangentialMeshPolicy::Prescribed, "Prescribed",
                       {"prescribedtangential", "prescribedvelocity"}),
             }})
        .key("Prescribed_tangential_mesh_velocity", &B::prescribed_tangential_mesh_velocity,
             ScalarVector3Codec{})
        .key("Tangential_mesh_penalty", &B::tangential_mesh_penalty,
             ScalarValueCodec{.real = opt::RealCodec{.minimum = 0.0, .open_minimum = true}})
        .key("Kinematic_nitsche_gamma", &B::kinematic_nitsche_gamma, anyReal())
        .key("Small_cut_aggregation", &B::small_cut_aggregation, relaxedBool())
        .key("Small_cut_aggregation_maximum_root_path_length",
             opt::nestedMember(&B::small_cut_aggregation_guards, &Guards::maximum_root_path_length),
             opt::IntCodec{.minimum = 1})
        .key("Small_cut_aggregation_maximum_reference_extrapolation_distance",
             opt::nestedMember(&B::small_cut_aggregation_guards,
                               &Guards::maximum_reference_extrapolation_distance),
             opt::RealCodec{.minimum = 0.0})
        .key("Small_cut_aggregation_maximum_absolute_coefficient",
             opt::nestedMember(&B::small_cut_aggregation_guards, &Guards::maximum_absolute_coefficient),
             opt::RealCodec{.minimum = 1.0})
        .key("Small_cut_aggregation_maximum_row_l1_norm",
             opt::nestedMember(&B::small_cut_aggregation_guards, &Guards::maximum_row_l1_norm),
             opt::RealCodec{.minimum = 1.0})
        .key("Small_cut_aggregation_rootless_fallback",
             opt::nestedMember(&B::small_cut_aggregation_guards, &Guards::rootless_fallback),
             relaxedBool())
        .key("Enable_cut_cell_stabilization",
             opt::nestedMember(&B::cut_cell_stabilization, &FreeSurfaceCutCellStabilization::enabled),
             relaxedBool())
        .key("Cut_cell_pressure_gradient_penalty",
             opt::nestedMember(&B::cut_cell_stabilization,
                               &FreeSurfaceCutCellStabilization::pressure_gradient_penalty),
             anyScalar())
        // Defaults that depend on another key (the register's resolution).
        // D47 item 3: an unfitted surface is liquid on the negative side of
        // the level set unless Active_domain says otherwise.
        .implies("Implementation", "Active_domain", "LevelSetNegative",
                 opt::ImplicationSettings{.when_trigger_is = "UnfittedLevelSet"})
        .implies("Prescribed_tangential_mesh_velocity", "Tangential_mesh_policy", "Prescribed")
        .implies("Cut_cell_pressure_gradient_penalty", "Enable_cut_cell_stabilization", "true");
    return schema;
}

opt::OptionSchema<FreeSurfaceContactLineInputView> makeContactLineInputSchema()
{
    using V = FreeSurfaceContactLineInputView;
    opt::OptionSchema<V> schema("Free_surface contact line");
    schema
        .key("Contact_line_model", &V::model,
             opt::EnumCodec<FreeSurfaceContactLineModelInput>{{
                 token(FreeSurfaceContactLineModelInput::None, "None", {"disabled", "off"}),
                 token(FreeSurfaceContactLineModelInput::Pinned, "Pinned", {"fixed", "fixedposition"}),
                 token(FreeSurfaceContactLineModelInput::PrescribedAngle, "PrescribedAngle",
                       {"prescribedcontactangle", "contactangle"}),
                 token(FreeSurfaceContactLineModelInput::DynamicRenE, "DynamicRenE",
                       {"dynamiccontactangle", "dynamicangle", "rene"}),
             }},
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Contact_angle_degrees", &V::contact_angle_degrees, anyReal(),
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Contact_line_wall_markers", &V::wall_markers, opt::ListCodec<opt::IntCodec>{},
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Contact_line_marker", &V::contact_line_marker, opt::IntCodec{},
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Contact_line_wall_normal", &V::wall_normals, opt::ListCodec<opt::Vector3Codec>{},
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Contact_line_mobility", &V::mobility, anyScalar(),
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Wall_slip_model", &V::wall_slip_model,
             opt::EnumCodec<FreeSurfaceWallSlipModelInput>{{
                 token(FreeSurfaceWallSlipModelInput::None, "None", {"noslip", "disabled", "off"}),
                 token(FreeSurfaceWallSlipModelInput::Navier, "Navier", {"navierslip", "slip"}),
             }},
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Wall_slip_length", &V::wall_slip_length, anyScalar(),
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Contact_line_law_enforcement", &V::law_enforcement,
             opt::EnumCodec<FreeSurfaceContactLawEnforcementInput>{{
                 token(FreeSurfaceContactLawEnforcementInput::Weak, "Weak"),
                 token(FreeSurfaceContactLawEnforcementInput::Strong, "Strong"),
             }});
    return schema;
}

/// The keys the equation and the domain element share: the moving domain,
/// the momentum source and the rotating frame.  When an element names a key
/// twice the register kept the later spelling of the moving domain and the
/// first of the momentum source; the declared spelling order says the same.
void addSharedNavierStokesKeys(opt::OptionSchema<NavierStokesInputView>& schema)
{
    using V = NavierStokesInputView;
    schema
        .key("Enable_ALE", &V::enable_ale, relaxedBool())
        .key("Mesh_velocity_field", &V::mesh_velocity_field, opt::StringCodec{})
        .key("Mesh_velocity_source", &V::mesh_velocity_source,
             opt::EnumCodec<ALEMeshVelocitySource>{{
                 token(ALEMeshVelocitySource::PrescribedData, "prescribed_data",
                       {"prescribed", "data", "mesh_motion_data"}),
                 token(ALEMeshVelocitySource::CoupledDisplacement, "coupled_displacement",
                       {"coupled", "derived", "derived_from_displacement", "monolithic"}),
             }})
        .key("Mesh_displacement_field", &V::mesh_displacement_field, opt::StringCodec{})
        .key("Auto_register_mesh_displacement_field", &V::auto_register_mesh_displacement_field,
             relaxedBool())
        .key("Momentum_source_field_name", &V::momentum_source_field_name, opt::StringCodec{})
        .key("Auto_register_momentum_source_field", &V::auto_register_momentum_source_field,
             relaxedBool())
        .key("Momentum_source_temporal_and_spatial_values_file_path", &V::momentum_source_file,
             opt::StringCodec{})
        .key("Rotating_frame_angular_velocity", &V::rotating_frame_angular_velocity, opt::Vector3Codec{})
        .key("Rotating_frame_angular_velocity_temporal_values_file_path",
             &V::rotating_frame_angular_velocity_file, opt::StringCodec{});
}

void addPhysicalModelKey(auto& schema, auto physical_model)
{
    schema.key("Free_surface_physical_model", physical_model, opt::StringCodec{},
               opt::KeySettings{.doc = "checked first by the scope check of the register"});
}

opt::OptionSchema<NavierStokesInputView> makeNavierStokesEquationSchema()
{
    using V = NavierStokesInputView;
    opt::OptionSchema<V> schema("Navier-Stokes equation");
    addPhysicalModelKey(schema, &V::physical_model);
    schema.key("Use_taylor_hood_type_basis", &V::use_taylor_hood_type_basis, relaxedBool());
    addSharedNavierStokesKeys(schema);
    return schema;
}

opt::OptionSchema<NavierStokesInputView> makeNavierStokesDomainSchema()
{
    using V = NavierStokesInputView;
    opt::OptionSchema<V> schema("Navier-Stokes domain");
    schema
        .key("Density", &V::density, anyReal(),
             opt::KeySettings{.spellings = {{"Density", false}, {"Fluid_density", false}}})
        .key("Force_x", [](V& v) -> FE::Real& { return v.body_force[0]; }, anyReal())
        .key("Force_y", [](V& v) -> FE::Real& { return v.body_force[1]; }, anyReal())
        .key("Force_z", [](V& v) -> FE::Real& { return v.body_force[2]; }, anyReal())
        .key("Hydrostatic_pressure_initialization", &V::hydrostatic_pressure_initialization,
             relaxedBool())
        .key("Hydrostatic_pressure_reference", &V::hydrostatic_pressure_reference, anyReal())
        .key("Hydrostatic_pressure_reference_point", &V::hydrostatic_pressure_reference_point,
             opt::Vector3Codec{})
        // The register reads the parameters of the stated viscosity model
        // only, and reads them with their Parameters defaults when the deck
        // leaves them out.
        .key("Viscosity.model", &V::viscosity_model, opt::StringCodec{})
        .key("Viscosity.Value", &V::viscosity_value, anyReal())
        .key("Viscosity.Limiting_high_shear_rate_viscosity",
             &V::limiting_high_shear_rate_viscosity, anyReal())
        .key("Viscosity.Limiting_low_shear_rate_viscosity",
             &V::limiting_low_shear_rate_viscosity, anyReal())
        .key("Viscosity.Shear_rate_tensor_multiplier", &V::shear_rate_tensor_multiplier, anyReal())
        .key("Viscosity.Power_law_index", &V::power_law_index, anyReal())
        .key("Viscosity.Shear_rate_tensor_exponent", &V::shear_rate_tensor_exponent, anyReal())
        .key("Backflow_stabilization_coefficient", &V::backflow_stabilization_coefficient, anyReal(),
             opt::KeySettings{.empty = opt::EmptyPolicy::Absent});
    addSharedNavierStokesKeys(schema);
    return schema;
}

opt::OptionSchema<IncompressibleTwoFluidInputView> makeIncompressibleTwoFluidSchema()
{
    using V = IncompressibleTwoFluidInputView;
    opt::OptionSchema<V> schema("Incompressible two-fluid");
    addPhysicalModelKey(schema, &V::physical_model);
    const auto optional_real = SetOptionalRealCodec{anyReal()};
    schema
        .key("Level_set_field_name", &V::level_set_field_name, opt::StringCodec{})
        .key("Generated_interface_domain_id", &V::generated_interface_domain_id, opt::StringCodec{})
        .key("Material_interface_marker", &V::material_interface_marker, opt::IntCodec{})
        .key("Negative_phase_density", &V::negative_phase_density, anyReal())
        .key("Negative_phase_dynamic_viscosity", &V::negative_phase_dynamic_viscosity, anyReal())
        .key("Positive_phase_density", &V::positive_phase_density, anyReal())
        .key("Positive_phase_dynamic_viscosity", &V::positive_phase_dynamic_viscosity, anyReal())
        .key("Two_fluid_surface_tension", &V::two_fluid_surface_tension, anyReal())
        .key("Two_fluid_interface_nitsche_gamma", &V::two_fluid_interface_nitsche_gamma, anyReal())
        .key("Prescribed_pressure_jump", &V::prescribed_pressure_jump, optional_real)
        .key("Prescribed_viscous_traction_jump_x", &V::prescribed_viscous_traction_jump_x,
             optional_real)
        .key("Prescribed_viscous_traction_jump_y", &V::prescribed_viscous_traction_jump_y,
             optional_real)
        .key("Prescribed_viscous_traction_jump_z", &V::prescribed_viscous_traction_jump_z,
             optional_real)
        .key("Use_taylor_hood_type_basis", &V::use_taylor_hood_type_basis, relaxedBool());
    return schema;
}

opt::OptionSchema<NavierStokesInputView> makeIncompressibleTwoFluidDomainSchema()
{
    using V = NavierStokesInputView;
    opt::OptionSchema<V> schema("Incompressible two-fluid domain");
    schema
        .key("Force_x", [](V& v) -> FE::Real& { return v.body_force[0]; }, anyReal())
        .key("Force_y", [](V& v) -> FE::Real& { return v.body_force[1]; }, anyReal())
        .key("Force_z", [](V& v) -> FE::Real& { return v.body_force[2]; }, anyReal());
    return schema;
}

[[nodiscard]] FE::Real constantOr(const ScalarValue& value, FE::Real fallback)
{
    const auto* literal = std::get_if<FE::Real>(&value);
    return literal != nullptr ? *literal : fallback;
}

void appendWallNormal(FreeSurfaceContactLineInputView& view, const std::array<ScalarValue, 3>& normal)
{
    const FE::Real nan = std::numeric_limits<FE::Real>::quiet_NaN();
    view.wall_normals.push_back(
        {constantOr(normal[0], nan), constantOr(normal[1], nan), constantOr(normal[2], nan)});
}

} // namespace

ScalarValueCodec::value_type ScalarValueCodec::parse(std::string_view text, std::string& error) const
{
    return value_type{real.parse(text, error)};
}

std::string ScalarValueCodec::format(const value_type& value) const
{
    if (const auto* literal = std::get_if<FE::Real>(&value)) {
        return real.format(*literal);
    }
    if (std::holds_alternative<FE::forms::ScalarCoefficient>(value)) {
        return "<spatial coefficient>";
    }
    if (std::holds_alternative<FE::forms::TimeScalarCoefficient>(value)) {
        return "<time coefficient>";
    }
    return "<form expression>";
}

ScalarVector3Codec::value_type ScalarVector3Codec::parse(std::string_view text, std::string& error) const
{
    const auto components = opt::Vector3Codec{}.parse(text, error);
    return value_type{ScalarValue{components[0]}, ScalarValue{components[1]}, ScalarValue{components[2]}};
}

std::string ScalarVector3Codec::format(const value_type& value) const
{
    const ScalarValueCodec component{};
    return component.format(value[0]) + " " + component.format(value[1]) + " " + component.format(value[2]);
}

const opt::OptionSchema<FreeSurfaceBoundary>& freeSurfaceBoundarySchema()
{
    static const auto schema = makeFreeSurfaceBoundarySchema();
    return schema;
}

const opt::OptionSchema<FreeSurfaceContactLineInputView>& freeSurfaceContactLineInputSchema()
{
    static const auto schema = makeContactLineInputSchema();
    return schema;
}

const opt::OptionSchema<NavierStokesInputView>& navierStokesEquationSchema()
{
    static const auto schema = makeNavierStokesEquationSchema();
    return schema;
}

const opt::OptionSchema<NavierStokesInputView>& navierStokesDomainSchema()
{
    static const auto schema = makeNavierStokesDomainSchema();
    return schema;
}

const opt::OptionSchema<IncompressibleTwoFluidInputView>& incompressibleTwoFluidSchema()
{
    static const auto schema = makeIncompressibleTwoFluidSchema();
    return schema;
}

const opt::OptionSchema<NavierStokesInputView>& incompressibleTwoFluidDomainSchema()
{
    static const auto schema = makeIncompressibleTwoFluidDomainSchema();
    return schema;
}

FreeSurfaceContactLineInputView contactLineInputView(const FreeSurfaceBoundary& boundary)
{
    FreeSurfaceContactLineInputView view{};
    bool first = true;
    bool has_angle = false;
    FE::forms::bc::ScalarValue contact_angle_radians{FE::Real{1.57079632679489661923}};
    for (const auto& line : boundary.contact_lines) {
        std::visit(
            [&](const auto& configuration) {
                using C = std::decay_t<decltype(configuration)>;
                if constexpr (std::is_same_v<C, FreeSurfaceContactLine::None>) {
                    view.model = FreeSurfaceContactLineModelInput::None;
                } else {
                    if (configuration.wall_boundary_marker >= 0) {
                        view.wall_markers.push_back(configuration.wall_boundary_marker);
                    }
                    if (first) {
                        view.contact_line_marker = configuration.contact_line_marker;
                    }
                    if constexpr (std::is_same_v<C, FreeSurfaceContactLine::Pinned>) {
                        view.model = FreeSurfaceContactLineModelInput::Pinned;
                    } else if constexpr (std::is_same_v<C, FreeSurfaceContactLine::PrescribedAngle>) {
                        view.model = FreeSurfaceContactLineModelInput::PrescribedAngle;
                        has_angle = true;
                        if (first) {
                            contact_angle_radians = configuration.contact_angle_radians;
                            if (configuration.slip_length.has_value()) {
                                view.wall_slip_model = FreeSurfaceWallSlipModelInput::Navier;
                                view.wall_slip_length = *configuration.slip_length;
                            }
                        }
                        appendWallNormal(view, configuration.wall_normal);
                    } else {
                        view.model = FreeSurfaceContactLineModelInput::DynamicRenE;
                        has_angle = true;
                        if (first) {
                            contact_angle_radians = configuration.equilibrium_contact_angle_radians;
                            view.mobility = configuration.mobility;
                            view.wall_slip_model = FreeSurfaceWallSlipModelInput::Navier;
                            view.wall_slip_length = configuration.slip_length;
                            view.law_enforcement = configuration.strong_law
                                                       ? FreeSurfaceContactLawEnforcementInput::Strong
                                                       : FreeSurfaceContactLawEnforcementInput::Weak;
                        }
                        appendWallNormal(view, configuration.wall_normal);
                    }
                }
            },
            line.configuration);
        first = false;
    }
    if (has_angle) {
        // The register converts Contact_angle_degrees to radians; this is the
        // inverse map of the resolved angle.
        view.contact_angle_degrees =
            constantOr(contact_angle_radians, std::numeric_limits<FE::Real>::quiet_NaN()) *
            static_cast<FE::Real>(180.0 / kPi);
    }
    return view;
}

opt::OptionInput optionInputFromParameters(const ParameterMap& params)
{
    opt::OptionInput input;
    for (const auto& [name, parameter] : params) {
        if (parameter.defined) {
            input.add(name, parameter.value, parameter.origin);
        }
    }
    return input;
}

std::string resolvedFreeSurfacesJson(const std::vector<FreeSurfaceBoundary>& free_surfaces,
                                     const std::string& resolved_input_json)
{
    const auto& boundary_schema = freeSurfaceBoundarySchema();
    const auto& contact_schema = freeSurfaceContactLineInputSchema();
    std::string out = "{\"version\":1,\"free_surfaces\":[";
    for (std::size_t i = 0; i < free_surfaces.size(); ++i) {
        const auto& boundary = free_surfaces[i];
        if (i != 0u) {
            out += ',';
        }
        out += "{\"boundary\":";
        out += boundary_schema.toJson(boundary_schema.describe(boundary.input, boundary, FreeSurfaceBoundary{}));
        out += ",\"contact_line\":";
        out += contact_schema.toJson(contact_schema.describe(
            boundary.input, contactLineInputView(boundary), FreeSurfaceContactLineInputView{}));
        // The method profile the boundary names (decision D47, P4); the keys
        // it stated carry the source "profile".
        if (const auto* profile = boundary.input.find(svmp::Physics::options::kFreeSurfaceProfileKey)) {
            out += ",\"profile\":";
            out += json::quote(svmp::Physics::options::freeSurfaceProfile(profile->text).label());
        }
        out += '}';
    }
    out += ']';
    if (!resolved_input_json.empty()) {
        out += ',';
        out += resolved_input_json;
    }
    out += '}';
    return out;
}

[[nodiscard]] const char* freeSurfaceGeneratedInterfaceGeometryName(
    FreeSurfaceGeneratedInterfaceGeometry geometry) noexcept
{
    switch (geometry) {
    case FreeSurfaceGeneratedInterfaceGeometry::LinearCorner:
        return "LinearCorner";
    case FreeSurfaceGeneratedInterfaceGeometry::HighOrderImplicit:
        return "HighOrderImplicit";
    }
    return "Unknown";
}

[[nodiscard]] const char* freeSurfaceGeometryTangentPolicyName(
    FreeSurfaceGeometryTangentPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfaceGeometryTangentPolicy::RefreshedFrozenQuadrature:
        return "RefreshedFrozenQuadrature";
    case FreeSurfaceGeometryTangentPolicy::DifferentiatedQuadrature:
        return "DifferentiatedQuadrature";
    }
    return "Unknown";
}

const char* freeSurfaceSurfaceTensionSemiImplicitName(
    FreeSurfaceSurfaceTensionSemiImplicit value) noexcept
{
    switch (value) {
    case FreeSurfaceSurfaceTensionSemiImplicit::None:
        return "None";
    case FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement:
        return "NormalIncrement";
    }
    return "Unsupported";
}

[[nodiscard]] std::string freeSurfaceValueName(std::string_view prefix,
                                               const FreeSurfaceBoundary& bc)
{
    const char* kind = isUnfittedLevelSet(bc) ? "_i" : "_b";
    return std::string(prefix) + kind + std::to_string(freeSurfaceMarker(bc));
}

// Input-facing name of a wall-owning contact law, used in validation messages
// shared by PrescribedContactAngle and DynamicContactAngle walls.
[[nodiscard]] std::string contactWallLawName(
    const FreeSurfaceContactLine& contact_line)
{
    return contactLineKind(contact_line) == ContactLineKind::PrescribedAngle
               ? "PrescribedContactAngle"
               : "DynamicContactAngle";
}

[[nodiscard]] std::string freeSurfaceCurvaturePolicyName(
    const FreeSurfaceBoundary& bc)
{
    if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
        return "inactive_zero_surface_tension";
    }
    // SurfaceStress, the only form (decision D48).
    return "variational_surface_stress_generated_geometry";
}

std::string freeSurfaceCurvatureTangentPolicyName(const FreeSurfaceBoundary& bc)
{
    if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
        return "not_applicable";
    }
    return isUnfittedLevelSet(bc) ? "generated_geometry_refreshed_frozen"
                                  : "fitted_geometry_surface_variation";
}

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "None";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "LevelSetNegative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "LevelSetPositive";
    }
    return "Unknown";
}

[[nodiscard]] const char* kinematicEnforcementName(
    FreeSurfaceKinematicEnforcement enforcement) noexcept
{
    switch (enforcement) {
    case FreeSurfaceKinematicEnforcement::None:
        return "None";
    case FreeSurfaceKinematicEnforcement::MeshNitsche:
        return "MeshNitsche";
    }
    return "Unknown";
}

[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept
{
    return side == FE::forms::CutVolumeSide::Negative ? "Negative" : "Positive";
}

[[nodiscard]] FE::systems::MeshTangentialBoundaryPolicy
systemTangentialMeshPolicy(FreeSurfaceTangentialMeshPolicy policy)
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return FE::systems::MeshTangentialBoundaryPolicy::Free;
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return FE::systems::MeshTangentialBoundaryPolicy::SmoothingOnly;
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return FE::systems::MeshTangentialBoundaryPolicy::Prescribed;
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unknown fitted "
        "free-surface tangential mesh policy");
}

[[nodiscard]] std::string fittedTangentialOperatorSource(
    int boundary_marker,
    FreeSurfaceTangentialMeshPolicy policy)
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return "Fitted free-surface natural tangential state on marker " +
               std::to_string(boundary_marker);
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return "Fitted free-surface tangential surface smoothing on marker " +
               std::to_string(boundary_marker);
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return "Fitted free-surface prescribed tangential mesh velocity on "
               "marker " +
               std::to_string(boundary_marker);
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unknown fitted tangential "
        "mesh policy source");
}

[[nodiscard]] std::string fittedFluidNormalDescriptorSource(
    const FreeSurfaceBoundary& bc)
{
    // MeshNitsche, the only normal relation, adds no fluid kinematic row.
    return "Fitted free-surface fluid natural dynamic condition on marker " +
           std::to_string(bc.boundary_marker);
}

[[nodiscard]] const char* freeSurfaceImplementationName(
    FreeSurfaceImplementation implementation) noexcept
{
    return implementation == FreeSurfaceImplementation::FittedALE
               ? "FittedALE"
               : "UnfittedLevelSet";
}

[[nodiscard]] const char* freeSurfaceBoundaryRoleName(
    FreeSurfaceBoundaryRole role) noexcept
{
    switch (role) {
    case FreeSurfaceBoundaryRole::ExteriorOnePhaseBoundary:
        return "ExteriorOnePhaseBoundary";
    case FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume:
        return "InternalMaterialInterfaceVolume";
    }
    return "Unknown";
}

[[nodiscard]] const char* freeSurfacePhysicalModelName(
    FreeSurfacePhysicalModel model) noexcept
{
    switch (model) {
    case FreeSurfacePhysicalModel::
        OnePhaseLiquidPrescribedExteriorPressure:
        return "one_phase_liquid_prescribed_exterior_pressure";
    }
    return nullptr;
}

[[nodiscard]] const char* normalKinematicPolicyName(
    FreeSurfaceNormalKinematicPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfaceNormalKinematicPolicy::None:
        return "None";
    case FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity:
        return "MatchFluidNormalVelocity";
    }
    return "Unknown";
}

[[nodiscard]] const char* tangentialMeshPolicyName(
    FreeSurfaceTangentialMeshPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return "Free";
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return "SmoothingOnly";
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return "Prescribed";
    }
    return "Unknown";
}

[[nodiscard]] const char* contactLineKindName(ContactLineKind kind) noexcept
{
    switch (kind) {
    case ContactLineKind::None:
        return "None";
    case ContactLineKind::Pinned:
        return "Pinned";
    case ContactLineKind::PrescribedAngle:
        return "PrescribedAngle";
    case ContactLineKind::DynamicRenE:
        return "DynamicRenE";
    }
    return "Unknown";
}

[[nodiscard]] const char* geometryTangentPathName(
    FE::forms::GeometryTangentPath path) noexcept
{
    switch (path) {
    case FE::forms::GeometryTangentPath::Auto:
        return "Auto";
    case FE::forms::GeometryTangentPath::ADReference:
        return "ADReference";
    case FE::forms::GeometryTangentPath::SymbolicRequired:
        return "SymbolicRequired";
    case FE::forms::GeometryTangentPath::SymbolicWithADCheck:
        return "SymbolicWithADCheck";
    }
    return "Unknown";
}

[[nodiscard]] std::string activePhaseSignName(
    FreeSurfaceActiveDomain domain)
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "full_domain";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "negative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "positive";
    }
    return "unknown";
}

namespace {

[[nodiscard]] std::string jsonReal(FE::Real value)
{
    return FE::options::json::number(
        value,
        "IncompressibleNavierStokesVMSModule: effective configuration contains a non-finite scalar");
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
    out << "{\"model\":" << json::quote(contactLineKindName(kind));
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
    bool effective_enable_vms)
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
        << IncompressibleNavierStokesVMSOptions::
               current_configuration_schema_version
        << ",\"effective_version\":"
        << IncompressibleNavierStokesVMSOptions::
               current_configuration_schema_version
        << ",\"migration_mode\":\"current\"}"
        << ",\"capability_label\":"
        << json::quote(has_internal_material_interface_volume &&
                              !has_exterior_one_phase_boundary
                          ? "internal_material_interface_phase_volume"
                          : "one_phase_liquid_sharp_interface")
        << ",\"physical_model\":";
    if (has_internal_material_interface_volume &&
        !has_exterior_one_phase_boundary) {
        out << "{\"name\":\"internal_material_interface_phase_volume\""
            << ",\"interface_traction_owner\":\"coupled_parent\""
            << ",\"pressure_gauge_owner\":\"coupled_parent\""
            << ",\"external_boundary_model\":false}";
    } else {
        out << "{\"name\":"
            << json::quote(freeSurfacePhysicalModelName(
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
        << json::quote(options.velocity_field_name)
        << ",\"pressure\":" << json::quote(options.pressure_field_name)
        << ",\"operator\":" << json::quote(options.operator_tag)
        << ",\"dimension\":" << dimension << '}'
        << ",\"ale\":{\"enabled\":" << json::boolean(options.enable_ale)
        << ",\"mesh_velocity_source\":"
        << json::quote(options.mesh_velocity_source ==
                              ALEMeshVelocitySource::CoupledDisplacement
                          ? "CoupledDisplacement"
                          : "PrescribedData")
        << ",\"mesh_velocity_field\":"
        << json::quote(options.mesh_velocity_field_name)
        << ",\"mesh_displacement_field\":"
        << json::quote(options.mesh_displacement_field_name)
        << ",\"geometry_tangent_path\":"
        << json::quote(geometryTangentPathName(options.moving_mesh_tangent_path))
        << '}'
        << ",\"generic_velocity_nitsche\":{\"gamma\":"
        << jsonReal(options.nitsche_gamma)
        << ",\"symmetric\":" << json::boolean(options.nitsche_symmetric)
        << ",\"scale_with_polynomial_order\":"
        << json::boolean(options.nitsche_scale_with_p)
        << ",\"generated_active_boundary_minimum_energy_ratio\":"
        << jsonReal(
               options
                   .generated_boundary_nitsche_minimum_energy_ratio)
        << '}'
        << ",\"stabilization\":{\"vms_enabled\":"
        << json::boolean(effective_enable_vms)
        << ",\"ct_m\":" << jsonReal(options.ct_m)
        << ",\"ct_c\":" << jsonReal(options.ct_c)
        << ",\"epsilon\":" << jsonReal(options.stabilization_epsilon)
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
            << json::quote(freeSurfaceBoundaryRoleName(boundary.role))
            << ",\"implementation\":"
            << json::quote(freeSurfaceImplementationName(boundary.implementation))
            << ",\"boundary_marker\":" << boundary.boundary_marker
            << ",\"interface_marker\":" << boundary.interface_marker
            << ",\"level_set_field\":"
            << json::quote(boundary.level_set_field_name)
            << ",\"generated_interface_domain\":"
            << json::quote(boundary.generated_interface_domain_id)
            << ",\"generated_interface_geometry\":"
            << json::quote(freeSurfaceGeneratedInterfaceGeometryName(
                   boundary.generated_interface_geometry))
            << ",\"geometry_tangent_policy\":"
            << json::quote(freeSurfaceGeometryTangentPolicyName(
                   boundary.geometry_tangent_policy))
            << ",\"level_set_isovalue\":"
            << jsonReal(boundary.level_set_isovalue)
            << ",\"active_domain\":"
            << json::quote(activeDomainName(boundary.active_domain))
            << ",\"active_phase_sign\":"
            << json::quote(activePhaseSignName(boundary.active_domain))
            << ",\"external_pressure\":"
            << jsonScalarValue(boundary.external_pressure)
            << ",\"surface_tension\":"
            << jsonScalarValue(boundary.surface_tension)
            << ",\"surface_tension_form_effective\":"
            << json::quote(
                   isInternalMaterialInterfaceVolume(boundary)
                       ? "NotApplicableInternalVolume"
                       : "SurfaceStress");
        if (usesSemiImplicitNormalIncrement(boundary)) {
            out << ",\"surface_tension_semi_implicit\":"
                << json::quote(freeSurfaceSurfaceTensionSemiImplicitName(
                       boundary.surface_tension_semi_implicit))
                << ",\"surface_tension_semi_implicit_reference_velocity\":"
                << json::quote(std::string(
                       kFreeSurfaceSemiImplicitReferenceVelocityFieldName));
        }
        out << ",\"curvature_policy\":"
            << json::quote(freeSurfaceCurvaturePolicyName(boundary))
            << ",\"curvature_tangent_policy\":"
            << json::quote(
                   freeSurfaceCurvatureTangentPolicyName(boundary))
            << ",\"kinematic\":{\"normal_policy\":"
            << json::quote(normalKinematicPolicyName(
                   boundary.normal_kinematic_policy))
            << ",\"tangential_mesh_policy\":"
            << json::quote(tangentialMeshPolicyName(
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
            out << json::quote(
                tangential_provenance.declaration->owner_component);
        } else {
            out << "null";
        }
        out << ",\"policy_consumed\":"
            << json::boolean(
                   tangential_provenance.consumed_operator != nullptr)
            << ",\"operator_tag\":";
        if (tangential_provenance.consumed_operator != nullptr) {
            out << json::quote(
                tangential_provenance.declaration
                    ->consumer_operator_tag);
        } else {
            out << "null";
        }
        out << ",\"operator_source\":";
        if (tangential_provenance.consumed_operator != nullptr) {
            out << json::quote(
                tangential_provenance.consumed_operator->source);
        } else {
            out << "null";
        }
        out << ",\"policy_qualification\":"
            << json::quote(
                   boundary.implementation !=
                           FreeSurfaceImplementation::FittedALE
                       ? "not_applicable"
                       : "supported_configuration_envelope")
            << ",\"enforcement\":"
            << json::quote(kinematicEnforcementName(
                   boundary.kinematic_enforcement))
            << ",\"nitsche\":{\"gamma\":"
            << jsonReal(boundary.kinematic_nitsche_gamma) << "}}"
            << ",\"stabilization\":{\"enabled\":"
            << json::boolean(boundary.cut_cell_stabilization.enabled)
            << ",\"small_cut_aggregation\":"
            << json::boolean(boundary.small_cut_aggregation)
            << ",\"pressure_gradient_penalty\":"
            << jsonScalarValue(
                   boundary.cut_cell_stabilization.pressure_gradient_penalty)
            << ",\"aggregation_guards\":{\"maximum_root_path_length\":"
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
            << json::boolean(
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
    out << ",\"resolved\":"
        << resolvedFreeSurfacesJson(free_surfaces, options.resolved_input_json);
    out << "}";

    return EffectiveConfigurationArtifact{
        .component = "incompressible_navier_stokes_free_surface",
        .json = out.str(),
    };
}


// ---------------------------------------------------------------------------
// The two-fluid artifact (artifact_schema_version 3; was IncompressibleTwoFluidModule)
// ---------------------------------------------------------------------------

namespace {
namespace two_fluid_artifact {

[[nodiscard]] std::string jsonReal(FE::Real value)
{
    return FE::options::json::number(
        value,
        "IncompressibleTwoFluidModule: effective configuration contains a non-finite scalar");
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
        << json::quote(
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
            out << json::boolean(
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

[[nodiscard]] EffectiveConfigurationArtifact makeArtifact(
    const IncompressibleTwoFluidOptions& options,
    int dimension,
    int interface_marker,
    const Factories::IncompressibleTwoFluidInterfaceWeights& weights)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"artifact_schema_version\":3"
        << ",\"component\":\"incompressible_two_fluid\""
        << ",\"capability_label\":\"incompressible_two_phase_sharp_interface_initial_envelope\""
        << ",\"momentum_operator\":"
        << json::quote(options.enable_convection ? "navier_stokes" : "stokes")
        << ",\"fields\":{\"negative_velocity\":"
        << json::quote(options.negative_phase.velocity_field_name)
        << ",\"negative_pressure\":"
        << json::quote(options.negative_phase.pressure_field_name)
        << ",\"positive_velocity\":"
        << json::quote(options.positive_phase.velocity_field_name)
        << ",\"positive_pressure\":"
        << json::quote(options.positive_phase.pressure_field_name)
        << ",\"level_set\":" << json::quote(options.level_set_field_name)
        << ",\"operator\":" << json::quote(options.operator_tag)
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
        << json::quote(options.generated_interface_domain_id)
        << ",\"isovalue\":" << jsonReal(options.level_set_isovalue)
        << ",\"geometry\":"
        << json::quote(freeSurfaceGeneratedInterfaceGeometryName(
               options.generated_interface_geometry))
        << ",\"geometry_tangent_policy\":"
        << json::quote(freeSurfaceGeometryTangentPolicyName(
               options.geometry_tangent_policy))
        << ",\"surface_tension\":" << jsonReal(options.surface_tension)
        << ",\"prescribed_pressure_jump_mode\":\"manufactured_normal_traction_and_diagnostic_target\""
        << ",\"prescribed_pressure_jump_applicable\":"
        << json::boolean(options.prescribed_pressure_jump.has_value());
    if (options.prescribed_pressure_jump.has_value()) {
        out << ",\"prescribed_pressure_jump\":"
            << jsonReal(*options.prescribed_pressure_jump);
    }
    out << ",\"prescribed_viscous_traction_jump_mode\":\"manufactured_global_viscous_traction_and_composed_stress_target\""
        << ",\"prescribed_viscous_traction_jump_applicable\":"
        << json::boolean(options.prescribed_viscous_traction_jump.has_value());
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
        << json::boolean(options.include_transient_interface_penalty)
        << ",\"negative_traction_weight\":"
        << jsonReal(weights.negative_traction)
        << ",\"positive_traction_weight\":"
        << jsonReal(weights.positive_traction)
        << ",\"harmonic_viscosity\":"
        << jsonReal(weights.harmonic_viscosity)
        << ",\"harmonic_density\":"
        << jsonReal(weights.harmonic_density) << '}'
        << ",\"stabilization\":{\"phasewise_vms_pspg\":"
        << json::boolean(options.enable_vms)
        << ",\"phasewise_pressure_ghost_penalty\":true"
        << ",\"phasewise_small_cut_aggregation\":true"
        << ",\"pressure_gradient_penalty\":"
        << jsonReal(options.pressure_gradient_penalty) << '}';
    appendVelocityBoundaryArtifact(out, options, dimension);
    out << ",\"pressure_space\":{\"representation\":\"separate_phase_fields\",\"shared_gauge_count\":1";
    if (options.shared_pressure_gauge.has_value()) {
        out << ",\"shared_gauge_policy\":\"explicit_global_vertex_gid\""
            << ",\"shared_gauge_field\":"
            << json::quote(options.negative_phase.pressure_field_name)
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
        << json::boolean(
               options.require_conservative_phase_momentum_reconciliation)
        << ",\"momentum_reconciliation_policy\":\"phasewise_fail_closed_no_hidden_velocity_update\"}"
        << ",\"accepted_stage_evidence\":{\"required\":true,\"operator_stage_interface_mass_momentum_flux\":true,\"canonical_phase_aggregation\":true,\"pressure_stabilization_configuration\":true,\"shared_coupled_nonlinear_linear_report\":true,\"timing_values_retained\":false,\"rank_local_aggregation_lineage_retained\":false,\"phase_resolved_iterations\":\"unavailable_from_shared_backend\",\"pressure_stabilization_work\":\"not_separately_resolved_by_coupled_operator\"}"
        << ",\"exclusions\":[\"compressible_gas\",\"trapped_gas_pressure\",\"air_cushioning\",\"phase_change\",\"contact\",\"moving_mesh\",\"variable_material_laws\",\"turbulence\"]";
    // The two-fluid keys of the input with their resolved values and sources
    // (decision D47); the keys above are unchanged.
    if (!options.resolved_input_json.empty()) {
        out << ",\"resolved\":{\"version\":1," << options.resolved_input_json << '}';
    }
    out << '}';
    return EffectiveConfigurationArtifact{
        .component = "incompressible_two_fluid",
        .json = out.str(),
    };
}

} // namespace two_fluid_artifact
} // namespace

EffectiveConfigurationArtifact makeIncompressibleTwoFluidConfigurationArtifact(
    const IncompressibleTwoFluidOptions& options,
    int dimension,
    int interface_marker,
    const Factories::IncompressibleTwoFluidInterfaceWeights& weights)
{
    return two_fluid_artifact::makeArtifact(options, dimension, interface_marker, weights);
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

namespace svmp::Physics::options {

// The parser of a Free_surface boundary condition and of its contact lines:
// the keys resolve through freeSurfaceBoundarySchema() and
// freeSurfaceContactLineInputSchema() (decision D47, step P2), with the
// value and format texts of the schemas (D52).  The relations between keys
// keep the register's texts.

namespace {

std::string normalized_token(std::string_view raw)
{
  auto out = lower_copy(trim_copy(std::string(raw)));
  out.erase(std::remove_if(out.begin(), out.end(), [](unsigned char ch) {
              return ch == '_' || ch == '-' || std::isspace(ch);
            }),
            out.end());
  return out;
}

} // namespace

bool is_free_surface_type(std::string_view raw)
{
  return normalized_token(raw) == "freesurface";
}

namespace {

namespace navier_stokes = svmp::Physics::formulations::navier_stokes;
namespace opt = svmp::FE::options;

/// The level-set keys of an unfitted interface; a fitted boundary does not
/// read them.
const std::vector<std::string>& unfitted_interface_spellings()
{
  static const std::vector<std::string> spellings{
      "Interface_marker",
      "Level_set_field_name",
      "Generated_interface_domain_id",
      "Generated_interface_geometry",
      "Geometry_tangent_policy",
      "Level_set_isovalue",
  };
  return spellings;
}

/// The implementation of a Free_surface boundary condition, which the input
/// must state (decision D47 item 3).
navier_stokes::FreeSurfaceImplementation
implementation_of(const opt::OptionInput& input)
{
  const auto& schema = navier_stokes::freeSurfaceBoundarySchema();
  const auto resolved = schema.resolve(input.selectSpellings({"Implementation"}));
  if (!resolved.isExplicit("Implementation")) {
    throw opt::OptionError(schema.element(), "Implementation",
                           "required: FittedALE or UnfittedLevelSet "
                           "(decision D47)");
  }
  return resolved.value.implementation;
}

/// The relations between the resolved keys of a boundary, with the
/// register's texts.
void check_free_surface_relations(
    const opt::ResolvedOptions<navier_stokes::FreeSurfaceBoundary>& resolved)
{
  using navier_stokes::FreeSurfaceImplementation;
  using navier_stokes::FreeSurfaceTangentialMeshPolicy;
  const auto& fs = resolved.value;
  const auto any_explicit = [&](std::initializer_list<std::string_view> keys) {
    return std::any_of(keys.begin(), keys.end(), [&](std::string_view key) {
      return resolved.isExplicit(key);
    });
  };

  // A prescribed tangential mesh velocity implies Tangential_mesh_policy=
  // Prescribed (schema implication) unless the input gives another policy.
  if (resolved.isExplicit("Prescribed_tangential_mesh_velocity") &&
      fs.tangential_mesh_policy != FreeSurfaceTangentialMeshPolicy::Prescribed) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Prescribed tangential mesh velocity "
        "conflicts with an explicit non-Prescribed tangential policy.");
  }
  if (resolved.isExplicit("Tangential_mesh_penalty") &&
      fs.tangential_mesh_policy !=
          FreeSurfaceTangentialMeshPolicy::Prescribed &&
      fs.tangential_mesh_policy !=
          FreeSurfaceTangentialMeshPolicy::SmoothingOnly) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Tangential_mesh_penalty requires "
        "Tangential_mesh_policy=Prescribed or SmoothingOnly; Free has no "
        "tangential boundary row.");
  }
  // The MeshNitsche normal enforcement of a fitted ALE surface is the only
  // reader of its penalty (decisions D35 and D47 Q9).
  if (resolved.isExplicit("Kinematic_nitsche_gamma") &&
      fs.implementation != FreeSurfaceImplementation::FittedALE) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Kinematic_nitsche_gamma requires a FittedALE free surface (MeshNitsche normal enforcement).");
  }

  if (fs.small_cut_aggregation_guards.maximum_row_l1_norm <
      fs.small_cut_aggregation_guards.maximum_absolute_coefficient) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
        "maximum row L1 norm must be no smaller than the maximum absolute "
        "coefficient.");
  }
  if (!fs.small_cut_aggregation &&
      any_explicit({"Small_cut_aggregation_maximum_root_path_length",
                    "Small_cut_aggregation_maximum_reference_extrapolation_distance",
                    "Small_cut_aggregation_maximum_absolute_coefficient",
                    "Small_cut_aggregation_maximum_row_l1_norm",
                    "Small_cut_aggregation_rootless_fallback"})) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Small_cut_aggregation=false cannot be "
        "combined with aggregation guard settings.");
  }

  // A cut-cell suboption implies Enable_cut_cell_stabilization=true (schema
  // implication) unless the input gives the switch.
  if (resolved.isExplicit("Enable_cut_cell_stabilization") &&
      !fs.cut_cell_stabilization.enabled &&
      any_explicit({"Cut_cell_pressure_gradient_penalty"})) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Enable_cut_cell_stabilization=false cannot be combined with cut-cell stabilization suboptions.");
  }
}

void resolve_free_surface_contact_lines(
    const svmp::Physics::ParameterMap& params,
    const opt::OptionInput& input,
    navier_stokes::FreeSurfaceBoundary& fs)
{
  using ContactLine = navier_stokes::FreeSurfaceContactLine;
  using Model = navier_stokes::FreeSurfaceContactLineModelInput;
  using navier_stokes::FreeSurfaceContactLawEnforcementInput;
  using navier_stokes::FreeSurfaceWallSlipModelInput;
  using ScalarValue = navier_stokes::FreeSurfaceBoundary::ScalarValue;

  constexpr std::string_view contact_keys[]{
      "Contact_line_model",
      "Contact_angle_degrees",
      "Contact_line_wall_markers",
      "Contact_line_wall_face",
      "Contact_line_marker",
      "Contact_line_wall_normal",
      "Contact_line_mobility",
      "Wall_slip_model",
      "Wall_slip_length",
      "Contact_line_law_enforcement"};

  const auto is_known_contact_key = [&](std::string_view key) {
    return std::find(std::begin(contact_keys), std::end(contact_keys), key) !=
           std::end(contact_keys);
  };
  const auto looks_like_contact_key = [](std::string_view key) {
    const auto token = normalized_token(key);
    const auto starts_with = [&](std::string_view prefix) {
      return token.rfind(prefix, 0) == 0;
    };
    return starts_with("contact") || starts_with("freesurfacecontact") ||
           starts_with("prescribedcontact") ||
           starts_with("wallboundarymarker") ||
           starts_with("wallboundaryface") || starts_with("wallnormal") ||
           starts_with("wallslip") || token == "mobility" ||
           token == "sliplength";
  };
  for (const auto& [key, value] : params) {
    if (value.defined && looks_like_contact_key(key) &&
        !is_known_contact_key(key)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Unknown free-surface contact key '" +
          key + "'. Contact configuration is fail-closed.");
    }
  }

  const bool contact_requested = std::any_of(
      std::begin(contact_keys), std::end(contact_keys), [&](std::string_view key) {
        const auto* value = find_param(params, key);
        return value != nullptr && value->defined;
      });
  if (!contact_requested) {
    return;
  }

  if (any_parameter_defined(
          params,
          {"Contact_line_wall_face"})) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Contact wall face names are not supported by the typed free-surface contact configuration; provide explicit wall marker ids.");
  }

  const auto resolved =
      navier_stokes::freeSurfaceContactLineInputSchema().resolve(input);
  const auto& view = resolved.value;
  const bool has_degrees = resolved.isExplicit("Contact_angle_degrees");

  if (!resolved.isExplicit("Contact_line_model")) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface contact configuration requires an explicit Contact_line_model (None, Pinned, PrescribedAngle, or DynamicRenE).");
  }

  // Presence counts a key given with an empty value, as the register did.
  const bool has_angle = has_degrees;
  const bool has_wall_marker = any_parameter_defined(
      params,
      {"Contact_line_wall_markers"});
  const bool has_wall_normal = any_parameter_defined(
      params,
      {"Contact_line_wall_normal"});
  const bool has_mobility = any_parameter_defined(
      params,
      {"Contact_line_mobility"});
  const bool has_slip_model = any_parameter_defined(
      params,
      {"Wall_slip_model"});
  const bool has_slip_length = any_parameter_defined(
      params,
      {"Wall_slip_length"});
  const bool has_contact_marker = any_parameter_defined(
      params, {"Contact_line_marker"});
  const bool navier_slip = resolved.isExplicit("Wall_slip_model") &&
                           view.wall_slip_model ==
                               FreeSurfaceWallSlipModelInput::Navier;

  if (view.model == Model::None) {
    if (has_angle || has_wall_marker || has_wall_normal || has_mobility ||
        has_slip_model || has_slip_length ||
        has_contact_marker) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_model=None does not accept contact-angle, wall, mobility, slip, or contact-marker parameters.");
    }
    fs.contact_lines.push_back(ContactLine{
        .configuration = ContactLine::None{},
    });
    return;
  }
  if (view.model == Model::Pinned) {
    if (!has_wall_marker && !has_contact_marker) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_model=Pinned requires Contact_line_marker or a contact wall marker.");
    }
    if (has_angle || has_wall_normal || has_mobility ||
        has_slip_model || has_slip_length) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_model=Pinned accepts only its contact-line or wall marker.");
    }
  } else {
    if (!has_angle || !has_wall_marker || !has_wall_normal) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] PrescribedAngle and DynamicRenE require an explicit angle, wall marker(s), and wall normal(s).");
    }
    if (view.model == Model::PrescribedAngle) {
      if (has_mobility) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] PrescribedAngle does not accept contact-line mobility.");
      }
      if (has_slip_model != has_slip_length) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] PrescribedAngle Navier slip requires both Wall_slip_model and Wall_slip_length, or neither.");
      }
      if (has_slip_model && !navier_slip) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] PrescribedAngle wall slip requires Wall_slip_model=Navier.");
      }
    } else if (!has_mobility || !has_slip_model || !has_slip_length) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] DynamicRenE requires explicit mobility, wall-slip model, and wall-slip length parameters.");
    }
  }
  if (view.model == Model::DynamicRenE && !navier_slip) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] DynamicRenE requires Wall_slip_model=Navier.");
  }

  ScalarValue contact_angle{svmp::FE::Real{1.57079632679489661923}};
  if (has_degrees) {
    constexpr double pi = 3.14159265358979323846;
    contact_angle = static_cast<svmp::FE::Real>(
        static_cast<double>(view.contact_angle_degrees) * pi / 180.0);
  }
  const bool strong_contact_law =
      view.law_enforcement == FreeSurfaceContactLawEnforcementInput::Strong;
  const bool wall_markers_given =
      resolved.isExplicit("Contact_line_wall_markers");
  const bool wall_normals_given =
      resolved.isExplicit("Contact_line_wall_normal");

  const std::size_t contact_line_count =
      wall_markers_given ? view.wall_markers.size() : 1u;
  if (contact_line_count > 1u && has_contact_marker) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] A single Contact_line_marker cannot be shared by multiple contact walls; omit it to use generated markers.");
  }
  if (wall_normals_given && view.wall_normals.size() != 1u &&
      view.wall_normals.size() != contact_line_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface Contact_line_wall_normals must contain either one vector or one vector per wall marker.");
  }
  for (std::size_t i = 0; i < contact_line_count; ++i) {
    const int wall_marker = wall_markers_given ? view.wall_markers[i] : -1;
    std::array<ScalarValue, 3> wall_normal{svmp::FE::Real{0.0},
                                           svmp::FE::Real{0.0},
                                           svmp::FE::Real{0.0}};
    if (wall_normals_given) {
      const auto& normal = view.wall_normals.size() == 1u
                               ? view.wall_normals[0]
                               : view.wall_normals[i];
      wall_normal = {normal[0], normal[1], normal[2]};
    }

    if (strong_contact_law && view.model != Model::DynamicRenE) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_law_enforcement=Strong requires "
          "Contact_line_model=DynamicRenE.");
    }
    switch (view.model) {
    case Model::None:
      throw std::logic_error("None contact configuration expanded unexpectedly");
    case Model::Pinned:
      fs.contact_lines.push_back(ContactLine{
          .configuration = ContactLine::Pinned{
              .wall_boundary_marker = wall_marker,
              .contact_line_marker = view.contact_line_marker,
          },
      });
      break;
    case Model::PrescribedAngle:
      fs.contact_lines.push_back(ContactLine{
          .configuration = ContactLine::PrescribedAngle{
              .wall_boundary_marker = wall_marker,
              .contact_line_marker = view.contact_line_marker,
              .contact_angle_radians = contact_angle,
              .wall_normal = wall_normal,
              .slip_length = has_slip_length
                  ? std::optional<ScalarValue>{view.wall_slip_length}
                  : std::nullopt,
          },
      });
      break;
    case Model::DynamicRenE:
      fs.contact_lines.push_back(ContactLine{
          .configuration = ContactLine::DynamicRenE{
              .wall_boundary_marker = wall_marker,
              .contact_line_marker = view.contact_line_marker,
              .equilibrium_contact_angle_radians = contact_angle,
              .wall_normal = wall_normal,
              .mobility = view.mobility,
              .slip_length = view.wall_slip_length,
              .strong_law = strong_contact_law,
          },
      });
      break;
    }
  }
}

} // namespace

svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation
free_surface_implementation_from_params(const svmp::Physics::ParameterMap& params)
{
  return implementation_of(
      svmp::Physics::formulations::navier_stokes::optionInputFromParameters(params));
}

FreeSurfaceApplicationKeys
free_surface_application_keys(const svmp::FE::options::OptionInput& input)
{
  const auto& schema =
      svmp::Physics::formulations::navier_stokes::freeSurfaceBoundarySchema();
  FreeSurfaceApplicationKeys keys;
  try {
    const auto resolved =
        schema.resolve(input.selectSpellings({"Implementation"}));
    if (resolved.isExplicit("Implementation")) {
      keys.implementation = resolved.value.implementation;
      keys.default_active_domain = resolved.value.active_domain;
    }
  } catch (const svmp::FE::options::OptionError&) {
    // The register reports the value.
  }
  // A relaxed boolean: every value parses.
  keys.small_cut_aggregation =
      schema.resolve(input.selectSpellings({"Small_cut_aggregation"}))
          .value.small_cut_aggregation;
  return keys;
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceBoundary
resolve_free_surface_bc(
    const svmp::Physics::BoundaryConditionInput& bc,
    bool is_steady,
    bool has_temp_spat,
    bool has_other_files)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation;

  if (!is_steady) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "' currently supports only steady scalar parameters in the new solver Navier-Stokes input translator.");
  }
  if (has_temp_spat || has_other_files) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "' does not support spatial/temporal files in the new solver Navier-Stokes input translator.");
  }

  const auto& schema =
      svmp::Physics::formulations::navier_stokes::freeSurfaceBoundarySchema();
  const auto input =
      svmp::Physics::formulations::navier_stokes::optionInputFromParameters(bc.params);
  const auto implementation = implementation_of(input);

  // A fitted boundary does not read the level-set keys of an unfitted
  // interface.
  auto keys = input;
  if (implementation == FreeSurfaceImplementation::FittedALE) {
    keys = keys.excludeSpellings(unfitted_interface_spellings());
  }
  const auto resolved = schema.resolve(keys);
  check_free_surface_relations(resolved);

  auto fs = resolved.value;
  // D35: MeshNitsche is the normal enforcement of a fitted ALE surface; an
  // unfitted surface has none.
  if (implementation == FreeSurfaceImplementation::FittedALE) {
    fs.kinematic_enforcement =
        svmp::Physics::formulations::navier_stokes::
            FreeSurfaceKinematicEnforcement::MeshNitsche;
  }
  // Kept for the provenance of the resolved values (effective-configuration
  // artifact, decision D47).
  fs.input = input;
  if (implementation == FreeSurfaceImplementation::FittedALE) {
    fs.boundary_marker = bc.boundary_marker;
  }
  resolve_free_surface_contact_lines(bc.params, input, fs);
  return fs;
}

svmp::Physics::formulations::navier_stokes::
    IncompressibleNavierStokesVMSOptions
translate_fitted_surface_contact_capability(
    const svmp::Physics::EquationModuleInput& input)
{
  using svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSOptions;

  IncompressibleNavierStokesVMSOptions options{};
  for (const auto& bc : input.boundary_conditions) {
    const auto* bc_type = find_param(bc.params, "Type");
    if (bc_type == nullptr || !bc_type->defined ||
        !is_free_surface_type(bc_type->value)) {
      continue;
    }

    const auto* time_dependence =
        find_param(bc.params, "Time_dependence");
    const auto time_value =
        time_dependence != nullptr && time_dependence->defined
            ? lower_copy(trim_copy(time_dependence->value))
            : std::string{"steady"};
    const bool is_steady = time_value.empty() || time_value == "steady";
    const bool has_temporal_spatial_file = has_nonempty_defined(
        bc.params, "Temporal_and_spatial_values_file_path");
    const bool has_other_files =
        has_nonempty_defined(bc.params, "Temporal_values_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_values_file_path") ||
        has_nonempty_defined(bc.params, "Bct_file_path") ||
        has_nonempty_defined(bc.params, "Traction_values_file_path") ||
        has_nonempty_defined(bc.params, "Fourier_coefficients_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_profile_file_path");
    options.free_surface.push_back(resolve_free_surface_bc(
        bc,
        is_steady,
        has_temporal_spatial_file,
        has_other_files));
  }
  return options;
}

} // namespace svmp::Physics::options
