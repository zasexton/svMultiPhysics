/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#include "Physics/Options/NavierStokesOptionSchema.h"

#include "Physics/Options/InputReaders.h"
#include "Physics/Options/NavierStokesRegistration.h"

#include "FE/Systems/FESystem.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Options/NavierStokesOptionRules.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesGeneratedNitsche.h"
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
#include <iomanip>
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
        .key("Active_domain_method", &B::active_domain_method,
             opt::EnumCodec<FreeSurfaceActiveDomainMethod>{{
                 token(FreeSurfaceActiveDomainMethod::CutVolume, "CutVolume",
                       {"cutcellvolume", "exactcutvolume"}),
             }})
        .key("Active_domain_smoothing_width", &B::active_domain_smoothing_width,
             opt::RealCodec{.minimum = 0.0, .require_finite = false})
        .key("Allow_full_domain_unfitted_free_surface", &B::allow_full_domain_unfitted_free_surface,
             relaxedBool())
        .key("Interface_marker", &B::interface_marker, opt::IntCodec{})
        .key("Level_set_field_name", &B::level_set_field_name, opt::StringCodec{})
        .key("Generated_interface_domain_id", &B::generated_interface_domain_id, opt::StringCodec{})
        .key("Generated_interface_geometry", &B::generated_interface_geometry, opt::StringCodec{})
        .key("Geometry_tangent_policy", &B::geometry_tangent_policy, opt::StringCodec{})
        .key("Level_set_isovalue", &B::level_set_isovalue, anyReal())
        .key("External_pressure", &B::external_pressure, anyScalar(),
             doc("prescribed exterior pressure"))
        .key("Surface_tension", &B::surface_tension, anyScalar(), doc("surface tension gamma"))
        .key("Surface_tension_form", &B::surface_tension_form,
             opt::EnumCodec<FreeSurfaceSurfaceTensionForm>{{
                 token(FreeSurfaceSurfaceTensionForm::Automatic, "Automatic", {"auto"}),
                 token(FreeSurfaceSurfaceTensionForm::SurfaceStress, "SurfaceStress",
                       {"surfaceenergy", "laplacebeltrami", "variational"}),
             }})
        .key("Surface_tension_semi_implicit", &B::surface_tension_semi_implicit,
             opt::EnumCodec<FreeSurfaceSurfaceTensionSemiImplicit>{{
                 token(FreeSurfaceSurfaceTensionSemiImplicit::None, "None", {"off", "disabled"}),
                 token(FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement, "NormalIncrement",
                       {"laggednormalincrement"}),
             }})
        .key("Normal_kinematic_policy", &B::normal_kinematic_policy,
             opt::EnumCodec<FreeSurfaceNormalKinematicPolicy>{{
                 token(FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity,
                       "MatchFluidNormalVelocity", {"matchfluid", "fluidnormalvelocity"}),
                 token(FreeSurfaceNormalKinematicPolicy::None, "None", {"disabled", "off"}),
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
        .key("Kinematic_enforcement", &B::kinematic_enforcement,
             opt::EnumCodec<FreeSurfaceKinematicEnforcement>{{
                 token(FreeSurfaceKinematicEnforcement::None, "None"),
                 token(FreeSurfaceKinematicEnforcement::MeshNitsche, "MeshNitsche"),
             }})
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
        .key("Cut_cell_pressure_stabilization_policy",
             opt::nestedMember(&B::cut_cell_stabilization, &FreeSurfaceCutCellStabilization::pressure_policy),
             opt::EnumCodec<FreeSurfacePressureStabilizationPolicy>{{
                 token(FreeSurfacePressureStabilizationPolicy::Enabled, "Enabled", {"enable", "on", "true"}),
                 token(FreeSurfacePressureStabilizationPolicy::Incremental, "Incremental",
                       {"pressureincrement", "incrementalpressure", "incrementalenabled"}),
                 token(FreeSurfacePressureStabilizationPolicy::Disabled, "Disabled", {"disable", "off", "false"}),
                 token(FreeSurfacePressureStabilizationPolicy::DisabledForRefreshedFrozenHighOrder,
                       "DisabledForRefreshedFrozenHighOrder",
                       {"disableforrefreshedfrozenhighorder", "refreshedfrozenhighorderdisabled",
                        "highorderrefreshedfrozendisabled"}),
             }})
        .key("Use_cut_metadata_scale",
             opt::nestedMember(&B::cut_cell_stabilization,
                               &FreeSurfaceCutCellStabilization::use_cut_metadata_scale),
             relaxedBool())
        .key("Cut_cell_metadata_scale_cap",
             opt::nestedMember(&B::cut_cell_stabilization,
                               &FreeSurfaceCutCellStabilization::cut_metadata_scale_cap),
             SetOptionalRealCodec{opt::RealCodec{.minimum = 1.0}})
        // Defaults that depend on another key (the register's resolution).
        // D35: MeshNitsche is the qualified default of a fitted ALE surface.
        .implies("Implementation", "Kinematic_enforcement", "MeshNitsche",
                 opt::ImplicationSettings{.when_trigger_is = "FittedALE",
                                          .trigger_must_be_explicit = false})
        .implies("Prescribed_tangential_mesh_velocity", "Tangential_mesh_policy", "Prescribed")
        .implies("Cut_cell_pressure_gradient_penalty", "Enable_cut_cell_stabilization", "true")
        .implies("Cut_cell_pressure_stabilization_policy", "Enable_cut_cell_stabilization", "true")
        .implies("Use_cut_metadata_scale", "Enable_cut_cell_stabilization", "true")
        .implies("Cut_cell_metadata_scale_cap", "Enable_cut_cell_stabilization", "true")
        // Retired keys (decisions D42, D47 Q9, D47 Q2 and D48), for the
        // documentation and the acceptance lists.  The register checks them
        // before resolving, with its relaxed booleans and its texts.
        .retired({"Use_current_geometry_curvature"},
                 "retired (decision D42): identically zero on affine faces",
                 opt::RetiredAcceptance{.value = "false"})
        .retired({"Cut_cell_velocity_gradient_penalty"},
                 "retired with the velocity ghost penalty (small-cut aggregation replaces it)")
        .retired({"Cut_cell_velocity_max_derivative_order"},
                 "retired with the velocity ghost penalty")
        .retired({"Enable_velocity_extension"},
                 "the same-field dry-velocity diffusion was retired (decision D42)",
                 opt::RetiredAcceptance{.value = "false"})
        .retired({"Velocity_extension_diffusivity"},
                 "belongs to the retired dry-velocity diffusion (decision D42)")
        .retired({"Kinematic_penalty"},
                 "belongs to the retired Penalty kinematic enforcement (decision D47 Q9); "
                 "use Kinematic_enforcement=MeshNitsche")
        .retired({"Curvature"},
                 "the supplied curvature of the retired CurvatureTraction route "
                 "(decision D47 Q2); use Surface_tension_form=SurfaceStress")
        .retired({"Use_level_set_curvature"},
                 "the raw level-set curvature of the retired CurvatureTraction route "
                 "(decision D47 Q2); use Surface_tension_form=SurfaceStress",
                 opt::RetiredAcceptance{.value = "false"})
        .retired({"Allow_fitted_surface_stress"},
                 "fitted SurfaceStress needs no opt-in (decision D47 Q2)",
                 opt::RetiredAcceptance{.value = "true"})
        .retired({"Curvature_field", "CurvatureField", "Curvature_field_name",
                  "CurvatureFieldName", "Projected_curvature_field",
                  "ProjectedCurvatureField", "Free_surface_curvature_field",
                  "FreeSurfaceCurvatureField"},
                 "the projected curvature field of the retired "
                 "KinematicAreaGradientTraction route (decision D48); use "
                 "Surface_tension_form=SurfaceStress");
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
        .key("Contact_angle_radians", &V::contact_angle_radians, anyScalar(),
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Contact_angle_degrees", &V::contact_angle_degrees, anyReal(),
             opt::KeySettings{.duplicates = opt::DuplicatePolicy::Reject})
        .key("Contact_line_wall_marker", &V::wall_markers, opt::ListCodec<opt::IntCodec>{},
             opt::KeySettings{.spellings = {{"Contact_line_wall_markers", false},
                                            {"Contact_line_wall_marker", false}},
                              .duplicates = opt::DuplicatePolicy::Reject})
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

FreeSurfaceContactLineInputView contactLineInputView(const FreeSurfaceBoundary& boundary)
{
    FreeSurfaceContactLineInputView view{};
    bool first = true;
    bool has_angle = false;
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
                            view.contact_angle_radians = configuration.contact_angle_radians;
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
                            view.contact_angle_radians = configuration.equilibrium_contact_angle_radians;
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
            constantOr(view.contact_angle_radians, std::numeric_limits<FE::Real>::quiet_NaN()) *
            static_cast<FE::Real>(180.0 / kPi);
    }
    return view;
}

opt::OptionInput optionInputFromParameters(const ParameterMap& params)
{
    opt::OptionInput input;
    for (const auto& [name, parameter] : params) {
        if (parameter.defined) {
            input.add(name, parameter.value);
        }
    }
    return input;
}

std::string resolvedFreeSurfacesJson(const std::vector<FreeSurfaceBoundary>& free_surfaces)
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
        out += '}';
    }
    out += "]}";
    return out;
}

[[nodiscard]] const char* pressureActiveDomainName(
    FreeSurfaceActiveDomain domain) noexcept
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

[[nodiscard]] const char* surfaceTensionFormName(
    const FreeSurfaceBoundary& bc) noexcept
{
    switch (bc.surface_tension_form) {
    case FreeSurfaceSurfaceTensionForm::Automatic:
        return "SurfaceStress";
    case FreeSurfaceSurfaceTensionForm::SurfaceStress:
        return "SurfaceStress";
    }
    return "Unsupported";
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

[[nodiscard]] const char* pressureStabilizationPolicyName(
    FreeSurfacePressureStabilizationPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfacePressureStabilizationPolicy::Enabled:
        return "Enabled";
    case FreeSurfacePressureStabilizationPolicy::Incremental:
        return "Incremental";
    case FreeSurfacePressureStabilizationPolicy::Disabled:
        return "Disabled";
    case FreeSurfacePressureStabilizationPolicy::DisabledForRefreshedFrozenHighOrder:
        return "DisabledForRefreshedFrozenHighOrder";
    }
    return "Unknown";
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

[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept
{
    switch (method) {
    case FreeSurfaceActiveDomainMethod::CutVolume:
        return "CutVolume";
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


// ---------------------------------------------------------------------------
// The two-fluid artifact (artifact_schema_version 3; was IncompressibleTwoFluidModule)
// ---------------------------------------------------------------------------

namespace {
namespace two_fluid_artifact {

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
// value and format texts of the schemas (D52).  The retired keys and values
// keep the register's acceptance and texts, and so do the relations between
// keys.

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

/// The input without the given spellings.
opt::OptionInput without_spellings(const opt::OptionInput& input,
                                   const std::vector<std::string>& spellings)
{
  opt::OptionInput out;
  for (const auto& entry : input.entries()) {
    if (std::find(spellings.begin(), spellings.end(), entry.spelling) ==
        spellings.end()) {
      out.add(entry.spelling, entry.text);
    }
  }
  return out;
}

/// The input restricted to the given spellings.
opt::OptionInput only_spellings(const opt::OptionInput& input,
                                const std::vector<std::string>& spellings)
{
  opt::OptionInput out;
  for (const auto& entry : input.entries()) {
    if (std::find(spellings.begin(), spellings.end(), entry.spelling) !=
        spellings.end()) {
      out.add(entry.spelling, entry.text);
    }
  }
  return out;
}

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

navier_stokes::FreeSurfaceImplementation
implementation_of(const opt::OptionInput& input)
{
  return navier_stokes::freeSurfaceBoundarySchema()
      .resolve(only_spellings(input, {"Implementation"}))
      .value.implementation;
}

/// A retired value of an enumerated key: the normalized tokens that select
/// it, its name, the decision that retired it and the replacement.
struct RetiredFreeSurfaceValue {
  std::string_view key;
  std::vector<std::string_view> tokens;
  std::string_view name;
  std::string_view decision;
  std::string_view replacement;
};

const std::vector<RetiredFreeSurfaceValue>& retired_free_surface_values()
{
  static const std::vector<RetiredFreeSurfaceValue> values{
      {"Active_domain_method",
       {"smoothedindicator", "smoothindicator", "indicator"},
       "SmoothedIndicator", "D42", "CutVolume"},
      // The supplied- or raw-curvature traction was used by no accepted
      // benchmark; SurfaceStress is the variation of the discrete area.
      {"Surface_tension_form",
       {"curvaturetraction", "explicitcurvature", "younglaplacetraction"},
       "CurvatureTraction", "D47 Q2", "SurfaceStress"},
      {"Surface_tension_form",
       {"generatedcurvaturetraction", "generatedgeometrycurvaturetraction",
        "generatednormalcurvaturetraction"},
       "GeneratedCurvatureTraction", "D42", "SurfaceStress"},
      // The projected-curvature traction was retired with its curvature
      // projection.
      {"Surface_tension_form",
       {"kinematicareagradienttraction", "totalenergygradienttraction",
        "discreteenergygradienttraction"},
       "KinematicAreaGradientTraction", "D48", "SurfaceStress"},
      // The fitted Penalty and Nitsche enforcements imposed the kinematic
      // relation on the fluid rows as well.
      {"Kinematic_enforcement", {"penalty"}, "Penalty", "D47 Q9",
       "MeshNitsche"},
      {"Kinematic_enforcement", {"nitsche"}, "Nitsche", "D47 Q9",
       "MeshNitsche"},
  };
  return values;
}

/// The retired keys and values of a Free_surface boundary condition, with
/// the register's acceptance (relaxed booleans; a restated no-op value
/// passes) and texts.  The schema's retired entries document the same keys
/// for the acceptance lists; the register applies these checks.
void reject_retired_free_surface_input(
    const svmp::Physics::BoundaryConditionInput& bc)
{
  for (const auto& retired : retired_free_surface_values()) {
    const auto raw = first_defined_string(bc.params, {retired.key});
    if (!raw.has_value()) {
      continue;
    }
    const auto token = normalized_token(*raw);
    if (std::find(retired.tokens.begin(), retired.tokens.end(), token) !=
        retired.tokens.end()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface " +
          std::string(retired.key) + "=" + *raw + " (" +
          std::string(retired.name) + ") was retired (decision " +
          std::string(retired.decision) + "); use " +
          std::string(retired.replacement) + ".");
    }
  }

  // Decision D47 Q2 retired the CurvatureTraction route with its supplied
  // curvature and its raw level-set curvature, and the opt-in of fitted
  // SurfaceStress.  The keys stay known to Parameters, which reports the
  // restated no-op values Use_level_set_curvature=false and
  // Allow_fitted_surface_stress=true as input deprecations; every other
  // value fails here.
  if (const auto allow_fitted_surface_stress = first_defined_bool(
          bc.params,
          {"Allow_fitted_surface_stress"});
      allow_fitted_surface_stress.has_value() &&
      !*allow_fitted_surface_stress) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Allow_fitted_surface_stress=false was retired (decision D47 "
        "Q2); fitted SurfaceStress needs no opt-in.  Remove the key and use "
        "Surface_tension_form=SurfaceStress.");
  }
  if (first_defined_string(bc.params, {"Curvature"}).has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Curvature (the supplied curvature of the CurvatureTraction "
        "route) was retired (decision D47 Q2); remove it and use "
        "Surface_tension_form=SurfaceStress.");
  }
  if (first_defined_bool(bc.params, {"Use_level_set_curvature"})
          .value_or(false)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Use_level_set_curvature=true (the raw level-set curvature of "
        "the CurvatureTraction route) was retired (decision D47 Q2); use "
        "Surface_tension_form=SurfaceStress.");
  }
  // Decision D48 retired KinematicAreaGradientTraction, the only reader of
  // the projected curvature field of a boundary.  Every spelling of the key
  // fails whenever it is given, whatever its value; the keys stay known to
  // Parameters.
  for (const std::string_view key :
       {"Curvature_field", "CurvatureField", "Curvature_field_name",
        "CurvatureFieldName", "Projected_curvature_field",
        "ProjectedCurvatureField", "Free_surface_curvature_field",
        "FreeSurfaceCurvatureField"}) {
    if (const auto* parameter = find_param(bc.params, key);
        parameter != nullptr && parameter->defined) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free_surface BC '" + bc.name + "': " +
          std::string(key) +
          " (the projected curvature field of the "
          "KinematicAreaGradientTraction route) was retired (decision D48); "
          "remove it and use Surface_tension_form=SurfaceStress.");
    }
  }
  // Fitted CurvatureTraction with pointwise current-geometry curvature was
  // withdrawn by D5 (identically zero on affine faces) and retired by D42.
  if (first_defined_bool(
          bc.params,
          {"Use_current_geometry_curvature"})
          .value_or(false)) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Use_current_geometry_curvature=true (pointwise current-geometry "
        "curvature of the fitted CurvatureTraction) was retired (decision "
        "D42); it is identically zero on affine faces.  Use "
        "Surface_tension_form=SurfaceStress, the default.");
  }
  // Kinematic_penalty set the penalty of the Penalty enforcement; both were
  // retired by decision D47 Q9, so the key fails whenever it is given.
  if (first_defined_string(bc.params, {"Kinematic_penalty"}).has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Kinematic_penalty (the penalty of the Penalty kinematic "
        "enforcement) was retired (decision D47 Q9); remove it and use "
        "Kinematic_enforcement=MeshNitsche.");
  }
  if (first_defined_double(
          bc.params,
          {"Cut_cell_velocity_gradient_penalty"})) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Cut-cell velocity ghost-penalty settings are retired; small-cut aggregation replaces them.");
  }
  if (first_defined_int(
          bc.params,
          {"Cut_cell_velocity_max_derivative_order"})) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Cut-cell velocity derivative-order settings are retired with the velocity ghost penalty.");
  }

  // The legacy same-field dry-velocity diffusion (Velocity_extension) was
  // retired with decision D42; it modified the physical momentum rows.
  // Decks may still state it as disabled.
  const auto velocity_extension_enabled = first_defined_bool(
      bc.params,
      {"Enable_velocity_extension"});
  const bool velocity_extension_explicitly_disabled =
      velocity_extension_enabled.has_value() &&
      !*velocity_extension_enabled;
  const bool velocity_extension_diffusivity_present =
      first_defined_double(
          bc.params,
          {"Velocity_extension_diffusivity"})
          .has_value();
  if (velocity_extension_enabled.value_or(false) ||
      (velocity_extension_diffusivity_present &&
       !velocity_extension_explicitly_disabled)) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': the legacy same-field dry-velocity diffusion "
        "(Enable_velocity_extension / Velocity_extension_diffusivity) was "
        "retired (decision D42) because it modified the physical momentum; "
        "the level-set advection velocity extension and small-cut "
        "aggregation replace it.  Remove these keys or set "
        "Enable_velocity_extension=false.");
  }
  if (velocity_extension_diffusivity_present) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "': Velocity_extension_diffusivity belongs to the retired legacy "
        "dry-velocity diffusion (decision D42).");
  }
}

/// The relations between the resolved keys of a boundary, with the
/// register's texts.
void check_free_surface_relations(
    const opt::ResolvedOptions<navier_stokes::FreeSurfaceBoundary>& resolved)
{
  using navier_stokes::FreeSurfaceKinematicEnforcement;
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
  // Kinematic_nitsche_symmetric and Kinematic_nitsche_scale_with_p, which
  // no deck, runner or test set, were removed (decision D47); the Nitsche
  // enforcement they configured was retired by D47 Q9.
  if (resolved.isExplicit("Kinematic_nitsche_gamma") &&
      fs.kinematic_enforcement != FreeSurfaceKinematicEnforcement::MeshNitsche) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Kinematic_nitsche_gamma requires Kinematic_enforcement=MeshNitsche.");
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
      any_explicit({"Cut_cell_pressure_gradient_penalty",
                    "Cut_cell_pressure_stabilization_policy",
                    "Use_cut_metadata_scale",
                    "Cut_cell_metadata_scale_cap"})) {
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
      "Contact_angle_radians",
      "Contact_angle_degrees",
      "Contact_line_wall_markers",
      "Contact_line_wall_marker",
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
  const bool has_radians = resolved.isExplicit("Contact_angle_radians");
  const bool has_degrees = resolved.isExplicit("Contact_angle_degrees");

  if (!resolved.isExplicit("Contact_line_model")) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface contact configuration requires an explicit Contact_line_model (None, Pinned, PrescribedAngle, or DynamicRenE).");
  }
  if (has_radians && has_degrees) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface contact configuration must specify the contact angle in either radians or degrees, not both.");
  }

  // Presence counts a key given with an empty value, as the register did.
  const bool has_angle = has_radians || has_degrees;
  const bool has_wall_marker = any_parameter_defined(
      params,
      {"Contact_line_wall_markers",
       "Contact_line_wall_marker"});
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
  if (has_radians) {
    contact_angle = view.contact_angle_radians;
  } else if (has_degrees) {
    constexpr double pi = 3.14159265358979323846;
    contact_angle = static_cast<svmp::FE::Real>(
        static_cast<double>(view.contact_angle_degrees) * pi / 180.0);
  }
  const bool strong_contact_law =
      view.law_enforcement == FreeSurfaceContactLawEnforcementInput::Strong;
  const bool wall_markers_given =
      resolved.isExplicit("Contact_line_wall_marker");
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
        schema.resolve(only_spellings(input, {"Implementation"}));
    if (resolved.isExplicit("Implementation")) {
      keys.implementation = resolved.value.implementation;
    }
  } catch (const svmp::FE::options::OptionError&) {
    // The register reports the value.
  }
  // A relaxed boolean: every value parses.
  keys.small_cut_aggregation =
      schema.resolve(only_spellings(input, {"Small_cut_aggregation"}))
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
  reject_retired_free_surface_input(bc);

  // The retired keys were checked above; a fitted boundary does not read the
  // level-set keys of an unfitted interface.
  auto keys = without_spellings(input, schema.retiredSpellings());
  if (implementation == FreeSurfaceImplementation::FittedALE) {
    keys = without_spellings(keys, unfitted_interface_spellings());
  }
  const auto resolved = schema.resolve(keys);
  check_free_surface_relations(resolved);

  auto fs = resolved.value;
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
