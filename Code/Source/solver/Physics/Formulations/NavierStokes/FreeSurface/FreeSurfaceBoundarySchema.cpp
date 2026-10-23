/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceBoundarySchema.h"

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>

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
                 token(FreeSurfaceSurfaceTensionForm::KinematicAreaGradientTraction,
                       "KinematicAreaGradientTraction",
                       {"totalenergygradienttraction", "discreteenergygradienttraction"}),
             }})
        .key("Surface_tension_semi_implicit", &B::surface_tension_semi_implicit,
             opt::EnumCodec<FreeSurfaceSurfaceTensionSemiImplicit>{{
                 token(FreeSurfaceSurfaceTensionSemiImplicit::None, "None", {"off", "disabled"}),
                 token(FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement, "NormalIncrement",
                       {"laggednormalincrement"}),
             }})
        .key("Curvature_field", &B::curvature_field_name, opt::StringCodec{},
             opt::KeySettings{.spellings = {{"Curvature_field", false},
                                            {"CurvatureField", true},
                                            {"Projected_curvature_field", true},
                                            {"ProjectedCurvatureField", true},
                                            {"Free_surface_curvature_field", true},
                                            {"FreeSurfaceCurvatureField", true}}})
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
             opt::OptionalCodec<opt::RealCodec>{opt::RealCodec{.minimum = 1.0}})
        // Retired keys the register still recognizes (decisions D42, D47 Q9
        // and D47 Q2).  The register applies the rules; these entries
        // document them.
        .retired({"Use_current_geometry_curvature"},
                 "retired (decision D42): identically zero on affine faces",
                 opt::RetiredAcceptance{.value = "false"})
        .retired({"Cut_cell_velocity_gradient_penalty"},
                 "retired with the velocity ghost penalty (small-cut aggregation replaces it)",
                 opt::RetiredAcceptance{.any_value_in_mode = "explicit_legacy"})
        .retired({"Cut_cell_velocity_max_derivative_order"},
                 "retired with the velocity ghost penalty",
                 opt::RetiredAcceptance{.any_value_in_mode = "explicit_legacy"})
        .retired({"Enable_velocity_extension"},
                 "the same-field dry-velocity diffusion was retired (decision D42)",
                 opt::RetiredAcceptance{.value = "false"})
        .retired({"Velocity_extension_diffusivity"},
                 "belongs to the retired dry-velocity diffusion (decision D42)",
                 opt::RetiredAcceptance{.any_value_in_mode = "explicit_legacy"})
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
                 opt::RetiredAcceptance{.value = "true"});
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

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
