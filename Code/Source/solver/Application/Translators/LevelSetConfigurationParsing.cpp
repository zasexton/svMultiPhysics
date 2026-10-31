#include "Application/Translators/LevelSetConfigurationParsing.h"

#include "Parameters.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace application::translators::level_set::configuration {
namespace {

namespace opt = svmp::FE::options;
namespace ls = svmp::FE::level_set;
using Keys = application::core::LevelSetEquationKeys;

// Every key reads like the parser it replaced: an empty value counts as
// absent, and the first spelling with a value wins.
opt::KeySettings setting(std::string doc,
                         std::vector<opt::OptionSpelling> spellings = {}) {
  return opt::KeySettings{.spellings = std::move(spellings),
                          .empty = opt::EmptyPolicy::Absent,
                          .doc = std::move(doc)};
}

// true, 1, yes or on (any case) is true; any other text is false.
opt::BoolCodec relaxedBool() { return opt::BoolCodec{opt::BoolLexicon::Relaxed}; }

// A real of the keys that the installation never read (std::stod in the
// maintenance or the driver): the whole text; non-finite values pass.
opt::RealCodec anyReal() { return opt::RealCodec{.require_finite = false}; }

// A member of the static-capillary options (some belong to the minimizer
// options it derives from).
template <class Member, class Owner>
auto staticCapillary(Member Owner::*member) {
  return [member](Keys &keys) -> auto & {
    return keys.static_capillary_equilibrium.*member;
  };
}

template <class E>
opt::EnumToken<E> token(E value, std::string canonical,
                        std::vector<std::string> synonyms = {}) {
  return opt::EnumToken<E>{value, std::move(canonical), std::move(synonyms)};
}

// A value that the input can only set: unset unless given ("none" is not an
// accepted text, unlike opt::OptionalCodec).
template <class Codec> struct SetOptionalCodec {
  using value_type = std::optional<typename Codec::value_type>;
  static constexpr opt::EmptyPolicy empty_policy = opt::EmptyPolicy::Absent;
  Codec inner{};

  [[nodiscard]] value_type parse(std::string_view text,
                                 std::string &error) const {
    return inner.parse(text, error);
  }
  [[nodiscard]] std::string format(const value_type &value) const {
    return value ? inner.format(*value) : std::string("none");
  }
  [[nodiscard]] std::string describe() const { return inner.describe(); }
};

// Fixed-flux region boxes, "name|kind|xmin|xmax|ymin|ymax|zmin|zmax"
// separated by ';' (ls::parseLevelSetPhaseRegionBoxes); '*' leaves a side
// unbounded.
struct PhaseRegionBoxesCodec {
  using value_type = std::vector<ls::LevelSetPhaseRegionBox>;
  static constexpr opt::EmptyPolicy empty_policy = opt::EmptyPolicy::Absent;

  [[nodiscard]] value_type parse(std::string_view text,
                                 std::string &error) const {
    try {
      return ls::parseLevelSetPhaseRegionBoxes(text);
    } catch (const std::exception &failure) {
      error = failure.what();
      return {};
    }
  }
  [[nodiscard]] std::string format(const value_type &boxes) const {
    const opt::RealCodec coordinate{};
    const auto bound = [&](svmp::FE::Real value, svmp::FE::Real unbounded) {
      return value == unbounded ? std::string("*") : coordinate.format(value);
    };
    std::string out;
    for (const auto &box : boxes) {
      if (!out.empty()) {
        out += "; ";
      }
      out += box.name + '|' + ls::levelSetPhaseRegionKindName(box.kind);
      for (std::size_t d = 0; d < 3u; ++d) {
        out += '|' + bound(box.minimum[d],
                           std::numeric_limits<svmp::FE::Real>::lowest());
        out += '|' + bound(box.maximum[d],
                           std::numeric_limits<svmp::FE::Real>::max());
      }
    }
    return out;
  }
  [[nodiscard]] std::string describe() const {
    return "list of name|kind|xmin|xmax|ymin|ymax|zmin|zmax";
  }
};

// The keys that only the equation element gives: the domain elements of
// the installation do not read them.
const std::initializer_list<std::string_view> kEquationElementKeys{
    "Level_set_isovalue",
    "Advection_velocity_extension_method",
    "Advection_velocity_extension_coupling",
    "Source_velocity_field_name",
    "Wet_extension_wall_faces",
    "Enable_static_capillary_equilibrium_initialization",
    "Static_capillary_volume_tolerance",
    "Static_capillary_projected_gradient_tolerance",
    "Static_capillary_pressure_representability_max_residual_norm",
    "Static_capillary_pressure_representability_max_relative_distance",
    "Static_capillary_physical_equilibrium_max_residual_norm",
    "Static_capillary_constant_pressure_kkt_max_residual_norm",
    "Static_capillary_constant_pressure_kkt_max_relative_distance",
    "Static_capillary_finite_difference_reference_coefficient_scale",
    "Static_capillary_finite_difference_relative_step",
    "Static_capillary_minimum_finite_difference_step",
    "Static_capillary_finite_difference_max_shrinks",
    "Static_capillary_max_iterations",
    "Static_capillary_max_line_search_iterations",
    "Static_capillary_max_topology_epoch_transitions",
    "Static_capillary_projected_gradient_inverse_stiffness",
    "Static_capillary_tangent_trust_radius",
    "Static_capillary_maximum_coefficient_update_linf",
    "Static_capillary_line_search_shrink",
    "Static_capillary_armijo_fraction",
    "Static_capillary_limited_memory_history_size",
    "Static_capillary_limited_memory_curvature_tolerance",
    "Static_capillary_minimum_volume_merit_penalty"};

opt::OptionSchema<Keys> makeLevelSetEquationSchema() {
  using Form = ls::LevelSetTransportForm;
  using FieldSource = ls::LevelSetFieldSource;
  using VelocitySource = ls::LevelSetVelocitySource;
  using Side = ls::LevelSetPhaseSide;
  using FluxPolicy = ls::LevelSetConservativePhaseBoundaryFluxPolicy;
  using Extension = ls::PdeVelocityExtensionOperator;
  using Coupling = application::core::LevelSetVelocityExtensionCoupling;
  using Static = application::core::StaticCapillaryEquilibriumOptions;
  opt::OptionSchema<Keys> schema("Level-set equation");
  schema
      .key("Coupled", &Keys::coupled, relaxedBool(),
           setting("the transport joins the monolithic operator (equations)"))
      .key("Transport_form",
           [](Keys &keys) -> auto & { return keys.transport.transport_form; },
           opt::EnumCodec<Form>{{
               token(Form::Advective, "advective", {"classical", "standard"}),
               token(Form::ConservativeDivergence, "conservative_divergence",
                     {"conservative", "divergence", "divergence_form"}),
           }},
           setting("form of the transport equation"))
      .key("Level_set_field_name",
           [](Keys &keys) -> auto & { return keys.transport.level_set.field_name; },
           opt::StringCodec{},
           setting("name of the level-set field",
                   {{"Level_set_field_name", false}, {"Field_name", false}}))
      .key("Level_set_source",
           [](Keys &keys) -> auto & { return keys.transport.level_set.source; },
           opt::EnumCodec<FieldSource>{{
               token(FieldSource::Unknown, "unknown", {"solution"}),
               token(FieldSource::PrescribedData, "prescribed_data",
                     {"prescribed", "data"}),
           }},
           setting("an unknown, or prescribed data initialized from the mesh"))
      .key("Auto_register_level_set_field",
           [](Keys &keys) -> auto & {
             return keys.transport.level_set.auto_register_field;
           },
           relaxedBool(), setting("register the level-set field if missing"))
      .key("Enable_conservative_phase_transport",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.enabled;
           },
           relaxedBool(),
           setting("conservative P1 phase transport that reconciles the "
                   "level set"))
      .key("Conservative_phase_field_name",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.liquid_indicator
                 .field_name;
           },
           opt::StringCodec{}, setting("name of the liquid-indicator field"))
      .key("Auto_register_conservative_phase_field",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.liquid_indicator
                 .auto_register_field;
           },
           relaxedBool(), setting("register the liquid-indicator field if missing"))
      .key("Conservative_phase_liquid_side",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.liquid_side;
           },
           opt::EnumCodec<Side>{{
               token(Side::Negative, "negative", {"minus"}),
               token(Side::Positive, "positive", {"plus"}),
           }},
           setting("side of the level set that holds the liquid"))
      .key("Conservative_phase_invariant_tolerance",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.invariant_tolerance;
           },
           opt::RealCodec{},
           setting("tolerance of the phase invariants and boundary flux"))
      .key("Conservative_phase_component_activity_tolerance",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase
                 .component_activity_tolerance;
           },
           opt::RealCodec{},
           setting("indicator measure below which a component is inactive"))
      .key("Conservative_phase_maximum_courant",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.maximum_courant;
           },
           opt::RealCodec{}, setting("largest graph Courant number of a stage"))
      .key("Conservative_phase_enforce_courant_limit",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.enforce_courant_limit;
           },
           relaxedBool(), setting("fail a stage above the Courant limit"))
      .key("Conservative_phase_require_constant_preservation",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase
                 .require_constant_preservation;
           },
           relaxedBool(),
           setting("require that a stage keeps a constant indicator"))
      .key("Conservative_phase_momentum_relative_tolerance",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase
                 .momentum_relative_tolerance;
           },
           opt::RealCodec{},
           setting("relative phasewise momentum gate after reconciliation"))
      .key("Conservative_phase_write_flux_artifacts",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.write_flux_artifacts;
           },
           relaxedBool(), setting("write the flux artifacts of the stages"))
      .key("Conservative_phase_flux_artifact_cadence_steps",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase
                 .flux_artifact_cadence_steps;
           },
           opt::IntCodec{}, setting("steps between two flux artifacts"))
      .key("Conservative_phase_classify_nonprimary_components_as_satellites",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase
                 .classify_nonprimary_components_as_satellites;
           },
           relaxedBool(),
           setting("ledger every component but the largest as a satellite"))
      .key("Conservative_phase_boundary_flux_policy",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.boundary_flux_policy;
           },
           opt::EnumCodec<FluxPolicy>{{
               token(FluxPolicy::ClosedDomainDiscreteQFluxOnly,
                     "closed_domain_discrete_q_flux_only"),
               token(FluxPolicy::GloballyBalancedDiscreteQFlux,
                     "globally_balanced_discrete_q_flux"),
           }},
           setting("discrete boundary flux that a stage admits"))
      .key("Conservative_phase_fixed_flux_regions",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.fixed_flux_regions;
           },
           PhaseRegionBoxesCodec{},
           setting("fixed box regions of the flux ledger"))
      .key("Conservative_phase_impermeable_normal_velocity_tolerance",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase
                 .impermeable_normal_velocity_tolerance;
           },
           opt::RealCodec{},
           setting("pointwise wall-velocity tolerance (not implemented: "
                   "stating it fails with the phase transport)"))
      .key("Conservative_phase_reconcile_geometry",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.reconcile_geometry;
           },
           relaxedBool(),
           setting("reconcile the level set with the indicator measure"))
      .key("Conservative_phase_geometry_measure_tolerance",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase
                 .geometry_measure_tolerance;
           },
           opt::RealCodec{}, setting("tolerance of the reconciled measure"))
      .key("Conservative_phase_geometry_correction_max_iterations",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase
                 .geometry_correction_max_iterations;
           },
           opt::IntCodec{.minimum = 1},
           setting("iterations of the geometry correction"))
      .key("Conservative_phase_maximum_geometry_displacement_fraction",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase
                 .maximum_geometry_displacement_fraction;
           },
           opt::RealCodec{},
           setting("largest correction displacement, as a fraction of a "
                   "cell"))
      .key("Velocity_field_name",
           [](Keys &keys) -> auto & { return keys.transport.velocity.field_name; },
           opt::StringCodec{}, setting("name of the advection velocity field"))
      .key("Velocity_source",
           [](Keys &keys) -> auto & { return keys.transport.velocity.source; },
           opt::EnumCodec<VelocitySource>{{
               token(VelocitySource::CoupledField, "coupled_field",
                     {"coupled", "unknown"}),
               token(VelocitySource::PrescribedData, "prescribed_data",
                     {"prescribed", "data"}),
               token(VelocitySource::ConstantVector, "constant_vector",
                     {"constant"}),
               token(VelocitySource::MaterialInterfacePhasePair,
                     "material_interface_phase_pair"),
           }},
           setting("where the advection velocity comes from"))
      .key("Material_interface_marker",
           [](Keys &keys) -> auto & {
             return keys.transport.velocity.material_interface_marker;
           },
           opt::IntCodec{},
           setting("interface of the material-interface phase-pair velocity"))
      .key("Auto_register_velocity_field",
           [](Keys &keys) -> auto & {
             return keys.transport.velocity.auto_register_field;
           },
           relaxedBool(), setting("register the velocity field if missing"))
      .key("Constant_velocity",
           [](Keys &keys) -> auto & { return keys.transport.velocity.constant_value; },
           opt::Vector3Codec{},
           setting("a constant advection velocity (selects constant_vector)"))
      .key("Enable_SUPG",
           [](Keys &keys) -> auto & { return keys.transport.supg.enabled; },
           relaxedBool(), setting("SUPG stabilization"))
      .key("SUPG_tau_scale",
           [](Keys &keys) -> auto & { return keys.transport.supg.tau_scale; },
           opt::RealCodec{}, setting("SUPG scale c_tau"))
      .key("SUPG_transient_scale",
           [](Keys &keys) -> auto & { return keys.transport.supg.transient_scale; },
           opt::RealCodec{}, setting("SUPG transient scale c_t"))
      .key("Enable_interface_kinematic",
           [](Keys &keys) -> auto & {
             return keys.transport.interface_kinematic.enabled;
           },
           relaxedBool(), setting("interface kinematic term"))
      .key("Interface_kinematic_marker",
           [](Keys &keys) -> auto & {
             return keys.transport.interface_kinematic.interface_marker;
           },
           opt::IntCodec{},
           setting("interface of the kinematic term (enables it)"))
      .key("Interface_kinematic_weight_scale",
           [](Keys &keys) -> auto & {
             return keys.transport.interface_kinematic.weight_scale;
           },
           opt::RealCodec{}, setting("weight of the interface kinematic term"))
      .key("Enable_kinematic_reconciliation",
           [](Keys &keys) -> auto & {
             return keys.transport.kinematic_reconciliation.enabled;
           },
           relaxedBool(), setting("accepted-step kinematic reconciliation"))
      .key("Enable_sign_definite_patch_bounds",
           [](Keys &keys) -> auto & {
             return keys.transport.sign_definite_patch_bounds.enabled;
           },
           relaxedBool(), setting("accepted-step sign-definite patch bounds"))
      .key("Enable_interface_preserving_redistancing",
           [](Keys &keys) -> auto & {
             return keys.transport.interface_preserving_redistancing.enabled;
           },
           relaxedBool(),
           setting("triggered accepted-step interface-preserving redistancing"))
      // Read from the equation element only (levelSetDomainInput).
      .key("Level_set_isovalue", &Keys::level_set_isovalue, anyReal(),
           setting("isovalue of the interface of the maintained field"))
      .key("Advection_velocity_extension_method",
           &Keys::advection_velocity_extension_method,
           SetOptionalCodec<opt::EnumCodec<Extension>>{{{
               token(Extension::Harmonic, "pde_harmonic"),
               token(Extension::LeastSquaresNormal, "pde_normal",
                     {"pde_least_squares_normal"}),
           }}},
           setting("PDE extension of the advection velocity into the dry "
                   "region"))
      .key("Advection_velocity_extension_coupling",
           &Keys::advection_velocity_extension_coupling,
           SetOptionalCodec<opt::EnumCodec<Coupling>>{{{
               token(Coupling::Monolithic, "monolithic"),
               token(Coupling::Prescribed, "prescribed"),
           }}},
           setting("frozen rows of an extension unknown, or a prescribed "
                   "advection field"))
      .key("Source_velocity_field_name", &Keys::source_velocity_field_name,
           opt::StringCodec{},
           setting("physical velocity that the extension extends (default "
                   "Velocity)"))
      .key("Wet_extension_wall_faces", &Keys::wet_extension_wall_faces,
           opt::StringCodec{},
           setting("wall faces of the extension's impermeability rows, "
                   "separated by ';' or ','"))
      .key("Enable_static_capillary_equilibrium_initialization",
           &Keys::static_capillary_equilibrium_enabled, relaxedBool(),
           setting("start from the discrete static capillary equilibrium"))
      .key("Static_capillary_volume_tolerance",
           staticCapillary(&Static::volume_tolerance), anyReal(),
           setting("liquid-volume tolerance of the minimizer"))
      .key("Static_capillary_projected_gradient_tolerance",
           staticCapillary(&Static::projected_gradient_tolerance), anyReal(),
           setting("projected-gradient tolerance of the minimizer"))
      .key("Static_capillary_pressure_representability_max_residual_norm",
           staticCapillary(
               &Static::pressure_representability_max_residual_norm),
           anyReal(),
           setting("pressure-representability certificate: largest residual"))
      .key("Static_capillary_pressure_representability_max_relative_distance",
           staticCapillary(
               &Static::pressure_representability_max_relative_distance),
           anyReal(),
           setting("pressure-representability certificate: largest relative "
                   "distance"))
      .key("Static_capillary_physical_equilibrium_max_residual_norm",
           staticCapillary(&Static::physical_equilibrium_max_residual_norm),
           anyReal(),
           setting("physical-equilibrium certificate: largest residual"))
      .key("Static_capillary_constant_pressure_kkt_max_residual_norm",
           staticCapillary(&Static::constant_pressure_kkt_max_residual_norm),
           anyReal(),
           setting("constant-pressure KKT certificate: largest residual"))
      .key("Static_capillary_constant_pressure_kkt_max_relative_distance",
           staticCapillary(
               &Static::constant_pressure_kkt_max_relative_distance),
           anyReal(),
           setting("constant-pressure KKT certificate: largest relative "
                   "distance"))
      .key("Static_capillary_finite_difference_reference_coefficient_scale",
           staticCapillary(
               &Static::finite_difference_reference_coefficient_scale),
           anyReal(),
           setting("coefficient scale of the finite-difference step"))
      .key("Static_capillary_finite_difference_relative_step",
           staticCapillary(&Static::finite_difference_relative_step),
           anyReal(), setting("relative finite-difference step"))
      .key("Static_capillary_minimum_finite_difference_step",
           staticCapillary(&Static::minimum_finite_difference_step),
           anyReal(), setting("smallest finite-difference step"))
      .key("Static_capillary_finite_difference_max_shrinks",
           staticCapillary(&Static::finite_difference_max_shrinks),
           opt::IntCodec{}, setting("reductions of the finite-difference step"))
      .key("Static_capillary_max_iterations",
           staticCapillary(&Static::max_iterations), opt::IntCodec{},
           setting("iterations of the minimizer"))
      .key("Static_capillary_max_line_search_iterations",
           staticCapillary(&Static::max_line_search_iterations),
           opt::IntCodec{}, setting("line-search iterations"))
      .key("Static_capillary_max_topology_epoch_transitions",
           staticCapillary(&Static::max_topology_epoch_transitions),
           opt::IntCodec{},
           setting("cut-topology epochs of an exact derivative"))
      .key("Static_capillary_projected_gradient_inverse_stiffness",
           staticCapillary(&Static::projected_gradient_inverse_stiffness),
           anyReal(),
           setting("inverse stiffness of the projected-gradient step"))
      .key("Static_capillary_tangent_trust_radius",
           staticCapillary(&Static::tangent_trust_radius), anyReal(),
           setting("trust radius of the tangent step"))
      .key("Static_capillary_maximum_coefficient_update_linf",
           staticCapillary(&Static::maximum_coefficient_update_linf),
           anyReal(), setting("largest coefficient update (max norm)"))
      .key("Static_capillary_line_search_shrink",
           staticCapillary(&Static::line_search_shrink), anyReal(),
           setting("shrink factor of the line search"))
      .key("Static_capillary_armijo_fraction",
           staticCapillary(&Static::armijo_fraction), anyReal(),
           setting("Armijo fraction of the line search"))
      .key("Static_capillary_limited_memory_history_size",
           staticCapillary(&Static::limited_memory_history_size),
           opt::IntCodec{},
           setting("limited-memory history (0: projected gradient alone)"))
      .key("Static_capillary_limited_memory_curvature_tolerance",
           staticCapillary(&Static::limited_memory_curvature_tolerance),
           anyReal(), setting("curvature tolerance of the limited-memory update"))
      .key("Static_capillary_minimum_volume_merit_penalty",
           staticCapillary(&Static::minimum_volume_merit_penalty), anyReal(),
           setting("smallest penalty of the volume merit"))
      // A velocity source decides whether the velocity field is registered,
      // unless the element states it.
      .implies("Velocity_source", "Auto_register_velocity_field", "true",
               opt::ImplicationSettings{.when_trigger_is = "prescribed_data"})
      .implies("Velocity_source", "Auto_register_velocity_field", "false",
               opt::ImplicationSettings{.when_trigger_is = "constant_vector"})
      .implies("Velocity_source", "Auto_register_velocity_field", "false",
               opt::ImplicationSettings{
                   .when_trigger_is = "material_interface_phase_pair"})
      .rule("Source_velocity_field_name", [](const opt::ResolvedOptions<Keys>
                                                 &resolved) {
        return !resolved.value.source_velocity_field_name.empty() &&
                       !resolved.value.advection_velocity_extension_method
                   ? std::string(
                         "requires Advection_velocity_extension_method")
                   : std::string{};
      });
  return schema;
}

std::string lowerCopy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

} // namespace

const opt::OptionSchema<Keys> &levelSetEquationSchema() {
  static const auto schema = makeLevelSetEquationSchema();
  return schema;
}

opt::OptionInput
levelSetEquationInput(const svmp::Physics::ParameterMap &parameters) {
  opt::OptionInput input;
  for (const auto &[name, parameter] : parameters) {
    if (parameter.defined) {
      input.add(name, parameter.value);
    }
  }
  return input;
}

opt::OptionInput levelSetEquationInput(const EquationParameters &equation) {
  opt::OptionInput input;
  for (const auto &[name, parameter] : equation.params_map) {
    std::visit(
        [&](const auto *value) {
          if (value != nullptr && value->defined()) {
            input.add(name, value->svalue());
          }
        },
        parameter);
  }
  return input;
}

opt::OptionInput
levelSetDomainInput(const svmp::Physics::ParameterMap &parameters) {
  return levelSetEquationInput(parameters).excludeSpellings(
      levelSetEquationSchema().spellings(kEquationElementKeys));
}

opt::ResolvedOptions<Keys> resolveLevelSetEquationKeys(const opt::OptionInput &input,
                                                       Keys previous) {
  auto resolved = levelSetEquationSchema().resolve(input, std::move(previous));
  auto &transport = resolved.value.transport;
  if (resolved.isExplicit("Coupled") && resolved.value.coupled) {
    transport.operator_tag = "equations";
  }
  if (resolved.isExplicit("Constant_velocity")) {
    transport.velocity.source = ls::LevelSetVelocitySource::ConstantVector;
    transport.velocity.auto_register_field = false;
  }
  if (resolved.isExplicit("Interface_kinematic_marker")) {
    transport.interface_kinematic.enabled = true;
  }
  if (resolved.isExplicit(
          "Conservative_phase_impermeable_normal_velocity_tolerance")) {
    transport.conservative_phase
        .pointwise_impermeable_velocity_tolerance_explicitly_requested = true;
  }
  return resolved;
}

std::string resolvedLevelSetEquationJson(const opt::OptionInput &input,
                                         const Keys &keys) {
  const auto &schema = levelSetEquationSchema();
  return "{\"version\":1,\"equation\":" +
         schema.toJson(schema.describe(input, keys, Keys{})) + '}';
}

std::string trimCopy(std::string value) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  return value;
}

std::string normalizedToken(std::string value) {
  value = lowerCopy(trimCopy(std::move(value)));
  value.erase(std::remove_if(value.begin(), value.end(),
                             [](unsigned char c) {
                               return c == '_' || c == '-' || std::isspace(c);
                             }),
              value.end());
  return value;
}

int parseStrictPositiveInteger(std::string_view raw,
                               std::string_view context) {
  const auto text = trimCopy(std::string(raw));
  try {
    std::size_t consumed = 0u;
    const auto value = std::stoi(text, &consumed);
    if (consumed != text.size() || value < 1) {
      throw std::runtime_error("");
    }
    return value;
  } catch (...) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to parse positive integer "
        "value '" +
        std::string(raw) + "' for " + std::string(context) + ".");
  }
}

} // namespace application::translators::level_set::configuration
