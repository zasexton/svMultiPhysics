#include "Application/Translators/LevelSetConfigurationParsing.h"

#include "Parameters.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
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

// The keys that only the equation element gives: the domain elements of
// the installation do not read them.
const std::initializer_list<std::string_view> kEquationElementKeys{
    "Advection_velocity_extension_method",
    "Advection_velocity_extension_coupling",
    "Source_velocity_field_name",
    "Enable_static_capillary_equilibrium_initialization",
    "Static_capillary_volume_tolerance",
    "Static_capillary_projected_gradient_tolerance",
    "Static_capillary_pressure_representability_max_residual_norm",
    "Static_capillary_pressure_representability_max_relative_distance",
    "Static_capillary_physical_equilibrium_max_residual_norm",
    "Static_capillary_constant_pressure_kkt_max_residual_norm",
    "Static_capillary_constant_pressure_kkt_max_relative_distance",
    "Static_capillary_finite_difference_relative_step",
    "Static_capillary_max_iterations",
    "Static_capillary_max_topology_epoch_transitions",
    "Static_capillary_limited_memory_history_size",
    "Static_capillary_limited_memory_curvature_tolerance"};

opt::OptionSchema<Keys> makeLevelSetEquationSchema() {
  using Form = ls::LevelSetTransportForm;
  using FieldSource = ls::LevelSetFieldSource;
  using VelocitySource = ls::LevelSetVelocitySource;
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
           setting("name of the level-set field"))
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
      .key("Conservative_phase_reconcile_geometry",
           [](Keys &keys) -> auto & {
             return keys.transport.conservative_phase.reconcile_geometry;
           },
           relaxedBool(),
           setting("reconcile the level set with the indicator measure"))
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
      .key("Static_capillary_finite_difference_relative_step",
           staticCapillary(&Static::finite_difference_relative_step),
           anyReal(), setting("relative finite-difference step"))
      .key("Static_capillary_max_iterations",
           staticCapillary(&Static::max_iterations), opt::IntCodec{},
           setting("iterations of the minimizer"))
      .key("Static_capillary_max_topology_epoch_transitions",
           staticCapillary(&Static::max_topology_epoch_transitions),
           opt::IntCodec{},
           setting("cut-topology epochs of an exact derivative"))
      .key("Static_capillary_limited_memory_history_size",
           staticCapillary(&Static::limited_memory_history_size),
           opt::IntCodec{},
           setting("limited-memory history (0: projected gradient alone)"))
      .key("Static_capillary_limited_memory_curvature_tolerance",
           staticCapillary(&Static::limited_memory_curvature_tolerance),
           anyReal(), setting("curvature tolerance of the limited-memory update"))
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
      input.add(name, parameter.value, parameter.origin);
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
            input.add(name, value->svalue(), equation.parameter_origin(name));
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
