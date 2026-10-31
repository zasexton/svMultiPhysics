#include "Application/Translators/LevelSetConfigurationParsing.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

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

template <class E>
opt::EnumToken<E> token(E value, std::string canonical,
                        std::vector<std::string> synonyms = {}) {
  return opt::EnumToken<E>{value, std::move(canonical), std::move(synonyms)};
}

opt::OptionSchema<Keys> makeLevelSetEquationSchema() {
  using Form = ls::LevelSetTransportForm;
  using FieldSource = ls::LevelSetFieldSource;
  using VelocitySource = ls::LevelSetVelocitySource;
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
      // A velocity source decides whether the velocity field is registered,
      // unless the element states it.
      .implies("Velocity_source", "Auto_register_velocity_field", "true",
               opt::ImplicationSettings{.when_trigger_is = "prescribed_data"})
      .implies("Velocity_source", "Auto_register_velocity_field", "false",
               opt::ImplicationSettings{.when_trigger_is = "constant_vector"})
      .implies("Velocity_source", "Auto_register_velocity_field", "false",
               opt::ImplicationSettings{
                   .when_trigger_is = "material_interface_phase_pair"});
  return schema;
}

std::string lowerCopy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool relaxedBoolean(std::string_view raw) {
  const auto value = lowerCopy(trimCopy(std::string(raw)));
  return value == "true" || value == "1" || value == "yes" || value == "on";
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

LevelSetConfigurationReader::LevelSetConfigurationReader(
    const svmp::Physics::ParameterMap &parameters,
    LevelSetInputPolicy policy) noexcept
    : parameters_(&parameters), policy_(policy) {}

std::optional<LevelSetSelectedParameter>
LevelSetConfigurationReader::selected(std::span<const std::string_view> aliases,
                                      std::string_view canonical_key) const {
  for (const auto alias : aliases) {
    const auto found = parameters_->find(std::string(alias));
    if (found == parameters_->end()) {
      continue;
    }
    const auto trimmed = trimCopy(found->second.value);
    if (trimmed.empty() || (policy_ == LevelSetInputPolicy::Installation &&
                            !found->second.defined)) {
      continue;
    }
    return LevelSetSelectedParameter{
        .canonical_key = canonical_key.empty() ? std::string(alias)
                                               : std::string(canonical_key),
        .selected_spelling = std::string(alias),
        .text = policy_ == LevelSetInputPolicy::Installation
                    ? trimmed
                    : found->second.value,
        .supplied = found->second.defined,
        .compatibility_fallback =
            policy_ == LevelSetInputPolicy::LegacyMaintenance &&
            !found->second.defined,
    };
  }
  return std::nullopt;
}

std::optional<LevelSetSelectedParameter> LevelSetConfigurationReader::selected(
    std::initializer_list<std::string_view> aliases,
    std::string_view canonical_key) const {
  return selected(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      canonical_key);
}

std::optional<LevelSetSelectedParameter>
LevelSetConfigurationReader::string(std::span<const std::string_view> aliases,
                                    std::string_view canonical_key) {
  const auto selection = selected(aliases, canonical_key);
  if (selection) {
    observe(*selection);
  }
  return selection;
}

std::optional<LevelSetSelectedParameter> LevelSetConfigurationReader::string(
    std::initializer_list<std::string_view> aliases,
    std::string_view canonical_key) {
  return string(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      canonical_key);
}

void LevelSetConfigurationReader::observe(
    const LevelSetSelectedParameter &selection) {
  selections_.push_back(selection);
}

std::optional<bool>
LevelSetConfigurationReader::boolean(std::span<const std::string_view> aliases,
                                     std::string_view canonical_key) {
  const auto selection = selected(aliases, canonical_key);
  if (!selection) {
    return std::nullopt;
  }
  observe(*selection);
  return relaxedBoolean(selection->text);
}

std::optional<bool> LevelSetConfigurationReader::boolean(
    std::initializer_list<std::string_view> aliases,
    std::string_view canonical_key) {
  return boolean(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      canonical_key);
}

std::optional<svmp::FE::Real>
LevelSetConfigurationReader::real(std::span<const std::string_view> aliases,
                                  std::string_view context) {
  const auto selection = selected(aliases, context);
  if (!selection) {
    return std::nullopt;
  }
  observe(*selection);
  if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
    return static_cast<svmp::FE::Real>(std::stod(selection->text));
  }
  try {
    std::size_t consumed = 0u;
    const auto value = std::stod(selection->text, &consumed);
    if (consumed != selection->text.size() || !std::isfinite(value)) {
      throw std::runtime_error("");
    }
    return static_cast<svmp::FE::Real>(value);
  } catch (...) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to parse numeric value '" +
        selection->text + "' for " + std::string(context) + ".");
  }
}

std::optional<svmp::FE::Real> LevelSetConfigurationReader::real(
    std::initializer_list<std::string_view> aliases, std::string_view context) {
  return real(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      context);
}

std::optional<int>
LevelSetConfigurationReader::integer(std::span<const std::string_view> aliases,
                                     std::string_view context,
                                     bool positive_for_installation) {
  const auto selection = selected(aliases, context);
  if (!selection) {
    return std::nullopt;
  }
  observe(*selection);
  if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
    return std::stoi(selection->text);
  }
  if (positive_for_installation) {
    return parseStrictPositiveInteger(selection->text, context);
  }
  try {
    std::size_t consumed = 0u;
    const auto value = std::stoi(selection->text, &consumed);
    if (consumed != selection->text.size()) {
      throw std::runtime_error("");
    }
    return value;
  } catch (...) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to parse integer value '" +
        selection->text + "' for " + std::string(context) + ".");
  }
}

std::optional<int> LevelSetConfigurationReader::integer(
    std::initializer_list<std::string_view> aliases, std::string_view context,
    bool positive_for_installation) {
  return integer(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      context, positive_for_installation);
}

std::optional<std::array<svmp::FE::Real, 3>>
LevelSetConfigurationReader::vector3(std::span<const std::string_view> aliases,
                                     std::string_view context,
                                     std::string_view canonical_key) {
  const auto selection =
      selected(aliases, canonical_key.empty() ? context : canonical_key);
  if (!selection) {
    return std::nullopt;
  }
  observe(*selection);
  auto text = selection->text;
  if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
    std::replace(text.begin(), text.end(), ',', ' ');
  }
  std::istringstream input(text);
  std::array<svmp::FE::Real, 3> result{};
  for (auto &value : result) {
    double parsed = 0.0;
    if (!(input >> parsed) || !std::isfinite(parsed)) {
      if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] " + std::string(context) +
            " must contain exactly three finite numeric components.");
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Failed to parse three numeric "
          "components for " +
          std::string(context) + ".");
    }
    value = static_cast<svmp::FE::Real>(parsed);
  }
  std::string trailing;
  if (input >> trailing) {
    if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] " + std::string(context) +
          " must contain exactly three finite numeric components.");
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to parse three numeric "
        "components for " +
        std::string(context) + ".");
  }
  return result;
}

std::optional<std::array<svmp::FE::Real, 3>>
LevelSetConfigurationReader::vector3(
    std::initializer_list<std::string_view> aliases, std::string_view context,
    std::string_view canonical_key) {
  return vector3(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      context, canonical_key);
}

const std::vector<LevelSetSelectedParameter> &
LevelSetConfigurationReader::selections() const noexcept {
  return selections_;
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

svmp::FE::level_set::LevelSetTransportForm
parseTransportForm(std::string_view raw) {
  const auto value = normalizedToken(std::string(raw));
  using Form = svmp::FE::level_set::LevelSetTransportForm;
  if (value == "advective" || value == "classical" || value == "standard") {
    return Form::Advective;
  }
  if (value == "conservative" || value == "conservativedivergence" ||
      value == "divergence" || value == "divergenceform") {
    return Form::ConservativeDivergence;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Level-set Transport_form must be one of "
      "'advective' or 'conservative_divergence'.");
}

svmp::FE::level_set::LevelSetVelocitySource
parseVelocitySource(std::string_view raw, LevelSetInputPolicy policy) {
  const auto value = normalizedToken(std::string(raw));
  using Source = svmp::FE::level_set::LevelSetVelocitySource;
  if ((policy == LevelSetInputPolicy::LegacyMaintenance && value.empty()) ||
      value == "coupled" || value == "coupledfield" ||
      value == "unknown" || (policy == LevelSetInputPolicy::Installation &&
                             value == "navierstokes")) {
    return Source::CoupledField;
  }
  if (value == "prescribed" || value == "prescribeddata" || value == "data") {
    return Source::PrescribedData;
  }
  if (value == "constant" || value == "constantvector") {
    return Source::ConstantVector;
  }
  if (value == "materialinterfacephasepair") {
    return Source::MaterialInterfacePhasePair;
  }
  if (policy == LevelSetInputPolicy::LegacyMaintenance) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set Velocity_source must be one "
        "of 'coupled_field', 'prescribed_data', 'constant_vector', or "
        "'material_interface_phase_pair'.");
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Velocity_source must be one of "
      "'coupled_field', 'prescribed_data', 'constant', or "
      "'material_interface_phase_pair'.");
}

svmp::FE::level_set::LevelSetPhaseSide parsePhaseSide(std::string_view raw) {
  const auto value = normalizedToken(std::string(raw));
  using Side = svmp::FE::level_set::LevelSetPhaseSide;
  if (value == "negative" || value == "minus") {
    return Side::Negative;
  }
  if (value == "positive" || value == "plus") {
    return Side::Positive;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Conservative_phase_liquid_side must be "
      "'negative' or 'positive'.");
}

svmp::FE::level_set::LevelSetConservativePhaseBoundaryFluxPolicy
parseConservativePhaseBoundaryFluxPolicy(std::string_view raw) {
  const auto value = normalizedToken(std::string(raw));
  using Policy =
      svmp::FE::level_set::LevelSetConservativePhaseBoundaryFluxPolicy;
  if (value == "closeddomaindiscreteqfluxonly") {
    return Policy::ClosedDomainDiscreteQFluxOnly;
  }
  if (value == "globallybalanceddiscreteqflux") {
    return Policy::GloballyBalancedDiscreteQFlux;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Conservative_phase_boundary_flux_policy "
      "must be 'closed_domain_discrete_q_flux_only' or "
      "'globally_balanced_discrete_q_flux'.");
}

} // namespace application::translators::level_set::configuration
