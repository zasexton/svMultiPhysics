#include "Application/Core/LevelSetMaintenanceConfiguration.h"

#include "Application/Translators/LevelSetConfigurationParsing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace application::core {
namespace {

namespace ls = svmp::FE::level_set;
namespace parsing = application::translators::level_set::configuration;
namespace aliases = parsing::aliases;

using Reader = parsing::LevelSetConfigurationReader;
using Selection = parsing::LevelSetSelectedParameter;

constexpr auto kInstallationPolicy = parsing::LevelSetInputPolicy::Installation;
constexpr auto kLegacyPolicy = parsing::LevelSetInputPolicy::LegacyMaintenance;

std::string jsonString(std::string_view value) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() + 2u);
  result.push_back('"');
  for (const unsigned char c : value) {
    switch (c) {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\b':
      result += "\\b";
      break;
    case '\f':
      result += "\\f";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (c < 0x20u) {
        result += "\\u00";
        result.push_back(hex[(c >> 4u) & 0x0fu]);
        result.push_back(hex[c & 0x0fu]);
      } else {
        result.push_back(static_cast<char>(c));
      }
      break;
    }
  }
  result.push_back('"');
  return result;
}

std::string jsonReal(svmp::FE::Real value) {
  if (std::isnan(value)) {
    return "\"nan\"";
  }
  if (std::isinf(value)) {
    return value < 0.0 ? "\"-inf\"" : "\"inf\"";
  }
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<svmp::FE::Real>::max_digits10)
      << value;
  return out.str();
}

constexpr const char *jsonBool(bool value) noexcept {
  return value ? "true" : "false";
}

const char *transportFormName(ls::LevelSetTransportForm value) noexcept {
  return value == ls::LevelSetTransportForm::Advective
             ? "advective"
             : "conservative_divergence";
}

const char *velocitySourceName(ls::LevelSetVelocitySource value) noexcept {
  switch (value) {
  case ls::LevelSetVelocitySource::CoupledField:
    return "coupled_field";
  case ls::LevelSetVelocitySource::PrescribedData:
    return "prescribed_data";
  case ls::LevelSetVelocitySource::ConstantVector:
    return "constant_vector";
  case ls::LevelSetVelocitySource::MaterialInterfacePhasePair:
    return "material_interface_phase_pair";
  }
  return "unknown";
}

const char *phaseSideName(ls::LevelSetPhaseSide value) noexcept {
  return value == ls::LevelSetPhaseSide::Negative ? "negative" : "positive";
}

const char *
reinitializationMethodName(ls::LevelSetReinitializationMethod value) noexcept {
  switch (value) {
  case ls::LevelSetReinitializationMethod::HamiltonJacobiPDE:
    return "hamilton_jacobi_pde";
  case ls::LevelSetReinitializationMethod::FastMarching:
    return "fast_marching";
  case ls::LevelSetReinitializationMethod::Projection:
    return "projection";
  }
  return "unknown";
}

void appendSelections(const Reader &reader, std::size_t &cursor,
                      std::vector<LevelSetInputObservation> &observations,
                      std::string_view source_layer,
                      std::string_view representation) {
  const auto &selections = reader.selections();
  while (cursor < selections.size()) {
    const auto &selected = selections[cursor++];
    observations.push_back(LevelSetInputObservation{
        .canonical_key = selected.canonical_key,
        .selected_spelling = selected.selected_spelling,
        .source_layer = std::string(source_layer),
        .supplied = selected.supplied,
        .representation = std::string(representation),
        .compatibility_fallback = selected.compatibility_fallback,
        .ordered_overrides = {},
    });
  }
}

void appendDerivedObservation(
    std::vector<LevelSetInputObservation> &observations,
    std::string_view canonical_key, std::string_view source_layer,
    std::string_view representation = "derived") {
  observations.push_back(LevelSetInputObservation{
      .canonical_key = std::string(canonical_key),
      .selected_spelling = {},
      .source_layer = std::string(source_layer),
      .supplied = false,
      .representation = std::string(representation),
      .compatibility_fallback = false,
      .ordered_overrides = {},
  });
}

std::optional<Selection>
selectedFromLayer(const svmp::Physics::ParameterMap &parameters,
                  std::span<const std::string_view> names,
                  std::string_view canonical) {
  Reader reader(parameters, kInstallationPolicy);
  return reader.selected(names, canonical);
}

void appendInstallationObservationImpl(
    std::vector<LevelSetInputObservation> &observations,
    const svmp::Physics::EquationModuleInput &input,
    std::span<const std::string_view> names, std::string_view canonical) {
  std::optional<Selection> effective;
  std::string effective_layer;
  std::vector<std::string> overrides;
  const auto visit = [&](const svmp::Physics::ParameterMap &parameters,
                         std::string layer) {
    const auto selected = selectedFromLayer(parameters, names, canonical);
    if (!selected) {
      return;
    }
    effective = selected;
    effective_layer = layer;
    overrides.push_back(std::move(layer));
  };
  visit(input.equation_params, "equation");
  visit(input.default_domain.params, "default_domain");
  for (const auto &domain : input.domains) {
    visit(domain.params, "domain:" + domain.id);
  }
  if (!effective) {
    return;
  }
  observations.push_back(LevelSetInputObservation{
      .canonical_key = effective->canonical_key,
      .selected_spelling = effective->selected_spelling,
      .source_layer = std::move(effective_layer),
      .supplied = effective->supplied,
      .representation = "installation_snapshot",
      .compatibility_fallback = false,
      .ordered_overrides = std::move(overrides),
  });
}

template <std::size_t N>
void appendInstallationObservation(
    std::vector<LevelSetInputObservation> &observations,
    const svmp::Physics::EquationModuleInput &input,
    const std::array<std::string_view, N> &names, std::string_view canonical) {
  appendInstallationObservationImpl(observations, input, names, canonical);
}

void appendInstallationObservations(
    std::vector<LevelSetInputObservation> &observations,
    const ResolvedLevelSetEquationHandle &installation) {
  if (!installation) {
    return;
  }
  const bool has_chronological_velocity_source_history =
      std::any_of(installation->input_observations.begin(),
                  installation->input_observations.end(),
                  [](const auto &observation) {
                    return observation.canonical_key == "velocity_source";
                  });
  const bool has_chronological_constant_velocity_history =
      std::any_of(installation->input_observations.begin(),
                  installation->input_observations.end(),
                  [](const auto &observation) {
                    return observation.canonical_key == "constant_velocity";
                  });
  if (installation->input_snapshot) {
    const auto &input = installation->input_snapshot->installation_input;
    appendInstallationObservation(observations, input,
                                  aliases::level_set_field_name,
                                  "level_set_field_name");
    appendInstallationObservation(observations, input, aliases::transport_form,
                                  "transport_form");
    if (!has_chronological_velocity_source_history) {
      appendInstallationObservation(
          observations, input, aliases::velocity_source, "velocity_source");
    }
    if (!has_chronological_constant_velocity_history) {
      appendInstallationObservation(observations, input,
                                    aliases::constant_velocity,
                                    "constant_velocity");
    }
    appendInstallationObservation(
        observations, input, aliases::conservative_phase_enable,
        "enable_conservative_phase_transport");
    appendInstallationObservation(observations, input,
                                  aliases::conservative_phase_side,
                                  "conservative_phase_liquid_side");
    appendInstallationObservation(observations, input,
                                  aliases::reinitialization_enable,
                                  "enable_reinitialization");
    appendInstallationObservation(observations, input,
                                  aliases::reinitialization_method,
                                  "reinitialization_method");
    appendInstallationObservation(observations, input,
                                  aliases::reinitialization_cadence,
                                  "reinitialization_cadence_steps");
    appendInstallationObservation(observations, input,
                                  aliases::volume_correction_enable,
                                  "enable_volume_correction");
    appendInstallationObservation(observations, input,
                                  aliases::volume_correction_tolerance,
                                  "volume_correction_tolerance");
    appendInstallationObservation(observations, input,
                                  aliases::curvature_field,
                                  "curvature_field_name");
  }

  observations.insert(observations.end(),
                      installation->input_observations.begin(),
                      installation->input_observations.end());

  const bool source_backed = !installation->input_observations.empty();
  constexpr std::array<std::string_view, 14> covered_keys{
      "level_set_field_name",
      "transport_form",
      "velocity_source",
      "velocity_auto_register_field",
      "velocity_source_field_name",
      "velocity_space_present",
      "enable_conservative_phase_transport",
      "conservative_phase_liquid_side",
      "enable_reinitialization",
      "reinitialization_method",
      "reinitialization_cadence_steps",
      "enable_volume_correction",
      "volume_correction_tolerance",
      "curvature_field_name",
  };
  for (const auto key : covered_keys) {
    const bool observed =
        std::any_of(observations.begin(), observations.end(),
                    [&](const auto &item) {
                      return item.canonical_key == key &&
                             (item.representation == "installation_snapshot" ||
                              item.representation == "installation_derived" ||
                              item.representation == "installation_default");
                    });
    if (observed) {
      continue;
    }
    observations.push_back(LevelSetInputObservation{
        .canonical_key = std::string(key),
        .selected_spelling = {},
        .source_layer = source_backed ? "fe_default" : "unavailable",
        .supplied = false,
        .representation =
            source_backed ? "installation_default" : "unavailable",
        .compatibility_fallback = false,
        .ordered_overrides = {},
    });
  }
}

void appendLegacyDefaults(std::vector<LevelSetInputObservation> &observations) {
  constexpr std::array<std::string_view, 17> keys{
      "level_set_field_name",
      "level_set_isovalue",
      "transport_form",
      "velocity_source",
      "velocity_auto_register_field",
      "velocity_source_field_name",
      "velocity_space_present",
      "enable_conservative_phase_transport",
      "conservative_phase_liquid_side",
      "enable_reinitialization",
      "reinitialization_method",
      "reinitialization_cadence_steps",
      "enable_volume_correction",
      "volume_correction_tolerance",
      "enable_static_capillary_equilibrium_initialization",
      "enable_curvature_projection",
      "curvature_field_name",
  };
  const auto has_effective_observation = [&](std::string_view key) {
    return std::any_of(observations.begin(), observations.end(),
                       [&](const auto &item) {
                         return item.canonical_key == key &&
                                (item.representation == "legacy_getter" ||
                                 item.representation == "derived");
                       });
  };
  for (const auto key : keys) {
    if (!has_effective_observation(key)) {
      observations.push_back(LevelSetInputObservation{
          .canonical_key = std::string(key),
          .selected_spelling = {},
          .source_layer = "fe_default",
          .supplied = false,
          .representation = "fe_default",
          .compatibility_fallback = false,
          .ordered_overrides = {},
      });
    }
  }
}

bool isInstallationObservation(
    const LevelSetInputObservation &observation) noexcept {
  return observation.representation == "installation_snapshot" ||
         observation.representation == "installation_derived" ||
         observation.representation == "installation_default" ||
         observation.representation == "unavailable";
}

std::optional<ActiveCutVolumeRequest> matchingActiveCutVolumeRequest(
    std::span<const ActiveCutVolumeRequest> active_requests,
    const std::string &level_set_field_name, double isovalue) {
  constexpr double tolerance = 1.0e-12;
  for (const auto &request : active_requests) {
    if (request.level_set_field_name == level_set_field_name &&
        std::abs(request.isovalue - isovalue) <= tolerance) {
      return request;
    }
  }
  for (const auto &request : active_requests) {
    if (request.level_set_field_name == level_set_field_name) {
      return request;
    }
  }
  return std::nullopt;
}

void appendObservationJson(std::ostringstream &out,
                           const LevelSetInputObservation &observation) {
  out << "{\"canonical_key\":" << jsonString(observation.canonical_key)
      << ",\"selected_spelling\":" << jsonString(observation.selected_spelling)
      << ",\"source_layer\":" << jsonString(observation.source_layer)
      << ",\"supplied\":" << jsonBool(observation.supplied)
      << ",\"representation\":" << jsonString(observation.representation)
      << ",\"compatibility_fallback\":"
      << jsonBool(observation.compatibility_fallback)
      << ",\"ordered_overrides\":[";
  for (std::size_t i = 0; i < observation.ordered_overrides.size(); ++i) {
    if (i != 0u) {
      out << ',';
    }
    out << jsonString(observation.ordered_overrides[i]);
  }
  out << "]}";
}

} // namespace

std::optional<LevelSetMaintenanceConfigurationHandle>
resolveLegacyLevelSetMaintenanceConfiguration(
    const LegacyLevelSetMaintenanceInput &input,
    std::span<const ActiveCutVolumeRequest> active_requests,
    ResolvedLevelSetEquationHandle installation) {
  if (!input.equation_type_defined) {
    return std::nullopt;
  }
  const auto equation_type = parsing::normalizedToken(input.equation_type);
  if (equation_type != "levelset" && equation_type != "levelsettransport") {
    return std::nullopt;
  }

  auto result =
      std::make_shared<ResolvedLevelSetMaintenanceCompatibilityConfiguration>();
  result->installation = std::move(installation);
  appendInstallationObservations(result->input_observations,
                                 result->installation);

  Reader reader(input.equation_parameters, kLegacyPolicy);
  std::size_t observation_cursor = 0u;
  const auto append_equation_observations = [&] {
    appendSelections(reader, observation_cursor, result->input_observations,
                     "equation", "legacy_getter");
  };
  const auto apply_boolean = [&](auto names, std::string_view canonical,
                                 bool &target) {
    if (const auto value = reader.boolean(names, canonical)) {
      target = *value;
    }
  };
  const auto apply_real = [&](auto names, std::string_view canonical,
                              auto &target) {
    if (const auto value = reader.real(names, canonical)) {
      target = static_cast<std::remove_reference_t<decltype(target)>>(*value);
    }
  };
  const auto apply_integer = [&](auto names, std::string_view canonical,
                                 int &target) {
    if (const auto value = reader.integer(names, canonical)) {
      target = *value;
    }
  };
  const auto apply_boolean_list =
      [&](std::initializer_list<std::string_view> names,
          std::string_view canonical, bool &target) {
        if (const auto value = reader.boolean(names, canonical)) {
          target = *value;
        }
      };
  const auto apply_real_list = [&](
      std::initializer_list<std::string_view> names, std::string_view canonical,
      auto &target) {
    if (const auto value = reader.real(names, canonical)) {
      target = static_cast<std::remove_reference_t<decltype(target)>>(*value);
    }
  };
  const auto apply_integer_list =
      [&](std::initializer_list<std::string_view> names,
          std::string_view canonical, int &target) {
        if (const auto value = reader.integer(names, canonical)) {
          target = *value;
        }
      };

  if (const auto field = reader.string(aliases::level_set_field_name,
                                       "level_set_field_name")) {
    result->transport.level_set.field_name = parsing::trimCopy(field->text);
  }
  apply_real(aliases::level_set_isovalue, "level_set_isovalue",
             result->isovalue);
  if (const auto form =
          reader.string(aliases::transport_form, "transport_form")) {
    result->transport.transport_form = parsing::parseTransportForm(form->text);
  }

  if (const auto field =
          reader.string(aliases::velocity_field_name, "velocity_field_name")) {
    result->transport.velocity.field_name = parsing::trimCopy(field->text);
  }
  if (const auto source =
          reader.string(aliases::velocity_source, "velocity_source")) {
    result->transport.velocity.source =
        parsing::parseVelocitySource(source->text, kLegacyPolicy);
  }
  apply_integer(aliases::material_interface_marker, "material_interface_marker",
                result->transport.velocity.material_interface_marker);
  const bool wet_extension_enabled =
      reader
          .boolean(aliases::wet_extension_enable,
                   "use_wet_extension_advection_velocity")
          .value_or(false) ||
      reader
          .string(aliases::wet_extension_source,
                  "advection_velocity_from_field")
          .has_value();
  if (wet_extension_enabled) {
    append_equation_observations();
    result->transport.velocity.source =
        ls::LevelSetVelocitySource::CoupledField;
    appendDerivedObservation(result->input_observations, "velocity_source",
                             "derived:wet_extension");
  }
  if (const auto constant =
          reader.vector3(aliases::constant_velocity, "Constant_velocity",
                         "constant_velocity")) {
    append_equation_observations();
    result->transport.velocity.source =
        ls::LevelSetVelocitySource::ConstantVector;
    result->transport.velocity.constant_value = *constant;
    appendDerivedObservation(result->input_observations, "velocity_source",
                             "derived:constant_velocity");
  }
  append_equation_observations();

  apply_boolean(aliases::conservative_phase_enable,
                "enable_conservative_phase_transport",
                result->transport.conservative_phase.enabled);
  if (const auto field = reader.string(aliases::conservative_phase_field,
                                       "conservative_phase_field_name")) {
    result->transport.conservative_phase.liquid_indicator.field_name =
        parsing::trimCopy(field->text);
  }
  apply_boolean(aliases::conservative_phase_auto_register,
                "auto_register_conservative_phase_field",
                result->transport.conservative_phase.liquid_indicator
                    .auto_register_field);
  if (const auto side = reader.string(aliases::conservative_phase_side,
                                      "conservative_phase_liquid_side")) {
    result->transport.conservative_phase.liquid_side =
        parsing::parsePhaseSide(side->text);
  }
  apply_real(
      std::array<std::string_view, 2>{"Conservative_phase_invariant_tolerance",
                                      "ConservativePhaseInvariantTolerance"},
      "conservative_phase_invariant_tolerance",
      result->transport.conservative_phase.invariant_tolerance);
  apply_real(
      std::array<std::string_view, 2>{
          "Conservative_phase_component_activity_tolerance",
          "ConservativePhaseComponentActivityTolerance"},
      "conservative_phase_component_activity_tolerance",
      result->transport.conservative_phase.component_activity_tolerance);
  apply_real(
      std::array<std::string_view, 2>{"Conservative_phase_maximum_courant",
                                      "ConservativePhaseMaximumCourant"},
      "conservative_phase_maximum_courant",
      result->transport.conservative_phase.maximum_courant);
  apply_boolean(
      std::array<std::string_view, 2>{
          "Conservative_phase_enforce_courant_limit",
          "ConservativePhaseEnforceCourantLimit"},
      "conservative_phase_enforce_courant_limit",
      result->transport.conservative_phase.enforce_courant_limit);
  apply_boolean(
      std::array<std::string_view, 2>{
          "Conservative_phase_require_constant_preservation",
          "ConservativePhaseRequireConstantPreservation"},
      "conservative_phase_require_constant_preservation",
      result->transport.conservative_phase.require_constant_preservation);
  apply_real(
      std::array<std::string_view, 2>{
          "Conservative_phase_momentum_relative_tolerance",
          "ConservativePhaseMomentumRelativeTolerance"},
      "conservative_phase_momentum_relative_tolerance",
      result->transport.conservative_phase.momentum_relative_tolerance);
  apply_boolean(
      std::array<std::string_view, 2>{"Conservative_phase_write_flux_artifacts",
                                      "ConservativePhaseWriteFluxArtifacts"},
      "conservative_phase_write_flux_artifacts",
      result->transport.conservative_phase.write_flux_artifacts);
  apply_integer(
      std::array<std::string_view, 2>{
          "Conservative_phase_flux_artifact_cadence_steps",
          "ConservativePhaseFluxArtifactCadenceSteps"},
      "conservative_phase_flux_artifact_cadence_steps",
      result->transport.conservative_phase.flux_artifact_cadence_steps);
  apply_boolean(
      std::array<std::string_view, 2>{
          "Conservative_phase_classify_nonprimary_components_as_satellites",
          "ConservativePhaseClassifyNonprimaryComponentsAsSatellites"},
      "conservative_phase_classify_nonprimary_components_as_satellites",
      result->transport.conservative_phase
          .classify_nonprimary_components_as_satellites);
  if (const auto policy =
          reader.string({"Conservative_phase_boundary_flux_policy",
                         "ConservativePhaseBoundaryFluxPolicy"},
                        "conservative_phase_boundary_flux_policy")) {
    result->transport.conservative_phase.boundary_flux_policy =
        parsing::parseConservativePhaseBoundaryFluxPolicy(policy->text);
  }
  if (const auto regions =
          reader.string({"Conservative_phase_fixed_flux_regions",
                         "ConservativePhaseFixedFluxRegions"},
                        "conservative_phase_fixed_flux_regions")) {
    result->transport.conservative_phase.fixed_flux_regions =
        ls::parseLevelSetPhaseRegionBoxes(regions->text);
  }
  if (const auto tolerance = reader.real(
          {"Conservative_phase_impermeable_normal_velocity_tolerance",
           "ConservativePhaseImpermeableNormalVelocityTolerance"},
          "conservative_phase_impermeable_normal_velocity_tolerance")) {
    result->transport.conservative_phase.impermeable_normal_velocity_tolerance =
        *tolerance;
    result->transport.conservative_phase
        .pointwise_impermeable_velocity_tolerance_explicitly_requested = true;
  }
  apply_boolean(
      std::array<std::string_view, 2>{"Conservative_phase_reconcile_geometry",
                                      "ConservativePhaseReconcileGeometry"},
      "conservative_phase_reconcile_geometry",
      result->transport.conservative_phase.reconcile_geometry);
  apply_real(
      std::array<std::string_view, 2>{
          "Conservative_phase_geometry_measure_tolerance",
          "ConservativePhaseGeometryMeasureTolerance"},
      "conservative_phase_geometry_measure_tolerance",
      result->transport.conservative_phase.geometry_measure_tolerance);
  apply_integer(
      std::array<std::string_view, 2>{
          "Conservative_phase_geometry_correction_max_iterations",
          "ConservativePhaseGeometryCorrectionMaxIterations"},
      "conservative_phase_geometry_correction_max_iterations",
      result->transport.conservative_phase.geometry_correction_max_iterations);
  apply_real(
      std::array<std::string_view, 2>{
          "Conservative_phase_maximum_geometry_displacement_fraction",
          "ConservativePhaseMaximumGeometryDisplacementFraction"},
      "conservative_phase_maximum_geometry_displacement_fraction",
      result->transport.conservative_phase
          .maximum_geometry_displacement_fraction);

  apply_boolean_list({"Enable_bound_preserving_limiter",
                      "EnableBoundPreservingLimiter",
                      "Bound_preserving_limiter", "BoundPreservingLimiter"},
                     "enable_bound_preserving_limiter",
                     result->transport.bound_preserving.enabled);
  apply_real_list(
      {"Bound_preserving_bound_tolerance", "BoundPreservingBoundTolerance"},
      "bound_preserving_bound_tolerance",
      result->transport.bound_preserving.bound_tolerance);
  apply_real_list(
      {"Bound_preserving_sign_tolerance", "BoundPreservingSignTolerance"},
      "bound_preserving_sign_tolerance",
      result->transport.bound_preserving.sign_tolerance);
  apply_real_list(
      {"Bound_preserving_courant_tolerance", "BoundPreservingCourantTolerance"},
      "bound_preserving_courant_tolerance",
      result->transport.bound_preserving.courant_tolerance);
  apply_real_list(
      {"Bound_preserving_maximum_courant", "BoundPreservingMaximumCourant"},
      "bound_preserving_maximum_courant",
      result->transport.bound_preserving.maximum_courant);
  apply_boolean_list({"Bound_preserving_enforce_courant_limit",
                      "BoundPreservingEnforceCourantLimit"},
                     "bound_preserving_enforce_courant_limit",
                     result->transport.bound_preserving.enforce_courant_limit);
  apply_boolean_list(
      {"Bound_preserving_enforce_impermeable_boundaries",
       "BoundPreservingEnforceImpermeableBoundaries"},
      "bound_preserving_enforce_impermeable_boundaries",
      result->transport.bound_preserving.enforce_impermeable_boundaries);
  apply_real_list(
      {"Bound_preserving_impermeable_normal_velocity_tolerance",
       "BoundPreservingImpermeableNormalVelocityTolerance"},
      "bound_preserving_impermeable_normal_velocity_tolerance",
      result->transport.bound_preserving.impermeable_normal_velocity_tolerance);
  append_equation_observations();

  for (const auto &boundary : input.boundaries) {
    if (!boundary.type_defined || !boundary.name_defined) {
      continue;
    }
    const auto type = parsing::normalizedToken(boundary.type);
    const bool inflow = type == "levelsetinflow" || type == "inflow" ||
                        type == "levelsetdirichlet";
    const bool outflow = type == "levelsetoutflow" || type == "outflow";
    if (!inflow && !outflow) {
      continue;
    }
    ResolvedLevelSetMaintenanceCompatibilityConfiguration::OpenBoundary open;
    open.face_name = parsing::trimCopy(boundary.name);
    open.inflow = inflow;
    if (inflow) {
      Reader boundary_reader(boundary.parameters, kLegacyPolicy);
      if (const auto value = boundary_reader.real({"Value", "Level_set_value"},
                                                  "open_boundary_value")) {
        open.literal_inflow_value = *value;
      }
      std::size_t boundary_cursor = 0u;
      appendSelections(boundary_reader, boundary_cursor,
                       result->input_observations, "boundary:" + open.face_name,
                       "legacy_getter");
    }
    result->open_boundaries.push_back(std::move(open));
  }

  apply_boolean(aliases::reinitialization_enable, "enable_reinitialization",
                result->transport.reinitialization.enabled);
  if (const auto method = reader.string(aliases::reinitialization_method,
                                        "reinitialization_method")) {
    result->transport.reinitialization.method =
        parsing::parseReinitializationMethod(method->text, kLegacyPolicy);
  }
  apply_integer(aliases::reinitialization_cadence,
                "reinitialization_cadence_steps",
                result->transport.reinitialization.cadence_steps);
  apply_integer(aliases::reinitialization_iterations,
                "reinitialization_max_iterations",
                result->transport.reinitialization.max_iterations);
  apply_real(aliases::reinitialization_pseudo_time_step,
             "reinitialization_pseudo_time_step_scale",
             result->transport.reinitialization.pseudo_time_step_scale);
  apply_real(aliases::reinitialization_interface_band,
             "reinitialization_interface_band_width",
             result->transport.reinitialization.interface_band_width);
  apply_real(aliases::reinitialization_signed_distance_tolerance,
             "reinitialization_signed_distance_tolerance",
             result->transport.reinitialization.signed_distance_tolerance);
  apply_real(aliases::reinitialization_max_zero_set_displacement,
             "reinitialization_max_zero_set_displacement",
             result->transport.reinitialization.max_zero_set_displacement);
  append_equation_observations();

  apply_boolean(aliases::volume_correction_enable, "enable_volume_correction",
                result->transport.volume_correction.enabled);
  apply_integer(aliases::volume_correction_cadence,
                "volume_correction_cadence_steps",
                result->transport.volume_correction.cadence_steps);
  apply_boolean(aliases::volume_correction_use_initial,
                "volume_correction_use_initial_volume",
                result->transport.volume_correction
                    .use_initial_negative_volume_as_target);
  if (const auto target =
          reader.real(aliases::volume_correction_target,
                      "volume_correction_target_negative_volume")) {
    result->transport.volume_correction.target_negative_volume = *target;
    result->transport.volume_correction.use_initial_negative_volume_as_target =
        false;
  }
  apply_real(aliases::volume_correction_tolerance,
             "volume_correction_tolerance",
             result->transport.volume_correction.volume_tolerance);
  apply_integer(aliases::volume_correction_iterations,
                "volume_correction_max_iterations",
                result->transport.volume_correction.max_iterations);
  apply_real_list(
      {"Volume_correction_minimum_relative_error",
       "VolumeCorrectionMinimumRelativeError"},
      "volume_correction_minimum_relative_error",
      result->transport.volume_correction.minimum_relative_volume_error);
  apply_real_list({"Volume_correction_maximum_interface_displacement_fraction",
                   "VolumeCorrectionMaximumInterfaceDisplacementFraction"},
                  "volume_correction_maximum_interface_displacement_fraction",
                  result->transport.volume_correction
                      .maximum_interface_displacement_fraction);
  apply_real_list(
      {"Volume_correction_maximum_cumulative_interface_displacement_fraction",
       "VolumeCorrectionMaximumCumulativeInterfaceDisplacementFraction"},
      "volume_correction_maximum_cumulative_interface_displacement_fraction",
      result->transport.volume_correction
          .maximum_cumulative_interface_displacement_fraction);
  append_equation_observations();

  apply_boolean_list({"Enable_static_capillary_equilibrium_initialization",
                      "EnableStaticCapillaryEquilibriumInitialization",
                      "Initialize_discrete_static_capillary_equilibrium",
                      "InitializeDiscreteStaticCapillaryEquilibrium"},
                     "enable_static_capillary_equilibrium_initialization",
                     result->static_capillary_equilibrium_enabled);
  apply_real_list(
      {"Static_capillary_volume_tolerance", "StaticCapillaryVolumeTolerance"},
      "static_capillary_volume_tolerance",
      result->static_capillary_equilibrium.volume_tolerance);
  apply_real_list(
      {"Static_capillary_projected_gradient_tolerance",
       "StaticCapillaryProjectedGradientTolerance"},
      "static_capillary_projected_gradient_tolerance",
      result->static_capillary_equilibrium.projected_gradient_tolerance);
  apply_real_list(
      {"Static_capillary_pressure_representability_max_residual_norm",
       "StaticCapillaryPressureRepresentabilityMaxResidualNorm"},
      "static_capillary_pressure_representability_max_residual_norm",
      result->static_capillary_equilibrium
          .pressure_representability_max_residual_norm);
  apply_real_list(
      {"Static_capillary_pressure_representability_max_relative_distance",
       "StaticCapillaryPressureRepresentabilityMaxRelativeDistance"},
      "static_capillary_pressure_representability_max_relative_distance",
      result->static_capillary_equilibrium
          .pressure_representability_max_relative_distance);
  apply_real_list({"Static_capillary_physical_equilibrium_max_residual_norm",
                   "StaticCapillaryPhysicalEquilibriumMaxResidualNorm"},
                  "static_capillary_physical_equilibrium_max_residual_norm",
                  result->static_capillary_equilibrium
                      .physical_equilibrium_max_residual_norm);
  apply_real_list({"Static_capillary_constant_pressure_kkt_max_residual_norm",
                   "StaticCapillaryConstantPressureKktMaxResidualNorm"},
                  "static_capillary_constant_pressure_kkt_max_residual_norm",
                  result->static_capillary_equilibrium
                      .constant_pressure_kkt_max_residual_norm);
  apply_real_list(
      {"Static_capillary_constant_pressure_kkt_max_relative_distance",
       "StaticCapillaryConstantPressureKktMaxRelativeDistance"},
      "static_capillary_constant_pressure_kkt_max_relative_distance",
      result->static_capillary_equilibrium
          .constant_pressure_kkt_max_relative_distance);
  apply_real_list(
      {"Static_capillary_finite_difference_reference_coefficient_scale",
       "StaticCapillaryFiniteDifferenceReferenceCoefficientScale"},
      "static_capillary_finite_difference_reference_coefficient_scale",
      result->static_capillary_equilibrium
          .finite_difference_reference_coefficient_scale);
  apply_real_list(
      {"Static_capillary_finite_difference_relative_step",
       "StaticCapillaryFiniteDifferenceRelativeStep"},
      "static_capillary_finite_difference_relative_step",
      result->static_capillary_equilibrium.finite_difference_relative_step);
  apply_real_list(
      {"Static_capillary_minimum_finite_difference_step",
       "StaticCapillaryMinimumFiniteDifferenceStep"},
      "static_capillary_minimum_finite_difference_step",
      result->static_capillary_equilibrium.minimum_finite_difference_step);
  apply_integer_list(
      {"Static_capillary_finite_difference_max_shrinks",
       "StaticCapillaryFiniteDifferenceMaxShrinks"},
      "static_capillary_finite_difference_max_shrinks",
      result->static_capillary_equilibrium.finite_difference_max_shrinks);
  apply_integer_list(
      {"Static_capillary_max_iterations", "StaticCapillaryMaxIterations"},
      "static_capillary_max_iterations",
      result->static_capillary_equilibrium.max_iterations);
  apply_integer_list(
      {"Static_capillary_max_line_search_iterations",
       "StaticCapillaryMaxLineSearchIterations"},
      "static_capillary_max_line_search_iterations",
      result->static_capillary_equilibrium.max_line_search_iterations);
  apply_integer_list(
      {"Static_capillary_max_topology_epoch_transitions",
       "StaticCapillaryMaxTopologyEpochTransitions"},
      "static_capillary_max_topology_epoch_transitions",
      result->static_capillary_equilibrium.max_topology_epoch_transitions);
  apply_real_list({"Static_capillary_projected_gradient_inverse_stiffness",
                   "StaticCapillaryProjectedGradientInverseStiffness"},
                  "static_capillary_projected_gradient_inverse_stiffness",
                  result->static_capillary_equilibrium
                      .projected_gradient_inverse_stiffness);
  apply_real_list({"Static_capillary_tangent_trust_radius",
                   "StaticCapillaryTangentTrustRadius"},
                  "static_capillary_tangent_trust_radius",
                  result->static_capillary_equilibrium.tangent_trust_radius);
  apply_real_list(
      {"Static_capillary_maximum_coefficient_update_linf",
       "StaticCapillaryMaximumCoefficientUpdateLinf"},
      "static_capillary_maximum_coefficient_update_linf",
      result->static_capillary_equilibrium.maximum_coefficient_update_linf);
  apply_real_list({"Static_capillary_line_search_shrink",
                   "StaticCapillaryLineSearchShrink"},
                  "static_capillary_line_search_shrink",
                  result->static_capillary_equilibrium.line_search_shrink);
  apply_real_list(
      {"Static_capillary_armijo_fraction", "StaticCapillaryArmijoFraction"},
      "static_capillary_armijo_fraction",
      result->static_capillary_equilibrium.armijo_fraction);
  apply_integer_list(
      {"Static_capillary_limited_memory_history_size",
       "StaticCapillaryLimitedMemoryHistorySize"},
      "static_capillary_limited_memory_history_size",
      result->static_capillary_equilibrium.limited_memory_history_size);
  apply_real_list(
      {"Static_capillary_limited_memory_curvature_tolerance",
       "StaticCapillaryLimitedMemoryCurvatureTolerance"},
      "static_capillary_limited_memory_curvature_tolerance",
      result->static_capillary_equilibrium.limited_memory_curvature_tolerance);
  apply_real_list(
      {"Static_capillary_minimum_volume_merit_penalty",
       "StaticCapillaryMinimumVolumeMeritPenalty"},
      "static_capillary_minimum_volume_merit_penalty",
      result->static_capillary_equilibrium.minimum_volume_merit_penalty);
  append_equation_observations();

  apply_boolean_list(
      {"Enable_curvature_projection", "Enable_projected_curvature",
       "Project_level_set_curvature", "Maintain_projected_curvature",
       "Curvature_projection"},
      "enable_curvature_projection", result->curvature_projection_enabled);
  if (const auto field =
          reader.string(aliases::curvature_field, "curvature_field_name")) {
    result->curvature_field_name = parsing::trimCopy(field->text);
    result->curvature_projection_enabled = true;
    append_equation_observations();
    appendDerivedObservation(result->input_observations,
                             "enable_curvature_projection",
                             "derived:curvature_field_name");
  }
  apply_integer_list(
      {"Curvature_projection_cadence_steps", "CurvatureProjectionCadenceSteps",
       "Projected_curvature_cadence_steps", "ProjectedCurvatureCadenceSteps"},
      "curvature_projection_cadence_steps",
      result->curvature_projection_cadence_steps);
  apply_real_list({"Curvature_projection_gradient_tolerance",
                   "CurvatureProjectionGradientTolerance"},
                  "curvature_projection_gradient_tolerance",
                  result->curvature_projection.gradient_tolerance);
  apply_real_list({"Curvature_projection_least_squares_rank_tolerance",
                   "CurvatureProjectionLeastSquaresRankTolerance",
                   "Curvature_projection_normal_equation_tolerance",
                   "CurvatureProjectionNormalEquationTolerance"},
                  "curvature_projection_least_squares_rank_tolerance",
                  result->curvature_projection.normal_equation_tolerance);
  apply_real_list({"Curvature_projection_max_normalized_fit_residual",
                   "CurvatureProjectionMaxNormalizedFitResidual",
                   "Projected_curvature_max_normalized_fit_residual",
                   "ProjectedCurvatureMaxNormalizedFitResidual"},
                  "curvature_projection_max_normalized_fit_residual",
                  result->curvature_projection.max_normalized_fit_residual);
  apply_integer_list({"Curvature_projection_neighbor_rings",
                      "CurvatureProjectionNeighborRings"},
                     "curvature_projection_neighbor_rings",
                     result->curvature_projection.max_neighbor_rings);
  apply_integer_list(
      {"Curvature_projection_max_neighbor_fallback_vertices",
       "CurvatureProjectionMaxNeighborFallbackVertices",
       "Projected_curvature_max_neighbor_fallback_vertices",
       "ProjectedCurvatureMaxNeighborFallbackVertices"},
      "curvature_projection_max_neighbor_fallback_vertices",
      result->curvature_projection.max_neighbor_fallback_vertices);
  apply_integer_list({"Curvature_projection_max_zero_fallback_vertices",
                      "CurvatureProjectionMaxZeroFallbackVertices",
                      "Projected_curvature_max_zero_fallback_vertices",
                      "ProjectedCurvatureMaxZeroFallbackVertices"},
                     "curvature_projection_max_zero_fallback_vertices",
                     result->curvature_projection.max_zero_fallback_vertices);
  apply_real_list({"Curvature_projection_supplemental_sample_weight",
                   "CurvatureProjectionSupplementalSampleWeight",
                   "Projected_curvature_supplemental_sample_weight",
                   "ProjectedCurvatureSupplementalSampleWeight",
                   "Curvature_projection_interface_sample_weight",
                   "CurvatureProjectionInterfaceSampleWeight"},
                  "curvature_projection_supplemental_sample_weight",
                  result->curvature_projection.supplemental_sample_weight);
  if (const auto mode = reader.string({"Curvature_projection_recovery_mode",
                                       "CurvatureProjectionRecoveryMode",
                                       "Projected_curvature_recovery_mode",
                                       "ProjectedCurvatureRecoveryMode"},
                                      "curvature_projection_recovery_mode")) {
    result->curvature_projection.recovery_mode =
        ls::parseLevelSetCurvatureRecoveryMode(mode->text);
  }
  apply_real_list(
      {"Curvature_projection_kinematic_area_gradient_filter_coefficient",
       "CurvatureProjectionKinematicAreaGradientFilterCoefficient",
       "Projected_curvature_kinematic_area_gradient_filter_coefficient",
       "ProjectedCurvatureKinematicAreaGradientFilterCoefficient"},
      "curvature_projection_kinematic_area_gradient_filter_coefficient",
      result->curvature_projection.kinematic_area_gradient_filter_coefficient);
  apply_real_list({"Curvature_projection_narrow_band_width",
                   "CurvatureProjectionNarrowBandWidth",
                   "Projected_curvature_narrow_band_width",
                   "ProjectedCurvatureNarrowBandWidth",
                   "Curvature_projection_interface_band_width",
                   "CurvatureProjectionInterfaceBandWidth"},
                  "curvature_projection_narrow_band_width",
                  result->curvature_projection.narrow_band_width);
  apply_integer_list({"Curvature_projection_smoothing_iterations",
                      "CurvatureProjectionSmoothingIterations",
                      "Projected_curvature_smoothing_iterations",
                      "ProjectedCurvatureSmoothingIterations"},
                     "curvature_projection_smoothing_iterations",
                     result->curvature_projection.smoothing_iterations);
  apply_real_list({"Curvature_projection_smoothing_relaxation",
                   "CurvatureProjectionSmoothingRelaxation",
                   "Projected_curvature_smoothing_relaxation",
                   "ProjectedCurvatureSmoothingRelaxation"},
                  "curvature_projection_smoothing_relaxation",
                  result->curvature_projection.smoothing_relaxation);
  if (const auto mode = reader.string({"Curvature_projection_smoothing_mode",
                                       "CurvatureProjectionSmoothingMode",
                                       "Projected_curvature_smoothing_mode",
                                       "ProjectedCurvatureSmoothingMode",
                                       "Curvature_projection_regularization",
                                       "CurvatureProjectionRegularization"},
                                      "curvature_projection_smoothing_mode")) {
    result->curvature_projection.smoothing_mode =
        ls::parseLevelSetCurvatureSmoothingMode(mode->text);
  }
  append_equation_observations();

  result->curvature_projection.isovalue =
      static_cast<svmp::FE::Real>(result->isovalue);
  result->volume_cut_request = matchingActiveCutVolumeRequest(
      active_requests, result->transport.level_set.field_name,
      result->isovalue);
  appendLegacyDefaults(result->input_observations);

  const bool enabled = result->transport.bound_preserving.enabled ||
                       result->transport.conservative_phase.enabled ||
                       result->transport.reinitialization.enabled ||
                       result->transport.volume_correction.enabled ||
                       result->static_capillary_equilibrium_enabled ||
                       result->curvature_projection_enabled;
  if (!enabled) {
    return std::nullopt;
  }
  return LevelSetMaintenanceConfigurationHandle{std::move(result)};
}

std::vector<LevelSetMaintenanceConfigurationHandle>
resolveLevelSetMaintenanceConfigurations(
    std::span<const ResolvedLevelSetEquationHandle> equations_by_input_index,
    std::span<const ActiveCutVolumeRequest> active_requests) {
  std::vector<LevelSetMaintenanceConfigurationHandle> configurations;
  for (const auto &equation : equations_by_input_index) {
    if (!equation) {
      continue;
    }
    if (!equation->input_snapshot ||
        !equation->input_snapshot->legacy_maintenance_input) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Cannot resolve level-set "
          "maintenance from a resolved equation without a legacy "
          "maintenance input attachment.");
    }
    const auto resolved = resolveLegacyLevelSetMaintenanceConfiguration(
        *equation->input_snapshot->legacy_maintenance_input, active_requests,
        equation);
    if (resolved) {
      configurations.push_back(*resolved);
    }
  }
  return configurations;
}

std::string serializeLevelSetMaintenanceCompatibility(
    const ResolvedLevelSetMaintenanceCompatibilityConfiguration
        &configuration) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "{\"installation\":{";
  if (!configuration.installation) {
    out << "\"availability\":\"unavailable\",\"policy\":\"installation\"}";
  } else {
    const auto &installation = *configuration.installation;
    out << "\"availability\":\"available\",\"policy\":\"installation\""
        << ",\"level_set_field_name\":"
        << jsonString(installation.options.level_set.field_name)
        << ",\"transport_form\":"
        << jsonString(transportFormName(installation.options.transport_form))
        << ",\"velocity\":{\"source\":"
        << jsonString(velocitySourceName(installation.options.velocity.source))
        << ",\"auto_register_field\":"
        << jsonBool(installation.options.velocity.auto_register_field)
        << ",\"source_field_name\":"
        << jsonString(installation.options.velocity
                          .algebraic_extension_source_field_name)
        << ",\"space_present\":"
        << jsonBool(static_cast<bool>(installation.options.velocity.space))
        << "},\"conservative_phase\":{\"enabled\":"
        << jsonBool(installation.options.conservative_phase.enabled)
        << ",\"liquid_side\":"
        << jsonString(phaseSideName(
               installation.options.conservative_phase.liquid_side))
        << "},\"reinitialization\":{\"enabled\":"
        << jsonBool(installation.options.reinitialization.enabled)
        << ",\"method\":" << jsonString(reinitializationMethodName(
                                 installation.options.reinitialization.method))
        << ",\"cadence_steps\":"
        << installation.options.reinitialization.cadence_steps
        << "},\"volume_correction\":{\"enabled\":"
        << jsonBool(installation.options.volume_correction.enabled);
    if (installation.options.volume_correction.enabled) {
      out << ",\"volume_tolerance\":"
          << jsonReal(installation.options.volume_correction.volume_tolerance);
    }
    out << "},\"curvature_targets\":[";
    for (std::size_t i = 0; i < installation.projected_curvature_fields.size();
         ++i) {
      if (i != 0u) {
        out << ',';
      }
      out << jsonString(installation.projected_curvature_fields[i]);
    }
    out << "],\"unobserved\":[]}";
  }

  const auto &transport = configuration.transport;
  out << ",\"legacy_equation_maintenance\":{\"availability\":\"available\""
      << ",\"policy\":\"legacy_maintenance\""
      << ",\"level_set_field_name\":"
      << jsonString(transport.level_set.field_name) << ",\"transport_form\":"
      << jsonString(transportFormName(transport.transport_form))
      << ",\"velocity\":{\"source\":"
      << jsonString(velocitySourceName(transport.velocity.source))
      << ",\"auto_register_field\":"
      << jsonBool(transport.velocity.auto_register_field)
      << ",\"source_field_name\":"
      << jsonString(transport.velocity.algebraic_extension_source_field_name)
      << ",\"space_present\":"
      << jsonBool(static_cast<bool>(transport.velocity.space));
  if (transport.velocity.source == ls::LevelSetVelocitySource::ConstantVector) {
    out << ",\"constant_value\":["
        << jsonReal(transport.velocity.constant_value[0]) << ','
        << jsonReal(transport.velocity.constant_value[1]) << ','
        << jsonReal(transport.velocity.constant_value[2]) << ']';
  }
  out << "},\"conservative_phase\":{\"enabled\":"
      << jsonBool(transport.conservative_phase.enabled) << ",\"liquid_side\":"
      << jsonString(phaseSideName(transport.conservative_phase.liquid_side))
      << "},\"reinitialization\":{\"enabled\":"
      << jsonBool(transport.reinitialization.enabled)
      << ",\"method\":" << jsonString(reinitializationMethodName(
                               transport.reinitialization.method))
      << ",\"cadence_steps\":" << transport.reinitialization.cadence_steps
      << "},\"volume_correction\":{\"enabled\":"
      << jsonBool(transport.volume_correction.enabled);
  if (transport.volume_correction.enabled) {
    out << ",\"volume_tolerance\":"
        << jsonReal(transport.volume_correction.volume_tolerance);
  }
  out << "},\"maintenance_only\":{\"isovalue\":"
      << jsonReal(static_cast<svmp::FE::Real>(configuration.isovalue))
      << ",\"open_boundaries\":[";
  for (std::size_t i = 0; i < configuration.open_boundaries.size(); ++i) {
    if (i != 0u) {
      out << ',';
    }
    const auto &boundary = configuration.open_boundaries[i];
    out << "{\"face_name\":" << jsonString(boundary.face_name)
        << ",\"inflow\":" << jsonBool(boundary.inflow)
        << ",\"literal_inflow_value\":";
    if (boundary.literal_inflow_value) {
      out << jsonReal(*boundary.literal_inflow_value);
    } else {
      out << "null";
    }
    out << '}';
  }
  out << "],\"active_cut_volume_request\":";
  if (!configuration.volume_cut_request) {
    out << "null";
  } else {
    const auto &request = *configuration.volume_cut_request;
    out << "{\"level_set_field_name\":"
        << jsonString(request.level_set_field_name)
        << ",\"domain_id\":" << jsonString(request.domain_id)
        << ",\"isovalue\":"
        << jsonReal(static_cast<svmp::FE::Real>(request.isovalue)) << '}';
  }
  out << ",\"static_capillary_equilibrium_enabled\":"
      << jsonBool(configuration.static_capillary_equilibrium_enabled)
      << ",\"curvature_projection_enabled\":"
      << jsonBool(configuration.curvature_projection_enabled)
      << ",\"curvature_target\":"
      << jsonString(configuration.curvature_field_name)
      << "},\"unobserved\":[\"operator_tag\",\"supg\","
         "\"interface_kinematic\"]}";

  out << ",\"observations\":{\"installation\":[";
  bool first = true;
  for (const auto &observation : configuration.input_observations) {
    if (!isInstallationObservation(observation)) {
      continue;
    }
    if (!first) {
      out << ',';
    }
    first = false;
    appendObservationJson(out, observation);
  }
  out << "],\"legacy_equation_maintenance\":[";
  first = true;
  for (const auto &observation : configuration.input_observations) {
    if (isInstallationObservation(observation)) {
      continue;
    }
    if (!first) {
      out << ',';
    }
    first = false;
    appendObservationJson(out, observation);
  }
  out << "]}}";
  return out.str();
}

} // namespace application::core
