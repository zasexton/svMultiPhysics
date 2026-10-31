#include "Application/Core/LevelSetMaintenanceConfiguration.h"

#include "Application/Translators/LevelSetConfigurationParsing.h"

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace application::core {
namespace {

namespace ls = svmp::FE::level_set;
namespace parsing = application::translators::level_set::configuration;

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

  if (!installation) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] Level-set maintenance requires the "
        "resolved level-set equation.");
  }

  auto result =
      std::make_shared<ResolvedLevelSetMaintenanceCompatibilityConfiguration>();
  result->installation = std::move(installation);

  // The maintained field reads the keys of the equation element as the
  // installation resolved them (levelSetEquationSchema()).  Its transport
  // holds the keys the maintenance uses; the others keep their defaults.
  const auto &keys = result->installation->equation_keys;
  auto &transport = result->transport;
  transport.level_set.field_name = keys.transport.level_set.field_name;
  transport.transport_form = keys.transport.transport_form;
  transport.velocity.field_name = keys.transport.velocity.field_name;
  transport.velocity.source = keys.transport.velocity.source;
  transport.velocity.material_interface_marker =
      keys.transport.velocity.material_interface_marker;
  transport.velocity.constant_value = keys.transport.velocity.constant_value;
  transport.kinematic_reconciliation = keys.transport.kinematic_reconciliation;
  transport.sign_definite_patch_bounds =
      keys.transport.sign_definite_patch_bounds;
  transport.interface_preserving_redistancing =
      keys.transport.interface_preserving_redistancing;
  transport.conservative_phase = keys.transport.conservative_phase;
  result->isovalue = keys.level_set_isovalue;
  result->static_capillary_equilibrium_enabled =
      keys.static_capillary_equilibrium_enabled;
  result->static_capillary_equilibrium = keys.static_capillary_equilibrium;
  // A source velocity field with an extension method selects the velocity
  // extension, whose field is a coupled unknown; a constant velocity stays
  // constant, and without a method a prescribed velocity is advected
  // directly.
  if (!keys.source_velocity_field_name.empty() &&
      keys.advection_velocity_extension_method.has_value() &&
      transport.velocity.source != ls::LevelSetVelocitySource::ConstantVector) {
    transport.velocity.source = ls::LevelSetVelocitySource::CoupledField;
  }

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
      // The literal inflow value: the first of Value and Level_set_value with
      // a text (an unset Value gives its default), read with std::stod.
      for (const auto *name : {"Value", "Level_set_value"}) {
        const auto found = boundary.parameters.find(name);
        if (found != boundary.parameters.end() &&
            !parsing::trimCopy(found->second.value).empty()) {
          open.literal_inflow_value =
              static_cast<svmp::FE::Real>(std::stod(found->second.value));
          break;
        }
      }
    }
    result->open_boundaries.push_back(std::move(open));
  }

  result->volume_cut_request = matchingActiveCutVolumeRequest(
      active_requests, result->transport.level_set.field_name,
      result->isovalue);

  const bool enabled = result->transport.conservative_phase.enabled ||
                       result->transport.kinematic_reconciliation.enabled ||
                       result->transport.sign_definite_patch_bounds.enabled ||
                       result->transport.interface_preserving_redistancing.enabled ||
                       result->static_capillary_equilibrium_enabled;
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

} // namespace application::core
