#include "Application/Core/LevelSetMaintenanceConfiguration.h"

#include "Application/Translators/LevelSetConfigurationParsing.h"

#include <algorithm>
#include <array>
#include <cmath>
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

constexpr auto kLegacyPolicy = parsing::LevelSetInputPolicy::LegacyMaintenance;

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
  result->isovalue = keys.level_set_isovalue;

  Reader reader(input.equation_parameters, kLegacyPolicy);
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

  // A source velocity field selects the velocity extension, whose field is
  // a coupled unknown; a constant velocity stays constant.
  if (reader.selected(aliases::wet_extension_source).has_value() &&
      transport.velocity.source != ls::LevelSetVelocitySource::ConstantVector) {
    transport.velocity.source = ls::LevelSetVelocitySource::CoupledField;
  }

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
      std::array<std::string_view, 1>{"Conservative_phase_invariant_tolerance"},
      "conservative_phase_invariant_tolerance",
      result->transport.conservative_phase.invariant_tolerance);
  apply_real(
      std::array<std::string_view, 1>{
          "Conservative_phase_component_activity_tolerance"},
      "conservative_phase_component_activity_tolerance",
      result->transport.conservative_phase.component_activity_tolerance);
  apply_real(
      std::array<std::string_view, 1>{"Conservative_phase_maximum_courant"},
      "conservative_phase_maximum_courant",
      result->transport.conservative_phase.maximum_courant);
  apply_boolean(
      std::array<std::string_view, 1>{
          "Conservative_phase_enforce_courant_limit"},
      "conservative_phase_enforce_courant_limit",
      result->transport.conservative_phase.enforce_courant_limit);
  apply_boolean(
      std::array<std::string_view, 1>{
          "Conservative_phase_require_constant_preservation"},
      "conservative_phase_require_constant_preservation",
      result->transport.conservative_phase.require_constant_preservation);
  apply_real(
      std::array<std::string_view, 1>{
          "Conservative_phase_momentum_relative_tolerance"},
      "conservative_phase_momentum_relative_tolerance",
      result->transport.conservative_phase.momentum_relative_tolerance);
  apply_boolean(
      std::array<std::string_view, 1>{"Conservative_phase_write_flux_artifacts"},
      "conservative_phase_write_flux_artifacts",
      result->transport.conservative_phase.write_flux_artifacts);
  apply_integer(
      std::array<std::string_view, 1>{
          "Conservative_phase_flux_artifact_cadence_steps"},
      "conservative_phase_flux_artifact_cadence_steps",
      result->transport.conservative_phase.flux_artifact_cadence_steps);
  apply_boolean(
      std::array<std::string_view, 1>{
          "Conservative_phase_classify_nonprimary_components_as_satellites"},
      "conservative_phase_classify_nonprimary_components_as_satellites",
      result->transport.conservative_phase
          .classify_nonprimary_components_as_satellites);
  if (const auto policy =
          reader.string({"Conservative_phase_boundary_flux_policy"},
                        "conservative_phase_boundary_flux_policy")) {
    result->transport.conservative_phase.boundary_flux_policy =
        parsing::parseConservativePhaseBoundaryFluxPolicy(policy->text);
  }
  if (const auto regions =
          reader.string({"Conservative_phase_fixed_flux_regions"},
                        "conservative_phase_fixed_flux_regions")) {
    result->transport.conservative_phase.fixed_flux_regions =
        ls::parseLevelSetPhaseRegionBoxes(regions->text);
  }
  if (const auto tolerance = reader.real(
          {"Conservative_phase_impermeable_normal_velocity_tolerance"},
          "conservative_phase_impermeable_normal_velocity_tolerance")) {
    result->transport.conservative_phase.impermeable_normal_velocity_tolerance =
        *tolerance;
    result->transport.conservative_phase
        .pointwise_impermeable_velocity_tolerance_explicitly_requested = true;
  }
  apply_boolean(
      std::array<std::string_view, 1>{"Conservative_phase_reconcile_geometry"},
      "conservative_phase_reconcile_geometry",
      result->transport.conservative_phase.reconcile_geometry);
  apply_real(
      std::array<std::string_view, 1>{
          "Conservative_phase_geometry_measure_tolerance"},
      "conservative_phase_geometry_measure_tolerance",
      result->transport.conservative_phase.geometry_measure_tolerance);
  apply_integer(
      std::array<std::string_view, 1>{
          "Conservative_phase_geometry_correction_max_iterations"},
      "conservative_phase_geometry_correction_max_iterations",
      result->transport.conservative_phase.geometry_correction_max_iterations);
  apply_real(
      std::array<std::string_view, 1>{
          "Conservative_phase_maximum_geometry_displacement_fraction"},
      "conservative_phase_maximum_geometry_displacement_fraction",
      result->transport.conservative_phase
          .maximum_geometry_displacement_fraction);

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
    }
    result->open_boundaries.push_back(std::move(open));
  }

  apply_boolean_list({"Enable_static_capillary_equilibrium_initialization"},
                     "enable_static_capillary_equilibrium_initialization",
                     result->static_capillary_equilibrium_enabled);
  apply_real_list(
      {"Static_capillary_volume_tolerance"},
      "static_capillary_volume_tolerance",
      result->static_capillary_equilibrium.volume_tolerance);
  apply_real_list(
      {"Static_capillary_projected_gradient_tolerance"},
      "static_capillary_projected_gradient_tolerance",
      result->static_capillary_equilibrium.projected_gradient_tolerance);
  apply_real_list(
      {"Static_capillary_pressure_representability_max_residual_norm"},
      "static_capillary_pressure_representability_max_residual_norm",
      result->static_capillary_equilibrium
          .pressure_representability_max_residual_norm);
  apply_real_list(
      {"Static_capillary_pressure_representability_max_relative_distance"},
      "static_capillary_pressure_representability_max_relative_distance",
      result->static_capillary_equilibrium
          .pressure_representability_max_relative_distance);
  apply_real_list({"Static_capillary_physical_equilibrium_max_residual_norm"},
                  "static_capillary_physical_equilibrium_max_residual_norm",
                  result->static_capillary_equilibrium
                      .physical_equilibrium_max_residual_norm);
  apply_real_list({"Static_capillary_constant_pressure_kkt_max_residual_norm"},
                  "static_capillary_constant_pressure_kkt_max_residual_norm",
                  result->static_capillary_equilibrium
                      .constant_pressure_kkt_max_residual_norm);
  apply_real_list(
      {"Static_capillary_constant_pressure_kkt_max_relative_distance"},
      "static_capillary_constant_pressure_kkt_max_relative_distance",
      result->static_capillary_equilibrium
          .constant_pressure_kkt_max_relative_distance);
  apply_real_list(
      {"Static_capillary_finite_difference_reference_coefficient_scale"},
      "static_capillary_finite_difference_reference_coefficient_scale",
      result->static_capillary_equilibrium
          .finite_difference_reference_coefficient_scale);
  apply_real_list(
      {"Static_capillary_finite_difference_relative_step"},
      "static_capillary_finite_difference_relative_step",
      result->static_capillary_equilibrium.finite_difference_relative_step);
  apply_real_list(
      {"Static_capillary_minimum_finite_difference_step"},
      "static_capillary_minimum_finite_difference_step",
      result->static_capillary_equilibrium.minimum_finite_difference_step);
  apply_integer_list(
      {"Static_capillary_finite_difference_max_shrinks"},
      "static_capillary_finite_difference_max_shrinks",
      result->static_capillary_equilibrium.finite_difference_max_shrinks);
  apply_integer_list(
      {"Static_capillary_max_iterations"},
      "static_capillary_max_iterations",
      result->static_capillary_equilibrium.max_iterations);
  apply_integer_list(
      {"Static_capillary_max_line_search_iterations"},
      "static_capillary_max_line_search_iterations",
      result->static_capillary_equilibrium.max_line_search_iterations);
  apply_integer_list(
      {"Static_capillary_max_topology_epoch_transitions"},
      "static_capillary_max_topology_epoch_transitions",
      result->static_capillary_equilibrium.max_topology_epoch_transitions);
  apply_real_list({"Static_capillary_projected_gradient_inverse_stiffness"},
                  "static_capillary_projected_gradient_inverse_stiffness",
                  result->static_capillary_equilibrium
                      .projected_gradient_inverse_stiffness);
  apply_real_list({"Static_capillary_tangent_trust_radius"},
                  "static_capillary_tangent_trust_radius",
                  result->static_capillary_equilibrium.tangent_trust_radius);
  apply_real_list(
      {"Static_capillary_maximum_coefficient_update_linf"},
      "static_capillary_maximum_coefficient_update_linf",
      result->static_capillary_equilibrium.maximum_coefficient_update_linf);
  apply_real_list({"Static_capillary_line_search_shrink"},
                  "static_capillary_line_search_shrink",
                  result->static_capillary_equilibrium.line_search_shrink);
  apply_real_list(
      {"Static_capillary_armijo_fraction"},
      "static_capillary_armijo_fraction",
      result->static_capillary_equilibrium.armijo_fraction);
  apply_integer_list(
      {"Static_capillary_limited_memory_history_size"},
      "static_capillary_limited_memory_history_size",
      result->static_capillary_equilibrium.limited_memory_history_size);
  apply_real_list(
      {"Static_capillary_limited_memory_curvature_tolerance"},
      "static_capillary_limited_memory_curvature_tolerance",
      result->static_capillary_equilibrium.limited_memory_curvature_tolerance);
  apply_real_list(
      {"Static_capillary_minimum_volume_merit_penalty"},
      "static_capillary_minimum_volume_merit_penalty",
      result->static_capillary_equilibrium.minimum_volume_merit_penalty);

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
