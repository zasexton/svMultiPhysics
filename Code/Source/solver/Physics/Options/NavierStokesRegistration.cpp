/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#include "Physics/Options/NavierStokesRegistration.h"

#include "Physics/Options/InputReaders.h"
#include "Physics/Options/NavierStokesOptionRules.h"

#include "FE/Spaces/FunctionSpace.h"
#include "FE/Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp::Physics::options {

namespace {

std::string normalized_scope_token(std::string_view raw)
{
  std::string token;
  token.reserve(raw.size());
  for (char ch : raw) {
    if (ch >= 'A' && ch <= 'Z') {
      token.push_back(static_cast<char>(ch - 'A' + 'a'));
    } else if ((ch >= 'a' && ch <= 'z') ||
               (ch >= '0' && ch <= '9')) {
      token.push_back(ch);
    }
  }
  return token;
}

} // namespace

namespace {

[[noreturn]] void reject_unsupported_free_surface_scope()
{
  throw std::runtime_error(
      "[svMultiPhysics::Physics] "
      "unsupported_two_phase_or_jump_free_surface_scope");
}

} // namespace

namespace {

[[nodiscard]] bool contains_unsupported_free_surface_scope_marker(
    std::string_view value)
{
  static constexpr std::array<std::string_view, 8> markers = {
      "twophase",
      "twofluid",
      "multiphase",
      "pressureenrichment",
      "jump",
      "gas",
      "gasdensity",
      "gasviscosity",
  };
  const auto token = normalized_scope_token(value);
  return std::any_of(
      markers.begin(), markers.end(), [&](std::string_view marker) {
        return token.find(marker) != std::string::npos;
      });
}

} // namespace

namespace {

[[nodiscard]] bool is_free_surface_scope_selector(std::string_view token)
{
  static constexpr std::array<std::string_view, 18> selectors = {
      "capability",
      "capabilityscope",
      "fluidmodel",
      "formulation",
      "freesurfacemodel",
      "freesurfacephysicalmodel",
      "implementation",
      "interfacephysics",
      "interfacemodel",
      "materialmodel",
      "model",
      "modelscope",
      "phasemodel",
      "physicalmodel",
      "physics",
      "physicsscope",
      "scope",
      "type",
  };
  return std::find(selectors.begin(), selectors.end(), token) !=
         selectors.end();
}

} // namespace

namespace {

void validate_free_surface_scope_entry(std::string_view key,
                                       std::string_view value,
                                       bool scan_value)
{
  const auto normalized_key = normalized_scope_token(key);
  if (contains_unsupported_free_surface_scope_marker(normalized_key) ||
      (scan_value && is_free_surface_scope_selector(normalized_key) &&
       contains_unsupported_free_surface_scope_marker(value))) {
    reject_unsupported_free_surface_scope();
  }
}

} // namespace

namespace {

[[nodiscard]] bool is_canonical_free_surface_physical_model_key(
    std::string_view key)
{
  return key == "Free_surface_physical_model" ||
         key == "FreeSurfacePhysicalModel";
}

} // namespace

namespace {

[[nodiscard]] bool is_two_fluid_configuration_key(
    std::string_view key)
{
  static constexpr std::array<std::string_view, 11> keys = {
      "Material_interface_marker",
      "Negative_phase_density",
      "Negative_phase_dynamic_viscosity",
      "Positive_phase_density",
      "Positive_phase_dynamic_viscosity",
      "Two_fluid_surface_tension",
      "Two_fluid_interface_nitsche_gamma",
      "Prescribed_pressure_jump",
      "Prescribed_viscous_traction_jump_x",
      "Prescribed_viscous_traction_jump_y",
      "Prescribed_viscous_traction_jump_z",
  };
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

} // namespace

namespace {

enum class FreeSurfaceScopeLocation {
  Equation,
  Domain,
  Boundary,
};

} // namespace

namespace {

[[nodiscard]] bool is_free_surface_implementation_selector(
    std::string_view normalized_key)
{
  return normalized_key == "implementation" ||
         normalized_key == "freesurfaceimplementation";
}

} // namespace

namespace {

[[nodiscard]] bool is_canonical_free_surface_implementation_key(
    std::string_view key)
{
  return key == "Implementation";
}

} // namespace

namespace {

void validate_free_surface_scope_map(
    const svmp::Physics::ParameterMap& params,
    FreeSurfaceScopeLocation location,
    NavierStokesPhysicalModel physical_model)
{
  for (const auto& [key, parameter] : params) {
    const auto normalized_key = normalized_scope_token(key);
    if (normalized_key == "freesurfacephysicalmodel") {
      if (!parameter.defined) {
        continue;
      }
      if (location != FreeSurfaceScopeLocation::Equation) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] "
            "misplaced_free_surface_physical_model");
      }
      if (!is_canonical_free_surface_physical_model_key(key)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] "
            "unsupported_free_surface_physical_model");
      }
      continue;
    }
    if (is_two_fluid_configuration_key(key)) {
      if (location != FreeSurfaceScopeLocation::Equation ||
          physical_model !=
              NavierStokesPhysicalModel::IncompressibleTwoFluid) {
        reject_unsupported_free_surface_scope();
      }
      continue;
    }
    validate_free_surface_scope_entry(
        key, parameter.value, parameter.defined);
    if (parameter.defined &&
        is_free_surface_implementation_selector(normalized_key)) {
      validate_free_surface_scope_entry(
          "Implementation", parameter.value, true);
    }
    if (!parameter.defined) {
      continue;
    }
    const bool boundary_control =
        location == FreeSurfaceScopeLocation::Boundary &&
        (key == "Type" ||
         (is_free_surface_implementation_selector(normalized_key) &&
          is_canonical_free_surface_implementation_key(key)));
    if ((is_free_surface_scope_selector(normalized_key) ||
         is_free_surface_implementation_selector(normalized_key)) &&
        !boundary_control) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
  }
}

} // namespace

namespace {

void validate_free_surface_boundary_scope(
    const svmp::Physics::ParameterMap& params,
    NavierStokesPhysicalModel physical_model)
{
  validate_free_surface_scope_map(
      params, FreeSurfaceScopeLocation::Boundary, physical_model);

  const auto type_it = params.find("Type");
  if (type_it == params.end() || !type_it->second.defined) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_free_surface_physical_model");
  }
  const auto type = normalized_scope_token(type_it->second.value);
  static constexpr std::array<std::string_view, 9> supported_types = {
      "dir",
      "dirichlet",
      "neu",
      "neumann",
      "trac",
      "traction",
      "rbn",
      "robin",
      "freesurface",
  };
  if (std::find(supported_types.begin(), supported_types.end(), type) ==
      supported_types.end()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_free_surface_physical_model");
  }

  std::size_t implementation_count = 0;
  for (const auto key : {
           "Implementation",
       }) {
    const auto it = params.find(key);
    if (it == params.end() || !it->second.defined) {
      continue;
    }
    ++implementation_count;
    const auto implementation =
        normalized_scope_token(it->second.value);
    const bool supported =
        implementation == "fitted" ||
        implementation == "fittedale" ||
        implementation == "ale" ||
        implementation == "unfitted" ||
        implementation == "unfittedlevelset" ||
        implementation == "levelset" ||
        implementation == "embeddedlevelset";
    if (type != "freesurface" || !supported) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
  }
  if (implementation_count > 1) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_free_surface_physical_model");
  }
}

} // namespace

namespace {

void validate_free_surface_module_options(std::string_view module_options)
{
  std::string normalized(module_options);
  for (char& ch : normalized) {
    if (ch == ';' || ch == '\n' || ch == '\t') {
      ch = ',';
    }
  }

  std::size_t start = 0;
  while (start < normalized.size()) {
    const auto end = normalized.find(',', start);
    const auto token = trim_copy(normalized.substr(start, end - start));
    if (!token.empty()) {
      const auto separator = token.find_first_of("=:");
      const auto key = trim_copy(token.substr(0, separator));
      const auto value =
          separator == std::string::npos
              ? std::string{}
              : trim_copy(token.substr(separator + 1));
      validate_free_surface_scope_entry(key, value, true);
      if (normalized_scope_token(key) == "freesurfacephysicalmodel") {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] "
            "misplaced_free_surface_physical_model");
      }
      if (is_free_surface_scope_selector(
              normalized_scope_token(key)) ||
          is_free_surface_implementation_selector(
              normalized_scope_token(key))) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] "
            "unsupported_free_surface_physical_model");
      }
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
}

} // namespace

FreeSurfaceSchemaContract resolve_free_surface_schema_contract(
    const svmp::Physics::ParameterMap& params)
{
  FreeSurfaceSchemaContract contract;
  bool schema_seen = false;
  for (const auto key : {
           "Free_surface_configuration_schema_version",
           "FreeSurfaceConfigurationSchemaVersion",
           "Free_surface_schema_version",
       }) {
    const auto it = params.find(key);
    if (it == params.end() || !it->second.defined) {
      continue;
    }
    if (schema_seen) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
    schema_seen = true;
    const auto value = trim_copy(it->second.value);
    std::size_t parsed = 0;
    try {
      contract.version = std::stoi(value, &parsed);
    } catch (...) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
    if (parsed != value.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
  }

  bool legacy_seen = false;
  for (const auto key : {
           "Enable_explicit_legacy_free_surface_configuration",
           "EnableExplicitLegacyFreeSurfaceConfiguration",
           "Free_surface_legacy_behavior",
       }) {
    const auto it = params.find(key);
    if (it == params.end() || !it->second.defined) {
      continue;
    }
    if (legacy_seen) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
    legacy_seen = true;
    const auto value = lower_copy(trim_copy(it->second.value));
    if (value == "true" || value == "1" ||
        value == "yes" || value == "on") {
      contract.explicit_legacy = true;
    } else if (
        value == "false" || value == "0" ||
        value == "no" || value == "off") {
      contract.explicit_legacy = false;
    } else {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
  }

  const auto current =
      svmp::Physics::formulations::navier_stokes::
          IncompressibleNavierStokesVMSOptions::
              current_configuration_schema_version;
  const bool current_contract =
      contract.version == current && !contract.explicit_legacy;
  const bool explicit_legacy_contract =
      contract.version == 1 && contract.explicit_legacy;
  if (!current_contract && !explicit_legacy_contract) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_free_surface_physical_model");
  }
  return contract;
}

NavierStokesPhysicalModel
validate_and_resolve_free_surface_physical_model(
    const svmp::Physics::EquationModuleInput& input)
{
  const svmp::Physics::ParameterValue* selected = nullptr;
  for (const auto key : {
           "Free_surface_physical_model",
           "FreeSurfacePhysicalModel",
       }) {
    const auto it = input.equation_params.find(key);
    if (it == input.equation_params.end() || !it->second.defined) {
      continue;
    }
    if (selected != nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "ambiguous_free_surface_physical_model");
    }
    selected = &it->second;
  }

  const auto schema_contract =
      resolve_free_surface_schema_contract(input.equation_params);
  auto physical_model =
      NavierStokesPhysicalModel::
          OnePhaseLiquidPrescribedExteriorPressure;
  if (selected == nullptr) {
    physical_model = NavierStokesPhysicalModel::
        OnePhaseLiquidPrescribedExteriorPressure;
  } else {
    if (schema_contract.explicit_legacy) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
    const auto exact_value = trim_copy(selected->value);
    if (exact_value == "IncompressibleTwoFluid") {
      physical_model =
          NavierStokesPhysicalModel::IncompressibleTwoFluid;
    } else if (normalized_scope_token(exact_value) ==
               "onephaseliquidprescribedexteriorpressure") {
      physical_model = NavierStokesPhysicalModel::
          OnePhaseLiquidPrescribedExteriorPressure;
    } else if (contains_unsupported_free_surface_scope_marker(
                   exact_value)) {
      reject_unsupported_free_surface_scope();
    } else {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
  }

  validate_free_surface_scope_map(
      input.equation_params,
      FreeSurfaceScopeLocation::Equation,
      physical_model);
  validate_free_surface_scope_map(
      input.default_domain.params,
      FreeSurfaceScopeLocation::Domain,
      physical_model);
  for (const auto& domain : input.domains) {
    validate_free_surface_scope_map(
        domain.params,
        FreeSurfaceScopeLocation::Domain,
        physical_model);
  }
  for (const auto& boundary : input.boundary_conditions) {
    validate_free_surface_boundary_scope(
        boundary.params, physical_model);
  }
  validate_free_surface_module_options(input.module_options);
  return physical_model;
}

namespace {

const svmp::Physics::ParameterValue& require_two_fluid_parameter(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view canonical_name)
{
  const svmp::Physics::ParameterValue* selected = nullptr;
  for (const auto key : keys) {
    const auto it = params.find(std::string(key));
    if (it == params.end() || !it->second.defined) {
      continue;
    }
    if (selected != nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "ambiguous_two_fluid_parameter:" +
          std::string(canonical_name));
    }
    selected = &it->second;
  }
  if (selected == nullptr || trim_copy(selected->value).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "missing_two_fluid_parameter:" +
        std::string(canonical_name));
  }
  return *selected;
}

} // namespace

std::string require_two_fluid_string(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view canonical_name)
{
  return trim_copy(
      require_two_fluid_parameter(params, keys, canonical_name).value);
}

svmp::FE::Real require_two_fluid_real(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view canonical_name)
{
  const auto& parameter =
      require_two_fluid_parameter(params, keys, canonical_name);
  return static_cast<svmp::FE::Real>(
      parse_double(parameter.value, canonical_name));
}

int require_two_fluid_int(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view canonical_name)
{
  const auto& parameter =
      require_two_fluid_parameter(params, keys, canonical_name);
  const auto value = trim_copy(parameter.value);
  try {
    std::size_t parsed = 0u;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size()) {
      throw std::runtime_error("");
    }
    return result;
  } catch (...) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Failed to parse integer value '" +
        parameter.value + "' for " + std::string(canonical_name) + ".");
  }
}

void reject_unsupported_two_fluid_default_domain_parameters(
    const svmp::Physics::ParameterMap& params)
{
  static constexpr std::array<std::string_view, 3> supported{
      "Force_x", "Force_y", "Force_z"};
  for (const auto& [key, parameter] : params) {
    if (!parameter.defined ||
        std::find(supported.begin(), supported.end(), key) !=
            supported.end()) {
      continue;
    }
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_two_fluid_default_domain_parameter:" +
        key);
  }
}

void reject_unsupported_two_fluid_equation_parameters(
    const svmp::Physics::ParameterMap& params)
{
  static constexpr std::array<std::string_view, 24> supported{
      "Coupled",
      "Max_iterations",
      "Min_iterations",
      "Tolerance",
      "Use_taylor_hood_type_basis",
      "Free_surface_physical_model",
      "FreeSurfacePhysicalModel",
      "Free_surface_configuration_schema_version",
      "FreeSurfaceConfigurationSchemaVersion",
      "Free_surface_schema_version",
      "Enable_explicit_legacy_free_surface_configuration",
      "EnableExplicitLegacyFreeSurfaceConfiguration",
      "Free_surface_legacy_behavior",
      "Level_set_field_name",
      "Generated_interface_domain_id",
      "Operator_tag",
      "OperatorTag",
      "Element_order",
      "Jit_enable",
      "Enable_jit",
      "Use_jit",
      "Jit_specialization_enable",
      "Enable_jit_specialization",
      "Use_jit_specialization",
  };
  for (const auto& [key, parameter] : params) {
    if (!parameter.defined || is_two_fluid_configuration_key(key) ||
        std::find(supported.begin(), supported.end(), key) !=
            supported.end()) {
      continue;
    }
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_two_fluid_equation_parameter:" +
        key);
  }
}

void reject_unsupported_two_fluid_boundary_parameters(
    std::span<const svmp::Physics::BoundaryConditionInput> boundaries)
{
  static constexpr std::array<std::string_view, 6> supported{
      "Type",
      "Time_dependence",
      "Value",
      "Effective_direction",
      "Weakly_applied",
      "Temporal_and_spatial_values_file_path",
  };
  for (const auto& boundary : boundaries) {
    if (!boundary.nested_configuration_blocks.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_two_fluid_boundary_block:" +
          boundary.nested_configuration_blocks.front());
    }
    for (const auto& [key, parameter] : boundary.params) {
      if (!parameter.defined ||
          std::find(supported.begin(), supported.end(), key) !=
              supported.end()) {
        continue;
      }
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_two_fluid_boundary_parameter:" +
          key);
    }
  }
}

[[nodiscard]] bool dependencySpacesCompatible(
    const svmp::FE::spaces::FunctionSpace& lhs,
    const svmp::FE::spaces::FunctionSpace& rhs) noexcept
{
  return lhs.space_type() == rhs.space_type() &&
         lhs.field_type() == rhs.field_type() &&
         lhs.value_dimension() == rhs.value_dimension() &&
         lhs.topological_dimension() == rhs.topological_dimension() &&
         lhs.polynomial_order() == rhs.polynomial_order() &&
         lhs.element_type() == rhs.element_type();
}

void preflightTwoFluidUnknownField(
    const svmp::FE::systems::FESystem& system,
    std::string_view name,
    const std::shared_ptr<const svmp::FE::spaces::FunctionSpace>& space,
    int components)
{
  if (name.empty() || !space || components <= 0) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] invalid two-fluid dependency field specification");
  }
  const auto existing = system.findFieldByName(name);
  if (existing == svmp::FE::INVALID_FIELD_ID) {
    return;
  }
  const auto& record = system.fieldRecord(existing);
  if (record.source_kind !=
          svmp::FE::systems::FieldSourceKind::Unknown ||
      record.components != components || !record.space ||
      !dependencySpacesCompatible(*record.space, *space)) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] two-fluid dependency field '" +
        std::string(name) + "' is incompatible with the requested unknown");
  }
}

[[nodiscard]] svmp::FE::FieldId ensureTwoFluidUnknownField(
    svmp::FE::systems::FESystem& system,
    std::string name,
    const std::shared_ptr<const svmp::FE::spaces::FunctionSpace>& space,
    int components)
{
  const auto existing = system.findFieldByName(name);
  if (existing != svmp::FE::INVALID_FIELD_ID) {
    return existing;
  }
  return system.addField(svmp::FE::systems::FieldSpec{
      .name = std::move(name),
      .space = space,
      .components = components,
      .source_kind = svmp::FE::systems::FieldSourceKind::Unknown,
  });
}

void preflightTwoFluidTranslationBeforeMutation(
    const TranslatedIncompressibleTwoFluidInput& translated,
    const svmp::FE::systems::FESystem& system)
{
  const auto& options = translated.options;
  const int dimension = translated.velocity_space
                            ? translated.velocity_space->value_dimension()
                            : 0;
  if ((dimension != 2 && dimension != 3) ||
      !translated.pressure_space || options.interface_marker < 0 ||
      options.operator_tag.empty() ||
      options.generated_interface_domain_id.empty() ||
      !std::isfinite(options.level_set_isovalue) ||
      !std::isfinite(options.surface_tension) ||
      options.surface_tension < svmp::FE::Real{0.0} ||
      !std::isfinite(options.interface_nitsche_gamma) ||
      !(options.interface_nitsche_gamma > svmp::FE::Real{0.0}) ||
      !std::isfinite(options.negative_phase.density) ||
      !(options.negative_phase.density > svmp::FE::Real{0.0}) ||
      !std::isfinite(options.positive_phase.density) ||
      !(options.positive_phase.density > svmp::FE::Real{0.0}) ||
      !std::isfinite(options.negative_phase.viscosity) ||
      !(options.negative_phase.viscosity > svmp::FE::Real{0.0}) ||
      !std::isfinite(options.positive_phase.viscosity) ||
      !(options.positive_phase.viscosity > svmp::FE::Real{0.0})) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] invalid incompressible two-fluid dependency definition");
  }

  svmp::Physics::formulations::navier_stokes::
      validateIncompressibleTwoFluidConfigurationSemantics(options);

  const std::array<std::string_view, 5> names{
      options.level_set_field_name,
      options.negative_phase.velocity_field_name,
      options.positive_phase.velocity_field_name,
      options.negative_phase.pressure_field_name,
      options.positive_phase.pressure_field_name,
  };
  for (std::size_t i = 0u; i < names.size(); ++i) {
    if (names[i].empty()) {
      throw std::invalid_argument(
          "[svMultiPhysics::Physics] two-fluid dependency field names must be nonempty");
    }
    for (std::size_t j = i + 1u; j < names.size(); ++j) {
      if (names[i] == names[j]) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] two-fluid dependency field names must be distinct");
      }
    }
  }

  const auto level_set = system.findFieldByName(options.level_set_field_name);
  if (level_set == svmp::FE::INVALID_FIELD_ID) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] two-fluid dependency requires its level-set field to be pre-registered");
  }
  const auto& level_set_record = system.fieldRecord(level_set);
  const bool supported_level_set_source =
      level_set_record.source_kind ==
          svmp::FE::systems::FieldSourceKind::Unknown ||
      level_set_record.source_kind ==
          svmp::FE::systems::FieldSourceKind::PrescribedData;
  if (!supported_level_set_source || level_set_record.components != 1 ||
      !level_set_record.space ||
      !dependencySpacesCompatible(
          *level_set_record.space, *translated.pressure_space)) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] two-fluid dependency requires a compatible scalar level-set field");
  }

  preflightTwoFluidUnknownField(
      system,
      options.negative_phase.velocity_field_name,
      translated.velocity_space,
      dimension);
  preflightTwoFluidUnknownField(
      system,
      options.positive_phase.velocity_field_name,
      translated.velocity_space,
      dimension);
  preflightTwoFluidUnknownField(
      system,
      options.negative_phase.pressure_field_name,
      translated.pressure_space,
      1);
  preflightTwoFluidUnknownField(
      system,
      options.positive_phase.pressure_field_name,
      translated.pressure_space,
      1);
}

} // namespace svmp::Physics::options
