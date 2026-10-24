/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#include "Physics/Options/NavierStokesRegistration.h"

#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/TemporalValues.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"
#include "Physics/Options/BoundaryDataReaders.h"
#include "Physics/Options/NavierStokesOptionSchema.h"

#include "Physics/Options/InputReaders.h"
#include "Physics/Options/NavierStokesOptionRules.h"

#include "FE/Core/Logger.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/Interfaces/MaterialInterfaceTransportVelocity.h"
#include "FE/Spaces/FunctionSpace.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellTopology.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
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

  // Implementation belongs to a Free_surface BC and must resolve through
  // the free-surface schema.
  const auto implementation_it = params.find("Implementation");
  if (implementation_it != params.end() && implementation_it->second.defined) {
    svmp::FE::options::OptionInput implementation;
    implementation.add("Implementation", implementation_it->second.value);
    if (type != "freesurface" ||
        !free_surface_application_keys(implementation)
             .implementation.has_value()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
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
      // The explicit schema-1 legacy input was retired (D47 P2); the
      // module keeps the legacy options for programmatic tests only.
      throw std::runtime_error(
          "[svMultiPhysics::Physics] The explicit schema-1 legacy "
          "free-surface configuration (" + std::string(key) +
          "=true) was retired (decision D47); remove it and use the "
          "current configuration schema.");
    }
    if (value != "false" && value != "0" &&
        value != "no" && value != "off") {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_free_surface_physical_model");
    }
  }

  const auto current =
      svmp::Physics::formulations::navier_stokes::
          IncompressibleNavierStokesVMSOptions::
              current_configuration_schema_version;
  if (contract.version != current) {
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

  (void)resolve_free_surface_schema_contract(input.equation_params);
  auto physical_model =
      NavierStokesPhysicalModel::
          OnePhaseLiquidPrescribedExteriorPressure;
  if (selected == nullptr) {
    physical_model = NavierStokesPhysicalModel::
        OnePhaseLiquidPrescribedExteriorPressure;
  } else {
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

void reject_unsupported_two_fluid_default_domain_parameters(
    const svmp::Physics::ParameterMap& params)
{
  // The keys of incompressibleTwoFluidDomainSchema() (the body force).
  static const auto supported = svmp::Physics::formulations::navier_stokes::
      incompressibleTwoFluidDomainSchema().spellings();
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
  // The keys of incompressibleTwoFluidSchema(), the generic keys of every
  // module (nonlinear solver, element order, JIT) and the retired legacy
  // switches, which the configuration contract reports.
  static const auto supported = [] {
    auto keys = svmp::Physics::formulations::navier_stokes::
        incompressibleTwoFluidSchema().spellings();
    for (const char* key : {
             "Coupled",
             "Max_iterations",
             "Min_iterations",
             "Tolerance",
             "Element_order",
             "Jit_enable",
             "Enable_jit",
             "Use_jit",
             "Jit_specialization_enable",
             "Enable_jit_specialization",
             "Use_jit_specialization",
             "Enable_explicit_legacy_free_surface_configuration",
             "EnableExplicitLegacyFreeSurfaceConfiguration",
             "Free_surface_legacy_behavior",
         }) {
      keys.emplace_back(key);
    }
    return keys;
  }();
  for (const auto& [key, parameter] : params) {
    if (!parameter.defined ||
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

// The Navier-Stokes input translation: the equation, domain and boundary
// keys, the boundary data files and the two-fluid translation (moved from
// NavierStokesRegister.cpp, decision D50).  The equation, domain and
// two-fluid keys resolve through the option schemas (D47 P2); the standard
// boundary conditions keep their readers.

namespace {

namespace ns = svmp::Physics::formulations::navier_stokes;
namespace opt = svmp::FE::options;

/// The input restricted to the given spellings.
opt::OptionInput select_spellings(const opt::OptionInput& input,
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

/// Every spelling of the given keys of a schema.
template <class T>
std::vector<std::string> spellings_of(const opt::OptionSchema<T>& schema,
                                      std::initializer_list<std::string_view> keys)
{
  std::vector<std::string> out;
  for (const auto& documentation : schema.documentation()) {
    if (std::find(keys.begin(), keys.end(), documentation.key) != keys.end()) {
      out.insert(out.end(), documentation.spellings.begin(), documentation.spellings.end());
      out.insert(out.end(), documentation.deprecated_spellings.begin(),
                 documentation.deprecated_spellings.end());
    }
  }
  return out;
}

/// The keys of the moving domain and the momentum source (the keys the
/// two-fluid mode reads from both elements, to reject them).
const std::vector<std::string>& moving_domain_and_momentum_source_spellings()
{
  static const auto spellings = spellings_of(
      ns::navierStokesDomainSchema(),
      {"Enable_ALE", "Mesh_velocity_field", "Mesh_velocity_source",
       "Mesh_displacement_field", "Auto_register_mesh_displacement_field",
       "Moving_mesh_tangent_path", "Momentum_source_field_name",
       "Auto_register_momentum_source_field"});
  return spellings;
}

/// The domain keys as the register reads them: the defined entries, and the
/// parameters of the stated viscosity model and the backflow coefficient
/// whether the deck defines them or not (the domain snapshot holds their
/// Parameters defaults, which the register uses).
opt::OptionInput domain_keys_as_read(const svmp::Physics::ParameterMap& params)
{
  const auto* model = find_param(params, "Viscosity.model");
  const auto model_token = model != nullptr && model->defined
                               ? lower_copy(trim_copy(model->value))
                               : std::string{};
  std::vector<std::string_view> model_keys;
  if (model_token == "constant") {
    model_keys = {"Viscosity.Value"};
  } else if (model_token == "carreau-yasuda" || model_token == "carreau_yasuda") {
    model_keys = {"Viscosity.Limiting_high_shear_rate_viscosity",
                  "Viscosity.Limiting_low_shear_rate_viscosity",
                  "Viscosity.Shear_rate_tensor_multiplier",
                  "Viscosity.Power_law_index",
                  "Viscosity.Shear_rate_tensor_exponent"};
  }
  opt::OptionInput input;
  for (const auto& [name, parameter] : params) {
    if (name == "Backflow_stabilization_coefficient") {
      if (!trim_copy(parameter.value).empty()) {
        input.add(name, parameter.value);
      }
    } else if (name.rfind("Viscosity.", 0) == 0 && name != "Viscosity.model") {
      if (std::find(model_keys.begin(), model_keys.end(), name) != model_keys.end()) {
        input.add(name, parameter.value);
      }
    } else if (parameter.defined) {
      input.add(name, parameter.value);
    }
  }
  return input;
}

/// The equation and domain keys of an input, resolved through
/// navierStokesEquationSchema() and navierStokesDomainSchema().
struct NavierStokesKeys {
  opt::OptionInput equation_input{};
  opt::OptionInput domain_input{};
  opt::ResolvedOptions<ns::NavierStokesInputView> equation{};
  opt::ResolvedOptions<ns::NavierStokesInputView> domain{};

  /// The element that sets a key both elements read: the domain when it
  /// gives the key, else the equation when it does, else none.
  [[nodiscard]] const ns::NavierStokesInputView* domainFirst(std::string_view key) const
  {
    if (domain.isExplicit(key)) {
      return &domain.value;
    }
    return equation.isExplicit(key) ? &equation.value : nullptr;
  }
  /// The equation when it gives the key, else the domain when it does.
  [[nodiscard]] const ns::NavierStokesInputView* equationFirst(std::string_view key) const
  {
    if (equation.isExplicit(key)) {
      return &equation.value;
    }
    return domain.isExplicit(key) ? &domain.value : nullptr;
  }
};

NavierStokesKeys resolve_navier_stokes_keys(const svmp::Physics::ParameterMap& equation_params,
                                            const svmp::Physics::ParameterMap& domain_params)
{
  NavierStokesKeys keys;
  keys.equation_input = ns::optionInputFromParameters(equation_params);
  keys.domain_input = ns::optionInputFromParameters(domain_params);
  keys.equation = ns::navierStokesEquationSchema().resolve(keys.equation_input);
  keys.domain = ns::navierStokesDomainSchema().resolve(domain_keys_as_read(domain_params));
  return keys;
}

/// The moving domain and momentum source keys only (the two-fluid mode).
NavierStokesKeys resolve_moving_domain_and_momentum_source_keys(
    const svmp::Physics::ParameterMap& equation_params,
    const svmp::Physics::ParameterMap& domain_params)
{
  const auto& spellings = moving_domain_and_momentum_source_spellings();
  NavierStokesKeys keys;
  keys.equation_input = select_spellings(ns::optionInputFromParameters(equation_params), spellings);
  keys.domain_input = select_spellings(ns::optionInputFromParameters(domain_params), spellings);
  keys.equation = ns::navierStokesEquationSchema().resolve(keys.equation_input);
  keys.domain = ns::navierStokesDomainSchema().resolve(keys.domain_input);
  return keys;
}

svmp::FE::Real backflow_coefficient(const svmp::Physics::ParameterMap& domain_params)
{
  return ns::navierStokesDomainSchema()
      .resolve(select_spellings(domain_keys_as_read(domain_params),
                                {"Backflow_stabilization_coefficient"}))
      .value.backflow_stabilization_coefficient;
}

void apply_node_pressure_constraints(
    const svmp::Physics::EquationModuleInput& input,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  if (!input.node_pressure_constraints.has_value()) {
    return;
  }

  const auto& node_constraints = *input.node_pressure_constraints;
  options.node_pressure_constraints.id_type =
      parse_node_pressure_constraint_id_type(node_constraints.id_type);
  options.node_pressure_constraints.values =
      read_node_pressure_constraints_csv(node_constraints.values_file_path);
}

/// The moving-domain keys: a domain value replaces an equation value.
void apply_fluid_moving_domain_options(
    const NavierStokesKeys& keys,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  if (const auto* view = keys.domainFirst("Enable_ALE")) {
    options.enable_ale = view->enable_ale;
  }
  if (const auto* view = keys.domainFirst("Mesh_velocity_field")) {
    options.mesh_velocity_field_name = view->mesh_velocity_field;
  }
  if (const auto* view = keys.domainFirst("Mesh_velocity_source")) {
    options.mesh_velocity_source = view->mesh_velocity_source;
  }
  if (const auto* view = keys.domainFirst("Mesh_displacement_field")) {
    options.mesh_displacement_field_name = view->mesh_displacement_field;
  }
  if (const auto* view = keys.domainFirst("Auto_register_mesh_displacement_field")) {
    options.auto_register_mesh_displacement_field =
        view->auto_register_mesh_displacement_field;
  }
  if (const auto* view = keys.domainFirst("Moving_mesh_tangent_path")) {
    options.moving_mesh_tangent_path = view->moving_mesh_tangent_path;
  }
  if (trim_copy(options.mesh_velocity_field_name).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Mesh_velocity_field must be non-empty when configuring Navier-Stokes ALE.");
  }
  if (options.mesh_velocity_source ==
          svmp::Physics::formulations::navier_stokes::ALEMeshVelocitySource::CoupledDisplacement &&
      trim_copy(options.mesh_displacement_field_name).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Mesh_displacement_field must be non-empty when configuring coupled ALE.");
  }
}

/// The momentum-source field keys: a domain value replaces an equation value.
void apply_fluid_momentum_source_params(
    const NavierStokesKeys& keys,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  if (const auto* view = keys.domainFirst("Momentum_source_field_name")) {
    options.body_force_field_name = view->momentum_source_field_name;
  }
  if (const auto* view = keys.domainFirst("Auto_register_momentum_source_field")) {
    options.auto_register_body_force_field = view->auto_register_momentum_source_field;
  }
}

void apply_fluid_properties(const svmp::Physics::DomainInput& domain,
                            const NavierStokesKeys& keys,
                            svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  const auto& view = keys.domain.value;
  if (keys.domain.isExplicit("Density")) {
    options.density = view.density;
  }
  for (std::size_t d = 0; d < 3u; ++d) {
    static constexpr std::array<std::string_view, 3> force_keys{"Force_x", "Force_y", "Force_z"};
    if (keys.domain.isExplicit(force_keys[d])) {
      options.body_force[d] = view.body_force[d];
    }
  }
  if (keys.domain.isExplicit("Hydrostatic_pressure_initialization")) {
    options.hydrostatic_pressure_initialization.enabled = view.hydrostatic_pressure_initialization;
  }
  if (keys.domain.isExplicit("Hydrostatic_pressure_reference")) {
    options.hydrostatic_pressure_initialization.reference_pressure =
        view.hydrostatic_pressure_reference;
  }
  if (keys.domain.isExplicit("Hydrostatic_pressure_reference_point")) {
    options.hydrostatic_pressure_initialization.reference_point =
        view.hydrostatic_pressure_reference_point;
  }
  if (keys.domain.isExplicit("Hydrostatic_pressure_field_name")) {
    options.hydrostatic_pressure_initialization.field_name = view.hydrostatic_pressure_field_name;
  }

  if (!keys.domain.isExplicit("Viscosity.model")) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] <Viscosity model=\"...\"> is required for the new solver Navier-Stokes module. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto& model_raw = view.viscosity_model;
  const auto model = lower_copy(model_raw);

  if (model == "constant") {
    const double mu = static_cast<double>(view.viscosity_value);
    if (!(mu > 0.0)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Constant viscosity must be > 0 for the new solver Navier-Stokes module. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
    options.viscosity = static_cast<svmp::FE::Real>(mu);
    options.viscosity_model.reset();
    return;
  }

  if (model == "carreau-yasuda" || model == "carreau_yasuda") {
    const auto mu_inf = view.limiting_high_shear_rate_viscosity;
    const auto mu0 = view.limiting_low_shear_rate_viscosity;
    const auto lambda = view.shear_rate_tensor_multiplier;
    const auto n = view.power_law_index;
    const auto a = view.shear_rate_tensor_exponent;

    try {
      options.viscosity_model = std::make_shared<svmp::Physics::materials::fluid::CarreauYasudaViscosity>(
          mu0, mu_inf, lambda, n, a);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string("[svMultiPhysics::Physics] Invalid Carreau-Yasuda viscosity parameters for the new solver "
                      "Navier-Stokes module: ") +
          e.what() + ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    // The deck's own limiting low-shear viscosity (not its Parameters
    // default) is also the reference viscosity.
    if (const auto* p_mu0 = find_param(domain.params, "Viscosity.Limiting_low_shear_rate_viscosity");
        p_mu0 != nullptr && p_mu0->defined) {
      options.viscosity = mu0;
    }
    return;
  }

  if (model == "cassons" || model == "casson") {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Cassons viscosity model is not supported by the new solver Navier-Stokes module yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  throw std::runtime_error(
      "[svMultiPhysics::Physics] Fluid viscosity model '" + model_raw +
      "' is not supported by the new solver Navier-Stokes module. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> "
      "to use the legacy solver.");
}

/// The rotating frame: an equation value replaces a domain value.
void apply_fluid_rotating_frame_coriolis(
    const NavierStokesKeys& keys,
    int dim,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using Options = svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  const auto* switch_view = keys.equationFirst("Rotating_frame_coriolis");
  const bool enabled = switch_view != nullptr && switch_view->rotating_frame_coriolis;

  constexpr std::string_view omega_key = "Rotating_frame_angular_velocity";
  constexpr std::string_view file_key = "Rotating_frame_angular_velocity_temporal_values_file_path";
  if (keys.equation.isExplicit(omega_key) && keys.equation.isExplicit(file_key)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Specify either Rotating_frame_angular_velocity or "
        "Rotating_frame_angular_velocity_temporal_values_file_path, not both.");
  }
  if (keys.domain.isExplicit(omega_key) && keys.domain.isExplicit(file_key)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Specify either Rotating_frame_angular_velocity or "
        "Rotating_frame_angular_velocity_temporal_values_file_path in a domain, not both.");
  }

  // The equation's angular velocity (value or file) replaces the domain's.
  const bool equation_has_omega =
      keys.equation.isExplicit(omega_key) || keys.equation.isExplicit(file_key);
  const auto& omega_keys = equation_has_omega ? keys.equation : keys.domain;
  const ns::NavierStokesInputView* omega =
      omega_keys.isExplicit(omega_key) ? &omega_keys.value : nullptr;
  const ns::NavierStokesInputView* omega_file =
      omega_keys.isExplicit(file_key) ? &omega_keys.value : nullptr;

  if (!enabled && omega == nullptr && omega_file == nullptr) {
    return;
  }
  if (dim != 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Rotating-frame Coriolis forcing requires a 3D Navier-Stokes mesh.");
  }

  options.rotating_frame_coriolis_enabled = true;
  if (omega != nullptr) {
    for (std::size_t d = 0; d < 3u; ++d) {
      options.rotating_frame_angular_velocity[d] =
          Options::ScalarValue{omega->rotating_frame_angular_velocity[d]};
    }
  }
  if (omega_file != nullptr) {
    auto temporal = svmp::Physics::readTemporalValuesFile(
        omega_file->rotating_frame_angular_velocity_file, /*num_components=*/3,
        svmp::Physics::TemporalEndBehavior::Clamp);
    for (int d = 0; d < 3; ++d) {
      const int comp = d;
      options.rotating_frame_angular_velocity[static_cast<std::size_t>(d)] =
          svmp::FE::forms::TimeScalarCoefficient(
              [temporal, comp](svmp::FE::Real /*x*/,
                               svmp::FE::Real /*y*/,
                               svmp::FE::Real /*z*/,
                               svmp::FE::Real t) -> svmp::FE::Real {
                return temporal->interpolate(t, comp);
              });
    }
  }
}

/// The time/space momentum-source file: an equation value replaces a domain
/// value.
void apply_fluid_momentum_source_spacetime_file(
    const svmp::Physics::EquationModuleInput& input,
    const NavierStokesKeys& keys,
    int dim,
    MarkerCommunicator comm,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  const auto* source = keys.equationFirst("Momentum_source_temporal_and_spatial_values_file_path");
  if (source == nullptr || source->momentum_source_file.empty()) {
    return;
  }
  if (!input.mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Time/spatial Navier-Stokes momentum source requires a mesh.");
  }

  auto data = read_temporal_and_spatial_values_file(
      *input.mesh,
      /*boundary_marker=*/-1,
      source->momentum_source_file,
      comm);
  options.has_body_force_spacetime = true;
  for (int d = 0; d < dim; ++d) {
    if (d < data->dof) {
      const int comp = d;
      options.body_force_spacetime[static_cast<std::size_t>(d)] =
          svmp::FE::forms::TimeScalarCoefficient(
              [data, comp](svmp::FE::Real x,
                           svmp::FE::Real y,
                           svmp::FE::Real z,
                           svmp::FE::Real t) -> svmp::FE::Real {
                const std::array<svmp::FE::Real, 3> p{x, y, z};
                return data->interpolateSpatial(p, t, comp);
              });
    } else {
      options.body_force_spacetime[static_cast<std::size_t>(d)] =
          svmp::Physics::formulations::navier_stokes::
              IncompressibleNavierStokesVMSOptions::ScalarValue{0.0};
    }
  }
}

/// The equation and domain members of the artifact's "resolved" section:
/// each key with the value the module uses and its source (the input of
/// its element; "derived" when the value comes from the other element or
/// from a Parameters default).
std::string resolved_navier_stokes_input_json(
    const NavierStokesKeys& keys,
    const svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  auto used = keys.domain.value;
  const auto& equation = keys.equation.value;
  used.physical_model = equation.physical_model;
  used.configuration_schema_version = equation.configuration_schema_version;
  used.operator_tag = options.operator_tag;
  used.use_taylor_hood_type_basis = equation.use_taylor_hood_type_basis;
  used.enable_ale = options.enable_ale;
  used.mesh_velocity_field = options.mesh_velocity_field_name;
  used.mesh_velocity_source = options.mesh_velocity_source;
  used.mesh_displacement_field = options.mesh_displacement_field_name;
  used.auto_register_mesh_displacement_field = options.auto_register_mesh_displacement_field;
  used.moving_mesh_tangent_path = options.moving_mesh_tangent_path;
  used.momentum_source_field_name = options.body_force_field_name;
  used.auto_register_momentum_source_field = options.auto_register_body_force_field;
  const auto* source_file = keys.equationFirst("Momentum_source_temporal_and_spatial_values_file_path");
  used.momentum_source_file = source_file != nullptr ? source_file->momentum_source_file : std::string{};
  used.rotating_frame_coriolis = options.rotating_frame_coriolis_enabled;
  const bool equation_has_omega =
      keys.equation.isExplicit("Rotating_frame_angular_velocity") ||
      keys.equation.isExplicit("Rotating_frame_angular_velocity_temporal_values_file_path");
  const auto& omega = equation_has_omega ? keys.equation.value : keys.domain.value;
  used.rotating_frame_angular_velocity = omega.rotating_frame_angular_velocity;
  used.rotating_frame_angular_velocity_file = omega.rotating_frame_angular_velocity_file;
  used.density = options.density;
  used.body_force = options.body_force;
  used.hydrostatic_pressure_initialization = options.hydrostatic_pressure_initialization.enabled;
  used.hydrostatic_pressure_reference = options.hydrostatic_pressure_initialization.reference_pressure;
  used.hydrostatic_pressure_reference_point = options.hydrostatic_pressure_initialization.reference_point;
  used.hydrostatic_pressure_field_name = options.hydrostatic_pressure_initialization.field_name;

  const auto& equation_schema = ns::navierStokesEquationSchema();
  const auto& domain_schema = ns::navierStokesDomainSchema();
  const ns::NavierStokesInputView defaults{};
  return "\"equation\":" +
         equation_schema.toJson(equation_schema.describe(keys.equation_input, used, defaults)) +
         ",\"domain\":" +
         domain_schema.toJson(domain_schema.describe(keys.domain_input, used, defaults));
}

void apply_fluid_bcs(const svmp::Physics::EquationModuleInput& input,
                     const svmp::Physics::DomainInput& domain,
                     MarkerCommunicator comm,
                     svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation;

  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Navier-Stokes BC translation received null mesh.");
  }

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes BC translation: " +
        std::to_string(dim) +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const svmp::FE::Real backflow_beta = backflow_coefficient(domain.params);

  for (const auto& bc : input.boundary_conditions) {
    const auto* bc_type_raw = find_param(bc.params, "Type");
    const std::string bc_type = bc_type_raw ? trim_copy(bc_type_raw->value) : std::string{};
    const std::string bc_type_lc = lower_copy(bc_type);
    const bool free_surface_type = is_free_surface_type(bc_type);
    const bool unfitted_free_surface =
        free_surface_type &&
        free_surface_implementation_from_params(bc.params) == FreeSurfaceImplementation::UnfittedLevelSet;

    if (bc.boundary_marker == svmp::INVALID_LABEL && !unfitted_free_surface) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto* time_dep = find_param(bc.params, "Time_dependence");
    const std::string time_value_raw =
        (time_dep && time_dep->defined) ? trim_copy(time_dep->value) : std::string("Steady");
    const std::string time_value_lc = lower_copy(time_value_raw);

    const bool is_steady = time_value_lc.empty() || time_value_lc == "steady";
    const bool is_general = time_value_lc == "general";
    const bool is_unsteady = time_value_lc == "unsteady";
    const bool is_resistance = time_value_lc == "resistance";
    const bool is_rcr = (time_value_lc == "rcr" || time_value_lc == "windkessel");
    const bool is_rcrcr =
        (time_value_lc == "rcrcr" || time_value_lc == "windkessel2c" || time_value_lc == "windkessel_2c");

    const bool has_temp_spat = has_nonempty_defined(bc.params, "Temporal_and_spatial_values_file_path");
    const bool has_other_files = has_nonempty_defined(bc.params, "Temporal_values_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_values_file_path") || has_nonempty_defined(bc.params, "Bct_file_path") ||
        has_nonempty_defined(bc.params, "Traction_values_file_path") ||
        has_nonempty_defined(bc.params, "Fourier_coefficients_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_profile_file_path");

    if (is_steady && (has_temp_spat || has_other_files)) {
      throw std::runtime_error(
        "[svMultiPhysics::Physics] Spatial/temporal boundary-condition files are not supported for the new solver "
        "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (free_surface_type) {
      options.free_surface.push_back(
          resolve_free_surface_bc(bc, is_steady, has_temp_spat, has_other_files));
      continue;
    }

    const auto* value_param = find_param(bc.params, "Value");
    const svmp::FE::Real magnitude =
        static_cast<svmp::FE::Real>(value_param ? parse_double(value_param->value, "Add_BC/Value") : 0.0);

    std::vector<int> effective_dir{};
    if (const auto* dir_param = find_param(bc.params, "Effective_direction")) {
      const auto s = trim_copy(dir_param->value);
      if (!s.empty()) {
        effective_dir = parse_int_list(s);
      }
    }

    if (bc_type_lc == "dir" || bc_type_lc == "dirichlet") {
      if (!is_steady && !is_general && !is_unsteady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Only Steady, General, and Unsteady boundary conditions are supported for the new solver "
            "Navier-Stokes module Dirichlet BCs (got Time_dependence='" +
            time_value_raw + "'). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }

      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        // Legacy inputs sometimes set this flag for clarity; the new solver applies
        // velocity Dirichlet along the boundary normal by default when Effective_direction is unset.
        // Accept the flag for Dirichlet to improve legacy compatibility.
      }

      const auto* weak_param = find_param(bc.params, "Weakly_applied");
      const bool weak = weak_param && weak_param->defined && parse_bool_relaxed(weak_param->value);

      if (is_general) {
        if (!has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Navier-Stokes Dirichlet BC currently supports only "
              "<Temporal_and_spatial_values_file_path>. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }
        if (weak) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Dirichlet BCs from temporal/spatial files are only supported as strong "
              "Dirichlet (Weakly_applied=false). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto* file_param = find_param(bc.params, "Temporal_and_spatial_values_file_path");
        const std::string file_path =
            (file_param && file_param->defined) ? trim_copy(file_param->value) : std::string{};
        if (file_path.empty()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Dirichlet BC is missing Temporal_and_spatial_values_file_path.");
        }

        auto data = read_temporal_and_spatial_values_file(
            *input.mesh, bc.boundary_marker, file_path, comm);

        IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir{};
        dir.boundary_marker = bc.boundary_marker;

        for (int d = 0; d < dim; ++d) {
          if (d < data->dof) {
            const int comp = d;
            dir.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
                [data, comp](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real t) -> svmp::FE::Real {
                  const std::array<svmp::FE::Real, 3> p{x, y, z};
                  const auto node = data->findNodeIndex(p);
                  return data->interpolate(node, t, comp);
                });
          } else {
            dir.value[static_cast<std::size_t>(d)] = IncompressibleNavierStokesVMSOptions::ScalarValue{0.0};
          }
        }

        options.velocity_dirichlet.push_back(std::move(dir));
        continue;
      }

      if (is_unsteady) {
        // Unsteady Dirichlet BC: time-varying magnitude from a .flow file + spatial profile (Flat/Parabolic).
        const auto* file_param = find_param(bc.params, "Temporal_values_file_path");
        const std::string flow_file =
            (file_param && file_param->defined) ? trim_copy(file_param->value) : std::string{};
        if (flow_file.empty()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Unsteady Dirichlet BC '" + bc.name +
              "' is missing <Temporal_values_file_path>.");
        }

        auto temporal = svmp::Physics::readTemporalValuesFile(
            flow_file, /*num_components=*/1, svmp::Physics::TemporalEndBehavior::Periodic);

        const auto* impose_flux_param_u = find_param(bc.params, "Impose_flux");
        const bool impose_flux_u = impose_flux_param_u ? parse_bool_relaxed(impose_flux_param_u->value) : false;

        const auto* profile_param_u = find_param(bc.params, "Profile");
        const std::string profile_raw_u =
            (profile_param_u && profile_param_u->defined) ? trim_copy(profile_param_u->value) : std::string("Flat");
        const std::string profile_lc_u = lower_copy(profile_raw_u);

        InletProfileType profile_u = InletProfileType::Flat;
        if (profile_lc_u == "flat") {
          profile_u = InletProfileType::Flat;
        } else if (profile_lc_u == "parabolic") {
          profile_u = InletProfileType::Parabolic;
        } else {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Unknown Dirichlet BC Profile='" + profile_raw_u +
              "' for Unsteady BC '" + bc.name + "'. Supported profiles: Flat, Parabolic.");
        }

        int active_count_u = 0;
        std::array<int, 3> active_u{0, 0, 0};
        for (int d = 0; d < dim; ++d) {
          const int flag = (static_cast<std::size_t>(d) < effective_dir.size() && effective_dir[static_cast<std::size_t>(d)] != 0) ? 1 : 0;
          active_u[static_cast<std::size_t>(d)] = flag;
          active_count_u += flag;
        }
        const bool use_normal_u = (active_count_u == 0 || active_count_u == dim);

        auto ctx_u = std::make_shared<InletProfileContext>();
        ctx_u->dim = dim;
        ctx_u->profile = profile_u;
        ctx_u->use_normal_direction = use_normal_u;
        ctx_u->active_components = active_u;

        const auto g_u =
            global_marker_geometry(*input.mesh, bc.boundary_marker, comm);
        if (!(g_u.area > 0.0)) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
              " has zero area; cannot apply Unsteady Dirichlet BC '" + bc.name + "'.");
        }

        if (use_normal_u) {
          const Vec3d n = normalized(g_u.normal_sum);
          if (!(norm2(n) > 0.0)) {
            throw std::runtime_error(
                "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
                " has a degenerate normal; cannot apply Unsteady Dirichlet BC '" + bc.name + "'.");
          }
          ctx_u->normal = n;
        }

        if (profile_u == InletProfileType::Parabolic) {
          ctx_u->parabolic = build_parabolic_profile_data(
              *input.mesh, bc.boundary_marker, comm);
        }

      double normalization_u = 1.0;
      if (impose_flux_u) {
        if (profile_u == InletProfileType::Flat) {
          normalization_u = g_u.area;
        } else {
            normalization_u = integrate_parabolic_weight_over_marker(
                *input.mesh, bc.boundary_marker, *ctx_u->parabolic, comm);
          }
          if (!(normalization_u > 0.0)) {
            throw std::runtime_error(
                "[svMultiPhysics::Physics] Failed to compute positive normalization for <Impose_flux> on Unsteady "
                "Dirichlet BC '" + bc.name + "'.");
        }
      }

        // Set scale=1/normalization; the time-dependent flow rate is multiplied in the callback.
        ctx_u->scale = 1.0 / normalization_u;

        if (navierStokesTraceEnabled()) {
          const auto local_g = local_marker_geometry(*input.mesh, bc.boundary_marker);
          const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
          std::ostringstream oss;
          oss << "NavierStokes BC setup: unsteady Dirichlet marker=" << bc.boundary_marker
              << " name='" << bc.name << "'"
              << " profile=" << ((profile_u == InletProfileType::Parabolic) ? "Parabolic" : "Flat")
              << " impose_flux=" << (impose_flux_u ? 1 : 0)
              << " local_faces=" << local_faces.size()
              << " local_area=" << local_g.area
              << " global_area=" << g_u.area
              << " normalization=" << normalization_u
              << " scale=" << ctx_u->scale
              << " center=(" << ((g_u.area > 0.0) ? g_u.center_sum.x / g_u.area : 0.0)
              << "," << ((g_u.area > 0.0) ? g_u.center_sum.y / g_u.area : 0.0)
              << "," << ((g_u.area > 0.0) ? g_u.center_sum.z / g_u.area : 0.0) << ")"
              << " normal=(" << ctx_u->normal.x << "," << ctx_u->normal.y << "," << ctx_u->normal.z << ")"
              << " perim_points=" << (ctx_u->parabolic.has_value() ? ctx_u->parabolic->perimeter_unit_dirs.size() : 0u);
          navierStokesTraceLog(oss.str());
        }

        IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir_u{};
        dir_u.boundary_marker = bc.boundary_marker;
        dir_u.active_components = use_normal_u
                                      ? std::array<bool, 3>{true, true, true}
                                      : active_components_from_flags(active_u, dim);

        for (int d = 0; d < dim; ++d) {
          const int comp = d;
          dir_u.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
              [ctx_u, comp, temporal](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real t) -> svmp::FE::Real {
                const double flow_rate = temporal->interpolate(static_cast<double>(t));
                const Vec3d p{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
                // componentValue = scale * weight(x) * direction_factor
                // With scale = 1/normalization, this gives flow_rate / normalization * weight * direction
                return static_cast<svmp::FE::Real>(flow_rate * ctx_u->componentValue(comp, p));
              });
        }

        if (weak) {
          options.velocity_dirichlet_weak.push_back(std::move(dir_u));
        } else {
          options.velocity_dirichlet.push_back(std::move(dir_u));
        }
        continue;
      }

      IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir{};
      dir.boundary_marker = bc.boundary_marker;

      // Legacy-style Dirichlet velocity BCs:
      // - default direction: boundary outward normal
      // - profile: Flat or Parabolic
      // - Impose_flux: if true, normalize profile so ∫ profile ds = 1 and interpret Value as flow rate
      const auto* impose_flux_param = find_param(bc.params, "Impose_flux");
      const bool impose_flux = impose_flux_param ? parse_bool_relaxed(impose_flux_param->value) : false;

      const auto* profile_param = find_param(bc.params, "Profile");
      const std::string profile_raw =
          (profile_param && profile_param->defined) ? trim_copy(profile_param->value) : std::string("Flat");
      const std::string profile_lc = lower_copy(profile_raw);

      InletProfileType profile = InletProfileType::Flat;
      if (profile_lc == "flat") {
        profile = InletProfileType::Flat;
      } else if (profile_lc == "parabolic") {
        profile = InletProfileType::Parabolic;
      } else if (profile_lc == "user_defined" || profile_lc == "user-defined" || profile_lc == "userdefined") {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Dirichlet BC Profile='User_defined' is not supported for the new solver "
            "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      } else {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Unknown Dirichlet BC Profile='" + profile_raw +
            "' for the new solver Navier-Stokes module. Supported profiles: Flat, Parabolic.");
      }

      // Common case: no-slip wall (Value=0) and other zero Dirichlet constraints.
      // Avoid expensive global geometry computations (MPI allgathers) when the imposed value is zero.
      if (magnitude == static_cast<svmp::FE::Real>(0.0)) {
        if (!effective_dir.empty()) {
          std::array<int, 3> active_zero{0, 0, 0};
          int active_count_zero = 0;
          for (int d = 0; d < dim; ++d) {
            const int flag =
                (static_cast<std::size_t>(d) < effective_dir.size() &&
                 effective_dir[static_cast<std::size_t>(d)] != 0)
                    ? 1
                    : 0;
            active_zero[static_cast<std::size_t>(d)] = flag;
            active_count_zero += flag;
          }
          if (active_count_zero > 0 && active_count_zero < dim) {
            dir.active_components = active_components_from_flags(active_zero, dim);
          }
        }
        if (weak) {
          options.velocity_dirichlet_weak.push_back(std::move(dir));
        } else {
          options.velocity_dirichlet.push_back(std::move(dir));
        }
        continue;
      }

      int active_count = 0;
      std::array<int, 3> active{0, 0, 0};
      for (int d = 0; d < dim; ++d) {
        const int flag = (static_cast<std::size_t>(d) < effective_dir.size() && effective_dir[static_cast<std::size_t>(d)] != 0) ? 1 : 0;
        active[static_cast<std::size_t>(d)] = flag;
        active_count += flag;
      }
      const bool use_normal_direction = (active_count == 0 || active_count == dim);

      auto ctx = std::make_shared<InletProfileContext>();
      ctx->dim = dim;
      ctx->profile = profile;
      ctx->use_normal_direction = use_normal_direction;
      ctx->active_components = active;

      // Compute normal/area/center and profile normalization as needed.
      const auto g =
          global_marker_geometry(*input.mesh, bc.boundary_marker, comm);
      if (!(g.area > 0.0)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
            " has zero area; cannot apply Dirichlet BC '" + bc.name + "'.");
      }

      if (use_normal_direction) {
        const Vec3d n = normalized(g.normal_sum);
        if (!(norm2(n) > 0.0)) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
              " has a degenerate normal; cannot apply Dirichlet BC '" + bc.name + "'.");
        }
        ctx->normal = n;
      }

      if (profile == InletProfileType::Parabolic) {
        ctx->parabolic = build_parabolic_profile_data(
            *input.mesh, bc.boundary_marker, comm);
      }

      double normalization = 1.0;
      if (impose_flux) {
        if (profile == InletProfileType::Flat) {
          normalization = g.area;
        } else {
          normalization = integrate_parabolic_weight_over_marker(
              *input.mesh, bc.boundary_marker, *ctx->parabolic, comm);
        }
        if (!(normalization > 0.0)) {
          const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
          const std::size_t perim_points =
              (ctx->parabolic.has_value()) ? ctx->parabolic->perimeter_unit_dirs.size() : 0u;
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Failed to compute a positive normalization for <Impose_flux>true</Impose_flux> "
              "on Dirichlet BC '" + bc.name + "' (marker=" + std::to_string(bc.boundary_marker) +
              ", normalization=" + std::to_string(normalization) + ", area=" + std::to_string(g.area) +
              ", perim_points=" + std::to_string(perim_points) + ", local_faces=" + std::to_string(local_faces.size()) +
              ").");
        }
      }

      ctx->scale = static_cast<double>(magnitude) / normalization;

      if (navierStokesTraceEnabled()) {
        const auto local_g = local_marker_geometry(*input.mesh, bc.boundary_marker);
        const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
        std::ostringstream oss;
        oss << "NavierStokes BC setup: steady Dirichlet marker=" << bc.boundary_marker
            << " name='" << bc.name << "'"
            << " profile=" << ((profile == InletProfileType::Parabolic) ? "Parabolic" : "Flat")
            << " impose_flux=" << (impose_flux ? 1 : 0)
            << " magnitude=" << magnitude
            << " local_faces=" << local_faces.size()
            << " local_area=" << local_g.area
            << " global_area=" << g.area
            << " normalization=" << normalization
            << " scale=" << ctx->scale
            << " center=(" << ((g.area > 0.0) ? g.center_sum.x / g.area : 0.0)
            << "," << ((g.area > 0.0) ? g.center_sum.y / g.area : 0.0)
            << "," << ((g.area > 0.0) ? g.center_sum.z / g.area : 0.0) << ")"
            << " normal=(" << ctx->normal.x << "," << ctx->normal.y << "," << ctx->normal.z << ")"
            << " perim_points=" << (ctx->parabolic.has_value() ? ctx->parabolic->perimeter_unit_dirs.size() : 0u);
        navierStokesTraceLog(oss.str());
      }

      for (int d = 0; d < dim; ++d) {
        const int comp = d;
        dir.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
            [ctx, comp](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real /*t*/) -> svmp::FE::Real {
              const Vec3d p{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
              return static_cast<svmp::FE::Real>(ctx->componentValue(comp, p));
            });
      }
      dir.active_components = use_normal_direction
                                  ? std::array<bool, 3>{true, true, true}
                                  : active_components_from_flags(active, dim);

      if (weak) {
        options.velocity_dirichlet_weak.push_back(std::move(dir));
      } else {
        options.velocity_dirichlet.push_back(std::move(dir));
      }
      continue;
    }

    if (bc_type_lc == "neu" || bc_type_lc == "neumann") {
      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] <Apply_along_normal_direction>true</Apply_along_normal_direction> is not "
            "supported for the new solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to "
            "use the legacy solver.");
      }

      if (is_steady) {
        IncompressibleNavierStokesVMSOptions::PressureOutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.pressure = IncompressibleNavierStokesVMSOptions::ScalarValue{magnitude};
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.pressure_outflow.push_back(std::move(out));
        continue;
      }

      if (is_resistance) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Resistance outflow BCs do not support spatial/temporal files for the new "
              "solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = 0.0;
        out.C = 0.0;
        out.Rd = static_cast<svmp::FE::Real>(magnitude);
        out.Pd = 0.0;
        out.X0 = 0.0;
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcr.push_back(std::move(out));
        continue;
      }

      if (is_rcr) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCR outflow BCs do not support spatial/temporal files for the new solver "
              "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto Rp = get_defined_double(bc.params, "RCR.Proximal_resistance");
        const auto C = get_defined_double(bc.params, "RCR.Capacitance");
        const auto Rd = get_defined_double(bc.params, "RCR.Distal_resistance");
        const auto Pd = get_defined_double(bc.params, "RCR.Distal_pressure");
        const auto X0 = get_defined_double(bc.params, "RCR.Initial_pressure");

        if (!Rp.has_value() || !C.has_value() || !Rd.has_value()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCR outflow BC '" + bc.name + "' is missing required <RCR_values> entries "
              "(Proximal_resistance, Capacitance, Distal_resistance).");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = static_cast<svmp::FE::Real>(*Rp);
        out.C = static_cast<svmp::FE::Real>(*C);
        out.Rd = static_cast<svmp::FE::Real>(*Rd);
        out.Pd = static_cast<svmp::FE::Real>(Pd.value_or(0.0));
        out.X0 = static_cast<svmp::FE::Real>(X0.value_or(0.0));
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcr.push_back(std::move(out));
        continue;
      }

      if (is_rcrcr) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCRCR outflow BCs do not support spatial/temporal files for the new solver "
              "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto Rp = get_defined_double(bc.params, "RCRCR.Proximal_resistance");
        const auto C1 = get_defined_double(bc.params, "RCRCR.Proximal_capacitance");
        const auto Rm = get_defined_double(bc.params, "RCRCR.Intermediate_resistance");
        const auto C2 = get_defined_double(bc.params, "RCRCR.Distal_capacitance");
        const auto Rd = get_defined_double(bc.params, "RCRCR.Distal_resistance");
        const auto Pd = get_defined_double(bc.params, "RCRCR.Distal_pressure");
        const auto P10 = get_defined_double(bc.params, "RCRCR.Initial_pressure_1");
        const auto P20 = get_defined_double(bc.params, "RCRCR.Initial_pressure_2");

        if (!Rp.has_value() || !C1.has_value() || !Rm.has_value() || !C2.has_value() || !Rd.has_value()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCRCR outflow BC '" + bc.name +
              "' is missing required <RCRCR_values> entries "
              "(Proximal_resistance, Proximal_capacitance, Intermediate_resistance, Distal_capacitance, "
              "Distal_resistance).");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = static_cast<svmp::FE::Real>(*Rp);
        out.C1 = static_cast<svmp::FE::Real>(*C1);
        out.Rm = static_cast<svmp::FE::Real>(*Rm);
        out.C2 = static_cast<svmp::FE::Real>(*C2);
        out.Rd = static_cast<svmp::FE::Real>(*Rd);
        out.Pd = static_cast<svmp::FE::Real>(Pd.value_or(0.0));
        out.P10 = static_cast<svmp::FE::Real>(P10.value_or(Pd.value_or(0.0)));
        out.P20 = static_cast<svmp::FE::Real>(P20.value_or(Pd.value_or(0.0)));
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcrcr.push_back(std::move(out));
        continue;
      }

      throw std::runtime_error(
          "[svMultiPhysics::Physics] Neumann BC Time_dependence='" + time_value_raw +
          "' is not supported for the new solver Navier-Stokes module. Supported: Steady, Resistance, RCR, RCRCR. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (bc_type_lc == "trac" || bc_type_lc == "traction") {
      if (!is_steady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] General/Unsteady traction BCs are not supported for the new solver Navier-Stokes "
            "module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
      IncompressibleNavierStokesVMSOptions::TractionNeumannBC trac{};
      trac.boundary_marker = bc.boundary_marker;
      fill_vector(trac.traction, dim, effective_dir, magnitude);
      options.traction_neumann.push_back(std::move(trac));
      continue;
    }

    if (bc_type_lc == "rbn" || bc_type_lc == "robin") {
      if (!is_steady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] General/Unsteady Robin BCs are not supported for the new solver Navier-Stokes "
            "module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] <Apply_along_normal_direction>true</Apply_along_normal_direction> is not "
            "supported for the new solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to "
            "use the legacy solver.");
      }
      const auto* stiff_param = find_param(bc.params, "Stiffness");
      const svmp::FE::Real stiff =
          static_cast<svmp::FE::Real>(stiff_param ? parse_double(stiff_param->value, "Add_BC/Stiffness") : 0.0);

      IncompressibleNavierStokesVMSOptions::TractionRobinBC robin{};
      robin.boundary_marker = bc.boundary_marker;
      robin.alpha = IncompressibleNavierStokesVMSOptions::ScalarValue{stiff};
      fill_vector(robin.rhs, dim, effective_dir, magnitude);
      options.traction_robin.push_back(std::move(robin));
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary condition type '" + bc_type +
        "' is not supported for the new solver Navier-Stokes module. Supported types: Dir, Dirichlet, Neu, Neumann, "
        "Trac, Traction, Robin, Rbn, Free_surface. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

} // namespace

TranslatedIncompressibleTwoFluidInput
translate_incompressible_two_fluid_input(
    const svmp::Physics::EquationModuleInput& input,
    const svmp::FE::systems::FESystem& system)
{
  using Options = svmp::Physics::formulations::navier_stokes::
      IncompressibleTwoFluidOptions;
  using BoundaryOptions = svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSOptions;

  if (!input.mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Incompressible two-fluid module factory received null mesh.");
  }
  if (input.body_force_block_count != 0u) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_body_force_configuration");
  }
  if (!input.domains.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_explicit_domain_configuration");
  }
  if (!trim_copy(input.module_options).empty() ||
      !trim_copy(input.module_options_file_path).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_module_options");
  }
  if (!input.outputs.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_output_configuration");
  }
  if (!input.nested_configuration_blocks.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_two_fluid_equation_block:" +
        input.nested_configuration_blocks.front());
  }
  if (!input.default_domain.nested_configuration_blocks.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] "
        "unsupported_two_fluid_default_domain_block:" +
        input.default_domain.nested_configuration_blocks.front());
  }

  const int dimension = input.mesh->dim();
  if (dimension != 2 && dimension != 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Incompressible two-fluid spaces require dimension two or three.");
  }
  const auto element_type = infer_base_element_type(*input.mesh);
  const bool supported_cell =
      (dimension == 2 && element_type == svmp::FE::ElementType::Triangle3) ||
      (dimension == 3 && element_type == svmp::FE::ElementType::Tetra4);
  const int order =
      resolve_element_order(input, infer_polynomial_order(*input.mesh));
  // The two-fluid keys resolve through incompressibleTwoFluidSchema()
  // (decision D47, P2); the basis switch is read first, as before.
  const auto& schema = ns::incompressibleTwoFluidSchema();
  const auto equation_input = ns::optionInputFromParameters(input.equation_params);
  const bool taylor_hood =
      schema.resolve(select_spellings(equation_input, {"Use_taylor_hood_type_basis"}))
          .value.use_taylor_hood_type_basis;
  if (!supported_cell || order != 1 || taylor_hood) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Incompressible two-fluid input requires affine equal-order P1 Triangle3 or Tetra4 spaces.");
  }

  Options options;
  if (input.node_pressure_constraints.has_value()) {
    const auto& gauge_input = *input.node_pressure_constraints;
    (void)parse_node_pressure_constraint_id_type(gauge_input.id_type);
    const auto values =
        read_node_pressure_constraints_csv(gauge_input.values_file_path);
    if (values.size() != 1u) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_two_fluid_shared_pressure_gauge_count");
    }
    options.shared_pressure_gauge =
        svmp::Physics::formulations::navier_stokes::
            IncompressibleTwoFluidSharedPressureGauge{
            .vertex_gid = values.front().node_id,
            .pressure = values.front().pressure,
        };
  }
  // Required keys: absent or empty values and two spellings fail with the
  // register's tokens, in the register's order; then every value resolves.
  const auto require = [&](std::initializer_list<std::string_view> keys,
                           std::string_view canonical) {
    (void)require_two_fluid_string(input.equation_params, keys, canonical);
  };
  for (const std::string_view key :
       {"Level_set_field_name", "Generated_interface_domain_id", "Material_interface_marker",
        "Negative_phase_density", "Negative_phase_dynamic_viscosity", "Positive_phase_density",
        "Positive_phase_dynamic_viscosity", "Two_fluid_surface_tension",
        "Two_fluid_interface_nitsche_gamma"}) {
    require({key}, key);
  }
  if (any_parameter_defined(input.equation_params, {"Prescribed_pressure_jump"})) {
    require({"Prescribed_pressure_jump"}, "Prescribed_pressure_jump");
  }
  const bool traction_jump = any_parameter_defined(
      input.equation_params,
      {"Prescribed_viscous_traction_jump_x",
       "Prescribed_viscous_traction_jump_y",
       "Prescribed_viscous_traction_jump_z"});
  if (traction_jump) {
    if (dimension == 2 &&
        any_parameter_defined(
            input.equation_params,
            {"Prescribed_viscous_traction_jump_z"})) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_two_fluid_out_of_plane_prescribed_viscous_traction_jump");
    }
    require({"Prescribed_viscous_traction_jump_x"}, "Prescribed_viscous_traction_jump_x");
    require({"Prescribed_viscous_traction_jump_y"}, "Prescribed_viscous_traction_jump_y");
    if (dimension == 3) {
      require({"Prescribed_viscous_traction_jump_z"}, "Prescribed_viscous_traction_jump_z");
    }
  }
  if (any_parameter_defined(
          input.equation_params, {"Operator_tag", "OperatorTag"})) {
    require({"Operator_tag", "OperatorTag"}, "Operator_tag");
  }
  const auto resolved = schema.resolve(equation_input);
  const auto& keys = resolved.value;
  options.level_set_field_name = keys.level_set_field_name;
  options.generated_interface_domain_id = keys.generated_interface_domain_id;
  options.interface_marker = keys.material_interface_marker;
  options.negative_phase.density = keys.negative_phase_density;
  options.negative_phase.viscosity = keys.negative_phase_dynamic_viscosity;
  options.positive_phase.density = keys.positive_phase_density;
  options.positive_phase.viscosity = keys.positive_phase_dynamic_viscosity;
  options.surface_tension = keys.two_fluid_surface_tension;
  options.interface_nitsche_gamma = keys.two_fluid_interface_nitsche_gamma;
  if (keys.prescribed_pressure_jump.has_value()) {
    options.prescribed_pressure_jump = *keys.prescribed_pressure_jump;
  }
  if (traction_jump) {
    std::array<svmp::FE::Real, 3> target{};
    target[0] = keys.prescribed_viscous_traction_jump_x.value_or(svmp::FE::Real{0.0});
    target[1] = keys.prescribed_viscous_traction_jump_y.value_or(svmp::FE::Real{0.0});
    if (dimension == 3) {
      target[2] = keys.prescribed_viscous_traction_jump_z.value_or(svmp::FE::Real{0.0});
    }
    options.prescribed_viscous_traction_jump = target;
  }
  options.require_conservative_phase_momentum_reconciliation = true;
  if (options.interface_marker < 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Material_interface_marker must be nonnegative for incompressible two-fluid input.");
  }
  if (resolved.isExplicit("Operator_tag")) {
    options.operator_tag = keys.operator_tag;
  }
  options.enable_convection = input.equation_type != "stokes";
  options.jit_policy =
      svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);

  const auto& domain_schema = ns::incompressibleTwoFluidDomainSchema();
  const auto domain_input = ns::optionInputFromParameters(input.default_domain.params);
  const auto domain_keys = domain_schema.resolve(domain_input);
  for (std::size_t component = 0u; component < options.body_force.size();
       ++component) {
    static constexpr std::array<std::string_view, 3> force_keys{
        "Force_x", "Force_y", "Force_z"};
    if (domain_keys.isExplicit(force_keys[component])) {
      options.body_force[component] = domain_keys.value.body_force[component];
    }
  }

  BoundaryOptions translated_boundaries;
  const auto shared_keys = resolve_moving_domain_and_momentum_source_keys(
      input.equation_params, input.default_domain.params);
  apply_fluid_moving_domain_options(shared_keys, translated_boundaries);
  if (translated_boundaries.enable_ale) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_moving_mesh");
  }
  apply_fluid_momentum_source_params(shared_keys, translated_boundaries);
  if (!translated_boundaries.body_force_field_name.empty() ||
      any_parameter_defined(
          input.equation_params,
          {"Momentum_source_temporal_and_spatial_values_file_path",
           "MomentumSourceTemporalAndSpatialValuesFilePath",
           "Body_force_temporal_and_spatial_values_file_path",
           "BodyForceTemporalAndSpatialValuesFilePath"}) ||
      any_parameter_defined(
          input.default_domain.params,
          {"Momentum_source_temporal_and_spatial_values_file_path",
           "MomentumSourceTemporalAndSpatialValuesFilePath",
           "Body_force_temporal_and_spatial_values_file_path",
           "BodyForceTemporalAndSpatialValuesFilePath"})) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_nonconstant_momentum_source");
  }
  reject_unsupported_two_fluid_equation_parameters(
      input.equation_params);
  reject_unsupported_two_fluid_default_domain_parameters(
      input.default_domain.params);
  apply_fluid_bcs(
      input,
      input.default_domain,
      markerCommunicator(system),
      translated_boundaries);
  if (!translated_boundaries.velocity_dirichlet_weak.empty() ||
      !translated_boundaries.pressure_dirichlet.empty() ||
      !translated_boundaries.traction_neumann.empty() ||
      !translated_boundaries.traction_robin.empty() ||
      !translated_boundaries.pressure_outflow.empty() ||
      !translated_boundaries.free_surface.empty() ||
      !translated_boundaries.coupled_outflow_rcr.empty() ||
      !translated_boundaries.coupled_outflow_rcrcr.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_boundary_condition");
  }
  reject_unsupported_two_fluid_boundary_parameters(
      input.boundary_conditions);
  options.shared_velocity_dirichlet =
      std::move(translated_boundaries.velocity_dirichlet);
  // The two-fluid members of the artifact's "resolved" section.
  options.resolved_input_json =
      "\"two_fluid\":" +
      schema.toJson(schema.describe(equation_input, keys, ns::IncompressibleTwoFluidInputView{})) +
      ",\"domain\":" +
      domain_schema.toJson(domain_schema.describe(domain_input, domain_keys.value,
                                                  ns::NavierStokesInputView{}));

  auto velocity_space = svmp::FE::spaces::VectorSpace(
      svmp::FE::spaces::SpaceType::H1,
      element_type,
      1,
      dimension);
  auto pressure_space =
      svmp::FE::spaces::SpaceFactory::create_h1(element_type, 1);
  return TranslatedIncompressibleTwoFluidInput{
      .velocity_space = std::move(velocity_space),
      .pressure_space = std::move(pressure_space),
      .options = std::move(options),
  };
}

namespace {

// The one-phase options and spaces of a fluid or stokes input (the
// translation that create_navier_stokes_from_input ran before it constructed
// the module).
ResolvedOnePhaseNavierStokesInput resolve_one_phase_navier_stokes_input(
    const svmp::Physics::EquationModuleInput& input,
    svmp::FE::systems::FESystem& system)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Navier-Stokes module factory received null mesh.");
  }

  const auto& domain = select_single_domain(input, "Navier-Stokes");

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes spaces: " + std::to_string(dim) +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int vel_order = resolve_element_order(input, infer_polynomial_order(*input.mesh));

  // The equation and domain keys (decision D47, P2): every value and format
  // error of these keys is reported here.
  const auto keys = resolve_navier_stokes_keys(input.equation_params, domain.params);
  const bool taylor_hood = keys.equation.value.use_taylor_hood_type_basis;
  const int p_order = taylor_hood ? std::max(1, vel_order - 1) : vel_order;

  auto velocity_space =
      svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, element_type, vel_order, dim);
  auto pressure_space = svmp::FE::spaces::SpaceFactory::create_h1(element_type, p_order);

  svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions options{};
  options.velocity_field_name = "Velocity";
  options.pressure_field_name = "Pressure";
  options.free_surface_physical_model =
      svmp::Physics::formulations::navier_stokes::
          FreeSurfacePhysicalModel::
              OnePhaseLiquidPrescribedExteriorPressure;
  if (keys.equation.isExplicit("Operator_tag")) {
    options.operator_tag = keys.equation.value.operator_tag;
  }
  options.enable_convection = (input.equation_type != "stokes");
  options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);
  options.conservative_balance_diagnostics =
      input.derived_requests.conservative_balance_diagnostics;

  apply_fluid_moving_domain_options(keys, options);
  apply_fluid_momentum_source_params(keys, options);
  apply_fluid_properties(domain, keys, options);
  apply_fluid_momentum_source_spacetime_file(
      input, keys, dim, markerCommunicator(system), options);
  apply_fluid_rotating_frame_coriolis(keys, dim, options);
  apply_node_pressure_constraints(input, options);
  apply_fluid_bcs(input, domain, markerCommunicator(system), options);
  options.resolved_input_json = resolved_navier_stokes_input_json(keys, options);

  return ResolvedOnePhaseNavierStokesInput{
      .velocity_space = std::move(velocity_space),
      .pressure_space = std::move(pressure_space),
      .options = std::move(options),
  };
}

} // namespace

ResolvedNavierStokesInput resolveNavierStokesInput(
    const svmp::Physics::EquationModuleInput& input,
    svmp::FE::systems::FESystem& system)
{
  svmp::Physics::formulations::navier_stokes::
      preflightFittedSurfaceContactCapability(input);
  ResolvedNavierStokesInput resolved;
  resolved.physical_model =
      validate_and_resolve_free_surface_physical_model(input);
  if (resolved.physical_model ==
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    resolved.two_fluid = translate_incompressible_two_fluid_input(input, system);
    return resolved;
  }
  resolved.one_phase = resolve_one_phase_navier_stokes_input(input, system);
  return resolved;
}

} // namespace svmp::Physics::options

namespace svmp::Physics::formulations::navier_stokes {

// The register API (declared in this file's header) reads the translation
// above.
using namespace svmp::Physics::options;

void preflightFittedSurfaceContactCapability(
    const svmp::Physics::EquationModuleInput& input)
{
  if (input.equation_type != "fluid" && input.equation_type != "stokes") {
    throw std::invalid_argument(
        "preflightFittedSurfaceContactCapability requires a fluid or stokes equation input");
  }
  const auto physical_model =
      validate_and_resolve_free_surface_physical_model(input);
  if (physical_model ==
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    return;
  }
  validate_fitted_surface_contact_capability(
      translate_fitted_surface_contact_capability(input));
}

std::optional<IncompressibleTwoFluidDependency>
incompressibleTwoFluidDependency(
    const svmp::Physics::EquationModuleInput& input,
    const FE::systems::FESystem& system)
{
  if (input.equation_type != "fluid" && input.equation_type != "stokes") {
    throw std::invalid_argument(
        "incompressibleTwoFluidDependency requires a fluid or stokes equation input");
  }
  if (validate_and_resolve_free_surface_physical_model(input) !=
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    return std::nullopt;
  }
  auto translated =
      translate_incompressible_two_fluid_input(input, system);
  const auto& options = translated.options;
  if (!input.mesh || !translated.velocity_space ||
      !translated.pressure_space) {
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] incomplete incompressible two-fluid dependency");
  }
  return IncompressibleTwoFluidDependency{
      .mesh = input.mesh.get(),
      .mesh_name = input.mesh_name,
      .dimension = translated.velocity_space->value_dimension(),
      .interface_marker = options.interface_marker,
      .level_set_field_name = options.level_set_field_name,
      .negative_velocity_field_name =
          options.negative_phase.velocity_field_name,
      .positive_velocity_field_name =
          options.positive_phase.velocity_field_name,
      .negative_pressure_field_name =
          options.negative_phase.pressure_field_name,
      .positive_pressure_field_name =
          options.positive_phase.pressure_field_name,
      .operator_tag = options.operator_tag,
      .generated_interface_domain_id =
          options.generated_interface_domain_id,
  };
}

void preRegisterIncompressibleTwoFluidDependencyFields(
    const svmp::Physics::EquationModuleInput& input,
    FE::systems::FESystem& system)
{
  if (validate_and_resolve_free_surface_physical_model(input) !=
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    throw std::invalid_argument(
        "preRegisterIncompressibleTwoFluidDependencyFields requires incompressible two-fluid input");
  }
  auto translated =
      translate_incompressible_two_fluid_input(input, system);
  preflightTwoFluidTranslationBeforeMutation(translated, system);

  const auto& options = translated.options;
  const int dimension = translated.velocity_space->value_dimension();
  const auto u_negative = ensureTwoFluidUnknownField(
      system,
      options.negative_phase.velocity_field_name,
      translated.velocity_space,
      dimension);
  const auto u_positive = ensureTwoFluidUnknownField(
      system,
      options.positive_phase.velocity_field_name,
      translated.velocity_space,
      dimension);
  const auto p_negative = ensureTwoFluidUnknownField(
      system,
      options.negative_phase.pressure_field_name,
      translated.pressure_space,
      1);
  const auto p_positive = ensureTwoFluidUnknownField(
      system,
      options.positive_phase.pressure_field_name,
      translated.pressure_space,
      1);
  (void)p_negative;
  (void)p_positive;

  const auto level_set =
      system.findFieldByName(options.level_set_field_name);
  FE::interfaces::GeneratedInterfaceMarkerKey marker_key{};
  marker_key.source =
      FE::interfaces::LevelSetInterfaceSource::fromField(level_set);
  marker_key.domain_id = options.generated_interface_domain_id;
  marker_key.isovalue = options.level_set_isovalue;
  marker_key.requested_marker = options.interface_marker;
  const int interface_marker =
      FE::interfaces::stableGeneratedInterfaceMarker(marker_key);

  const Factories::IncompressibleTwoFluidInterfaceParameters parameters{
      .dimension = dimension,
      .interface_marker = interface_marker,
      .negative_density = options.negative_phase.density,
      .positive_density = options.positive_phase.density,
      .negative_viscosity = options.negative_phase.viscosity,
      .positive_viscosity = options.positive_phase.viscosity,
      .nitsche_gamma = options.interface_nitsche_gamma,
      .surface_tension = options.surface_tension,
      .include_transient_penalty =
          options.include_transient_interface_penalty,
      .prescribed_pressure_jump = options.prescribed_pressure_jump,
      .prescribed_viscous_traction_jump =
          options.prescribed_viscous_traction_jump,
  };
  const auto weights = Factories::incompressibleTwoFluidInterfaceWeights(parameters);
  system.declareMaterialInterfaceTransportVelocity(
      FE::interfaces::MaterialInterfaceTransportVelocityDeclaration{
          .dimension = dimension,
          .interface_marker = interface_marker,
          .level_set_field = level_set,
          .negative_velocity_field = u_negative,
          .positive_velocity_field = u_positive,
          .level_set_isovalue = options.level_set_isovalue,
          .negative_trace_weight = weights.negative_complement,
          .positive_trace_weight = weights.positive_complement,
          .geometry_domain_id = options.generated_interface_domain_id,
          .owner_component = "incompressible_two_fluid",
      });
}

FE::FieldId preRegisterPrimaryVelocityField(
    const svmp::Physics::EquationModuleInput& input,
    FE::systems::FESystem& system)
{
  preflightFittedSurfaceContactCapability(input);
  if (validate_and_resolve_free_surface_physical_model(input) ==
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    throw std::invalid_argument(
        "Incompressible two-fluid input owns separate phase velocities and cannot pre-register a field named 'Velocity'");
  }
  if (input.equation_type != "fluid" && input.equation_type != "stokes") {
    throw std::invalid_argument(
        "preRegisterPrimaryVelocityField requires a fluid or stokes equation input");
  }
  if (!input.mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Navier-Stokes velocity pre-registration received null mesh.");
  }

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes velocity pre-registration: " +
        std::to_string(dim));
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int velocity_order =
      resolve_element_order(input, infer_polynomial_order(*input.mesh));
  auto velocity_space = svmp::FE::spaces::VectorSpace(
      svmp::FE::spaces::SpaceType::H1,
      element_type,
      velocity_order,
      dim);

  FE::systems::FieldSpec velocity_spec;
  velocity_spec.name = "Velocity";
  velocity_spec.space = std::move(velocity_space);
  velocity_spec.components = dim;
  const auto existing = system.findFieldByName(velocity_spec.name);
  if (existing == FE::INVALID_FIELD_ID) {
    return system.addField(std::move(velocity_spec));
  }

  const auto& record = system.fieldRecord(existing);
  if (record.source_kind != FE::systems::FieldSourceKind::Unknown ||
      record.components != velocity_spec.components || !record.space ||
      !velocity_spec.space ||
      record.space->space_type() != velocity_spec.space->space_type() ||
      record.space->field_type() != velocity_spec.space->field_type() ||
      record.space->value_dimension() !=
          velocity_spec.space->value_dimension() ||
      record.space->topological_dimension() !=
          velocity_spec.space->topological_dimension() ||
      record.space->polynomial_order() !=
          velocity_spec.space->polynomial_order() ||
      record.space->element_type() != velocity_spec.space->element_type()) {
    throw std::invalid_argument(
        "Navier-Stokes future velocity pre-registration found an incompatible existing field 'Velocity'");
  }
  return existing;
}

} // namespace svmp::Physics::formulations::navier_stokes
