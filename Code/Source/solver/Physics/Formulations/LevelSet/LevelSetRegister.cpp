#include "Physics/Formulations/LevelSet/LevelSetTransportModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Core/JITRuntimePolicy.h"

#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace ls = svmp::Physics::formulations::level_set;

std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string normalized_token(std::string s)
{
  s = lower_copy(trim_copy(std::move(s)));
  s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch) {
            return ch == '_' || ch == '-' || std::isspace(ch);
          }),
          s.end());
  return s;
}

const svmp::Physics::ParameterValue* find_param(const svmp::Physics::ParameterMap& params,
                                                std::string_view key)
{
  const auto it = params.find(std::string(key));
  return (it == params.end()) ? nullptr : &it->second;
}

std::optional<std::string> get_defined_string(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    const auto* p = find_param(params, key);
    if (!p || !p->defined) {
      continue;
    }
    auto value = trim_copy(p->value);
    if (!value.empty()) {
      return value;
    }
  }
  return std::nullopt;
}

bool parse_bool_relaxed(std::string_view raw)
{
  const auto value = lower_copy(trim_copy(std::string(raw)));
  return value == "true" || value == "1" || value == "yes" || value == "on";
}

std::optional<bool> get_defined_bool(const svmp::Physics::ParameterMap& params,
                                     std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    const auto* p = find_param(params, key);
    if (!p || !p->defined) {
      continue;
    }
    const auto value = trim_copy(p->value);
    if (!value.empty()) {
      return parse_bool_relaxed(value);
    }
  }
  return std::nullopt;
}

double parse_double(std::string_view raw, std::string_view context)
{
  const auto s = trim_copy(std::string(raw));
  try {
    size_t pos = 0;
    const double v = std::stod(s, &pos);
    if (pos != s.size() || !std::isfinite(v)) {
      throw std::runtime_error("");
    }
    return v;
  } catch (...) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse numeric value '" + std::string(raw) +
                             "' for " + std::string(context) + ".");
  }
}

std::optional<svmp::FE::Real> get_defined_real(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  for (const auto key : keys) {
    const auto* p = find_param(params, key);
    if (!p || !p->defined) {
      continue;
    }
    const auto value = trim_copy(p->value);
    if (!value.empty()) {
      return static_cast<svmp::FE::Real>(parse_double(value, context));
    }
  }
  return std::nullopt;
}

int parse_positive_int(std::string_view raw, std::string_view context)
{
  const auto s = trim_copy(std::string(raw));
  try {
    size_t pos = 0;
    const int v = std::stoi(s, &pos);
    if (pos != s.size() || v < 1) {
      throw std::runtime_error("");
    }
    return v;
  } catch (...) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse positive integer value '" +
                             std::string(raw) + "' for " + std::string(context) + ".");
  }
}

svmp::FE::ElementType infer_base_element_type(const svmp::MeshBase& mesh)
{
  if (mesh.n_cells() == 0) {
    throw std::runtime_error("[svMultiPhysics::Physics] Mesh has no cells; cannot infer FE element type.");
  }

  const auto& shapes = mesh.cell_shapes();
  if (shapes.empty()) {
    throw std::runtime_error("[svMultiPhysics::Physics] Mesh has no cell shapes; cannot infer FE element type.");
  }

  if (shapes.front().is_mixed_order) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Mixed-order meshes are not supported by the level-set transport module.");
  }

  const auto family = shapes.front().family;
  for (const auto& s : shapes) {
    if (s.family != family) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Mixed cell families are not supported by the level-set transport module.");
    }
  }

  switch (family) {
    case svmp::CellFamily::Line: return svmp::FE::ElementType::Line2;
    case svmp::CellFamily::Triangle: return svmp::FE::ElementType::Triangle3;
    case svmp::CellFamily::Quad: return svmp::FE::ElementType::Quad4;
    case svmp::CellFamily::Tetra: return svmp::FE::ElementType::Tetra4;
    case svmp::CellFamily::Hex: return svmp::FE::ElementType::Hex8;
    case svmp::CellFamily::Wedge: return svmp::FE::ElementType::Wedge6;
    case svmp::CellFamily::Pyramid: return svmp::FE::ElementType::Pyramid5;
    default:
      break;
  }

  throw std::runtime_error("[svMultiPhysics::Physics] Unsupported mesh cell family for level-set transport.");
}

int infer_polynomial_order(const svmp::MeshBase& mesh)
{
  const auto& shapes = mesh.cell_shapes();
  if (shapes.empty()) {
    return 1;
  }

  const int order = shapes.front().order > 0 ? shapes.front().order : 1;
  for (const auto& s : shapes) {
    const int s_order = s.order > 0 ? s.order : 1;
    if (s_order != order) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Mixed polynomial orders are not supported by the level-set transport module.");
    }
  }

  return order;
}

int resolve_element_order(const svmp::Physics::EquationModuleInput& input, int inferred_order)
{
  if (const auto* p = find_param(input.equation_params, "Element_order"); p && p->defined) {
    return parse_positive_int(p->value, "Element_order");
  }
  return inferred_order;
}

ls::LevelSetFieldSource parse_level_set_source(std::string_view raw)
{
  const auto value = normalized_token(std::string(raw));
  if (value == "unknown" || value == "solution") {
    return ls::LevelSetFieldSource::Unknown;
  }
  if (value == "prescribed" || value == "prescribeddata" || value == "data") {
    return ls::LevelSetFieldSource::PrescribedData;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] Level_set_source must be one of 'unknown' or 'prescribed_data'.");
}

ls::LevelSetVelocitySource parse_velocity_source(std::string_view raw)
{
  const auto value = normalized_token(std::string(raw));
  if (value == "coupled" || value == "coupledfield" || value == "unknown" || value == "navierstokes") {
    return ls::LevelSetVelocitySource::CoupledField;
  }
  if (value == "prescribed" || value == "prescribeddata" || value == "data" || value == "constant") {
    return ls::LevelSetVelocitySource::PrescribedData;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] Velocity_source must be one of 'coupled_field' or 'prescribed_data'.");
}

void apply_level_set_params(const svmp::Physics::ParameterMap& params,
                            ls::LevelSetTransportOptions& options)
{
  if (const auto value = get_defined_string(
          params,
          {"Level_set_field_name", "LevelSetFieldName", "Level_set_field", "LevelSetField", "Field_name"})) {
    options.level_set.field_name = *value;
  }
  if (const auto value = get_defined_string(params, {"Level_set_source", "LevelSetSource"})) {
    options.level_set.source = parse_level_set_source(*value);
  }
  if (const auto value = get_defined_bool(
          params,
          {"Auto_register_level_set_field", "AutoRegisterLevelSetField"})) {
    options.level_set.auto_register_field = *value;
  }

  if (const auto value = get_defined_string(
          params,
          {"Velocity_field_name", "VelocityFieldName", "Advection_velocity_field", "AdvectionVelocityField"})) {
    options.velocity.field_name = *value;
  }
  if (const auto value = get_defined_string(params, {"Velocity_source", "VelocitySource"})) {
    options.velocity.source = parse_velocity_source(*value);
    if (options.velocity.source == ls::LevelSetVelocitySource::PrescribedData) {
      options.velocity.auto_register_field = true;
    }
  }
  if (const auto value = get_defined_bool(
          params,
          {"Auto_register_velocity_field", "AutoRegisterVelocityField"})) {
    options.velocity.auto_register_field = *value;
  }

  if (const auto value = get_defined_bool(params, {"Enable_SUPG", "SUPG", "SUPG_enabled"})) {
    options.supg.enabled = *value;
  }
  if (const auto value = get_defined_real(params, {"SUPG_tau_scale", "SUPGScale"}, "SUPG_tau_scale")) {
    options.supg.tau_scale = *value;
  }
  if (const auto value = get_defined_real(
          params,
          {"SUPG_velocity_epsilon", "SUPGVelocityEpsilon"},
          "SUPG_velocity_epsilon")) {
    options.supg.velocity_epsilon = *value;
  }
}

void apply_level_set_bcs(const svmp::Physics::EquationModuleInput& input,
                         ls::LevelSetTransportOptions& options)
{
  for (const auto& bc : input.boundary_conditions) {
    if (bc.boundary_marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Level-set boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto* type_param = find_param(bc.params, "Type");
    const std::string type = type_param ? normalized_token(type_param->value) : std::string{};

    if (type == "levelsetinflow" || type == "inflow" || type == "levelsetdirichlet") {
      ls::LevelSetInflowBoundary inflow{};
      inflow.boundary_marker = bc.boundary_marker;
      if (const auto value = get_defined_real(bc.params, {"Value", "Level_set_value"}, "Add_BC/Value")) {
        inflow.value = *value;
      }
      if (const auto penalty = get_defined_real(
              bc.params,
              {"Penalty_scale", "Penalty", "Inflow_penalty_scale"},
              "Add_BC/Penalty_scale")) {
        inflow.penalty_scale = *penalty;
      }
      options.boundaries.inflow.push_back(std::move(inflow));
      continue;
    }

    if (type == "levelsetoutflow" || type == "outflow") {
      options.boundaries.outflow.push_back(ls::LevelSetOutflowBoundary{.boundary_marker = bc.boundary_marker});
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary condition type '" +
        (type_param ? trim_copy(type_param->value) : std::string{}) +
        "' is not supported for the level-set transport module. Supported types: LevelSetInflow, LevelSetOutflow.");
  }
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_level_set_transport_from_input(const svmp::Physics::EquationModuleInput& input,
                                      svmp::FE::systems::FESystem& system)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Level-set transport module factory received null mesh.");
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int order = resolve_element_order(input, infer_polynomial_order(*input.mesh));
  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error("[svMultiPhysics::Physics] Level-set transport requires a mesh dimension in [1, 3].");
  }

  auto level_set_space = svmp::FE::spaces::SpaceFactory::create_h1(element_type, order);
  auto velocity_space = svmp::FE::spaces::SpaceFactory::create_vector_h1(element_type, order, dim);

  ls::LevelSetTransportOptions options{};
  options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);

  apply_level_set_params(input.equation_params, options);
  apply_level_set_params(input.default_domain.params, options);
  if (!input.domains.empty()) {
    if (input.domains.size() != 1u) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Multiple <Domain> blocks are not supported by the level-set transport module.");
    }
    apply_level_set_params(input.domains.front().params, options);
  }
  apply_level_set_bcs(input, options);

  options.velocity.space = velocity_space;

  auto module = std::make_unique<ls::LevelSetTransportModule>(std::move(level_set_space),
                                                              std::move(options));
  module->registerOn(system);
  return module;
}

} // namespace

SVMP_REGISTER_EQUATION("level_set", &create_level_set_transport_from_input);
SVMP_REGISTER_EQUATION("levelSet", &create_level_set_transport_from_input);
SVMP_REGISTER_EQUATION("level_set_transport", &create_level_set_transport_from_input);

namespace svmp::Physics::formulations::level_set {

void forceLink_LevelSetRegister() {}

} // namespace svmp::Physics::formulations::level_set
