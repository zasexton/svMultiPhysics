/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Options/InputReaders.h"

#include "FE/Core/Logger.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellTopology.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace svmp::Physics::options {

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

bool parse_bool_relaxed(std::string_view raw)
{
  const auto v = lower_copy(trim_copy(std::string(raw)));
  if (v == "true" || v == "1" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "false" || v == "0" || v == "no" || v == "off") {
    return false;
  }
  return false;
}

[[nodiscard]] bool navierStokesTraceEnabled() noexcept
{
  const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE");
  return env != nullptr && env[0] != '\0';
}

void navierStokesTraceLog(const std::string& message)
{
  if (navierStokesTraceEnabled()) {
    FE_LOG_INFO(message);
  }
}

[[nodiscard]] bool temporalSpatialBcTraceEnabled() noexcept
{
  return navierStokesTraceEnabled();
}

double parse_double(std::string_view raw, std::string_view context)
{
  const auto s = trim_copy(std::string(raw));
  try {
    size_t pos = 0;
    const double v = std::stod(s, &pos);
    if (pos != s.size()) {
      throw std::runtime_error("");
    }
    return v;
  } catch (...) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse numeric value '" + std::string(raw) +
                             "' for " + std::string(context) + ".");
  }
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

const svmp::Physics::ParameterValue* find_param(const svmp::Physics::ParameterMap& params,
                                                std::string_view key)
{
  const auto it = params.find(std::string(key));
  if (it == params.end()) {
    return nullptr;
  }
  return &it->second;
}

bool has_nonempty_defined(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p) {
    return false;
  }
  return p->defined && !trim_copy(p->value).empty();
}

std::optional<double> get_defined_double(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p || !p->defined) {
    return std::nullopt;
  }
  return parse_double(p->value, key);
}

std::optional<int> get_defined_int(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p || !p->defined) {
    return std::nullopt;
  }

  const auto s = trim_copy(p->value);
  try {
    size_t pos = 0;
    const int v = std::stoi(s, &pos);
    if (pos != s.size()) {
      throw std::runtime_error("");
    }
    return v;
  } catch (...) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse integer value '" +
                             p->value + "' for " + std::string(key) + ".");
  }
}

std::optional<bool> get_defined_bool(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p || !p->defined) {
    return std::nullopt;
  }
  return parse_bool_relaxed(p->value);
}

std::optional<std::string> get_defined_string(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p || !p->defined) {
    return std::nullopt;
  }
  auto value = trim_copy(p->value);
  if (value.empty()) {
    return std::nullopt;
  }
  return value;
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
        "[svMultiPhysics::Physics] Mixed-order meshes are not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto family = shapes.front().family;
  for (const auto& s : shapes) {
    if (s.family != family) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Mixed cell families are not supported by the new solver yet. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
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

  throw std::runtime_error(
      "[svMultiPhysics::Physics] Unsupported mesh cell family for new solver. "
      "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
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
          "[svMultiPhysics::Physics] Mixed polynomial orders are not supported by the new solver yet. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
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

const svmp::Physics::DomainInput& select_single_domain(const svmp::Physics::EquationModuleInput& input,
                                                       std::string_view module_name)
{
  if (!input.domains.empty()) {
    if (input.domains.size() != 1) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Multiple <Domain> blocks are not supported for the new solver " +
          std::string(module_name) +
          " module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
    return input.domains.front();
  }
  return input.default_domain;
}

std::vector<std::string> split_csv_line(std::string_view line)
{
  std::vector<std::string> fields;
  std::string field;
  for (const char ch : line) {
    if (ch == ',') {
      fields.push_back(trim_copy(field));
      field.clear();
      continue;
    }
    field.push_back(ch);
  }
  fields.push_back(trim_copy(field));
  return fields;
}

svmp::FE::GlobalIndex parse_global_index(std::string_view raw, std::string_view context)
{
  const auto s = trim_copy(std::string(raw));
  try {
    size_t pos = 0;
    const long long value = std::stoll(s, &pos);
    if (pos != s.size() || value < 0) {
      throw std::runtime_error("");
    }
    return static_cast<svmp::FE::GlobalIndex>(value);
  } catch (...) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse non-negative integer value '" +
                             std::string(raw) + "' for " + std::string(context) + ".");
  }
}

bool values_match(svmp::FE::Real a, svmp::FE::Real b)
{
  const double da = static_cast<double>(a);
  const double db = static_cast<double>(b);
  const double scale = 1.0 + std::max(std::abs(da), std::abs(db));
  return std::abs(da - db) <= 1e-12 * scale;
}

std::vector<int> parse_int_list(std::string_view raw)
{
  std::string normalized;
  normalized.reserve(raw.size());
  for (const char ch : raw) {
    const auto uch = static_cast<unsigned char>(ch);
    normalized.push_back(
        (std::isdigit(uch) || ch == '-' || ch == '+') ? ch : ' ');
  }

  std::istringstream iss{normalized};
  std::vector<int> out;
  int v = 0;
  while (iss >> v) {
    out.push_back(v);
  }
  return out;
}

std::array<bool, 3> active_components_from_flags(const std::array<int, 3>& active, int dim)
{
  std::array<bool, 3> out{true, true, true};
  for (int d = 0; d < 3; ++d) {
    out[static_cast<std::size_t>(d)] = (d < dim) ? (active[static_cast<std::size_t>(d)] != 0) : false;
  }
  return out;
}

svmp::FE::Real direction_component(const std::vector<int>& effective_dir, int component)
{
  if (component < 0) {
    return static_cast<svmp::FE::Real>(0.0);
  }
  const auto idx = static_cast<std::size_t>(component);
  if (idx >= effective_dir.size()) {
    return static_cast<svmp::FE::Real>(1.0);
  }
  return static_cast<svmp::FE::Real>(effective_dir[idx]);
}

std::array<svmp::FE::Real, 3> parse_real_vector3(std::string_view raw,
                                                 std::string_view context)
{
  auto s = trim_copy(std::string(raw));
  std::replace(s.begin(), s.end(), ',', ' ');
  std::istringstream iss{s};
  std::array<svmp::FE::Real, 3> out{0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < out.size(); ++i) {
    double value = 0.0;
    if (!(iss >> value)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Failed to parse three numeric components for " +
          std::string(context) + ".");
    }
    out[i] = static_cast<svmp::FE::Real>(value);
  }
  double extra = 0.0;
  if (iss >> extra) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Expected exactly three numeric components for " +
        std::string(context) + ".");
  }
  return out;
}

std::optional<double> first_defined_double(const svmp::Physics::ParameterMap& params,
                                           std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    if (const auto value = get_defined_double(params, key)) {
      return value;
    }
  }
  return std::nullopt;
}

std::optional<std::string> first_defined_string(const svmp::Physics::ParameterMap& params,
                                                std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    if (const auto value = get_defined_string(params, key)) {
      return value;
    }
  }
  return std::nullopt;
}

std::optional<bool> first_defined_bool(const svmp::Physics::ParameterMap& params,
                                       std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    if (const auto value = get_defined_bool(params, key)) {
      return value;
    }
  }
  return std::nullopt;
}

std::optional<int> first_defined_int(const svmp::Physics::ParameterMap& params,
                                     std::initializer_list<std::string_view> keys)
{
  for (const auto key : keys) {
    if (const auto value = get_defined_int(params, key)) {
      return value;
    }
  }
  return std::nullopt;
}

std::size_t defined_parameter_count(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys)
{
  return static_cast<std::size_t>(std::count_if(
      keys.begin(), keys.end(), [&](std::string_view key) {
        const auto* value = find_param(params, key);
        return value != nullptr && value->defined;
      }));
}

bool any_parameter_defined(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys)
{
  return defined_parameter_count(params, keys) != 0u;
}

} // namespace svmp::Physics::options
