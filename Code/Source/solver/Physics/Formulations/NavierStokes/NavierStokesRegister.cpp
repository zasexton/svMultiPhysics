#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidInterface.h"
#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesRegister.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Core/TemporalValues.h"
#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"

#include "FE/Core/Logger.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/Interfaces/MaterialInterfaceTransportVelocity.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellTopology.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace {

#if FE_HAS_MPI
using MarkerCommunicator = MPI_Comm;
#else
using MarkerCommunicator = int;
#endif

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

[[noreturn]] void reject_unsupported_free_surface_scope()
{
  throw std::runtime_error(
      "[svMultiPhysics::Physics] "
      "unsupported_two_phase_or_jump_free_surface_scope");
}

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

[[nodiscard]] bool is_canonical_free_surface_physical_model_key(
    std::string_view key)
{
  return key == "Free_surface_physical_model" ||
         key == "FreeSurfacePhysicalModel";
}

enum class NavierStokesPhysicalModel {
  OnePhaseLiquidPrescribedExteriorPressure,
  IncompressibleTwoFluid,
};

[[nodiscard]] bool is_two_fluid_configuration_key(
    std::string_view key)
{
  static constexpr std::array<std::string_view, 22> keys = {
      "Material_interface_marker",
      "MaterialInterfaceMarker",
      "Negative_phase_density",
      "NegativePhaseDensity",
      "Negative_phase_dynamic_viscosity",
      "NegativePhaseDynamicViscosity",
      "Positive_phase_density",
      "PositivePhaseDensity",
      "Positive_phase_dynamic_viscosity",
      "PositivePhaseDynamicViscosity",
      "Two_fluid_surface_tension",
      "TwoFluidSurfaceTension",
      "Two_fluid_interface_nitsche_gamma",
      "TwoFluidInterfaceNitscheGamma",
      "Prescribed_pressure_jump",
      "PrescribedPressureJump",
      "Prescribed_viscous_traction_jump_x",
      "PrescribedViscousTractionJumpX",
      "Prescribed_viscous_traction_jump_y",
      "PrescribedViscousTractionJumpY",
      "Prescribed_viscous_traction_jump_z",
      "PrescribedViscousTractionJumpZ",
  };
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

enum class FreeSurfaceScopeLocation {
  Equation,
  Domain,
  Boundary,
};

[[nodiscard]] bool is_free_surface_implementation_selector(
    std::string_view normalized_key)
{
  return normalized_key == "implementation" ||
         normalized_key == "freesurfaceimplementation";
}

[[nodiscard]] bool is_canonical_free_surface_implementation_key(
    std::string_view key)
{
  return key == "Implementation" ||
         key == "Free_surface_implementation" ||
         key == "FreeSurfaceImplementation";
}

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
           "Free_surface_implementation",
           "FreeSurfaceImplementation",
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

struct FreeSurfaceSchemaContract {
  int version{
      svmp::Physics::formulations::navier_stokes::
          IncompressibleNavierStokesVMSOptions::
              current_configuration_schema_version};
  bool explicit_legacy{false};
};

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

std::array<svmp::FE::Real, 3> parse_real_vector3(std::string_view raw,
                                                 std::string_view context);

svmp::FE::forms::GeometryTangentPath parse_geometry_tangent_path(std::string_view raw,
                                                                 std::string_view context)
{
  const auto path = lower_copy(trim_copy(std::string(raw)));
  if (path == "symbolic" || path == "symbolic_required" ||
      path == "symbolicrequired" || path == "required") {
    return svmp::FE::forms::GeometryTangentPath::SymbolicRequired;
  }
  if (path == "ad" || path == "ad_reference" || path == "adreference" ||
      path == "reference_ad" || path == "referencead") {
    return svmp::FE::forms::GeometryTangentPath::ADReference;
  }
  if (path == "symbolic_ad_check" || path == "symbolic_with_ad_check" ||
      path == "symbolicadcheck" || path == "symbolicwithadcheck" ||
      path == "check" || path == "parity_check" || path == "paritycheck") {
    return svmp::FE::forms::GeometryTangentPath::SymbolicWithADCheck;
  }
  if (path == "auto") {
    return svmp::FE::forms::GeometryTangentPath::Auto;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of 'symbolic', 'ad', 'symbolic_ad_check', or 'auto'.");
}

struct TemporalSpatialValues {
  struct Key {
    std::int64_t x{0};
    std::int64_t y{0};
    std::int64_t z{0};

    friend bool operator==(const Key& a, const Key& b) noexcept
    {
      return a.x == b.x && a.y == b.y && a.z == b.z;
    }
  };

  struct KeyHash {
    size_t operator()(const Key& k) const noexcept
    {
      size_t h = 1469598103934665603ull;
      auto mix = [&](std::int64_t v) {
        const size_t x = std::hash<std::int64_t>{}(v);
        h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      };
      mix(k.x);
      mix(k.y);
      mix(k.z);
      return h;
    }
  };

  int dim{0};
  int dof{0};
  int num_time_points{0};
  int boundary_marker{0};
  std::string file_path{};

  std::vector<double> t{};
  double period{0.0};

  std::vector<svmp::index_t> node_ids{};
  std::vector<std::array<svmp::FE::Real, 3>> coords{};

  // Data layout: d[((node * num_time_points) + it) * dof + comp]
  std::vector<svmp::FE::Real> d{};

  std::unordered_map<Key, std::size_t, KeyHash> node_index_by_key{};
  bool has_x_only_interpolant{false};
  std::vector<svmp::FE::Real> x_interpolation_coords{};
  std::vector<std::size_t> x_interpolation_nodes{};

  [[nodiscard]] static Key quantize(const std::array<svmp::FE::Real, 3>& p, int dim_in) noexcept
  {
    constexpr double scale = 1e12;
    auto q = [&](svmp::FE::Real v) { return static_cast<std::int64_t>(std::llround(static_cast<double>(v) * scale)); };
    Key k{};
    k.x = q(p[0]);
    k.y = (dim_in >= 2) ? q(p[1]) : 0;
    k.z = (dim_in >= 3) ? q(p[2]) : 0;
    return k;
  }

  [[nodiscard]] svmp::FE::Real sample(std::size_t node_idx, int time_idx, int comp) const
  {
    const auto idx = ((node_idx * static_cast<std::size_t>(num_time_points) + static_cast<std::size_t>(time_idx)) *
                          static_cast<std::size_t>(dof) +
                      static_cast<std::size_t>(comp));
    if (idx >= d.size()) {
      throw std::runtime_error("[svMultiPhysics::Physics] Internal error: temporal/spatial BC index out of range.");
    }
    return d[idx];
  }

  [[nodiscard]] static bool closeValues(svmp::FE::Real a, svmp::FE::Real b) noexcept
  {
    const double da = static_cast<double>(a);
    const double db = static_cast<double>(b);
    const double scale = std::max({1.0, std::abs(da), std::abs(db)});
    return std::abs(da - db) <= 1.0e-10 * scale;
  }

  void buildInterpolationMetadata()
  {
    has_x_only_interpolant = false;
    x_interpolation_coords.clear();
    x_interpolation_nodes.clear();
    if (coords.size() < 2 || num_time_points <= 0 || dof <= 0) {
      return;
    }

    std::vector<std::size_t> order(coords.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    std::sort(order.begin(), order.end(), [this](std::size_t a, std::size_t b) {
      return coords[a][0] < coords[b][0];
    });

    constexpr double coord_tol = 1.0e-12;
    std::size_t group_begin = 0;
    bool values_depend_only_on_x = true;
    while (group_begin < order.size()) {
      const auto representative = order[group_begin];
      const double x0 = static_cast<double>(coords[representative][0]);
      std::size_t group_end = group_begin + 1;
      while (group_end < order.size() &&
             std::abs(static_cast<double>(coords[order[group_end]][0]) - x0) <= coord_tol) {
        ++group_end;
      }

      for (std::size_t j = group_begin + 1; j < group_end && values_depend_only_on_x; ++j) {
        const auto node = order[j];
        for (int it = 0; it < num_time_points && values_depend_only_on_x; ++it) {
          for (int comp = 0; comp < dof; ++comp) {
            if (!closeValues(sample(representative, it, comp), sample(node, it, comp))) {
              values_depend_only_on_x = false;
              break;
            }
          }
        }
      }
      if (!values_depend_only_on_x) {
        break;
      }

      x_interpolation_coords.push_back(coords[representative][0]);
      x_interpolation_nodes.push_back(representative);
      group_begin = group_end;
    }

    has_x_only_interpolant =
        values_depend_only_on_x && x_interpolation_coords.size() >= 2 &&
        std::abs(static_cast<double>(x_interpolation_coords.back() - x_interpolation_coords.front())) > 1.0e-14;
    if (!has_x_only_interpolant) {
      x_interpolation_coords.clear();
      x_interpolation_nodes.clear();
    }
  }

  [[nodiscard]] std::size_t findNodeIndex(const std::array<svmp::FE::Real, 3>& p) const
  {
    const auto key = quantize(p, dim);
    if (const auto it = node_index_by_key.find(key); it != node_index_by_key.end()) {
      return it->second;
    }

    // Fallback: nearest-node match (robust to minor floating differences).
    std::size_t best = coords.size();
    double best_d2 = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < coords.size(); ++i) {
      const auto& c = coords[i];
      const double dx = static_cast<double>(p[0] - c[0]);
      const double dy = static_cast<double>(p[1] - c[1]);
      const double dz = static_cast<double>(p[2] - c[2]);
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = i;
      }
    }

    constexpr double tol = 1e-8;
    if (best < coords.size() && best_d2 <= tol * tol) {
      return best;
    }

    if (temporalSpatialBcTraceEnabled()) {
      int rank = 0;
#if FE_HAS_MPI
      int initialized = 0;
      MPI_Initialized(&initialized);
      if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      }
#endif
      std::ostringstream oss;
      oss << "TemporalSpatialValues: rank=" << rank
          << " marker=" << boundary_marker
          << " file='" << file_path << "'"
          << " failed coordinate match"
          << " query=(" << p[0] << "," << p[1] << "," << p[2] << ")"
          << " stored_nodes=" << coords.size();
      if (best < coords.size()) {
        const auto& c = coords[best];
        oss << " nearest=(" << c[0] << "," << c[1] << "," << c[2] << ")"
            << " nearest_d=" << std::sqrt(best_d2);
      }
      navierStokesTraceLog(oss.str());
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Failed to match a temporal/spatial BC value to a boundary node by coordinate.");
  }

  [[nodiscard]] double wrapTime(double time) const noexcept
  {
    if (!(period > 0.0) || !std::isfinite(period) || num_time_points < 2) {
      return time;
    }
    double tmod = std::fmod(time, period);
    if (tmod < 0.0) {
      tmod += period;
    }
    return tmod;
  }

  [[nodiscard]] svmp::FE::Real interpolate(std::size_t node_idx, svmp::FE::Real time, int comp) const
  {
    if (num_time_points <= 0) {
      return svmp::FE::Real{0.0};
    }
    if (num_time_points == 1) {
      return sample(node_idx, 0, comp);
    }

    const double tt = wrapTime(static_cast<double>(time));

    int i0 = 0;
    for (int i = 0; i < num_time_points - 1; ++i) {
      if (t[static_cast<std::size_t>(i + 1)] >= tt) {
        i0 = i;
        break;
      }
    }

    const double t0 = t[static_cast<std::size_t>(i0)];
    const double t1 = t[static_cast<std::size_t>(i0 + 1)];
    const double dt = t1 - t0;
    const double alpha = (dt > 0.0) ? ((tt - t0) / dt) : 0.0;

    const auto v0 = static_cast<double>(sample(node_idx, i0, comp));
    const auto v1 = static_cast<double>(sample(node_idx, i0 + 1, comp));
    return static_cast<svmp::FE::Real>((1.0 - alpha) * v0 + alpha * v1);
  }

  [[nodiscard]] svmp::FE::Real interpolateAlongX(svmp::FE::Real x, svmp::FE::Real time, int comp) const
  {
    if (!has_x_only_interpolant || x_interpolation_coords.empty()) {
      throw std::runtime_error("[svMultiPhysics::Physics] Internal error: missing x-axis temporal/spatial interpolant.");
    }

    if (x <= x_interpolation_coords.front()) {
      return interpolate(x_interpolation_nodes.front(), time, comp);
    }
    if (x >= x_interpolation_coords.back()) {
      return interpolate(x_interpolation_nodes.back(), time, comp);
    }

    const auto upper = std::upper_bound(x_interpolation_coords.begin(), x_interpolation_coords.end(), x);
    const auto hi = static_cast<std::size_t>(std::distance(x_interpolation_coords.begin(), upper));
    const auto lo = hi - 1;
    const double x0 = static_cast<double>(x_interpolation_coords[lo]);
    const double x1 = static_cast<double>(x_interpolation_coords[hi]);
    const double denom = x1 - x0;
    const double alpha = (denom > 0.0) ? ((static_cast<double>(x) - x0) / denom) : 0.0;
    const double v0 = static_cast<double>(interpolate(x_interpolation_nodes[lo], time, comp));
    const double v1 = static_cast<double>(interpolate(x_interpolation_nodes[hi], time, comp));
    return static_cast<svmp::FE::Real>((1.0 - alpha) * v0 + alpha * v1);
  }

  [[nodiscard]] svmp::FE::Real interpolateNearestSpatial(const std::array<svmp::FE::Real, 3>& p,
                                                         svmp::FE::Real time,
                                                         int comp) const
  {
    if (coords.empty()) {
      return svmp::FE::Real{0.0};
    }

    std::vector<std::pair<double, std::size_t>> distances;
    distances.reserve(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i) {
      const auto& c = coords[i];
      const double dx = static_cast<double>(p[0] - c[0]);
      const double dy = static_cast<double>(p[1] - c[1]);
      const double dz = static_cast<double>(p[2] - c[2]);
      distances.emplace_back(dx * dx + dy * dy + dz * dz, i);
    }
    const auto k = std::min<std::size_t>(8, distances.size());
    std::partial_sort(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(k), distances.end());

    constexpr double exact_tol2 = 1.0e-16;
    if (distances.front().first <= exact_tol2) {
      return interpolate(distances.front().second, time, comp);
    }

    double weighted_sum = 0.0;
    double weight_total = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      const double w = 1.0 / std::max(distances[i].first, 1.0e-24);
      weighted_sum += w * static_cast<double>(interpolate(distances[i].second, time, comp));
      weight_total += w;
    }
    return static_cast<svmp::FE::Real>(weight_total > 0.0 ? weighted_sum / weight_total : 0.0);
  }

  [[nodiscard]] svmp::FE::Real interpolateSpatial(const std::array<svmp::FE::Real, 3>& p,
                                                  svmp::FE::Real time,
                                                  int comp) const
  {
    const auto key = quantize(p, dim);
    if (const auto it = node_index_by_key.find(key); it != node_index_by_key.end()) {
      return interpolate(it->second, time, comp);
    }
    if (has_x_only_interpolant) {
      return interpolateAlongX(p[0], time, comp);
    }
    return interpolateNearestSpatial(p, time, comp);
  }
};

std::unordered_set<svmp::gid_t> collect_temporal_spatial_target_vertex_gids(
    const svmp::MeshBase& mesh,
    int boundary_marker)
{
  std::unordered_set<svmp::gid_t> gids;
  const auto& vgids = mesh.vertex_gids();
  if (boundary_marker < 0) {
    gids.insert(vgids.begin(), vgids.end());
    return gids;
  }
  const auto add_vertex = [&](svmp::index_t v) {
    if (v < 0) {
      return;
    }
    const auto idx = static_cast<std::size_t>(v);
    if (idx >= vgids.size()) {
      return;
    }
    gids.insert(vgids[idx]);
  };
  const auto sorted_unique = [](std::vector<svmp::index_t> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
  };

  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
  for (const auto f : faces) {
    const auto stored_vertices = mesh.face_vertices(f);
    for (const auto v : stored_vertices) {
      add_vertex(v);
    }

    auto stored_key = sorted_unique(stored_vertices);
    if (stored_key.empty() || static_cast<std::size_t>(f) >= mesh.face2cell().size()) {
      continue;
    }

    const auto& adjacent = mesh.face2cell()[static_cast<std::size_t>(f)];
    for (const auto cell : adjacent) {
      if (cell == svmp::INVALID_INDEX || static_cast<std::size_t>(cell) >= mesh.n_cells()) {
        continue;
      }
      const auto& shape = mesh.cell_shape(cell);
      const auto face_view = svmp::CellTopology::get_oriented_boundary_faces_view(shape.family);
      if (!face_view.indices || !face_view.offsets || face_view.face_count <= 0) {
        continue;
      }

      for (int lf = 0; lf < face_view.face_count; ++lf) {
        std::vector<svmp::index_t> geometry_vertices;
        try {
          geometry_vertices = mesh.cell_face_geometry_dofs(cell, lf);
        } catch (const std::exception&) {
          geometry_vertices.clear();
        }
        if (geometry_vertices.empty()) {
          const auto [cell_vertices, n_cell_vertices] = mesh.cell_vertices_span(cell);
          const int begin = face_view.offsets[lf];
          const int end = face_view.offsets[lf + 1];
          for (int j = begin; j < end; ++j) {
            const auto local = face_view.indices[j];
            if (local >= 0 && static_cast<std::size_t>(local) < n_cell_vertices) {
              geometry_vertices.push_back(cell_vertices[local]);
            }
          }
        }

        auto geometry_key = sorted_unique(geometry_vertices);
        if (!std::includes(geometry_key.begin(), geometry_key.end(),
                           stored_key.begin(), stored_key.end())) {
          continue;
        }
        for (const auto v : geometry_vertices) {
          add_vertex(v);
        }
        break;
      }
    }
  }
  return gids;
}

std::shared_ptr<TemporalSpatialValues> read_temporal_and_spatial_values_file(const svmp::MeshBase& mesh,
                                                                            int boundary_marker,
                                                                            const std::string& file_path,
                                                                            MarkerCommunicator comm)
{
  std::ifstream in(file_path);
  if (!in.is_open()) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to open temporal/spatial BC file '" + file_path + "'.");
  }

  int ndof = 0;
  int num_ts = 0;
  int num_nodes = 0;
  in >> ndof >> num_ts >> num_nodes;
  if (ndof <= 0 || num_ts <= 0 || num_nodes <= 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Invalid header in temporal/spatial BC file '" + file_path +
        "' (expected: <ndof> <num_ts> <num_nodes>).");
  }

  const int dim = mesh.dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error("[svMultiPhysics::Physics] Invalid mesh dimension for temporal/spatial BC parsing.");
  }
  if (ndof > dim) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "' specifies ndof=" + std::to_string(ndof) +
        ", but mesh dimension is " + std::to_string(dim) + ".");
  }

  const auto target_gids =
      collect_temporal_spatial_target_vertex_gids(mesh, boundary_marker);
  if (mesh.vertex_gids().size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
        "' requires stable global vertex IDs on the boundary mesh.");
  }

  auto out = std::make_shared<TemporalSpatialValues>();
  out->dim = dim;
  out->dof = ndof;
  out->num_time_points = num_ts;
  out->boundary_marker = boundary_marker;
  out->file_path = file_path;
  out->t.resize(static_cast<std::size_t>(num_ts));

  // Time sequence (t0 must be 0 and increasing).
  for (int i = 0; i < num_ts; ++i) {
    double ti = 0.0;
    if (!(in >> ti) || !std::isfinite(ti)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "' contains a missing or nonfinite time value at index " +
          std::to_string(i) + ".");
    }
    out->t[static_cast<std::size_t>(i)] = ti;
    if (i == 0) {
      if (std::abs(ti) > 1e-14) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "': first time value must be 0.");
      }
    } else {
      const double dt = ti - out->t[static_cast<std::size_t>(i - 1)];
      if (!(dt > 0.0)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path + "': time values must be increasing.");
      }
    }
  }
  out->period = out->t.back();

  struct FileNodeRecord {
    long long file_node_id{0};
    std::vector<svmp::FE::Real> values{};
  };
  std::vector<FileNodeRecord> file_nodes;
  file_nodes.reserve(static_cast<std::size_t>(num_nodes));
  std::unordered_set<long long> seen_file_node_ids;
  for (int b = 0; b < num_nodes; ++b) {
    FileNodeRecord record;
    if (!(in >> record.file_node_id)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "' ended before node record " + std::to_string(b + 1) + ".");
    }
    if (record.file_node_id <= 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "' contains a nonpositive node id: " + std::to_string(record.file_node_id) + ".");
    }
    if (!seen_file_node_ids.insert(record.file_node_id).second) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
          "' contains duplicate node id " + std::to_string(record.file_node_id) + ".");
    }

    record.values.reserve(static_cast<std::size_t>(num_ts) * static_cast<std::size_t>(ndof));
    for (int i = 0; i < num_ts; ++i) {
      for (int k = 0; k < ndof; ++k) {
        double value = 0.0;
        if (!(in >> value) || !std::isfinite(value)) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
              "' contains a missing or nonfinite value for node id " +
              std::to_string(record.file_node_id) + ".");
        }
        record.values.push_back(static_cast<svmp::FE::Real>(value));
      }
    }
    file_nodes.push_back(std::move(record));
  }
  std::string trailing_token;
  if (in >> trailing_token) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal/spatial BC file '" + file_path +
        "' contains trailing data after " + std::to_string(num_nodes) +
        " node records (first unexpected token: '" + trailing_token + "').");
  }

  enum class NodeIdConvention {
    LegacyOneBasedOrdinal,
    ImportedGlobalId,
  };
  struct NodeIdCoverage {
    NodeIdConvention convention{NodeIdConvention::LegacyOneBasedOrdinal};
    std::vector<svmp::gid_t> mapped_gids{};
    std::size_t globally_matched_nodes{0};
    bool covers_global_boundary{false};
    bool exact{false};
  };

  const auto evaluate_convention = [&](NodeIdConvention convention) {
    NodeIdCoverage coverage;
    coverage.convention = convention;
    coverage.mapped_gids.reserve(file_nodes.size());
    std::unordered_set<svmp::gid_t> mapped_gid_set;
    mapped_gid_set.reserve(file_nodes.size());
    for (const auto& record : file_nodes) {
      const auto mapped =
          convention == NodeIdConvention::ImportedGlobalId
              ? static_cast<svmp::gid_t>(record.file_node_id)
              : static_cast<svmp::gid_t>(record.file_node_id - 1);
      coverage.mapped_gids.push_back(mapped);
      mapped_gid_set.insert(mapped);
    }

    std::vector<int> globally_matched(file_nodes.size(), 0);
    for (std::size_t i = 0; i < coverage.mapped_gids.size(); ++i) {
      globally_matched[i] = target_gids.count(coverage.mapped_gids[i]) != 0u ? 1 : 0;
    }
    int covers_global_boundary = std::all_of(
                                      target_gids.begin(),
                                      target_gids.end(),
                                      [&](svmp::gid_t gid) {
                                        return mapped_gid_set.count(gid) != 0u;
                                      })
                                      ? 1
                                      : 0;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
      MPI_Finalized(&mpi_finalized);
    }
    if (mpi_initialized && !mpi_finalized && comm != MPI_COMM_NULL) {
      MPI_Allreduce(
          MPI_IN_PLACE,
          globally_matched.data(),
          static_cast<int>(globally_matched.size()),
          MPI_INT,
          MPI_MAX,
          comm);
      MPI_Allreduce(
          MPI_IN_PLACE,
          &covers_global_boundary,
          1,
          MPI_INT,
          MPI_MIN,
          comm);
    }
#endif
    coverage.globally_matched_nodes = static_cast<std::size_t>(
        std::count(globally_matched.begin(), globally_matched.end(), 1));
    coverage.covers_global_boundary = covers_global_boundary != 0;
    coverage.exact =
        coverage.covers_global_boundary && coverage.globally_matched_nodes == file_nodes.size();
    return coverage;
  };

  const auto legacy_coverage = evaluate_convention(NodeIdConvention::LegacyOneBasedOrdinal);
  const auto imported_coverage = evaluate_convention(NodeIdConvention::ImportedGlobalId);
  if (legacy_coverage.exact == imported_coverage.exact) {
    std::ostringstream message;
    message << "[svMultiPhysics::Physics] Temporal/spatial BC file '" << file_path << "' node IDs ";
    if (legacy_coverage.exact) {
      message << "are ambiguous";
    } else {
      message << "do not exactly cover ";
      if (boundary_marker < 0) {
        message << "the mesh vertex set";
      } else {
        message << "boundary marker " << boundary_marker;
      }
    }
    message << " under either the legacy one-based ordinal convention or the imported global-ID convention"
            << " (file_nodes=" << file_nodes.size()
            << ", legacy_matched=" << legacy_coverage.globally_matched_nodes
            << ", legacy_covers_boundary=" << (legacy_coverage.covers_global_boundary ? 1 : 0)
            << ", imported_matched=" << imported_coverage.globally_matched_nodes
            << ", imported_covers_boundary=" << (imported_coverage.covers_global_boundary ? 1 : 0) << ").";
    throw std::runtime_error(message.str());
  }
  const auto& selected_coverage = imported_coverage.exact ? imported_coverage : legacy_coverage;

  out->node_ids.clear();
  out->coords.clear();
  out->d.clear();
  out->node_ids.reserve(static_cast<std::size_t>(num_nodes));
  out->coords.reserve(static_cast<std::size_t>(num_nodes));
  out->d.reserve(static_cast<std::size_t>(num_nodes) * static_cast<std::size_t>(num_ts) * static_cast<std::size_t>(ndof));

  int missing_local_vertex_count = 0;
  int non_boundary_file_node_count = 0;
  for (std::size_t b = 0; b < file_nodes.size(); ++b) {
    const auto node_gid = selected_coverage.mapped_gids[b];
    const auto node_idx = mesh.global_to_local_vertex(node_gid);

    const bool has_local_vertex = node_idx != svmp::INVALID_INDEX;
    const bool is_boundary_node = target_gids.count(node_gid) != 0u;
    if (!has_local_vertex) {
      ++missing_local_vertex_count;
    } else if (!is_boundary_node) {
      ++non_boundary_file_node_count;
    }
    const bool keep = has_local_vertex && is_boundary_node;
    if (keep) {
      out->node_ids.push_back(node_idx);

      const auto& X = mesh.X_ref();
      std::array<svmp::FE::Real, 3> p{0.0, 0.0, 0.0};
      const auto base = static_cast<std::size_t>(node_idx) * static_cast<std::size_t>(dim);
      p[0] = static_cast<svmp::FE::Real>(X.at(base + 0));
      if (dim >= 2) {
        p[1] = static_cast<svmp::FE::Real>(X.at(base + 1));
      }
      if (dim >= 3) {
        p[2] = static_cast<svmp::FE::Real>(X.at(base + 2));
      }
      out->coords.push_back(p);

      const auto stored_idx = out->coords.size() - 1;
      out->node_index_by_key.emplace(TemporalSpatialValues::quantize(p, dim), stored_idx);
    }

    if (keep) {
      out->d.insert(out->d.end(), file_nodes[b].values.begin(), file_nodes[b].values.end());
    }
  }

  if (temporalSpatialBcTraceEnabled()) {
    int rank = 0;
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    int finalized = 0;
    if (initialized) {
      MPI_Finalized(&finalized);
    }
    if (initialized && !finalized && comm != MPI_COMM_NULL) {
      MPI_Comm_rank(comm, &rank);
    }
#endif
    std::ostringstream oss;
    oss << "TemporalSpatialValues: rank=" << rank
        << " marker=" << boundary_marker
        << " file='" << file_path << "'"
        << " ndof=" << ndof
        << " time_points=" << num_ts
        << " file_nodes=" << num_nodes
        << " kept_nodes=" << out->coords.size()
        << " mapping_scope="
        << (boundary_marker < 0 ? "mesh_vertices" : "boundary_marker")
        << " target_nodes=" << target_gids.size()
        << " missing_local_vertex_nodes=" << missing_local_vertex_count
        << " non_boundary_file_nodes=" << non_boundary_file_node_count
        << " node_id_convention="
        << (selected_coverage.convention == NodeIdConvention::ImportedGlobalId ? "imported_global_id"
                                                                               : "legacy_one_based_ordinal")
        << " legacy_matched_nodes=" << legacy_coverage.globally_matched_nodes
        << " imported_matched_nodes=" << imported_coverage.globally_matched_nodes;
    navierStokesTraceLog(oss.str());
  }

  out->buildInterpolationMetadata();

  return out;
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

svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType
parse_node_pressure_constraint_id_type(std::string_view raw)
{
  using Options = svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  const auto value = trim_copy(std::string(raw));
  if (value == "Global_vertex_gid") {
    return Options::NodePressureConstraintIdType::GlobalVertexGid;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] Unsupported <Node_pressure_constraints><Id_type> '" + value +
      "'. Supported value: Global_vertex_gid.");
}

std::vector<svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::NodePressureConstraint>
read_node_pressure_constraints_csv(const std::string& path)
{
  using Options = svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to open node pressure constraints CSV file '" +
                             path + "'.");
  }

  std::vector<Options::NodePressureConstraint> out;
  std::unordered_map<svmp::FE::GlobalIndex, svmp::FE::Real> seen;

  bool header_seen = false;
  bool data_seen = false;
  int node_col = 0;
  int pressure_col = 1;

  std::string line;
  int line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    const auto trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }

    const auto fields = split_csv_line(trimmed);
    if (fields.size() < 2u) {
      throw std::runtime_error("[svMultiPhysics::Physics] Malformed node pressure CSV row in '" + path +
                               "' at line " + std::to_string(line_number) +
                               ": expected at least two comma-separated fields.");
    }

    if (!header_seen) {
      const auto c0 = lower_copy(fields[0]);
      const auto c1 = lower_copy(fields[1]);
      if (c0 == "node_id" || c0 == "pressure" || c1 == "node_id" || c1 == "pressure") {
        if (c0 == "node_id") {
          node_col = 0;
        } else if (c1 == "node_id") {
          node_col = 1;
        } else {
          throw std::runtime_error("[svMultiPhysics::Physics] Node pressure CSV header in '" + path +
                                   "' must contain a node_id column.");
        }

        if (c0 == "pressure") {
          pressure_col = 0;
        } else if (c1 == "pressure") {
          pressure_col = 1;
        } else {
          throw std::runtime_error("[svMultiPhysics::Physics] Node pressure CSV header in '" + path +
                                   "' must contain a pressure column.");
        }

        header_seen = true;
        continue;
      }
      header_seen = true;
    }

    const auto max_col = static_cast<std::size_t>(std::max(node_col, pressure_col));
    if (fields.size() <= max_col) {
      throw std::runtime_error("[svMultiPhysics::Physics] Malformed node pressure CSV row in '" + path +
                               "' at line " + std::to_string(line_number) +
                               ": missing node_id or pressure field.");
    }

    const auto context = std::string("node pressure CSV '") + path + "' line " + std::to_string(line_number);
    const auto node_id = parse_global_index(fields[static_cast<std::size_t>(node_col)], context + " node_id");
    const auto pressure_value = parse_double(fields[static_cast<std::size_t>(pressure_col)], context + " pressure");
    if (!std::isfinite(pressure_value)) {
      throw std::runtime_error("[svMultiPhysics::Physics] Non-finite pressure value in '" + path +
                               "' at line " + std::to_string(line_number) + ".");
    }
    const auto pressure = static_cast<svmp::FE::Real>(pressure_value);

    const auto it = seen.find(node_id);
    if (it != seen.end()) {
      if (!values_match(it->second, pressure)) {
        throw std::runtime_error("[svMultiPhysics::Physics] Conflicting duplicate node pressure value for node_id " +
                                 std::to_string(node_id) + " in '" + path + "'.");
      }
      continue;
    }

    seen.emplace(node_id, pressure);
    out.push_back(Options::NodePressureConstraint{node_id, pressure});
    data_seen = true;
  }

  if (!data_seen) {
    throw std::runtime_error("[svMultiPhysics::Physics] Node pressure constraints CSV file '" + path +
                             "' did not contain any node pressure values.");
  }

  return out;
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

void apply_fluid_momentum_source_params(
    const svmp::Physics::ParameterMap& params,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  for (const auto key : {"Momentum_source_field_name", "MomentumSourceFieldName",
                         "Body_force_field_name", "BodyForceFieldName",
                         "Body_force_field", "BodyForceField"}) {
    if (const auto field_name = get_defined_string(params, key)) {
      options.body_force_field_name = *field_name;
      break;
    }
  }
  for (const auto key : {"Auto_register_momentum_source_field", "AutoRegisterMomentumSourceField",
                         "Auto_register_body_force_field", "AutoRegisterBodyForceField"}) {
    if (const auto auto_register = get_defined_bool(params, key)) {
      options.auto_register_body_force_field = *auto_register;
      break;
    }
  }
}

void apply_fluid_properties(const svmp::Physics::DomainInput& domain,
                            svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  if (const auto rho = get_defined_double(domain.params, "Density")) {
    options.density = static_cast<svmp::FE::Real>(*rho);
  } else if (const auto rho2 = get_defined_double(domain.params, "Fluid_density")) {
    options.density = static_cast<svmp::FE::Real>(*rho2);
  }

  if (const auto fx = get_defined_double(domain.params, "Force_x")) {
    options.body_force[0] = static_cast<svmp::FE::Real>(*fx);
  }
  if (const auto fy = get_defined_double(domain.params, "Force_y")) {
    options.body_force[1] = static_cast<svmp::FE::Real>(*fy);
  }
  if (const auto fz = get_defined_double(domain.params, "Force_z")) {
    options.body_force[2] = static_cast<svmp::FE::Real>(*fz);
  }
  apply_fluid_momentum_source_params(domain.params, options);

  if (const auto enabled = get_defined_bool(domain.params, "Hydrostatic_pressure_initialization")) {
    options.hydrostatic_pressure_initialization.enabled = *enabled;
  }
  if (const auto reference = get_defined_double(domain.params, "Hydrostatic_pressure_reference")) {
    options.hydrostatic_pressure_initialization.reference_pressure =
        static_cast<svmp::FE::Real>(*reference);
  }
  if (const auto reference_point =
          get_defined_string(domain.params, "Hydrostatic_pressure_reference_point")) {
    options.hydrostatic_pressure_initialization.reference_point =
        parse_real_vector3(*reference_point, "Hydrostatic_pressure_reference_point");
  }
  if (const auto field_name = get_defined_string(domain.params, "Hydrostatic_pressure_field_name")) {
    options.hydrostatic_pressure_initialization.field_name = *field_name;
  }

  const auto* model_param = find_param(domain.params, "Viscosity.model");
  if (!model_param || !model_param->defined || trim_copy(model_param->value).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] <Viscosity model=\"...\"> is required for the new solver Navier-Stokes module. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto model_raw = trim_copy(model_param->value);
  const auto model = lower_copy(model_raw);

  if (model == "constant") {
    const auto* mu_param = find_param(domain.params, "Viscosity.Value");
    const double mu = mu_param ? parse_double(mu_param->value, "Viscosity/Value") : 0.0;
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
    const auto* p_mu_inf = find_param(domain.params, "Viscosity.Limiting_high_shear_rate_viscosity");
    const auto* p_mu0 = find_param(domain.params, "Viscosity.Limiting_low_shear_rate_viscosity");
    const auto* p_lambda = find_param(domain.params, "Viscosity.Shear_rate_tensor_multiplier");
    const auto* p_n = find_param(domain.params, "Viscosity.Power_law_index");
    const auto* p_a = find_param(domain.params, "Viscosity.Shear_rate_tensor_exponent");

    const auto mu_inf = static_cast<svmp::FE::Real>(
        p_mu_inf ? parse_double(p_mu_inf->value, "Viscosity/Limiting_high_shear_rate_viscosity") : 0.0);
    const auto mu0 = static_cast<svmp::FE::Real>(
        p_mu0 ? parse_double(p_mu0->value, "Viscosity/Limiting_low_shear_rate_viscosity") : 0.0);
    const auto lambda = static_cast<svmp::FE::Real>(
        p_lambda ? parse_double(p_lambda->value, "Viscosity/Shear_rate_tensor_multiplier") : 0.0);
    const auto n =
        static_cast<svmp::FE::Real>(p_n ? parse_double(p_n->value, "Viscosity/Power_law_index") : 0.0);
    const auto a = static_cast<svmp::FE::Real>(
        p_a ? parse_double(p_a->value, "Viscosity/Shear_rate_tensor_exponent") : 0.0);

    try {
      options.viscosity_model = std::make_shared<svmp::Physics::materials::fluid::CarreauYasudaViscosity>(
          mu0, mu_inf, lambda, n, a);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string("[svMultiPhysics::Physics] Invalid Carreau-Yasuda viscosity parameters for the new solver "
                      "Navier-Stokes module: ") +
          e.what() + ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (p_mu0 && p_mu0->defined) {
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

void apply_fluid_moving_domain_params(
    const svmp::Physics::ParameterMap& params,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  namespace ns = svmp::Physics::formulations::navier_stokes;

  constexpr std::array<std::string_view, 5> kAleKeys = {
      "ALE",
      "Enable_ALE",
      "Use_ALE",
      "Moving_mesh",
      "Use_moving_mesh",
  };
  for (const auto key : kAleKeys) {
    if (const auto value = get_defined_bool(params, key)) {
      options.enable_ale = *value;
    }
  }

  constexpr std::array<std::string_view, 3> kMovingVolumeKeys = {
      "Moving_control_volume_transient",
      "Include_moving_control_volume_transient",
      "ALE_moving_control_volume_transient",
  };
  for (const auto key : kMovingVolumeKeys) {
    if (const auto value = get_defined_bool(params, key)) {
      options.include_moving_control_volume_transient = *value;
    }
  }

  constexpr std::array<std::string_view, 3> kMeshVelocityFieldKeys = {
      "Mesh_velocity_field",
      "MeshVelocityField",
      "Mesh_motion_velocity_field",
  };
  for (const auto key : kMeshVelocityFieldKeys) {
    if (const auto value = get_defined_string(params, key)) {
      options.mesh_velocity_field_name = *value;
    }
  }

  constexpr std::array<std::string_view, 4> kMeshVelocitySourceKeys = {
      "Mesh_velocity_source",
      "MeshVelocitySource",
      "ALE_mesh_velocity_source",
      "ALEMeshVelocitySource",
  };
  for (const auto key : kMeshVelocitySourceKeys) {
    if (const auto value = get_defined_string(params, key)) {
      const auto source = lower_copy(trim_copy(*value));
      if (source == "prescribed" || source == "prescribed_data" ||
          source == "data" || source == "mesh_motion_data") {
        options.mesh_velocity_source = ns::ALEMeshVelocitySource::PrescribedData;
      } else if (source == "coupled" || source == "coupled_displacement" ||
                 source == "derived" || source == "derived_from_displacement" ||
                 source == "monolithic") {
        options.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
      } else {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Mesh_velocity_source must be one of "
            "'prescribed_data' or 'coupled_displacement'.");
      }
    }
  }

  constexpr std::array<std::string_view, 3> kMeshDisplacementFieldKeys = {
      "Mesh_displacement_field",
      "MeshDisplacementField",
      "Mesh_motion_displacement_field",
  };
  for (const auto key : kMeshDisplacementFieldKeys) {
    if (const auto value = get_defined_string(params, key)) {
      options.mesh_displacement_field_name = *value;
    }
  }

  constexpr std::array<std::string_view, 3> kAutoRegisterMeshDisplacementKeys = {
      "Auto_register_mesh_displacement_field",
      "AutoRegisterMeshDisplacementField",
      "ALE_auto_register_mesh_displacement_field",
  };
  for (const auto key : kAutoRegisterMeshDisplacementKeys) {
    if (const auto value = get_defined_bool(params, key)) {
      options.auto_register_mesh_displacement_field = *value;
    }
  }

  constexpr std::array<std::string_view, 4> kMovingMeshTangentPathKeys = {
      "MovingMeshTangentPath",
      "Moving_mesh_tangent_path",
      "Moving_mesh_geometry_tangent_path",
      "ALE_moving_mesh_tangent_path",
  };
  for (const auto key : kMovingMeshTangentPathKeys) {
    if (const auto value = get_defined_string(params, key)) {
      options.moving_mesh_tangent_path =
          parse_geometry_tangent_path(*value, key);
    }
  }
}

void apply_fluid_moving_domain_options(
    const svmp::Physics::EquationModuleInput& input,
    const svmp::Physics::DomainInput& domain,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  apply_fluid_moving_domain_params(input.equation_params, options);
  apply_fluid_moving_domain_params(domain.params, options);
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

void apply_free_surface_schema_options(
    const svmp::Physics::ParameterMap& params,
    svmp::Physics::formulations::navier_stokes::
        IncompressibleNavierStokesVMSOptions& options)
{
  const auto contract = resolve_free_surface_schema_contract(params);
  options.input_configuration_schema_version = contract.version;
  options.explicit_legacy_configuration = contract.explicit_legacy;
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

struct Vec3d {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

Vec3d to_vec3(const std::array<svmp::FE::Real, 3>& v)
{
  return Vec3d{static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

std::array<svmp::FE::Real, 3> to_array(const Vec3d& v)
{
  return {static_cast<svmp::FE::Real>(v.x), static_cast<svmp::FE::Real>(v.y), static_cast<svmp::FE::Real>(v.z)};
}

Vec3d operator+(const Vec3d& a, const Vec3d& b) { return Vec3d{a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3d operator-(const Vec3d& a, const Vec3d& b) { return Vec3d{a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3d operator*(double s, const Vec3d& a) { return Vec3d{s * a.x, s * a.y, s * a.z}; }

double dot(const Vec3d& a, const Vec3d& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

double norm2(const Vec3d& a) { return dot(a, a); }

double norm(const Vec3d& a) { return std::sqrt(norm2(a)); }

Vec3d cross(const Vec3d& a, const Vec3d& b)
{
  return Vec3d{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vec3d normalized(const Vec3d& v)
{
  const double n = norm(v);
  if (!(n > 0.0)) {
    return Vec3d{};
  }
  return (1.0 / n) * v;
}

struct MarkerGeometry {
  double area{0.0};
  Vec3d center_sum{};
  Vec3d normal_sum{};
};

MarkerGeometry local_marker_geometry(const svmp::MeshBase& mesh, int boundary_marker)
{
  MarkerGeometry out{};
  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
  for (const auto f : faces) {
    const double a = static_cast<double>(mesh.face_area(f));
    const auto c = to_vec3(mesh.face_center(f));
    const auto n = to_vec3(mesh.face_normal(f));
    out.area += a;
    out.center_sum = out.center_sum + (a * c);
    out.normal_sum = out.normal_sum + (a * n);
  }
  return out;
}

[[nodiscard]] MarkerCommunicator markerCommunicator(
    const svmp::FE::systems::FESystem& system) noexcept
{
#if FE_HAS_MPI
  return system.activeMpiCommunicator();
#else
  (void)system;
  return 0;
#endif
}

MarkerGeometry global_marker_geometry(const svmp::MeshBase& mesh,
                                      int boundary_marker,
                                      MarkerCommunicator comm);

struct ParabolicProfileData {
  Vec3d center{};
  std::vector<Vec3d> perimeter_unit_dirs{};
  std::vector<double> perimeter_r2{};
};

struct GidPairHash {
  std::size_t operator()(const std::pair<svmp::gid_t, svmp::gid_t>& p) const noexcept
  {
    const std::size_t h1 = std::hash<svmp::gid_t>{}(p.first);
    const std::size_t h2 = std::hash<svmp::gid_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct GidSequenceHash {
  std::size_t operator()(const std::vector<svmp::gid_t>& values) const noexcept
  {
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto value : values) {
      const auto mixed = static_cast<std::uint64_t>(std::hash<svmp::gid_t>{}(value));
      hash ^= mixed + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    hash ^= static_cast<std::uint64_t>(values.size()) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return static_cast<std::size_t>(hash);
  }
};

[[nodiscard]] bool mpiMultiRankActive(MarkerCommunicator comm) noexcept
{
#if FE_HAS_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    return false;
  }

  int comm_size = 1;
  MPI_Comm_size(comm, &comm_size);
  return comm_size > 1;
#else
  (void)comm;
  return false;
#endif
}

template <std::size_t N>
struct MarkerFacePayload {
  std::vector<svmp::gid_t> ordered_vertex_gids{};
  std::array<double, N> values{};
};

template <std::size_t N, class PayloadBuilder>
std::vector<MarkerFacePayload<N>> gather_unique_marker_face_payloads(
    const svmp::MeshBase& mesh,
    int boundary_marker,
    MarkerCommunicator comm,
    PayloadBuilder&& payload_builder)
{
  const bool mpi_active = mpiMultiRankActive(comm);
  const auto& vgids = mesh.vertex_gids();
  if (mpi_active && vgids.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] MPI boundary-face normalization for marker " +
        std::to_string(boundary_marker) + " requires stable vertex GIDs.");
  }

  std::vector<MarkerFacePayload<N>> local_payloads;
  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));
  local_payloads.reserve(faces.size());

  for (const auto f : faces) {
    MarkerFacePayload<N> payload;
    const auto verts = mesh.face_vertices(f);
    payload.ordered_vertex_gids.reserve(verts.size());
    for (const auto v : verts) {
      if (v == svmp::INVALID_INDEX) {
        continue;
      }
      const auto idx = static_cast<std::size_t>(v);
      if (idx >= mesh.n_vertices()) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Boundary face references an out-of-range vertex while processing marker " +
            std::to_string(boundary_marker) + ".");
      }
      const auto gid = (vgids.size() == mesh.n_vertices())
                           ? vgids[idx]
                           : static_cast<svmp::gid_t>(v);
      payload.ordered_vertex_gids.push_back(gid);
    }
    if (payload.ordered_vertex_gids.size() < 2u) {
      continue;
    }
    payload_builder(f, payload.values);
    local_payloads.push_back(std::move(payload));
  }

  auto deduplicate = [](std::vector<MarkerFacePayload<N>> payloads) {
    std::unordered_set<std::vector<svmp::gid_t>, GidSequenceHash> seen;
    std::vector<MarkerFacePayload<N>> unique;
    unique.reserve(payloads.size());
    for (auto& payload : payloads) {
      auto key = payload.ordered_vertex_gids;
      std::sort(key.begin(), key.end());
      key.erase(std::unique(key.begin(), key.end()), key.end());
      if (key.size() < 2u) {
        continue;
      }
      if (!seen.emplace(std::move(key)).second) {
        continue;
      }
      unique.push_back(std::move(payload));
    }
    return unique;
  };

  if (!mpi_active) {
    return deduplicate(std::move(local_payloads));
  }

#if FE_HAS_MPI
  int comm_size = 1;
  MPI_Comm_size(comm, &comm_size);

  const int local_face_count = static_cast<int>(local_payloads.size());
  std::vector<int> face_counts(static_cast<std::size_t>(comm_size), 0);
  MPI_Allgather(&local_face_count, 1, MPI_INT, face_counts.data(), 1, MPI_INT, comm);

  std::vector<int> face_displs(static_cast<std::size_t>(comm_size), 0);
  int total_faces = 0;
  for (int r = 0; r < comm_size; ++r) {
    face_displs[static_cast<std::size_t>(r)] = total_faces;
    total_faces += face_counts[static_cast<std::size_t>(r)];
  }

  std::vector<int> local_gid_sizes;
  local_gid_sizes.reserve(local_payloads.size());
  std::vector<svmp::gid_t> local_gid_data;
  for (const auto& payload : local_payloads) {
    local_gid_sizes.push_back(static_cast<int>(payload.ordered_vertex_gids.size()));
    local_gid_data.insert(local_gid_data.end(),
                          payload.ordered_vertex_gids.begin(),
                          payload.ordered_vertex_gids.end());
  }

  const int local_gid_count = static_cast<int>(local_gid_data.size());
  std::vector<int> gid_counts(static_cast<std::size_t>(comm_size), 0);
  MPI_Allgather(&local_gid_count, 1, MPI_INT, gid_counts.data(), 1, MPI_INT, comm);

  std::vector<int> gid_displs(static_cast<std::size_t>(comm_size), 0);
  int total_gid_count = 0;
  for (int r = 0; r < comm_size; ++r) {
    gid_displs[static_cast<std::size_t>(r)] = total_gid_count;
    total_gid_count += gid_counts[static_cast<std::size_t>(r)];
  }

  std::vector<int> all_gid_sizes(static_cast<std::size_t>(total_faces), 0);
  MPI_Allgatherv(local_gid_sizes.data(),
                 local_face_count,
                 MPI_INT,
                 all_gid_sizes.data(),
                 face_counts.data(),
                 face_displs.data(),
                 MPI_INT,
                 comm);

  std::vector<svmp::gid_t> all_gid_data(static_cast<std::size_t>(total_gid_count), svmp::gid_t{0});
  MPI_Allgatherv(local_gid_data.data(),
                 local_gid_count,
                 MPI_INT64_T,
                 all_gid_data.data(),
                 gid_counts.data(),
                 gid_displs.data(),
                 MPI_INT64_T,
                 comm);

  std::vector<double> local_values;
  local_values.reserve(local_payloads.size() * N);
  for (const auto& payload : local_payloads) {
    local_values.insert(local_values.end(), payload.values.begin(), payload.values.end());
  }

  std::vector<int> value_counts(static_cast<std::size_t>(comm_size), 0);
  std::vector<int> value_displs(static_cast<std::size_t>(comm_size), 0);
  int total_value_count = 0;
  for (int r = 0; r < comm_size; ++r) {
    value_displs[static_cast<std::size_t>(r)] = total_value_count;
    value_counts[static_cast<std::size_t>(r)] = face_counts[static_cast<std::size_t>(r)] * static_cast<int>(N);
    total_value_count += value_counts[static_cast<std::size_t>(r)];
  }

  std::vector<double> all_values(static_cast<std::size_t>(total_value_count), 0.0);
  MPI_Allgatherv(local_values.data(),
                 static_cast<int>(local_values.size()),
                 MPI_DOUBLE,
                 all_values.data(),
                 value_counts.data(),
                 value_displs.data(),
                 MPI_DOUBLE,
                 comm);

  std::vector<MarkerFacePayload<N>> gathered;
  gathered.reserve(static_cast<std::size_t>(total_faces));
  std::size_t gid_offset = 0u;
  for (int i = 0; i < total_faces; ++i) {
    const auto count = static_cast<std::size_t>(all_gid_sizes[static_cast<std::size_t>(i)]);
    if (gid_offset + count > all_gid_data.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Corrupt MPI face-GID gather while processing marker " +
          std::to_string(boundary_marker) + ".");
    }
    MarkerFacePayload<N> payload;
    payload.ordered_vertex_gids.assign(all_gid_data.begin() + static_cast<std::ptrdiff_t>(gid_offset),
                                       all_gid_data.begin() + static_cast<std::ptrdiff_t>(gid_offset + count));
    gid_offset += count;

    const std::size_t value_offset = static_cast<std::size_t>(i) * N;
    if (value_offset + N > all_values.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Corrupt MPI face-payload gather while processing marker " +
          std::to_string(boundary_marker) + ".");
    }
    std::copy_n(all_values.begin() + static_cast<std::ptrdiff_t>(value_offset), N, payload.values.begin());
    gathered.push_back(std::move(payload));
  }

  return deduplicate(std::move(gathered));
#else
  return deduplicate(std::move(local_payloads));
#endif
}

MarkerGeometry global_marker_geometry(const svmp::MeshBase& mesh,
                                      int boundary_marker,
                                      MarkerCommunicator comm)
{
  MarkerGeometry out{};
  const auto payloads = gather_unique_marker_face_payloads<7>(
      mesh, boundary_marker, comm,
      [&](svmp::index_t f, std::array<double, 7>& values) {
        const double a = static_cast<double>(mesh.face_area(f));
        const auto c = to_vec3(mesh.face_center(f));
        const auto n = to_vec3(mesh.face_normal(f));
        values[0] = a;
        values[1] = a * c.x;
        values[2] = a * c.y;
        values[3] = a * c.z;
        values[4] = a * n.x;
        values[5] = a * n.y;
        values[6] = a * n.z;
      });
  for (const auto& payload : payloads) {
    out.area += payload.values[0];
    out.center_sum = out.center_sum + Vec3d{payload.values[1], payload.values[2], payload.values[3]};
    out.normal_sum = out.normal_sum + Vec3d{payload.values[4], payload.values[5], payload.values[6]};
  }
  return out;
}

std::vector<std::pair<svmp::gid_t, Vec3d>> gather_perimeter_vertex_coords(const svmp::MeshBase& mesh,
                                                                          int boundary_marker,
                                                                          MarkerCommunicator comm)
{
  const auto& vgids = mesh.vertex_gids();
  if (vgids.size() != mesh.n_vertices()) {
    return {};
  }

  // Determine the perimeter of the marker surface as the set of boundary edges of the
  // marker patch itself (edges that appear only once among unique marker faces).
  const auto marker_faces = gather_unique_marker_face_payloads<1>(
      mesh, boundary_marker, comm,
      [](svmp::index_t /*face*/, std::array<double, 1>& values) {
        values[0] = 0.0;
      });

  std::vector<svmp::gid_t> local_edge_gids;
  local_edge_gids.reserve(marker_faces.size() * 6u);

  for (const auto& face : marker_faces) {
    const auto& face_vgids = face.ordered_vertex_gids;
    for (std::size_t i = 0; i < face_vgids.size(); ++i) {
      svmp::gid_t a = face_vgids[i];
      svmp::gid_t b = face_vgids[(i + 1u) % face_vgids.size()];
      if (a == b) {
        continue;
      }
      if (a > b) {
        std::swap(a, b);
      }
      local_edge_gids.push_back(a);
      local_edge_gids.push_back(b);
    }
  }

  const auto& all_edge_gids = local_edge_gids;

  std::unordered_map<std::pair<svmp::gid_t, svmp::gid_t>, int, GidPairHash> edge_counts;
  edge_counts.reserve(all_edge_gids.size() / 2u);
  for (std::size_t i = 0; i + 1u < all_edge_gids.size(); i += 2u) {
    const auto key = std::make_pair(all_edge_gids[i], all_edge_gids[i + 1u]);
    ++edge_counts[key];
  }

  std::unordered_set<svmp::gid_t> perimeter_gids;
  perimeter_gids.reserve(edge_counts.size() * 2u);
  for (const auto& [e, count] : edge_counts) {
    if (count == 1) {
      perimeter_gids.insert(e.first);
      perimeter_gids.insert(e.second);
    }
  }

  std::vector<svmp::gid_t> local_gids;
  std::vector<double> local_xyz;
  local_gids.reserve(perimeter_gids.size());
  local_xyz.reserve(perimeter_gids.size() * 3u);

  for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
    const auto idx = static_cast<std::size_t>(v);
    if (idx >= vgids.size()) {
      continue;
    }
    const auto gid = vgids[idx];
    if (perimeter_gids.find(gid) == perimeter_gids.end()) {
      continue;
    }
    local_gids.push_back(gid);
    const auto p = to_vec3(mesh.get_vertex_coords(v));
    local_xyz.push_back(p.x);
    local_xyz.push_back(p.y);
    local_xyz.push_back(p.z);
  }

  std::vector<svmp::gid_t> all_gids = local_gids;
  std::vector<double> all_xyz = local_xyz;

#if FE_HAS_MPI
  if (mpiMultiRankActive(comm)) {
    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);

    const int local_gid_count = static_cast<int>(local_gids.size());
    std::vector<int> gid_counts(static_cast<std::size_t>(comm_size), 0);
    MPI_Allgather(&local_gid_count, 1, MPI_INT, gid_counts.data(), 1, MPI_INT, comm);

    std::vector<int> gid_displs(static_cast<std::size_t>(comm_size), 0);
    int total_gid_count = 0;
    for (int r = 0; r < comm_size; ++r) {
      gid_displs[static_cast<std::size_t>(r)] = total_gid_count;
      total_gid_count += gid_counts[static_cast<std::size_t>(r)];
    }

    all_gids.assign(static_cast<std::size_t>(total_gid_count), svmp::gid_t{0});
    MPI_Allgatherv(local_gids.data(),
                   local_gid_count,
                   MPI_INT64_T,
                   all_gids.data(),
                   gid_counts.data(),
                   gid_displs.data(),
                   MPI_INT64_T,
                   comm);

    std::vector<int> xyz_counts(static_cast<std::size_t>(comm_size), 0);
    std::vector<int> xyz_displs(static_cast<std::size_t>(comm_size), 0);
    int total_xyz_count = 0;
    for (int r = 0; r < comm_size; ++r) {
      xyz_displs[static_cast<std::size_t>(r)] = total_xyz_count;
      xyz_counts[static_cast<std::size_t>(r)] = gid_counts[static_cast<std::size_t>(r)] * 3;
      total_xyz_count += xyz_counts[static_cast<std::size_t>(r)];
    }

    all_xyz.assign(static_cast<std::size_t>(total_xyz_count), 0.0);
    MPI_Allgatherv(local_xyz.data(),
                   static_cast<int>(local_xyz.size()),
                   MPI_DOUBLE,
                   all_xyz.data(),
                   xyz_counts.data(),
                   xyz_displs.data(),
                   MPI_DOUBLE,
                   comm);
  }
#endif

  std::vector<std::pair<svmp::gid_t, Vec3d>> out;
  out.reserve(all_gids.size());
  for (std::size_t i = 0; i < all_gids.size(); ++i) {
    const std::size_t j = i * 3u;
    if (j + 2u >= all_xyz.size()) {
      break;
    }
    out.emplace_back(all_gids[i], Vec3d{all_xyz[j + 0], all_xyz[j + 1], all_xyz[j + 2]});
  }
  return out;
}

ParabolicProfileData build_parabolic_profile_data(const svmp::MeshBase& mesh,
                                                  int boundary_marker,
                                                  MarkerCommunicator comm)
{
  const auto g = global_marker_geometry(mesh, boundary_marker, comm);
  if (!(g.area > 0.0)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary marker " + std::to_string(boundary_marker) +
        " has zero area; cannot construct parabolic profile.");
  }
  ParabolicProfileData out{};
  out.center = (1.0 / g.area) * g.center_sum;

  const auto all_perim =
      gather_perimeter_vertex_coords(mesh, boundary_marker, comm);
  std::unordered_map<svmp::gid_t, Vec3d> unique;
  unique.reserve(all_perim.size());
  for (const auto& [gid, p] : all_perim) {
    if (unique.find(gid) == unique.end()) {
      unique.emplace(gid, p);
    }
  }

  out.perimeter_unit_dirs.clear();
  out.perimeter_r2.clear();
  out.perimeter_unit_dirs.reserve(unique.size());
  out.perimeter_r2.reserve(unique.size());

  for (const auto& [_, p] : unique) {
    const Vec3d v = p - out.center;
    const double r2 = norm2(v);
    if (!(r2 > 0.0)) {
      continue;
    }
    out.perimeter_r2.push_back(r2);
    out.perimeter_unit_dirs.push_back((1.0 / std::sqrt(r2)) * v);
  }

  if (out.perimeter_unit_dirs.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Parabolic inflow profile for boundary marker " + std::to_string(boundary_marker) +
        " requires a non-empty perimeter. Ensure this face shares an edge with another boundary face "
        "(e.g., a wall surface), or set <Profile>Flat</Profile>.");
  }

  return out;
}

double parabolic_weight(const ParabolicProfileData& data, const Vec3d& x)
{
  const Vec3d r = x - data.center;
  const double r2 = norm2(r);
  if (!(r2 > 0.0)) {
    return 1.0;
  }

  double best = -std::numeric_limits<double>::infinity();
  std::size_t best_i = 0;
  for (std::size_t i = 0; i < data.perimeter_unit_dirs.size(); ++i) {
    const double d = dot(r, data.perimeter_unit_dirs[i]);
    if (d > best) {
      best = d;
      best_i = i;
    }
  }

  const double R2 = data.perimeter_r2[best_i];
  if (!(R2 > 0.0)) {
    return 0.0;
  }

  const double w = 1.0 - (r2 / R2);
  return (w > 0.0) ? w : 0.0;
}

double integrate_parabolic_weight_over_marker(const svmp::MeshBase& mesh,
                                              int boundary_marker,
                                              const ParabolicProfileData& data,
                                              MarkerCommunicator comm)
{
  auto tri_area = [](const Vec3d& a, const Vec3d& b, const Vec3d& c) {
    const Vec3d ab = b - a;
    const Vec3d ac = c - a;
    return 0.5 * norm(cross(ab, ac));
  };

  auto tri_integral = [&](const Vec3d& p0, const Vec3d& p1, const Vec3d& p2) {
    const double a = tri_area(p0, p1, p2);
    if (!(a > 0.0)) {
      return 0.0;
    }
    // Degree-2 symmetric rule (3 points), weights sum to 1.
    const Vec3d x1 = (1.0 / 6.0) * p0 + (1.0 / 6.0) * p1 + (2.0 / 3.0) * p2;
    const Vec3d x2 = (1.0 / 6.0) * p0 + (2.0 / 3.0) * p1 + (1.0 / 6.0) * p2;
    const Vec3d x3 = (2.0 / 3.0) * p0 + (1.0 / 6.0) * p1 + (1.0 / 6.0) * p2;

    const double f = (parabolic_weight(data, x1) + parabolic_weight(data, x2) + parabolic_weight(data, x3)) / 3.0;
    return a * f;
  };

  auto face_integral = [&](const std::vector<svmp::index_t>& verts) {
    if (verts.size() < 3u) {
      return 0.0;
    }

    // Treat any face as a polygon and fan-triangulate it.
    // This is robust to meshes that store boundary faces as CellFamily::Polygon.
    std::vector<Vec3d> pts;
    pts.reserve(verts.size());
    for (const auto v : verts) {
      if (v == svmp::INVALID_INDEX) {
        continue;
      }
      pts.push_back(to_vec3(mesh.get_vertex_coords(v)));
    }
    if (pts.size() < 3u) {
      return 0.0;
    }

    double val = 0.0;
    const Vec3d p0 = pts[0];
    for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
      val += tri_integral(p0, pts[i], pts[i + 1]);
    }
    return val;
  };

  double sum = 0.0;
  const auto payloads = gather_unique_marker_face_payloads<1>(
      mesh, boundary_marker, comm,
      [&](svmp::index_t f, std::array<double, 1>& values) {
        values[0] = face_integral(mesh.face_vertices(f));
      });
  for (const auto& payload : payloads) {
    sum += payload.values[0];
  }
  return sum;
}

enum class InletProfileType { Flat, Parabolic };

struct InletProfileContext {
  int dim{0};
  InletProfileType profile{InletProfileType::Flat};
  bool use_normal_direction{true};
  std::array<int, 3> active_components{1, 1, 1}; // used when use_normal_direction==false

  Vec3d normal{};   // unit
  double scale{0.0};

  std::optional<ParabolicProfileData> parabolic{};

  double weight(const Vec3d& x) const
  {
    switch (profile) {
      case InletProfileType::Flat: return 1.0;
      case InletProfileType::Parabolic:
        if (!parabolic.has_value()) {
          return 0.0;
        }
        return parabolic_weight(*parabolic, x);
    }
    return 1.0;
  }

  double componentValue(int component, const Vec3d& x) const
  {
    if (component < 0 || component >= 3) {
      return 0.0;
    }
    const double w = weight(x);
    if (use_normal_direction) {
      const double nd = (component == 0) ? normal.x : (component == 1) ? normal.y : normal.z;
      return scale * w * nd;
    }
    const int active = active_components[static_cast<std::size_t>(component)];
    if (active == 0) {
      return 0.0;
    }
    return scale * w;
  }
};

template <class ScalarValue>
void fill_vector(std::array<ScalarValue, 3>& dst, int dim, const std::vector<int>& effective_dir,
                 svmp::FE::Real magnitude)
{
  dst = {ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
  for (int d = 0; d < dim; ++d) {
    const auto scale =
        effective_dir.empty() ? static_cast<svmp::FE::Real>(1.0) : direction_component(effective_dir, d);
    dst[static_cast<std::size_t>(d)] = ScalarValue{static_cast<svmp::FE::Real>(magnitude * scale)};
  }
}

std::string normalized_token(std::string_view raw)
{
  auto out = lower_copy(trim_copy(std::string(raw)));
  out.erase(std::remove_if(out.begin(), out.end(), [](unsigned char ch) {
              return ch == '_' || ch == '-' || std::isspace(ch);
            }),
            out.end());
  return out;
}

bool is_free_surface_type(std::string_view raw)
{
  return normalized_token(raw) == "freesurface";
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation
parse_free_surface_implementation(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "fitted" || token == "fittedale" || token == "ale") {
    return FreeSurfaceImplementation::FittedALE;
  }
  if (token == "unfitted" || token == "unfittedlevelset" ||
      token == "levelset" || token == "embeddedlevelset") {
    return FreeSurfaceImplementation::UnfittedLevelSet;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of FittedALE or UnfittedLevelSet.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceKinematicEnforcement
parse_free_surface_kinematic_enforcement(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceKinematicEnforcement;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "none") {
    return FreeSurfaceKinematicEnforcement::None;
  }
  if (token == "penalty") {
    return FreeSurfaceKinematicEnforcement::Penalty;
  }
  if (token == "nitsche") {
    return FreeSurfaceKinematicEnforcement::Nitsche;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of None, Penalty, or Nitsche.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomain
parse_free_surface_active_domain(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomain;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "none" || token == "disabled" ||
      token == "off" || token == "inactive") {
    return FreeSurfaceActiveDomain::None;
  }
  if (token == "levelsetnegative" || token == "negative" ||
      token == "negativelevelset" || token == "phinegative") {
    return FreeSurfaceActiveDomain::LevelSetNegative;
  }
  if (token == "levelsetpositive" || token == "positive" ||
      token == "positivelevelset" || token == "phipositive") {
    return FreeSurfaceActiveDomain::LevelSetPositive;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of None, LevelSetNegative, or LevelSetPositive.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomainMethod
parse_free_surface_active_domain_method(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomainMethod;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "cutvolume" || token == "cutcellvolume" ||
      token == "exactcutvolume") {
    return FreeSurfaceActiveDomainMethod::CutVolume;
  }
  if (token == "smoothedindicator" || token == "smoothindicator" ||
      token == "indicator") {
    return FreeSurfaceActiveDomainMethod::SmoothedIndicator;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of CutVolume or SmoothedIndicator.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceSurfaceTensionForm
parse_free_surface_surface_tension_form(std::string_view raw,
                                       std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceSurfaceTensionForm;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "automatic" || token == "auto") {
    return FreeSurfaceSurfaceTensionForm::Automatic;
  }
  if (token == "curvaturetraction" || token == "explicitcurvature" ||
      token == "younglaplacetraction") {
    return FreeSurfaceSurfaceTensionForm::CurvatureTraction;
  }
  if (token == "generatedcurvaturetraction" ||
      token == "generatedgeometrycurvaturetraction" ||
      token == "generatednormalcurvaturetraction") {
    return FreeSurfaceSurfaceTensionForm::GeneratedCurvatureTraction;
  }
  if (token == "kinematicareagradienttraction" ||
      token == "totalenergygradienttraction" ||
      token == "discreteenergygradienttraction") {
    return FreeSurfaceSurfaceTensionForm::KinematicAreaGradientTraction;
  }
  if (token == "surfacestress" || token == "surfaceenergy" ||
      token == "laplacebeltrami" || token == "variational") {
    return FreeSurfaceSurfaceTensionForm::SurfaceStress;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of Automatic, CurvatureTraction, "
      "GeneratedCurvatureTraction, KinematicAreaGradientTraction, or SurfaceStress.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceNormalKinematicPolicy
parse_free_surface_normal_kinematic_policy(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceNormalKinematicPolicy;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "matchfluidnormalvelocity" ||
      token == "matchfluid" || token == "fluidnormalvelocity") {
    return FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity;
  }
  if (token == "none" || token == "disabled" || token == "off") {
    return FreeSurfaceNormalKinematicPolicy::None;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of MatchFluidNormalVelocity or None.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfaceTangentialMeshPolicy
parse_free_surface_tangential_mesh_policy(std::string_view raw, std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceTangentialMeshPolicy;
  const auto token = normalized_token(raw);
  if (token == "free" || token == "freetangential" || token == "unconstrained") {
    return FreeSurfaceTangentialMeshPolicy::Free;
  }
  if (token.empty() || token == "smoothing" || token == "smoothingonly" ||
      token == "meshsmoothing" || token == "smooth") {
    return FreeSurfaceTangentialMeshPolicy::SmoothingOnly;
  }
  if (token == "prescribed" || token == "prescribedtangential" ||
      token == "prescribedvelocity") {
    return FreeSurfaceTangentialMeshPolicy::Prescribed;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of Free, SmoothingOnly, or Prescribed.");
}

enum class ParsedFreeSurfaceContactModel : std::uint8_t {
  None,
  Pinned,
  PrescribedAngle,
  DynamicRenE
};

ParsedFreeSurfaceContactModel
parse_free_surface_contact_line_model(std::string_view raw, std::string_view context)
{
  const auto token = normalized_token(raw);
  if (token.empty() || token == "none" || token == "disabled" || token == "off") {
    return ParsedFreeSurfaceContactModel::None;
  }
  if (token == "pinned" || token == "fixed" || token == "fixedposition") {
    return ParsedFreeSurfaceContactModel::Pinned;
  }
  if (token == "prescribedcontactangle" || token == "contactangle" ||
      token == "prescribedangle") {
    return ParsedFreeSurfaceContactModel::PrescribedAngle;
  }
  if (token == "dynamiccontactangle" || token == "dynamicangle" ||
      token == "dynamicrene" || token == "rene") {
    return ParsedFreeSurfaceContactModel::DynamicRenE;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of None, Pinned, PrescribedAngle, or DynamicRenE.");
}

enum class ParsedFreeSurfaceWallSlipModel : std::uint8_t {
  None,
  Navier
};

ParsedFreeSurfaceWallSlipModel
parse_free_surface_wall_slip_model(std::string_view raw, std::string_view context)
{
  const auto token = normalized_token(raw);
  if (token.empty() || token == "none" || token == "noslip" ||
      token == "disabled" || token == "off") {
    return ParsedFreeSurfaceWallSlipModel::None;
  }
  if (token == "navier" || token == "navierslip" || token == "slip") {
    return ParsedFreeSurfaceWallSlipModel::Navier;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of None or Navier.");
}

svmp::Physics::formulations::navier_stokes::FreeSurfacePressureStabilizationPolicy
parse_free_surface_pressure_stabilization_policy(std::string_view raw,
                                                 std::string_view context)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfacePressureStabilizationPolicy;
  const auto token = normalized_token(raw);
  if (token.empty() || token == "enabled" || token == "enable" ||
      token == "on" || token == "true") {
    return FreeSurfacePressureStabilizationPolicy::Enabled;
  }
  if (token == "incremental" || token == "pressureincrement" ||
      token == "incrementalpressure" || token == "incrementalenabled") {
    return FreeSurfacePressureStabilizationPolicy::Incremental;
  }
  if (token == "disabled" || token == "disable" ||
      token == "off" || token == "false") {
    return FreeSurfacePressureStabilizationPolicy::Disabled;
  }
  if (token == "disabledforrefreshedfrozenhighorder" ||
      token == "disableforrefreshedfrozenhighorder" ||
      token == "refreshedfrozenhighorderdisabled" ||
      token == "highorderrefreshedfrozendisabled") {
    return FreeSurfacePressureStabilizationPolicy::
        DisabledForRefreshedFrozenHighOrder;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] " + std::string(context) +
      " must be one of Enabled, Incremental, Disabled, or DisabledForRefreshedFrozenHighOrder.");
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
  static constexpr std::array<std::string_view, 26> supported{
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
      "LevelSetFieldName",
      "Generated_interface_domain_id",
      "GeneratedInterfaceDomainId",
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

void reject_duplicate_aliases(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view property)
{
  if (defined_parameter_count(params, keys) > 1u) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface contact configuration defines multiple aliases for " +
        std::string(property) + ". Specify exactly one spelling.");
  }
}

std::vector<std::string> split_semicolon_entries(std::string_view raw)
{
  std::vector<std::string> entries;
  std::string current;
  for (const char ch : raw) {
    if (ch == ';') {
      auto entry = trim_copy(std::move(current));
      if (!entry.empty()) {
        entries.push_back(std::move(entry));
      }
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  auto entry = trim_copy(std::move(current));
  if (!entry.empty()) {
    entries.push_back(std::move(entry));
  }
  return entries;
}

std::optional<std::vector<int>> first_defined_int_entries(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view context)
{
  for (const auto key : keys) {
    if (const auto value = get_defined_string(params, key)) {
      std::vector<int> parsed;
      for (const auto& entry : split_semicolon_entries(*value)) {
        try {
          size_t pos = 0;
          const int marker = std::stoi(entry, &pos);
          if (pos != entry.size()) {
            throw std::runtime_error("");
          }
          parsed.push_back(marker);
        } catch (...) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Failed to parse integer value '" +
              entry + "' for " + std::string(context) + ".");
        }
      }
      if (parsed.empty()) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] " + std::string(context) +
            " must contain at least one integer marker.");
      }
      return parsed;
    }
  }
  return std::nullopt;
}

std::optional<std::vector<std::array<svmp::FE::Real, 3>>>
first_defined_vector3_entries(const svmp::Physics::ParameterMap& params,
                              std::initializer_list<std::string_view> keys,
                              std::string_view context)
{
  for (const auto key : keys) {
    if (const auto value = get_defined_string(params, key)) {
      std::vector<std::array<svmp::FE::Real, 3>> parsed;
      for (const auto& entry : split_semicolon_entries(*value)) {
        parsed.push_back(parse_real_vector3(entry, context));
      }
      if (parsed.empty()) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] " + std::string(context) +
            " must contain at least one vector entry.");
      }
      return parsed;
    }
  }
  return std::nullopt;
}

void apply_fluid_rotating_frame_coriolis(
    const svmp::Physics::EquationModuleInput& input,
    const svmp::Physics::DomainInput& domain,
    int dim,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using Options = svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  const auto enabled = first_defined_bool(
      input.equation_params,
      {"Rotating_frame_coriolis", "RotatingFrameCoriolis",
       "Enable_rotating_frame_coriolis", "EnableRotatingFrameCoriolis"})
      .value_or(first_defined_bool(
          domain.params,
          {"Rotating_frame_coriolis", "RotatingFrameCoriolis",
           "Enable_rotating_frame_coriolis", "EnableRotatingFrameCoriolis"})
          .value_or(false));

  const auto equation_omega = first_defined_string(
      input.equation_params,
      {"Rotating_frame_angular_velocity", "RotatingFrameAngularVelocity",
       "Angular_velocity", "AngularVelocity"});
  const auto domain_omega = first_defined_string(
      domain.params,
      {"Rotating_frame_angular_velocity", "RotatingFrameAngularVelocity",
       "Angular_velocity", "AngularVelocity"});
  const auto equation_omega_file = first_defined_string(
      input.equation_params,
      {"Rotating_frame_angular_velocity_temporal_values_file_path",
       "RotatingFrameAngularVelocityTemporalValuesFilePath",
       "Angular_velocity_temporal_values_file_path",
       "AngularVelocityTemporalValuesFilePath"});
  const auto domain_omega_file = first_defined_string(
      domain.params,
      {"Rotating_frame_angular_velocity_temporal_values_file_path",
       "RotatingFrameAngularVelocityTemporalValuesFilePath",
       "Angular_velocity_temporal_values_file_path",
       "AngularVelocityTemporalValuesFilePath"});

  if (equation_omega && equation_omega_file) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Specify either Rotating_frame_angular_velocity or "
        "Rotating_frame_angular_velocity_temporal_values_file_path, not both.");
  }
  if (domain_omega && domain_omega_file) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Specify either Rotating_frame_angular_velocity or "
        "Rotating_frame_angular_velocity_temporal_values_file_path in a domain, not both.");
  }

  const bool equation_has_omega = equation_omega || equation_omega_file;
  const auto omega = equation_omega ? equation_omega : (!equation_has_omega ? domain_omega : std::nullopt);
  const auto omega_file =
      equation_omega_file ? equation_omega_file : (!equation_has_omega ? domain_omega_file : std::nullopt);

  if (!enabled && !omega && !omega_file) {
    return;
  }
  if (dim != 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Rotating-frame Coriolis forcing requires a 3D Navier-Stokes mesh.");
  }

  options.rotating_frame_coriolis_enabled = true;
  if (omega) {
    const auto parsed = parse_real_vector3(*omega, "Rotating_frame_angular_velocity");
    for (std::size_t d = 0; d < parsed.size(); ++d) {
      options.rotating_frame_angular_velocity[d] = Options::ScalarValue{parsed[d]};
    }
  }
  if (omega_file) {
    auto temporal = svmp::Physics::readTemporalValuesFile(
        *omega_file, /*num_components=*/3, svmp::Physics::TemporalEndBehavior::Clamp);
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

void apply_fluid_momentum_source_spacetime_file(
    const svmp::Physics::EquationModuleInput& input,
    const svmp::Physics::DomainInput& domain,
    int dim,
    MarkerCommunicator comm,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  const auto path = first_defined_string(
      input.equation_params,
      {"Momentum_source_temporal_and_spatial_values_file_path",
       "MomentumSourceTemporalAndSpatialValuesFilePath",
       "Body_force_temporal_and_spatial_values_file_path",
       "BodyForceTemporalAndSpatialValuesFilePath"});
  const auto domain_path = first_defined_string(
      domain.params,
      {"Momentum_source_temporal_and_spatial_values_file_path",
       "MomentumSourceTemporalAndSpatialValuesFilePath",
       "Body_force_temporal_and_spatial_values_file_path",
       "BodyForceTemporalAndSpatialValuesFilePath"});

  const auto source_path = path ? path : domain_path;
  if (!source_path || source_path->empty()) {
    return;
  }
  if (!input.mesh) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Time/spatial Navier-Stokes momentum source requires a mesh.");
  }

  auto data = read_temporal_and_spatial_values_file(
      *input.mesh,
      /*boundary_marker=*/-1,
      *source_path,
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

svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation
free_surface_implementation_from_params(const svmp::Physics::ParameterMap& params)
{
  const auto raw = first_defined_string(
      params,
      {"Implementation", "Free_surface_implementation", "FreeSurfaceImplementation"});
  return parse_free_surface_implementation(raw.value_or("FittedALE"), "Free-surface Implementation");
}

void append_free_surface_contact_line(
    const svmp::Physics::ParameterMap& params,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary& fs)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;
  using ContactLine = IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;

  constexpr std::string_view contact_keys[]{
      "Contact_line_model", "ContactLineModel",
      "Free_surface_contact_line_model", "FreeSurfaceContactLineModel",
      "Contact_angle_radians", "ContactAngleRadians",
      "Prescribed_contact_angle_radians", "PrescribedContactAngleRadians",
      "Contact_angle_degrees", "ContactAngleDegrees",
      "Prescribed_contact_angle_degrees", "PrescribedContactAngleDegrees",
      "Contact_line_wall_markers", "ContactLineWallMarkers",
      "Wall_boundary_markers", "WallBoundaryMarkers",
      "Contact_line_wall_marker", "ContactLineWallMarker",
      "Wall_boundary_marker", "WallBoundaryMarker",
      "Contact_line_wall_faces", "ContactLineWallFaces",
      "Wall_boundary_faces", "WallBoundaryFaces",
      "Contact_line_wall_face", "ContactLineWallFace",
      "Wall_boundary_face", "WallBoundaryFace",
      "Contact_line_marker", "ContactLineMarker",
      "Contact_line_wall_normals", "ContactLineWallNormals",
      "Contact_angle_wall_normals", "ContactAngleWallNormals",
      "Wall_normals", "WallNormals",
      "Contact_line_wall_normal", "ContactLineWallNormal",
      "Contact_angle_wall_normal", "ContactAngleWallNormal",
      "Wall_normal", "WallNormal",
      "Contact_line_mobility", "ContactLineMobility", "Mobility",
      "Wall_slip_model", "WallSlipModel",
      "Contact_line_wall_slip_model", "ContactLineWallSlipModel",
      "Wall_slip_length", "WallSlipLength", "Slip_length", "SlipLength"};

  const auto is_known_contact_key = [&](std::string_view key) {
    return std::find(std::begin(contact_keys), std::end(contact_keys), key) !=
           std::end(contact_keys);
  };
  const auto looks_like_contact_key = [](std::string_view key) {
    const auto token = normalized_token(key);
    const auto starts_with = [&](std::string_view prefix) {
      return token.rfind(prefix, 0) == 0;
    };
    return starts_with("contact") || starts_with("freesurfacecontact") ||
           starts_with("prescribedcontact") ||
           starts_with("wallboundarymarker") ||
           starts_with("wallboundaryface") || starts_with("wallnormal") ||
           starts_with("wallslip") || token == "mobility" ||
           token == "sliplength";
  };
  for (const auto& [key, value] : params) {
    if (value.defined && looks_like_contact_key(key) &&
        !is_known_contact_key(key)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Unknown free-surface contact key '" +
          key + "'. Contact configuration is fail-closed.");
    }
  }

  const bool contact_requested = std::any_of(
      std::begin(contact_keys), std::end(contact_keys), [&](std::string_view key) {
        const auto* value = find_param(params, key);
        return value != nullptr && value->defined;
      });
  if (!contact_requested) {
    return;
  }

  reject_duplicate_aliases(
      params,
      {"Contact_line_model", "ContactLineModel",
       "Free_surface_contact_line_model", "FreeSurfaceContactLineModel"},
      "contact-line model");
  reject_duplicate_aliases(
      params,
      {"Contact_angle_radians", "ContactAngleRadians",
       "Prescribed_contact_angle_radians", "PrescribedContactAngleRadians"},
      "contact angle in radians");
  reject_duplicate_aliases(
      params,
      {"Contact_angle_degrees", "ContactAngleDegrees",
       "Prescribed_contact_angle_degrees", "PrescribedContactAngleDegrees"},
      "contact angle in degrees");
  reject_duplicate_aliases(
      params,
      {"Contact_line_wall_markers", "ContactLineWallMarkers",
       "Wall_boundary_markers", "WallBoundaryMarkers",
       "Contact_line_wall_marker", "ContactLineWallMarker",
       "Wall_boundary_marker", "WallBoundaryMarker"},
      "contact wall marker(s)");
  reject_duplicate_aliases(
      params,
      {"Contact_line_wall_faces", "ContactLineWallFaces",
       "Wall_boundary_faces", "WallBoundaryFaces",
       "Contact_line_wall_face", "ContactLineWallFace",
       "Wall_boundary_face", "WallBoundaryFace"},
      "contact wall face name(s)");
  reject_duplicate_aliases(
      params,
      {"Contact_line_marker", "ContactLineMarker"},
      "contact-line marker");
  reject_duplicate_aliases(
      params,
      {"Contact_line_wall_normals", "ContactLineWallNormals",
       "Contact_angle_wall_normals", "ContactAngleWallNormals",
       "Wall_normals", "WallNormals",
       "Contact_line_wall_normal", "ContactLineWallNormal",
       "Contact_angle_wall_normal", "ContactAngleWallNormal",
       "Wall_normal", "WallNormal"},
      "contact wall normal(s)");
  reject_duplicate_aliases(
      params,
      {"Contact_line_mobility", "ContactLineMobility", "Mobility"},
      "contact-line mobility");
  reject_duplicate_aliases(
      params,
      {"Wall_slip_model", "WallSlipModel",
       "Contact_line_wall_slip_model", "ContactLineWallSlipModel"},
      "contact wall-slip model");
  reject_duplicate_aliases(
      params,
      {"Wall_slip_length", "WallSlipLength", "Slip_length", "SlipLength"},
      "contact wall-slip length");

  if (any_parameter_defined(
          params,
          {"Contact_line_wall_faces", "ContactLineWallFaces",
           "Wall_boundary_faces", "WallBoundaryFaces",
           "Contact_line_wall_face", "ContactLineWallFace",
           "Wall_boundary_face", "WallBoundaryFace"})) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Contact wall face names are not supported by the typed free-surface contact configuration; provide explicit wall marker ids.");
  }

  const auto model = first_defined_string(
      params,
      {"Contact_line_model", "ContactLineModel",
       "Free_surface_contact_line_model", "FreeSurfaceContactLineModel"});
  const auto angle_radians = first_defined_double(
      params,
      {"Contact_angle_radians", "ContactAngleRadians",
       "Prescribed_contact_angle_radians", "PrescribedContactAngleRadians"});
  const auto angle_degrees = first_defined_double(
      params,
      {"Contact_angle_degrees", "ContactAngleDegrees",
       "Prescribed_contact_angle_degrees", "PrescribedContactAngleDegrees"});

  if (!model.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface contact configuration requires an explicit Contact_line_model (None, Pinned, PrescribedAngle, or DynamicRenE).");
  }
  if (angle_radians.has_value() && angle_degrees.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface contact configuration must specify the contact angle in either radians or degrees, not both.");
  }

  const auto parsed_model = parse_free_surface_contact_line_model(
      *model, "Free-surface Contact_line_model");

  const bool has_angle = angle_radians.has_value() || angle_degrees.has_value();
  const bool has_wall_marker = any_parameter_defined(
      params,
      {"Contact_line_wall_markers", "ContactLineWallMarkers",
       "Wall_boundary_markers", "WallBoundaryMarkers",
       "Contact_line_wall_marker", "ContactLineWallMarker",
       "Wall_boundary_marker", "WallBoundaryMarker"});
  const bool has_wall_normal = any_parameter_defined(
      params,
      {"Contact_line_wall_normals", "ContactLineWallNormals",
       "Contact_angle_wall_normals", "ContactAngleWallNormals",
       "Wall_normals", "WallNormals",
       "Contact_line_wall_normal", "ContactLineWallNormal",
       "Contact_angle_wall_normal", "ContactAngleWallNormal",
       "Wall_normal", "WallNormal"});
  const bool has_mobility = any_parameter_defined(
      params,
      {"Contact_line_mobility", "ContactLineMobility", "Mobility"});
  const bool has_slip_model = any_parameter_defined(
      params,
      {"Wall_slip_model", "WallSlipModel",
       "Contact_line_wall_slip_model", "ContactLineWallSlipModel"});
  const bool has_slip_length = any_parameter_defined(
      params,
      {"Wall_slip_length", "WallSlipLength", "Slip_length", "SlipLength"});
  const bool has_contact_marker = any_parameter_defined(
      params, {"Contact_line_marker", "ContactLineMarker"});
  const auto slip_model = first_defined_string(
      params,
      {"Wall_slip_model", "WallSlipModel",
       "Contact_line_wall_slip_model", "ContactLineWallSlipModel"});

  if (parsed_model == ParsedFreeSurfaceContactModel::None) {
    if (has_angle || has_wall_marker || has_wall_normal || has_mobility ||
        has_slip_model || has_slip_length ||
        has_contact_marker) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_model=None does not accept contact-angle, wall, mobility, slip, or contact-marker parameters.");
    }
    fs.contact_lines.push_back(ContactLine{
        .configuration = ContactLine::None{},
    });
    return;
  }
  if (parsed_model == ParsedFreeSurfaceContactModel::Pinned) {
    if (!has_wall_marker && !has_contact_marker) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_model=Pinned requires Contact_line_marker or a contact wall marker.");
    }
    if (has_angle || has_wall_normal || has_mobility ||
        has_slip_model || has_slip_length) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Contact_line_model=Pinned accepts only its contact-line or wall marker.");
    }
  } else {
    if (!has_angle || !has_wall_marker || !has_wall_normal) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] PrescribedAngle and DynamicRenE require an explicit angle, wall marker(s), and wall normal(s).");
    }
    if (parsed_model == ParsedFreeSurfaceContactModel::PrescribedAngle) {
      if (has_mobility) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] PrescribedAngle does not accept contact-line mobility.");
      }
      if (has_slip_model != has_slip_length) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] PrescribedAngle Navier slip requires both Wall_slip_model and Wall_slip_length, or neither.");
      }
      if (has_slip_model &&
          parse_free_surface_wall_slip_model(
              *slip_model, "Free-surface Wall_slip_model") !=
              ParsedFreeSurfaceWallSlipModel::Navier) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] PrescribedAngle wall slip requires Wall_slip_model=Navier.");
      }
    } else if (!has_mobility || !has_slip_model || !has_slip_length) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] DynamicRenE requires explicit mobility, wall-slip model, and wall-slip length parameters.");
    }
  }

  const auto wall_markers = first_defined_int_entries(
      params,
      {"Contact_line_wall_markers", "ContactLineWallMarkers",
       "Wall_boundary_markers", "WallBoundaryMarkers",
       "Contact_line_wall_marker", "ContactLineWallMarker",
       "Wall_boundary_marker", "WallBoundaryMarker"},
      "Free-surface Contact_line_wall_marker");
  const int contact_marker = first_defined_int(
      params, {"Contact_line_marker", "ContactLineMarker"})
                                 .value_or(-1);

  IncompressibleNavierStokesVMSOptions::ScalarValue contact_angle{
      svmp::FE::Real{1.57079632679489661923}};
  if (angle_radians.has_value()) {
    contact_angle = static_cast<svmp::FE::Real>(*angle_radians);
  } else if (angle_degrees.has_value()) {
    constexpr double pi = 3.14159265358979323846;
    contact_angle =
        static_cast<svmp::FE::Real>((*angle_degrees) * pi / 180.0);
  }

  const auto wall_normals = first_defined_vector3_entries(
      params,
      {"Contact_line_wall_normals", "ContactLineWallNormals",
       "Contact_angle_wall_normals", "ContactAngleWallNormals",
       "Wall_normals", "WallNormals",
       "Contact_line_wall_normal", "ContactLineWallNormal",
       "Contact_angle_wall_normal", "ContactAngleWallNormal",
       "Wall_normal", "WallNormal"},
      "Free-surface Contact_line_wall_normal");

  IncompressibleNavierStokesVMSOptions::ScalarValue mobility{
      svmp::FE::Real{0.0}};
  if (const auto parsed_mobility = first_defined_double(
          params,
          {"Contact_line_mobility", "ContactLineMobility", "Mobility"})) {
    mobility = static_cast<svmp::FE::Real>(*parsed_mobility);
  }

  if (parsed_model == ParsedFreeSurfaceContactModel::DynamicRenE &&
      (!slip_model.has_value() ||
       parse_free_surface_wall_slip_model(
           *slip_model, "Free-surface Wall_slip_model") !=
           ParsedFreeSurfaceWallSlipModel::Navier)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] DynamicRenE requires Wall_slip_model=Navier.");
  }

  IncompressibleNavierStokesVMSOptions::ScalarValue slip_length{
      svmp::FE::Real{0.0}};
  if (const auto parsed_slip_length = first_defined_double(
          params,
          {"Wall_slip_length", "WallSlipLength", "Slip_length", "SlipLength"})) {
    slip_length = static_cast<svmp::FE::Real>(*parsed_slip_length);
  }

  const std::size_t contact_line_count =
      wall_markers.has_value() ? wall_markers->size() : 1u;
  if (contact_line_count > 1u && has_contact_marker) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] A single Contact_line_marker cannot be shared by multiple contact walls; omit it to use generated markers.");
  }
  if (wall_normals.has_value() && wall_normals->size() != 1u &&
      wall_normals->size() != contact_line_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface Contact_line_wall_normals must contain either one vector or one vector per wall marker.");
  }
  for (std::size_t i = 0; i < contact_line_count; ++i) {
    const int wall_marker = wall_markers.has_value()
                                ? (*wall_markers)[i]
                                : -1;
    std::array<IncompressibleNavierStokesVMSOptions::ScalarValue, 3>
        wall_normal{svmp::FE::Real{0.0},
                    svmp::FE::Real{0.0},
                    svmp::FE::Real{0.0}};
    if (wall_normals.has_value()) {
      const auto& normal =
          wall_normals->size() == 1u ? (*wall_normals)[0] : (*wall_normals)[i];
      wall_normal = {normal[0], normal[1], normal[2]};
    }

    switch (parsed_model) {
    case ParsedFreeSurfaceContactModel::None:
      throw std::logic_error("None contact configuration expanded unexpectedly");
    case ParsedFreeSurfaceContactModel::Pinned:
      fs.contact_lines.push_back(ContactLine{
          .configuration = ContactLine::Pinned{
              .wall_boundary_marker = wall_marker,
              .contact_line_marker = contact_marker,
          },
      });
      break;
    case ParsedFreeSurfaceContactModel::PrescribedAngle:
      fs.contact_lines.push_back(ContactLine{
          .configuration = ContactLine::PrescribedAngle{
              .wall_boundary_marker = wall_marker,
              .contact_line_marker = contact_marker,
              .contact_angle_radians = contact_angle,
              .wall_normal = wall_normal,
              .slip_length = has_slip_length
                  ? std::optional<
                        IncompressibleNavierStokesVMSOptions::ScalarValue>{
                        slip_length}
                  : std::nullopt,
          },
      });
      break;
    case ParsedFreeSurfaceContactModel::DynamicRenE:
      fs.contact_lines.push_back(ContactLine{
          .configuration = ContactLine::DynamicRenE{
              .wall_boundary_marker = wall_marker,
              .contact_line_marker = contact_marker,
              .equilibrium_contact_angle_radians = contact_angle,
              .wall_normal = wall_normal,
              .mobility = mobility,
              .slip_length = slip_length,
          },
      });
      break;
    }
  }
}

void append_free_surface_bc(
    const svmp::Physics::BoundaryConditionInput& bc,
    bool is_steady,
    bool has_temp_spat,
    bool has_other_files,
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation;
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceKinematicEnforcement;
  using svmp::Physics::formulations::navier_stokes::FreeSurfaceTangentialMeshPolicy;
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  if (!is_steady) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "' currently supports only steady scalar parameters in the new solver Navier-Stokes input translator.");
  }
  if (has_temp_spat || has_other_files) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free_surface BC '" + bc.name +
        "' does not support spatial/temporal files in the new solver Navier-Stokes input translator.");
  }

  IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary fs{};
  const bool explicit_legacy_configuration =
      options.input_configuration_schema_version == 1 &&
      options.explicit_legacy_configuration;
  fs.implementation = free_surface_implementation_from_params(bc.params);
  if (const auto active_domain = first_defined_string(
          bc.params,
          {"Active_domain", "ActiveDomain",
           "Free_surface_active_domain", "FreeSurfaceActiveDomain"})) {
    fs.active_domain =
        parse_free_surface_active_domain(*active_domain, "Free-surface Active_domain");
  }
  if (const auto active_domain_method = first_defined_string(
          bc.params,
          {"Active_domain_method", "ActiveDomainMethod",
           "Free_surface_active_domain_method", "FreeSurfaceActiveDomainMethod"})) {
    fs.active_domain_method = parse_free_surface_active_domain_method(
        *active_domain_method, "Free-surface Active_domain_method");
  }
  if (const auto active_domain_smoothing_width = first_defined_double(
          bc.params,
          {"Active_domain_smoothing_width", "ActiveDomainSmoothingWidth",
           "Free_surface_active_domain_smoothing_width",
           "FreeSurfaceActiveDomainSmoothingWidth"})) {
    if (*active_domain_smoothing_width < 0.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free_surface Active_domain_smoothing_width must be nonnegative.");
    }
    fs.active_domain_smoothing_width =
        static_cast<svmp::FE::Real>(*active_domain_smoothing_width);
  }
  if (const auto allow_full_domain = first_defined_bool(
          bc.params,
          {"Allow_full_domain_unfitted_free_surface",
           "AllowFullDomainUnfittedFreeSurface",
           "Allow_full_domain_free_surface",
           "AllowFullDomainFreeSurface"})) {
    fs.allow_full_domain_unfitted_free_surface = *allow_full_domain;
  }
  if (fs.implementation == FreeSurfaceImplementation::FittedALE) {
    fs.boundary_marker = bc.boundary_marker;
  } else {
    const auto marker = first_defined_int(bc.params, {"Interface_marker", "InterfaceMarker"});
    if (marker.has_value()) {
      fs.interface_marker = *marker;
    }
    if (const auto field_name = first_defined_string(
            bc.params,
            {"Level_set_field_name", "Level_set_field", "LevelSetFieldName", "LevelSetField"})) {
      fs.level_set_field_name = *field_name;
    }
    if (const auto domain_id = first_defined_string(
            bc.params,
            {"Generated_interface_domain_id", "GeneratedInterfaceDomainId",
             "Interface_domain_id", "InterfaceDomainId"})) {
      fs.generated_interface_domain_id = *domain_id;
    }
    if (const auto geometry = first_defined_string(
            bc.params,
            {"Generated_interface_geometry", "GeneratedInterfaceGeometry",
             "Implicit_geometry_mode", "ImplicitGeometryMode",
             "Generated_interface_geometry_mode",
             "GeneratedInterfaceGeometryMode"})) {
      fs.generated_interface_geometry = *geometry;
    }
    if (const auto tangent_policy = first_defined_string(
            bc.params,
            {"Geometry_tangent_policy", "GeometryTangentPolicy",
             "Generated_interface_geometry_tangent_policy",
             "GeneratedInterfaceGeometryTangentPolicy",
             "Implicit_geometry_tangent_policy",
             "ImplicitGeometryTangentPolicy"})) {
      fs.geometry_tangent_policy = *tangent_policy;
    }
    if (const auto isovalue = first_defined_double(
            bc.params,
            {"Level_set_isovalue", "LevelSetIsovalue", "Interface_isovalue", "InterfaceIsovalue"})) {
      fs.level_set_isovalue = static_cast<svmp::FE::Real>(*isovalue);
    }
  }

  if (const auto pressure = first_defined_double(
          bc.params,
          {"External_pressure", "ExternalPressure", "Pressure", "Value"})) {
    fs.external_pressure = IncompressibleNavierStokesVMSOptions::ScalarValue{
        static_cast<svmp::FE::Real>(*pressure)};
  }
  if (const auto surface_tension = first_defined_double(
          bc.params,
          {"Surface_tension", "SurfaceTension"})) {
    fs.surface_tension = IncompressibleNavierStokesVMSOptions::ScalarValue{
        static_cast<svmp::FE::Real>(*surface_tension)};
  }
  if (const auto surface_tension_form = first_defined_string(
          bc.params,
          {"Surface_tension_form", "SurfaceTensionForm",
           "Free_surface_surface_tension_form",
           "FreeSurfaceSurfaceTensionForm",
           "Capillary_force_form", "CapillaryForceForm"})) {
    fs.surface_tension_form = parse_free_surface_surface_tension_form(
        *surface_tension_form, "Free-surface Surface_tension_form");
  }
  if (const auto curvature = first_defined_double(bc.params, {"Curvature"})) {
    fs.curvature = IncompressibleNavierStokesVMSOptions::ScalarValue{
        static_cast<svmp::FE::Real>(*curvature)};
  }
  if (const auto curvature_field = first_defined_string(
          bc.params,
          {"Curvature_field", "CurvatureField",
           "Projected_curvature_field", "ProjectedCurvatureField",
           "Free_surface_curvature_field", "FreeSurfaceCurvatureField"})) {
    fs.curvature_field_name = *curvature_field;
  }
  if (const auto use_level_set_curvature = first_defined_bool(
          bc.params,
          {"Use_level_set_curvature", "UseLevelSetCurvature"})) {
    fs.use_level_set_curvature = *use_level_set_curvature;
  }
  if (const auto use_current_geometry_curvature = first_defined_bool(
          bc.params,
          {"Use_current_geometry_curvature", "UseCurrentGeometryCurvature",
           "Use_fitted_current_geometry_curvature", "UseFittedCurrentGeometryCurvature"})) {
    fs.use_current_geometry_curvature = *use_current_geometry_curvature;
  }

  if (const auto normal_policy = first_defined_string(
          bc.params,
          {"Normal_kinematic_policy", "NormalKinematicPolicy",
           "Free_surface_normal_kinematic_policy", "FreeSurfaceNormalKinematicPolicy"})) {
    fs.normal_kinematic_policy = parse_free_surface_normal_kinematic_policy(
        *normal_policy, "Free-surface Normal_kinematic_policy");
  }
  const auto tangential_policy = first_defined_string(
          bc.params,
          {"Tangential_mesh_policy", "TangentialMeshPolicy",
           "Free_surface_tangential_mesh_policy", "FreeSurfaceTangentialMeshPolicy"});
  if (tangential_policy.has_value()) {
    fs.tangential_mesh_policy = parse_free_surface_tangential_mesh_policy(
        *tangential_policy, "Free-surface Tangential_mesh_policy");
  }
  const auto tangential_velocity = first_defined_string(
          bc.params,
          {"Prescribed_tangential_mesh_velocity", "PrescribedTangentialMeshVelocity",
           "Tangential_mesh_velocity", "TangentialMeshVelocity"});
  if (tangential_velocity.has_value()) {
    const auto v = parse_real_vector3(
        *tangential_velocity, "Free-surface Prescribed_tangential_mesh_velocity");
    fs.prescribed_tangential_mesh_velocity = {
        IncompressibleNavierStokesVMSOptions::ScalarValue{v[0]},
        IncompressibleNavierStokesVMSOptions::ScalarValue{v[1]},
        IncompressibleNavierStokesVMSOptions::ScalarValue{v[2]}};
    if (fs.tangential_mesh_policy != FreeSurfaceTangentialMeshPolicy::Prescribed) {
      if (tangential_policy.has_value() && !explicit_legacy_configuration) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Prescribed tangential mesh velocity "
            "conflicts with an explicit non-Prescribed tangential policy.");
      }
      fs.tangential_mesh_policy = FreeSurfaceTangentialMeshPolicy::Prescribed;
    }
  }
  const auto tangential_mesh_penalty = first_defined_double(
      bc.params,
      {"Tangential_mesh_penalty", "TangentialMeshPenalty",
       "Prescribed_tangential_mesh_penalty",
       "PrescribedTangentialMeshPenalty"});
  if (tangential_mesh_penalty.has_value()) {
    if (!std::isfinite(*tangential_mesh_penalty) ||
        !(*tangential_mesh_penalty > 0.0)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface tangential mesh penalty "
          "must be finite and positive.");
    }
    fs.tangential_mesh_penalty =
        IncompressibleNavierStokesVMSOptions::ScalarValue{
            static_cast<svmp::FE::Real>(*tangential_mesh_penalty)};
  }
  if (tangential_mesh_penalty.has_value() &&
      fs.tangential_mesh_policy !=
          FreeSurfaceTangentialMeshPolicy::Prescribed &&
      fs.tangential_mesh_policy !=
          FreeSurfaceTangentialMeshPolicy::SmoothingOnly &&
      !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Tangential_mesh_penalty requires "
        "Tangential_mesh_policy=Prescribed or SmoothingOnly; Free has no "
        "tangential boundary row, and unused tangential settings are "
        "accepted only by the explicit schema-1 legacy mode.");
  }

  const auto enforcement = first_defined_string(
      bc.params,
      {"Kinematic_enforcement", "KinematicEnforcement"});
  if (enforcement.has_value()) {
    fs.kinematic_enforcement =
        parse_free_surface_kinematic_enforcement(
            *enforcement, "Free-surface Kinematic_enforcement");
  }
  const auto penalty = first_defined_double(
      bc.params,
      {"Kinematic_penalty", "KinematicPenalty"});
  if (penalty.has_value()) {
    fs.kinematic_penalty = IncompressibleNavierStokesVMSOptions::ScalarValue{
        static_cast<svmp::FE::Real>(*penalty)};
    if (!enforcement.has_value() && explicit_legacy_configuration) {
      fs.kinematic_enforcement = FreeSurfaceKinematicEnforcement::Penalty;
    }
  }
  if (penalty.has_value() &&
      fs.kinematic_enforcement != FreeSurfaceKinematicEnforcement::Penalty &&
      !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Kinematic_penalty requires explicit "
        "Kinematic_enforcement=Penalty; it cannot promote None and unused "
        "penalty settings are accepted only by the explicit schema-1 "
        "legacy mode.");
  }
  const auto kinematic_nitsche_gamma = first_defined_double(
          bc.params,
          {"Kinematic_nitsche_gamma", "KinematicNitscheGamma",
           "Free_surface_nitsche_gamma", "FreeSurfaceNitscheGamma",
           "Nitsche_gamma", "NitscheGamma"});
  if (kinematic_nitsche_gamma.has_value()) {
    fs.kinematic_nitsche_gamma =
        static_cast<svmp::FE::Real>(*kinematic_nitsche_gamma);
  }
  const auto kinematic_nitsche_symmetric = first_defined_bool(
          bc.params,
          {"Kinematic_nitsche_symmetric", "KinematicNitscheSymmetric",
           "Nitsche_symmetric", "NitscheSymmetric"});
  if (kinematic_nitsche_symmetric.has_value()) {
    fs.kinematic_nitsche_symmetric = *kinematic_nitsche_symmetric;
  }
  const auto kinematic_nitsche_scale_with_p = first_defined_bool(
          bc.params,
          {"Kinematic_nitsche_scale_with_p", "KinematicNitscheScaleWithP",
           "Nitsche_scale_with_p", "NitscheScaleWithP"});
  if (kinematic_nitsche_scale_with_p.has_value()) {
    fs.kinematic_nitsche_scale_with_p = *kinematic_nitsche_scale_with_p;
  }
  if ((kinematic_nitsche_gamma.has_value() ||
       kinematic_nitsche_symmetric.has_value() ||
       kinematic_nitsche_scale_with_p.has_value()) &&
      fs.kinematic_enforcement != FreeSurfaceKinematicEnforcement::Nitsche &&
      !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary-local free-surface Nitsche settings require Kinematic_enforcement=Nitsche; unused Nitsche settings are accepted only by the explicit schema-1 legacy mode.");
  }

  const auto small_cut_aggregation = first_defined_bool(
          bc.params,
          {"Small_cut_aggregation", "SmallCutAggregation",
           "Enable_small_cut_aggregation", "EnableSmallCutAggregation"});
  if (small_cut_aggregation.has_value()) {
    fs.small_cut_aggregation = *small_cut_aggregation;
  }

  bool aggregation_guard_suboption_present = false;
  if (const auto maximum_root_path = first_defined_int(
          bc.params,
          {"Small_cut_aggregation_maximum_root_path_length",
           "SmallCutAggregationMaximumRootPathLength"})) {
    aggregation_guard_suboption_present = true;
    if (*maximum_root_path <= 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
          "maximum root path length must be positive.");
    }
    fs.small_cut_aggregation_guards.maximum_root_path_length =
        static_cast<std::size_t>(*maximum_root_path);
  }
  if (const auto maximum_extrapolation = first_defined_double(
          bc.params,
          {"Small_cut_aggregation_maximum_reference_extrapolation_distance",
           "SmallCutAggregationMaximumReferenceExtrapolationDistance"})) {
    aggregation_guard_suboption_present = true;
    if (!std::isfinite(*maximum_extrapolation) ||
        *maximum_extrapolation < 0.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
          "maximum reference extrapolation distance must be finite and "
          "nonnegative.");
    }
    fs.small_cut_aggregation_guards
        .maximum_reference_extrapolation_distance =
        static_cast<svmp::FE::Real>(*maximum_extrapolation);
  }
  if (const auto maximum_coefficient = first_defined_double(
          bc.params,
          {"Small_cut_aggregation_maximum_absolute_coefficient",
           "SmallCutAggregationMaximumAbsoluteCoefficient"})) {
    aggregation_guard_suboption_present = true;
    if (!std::isfinite(*maximum_coefficient) ||
        *maximum_coefficient < 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
          "maximum absolute coefficient must be finite and at least 1.");
    }
    fs.small_cut_aggregation_guards.maximum_absolute_coefficient =
        static_cast<svmp::FE::Real>(*maximum_coefficient);
  }
  if (const auto maximum_row_norm = first_defined_double(
          bc.params,
          {"Small_cut_aggregation_maximum_row_l1_norm",
           "SmallCutAggregationMaximumRowL1Norm"})) {
    aggregation_guard_suboption_present = true;
    if (!std::isfinite(*maximum_row_norm) || *maximum_row_norm < 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
          "maximum row L1 norm must be finite and at least 1.");
    }
    fs.small_cut_aggregation_guards.maximum_row_l1_norm =
        static_cast<svmp::FE::Real>(*maximum_row_norm);
  }
  if (fs.small_cut_aggregation_guards.maximum_row_l1_norm <
      fs.small_cut_aggregation_guards.maximum_absolute_coefficient) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Free-surface small-cut aggregation "
        "maximum row L1 norm must be no smaller than the maximum absolute "
        "coefficient.");
  }
  if (!fs.small_cut_aggregation && aggregation_guard_suboption_present &&
      !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Small_cut_aggregation=false cannot be "
        "combined with aggregation guard settings; unused guards are "
        "accepted only by the explicit schema-1 legacy mode.");
  }

  const auto cut_cell_stabilization_enabled = first_defined_bool(
          bc.params,
          {"Enable_cut_cell_stabilization", "EnableCutCellStabilization",
           "Cut_cell_stabilization", "CutCellStabilization"});
  const bool cut_cell_stabilization_explicitly_disabled =
      cut_cell_stabilization_enabled.has_value() &&
      !*cut_cell_stabilization_enabled;
  bool cut_cell_suboption_present = false;
  if (cut_cell_stabilization_enabled.has_value()) {
    fs.cut_cell_stabilization.enabled = *cut_cell_stabilization_enabled;
  }
  if (first_defined_double(
          bc.params,
          {"Cut_cell_velocity_gradient_penalty", "CutCellVelocityGradientPenalty",
           "Velocity_gradient_ghost_penalty", "VelocityGradientGhostPenalty"})) {
    if (!explicit_legacy_configuration) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Cut-cell velocity ghost-penalty settings are retired; small-cut aggregation replaces them. Only explicit schema-1 legacy input may retain an ignored archived value.");
    }
    std::cerr << "[svMultiPhysics::Physics] WARNING: "
                 "Cut_cell_velocity_gradient_penalty is deprecated and "
                 "ignored by explicit schema-1 legacy mode; small-cut aggregation replaces the velocity "
                 "ghost penalty." << std::endl;
  }
  if (const auto pressure_penalty = first_defined_double(
          bc.params,
          {"Cut_cell_pressure_gradient_penalty", "CutCellPressureGradientPenalty",
           "Pressure_gradient_ghost_penalty", "PressureGradientGhostPenalty"})) {
    cut_cell_suboption_present = true;
    fs.cut_cell_stabilization.pressure_gradient_penalty =
        IncompressibleNavierStokesVMSOptions::ScalarValue{
            static_cast<svmp::FE::Real>(*pressure_penalty)};
    if (!cut_cell_stabilization_explicitly_disabled) {
      fs.cut_cell_stabilization.enabled = true;
    }
  }
  if (const auto pressure_policy = first_defined_string(
          bc.params,
          {"Cut_cell_pressure_stabilization_policy",
           "CutCellPressureStabilizationPolicy",
           "Pressure_ghost_penalty_policy",
           "PressureGhostPenaltyPolicy"})) {
    cut_cell_suboption_present = true;
    fs.cut_cell_stabilization.pressure_policy =
        parse_free_surface_pressure_stabilization_policy(
            *pressure_policy,
            "Free-surface Cut_cell_pressure_stabilization_policy");
  }
  if (const auto use_cut_scale = first_defined_bool(
          bc.params,
          {"Use_cut_metadata_scale", "UseCutMetadataScale",
           "Use_cut_stabilization_scale", "UseCutStabilizationScale"})) {
    cut_cell_suboption_present = true;
    fs.cut_cell_stabilization.use_cut_metadata_scale = *use_cut_scale;
  }
  if (const auto cut_scale_cap = first_defined_double(
          bc.params,
          {"Cut_cell_metadata_scale_cap", "CutCellMetadataScaleCap",
           "Cut_cell_stabilization_scale_cap", "CutCellStabilizationScaleCap",
           "Cut_metadata_scale_cap", "CutMetadataScaleCap"})) {
    cut_cell_suboption_present = true;
    if (!std::isfinite(*cut_scale_cap) || *cut_scale_cap < 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Free-surface "
          "Cut_cell_metadata_scale_cap must be finite and at least 1.");
    }
    fs.cut_cell_stabilization.cut_metadata_scale_cap =
        static_cast<svmp::FE::Real>(*cut_scale_cap);
  }
  if (cut_cell_stabilization_explicitly_disabled &&
      cut_cell_suboption_present && !explicit_legacy_configuration) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Enable_cut_cell_stabilization=false cannot be combined with cut-cell stabilization suboptions; unused stabilization settings are accepted only by the explicit schema-1 legacy mode.");
  }
  if (!cut_cell_stabilization_enabled.has_value() &&
      cut_cell_suboption_present) {
    fs.cut_cell_stabilization.enabled = true;
  }
  if (first_defined_int(
          bc.params,
          {"Cut_cell_velocity_max_derivative_order",
           "CutCellVelocityMaxDerivativeOrder",
           "Velocity_ghost_penalty_max_derivative_order",
           "VelocityGhostPenaltyMaxDerivativeOrder",
           "Velocity_gradient_ghost_penalty_max_derivative_order",
           "VelocityGradientGhostPenaltyMaxDerivativeOrder"})) {
    if (!explicit_legacy_configuration) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Cut-cell velocity derivative-order settings are retired with the velocity ghost penalty. Only explicit schema-1 legacy input may retain an ignored archived value.");
    }
    std::cerr << "[svMultiPhysics::Physics] WARNING: "
                 "Cut_cell_velocity_max_derivative_order is deprecated and "
                 "ignored by explicit schema-1 legacy mode (velocity ghost penalty retired)." << std::endl;
  }

  const auto velocity_extension_enabled = first_defined_bool(
      bc.params,
      {"Enable_velocity_extension", "EnableVelocityExtension",
       "Velocity_extension", "VelocityExtension",
       "Extend_velocity_to_inactive_domain",
       "ExtendVelocityToInactiveDomain",
       "Free_surface_velocity_extension",
       "FreeSurfaceVelocityExtension"});
  if (velocity_extension_enabled.has_value()) {
    fs.velocity_extension.enabled = *velocity_extension_enabled;
  }
  const bool velocity_extension_explicitly_disabled =
      velocity_extension_enabled.has_value() &&
      !*velocity_extension_enabled;
  if (const auto velocity_extension_diffusivity = first_defined_double(
          bc.params,
          {"Velocity_extension_diffusivity", "VelocityExtensionDiffusivity",
           "Inactive_velocity_extension_diffusivity",
           "InactiveVelocityExtensionDiffusivity"})) {
    if (velocity_extension_explicitly_disabled &&
        !explicit_legacy_configuration) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Velocity_extension_diffusivity cannot accompany an explicitly disabled velocity extension; unused extension settings are accepted only by the explicit schema-1 legacy mode.");
    }
    fs.velocity_extension.diffusivity =
        IncompressibleNavierStokesVMSOptions::ScalarValue{
            static_cast<svmp::FE::Real>(*velocity_extension_diffusivity)};
    if (!velocity_extension_explicitly_disabled) {
      fs.velocity_extension.enabled = true;
    }
  }

  append_free_surface_contact_line(bc.params, fs);
  options.free_surface.push_back(std::move(fs));
}

svmp::Physics::formulations::navier_stokes::
    IncompressibleNavierStokesVMSOptions
translate_fitted_surface_contact_capability(
    const svmp::Physics::EquationModuleInput& input)
{
  using svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSOptions;

  IncompressibleNavierStokesVMSOptions options{};
  apply_free_surface_schema_options(input.equation_params, options);
  for (const auto& bc : input.boundary_conditions) {
    const auto* bc_type = find_param(bc.params, "Type");
    if (bc_type == nullptr || !bc_type->defined ||
        !is_free_surface_type(bc_type->value)) {
      continue;
    }

    const auto* time_dependence =
        find_param(bc.params, "Time_dependence");
    const auto time_value =
        time_dependence != nullptr && time_dependence->defined
            ? lower_copy(trim_copy(time_dependence->value))
            : std::string{"steady"};
    const bool is_steady = time_value.empty() || time_value == "steady";
    const bool has_temporal_spatial_file = has_nonempty_defined(
        bc.params, "Temporal_and_spatial_values_file_path");
    const bool has_other_files =
        has_nonempty_defined(bc.params, "Temporal_values_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_values_file_path") ||
        has_nonempty_defined(bc.params, "Bct_file_path") ||
        has_nonempty_defined(bc.params, "Traction_values_file_path") ||
        has_nonempty_defined(bc.params, "Fourier_coefficients_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_profile_file_path");
    append_free_surface_bc(
        bc,
        is_steady,
        has_temporal_spatial_file,
        has_other_files,
        options);
  }
  return options;
}

void validate_fitted_surface_contact_capability(
    const svmp::Physics::formulations::navier_stokes::
        IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::
      FreeSurfaceImplementation;
  using svmp::Physics::formulations::navier_stokes::
      FreeSurfaceSurfaceTensionForm;
  using ContactLine = svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;

  for (const auto& boundary : options.free_surface) {
    if (boundary.implementation != FreeSurfaceImplementation::FittedALE) {
      continue;
    }
    if (boundary.surface_tension_form ==
        FreeSurfaceSurfaceTensionForm::SurfaceStress) {
      throw std::invalid_argument(
          "IncompressibleNavierStokesVMSModule: fitted-ALE SurfaceStress is not yet qualified for current-frame test-function gradients; use Automatic/CurvatureTraction for fitted boundaries");
    }
    if (boundary.surface_tension_form ==
        FreeSurfaceSurfaceTensionForm::GeneratedCurvatureTraction) {
      throw std::invalid_argument(
          "IncompressibleNavierStokesVMSModule: GeneratedCurvatureTraction is available only for unfitted level-set free surfaces; use Automatic/CurvatureTraction for fitted boundaries");
    }
    if (boundary.surface_tension_form ==
        FreeSurfaceSurfaceTensionForm::KinematicAreaGradientTraction) {
      throw std::invalid_argument(
          "IncompressibleNavierStokesVMSModule: KinematicAreaGradientTraction is available only for unfitted level-set free surfaces; use Automatic/CurvatureTraction for fitted boundaries");
    }
    for (const auto& contact_line : boundary.contact_lines) {
      if (std::holds_alternative<ContactLine::PrescribedAngle>(
              contact_line.configuration)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed fitted contact angles are unsupported until a true fitted contact-line (codimension-two) integration entity is available; the condition must not be integrated over the complete free-surface boundary");
      }
      if (std::holds_alternative<ContactLine::DynamicRenE>(
              contact_line.configuration)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: DynamicContactAngle is currently supported only for sharp unfitted level-set free surfaces");
      }
    }
  }
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

  svmp::FE::Real backflow_beta = 0.0;
  if (const auto* p = find_param(domain.params, "Backflow_stabilization_coefficient")) {
    if (!trim_copy(p->value).empty()) {
      backflow_beta = static_cast<svmp::FE::Real>(parse_double(p->value, "Backflow_stabilization_coefficient"));
    }
  }

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
      append_free_surface_bc(bc, is_steady, has_temp_spat, has_other_files, options);
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

struct TranslatedIncompressibleTwoFluidInput {
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> velocity_space{};
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> pressure_space{};
  svmp::Physics::formulations::navier_stokes::
      IncompressibleTwoFluidOptions options{};
};

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
  if (input.node_pressure_constraints.has_value()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_node_pressure_constraints");
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
  const bool taylor_hood = first_defined_bool(
      input.equation_params,
      {"Use_taylor_hood_type_basis"}).value_or(false);
  if (!supported_cell || order != 1 || taylor_hood) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Incompressible two-fluid input requires affine equal-order P1 Triangle3 or Tetra4 spaces.");
  }

  Options options;
  options.level_set_field_name = require_two_fluid_string(
      input.equation_params,
      {"Level_set_field_name", "LevelSetFieldName"},
      "Level_set_field_name");
  options.generated_interface_domain_id = require_two_fluid_string(
      input.equation_params,
      {"Generated_interface_domain_id", "GeneratedInterfaceDomainId"},
      "Generated_interface_domain_id");
  options.interface_marker = require_two_fluid_int(
      input.equation_params,
      {"Material_interface_marker", "MaterialInterfaceMarker"},
      "Material_interface_marker");
  options.negative_phase.density = require_two_fluid_real(
      input.equation_params,
      {"Negative_phase_density", "NegativePhaseDensity"},
      "Negative_phase_density");
  options.negative_phase.viscosity = require_two_fluid_real(
      input.equation_params,
      {"Negative_phase_dynamic_viscosity",
       "NegativePhaseDynamicViscosity"},
      "Negative_phase_dynamic_viscosity");
  options.positive_phase.density = require_two_fluid_real(
      input.equation_params,
      {"Positive_phase_density", "PositivePhaseDensity"},
      "Positive_phase_density");
  options.positive_phase.viscosity = require_two_fluid_real(
      input.equation_params,
      {"Positive_phase_dynamic_viscosity",
       "PositivePhaseDynamicViscosity"},
      "Positive_phase_dynamic_viscosity");
  options.surface_tension = require_two_fluid_real(
      input.equation_params,
      {"Two_fluid_surface_tension", "TwoFluidSurfaceTension"},
      "Two_fluid_surface_tension");
  options.interface_nitsche_gamma = require_two_fluid_real(
      input.equation_params,
      {"Two_fluid_interface_nitsche_gamma",
       "TwoFluidInterfaceNitscheGamma"},
      "Two_fluid_interface_nitsche_gamma");
  if (any_parameter_defined(
          input.equation_params,
          {"Prescribed_pressure_jump", "PrescribedPressureJump"})) {
    options.prescribed_pressure_jump = require_two_fluid_real(
        input.equation_params,
        {"Prescribed_pressure_jump", "PrescribedPressureJump"},
        "Prescribed_pressure_jump");
  }
  if (any_parameter_defined(
          input.equation_params,
          {"Prescribed_viscous_traction_jump_x",
           "PrescribedViscousTractionJumpX",
           "Prescribed_viscous_traction_jump_y",
           "PrescribedViscousTractionJumpY",
           "Prescribed_viscous_traction_jump_z",
           "PrescribedViscousTractionJumpZ"})) {
    if (dimension == 2 &&
        any_parameter_defined(
            input.equation_params,
            {"Prescribed_viscous_traction_jump_z",
             "PrescribedViscousTractionJumpZ"})) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] "
          "unsupported_two_fluid_out_of_plane_prescribed_viscous_traction_jump");
    }
    std::array<svmp::FE::Real, 3> target{};
    target[0] = require_two_fluid_real(
        input.equation_params,
        {"Prescribed_viscous_traction_jump_x",
         "PrescribedViscousTractionJumpX"},
        "Prescribed_viscous_traction_jump_x");
    target[1] = require_two_fluid_real(
        input.equation_params,
        {"Prescribed_viscous_traction_jump_y",
         "PrescribedViscousTractionJumpY"},
        "Prescribed_viscous_traction_jump_y");
    if (dimension == 3) {
      target[2] = require_two_fluid_real(
          input.equation_params,
          {"Prescribed_viscous_traction_jump_z",
           "PrescribedViscousTractionJumpZ"},
          "Prescribed_viscous_traction_jump_z");
    }
    options.prescribed_viscous_traction_jump = target;
  }
  options.require_conservative_phase_momentum_reconciliation = true;
  if (options.interface_marker < 0) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Material_interface_marker must be nonnegative for incompressible two-fluid input.");
  }
  if (any_parameter_defined(
          input.equation_params, {"Operator_tag", "OperatorTag"})) {
    options.operator_tag = require_two_fluid_string(
        input.equation_params,
        {"Operator_tag", "OperatorTag"},
        "Operator_tag");
  }
  options.enable_convection = input.equation_type != "stokes";
  options.jit_policy =
      svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);

  for (std::size_t component = 0u; component < options.body_force.size();
       ++component) {
    static constexpr std::array<std::string_view, 3> force_keys{
        "Force_x", "Force_y", "Force_z"};
    if (const auto value = get_defined_double(
            input.default_domain.params, force_keys[component])) {
      options.body_force[component] = static_cast<svmp::FE::Real>(*value);
    }
  }

  BoundaryOptions translated_boundaries;
  apply_fluid_moving_domain_options(
      input, input.default_domain, translated_boundaries);
  if (translated_boundaries.enable_ale) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] unsupported_two_fluid_moving_mesh");
  }
  apply_fluid_momentum_source_params(
      input.equation_params, translated_boundaries);
  apply_fluid_momentum_source_params(
      input.default_domain.params, translated_boundaries);
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

std::unique_ptr<svmp::Physics::PhysicsModule>
create_incompressible_two_fluid_from_input(
    const svmp::Physics::EquationModuleInput& input,
    svmp::FE::systems::FESystem& system)
{
  using Module = svmp::Physics::formulations::navier_stokes::
      IncompressibleTwoFluidModule;
  auto translated = translate_incompressible_two_fluid_input(input, system);
  auto module = std::make_unique<Module>(
      translated.velocity_space,
      translated.pressure_space,
      translated.velocity_space,
      translated.pressure_space,
      std::move(translated.options));
  module->registerOn(system);
  return module;
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_navier_stokes_from_input(const svmp::Physics::EquationModuleInput& input,
                                svmp::FE::systems::FESystem& system)
{
  svmp::Physics::formulations::navier_stokes::
      preflightFittedSurfaceContactCapability(input);
  const auto physical_model =
      validate_and_resolve_free_surface_physical_model(input);
  if (physical_model ==
      NavierStokesPhysicalModel::IncompressibleTwoFluid) {
    return create_incompressible_two_fluid_from_input(input, system);
  }
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

  bool taylor_hood = false;
  if (const auto* p = find_param(input.equation_params, "Use_taylor_hood_type_basis"); p && p->defined) {
    taylor_hood = parse_bool_relaxed(p->value);
  }
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
  if (const auto operator_tag = first_defined_string(
          input.equation_params, {"Operator_tag", "OperatorTag"})) {
    options.operator_tag = *operator_tag;
  }
  options.enable_convection = (input.equation_type != "stokes");
  options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);

  apply_free_surface_schema_options(input.equation_params, options);
  apply_fluid_moving_domain_options(input, domain, options);
  apply_fluid_momentum_source_params(input.equation_params, options);
  apply_fluid_properties(domain, options);
  apply_fluid_momentum_source_spacetime_file(
      input, domain, dim, markerCommunicator(system), options);
  apply_fluid_rotating_frame_coriolis(input, domain, dim, options);
  apply_node_pressure_constraints(input, options);
  apply_fluid_bcs(input, domain, markerCommunicator(system), options);

  auto module = std::make_unique<svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule>(
      std::move(velocity_space), std::move(pressure_space), std::move(options));
  module->registerOn(system);
  return module;
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

} // namespace

SVMP_REGISTER_EQUATION("fluid", &create_navier_stokes_from_input);
SVMP_REGISTER_EQUATION("stokes", &create_navier_stokes_from_input);

namespace svmp::Physics::formulations::navier_stokes {

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

  const IncompressibleTwoFluidInterfaceParameters parameters{
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
  const auto weights = incompressibleTwoFluidInterfaceWeights(parameters);
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

void forceLink_NavierStokesRegister() {}

} // namespace svmp::Physics::formulations::navier_stokes
