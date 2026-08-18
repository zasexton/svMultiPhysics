#include "Physics/Formulations/Ustruct/UstructModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/TemporalValues.h"
#include "Physics/Materials/Solid/IsochoricNeoHookeanPK1.h"

#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct SolidMaterialProperties {
  svmp::FE::Real density{1.0};
  svmp::FE::Real elasticity_modulus{1.0};
  svmp::FE::Real poisson_ratio{0.49};
  svmp::FE::Real shear_modulus{0.0};
  svmp::FE::Real bulk_modulus{0.0};
  svmp::FE::Real bulk_wave_speed{0.0};
};

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

std::string normalize_output_name(std::string s)
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

bool parse_bool_relaxed(std::string_view raw)
{
  const auto v = lower_copy(trim_copy(std::string(raw)));
  return v == "true" || v == "1" || v == "yes" || v == "on";
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

std::optional<double> get_defined_double(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p || !p->defined) {
    return std::nullopt;
  }
  return parse_double(p->value, key);
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

std::optional<bool> get_defined_bool(const svmp::Physics::ParameterMap& params, std::string_view key)
{
  const auto* p = find_param(params, key);
  if (!p || !p->defined) {
    return std::nullopt;
  }
  const auto value = trim_copy(p->value);
  if (value.empty()) {
    return std::nullopt;
  }
  return parse_bool_relaxed(value);
}

std::optional<bool> module_option_bool(std::string_view module_options,
                                       std::initializer_list<std::string_view> keys)
{
  if (module_options.empty()) {
    return std::nullopt;
  }

  std::string normalized(module_options);
  for (char& ch : normalized) {
    if (ch == ';' || ch == '\n' || ch == '\t') {
      ch = ',';
    }
  }

  std::size_t start = 0;
  while (start < normalized.size()) {
    const std::size_t end = normalized.find(',', start);
    const std::string token = trim_copy(normalized.substr(start, end - start));
    if (!token.empty()) {
      const std::size_t sep = token.find_first_of("=:");
      if (sep != std::string::npos) {
        const auto key = lower_copy(trim_copy(token.substr(0, sep)));
        for (const auto candidate : keys) {
          if (key == candidate) {
            return parse_bool_relaxed(token.substr(sep + 1));
          }
        }
      }
    }

    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }

  return std::nullopt;
}

bool is_spatial_output_request(const svmp::Physics::OutputRequestInput& output)
{
  return lower_copy(trim_copy(output.type)) == "spatial";
}

std::optional<bool> spatial_output_bool(const svmp::Physics::EquationModuleInput& input,
                                        std::initializer_list<std::string_view> normalized_keys)
{
  std::optional<bool> value{};
  for (const auto& output : input.outputs) {
    if (!is_spatial_output_request(output)) {
      continue;
    }

    for (const auto& [raw_name, param] : output.params) {
      if (!param.defined) {
        continue;
      }

      const auto name = normalize_output_name(raw_name);
      for (const auto key : normalized_keys) {
        if (name == key) {
          value = parse_bool_relaxed(param.value);
        }
      }
    }
  }
  return value;
}

bool resolve_ustruct_time_derivative_terms(const svmp::Physics::EquationModuleInput& input,
                                           bool default_enabled)
{
  if (const auto value = module_option_bool(
          input.module_options,
          {"enable_time_derivative_terms", "time_derivative_terms", "time_derivatives", "dynamic"})) {
    return *value;
  }
  if (const auto value = module_option_bool(
          input.module_options,
          {"quasi_static", "quasistatic", "static"})) {
    return !*value;
  }
  return default_enabled;
}

std::vector<int> parse_int_list(std::string raw)
{
  for (char& ch : raw) {
    if (ch == '(' || ch == ')' || ch == ',' || ch == ';') {
      ch = ' ';
    }
  }

  std::istringstream in(raw);
  std::vector<int> out;
  int value = 0;
  while (in >> value) {
    out.push_back(value);
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

template <class ScalarValue>
void fill_vector(std::array<ScalarValue, 3>& dst,
                 int dim,
                 const std::vector<int>& effective_dir,
                 svmp::FE::Real magnitude)
{
  dst = {ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
  for (int d = 0; d < dim; ++d) {
    const auto scale = effective_dir.empty()
                           ? static_cast<svmp::FE::Real>(1.0)
                           : direction_component(effective_dir, d);
    dst[static_cast<std::size_t>(d)] = ScalarValue{static_cast<svmp::FE::Real>(magnitude * scale)};
  }
}

template <class ScalarValue>
std::array<bool, 3> fill_dirichlet_vector(std::array<ScalarValue, 3>& dst,
                                          int dim,
                                          const std::vector<int>& effective_dir,
                                          svmp::FE::Real magnitude)
{
  std::array<bool, 3> active{false, false, false};
  dst = {ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
  for (int d = 0; d < dim; ++d) {
    const auto scale = effective_dir.empty()
                           ? static_cast<svmp::FE::Real>(1.0)
                           : direction_component(effective_dir, d);
    active[static_cast<std::size_t>(d)] = (scale != static_cast<svmp::FE::Real>(0.0));
    if (active[static_cast<std::size_t>(d)]) {
      dst[static_cast<std::size_t>(d)] = ScalarValue{static_cast<svmp::FE::Real>(magnitude * scale)};
    }
  }
  return active;
}

const svmp::Physics::DomainInput&
select_single_domain(const svmp::Physics::EquationModuleInput& input)
{
  if (!input.domains.empty()) {
    if (input.domains.size() != 1u) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Multiple <Domain> blocks are not supported for the new solver ustruct module yet. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
    return input.domains.front();
  }
  return input.default_domain;
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
    case svmp::CellFamily::Triangle: return svmp::FE::ElementType::Triangle3;
    case svmp::CellFamily::Quad: return svmp::FE::ElementType::Quad4;
    case svmp::CellFamily::Tetra: return svmp::FE::ElementType::Tetra4;
    case svmp::CellFamily::Hex: return svmp::FE::ElementType::Hex8;
    case svmp::CellFamily::Wedge: return svmp::FE::ElementType::Wedge6;
    case svmp::CellFamily::Pyramid: return svmp::FE::ElementType::Pyramid5;
    default: break;
  }

  throw std::runtime_error(
      "[svMultiPhysics::Physics] Unsupported mesh cell family for new solver ustruct module. "
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

svmp::Physics::formulations::ustruct::VolumetricPenaltyModel
parse_volumetric_model(std::string_view raw)
{
  const auto value = lower_copy(trim_copy(std::string(raw)));
  if (value.empty() || value == "st91" || value == "simo-taylor91") {
    return svmp::Physics::formulations::ustruct::VolumetricPenaltyModel::ST91;
  }
  if (value == "quad" || value == "quadratic") {
    return svmp::Physics::formulations::ustruct::VolumetricPenaltyModel::Quadratic;
  }
  if (value == "m94" || value == "miehe94") {
    return svmp::Physics::formulations::ustruct::VolumetricPenaltyModel::M94;
  }
  if (value == "none" || value == "na") {
    return svmp::Physics::formulations::ustruct::VolumetricPenaltyModel::None;
  }
  throw std::runtime_error("[svMultiPhysics::Physics] Unsupported ustruct Dilational_penalty_model '" +
                           std::string(raw) + "'. Supported values: ST91, quadratic, M94, none.");
}

SolidMaterialProperties parse_solid_properties(const svmp::Physics::DomainInput& domain)
{
  SolidMaterialProperties props{};

  if (const auto model = get_defined_string(domain.params, "Constitutive_model.type")) {
    const auto lower = lower_copy(*model);
    if (lower != "neohookean" && lower != "nhk") {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] The new solver ustruct module currently supports only "
          "Constitutive_model type='neoHookean'.");
    }
  }

  if (const auto rho = get_defined_double(domain.params, "Solid_density")) {
    props.density = static_cast<svmp::FE::Real>(*rho);
  } else if (const auto rho_generic = get_defined_double(domain.params, "Density")) {
    props.density = static_cast<svmp::FE::Real>(*rho_generic);
  }

  if (const auto E = get_defined_double(domain.params, "Elasticity_modulus")) {
    props.elasticity_modulus = static_cast<svmp::FE::Real>(*E);
  }
  if (const auto nu = get_defined_double(domain.params, "Poisson_ratio")) {
    props.poisson_ratio = static_cast<svmp::FE::Real>(*nu);
  }

  if (!(props.density > 0.0)) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct density must be > 0.");
  }
  if (!(props.elasticity_modulus > 0.0)) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct Elasticity_modulus must be > 0.");
  }
  if (!(props.poisson_ratio > -1.0 && props.poisson_ratio <= 0.5)) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct Poisson_ratio must be in (-1, 0.5].");
  }

  props.shear_modulus = static_cast<svmp::FE::Real>(0.5) * props.elasticity_modulus /
                        (static_cast<svmp::FE::Real>(1.0) + props.poisson_ratio);

  if (std::abs(props.poisson_ratio - static_cast<svmp::FE::Real>(0.5)) <=
      static_cast<svmp::FE::Real>(1e-12)) {
    props.bulk_modulus = 0.0;
    props.bulk_wave_speed = std::sqrt(props.shear_modulus / props.density);
  } else {
    props.bulk_modulus = props.elasticity_modulus /
                         (static_cast<svmp::FE::Real>(3.0) *
                          (static_cast<svmp::FE::Real>(1.0) -
                           static_cast<svmp::FE::Real>(2.0) * props.poisson_ratio));
    const auto lambda = props.elasticity_modulus * props.poisson_ratio /
                        ((static_cast<svmp::FE::Real>(1.0) + props.poisson_ratio) *
                         (static_cast<svmp::FE::Real>(1.0) -
                          static_cast<svmp::FE::Real>(2.0) * props.poisson_ratio));
    props.bulk_wave_speed =
        std::sqrt((lambda + static_cast<svmp::FE::Real>(2.0) * props.shear_modulus) / props.density);
  }

  return props;
}

void apply_solid_options(const svmp::Physics::DomainInput& domain,
                         const SolidMaterialProperties& props,
                         svmp::Physics::formulations::ustruct::UstructOptions& options)
{
  options.density = props.density;
  options.bulk_wave_speed = props.bulk_wave_speed;
  options.kinematic_residual_scale = std::sqrt(props.density * props.shear_modulus);
  options.deviatoric_pk1_model =
      std::make_shared<svmp::Physics::materials::solid::IsochoricNeoHookeanPK1>(props.shear_modulus);

  if (const auto penalty = get_defined_double(domain.params, "Penalty_parameter")) {
    options.penalty_parameter = static_cast<svmp::FE::Real>(*penalty);
  } else {
    options.penalty_parameter = props.bulk_modulus;
  }
  if (options.penalty_parameter < 0.0) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct Penalty_parameter must be >= 0.");
  }

  if (const auto model = get_defined_string(domain.params, "Dilational_penalty_model")) {
    options.volumetric_model = parse_volumetric_model(*model);
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

  if (const auto ct_m = get_defined_double(domain.params, "Momentum_stabilization_coefficient")) {
    options.ct_m = static_cast<svmp::FE::Real>(*ct_m);
  }
  if (const auto ct_c = get_defined_double(domain.params, "Continuity_stabilization_coefficient")) {
    options.ct_c = static_cast<svmp::FE::Real>(*ct_c);
  }
}

void apply_ustruct_output_options(const svmp::Physics::EquationModuleInput& input,
                                  svmp::Physics::formulations::ustruct::UstructOptions& options)
{
  options.register_def_grad_output =
      spatial_output_bool(input, {"defgrad", "deformationgradient"}).value_or(false);
  options.register_jacobian_output =
      spatial_output_bool(input, {"jacobian", "j"}).value_or(false);
  options.register_cauchy_stress_output =
      spatial_output_bool(input, {"cauchystress"}).value_or(false);
  options.register_divergence_output =
      spatial_output_bool(input, {"divergence", "div"}).value_or(false);
  options.register_strain_output =
      spatial_output_bool(input, {"strain", "greenlagrangestrain"}).value_or(false);
  options.register_stress_output =
      spatial_output_bool(input, {"stress", "secondpiolastress", "pk2stress"}).value_or(false);
  options.register_von_mises_stress_output =
      spatial_output_bool(input, {"vonmisesstress", "vonmises", "misesstress"}).value_or(false);
}

void apply_ustruct_bcs(const svmp::Physics::EquationModuleInput& input,
                       svmp::Physics::formulations::ustruct::UstructOptions& options)
{
  using svmp::Physics::formulations::ustruct::UstructOptions;

  const int dim = input.mesh ? input.mesh->dim() : 0;
  if (dim != 2 && dim != 3) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct requires a 2D or 3D mesh for BC translation.");
  }

  for (const auto& bc : input.boundary_conditions) {
    if (bc.boundary_marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto* time_dep = find_param(bc.params, "Time_dependence");
    const std::string time_value =
        (time_dep && time_dep->defined) ? trim_copy(time_dep->value) : std::string("Steady");
    const auto time_value_lc = lower_copy(time_value);
    const bool is_steady = time_value_lc.empty() || time_value_lc == "steady";
    const bool is_unsteady = time_value_lc == "unsteady";

    const auto has_nonempty_defined = [&](std::string_view key) {
      const auto* p = find_param(bc.params, key);
      return p && p->defined && !trim_copy(p->value).empty();
    };
    const bool has_temporal_values_file = has_nonempty_defined("Temporal_values_file_path");
    const bool has_unsupported_file =
        has_nonempty_defined("Spatial_values_file_path") ||
        has_nonempty_defined("Temporal_and_spatial_values_file_path") ||
        has_nonempty_defined("Bct_file_path") ||
        has_nonempty_defined("Traction_values_file_path");

    const auto* type_param = find_param(bc.params, "Type");
    const std::string bc_type = type_param ? trim_copy(type_param->value) : std::string{};
    const auto bc_type_lc = lower_copy(bc_type);
    const bool is_neumann_like =
        (bc_type_lc == "neumann" || bc_type_lc == "neu" ||
         bc_type_lc == "traction" || bc_type_lc == "trac");

    const bool follower_pressure =
        (find_param(bc.params, "Follower_pressure_load") &&
         find_param(bc.params, "Follower_pressure_load")->defined &&
         parse_bool_relaxed(find_param(bc.params, "Follower_pressure_load")->value));

    if (!is_steady && !is_unsteady) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Only Steady ustruct boundary conditions and Unsteady file-driven follower-pressure "
          "BCs are supported for the new solver ustruct module (got Time_dependence='" + time_value +
          "'). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (is_unsteady) {
      if (!(is_neumann_like && follower_pressure && has_temporal_values_file) || has_unsupported_file) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Unsteady ustruct BC '" + bc.name +
            "' is only supported for file-driven follower pressure using <Temporal_values_file_path>. "
            "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
    } else if (has_temporal_values_file || has_unsupported_file) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Spatial/temporal/file-driven ustruct boundary conditions are only supported "
          "for Unsteady follower pressure in the new solver module. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> "
          "to use the legacy solver.");
    }

    const auto* value_param = find_param(bc.params, "Value");
    const auto value =
        static_cast<svmp::FE::Real>(value_param ? parse_double(value_param->value, "Add_BC/Value") : 0.0);

    std::vector<int> effective_dir;
    if (const auto* p = find_param(bc.params, "Effective_direction");
        p && !trim_copy(p->value).empty()) {
      effective_dir = parse_int_list(p->value);
    }

    const bool along_normal =
        (find_param(bc.params, "Apply_along_normal_direction") &&
         find_param(bc.params, "Apply_along_normal_direction")->defined &&
         parse_bool_relaxed(find_param(bc.params, "Apply_along_normal_direction")->value));

    if (bc_type_lc == "dirichlet" || bc_type_lc == "dir") {
      const bool impose_on_integral =
          !options.enable_time_derivative_terms ||
          get_defined_bool(bc.params, "Impose_on_state_variable_integral").value_or(false);
      if (impose_on_integral) {
        UstructOptions::DisplacementDirichletBC dir{};
        dir.boundary_marker = bc.boundary_marker;
        dir.active_components = fill_dirichlet_vector(dir.value, dim, effective_dir, value);
        if (std::any_of(dir.active_components.begin(), dir.active_components.end(), [](bool b) { return b; })) {
          options.displacement_dirichlet.push_back(std::move(dir));
        }

        UstructOptions::VelocityDirichletBC vel{};
        vel.boundary_marker = bc.boundary_marker;
        vel.active_components =
            fill_dirichlet_vector(vel.value, dim, effective_dir, static_cast<svmp::FE::Real>(0.0));
        if (std::any_of(vel.active_components.begin(), vel.active_components.end(), [](bool b) { return b; })) {
          options.velocity_dirichlet.push_back(std::move(vel));
        }
      } else {
        UstructOptions::VelocityDirichletBC vel{};
        vel.boundary_marker = bc.boundary_marker;
        vel.active_components = fill_dirichlet_vector(vel.value, dim, effective_dir, value);
        if (std::any_of(vel.active_components.begin(), vel.active_components.end(), [](bool b) { return b; })) {
          options.velocity_dirichlet.push_back(std::move(vel));
        }
      }
      continue;
    }

    if (bc_type_lc == "neumann" || bc_type_lc == "neu" ||
        bc_type_lc == "traction" || bc_type_lc == "trac") {
      if (follower_pressure) {
        UstructOptions::FollowerPressureBC pressure{};
        pressure.boundary_marker = bc.boundary_marker;
        if (is_unsteady) {
          const auto file_path = get_defined_string(bc.params, "Temporal_values_file_path");
          if (!file_path.has_value()) {
            throw std::runtime_error(
                "[svMultiPhysics::Physics] Unsteady follower-pressure BC '" + bc.name +
                "' is missing <Temporal_values_file_path>.");
          }

          auto temporal = svmp::Physics::readTemporalValuesFile(
              *file_path, /*num_components=*/1, svmp::Physics::TemporalEndBehavior::Clamp);
          const bool ramp = get_defined_bool(bc.params, "Ramp_function").value_or(false);
          if (ramp) {
            pressure.ramp = UstructOptions::FollowerPressureBC::LinearRamp{
                static_cast<svmp::FE::Real>(temporal->firstTime()),
                static_cast<svmp::FE::Real>(temporal->lastTime()),
                static_cast<svmp::FE::Real>(temporal->firstValue()),
                static_cast<svmp::FE::Real>(temporal->lastValue())};
          } else {
            pressure.pressure = UstructOptions::ScalarValue{
                svmp::FE::forms::TimeScalarCoefficient(
                    [temporal](svmp::FE::Real /*x*/,
                               svmp::FE::Real /*y*/,
                               svmp::FE::Real /*z*/,
                               svmp::FE::Real t) -> svmp::FE::Real {
                      return static_cast<svmp::FE::Real>(temporal->interpolate(static_cast<double>(t)));
                    })};
          }
        } else {
          pressure.pressure = UstructOptions::ScalarValue{value};
        }
        options.follower_pressure.push_back(std::move(pressure));
      } else if (along_normal) {
        UstructOptions::NormalTractionBC normal{};
        normal.boundary_marker = bc.boundary_marker;
        normal.traction = UstructOptions::ScalarValue{value};
        options.normal_traction.push_back(std::move(normal));
      } else {
        UstructOptions::TractionNeumannBC traction{};
        traction.boundary_marker = bc.boundary_marker;
        fill_vector(traction.traction, dim, effective_dir, value);
        options.traction_neumann.push_back(std::move(traction));
      }
      continue;
    }

    if (bc_type_lc == "pressure_dirichlet" || bc_type_lc == "pressure-dirichlet") {
      UstructOptions::PressureDirichletBC pbc{};
      pbc.boundary_marker = bc.boundary_marker;
      pbc.value = UstructOptions::ScalarValue{value};
      options.pressure_dirichlet.push_back(std::move(pbc));
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary condition type '" + bc_type +
        "' is not supported for the new solver ustruct module. Supported types: Dir, Dirichlet, Neu, Neumann, Trac, Traction. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_ustruct_from_input(const svmp::Physics::EquationModuleInput& input,
                          svmp::FE::systems::FESystem& system)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct module factory received null mesh.");
  }

  const auto& domain = select_single_domain(input);
  const int dim = input.mesh->dim();
  if (dim != 2 && dim != 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for ustruct spaces: " + std::to_string(dim) +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int order = resolve_element_order(input, infer_polynomial_order(*input.mesh));

  auto displacement_space =
      svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, element_type, order, dim);
  auto pressure_space = svmp::FE::spaces::SpaceFactory::create_h1(element_type, order);

  svmp::Physics::formulations::ustruct::UstructOptions options{};
  options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);
  options.enable_time_derivative_terms =
      resolve_ustruct_time_derivative_terms(input, options.enable_time_derivative_terms);

  const auto solid_props = parse_solid_properties(domain);
  apply_solid_options(domain, solid_props, options);
  apply_ustruct_output_options(input, options);
  apply_ustruct_bcs(input, options);

  auto module = std::make_unique<svmp::Physics::formulations::ustruct::UstructModule>(
      std::move(displacement_space), std::move(pressure_space), std::move(options));
  module->registerOn(system);
  return module;
}

} // namespace

SVMP_REGISTER_EQUATION("ustruct", &create_ustruct_from_input);

namespace svmp::Physics::formulations::ustruct {

void forceLink_UstructRegister() {}

} // namespace svmp::Physics::formulations::ustruct
