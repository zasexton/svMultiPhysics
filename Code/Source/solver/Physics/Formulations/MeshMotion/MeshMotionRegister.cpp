#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"
#include "Physics/Formulations/MeshMotion/PseudoElasticMeshMotionModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Core/JITRuntimePolicy.h"

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

namespace mm = svmp::Physics::formulations::mesh_motion;

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
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse numeric value '" +
                             std::string(raw) + "' for " + std::string(context) + ".");
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

std::vector<svmp::FE::Real> parse_real_list(std::string raw, std::string_view context)
{
  for (char& ch : raw) {
    if (ch == '(' || ch == ')' || ch == ',' || ch == ';') {
      ch = ' ';
    }
  }

  std::istringstream in(raw);
  std::vector<svmp::FE::Real> out;
  double value = 0.0;
  while (in >> value) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse finite numeric components for " +
                               std::string(context) + ".");
    }
    out.push_back(static_cast<svmp::FE::Real>(value));
  }
  if (out.empty()) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to parse numeric components for " +
                             std::string(context) + ".");
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

std::array<mm::HarmonicMeshMotionOptions::ScalarValue, 3>
parse_vector_value(const svmp::Physics::ParameterMap& params,
                   std::initializer_list<std::string_view> keys,
                   int dim,
                   std::string_view context)
{
  std::array<mm::HarmonicMeshMotionOptions::ScalarValue, 3> values{
      mm::HarmonicMeshMotionOptions::ScalarValue{0.0},
      mm::HarmonicMeshMotionOptions::ScalarValue{0.0},
      mm::HarmonicMeshMotionOptions::ScalarValue{0.0}};
  const auto raw = get_defined_string(params, keys).value_or("0.0");
  const auto parsed = parse_real_list(raw, context);
  if (parsed.size() == 1u) {
    std::vector<int> effective_dir;
    if (const auto raw_direction =
            get_defined_string(params, {"Effective_direction", "EffectiveDirection"})) {
      effective_dir = parse_int_list(*raw_direction);
    }
    for (int d = 0; d < dim; ++d) {
      values[static_cast<std::size_t>(d)] =
          parsed.front() * direction_component(effective_dir, d);
    }
    return values;
  }

  if (parsed.size() != 2u && parsed.size() != 3u) {
    throw std::runtime_error("[svMultiPhysics::Physics] " + std::string(context) +
                             " must contain one, two, or three numeric values.");
  }
  if (static_cast<int>(parsed.size()) < dim) {
    throw std::runtime_error("[svMultiPhysics::Physics] " + std::string(context) +
                             " does not provide enough components for the mesh dimension.");
  }
  for (int d = 0; d < dim; ++d) {
    values[static_cast<std::size_t>(d)] = parsed[static_cast<std::size_t>(d)];
  }
  return values;
}

svmp::FE::forms::GeometryTangentPath parse_geometry_tangent_path(std::string_view raw,
                                                                 std::string_view context)
{
  const auto path = normalized_token(std::string(raw));
  if (path == "symbolic" || path == "symbolicrequired" || path == "required") {
    return svmp::FE::forms::GeometryTangentPath::SymbolicRequired;
  }
  if (path == "ad" || path == "adreference" || path == "referencead") {
    return svmp::FE::forms::GeometryTangentPath::ADReference;
  }
  if (path == "symbolicadcheck" || path == "symbolicwithadcheck" ||
      path == "check" || path == "paritycheck") {
    return svmp::FE::forms::GeometryTangentPath::SymbolicWithADCheck;
  }
  if (path == "auto") {
    return svmp::FE::forms::GeometryTangentPath::Auto;
  }
  throw std::runtime_error("[svMultiPhysics::Physics] " + std::string(context) +
                           " must be one of 'symbolic', 'ad', 'symbolic_ad_check', or 'auto'.");
}

mm::NormalConstraintQuantity parse_normal_quantity(std::string_view raw,
                                                   std::string_view context)
{
  const auto token = normalized_token(std::string(raw));
  if (token.empty() || token == "displacement" || token == "meshdisplacement") {
    return mm::NormalConstraintQuantity::Displacement;
  }
  if (token == "velocity" || token == "meshvelocity") {
    return mm::NormalConstraintQuantity::Velocity;
  }
  throw std::runtime_error("[svMultiPhysics::Physics] " + std::string(context) +
                           " quantity must be 'displacement' or 'velocity'.");
}

mm::TangentialConstraintQuantity parse_tangential_quantity(std::string_view raw,
                                                           std::string_view context)
{
  const auto token = normalized_token(std::string(raw));
  if (token.empty() || token == "displacement" || token == "meshdisplacement") {
    return mm::TangentialConstraintQuantity::Displacement;
  }
  if (token == "velocity" || token == "meshvelocity") {
    return mm::TangentialConstraintQuantity::Velocity;
  }
  throw std::runtime_error("[svMultiPhysics::Physics] " + std::string(context) +
                           " quantity must be 'displacement' or 'velocity'.");
}

mm::TangentialMeshPolicy parse_tangential_policy(std::string_view raw,
                                                 std::string_view context)
{
  const auto token = normalized_token(std::string(raw));
  if (token.empty() || token == "smoothing" || token == "smoothingonly" ||
      token == "meshsmoothing" || token == "smooth") {
    return mm::TangentialMeshPolicy::SmoothingOnly;
  }
  if (token == "free") {
    return mm::TangentialMeshPolicy::Free;
  }
  if (token == "prescribed") {
    return mm::TangentialMeshPolicy::Prescribed;
  }
  throw std::runtime_error("[svMultiPhysics::Physics] " + std::string(context) +
                           " policy must be 'free', 'smoothing_only', or 'prescribed'.");
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
    throw std::runtime_error("[svMultiPhysics::Physics] Mixed-order meshes are not supported by mesh motion.");
  }

  const auto family = shapes.front().family;
  for (const auto& s : shapes) {
    if (s.family != family) {
      throw std::runtime_error("[svMultiPhysics::Physics] Mixed cell families are not supported by mesh motion.");
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
    default: break;
  }
  throw std::runtime_error("[svMultiPhysics::Physics] Unsupported mesh cell family for mesh motion.");
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
      throw std::runtime_error("[svMultiPhysics::Physics] Mixed polynomial orders are not supported by mesh motion.");
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

template <class Options>
void apply_common_params(const svmp::Physics::ParameterMap& params, Options& options)
{
  if (const auto value = get_defined_string(
          params,
          {"Field_name", "FieldName", "Mesh_displacement_field", "MeshDisplacementField"})) {
    options.field_name = *value;
  }
  if (const auto value = get_defined_string(params, {"Operator_tag", "OperatorTag"})) {
    options.operator_tag = *value;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Auto_register_field", "AutoRegisterField", "Auto_register_mesh_displacement_field"})) {
    options.auto_register_field = *value;
  }
  if (const auto value = get_defined_bool(
          params,
          {"Bind_as_mesh_displacement", "BindAsMeshDisplacement"})) {
    options.bind_as_mesh_displacement = *value;
  }
  if (const auto value = get_defined_string(
          params,
          {"Moving_mesh_tangent_path", "MovingMeshTangentPath",
           "Geometry_tangent_path", "GeometryTangentPath"})) {
    options.tangent_path = parse_geometry_tangent_path(*value, "Moving_mesh_tangent_path");
  }
}

void apply_harmonic_params(const svmp::Physics::ParameterMap& params,
                           mm::HarmonicMeshMotionOptions& options)
{
  apply_common_params(params, options);
  if (const auto value = get_defined_real(params, {"Kappa", "Mesh_motion_kappa"}, "Kappa")) {
    options.kappa = *value;
  }
  if (const auto value = get_defined_real(params, {"Stiffness", "Mesh_motion_stiffness"}, "Stiffness")) {
    options.stiffness = *value;
  }
}

void apply_pseudo_elastic_params(const svmp::Physics::ParameterMap& params,
                                 mm::PseudoElasticMeshMotionOptions& options)
{
  apply_common_params(params, options);
  if (const auto value = get_defined_real(params, {"Lambda_mesh", "Mesh_lambda"}, "Lambda_mesh")) {
    options.lambda_mesh = *value;
  }
  if (const auto value = get_defined_real(params, {"Mu_mesh", "Mesh_mu"}, "Mu_mesh")) {
    options.mu_mesh = *value;
  }
}

mm::NormalConstraintBC parse_normal_constraint_bc(const svmp::Physics::BoundaryConditionInput& bc)
{
  mm::NormalConstraintBC out{};
  out.boundary_marker = bc.boundary_marker;
  out.quantity = parse_normal_quantity(
      get_defined_string(bc.params, {"Quantity", "Constraint_quantity"}).value_or("displacement"),
      "mesh-motion normal constraint");
  out.target = get_defined_real(bc.params, {"Target", "Value"}, "Normal_constraint/Target").value_or(0.0);
  out.penalty = get_defined_real(bc.params, {"Penalty", "Penalty_scale"}, "Normal_constraint/Penalty")
                    .value_or(1.0);
  out.velocity_time_scale =
      get_defined_real(bc.params, {"Velocity_time_scale", "Time_scale"}, "Normal_constraint/Velocity_time_scale")
          .value_or(1.0);
  return out;
}

mm::TangentialPolicyBC parse_tangential_policy_bc(const svmp::Physics::BoundaryConditionInput& bc,
                                                  int dim)
{
  mm::TangentialPolicyBC out{};
  out.boundary_marker = bc.boundary_marker;
  out.policy = parse_tangential_policy(
      get_defined_string(bc.params, {"Policy", "Tangential_policy"}).value_or("smoothing_only"),
      "mesh-motion tangential policy");
  out.quantity = parse_tangential_quantity(
      get_defined_string(bc.params, {"Quantity", "Constraint_quantity"}).value_or("displacement"),
      "mesh-motion tangential policy");
  out.target = parse_vector_value(bc.params, {"Target", "Value"}, dim, "Tangential_policy/Target");
  out.penalty =
      get_defined_real(bc.params, {"Penalty", "Penalty_scale"}, "Tangential_policy/Penalty").value_or(1.0);
  out.velocity_time_scale =
      get_defined_real(bc.params, {"Velocity_time_scale", "Time_scale"}, "Tangential_policy/Velocity_time_scale")
          .value_or(1.0);
  return out;
}

template <class Options>
void apply_mesh_motion_bcs(const svmp::Physics::EquationModuleInput& input,
                           Options& options,
                           int dim)
{
  for (const auto& bc : input.boundary_conditions) {
    if (bc.boundary_marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Mesh-motion boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto type = normalized_token(
        get_defined_string(bc.params, {"Type"}).value_or(std::string{}));

    if (type == "dir" || type == "dirichlet") {
      typename Options::DirichletBC dirichlet{};
      dirichlet.boundary_marker = bc.boundary_marker;
      dirichlet.value = parse_vector_value(bc.params, {"Value"}, dim, "Mesh-motion Dirichlet Value");
      options.dirichlet.push_back(std::move(dirichlet));
      continue;
    }

    if (type == "natural" || type == "neumann" || type == "neu" ||
        type == "traction" || type == "trac") {
      typename Options::NaturalBC natural{};
      natural.boundary_marker = bc.boundary_marker;
      natural.value = parse_vector_value(bc.params, {"Value"}, dim, "Mesh-motion natural Value");
      options.natural.push_back(std::move(natural));
      continue;
    }

    if (type == "robin") {
      typename Options::RobinBC robin{};
      robin.boundary_marker = bc.boundary_marker;
      robin.alpha = get_defined_real(bc.params, {"Alpha", "Penalty", "Penalty_scale"}, "Mesh-motion Robin Alpha")
                        .value_or(1.0);
      robin.target = parse_vector_value(bc.params, {"Target", "Value"}, dim, "Mesh-motion Robin Target");
      options.robin.push_back(std::move(robin));
      continue;
    }

    if (type == "normalconstraint" || type == "normal") {
      options.normal_constraint.push_back(parse_normal_constraint_bc(bc));
      continue;
    }

    if (type == "tangentialpolicy" || type == "tangential") {
      options.tangential_policy.push_back(parse_tangential_policy_bc(bc, dim));
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary condition type '" +
        get_defined_string(bc.params, {"Type"}).value_or(std::string{}) +
        "' is not supported for mesh motion. Supported types: Dir, Neumann, Robin, NormalConstraint, TangentialPolicy.");
  }
}

std::string resolve_model_token(const svmp::Physics::EquationModuleInput& input)
{
  if (const auto model = get_defined_string(
          input.equation_params,
          {"Model", "Formulation", "Mesh_motion_model", "MeshMotionModel"})) {
    return normalized_token(*model);
  }
  const auto type = normalized_token(input.equation_type);
  if (type.find("pseudoelastic") != std::string::npos) {
    return "pseudoelastic";
  }
  return "harmonic";
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_mesh_motion_from_input(const svmp::Physics::EquationModuleInput& input,
                              svmp::FE::systems::FESystem& system)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Mesh-motion module factory received null mesh.");
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int order = resolve_element_order(input, infer_polynomial_order(*input.mesh));
  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error("[svMultiPhysics::Physics] Mesh motion requires a mesh dimension in [1, 3].");
  }

  auto displacement_space = svmp::FE::spaces::SpaceFactory::create_vector_h1(element_type, order, dim);
  const auto model = resolve_model_token(input);

  if (model == "harmonic" || model == "laplace" || model == "laplacian") {
    mm::HarmonicMeshMotionOptions options{};
    options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);
    apply_harmonic_params(input.equation_params, options);
    apply_harmonic_params(input.default_domain.params, options);
    if (!input.domains.empty()) {
      if (input.domains.size() != 1u) {
        throw std::runtime_error("[svMultiPhysics::Physics] Multiple <Domain> blocks are not supported by mesh motion.");
      }
      apply_harmonic_params(input.domains.front().params, options);
    }
    apply_mesh_motion_bcs(input, options, dim);

    auto module = std::make_unique<mm::HarmonicMeshMotionModule>(
        std::move(displacement_space), std::move(options));
    module->registerOn(system);
    return module;
  }

  if (model == "pseudoelastic" || model == "elastic" || model == "linearelastic") {
    mm::PseudoElasticMeshMotionOptions options{};
    options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);
    apply_pseudo_elastic_params(input.equation_params, options);
    apply_pseudo_elastic_params(input.default_domain.params, options);
    if (!input.domains.empty()) {
      if (input.domains.size() != 1u) {
        throw std::runtime_error("[svMultiPhysics::Physics] Multiple <Domain> blocks are not supported by mesh motion.");
      }
      apply_pseudo_elastic_params(input.domains.front().params, options);
    }
    apply_mesh_motion_bcs(input, options, dim);

    auto module = std::make_unique<mm::PseudoElasticMeshMotionModule>(
        std::move(displacement_space), std::move(options));
    module->registerOn(system);
    return module;
  }

  throw std::runtime_error(
      "[svMultiPhysics::Physics] Mesh_motion model must be one of 'harmonic' or 'pseudo_elastic'.");
}

} // namespace

SVMP_REGISTER_EQUATION("mesh_motion", &create_mesh_motion_from_input);
SVMP_REGISTER_EQUATION("meshMotion", &create_mesh_motion_from_input);
SVMP_REGISTER_EQUATION("harmonic_mesh_motion", &create_mesh_motion_from_input);
SVMP_REGISTER_EQUATION("mesh_motion_harmonic", &create_mesh_motion_from_input);
SVMP_REGISTER_EQUATION("pseudo_elastic_mesh_motion", &create_mesh_motion_from_input);
SVMP_REGISTER_EQUATION("mesh_motion_pseudo_elastic", &create_mesh_motion_from_input);

namespace svmp::Physics::formulations::mesh_motion {

void forceLink_MeshMotionRegister() {}

} // namespace svmp::Physics::formulations::mesh_motion
