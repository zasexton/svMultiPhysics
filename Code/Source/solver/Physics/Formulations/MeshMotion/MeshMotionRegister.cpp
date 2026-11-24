#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"
#include "Physics/Formulations/MeshMotion/PseudoElasticMeshMotionModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Options/InputReaders.h"

#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
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

// The generic readers of equation input (decision D52).
using namespace svmp::Physics::options;
namespace opt = svmp::FE::options;
using svmp::FE::Real;

enum class MeshMotionModel : std::uint8_t { FromEquationType, Harmonic, PseudoElastic };

// The equation-only key of mesh motion.
struct MeshMotionEquationKeys {
  MeshMotionModel model{MeshMotionModel::FromEquationType};
};

// The keys of the equation, read from the equation, then the default domain,
// then the domain, each overriding the earlier ones.
struct MeshMotionKeys {
  std::string field_name{"mesh_displacement"};
  bool auto_register_field{true};
  bool bind_as_mesh_displacement{true};
  Real kappa{1.0};
  std::optional<Real> stiffness{};
  mm::HarmonicQuantity harmonic_quantity{mm::HarmonicQuantity::Displacement};
  Real lambda_mesh{1.0};
  Real mu_mesh{1.0};
};

enum class MeshMotionBoundaryType : std::uint8_t { None, Dirichlet, Natural, Robin, NormalConstraint, TangentialPolicy };
enum class MeshMotionQuantity : std::uint8_t { Displacement, Velocity };

// The keys of one boundary condition.  Value and Target are kept as text:
// one number along Effective_direction, or two or three components.
struct MeshMotionBoundaryKeys {
  MeshMotionBoundaryType type{MeshMotionBoundaryType::None};
  std::string value{};
  std::string target{};
  std::string effective_direction{};
  MeshMotionQuantity quantity{MeshMotionQuantity::Displacement};
  std::optional<Real> alpha{};
  std::optional<Real> penalty{};
  Real velocity_time_scale{1.0};
  mm::TangentialMeshPolicy tangential_policy{mm::TangentialMeshPolicy::SmoothingOnly};
};

const opt::OptionSchema<MeshMotionEquationKeys>& mesh_motion_equation_schema()
{
  static const auto schema = [] {
    opt::OptionSchema<MeshMotionEquationKeys> out("Mesh motion equation");
    out.key("Model", &MeshMotionEquationKeys::model,
            opt::EnumCodec<MeshMotionModel>{{{MeshMotionModel::Harmonic, "harmonic", {"laplace", "laplacian"}},
                                             {MeshMotionModel::PseudoElastic, "pseudo_elastic",
                                              {"elastic", "linear_elastic"}}}},
            {.doc = "the mesh-motion operator (default: from the equation type)"});
    return out;
  }();
  return schema;
}

void add_common_mesh_motion_keys(opt::OptionSchema<MeshMotionKeys>& schema)
{
  const auto relaxed = opt::BoolCodec{opt::BoolLexicon::Relaxed};
  schema
      .key("Field_name", &MeshMotionKeys::field_name, opt::StringCodec{},
           {.spellings = {{"Field_name"}, {"Mesh_displacement_field"}}, .doc = "the mesh displacement field"})
      .key("Auto_register_field", &MeshMotionKeys::auto_register_field, relaxed,
           {.spellings = {{"Auto_register_field"}, {"Auto_register_mesh_displacement_field"}},
            .empty = opt::EmptyPolicy::Absent,
            .doc = "register the field when missing"})
      .key("Bind_as_mesh_displacement", &MeshMotionKeys::bind_as_mesh_displacement, relaxed,
           {.empty = opt::EmptyPolicy::Absent, .doc = "the field moves the mesh"});
}

const opt::OptionSchema<MeshMotionKeys>& harmonic_mesh_motion_schema()
{
  static const auto schema = [] {
    opt::OptionSchema<MeshMotionKeys> out("Harmonic mesh motion");
    add_common_mesh_motion_keys(out);
    out.key("Kappa", &MeshMotionKeys::kappa, opt::RealCodec{}, {.empty = opt::EmptyPolicy::Absent, .doc = "kappa"})
        .key("Stiffness", &MeshMotionKeys::stiffness, opt::OptionalCodec<opt::RealCodec>{},
             {.doc = "the older name of Kappa (none: unset)"})
        .key("Harmonic_quantity", &MeshMotionKeys::harmonic_quantity,
             opt::EnumCodec<mm::HarmonicQuantity>{
                 {{mm::HarmonicQuantity::Displacement, "displacement", {"mesh_displacement"}},
                  {mm::HarmonicQuantity::Velocity, "velocity", {"mesh_velocity"}}}},
             {.doc = "the harmonic unknown"});
    return out;
  }();
  return schema;
}

const opt::OptionSchema<MeshMotionKeys>& pseudo_elastic_mesh_motion_schema()
{
  static const auto schema = [] {
    opt::OptionSchema<MeshMotionKeys> out("Pseudo-elastic mesh motion");
    add_common_mesh_motion_keys(out);
    out.key("Lambda_mesh", &MeshMotionKeys::lambda_mesh, opt::RealCodec{},
            {.empty = opt::EmptyPolicy::Absent, .doc = "the Lame lambda"})
        .key("Mu_mesh", &MeshMotionKeys::mu_mesh, opt::RealCodec{},
             {.empty = opt::EmptyPolicy::Absent, .doc = "the Lame mu"});
    return out;
  }();
  return schema;
}

const opt::OptionSchema<MeshMotionBoundaryKeys>& mesh_motion_boundary_schema()
{
  using K = MeshMotionBoundaryKeys;
  using Type = MeshMotionBoundaryType;
  using Policy = mm::TangentialMeshPolicy;
  static const auto schema = [] {
    opt::OptionSchema<K> out("Mesh-motion boundary condition");
    const opt::KeySettings finite_real{.empty = opt::EmptyPolicy::Absent};
    out.key("Type", &K::type,
            opt::EnumCodec<Type>{{{Type::Dirichlet, "Dirichlet", {"Dir"}},
                                  {Type::Natural, "Natural", {"Neumann", "Neu", "Traction", "Trac"}},
                                  {Type::Robin, "Robin", {}},
                                  {Type::NormalConstraint, "NormalConstraint", {"Normal"}},
                                  {Type::TangentialPolicy, "TangentialPolicy", {"Tangential"}}}},
            {.doc = "the condition"})
        .key("Value", &K::value, opt::StringCodec{}, {.doc = "one number, or two or three components"})
        .key("Target", &K::target, opt::StringCodec{}, {.doc = "the Robin or constraint target (default: Value)"})
        .key("Effective_direction", &K::effective_direction, opt::StringCodec{},
             {.spellings = {{"Effective_direction"}, {"EffectiveDirection"}},
              .doc = "integer direction of a single number"})
        .key("Quantity", &K::quantity,
             opt::EnumCodec<MeshMotionQuantity>{{{MeshMotionQuantity::Displacement, "displacement", {"mesh_displacement"}},
                                                 {MeshMotionQuantity::Velocity, "velocity", {"mesh_velocity"}}}},
             {.doc = "the constrained quantity"})
        .key("Alpha", &K::alpha, opt::OptionalCodec<opt::RealCodec>{}, {.doc = "the Robin alpha (default: Penalty, 1)"})
        .key("Penalty", &K::penalty, opt::OptionalCodec<opt::RealCodec>{}, {.doc = "the constraint penalty (default 1)"})
        .key("Velocity_time_scale", &K::velocity_time_scale, opt::RealCodec{}, finite_real)
        .key("Tangential_policy", &K::tangential_policy,
             opt::EnumCodec<Policy>{{{Policy::SmoothingOnly, "smoothing_only", {"smoothing", "mesh_smoothing", "smooth"}},
                                     {Policy::Free, "free", {}},
                                     {Policy::Prescribed, "prescribed", {}}}},
             {.doc = "the tangential motion"});
    return out;
  }();
  return schema;
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

// One number along the direction, or two or three components (Value or Target).
std::array<mm::HarmonicMeshMotionOptions::ScalarValue, 3> vector_value(const std::string& text,
                                                                     const std::string& direction,
                                                                     int dim,
                                                                     std::string_view context)
{
  using ScalarValue = mm::HarmonicMeshMotionOptions::ScalarValue;
  std::array<ScalarValue, 3> values{ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
  const auto parsed = parse_real_list(text.empty() ? std::string("0.0") : text, context);
  if (parsed.size() == 1u) {
    std::vector<int> effective_dir;
    if (!direction.empty()) {
      effective_dir = parse_int_list(direction);
    }
    for (int d = 0; d < dim; ++d) {
      values[static_cast<std::size_t>(d)] = parsed.front() * direction_component(effective_dir, d);
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

// Component mask of a zero-valued mesh-motion Dirichlet condition.
//
// As for the fluid velocity, `Effective_direction` on a Dirichlet condition
// with the single literal value 0 selects the constrained Cartesian
// components: `1 0` on a wall x = const constrains only the x displacement,
// so the mesh slides along the wall (normal-only, "slip" mesh motion).  A
// direction that selects every component or none constrains all components
// (the historical behavior).  For a nonzero single value the direction keeps
// its historical meaning as the direction of the prescribed displacement and
// every component is constrained.
std::array<bool, 3> zero_dirichlet_component_mask(const MeshMotionBoundaryKeys& keys,
                                                  int dim,
                                                  const std::string& context)
{
  std::array<bool, 3> active{true, true, true};
  if (keys.effective_direction.empty()) {
    return active;
  }
  const auto parsed = parse_real_list(keys.value.empty() ? std::string("0.0") : keys.value, context);
  if (parsed.size() != 1u || parsed.front() != 0.0) {
    return active;
  }
  const auto direction = parse_int_list(keys.effective_direction);
  int selected = 0;
  std::array<bool, 3> mask{false, false, false};
  for (int d = 0; d < dim; ++d) {
    const auto idx = static_cast<std::size_t>(d);
    mask[idx] = idx < direction.size() && direction[idx] != 0;
    selected += mask[idx] ? 1 : 0;
  }
  if (selected == 0 || selected == dim) {
    return active;
  }
  return mask;
}

// Target, or Value when Target is absent.
const std::string& target_text(const MeshMotionBoundaryKeys& keys)
{
  return keys.target.empty() ? keys.value : keys.target;
}

template <class Options>
void apply_mesh_motion_bcs(const svmp::Physics::EquationModuleInput& input,
                           Options& options,
                           int dim)
{
  using Type = MeshMotionBoundaryType;
  for (const auto& bc : input.boundary_conditions) {
    if (bc.boundary_marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Mesh-motion boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto keys = mesh_motion_boundary_schema().resolve(option_input(bc.params)).value;
    switch (keys.type) {
    case Type::Dirichlet: {
      typename Options::DirichletBC dirichlet{};
      dirichlet.boundary_marker = bc.boundary_marker;
      dirichlet.value = vector_value(keys.value, keys.effective_direction, dim, "Mesh-motion Dirichlet Value");
      dirichlet.active_components =
          zero_dirichlet_component_mask(keys, dim, "Mesh-motion Dirichlet '" + bc.name + "'");
      options.dirichlet.push_back(std::move(dirichlet));
      break;
    }
    case Type::Natural: {
      typename Options::NaturalBC natural{};
      natural.boundary_marker = bc.boundary_marker;
      natural.value = vector_value(keys.value, keys.effective_direction, dim, "Mesh-motion natural Value");
      options.natural.push_back(std::move(natural));
      break;
    }
    case Type::Robin: {
      typename Options::RobinBC robin{};
      robin.boundary_marker = bc.boundary_marker;
      robin.alpha = keys.alpha.value_or(keys.penalty.value_or(1.0));
      robin.target = vector_value(target_text(keys), keys.effective_direction, dim, "Mesh-motion Robin Target");
      options.robin.push_back(std::move(robin));
      break;
    }
    case Type::NormalConstraint: {
      mm::NormalConstraintBC normal{};
      normal.boundary_marker = bc.boundary_marker;
      normal.quantity = keys.quantity == MeshMotionQuantity::Velocity ? mm::NormalConstraintQuantity::Velocity
                                                                      : mm::NormalConstraintQuantity::Displacement;
      const auto& target = target_text(keys);
      normal.target = target.empty() ? 0.0 : parse_finite_double(target, "Normal_constraint/Target");
      normal.penalty = keys.penalty.value_or(1.0);
      normal.velocity_time_scale = keys.velocity_time_scale;
      options.normal_constraint.push_back(std::move(normal));
      break;
    }
    case Type::TangentialPolicy: {
      mm::TangentialPolicyBC tangential{};
      tangential.boundary_marker = bc.boundary_marker;
      tangential.policy = keys.tangential_policy;
      tangential.quantity = keys.quantity == MeshMotionQuantity::Velocity
                                ? mm::TangentialConstraintQuantity::Velocity
                                : mm::TangentialConstraintQuantity::Displacement;
      tangential.target = vector_value(target_text(keys), keys.effective_direction, dim, "Tangential_policy/Target");
      tangential.penalty = keys.penalty.value_or(1.0);
      tangential.velocity_time_scale = keys.velocity_time_scale;
      options.tangential_policy.push_back(std::move(tangential));
      break;
    }
    case Type::None:
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Mesh-motion boundary condition '" + bc.name +
          "' has no Type. Supported types: Dir, Neumann, Robin, NormalConstraint, TangentialPolicy.");
    }
  }
}

// The keys of the equation, then of the default domain, then of the domain.
MeshMotionKeys mesh_motion_keys(const svmp::Physics::EquationModuleInput& input,
                                const opt::OptionSchema<MeshMotionKeys>& schema)
{
  auto keys = schema.resolve(option_input(input.equation_params)).value;
  keys = schema.resolve(option_input(input.default_domain.params), keys).value;
  if (!input.domains.empty()) {
    if (input.domains.size() != 1u) {
      throw std::runtime_error("[svMultiPhysics::Physics] Multiple <Domain> blocks are not supported by mesh motion.");
    }
    keys = schema.resolve(option_input(input.domains.front().params), keys).value;
  }
  return keys;
}

MeshMotionModel mesh_motion_model(const svmp::Physics::EquationModuleInput& input)
{
  const auto model = mesh_motion_equation_schema().resolve(option_input(input.equation_params)).value.model;
  if (model != MeshMotionModel::FromEquationType) {
    return model;
  }
  return normalized_token(input.equation_type).find("pseudoelastic") != std::string::npos
             ? MeshMotionModel::PseudoElastic
             : MeshMotionModel::Harmonic;
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

  if (mesh_motion_model(input) == MeshMotionModel::Harmonic) {
    const auto keys = mesh_motion_keys(input, harmonic_mesh_motion_schema());
    mm::HarmonicMeshMotionOptions options{};
    options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);
    options.field_name = keys.field_name;
    options.auto_register_field = keys.auto_register_field;
    options.bind_as_mesh_displacement = keys.bind_as_mesh_displacement;
    options.kappa = keys.kappa;
    if (keys.stiffness) {
      options.stiffness = mm::HarmonicMeshMotionOptions::ScalarValue{*keys.stiffness};
    }
    options.quantity = keys.harmonic_quantity;
    apply_mesh_motion_bcs(input, options, dim);

    auto module = std::make_unique<mm::HarmonicMeshMotionModule>(
        std::move(displacement_space), std::move(options));
    module->registerOn(system);
    return module;
  }

  const auto keys = mesh_motion_keys(input, pseudo_elastic_mesh_motion_schema());
  mm::PseudoElasticMeshMotionOptions options{};
  options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);
  options.field_name = keys.field_name;
  options.auto_register_field = keys.auto_register_field;
  options.bind_as_mesh_displacement = keys.bind_as_mesh_displacement;
  options.lambda_mesh = keys.lambda_mesh;
  options.mu_mesh = keys.mu_mesh;
  apply_mesh_motion_bcs(input, options, dim);

  auto module = std::make_unique<mm::PseudoElasticMeshMotionModule>(
      std::move(displacement_space), std::move(options));
  module->registerOn(system);
  return module;
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

std::string resolvedInputKeys(const EquationModuleInput& input)
{
  const auto& equation_schema = mesh_motion_equation_schema();
  const auto& model_schema = mesh_motion_model(input) == MeshMotionModel::Harmonic
                                 ? harmonic_mesh_motion_schema()
                                 : pseudo_elastic_mesh_motion_schema();
  const auto& boundary_schema = mesh_motion_boundary_schema();
  std::string out = option_line(
      equation_schema.element(),
      equation_schema.serialize(equation_schema.resolve(option_input(input.equation_params)).value));
  out += option_line(model_schema.element(), model_schema.serialize(mesh_motion_keys(input, model_schema)));
  for (const auto& bc : input.boundary_conditions) {
    out += option_line(boundary_schema.element() + " '" + bc.name + "'",
                           boundary_schema.serialize(boundary_schema.resolve(option_input(bc.params)).value));
  }
  return out;
}

} // namespace svmp::Physics::formulations::mesh_motion
