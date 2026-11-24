#include "Physics/Formulations/Ustruct/UstructModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/TemporalValues.h"
#include "Physics/Options/InputReaders.h"
#include "Physics/Materials/Solid/IsochoricNeoHookeanPK1.h"

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

// The generic readers of equation input (decision D52).
using namespace svmp::Physics::options;
namespace opt = svmp::FE::options;
using svmp::FE::Real;
using svmp::Physics::formulations::ustruct::VolumetricPenaltyModel;

enum class SolidModel : std::uint8_t { NeoHookean };

// The domain keys of a ustruct equation.
struct UstructDomainKeys {
  SolidModel constitutive_model{SolidModel::NeoHookean};
  Real density{1.0};
  Real elasticity_modulus{1.0};
  Real poisson_ratio{0.49};
  std::optional<Real> penalty_parameter{};  // none: the bulk modulus
  VolumetricPenaltyModel volumetric_model{VolumetricPenaltyModel::ST91};
  Real force_x{0.0};
  Real force_y{0.0};
  Real force_z{0.0};
  Real momentum_stabilization_coefficient{0.0};
  Real continuity_stabilization_coefficient{0.0};
};

enum class UstructBoundaryType : std::uint8_t { None, Dirichlet, Neumann, PressureDirichlet };

// The keys of one boundary condition.  An omitted Value is 0, the default of
// the input element.
struct UstructBoundaryKeys {
  std::string time_dependence{"Steady"};
  UstructBoundaryType type{UstructBoundaryType::None};
  Real value{0.0};
  std::string effective_direction{};
  bool follower_pressure_load{false};
  bool apply_along_normal_direction{false};
  bool impose_on_state_variable_integral{false};
  bool ramp_function{false};
  std::string temporal_values_file_path{};
};

// A number as std::stod reads it, infinite values included.
opt::RealCodec any_real()
{
  opt::RealCodec codec;
  codec.require_finite = false;
  return codec;
}

const opt::OptionSchema<UstructDomainKeys>& ustruct_domain_schema()
{
  using K = UstructDomainKeys;
  using Model = VolumetricPenaltyModel;
  static const auto schema = [] {
    opt::OptionSchema<K> out("ustruct domain");
    out.key("Constitutive_model.type", &K::constitutive_model,
            opt::EnumCodec<SolidModel>{{{SolidModel::NeoHookean, "neoHookean", {"nhk"}}}},
            {.doc = "the deviatoric model"})
        .key("Solid_density", &K::density, any_real(),
             {.spellings = {{"Solid_density"}, {"Density"}}, .doc = "the density rho"})
        .key("Elasticity_modulus", &K::elasticity_modulus, any_real(), {.doc = "Young's modulus E"})
        .key("Poisson_ratio", &K::poisson_ratio, any_real(), {.doc = "Poisson's ratio nu"})
        .key("Penalty_parameter", &K::penalty_parameter, opt::OptionalCodec<opt::RealCodec>{any_real()},
             {.empty = opt::EmptyPolicy::Parse, .doc = "the volumetric penalty (default: the bulk modulus)"})
        .key("Dilational_penalty_model", &K::volumetric_model,
             opt::EnumCodec<Model>{{{Model::ST91, "ST91", {"Simo-Taylor91"}},
                                    {Model::Quadratic, "quadratic", {"quad"}},
                                    {Model::M94, "M94", {"Miehe94"}},
                                    {Model::None, "none", {"na"}}}},
             {.doc = "the volumetric energy"})
        .key("Force_x", &K::force_x, any_real(), {.doc = "body force, x"})
        .key("Force_y", &K::force_y, any_real(), {.doc = "body force, y"})
        .key("Force_z", &K::force_z, any_real(), {.doc = "body force, z"})
        .key("Momentum_stabilization_coefficient", &K::momentum_stabilization_coefficient, any_real(),
             {.doc = "ct_m"})
        .key("Continuity_stabilization_coefficient", &K::continuity_stabilization_coefficient, any_real(),
             {.doc = "ct_c"});
    return out;
  }();
  return schema;
}

const opt::OptionSchema<UstructBoundaryKeys>& ustruct_boundary_schema()
{
  using K = UstructBoundaryKeys;
  using Type = UstructBoundaryType;
  const auto relaxed = opt::BoolCodec{opt::BoolLexicon::Relaxed};
  static const auto schema = [relaxed] {
    opt::OptionSchema<K> out("ustruct boundary condition");
    out.key("Time_dependence", &K::time_dependence, opt::StringCodec{},
            {.doc = "Steady, or Unsteady for a file-driven follower pressure"})
        .key("Type", &K::type,
             opt::EnumCodec<Type>{{{Type::Dirichlet, "Dirichlet", {"Dir"}},
                                   {Type::Neumann, "Neumann", {"Neu", "Traction", "Trac"}},
                                   {Type::PressureDirichlet, "Pressure_dirichlet", {}}}},
             {.doc = "the condition"})
        .key("Value", &K::value, any_real(), {.doc = "the magnitude"})
        .key("Effective_direction", &K::effective_direction, opt::StringCodec{},
             {.doc = "integer direction of a vector value"})
        .key("Follower_pressure_load", &K::follower_pressure_load, relaxed, {.doc = "a follower pressure"})
        .key("Apply_along_normal_direction", &K::apply_along_normal_direction, relaxed,
             {.doc = "a traction along the normal"})
        .key("Impose_on_state_variable_integral", &K::impose_on_state_variable_integral, relaxed,
             {.empty = opt::EmptyPolicy::Absent, .doc = "a Dirichlet value on the displacement"})
        .key("Ramp_function", &K::ramp_function, relaxed,
             {.empty = opt::EmptyPolicy::Absent, .doc = "a linear ramp between the file's end values"})
        .key("Temporal_values_file_path", &K::temporal_values_file_path, opt::StringCodec{},
             {.doc = "the follower pressure in time"});
    return out;
  }();
  return schema;
}

struct SolidMaterialProperties {
  svmp::FE::Real density{1.0};
  svmp::FE::Real elasticity_modulus{1.0};
  svmp::FE::Real poisson_ratio{0.49};
  svmp::FE::Real shear_modulus{0.0};
  svmp::FE::Real bulk_modulus{0.0};
  svmp::FE::Real bulk_wave_speed{0.0};
};

std::string normalize_output_name(std::string s)
{
  s = lower_copy(trim_copy(std::move(s)));
  s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch) {
            return ch == '_' || ch == '-' || std::isspace(ch);
          }),
          s.end());
  return s;
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

// Validates the moduli and derives the shear and bulk moduli and the bulk wave speed.
SolidMaterialProperties solid_properties(const UstructDomainKeys& keys)
{
  SolidMaterialProperties props{};
  props.density = keys.density;
  props.elasticity_modulus = keys.elasticity_modulus;
  props.poisson_ratio = keys.poisson_ratio;

  if (!(props.density > 0.0)) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct density must be > 0.");
  }
  if (!(props.elasticity_modulus > 0.0)) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct Elasticity_modulus must be > 0.");
  }
  if (!(props.poisson_ratio > -1.0 && props.poisson_ratio <= 0.5)) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct Poisson_ratio must be in (-1, 0.5].");
  }

  props.shear_modulus = static_cast<Real>(0.5) * props.elasticity_modulus /
                        (static_cast<Real>(1.0) + props.poisson_ratio);

  if (std::abs(props.poisson_ratio - static_cast<Real>(0.5)) <= static_cast<Real>(1e-12)) {
    props.bulk_modulus = 0.0;
    props.bulk_wave_speed = std::sqrt(props.shear_modulus / props.density);
  } else {
    props.bulk_modulus = props.elasticity_modulus /
                         (static_cast<Real>(3.0) *
                          (static_cast<Real>(1.0) - static_cast<Real>(2.0) * props.poisson_ratio));
    const auto lambda = props.elasticity_modulus * props.poisson_ratio /
                        ((static_cast<Real>(1.0) + props.poisson_ratio) *
                         (static_cast<Real>(1.0) - static_cast<Real>(2.0) * props.poisson_ratio));
    props.bulk_wave_speed =
        std::sqrt((lambda + static_cast<Real>(2.0) * props.shear_modulus) / props.density);
  }

  return props;
}

void apply_solid_options(const UstructDomainKeys& keys,
                         const SolidMaterialProperties& props,
                         svmp::Physics::formulations::ustruct::UstructOptions& options)
{
  options.density = props.density;
  options.bulk_wave_speed = props.bulk_wave_speed;
  options.kinematic_residual_scale = std::sqrt(props.density * props.shear_modulus);
  options.deviatoric_pk1_model =
      svmp::Physics::materials::solid::isochoricNeoHookeanPK1(props.shear_modulus).model();

  options.penalty_parameter = keys.penalty_parameter.value_or(props.bulk_modulus);
  if (options.penalty_parameter < 0.0) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct Penalty_parameter must be >= 0.");
  }
  options.volumetric_model = keys.volumetric_model;
  options.body_force = {keys.force_x, keys.force_y, keys.force_z};
  options.ct_m = keys.momentum_stabilization_coefficient;
  options.ct_c = keys.continuity_stabilization_coefficient;
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

    const auto keys = ustruct_boundary_schema().resolve(option_input(bc.params)).value;
    const auto time_value_lc = lower_copy(keys.time_dependence);
    const bool is_steady = time_value_lc == "steady";
    const bool is_unsteady = time_value_lc == "unsteady";
    const bool has_temporal_values_file = !keys.temporal_values_file_path.empty();
    const bool has_unsupported_file =
        has_nonempty_defined(bc.params, "Spatial_values_file_path") ||
        has_nonempty_defined(bc.params, "Temporal_and_spatial_values_file_path") ||
        has_nonempty_defined(bc.params, "Bct_file_path") ||
        has_nonempty_defined(bc.params, "Traction_values_file_path");
    const bool is_neumann_like = keys.type == UstructBoundaryType::Neumann;
    const bool follower_pressure = keys.follower_pressure_load;

    if (!is_steady && !is_unsteady) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Only Steady ustruct boundary conditions and Unsteady file-driven follower-pressure "
          "BCs are supported for the new solver ustruct module (got Time_dependence='" + keys.time_dependence +
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

    const auto value = keys.value;
    std::vector<int> effective_dir;
    if (!keys.effective_direction.empty()) {
      effective_dir = parse_int_list(keys.effective_direction);
    }

    switch (keys.type) {
    case UstructBoundaryType::Dirichlet: {
      const bool impose_on_integral =
          !options.enable_time_derivative_terms || keys.impose_on_state_variable_integral;
      if (impose_on_integral) {
        UstructOptions::DisplacementDirichletBC dir{};
        dir.boundary_marker = bc.boundary_marker;
        dir.active_components = fill_dirichlet_vector(dir.value, dim, effective_dir, value);
        if (std::any_of(dir.active_components.begin(), dir.active_components.end(), [](bool b) { return b; })) {
          options.displacement_dirichlet.push_back(std::move(dir));
        }

        UstructOptions::VelocityDirichletBC vel{};
        vel.boundary_marker = bc.boundary_marker;
        vel.active_components = fill_dirichlet_vector(vel.value, dim, effective_dir, static_cast<Real>(0.0));
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
      break;
    }
    case UstructBoundaryType::Neumann: {
      if (follower_pressure) {
        UstructOptions::FollowerPressureBC pressure{};
        pressure.boundary_marker = bc.boundary_marker;
        if (is_unsteady) {
          auto temporal = svmp::Physics::readTemporalValuesFile(
              keys.temporal_values_file_path, /*num_components=*/1, svmp::Physics::TemporalEndBehavior::Clamp);
          if (keys.ramp_function) {
            pressure.ramp = UstructOptions::FollowerPressureBC::LinearRamp{
                static_cast<Real>(temporal->firstTime()),
                static_cast<Real>(temporal->lastTime()),
                static_cast<Real>(temporal->firstValue()),
                static_cast<Real>(temporal->lastValue())};
          } else {
            pressure.pressure = UstructOptions::ScalarValue{
                svmp::FE::forms::TimeScalarCoefficient(
                    [temporal](Real /*x*/, Real /*y*/, Real /*z*/, Real t) -> Real {
                      return static_cast<Real>(temporal->interpolate(static_cast<double>(t)));
                    })};
          }
        } else {
          pressure.pressure = UstructOptions::ScalarValue{value};
        }
        options.follower_pressure.push_back(std::move(pressure));
      } else if (keys.apply_along_normal_direction) {
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
      break;
    }
    case UstructBoundaryType::PressureDirichlet: {
      UstructOptions::PressureDirichletBC pbc{};
      pbc.boundary_marker = bc.boundary_marker;
      pbc.value = UstructOptions::ScalarValue{value};
      options.pressure_dirichlet.push_back(std::move(pbc));
      break;
    }
    case UstructBoundaryType::None:
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Boundary condition '" + bc.name +
          "' has no Type. Supported types: Dir, Dirichlet, Neu, Neumann, Trac, Traction.");
    }
  }
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_ustruct_from_input(const svmp::Physics::EquationModuleInput& input,
                          svmp::FE::systems::FESystem& system)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] ustruct module factory received null mesh.");
  }

  const auto& domain = select_single_domain(input, "ustruct");
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

  const auto domain_keys = ustruct_domain_schema().resolve(option_input(domain.params)).value;
  apply_solid_options(domain_keys, solid_properties(domain_keys), options);
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

std::string resolvedInputKeys(const EquationModuleInput& input)
{
  const auto& domain_schema = ustruct_domain_schema();
  const auto& boundary_schema = ustruct_boundary_schema();
  const auto& domain = select_single_domain(input, "ustruct");
  std::string out = option_line(domain_schema.element(),
                                    domain_schema.serialize(domain_schema.resolve(option_input(domain.params)).value));
  for (const auto& bc : input.boundary_conditions) {
    out += option_line(boundary_schema.element() + " '" + bc.name + "'",
                           boundary_schema.serialize(boundary_schema.resolve(option_input(bc.params)).value));
  }
  return out;
}

} // namespace svmp::Physics::formulations::ustruct
