#include "Physics/Formulations/Poisson/PoissonModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Options/InputReaders.h"

#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// The generic readers of equation input (decision D52).
using namespace svmp::Physics::options;

svmp::Physics::formulations::poisson::NodeIdType parse_node_id_type(std::string_view id_type)
{
  const auto value = trim_copy(std::string(id_type));
  if (value == "Global_vertex_gid") {
    return svmp::Physics::formulations::poisson::NodeIdType::GlobalVertexGid;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Physics] Unsupported <Node_pressure_constraints><Id_type> '" + value +
      "'. Supported value: Global_vertex_gid.");
}

std::vector<svmp::Physics::formulations::poisson::PoissonOptions::NodeDirichletBC>
read_node_pressure_csv(const std::string& path)
{
  using svmp::Physics::formulations::poisson::PoissonOptions;

  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to open node pressure constraints CSV file '" +
                             path + "'.");
  }

  std::vector<PoissonOptions::NodeDirichletBC> out;
  std::unordered_map<svmp::FE::GlobalIndex, svmp::FE::Real> seen;

  bool header_seen = false;
  bool data_seen = false;
  int node_col = 0;
  int pressure_col = 1;

  std::string line;
  int line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    auto trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }

    auto fields = split_csv_line(trimmed);
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
    out.push_back(PoissonOptions::NodeDirichletBC{node_id, pressure});
    data_seen = true;
  }

  if (!data_seen) {
    throw std::runtime_error("[svMultiPhysics::Physics] Node pressure constraints CSV file '" + path +
                             "' did not contain any node pressure values.");
  }

  return out;
}

void apply_thermal_properties(const svmp::Physics::EquationModuleInput& input,
                              svmp::Physics::formulations::poisson::PoissonOptions& options)
{
  const auto& dom = select_single_domain(input, "Poisson");
  if (const auto* p = find_param(dom.params, "Conductivity"); p && p->defined) {
    options.diffusion = static_cast<svmp::FE::Real>(parse_double(p->value, "Domain/Conductivity"));
  } else if (const auto* q = find_param(dom.params, "Isotropic_conductivity"); q && q->defined) {
    options.diffusion = static_cast<svmp::FE::Real>(parse_double(q->value, "Domain/Isotropic_conductivity"));
  }

  if (const auto* p = find_param(dom.params, "Source_term"); p && p->defined) {
    options.source = static_cast<svmp::FE::Real>(parse_double(p->value, "Domain/Source_term"));
  }

  if (has_nonempty_defined(dom.params, "Anisotropic_conductivity")) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Anisotropic_conductivity is not supported for the new solver Poisson module yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

void apply_scalar_bcs(const svmp::Physics::EquationModuleInput& input,
                      svmp::Physics::formulations::poisson::PoissonOptions& options)
{
  using svmp::Physics::formulations::poisson::PoissonOptions;

  for (const auto& bc : input.boundary_conditions) {
    if (bc.boundary_marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto* time_dep = find_param(bc.params, "Time_dependence");
    const std::string time_value =
        (time_dep && time_dep->defined) ? trim_copy(time_dep->value) : std::string("Steady");
    if (time_value != "Steady") {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Only steady boundary conditions are supported for the new solver Poisson module "
          "(got Time_dependence='" +
          time_value + "'). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (has_nonempty_defined(bc.params, "Temporal_values_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_values_file_path") ||
        has_nonempty_defined(bc.params, "Temporal_and_spatial_values_file_path") ||
        has_nonempty_defined(bc.params, "Bct_file_path")) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Spatial/temporal boundary condition files are not supported for the new solver "
          "Poisson module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    const auto* type_param = find_param(bc.params, "Type");
    const std::string bc_type = type_param ? trim_copy(type_param->value) : std::string{};

    const auto* value_param = find_param(bc.params, "Value");
    const svmp::FE::Real value =
        static_cast<svmp::FE::Real>(value_param ? parse_double(value_param->value, "Add_BC/Value") : 0.0);

    const bool weak = get_defined_bool(bc.params, "Weakly_applied").value_or(false);

    if (bc_type == "Dirichlet" || bc_type == "Dir") {
      PoissonOptions::DirichletBC dir{};
      dir.boundary_marker = bc.boundary_marker;
      dir.value = value;
      if (weak) {
        options.dirichlet_weak.push_back(std::move(dir));
      } else {
        options.dirichlet.push_back(std::move(dir));
      }
      continue;
    }

    if (bc_type == "Neumann" || bc_type == "Neu") {
      PoissonOptions::NeumannBC neu{};
      neu.boundary_marker = bc.boundary_marker;
      neu.value = value;
      options.neumann.push_back(std::move(neu));
      continue;
    }

    if (bc_type == "Robin" || bc_type == "Rbn") {
      const auto* stiff_param = find_param(bc.params, "Stiffness");
      const svmp::FE::Real stiff =
          static_cast<svmp::FE::Real>(stiff_param ? parse_double(stiff_param->value, "Add_BC/Stiffness") : 0.0);

      PoissonOptions::RobinBC robin{};
      robin.boundary_marker = bc.boundary_marker;
      robin.value.alpha = stiff;
      robin.value.rhs = value;
      options.robin.push_back(std::move(robin));
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary condition type '" + bc_type +
        "' is not supported for the new solver Poisson module. Supported types: Dir, Dirichlet, Neu, Neumann, Robin, Rbn. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

void apply_node_pressure_constraints(const svmp::Physics::EquationModuleInput& input,
                                     svmp::Physics::formulations::poisson::PoissonOptions& options)
{
  if (!input.node_pressure_constraints.has_value()) {
    return;
  }

  const auto& node_constraints = *input.node_pressure_constraints;
  options.node_dirichlet.id_type = parse_node_id_type(node_constraints.id_type);
  options.node_dirichlet.values = read_node_pressure_csv(node_constraints.values_file_path);
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_poisson_with_field_name(const svmp::Physics::EquationModuleInput& input,
                               svmp::FE::systems::FESystem& system,
                               std::string field_name,
                               bool register_darcy_flux_output = false)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Poisson module factory received null mesh.");
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int order = resolve_element_order(input, infer_polynomial_order(*input.mesh));

  auto space = svmp::FE::spaces::SpaceFactory::create_h1(element_type, order);

  svmp::Physics::formulations::poisson::PoissonOptions options{};
  options.field_name = std::move(field_name);
  options.register_darcy_flux_output = register_darcy_flux_output;
  options.jit_policy = svmp::Physics::core::resolveOopJitPolicy(input, options.jit_policy);

  apply_thermal_properties(input, options);
  apply_scalar_bcs(input, options);
  apply_node_pressure_constraints(input, options);

  auto module = std::make_unique<svmp::Physics::formulations::poisson::PoissonModule>(std::move(space),
                                                                                      std::move(options));
  module->registerOn(system);
  return module;
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_poisson_from_input(const svmp::Physics::EquationModuleInput& input,
                          svmp::FE::systems::FESystem& system)
{
  return create_poisson_with_field_name(input, system, "Temperature");
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_darcy_pressure_from_input(const svmp::Physics::EquationModuleInput& input,
                                 svmp::FE::systems::FESystem& system)
{
  return create_poisson_with_field_name(input, system, "Pressure", true);
}

} // namespace

SVMP_REGISTER_EQUATION("heatS", &create_poisson_from_input);
SVMP_REGISTER_EQUATION("heatF", &create_poisson_from_input);
SVMP_REGISTER_EQUATION("darcy", &create_darcy_pressure_from_input);

namespace svmp::Physics::formulations::poisson {

void forceLink_PoissonRegister() {}

} // namespace svmp::Physics::formulations::poisson
