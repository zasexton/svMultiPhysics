#include "Physics/Formulations/Poisson/PoissonModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"

#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

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

const svmp::Physics::ParameterValue* find_param(const svmp::Physics::ParameterMap& params,
                                                std::string_view key)
{
  const auto it = params.find(std::string(key));
  if (it == params.end()) {
    return nullptr;
  }
  return &it->second;
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

void apply_thermal_properties(const svmp::Physics::EquationModuleInput& input,
                              svmp::Physics::formulations::poisson::PoissonOptions& options)
{
  using svmp::Physics::ParameterValue;

  if (!input.domains.empty()) {
    if (input.domains.size() != 1) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Multiple <Domain> blocks are not supported for the new solver Poisson module yet. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    const auto& dom = input.domains.front();
    if (const auto* p = find_param(dom.params, "Conductivity"); p && p->defined) {
      options.diffusion = static_cast<svmp::FE::Real>(parse_double(p->value, "Domain/Conductivity"));
    } else if (const auto* q = find_param(dom.params, "Isotropic_conductivity"); q && q->defined) {
      options.diffusion = static_cast<svmp::FE::Real>(parse_double(q->value, "Domain/Isotropic_conductivity"));
    }

    if (const auto* p = find_param(dom.params, "Source_term"); p && p->defined) {
      options.source = static_cast<svmp::FE::Real>(parse_double(p->value, "Domain/Source_term"));
    }

    if (const auto* p = find_param(dom.params, "Anisotropic_conductivity"); p && p->defined &&
                           !trim_copy(p->value).empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Anisotropic_conductivity is not supported for the new solver Poisson module yet. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
    return;
  }

  const auto& dom = input.default_domain;
  if (const auto* p = find_param(dom.params, "Conductivity"); p && p->defined) {
    options.diffusion = static_cast<svmp::FE::Real>(parse_double(p->value, "Domain/Conductivity"));
  } else if (const auto* q = find_param(dom.params, "Isotropic_conductivity"); q && q->defined) {
    options.diffusion = static_cast<svmp::FE::Real>(parse_double(q->value, "Domain/Isotropic_conductivity"));
  }

  if (const auto* p = find_param(dom.params, "Source_term"); p && p->defined) {
    options.source = static_cast<svmp::FE::Real>(parse_double(p->value, "Domain/Source_term"));
  }

  if (const auto* p = find_param(dom.params, "Anisotropic_conductivity"); p && p->defined &&
                         !trim_copy(p->value).empty()) {
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

    const auto has_nonempty_defined = [&](std::string_view key) {
      const auto* p = find_param(bc.params, key);
      return (p && p->defined && !trim_copy(p->value).empty());
    };

    if (has_nonempty_defined("Temporal_values_file_path") || has_nonempty_defined("Spatial_values_file_path") ||
        has_nonempty_defined("Temporal_and_spatial_values_file_path") || has_nonempty_defined("Bct_file_path")) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Spatial/temporal boundary condition files are not supported for the new solver "
          "Poisson module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    const auto* type_param = find_param(bc.params, "Type");
    const std::string bc_type = type_param ? trim_copy(type_param->value) : std::string{};

    const auto* value_param = find_param(bc.params, "Value");
    const svmp::FE::Real value =
        static_cast<svmp::FE::Real>(value_param ? parse_double(value_param->value, "Add_BC/Value") : 0.0);

    const auto* weak_param = find_param(bc.params, "Weakly_applied");
    const bool weak = weak_param && weak_param->defined && (lower_copy(trim_copy(weak_param->value)) == "1" ||
                                                            lower_copy(trim_copy(weak_param->value)) == "true" ||
                                                            lower_copy(trim_copy(weak_param->value)) == "yes" ||
                                                            lower_copy(trim_copy(weak_param->value)) == "on");

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
      neu.flux = value;
      options.neumann.push_back(std::move(neu));
      continue;
    }

    if (bc_type == "Robin" || bc_type == "Rbn") {
      const auto* stiff_param = find_param(bc.params, "Stiffness");
      const svmp::FE::Real stiff =
          static_cast<svmp::FE::Real>(stiff_param ? parse_double(stiff_param->value, "Add_BC/Stiffness") : 0.0);

      PoissonOptions::RobinBC robin{};
      robin.boundary_marker = bc.boundary_marker;
      robin.alpha = stiff;
      robin.rhs = value;
      options.robin.push_back(std::move(robin));
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary condition type '" + bc_type +
        "' is not supported for the new solver Poisson module. Supported types: Dir, Dirichlet, Neu, Neumann, Robin, Rbn. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_poisson_from_input(const svmp::Physics::EquationModuleInput& input,
                          svmp::FE::systems::FESystem& system)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Poisson module factory received null mesh.");
  }

  const auto element_type = infer_base_element_type(*input.mesh);
  const int order = infer_polynomial_order(*input.mesh);

  auto space = svmp::FE::spaces::SpaceFactory::create_h1(element_type, order);

  svmp::Physics::formulations::poisson::PoissonOptions options{};
  options.field_name = "Temperature";

  apply_thermal_properties(input, options);
  apply_scalar_bcs(input, options);

  auto module = std::make_unique<svmp::Physics::formulations::poisson::PoissonModule>(std::move(space),
                                                                                      std::move(options));
  module->registerOn(system);
  return module;
}

} // namespace

SVMP_REGISTER_EQUATION("heatS", &create_poisson_from_input);
SVMP_REGISTER_EQUATION("heatF", &create_poisson_from_input);

