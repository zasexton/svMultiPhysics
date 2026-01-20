#include "Application/Translators/BoundaryConditionTranslator.h"

#include "Mesh/Core/MeshBase.h"
#include "Parameters.h"
#include "Physics/Formulations/Poisson/PoissonModule.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace {

std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

} // namespace

namespace application {
namespace translators {

void BoundaryConditionTranslator::applyScalarBCs(
    const std::vector<BoundaryConditionParameters*>& bc_params, const svmp::MeshBase& mesh,
    svmp::Physics::formulations::poisson::PoissonOptions& options)
{
  using svmp::Physics::formulations::poisson::PoissonOptions;

  for (const auto* bc : bc_params) {
    if (!bc) {
      continue;
    }

    const auto face_name = bc->name.value();
    const auto marker = mesh.label_from_name(face_name);
    if (marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Boundary condition references face '" + face_name +
          "', but that face is not registered. Ensure <Add_face name=\"" + face_name +
          "\"> exists under the mesh and <Add_BC name=\"" + face_name +
          "\"> references it, or set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    const auto time_dep = bc->time_dependence.defined() ? trim_copy(bc->time_dependence.value()) : "Steady";
    if (time_dep != "Steady") {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Only steady boundary conditions are supported for the new solver Poisson module "
          "(got Time_dependence='" +
          time_dep +
          "'). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if ((bc->temporal_values_file_path.defined() && !bc->temporal_values_file_path.value().empty()) ||
        (bc->spatial_values_file_path.defined() && !bc->spatial_values_file_path.value().empty()) ||
        (bc->temporal_and_spatial_values_file_path.defined() &&
         !bc->temporal_and_spatial_values_file_path.value().empty()) ||
        (bc->bct_file_path.defined() && !bc->bct_file_path.value().empty())) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Spatial/temporal boundary condition files are not supported for the new solver "
          "Poisson module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    const auto bc_type = trim_copy(bc->type.value());
    const auto value = static_cast<svmp::FE::Real>(bc->value.value());

    if (bc_type == "Dirichlet" || bc_type == "Dir") {
      PoissonOptions::DirichletBC dir{};
      dir.boundary_marker = marker;
      dir.value = value;

      const bool weak = (bc->weakly_applied.defined() && bc->weakly_applied.value());
      if (weak) {
        options.dirichlet_weak.push_back(std::move(dir));
      } else {
        options.dirichlet.push_back(std::move(dir));
      }
      continue;
    }

    if (bc_type == "Neumann" || bc_type == "Neu") {
      PoissonOptions::NeumannBC neu{};
      neu.boundary_marker = marker;
      neu.flux = value;
      options.neumann.push_back(std::move(neu));
      continue;
    }

    if (bc_type == "Robin" || bc_type == "Rbn") {
      PoissonOptions::RobinBC robin{};
      robin.boundary_marker = marker;
      robin.alpha = static_cast<svmp::FE::Real>(bc->stiffness.value());
      robin.rhs = value;
      options.robin.push_back(std::move(robin));
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Application] Boundary condition type '" + bc_type +
        "' is not supported for the new solver Poisson module. Supported types: Dir, Dirichlet, Neu, Neumann, Robin, Rbn. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

} // namespace translators
} // namespace application
