#include "Application/Translators/EquationTranslator.h"

#include "Application/Core/OopMpiLog.h"
#include "Mesh/Core/MeshBase.h"
#include "Parameters.h"
#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <variant>

namespace {

svmp::Physics::ParameterMap snapshot_params(const ParameterLists& list)
{
  svmp::Physics::ParameterMap out;
  for (const auto& [_, v] : list.params_map) {
    std::visit(
        [&](const auto* p) {
          if (!p) {
            return;
          }
          out[p->name()] = svmp::Physics::ParameterValue{p->defined(), p->svalue()};
        },
        v);
  }
  return out;
}

svmp::Physics::DomainInput snapshot_domain(const DomainParameters& domain)
{
  svmp::Physics::DomainInput out{};
  out.id = domain.id.value();
  out.params = snapshot_params(domain);

  if (domain.fluid_viscosity.model.defined()) {
    out.params["Viscosity.model"] =
        svmp::Physics::ParameterValue{domain.fluid_viscosity.model.defined(), domain.fluid_viscosity.model.value()};

    const auto append = [&](const ParameterLists& list, const std::string& prefix) {
      const auto block = snapshot_params(list);
      for (const auto& [k, v] : block) {
        out.params[prefix + k] = v;
      }
    };

    append(domain.fluid_viscosity.newtonian_model, "Viscosity.");
    append(domain.fluid_viscosity.carreau_yasuda_model, "Viscosity.");
    append(domain.fluid_viscosity.cassons_model, "Viscosity.");
  }

  return out;
}

} // namespace

namespace application {
namespace translators {

std::unique_ptr<svmp::Physics::PhysicsModule> EquationTranslator::createModule(
    const EquationParameters& eq_params, svmp::FE::systems::FESystem& system,
    const std::map<std::string, std::shared_ptr<svmp::Mesh>>& meshes)
{
  const std::string eq_type = eq_params.type.value();

  application::core::oopCout() << "[svMultiPhysics::Application] EquationTranslator: createModule(type='" << eq_type
                               << "')" << std::endl;

  if (meshes.empty()) {
    throw std::runtime_error("[svMultiPhysics::Application] No meshes are available for equation translation.");
  }

  if (meshes.size() != 1) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Multiple <Add_mesh> blocks are not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto mesh_name = meshes.begin()->first;
  auto mesh = meshes.begin()->second;
  if (!mesh) {
    throw std::runtime_error("[svMultiPhysics::Application] Null mesh encountered during equation translation.");
  }

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = eq_type;
  input.equation_params = snapshot_params(eq_params);
  input.mesh_name = mesh_name;
  input.mesh = mesh->local_mesh_ptr();

  application::core::oopCout() << "[svMultiPhysics::Application] EquationTranslator: mesh='" << mesh_name << "'"
                               << " domains=" << static_cast<int>(eq_params.domains.size())
                               << " boundary_conditions=" << static_cast<int>(eq_params.boundary_conditions.size())
                               << std::endl;

  // Generic module-specific options hook (added to EquationParameters once).
  // If unset, these remain empty and the formulation uses its defaults.
  if (eq_params.module_options.defined()) {
    input.module_options = eq_params.module_options.value();
    application::core::oopCout() << "[svMultiPhysics::Application] EquationTranslator: module_options='"
                                 << input.module_options << "'" << std::endl;
  }
  if (eq_params.module_options_file_path.defined()) {
    input.module_options_file_path = eq_params.module_options_file_path.value();
    application::core::oopCout() << "[svMultiPhysics::Application] EquationTranslator: module_options_file_path='"
                                 << input.module_options_file_path << "'" << std::endl;
  }

  if (eq_params.default_domain) {
    input.default_domain = snapshot_domain(*eq_params.default_domain);
  }

  if (!eq_params.domains.empty()) {
    input.domains.reserve(eq_params.domains.size());
    for (const auto* d : eq_params.domains) {
      if (!d) {
        continue;
      }
      input.domains.push_back(snapshot_domain(*d));
    }
  }

  if (!eq_params.boundary_conditions.empty()) {
    input.boundary_conditions.reserve(eq_params.boundary_conditions.size());
    for (const auto* bc : eq_params.boundary_conditions) {
      if (!bc) {
        continue;
      }

      svmp::Physics::BoundaryConditionInput bc_in{};
      bc_in.name = bc->name.value();
      bc_in.boundary_marker = mesh->label_from_name(bc_in.name);
      application::core::oopCout() << "[svMultiPhysics::Application]   BC '" << bc_in.name
                                   << "': boundary_marker=" << bc_in.boundary_marker << std::endl;
      if (bc_in.boundary_marker == svmp::INVALID_LABEL) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Boundary condition references face '" + bc_in.name +
            "', but that face is not registered. Ensure <Add_face name=\"" + bc_in.name +
            "\"> exists under the mesh and <Add_BC name=\"" + bc_in.name +
            "\"> references it, or set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }

      bc_in.params = snapshot_params(*bc);
      if (bc->rcr.value_set) {
        const auto rcr = snapshot_params(bc->rcr);
        for (const auto& [k, v] : rcr) {
          bc_in.params["RCR." + k] = v;
        }
      }
      input.boundary_conditions.push_back(std::move(bc_in));
    }
  }

  auto& registry = svmp::Physics::EquationModuleRegistry::instance();
  const auto types = registry.registeredTypes();
  const auto supported = std::find(types.begin(), types.end(), eq_type) != types.end();
  if (!supported) {
    std::string supported_list = types.empty() ? "(none)" : types.front();
    for (std::size_t i = 1; i < types.size(); ++i) {
      supported_list += ", " + types[i];
    }

    throw std::runtime_error(
        "[svMultiPhysics::Application] Equation type '" + eq_type +
        "' is not registered for the new OOP solver. Registered types: " + supported_list +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  return registry.create(eq_type, input, system);
}

} // namespace translators
} // namespace application
