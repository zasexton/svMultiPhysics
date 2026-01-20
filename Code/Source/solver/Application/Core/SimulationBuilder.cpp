#include "Application/Core/SimulationBuilder.h"

#include "Application/Translators/MeshTranslator.h"

#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/LinearSolver.h"
#include "FE/Systems/FESystem.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "Physics/Core/PhysicsModule.h"
#include "Parameters.h"

#include <iostream>
#include <stdexcept>

namespace application {
namespace core {

SimulationComponents::SimulationComponents() = default;

SimulationComponents::SimulationComponents(SimulationComponents&&) noexcept = default;

SimulationComponents& SimulationComponents::operator=(SimulationComponents&&) noexcept = default;

SimulationComponents::~SimulationComponents() = default;

SimulationBuilder::SimulationBuilder(const Parameters& params)
  : params_(params)
{
}

SimulationComponents SimulationBuilder::build()
{
  std::cout << "[svMultiPhysics::Application] SimulationBuilder: loadMeshes()" << std::endl;
  loadMeshes();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: createFESystem() (not yet implemented)" << std::endl;
  createFESystem();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: createPhysicsModules() (not yet implemented)" << std::endl;
  createPhysicsModules();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: setupSystem() (not yet implemented)" << std::endl;
  setupSystem();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: createSolvers() (not yet implemented)" << std::endl;
  createSolvers();

  std::cout << "[svMultiPhysics::Application] SimulationBuilder: allocateHistory() (not yet implemented)" << std::endl;
  allocateHistory();

  return std::move(components_);
}

void SimulationBuilder::loadMeshes()
{
  for (const auto* mesh_params : params_.mesh_parameters) {
    if (!mesh_params) {
      continue;
    }

    const auto mesh_name = mesh_params->name.value();
    if (mesh_name.empty()) {
      throw std::runtime_error("[svMultiPhysics::Application] <Add_mesh> is missing required name attribute.");
    }

    if (components_.meshes.count(mesh_name) != 0) {
      throw std::runtime_error("[svMultiPhysics::Application] Duplicate <Add_mesh name=\"" + mesh_name +
                               "\"> detected.");
    }

    auto mesh = application::translators::MeshTranslator::loadMesh(*mesh_params);
    components_.meshes.emplace(mesh_name, mesh);

    if (!components_.primary_mesh) {
      components_.primary_mesh = mesh;
      components_.primary_mesh_name = mesh_name;
    }
  }
}

void SimulationBuilder::createFESystem()
{
}

void SimulationBuilder::createPhysicsModules()
{
}

void SimulationBuilder::setupSystem()
{
}

void SimulationBuilder::createSolvers()
{
}

void SimulationBuilder::allocateHistory()
{
}

} // namespace core
} // namespace application
