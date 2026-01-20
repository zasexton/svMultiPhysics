#include "Application/Translators/EquationTranslator.h"

#include "Application/Translators/BoundaryConditionTranslator.h"
#include "Application/Translators/MaterialTranslator.h"

#include "FE/Core/Types.h"
#include "FE/Spaces/SpaceFactory.h"
#include "Mesh/Core/MeshBase.h"
#include "Parameters.h"
#include "Physics/Formulations/Poisson/PoissonModule.h"

#include <stdexcept>
#include <vector>

namespace {

svmp::FE::ElementType infer_base_element_type(const svmp::MeshBase& mesh)
{
  if (mesh.n_cells() == 0) {
    throw std::runtime_error("[svMultiPhysics::Application] Mesh has no cells; cannot infer FE element type.");
  }

  const auto& shapes = mesh.cell_shapes();
  if (shapes.empty()) {
    throw std::runtime_error("[svMultiPhysics::Application] Mesh has no cell shapes; cannot infer FE element type.");
  }

  if (shapes.front().is_mixed_order) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Mixed-order meshes are not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto family = shapes.front().family;
  for (const auto& s : shapes) {
    if (s.family != family) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Mixed cell families are not supported by the new solver yet. "
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
      "[svMultiPhysics::Application] Unsupported mesh cell family for new solver. "
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
          "[svMultiPhysics::Application] Mixed polynomial orders are not supported by the new solver yet. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
  }

  return order;
}

} // namespace

namespace application {
namespace translators {

std::unique_ptr<svmp::Physics::PhysicsModule> EquationTranslator::createModule(
    const EquationParameters& eq_params, svmp::FE::systems::FESystem& system,
    const std::map<std::string, std::shared_ptr<svmp::MeshBase>>& meshes)
{
  const std::string eq_type = eq_params.type.value();

  if (meshes.empty()) {
    throw std::runtime_error("[svMultiPhysics::Application] No meshes are available for equation translation.");
  }

  if (meshes.size() != 1) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Multiple <Add_mesh> blocks are not supported by the new solver yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  auto mesh = meshes.begin()->second;
  if (!mesh) {
    throw std::runtime_error("[svMultiPhysics::Application] Null mesh encountered during equation translation.");
  }

  if (eq_type == "heatS" || eq_type == "heatF") {
    return createHeatModule(eq_params, system, mesh);
  }

  throw std::runtime_error(
      "[svMultiPhysics::Application] Equation type '" + eq_type +
      "' is not yet supported by the new OOP solver. Supported types: heatS, heatF. "
      "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
}

std::unique_ptr<svmp::Physics::PhysicsModule> EquationTranslator::createHeatModule(
    const EquationParameters& eq_params, svmp::FE::systems::FESystem& system,
    const std::shared_ptr<svmp::MeshBase>& mesh)
{
  auto space = createScalarSpace(eq_params, mesh);

  svmp::Physics::formulations::poisson::PoissonOptions options{};
  options.field_name = "Temperature";

  std::vector<const DomainParameters*> domains;
  if (!eq_params.domains.empty()) {
    domains.reserve(eq_params.domains.size());
    for (const auto* d : eq_params.domains) {
      domains.push_back(d);
    }
  } else if (eq_params.default_domain) {
    domains.push_back(eq_params.default_domain);
  }

  MaterialTranslator::applyThermalProperties(domains, options);
  BoundaryConditionTranslator::applyScalarBCs(eq_params.boundary_conditions, *mesh, options);

  auto module = std::make_unique<svmp::Physics::formulations::poisson::PoissonModule>(std::move(space),
                                                                                      std::move(options));
  module->registerOn(system);
  return module;
}

std::shared_ptr<const svmp::FE::spaces::FunctionSpace> EquationTranslator::createScalarSpace(
    const EquationParameters& /*eq_params*/, const std::shared_ptr<svmp::MeshBase>& mesh)
{
  if (!mesh) {
    throw std::runtime_error("[svMultiPhysics::Application] createScalarSpace() called with null mesh.");
  }

  const auto element_type = infer_base_element_type(*mesh);
  const int order = infer_polynomial_order(*mesh);

  auto space = svmp::FE::spaces::SpaceFactory::create_h1(element_type, order);
  return space;
}

} // namespace translators
} // namespace application
