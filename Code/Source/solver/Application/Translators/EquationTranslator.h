#pragma once

#include <map>
#include <memory>
#include <string>

#include "Mesh/Mesh.h"
#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/PhysicsModule.h"

class EquationParameters;

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
}
namespace spaces {
class FunctionSpace;
}
} // namespace FE
} // namespace svmp

namespace application {
namespace translators {

class EquationTranslator {
public:
  static svmp::Physics::EquationModuleInput buildInput(
      const EquationParameters& eq_params,
      const std::map<std::string, std::shared_ptr<svmp::Mesh>>& meshes);

  static std::unique_ptr<svmp::Physics::PhysicsModule> createModule(
      const EquationParameters& eq_params, svmp::FE::systems::FESystem& system,
      const std::map<std::string, std::shared_ptr<svmp::Mesh>>& meshes);
};

} // namespace translators
} // namespace application
