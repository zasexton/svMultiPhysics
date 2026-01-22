#pragma once

#include <map>
#include <memory>
#include <string>

#include "Mesh/Mesh.h"

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
namespace Physics {
class PhysicsModule;
}
} // namespace svmp

namespace application {
namespace translators {

class EquationTranslator {
public:
  static std::unique_ptr<svmp::Physics::PhysicsModule> createModule(
      const EquationParameters& eq_params, svmp::FE::systems::FESystem& system,
      const std::map<std::string, std::shared_ptr<svmp::Mesh>>& meshes);
};

} // namespace translators
} // namespace application
