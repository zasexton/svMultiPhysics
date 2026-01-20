#pragma once

#include <map>
#include <memory>
#include <string>

class EquationParameters;

namespace svmp {
class MeshBase;
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
      const std::map<std::string, std::shared_ptr<svmp::MeshBase>>& meshes);

private:
  static std::unique_ptr<svmp::Physics::PhysicsModule> createHeatModule(
      const EquationParameters& eq_params, svmp::FE::systems::FESystem& system,
      const std::shared_ptr<svmp::MeshBase>& mesh);

  static std::shared_ptr<const svmp::FE::spaces::FunctionSpace> createScalarSpace(
      const EquationParameters& eq_params, const std::shared_ptr<svmp::MeshBase>& mesh);
};

} // namespace translators
} // namespace application
