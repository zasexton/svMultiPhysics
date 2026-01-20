#pragma once

#include <vector>

class BoundaryConditionParameters;

namespace svmp {
class MeshBase;
namespace Physics {
namespace formulations {
namespace poisson {
struct PoissonOptions;
}
} // namespace formulations
} // namespace Physics
} // namespace svmp

namespace application {
namespace translators {

class BoundaryConditionTranslator {
public:
  BoundaryConditionTranslator() = delete;

  static void applyScalarBCs(const std::vector<BoundaryConditionParameters*>& bc_params,
                             const svmp::MeshBase& mesh,
                             svmp::Physics::formulations::poisson::PoissonOptions& options);
};

} // namespace translators
} // namespace application
