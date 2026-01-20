#pragma once

#include <vector>

class DomainParameters;

namespace svmp {
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

class MaterialTranslator {
public:
  MaterialTranslator() = delete;

  static void applyThermalProperties(const std::vector<const DomainParameters*>& domain_params,
                                     svmp::Physics::formulations::poisson::PoissonOptions& options);
};

} // namespace translators
} // namespace application
