#pragma once

#include "Physics/Core/PhysicsModule.h"

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::FE::systems {
class FESystem;
}

namespace svmp::Physics {

struct EquationModuleInput;

class EquationModuleRegistry {
public:
  using FactoryFn = std::unique_ptr<PhysicsModule> (*)(const EquationModuleInput&,
                                                       svmp::FE::systems::FESystem&);

  static EquationModuleRegistry& instance();

  void registerFactory(std::string type, FactoryFn factory);

  [[nodiscard]] std::unique_ptr<PhysicsModule> create(std::string_view type,
                                                      const EquationModuleInput& input,
                                                      svmp::FE::systems::FESystem& system) const;

  [[nodiscard]] std::vector<std::string> registeredTypes() const;

private:
  std::map<std::string, FactoryFn> factories_{};
};

#define SVMP_DETAIL_CONCAT_INNER(a, b) a##b
#define SVMP_DETAIL_CONCAT(a, b) SVMP_DETAIL_CONCAT_INNER(a, b)

#define SVMP_REGISTER_EQUATION(TYPE_STRING, FACTORY_FN)                                            \
  namespace {                                                                                       \
  const bool SVMP_DETAIL_CONCAT(svmp_equation_registered_, __COUNTER__) = []() {                    \
    ::svmp::Physics::EquationModuleRegistry::instance().registerFactory((TYPE_STRING), (FACTORY_FN)); \
    return true;                                                                                   \
  }();                                                                                              \
  } // namespace

} // namespace svmp::Physics
