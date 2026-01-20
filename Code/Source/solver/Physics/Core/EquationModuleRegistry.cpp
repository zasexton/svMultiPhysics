#include "Physics/Core/EquationModuleRegistry.h"

#include "Physics/Core/EquationModuleInput.h"

#include <stdexcept>

namespace svmp::Physics {

EquationModuleRegistry& EquationModuleRegistry::instance()
{
  static EquationModuleRegistry registry;
  return registry;
}

void EquationModuleRegistry::registerFactory(std::string type, FactoryFn factory)
{
  if (type.empty()) {
    throw std::runtime_error("[svMultiPhysics::Physics] Cannot register equation module with empty type.");
  }
  if (!factory) {
    throw std::runtime_error("[svMultiPhysics::Physics] Cannot register equation module '" + type +
                             "' with null factory.");
  }

  const auto [it, inserted] = factories_.emplace(std::move(type), factory);
  if (!inserted) {
    throw std::runtime_error("[svMultiPhysics::Physics] Duplicate equation module registration for type '" +
                             it->first + "'.");
  }
}

std::unique_ptr<PhysicsModule> EquationModuleRegistry::create(std::string_view type,
                                                              const EquationModuleInput& input,
                                                              svmp::FE::systems::FESystem& system) const
{
  const auto it = factories_.find(std::string(type));
  if (it == factories_.end() || !it->second) {
    throw std::runtime_error("[svMultiPhysics::Physics] No equation module registered for type '" +
                             std::string(type) + "'.");
  }
  return (it->second)(input, system);
}

std::vector<std::string> EquationModuleRegistry::registeredTypes() const
{
  std::vector<std::string> types;
  types.reserve(factories_.size());
  for (const auto& [k, _] : factories_) {
    types.push_back(k);
  }
  return types;
}

} // namespace svmp::Physics

