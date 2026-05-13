#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "Physics/Core/PhysicsModule.h"

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
} // namespace systems
} // namespace FE

namespace Physics {
struct EquationModuleInput;
} // namespace Physics
} // namespace svmp

namespace application {
namespace translators {
namespace level_set {

[[nodiscard]] bool isEquationType(std::string_view type);

[[nodiscard]] std::vector<std::string> equationTypes();

[[nodiscard]] std::unique_ptr<svmp::Physics::PhysicsModule>
createModule(const svmp::Physics::EquationModuleInput& input,
             svmp::FE::systems::FESystem& system);

} // namespace level_set
} // namespace translators
} // namespace application
