#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "FE/Core/Types.h"
#include "Physics/Core/PhysicsModule.h"

namespace svmp {
class MeshBase;

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

/** Pure cross-equation dependency advertised by material-interface transport. */
struct MaterialInterfaceTransportDependency {
  const svmp::MeshBase* mesh{nullptr};
  std::string mesh_name{};
  int dimension{0};
  int interface_marker{-1};
  std::string level_set_field_name{};
  std::string conservative_phase_field_name{};
  std::string operator_tag{};
};

[[nodiscard]] bool isEquationType(std::string_view type);

[[nodiscard]] std::vector<std::string> equationTypes();

/**
 * Translate and validate the material-interface dependency without mutating
 * an FE system. Non-material-interface level-set inputs return nullopt.
 */
[[nodiscard]] std::optional<MaterialInterfaceTransportDependency>
materialInterfaceTransportDependency(
    const svmp::Physics::EquationModuleInput& input);

/**
 * Predeclare the paired scalar unknowns in canonical level-set/phase order.
 * All requested fields are validated before either field is added.
 */
void preRegisterMaterialInterfaceTransportFields(
    const svmp::Physics::EquationModuleInput& input,
    svmp::FE::systems::FESystem& system);

[[nodiscard]] std::unique_ptr<svmp::Physics::PhysicsModule>
createModule(const svmp::Physics::EquationModuleInput& input,
             svmp::FE::systems::FESystem& system);

} // namespace level_set
} // namespace translators
} // namespace application
