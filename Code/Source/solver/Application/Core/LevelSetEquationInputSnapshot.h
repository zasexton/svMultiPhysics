#pragma once

#include "Physics/Core/EquationModuleInput.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace application::core {

struct LegacyLevelSetBoundaryInput {
  bool type_defined{false};
  bool name_defined{false};
  std::string type{};
  std::string name{};
  svmp::Physics::ParameterMap parameters{};
};

// The equation type and the boundaries of a level-set equation as the
// maintenance reads them; its keys come from the resolved installation.
struct LegacyLevelSetMaintenanceInput {
  bool equation_type_defined{false};
  std::string equation_type{};
  std::vector<LegacyLevelSetBoundaryInput> boundaries{};
};

struct LevelSetEquationInputSnapshot {
  svmp::Physics::EquationModuleInput installation_input{};
  std::optional<LegacyLevelSetMaintenanceInput> legacy_maintenance_input{};
};

using LevelSetEquationInputHandle =
    std::shared_ptr<const LevelSetEquationInputSnapshot>;

} // namespace application::core
