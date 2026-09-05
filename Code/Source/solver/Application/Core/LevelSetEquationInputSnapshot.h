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

struct LegacyLevelSetMaintenanceInput {
  bool equation_type_defined{false};
  std::string equation_type{};
  svmp::Physics::ParameterMap equation_parameters{};
  std::vector<LegacyLevelSetBoundaryInput> boundaries{};
};

struct LevelSetEquationInputSnapshot {
  svmp::Physics::EquationModuleInput installation_input{};
  std::optional<LegacyLevelSetMaintenanceInput> legacy_maintenance_input{};
};

using LevelSetEquationInputHandle =
    std::shared_ptr<const LevelSetEquationInputSnapshot>;

} // namespace application::core
