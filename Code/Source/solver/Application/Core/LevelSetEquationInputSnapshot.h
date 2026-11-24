#pragma once

#include "FE/LevelSet/LevelSetOptions.h"
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
  // The deck has no other equation: the transport is the monolithic
  // operator whether or not it is declared Coupled.
  bool sole_equation{false};
  // A Coupled equation with a positive Tolerance declares the convergence of
  // its field: that relative tolerance, and the absolute tolerance of its
  // linear solver (1e-10 when unstated, 0 when not positive and finite).
  svmp::FE::level_set::LevelSetResidualCriterionOptions residual_criterion{};
};

using LevelSetEquationInputHandle =
    std::shared_ptr<const LevelSetEquationInputSnapshot>;

} // namespace application::core
