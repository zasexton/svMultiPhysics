#pragma once

#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/ResolvedMovingDomainConfiguration.h"
#include "FE/LevelSet/LevelSetOptions.h"
#include "FE/LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h"
#include "FE/LevelSet/LevelSetVolumeConstrainedMinimizer.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace application::core {

// The options of the static-capillary initialization: the FE minimizer's
// options and the limits of the Navier-Stokes pressure certificates it must
// pass (the acceptance checks of the minimizer's problem).
struct StaticCapillaryEquilibriumOptions
    : svmp::FE::level_set::LevelSetVolumeConstrainedMinimizerOptions {
  svmp::FE::Real pressure_representability_max_residual_norm{1.0e-10};
  svmp::FE::Real pressure_representability_max_relative_distance{1.0e-8};
  svmp::FE::Real physical_equilibrium_max_residual_norm{1.0e-10};
  svmp::FE::Real constant_pressure_kkt_max_residual_norm{1.0e-10};
  svmp::FE::Real constant_pressure_kkt_max_relative_distance{1.0e-8};
};

// The FE configuration of a maintained level-set field
// (FE/LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h), resolved from
// the keys of the equation element that the installation resolved, with the
// installation itself and the static-capillary initialization (a
// Navier-Stokes free-surface option).
struct ResolvedLevelSetMaintenanceCompatibilityConfiguration
    : svmp::FE::level_set::LevelSetMaintenanceConfiguration {
  ResolvedLevelSetEquationHandle installation{};
  bool static_capillary_equilibrium_enabled{false};
  StaticCapillaryEquilibriumOptions static_capillary_equilibrium{};
};

using LevelSetMaintenanceConfigurationHandle = std::shared_ptr<
    const ResolvedLevelSetMaintenanceCompatibilityConfiguration>;

// The maintenance of a level-set equation: the equation type and the open
// boundaries come from the legacy input, the keys from the installation
// (required for a level-set equation).
[[nodiscard]] std::optional<LevelSetMaintenanceConfigurationHandle>
resolveLegacyLevelSetMaintenanceConfiguration(
    const LegacyLevelSetMaintenanceInput &input,
    std::span<const ActiveCutVolumeRequest> active_requests,
    ResolvedLevelSetEquationHandle installation);

[[nodiscard]] std::vector<LevelSetMaintenanceConfigurationHandle>
resolveLevelSetMaintenanceConfigurations(
    std::span<const ResolvedLevelSetEquationHandle> equations_by_input_index,
    std::span<const ActiveCutVolumeRequest> active_requests);

} // namespace application::core
