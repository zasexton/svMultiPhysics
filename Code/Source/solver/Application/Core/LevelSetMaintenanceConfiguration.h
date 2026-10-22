#pragma once

#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/ResolvedMovingDomainConfiguration.h"
#include "FE/LevelSet/LevelSetCurvatureProjection.h"
#include "FE/LevelSet/LevelSetOptions.h"
#include "FE/LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/StaticCapillaryEquilibrium.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace application::core {

// The FE configuration of a maintained level-set field
// (FE/LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h), resolved from
// the legacy equation input, with the resolved installation it came from,
// the static-capillary initialization (a Navier-Stokes free-surface option)
// and the input provenance.
struct ResolvedLevelSetMaintenanceCompatibilityConfiguration
    : svmp::FE::level_set::LevelSetMaintenanceConfiguration {
  ResolvedLevelSetEquationHandle installation{};
  bool static_capillary_equilibrium_enabled{false};
  svmp::Physics::formulations::navier_stokes::static_capillary::LevelSetStaticCapillaryEquilibriumOptions
      static_capillary_equilibrium{};
  std::vector<LevelSetInputObservation> input_observations{};
};

using LevelSetMaintenanceConfigurationHandle = std::shared_ptr<
    const ResolvedLevelSetMaintenanceCompatibilityConfiguration>;

[[nodiscard]] std::optional<LevelSetMaintenanceConfigurationHandle>
resolveLegacyLevelSetMaintenanceConfiguration(
    const LegacyLevelSetMaintenanceInput &input,
    std::span<const ActiveCutVolumeRequest> active_requests,
    ResolvedLevelSetEquationHandle installation = {});

[[nodiscard]] std::vector<LevelSetMaintenanceConfigurationHandle>
resolveLevelSetMaintenanceConfigurations(
    std::span<const ResolvedLevelSetEquationHandle> equations_by_input_index,
    std::span<const ActiveCutVolumeRequest> active_requests);

} // namespace application::core
