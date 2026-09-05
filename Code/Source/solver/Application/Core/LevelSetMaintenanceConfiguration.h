#pragma once

#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/ResolvedMovingDomainConfiguration.h"
#include "FE/LevelSet/LevelSetCurvatureProjection.h"
#include "FE/LevelSet/LevelSetOptions.h"
#include "FE/LevelSet/LevelSetStaticCapillaryEquilibrium.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace application::core {

struct ResolvedLevelSetMaintenanceCompatibilityConfiguration {
  struct OpenBoundary {
    std::string face_name{};
    bool inflow{false};
    std::optional<svmp::FE::Real> literal_inflow_value{};
  };

  ResolvedLevelSetEquationHandle installation{};
  svmp::FE::level_set::LevelSetTransportOptions transport{};
  double isovalue{0.0};
  std::vector<OpenBoundary> open_boundaries{};
  std::optional<ActiveCutVolumeRequest> volume_cut_request{};
  bool static_capillary_equilibrium_enabled{false};
  svmp::FE::level_set::LevelSetStaticCapillaryEquilibriumOptions
      static_capillary_equilibrium{};
  bool curvature_projection_enabled{false};
  std::string curvature_field_name{};
  int curvature_projection_cadence_steps{1};
  svmp::FE::level_set::LevelSetCurvatureProjectionOptions
      curvature_projection{};
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

[[nodiscard]] std::string serializeLevelSetMaintenanceCompatibility(
    const ResolvedLevelSetMaintenanceCompatibilityConfiguration &configuration);

} // namespace application::core
