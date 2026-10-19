#pragma once

#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"
#include "FE/LevelSet/LevelSetVolume.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

class Parameters;
class EquationParameters;

namespace application {
namespace core {

// The cut-volume request and its pure functions live in
// FE/LevelSet/LevelSetGeometry.h; the Application keeps the XML parsing.
using LevelSetActiveSide = svmp::FE::level_set::LevelSetActiveSide;
using ActiveCutVolumeRequestOrigin =
    svmp::FE::level_set::ActiveCutVolumeRequestOrigin;
using ActiveCutVolumeRetention = svmp::FE::level_set::ActiveCutVolumeRetention;
using ActiveCutVolumeRequest = svmp::FE::level_set::ActiveCutVolumeRequest;
using svmp::FE::level_set::generatedInterfaceOptionsForActiveCut;
using svmp::FE::level_set::volumeOptionsForCutMaintenance;
using svmp::FE::level_set::hasHighOrderGeneratedInterfaceGeometry;
using svmp::FE::level_set::activeCutVolumeRequestPolicyKey;
using svmp::FE::level_set::resolvedActiveCutVolumeInterfaceMarker;

[[nodiscard]] svmp::FE::level_set::GeneratedInterfaceGeometryMode
parseGeneratedInterfaceGeometryMode(const std::string& raw);

[[nodiscard]] svmp::FE::level_set::ImplicitCutQuadratureBackend
parseImplicitCutQuadratureBackend(const std::string& raw);

[[nodiscard]] svmp::FE::level_set::ImplicitCutFallbackPolicy
parseImplicitCutFallbackPolicy(const std::string& raw);

[[nodiscard]] svmp::FE::level_set::GeometryTangentPolicy
parseGeometryTangentPolicy(const std::string& raw);

[[nodiscard]] std::vector<ActiveCutVolumeRequest>
activeCutVolumeRequests(const Parameters& params);

[[nodiscard]] std::vector<ActiveCutVolumeRequest>
activeCutVolumeRequests(const EquationParameters& equation);

[[nodiscard]] int requireResolvedActiveCutVolumeInterfaceMarker(
    const svmp::FE::systems::FESystem& system,
    const ActiveCutVolumeRequest& request);

void validateEquationLevelCutVolumeConsumer(
    const svmp::FE::systems::FESystem& system,
    const ActiveCutVolumeRequest& request,
    int resolved_marker);

} // namespace core
} // namespace application
