#pragma once

#include "FE/Core/Types.h"
#include "FE/Geometry/CutQuadrature.h"
#include "FE/LevelSet/LevelSetCurvatureProjection.h"
#include "FE/Systems/SystemState.h"

#include <array>
#include <optional>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
class IMeshAccess;
} // namespace assembly
namespace systems {
class FESystem;
} // namespace systems
} // namespace FE
} // namespace svmp

namespace application {
namespace core {

/** Map the exact reference sample used for a curvature value to physical space. */
[[nodiscard]] std::optional<std::array<svmp::FE::Real, 3>>
mapLevelSetCurvatureReferenceSampleToPhysical(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell,
    const std::array<svmp::FE::Real, 3>& reference_point);

[[nodiscard]] std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample>
collectLevelSetCurvatureCutVolumeSupplementalSamples(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId field,
    int interface_marker,
    svmp::FE::geometry::CutIntegrationSide side);

} // namespace core
} // namespace application
