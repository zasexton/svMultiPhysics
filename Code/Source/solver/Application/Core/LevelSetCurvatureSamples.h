#pragma once

#include "FE/Core/Types.h"
#include "FE/Geometry/CutQuadrature.h"
#include "FE/LevelSet/LevelSetCurvatureProjection.h"
#include "FE/Systems/SystemState.h"

#include <vector>

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
} // namespace systems
} // namespace FE
} // namespace svmp

namespace application {
namespace core {

[[nodiscard]] std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample>
collectLevelSetCurvatureCutVolumeSupplementalSamples(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId field,
    int interface_marker,
    svmp::FE::geometry::CutIntegrationSide side);

} // namespace core
} // namespace application

