#pragma once

#include "FE/LevelSet/LevelSetReinitialization.h"

namespace svmp::Physics::formulations::level_set {

// Compatibility aliases; remove when Physics callers include FE/LevelSet directly.
using LevelSetReinitializationMethod =
    FE::level_set::LevelSetReinitializationMethod;
using LevelSetReinitializationOptions =
    FE::level_set::LevelSetReinitializationOptions;
using LevelSetSignedDistanceRepairResult =
    FE::level_set::LevelSetSignedDistanceRepairResult;

using FE::level_set::repairLevelSetSignedDistanceByProjection;

} // namespace svmp::Physics::formulations::level_set
