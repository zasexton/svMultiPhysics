#pragma once

#include "FE/LevelSet/LevelSetRestart.h"

namespace svmp::Physics::formulations::level_set {

// Compatibility aliases; remove when Physics callers include FE/LevelSet directly.
using LevelSetFieldRestartRecord = FE::level_set::LevelSetFieldRestartRecord;
using LevelSetGeneratedInterfaceRestartRecord =
    FE::level_set::LevelSetGeneratedInterfaceRestartRecord;
using LevelSetRestartSnapshot = FE::level_set::LevelSetRestartSnapshot;

using FE::level_set::captureLevelSetFieldRestartRecord;
using FE::level_set::captureLevelSetGeneratedInterfaceRestartRecord;
using FE::level_set::levelSetGeneratedInterfaceRestartRecordMatches;
using FE::level_set::optionsFromLevelSetGeneratedInterfaceRestartRecord;

} // namespace svmp::Physics::formulations::level_set
