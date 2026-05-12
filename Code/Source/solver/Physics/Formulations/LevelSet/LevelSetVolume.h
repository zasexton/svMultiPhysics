#pragma once

#include "FE/LevelSet/LevelSetVolume.h"

namespace svmp::Physics::formulations::level_set {

// Compatibility aliases; remove when Physics callers include FE/LevelSet directly.
using LevelSetVolumeOptions = FE::level_set::LevelSetVolumeOptions;
using LevelSetVolumeResult = FE::level_set::LevelSetVolumeResult;
using LevelSetGlobalShiftCorrectionOptions =
    FE::level_set::LevelSetGlobalShiftCorrectionOptions;
using LevelSetGlobalShiftCorrectionResult =
    FE::level_set::LevelSetGlobalShiftCorrectionResult;

using FE::level_set::applyGlobalLevelSetShiftCorrection;
using FE::level_set::computeLevelSetCutCellVolume;

} // namespace svmp::Physics::formulations::level_set
