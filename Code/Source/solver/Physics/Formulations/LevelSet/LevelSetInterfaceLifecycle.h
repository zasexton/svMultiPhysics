#pragma once

#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"

namespace svmp::Physics::formulations::level_set {

// Compatibility aliases; remove when Physics callers include FE/LevelSet directly.
using LevelSetGeneratedInterfaceOptions =
    FE::level_set::LevelSetGeneratedInterfaceOptions;
using LevelSetGeneratedInterfaceResult =
    FE::level_set::LevelSetGeneratedInterfaceResult;
using LevelSetGeneratedInterfaceLifecycle =
    FE::level_set::LevelSetGeneratedInterfaceLifecycle;

} // namespace svmp::Physics::formulations::level_set
