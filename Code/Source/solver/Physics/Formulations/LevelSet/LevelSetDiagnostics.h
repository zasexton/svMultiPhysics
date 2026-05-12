#pragma once

#include "FE/LevelSet/LevelSetDiagnostics.h"

namespace svmp::Physics::formulations::level_set {

// Compatibility aliases; remove when Physics callers include FE/LevelSet directly.
using LevelSetScalarDiagnostic = FE::level_set::LevelSetScalarDiagnostic;
using LevelSetOutputDiagnosticsOptions =
    FE::level_set::LevelSetOutputDiagnosticsOptions;
using LevelSetOutputDiagnostics = FE::level_set::LevelSetOutputDiagnostics;

using FE::level_set::computeLevelSetOutputDiagnostics;

} // namespace svmp::Physics::formulations::level_set
