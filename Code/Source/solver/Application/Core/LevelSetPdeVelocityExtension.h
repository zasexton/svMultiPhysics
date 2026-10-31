/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

// The PDE velocity extension is in
// FE/LevelSet/Extension/LevelSetPdeVelocityExtension.h.  These names keep its
// Application callers (the driver, the level-set translator and the
// semi-implicit reference) unchanged.

#include "FE/LevelSet/Extension/LevelSetPdeVelocityExtension.h"

namespace application::core {

using svmp::FE::level_set::PdeVelocityExtensionCache;
using svmp::FE::level_set::PdeVelocityExtensionFactorization;
using svmp::FE::level_set::PdeVelocityExtensionMeshRevisions;
using svmp::FE::level_set::PdeVelocityExtensionOperator;
using svmp::FE::level_set::PdeVelocityExtensionOptions;
using svmp::FE::level_set::PdeVelocityExtensionReport;
using svmp::FE::level_set::communicatorCombinedRevision;
using svmp::FE::level_set::extendVelocityByPde;
using svmp::FE::level_set::pdeVelocityExtensionFactorizationFromToken;
using svmp::FE::level_set::pdeVelocityExtensionFactorizationName;
using svmp::FE::level_set::pdeVelocityExtensionOperatorFromToken;
using svmp::FE::level_set::pdeVelocityExtensionOperatorName;

} // namespace application::core
