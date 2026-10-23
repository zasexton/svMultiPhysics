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

#include <string_view>

namespace application::core {

// The settings of the algebraic wall-compatible wet extension fail with this
// message, at translation and when the extension requests are read.
inline constexpr std::string_view kRetiredWetExtensionMessage =
    "[svMultiPhysics::Application] The algebraic wall-compatible wet "
    "extension (Use_wet_extension_advection_velocity=true, "
    "Advection_velocity_extension_method=wall_compatible_normal or a "
    "nearest-* method, a source velocity field without an extension method, "
    "and Write_velocity_extension_maps=true) was retired (decisions D47 and "
    "D55).  Set Advection_velocity_extension_method=pde_harmonic or "
    "pde_normal with Advection_velocity_extension_coupling=monolithic or "
    "prescribed, and remove the retired settings or set them to false.";

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
