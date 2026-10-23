/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#ifndef SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_RULES_H
#define SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_RULES_H

/**
 * @file NavierStokesOptionRules.h
 * @brief Validation rules of the Navier-Stokes options
 *
 * Checks of option values against each other and against the installation
 * (decisions D47 and D52).  Moved unchanged from the free-surface support
 * files of Physics/Formulations/NavierStokes.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include <string_view>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

void validateWallNormalDimension(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim,
    std::string_view context);

void validateFiniteConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

void validateNonnegativeConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

void validatePositiveConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_RULES_H
