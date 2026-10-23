/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

/**
 * @file
 * @brief The step participant of the accepted contact stages is an FE
 *        service (FE/LevelSet/LevelSetContactStageParticipant.h).  This
 *        forward keeps the callers until they name it.
 */

#include "FE/LevelSet/LevelSetContactStageParticipant.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactStages.h"

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using svmp::FE::level_set::ContactStageParticipant;

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
