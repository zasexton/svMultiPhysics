#pragma once

/**
 * @file
 * @defgroup fe_level_set FE Level-Set Services
 * @brief Physics-neutral level-set transport, interface, volume, diagnostic,
 * and restart services.
 *
 * Public APIs live in `svmp::FE::level_set`. Include this aggregate header for
 * the full service set, or include a narrower `LevelSet/...` header for a
 * specific service.
 */

#include "LevelSet/LevelSetDiagnostics.h"
#include "LevelSet/LevelSetCurvatureProjection.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetOptions.h"
#include "LevelSet/LevelSetReinitialization.h"
#include "LevelSet/LevelSetRestart.h"
#include "LevelSet/LevelSetTransport.h"
#include "LevelSet/LevelSetVolume.h"
