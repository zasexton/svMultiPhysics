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
#include "LevelSet/LevelSetConservativePhaseOperator.h"
#include "LevelSet/LevelSetConservativePhaseArtifact.h"
#include "LevelSet/LevelSetConservativePhaseRegions.h"
#include "LevelSet/LevelSetConservativePhaseState.h"
#include "LevelSet/LevelSetConservativePhaseTransport.h"
#include "LevelSet/LevelSetCurvatureProjection.h"
#include "LevelSet/LevelSetCurvatureSamples.h"
#include "LevelSet/LevelSetCutTopology.h"
#include "LevelSet/LevelSetCutVolumeMeasures.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetMaintenanceSubstage.h"
#include "LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h"
#include "LevelSet/Maintenance/LevelSetMaintenancePipeline.h"
#include "LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h"
#include "LevelSet/LevelSetKinematicReconciliation.h"
#include "LevelSet/LevelSetOptions.h"
#include "LevelSet/LevelSetRestart.h"
#include "LevelSet/LevelSetSignDefinitePatchBounds.h"
#include "LevelSet/LevelSetTransport.h"
#include "LevelSet/LevelSetVolume.h"
#include "LevelSet/LevelSetWallContactConstraint.h"
