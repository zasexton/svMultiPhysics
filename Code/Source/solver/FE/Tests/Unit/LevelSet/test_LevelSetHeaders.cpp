#include "LevelSet/LevelSet.h"
#include "LevelSet/LevelSetConservativePhaseOperator.h"
#include "LevelSet/LevelSetConservativePhaseTransport.h"
#include "LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "LevelSet/LevelSetCurvatureProjection.h"
#include "LevelSet/LevelSetCurvatureSamples.h"
#include "LevelSet/LevelSetCutTopology.h"
#include "LevelSet/LevelSetCutVolumeMeasures.h"
#include "LevelSet/LevelSetDiagnostics.h"
#include "LevelSet/LevelSetGeometry.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetMaintenanceSchedule.h"
#include "LevelSet/LevelSetMaintenanceSubstage.h"
#include "LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h"
#include "LevelSet/Maintenance/LevelSetMaintenancePipeline.h"
#include "LevelSet/Maintenance/LevelSetMaintenanceRequestSchedule.h"
#include "LevelSet/Maintenance/LevelSetWallAwareMaintenance.h"
#include "LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h"
#include "LevelSet/LevelSetOptions.h"
#include "LevelSet/LevelSetPhaseGeometryReconciliation.h"
#include "LevelSet/LevelSetWallContactConstraint.h"
#include "LevelSet/LevelSetRestart.h"
#include "LevelSet/LevelSetTransport.h"
#include "LevelSet/LevelSetVolume.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "LevelSet/Extension/LevelSetPdeVelocityExtension.h"
#include "LevelSet/LevelSetContactStageParticipant.h"
#include "LevelSet/LevelSetEnergyParticipant.h"
#include "LevelSet/Extension/LevelSetVelocityExtensionWall.h"
#endif

#include <gtest/gtest.h>

TEST(LevelSetHeaders, PublicHeadersCompile)
{
    SUCCEED();
}
