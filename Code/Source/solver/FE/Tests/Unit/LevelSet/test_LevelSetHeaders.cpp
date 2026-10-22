#include "LevelSet/LevelSet.h"
#include "LevelSet/LevelSetConservativePhaseOperator.h"
#include "LevelSet/LevelSetConservativePhaseTransport.h"
#include "LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "LevelSet/LevelSetCurvatureProjection.h"
#include "LevelSet/LevelSetCurvatureSamples.h"
#include "LevelSet/LevelSetCutVolumeMeasures.h"
#include "LevelSet/LevelSetDiagnostics.h"
#include "LevelSet/LevelSetGeometry.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetMaintenanceSchedule.h"
#include "LevelSet/LevelSetMaintenanceSubstage.h"
#include "LevelSet/LevelSetOptions.h"
#include "LevelSet/LevelSetPhaseGeometryReconciliation.h"
#include "LevelSet/LevelSetReinitialization.h"
#include "LevelSet/LevelSetRestart.h"
#include "LevelSet/LevelSetTransport.h"
#include "LevelSet/LevelSetVolume.h"

#include <gtest/gtest.h>

TEST(LevelSetHeaders, PublicHeadersCompile)
{
    SUCCEED();
}
