#include "LevelSet/LevelSet.h"
#include "LevelSet/LevelSetCurvatureProjection.h"
#include "LevelSet/LevelSetDiagnostics.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetOptions.h"
#include "LevelSet/LevelSetReinitialization.h"
#include "LevelSet/LevelSetRestart.h"
#include "LevelSet/LevelSetTransport.h"
#include "LevelSet/LevelSetVolume.h"

#include <gtest/gtest.h>

TEST(LevelSetHeaders, PublicHeadersCompile)
{
    SUCCEED();
}
