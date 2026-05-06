#include "Coupling/CouplingContext.h"
#include "Coupling/CouplingContract.h"
#include "Coupling/CouplingDefinitionBuilder.h"
#include "Coupling/CouplingDeclaration.h"
#include "Coupling/CouplingDiagnostics.h"
#include "Coupling/CouplingFormBuilder.h"
#include "Coupling/CouplingGeometryRequirements.h"
#include "Coupling/CouplingGraph.h"
#include "Coupling/CouplingRegistry.h"
#include "Coupling/CouplingTemporalRequirements.h"
#include "Coupling/CouplingTypes.h"
#include "Coupling/DefinitionBackedCouplingContract.h"
#include "Coupling/MonolithicCouplingBuilder.h"
#include "Coupling/CouplingPayloadDetangler.h"
#include "Coupling/PartitionedCouplingBuilder.h"
#include "Coupling/PartitionedCouplingPlan.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"
#include "Coupling/SharedRegionRegistry.h"
#include "Coupling/TransferPlan.h"

#include <gtest/gtest.h>

TEST(CouplingHeaders, PublicHeadersCompile)
{
    SUCCEED();
}
