#include "Assembly/CutIntegrationContext.h"
#include "Geometry/CutQuadrature.h"
#include "Systems/CutIntegrationInvalidation.h"

#include <gtest/gtest.h>

using namespace svmp::FE;
using namespace svmp::FE::assembly;
using namespace svmp::FE::geometry;
using namespace svmp::FE::systems;

TEST(CutIntegrationInfrastructure, AssemblyContextCarriesQuadratureMetadataAndHooks)
{
    CutIntegrationContext context;
    CutCellAssemblyMetadata metadata;
    metadata.cell = 4;
    metadata.volume_fraction = 0.5;
    metadata.embedded_normal = {{1.0, 0.0, 0.0}};
    metadata.provenance_id = "embedded-plane";
    metadata.revision_key = 99;

    auto rule = makeAxisAlignedBoxCutVolumeQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        0,
        0.5,
        CutIntegrationSide::Negative,
        "embedded-plane");
    context.addVolumeRule(metadata, rule);
    context.addInterfaceRule(makeAxisAlignedBoxCutInterfaceQuadrature(
        {{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}, 0, 0.5, "embedded-plane"));

    EmbeddedBoundaryKinematicData kinematics;
    kinematics.constraint_id = "moving-plane";
    kinematics.relation_map_id = "relation";
    kinematics.source_geometry_id = "plane";
    kinematics.provenance_id = "embedded-plane";
    context.addKinematicData(kinematics);

    CutStabilizationHook hook;
    hook.name = "aggregation-candidate";
    hook.geometry_scale = 0.5;
    hook.conditioning_indicator = 1.0e-9;
    hook.enabled = true;
    context.addStabilizationHook(hook);

    ASSERT_EQ(context.metadata().size(), 1u);
    ASSERT_EQ(context.volumeRules().size(), 1u);
    ASSERT_EQ(context.interfaceRules().size(), 1u);
    ASSERT_EQ(context.kinematicData().size(), 1u);
    ASSERT_EQ(context.stabilizationHooks().size(), 1u);
    EXPECT_DOUBLE_EQ(context.volumeRules()[0].measure, 0.5);
    EXPECT_EQ(context.metadata()[0].provenance_id, "embedded-plane");
}

TEST(CutIntegrationInfrastructure, InvalidationSeparatesGeometryAndFELayoutChanges)
{
    CutIntegrationRevisionSnapshot cached;
    cached.valid = true;
    cached.cut_revision_key = 10;
    cached.geometry_revision = 1;
    cached.fe_dof_layout_revision = 2;
    cached.cut_cell_count = 1;

    auto current = cached;
    EXPECT_FALSE(classifyCutIntegrationRefresh(cached, current).any());

    current.geometry_revision = 2;
    auto geometry_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_TRUE(geometry_decision.rebuild_cut_classification);
    EXPECT_TRUE(geometry_decision.rebuild_quadrature);
    EXPECT_TRUE(geometry_decision.rebuild_matrix_free_data);

    current = cached;
    current.fe_dof_layout_revision = 3;
    auto layout_decision = classifyCutIntegrationRefresh(cached, current);
    EXPECT_FALSE(layout_decision.rebuild_cut_classification);
    EXPECT_FALSE(layout_decision.rebuild_quadrature);
    EXPECT_TRUE(layout_decision.rebuild_matrix);
    EXPECT_TRUE(layout_decision.refresh_preconditioner);
}

TEST(CutIntegrationInfrastructure, ConditioningDiagnosticsIdentifySmallAndDegenerateCuts)
{
    const auto diagnostic = diagnoseCutConditioning({0.5, 1.0e-9, 0.0}, 1.0e-6, 1.0e-12);
    EXPECT_FALSE(diagnostic.ok);
    EXPECT_EQ(diagnostic.small_cut_cell_count, 1u);
    EXPECT_EQ(diagnostic.degenerate_cut_count, 1u);
}
