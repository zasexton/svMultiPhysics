#include "Systems/InterfaceOperators.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

using namespace svmp::FE::systems;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

namespace {

svmp::search::InterfaceMap make_single_pair_map()
{
    svmp::search::InterfaceMap map;
    map.name = "sliding";
    map.state = svmp::search::InterfaceMapState::Trial;

    svmp::search::InterfacePair pair;
    pair.source_face = 0;
    pair.target_face = 0;
    pair.source_measure = 2.0;
    pair.target_measure = 2.0;
    map.pairs.push_back(pair);
    return map;
}

} // namespace

TEST(SlidingInterfaceOperators, AppliesVectorFrameTransformBeforeProjection)
{
    const auto map = make_single_pair_map();
    const auto op = makeInterfaceTransferOperator(InterfaceOperatorKind::PointwiseInterpolation);

    InterfaceTransferOptions options;
    options.field_kind = InterfaceFieldKind::Vector;
    options.frame_policy = InterfaceFrameTransformPolicy::SourceToTargetVector;
    options.component_count = 3;
    options.source_to_target_rotation = {{
        {{0.0, -1.0, 0.0}},
        {{1.0,  0.0, 0.0}},
        {{0.0,  0.0, 1.0}}}};

    const auto result = applyInterfaceTransfer(*op, map, std::vector<svmp::FE::Real>{1.0, 0.0, 2.0}, options);
    ASSERT_EQ(result.target_values.size(), 3u);
    EXPECT_DOUBLE_EQ(result.target_values[0], 0.0);
    EXPECT_DOUBLE_EQ(result.target_values[1], 1.0);
    EXPECT_DOUBLE_EQ(result.target_values[2], 2.0);
}

TEST(SlidingInterfaceOperators, ConservativeDiagnosticsTrackBalance)
{
    const auto map = make_single_pair_map();
    const auto op = makeInterfaceTransferOperator(InterfaceOperatorKind::ConservativeProjection);
    const auto result = applyInterfaceTransfer(*op, map, std::vector<svmp::FE::Real>{3.0});
    const auto diagnostic = diagnoseInterfaceTransfer(map, result, 1e-12);

    EXPECT_TRUE(diagnostic.ok);
    EXPECT_DOUBLE_EQ(diagnostic.source_integral, 6.0);
    EXPECT_DOUBLE_EQ(diagnostic.target_integral, 6.0);
}

TEST(SlidingInterfaceOperators, SlidingTransferCarriesMapMetadata)
{
    SlidingInterfaceMap sliding;
    auto map = make_single_pair_map();
    map.name = "sliding_meta";
    sliding.map_kind = SlidingInterfaceMapKind::RotatingSliding;
    sliding.set_trial_map(std::move(map), 2.5, 8);
    sliding.accept_trial(InterfaceOperatorState::AcceptedTimeStep);

    const auto op = makeInterfaceTransferOperator(InterfaceOperatorKind::ConservativeProjection);
    const auto result = applySlidingInterfaceTransfer(
        *op, sliding, std::vector<svmp::FE::Real>{3.0});

    EXPECT_EQ(result.interface_name, "sliding_meta");
    EXPECT_EQ(result.sliding_map_kind, SlidingInterfaceMapKind::RotatingSliding);
    EXPECT_EQ(result.interface_state, InterfaceOperatorState::AcceptedTimeStep);
    EXPECT_DOUBLE_EQ(result.interface_time, 2.5);
    EXPECT_EQ(result.interface_time_level_epoch, 8u);
    EXPECT_EQ(result.interface_revision_key, sliding.accepted_revision_key);
    EXPECT_DOUBLE_EQ(result.source_integral, 6.0);
    EXPECT_DOUBLE_EQ(result.target_integral, 6.0);
}

TEST(SlidingInterfaceOperators, InvalidationIncludesMeshAndDofLayoutRebuilds)
{
    SlidingInterfaceMap sliding;
    sliding.set_trial_map(make_single_pair_map(), 0.0, 1);
    sliding.accept_trial(InterfaceOperatorState::AcceptedTimeStep);

    const auto policy = interfaceOperatorInvalidation(sliding, 3, 2);
    EXPECT_TRUE(policy.rebuild_interface_operator);
    EXPECT_TRUE(policy.rebuild_matrix);
    EXPECT_TRUE(policy.refresh_preconditioner);
    EXPECT_TRUE(policy.refresh_restart_metadata);
}

TEST(SlidingInterfaceOperators, TimeLevelInvalidationRequestsMapRefresh)
{
    SlidingInterfaceMap sliding;
    sliding.set_trial_map(make_single_pair_map(), 1.0, 4);
    sliding.accept_trial(InterfaceOperatorState::AcceptedTimeStep);

    EXPECT_TRUE(sliding.valid_for_time_level(1.0, 4));
    EXPECT_FALSE(sliding.valid_for_time_level(1.1, 4));
    EXPECT_FALSE(sliding.valid_for_time_level(1.0, 5));

    const auto policy = interfaceOperatorInvalidationForTime(
        sliding,
        /*current_time=*/1.1,
        /*current_time_level_epoch=*/5,
        /*fe_dof_layout_revision=*/3,
        /*cached_fe_dof_layout_revision=*/3);

    EXPECT_TRUE(policy.rebuild_search);
    EXPECT_TRUE(policy.rebuild_interface_operator);
    EXPECT_TRUE(policy.refresh_matrix_free_geometry);
    EXPECT_TRUE(policy.rebuild_matrix);
    EXPECT_TRUE(policy.refresh_preconditioner);
    EXPECT_TRUE(policy.refresh_restart_metadata);
}

#endif
