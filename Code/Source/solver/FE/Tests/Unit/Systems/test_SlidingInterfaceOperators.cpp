#include "Systems/InterfaceOperators.h"

#include <gtest/gtest.h>

#include <memory>
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

#endif
