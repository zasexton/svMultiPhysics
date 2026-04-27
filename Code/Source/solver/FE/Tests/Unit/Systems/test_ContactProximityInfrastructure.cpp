#include "Assembly/ContactPairContext.h"
#include "Systems/ContactPenaltyKernel.h"
#include "Systems/ContactOperatorInvalidation.h"
#include "Systems/SurfaceContactKernel.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include <gtest/gtest.h>

#include <iterator>

namespace {

svmp::search::ContactPair make_pair(
    svmp::gid_t source_gid,
    svmp::gid_t target_gid,
    svmp::search::ContactPairState state,
    svmp::FE::Real gap) {
    svmp::search::ContactPair pair;
    pair.state = state;
    pair.provenance.pair_id = static_cast<std::uint64_t>(source_gid * 1000 + target_gid);
    pair.provenance.source_surface_name = "slave-surface";
    pair.provenance.target_surface_name = "master-surface";
    pair.provenance.source_entity = static_cast<svmp::index_t>(source_gid);
    pair.provenance.target_entity = static_cast<svmp::index_t>(target_gid);
    pair.provenance.source_gid = source_gid;
    pair.provenance.target_gid = target_gid;
    pair.provenance.source_label = 11;
    pair.provenance.target_label = 12;
    pair.provenance.source_kind = svmp::search::ContactEntityKind::Surface;
    pair.provenance.target_kind = svmp::search::ContactEntityKind::Surface;
    pair.provenance.canonical_owner_rank = 0;
    pair.provenance.source_configuration = svmp::Configuration::Current;
    pair.provenance.target_configuration = svmp::Configuration::Current;
    pair.provenance.time_level = 0.25;
    pair.provenance.generation_policy = "unit-test-closest-point";
    pair.lifecycle_stage = svmp::search::ContactLifecycleStage::Classified;
    pair.projection.valid = true;
    pair.projection.source_point = {{0.0, 0.5, 0.5}};
    pair.projection.target_point = {{gap, 0.5, 0.5}};
    pair.projection.source_normal = {{1.0, 0.0, 0.0}};
    pair.projection.target_normal = {{-1.0, 0.0, 0.0}};
    pair.projection.tangent0 = {{0.0, 1.0, 0.0}};
    pair.projection.tangent1 = {{0.0, 0.0, 1.0}};
    pair.projection.tangential_reference0 = pair.projection.tangent0;
    pair.projection.tangential_reference1 = pair.projection.tangent1;
    pair.projection.unsigned_gap = gap;
    pair.projection.signed_gap = gap;
    pair.projection.tangential_frame_valid = true;
    pair.projection.tangential_slip_magnitude = 0.0;
    pair.projection.side = "positive";
    return pair;
}

svmp::search::ContactProximityMap make_contact_map() {
    svmp::search::ContactProximityMap map;
    map.name = "contact-map";
    map.state = svmp::search::ContactTransactionState::AcceptedTimeStep;
    map.source.configuration = svmp::Configuration::Current;
    map.target.configuration = svmp::Configuration::Current;
    map.candidate_generation_epoch = 1;
    map.active_set_epoch = 2;
    map.pairs.push_back(make_pair(1, 2, svmp::search::ContactPairState::Active, 0.05));
    map.pairs.push_back(make_pair(3, 4, svmp::search::ContactPairState::Inactive, 0.50));
    return map;
}

} // namespace

TEST(ContactProximityInfrastructureTest, AssemblyContextExposesPhysicsNeutralContactGeometry) {
    const auto map = make_contact_map();
    svmp::FE::assembly::ContactPairContext all_pairs(map);
    svmp::FE::assembly::ContactPairContext active_pairs(
        map, svmp::FE::assembly::ContactPairSelection::ActivePairsOnly);

    ASSERT_EQ(all_pairs.quadraturePairs().size(), 2u);
    ASSERT_EQ(active_pairs.quadraturePairs().size(), 1u);
    const auto& qp = active_pairs.quadraturePairs().front();
    EXPECT_EQ(qp.source_gid, 1);
    EXPECT_EQ(qp.target_gid, 2);
    EXPECT_EQ(qp.source_surface_name, "slave-surface");
    EXPECT_EQ(qp.target_surface_name, "master-surface");
    EXPECT_EQ(qp.state, svmp::search::ContactPairState::Active);
    EXPECT_DOUBLE_EQ(qp.unsigned_gap, 0.05);
    EXPECT_DOUBLE_EQ(qp.normal[0], 1.0);
    EXPECT_DOUBLE_EQ(qp.tangent0[1], 1.0);
    EXPECT_TRUE(qp.projection_valid);
    EXPECT_TRUE(qp.tangential_frame_valid);
    EXPECT_EQ(qp.side, "positive");
    EXPECT_DOUBLE_EQ(qp.time_level, 0.25);
    EXPECT_EQ(qp.generation_policy, "unit-test-closest-point");
    EXPECT_TRUE(qp.projection_operator.available);
    EXPECT_EQ(qp.projection_operator.operator_family, "closest-point");
    EXPECT_FALSE(qp.projection_operator.conservative);
    EXPECT_EQ(active_pairs.sourceConfiguration(), svmp::Configuration::Current);
    EXPECT_NE(active_pairs.contactRevisionKey(), 0u);

    ASSERT_EQ(all_pairs.contactPatches().size(), 1u);
    const auto& patch = all_pairs.contactPatches().front();
    EXPECT_EQ(patch.source_label, 11);
    EXPECT_EQ(patch.target_label, 12);
    ASSERT_EQ(patch.pair_indices.size(), 2u);
    EXPECT_EQ(all_pairs.pair(patch.pair_indices.front()).source_gid, 1);
    EXPECT_EQ(std::distance(all_pairs.begin(), all_pairs.end()), 2);
    EXPECT_EQ(std::distance(all_pairs.patchesBegin(), all_pairs.patchesEnd()), 1);
}

TEST(ContactProximityInfrastructureTest, OperatorInvalidationTracksPairsActiveSetAndFELayouts) {
    auto map = make_contact_map();
    svmp::search::ContactExternalRevisions external;
    external.fe_space_revision = 10;
    external.fe_dof_layout_revision = 20;

    const auto cached =
        svmp::FE::systems::ContactOperatorRevisionSnapshot::capture(map, external);
    auto same = svmp::FE::systems::ContactOperatorRevisionSnapshot::capture(map, external);
    auto decision = svmp::FE::systems::classifyContactOperatorRefresh(cached, same);
    EXPECT_FALSE(decision.structural_rebuild);
    EXPECT_FALSE(decision.value_update);
    EXPECT_FALSE(decision.matrix_rebuild);

    map.pairs.push_back(make_pair(5, 6, svmp::search::ContactPairState::Active, 0.02));
    auto pair_changed = svmp::FE::systems::ContactOperatorRevisionSnapshot::capture(map, external);
    decision = svmp::FE::systems::classifyContactOperatorRefresh(cached, pair_changed);
    EXPECT_TRUE(decision.structural_rebuild);
    EXPECT_TRUE(decision.matrix_rebuild);
    EXPECT_TRUE(decision.matrix_free_rebuild);
    EXPECT_TRUE(decision.preconditioner_refresh);

    map = make_contact_map();
    map.active_set_epoch += 1;
    map.pairs.front().state = svmp::search::ContactPairState::Inactive;
    auto active_changed = svmp::FE::systems::ContactOperatorRevisionSnapshot::capture(map, external);
    decision = svmp::FE::systems::classifyContactOperatorRefresh(cached, active_changed);
    EXPECT_FALSE(decision.structural_rebuild);
    EXPECT_TRUE(decision.value_update);
    EXPECT_TRUE(decision.matrix_rebuild);
    EXPECT_TRUE(decision.preconditioner_refresh);

    map = make_contact_map();
    map.source_revision.current_geometry_revision += 1;
    map.source_revision.geometry_revision += 1;
    auto geometry_changed =
        svmp::FE::systems::ContactOperatorRevisionSnapshot::capture(map, external);
    decision = svmp::FE::systems::classifyContactOperatorRefresh(cached, geometry_changed);
    EXPECT_FALSE(decision.structural_rebuild);
    EXPECT_TRUE(decision.value_update);
    EXPECT_TRUE(decision.matrix_rebuild);
    EXPECT_TRUE(decision.matrix_free_rebuild);
    EXPECT_TRUE(decision.preconditioner_refresh);

    auto restart_external = external;
    restart_external.restart_layout_revision += 1;
    auto restart_changed =
        svmp::FE::systems::ContactOperatorRevisionSnapshot::capture(make_contact_map(), restart_external);
    decision = svmp::FE::systems::classifyContactOperatorRefresh(cached, restart_changed);
    EXPECT_FALSE(decision.structural_rebuild);
    EXPECT_FALSE(decision.matrix_rebuild);
    EXPECT_TRUE(decision.restart_metadata_update);

    auto assert_layout_rebuild = [&](svmp::search::ContactExternalRevisions changed) {
        auto changed_snapshot =
            svmp::FE::systems::ContactOperatorRevisionSnapshot::capture(make_contact_map(), changed);
        auto changed_decision =
            svmp::FE::systems::classifyContactOperatorRefresh(cached, changed_snapshot);
        EXPECT_TRUE(changed_decision.structural_rebuild);
        EXPECT_TRUE(changed_decision.matrix_rebuild);
        EXPECT_TRUE(changed_decision.matrix_free_rebuild);
        EXPECT_TRUE(changed_decision.preconditioner_refresh);
        EXPECT_TRUE(changed_decision.restart_metadata_update);
    };

    auto space_changed = external;
    space_changed.fe_space_revision += 1;
    assert_layout_rebuild(space_changed);

    external.fe_dof_layout_revision += 1;
    assert_layout_rebuild(external);

    auto constraint_changed = svmp::search::ContactExternalRevisions{};
    constraint_changed.fe_space_revision = 10;
    constraint_changed.fe_dof_layout_revision = 20;
    constraint_changed.fe_constraint_layout_revision = 1;
    assert_layout_rebuild(constraint_changed);

    auto block_changed = svmp::search::ContactExternalRevisions{};
    block_changed.fe_space_revision = 10;
    block_changed.fe_dof_layout_revision = 20;
    block_changed.fe_block_layout_revision = 1;
    assert_layout_rebuild(block_changed);
}

TEST(ContactProximityInfrastructureTest, SimplePenaltyKernelsDoNotOwnGenericContactState) {
    const auto map = make_contact_map();
    const svmp::FE::assembly::ContactPairContext context(map);

    svmp::FE::systems::PenaltyContactConfig point_cfg;
    point_cfg.field = svmp::FE::FieldId{0};
    point_cfg.slave_marker = 11;
    point_cfg.master_marker = 12;
    point_cfg.search_radius = 1.0;
    point_cfg.activation_distance = 0.1;
    point_cfg.penalty = 10.0;
    const svmp::FE::systems::PenaltyPointContactKernel point_kernel(point_cfg);
    EXPECT_EQ(point_kernel.name(), "PenaltyPointContactKernel");
    const auto point_options = svmp::FE::systems::makeContactCandidateOptions(point_cfg);
    EXPECT_DOUBLE_EQ(point_options.search_radius, point_cfg.search_radius);
    EXPECT_DOUBLE_EQ(point_options.activation_distance, point_cfg.activation_distance);
    EXPECT_TRUE(point_options.only_nearest_per_source);
    EXPECT_TRUE(point_options.remove_duplicate_pairs);
    EXPECT_EQ(point_options.generation_policy, "penalty-point-contact");

    svmp::FE::systems::PenaltySurfaceContactConfig surface_cfg;
    surface_cfg.field = svmp::FE::FieldId{0};
    surface_cfg.slave_marker = 11;
    surface_cfg.master_marker = 12;
    surface_cfg.search_radius = 1.0;
    surface_cfg.activation_distance = 0.1;
    surface_cfg.penalty = 10.0;
    const svmp::FE::systems::PenaltySurfaceContactKernel surface_kernel(surface_cfg);
    EXPECT_EQ(surface_kernel.name(), "PenaltySurfaceContactKernel");
    const auto surface_options = svmp::FE::systems::makeContactCandidateOptions(surface_cfg);
    EXPECT_DOUBLE_EQ(surface_options.search_radius, surface_cfg.search_radius);
    EXPECT_DOUBLE_EQ(surface_options.activation_distance, surface_cfg.activation_distance);
    EXPECT_FALSE(surface_options.only_nearest_per_source);
    EXPECT_TRUE(surface_options.remove_duplicate_pairs);
    EXPECT_EQ(surface_options.generation_policy, "penalty-surface-contact");

    ASSERT_EQ(context.quadraturePairs().size(), map.pairs.size());
    EXPECT_EQ(map.state, svmp::search::ContactTransactionState::AcceptedTimeStep);
    EXPECT_EQ(map.candidate_generation_epoch, 1u);
    EXPECT_EQ(map.active_set_epoch, 2u);
}

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
