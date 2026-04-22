/**
 * @file test_AuxiliaryStateManagerMPI.cpp
 * @brief MPI regression tests for ghosted auxiliary state lifecycle behavior.
 */

#include <gtest/gtest.h>

#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryStateStepper.h"
#include "Auxiliary/AuxiliaryStateManager.h"

#include <mpi.h>

#include <array>
#include <span>
#include <string_view>
#include <vector>

namespace svmp::FE::assembly::test {

TEST(AuxiliaryStateManagerMPITest, GhostedLifecycleOpsRefreshGhostEntries)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    using svmp::FE::systems::AuxiliaryStateManager;
    using svmp::FE::systems::AuxiliaryStateSpec;

    AuxiliaryStateManager mgr;
    const auto spec = AuxiliaryStateSpec::nodeField("ghosted", 1);

    const std::array<Real, 3> initial = {
        Real(10 * rank + 1),
        Real(10 * rank + 2),
        Real(-1.0)
    };
    mgr.registerBlock(spec, /*entity_count=*/3, /*owned_entity_count=*/2, initial);

    const std::vector<std::size_t> entity_ids =
        rank == 0 ? std::vector<std::size_t>{0u, 1u, 3u}
                  : std::vector<std::size_t>{3u, 4u, 1u};
    svmp::FE::systems::AuxiliaryDeploymentRegion region;
    region.kind = svmp::FE::systems::AuxiliaryRegionKind::FormulationDefined;
    region.identity = "mpi-restricted-node-subset";
    region.explicit_entities = entity_ids;
    svmp::FE::systems::AuxiliaryEntityRemapMetadata metadata;
    metadata.deployment_region = region;
    metadata.entity_ids = entity_ids;
    mgr.setEntityRemapMetadata("ghosted", metadata);

    const auto schema = mgr.restartSchema("ghosted");
    EXPECT_EQ(schema.scope_name, "Node");
    EXPECT_EQ(schema.deployment_region_kind, "FormulationDefined");
    EXPECT_EQ(schema.region_identity, "mpi-restricted-node-subset");
    EXPECT_EQ(schema.entity_ids, entity_ids);
    EXPECT_EQ(schema.owned_entity_count, 2u);

    auto sync_neighbor = [&](std::string_view, std::span<Real> values) {
        const Real send_value = (rank == 0) ? values[1] : values[0];
        Real recv_value = Real(-1.0);
        MPI_Sendrecv(&send_value, 1, MPI_DOUBLE, 1 - rank, 0,
                     &recv_value, 1, MPI_DOUBLE, 1 - rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        values[2] = recv_value;
    };
    mgr.setGhostSyncHook("ghosted", sync_neighbor);

    mgr.syncGhosts();
    {
        const auto& blk = mgr.getBlock("ghosted");
        EXPECT_DOUBLE_EQ(blk.work()[2], rank == 0 ? 11.0 : 2.0);
    }

    {
        auto& blk = mgr.getBlock("ghosted");
        blk.work()[0] = Real(100 + 10 * rank + 1);
        blk.work()[1] = Real(100 + 10 * rank + 2);
        blk.work()[2] = Real(-99.0);
    }

    mgr.commitAll(/*time=*/0.25);
    {
        const auto& blk = mgr.getBlock("ghosted");
        EXPECT_DOUBLE_EQ(blk.committed()[0], Real(100 + 10 * rank + 1));
        EXPECT_DOUBLE_EQ(blk.committed()[1], Real(100 + 10 * rank + 2));
        EXPECT_DOUBLE_EQ(blk.committed()[2], rank == 0 ? 111.0 : 102.0);
    }

    auto packed = mgr.packAll();

    {
        auto& blk = mgr.getBlock("ghosted");
        blk.work()[0] = Real(-5.0);
        blk.work()[1] = Real(-6.0);
        blk.work()[2] = Real(-7.0);
    }

    mgr.rollbackAll();
    {
        const auto& blk = mgr.getBlock("ghosted");
        EXPECT_DOUBLE_EQ(blk.work()[0], Real(100 + 10 * rank + 1));
        EXPECT_DOUBLE_EQ(blk.work()[1], Real(100 + 10 * rank + 2));
        EXPECT_DOUBLE_EQ(blk.work()[2], rank == 0 ? 111.0 : 102.0);
    }

    {
        auto& blk = mgr.getBlock("ghosted");
        blk.work()[0] = Real(-15.0);
        blk.work()[1] = Real(-16.0);
        blk.work()[2] = Real(-17.0);
    }
    mgr.unpackAll(packed);
    {
        const auto& blk = mgr.getBlock("ghosted");
        EXPECT_DOUBLE_EQ(blk.committed()[0], Real(100 + 10 * rank + 1));
        EXPECT_DOUBLE_EQ(blk.committed()[1], Real(100 + 10 * rank + 2));
        EXPECT_DOUBLE_EQ(blk.committed()[2], rank == 0 ? 111.0 : 102.0);
        EXPECT_DOUBLE_EQ(blk.work()[0], Real(100 + 10 * rank + 1));
        EXPECT_DOUBLE_EQ(blk.work()[1], Real(100 + 10 * rank + 2));
        EXPECT_DOUBLE_EQ(blk.work()[2], rank == 0 ? 111.0 : 102.0);
    }
}

TEST(AuxiliaryStateManagerMPITest, GhostedAdvanceRefreshesGhostEntries)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    using svmp::FE::systems::AuxiliaryDerivativeProvider;
    using svmp::FE::systems::AuxiliaryModelBuilder;
    using svmp::FE::systems::AuxiliaryStateManager;
    using svmp::FE::systems::AuxiliaryStateSpec;
    using svmp::FE::systems::AuxiliaryStepperSpec;
    using svmp::FE::systems::createStepper;
    using svmp::FE::forms::FormExpr;

    AuxiliaryStateManager mgr;
    const auto spec = AuxiliaryStateSpec::nodeField("ghosted", 1);

    const std::array<Real, 3> initial = {
        Real(10 * rank + 1),
        Real(10 * rank + 2),
        Real(-1.0)
    };
    mgr.registerBlock(spec, /*entity_count=*/3, /*owned_entity_count=*/2, initial);

    auto sync_neighbor = [&](std::string_view, std::span<Real> values) {
        const Real send_value = (rank == 0) ? values[1] : values[0];
        Real recv_value = Real(-1.0);
        MPI_Sendrecv(&send_value, 1, MPI_DOUBLE, 1 - rank, 0,
                     &recv_value, 1, MPI_DOUBLE, 1 - rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        values[2] = recv_value;
    };
    mgr.setGhostSyncHook("ghosted", sync_neighbor);

    const auto model = AuxiliaryModelBuilder("ghosted_advance")
        .state("x")
        .ode("x", FormExpr::constant(10.0))
        .build();

    auto stepper = createStepper("ForwardEuler");
    AuxiliaryStepperSpec stepper_spec;
    stepper_spec.method_name = "ForwardEuler";
    stepper->setup(/*dimension=*/1, stepper_spec);

    AuxiliaryDerivativeProvider deriv;
    auto& blk = mgr.getBlock("ghosted");
    std::vector<std::span<const Real>> history;
    for (std::size_t e = 0; e < blk.ownedEntityCount(); ++e) {
        auto work = blk.gatherEntityWork(e);
        const auto committed = blk.gatherEntityCommitted(e);
        const auto result = stepper->advance(*model,
                                             deriv,
                                             work,
                                             committed,
                                             history,
                                             {},
                                             {},
                                             /*t=*/0.0,
                                             /*dt=*/0.1,
                                             /*substep_count=*/1,
                                             e);
        EXPECT_TRUE(result.converged);
        blk.scatterEntityWork(e, work);
    }

    {
        const auto& advanced = mgr.getBlock("ghosted");
        EXPECT_DOUBLE_EQ(advanced.work()[0], Real(10 * rank + 2));
        EXPECT_DOUBLE_EQ(advanced.work()[1], Real(10 * rank + 3));
        EXPECT_DOUBLE_EQ(advanced.work()[2], Real(-1.0));
        EXPECT_DOUBLE_EQ(advanced.committed()[2], Real(-1.0));
    }

    mgr.syncGhosts();
    {
        const auto& synced = mgr.getBlock("ghosted");
        EXPECT_DOUBLE_EQ(synced.work()[0], Real(10 * rank + 2));
        EXPECT_DOUBLE_EQ(synced.work()[1], Real(10 * rank + 3));
        EXPECT_DOUBLE_EQ(synced.work()[2], rank == 0 ? 12.0 : 3.0);
        EXPECT_DOUBLE_EQ(synced.committed()[2], Real(-1.0));
    }
}

TEST(AuxiliaryStateManagerMPITest, RestrictedNodeOwnerUpdatesRefreshGhostEntries)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    using svmp::FE::systems::AuxiliaryStateManager;
    using svmp::FE::systems::AuxiliaryStateSpec;

    AuxiliaryStateManager mgr;
    const auto spec = AuxiliaryStateSpec::nodeField("node_solution", 1);
    const std::array<Real, 3> initial = {Real(0.0), Real(0.0), Real(-1.0)};
    mgr.registerBlock(spec,
                      /*entity_count=*/3,
                      /*owned_entity_count=*/2,
                      initial);

    const std::vector<std::size_t> entity_ids =
        rank == 0 ? std::vector<std::size_t>{0u, 1u, 3u}
                  : std::vector<std::size_t>{3u, 4u, 1u};
    svmp::FE::systems::AuxiliaryDeploymentRegion region;
    region.kind = svmp::FE::systems::AuxiliaryRegionKind::FormulationDefined;
    region.identity = "mpi-restricted-node-subset";
    region.explicit_entities = entity_ids;
    svmp::FE::systems::AuxiliaryEntityRemapMetadata metadata;
    metadata.deployment_region = region;
    metadata.entity_ids = entity_ids;
    mgr.setEntityRemapMetadata("node_solution", metadata);

    auto sync_neighbor = [&](std::string_view, std::span<Real> values) {
        const Real send_value = (rank == 0) ? values[1] : values[0];
        Real recv_value = Real(-1.0);
        MPI_Sendrecv(&send_value, 1, MPI_DOUBLE, 1 - rank, 1,
                     &recv_value, 1, MPI_DOUBLE, 1 - rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        values[2] = recv_value;
    };
    mgr.setGhostSyncHook("node_solution", sync_neighbor);

    auto& blk = mgr.getBlock("node_solution");
    blk.work()[0] = Real(100 + 10 * rank);
    blk.work()[1] = Real(101 + 10 * rank);
    blk.work()[2] = Real(-99.0);

    mgr.syncGhosts("node_solution");
    EXPECT_DOUBLE_EQ(blk.work()[2], rank == 0 ? 110.0 : 101.0);

    blk.work()[2] = Real(-88.0);
    mgr.commitAll(/*time=*/0.5);
    EXPECT_DOUBLE_EQ(blk.committed()[0], Real(100 + 10 * rank));
    EXPECT_DOUBLE_EQ(blk.committed()[1], Real(101 + 10 * rank));
    EXPECT_DOUBLE_EQ(blk.committed()[2], rank == 0 ? 110.0 : 101.0);
}

TEST(AuxiliaryStateManagerMPITest,
     RaggedRestrictedNodeOwnerUpdatesRefreshGhostComponents)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    using svmp::FE::systems::AuxiliaryDeploymentRegion;
    using svmp::FE::systems::AuxiliaryEntityRemapMetadata;
    using svmp::FE::systems::AuxiliaryLayoutMode;
    using svmp::FE::systems::AuxiliaryRegionKind;
    using svmp::FE::systems::AuxiliaryStateManager;
    using svmp::FE::systems::AuxiliaryStateSpec;

    AuxiliaryStateManager mgr;
    auto spec = AuxiliaryStateSpec::nodeField("ragged_node_solution", 1);
    spec.layout_mode = AuxiliaryLayoutMode::Ragged;

    const std::vector<std::size_t> entity_ids =
        rank == 0 ? std::vector<std::size_t>{0u, 1u, 3u}
                  : std::vector<std::size_t>{3u, 4u, 1u};
    const std::vector<std::size_t> component_offsets =
        rank == 0 ? std::vector<std::size_t>{0u, 1u, 3u, 6u}
                  : std::vector<std::size_t>{0u, 3u, 4u, 6u};

    const std::vector<Real> initial(component_offsets.back(), Real(-1.0));
    mgr.registerBlockRagged(
        spec, component_offsets, /*owned_entity_count=*/2, initial);

    AuxiliaryDeploymentRegion region;
    region.kind = AuxiliaryRegionKind::FormulationDefined;
    region.identity = "mpi-ragged-restricted-node-subset";
    region.explicit_entities = entity_ids;
    AuxiliaryEntityRemapMetadata metadata;
    metadata.deployment_region = region;
    metadata.entity_ids = entity_ids;
    metadata.component_offsets = component_offsets;
    mgr.setEntityRemapMetadata("ragged_node_solution", metadata);

    const auto schema = mgr.restartSchema("ragged_node_solution");
    EXPECT_EQ(schema.scope_name, "Node");
    EXPECT_EQ(schema.entity_ids, entity_ids);
    EXPECT_EQ(schema.component_offsets, component_offsets);
    EXPECT_EQ(schema.owned_entity_count, 2u);

    auto sync_neighbor = [&](std::string_view, std::span<Real> values) {
        std::array<Real, 3> send_buf{Real(-77.0), Real(-77.0), Real(-77.0)};
        std::array<Real, 3> recv_buf{Real(-88.0), Real(-88.0), Real(-88.0)};
        int send_count = 0;
        int recv_count = 0;
        std::size_t ghost_begin = 0;

        if (rank == 0) {
            // Owned node 1 has width 2 on rank 0; it is rank 1's ghost.
            send_buf[0] = values[1];
            send_buf[1] = values[2];
            send_count = 2;
            recv_count = 3;
            ghost_begin = 3;
        } else {
            // Owned node 3 has width 3 on rank 1; it is rank 0's ghost.
            send_buf[0] = values[0];
            send_buf[1] = values[1];
            send_buf[2] = values[2];
            send_count = 3;
            recv_count = 2;
            ghost_begin = 4;
        }

        MPI_Sendrecv(send_buf.data(), send_count, MPI_DOUBLE, 1 - rank, 2,
                     recv_buf.data(), recv_count, MPI_DOUBLE, 1 - rank, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < recv_count; ++i) {
            values[ghost_begin + static_cast<std::size_t>(i)] =
                recv_buf[static_cast<std::size_t>(i)];
        }
    };
    mgr.setGhostSyncHook("ragged_node_solution", sync_neighbor);

    auto& blk = mgr.getBlock("ragged_node_solution");
    if (rank == 0) {
        blk.work()[0] = Real(10.0);
        blk.work()[1] = Real(11.0);
        blk.work()[2] = Real(12.0);
        blk.work()[3] = Real(-900.0);
        blk.work()[4] = Real(-901.0);
        blk.work()[5] = Real(-902.0);
    } else {
        blk.work()[0] = Real(30.0);
        blk.work()[1] = Real(31.0);
        blk.work()[2] = Real(32.0);
        blk.work()[3] = Real(40.0);
        blk.work()[4] = Real(-910.0);
        blk.work()[5] = Real(-911.0);
    }

    mgr.syncGhosts("ragged_node_solution");
    if (rank == 0) {
        EXPECT_EQ(std::vector<Real>(blk.work().begin(), blk.work().end()),
                  (std::vector<Real>{10.0, 11.0, 12.0, 30.0, 31.0, 32.0}));
    } else {
        EXPECT_EQ(std::vector<Real>(blk.work().begin(), blk.work().end()),
                  (std::vector<Real>{30.0, 31.0, 32.0, 40.0, 11.0, 12.0}));
    }

    // Mutate ghost values locally; commit must copy only the owned prefix and
    // repopulate ghosts from owner ranks.
    if (rank == 0) {
        blk.work()[0] = Real(100.0);
        blk.work()[1] = Real(101.0);
        blk.work()[2] = Real(102.0);
        blk.work()[3] = Real(-700.0);
        blk.work()[4] = Real(-701.0);
        blk.work()[5] = Real(-702.0);
    } else {
        blk.work()[0] = Real(300.0);
        blk.work()[1] = Real(301.0);
        blk.work()[2] = Real(302.0);
        blk.work()[3] = Real(400.0);
        blk.work()[4] = Real(-710.0);
        blk.work()[5] = Real(-711.0);
    }

    mgr.commitAll(/*time=*/0.75);
    if (rank == 0) {
        EXPECT_EQ(std::vector<Real>(blk.committed().begin(), blk.committed().end()),
                  (std::vector<Real>{100.0, 101.0, 102.0, 300.0, 301.0, 302.0}));
    } else {
        EXPECT_EQ(std::vector<Real>(blk.committed().begin(), blk.committed().end()),
                  (std::vector<Real>{300.0, 301.0, 302.0, 400.0, 101.0, 102.0}));
    }
}

} // namespace svmp::FE::assembly::test
