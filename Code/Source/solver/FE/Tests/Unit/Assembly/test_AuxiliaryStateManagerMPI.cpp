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

} // namespace svmp::FE::assembly::test
