/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GhostContributionManager.h"
#include "Dofs/DofMap.h"

#include <array>
#include <cmath>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp::FE::assembly::test {

#if FE_HAS_MPI
TEST(GhostContributionManagerMPITest, RepeatedExchangeMicrobenchmark)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "Run with 2 MPI ranks to enable this test";
    }

    constexpr GlobalIndex total_dofs = 20;
    constexpr int exchanges = 16;
    constexpr int matrix_entries_per_exchange = 32;
    constexpr int vector_entries_per_exchange = 16;

    dofs::DofMap dof_map(2, total_dofs, 4);

    if (rank == 0) {
        dof_map.setCellDofs(0, std::array<GlobalIndex, 4>{0, 1, 2, 3});
        dof_map.setCellDofs(1, std::array<GlobalIndex, 4>{5, 6, 10, 11});
    } else {
        dof_map.setCellDofs(0, std::array<GlobalIndex, 4>{10, 11, 12, 13});
        dof_map.setCellDofs(1, std::array<GlobalIndex, 4>{8, 9, 14, 15});
    }

    dof_map.setNumDofs(total_dofs);
    dof_map.setNumLocalDofs(0);
    dof_map.setMyRank(rank);
    dof_map.setDofOwnership([](GlobalIndex dof) -> int {
        return (dof < 10) ? 0 : 1;
    });
    dof_map.finalize();

    GhostContributionManager manager(dof_map, MPI_COMM_WORLD);
    manager.setPolicy(GhostPolicy::ReverseScatter);
    manager.setDeterministic(true);
    manager.initialize();

    const GlobalIndex remote_row = (rank == 0) ? 10 : 8;
    const GlobalIndex local_col_base = (rank == 0) ? 0 : 10;

    double total_exchange_time = 0.0;
    std::size_t total_bytes_sent = 0;
    std::size_t total_bytes_received = 0;

    for (int exchange = 0; exchange < exchanges; ++exchange) {
        for (int i = 0; i < matrix_entries_per_exchange; ++i) {
            const GlobalIndex col = local_col_base + static_cast<GlobalIndex>(i % 4);
            const Real value = static_cast<Real>(exchange + 1) * static_cast<Real>(i + 1);
            EXPECT_FALSE(manager.addMatrixContribution(remote_row, col, value));
        }
        for (int i = 0; i < vector_entries_per_exchange; ++i) {
            const Real value = static_cast<Real>(exchange + 1) * static_cast<Real>(i + 1) * 0.25;
            EXPECT_FALSE(manager.addVectorContribution(remote_row, value));
        }

        manager.exchangeContributions();

        const auto& stats = manager.getLastExchangeStats();
        EXPECT_EQ(stats.matrix_entries_sent, static_cast<std::size_t>(matrix_entries_per_exchange));
        EXPECT_EQ(stats.matrix_entries_received, static_cast<std::size_t>(matrix_entries_per_exchange));
        EXPECT_EQ(stats.vector_entries_sent, static_cast<std::size_t>(vector_entries_per_exchange));
        EXPECT_EQ(stats.vector_entries_received, static_cast<std::size_t>(vector_entries_per_exchange));
        EXPECT_GT(stats.bytes_sent, 0u);
        EXPECT_GT(stats.bytes_received, 0u);
        EXPECT_TRUE(std::isfinite(stats.exchange_time_seconds));
        EXPECT_GE(stats.exchange_time_seconds, 0.0);

        total_exchange_time += stats.exchange_time_seconds;
        total_bytes_sent += stats.bytes_sent;
        total_bytes_received += stats.bytes_received;

        manager.clearReceivedContributions();
        manager.clearSendBuffers();
    }

    const double local_avg_exchange_time = total_exchange_time / static_cast<double>(exchanges);
    double max_avg_exchange_time = 0.0;
    MPI_Allreduce(&local_avg_exchange_time, &max_avg_exchange_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        RecordProperty("avg_exchange_time_seconds", max_avg_exchange_time);
        RecordProperty("avg_bytes_sent", static_cast<double>(total_bytes_sent) / static_cast<double>(exchanges));
        RecordProperty("avg_bytes_received", static_cast<double>(total_bytes_received) / static_cast<double>(exchanges));
    }

    EXPECT_LT(max_avg_exchange_time, 5.0);
}
#endif

} // namespace svmp::FE::assembly::test
