/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GhostContributionManager.h"
#include "Dofs/DofMap.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <cmath>
#include "Tests/Unit/TimingAssertions.h"

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp::FE::assembly::test {

#if FE_HAS_MPI
TEST(GhostContributionManagerMPITest, TwoRankGhostExchange) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "Run with 2 MPI ranks to enable this test";
    }

    constexpr GlobalIndex total_dofs = 20;
    dofs::DofMap dof_map(2, total_dofs, 4);

    if (rank == 0) {
        dof_map.setCellDofs(0, std::array<GlobalIndex, 4>{0, 1, 2, 3});
        dof_map.setCellDofs(1, std::array<GlobalIndex, 4>{5, 6, 10, 11});  // includes ghosts 10,11
    } else {
        dof_map.setCellDofs(0, std::array<GlobalIndex, 4>{10, 11, 12, 13});
        dof_map.setCellDofs(1, std::array<GlobalIndex, 4>{8, 9, 14, 15});  // includes ghosts 8,9
    }

    dof_map.setNumDofs(total_dofs);
    dof_map.setNumLocalDofs(0);  // disable contiguous heuristic
    dof_map.setMyRank(rank);
    dof_map.setDofOwnership([](GlobalIndex dof) -> int {
        return (dof < 10) ? 0 : 1;
    });
    dof_map.finalize();

    GhostContributionManager manager(dof_map, MPI_COMM_WORLD);
    manager.setPolicy(GhostPolicy::ReverseScatter);
    manager.setDeterministic(true);
    manager.initialize();

    if (rank == 0) {
        EXPECT_FALSE(manager.addMatrixContribution(/*row=*/10, /*col=*/0, /*value=*/2.0));
        EXPECT_FALSE(manager.addVectorContribution(/*row=*/10, /*value=*/-1.5));
    } else {
        EXPECT_FALSE(manager.addMatrixContribution(/*row=*/8, /*col=*/10, /*value=*/3.0));
        EXPECT_FALSE(manager.addVectorContribution(/*row=*/8, /*value=*/4.5));
    }

    manager.exchangeContributions();

    const auto received = manager.getReceivedMatrixContributions();
    const auto received_vec = manager.getReceivedVectorContributions();
    if (rank == 0) {
        bool found = false;
        for (const auto& e : received) {
            if (e.global_row == 8 && e.global_col == 10 && e.value == 3.0) {
                found = true;
            }
        }
        EXPECT_TRUE(found);

        bool found_vec = false;
        for (const auto& e : received_vec) {
            if (e.global_row == 8 && e.value == 4.5) {
                found_vec = true;
            }
        }
        EXPECT_TRUE(found_vec);
    } else {
        bool found = false;
        for (const auto& e : received) {
            if (e.global_row == 10 && e.global_col == 0 && e.value == 2.0) {
                found = true;
            }
        }
        EXPECT_TRUE(found);

        bool found_vec = false;
        for (const auto& e : received_vec) {
            if (e.global_row == 10 && e.value == -1.5) {
                found_vec = true;
            }
        }
        EXPECT_TRUE(found_vec);
    }
}

TEST(GhostContributionManagerMPITest, MultiRankGhostExchange) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 4) {
        GTEST_SKIP() << "Run with 4+ MPI ranks to enable this test";
    }

    constexpr GlobalIndex dofs_per_rank = 10;
    const GlobalIndex total_dofs = static_cast<GlobalIndex>(size) * dofs_per_rank;

    dofs::DofMap dof_map(2, total_dofs, 4);

    const int next = (rank + 1) % size;
    const int prev = (rank - 1 + size) % size;
    const GlobalIndex owned_base = static_cast<GlobalIndex>(rank) * dofs_per_rank;

    dof_map.setCellDofs(
        0, std::array<GlobalIndex, 4>{owned_base + 0, owned_base + 1, owned_base + 2,
                                      static_cast<GlobalIndex>(next) * dofs_per_rank});
    dof_map.setCellDofs(
        1, std::array<GlobalIndex, 4>{owned_base + 3, owned_base + 4, owned_base + 5,
                                      static_cast<GlobalIndex>(prev) * dofs_per_rank});

    dof_map.setNumDofs(total_dofs);
    dof_map.setNumLocalDofs(0);  // disable contiguous heuristic
    dof_map.setMyRank(rank);
    dof_map.setDofOwnership([dofs_per_rank](GlobalIndex dof) -> int {
        return static_cast<int>(dof / dofs_per_rank);
    });
    dof_map.finalize();

    GhostContributionManager manager(dof_map, MPI_COMM_WORLD);
    manager.setPolicy(GhostPolicy::ReverseScatter);
    manager.setDeterministic(true);
    manager.initialize();

    const GlobalIndex row = static_cast<GlobalIndex>(next) * dofs_per_rank;
    const GlobalIndex col = owned_base;
    const Real value = static_cast<Real>(rank + 1);

    EXPECT_FALSE(manager.addMatrixContribution(row, col, value));
    manager.exchangeContributions();

    const int expected_sender = prev;
    const GlobalIndex expected_row = owned_base;
    const GlobalIndex expected_col = static_cast<GlobalIndex>(expected_sender) * dofs_per_rank;
    const Real expected_value = static_cast<Real>(expected_sender + 1);

    bool found = false;
    for (const auto& e : manager.getReceivedMatrixContributions()) {
        if (e.global_row == expected_row && e.global_col == expected_col && e.value == expected_value) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(GhostContributionManagerMPITest, LargeMessageHandling) {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const char* env = std::getenv("SVMP_FE_RUN_STRESS_TESTS");
    if (env == nullptr || std::string(env) != "1") {
        GTEST_SKIP() << "Set SVMP_FE_RUN_STRESS_TESTS=1 to enable large-message exchange";
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int next = (rank + 1) % size;
    const int previous = (rank + size - 1) % size;
    dofs::DofMap dof_map(1, size, 3);
    dof_map.setCellDofs(0, std::array<GlobalIndex, 3>{rank, next, previous});
    dof_map.setNumDofs(size);
    dof_map.setNumLocalDofs(0);
    dof_map.setMyRank(rank);
    dof_map.setDofOwnership([](GlobalIndex dof) { return static_cast<int>(dof); });
    dof_map.finalize();

    GhostContributionManager manager(dof_map, MPI_COMM_WORLD);
    manager.setPolicy(GhostPolicy::ReverseScatter);
    manager.initialize();

    constexpr std::size_t entries = 50000;
    for (std::size_t i = 0; i < entries; ++i) {
        manager.addMatrixContribution(next, rank, 1.0);
    }
    manager.exchangeContributions();
    const auto received = manager.getReceivedMatrixContributions();
    ASSERT_EQ(received.size(), entries);
    EXPECT_TRUE(std::all_of(received.begin(), received.end(), [&](const auto& entry) {
        return entry.global_row == rank && entry.global_col == previous && entry.value == 1.0;
    }));

}

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

    // Opt-in (SVMP_ENABLE_TIMING_ASSERTIONS=1): wall-clock limits flake on shared nodes.
    if (::svmp::FE::test::timingAssertionsEnabled()) {
        EXPECT_LT(max_avg_exchange_time, 5.0);
    }
}
#endif

} // namespace svmp::FE::assembly::test
