/**
 * @file test_GhostDofManager.cpp
 * @brief Unit tests for GhostDofManager (mesh-independent parts)
 */

#include <gtest/gtest.h>

#include "FE/Dofs/GhostDofManager.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::dofs::GhostDofManager;
using svmp::FE::dofs::OwnerResolutionStrategy;

TEST(GhostDofManager, ManualGhostAndSharedSchedules) {
    GhostDofManager mgr;

    // Ghost DOFs needed locally but owned by other ranks.
    const std::vector<GlobalIndex> ghost = {100, 101};
    const std::vector<int> owners = {1, 2};
    mgr.setGhostDofs(ghost, owners);

    // Shared DOFs with neighbors.
    mgr.addSharedDofsWithNeighbor(1, std::vector<GlobalIndex>{5, 6});
    mgr.addSharedDofsWithNeighbor(2, std::vector<GlobalIndex>{6, 7});

    mgr.resolveSharedOwnership(OwnerResolutionStrategy::LowestRank);
    mgr.buildGhostExchange();

    const auto& sched = mgr.getCommSchedule();
    ASSERT_EQ(sched.neighbor_ranks.size(), 2u);
    EXPECT_EQ(sched.neighbor_ranks[0], 1);
    EXPECT_EQ(sched.neighbor_ranks[1], 2);

    // Recv lists come from ghost owners.
    EXPECT_EQ(sched.recv_lists[0], (std::vector<GlobalIndex>{100}));
    EXPECT_EQ(sched.recv_lists[1], (std::vector<GlobalIndex>{101}));

    // Send lists include shared DOFs we own (default my_rank_=0).
    EXPECT_EQ(sched.send_lists[0], (std::vector<GlobalIndex>{5, 6}));
    EXPECT_EQ(sched.send_lists[1], (std::vector<GlobalIndex>{6, 7}));
}

TEST(GhostDofManager, QueryGhostAndSharedSets) {
    GhostDofManager mgr;
    mgr.setGhostDofs(std::vector<GlobalIndex>{10}, std::vector<int>{1});
    mgr.addSharedDofsWithNeighbor(1, std::vector<GlobalIndex>{2, 3});

    EXPECT_TRUE(mgr.isGhost(10));
    EXPECT_FALSE(mgr.isGhost(2));
    EXPECT_TRUE(mgr.isShared(2));
    EXPECT_TRUE(mgr.isShared(3));
    EXPECT_FALSE(mgr.isShared(10));
}

TEST(GhostDofManager, ResolveSharedOwnershipRemovesOwnedDofsFromGhostSet) {
    GhostDofManager mgr;

    // Start by marking a shared DOF as a ghost owned by rank 1.
    mgr.setGhostDofs(std::vector<GlobalIndex>{6}, std::vector<int>{1});
    mgr.addSharedDofsWithNeighbor(1, std::vector<GlobalIndex>{6});

    // With LowestRank and default my_rank_=0, this DOF becomes locally owned.
    mgr.resolveSharedOwnership(OwnerResolutionStrategy::LowestRank);
    mgr.buildGhostExchange();

    EXPECT_TRUE(mgr.getGhostDofs().empty());

    const auto& sched = mgr.getCommSchedule();
    ASSERT_EQ(sched.neighbor_ranks.size(), 1u);
    EXPECT_EQ(sched.neighbor_ranks[0], 1);
    EXPECT_TRUE(sched.recv_lists[0].empty());
    EXPECT_EQ(sched.send_lists[0], (std::vector<GlobalIndex>{6}));
}

TEST(GhostDofManager, SyncGhostValuesMapsReceivedDofsToGhostOrdering) {
    GhostDofManager mgr;

    // Ghost set is globally sorted: [10, 30, 40, 50]
    mgr.setGhostDofs(std::vector<GlobalIndex>{10, 30, 40, 50},
                     std::vector<int>{1, 2, 1, 2});
    mgr.buildGhostExchange();

    std::vector<double> local_values(100, 0.0);
    std::vector<double> ghost_values(4, -1.0);

    const std::unordered_map<int, std::vector<double>> recv_by_rank = {
        {1, {100.0, 400.0}}, // DOFs {10, 40}
        {2, {300.0, 500.0}}  // DOFs {30, 50}
    };

    mgr.syncGhostValues(local_values, ghost_values,
                        [&](int /*send_rank*/, std::span<const double> /*send_data*/,
                            int recv_rank, std::span<double> recv_data) {
                            const auto it = recv_by_rank.find(recv_rank);
                            EXPECT_NE(it, recv_by_rank.end());
                            if (it == recv_by_rank.end()) return;
                            EXPECT_EQ(recv_data.size(), it->second.size());
                            const auto n = std::min(recv_data.size(), it->second.size());
                            for (std::size_t i = 0; i < n; ++i) {
                                recv_data[i] = it->second[i];
                            }
                        });

    EXPECT_EQ(ghost_values, (std::vector<double>{100.0, 300.0, 400.0, 500.0}));
}
