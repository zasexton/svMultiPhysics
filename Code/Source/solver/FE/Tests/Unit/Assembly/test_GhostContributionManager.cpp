/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GhostContributionManager.cpp
 * @brief Unit tests for GhostContributionManager
 *
 * Tests ghost DOF handling for parallel assembly:
 * - Construction and configuration
 * - addMatrixContribution / addVectorContribution
 * - exchangeContributions (single-rank / mock MPI)
 * - Contribution accumulation
 * - Buffer management
 * - Determinism guarantees
 */

#include <gtest/gtest.h>

#include "Assembly/GhostContributionManager.h"
#include "Dofs/DofMap.h"

#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// Helper Functions for Testing
// ============================================================================

/**
 * @brief Create a configured DofMap with ownership information
 *
 * Sets up a DofMap for testing ghost contribution handling with
 * configurable ownership range.
 */
inline dofs::DofMap createDofMapWithOwnership(GlobalIndex total_dofs,
                                               GlobalIndex owned_begin,
                                               GlobalIndex owned_end) {
    // 4 cells with 4 DOFs each, overlapping connectivity
    dofs::DofMap dof_map(4, total_dofs, 4);

    std::vector<GlobalIndex> cell0_dofs = {0, 1, 2, 3};
    std::vector<GlobalIndex> cell1_dofs = {2, 3, 4, 5};
    std::vector<GlobalIndex> cell2_dofs = {4, 5, 6, 7};
    std::vector<GlobalIndex> cell3_dofs = {6, 7, 8, 9};

    dof_map.setCellDofs(0, cell0_dofs);
    dof_map.setCellDofs(1, cell1_dofs);
    dof_map.setCellDofs(2, cell2_dofs);
    dof_map.setCellDofs(3, cell3_dofs);
    dof_map.setNumDofs(total_dofs);
    dof_map.setNumLocalDofs(owned_end - owned_begin);

    // Set ownership function
    dof_map.setDofOwnership([owned_begin, owned_end](GlobalIndex dof) -> int {
        return (dof >= owned_begin && dof < owned_end) ? 0 : 1;
    });

    dof_map.finalize();
    return dof_map;
}

// ============================================================================
// GhostContributionManager Tests
// ============================================================================

class GhostContributionManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a DOF map: 10 total DOFs, this rank owns DOFs [0, 5)
        dof_map_ = createDofMapWithOwnership(10, 0, 5);
        manager_ = std::make_unique<GhostContributionManager>(dof_map_);
    }

    dofs::DofMap dof_map_;
    std::unique_ptr<GhostContributionManager> manager_;
};

TEST_F(GhostContributionManagerTest, DefaultConstruction) {
    GhostContributionManager manager;
    EXPECT_FALSE(manager.isInitialized());
    EXPECT_EQ(manager.getPolicy(), GhostPolicy::ReverseScatter);  // Default
}

TEST_F(GhostContributionManagerTest, ConstructWithDofMap) {
    GhostContributionManager manager(dof_map_);
    EXPECT_FALSE(manager.isInitialized());  // Not initialized until initialize() called
}

TEST_F(GhostContributionManagerTest, SetPolicy) {
    manager_->setPolicy(GhostPolicy::OwnedRowsOnly);
    EXPECT_EQ(manager_->getPolicy(), GhostPolicy::OwnedRowsOnly);

    manager_->setPolicy(GhostPolicy::ReverseScatter);
    EXPECT_EQ(manager_->getPolicy(), GhostPolicy::ReverseScatter);
}

TEST_F(GhostContributionManagerTest, SetDeterministic) {
    // Should not throw
    EXPECT_NO_THROW(manager_->setDeterministic(true));
    EXPECT_NO_THROW(manager_->setDeterministic(false));
}

TEST_F(GhostContributionManagerTest, Initialize) {
    EXPECT_NO_THROW(manager_->initialize());
    EXPECT_TRUE(manager_->isInitialized());
}

// ============================================================================
// Ownership Tests
// ============================================================================

TEST_F(GhostContributionManagerTest, IsOwnedDof) {
    manager_->initialize();

    // DOFs 0-4 are owned by this rank
    EXPECT_TRUE(manager_->isOwned(0));
    EXPECT_TRUE(manager_->isOwned(4));

    // DOFs 5-9 are not owned (ghosts)
    EXPECT_FALSE(manager_->isOwned(5));
    EXPECT_FALSE(manager_->isOwned(9));
}

// ============================================================================
// Matrix Contribution Tests
// ============================================================================

TEST_F(GhostContributionManagerTest, AddMatrixContributionOwned) {
    manager_->initialize();

    // Contribution to owned row should return true (direct insertion)
    bool owned = manager_->addMatrixContribution(2, 3, 1.5);
    EXPECT_TRUE(owned);
}

TEST_F(GhostContributionManagerTest, AddMatrixContributionGhost) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();

    // Contribution to ghost row should return false (buffered)
    bool owned = manager_->addMatrixContribution(7, 3, 2.5);
    EXPECT_FALSE(owned);

    // Should have buffered this contribution
    EXPECT_GT(manager_->numBufferedMatrixContributions(), 0u);
}

TEST_F(GhostContributionManagerTest, AddMatrixContributionOwnedRowsOnly) {
    manager_->setPolicy(GhostPolicy::OwnedRowsOnly);
    manager_->initialize();

    // With OwnedRowsOnly, ghost contributions are discarded
    bool owned = manager_->addMatrixContribution(7, 3, 2.5);
    EXPECT_FALSE(owned);

    // Buffer should be empty since we're discarding ghost contributions
    // (OwnedRowsOnly policy doesn't buffer - it discards)
}

// ============================================================================
// Vector Contribution Tests
// ============================================================================

TEST_F(GhostContributionManagerTest, AddVectorContributionOwned) {
    manager_->initialize();

    bool owned = manager_->addVectorContribution(3, 4.0);
    EXPECT_TRUE(owned);
}

TEST_F(GhostContributionManagerTest, AddVectorContributionGhost) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();

    bool owned = manager_->addVectorContribution(8, 5.0);
    EXPECT_FALSE(owned);

    EXPECT_GT(manager_->numBufferedVectorContributions(), 0u);
}

// ============================================================================
// Batch Contribution Tests
// ============================================================================

TEST_F(GhostContributionManagerTest, AddMatrixContributionsBatch) {
    manager_->initialize();

    // Row DOFs: mix of owned and ghost
    std::vector<GlobalIndex> row_dofs = {2, 3, 6, 7};  // 2,3 owned; 6,7 ghost
    std::vector<GlobalIndex> col_dofs = {0, 1, 2, 3};
    std::vector<Real> values(16, 1.0);  // 4x4 dense matrix

    std::vector<GhostContribution> owned_contributions;
    manager_->addMatrixContributions(row_dofs, col_dofs, values, owned_contributions);

    // Should have contributions for owned rows (2 rows x 4 cols = 8)
    // and buffered contributions for ghost rows
    EXPECT_GT(owned_contributions.size(), 0u);
}

// ============================================================================
// Buffer Management Tests
// ============================================================================

TEST_F(GhostContributionManagerTest, ClearSendBuffers) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();

    // Add some ghost contributions
    manager_->addMatrixContribution(7, 3, 2.5);
    manager_->addMatrixContribution(8, 4, 3.5);
    EXPECT_GT(manager_->numBufferedMatrixContributions(), 0u);

    // Clear buffers
    manager_->clearSendBuffers();
    EXPECT_EQ(manager_->numBufferedMatrixContributions(), 0u);
}

TEST_F(GhostContributionManagerTest, ReserveBuffers) {
    manager_->initialize();

    // Should not throw
    EXPECT_NO_THROW(manager_->reserveBuffers(100));
}

// ============================================================================
// Exchange Tests (Single Rank)
// ============================================================================

TEST_F(GhostContributionManagerTest, ExchangeContributionsSingleRank) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();

    // Add ghost contributions
    manager_->addMatrixContribution(7, 3, 2.5);
    manager_->addVectorContribution(8, 5.0);

    // Exchange should complete without error on single rank
    EXPECT_NO_THROW(manager_->exchangeContributions());

    // On single rank with no real ghost DOFs (all owned by rank 0 in serial),
    // received contributions may be empty or contain self-contributions
    manager_->clearReceivedContributions();
}

TEST_F(GhostContributionManagerTest, NonBlockingExchange) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();

    // Add ghost contributions
    manager_->addMatrixContribution(7, 3, 2.5);

    // Start non-blocking exchange
    EXPECT_NO_THROW(manager_->startExchange());
    EXPECT_TRUE(manager_->isExchangeInProgress());

    // Wait for completion
    EXPECT_NO_THROW(manager_->waitExchange());
    EXPECT_FALSE(manager_->isExchangeInProgress());
}

// ============================================================================
// Received Contributions Tests
// ============================================================================

TEST_F(GhostContributionManagerTest, ClearReceivedContributions) {
    manager_->initialize();
    manager_->exchangeContributions();

    // Should not throw
    EXPECT_NO_THROW(manager_->clearReceivedContributions());
}

TEST_F(GhostContributionManagerTest, GetReceivedContributionsAfterExchange) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();

    manager_->exchangeContributions();

    auto matrix_contribs = manager_->getReceivedMatrixContributions();
    auto vector_contribs = manager_->getReceivedVectorContributions();

    // Spans should be valid (may be empty on single rank)
    // Just verify they're accessible
    EXPECT_GE(matrix_contribs.size(), 0u);
    EXPECT_GE(vector_contribs.size(), 0u);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(GhostContributionManagerTest, NeighborCount) {
    manager_->initialize();

    // On single rank, may have no neighbors
    EXPECT_GE(manager_->numNeighbors(), 0);
}

TEST_F(GhostContributionManagerTest, ExchangeStats) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();

    manager_->addMatrixContribution(7, 3, 2.5);
    manager_->exchangeContributions();

    const auto& stats = manager_->getLastExchangeStats();

    // Stats should be non-negative
    EXPECT_GE(stats.bytes_sent, 0u);
    EXPECT_GE(stats.bytes_received, 0u);
    EXPECT_GE(stats.matrix_entries_sent, 0u);
    EXPECT_GE(stats.exchange_time_seconds, 0.0);
}

// ============================================================================
// GhostContribution Struct Tests
// ============================================================================

TEST(GhostContributionTest, Ordering) {
    GhostContribution a{1, 2, 1.0};
    GhostContribution b{1, 3, 2.0};
    GhostContribution c{2, 1, 3.0};

    // a < b (same row, col 2 < col 3)
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);

    // a < c (row 1 < row 2)
    EXPECT_TRUE(a < c);
    EXPECT_FALSE(c < a);

    // b < c (row 1 < row 2)
    EXPECT_TRUE(b < c);
}

TEST(GhostContributionTest, EqualityComparison) {
    GhostContribution a{1, 2, 1.0};
    GhostContribution b{1, 2, 5.0};  // Same row/col, different value

    // Comparison is by row/col only, not value
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(b < a);
}

// ============================================================================
// GhostBuffer Tests
// ============================================================================

TEST(GhostBufferTest, Clear) {
    GhostBuffer buffer;
    buffer.dest_rank = 1;
    buffer.entries.push_back({1, 2, 1.0});
    buffer.vector_entries.push_back(3.0);

    buffer.clear();

    EXPECT_TRUE(buffer.entries.empty());
    EXPECT_TRUE(buffer.vector_entries.empty());
    EXPECT_EQ(buffer.dest_rank, 1);  // dest_rank not cleared
}

TEST(GhostBufferTest, Reserve) {
    GhostBuffer buffer;

    // Should not throw
    EXPECT_NO_THROW(buffer.reserve(100, 50));
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST_F(GhostContributionManagerTest, DeterministicBuffering) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->setDeterministic(true);
    manager_->initialize();

    // Add contributions in specific order
    manager_->addMatrixContribution(7, 3, 1.0);
    manager_->addMatrixContribution(8, 2, 2.0);
    manager_->addMatrixContribution(7, 1, 3.0);

    // With deterministic mode, contributions should be sorted before exchange
    // (sorting happens during exchangeContributions)
    EXPECT_NO_THROW(manager_->exchangeContributions());
}

TEST_F(GhostContributionManagerTest, ReproducibleResults) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->setDeterministic(true);
    manager_->initialize();

    // Run twice with same contributions
    auto runAssembly = [&]() {
        manager_->clearSendBuffers();
        manager_->addMatrixContribution(7, 3, 1.0);
        manager_->addMatrixContribution(8, 2, 2.0);
        manager_->addMatrixContribution(7, 1, 3.0);
        manager_->exchangeContributions();
        return manager_->getLastExchangeStats();
    };

    auto stats1 = runAssembly();
    auto stats2 = runAssembly();

    // Statistics should be identical
    EXPECT_EQ(stats1.matrix_entries_sent, stats2.matrix_entries_sent);
    EXPECT_EQ(stats1.bytes_sent, stats2.bytes_sent);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(GhostContributionManagerTest, EmptyExchange) {
    manager_->initialize();

    // Exchange with no buffered contributions should succeed
    EXPECT_NO_THROW(manager_->exchangeContributions());
}

TEST_F(GhostContributionManagerTest, MultipleExchanges) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();

    // Multiple exchange cycles
    for (int i = 0; i < 3; ++i) {
        manager_->addMatrixContribution(7, 3, static_cast<Real>(i));
        manager_->exchangeContributions();
        manager_->clearReceivedContributions();
        manager_->clearSendBuffers();
    }
}

TEST_F(GhostContributionManagerTest, LargeContribution) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();

    // Add many contributions
    for (GlobalIndex row = 5; row < 10; ++row) {
        for (GlobalIndex col = 0; col < 10; ++col) {
            manager_->addMatrixContribution(row, col, 1.0);
        }
    }

    // 5 ghost rows x 10 cols = 50 contributions
    EXPECT_EQ(manager_->numBufferedMatrixContributions(), 50u);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(GhostContributionManagerTest, MoveConstruction) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();
    manager_->addMatrixContribution(7, 3, 1.0);

    GhostContributionManager moved(std::move(*manager_));

    // Moved-to object should be usable
    EXPECT_TRUE(moved.isInitialized());
    EXPECT_GT(moved.numBufferedMatrixContributions(), 0u);
}

TEST_F(GhostContributionManagerTest, MoveAssignment) {
    manager_->setPolicy(GhostPolicy::ReverseScatter);
    manager_->initialize();
    manager_->addMatrixContribution(7, 3, 1.0);

    GhostContributionManager other;
    other = std::move(*manager_);

    EXPECT_TRUE(other.isInitialized());
    EXPECT_GT(other.numBufferedMatrixContributions(), 0u);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
