/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>

#include "Sparsity/ParallelSparsity.h"
#include "Sparsity/SparsityBuilder.h"
#include "Dofs/DofMap.h"
#include "Core/FEException.h"

#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::dofs;
using namespace svmp::FE::sparsity;

namespace {

DofMap makeDofMap(GlobalIndex n_dofs, const std::vector<std::vector<GlobalIndex>>& cell_dofs) {
    FE_CHECK_ARG(!cell_dofs.empty(), "cell_dofs must not be empty");

    const GlobalIndex n_cells = static_cast<GlobalIndex>(cell_dofs.size());
    const LocalIndex dofs_per_cell = static_cast<LocalIndex>(cell_dofs.front().size());
    FE_CHECK_ARG(dofs_per_cell > 0, "Each cell must have at least one DOF");

    DofMap map(n_cells, n_dofs, dofs_per_cell);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        map.setCellDofs(c, cell_dofs[static_cast<std::size_t>(c)]);
    }
    map.setNumDofs(n_dofs);
    return map;
}

} // namespace

// ============================================================================
// DofOwnership tests
// ============================================================================

TEST(DofOwnershipTest, BlockDistributionBalanced) {
    DofOwnership own0(/*n_global_dofs=*/10, /*n_ranks=*/3, /*my_rank=*/0);
    DofOwnership own1(/*n_global_dofs=*/10, /*n_ranks=*/3, /*my_rank=*/1);
    DofOwnership own2(/*n_global_dofs=*/10, /*n_ranks=*/3, /*my_rank=*/2);

    EXPECT_EQ(own0.numRanks(), 3);
    EXPECT_EQ(own1.numRanks(), 3);
    EXPECT_EQ(own2.numRanks(), 3);

    EXPECT_EQ(own0.ownedRange().first, 0);
    EXPECT_EQ(own0.ownedRange().last, 4);
    EXPECT_EQ(own1.ownedRange().first, 4);
    EXPECT_EQ(own1.ownedRange().last, 7);
    EXPECT_EQ(own2.ownedRange().first, 7);
    EXPECT_EQ(own2.ownedRange().last, 10);

    EXPECT_EQ(own0.getOwner(0), 0);
    EXPECT_EQ(own0.getOwner(3), 0);
    EXPECT_EQ(own0.getOwner(4), 1);
    EXPECT_EQ(own0.getOwner(7), 2);
}

TEST(DofOwnershipTest, ExplicitOffsets) {
    std::vector<GlobalIndex> offsets = {0, 3, 7, 10};
    DofOwnership own(offsets, /*my_rank=*/1);

    EXPECT_EQ(own.numRanks(), 3);
    EXPECT_EQ(own.globalNumDofs(), 10);
    EXPECT_EQ(own.ownedRange().first, 3);
    EXPECT_EQ(own.ownedRange().last, 7);

    EXPECT_EQ(own.getOwner(0), 0);
    EXPECT_EQ(own.getOwner(2), 0);
    EXPECT_EQ(own.getOwner(3), 1);
    EXPECT_EQ(own.getOwner(6), 1);
    EXPECT_EQ(own.getOwner(7), 2);
}

TEST(DofOwnershipTest, CustomOwnershipInfersRanksAndRange) {
    auto owner = [](GlobalIndex dof) -> int {
        if (dof < 2) return 0;
        if (dof < 5) return 1;
        return 2;
    };

    DofOwnership own(/*n_global_dofs=*/10, owner, /*my_rank=*/1);

    EXPECT_EQ(own.numRanks(), 3);
    EXPECT_FALSE(own.isSerial());
    EXPECT_EQ(own.ownedRange().first, 2);
    EXPECT_EQ(own.ownedRange().last, 5);
    EXPECT_EQ(own.numOwnedDofs(), 3);

    EXPECT_EQ(own.getOwner(0), 0);
    EXPECT_EQ(own.getOwner(3), 1);
    EXPECT_EQ(own.getOwner(9), 2);
}

TEST(DofOwnershipTest, CustomOwnershipNonContiguousThrows) {
    auto owner = [](GlobalIndex dof) -> int {
        return (dof % 2 == 0) ? 0 : 1;
    };

    EXPECT_THROW((DofOwnership(/*n_global_dofs=*/10, owner, /*my_rank=*/0)),
                 InvalidArgumentException);
}

TEST(DofOwnershipTest, GetOwnerOutOfRangeThrows) {
    DofOwnership own(/*n_global_dofs=*/5, /*n_ranks=*/1, /*my_rank=*/0);
    EXPECT_THROW((void)own.getOwner(-1), InvalidArgumentException);
    EXPECT_THROW((void)own.getOwner(5), InvalidArgumentException);
}

// ============================================================================
// DofMapAdapter owned range tests
// ============================================================================

TEST(DofMapAdapterTest, OwnedRangeSerialFastPath) {
    auto map = makeDofMap(6, {{0, 1}, {1, 2}});
    map.setNumLocalDofs(6);
    map.finalize();

    DofMapAdapter adapter(map);
    auto range = adapter.getOwnedRange();
    EXPECT_EQ(range.first, 0);
    EXPECT_EQ(range.second, 6);
}

TEST(DofMapAdapterTest, OwnedRangeContiguousDistributed) {
    auto map = makeDofMap(10, {{0, 4, 5}});
    map.setMyRank(1);
    map.setNumLocalDofs(6);
    map.setDofOwnership([](GlobalIndex dof) -> int { return (dof < 4) ? 0 : 1; });
    map.finalize();

    DofMapAdapter adapter(map);
    auto range = adapter.getOwnedRange();
    EXPECT_EQ(range.first, 4);
    EXPECT_EQ(range.second, 10);
}

TEST(DofMapAdapterTest, OwnedRangeNonContiguousThrows) {
    auto map = makeDofMap(10, {{0, 1}});
    map.setMyRank(0);
    map.setNumLocalDofs(5);
    map.setDofOwnership([](GlobalIndex dof) -> int { return (dof % 2 == 0) ? 0 : 1; });
    map.finalize();

    DofMapAdapter adapter(map);
    EXPECT_THROW((void)adapter.getOwnedRange(), InvalidArgumentException);
}

TEST(DofMapAdapterTest, OwnedRangeLocalCountMismatchThrows) {
    auto map = makeDofMap(10, {{0, 1}});
    map.setMyRank(1);
    map.setNumLocalDofs(5);  // Incorrect: inferred range is size 6
    map.setDofOwnership([](GlobalIndex dof) -> int { return (dof < 4) ? 0 : 1; });
    map.finalize();

    DofMapAdapter adapter(map);
    EXPECT_THROW((void)adapter.getOwnedRange(), InvalidArgumentException);
}

// ============================================================================
// ParallelSparsityManager tests (virtual ranks, no MPI needed)
// ============================================================================

TEST(ParallelSparsityManagerTest, BuildVirtualRanksGhostCols) {
    // Global DOFs: 0..5, rank0 owns [0,3), rank1 owns [3,6)
    auto rank0_map = makeDofMap(6, {{0, 1, 3}, {1, 2, 4}});
    rank0_map.finalize();

    ParallelSparsityManager mgr(/*n_ranks=*/2, /*my_rank=*/0);
    mgr.setBlockOwnership(/*n_global_dofs=*/6);
    mgr.setRowDofMap(rank0_map);

    SparsityBuildOptions opts;
    opts.ensure_diagonal = true;
    mgr.setOptions(opts);

    auto pattern = mgr.build();
    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_TRUE(pattern.validate());

    EXPECT_EQ(pattern.numOwnedRows(), 3);
    EXPECT_EQ(pattern.numGhostCols(), 2);
    auto ghost_cols = pattern.getGhostColMap();
    ASSERT_EQ(ghost_cols.size(), 2);
    EXPECT_EQ(ghost_cols[0], 3);
    EXPECT_EQ(ghost_cols[1], 4);

    EXPECT_TRUE(pattern.hasEntry(0, 3));
    EXPECT_TRUE(pattern.hasEntry(2, 4));
    EXPECT_FALSE(pattern.hasEntry(0, 5));
}

TEST(ParallelSparsityManagerTest, ExcludesGhostColsWhenDisabled) {
    auto rank0_map = makeDofMap(6, {{0, 1, 3}, {1, 2, 4}});
    rank0_map.finalize();

    ParallelSparsityManager mgr(/*n_ranks=*/2, /*my_rank=*/0);
    mgr.setBlockOwnership(/*n_global_dofs=*/6);
    mgr.setRowDofMap(rank0_map);

    SparsityBuildOptions opts;
    opts.ensure_diagonal = true;
    opts.include_ghost_rows = false;
    mgr.setOptions(opts);

    auto pattern = mgr.build();
    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_TRUE(pattern.validate());

    EXPECT_EQ(pattern.numGhostCols(), 0);
    EXPECT_FALSE(pattern.hasEntry(0, 3));
    EXPECT_FALSE(pattern.hasEntry(2, 4));
    EXPECT_TRUE(pattern.hasEntry(0, 0));  // diagonal preserved
}

TEST(ParallelSparsityManagerTest, GhostRowExchangeThrowsWhenEnabled) {
    auto rank0_map = makeDofMap(6, {{0, 1, 3}});
    rank0_map.finalize();

    ParallelSparsityManager mgr(/*n_ranks=*/2, /*my_rank=*/0);
    mgr.setBlockOwnership(/*n_global_dofs=*/6);
    mgr.setRowDofMap(rank0_map);
    mgr.setGhostRowExchange(true);

    EXPECT_THROW(mgr.build(), NotImplementedException);
}

TEST(ParallelSparsityManagerTest, CellCountMismatchThrows) {
    auto row_map = makeDofMap(6, {{0, 1, 3}, {1, 2, 4}});
    auto col_map = makeDofMap(6, {{0, 1, 3}});
    row_map.finalize();
    col_map.finalize();

    ParallelSparsityManager mgr(/*n_ranks=*/2, /*my_rank=*/0);
    mgr.setBlockOwnership(/*n_global_dofs=*/6);
    mgr.setRowDofMap(row_map);
    mgr.setColDofMap(col_map);

    EXPECT_THROW(mgr.build(), InvalidArgumentException);
}
