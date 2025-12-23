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
#include "Dofs/DofMap.h"

#include <mpi.h>

#include <algorithm>
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

std::vector<GlobalIndex> expectedPathRowCols(GlobalIndex row, GlobalIndex n_global) {
    std::vector<GlobalIndex> cols;
    cols.push_back(row);
    if (row > 0) cols.push_back(row - 1);
    if (row + 1 < n_global) cols.push_back(row + 1);
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    return cols;
}

} // namespace

TEST(ParallelSparsityManagerMPITest, GhostColumnsAndGhostRowsExchangePathGraph) {
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Use a simple path graph built from 2-node "elements".
    // Pick a size that yields an equal block distribution: 3 DOFs per rank.
    const GlobalIndex n_global = static_cast<GlobalIndex>(3 * n_ranks);
    const GlobalIndex owned_first = static_cast<GlobalIndex>(3 * my_rank);
    const GlobalIndex owned_last = owned_first + 3;

    std::vector<std::vector<GlobalIndex>> cell_dofs;
    cell_dofs.push_back({owned_first, owned_first + 1});
    cell_dofs.push_back({owned_first + 1, owned_first + 2});
    if (my_rank < n_ranks - 1) {
        // Boundary edge to the next rank (remote row couplings for owned_first+3).
        cell_dofs.push_back({owned_first + 2, owned_first + 3});
    }

    auto dof_map = makeDofMap(n_global, cell_dofs);
    dof_map.finalize();

    ParallelSparsityManager mgr(MPI_COMM_WORLD);
    mgr.setBlockOwnership(n_global);
    mgr.setRowDofMap(dof_map);

    SparsityBuildOptions opts;
    opts.ensure_diagonal = true;
    opts.ensure_non_empty_rows = true;
    opts.include_ghost_rows = true;
    mgr.setOptions(opts);
    mgr.setGhostRowExchange(true);

    auto pattern = mgr.build();
    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_TRUE(pattern.validate());

    // Expected ghost columns for this rank:
    // - from the previous rank's boundary element: owned_first-1 (if rank>0)
    // - from this rank's boundary element: owned_last (if rank<n_ranks-1)
    std::vector<GlobalIndex> expected_ghost_cols;
    if (my_rank > 0) {
        expected_ghost_cols.push_back(owned_first - 1);
    }
    if (my_rank < n_ranks - 1) {
        expected_ghost_cols.push_back(owned_last);
    }

    const auto ghost_map = pattern.getGhostColMap();
    ASSERT_EQ(ghost_map.size(), expected_ghost_cols.size());
    for (std::size_t i = 0; i < expected_ghost_cols.size(); ++i) {
        EXPECT_EQ(ghost_map[i], expected_ghost_cols[i]);
    }

    // Manager communication plan should match the pattern ghost map.
    const auto mgr_ghost_cols = mgr.ghostCols();
    ASSERT_EQ(mgr_ghost_cols.size(), ghost_map.size());
    for (std::size_t i = 0; i < ghost_map.size(); ++i) {
        EXPECT_EQ(mgr_ghost_cols[i], ghost_map[i]);
    }

    if (my_rank > 0) {
        const auto left_owned = mgr.ghostColsOwnedBy(my_rank - 1);
        ASSERT_EQ(left_owned.size(), 1);
        EXPECT_EQ(left_owned[0], owned_first - 1);

        // This entry exists only due to MPI exchange of remote row couplings.
        EXPECT_TRUE(pattern.hasEntry(owned_first, owned_first - 1));
    }

    if (my_rank < n_ranks - 1) {
        const auto right_owned = mgr.ghostColsOwnedBy(my_rank + 1);
        ASSERT_EQ(right_owned.size(), 1);
        EXPECT_EQ(right_owned[0], owned_last);

        // Locally owned row couples to a non-owned column through the boundary element.
        EXPECT_TRUE(pattern.hasEntry(owned_last - 1, owned_last));
    }

    // Owned rows should reflect the path adjacency (with diagonal).
    for (GlobalIndex row = owned_first; row < owned_last; ++row) {
        const auto expected_cols = expectedPathRowCols(row, n_global);
        for (GlobalIndex col : expected_cols) {
            EXPECT_TRUE(pattern.hasEntry(row, col));
        }
    }

    // Ghost-row exchange uses a default policy: request rows matching ghost columns.
    ASSERT_EQ(pattern.numGhostRows(), static_cast<GlobalIndex>(expected_ghost_cols.size()));
    const auto ghost_rows = pattern.getGhostRowMap();
    ASSERT_EQ(ghost_rows.size(), expected_ghost_cols.size());
    for (std::size_t i = 0; i < expected_ghost_cols.size(); ++i) {
        EXPECT_EQ(ghost_rows[i], expected_ghost_cols[i]);
    }

    for (GlobalIndex local = 0; local < pattern.numGhostRows(); ++local) {
        const GlobalIndex global_row = ghost_rows[static_cast<std::size_t>(local)];
        const auto cols_span = pattern.getGhostRowCols(local);
        std::vector<GlobalIndex> cols(cols_span.begin(), cols_span.end());

        const auto expected_cols = expectedPathRowCols(global_row, n_global);
        EXPECT_EQ(cols, expected_cols);
    }
}

