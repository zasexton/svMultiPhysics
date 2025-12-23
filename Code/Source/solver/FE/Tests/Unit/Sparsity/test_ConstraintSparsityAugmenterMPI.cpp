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

#include "Sparsity/ConstraintSparsityAugmenter.h"
#include "Sparsity/ParallelSparsity.h"
#include "Dofs/DofMap.h"

#include <mpi.h>

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

TEST(ConstraintSparsityAugmenterMPITest, BuildReducedDistributedPatternDirichlet) {
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const GlobalIndex n_global = static_cast<GlobalIndex>(3 * n_ranks);
    const GlobalIndex owned_first = static_cast<GlobalIndex>(3 * my_rank);
    const GlobalIndex owned_last = owned_first + 3;

    std::vector<std::vector<GlobalIndex>> cell_dofs;
    cell_dofs.push_back({owned_first, owned_first + 1});
    cell_dofs.push_back({owned_first + 1, owned_first + 2});
    if (my_rank < n_ranks - 1) {
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

    const auto full_pattern = mgr.build();
    EXPECT_TRUE(full_pattern.isFinalized());
    EXPECT_TRUE(full_pattern.validate());

    // Constrain the middle DOF on each rank: this removes one vertex per rank
    // but preserves cross-rank couplings between (3r+2) and (3(r+1)).
    auto constraints = std::make_shared<SimpleConstraintSet>();
    for (int r = 0; r < n_ranks; ++r) {
        constraints->addDirichlet(static_cast<GlobalIndex>(3 * r + 1));
    }

    ConstraintSparsityAugmenter augmenter(constraints);
    const auto reduced = augmenter.buildReducedDistributedPattern(full_pattern, MPI_COMM_WORLD);

    const GlobalIndex expected_global_reduced = static_cast<GlobalIndex>(2 * n_ranks);
    EXPECT_EQ(reduced.global_reduced_size, expected_global_reduced);
    EXPECT_EQ(reduced.owned_reduced_range.first, static_cast<GlobalIndex>(2 * my_rank));
    EXPECT_EQ(reduced.owned_reduced_range.last, static_cast<GlobalIndex>(2 * my_rank + 2));

    EXPECT_TRUE(reduced.pattern.isFinalized());
    EXPECT_TRUE(reduced.pattern.validate());
    EXPECT_EQ(reduced.pattern.globalRows(), expected_global_reduced);
    EXPECT_EQ(reduced.pattern.globalCols(), expected_global_reduced);

    ASSERT_EQ(reduced.full_to_reduced_owned.size(), static_cast<std::size_t>(owned_last - owned_first));
    ASSERT_EQ(reduced.reduced_to_full_owned.size(), static_cast<std::size_t>(reduced.owned_reduced_range.size()));

    // Local full->reduced mapping: [3r, 3r+1, 3r+2] -> [2r, -1, 2r+1]
    EXPECT_EQ(reduced.full_to_reduced_owned[0], static_cast<GlobalIndex>(2 * my_rank));
    EXPECT_EQ(reduced.full_to_reduced_owned[1], -1);
    EXPECT_EQ(reduced.full_to_reduced_owned[2], static_cast<GlobalIndex>(2 * my_rank + 1));

    // Local reduced->full mapping: [2r, 2r+1] -> [3r, 3r+2]
    EXPECT_EQ(reduced.reduced_to_full_owned[0], owned_first);
    EXPECT_EQ(reduced.reduced_to_full_owned[1], owned_first + 2);

    // Reduced pattern rows should at least contain diagonal entries.
    for (GlobalIndex row = reduced.owned_reduced_range.first; row < reduced.owned_reduced_range.last; ++row) {
        EXPECT_TRUE(reduced.pattern.hasEntry(row, row));
    }

    // Cross-rank couplings should survive in the reduced system:
    // A_r (2r) <-> B_{r-1} (2r-1) and B_r (2r+1) <-> A_{r+1} (2r+2).
    const GlobalIndex a_r = static_cast<GlobalIndex>(2 * my_rank);
    const GlobalIndex b_r = static_cast<GlobalIndex>(2 * my_rank + 1);
    if (my_rank > 0) {
        EXPECT_TRUE(reduced.pattern.hasEntry(a_r, a_r - 1));
    }
    if (my_rank < n_ranks - 1) {
        EXPECT_TRUE(reduced.pattern.hasEntry(b_r, b_r + 1));
    }
}

