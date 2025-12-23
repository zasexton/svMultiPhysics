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

#include "Sparsity/SparsityOptimizer.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

namespace {

DistributedSparsityPattern makePoorlyOrderedDistributedPattern(GlobalIndex n, int my_rank, int n_ranks) {
    FE_CHECK_ARG(n > 0, "n must be positive");
    FE_CHECK_ARG(n_ranks > 0, "n_ranks must be positive");
    FE_CHECK_ARG(n % n_ranks == 0, "n must be divisible by n_ranks for this test");

    const GlobalIndex local_n = n / static_cast<GlobalIndex>(n_ranks);
    const IndexRange owned{local_n * static_cast<GlobalIndex>(my_rank),
                           local_n * static_cast<GlobalIndex>(my_rank + 1)};

    DistributedSparsityPattern pattern(owned, owned, n, n);

    // Similar to createPoorlyOrderedPattern() used in serial tests:
    // connect i <-> n-1-i and add local neighbor edges.
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        pattern.addEntry(row, row);

        const GlobalIndex paired = n - 1 - row;
        pattern.addEntry(row, paired);

        if (row > 0) {
            pattern.addEntry(row, row - 1);
        }
        if (row + 1 < n) {
            pattern.addEntry(row, row + 1);
        }
    }

    pattern.finalize();
    return pattern;
}

bool isBijectionOldToNew(const std::vector<GlobalIndex>& perm) {
    std::vector<GlobalIndex> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(sorted.size()); ++i) {
        if (sorted[static_cast<std::size_t>(i)] != i) return false;
    }
    return true;
}

} // namespace

TEST(SparsityOptimizerMPITest, ParMetisNodeNDPermutationValidAndConsistent) {
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    if (!SparsityOptimizer::hasParMetis()) {
        GTEST_SKIP() << "ParMETIS not available in this build";
    }

    const GlobalIndex n = static_cast<GlobalIndex>(16 * n_ranks);
    auto dist_pattern = makePoorlyOrderedDistributedPattern(n, my_rank, n_ranks);

    SparsityOptimizer optimizer;
    const auto perm1 = optimizer.parmetisNodeNDPermutation(dist_pattern, MPI_COMM_WORLD);
    const auto perm2 = optimizer.parmetisNodeNDPermutation(dist_pattern, MPI_COMM_WORLD);

    ASSERT_EQ(perm1.size(), static_cast<std::size_t>(n));
    EXPECT_EQ(perm1, perm2);
    EXPECT_TRUE(isBijectionOldToNew(perm1));

    // Verify that all ranks received the same global permutation vector.
    std::vector<GlobalIndex> root_perm = perm1;
    const int n_int = static_cast<int>(n);
    MPI_Bcast(root_perm.data(), n_int, MPI_INT64_T, 0, MPI_COMM_WORLD);
    EXPECT_EQ(perm1, root_perm);
}

