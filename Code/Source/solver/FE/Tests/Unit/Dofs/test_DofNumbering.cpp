/**
 * @file test_DofNumbering.cpp
 * @brief Unit tests for DofNumbering strategies and utilities
 */

#include <gtest/gtest.h>

#include "FE/Dofs/DofNumbering.h"
#include "FE/Dofs/DofGraph.h"
#include "FE/Dofs/DofMap.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::dofs::applyNumbering;
using svmp::FE::dofs::BlockNumbering;
using svmp::FE::dofs::composePermutations;
using svmp::FE::dofs::CuthillMcKeeNumbering;
using svmp::FE::dofs::DofGraph;
using svmp::FE::dofs::DofMap;
using svmp::FE::dofs::InterleavedNumbering;
using svmp::FE::dofs::invertPermutation;
using svmp::FE::dofs::SequentialNumbering;

namespace {

bool isValidPermutation(const std::vector<GlobalIndex>& perm) {
    const auto n = static_cast<GlobalIndex>(perm.size());
    std::vector<bool> seen(static_cast<std::size_t>(n), false);
    for (auto v : perm) {
        if (v < 0 || v >= n) return false;
        if (seen[static_cast<std::size_t>(v)]) return false;
        seen[static_cast<std::size_t>(v)] = true;
    }
    return true;
}

DofMap makeLineMap(GlobalIndex n_vertices) {
    DofMap map;
    map.reserve(n_vertices - 1, 2);
    for (GlobalIndex c = 0; c < n_vertices - 1; ++c) {
        map.setCellDofs(c, std::vector<GlobalIndex>{c, c + 1});
    }
    map.setNumDofs(n_vertices);
    map.setNumLocalDofs(n_vertices);
    map.finalize();
    return map;
}

} // namespace

TEST(DofNumbering, SequentialIsIdentity) {
    SequentialNumbering strat;
    auto perm = strat.computeNumbering(10, {}, {});
    ASSERT_EQ(perm.size(), 10u);
    EXPECT_TRUE(isValidPermutation(perm));
    for (GlobalIndex i = 0; i < 10; ++i) {
        EXPECT_EQ(perm[static_cast<std::size_t>(i)], i);
    }
}

TEST(DofNumbering, InterleavedFromBlockLayout) {
    // 3 components, 4 nodes per component => 12 DOFs in block ordering
    //   [c0_0,c0_1,c0_2,c0_3, c1_0,c1_1,c1_2,c1_3, c2_0,c2_1,c2_2,c2_3]
    InterleavedNumbering strat(/*n_components=*/3);
    auto perm = strat.computeNumbering(12, {}, {});
    EXPECT_TRUE(isValidPermutation(perm));

    const std::vector<GlobalIndex> expected = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    EXPECT_EQ(perm, expected);
}

TEST(DofNumbering, BlockFromInterleavedLayoutUniformBlocks) {
    // 3 blocks of size 4, in interleaved ordering:
    //   [b0_0,b1_0,b2_0, b0_1,b1_1,b2_1, ...]
    BlockNumbering strat(std::vector<GlobalIndex>{4, 4, 4});
    auto perm = strat.computeNumbering(12, {}, {});
    EXPECT_TRUE(isValidPermutation(perm));

    const std::vector<GlobalIndex> expected = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    EXPECT_EQ(perm, expected);
}

TEST(DofNumbering, InterleavedAndBlockAreInversesForUniformCase) {
    InterleavedNumbering to_interleaved(3);
    BlockNumbering to_block(std::vector<GlobalIndex>{4, 4, 4});

    auto p = to_interleaved.computeNumbering(12, {}, {});
    auto q = to_block.computeNumbering(12, {}, {});

    auto composed = composePermutations(p, q); // apply p then q
    ASSERT_EQ(composed.size(), 12u);
    for (std::size_t i = 0; i < composed.size(); ++i) {
        EXPECT_EQ(composed[i], static_cast<GlobalIndex>(i));
    }
}

TEST(DofNumbering, ApplyNumberingRebuildsMapSafely) {
    DofMap map;
    map.reserve(2, 2);
    map.setCellDofs(0, std::vector<GlobalIndex>{0, 1});
    map.setCellDofs(1, std::vector<GlobalIndex>{1, 2});
    map.setNumDofs(3);
    map.setNumLocalDofs(3);

    // Reverse permutation: 0->2,1->1,2->0
    const std::vector<GlobalIndex> perm = {2, 1, 0};
    applyNumbering(map, perm);

    map.finalize();

    auto c0 = map.getCellDofs(0);
    EXPECT_EQ(std::vector<GlobalIndex>(c0.begin(), c0.end()),
              std::vector<GlobalIndex>({2, 1}));
    auto c1 = map.getCellDofs(1);
    EXPECT_EQ(std::vector<GlobalIndex>(c1.begin(), c1.end()),
              std::vector<GlobalIndex>({1, 0}));
}

TEST(DofNumbering, CuthillMcKeeProducesValidPermutation) {
    auto map = makeLineMap(10);
    DofGraph graph;
    graph.build(map);

    CuthillMcKeeNumbering rcm(/*reverse=*/true);
    auto perm = rcm.computeNumbering(map.getNumDofs(), graph.getAdjOffsets(), graph.getAdjIndices());

    EXPECT_EQ(perm.size(), static_cast<std::size_t>(map.getNumDofs()));
    EXPECT_TRUE(isValidPermutation(perm));

    // Inversion must also be a valid permutation.
    auto inv = invertPermutation(perm);
    EXPECT_TRUE(isValidPermutation(inv));
}

