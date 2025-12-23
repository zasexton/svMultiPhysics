/**
 * @file test_DofIndexSet.cpp
 * @brief Unit tests for IndexSet and DofPartition
 */

#include <gtest/gtest.h>

#include "FE/Dofs/DofIndexSet.h"

#include <algorithm>
#include <numeric>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::dofs::BackendMapHints;
using svmp::FE::dofs::DofPartition;
using svmp::FE::dofs::IndexInterval;
using svmp::FE::dofs::IndexSet;

TEST(IndexSet, DefaultConstruction) {
    IndexSet set;
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);
    EXPECT_FALSE(set.contains(0));
}

TEST(IndexSet, ConstructFromRange) {
    IndexSet set(10, 15); // [10, 15)
    EXPECT_FALSE(set.empty());
    EXPECT_TRUE(set.isContiguous());
    EXPECT_EQ(set.size(), 5);
    EXPECT_TRUE(set.contains(10));
    EXPECT_TRUE(set.contains(14));
    EXPECT_FALSE(set.contains(15));
}

TEST(IndexSet, ConstructFromExplicitIndicesSortsAndDedups) {
    IndexSet set(std::vector<GlobalIndex>{5, 3, 3, 10, 1});
    EXPECT_EQ(set.size(), 4);
    EXPECT_TRUE(set.contains(1));
    EXPECT_TRUE(set.contains(3));
    EXPECT_TRUE(set.contains(5));
    EXPECT_TRUE(set.contains(10));
    EXPECT_FALSE(set.contains(2));
}

TEST(IndexSet, ConstructFromIntervalsMerges) {
    IndexSet set(std::vector<IndexInterval>{{0, 3}, {3, 5}, {10, 12}});
    EXPECT_EQ(set.size(), 7);
    EXPECT_TRUE(set.contains(0));
    EXPECT_TRUE(set.contains(4));
    EXPECT_TRUE(set.contains(10));
    EXPECT_FALSE(set.contains(9));

    ASSERT_TRUE(set.isContiguous() == false); // two disjoint intervals
}

TEST(IndexSet, MinMax) {
    IndexSet set(std::vector<GlobalIndex>{100, 50, 200, 25, 150});
    EXPECT_EQ(set.minIndex(), 25);
    EXPECT_EQ(set.maxIndex(), 200);
}

TEST(IndexSet, ToVectorAndIterationAreSorted) {
    IndexSet set(std::vector<GlobalIndex>{5, 3, 1, 4, 2});
    auto vec = set.toVector();
    ASSERT_EQ(vec.size(), 5u);
    EXPECT_TRUE(std::is_sorted(vec.begin(), vec.end()));

    std::vector<GlobalIndex> iter;
    for (auto v : set) iter.push_back(v);
    EXPECT_EQ(iter, vec);
}

TEST(IndexSet, SetOperations) {
    IndexSet a(std::vector<GlobalIndex>{1, 2, 3, 4});
    IndexSet b(std::vector<GlobalIndex>{3, 4, 5});

    auto u = a.unionWith(b);
    EXPECT_EQ(u.size(), 5);
    for (GlobalIndex i = 1; i <= 5; ++i) {
        EXPECT_TRUE(u.contains(i));
    }

    auto isect = a.intersectionWith(b);
    EXPECT_EQ(isect.size(), 2);
    EXPECT_TRUE(isect.contains(3));
    EXPECT_TRUE(isect.contains(4));

    auto diff = a.difference(b);
    EXPECT_EQ(diff.size(), 2);
    EXPECT_TRUE(diff.contains(1));
    EXPECT_TRUE(diff.contains(2));
    EXPECT_FALSE(diff.contains(3));
}

TEST(IndexSet, AddRemoveReturnNewSets) {
    IndexSet a(0, 3); // {0,1,2}
    auto b = a.add(5);
    EXPECT_TRUE(a.contains(1));
    EXPECT_FALSE(a.contains(5));
    EXPECT_TRUE(b.contains(5));

    auto c = b.remove(1);
    EXPECT_FALSE(c.contains(1));
    EXPECT_TRUE(c.contains(0));
    EXPECT_TRUE(c.contains(2));
}

TEST(IndexSet, BackendMapHints) {
    IndexSet set(10, 20);
    set.setGlobalSize(100);
    set.setOwningRank(2);

    BackendMapHints hints = set.getBackendMapHints();
    EXPECT_TRUE(hints.is_contiguous);
    EXPECT_EQ(hints.range_begin, 10);
    EXPECT_EQ(hints.range_end, 20);
    EXPECT_EQ(hints.global_size, 100);
    EXPECT_EQ(hints.owning_rank, 2);
}

TEST(DofPartition, OwnedGhostRelevantSets) {
    IndexSet owned(0, 5); // {0..4}
    IndexSet ghost(std::vector<GlobalIndex>{10, 11});
    DofPartition part(owned, ghost);
    part.setGlobalSize(20);

    EXPECT_EQ(part.globalSize(), 20);
    EXPECT_EQ(part.localOwnedSize(), 5);
    EXPECT_EQ(part.ghostSize(), 2);
    EXPECT_EQ(part.localRelevantSize(), 7);

    EXPECT_TRUE(part.isOwned(0));
    EXPECT_FALSE(part.isOwned(10));

    EXPECT_TRUE(part.isGhost(10));
    EXPECT_FALSE(part.isGhost(4));

    EXPECT_TRUE(part.isRelevant(11));
    EXPECT_TRUE(part.isRelevant(2));
    EXPECT_FALSE(part.isRelevant(19));
}

