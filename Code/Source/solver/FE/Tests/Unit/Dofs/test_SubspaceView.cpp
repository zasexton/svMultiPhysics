/**
 * @file test_SubspaceView.cpp
 * @brief Unit tests for SubspaceView
 */

#include <gtest/gtest.h>

#include "FE/Dofs/SubspaceView.h"
#include "FE/Core/FEException.h"

#include <vector>

using svmp::FE::FEException;
using svmp::FE::GlobalIndex;
using svmp::FE::dofs::IndexSet;
using svmp::FE::dofs::SubspaceView;

TEST(SubspaceView, DefaultConstruction) {
    SubspaceView view;
    EXPECT_TRUE(view.empty());
    EXPECT_EQ(view.getLocalSize(), 0);
    EXPECT_EQ(view.getGlobalSize(), 0);
    EXPECT_FALSE(view.hasBlockIndex());
}

TEST(SubspaceView, ConstructFromIndexSet) {
    SubspaceView view(IndexSet(std::vector<GlobalIndex>{1, 3, 5, 7}), "TestView", 2);

    EXPECT_FALSE(view.empty());
    EXPECT_EQ(view.getLocalSize(), 4);
    EXPECT_EQ(view.name(), "TestView");
    EXPECT_EQ(view.getBlockIndex(), 2);
    EXPECT_TRUE(view.hasBlockIndex());
    EXPECT_TRUE(view.contains(1));
    EXPECT_TRUE(view.contains(7));
    EXPECT_FALSE(view.contains(2));
}

TEST(SubspaceView, ConstructFromRangeIsContiguous) {
    SubspaceView view(10, 15, "RangeView"); // [10, 15)
    EXPECT_EQ(view.getLocalSize(), 5);
    EXPECT_TRUE(view.isContiguous());

    auto range = view.contiguousRange();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 10);
    EXPECT_EQ(range->second, 15);
}

TEST(SubspaceView, ExtractAndScatter) {
    std::vector<double> full_vec(10, 0.0);
    for (int i = 0; i < 10; ++i) full_vec[static_cast<std::size_t>(i)] = i * 10.0;

    SubspaceView view(IndexSet(std::vector<GlobalIndex>{1, 3, 5}));
    auto sub = view.extractSubvector(full_vec);
    ASSERT_EQ(sub.size(), 3u);
    EXPECT_DOUBLE_EQ(sub[0], 10.0);
    EXPECT_DOUBLE_EQ(sub[1], 30.0);
    EXPECT_DOUBLE_EQ(sub[2], 50.0);

    std::vector<double> target(10, 0.0);
    view.scatterToFull(sub, target);
    EXPECT_DOUBLE_EQ(target[1], 10.0);
    EXPECT_DOUBLE_EQ(target[3], 30.0);
    EXPECT_DOUBLE_EQ(target[5], 50.0);
}

TEST(SubspaceView, LocalGlobalMapping) {
    SubspaceView view(IndexSet(std::vector<GlobalIndex>{30, 10, 20}));

    // Mapping uses sorted order internally.
    EXPECT_EQ(view.localToGlobal(0), 10);
    EXPECT_EQ(view.localToGlobal(1), 20);
    EXPECT_EQ(view.localToGlobal(2), 30);

    EXPECT_EQ(view.globalToLocal(10), 0);
    EXPECT_EQ(view.globalToLocal(20), 1);
    EXPECT_EQ(view.globalToLocal(30), 2);
    EXPECT_EQ(view.globalToLocal(15), -1);

    EXPECT_THROW(view.localToGlobal(3), FEException);
}

TEST(SubspaceView, SetOperations) {
    SubspaceView a(IndexSet(std::vector<GlobalIndex>{1, 2, 3}), "A");
    SubspaceView b(IndexSet(std::vector<GlobalIndex>{3, 4, 5}), "B");

    auto isect = a.intersection_with(b);
    EXPECT_EQ(isect.getLocalSize(), 1);
    EXPECT_TRUE(isect.contains(3));

    auto uni = a.union_with(b);
    EXPECT_EQ(uni.getLocalSize(), 5);
    EXPECT_TRUE(uni.contains(1));
    EXPECT_TRUE(uni.contains(5));

    auto diff = a.difference(b);
    EXPECT_EQ(diff.getLocalSize(), 2);
    EXPECT_TRUE(diff.contains(1));
    EXPECT_TRUE(diff.contains(2));
    EXPECT_FALSE(diff.contains(3));

    auto comp = a.complement(6);
    EXPECT_EQ(comp.getLocalSize(), 3);
    EXPECT_TRUE(comp.contains(0));
    EXPECT_TRUE(comp.contains(4));
    EXPECT_TRUE(comp.contains(5));
}

