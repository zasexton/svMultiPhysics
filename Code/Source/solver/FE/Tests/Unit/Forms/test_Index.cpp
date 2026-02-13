/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_Index.cpp
 * @brief Unit tests for FE/Forms Index and IndexSet
 */

#include <gtest/gtest.h>

#include "Forms/Index.h"

#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(IndexTest, IndexSetDefaultExtent)
{
    IndexSet set;
    EXPECT_EQ(set.extent(), 0);
}

TEST(IndexTest, IndexSetCustomExtent)
{
    EXPECT_EQ(IndexSet(2).extent(), 2);
    EXPECT_EQ(IndexSet(4).extent(), 4);
}

TEST(IndexTest, IndexDefaultConstruction)
{
    const Index a;
    const Index b;
    EXPECT_TRUE(a.name().empty());
    EXPECT_TRUE(b.name().empty());
    EXPECT_NE(a.id(), b.id());
}

TEST(IndexTest, IndexNamedConstruction)
{
    const Index i("i");
    EXPECT_EQ(i.name(), "i");
    EXPECT_EQ(i.extent(), 0);
}

TEST(IndexTest, IndexUniqueIds)
{
    std::unordered_set<int> ids;
    for (int k = 0; k < 50; ++k) {
        ids.insert(Index().id());
    }
    EXPECT_EQ(ids.size(), 50u);
}

TEST(IndexTest, IndexExtentFromSet)
{
    const Index i("i", IndexSet(4));
    EXPECT_EQ(i.name(), "i");
    EXPECT_EQ(i.extent(), 4);
}

TEST(IndexTest, IndexIdThreadSafety)
{
    constexpr int kThreads = 4;
    constexpr int kPerThread = 500;

    std::mutex m;
    std::vector<int> ids;
    ids.reserve(static_cast<std::size_t>(kThreads * kPerThread));

    auto worker = [&]() {
        std::vector<int> local;
        local.reserve(static_cast<std::size_t>(kPerThread));
        for (int i = 0; i < kPerThread; ++i) {
            local.push_back(Index().id());
        }
        std::lock_guard<std::mutex> lock(m);
        ids.insert(ids.end(), local.begin(), local.end());
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }

    std::unordered_set<int> unique(ids.begin(), ids.end());
    EXPECT_EQ(unique.size(), ids.size());
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
