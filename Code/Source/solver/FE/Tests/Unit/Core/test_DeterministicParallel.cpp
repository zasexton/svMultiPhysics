/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Core/DeterministicParallel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using svmp::FE::deterministicParallelFor;

// Every item runs exactly once and writes its own slot, for any thread count
// and block size.
TEST(DeterministicParallel, EveryItemRunsOnceForAnyThreadCount)
{
    for (const int threads : {1, 2, 3, 4, 8}) {
        for (const std::size_t block : {std::size_t{1}, std::size_t{7},
                                        std::size_t{16}, std::size_t{1000}}) {
            const std::size_t n = 1037u;
            std::vector<int> visits(n, 0);
            std::vector<int> participant_of(n, -1);
            deterministicParallelFor(
                n,
                threads,
                [&](std::size_t item, int participant) {
                    ++visits[item];
                    participant_of[item] = participant;
                },
                block,
                /*min_parallel_items=*/2u);
            for (std::size_t i = 0; i < n; ++i) {
                ASSERT_EQ(visits[i], 1) << threads << " " << block << " " << i;
                ASSERT_GE(participant_of[i], 0);
                ASSERT_LT(participant_of[i], std::max(threads, 1));
            }
            if (threads > 1 && block < n) {
                // Fixed assignment: block b on participant b mod P.
                const std::size_t n_blocks = (n + block - 1) / block;
                const auto participants = static_cast<int>(
                    std::min<std::size_t>(static_cast<std::size_t>(threads),
                                          n_blocks));
                for (std::size_t i = 0; i < n; ++i) {
                    ASSERT_EQ(participant_of[i],
                              static_cast<int>((i / block) %
                                               static_cast<std::size_t>(
                                                   participants)));
                }
            }
        }
    }
}

// The exception of the lowest throwing item is rethrown, as in the serial
// loop.
TEST(DeterministicParallel, RethrowsTheLowestFailingItem)
{
    for (const int threads : {1, 2, 4, 8}) {
        for (const std::size_t first_bad : {std::size_t{0}, std::size_t{5},
                                            std::size_t{63}, std::size_t{400}}) {
            try {
                deterministicParallelFor(
                    500u,
                    threads,
                    [&](std::size_t item, int) {
                        if (item >= first_bad && item % 3u == first_bad % 3u) {
                            throw std::runtime_error(std::to_string(item));
                        }
                    },
                    /*block_size=*/8u,
                    /*min_parallel_items=*/2u);
                FAIL() << "no exception";
            } catch (const std::runtime_error& error) {
                EXPECT_EQ(std::string(error.what()), std::to_string(first_bad))
                    << "threads " << threads;
            }
        }
    }
}

// A loop issued from inside a participant runs serially on that thread.
TEST(DeterministicParallel, NestedLoopsRunSerially)
{
    std::vector<std::vector<int>> inner_participant(64);
    deterministicParallelFor(
        64u,
        4,
        [&](std::size_t outer, int) {
            EXPECT_TRUE(svmp::FE::insideDeterministicParallelRegion());
            deterministicParallelFor(
                100u,
                4,
                [&](std::size_t, int participant) {
                    inner_participant[outer].push_back(participant);
                },
                1u,
                2u);
        },
        1u,
        2u);
    for (const auto& participants : inner_participant) {
        ASSERT_EQ(participants.size(), 100u);
        for (const int p : participants) {
            EXPECT_EQ(p, 0);
        }
    }
    EXPECT_FALSE(svmp::FE::insideDeterministicParallelRegion());
}

TEST(DeterministicParallel, FewItemsRunSeriallyOnTheCaller)
{
    std::vector<int> participant(10, -1);
    deterministicParallelFor(
        10u,
        8,
        [&](std::size_t item, int p) { participant[item] = p; },
        1u,
        /*min_parallel_items=*/64u);
    for (const int p : participant) {
        EXPECT_EQ(p, 0);
    }
}

} // namespace
