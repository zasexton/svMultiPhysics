/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SystemCollectives.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace systems = svmp::FE::systems;

int worldRank()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int worldSize()
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

std::vector<std::uint64_t> sequence(std::size_t count)
{
    std::vector<std::uint64_t> words(count);
    for (std::size_t i = 0; i < count; ++i) {
        words[i] = 0x9e3779b97f4a7c15ull * (static_cast<std::uint64_t>(i) + 1u);
    }
    return words;
}

} // namespace

TEST(SystemCollectivesMPI, IdenticalSequencesAgreeOnEveryRank)
{
    EXPECT_TRUE(systems::collectiveCanonicalWordsAgree(sequence(0), MPI_COMM_WORLD, "empty"));
    EXPECT_TRUE(systems::collectiveCanonicalWordsAgree(sequence(5), MPI_COMM_WORLD, "short"));
    // Longer than one 4096-word comparison chunk.
    EXPECT_TRUE(systems::collectiveCanonicalWordsAgree(sequence(9000), MPI_COMM_WORLD, "long"));
}

TEST(SystemCollectivesMPI, OneDifferingWordFailsOnEveryRank)
{
    if (worldSize() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }
    auto words = sequence(9000);
    if (worldRank() == worldSize() - 1) {
        words[4096 + 7] ^= 1u;  // second chunk
    }
    testing::internal::CaptureStderr();
    const bool agree = systems::collectiveCanonicalWordsAgree(
        words, MPI_COMM_WORLD, "unit context",
        systems::CanonicalWordsDisagreementReport{"[prefix]", "Words disagree", "unit_disagreement"});
    const std::string report = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(agree);
    const std::string expected_head =
        "[prefix] [R" + std::to_string(worldRank()) +
        "] Words disagree diagnostic=unit_disagreement context='unit context'"
        " local_word_count=9000 minimum_word_count=9000 maximum_word_count=9000"
        " differing_words_shown=1 [index=4103 ";
    EXPECT_EQ(report.rfind(expected_head, 0), 0u) << report;
}

TEST(SystemCollectivesMPI, DifferingLengthsFailOnEveryRank)
{
    if (worldSize() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }
    auto words = sequence(12);
    if (worldRank() == 0) {
        words.resize(11);
    }
    testing::internal::CaptureStderr();
    const bool agree = systems::collectiveCanonicalWordsAgree(words, MPI_COMM_WORLD, "lengths");
    const std::string report = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(agree);
    EXPECT_NE(report.find("diagnostic=canonical_words_disagreement context='lengths'"),
              std::string::npos)
        << report;
    EXPECT_NE(report.find(" minimum_word_count=11 maximum_word_count=12"), std::string::npos)
        << report;
}

TEST(SystemCollectivesMPI, SingleRankCommunicatorAgreesWithoutCommunicating)
{
    auto words = sequence(3);
    words[0] += static_cast<std::uint64_t>(worldRank());
    EXPECT_TRUE(systems::collectiveCanonicalWordsAgree(words, MPI_COMM_SELF, "self"));
}
