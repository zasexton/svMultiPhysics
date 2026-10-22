/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SystemCollectives.h"

#include "Interfaces/FreeSurfaceGeometrySnapshot.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
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

systems::CollectiveCommunicator world()
{
    return systems::CollectiveCommunicator(MPI_COMM_WORLD);
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
    EXPECT_TRUE(systems::collectiveCanonicalWordsAgree(sequence(0), world(), "empty"));
    EXPECT_TRUE(systems::collectiveCanonicalWordsAgree(sequence(5), world(), "short"));
    // Longer than one 4096-word comparison chunk.
    EXPECT_TRUE(systems::collectiveCanonicalWordsAgree(sequence(9000), world(), "long"));
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
        words, world(), "unit context",
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
    const bool agree = systems::collectiveCanonicalWordsAgree(words, world(), "lengths");
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
    EXPECT_TRUE(systems::collectiveCanonicalWordsAgree(
        words, systems::CollectiveCommunicator(MPI_COMM_SELF), "self"));
    EXPECT_TRUE(systems::collectiveCanonicalWordsAgree(words, systems::CollectiveCommunicator{}, "default"));
}

TEST(SystemCollectivesMPI, CommunicatorReadsRankAndSize)
{
    const auto comm = world();
    EXPECT_EQ(comm.rank(), worldRank());
    EXPECT_EQ(comm.size(), worldSize());
    EXPECT_EQ(comm.isParallel(), worldSize() > 1);
    const systems::CollectiveCommunicator self{};
    EXPECT_EQ(self.size(), 1);
    EXPECT_FALSE(self.isParallel());
}

TEST(SystemCollectivesMPI, ReductionsAgreeOnEveryRank)
{
    const auto comm = world();
    const int rank = worldRank();
    const int size = worldSize();
    EXPECT_EQ(systems::collectiveAnyTrue(rank == size - 1, comm), true);
    EXPECT_EQ(systems::collectiveAnyTrue(false, comm), false);
    EXPECT_EQ(systems::collectiveMinDouble(1.5 + rank, comm), 1.5);
    const auto [minimum, maximum] =
        systems::collectiveMinMaxUint64(static_cast<std::uint64_t>(10 + rank), comm);
    EXPECT_EQ(minimum, 10u);
    EXPECT_EQ(maximum, static_cast<std::uint64_t>(10 + size - 1));
}

TEST(SystemCollectivesMPI, PhaseFailureOnOneRankThrowsOnEveryRank)
{
    const auto comm = world();
    EXPECT_NO_THROW(systems::requireCollectivePhase(nullptr, comm, "failed unit phase", "ok"));
    if (worldSize() < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }
    std::exception_ptr local_failure;
    if (worldRank() == 0) {
        local_failure = std::make_exception_ptr(std::invalid_argument("local"));
    }
    if (worldRank() == 0) {
        EXPECT_THROW(systems::requireCollectivePhase(local_failure, comm, "failed unit phase", "p1"),
                     std::invalid_argument);
    } else {
        try {
            systems::requireCollectivePhase(local_failure, comm, "failed unit phase", "p1");
            ADD_FAILURE() << "expected a remote-failure exception";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()),
                      "[svMultiPhysics::Application] Another active FE communicator rank "
                      "failed unit phase 'p1'.");
        }
    }
}

TEST(SystemCollectivesMPI, SizeSumAndPreparationPhaseAgreeOnEveryRank)
{
    const auto comm = world();
    const int rank = worldRank();
    const int size = worldSize();
    // Rank r contributes r + 1.
    EXPECT_EQ(systems::collectiveSumSize(static_cast<std::size_t>(rank + 1), comm),
              static_cast<std::size_t>(size * (size + 1) / 2));
    EXPECT_EQ(systems::collectiveSumSize(7u, systems::CollectiveCommunicator{}), 7u);

    EXPECT_NO_THROW(systems::requireCollectivePhasePreparation(nullptr, comm, "ok"));
    if (size < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }
    std::exception_ptr local_failure;
    if (rank == size - 1) {
        local_failure = std::make_exception_ptr(std::invalid_argument("local"));
    }
    if (rank == size - 1) {
        EXPECT_THROW(systems::requireCollectivePhasePreparation(local_failure, comm, "p2"),
                     std::invalid_argument);
    } else {
        try {
            systems::requireCollectivePhasePreparation(local_failure, comm, "p2");
            ADD_FAILURE() << "expected a remote-failure exception";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()),
                      "[svMultiPhysics::Application] Another active FE communicator rank "
                      "failed local preparation for collective phase 'p2'.");
        }
    }
}

TEST(SystemCollectivesMPI, LocalPhaseOutcomeIsAgreedOnEveryRank)
{
    const auto comm = world();
    const int rank = worldRank();
    const int size = worldSize();
    const std::string_view owner = "UnitOwner";

    EXPECT_TRUE(systems::agreeOnLocalPhase(nullptr, true, comm, "all", owner));
    EXPECT_FALSE(systems::agreeOnLocalPhase(nullptr, rank != size - 1, comm, "one_unavailable", owner));
    EXPECT_FALSE(systems::agreeOnLocalPhase(nullptr, false, systems::CollectiveCommunicator{}, "single", owner));
    EXPECT_THROW((void)systems::agreeOnLocalPhase(
                     std::make_exception_ptr(std::invalid_argument("local")), true,
                     systems::CollectiveCommunicator{}, "single", owner),
                 std::invalid_argument);
    if (size < 2) {
        GTEST_SKIP() << "needs at least two ranks";
    }
    std::exception_ptr local_failure;
    if (rank == 0) {
        local_failure = std::make_exception_ptr(std::invalid_argument("local"));
    }
    if (rank == 0) {
        EXPECT_THROW((void)systems::agreeOnLocalPhase(local_failure, true, comm, "p3", owner),
                     std::invalid_argument);
    } else {
        try {
            (void)systems::agreeOnLocalPhase(local_failure, true, comm, "p3", owner);
            ADD_FAILURE() << "expected a remote-failure exception";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()),
                      "UnitOwner: another communicator rank rejected cut-context callback phase 'p3'");
        }
    }
}

TEST(SystemCollectivesMPI, MaximaAndIndexUnionCoverEveryRank)
{
    const auto comm = world();
    const int rank = worldRank();
    const int size = worldSize();
    EXPECT_EQ(systems::collectiveMaxDouble(0.5 + rank, comm), 0.5 + (size - 1));

    const std::vector<svmp::FE::Real> left = {1.0, 2.0, 3.0};
    std::vector<svmp::FE::Real> right = left;
    right[1] += 0.25 * (rank + 1);
    EXPECT_EQ(systems::collectiveMaxAbsDifference(left, right, comm), 0.25 * size);
    EXPECT_THROW((void)systems::collectiveMaxAbsDifference(
                     left, std::vector<svmp::FE::Real>(2, 0.0), comm),
                 std::runtime_error);

    // Rank r holds {r, r + 1, r}; the union is {0, ..., size}.
    const auto r = static_cast<std::size_t>(rank);
    const auto indices = systems::collectiveIndexUnion(
        {r, r + 1u, r}, static_cast<std::size_t>(size) + 1u, comm, "unit indices");
    ASSERT_EQ(indices.size(), static_cast<std::size_t>(size) + 1u);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        EXPECT_EQ(indices[i], i);
    }
    // One out-of-range index on the last rank throws on every rank.
    std::vector<std::size_t> bad = {0u};
    if (rank == size - 1) {
        bad.push_back(static_cast<std::size_t>(size));
    }
    EXPECT_THROW((void)systems::collectiveIndexUnion(
                     bad, static_cast<std::size_t>(size), comm, "unit indices"),
                 std::runtime_error);
}

TEST(SystemCollectivesMPI, OwnershipCollectiveGathersWordsInRankOrder)
{
    const auto collective = systems::ownershipCollective(world());
    EXPECT_EQ(collective.rank, worldRank());
    EXPECT_EQ(collective.size, worldSize());
    // Rank r contributes r + 1 words, each equal to r.
    const std::vector<std::uint64_t> local(
        static_cast<std::size_t>(worldRank()) + 1u,
        static_cast<std::uint64_t>(worldRank()));
    const auto gathered = collective.all_gather_revision_values(local);
    std::vector<std::uint64_t> expected;
    for (int r = 0; r < worldSize(); ++r) {
        expected.insert(expected.end(), static_cast<std::size_t>(r) + 1u,
                        static_cast<std::uint64_t>(r));
    }
    EXPECT_EQ(gathered, expected);
    EXPECT_EQ(collective.all_gather_owned_rule_identity_values(local), expected);
}
