/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ParallelConstraintsMPI.cpp
 * @brief MPI unit tests for ParallelConstraints
 */

#include <gtest/gtest.h>

#include "Constraints/AffineConstraints.h"
#include "Constraints/ParallelConstraints.h"
#include "Dofs/DofIndexSet.h"

#include <mpi.h>

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

struct CollectiveOutcome {
    int minimum_threw{0};
    int maximum_threw{0};
    std::string local_message{};

    [[nodiscard]] bool allThrew() const noexcept
    {
        return minimum_threw == 1 && maximum_threw == 1;
    }
};

template <typename Callable>
CollectiveOutcome invokeCollectively(MPI_Comm comm, Callable&& callable)
{
    int local_threw = 0;
    std::string local_message;
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception& error) {
        local_threw = 1;
        local_message = error.what();
    } catch (...) {
        local_threw = 1;
        local_message = "non-std exception";
    }

    CollectiveOutcome outcome;
    outcome.local_message = std::move(local_message);
    MPI_Allreduce(&local_threw,
                  &outcome.minimum_threw,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  comm);
    MPI_Allreduce(&local_threw,
                  &outcome.maximum_threw,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm);
    return outcome;
}

TEST(ParallelConstraintsMPITest, OwnerWinsResolvesGhostConflicts) {
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Global DOF layout: rank r owns [2r, 2r+2)
    const GlobalIndex n_global = static_cast<GlobalIndex>(2 * n_ranks);
    const GlobalIndex owned_begin = static_cast<GlobalIndex>(2 * my_rank);
    const GlobalIndex owned_end = owned_begin + 2;

    std::vector<GlobalIndex> ghosts;
    if (my_rank > 0) {
        // Import previous rank's interface DOFs to test ghost constraint import and conflict resolution.
        ghosts.push_back(owned_begin - 2); // a_prev
        ghosts.push_back(owned_begin - 1); // b_prev (constrained on owner)
    }

    dofs::DofPartition partition(owned_begin, owned_end, ghosts);
    partition.setGlobalSize(n_global);

    AffineConstraints constraints;

    // Owner constraint (always locally owned): b = a
    const GlobalIndex a = owned_begin;
    const GlobalIndex b = owned_begin + 1;
    constraints.addLine(b);
    constraints.addEntry(b, a, 1.0);

    // Ghost-side conflicting constraint for previous rank's b_prev: b_prev = 2 * a_prev.
    // Owner rank (r-1) defines b_prev = 1 * a_prev, so this should be overridden by OwnerWins.
    if (my_rank > 0) {
        const GlobalIndex a_prev = owned_begin - 2;
        const GlobalIndex b_prev = owned_begin - 1;
        constraints.addLine(b_prev);
        constraints.addEntry(b_prev, a_prev, 2.0);
    }

    ParallelConstraints parallel(MPI_COMM_WORLD, partition);
    ParallelConstraintOptions opts;
    opts.conflict_resolution = ParallelConstraintOptions::ConflictResolution::OwnerWins;
    parallel.setOptions(opts);

    const auto stats = parallel.synchronize(constraints);

    // Owned constraint unchanged
    auto owned_line = constraints.getConstraint(b);
    ASSERT_TRUE(owned_line.has_value());
    ASSERT_EQ(owned_line->entries.size(), 1u);
    EXPECT_EQ(owned_line->entries[0].master_dof, a);
    EXPECT_DOUBLE_EQ(owned_line->entries[0].weight, 1.0);

    // Ghost constraint for previous interface DOF imported and resolved to owner definition (weight 1.0)
    if (my_rank > 0) {
        const GlobalIndex a_prev = owned_begin - 2;
        const GlobalIndex b_prev = owned_begin - 1;
        auto ghost_line = constraints.getConstraint(b_prev);
        ASSERT_TRUE(ghost_line.has_value());
        ASSERT_EQ(ghost_line->entries.size(), 1u);
        EXPECT_EQ(ghost_line->entries[0].master_dof, a_prev);
        EXPECT_DOUBLE_EQ(ghost_line->entries[0].weight, 1.0);
    }

    EXPECT_TRUE(parallel.validateConsistency(constraints));

    // Each interface between ranks introduces one conflicting ghost definition.
    EXPECT_EQ(stats.n_conflicts_resolved, static_cast<GlobalIndex>(n_ranks - 1));
    EXPECT_EQ(stats.n_local_constraints, 1);
    EXPECT_EQ(stats.n_ghost_constraints, my_rank > 0 ? 1 : 0);
}

TEST(ParallelConstraintsMPITest, SmallestRankResolvesConflictsEvenAgainstOwner) {
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Each rank owns [2r, 2r+2) and ghosts neighbor ranges.
    const GlobalIndex owned_begin = static_cast<GlobalIndex>(2 * my_rank);
    const GlobalIndex owned_end = owned_begin + 2;
    const GlobalIndex n_global = static_cast<GlobalIndex>(2 * n_ranks);

    std::vector<GlobalIndex> ghosts;
    if (my_rank > 0) {
        ghosts.push_back(owned_begin - 2);
        ghosts.push_back(owned_begin - 1);
    }
    if (my_rank + 1 < n_ranks) {
        ghosts.push_back(owned_end);
        ghosts.push_back(owned_end + 1);
    }

    dofs::DofPartition partition(owned_begin, owned_end, ghosts);
    partition.setGlobalSize(n_global);

    // Create a conflict for DOF b_owned on its owner rank, and a different definition
    // on rank 0 via its ghost copy. SmallestRank should pick rank 0's definition.
    AffineConstraints constraints;

    if (my_rank + 1 < n_ranks) {
        const GlobalIndex a_other = owned_end;
        const GlobalIndex b_other = owned_end + 1;
        constraints.addLine(b_other);
        constraints.addEntry(b_other, a_other, 2.0);  // smaller rank prefers weight 2
    }

    if (my_rank > 0) {
        const GlobalIndex a_owned = owned_begin;
        const GlobalIndex b_owned = owned_begin + 1;
        constraints.addLine(b_owned);
        constraints.addEntry(b_owned, a_owned, 1.0);  // owner defines weight 1
    }

    ParallelConstraints parallel(MPI_COMM_WORLD, partition);
    ParallelConstraintOptions opts;
    opts.conflict_resolution = ParallelConstraintOptions::ConflictResolution::SmallestRank;
    parallel.setOptions(opts);

    parallel.synchronize(constraints);

    if (my_rank > 0) {
        const GlobalIndex a_owned = owned_begin;
        const GlobalIndex b_owned = owned_begin + 1;
        auto line = constraints.getConstraint(b_owned);
        ASSERT_TRUE(line.has_value());
        ASSERT_EQ(line->entries.size(), 1u);
        EXPECT_EQ(line->entries[0].master_dof, a_owned);
        EXPECT_DOUBLE_EQ(line->entries[0].weight, 2.0);
    }
}

TEST(ParallelConstraintsMPITest,
     RankPrivateDecodeFailureIsCoordinated)
{
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const GlobalIndex owned_begin =
        static_cast<GlobalIndex>(my_rank);
    const GlobalIndex owned_end = owned_begin + 1;
    std::vector<GlobalIndex> ghosts;
    if (my_rank != 0) {
        ghosts.push_back(0);
    }
    dofs::DofPartition partition(
        owned_begin, owned_end, ghosts);
    partition.setGlobalSize(
        static_cast<GlobalIndex>(n_ranks));

    // Every rank declares the rank-0 DOF, but nonowners disagree with its
    // owner. Rank 0 rejects the conflict while other ranks would resolve it;
    // the post-gather decode checkpoint must make the failure collective.
    AffineConstraints constraints;
    constraints.addDirichlet(
        0, my_rank == 0 ? 0.0 : 1.0);

    ParallelConstraints parallel(MPI_COMM_WORLD, partition);
    ParallelConstraintOptions options;
    options.conflict_resolution =
        my_rank == 0
            ? ParallelConstraintOptions::ConflictResolution::Error
            : ParallelConstraintOptions::ConflictResolution::OwnerWins;
    parallel.setOptions(options);

    const auto outcome = invokeCollectively(
        MPI_COMM_WORLD,
        [&] { static_cast<void>(parallel.synchronize(constraints)); });
    EXPECT_TRUE(outcome.allThrew());
    if (my_rank == 0) {
        EXPECT_NE(
            outcome.local_message.find(
                "Conflicting constraints from different ranks"),
            std::string::npos);
    } else {
        EXPECT_NE(
            outcome.local_message.find(
                "distributed_parallel_constraint_phase_failure"),
            std::string::npos);
        EXPECT_NE(
            outcome.local_message.find(
                "phase='decode_and_resolve'"),
            std::string::npos);
    }
}

TEST(ParallelConstraintsMPITest,
     ValidationResultIsReducedAcrossRanks)
{
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const GlobalIndex owned_begin =
        static_cast<GlobalIndex>(my_rank);
    const GlobalIndex owned_end = owned_begin + 1;
    std::vector<GlobalIndex> ghosts;
    if (my_rank != 0) {
        ghosts.push_back(0);
    }
    dofs::DofPartition partition(
        owned_begin, owned_end, ghosts);
    partition.setGlobalSize(
        static_cast<GlobalIndex>(n_ranks));

    // OwnerWins makes rank 0's value canonical. Only rank 1 retains a
    // disagreeing ghost value, so a local-only validation result would differ
    // across ranks.
    AffineConstraints constraints;
    constraints.addDirichlet(
        0, my_rank == 1 ? 1.0 : 0.0);

    ParallelConstraints parallel(MPI_COMM_WORLD, partition);
    const bool valid =
        parallel.validateConsistency(constraints);
    const int local_valid = valid ? 1 : 0;
    int minimum_valid = 0;
    int maximum_valid = 0;
    MPI_Allreduce(&local_valid,
                  &minimum_valid,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_valid,
                  &maximum_valid,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  MPI_COMM_WORLD);

    EXPECT_FALSE(valid);
    EXPECT_EQ(minimum_valid, 0);
    EXPECT_EQ(maximum_valid, 0);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
