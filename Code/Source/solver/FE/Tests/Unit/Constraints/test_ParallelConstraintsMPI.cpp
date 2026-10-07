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

#include <cstring>
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

namespace {

// Rank r owns DOFs [4r, 4r+4) and holds as ghosts the last two DOFs of
// rank r-1 and the first two of rank r+1.  Lines: a two-master line and an
// inhomogeneous Dirichlet line on owned DOFs; a line on the next rank's
// first DOF that only this (ghost-holding) rank declares; and a Dirichlet
// value on the previous rank's last DOF that conflicts with its owner's.
struct RoutedFixture {
    dofs::DofPartition partition;
    AffineConstraints constraints;
};

RoutedFixture makeRoutedFixture(int my_rank, int n_ranks)
{
    const GlobalIndex begin = static_cast<GlobalIndex>(4 * my_rank);
    std::vector<GlobalIndex> ghosts;
    if (my_rank > 0) {
        ghosts.push_back(begin - 2);
        ghosts.push_back(begin - 1);
    }
    if (my_rank + 1 < n_ranks) {
        ghosts.push_back(begin + 4);
        ghosts.push_back(begin + 5);
    }
    RoutedFixture fixture{dofs::DofPartition(begin, begin + 4, ghosts), AffineConstraints{}};
    fixture.partition.setGlobalSize(static_cast<GlobalIndex>(4 * n_ranks));
    auto& c = fixture.constraints;
    c.addLine(begin + 1);
    c.addEntry(begin + 1, begin, 0.5);
    c.addEntry(begin + 1, begin + 2, 0.5);
    c.addDirichlet(begin + 3, static_cast<double>(my_rank) + 0.25);
    if (my_rank + 1 < n_ranks) {
        c.addLine(begin + 4);
        c.addEntry(begin + 4, begin + 3, 2.0);
        c.addEntry(begin + 4, begin + 5, -1.0 / 3.0);
    }
    if (my_rank > 0) {
        c.addDirichlet(begin - 1, 100.0 + my_rank);
    }
    return fixture;
}

bool sameLineBits(const AffineConstraints& a, const AffineConstraints& b, GlobalIndex dof)
{
    const auto la = a.getConstraint(dof);
    const auto lb = b.getConstraint(dof);
    if (la.has_value() != lb.has_value()) {
        return false;
    }
    if (!la.has_value()) {
        return true;
    }
    if (std::memcmp(&la->inhomogeneity, &lb->inhomogeneity, sizeof(double)) != 0 ||
        la->entries.size() != lb->entries.size()) {
        return false;
    }
    for (std::size_t i = 0; i < la->entries.size(); ++i) {
        if (la->entries[i].master_dof != lb->entries[i].master_dof ||
            std::memcmp(&la->entries[i].weight, &lb->entries[i].weight, sizeof(double)) != 0) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST(ParallelConstraintsMPITest, OwnerRoutedResolutionMatchesAllGather)
{
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    if (n_ranks < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    for (const auto strategy : {ParallelConstraintOptions::ConflictResolution::OwnerWins,
                                ParallelConstraintOptions::ConflictResolution::SmallestRank}) {
        SCOPED_TRACE(static_cast<int>(strategy));
        auto gathered = makeRoutedFixture(my_rank, n_ranks);
        auto routed = makeRoutedFixture(my_rank, n_ranks);
        ParallelConstraintOptions opts;
        opts.conflict_resolution = strategy;

        ParallelConstraints all_gather(MPI_COMM_WORLD, gathered.partition);
        all_gather.setOptions(opts);
        ParallelConstraints owner_routed(MPI_COMM_WORLD, routed.partition);
        owner_routed.setOptions(opts);
        owner_routed.setDofOwnerFunction(
            [](GlobalIndex dof) { return static_cast<int>(dof / 4); });

        (void)all_gather.synchronize(gathered.constraints);
        (void)owner_routed.synchronize(routed.constraints);

        for (const auto dof : routed.partition.locallyRelevant()) {
            EXPECT_TRUE(sameLineBits(gathered.constraints, routed.constraints, dof))
                << "dof=" << dof;
        }
        EXPECT_EQ(gathered.constraints.getConstrainedDofs().size(),
                  routed.constraints.getConstrainedDofs().size());
        EXPECT_TRUE(owner_routed.validateConsistency(routed.constraints));
        EXPECT_TRUE(all_gather.validateConsistency(gathered.constraints));

        // The ghost-holder-only line on the next rank's first DOF is
        // installed on its owner.
        const auto first = routed.constraints.getConstraint(static_cast<GlobalIndex>(4 * my_rank));
        EXPECT_EQ(first.has_value(), my_rank > 0);
    }

    // An owner function that disagrees with the partition falls back to the
    // all-gather on every rank instead of failing.
    auto gathered = makeRoutedFixture(my_rank, n_ranks);
    auto routed = makeRoutedFixture(my_rank, n_ranks);
    ParallelConstraints all_gather(MPI_COMM_WORLD, gathered.partition);
    ParallelConstraints wrong_owner(MPI_COMM_WORLD, routed.partition);
    wrong_owner.setDofOwnerFunction([n_ranks](GlobalIndex dof) {
        return static_cast<int>((dof / 4 + 1) % n_ranks);
    });
    (void)all_gather.synchronize(gathered.constraints);
    const auto outcome = invokeCollectively(MPI_COMM_WORLD, [&] {
        (void)wrong_owner.synchronize(routed.constraints);
    });
    EXPECT_EQ(outcome.maximum_threw, 0) << outcome.local_message;
    for (const auto dof : routed.partition.locallyRelevant()) {
        EXPECT_TRUE(sameLineBits(gathered.constraints, routed.constraints, dof)) << "dof=" << dof;
    }
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
