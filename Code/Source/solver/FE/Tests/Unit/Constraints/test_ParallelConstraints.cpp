/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ParallelConstraints.cpp
 * @brief Unit tests for ParallelConstraints (serial behavior)
 */

#include <gtest/gtest.h>

#include "Constraints/AffineConstraints.h"
#include "Constraints/ParallelConstraints.h"

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(ParallelConstraintsTest, SerialNoOpButStatsReported) {
    AffineConstraints constraints;
    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.setInhomogeneity(0, 2.0);

    ParallelConstraints parallel;
    EXPECT_FALSE(parallel.isParallel());

    auto stats = parallel.makeConsistent(constraints);
    EXPECT_EQ(stats.n_local_constraints, 1);
    EXPECT_EQ(stats.n_ghost_constraints, 0);

    auto stats2 = parallel.importGhostConstraints(constraints);
    EXPECT_EQ(stats2.n_messages_sent, 0);
    EXPECT_EQ(stats2.n_messages_received, 0);

    EXPECT_TRUE(parallel.validateConsistency(constraints));
}

TEST(ParallelConstraintsTest, ExportConstraintsExtractsRequestedLines) {
    AffineConstraints constraints;

    constraints.addLine(0);
    constraints.addEntry(0, 1, 1.0);
    constraints.setInhomogeneity(0, 2.0);

    constraints.addDirichlet(5, 7.0);

    ParallelConstraints parallel;
    std::array<GlobalIndex, 3> requested{{0, 2, 5}};
    auto exported = parallel.exportConstraints(constraints, requested);

    ASSERT_EQ(exported.size(), 2u);

    // Deterministic order follows requested order
    EXPECT_EQ(exported[0].slave_dof, 0);
    EXPECT_DOUBLE_EQ(exported[0].inhomogeneity, 2.0);
    ASSERT_EQ(exported[0].entries.size(), 1u);
    EXPECT_EQ(exported[0].entries[0].master_dof, 1);
    EXPECT_DOUBLE_EQ(exported[0].entries[0].weight, 1.0);

    EXPECT_EQ(exported[1].slave_dof, 5);
    EXPECT_DOUBLE_EQ(exported[1].inhomogeneity, 7.0);
    EXPECT_TRUE(exported[1].entries.empty());
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp

