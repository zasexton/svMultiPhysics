/**
 * @file test_HDivTracePeriodicMPI.cpp
 * @brief MPI regression tests for replicated H(div) trace periodic helpers
 */

#include <gtest/gtest.h>

#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintTools.h"
#include "Spaces/HDivSpace.h"

#include <mpi.h>

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(HDivTracePeriodicMPITest, ReplicatedHelpersProduceDeterministicPairsAcrossRanks)
{
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    spaces::HDivSpace space(ElementType::Quad4, /*order=*/0);

    const std::array<TraceBoundaryEntity, 2> slave_entities = {{
        TraceBoundaryEntity{
            .dofs = {0},
            .vertices = {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}},
            .outward_normal = {{-1.0, 0.0, 0.0}},
        },
        TraceBoundaryEntity{
            .dofs = {1},
            .vertices = {{{0.0, 1.0, 0.0}, {0.0, 2.0, 0.0}}},
            .outward_normal = {{-1.0, 0.0, 0.0}},
        },
    }};

    // Reverse the master-entity container order to ensure matching uses
    // transformed trace geometry rather than container position.
    const std::array<TraceBoundaryEntity, 2> master_entities = {{
        TraceBoundaryEntity{
            .dofs = {11},
            .vertices = {{{1.0, 1.0, 0.0}, {1.0, 2.0, 0.0}}},
            .outward_normal = {{1.0, 0.0, 0.0}},
        },
        TraceBoundaryEntity{
            .dofs = {10},
            .vertices = {{{1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}}},
            .outward_normal = {{1.0, 0.0, 0.0}},
        },
    }};

    const auto pairs = makeHDivTracePeriodicPairsTranslation(
        space,
        slave_entities,
        master_entities,
        {{1.0, 0.0, 0.0}});

    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].slave_dof, 0);
    EXPECT_EQ(pairs[0].master_dof, 10);
    EXPECT_DOUBLE_EQ(pairs[0].weight, -1.0);
    EXPECT_EQ(pairs[1].slave_dof, 1);
    EXPECT_EQ(pairs[1].master_dof, 11);
    EXPECT_DOUBLE_EQ(pairs[1].weight, -1.0);

    auto bc = makeHDivTracePeriodicBCTranslation(
        space,
        slave_entities,
        master_entities,
        {{1.0, 0.0, 0.0}});

    AffineConstraints constraints;
    bc.apply(constraints);
    constraints.close();

    ASSERT_EQ(constraints.numConstraints(), 2u);
    EXPECT_TRUE(constraints.isConstrained(0));
    EXPECT_TRUE(constraints.isConstrained(1));
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
