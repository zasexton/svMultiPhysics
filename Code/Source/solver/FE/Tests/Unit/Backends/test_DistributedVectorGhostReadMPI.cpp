/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_DistributedVectorGhostReadMPI.cpp
 * @brief Backend read views expose refreshed overlap values in global numbering.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/BlockVector.h"

#if defined(FE_HAS_PETSC) && FE_HAS_PETSC
#include "Backends/PETSc/PetscVector.h"
#endif

#if defined(FE_HAS_TRILINOS) && FE_HAS_TRILINOS
#include "Backends/Trilinos/TrilinosVector.h"
#endif

#include <mpi.h>

#include <memory>
#include <vector>

namespace svmp::FE::backends {
namespace {

template <typename VectorFactory>
void verifyRefreshedGhostAndBlockReads(VectorFactory&& make_vector)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    const GlobalIndex neighbor =
        static_cast<GlobalIndex>((rank + 1) % size);
    const auto expected_neighbor =
        static_cast<Real>(((rank + 1) % size) + 1);

    auto vector = make_vector(
        std::vector<GlobalIndex>{neighbor});
    ASSERT_NE(vector, nullptr);
    {
        auto local = vector->localSpan();
        ASSERT_GE(local.size(), 1u);
        local[0] = static_cast<Real>(rank + 1);
    }
    vector->updateGhosts();
    auto view = vector->createGhostedReadView();
    ASSERT_NE(view, nullptr);
    EXPECT_NEAR(view->getVectorEntry(neighbor),
                expected_neighbor,
                1e-14);

    std::vector<std::unique_ptr<GenericVector>> blocks;
    blocks.push_back(make_vector(
        std::vector<GlobalIndex>{neighbor}));
    blocks.push_back(make_vector(
        std::vector<GlobalIndex>{neighbor}));
    BlockVector block_vector(std::move(blocks));
    {
        auto first = block_vector.block(0).localSpan();
        auto second = block_vector.block(1).localSpan();
        ASSERT_GE(first.size(), 1u);
        ASSERT_GE(second.size(), 1u);
        first[0] = static_cast<Real>(rank + 1);
        second[0] = static_cast<Real>(rank + 11);
    }
    block_vector.updateGhosts();
    auto block_view = block_vector.createGhostedReadView();
    ASSERT_NE(block_view, nullptr);
    EXPECT_NEAR(block_view->getVectorEntry(neighbor),
                expected_neighbor,
                1e-14);
    EXPECT_NEAR(
        block_view->getVectorEntry(
            static_cast<GlobalIndex>(size) + neighbor),
        expected_neighbor + Real{10.0},
        1e-14);
}

template <typename VectorFactory>
void verifyZeroGhostRankStillServesPeer(VectorFactory&& make_vector)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    // Rank zero imports nothing, but rank one imports rank zero's owned row.
    // The empty rank must still participate in the backend refresh and export.
    std::vector<GlobalIndex> ghosts;
    if (rank == 1) {
        ghosts.push_back(GlobalIndex{0});
    }
    auto vector = make_vector(std::move(ghosts));
    ASSERT_NE(vector, nullptr);
    {
        auto local = vector->localSpan();
        ASSERT_GE(local.size(), 1u);
        local[0] = static_cast<Real>(rank + 7);
    }
    vector->updateGhosts();

    auto view = vector->createGhostedReadView();
    ASSERT_NE(view, nullptr);
    if (rank == 1) {
        EXPECT_NEAR(view->getVectorEntry(GlobalIndex{0}),
                    Real{7.0},
                    1e-14);
    } else {
        EXPECT_NEAR(view->getVectorEntry(
                        static_cast<GlobalIndex>(rank)),
                    static_cast<Real>(rank + 7),
                    1e-14);
    }
    if (rank == 0) {
        EXPECT_THROW(
            static_cast<void>(
                view->getVectorEntry(GlobalIndex{1})),
            FEException);
    }
}

} // namespace

TEST(DistributedVectorGhostReadMPI,
     PetscViewReadsRefreshedGhosts)
{
#if defined(FE_HAS_PETSC) && FE_HAS_PETSC
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    verifyRefreshedGhostAndBlockReads(
        [size](std::vector<GlobalIndex> ghosts)
            -> std::unique_ptr<GenericVector> {
            return std::make_unique<PetscVector>(
                GlobalIndex{1},
                static_cast<GlobalIndex>(size),
                ghosts);
        });
    verifyZeroGhostRankStillServesPeer(
        [size](std::vector<GlobalIndex> ghosts)
            -> std::unique_ptr<GenericVector> {
            return std::make_unique<PetscVector>(
                GlobalIndex{1},
                static_cast<GlobalIndex>(size),
                ghosts);
        });
#else
    GTEST_SKIP() << "FE_HAS_PETSC not enabled";
#endif
}

TEST(DistributedVectorGhostReadMPI,
     TrilinosViewReadsRefreshedGhosts)
{
#if defined(FE_HAS_TRILINOS) && FE_HAS_TRILINOS
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    verifyRefreshedGhostAndBlockReads(
        [rank, size](std::vector<GlobalIndex> ghosts)
            -> std::unique_ptr<GenericVector> {
            return std::make_unique<TrilinosVector>(
                static_cast<GlobalIndex>(rank),
                GlobalIndex{1},
                static_cast<GlobalIndex>(size),
                ghosts);
        });
    verifyZeroGhostRankStillServesPeer(
        [rank, size](std::vector<GlobalIndex> ghosts)
            -> std::unique_ptr<GenericVector> {
            return std::make_unique<TrilinosVector>(
                static_cast<GlobalIndex>(rank),
                GlobalIndex{1},
                static_cast<GlobalIndex>(size),
                ghosts);
        });
    EXPECT_THROW(
        static_cast<void>(TrilinosVector(
            static_cast<GlobalIndex>(rank + 1),
            GlobalIndex{1},
            static_cast<GlobalIndex>(size),
            {})),
        InvalidArgumentException);
#else
    GTEST_SKIP() << "FE_HAS_TRILINOS not enabled";
#endif
}

} // namespace svmp::FE::backends
