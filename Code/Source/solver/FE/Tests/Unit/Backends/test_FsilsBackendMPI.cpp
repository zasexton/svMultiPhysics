/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Backends/FSILS/liner_solver/bcast.h"
#include "Backends/FSILS/liner_solver/fsils_api.hpp"
#include "Backends/FSILS/liner_solver/norm.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include "Array.h"
#include "Vector.h"

#include <mpi.h>
#include <cmath>
#include <utility>
#include <vector>

namespace svmp::FE::backends {

namespace {

fe_fsi_linear_solver::FSILS_commuType makeWorldCommu()
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    fe_fsi_linear_solver::FSILS_commuType commu{};
    commu.task = rank;
    commu.master = 0;
    commu.masF = (rank == 0) ? 1 : 0;
    commu.nTasks = size;
    commu.comm = MPI_COMM_WORLD;
    return commu;
}

svmp::FE::sparsity::DistributedSparsityPattern makeTwoRankOverlapPatternDof2(int rank)
{
    constexpr GlobalIndex n_global = 6;

    using svmp::FE::sparsity::DistributedSparsityPattern;
    using svmp::FE::sparsity::IndexRange;

    const IndexRange owned = (rank == 0) ? IndexRange{0, 2} : IndexRange{2, 6};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    pattern.setDofIndexing(DistributedSparsityPattern::DofIndexing::NodalInterleaved);

    if (rank == 0) {
        pattern.addEntry(0, 0);
        pattern.addEntry(0, 1);
        pattern.addEntry(0, 2);
        pattern.addEntry(0, 3);
        pattern.addEntry(1, 0);
        pattern.addEntry(1, 1);
        pattern.addEntry(1, 2);
        pattern.addEntry(1, 3);
    } else {
        for (GlobalIndex row = 2; row < 6; ++row) {
            for (GlobalIndex col = 2; col < 6; ++col) {
                pattern.addEntry(row, col);
            }
        }
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{2, 3};
        std::vector<GlobalIndex> ghost_row_ptr{0, 4, 8};
        std::vector<GlobalIndex> ghost_cols{0, 1, 2, 3, 0, 1, 2, 3};
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    return pattern;
}

} // namespace

TEST(FsilsBackendMPI, LegacyBcastAndNormMatchIndependentAllreduce)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        GTEST_SKIP() << "This test requires at least 2 MPI ranks";
    }

    auto commu = makeWorldCommu();

    double scalar = 1.25 + static_cast<double>(rank);
    double scalar_ref = 0.0;
    MPI_Allreduce(&scalar, &scalar_ref, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bcast::fsils_bcast(scalar, commu);
    EXPECT_NEAR(scalar, scalar_ref, 1e-14);

    Vector<double> legacy_vec(3);
    std::vector<double> local_vec(3);
    for (int i = 0; i < 3; ++i) {
        local_vec[static_cast<std::size_t>(i)] =
            static_cast<double>((rank + 1) * (i + 2)) - 0.5 * static_cast<double>(i);
        legacy_vec(i) = local_vec[static_cast<std::size_t>(i)];
    }
    std::vector<double> vec_ref(3, 0.0);
    MPI_Allreduce(local_vec.data(), vec_ref.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bcast::fsils_bcast_v(3, legacy_vec, commu);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(legacy_vec(i), vec_ref[static_cast<std::size_t>(i)], 1e-14);
    }

    std::vector<double> std_vec{local_vec[0] + 0.25, local_vec[1] - 1.0, local_vec[2] + 2.0};
    std::vector<double> std_ref(3, 0.0);
    MPI_Allreduce(std_vec.data(), std_ref.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bcast::fsils_bcast_v(3, std_vec, commu);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(std_vec[static_cast<std::size_t>(i)], std_ref[static_cast<std::size_t>(i)], 1e-14);
    }

    constexpr int dof = 3;
    constexpr int owned_nodes = 2;
    constexpr int local_nodes = 4;
    Array<double> u(dof, local_nodes);
    for (int node = 0; node < local_nodes; ++node) {
        for (int comp = 0; comp < dof; ++comp) {
            u(comp, node) = static_cast<double>(rank + 1) +
                            0.25 * static_cast<double>(node + 1) -
                            0.125 * static_cast<double>(comp + 2);
        }
    }
    for (int node = owned_nodes; node < local_nodes; ++node) {
        for (int comp = 0; comp < dof; ++comp) {
            u(comp, node) = 1.0e6 + 100.0 * rank + 10.0 * node + comp;
        }
    }

    double local_sq = 0.0;
    for (int node = 0; node < owned_nodes; ++node) {
        for (int comp = 0; comp < dof; ++comp) {
            local_sq += u(comp, node) * u(comp, node);
        }
    }
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_NEAR(norm::fsi_ls_normv(dof, owned_nodes, commu, u), std::sqrt(global_sq), 1e-12);
}

TEST(FsilsBackendMPI, SolveCGOverlap1DChain)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    // Global 1D chain with 3 nodes (dof=1):
    // A = [[ 2,-1, 0],
    //      [-1, 4,-1],
    //      [ 0,-1, 2]]
    // Choose b = [1,2,1] => x = [1,1,1].
    constexpr GlobalIndex n_global = 3;

    using svmp::FE::sparsity::DistributedSparsityPattern;
    using svmp::FE::sparsity::IndexRange;

    // Ownership (contiguous by node/DOF):
    // rank0 owns node0; rank1 owns nodes1-2.
    const IndexRange owned = (rank == 0) ? IndexRange{0, 1} : IndexRange{1, 3};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    // Owned row sparsity for local element contributions:
    // rank0 element (0-1), rank1 element (1-2).
    if (rank == 0) {
        pattern.addEntry(0, 0);
        pattern.addEntry(0, 1);
    } else {
        pattern.addEntry(1, 1);
        pattern.addEntry(1, 2);
        pattern.addEntry(2, 1);
        pattern.addEntry(2, 2);
    }

    pattern.ensureDiagonal();
    pattern.finalize();

    // Overlap: rank0 also stores a ghost row for node1 from its local element (0-1).
    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{1};
        std::vector<GlobalIndex> ghost_row_ptr{0, 2};
        std::vector<GlobalIndex> ghost_cols{0, 1};
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(/*dof_per_node=*/1);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);
    auto x = factory.createVector(n_global);

    // Assemble matrix (local contributions only).
    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    if (rank == 0) {
        const GlobalIndex dofs[2] = {0, 1};
        const Real Ke[4] = {2.0, -1.0,
                            -1.0, 2.0};
        viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    } else {
        const GlobalIndex dofs[2] = {1, 2};
        const Real Ke[4] = {2.0, -1.0,
                            -1.0, 2.0};
        viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    // Assemble RHS as overlap contributions:
    // rank0 contributes to nodes 0 and 1, rank1 contributes to nodes 1 and 2.
    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    if (rank == 0) {
        const GlobalIndex dofs[2] = {0, 1};
        const Real be[2] = {1.0, 1.0};
        viewb->addVectorEntries(dofs, be, assembly::AddMode::Add);
    } else {
        const GlobalIndex dofs[2] = {1, 2};
        const Real be[2] = {1.0, 1.0};
        viewb->addVectorEntries(dofs, be, assembly::AddMode::Add);
    }
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0, 1e-10);
    EXPECT_NEAR(xs[1], 1.0, 1e-10);

    // Shared node 1 should match between ranks.
    double shared_val = 0.0;
    if (rank == 0) {
        shared_val = xs[1]; // local nodes are [0,1]
    } else {
        shared_val = xs[0]; // local nodes are [1,2]
    }

    double gathered[2] = {0.0, 0.0};
    MPI_Gather(&shared_val, 1, MPI_DOUBLE, gathered, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        EXPECT_NEAR(gathered[0], gathered[1], 1e-12);
    }
}

TEST(FsilsBackendMPI, SharedFaceReductionUsesOwnedRowHalo)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 2;
    constexpr GlobalIndex n_nodes = 2;
    constexpr GlobalIndex n_global = n_nodes * dof;

    using svmp::FE::sparsity::DistributedSparsityPattern;
    using svmp::FE::sparsity::IndexRange;

    // rank0 owns node 0 and keeps node 1 as a ghost face node; rank1 owns node 1.
    const IndexRange owned = (rank == 0) ? IndexRange{0, 2} : IndexRange{2, 4};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    if (rank == 0) {
        pattern.addEntry(0, 0);
        pattern.addEntry(0, 2);
        pattern.addEntry(1, 1);
        pattern.addEntry(1, 3);
    } else {
        pattern.addEntry(2, 2);
        pattern.addEntry(3, 3);
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{2, 3};
        std::vector<GlobalIndex> ghost_row_ptr{0, 2, 4};
        std::vector<GlobalIndex> ghost_cols{0, 2, 1, 3};
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(/*dof_per_node=*/dof);
    auto A = factory.createMatrix(pattern);
    const auto* fsils = dynamic_cast<const FsilsMatrix*>(A.get());
    ASSERT_NE(fsils, nullptr);

    const auto shared_ptr = fsils->shared();
    ASSERT_NE(shared_ptr, nullptr);
    const auto& shared = *shared_ptr;
    ASSERT_TRUE(shared.lhs.owned_row_operator);
    ASSERT_FALSE(shared.lhs.owned_halo_neighbor_ranks.empty());

    const int internal_face_node = shared.globalNodeToInternal(1);
    ASSERT_GE(internal_face_node, 0);
    ASSERT_LT(internal_face_node, shared.lhs.nNo);

    Vector<int> face_nodes(1);
    face_nodes(0) = internal_face_node;
    Array<double> face_values(dof, 1);
    face_values(0, 0) = (rank == 0) ? 3.0 : 4.5;
    face_values(1, 0) = (rank == 0) ? -1.0 : 2.25;

    fe_fsi_linear_solver::fsils_reduce_shared_face_values_owned_row(
        shared.lhs, dof, face_nodes, face_values);

    EXPECT_NEAR(face_values(0, 0), 7.5, 1e-12);
    EXPECT_NEAR(face_values(1, 0), 1.25, 1e-12);

    Array<double> face_values_m(dof, 1);
    face_values_m(0, 0) = (rank == 0) ? 0.25 : 0.75;
    face_values_m(1, 0) = (rank == 0) ? 10.0 : -3.5;
    fe_fsi_linear_solver::fsils_reduce_shared_face_values_owned_row(
        shared.lhs, dof, face_nodes, face_values_m);

    EXPECT_NEAR(face_values_m(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(face_values_m(1, 0), 6.5, 1e-12);

    Array<double> dirichlet_mask(dof, 1);
    dirichlet_mask(0, 0) = 1.0;
    dirichlet_mask(1, 0) = (rank == 0) ? 1.0 : 0.0;
    fe_fsi_linear_solver::fsils_reduce_shared_face_values_owned_row(
        shared.lhs, dof, face_nodes, dirichlet_mask);
    fe_fsi_linear_solver::fsils_apply_shared_dirichlet_face_mask(
        shared.lhs, dof, face_nodes, dirichlet_mask);

    EXPECT_NEAR(dirichlet_mask(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(dirichlet_mask(1, 0), 0.0, 1e-12);
}

TEST(FsilsBackendMPI, FactoryCreateVectorUsesCachedDistributedOverlapLayout)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 2;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    auto pattern = makeTwoRankOverlapPatternDof2(rank);

    FsilsFactory factory(/*dof_per_node=*/dof);

    auto standalone = factory.createVector(/*local_size=*/1, n_global);
    auto* standalone_fsils = dynamic_cast<FsilsVector*>(standalone.get());
    ASSERT_NE(standalone_fsils, nullptr);
    EXPECT_EQ(standalone_fsils->shared(), nullptr);
    EXPECT_EQ(standalone->size(), n_global);

    auto matrix = factory.createMatrix(pattern);
    const auto* fsils_matrix = dynamic_cast<const FsilsMatrix*>(matrix.get());
    ASSERT_NE(fsils_matrix, nullptr);
    const auto shared = fsils_matrix->shared();
    ASSERT_NE(shared, nullptr);

    const GlobalIndex expected_local_size = static_cast<GlobalIndex>(shared->dof) * shared->lhs.nNo;
    ASSERT_GT(expected_local_size, 0);

    auto overlap = factory.createVector(expected_local_size, n_global);
    auto* overlap_fsils = dynamic_cast<FsilsVector*>(overlap.get());
    ASSERT_NE(overlap_fsils, nullptr);
    EXPECT_EQ(overlap_fsils->shared(), shared.get());
    EXPECT_TRUE(overlap_fsils->usesOwnedRowLayout());
    EXPECT_EQ(overlap->size(), n_global);
    EXPECT_EQ(static_cast<GlobalIndex>(overlap->localSpan().size()), expected_local_size);

    EXPECT_THROW((void)factory.createVector(expected_local_size - 1, n_global), InvalidArgumentException);
    EXPECT_THROW((void)factory.createVector(expected_local_size, n_global + 2), InvalidArgumentException);
}

TEST(FsilsBackendMPI, FactoryPropagatesProvidedCommunicatorHandle)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    MPI_Comm duplicated = MPI_COMM_NULL;
    ASSERT_EQ(MPI_Comm_dup(MPI_COMM_WORLD, &duplicated), MPI_SUCCESS);

    {
        auto pattern = makeTwoRankOverlapPatternDof2(rank);
        FsilsFactory factory(/*dof_per_node=*/2, {}, duplicated);
        auto matrix = factory.createMatrix(pattern);
        auto vector = factory.createVector(/*size=*/6);

        const auto* fsils_matrix = dynamic_cast<const FsilsMatrix*>(matrix.get());
        const auto* fsils_vector = dynamic_cast<const FsilsVector*>(vector.get());
        ASSERT_NE(fsils_matrix, nullptr);
        ASSERT_NE(fsils_vector, nullptr);
        ASSERT_NE(fsils_matrix->shared(), nullptr);
        EXPECT_EQ(fsils_vector->shared(), fsils_matrix->shared().get());

        int compare_dup = MPI_UNEQUAL;
        ASSERT_EQ(MPI_Comm_compare(fsils_matrix->shared()->lhs.commu.comm, duplicated, &compare_dup),
                  MPI_SUCCESS);
        EXPECT_EQ(compare_dup, MPI_IDENT);

        int compare_world = MPI_IDENT;
        ASSERT_EQ(MPI_Comm_compare(fsils_matrix->shared()->lhs.commu.comm, MPI_COMM_WORLD, &compare_world),
                  MPI_SUCCESS);
        EXPECT_NE(compare_world, MPI_IDENT);

        EXPECT_EQ(fsils_matrix->shared()->lhs.commu.task, rank);
        EXPECT_EQ(fsils_matrix->shared()->lhs.commu.nTasks, size);
    }

    ASSERT_EQ(MPI_Comm_free(&duplicated), MPI_SUCCESS);
}

TEST(FsilsBackendMPI, VectorCopyFromRejectsLocalSizeMismatchBetweenStandaloneAndOverlapLayouts)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr GlobalIndex n_global = 6;
    auto pattern = makeTwoRankOverlapPatternDof2(rank);

    FsilsFactory factory(/*dof_per_node=*/2);
    auto standalone = factory.createVector(/*local_size=*/1, n_global);
    auto matrix = factory.createMatrix(pattern);
    const auto* fsils_matrix = dynamic_cast<const FsilsMatrix*>(matrix.get());
    ASSERT_NE(fsils_matrix, nullptr);
    ASSERT_NE(fsils_matrix->shared(), nullptr);

    const GlobalIndex local_size =
        static_cast<GlobalIndex>(fsils_matrix->shared()->dof) *
        static_cast<GlobalIndex>(fsils_matrix->shared()->lhs.nNo);
    auto shared_src = factory.createVector(local_size, n_global);
    auto shared_dst = factory.createVector(local_size, n_global);
    ASSERT_NE(dynamic_cast<FsilsVector*>(standalone.get()), nullptr);

    {
        auto vals = shared_src->localSpan();
        for (std::size_t i = 0; i < vals.size(); ++i) {
            vals[i] = static_cast<Real>(rank * 10 + static_cast<int>(i) + 1);
        }
    }

    EXPECT_NO_THROW(shared_dst->copyFrom(*shared_src));
    {
        const auto src_vals = shared_src->localSpan();
        const auto dst_vals = shared_dst->localSpan();
        ASSERT_EQ(src_vals.size(), dst_vals.size());
        for (std::size_t i = 0; i < src_vals.size(); ++i) {
            EXPECT_DOUBLE_EQ(dst_vals[i], src_vals[i]);
        }
    }

    EXPECT_THROW(shared_dst->copyFrom(*standalone), InvalidArgumentException);
    EXPECT_THROW(standalone->copyFrom(*shared_src), InvalidArgumentException);
}

TEST(FsilsBackendMPI, OwnedFeDofsMatchesDistributedOverlapOwnership)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr GlobalIndex n_global = 6;
    FsilsFactory factory(/*dof_per_node=*/2);
    auto matrix = factory.createMatrix(makeTwoRankOverlapPatternDof2(rank));
    auto vector = factory.createVector(n_global);
    auto* fsils_matrix = dynamic_cast<FsilsMatrix*>(matrix.get());
    auto* fsils_vector = dynamic_cast<FsilsVector*>(vector.get());
    ASSERT_NE(fsils_matrix, nullptr);
    ASSERT_NE(fsils_vector, nullptr);
    ASSERT_NE(fsils_matrix->shared(), nullptr);
    ASSERT_EQ(fsils_vector->shared(), fsils_matrix->shared().get());
    EXPECT_TRUE(fsils_vector->usesOwnedRowLayout());

    const auto owned = fsils_vector->ownedFeDofs();
    if (rank == 0) {
        EXPECT_EQ(owned, (std::vector<GlobalIndex>{0, 1}));
        EXPECT_TRUE(fsils_vector->ownsFeDof(0));
        EXPECT_TRUE(fsils_vector->ownsFeDof(1));
        EXPECT_FALSE(fsils_vector->ownsFeDof(2));
        EXPECT_FALSE(fsils_vector->ownsFeDof(3));
        EXPECT_FALSE(fsils_vector->ownsFeDof(4));
        EXPECT_FALSE(fsils_vector->ownsFeDof(5));
    } else {
        EXPECT_EQ(owned, (std::vector<GlobalIndex>{2, 3, 4, 5}));
        EXPECT_FALSE(fsils_vector->ownsFeDof(0));
        EXPECT_FALSE(fsils_vector->ownsFeDof(1));
        EXPECT_TRUE(fsils_vector->ownsFeDof(2));
        EXPECT_TRUE(fsils_vector->ownsFeDof(3));
        EXPECT_TRUE(fsils_vector->ownsFeDof(4));
        EXPECT_TRUE(fsils_vector->ownsFeDof(5));
    }
}

} // namespace svmp::FE::backends
