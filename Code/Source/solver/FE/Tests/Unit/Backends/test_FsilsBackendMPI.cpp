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
#include "Backends/FSILS/liner_solver/fsils_api.hpp"
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

} // namespace svmp::FE::backends
