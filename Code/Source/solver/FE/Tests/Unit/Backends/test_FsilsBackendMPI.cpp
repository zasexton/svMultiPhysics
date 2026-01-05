/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include <mpi.h>
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

} // namespace svmp::FE::backends
