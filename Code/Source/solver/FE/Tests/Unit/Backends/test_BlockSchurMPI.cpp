/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/BlockMatrix.h"
#include "Backends/Interfaces/BlockVector.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include "Backends/FSILS/FsilsFactory.h"

#include <mpi.h>
#include <vector>
#include <cmath>

namespace svmp::FE::backends {

namespace {

using svmp::FE::sparsity::DistributedSparsityPattern;
using svmp::FE::sparsity::IndexRange;

} // namespace

#if defined(FE_HAS_PETSC) && FE_HAS_PETSC
TEST(PetscBackendMPI, SolveBlockSchur2x2)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        GTEST_SKIP() << "This test requires MPI ranks >= 2";
    }

    // Create a 2x2 Block System:
    // [ A00  A01 ] [ x0 ] = [ b0 ]
    // [ A10  A11 ] [ x1 ]   [ b1 ]
    //
    // For simplicity, let's make it diagonal-dominant to ensure easy convergence with Schur complement.
    // A00: Diagonal, A11: Diagonal. Off-diagonals zero for this basic connectivity test.
    
    constexpr GlobalIndex n_global_sub = 40; // Size of each block
    const GlobalIndex base = n_global_sub / size;
    const GlobalIndex rem = n_global_sub % size;
    const GlobalIndex start = rank * base + std::min<GlobalIndex>(rank, rem);
    const GlobalIndex count = base + ((static_cast<GlobalIndex>(rank) < rem) ? 1 : 0);
    const IndexRange owned = {start, start + count};

    // Shared sparsity pattern for diagonal blocks
    DistributedSparsityPattern pattern(owned, owned, n_global_sub, n_global_sub);
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        pattern.addEntry(row, row);
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    auto factory = BackendFactory::create(BackendKind::PETSc);

    // Create sub-matrices
    auto A00 = factory->createMatrix(pattern);
    auto A01 = factory->createMatrix(pattern); // Will be zero
    auto A10 = factory->createMatrix(pattern); // Will be zero
    auto A11 = factory->createMatrix(pattern);

    // Assemble A00 (Identity * 4)
    auto viewA00 = A00->createAssemblyView();
    viewA00->beginAssemblyPhase();
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        viewA00->addMatrixEntry(row, row, 4.0, assembly::AddMode::Insert);
    }
    viewA00->finalizeAssembly();
    A00->finalizeAssembly();

    // Assemble A11 (Identity * 2)
    auto viewA11 = A11->createAssemblyView();
    viewA11->beginAssemblyPhase();
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        viewA11->addMatrixEntry(row, row, 2.0, assembly::AddMode::Insert);
    }
    viewA11->finalizeAssembly();
    A11->finalizeAssembly();

    // Assemble empty off-diagonals
    A01->createAssemblyView()->beginAssemblyPhase();
    A01->createAssemblyView()->finalizeAssembly();
    A01->finalizeAssembly();
    
    A10->createAssemblyView()->beginAssemblyPhase();
    A10->createAssemblyView()->finalizeAssembly();
    A10->finalizeAssembly();

    // Create Block Matrix
    auto A = factory->createBlockMatrix(2, 2);
    A->setBlock(0, 0, std::move(A00));
    A->setBlock(0, 1, std::move(A01));
    A->setBlock(1, 0, std::move(A10));
    A->setBlock(1, 1, std::move(A11));

    // Create Block Vectors
    auto x = factory->createBlockVector(2);
    x->setBlock(0, factory->createVector(owned.size(), n_global_sub));
    x->setBlock(1, factory->createVector(owned.size(), n_global_sub));

    auto b = factory->createBlockVector(2);
    auto b0 = factory->createVector(owned.size(), n_global_sub);
    auto b1 = factory->createVector(owned.size(), n_global_sub);

    // b0 = 4.0, b1 = 2.0 => expected x0 = 1.0, x1 = 1.0
    auto viewB0 = b0->createAssemblyView();
    viewB0->beginAssemblyPhase();
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        viewB0->addVectorEntry(row, 4.0, assembly::AddMode::Insert);
    }
    viewB0->finalizeAssembly();

    auto viewB1 = b1->createAssemblyView();
    viewB1->beginAssemblyPhase();
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        viewB1->addVectorEntry(row, 2.0, assembly::AddMode::Insert);
    }
    viewB1->finalizeAssembly();

    b->setBlock(0, std::move(b0));
    b->setBlock(1, std::move(b1));

    SolverOptions opts;
    opts.method = SolverMethod::BlockSchur;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 1e-12;
    opts.max_iter = 100;
    
    // FieldSplit options
    opts.fieldsplit.kind = FieldSplitKind::Schur;
    opts.fieldsplit.split_names = {"u", "p"}; // Optional names

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    // Verify
    auto& x0 = x->block(0);
    auto& x1 = x->block(1);
    
    x0.updateGhosts();
    x1.updateGhosts();

    const auto s0 = x0.localSpan();
    const auto s1 = x1.localSpan();

    for (auto val : s0) EXPECT_NEAR(val, 1.0, 1e-8);
    for (auto val : s1) EXPECT_NEAR(val, 1.0, 1e-8);
}
#endif

TEST(FsilsBackendMPI, SolveNSBlockSchur3DOF)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    // 1D chain of nodes, but with 3 DOFs per node to simulate 2D flow (u, v, p).
    // Node 0 (Rank 0) -- Node 1 (Shared) -- Node 2 (Rank 1)
    
    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    // Rank 0 owns Node 0. Rank 1 owns Nodes 1 & 2.
    // (Note: FSILS usually partitions element-wise, but here we define node ownership).
    // Let's stick to the pattern used in other FSILS tests:
    // Rank 0 owns [0..dof-1], Rank 1 owns [dof..3*dof-1].
    
    const IndexRange owned = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    // Add diagonal blocks for all variables
    // And some coupling to make it interesting (but diagonally dominant)
    
    if (rank == 0) {
        // Node 0 coupling to Node 1
        // (Simplification: Just diagonals + intra-node coupling)
        for(int i=0; i<3; ++i) pattern.addEntry(i, i);
        // Coupling to Node 1 (indices 3,4,5)
        for(int i=0; i<3; ++i) pattern.addEntry(i, 3+i); 
    } else {
        // Node 1 (indices 3,4,5) and Node 2 (indices 6,7,8)
        for(int i=3; i<9; ++i) pattern.addEntry(i, i);
        // Node 1 coupling to Node 0 (indices 0,1,2) - stored on Rank 1 as ghost? 
        // No, standard FE assembly: Rank 1 sees Node 1 and Node 2. 
        // If element (0-1) is on Rank 0, Rank 0 adds entries for Node 1.
        // If element (1-2) is on Rank 1, Rank 1 adds entries for Node 1.
        
        // Let's strictly follow the 1D chain overlap logic:
        // Element 0-1 on Rank 0.
        // Element 1-2 on Rank 1.
        
        // Rank 1 owns rows for Node 1 (3,4,5) and Node 2 (6,7,8).
        // It sees contributions for Node 1 from Element 1-2.
        // Rank 0 sees contributions for Node 1 from Element 0-1 (off-process).
        
        // Rank 1 local entries:
        for(int i=3; i<9; ++i) pattern.addEntry(i, i); // Diagonals
        // Coupling 1-2
        for(int i=0; i<3; ++i) {
             pattern.addEntry(3+i, 6+i); // Node 1 -> Node 2
             pattern.addEntry(6+i, 3+i); // Node 2 -> Node 1
        }
    }
    
    pattern.ensureDiagonal();
    pattern.finalize();

    // Ghost info on Rank 0 for Node 1 (indices 3,4,5)
    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 4, 5};
        // Reuse pattern logic from previous tests...
        // Just minimal ghosts to allow assembly
        std::vector<GlobalIndex> ghost_row_ptr{0, 1, 2, 3}; 
        std::vector<GlobalIndex> ghost_cols{3, 4, 5}; // Dummy columns, not used for matrix-mult, just for sparsity placeholder?
        // Actually FSILS sparsity is monolithic.
        // Let's use the helper:
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(dof);
    auto A = factory.createMatrix(pattern);
    auto x = factory.createVector(n_global);
    auto b = factory.createVector(n_global);

    // Assembly
    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();

    // Simple Diagonally Dominant System
    // Diag = 10, Off-diag = -1
    // Matrix per element (size 2*dof = 6)
    
    const int edof = 2 * dof;
    std::vector<Real> Ke(edof * edof, 0.0);
    for(int i=0; i<edof; ++i) {
        Ke[i*edof + i] = 5.0; // Half of 10 (shared nodes sum to 10)
        // Add small coupling
        if (i+dof < edof) {
            Ke[i*edof + (i+dof)] = -1.0;
            Ke[(i+dof)*edof + i] = -1.0;
        }
    }

    if (rank == 0) {
        // Element 0-1 (Indices 0..5)
        std::vector<GlobalIndex> idx(edof);
        for(int i=0; i<edof; ++i) idx[i] = i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    } else {
        // Element 1-2 (Indices 3..8)
        std::vector<GlobalIndex> idx(edof);
        for(int i=0; i<edof; ++i) idx[i] = 3 + i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    // RHS = A * 1 = RowSum
    auto viewB = b->createAssemblyView();
    viewB->beginAssemblyPhase();
    
    // We want x = all ones.
    // Row sums:
    // Node 0 (Rank 0 only): 5 - 1 = 4.
    // Node 1 (Shared): (5-1) + (5-1) = 8.
    // Node 2 (Rank 1 only): 5 - 1 = 4.
    
    std::vector<Real> be(edof);
    // Local contribution to row sum
    for(int i=0; i<edof; ++i) {
        Real sum = 0.0;
        for(int j=0; j<edof; ++j) sum += Ke[i*edof + j];
        be[i] = sum;
    }

    if (rank == 0) {
        std::vector<GlobalIndex> idx(edof);
        for(int i=0; i<edof; ++i) idx[i] = i;
        viewB->addVectorEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(be.data(), be.size()),
                                assembly::AddMode::Add);
    } else {
         std::vector<GlobalIndex> idx(edof);
        for(int i=0; i<edof; ++i) idx[i] = 3 + i;
        viewB->addVectorEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(be.data(), be.size()),
                                assembly::AddMode::Add);
    }
    viewB->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::BlockSchur;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 1e-12;
    opts.max_iter = 50;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    
    // FSILS NS solver can be tricky with small problems, but let's see.
    EXPECT_TRUE(rep.converged);
    
    const auto xs = x->localSpan();
    for (auto val : xs) {
        EXPECT_NEAR(val, 1.0, 1e-6);
    }
}

} // namespace svmp::FE::backends
