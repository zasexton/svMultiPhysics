/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsVector.h"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace svmp::FE::backends {

namespace {

using svmp::FE::sparsity::DistributedSparsityPattern;
using svmp::FE::sparsity::IndexRange;

[[maybe_unused]] [[nodiscard]] IndexRange blockOwnedRange(GlobalIndex n_global, int comm_size, int rank)
{
    const GlobalIndex base = n_global / comm_size;
    const GlobalIndex rem = n_global % comm_size;
    const GlobalIndex start = rank * base + std::min<GlobalIndex>(rank, rem);
    const GlobalIndex count = base + ((static_cast<GlobalIndex>(rank) < rem) ? 1 : 0);
    return IndexRange{start, start + count};
}

[[maybe_unused]] [[nodiscard]] Real relativeResidualDistributed(const BackendFactory& factory,
                                               const GenericMatrix& A,
                                               const GenericVector& x,
                                               const GenericVector& b,
                                               GlobalIndex local_size,
                                               GlobalIndex global_size)
{
    auto Ax = factory.createVector(local_size, global_size);
    Ax->zero();
    A.mult(x, *Ax);

    auto r = factory.createVector(local_size, global_size);
    r->zero();
    auto rs = r->localSpan();
    const auto bs = b.localSpan();
    const auto axs = Ax->localSpan();
    FE_THROW_IF(rs.size() != bs.size() || rs.size() != axs.size(), FEException, "residual size mismatch");
    for (std::size_t i = 0; i < rs.size(); ++i) {
        rs[i] = bs[i] - axs[i];
    }

    const Real rn = r->norm();
    const Real bn = b.norm();
    return rn / std::max<Real>(bn, 1e-30);
}

} // namespace

TEST(FsilsBackendMPI, DotAndNormUseOwnedNodesOnly)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    // Minimal overlap layout (same as SolveCGOverlap1DChain):
    // rank0 owns node0, ghosts node1; rank1 owns nodes1-2.
    constexpr GlobalIndex n_global = 3;
    const IndexRange owned = (rank == 0) ? IndexRange{0, 1} : IndexRange{1, 3};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

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

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{1};
        std::vector<GlobalIndex> ghost_row_ptr{0, 2};
        std::vector<GlobalIndex> ghost_cols{0, 1};
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(/*dof_per_node=*/1);
    auto A = factory.createMatrix(pattern);
    auto v = factory.createVector(n_global);

    // Fill v with global node id + 1.0 on owned+ghost nodes.
    auto* fv = dynamic_cast<FsilsVector*>(v.get());
    ASSERT_TRUE(fv);
    const auto* shared = fv->shared();
    ASSERT_TRUE(shared);
    const auto& lhs = shared->lhs;
    const int nNo = lhs.nNo;
    const int mynNo = lhs.mynNo;

    auto vs = v->localSpan();
    ASSERT_EQ(static_cast<int>(vs.size()), nNo);
    for (int old = 0; old < nNo; ++old) {
        const int global_node = (old < shared->owned_node_count)
                                    ? (shared->owned_node_start + old)
                                    : shared->ghost_nodes[static_cast<std::size_t>(old - shared->owned_node_count)];
        vs[static_cast<std::size_t>(old)] = static_cast<Real>(global_node + 1);
    }

    const Real vnorm = v->norm();
    const Real vdot = v->dot(*v);

    // Manual owned-only reduction matching FsilsVector::dot implementation.
    Real local_sum = 0.0;
    for (int old = 0; old < nNo; ++old) {
        if (lhs.map(old) >= mynNo) continue;
        local_sum += vs[static_cast<std::size_t>(old)] * vs[static_cast<std::size_t>(old)];
    }
    Real global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    EXPECT_NEAR(vdot, global_sum, 1e-14);
    EXPECT_NEAR(vnorm, std::sqrt(global_sum), 1e-14);
}

TEST(FsilsBackendMPI, SolveCGOverlap1DChainDof2)
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

    // rank0 owns node0; rank1 owns nodes1-2.
    const IndexRange owned = (rank == 0) ? IndexRange{0, 2} : IndexRange{2, 6};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    // Owned row sparsity: tri-diagonal per component.
    if (rank == 0) {
        // Element (0-1) assembled on rank0: owned node0 rows couple to node0 and node1.
        pattern.addEntry(0, 0);
        pattern.addEntry(0, 2);
        pattern.addEntry(1, 1);
        pattern.addEntry(1, 3);
    } else {
        // Element (1-2) assembled on rank1: rows couple only within {node1,node2}.
        pattern.addEntry(2, 2);
        pattern.addEntry(2, 4);
        pattern.addEntry(3, 3);
        pattern.addEntry(3, 5);
        pattern.addEntry(4, 2);
        pattern.addEntry(4, 4);
        pattern.addEntry(5, 3);
        pattern.addEntry(5, 5);
    }

    pattern.ensureDiagonal();
    pattern.finalize();

    // Overlap: rank0 stores ghost rows for node1 (both components).
    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{2, 3};
        std::vector<GlobalIndex> ghost_row_ptr{0, 2, 4};
        std::vector<GlobalIndex> ghost_cols{0, 2, 1, 3};
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(/*dof_per_node=*/dof);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);
    auto x = factory.createVector(n_global);

    // Assemble 1D element stiffness per component.
    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const Real Ke[4] = {2.0, -1.0,
                        -1.0, 2.0};

    if (rank == 0) {
        // element (0-1)
        const GlobalIndex e0c0[2] = {0, 2};
        const GlobalIndex e0c1[2] = {1, 3};
        viewA->addMatrixEntries(e0c0, Ke, assembly::AddMode::Add);
        viewA->addMatrixEntries(e0c1, Ke, assembly::AddMode::Add);
    } else {
        // element (1-2)
        const GlobalIndex e1c0[2] = {2, 4};
        const GlobalIndex e1c1[2] = {3, 5};
        viewA->addMatrixEntries(e1c0, Ke, assembly::AddMode::Add);
        viewA->addMatrixEntries(e1c1, Ke, assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    // Assemble overlap RHS so node1 contributions sum across ranks.
    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    if (rank == 0) {
        const GlobalIndex dofs0[4] = {0, 1, 2, 3};
        const Real vals0[4] = {1.0, 1.0, 1.0, 1.0};
        viewb->addVectorEntries(dofs0, vals0, assembly::AddMode::Add);
    } else {
        const GlobalIndex dofs1[4] = {2, 3, 4, 5};
        const Real vals1[4] = {1.0, 1.0, 1.0, 1.0};
        viewb->addVectorEntries(dofs1, vals1, assembly::AddMode::Add);
    }
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 400;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    // Local vectors contain (owned + ghost) nodes; in this layout both ranks have 2 nodes locally.
    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 4u);
    for (const auto v : xs) {
        EXPECT_NEAR(v, 1.0, 1e-10);
    }

    // Shared node1 components should match across ranks.
    double shared_vals[2] = {0.0, 0.0};
    if (rank == 0) {
        shared_vals[0] = xs[2]; // node1 comp0
        shared_vals[1] = xs[3]; // node1 comp1
    } else {
        shared_vals[0] = xs[0]; // node1 comp0
        shared_vals[1] = xs[1]; // node1 comp1
    }

    double gathered[4] = {0.0, 0.0, 0.0, 0.0};
    MPI_Gather(shared_vals, 2, MPI_DOUBLE, gathered, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        EXPECT_NEAR(gathered[0], gathered[2], 1e-12);
        EXPECT_NEAR(gathered[1], gathered[3], 1e-12);
    }
}

TEST(FsilsBackendMPI, UpdateGhostsCopiesOwnedToGhost)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    // Minimal overlap layout:
    // - rank0 owns node0 and stores node1 as a ghost (overlap) node
    // - rank1 owns nodes1-2
    constexpr GlobalIndex n_global = 3;
    const IndexRange owned = (rank == 0) ? IndexRange{0, 1} : IndexRange{1, 3};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

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

    // Provide ghost row closure for the overlap node on rank0.
    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{1};
        std::vector<GlobalIndex> ghost_row_ptr{0, 2};
        std::vector<GlobalIndex> ghost_cols{0, 1};
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(/*dof_per_node=*/1);
    auto A = factory.createMatrix(pattern);
    (void)A;
    auto v = factory.createVector(n_global);

    auto* fv = dynamic_cast<FsilsVector*>(v.get());
    ASSERT_TRUE(fv);
    const auto* shared = fv->shared();
    ASSERT_TRUE(shared);

    auto vs = v->localSpan();
    ASSERT_EQ(vs.size(), static_cast<std::size_t>(shared->localNodeCount()));
    std::fill(vs.begin(), vs.end(), 0.0);

    constexpr Real owned_value = 123.0;
    if (rank == 1) {
        // node1 is owned on rank1 (old index 0).
        ASSERT_GE(shared->owned_node_count, 1);
        vs[0] = owned_value;
    }

    v->updateGhosts();

    const GenericVector& cv = *v;
    const auto cvs = cv.localSpan();

    if (rank == 0) {
        // node1 is ghost on rank0 (old index = owned_count + 0).
        ASSERT_EQ(shared->owned_node_count, 1);
        ASSERT_EQ(shared->ghost_nodes.size(), 1u);
        EXPECT_EQ(shared->ghost_nodes[0], 1);
        ASSERT_EQ(cvs.size(), 2u);
        EXPECT_NEAR(cvs[1], owned_value, 1e-14);
    }
}

#if defined(FE_HAS_PETSC) && FE_HAS_PETSC
TEST(PetscBackendMPI, SolveCG1DPoissonOwnedRows)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        GTEST_SKIP() << "This test requires MPI ranks >= 2";
    }

    constexpr GlobalIndex n_global = 80;
    const IndexRange owned = blockOwnedRange(n_global, size, rank);

    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        pattern.addEntry(row, row);
        if (row > 0) pattern.addEntry(row, row - 1);
        if (row + 1 < n_global) pattern.addEntry(row, row + 1);
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    auto factory = BackendFactory::create(BackendKind::PETSc);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(owned.size(), n_global);
    auto x = factory->createVector(owned.size(), n_global);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        const Real diag = (row == 0 || row + 1 == n_global) ? 2.0 : 4.0;
        viewA->addMatrixEntry(row, row, diag, assembly::AddMode::Insert);
        if (row > 0) viewA->addMatrixEntry(row, row - 1, -1.0, assembly::AddMode::Insert);
        if (row + 1 < n_global) viewA->addMatrixEntry(row, row + 1, -1.0, assembly::AddMode::Insert);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        const Real v = (row == 0 || row + 1 == n_global) ? 1.0 : 2.0; // A*1
        viewb->addVectorEntry(row, v, assembly::AddMode::Insert);
    }
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 0.0;
    opts.max_iter = 4000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    x->updateGhosts();
    const GenericVector& cx = *x;
    const auto xs = cx.localSpan();
    ASSERT_EQ(xs.size(), static_cast<std::size_t>(owned.size() + pattern.numGhostCols()));
    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(xs[i], 1.0, 1e-8);
    }

    const Real rel = relativeResidualDistributed(*factory, *A, *x, *b, owned.size(), n_global);
    EXPECT_LE(rel, opts.rel_tol * 50.0 + 1e-12);
}

TEST(PetscBackendMPI, UpdateGhostsCopiesOwnedToGhost)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr GlobalIndex n_global = 3;
    const IndexRange owned = (rank == 0) ? IndexRange{0, 1} : IndexRange{1, 3};

    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        pattern.addEntry(row, row);
        if (row > 0) pattern.addEntry(row, row - 1);
        if (row + 1 < n_global) pattern.addEntry(row, row + 1);
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    auto factory = BackendFactory::create(BackendKind::PETSc);
    auto A = factory->createMatrix(pattern);
    (void)A;
    auto v = factory->createVector(owned.size(), n_global);

    auto vs = v->localSpan();
    std::fill(vs.begin(), vs.end(), 0.0);

    constexpr Real owned_value = 123.0;
    if (rank == 1) {
        // gid=1 is owned on rank1 (local index 0).
        ASSERT_GE(vs.size(), 1u);
        vs[0] = owned_value;
    }

    v->updateGhosts();

    const GenericVector& cv = *v;
    const auto cvs = cv.localSpan();
    ASSERT_EQ(cvs.size(), static_cast<std::size_t>(owned.size() + pattern.numGhostCols()));

    if (rank == 0) {
        const auto ghosts = pattern.getGhostColMap();
        const auto it = std::find(ghosts.begin(), ghosts.end(), 1);
        ASSERT_NE(it, ghosts.end());
        const std::size_t ghost_pos = static_cast<std::size_t>(it - ghosts.begin());
        const std::size_t local_idx = static_cast<std::size_t>(owned.size()) + ghost_pos;
        ASSERT_LT(local_idx, cvs.size());
        EXPECT_NEAR(cvs[local_idx], owned_value, 1e-14);
    }
}

TEST(PetscBackendMPI, DotAndNormMatchManualReduction)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        GTEST_SKIP() << "This test requires MPI ranks >= 2";
    }

    constexpr GlobalIndex n_global = 101;
    const IndexRange owned = blockOwnedRange(n_global, size, rank);

    auto factory = BackendFactory::create(BackendKind::PETSc);
    auto v = factory->createVector(owned.size(), n_global);

    auto vs = v->localSpan();
    for (std::size_t i = 0; i < vs.size(); ++i) {
        const GlobalIndex gid = owned.first + static_cast<GlobalIndex>(i);
        vs[i] = static_cast<Real>(gid + 1);
    }

    const Real dot = v->dot(*v);
    const Real norm = v->norm();

    Real local_sum = 0.0;
    for (const auto val : vs) local_sum += val * val;
    Real global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    EXPECT_NEAR(dot, global_sum, 1e-10);
    EXPECT_NEAR(norm, std::sqrt(global_sum), 1e-10);
}
#endif

#if defined(FE_HAS_TRILINOS) && FE_HAS_TRILINOS
TEST(TrilinosBackendMPI, SolveCG1DPoissonOwnedRows)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        GTEST_SKIP() << "This test requires MPI ranks >= 2";
    }

    constexpr GlobalIndex n_global = 80;
    const IndexRange owned = blockOwnedRange(n_global, size, rank);

    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        pattern.addEntry(row, row);
        if (row > 0) pattern.addEntry(row, row - 1);
        if (row + 1 < n_global) pattern.addEntry(row, row + 1);
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    auto factory = BackendFactory::create(BackendKind::Trilinos);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(owned.size(), n_global);
    auto x = factory->createVector(owned.size(), n_global);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        const Real diag = (row == 0 || row + 1 == n_global) ? 2.0 : 4.0;
        viewA->addMatrixEntry(row, row, diag, assembly::AddMode::Insert);
        if (row > 0) viewA->addMatrixEntry(row, row - 1, -1.0, assembly::AddMode::Insert);
        if (row + 1 < n_global) viewA->addMatrixEntry(row, row + 1, -1.0, assembly::AddMode::Insert);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        const Real v = (row == 0 || row + 1 == n_global) ? 1.0 : 2.0;
        viewb->addVectorEntry(row, v, assembly::AddMode::Insert);
    }
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 0.0;
    opts.max_iter = 4000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    x->updateGhosts();
    const GenericVector& cx = *x;
    const auto xs = cx.localSpan();
    ASSERT_EQ(xs.size(), static_cast<std::size_t>(owned.size() + pattern.numGhostCols()));
    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(xs[i], 1.0, 1e-8);
    }

    const Real rel = relativeResidualDistributed(*factory, *A, *x, *b, owned.size(), n_global);
    EXPECT_LE(rel, opts.rel_tol * 50.0 + 1e-12);
}

TEST(TrilinosBackendMPI, UpdateGhostsCopiesOwnedToGhost)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr GlobalIndex n_global = 3;
    const IndexRange owned = (rank == 0) ? IndexRange{0, 1} : IndexRange{1, 3};

    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        pattern.addEntry(row, row);
        if (row > 0) pattern.addEntry(row, row - 1);
        if (row + 1 < n_global) pattern.addEntry(row, row + 1);
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    auto factory = BackendFactory::create(BackendKind::Trilinos);
    auto A = factory->createMatrix(pattern);
    (void)A;
    auto v = factory->createVector(owned.size(), n_global);

    auto vs = v->localSpan();
    std::fill(vs.begin(), vs.end(), 0.0);

    constexpr Real owned_value = 123.0;
    if (rank == 1) {
        ASSERT_GE(vs.size(), 1u);
        vs[0] = owned_value;
    }

    v->updateGhosts();

    const GenericVector& cv = *v;
    const auto cvs = cv.localSpan();
    ASSERT_EQ(cvs.size(), static_cast<std::size_t>(owned.size() + pattern.numGhostCols()));

    if (rank == 0) {
        const auto ghosts = pattern.getGhostColMap();
        const auto it = std::find(ghosts.begin(), ghosts.end(), 1);
        ASSERT_NE(it, ghosts.end());
        const std::size_t ghost_pos = static_cast<std::size_t>(it - ghosts.begin());
        const std::size_t local_idx = static_cast<std::size_t>(owned.size()) + ghost_pos;
        ASSERT_LT(local_idx, cvs.size());
        EXPECT_NEAR(cvs[local_idx], owned_value, 1e-14);
    }
}

TEST(TrilinosBackendMPI, DotAndNormMatchManualReduction)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        GTEST_SKIP() << "This test requires MPI ranks >= 2";
    }

    constexpr GlobalIndex n_global = 101;
    const IndexRange owned = blockOwnedRange(n_global, size, rank);

    auto factory = BackendFactory::create(BackendKind::Trilinos);
    auto v = factory->createVector(owned.size(), n_global);

    auto vs = v->localSpan();
    for (std::size_t i = 0; i < vs.size(); ++i) {
        const GlobalIndex gid = owned.first + static_cast<GlobalIndex>(i);
        vs[i] = static_cast<Real>(gid + 1);
    }

    const Real dot = v->dot(*v);
    const Real norm = v->norm();

    Real local_sum = 0.0;
    for (const auto val : vs) local_sum += val * val;
    Real global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    EXPECT_NEAR(dot, global_sum, 1e-10);
    EXPECT_NEAR(norm, std::sqrt(global_sum), 1e-10);
}
#endif

} // namespace svmp::FE::backends
