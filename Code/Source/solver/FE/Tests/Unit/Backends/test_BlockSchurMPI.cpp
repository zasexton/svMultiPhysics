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
#include "Backends/FSILS/FsilsVector.h"

#include <mpi.h>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace svmp::FE::backends {

namespace {

using svmp::FE::sparsity::DistributedSparsityPattern;
using svmp::FE::sparsity::IndexRange;

SolverOptions makeFsilsBlockSchurOptions(int dof_per_node,
                                         int primary_components,
                                         int constraint_components,
                                         FsilsBlockSchurSchurPreconditioner schur_pc =
                                             FsilsBlockSchurSchurPreconditioner::DiagL,
                                         FsilsBlockSchurMomentumApproximation momentum_hat =
                                             FsilsBlockSchurMomentumApproximation::DiagK)
{
    SolverOptions opts;
    opts.method = SolverMethod::BlockSchur;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 0.4;
    opts.abs_tol = 0.0;
    opts.max_iter = 20;
    opts.krylov_dim = 20;
    opts.fsils_blockschur_gm_max_iter = 80;
    opts.fsils_blockschur_cg_max_iter = 80;
    opts.fsils_blockschur_gm_rel_tol = 1e-10;
    opts.fsils_blockschur_cg_rel_tol = 1e-10;
    opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;
    opts.fsils_blockschur_schur_preconditioner = schur_pc;
    opts.fsils_blockschur_momentum_approximation = momentum_hat;

    BlockLayout layout;
    layout.blocks.push_back({"u", 0, primary_components, BlockRole::PrimaryField});
    layout.blocks.push_back({"p", primary_components, constraint_components, BlockRole::ConstraintField});
    layout.momentum_block = 0;
    layout.constraint_block = 1;
    opts.block_layout = std::move(layout);
    return opts;
}

void expectSolverReportSane(const SolverReport& rep, int max_iter)
{
    EXPECT_GE(rep.iterations, 0);
    EXPECT_LE(rep.iterations, max_iter);
    EXPECT_TRUE(std::isfinite(rep.initial_residual_norm));
    EXPECT_TRUE(std::isfinite(rep.final_residual_norm));
    EXPECT_TRUE(std::isfinite(rep.relative_residual));
    EXPECT_TRUE(std::isfinite(rep.setup_time_seconds));
    EXPECT_TRUE(std::isfinite(rep.validation_time_seconds));
    EXPECT_TRUE(std::isfinite(rep.collective_time_seconds));
}

void expectBlockSchurMetricsPresent(const SolverReport& rep)
{
    EXPECT_GT(rep.blockschur_outer_iterations, 0);
    EXPECT_GT(rep.blockschur_momentum_solve_calls, 0);
    EXPECT_GT(rep.blockschur_momentum_iterations, 0);
    EXPECT_LE(rep.blockschur_momentum_restart_cycles, rep.blockschur_momentum_solve_calls);
    EXPECT_GE(rep.blockschur_momentum_solve_time_seconds, 0.0);
    if (rep.blockschur_schur_solve_calls == 0) {
        EXPECT_EQ(rep.blockschur_schur_iterations, 0);
    } else {
        EXPECT_GT(rep.blockschur_schur_solve_calls, 0);
        EXPECT_GT(rep.blockschur_schur_iterations, 0);
    }
    EXPECT_GE(rep.blockschur_schur_setup_time_seconds, 0.0);
    EXPECT_GE(rep.blockschur_schur_solve_time_seconds, 0.0);
    EXPECT_GE(rep.blockschur_collective_calls_max_per_outer, 0u);
    EXPECT_GE(rep.blockschur_collective_time_max_per_outer, 0.0);
}

void expectBlockSchurOrExplicitRecovery(const SolverReport& rep)
{
    if (rep.blockschur_outer_iterations > 0) {
        expectBlockSchurMetricsPresent(rep);
        return;
    }

    EXPECT_NE(rep.message.find("fallback gmres"), std::string::npos);
}

Real sparseRankOneDot(const GenericVector& x, const RankOneUpdate& update, MPI_Comm comm)
{
    const auto* x_fs = dynamic_cast<const FsilsVector*>(&x);
    EXPECT_NE(x_fs, nullptr);
    if (x_fs == nullptr) {
        return 0.0;
    }

    std::vector<GlobalIndex> dofs;
    dofs.reserve(update.v.size());
    for (const auto& [dof, _] : update.v) {
        dofs.push_back(dof);
    }

    std::vector<GlobalIndex> resolved(dofs.size(), INVALID_GLOBAL_INDEX);
    x_fs->resolveEntriesCached(dofs, resolved);

    const auto xs = x_fs->localSpan();
    double local_dot = 0.0;
    for (std::size_t i = 0; i < update.v.size(); ++i) {
        const auto local_dof = resolved[i];
        if (local_dof == INVALID_GLOBAL_INDEX) {
            continue;
        }
        local_dot += static_cast<double>(update.v[i].second) *
                     static_cast<double>(xs[static_cast<std::size_t>(local_dof)]);
    }

    double global_dot = local_dot;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
    return static_cast<Real>(global_dot);
}

void addRankOneContribution(FsilsFactory& factory,
                            GenericVector& y,
                            const GenericVector& x,
                            std::span<const RankOneUpdate> updates,
                            MPI_Comm comm)
{
    auto corr = factory.createVector(y.size());
    corr->zero();
    auto view = corr->createAssemblyView();
    view->beginAssemblyPhase();
    for (const auto& update : updates) {
        const Real dot = sparseRankOneDot(x, update, comm);
        const Real scale = update.sigma * dot;
        if (std::abs(scale) <= Real(1e-30)) {
            continue;
        }
        for (const auto& [dof, val] : update.v) {
            view->addVectorEntry(dof, scale * val, assembly::AddMode::Add);
        }
    }
    view->finalizeAssembly();

    auto ys = y.localSpan();
    const auto cs = corr->localSpan();
    EXPECT_EQ(ys.size(), cs.size());
    if (ys.size() != cs.size()) {
        return;
    }
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ys[i] += cs[i];
    }
}

Real fullOperatorRelativeResidual(FsilsFactory& factory,
                                  const GenericMatrix& A,
                                  GenericVector& x,
                                  const GenericVector& b,
                                  std::span<const RankOneUpdate> updates,
                                  MPI_Comm comm)
{
    x.updateGhosts();

    auto Ax = factory.createVector(b.size());
    A.mult(x, *Ax);
    addRankOneContribution(factory, *Ax, x, updates, comm);

    auto b_acc = factory.createVector(b.size());
    b_acc->copyFrom(b);
    auto* b_fs = dynamic_cast<FsilsVector*>(b_acc.get());
    EXPECT_NE(b_fs, nullptr);
    if (b_fs != nullptr) {
        b_fs->accumulateOverlap();
    }

    auto r = factory.createVector(b.size());
    auto rs = r->localSpan();
    const auto bs = b_acc->localSpan();
    const auto axs = Ax->localSpan();
    EXPECT_EQ(rs.size(), bs.size());
    EXPECT_EQ(rs.size(), axs.size());
    if (rs.size() != bs.size() || rs.size() != axs.size()) {
        return std::numeric_limits<Real>::infinity();
    }
    for (std::size_t i = 0; i < rs.size(); ++i) {
        rs[i] = bs[i] - axs[i];
    }

    const Real denom = std::max<Real>(b_acc->norm(), 1e-30);
    return r->norm() / denom;
}


Real sparseReducedDot(const GenericVector& x,
                     std::span<const std::pair<GlobalIndex, Real>> entries,
                     MPI_Comm comm)
{
    const auto* x_fs = dynamic_cast<const FsilsVector*>(&x);
    EXPECT_NE(x_fs, nullptr);
    if (x_fs == nullptr) {
        return 0.0;
    }

    std::vector<GlobalIndex> dofs;
    dofs.reserve(entries.size());
    for (const auto& [dof, _] : entries) {
        dofs.push_back(dof);
    }

    std::vector<GlobalIndex> resolved(dofs.size(), INVALID_GLOBAL_INDEX);
    x_fs->resolveEntriesCached(dofs, resolved);

    const auto xs = x_fs->localSpan();
    double local_dot = 0.0;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto local_dof = resolved[i];
        if (local_dof == INVALID_GLOBAL_INDEX) {
            continue;
        }
        local_dot += static_cast<double>(entries[i].second) *
                     static_cast<double>(xs[static_cast<std::size_t>(local_dof)]);
    }

    double global_dot = local_dot;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
    return static_cast<Real>(global_dot);
}

void addReducedFieldContribution(FsilsFactory& factory,
                                 GenericVector& y,
                                 const GenericVector& x,
                                 std::span<const ReducedFieldUpdate> updates,
                                 MPI_Comm comm)
{
    auto corr = factory.createVector(y.size());
    corr->zero();
    auto view = corr->createAssemblyView();
    view->beginAssemblyPhase();
    for (const auto& update : updates) {
        const Real dot = sparseReducedDot(x,
                                          std::span<const std::pair<GlobalIndex, Real>>(update.right.data(),
                                                                                         update.right.size()),
                                          comm);
        const Real scale = update.sigma * dot;
        if (std::abs(scale) <= Real(1e-30)) {
            continue;
        }
        for (const auto& [dof, val] : update.left) {
            view->addVectorEntry(dof, scale * val, assembly::AddMode::Add);
        }
    }
    view->finalizeAssembly();

    auto ys = y.localSpan();
    const auto cs = corr->localSpan();
    EXPECT_EQ(ys.size(), cs.size());
    if (ys.size() != cs.size()) {
        return;
    }
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ys[i] += cs[i];
    }
}

Real fullOperatorRelativeResidual(FsilsFactory& factory,
                                  const GenericMatrix& A,
                                  GenericVector& x,
                                  const GenericVector& b,
                                  std::span<const ReducedFieldUpdate> updates,
                                  MPI_Comm comm)
{
    x.updateGhosts();

    auto Ax = factory.createVector(b.size());
    A.mult(x, *Ax);
    addReducedFieldContribution(factory, *Ax, x, updates, comm);

    auto b_acc = factory.createVector(b.size());
    b_acc->copyFrom(b);
    auto* b_fs = dynamic_cast<FsilsVector*>(b_acc.get());
    EXPECT_NE(b_fs, nullptr);
    if (b_fs != nullptr) {
        b_fs->accumulateOverlap();
    }

    auto r = factory.createVector(b.size());
    auto rs = r->localSpan();
    const auto bs = b_acc->localSpan();
    const auto axs = Ax->localSpan();
    EXPECT_EQ(rs.size(), bs.size());
    EXPECT_EQ(rs.size(), axs.size());
    if (rs.size() != bs.size() || rs.size() != axs.size()) {
        return std::numeric_limits<Real>::infinity();
    }
    for (std::size_t i = 0; i < rs.size(); ++i) {
        rs[i] = bs[i] - axs[i];
    }

    const Real denom = std::max<Real>(b_acc->norm(), 1e-30);
    return r->norm() / denom;
}

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

    // Build element-level couplings (overlap model):
    // - Rank 0 assembles element (node 0, node 1) => dofs [0..5]
    // - Rank 1 assembles element (node 1, node 2) => dofs [3..8]
    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs = {0, 1, 2, 3, 4, 5};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    } else {
        const std::array<GlobalIndex, 6> edofs = {3, 4, 5, 6, 7, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    }

    pattern.ensureDiagonal();
    pattern.finalize();

    // Ghost info on Rank 0 for Node 1 (indices 3,4,5)
    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 4, 5};
        // Rank 0 assembles the local element (0-1) which contributes to rows for Node 1 (global dofs 3..5).
        // In the overlap model used by FSILS, those rows must include the full element column closure (0..5)
        // so that scalar entry insertion via FsilsMatrixView::addMatrixEntries doesn't silently drop terms.
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        for (int r = 0; r < dof; ++r) {
            for (GlobalIndex c = 0; c < static_cast<GlobalIndex>(2 * dof); ++c) {
                ghost_cols.push_back(c);
            }
        }
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
    // A = [K G; D L] with D = -G^t, matching FSILS NS solver expectations.
    // Matrix per element (size 2*dof = 6; two nodes per element).
    
    const int edof = 2 * dof;
    std::vector<Real> Ke(edof * edof, 0.0);

    const auto setKe = [&](int r, int c, Real v) {
        Ke[static_cast<std::size_t>(r * edof + c)] = v;
    };

    // Node-local saddle-point block (u, v, p) used by the FSILS BlockSchur unit tests.
    const Real B[3][3] = {
        {4.0, 1.0, 1.0},
        {1.0, 3.0, 0.0},
        {1.0, 0.0, 1.0},
    };

    // Off-diagonal node coupling (velocity only; keep pressure decoupled between nodes for stability).
    const Real C[3][3] = {
        {-1.0, 0.0, 0.0},
        { 0.0,-1.0, 0.0},
        { 0.0, 0.0, 0.0},
    };

    // Assemble 6x6 local block matrix:
    // [ B  C ]
    // [ Cᵀ B ]
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            setKe(r, c, B[r][c]);
            setKe(r, c + 3, C[r][c]);
            setKe(r + 3, c, C[c][r]);
            setKe(r + 3, c + 3, B[r][c]);
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

    struct Variant {
        FsilsBlockSchurSchurPreconditioner schur_pc;
        FsilsBlockSchurMomentumApproximation momentum_hat;
        const char* label;
    };

    const std::array<Variant, 5> variants{{
        {FsilsBlockSchurSchurPreconditioner::DiagL, FsilsBlockSchurMomentumApproximation::DiagK, "diag-l"},
        {FsilsBlockSchurSchurPreconditioner::ILUL, FsilsBlockSchurMomentumApproximation::DiagK, "ilu-l"},
        {FsilsBlockSchurSchurPreconditioner::AlgebraicSchur, FsilsBlockSchurMomentumApproximation::BlockDiagK, "algebraic-shat-blockdiag-k"},
        {FsilsBlockSchurSchurPreconditioner::AlgebraicSchur, FsilsBlockSchurMomentumApproximation::ILUK, "algebraic-shat-ilu-k"},
        {FsilsBlockSchurSchurPreconditioner::AlgebraicSchur, FsilsBlockSchurMomentumApproximation::ASM, "algebraic-shat-asm-k"},
    }};

    for (const auto& variant : variants) {
        SCOPED_TRACE(variant.label);
        auto x_case = factory.createVector(n_global);
        auto opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/3, /*primary_components=*/2, /*constraint_components=*/1,
                                               variant.schur_pc, variant.momentum_hat);

        auto solver = factory.createLinearSolver(opts);
        const auto rep = solver->solve(*A, *x_case, *b);
        EXPECT_TRUE(rep.converged);
        expectSolverReportSane(rep, opts.max_iter);
        expectBlockSchurOrExplicitRecovery(rep);
        EXPECT_GT(rep.collective_calls, 0u);
        if (rep.blockschur_outer_iterations > 0) {
            EXPECT_GT(rep.blockschur_collective_calls_max_per_outer, 0u);
        }

        x_case->updateGhosts();

        auto b_acc = factory.createVector(n_global);
        {
            auto dst = b_acc->localSpan();
            const auto src = b->localSpan();
            ASSERT_EQ(dst.size(), src.size());
            std::copy(src.begin(), src.end(), dst.begin());
        }
        auto* b_fs = dynamic_cast<FsilsVector*>(b_acc.get());
        ASSERT_NE(b_fs, nullptr);
        b_fs->accumulateOverlap();

        auto Ax = factory.createVector(n_global);
        A->mult(*x_case, *Ax);

        auto r = factory.createVector(n_global);
        r->zero();
        auto rs = r->localSpan();
        const auto bs = b_acc->localSpan();
        const auto axs = Ax->localSpan();
        ASSERT_EQ(rs.size(), bs.size());
        ASSERT_EQ(rs.size(), axs.size());
        for (std::size_t i = 0; i < rs.size(); ++i) {
            rs[i] = bs[i] - axs[i];
        }

        const Real denom = std::max<Real>(b_acc->norm(), 1e-30);
        const Real rel = r->norm() / denom;
        EXPECT_LE(rel, opts.rel_tol + 1e-12);
    }
}

TEST(FsilsBackendMPI, SolveBlockSchur4DOFMultiConstraintPreconditioners)
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        GTEST_SKIP() << "This test uses a single-rank 4x4 multi-constraint system";
    }

    constexpr int dof = 4;
    constexpr GlobalIndex n_global = 4;
    const IndexRange owned{0, n_global};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    for (GlobalIndex row = 0; row < n_global; ++row) {
        for (GlobalIndex col = 0; col < n_global; ++col) {
            pattern.addEntry(row, col);
        }
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    FsilsFactory factory(dof);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);

    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    const std::array<Real, 16> Ke = {
        4.0, 1.0, 1.0, 0.2,
        1.0, 3.0, 0.1, 1.1,
        0.9, 0.2, 2.0, 0.4,
        0.3, 0.8, 0.3, 1.7,
    };

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewB = b->createAssemblyView();
    viewB->beginAssemblyPhase();
    std::array<Real, 4> rhs{};
    for (int r = 0; r < dof; ++r) {
        for (int c = 0; c < dof; ++c) {
            rhs[static_cast<std::size_t>(r)] += Ke[static_cast<std::size_t>(r * dof + c)];
        }
    }
    viewB->addVectorEntries(dofs, rhs, assembly::AddMode::Insert);
    viewB->finalizeAssembly();

    const std::array<std::pair<FsilsBlockSchurSchurPreconditioner, FsilsBlockSchurMomentumApproximation>, 4> variants{{
        {FsilsBlockSchurSchurPreconditioner::BlockDiagL, FsilsBlockSchurMomentumApproximation::DiagK},
        {FsilsBlockSchurSchurPreconditioner::ILUL, FsilsBlockSchurMomentumApproximation::DiagK},
        {FsilsBlockSchurSchurPreconditioner::AlgebraicSchur, FsilsBlockSchurMomentumApproximation::BlockDiagK},
        {FsilsBlockSchurSchurPreconditioner::AlgebraicSchur, FsilsBlockSchurMomentumApproximation::ILUK},
    }};

    for (const auto& [schur_pc, momentum_hat] : variants) {
        SCOPED_TRACE(std::string(fsilsBlockSchurPreconditionerToString(schur_pc)) + "/" +
                     std::string(fsilsBlockSchurMomentumApproximationToString(momentum_hat)));
        auto x = factory.createVector(n_global);
        auto opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/4, /*primary_components=*/2, /*constraint_components=*/2,
                                               schur_pc, momentum_hat);
        opts.rel_tol = 1e-10;
        opts.abs_tol = 1e-12;
        opts.max_iter = 40;
        opts.krylov_dim = 40;

        auto solver = factory.createLinearSolver(opts);
        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_TRUE(rep.converged);
        expectSolverReportSane(rep, opts.max_iter);
        expectBlockSchurMetricsPresent(rep);

        auto Ax = factory.createVector(n_global);
        A->mult(*x, *Ax);
        auto r = factory.createVector(n_global);
        auto rs = r->localSpan();
        const auto bs = b->localSpan();
        const auto axs = Ax->localSpan();
        ASSERT_EQ(rs.size(), bs.size());
        ASSERT_EQ(rs.size(), axs.size());
        for (std::size_t i = 0; i < rs.size(); ++i) {
            rs[i] = bs[i] - axs[i];
        }

        const Real denom = std::max<Real>(b->norm(), 1e-30);
        EXPECT_LE(r->norm() / denom, 1e-8);
    }
}

TEST(FsilsBackendMPI, RankOneUpdateSolversConvergeComparable)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    const IndexRange owned = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs = {0, 1, 2, 3, 4, 5};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    } else {
        const std::array<GlobalIndex, 6> edofs = {3, 4, 5, 6, 7, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    }

    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 4, 5};
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        for (int r = 0; r < dof; ++r) {
            for (GlobalIndex c = 0; c < static_cast<GlobalIndex>(2 * dof); ++c) {
                ghost_cols.push_back(c);
            }
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(dof);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();

    const int edof = 2 * dof;
    std::vector<Real> Ke(edof * edof, 0.0);
    const auto setKe = [&](int r, int c, Real v) {
        Ke[static_cast<std::size_t>(r * edof + c)] = v;
    };

    const Real B[3][3] = {
        {4.0, 1.0, 1.0},
        {1.0, 3.0, 0.0},
        {1.0, 0.0, 1.0},
    };
    const Real C[3][3] = {
        {-1.0, 0.0, 0.0},
        { 0.0,-1.0, 0.0},
        { 0.0, 0.0, 0.0},
    };

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            setKe(r, c, B[r][c]);
            setKe(r, c + 3, C[r][c]);
            setKe(r + 3, c, C[c][r]);
            setKe(r + 3, c + 3, B[r][c]);
        }
    }

    if (rank == 0) {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    } else {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = 3 + i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewB = b->createAssemblyView();
    viewB->beginAssemblyPhase();

    std::vector<Real> be(edof);
    for (int i = 0; i < edof; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < edof; ++j) {
            sum += Ke[static_cast<std::size_t>(i * edof + j)];
        }
        be[static_cast<std::size_t>(i)] = sum;
    }

    if (rank == 0) {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = i;
        viewB->addVectorEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(be.data(), be.size()),
                                assembly::AddMode::Add);
    } else {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = 3 + i;
        viewB->addVectorEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(be.data(), be.size()),
                                assembly::AddMode::Add);
    }

    RankOneUpdate upd{};
    upd.sigma = 2000.0;
    upd.active_components = {0, 1};
    // Route the distributed rank-one correction through FSILS' native face path.
    upd.prefer_native_face = true;
    if (rank == 1) {
        upd.v = {
            {6, 0.10},
            {7, 0.05},
        };
    }

    const Real dot_exact = 0.15;
    const Real scale_exact = upd.sigma * dot_exact;
    if (rank == 1) {
        viewB->addVectorEntry(6, scale_exact * 0.10, assembly::AddMode::Add);
        viewB->addVectorEntry(7, scale_exact * 0.05, assembly::AddMode::Add);
    }
    viewB->finalizeAssembly();

    const std::array<SolverMethod, 1> methods{
        SolverMethod::BlockSchur,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur" : "gmres");

        auto x_case = factory.createVector(n_global);
        SolverOptions opts;
        if (method == SolverMethod::BlockSchur) {
            opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/3, /*primary_components=*/2, /*constraint_components=*/1,
                                              FsilsBlockSchurSchurPreconditioner::DiagL,
                                              FsilsBlockSchurMomentumApproximation::DiagK);
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 20;
            opts.krylov_dim = 40;
            opts.fsils_blockschur_gm_max_iter = 120;
            opts.fsils_blockschur_cg_max_iter = 120;
            opts.fsils_blockschur_gm_rel_tol = 1e-10;
            opts.fsils_blockschur_cg_rel_tol = 1e-10;
        } else {
            opts.method = SolverMethod::GMRES;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 400;
            opts.krylov_dim = 120;
            opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;
        }

        auto solver = factory.createLinearSolver(opts);
        ASSERT_TRUE(solver->supportsNativeRankOneUpdates());
        solver->setRankOneUpdates(std::span<const RankOneUpdate>(&upd, 1));
        solver->setEffectiveTimeStep(1.0 / 300.0);

        const auto rep = solver->solve(*A, *x_case, *b);
        EXPECT_TRUE(rep.converged);
        expectSolverReportSane(rep, opts.max_iter);
        if (method == SolverMethod::BlockSchur) {
            EXPECT_EQ(rep.message.find("fallback"), std::string::npos);
            expectBlockSchurMetricsPresent(rep);
        }

        const Real rel = fullOperatorRelativeResidual(factory, *A, *x_case, *b,
                                                      std::span<const RankOneUpdate>(&upd, 1),
                                                      MPI_COMM_WORLD);
        EXPECT_LE(rel, 1e-8);
    }
}


TEST(FsilsBackendMPI, ReducedFieldUpdateSolversConvergeComparable)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    const IndexRange owned = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs = {0, 1, 2, 3, 4, 5};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    } else {
        const std::array<GlobalIndex, 6> edofs = {3, 4, 5, 6, 7, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    }

    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 4, 5};
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        for (int r = 0; r < dof; ++r) {
            for (GlobalIndex c = 0; c < static_cast<GlobalIndex>(2 * dof); ++c) {
                ghost_cols.push_back(c);
            }
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(dof);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();

    const int edof = 2 * dof;
    std::vector<Real> Ke(edof * edof, 0.0);
    const auto setKe = [&](int r, int c, Real v) {
        Ke[static_cast<std::size_t>(r * edof + c)] = v;
    };

    const Real B[3][3] = {
        {4.0, 1.0, 1.0},
        {1.0, 3.0, 0.0},
        {1.0, 0.0, 1.0},
    };
    const Real C[3][3] = {
        {-1.0, 0.0, 0.0},
        { 0.0,-1.0, 0.0},
        { 0.0, 0.0, 0.0},
    };

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            setKe(r, c, B[r][c]);
            setKe(r, c + 3, C[r][c]);
            setKe(r + 3, c, C[c][r]);
            setKe(r + 3, c + 3, B[r][c]);
        }
    }

    if (rank == 0) {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    } else {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = 3 + i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    ReducedFieldUpdate upd{};
    upd.sigma = 1500.0;
    upd.active_components = {0, 1};
    if (rank == 0) {
        upd.left = {
            {1, 0.03},
        };
        upd.right = {
            {0, 0.05},
            {1, 0.12},
        };
    } else {
        upd.left = {
            {6, 0.10},
            {7, -0.07},
        };
        upd.right = {
            {6, -0.02},
        };
    }

    auto x_exact = factory.createVector(n_global);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    x_exact->updateGhosts();
    A->mult(*x_exact, *b);
    addReducedFieldContribution(factory,
                                *b,
                                *x_exact,
                                std::span<const ReducedFieldUpdate>(&upd, 1),
                                MPI_COMM_WORLD);

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur_reduced" : "gmres_reduced");

        auto x_case = factory.createVector(n_global);
        SolverOptions opts;
        if (method == SolverMethod::BlockSchur) {
            opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/3, /*primary_components=*/2, /*constraint_components=*/1,
                                              FsilsBlockSchurSchurPreconditioner::DiagL,
                                              FsilsBlockSchurMomentumApproximation::DiagK);
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 20;
            opts.krylov_dim = 40;
            opts.fsils_blockschur_gm_max_iter = 120;
            opts.fsils_blockschur_cg_max_iter = 120;
            opts.fsils_blockschur_gm_rel_tol = 1e-10;
            opts.fsils_blockschur_cg_rel_tol = 1e-10;
        } else {
            opts.method = SolverMethod::GMRES;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 400;
            opts.krylov_dim = 120;
            opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;
        }

        auto solver = factory.createLinearSolver(opts);
        ASSERT_TRUE(solver->supportsNativeReducedFieldUpdates());
        solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(&upd, 1));
        solver->setEffectiveTimeStep(1.0 / 300.0);

        const auto rep = solver->solve(*A, *x_case, *b);
        EXPECT_TRUE(rep.converged);
        expectSolverReportSane(rep, (method == SolverMethod::GMRES) ? 400 : opts.max_iter);
        EXPECT_EQ(rep.message.find("fallback"), std::string::npos);
        if (method == SolverMethod::BlockSchur) {
            expectBlockSchurMetricsPresent(rep);
        }

        const Real rel = fullOperatorRelativeResidual(factory,
                                                      *A,
                                                      *x_case,
                                                      *b,
                                                      std::span<const ReducedFieldUpdate>(&upd, 1),
                                                      MPI_COMM_WORLD);
        EXPECT_LE(rel, 1e-8);
    }
}

TEST(FsilsBackendMPI, GroupedBorderedFieldCouplingSolversConvergeComparable)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    const IndexRange owned = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs = {0, 1, 2, 3, 4, 5};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    } else {
        const std::array<GlobalIndex, 6> edofs = {3, 4, 5, 6, 7, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    }

    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 4, 5};
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        for (int r = 0; r < dof; ++r) {
            for (GlobalIndex c = 0; c < static_cast<GlobalIndex>(2 * dof); ++c) {
                ghost_cols.push_back(c);
            }
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(dof);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();

    const int edof = 2 * dof;
    std::vector<Real> Ke(edof * edof, 0.0);
    const auto setKe = [&](int r, int c, Real v) {
        Ke[static_cast<std::size_t>(r * edof + c)] = v;
    };

    const Real B[3][3] = {
        {4.0, 1.0, 1.0},
        {1.0, 3.0, 0.0},
        {1.0, 0.0, 1.0},
    };
    const Real C[3][3] = {
        {-1.0, 0.0, 0.0},
        { 0.0,-1.0, 0.0},
        { 0.0, 0.0, 0.0},
    };

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            setKe(r, c, B[r][c]);
            setKe(r, c + 3, C[r][c]);
            setKe(r + 3, c, C[c][r]);
            setKe(r + 3, c + 3, B[r][c]);
        }
    }

    if (rank == 0) {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) {
            idx[static_cast<std::size_t>(i)] = i;
        }
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    } else {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) {
            idx[static_cast<std::size_t>(i)] = 3 + i;
        }
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    constexpr Real d00 = 2.0;
    constexpr Real d01 = -0.3;
    constexpr Real d10 = 0.4;
    constexpr Real d11 = 1.5;
    const Real det = d00 * d11 - d01 * d10;
    ASSERT_GT(std::abs(det), Real(1e-12));
    const Real dinv00 = d11 / det;
    const Real dinv01 = -d01 / det;
    const Real dinv10 = -d10 / det;
    const Real dinv11 = d00 / det;

    std::array<std::array<Real, n_global>, 2> c_rows{};
    std::array<std::array<Real, n_global>, 2> b_cols{};

    c_rows[0][0] = 0.05;
    c_rows[0][1] = 0.11;
    c_rows[0][6] = -0.02;

    c_rows[1][1] = -0.03;
    c_rows[1][6] = 0.07;
    c_rows[1][7] = 0.04;

    b_cols[0][1] = 0.08;
    b_cols[0][6] = -0.05;

    b_cols[1][0] = -0.04;
    b_cols[1][7] = 0.09;

    std::array<ReducedFieldUpdate, 2> updates{};
    for (int i = 0; i < 2; ++i) {
        updates[static_cast<std::size_t>(i)].sigma = -1.0;
        updates[static_cast<std::size_t>(i)].active_components = {0, 1};
        updates[static_cast<std::size_t>(i)].grouped_coupling_id = 0;
    }

    GroupedBorderedFieldCoupling grouped{};
    grouped.grouped_coupling_id = 0;
    grouped.aux_matrix = {d00, d01,
                          d10, d11};
    grouped.modes.resize(2);
    grouped.modes[0].active_components = {0, 1};
    grouped.modes[1].active_components = {0, 1};

    const auto ownedHere = [&](GlobalIndex dof_idx) {
        return owned.contains(dof_idx);
    };

    for (GlobalIndex dof_idx = 0; dof_idx < n_global; ++dof_idx) {
        const Real row0 = dinv00 * c_rows[0][static_cast<std::size_t>(dof_idx)] +
                          dinv01 * c_rows[1][static_cast<std::size_t>(dof_idx)];
        const Real row1 = dinv10 * c_rows[0][static_cast<std::size_t>(dof_idx)] +
                          dinv11 * c_rows[1][static_cast<std::size_t>(dof_idx)];
        if (!ownedHere(dof_idx)) {
            continue;
        }

        if (std::abs(b_cols[0][static_cast<std::size_t>(dof_idx)]) > Real(1e-30)) {
            updates[0].left.emplace_back(dof_idx, b_cols[0][static_cast<std::size_t>(dof_idx)]);
            grouped.modes[0].left.emplace_back(dof_idx, b_cols[0][static_cast<std::size_t>(dof_idx)]);
        }
        if (std::abs(b_cols[1][static_cast<std::size_t>(dof_idx)]) > Real(1e-30)) {
            updates[1].left.emplace_back(dof_idx, b_cols[1][static_cast<std::size_t>(dof_idx)]);
            grouped.modes[1].left.emplace_back(dof_idx, b_cols[1][static_cast<std::size_t>(dof_idx)]);
        }
        if (std::abs(row0) > Real(1e-30)) {
            updates[0].right.emplace_back(dof_idx, row0);
        }
        if (std::abs(row1) > Real(1e-30)) {
            updates[1].right.emplace_back(dof_idx, row1);
        }
        if (std::abs(c_rows[0][static_cast<std::size_t>(dof_idx)]) > Real(1e-30)) {
            grouped.modes[0].right.emplace_back(dof_idx, c_rows[0][static_cast<std::size_t>(dof_idx)]);
        }
        if (std::abs(c_rows[1][static_cast<std::size_t>(dof_idx)]) > Real(1e-30)) {
            grouped.modes[1].right.emplace_back(dof_idx, c_rows[1][static_cast<std::size_t>(dof_idx)]);
        }
    }

    auto x_exact = factory.createVector(n_global);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    x_exact->updateGhosts();
    A->mult(*x_exact, *b);
    addReducedFieldContribution(factory,
                                *b,
                                *x_exact,
                                std::span<const ReducedFieldUpdate>(updates.data(), updates.size()),
                                MPI_COMM_WORLD);

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur_grouped_bordered"
                                                        : "gmres_grouped_bordered");

        auto x_case = factory.createVector(n_global);
        SolverOptions opts;
        if (method == SolverMethod::BlockSchur) {
            opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/3,
                                              /*primary_components=*/2,
                                              /*constraint_components=*/1,
                                              FsilsBlockSchurSchurPreconditioner::DiagL,
                                              FsilsBlockSchurMomentumApproximation::DiagK);
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 20;
            opts.krylov_dim = 40;
            opts.fsils_blockschur_gm_max_iter = 120;
            opts.fsils_blockschur_cg_max_iter = 120;
            opts.fsils_blockschur_gm_rel_tol = 1e-10;
            opts.fsils_blockschur_cg_rel_tol = 1e-10;
        } else {
            opts.method = SolverMethod::GMRES;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 400;
            opts.krylov_dim = 120;
            opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;
        }

        auto solver = factory.createLinearSolver(opts);
        ASSERT_TRUE(solver->supportsNativeReducedFieldUpdates());
        solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
        solver->setGroupedBorderedFieldCouplings(
            std::span<const GroupedBorderedFieldCoupling>(&grouped, 1));
        solver->setEffectiveTimeStep(1.0 / 300.0);

        const auto rep = solver->solve(*A, *x_case, *b);
        EXPECT_TRUE(rep.converged);
        expectSolverReportSane(rep, (method == SolverMethod::GMRES) ? 400 : opts.max_iter);
        EXPECT_EQ(rep.message.find("fallback"), std::string::npos);
        if (method == SolverMethod::BlockSchur) {
            expectBlockSchurMetricsPresent(rep);
        }

        const Real rel = fullOperatorRelativeResidual(
            factory,
            *A,
            *x_case,
            *b,
            std::span<const ReducedFieldUpdate>(updates.data(), updates.size()),
            MPI_COMM_WORLD);
        EXPECT_LE(rel, 1e-8);
    }
}

TEST(FsilsBackendMPI, GroupedBorderedFieldCouplingSingleModeConvergesComparable)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    const IndexRange owned = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs = {0, 1, 2, 3, 4, 5};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    } else {
        const std::array<GlobalIndex, 6> edofs = {3, 4, 5, 6, 7, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    }

    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 4, 5};
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        for (int r = 0; r < dof; ++r) {
            for (GlobalIndex c = 0; c < static_cast<GlobalIndex>(2 * dof); ++c) {
                ghost_cols.push_back(c);
            }
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(dof);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();

    const int edof = 2 * dof;
    std::vector<Real> Ke(edof * edof, 0.0);
    const auto setKe = [&](int r, int c, Real v) {
        Ke[static_cast<std::size_t>(r * edof + c)] = v;
    };

    const Real B[3][3] = {
        {4.0, 1.0, 1.0},
        {1.0, 3.0, 0.0},
        {1.0, 0.0, 1.0},
    };
    const Real C[3][3] = {
        {-1.0, 0.0, 0.0},
        { 0.0,-1.0, 0.0},
        { 0.0, 0.0, 0.0},
    };

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            setKe(r, c, B[r][c]);
            setKe(r, c + 3, C[r][c]);
            setKe(r + 3, c, C[c][r]);
            setKe(r + 3, c + 3, B[r][c]);
        }
    }

    if (rank == 0) {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) {
            idx[static_cast<std::size_t>(i)] = i;
        }
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    } else {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) {
            idx[static_cast<std::size_t>(i)] = 3 + i;
        }
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    constexpr Real d00 = 1.75;
    std::array<Real, n_global> c_row{};
    std::array<Real, n_global> b_col{};

    c_row[0] = 0.05;
    c_row[1] = 0.11;
    c_row[6] = -0.02;

    b_col[1] = 0.08;
    b_col[6] = -0.05;

    ReducedFieldUpdate upd{};
    upd.sigma = -1.0;
    upd.active_components = {0, 1};
    upd.grouped_coupling_id = 0;

    GroupedBorderedFieldCoupling grouped{};
    grouped.grouped_coupling_id = 0;
    grouped.aux_matrix = {d00};
    grouped.modes.resize(1);
    grouped.modes[0].active_components = {0, 1};

    const auto ownedHere = [&](GlobalIndex dof_idx) {
        return owned.contains(dof_idx);
    };

    for (GlobalIndex dof_idx = 0; dof_idx < n_global; ++dof_idx) {
        if (!ownedHere(dof_idx)) {
            continue;
        }

        if (std::abs(b_col[static_cast<std::size_t>(dof_idx)]) > Real(1e-30)) {
            upd.left.emplace_back(dof_idx, b_col[static_cast<std::size_t>(dof_idx)]);
            grouped.modes[0].left.emplace_back(dof_idx, b_col[static_cast<std::size_t>(dof_idx)]);
        }
        if (std::abs(c_row[static_cast<std::size_t>(dof_idx)]) > Real(1e-30)) {
            upd.right.emplace_back(dof_idx,
                                   c_row[static_cast<std::size_t>(dof_idx)] / d00);
            grouped.modes[0].right.emplace_back(dof_idx, c_row[static_cast<std::size_t>(dof_idx)]);
        }
    }

    auto x_exact = factory.createVector(n_global);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    x_exact->updateGhosts();
    A->mult(*x_exact, *b);
    addReducedFieldContribution(factory,
                                *b,
                                *x_exact,
                                std::span<const ReducedFieldUpdate>(&upd, 1),
                                MPI_COMM_WORLD);

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur_grouped_bordered_single"
                                                        : "gmres_grouped_bordered_single");

        auto x_case = factory.createVector(n_global);
        SolverOptions opts;
        if (method == SolverMethod::BlockSchur) {
            opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/3,
                                              /*primary_components=*/2,
                                              /*constraint_components=*/1,
                                              FsilsBlockSchurSchurPreconditioner::DiagL,
                                              FsilsBlockSchurMomentumApproximation::DiagK);
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 20;
            opts.krylov_dim = 40;
            opts.fsils_blockschur_gm_max_iter = 120;
            opts.fsils_blockschur_cg_max_iter = 120;
            opts.fsils_blockschur_gm_rel_tol = 1e-10;
            opts.fsils_blockschur_cg_rel_tol = 1e-10;
        } else {
            opts.method = SolverMethod::GMRES;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 400;
            opts.krylov_dim = 120;
            opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;
        }

        auto solver = factory.createLinearSolver(opts);
        ASSERT_TRUE(solver->supportsNativeReducedFieldUpdates());
        solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(&upd, 1));
        solver->setGroupedBorderedFieldCouplings(
            std::span<const GroupedBorderedFieldCoupling>(&grouped, 1));
        solver->setEffectiveTimeStep(1.0 / 300.0);

        const auto rep = solver->solve(*A, *x_case, *b);
        EXPECT_TRUE(rep.converged);
        expectSolverReportSane(rep, (method == SolverMethod::GMRES) ? 400 : opts.max_iter);
        EXPECT_EQ(rep.message.find("fallback"), std::string::npos);
        if (method == SolverMethod::BlockSchur) {
            expectBlockSchurMetricsPresent(rep);
        }

        const Real rel = fullOperatorRelativeResidual(
            factory,
            *A,
            *x_case,
            *b,
            std::span<const ReducedFieldUpdate>(&upd, 1),
            MPI_COMM_WORLD);
        EXPECT_LE(rel, 1e-8);
    }
}

TEST(FsilsBackendMPI, RankOneUpdateSolversConvergeComparable4DOF)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    constexpr int dof = 4;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    const IndexRange owned = (rank == 0) ? IndexRange{0, 4} : IndexRange{4, 12};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    if (rank == 0) {
        const std::array<GlobalIndex, 8> edofs = {0, 1, 2, 3, 4, 5, 6, 7};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    } else {
        const std::array<GlobalIndex, 8> edofs = {4, 5, 6, 7, 8, 9, 10, 11};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    }

    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{4, 5, 6, 7};
        std::vector<GlobalIndex> ghost_row_ptr{0, 8, 16, 24, 32};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(32);
        for (int r = 0; r < dof; ++r) {
            for (GlobalIndex c = 0; c < static_cast<GlobalIndex>(2 * dof); ++c) {
                ghost_cols.push_back(c);
            }
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(dof);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n_global);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();

    const int edof = 2 * dof;
    std::vector<Real> Ke(edof * edof, 0.0);
    const auto setKe = [&](int r, int c, Real v) {
        Ke[static_cast<std::size_t>(r * edof + c)] = v;
    };

    const Real B[4][4] = {
        {6.0, 1.0, 0.5,  1.0},
        {1.0, 5.0, 0.3, -0.4},
        {0.5, 0.3, 4.5,  0.6},
        {1.0,-0.4, 0.6,  1.2},
    };
    const Real C[4][4] = {
        {-1.5, 0.0,  0.0, -0.2},
        { 0.0,-1.0,  0.0,  0.3},
        { 0.0, 0.0, -1.2, -0.1},
        {-0.2, 0.3, -0.1,  0.2},
    };

    for (int r = 0; r < dof; ++r) {
        for (int c = 0; c < dof; ++c) {
            setKe(r, c, B[r][c]);
            setKe(r, c + dof, C[r][c]);
            setKe(r + dof, c, C[c][r]);
            setKe(r + dof, c + dof, B[r][c]);
        }
    }

    if (rank == 0) {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) {
            idx[static_cast<std::size_t>(i)] = i;
        }
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    } else {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) {
            idx[static_cast<std::size_t>(i)] = 4 + i;
        }
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    RankOneUpdate upd{};
    upd.sigma = 1600.0;
    upd.active_components = {0, 1, 2};
    // Route the distributed rank-one correction through FSILS' native face path.
    upd.prefer_native_face = true;
    if (rank == 1) {
        upd.v = {
            {8,  0.10},
            {9,  0.05},
            {10, -0.08},
        };
    }

    auto x_exact = factory.createVector(n_global);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    x_exact->updateGhosts();
    A->mult(*x_exact, *b);
    addRankOneContribution(factory, *b, *x_exact, std::span<const RankOneUpdate>(&upd, 1), MPI_COMM_WORLD);

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur_4dof" : "gmres_4dof");

        auto x_case = factory.createVector(n_global);
        SolverOptions opts;
        if (method == SolverMethod::BlockSchur) {
            opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/4, /*primary_components=*/3, /*constraint_components=*/1,
                                              FsilsBlockSchurSchurPreconditioner::DiagL,
                                              FsilsBlockSchurMomentumApproximation::DiagK);
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 30;
            opts.krylov_dim = 40;
            opts.fsils_blockschur_gm_max_iter = 160;
            opts.fsils_blockschur_cg_max_iter = 160;
            opts.fsils_blockschur_gm_rel_tol = 1e-10;
            opts.fsils_blockschur_cg_rel_tol = 1e-10;
        } else {
            opts.method = SolverMethod::GMRES;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 400;
            opts.krylov_dim = 120;
            opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;
        }

        auto solver = factory.createLinearSolver(opts);
        ASSERT_TRUE(solver->supportsNativeRankOneUpdates());
        solver->setRankOneUpdates(std::span<const RankOneUpdate>(&upd, 1));
        solver->setEffectiveTimeStep(1.0 / 300.0);

        const auto rep = solver->solve(*A, *x_case, *b);
        EXPECT_TRUE(rep.converged);
        expectSolverReportSane(rep, (method == SolverMethod::GMRES) ? 400 : opts.max_iter);
        EXPECT_EQ(rep.message.find("fallback"), std::string::npos);
        if (method == SolverMethod::BlockSchur) {
            expectBlockSchurMetricsPresent(rep);
        }

        const Real rel = fullOperatorRelativeResidual(factory, *A, *x_case, *b,
                                                      std::span<const RankOneUpdate>(&upd, 1),
                                                      MPI_COMM_WORLD);
        EXPECT_LE(rel, 1e-8);
    }
}

TEST(FsilsBackendMPI, SolveNSBlockSchur3DOFSubcommunicator)
{
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 4) {
        GTEST_SKIP() << "This test requires exactly 4 MPI ranks";
    }

    MPI_Comm subcomm = MPI_COMM_NULL;
    const int color = world_rank / 2;
    const int key = world_rank % 2;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &subcomm);
    ASSERT_NE(subcomm, MPI_COMM_NULL);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(subcomm, &rank);
    MPI_Comm_size(subcomm, &size);
    ASSERT_EQ(size, 2);

    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    const IndexRange owned = (rank == 0) ? IndexRange{0, 3} : IndexRange{3, 9};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);

    if (rank == 0) {
        const std::array<GlobalIndex, 6> edofs = {0, 1, 2, 3, 4, 5};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    } else {
        const std::array<GlobalIndex, 6> edofs = {3, 4, 5, 6, 7, 8};
        pattern.addElementCouplings(std::span<const GlobalIndex>(edofs.data(), edofs.size()));
    }

    pattern.ensureDiagonal();
    pattern.finalize();

    if (rank == 0) {
        std::vector<GlobalIndex> ghost_rows{3, 4, 5};
        std::vector<GlobalIndex> ghost_row_ptr{0, 6, 12, 18};
        std::vector<GlobalIndex> ghost_cols;
        ghost_cols.reserve(18);
        for (int r = 0; r < dof; ++r) {
            for (GlobalIndex c = 0; c < static_cast<GlobalIndex>(2 * dof); ++c) {
                ghost_cols.push_back(c);
            }
        }
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_row_ptr), std::move(ghost_cols));
    }

    FsilsFactory factory(dof, {}, subcomm);
    auto A = factory.createMatrix(pattern);
    auto x = factory.createVector(n_global);
    auto b = factory.createVector(n_global);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();

    const int edof = 2 * dof;
    std::vector<Real> Ke(edof * edof, 0.0);
    const auto setKe = [&](int r, int c, Real v) {
        Ke[static_cast<std::size_t>(r * edof + c)] = v;
    };

    const Real B[3][3] = {
        {4.0, 1.0, 1.0},
        {1.0, 3.0, 0.0},
        {1.0, 0.0, 1.0},
    };
    const Real C[3][3] = {
        {-1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0},
        {0.0, 0.0, 0.0},
    };

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            setKe(r, c, B[r][c]);
            setKe(r, c + 3, C[r][c]);
            setKe(r + 3, c, C[c][r]);
            setKe(r + 3, c + 3, B[r][c]);
        }
    }

    if (rank == 0) {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    } else {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = 3 + i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewB = b->createAssemblyView();
    viewB->beginAssemblyPhase();

    std::vector<Real> be(edof);
    for (int i = 0; i < edof; ++i) {
        Real sum = 0.0;
        for (int j = 0; j < edof; ++j) {
            sum += Ke[static_cast<std::size_t>(i * edof + j)];
        }
        be[static_cast<std::size_t>(i)] = sum;
    }

    if (rank == 0) {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = i;
        viewB->addVectorEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(be.data(), be.size()),
                                assembly::AddMode::Add);
    } else {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = 3 + i;
        viewB->addVectorEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(be.data(), be.size()),
                                assembly::AddMode::Add);
    }
    viewB->finalizeAssembly();

    auto opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/3, /*primary_components=*/2, /*constraint_components=*/1,
                                           FsilsBlockSchurSchurPreconditioner::AlgebraicSchur,
                                           FsilsBlockSchurMomentumApproximation::BlockDiagK);

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);
    expectSolverReportSane(rep, opts.max_iter);
    expectBlockSchurOrExplicitRecovery(rep);
    EXPECT_GT(rep.collective_calls, 0u);
    if (rep.blockschur_outer_iterations > 0) {
        EXPECT_GT(rep.blockschur_collective_calls_max_per_outer, 0u);
    }

    auto* x_fs = dynamic_cast<FsilsVector*>(x.get());
    ASSERT_NE(x_fs, nullptr);
    const auto* shared = x_fs->shared();
    ASSERT_NE(shared, nullptr);

    int comm_compare = MPI_UNEQUAL;
    MPI_Comm_compare(subcomm, shared->lhs.commu.comm, &comm_compare);
    EXPECT_TRUE(comm_compare == MPI_IDENT || comm_compare == MPI_CONGRUENT);

    x->updateGhosts();

    auto b_acc = factory.createVector(n_global);
    {
        auto dst = b_acc->localSpan();
        const auto src = b->localSpan();
        ASSERT_EQ(dst.size(), src.size());
        std::copy(src.begin(), src.end(), dst.begin());
    }
    auto* b_fs = dynamic_cast<FsilsVector*>(b_acc.get());
    ASSERT_NE(b_fs, nullptr);
    b_fs->accumulateOverlap();

    auto Ax = factory.createVector(n_global);
    A->mult(*x, *Ax);

    auto r = factory.createVector(n_global);
    auto rs = r->localSpan();
    const auto bs = b_acc->localSpan();
    const auto axs = Ax->localSpan();
    ASSERT_EQ(rs.size(), bs.size());
    ASSERT_EQ(rs.size(), axs.size());
    for (std::size_t i = 0; i < rs.size(); ++i) {
        rs[i] = bs[i] - axs[i];
    }

    const Real denom = std::max<Real>(b_acc->norm(), 1e-30);
    const Real rel = r->norm() / denom;
    EXPECT_LE(rel, opts.rel_tol + 1e-12);

    MPI_Comm_free(&subcomm);
}

TEST(FsilsBackendMPI, NullspaceProjectionUsesSubcommunicator)
{
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 4) {
        GTEST_SKIP() << "This test requires exactly 4 MPI ranks";
    }

    MPI_Comm subcomm = MPI_COMM_NULL;
    const int color = world_rank / 2;
    const int key = world_rank % 2;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &subcomm);
    ASSERT_NE(subcomm, MPI_COMM_NULL);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(subcomm, &rank);
    MPI_Comm_size(subcomm, &size);
    ASSERT_EQ(size, 2);

    constexpr GlobalIndex n_global = 2;
    const IndexRange owned = (rank == 0) ? IndexRange{0, 1} : IndexRange{1, 2};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    pattern.addEntry(owned.first, owned.first);
    pattern.ensureDiagonal();
    pattern.finalize();

    FsilsFactory factory(/*dof_per_node=*/1, {}, subcomm);
    auto A = factory.createMatrix(pattern);
    auto x = factory.createVector(n_global);
    auto b = factory.createVector(n_global);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewA->addMatrixEntry(owned.first, owned.first, 1.0, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewB = b->createAssemblyView();
    viewB->beginAssemblyPhase();
    viewB->addVectorEntry(owned.first, static_cast<Real>(color + 1), assembly::AddMode::Insert);
    viewB->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-14;
    opts.abs_tol = 1e-14;
    opts.max_iter = 8;
    opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;

    auto solver = factory.createLinearSolver(opts);
    const auto local_size = x->localSpan().size();
    ASSERT_EQ(local_size, 1u);
    const double basis_entry = 1.0 / std::sqrt(static_cast<double>(n_global));
    std::vector<double> basis_local(local_size, basis_entry);
    std::vector<std::vector<double>> basis{basis_local};
    solver->setNullspaceBasis(basis);

    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);
    EXPECT_GT(rep.collective_calls, 0u);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 1u);
    EXPECT_NEAR(xs[0], 0.0, 1e-12);

    MPI_Comm_free(&subcomm);
}

} // namespace svmp::FE::backends
