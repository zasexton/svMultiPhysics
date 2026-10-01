/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Distributed checks of the FSILS right preconditioners: block-Jacobi block
// ILU(0) and SIMPLE across two ranks, and collective reuse decisions.

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/FSILS/FsilsBlockPreconditioners.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsLinearSolver.h"
#include "Backends/Utils/BackendOptions.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace svmp::FE::backends {
namespace {

using sparsity::DistributedSparsityPattern;
using sparsity::IndexRange;

constexpr int kDof = 3;
// Nodal blocks of a saddle-point-like chain operator (row-major 3 x 3): the
// last component plays the role of a stabilized pressure.
constexpr Real kDiag[9] = {6.0, 1.0, 0.5, 1.0, 6.0, 0.5, 0.5, 0.5, 1.0};
constexpr Real kLeft[9] = {-1.0, 0.2, -0.3, 0.1, -1.0, 0.2, -0.3, 0.2, -0.2};
constexpr Real kRight[9] = {-1.0, 0.1, 0.3, 0.2, -1.0, -0.2, 0.3, -0.2, -0.2};

struct ChainSystem {
    std::unique_ptr<GenericMatrix> A;
    std::unique_ptr<GenericVector> b;
    std::unique_ptr<GenericVector> x;
};

/// 1D chain of n_nodes nodes split into two contiguous halves.  Every rank
/// assembles its complete owned rows (couplings to the neighbouring node of
/// the other rank are ghost columns), so the exact solution is x = 1.
ChainSystem buildChain(const FsilsFactory& factory, GlobalIndex n_nodes, int rank, Real scale)
{
    const GlobalIndex half = n_nodes / 2;
    const GlobalIndex first = (rank == 0) ? 0 : half;
    const GlobalIndex last = (rank == 0) ? half - 1 : n_nodes - 1;
    const GlobalIndex n_global = n_nodes * kDof;
    const IndexRange owned{first * kDof, (last + 1) * kDof};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    auto neighbours = [&](GlobalIndex i) {
        std::vector<GlobalIndex> nb;
        for (GlobalIndex j = std::max<GlobalIndex>(0, i - 1); j <= std::min(n_nodes - 1, i + 1); ++j) {
            nb.push_back(j);
        }
        return nb;
    };
    for (GlobalIndex i = first; i <= last; ++i) {
        for (int r = 0; r < kDof; ++r) {
            for (const auto j : neighbours(i)) {
                for (int c = 0; c < kDof; ++c) {
                    pattern.addEntry(i * kDof + r, j * kDof + c);
                }
            }
        }
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    // Ghost rows of the neighbouring node, restricted to the local overlap set.
    const GlobalIndex ghost = (rank == 0) ? last + 1 : first - 1;
    const std::vector<GlobalIndex> ghost_cols_nodes =
        (rank == 0) ? std::vector<GlobalIndex>{ghost - 1, ghost} : std::vector<GlobalIndex>{ghost, ghost + 1};
    std::vector<GlobalIndex> ghost_rows;
    std::vector<GlobalIndex> ghost_ptr{0};
    std::vector<GlobalIndex> ghost_cols;
    for (int r = 0; r < kDof; ++r) {
        ghost_rows.push_back(ghost * kDof + r);
        for (const auto j : ghost_cols_nodes) {
            for (int c = 0; c < kDof; ++c) {
                ghost_cols.push_back(j * kDof + c);
            }
        }
        ghost_ptr.push_back(static_cast<GlobalIndex>(ghost_cols.size()));
    }
    pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_ptr), std::move(ghost_cols));

    ChainSystem sys;
    sys.A = factory.createMatrix(pattern);
    sys.b = factory.createVector(n_global);
    sys.x = factory.createVector(n_global);

    auto viewA = sys.A->createAssemblyView();
    auto viewb = sys.b->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewb->beginAssemblyPhase();
    for (GlobalIndex i = first; i <= last; ++i) {
        const auto nb = neighbours(i);
        std::vector<GlobalIndex> rows;
        std::vector<GlobalIndex> cols;
        for (int r = 0; r < kDof; ++r) {
            rows.push_back(i * kDof + r);
        }
        for (const auto j : nb) {
            for (int c = 0; c < kDof; ++c) {
                cols.push_back(j * kDof + c);
            }
        }
        std::vector<Real> values(rows.size() * cols.size(), 0.0);
        std::vector<Real> rhs(kDof, 0.0);
        for (std::size_t k = 0; k < nb.size(); ++k) {
            const Real* blk = (nb[k] == i) ? kDiag : (nb[k] < i ? kLeft : kRight);
            for (int r = 0; r < kDof; ++r) {
                for (int c = 0; c < kDof; ++c) {
                    const Real v = scale * blk[r * kDof + c];
                    values[static_cast<std::size_t>(r) * cols.size() + k * kDof + static_cast<std::size_t>(c)] = v;
                    rhs[static_cast<std::size_t>(r)] += v;
                }
            }
        }
        viewA->addMatrixEntries(rows, cols, values, assembly::AddMode::Insert);
        viewb->addVectorEntries(rows, rhs, assembly::AddMode::Insert);
    }
    viewA->finalizeAssembly();
    viewb->finalizeAssembly();
    sys.A->finalizeAssembly();
    return sys;
}

SolverOptions rightPcOptions(RightPreconditionerType pc)
{
    SolverOptions o;
    o.method = SolverMethod::GMRES;
    o.preconditioner = PreconditionerType::RowColumnScaling;
    o.fsils_use_rcs = true;
    o.rel_tol = 1e-10;
    o.abs_tol = 1e-14;
    o.max_iter = 400;
    o.krylov_dim = 30;
    o.right_preconditioner = pc;
    o.reuse_preconditioner = true;
    BlockLayout layout;
    layout.blocks.push_back({"Velocity", 0, kDof - 1, BlockRole::Generic});
    layout.blocks.push_back({"Pressure", kDof - 1, 1, BlockRole::Generic});
    o.block_layout = layout;
    o.right_preconditioner_constraint_block = "Pressure";
    return o;
}

class RightPreconditionerMPI : public ::testing::TestWithParam<RightPreconditionerType> {};

TEST_P(RightPreconditionerMPI, BlockJacobiSolvesMatchExactSolutionAndReuseIsCollective)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    FsilsFactory factory(kDof);
    FsilsLinearSolver solver(rightPcOptions(GetParam()));
    for (const Real scale : {1.0, 1.01}) {
        auto sys = buildChain(factory, 24, rank, scale);
        const auto rep = solver.solve(*sys.A, *sys.x, *sys.b);
        EXPECT_TRUE(rep.converged) << "scale=" << scale;
        for (const auto v : sys.x->localSpan()) {
            EXPECT_NEAR(v, 1.0, 1e-8) << "scale=" << scale;
        }
    }

    const auto* pc = solver.krylovPreconditioner();
    ASSERT_NE(pc, nullptr);
    EXPECT_EQ(pc->stats().solves, 2u);
    EXPECT_EQ(pc->stats().refreshes + pc->stats().reuses, 2u);
    // Refresh decisions must be identical on every rank.
    long long local[2] = {static_cast<long long>(pc->stats().refreshes),
                          static_cast<long long>(pc->stats().reuses)};
    long long lo[2] = {0, 0};
    long long hi[2] = {0, 0};
    MPI_Allreduce(local, lo, 2, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local, hi, 2, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(lo[0], hi[0]);
    EXPECT_EQ(lo[1], hi[1]);
}

INSTANTIATE_TEST_SUITE_P(Kinds,
                         RightPreconditionerMPI,
                         ::testing::Values(RightPreconditionerType::BlockILU0, RightPreconditionerType::Simple));

} // namespace
} // namespace svmp::FE::backends
