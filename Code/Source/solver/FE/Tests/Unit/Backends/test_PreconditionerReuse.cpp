/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Unit tests for the opt-in Krylov right preconditioners (block ILU(0), SIMPLE)
// and for preconditioner/factorization reuse in the FSILS and Eigen backends.

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/FSILS/FsilsBlockPreconditioners.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsLinearSolver.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsShared.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Backends/Utils/BackendOptions.h"
#include "Backends/Utils/PreconditionerReusePolicy.h"
#include "Sparsity/SparsityPattern.h"

#if defined(FE_HAS_EIGEN)
#include "Backends/Eigen/EigenFactory.h"
#include "Backends/Eigen/EigenLinearSolver.h"
#include "Backends/Eigen/EigenMatrix.h"
#include "Backends/Eigen/EigenVector.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace svmp::FE::backends {
namespace {

using Reason = PreconditionerReusePolicy::Reason;

// ---------------------------------------------------------------------------
// Small block systems on node graphs
// ---------------------------------------------------------------------------

struct BlockSystem {
    int n{0};
    int dof{0};
    std::vector<std::vector<int>> adjacency;   // sorted neighbours incl. self
    std::vector<std::vector<double>> blocks;   // per (row, k) row-major dof x dof
    std::vector<double> rhs;
};

std::vector<std::vector<int>> pathGraph(int n)
{
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = std::max(0, i - 1); j <= std::min(n - 1, i + 1); ++j) {
            adj[static_cast<std::size_t>(i)].push_back(j);
        }
    }
    return adj;
}

std::vector<std::vector<int>> gridGraph(int nx, int ny)
{
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(nx * ny));
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const int i = y * nx + x;
            auto& row = adj[static_cast<std::size_t>(i)];
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= nx || yy >= ny) {
                        continue;
                    }
                    row.push_back(yy * nx + xx);
                }
            }
            std::sort(row.begin(), row.end());
        }
    }
    return adj;
}

/// Random saddle-point-like block system: the last component of each node is
/// a "pressure" with a small stabilizing diagonal, the others form a
/// diagonally dominant "momentum" part.
BlockSystem makeSystem(const std::vector<std::vector<int>>& adj, int dof, unsigned seed, double perturbation = 0.0)
{
    BlockSystem s;
    s.n = static_cast<int>(adj.size());
    s.dof = dof;
    s.adjacency = adj;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::mt19937 prng(seed + 7777u);
    for (int i = 0; i < s.n; ++i) {
        for (const int j : adj[static_cast<std::size_t>(i)]) {
            std::vector<double> b(static_cast<std::size_t>(dof * dof), 0.0);
            for (int r = 0; r < dof; ++r) {
                for (int c = 0; c < dof; ++c) {
                    double v = 0.2 * u(rng);
                    if (i == j && r == c) {
                        v = (r == dof - 1) ? 0.5 : 4.0 + static_cast<double>(adj[static_cast<std::size_t>(i)].size());
                    }
                    if (r == dof - 1 && c == dof - 1 && i != j) {
                        v = -0.05;
                    }
                    if (perturbation != 0.0) {
                        v *= 1.0 + perturbation * u(prng);
                    }
                    b[static_cast<std::size_t>(r * dof + c)] = v;
                }
            }
            s.blocks.push_back(std::move(b));
        }
    }
    s.rhs.resize(static_cast<std::size_t>(s.n * dof));
    for (auto& v : s.rhs) {
        v = u(rng);
    }
    return s;
}

std::vector<double> denseMatrix(const BlockSystem& s)
{
    const int N = s.n * s.dof;
    std::vector<double> a(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 0.0);
    std::size_t k = 0;
    for (int i = 0; i < s.n; ++i) {
        for (const int j : s.adjacency[static_cast<std::size_t>(i)]) {
            const auto& b = s.blocks[k++];
            for (int r = 0; r < s.dof; ++r) {
                for (int c = 0; c < s.dof; ++c) {
                    a[static_cast<std::size_t>((i * s.dof + r) * N + j * s.dof + c)] =
                        b[static_cast<std::size_t>(r * s.dof + c)];
                }
            }
        }
    }
    return a;
}

std::vector<double> denseSolve(std::vector<double> a, std::vector<double> b)
{
    const int N = static_cast<int>(b.size());
    for (int col = 0; col < N; ++col) {
        int piv = col;
        for (int r = col + 1; r < N; ++r) {
            if (std::abs(a[static_cast<std::size_t>(r * N + col)]) >
                std::abs(a[static_cast<std::size_t>(piv * N + col)])) {
                piv = r;
            }
        }
        for (int c = 0; c < N; ++c) {
            std::swap(a[static_cast<std::size_t>(col * N + c)], a[static_cast<std::size_t>(piv * N + c)]);
        }
        std::swap(b[static_cast<std::size_t>(col)], b[static_cast<std::size_t>(piv)]);
        for (int r = col + 1; r < N; ++r) {
            const double f = a[static_cast<std::size_t>(r * N + col)] / a[static_cast<std::size_t>(col * N + col)];
            for (int c = col; c < N; ++c) {
                a[static_cast<std::size_t>(r * N + c)] -= f * a[static_cast<std::size_t>(col * N + c)];
            }
            b[static_cast<std::size_t>(r)] -= f * b[static_cast<std::size_t>(col)];
        }
    }
    std::vector<double> x(static_cast<std::size_t>(N), 0.0);
    for (int r = N - 1; r >= 0; --r) {
        double s = b[static_cast<std::size_t>(r)];
        for (int c = r + 1; c < N; ++c) {
            s -= a[static_cast<std::size_t>(r * N + c)] * x[static_cast<std::size_t>(c)];
        }
        x[static_cast<std::size_t>(r)] = s / a[static_cast<std::size_t>(r * N + r)];
    }
    return x;
}

double relativeDifference(const std::vector<double>& a, const std::vector<double>& b)
{
    double d = 0.0;
    double n = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d += (a[i] - b[i]) * (a[i] - b[i]);
        n += b[i] * b[i];
    }
    return std::sqrt(d) / std::max(std::sqrt(n), 1e-300);
}

sparsity::SparsityPattern scalarPattern(const BlockSystem& s)
{
    const GlobalIndex N = static_cast<GlobalIndex>(s.n) * s.dof;
    sparsity::SparsityPattern p(N, N);
    std::vector<GlobalIndex> cols;
    for (int i = 0; i < s.n; ++i) {
        cols.clear();
        for (const int j : s.adjacency[static_cast<std::size_t>(i)]) {
            for (int c = 0; c < s.dof; ++c) {
                cols.push_back(static_cast<GlobalIndex>(j) * s.dof + c);
            }
        }
        for (int r = 0; r < s.dof; ++r) {
            p.addEntries(static_cast<GlobalIndex>(i) * s.dof + r, cols);
        }
    }
    p.finalize();
    return p;
}

struct FsilsSystem {
    std::unique_ptr<GenericMatrix> A;
    std::unique_ptr<GenericVector> b;
    std::unique_ptr<GenericVector> x;
};

FsilsSystem buildFsils(const BlockSystem& s)
{
    FsilsFactory factory(s.dof);
    FsilsSystem sys;
    sys.A = factory.createMatrix(scalarPattern(s));
    auto* A = dynamic_cast<FsilsMatrix*>(sys.A.get());
    const auto shared = A->shared();
    std::size_t k = 0;
    for (int i = 0; i < s.n; ++i) {
        for (const int j : s.adjacency[static_cast<std::size_t>(i)]) {
            A->addBlock(shared->globalNodeToInternal(i), shared->globalNodeToInternal(j), s.blocks[k++].data(),
                        s.dof, assembly::AddMode::Insert);
        }
    }
    A->finalizeAssembly();
    sys.b = factory.createVector(static_cast<GlobalIndex>(s.n) * s.dof);
    sys.x = factory.createVector(static_cast<GlobalIndex>(s.n) * s.dof);
    auto& bd = dynamic_cast<FsilsVector*>(sys.b.get())->data();
    for (int g = 0; g < s.n; ++g) {
        const int old = shared->globalNodeToOld(g);
        for (int c = 0; c < s.dof; ++c) {
            bd[static_cast<std::size_t>(old * s.dof + c)] = s.rhs[static_cast<std::size_t>(g * s.dof + c)];
        }
    }
    return sys;
}

std::vector<double> fsilsSolution(const BlockSystem& s, const FsilsSystem& sys)
{
    const auto shared = dynamic_cast<const FsilsMatrix*>(sys.A.get())->shared();
    const auto& xd = dynamic_cast<const FsilsVector*>(sys.x.get())->data();
    std::vector<double> x(static_cast<std::size_t>(s.n * s.dof));
    for (int g = 0; g < s.n; ++g) {
        const int old = shared->globalNodeToOld(g);
        for (int c = 0; c < s.dof; ++c) {
            x[static_cast<std::size_t>(g * s.dof + c)] = xd[static_cast<std::size_t>(old * s.dof + c)];
        }
    }
    return x;
}

SolverOptions gmresOptions(int dof, RightPreconditionerType pc, bool reuse)
{
    SolverOptions o;
    o.method = SolverMethod::GMRES;
    o.preconditioner = PreconditionerType::RowColumnScaling;
    o.fsils_use_rcs = true;
    o.rel_tol = 1e-10;
    o.abs_tol = 1e-14;
    o.max_iter = 400;
    o.krylov_dim = 50;
    o.right_preconditioner = pc;
    o.reuse_preconditioner = reuse;
    BlockLayout layout;
    layout.blocks.push_back({"Velocity", 0, dof - 1, BlockRole::Generic});
    layout.blocks.push_back({"Pressure", dof - 1, 1, BlockRole::Generic});
    o.block_layout = layout;
    o.right_preconditioner_constraint_block = "Pressure";
    return o;
}

// ---------------------------------------------------------------------------
// Reuse policy
// ---------------------------------------------------------------------------

TEST(PreconditionerReusePolicy, InitialAndStructureChangesRefresh)
{
    PreconditionerReusePolicy policy;
    auto d = policy.beforeSolve(true, false);
    EXPECT_TRUE(d.refresh);
    EXPECT_EQ(d.reason, Reason::Initial);

    policy.recordRefresh();
    policy.recordSolve(10, true, 100.0);
    d = policy.beforeSolve(true, false);
    EXPECT_FALSE(d.refresh);
    EXPECT_EQ(d.reason, Reason::None);

    d = policy.beforeSolve(true, true);
    EXPECT_TRUE(d.refresh);
    EXPECT_EQ(d.reason, Reason::Structure);
}

TEST(PreconditionerReusePolicy, RefreshesWhenExtraIterationsReachSetupCost)
{
    PreconditionerReusePolicy policy;
    policy.recordRefresh();
    policy.recordSolve(10, true, 5.0);  // fresh solve: 10 iterations, setup = 5 iterations
    EXPECT_EQ(policy.freshIterations(), 10);
    EXPECT_DOUBLE_EQ(policy.refreshCostIterations(), 5.0);

    policy.recordSolve(12, false);  // +2
    EXPECT_FALSE(policy.beforeSolve(true, false).refresh);
    policy.recordSolve(9, false);   // fewer iterations never count negative
    EXPECT_DOUBLE_EQ(policy.excessIterations(), 2.0);
    EXPECT_FALSE(policy.beforeSolve(true, false).refresh);
    policy.recordSolve(13, false);  // +3 -> 5 >= 5
    const auto d = policy.beforeSolve(true, false);
    EXPECT_TRUE(d.refresh);
    EXPECT_EQ(d.reason, Reason::BreakEven);
    EXPECT_EQ(policy.reuseCount(), 3u);

    policy.recordRefresh();
    EXPECT_DOUBLE_EQ(policy.excessIterations(), 0.0);
    EXPECT_EQ(policy.refreshCount(), 2u);
}

TEST(PreconditionerReusePolicy, DisabledReuseRefreshesEverySolve)
{
    PreconditionerReusePolicy policy;
    policy.recordRefresh();
    policy.recordSolve(10, true, 1.0e6);
    const auto d = policy.beforeSolve(false, false);
    EXPECT_TRUE(d.refresh);
    EXPECT_EQ(d.reason, Reason::Disabled);
}

TEST(PreconditionerReusePolicy, ZeroSetupCostRefreshesEverySolve)
{
    PreconditionerReusePolicy policy;
    policy.recordRefresh();
    policy.recordSolve(10, true, 0.0);
    EXPECT_TRUE(policy.beforeSolve(true, false).refresh);
    policy.invalidate();
    EXPECT_EQ(policy.beforeSolve(true, false).reason, Reason::Initial);
}

// ---------------------------------------------------------------------------
// Block kernels
// ---------------------------------------------------------------------------

TEST(FsilsBlockPreconditioners, DenseBlockInverseMatchesIdentity)
{
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int d = 1; d <= 9; ++d) {
        std::vector<double> a(static_cast<std::size_t>(d * d));
        for (auto& v : a) {
            v = u(rng);
        }
        for (int i = 0; i < d; ++i) {
            a[static_cast<std::size_t>(i * d + i)] += 3.0;
        }
        std::vector<double> inv(a.size());
        EXPECT_EQ(invertDenseBlock(d, a.data(), inv.data()), 0);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = 0.0;
                for (int k = 0; k < d; ++k) {
                    s += a[static_cast<std::size_t>(r * d + k)] * inv[static_cast<std::size_t>(k * d + c)];
                }
                EXPECT_NEAR(s, r == c ? 1.0 : 0.0, 1e-12) << "d=" << d;
            }
        }
    }
}

TEST(FsilsBlockPreconditioners, SingularAndZeroBlocksAreRegularized)
{
    const double singular[4] = {1.0, 2.0, 2.0, 4.0};
    double inv[4] = {};
    EXPECT_GT(invertDenseBlock(2, singular, inv), 0);
    for (const double v : inv) {
        EXPECT_TRUE(std::isfinite(v));
    }
    const double zero[4] = {0.0, 0.0, 0.0, 0.0};
    EXPECT_EQ(invertDenseBlock(2, zero, inv), 2);
    EXPECT_DOUBLE_EQ(inv[0], 1.0);
    EXPECT_DOUBLE_EQ(inv[1], 0.0);
    EXPECT_DOUBLE_EQ(inv[3], 1.0);
}

TEST(FsilsBlockPreconditioners, BlockIlu0IsExactOnBlockTridiagonalSystems)
{
    // ILU(0) of a block-tridiagonal matrix has no dropped fill: it is exact.
    const auto sys = makeSystem(pathGraph(7), 3, 11u);
    FsilsOwnedBlockGraph graph;
    graph.n = sys.n;
    graph.row_ptr.push_back(0);
    std::vector<double> blocks;
    std::size_t k = 0;
    for (int i = 0; i < sys.n; ++i) {
        for (const int j : sys.adjacency[static_cast<std::size_t>(i)]) {
            if (j == i) {
                graph.diag.push_back(static_cast<int>(graph.cols.size()));
            }
            graph.cols.push_back(j);
            graph.src.push_back(static_cast<int>(k));
            blocks.insert(blocks.end(), sys.blocks[k].begin(), sys.blocks[k].end());
            ++k;
        }
        graph.row_ptr.push_back(static_cast<int>(graph.cols.size()));
    }
    BlockIlu0Factorization ilu;
    ilu.factor(graph, sys.dof, blocks);
    EXPECT_EQ(ilu.regularizedPivots(), 0);
    EXPECT_GT(ilu.factorFlops(), 0.0);
    EXPECT_GT(ilu.applyFlops(), 0.0);
    std::vector<double> x(sys.rhs.size());
    ilu.solve(sys.rhs.data(), x.data());
    const auto ref = denseSolve(denseMatrix(sys), sys.rhs);
    EXPECT_LT(relativeDifference(x, ref), 1e-12);
    // In-place application gives the same result.
    std::vector<double> y = sys.rhs;
    ilu.solve(y.data(), y.data());
    EXPECT_LT(relativeDifference(y, x), 1e-15);
}

// ---------------------------------------------------------------------------
// FSILS GMRES with right preconditioners
// ---------------------------------------------------------------------------

struct FsilsRun {
    SolverReport report;
    std::vector<double> x;
};

FsilsRun runFsils(LinearSolver& solver, const BlockSystem& s)
{
    auto sys = buildFsils(s);
    FsilsRun run;
    run.report = solver.solve(*sys.A, *sys.x, *sys.b);
    run.x = fsilsSolution(s, sys);
    return run;
}

TEST(FsilsRightPreconditioner, BlockIlu0AndSimpleMatchDirectSolve)
{
    const auto sys = makeSystem(gridGraph(6, 5), 4, 21u);
    const auto ref = denseSolve(denseMatrix(sys), sys.rhs);

    FsilsLinearSolver plain(gmresOptions(sys.dof, RightPreconditionerType::None, false));
    const auto r0 = runFsils(plain, sys);
    ASSERT_TRUE(r0.report.converged);
    EXPECT_LT(relativeDifference(r0.x, ref), 1e-7);

    for (const auto pc : {RightPreconditionerType::BlockILU0, RightPreconditionerType::Simple}) {
        FsilsLinearSolver solver(gmresOptions(sys.dof, pc, false));
        const auto r = runFsils(solver, sys);
        ASSERT_TRUE(r.report.converged) << rightPreconditionerToString(pc);
        EXPECT_LT(relativeDifference(r.x, ref), 1e-7) << rightPreconditionerToString(pc);
        EXPECT_LT(r.report.iterations, r0.report.iterations) << rightPreconditionerToString(pc);
        ASSERT_NE(solver.krylovPreconditioner(), nullptr);
        EXPECT_EQ(solver.krylovPreconditioner()->stats().refreshes, 1u);
    }
}

TEST(FsilsRightPreconditioner, SimpleFallsBackToBlockIlu0WithoutScalarConstraint)
{
    const auto sys = makeSystem(gridGraph(4, 4), 3, 5u);
    auto opts = gmresOptions(sys.dof, RightPreconditionerType::Simple, false);
    opts.right_preconditioner_constraint_block = "NoSuchBlock";
    opts.block_layout.reset();
    FsilsLinearSolver solver(opts);
    const auto r = runFsils(solver, sys);
    ASSERT_TRUE(r.report.converged);
    EXPECT_LT(relativeDifference(r.x, denseSolve(denseMatrix(sys), sys.rhs)), 1e-7);
}

TEST(FsilsRightPreconditioner, DisabledReuseGivesIdenticalResultsToFreshSolvers)
{
    const auto graph = gridGraph(5, 5);
    std::vector<BlockSystem> systems;
    for (int k = 0; k < 3; ++k) {
        systems.push_back(makeSystem(graph, 4, 31u, 0.02 * static_cast<double>(k)));
    }
    FsilsLinearSolver persistent(gmresOptions(4, RightPreconditionerType::BlockILU0, false));
    for (const auto& s : systems) {
        const auto a = runFsils(persistent, s);
        FsilsLinearSolver fresh(gmresOptions(4, RightPreconditionerType::BlockILU0, false));
        const auto b = runFsils(fresh, s);
        EXPECT_EQ(a.report.iterations, b.report.iterations);
        ASSERT_EQ(a.x.size(), b.x.size());
        for (std::size_t i = 0; i < a.x.size(); ++i) {
            EXPECT_EQ(a.x[i], b.x[i]);
        }
    }
    EXPECT_EQ(persistent.krylovPreconditioner()->stats().refreshes, 3u);
    EXPECT_EQ(persistent.krylovPreconditioner()->stats().reuses, 0u);
}

TEST(FsilsRightPreconditioner, ReuseKeepsTheFactorizationAndMeetsTolerance)
{
    const auto graph = gridGraph(6, 6);
    FsilsLinearSolver solver(gmresOptions(4, RightPreconditionerType::BlockILU0, true));
    const auto s0 = makeSystem(graph, 4, 41u);
    const auto r0 = runFsils(solver, s0);
    ASSERT_TRUE(r0.report.converged);
    const auto* pc = solver.krylovPreconditioner();
    ASSERT_NE(pc, nullptr);
    EXPECT_EQ(pc->stats().refreshes, 1u);

    // A nearby Jacobian (same sparsity) reuses the factorization unless the
    // break-even rule fires; either way the solve meets the tolerance.
    const auto s1 = makeSystem(graph, 4, 41u, 0.01);
    const auto r1 = runFsils(solver, s1);
    ASSERT_TRUE(r1.report.converged);
    EXPECT_LT(relativeDifference(r1.x, denseSolve(denseMatrix(s1), s1.rhs)), 1e-7);
    EXPECT_EQ(pc->stats().refreshes + pc->stats().reuses, 2u);
    EXPECT_EQ(pc->stats().solves, 2u);
}

TEST(FsilsRightPreconditioner, StructureChangeForcesRefresh)
{
    FsilsLinearSolver solver(gmresOptions(3, RightPreconditionerType::BlockILU0, true));
    const auto a = makeSystem(gridGraph(4, 4), 3, 51u);
    const auto b = makeSystem(pathGraph(16), 3, 52u);
    ASSERT_TRUE(runFsils(solver, a).report.converged);
    ASSERT_TRUE(runFsils(solver, b).report.converged);
    const auto* pc = solver.krylovPreconditioner();
    EXPECT_EQ(pc->stats().refreshes, 2u);
    EXPECT_EQ(pc->stats().last_reason, Reason::Structure);
}

TEST(FsilsRightPreconditioner, FailedReuseIsRepeatedWithAFreshFactorization)
{
    // Block ILU(0) is exact on a block-tridiagonal operator, so a fresh
    // preconditioner converges within a tiny iteration budget while a stale
    // one for a very different operator does not.
    const auto graph = pathGraph(12);
    auto opts = gmresOptions(3, RightPreconditionerType::BlockILU0, true);
    opts.max_iter = 3;
    opts.krylov_dim = 2;
    FsilsLinearSolver solver(opts);
    const auto a = makeSystem(graph, 3, 61u);
    ASSERT_TRUE(runFsils(solver, a).report.converged);
    auto b = makeSystem(graph, 3, 62u, 0.9);
    for (auto& blk : b.blocks) {
        std::reverse(blk.begin(), blk.end());
    }
    const auto rb = runFsils(solver, b);
    EXPECT_TRUE(rb.report.converged);
    const auto* pc = solver.krylovPreconditioner();
    EXPECT_EQ(pc->stats().refreshes, 2u);
    EXPECT_EQ(pc->stats().stale_retries + (pc->stats().last_reason == Reason::BreakEven ? 1u : 0u), 1u);
}

#if defined(FE_HAS_EIGEN)

// ---------------------------------------------------------------------------
// Eigen reuse
// ---------------------------------------------------------------------------

struct EigenSystem {
    std::unique_ptr<GenericMatrix> A;
    std::unique_ptr<GenericVector> b;
    std::unique_ptr<GenericVector> x;
};

EigenSystem buildEigen(const BlockSystem& s)
{
    EigenFactory factory;
    EigenSystem sys;
    sys.A = factory.createMatrix(scalarPattern(s));
    auto* A = dynamic_cast<EigenMatrix*>(sys.A.get());
    std::size_t k = 0;
    for (int i = 0; i < s.n; ++i) {
        for (const int j : s.adjacency[static_cast<std::size_t>(i)]) {
            const auto& blk = s.blocks[k++];
            for (int r = 0; r < s.dof; ++r) {
                for (int c = 0; c < s.dof; ++c) {
                    A->addValue(static_cast<GlobalIndex>(i) * s.dof + r, static_cast<GlobalIndex>(j) * s.dof + c,
                                blk[static_cast<std::size_t>(r * s.dof + c)], assembly::AddMode::Insert);
                }
            }
        }
    }
    A->finalizeAssembly();
    sys.b = factory.createVector(A->numRows());
    sys.x = factory.createVector(A->numRows());
    auto* b = dynamic_cast<EigenVector*>(sys.b.get());
    for (std::size_t i = 0; i < s.rhs.size(); ++i) {
        b->eigen()(static_cast<Eigen::Index>(i)) = s.rhs[i];
    }
    return sys;
}

FsilsRun runEigen(LinearSolver& solver, const BlockSystem& s)
{
    auto sys = buildEigen(s);
    FsilsRun run;
    run.report = solver.solve(*sys.A, *sys.x, *sys.b);
    const auto* x = dynamic_cast<const EigenVector*>(sys.x.get());
    run.x.assign(x->eigen().data(), x->eigen().data() + x->eigen().size());
    return run;
}

SolverOptions eigenOptions(SolverMethod method, PreconditionerType pc, bool reuse)
{
    SolverOptions o;
    o.method = method;
    o.preconditioner = pc;
    o.rel_tol = 1e-10;
    o.abs_tol = 1e-14;
    o.max_iter = 100;
    o.krylov_dim = 30;
    o.reuse_preconditioner = reuse;
    return o;
}

TEST(EigenPreconditionerReuse, DisabledReuseKeepsTheDirectPath)
{
    const auto s = makeSystem(gridGraph(4, 4), 3, 71u);
    EigenLinearSolver solver(eigenOptions(SolverMethod::Direct, PreconditionerType::None, false));
    const auto r = runEigen(solver, s);
    EXPECT_EQ(r.report.message, "direct");
    EXPECT_EQ(solver.reuseStats().solves, 0u);
    EXPECT_LT(relativeDifference(r.x, denseSolve(denseMatrix(s), s.rhs)), 1e-12);
}

TEST(EigenPreconditionerReuse, DirectFactorizationIsReusedAcrossNearbySystems)
{
    const auto graph = gridGraph(6, 6);
    EigenLinearSolver solver(eigenOptions(SolverMethod::Direct, PreconditionerType::None, true));
    const auto s0 = makeSystem(graph, 4, 81u);
    const auto r0 = runEigen(solver, s0);
    ASSERT_TRUE(r0.report.converged);
    EXPECT_EQ(r0.report.iterations, 1);
    EXPECT_LT(relativeDifference(r0.x, denseSolve(denseMatrix(s0), s0.rhs)), 1e-12);

    const auto s1 = makeSystem(graph, 4, 81u, 0.01);
    const auto r1 = runEigen(solver, s1);
    ASSERT_TRUE(r1.report.converged);
    EXPECT_LT(r1.report.relative_residual, 1e-10);
    EXPECT_LT(relativeDifference(r1.x, denseSolve(denseMatrix(s1), s1.rhs)), 1e-8);
    const auto st = solver.reuseStats();
    EXPECT_EQ(st.solves, 2u);
    EXPECT_EQ(st.refreshes + st.reuses, 2u);
}

TEST(EigenPreconditionerReuse, StructureChangeAndStaleFailureRefresh)
{
    auto opts = eigenOptions(SolverMethod::Direct, PreconditionerType::None, true);
    opts.max_iter = 1;
    EigenLinearSolver solver(opts);
    const auto a = makeSystem(gridGraph(4, 4), 3, 91u);
    ASSERT_TRUE(runEigen(solver, a).report.converged);
    // Same sparsity, very different values: one iteration with the stale
    // factorization cannot converge, so the solve is repeated after a refresh.
    auto b = makeSystem(gridGraph(4, 4), 3, 92u, 0.9);
    const auto rb = runEigen(solver, b);
    EXPECT_TRUE(rb.report.converged);
    EXPECT_EQ(solver.reuseStats().stale_retries, 1u);
    EXPECT_EQ(solver.reuseStats().refreshes, 2u);
    // Different sparsity: refresh without a failed attempt.
    const auto c = makeSystem(pathGraph(16), 3, 93u);
    EXPECT_TRUE(runEigen(solver, c).report.converged);
    EXPECT_EQ(solver.reuseStats().refreshes, 3u);
    EXPECT_EQ(solver.reuseStats().stale_retries, 1u);
}

TEST(EigenPreconditionerReuse, IlutReuseMatchesDirectSolve)
{
    const auto graph = gridGraph(6, 6);
    EigenLinearSolver solver(eigenOptions(SolverMethod::GMRES, PreconditionerType::ILU, true));
    for (int k = 0; k < 3; ++k) {
        const auto s = makeSystem(graph, 4, 101u, 0.01 * static_cast<double>(k));
        const auto r = runEigen(solver, s);
        ASSERT_TRUE(r.report.converged);
        EXPECT_LT(relativeDifference(r.x, denseSolve(denseMatrix(s), s.rhs)), 1e-8);
    }
    EXPECT_EQ(solver.reuseStats().solves, 3u);
}

#endif // FE_HAS_EIGEN

} // namespace
} // namespace svmp::FE::backends
