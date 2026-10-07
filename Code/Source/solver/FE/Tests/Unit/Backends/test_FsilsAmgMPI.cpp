/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Checks of the FSILS aggregation multigrid right preconditioner (Amg): the
// same 2D block operator numbered and split across the ranks in two different
// ways gives the same aggregates and, up to round-off, the same solution, and
// it needs far fewer GMRES iterations than the row-column scaling alone.

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/FSILS/FsilsAmg.h"
#include "Backends/FSILS/FsilsBlockPreconditioners.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsLinearSolver.h"
#include "Backends/Interfaces/DofPermutation.h"
#include "Backends/Utils/BackendOptions.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

namespace svmp::FE::backends {
namespace {

using sparsity::DistributedSparsityPattern;
using sparsity::IndexRange;

constexpr int kDof = 2;
constexpr int kNx = 37;
constexpr int kNy = 31;
constexpr GlobalIndex kNodes = static_cast<GlobalIndex>(kNx) * kNy;

/// Node id of grid point (ix, iy) in the row-major or column-major numbering.
GlobalIndex nodeId(int ix, int iy, bool row_major)
{
    return row_major ? static_cast<GlobalIndex>(iy) * kNx + ix : static_cast<GlobalIndex>(ix) * kNy + iy;
}

void gridPoint(GlobalIndex id, bool row_major, int& ix, int& iy)
{
    if (row_major) {
        iy = static_cast<int>(id / kNx);
        ix = static_cast<int>(id % kNx);
    } else {
        ix = static_cast<int>(id / kNy);
        iy = static_cast<int>(id % kNy);
    }
}

/// Physical (numbering-independent) index of a grid point.
std::size_t physical(int ix, int iy)
{
    return static_cast<std::size_t>(iy) * kNx + static_cast<std::size_t>(ix);
}

std::vector<std::pair<int, int>> stencil(int ix, int iy)
{
    std::vector<std::pair<int, int>> nb{{ix, iy}};
    if (ix > 0) nb.emplace_back(ix - 1, iy);
    if (ix + 1 < kNx) nb.emplace_back(ix + 1, iy);
    if (iy > 0) nb.emplace_back(ix, iy - 1);
    if (iy + 1 < kNy) nb.emplace_back(ix, iy + 1);
    return nb;
}

/// Row-major 2 x 2 block coupling grid point i to j: a diagonally dominant,
/// non-symmetric (convection-like) coupled Laplacian.
void block(int ix, int iy, int jx, int jy, Real* B)
{
    if (ix == jx && iy == jy) {
        B[0] = 4.5;
        B[1] = 0.3;
        B[2] = 0.2;
        B[3] = 4.53;
        return;
    }
    const Real sx = static_cast<Real>(jx - ix);
    const Real sy = static_cast<Real>(jy - iy);
    B[0] = -1.0 - 0.15 * sx;
    B[1] = 0.05;
    B[2] = 0.02 * sy;
    B[3] = -1.0 + 0.1 * sy;
}

/// Exact solution; zero on the Dirichlet unknowns (component 0 on the left
/// edge, component 1 on the bottom edge), so that it also solves the masked system.
Real exactValue(int ix, int iy, int c)
{
    if ((c == 0 && ix == 0) || (c == 1 && iy == 0)) {
        return 0.0;
    }
    const Real x = static_cast<Real>(ix) / (kNx - 1);
    const Real y = static_cast<Real>(iy) / (kNy - 1);
    return c == 0 ? std::sin(3.0 * x + 0.5) * std::cos(2.0 * y) : x * y + 0.25 * std::sin(5.0 * y);
}

struct GridSystem {
    std::unique_ptr<FsilsFactory> factory;
    std::unique_ptr<GenericMatrix> A;
    std::unique_ptr<GenericVector> b;
    std::unique_ptr<GenericVector> x;
    IndexRange owned_nodes{};
    std::vector<GlobalIndex> dirichlet{};
    bool row_major{true};
};

GridSystem buildGrid(bool row_major, int rank, int size)
{
    GridSystem sys;
    sys.row_major = row_major;
    const GlobalIndex first = kNodes * rank / size;
    const GlobalIndex last = kNodes * (rank + 1) / size;
    sys.owned_nodes = IndexRange{first, last};

    auto perm = std::make_shared<DofPermutation>();
    perm->forward.resize(static_cast<std::size_t>(kNodes * kDof));
    std::iota(perm->forward.begin(), perm->forward.end(), GlobalIndex{0});
    perm->inverse = perm->forward;
    perm->node_key.resize(static_cast<std::size_t>(kNodes));
    for (GlobalIndex id = 0; id < kNodes; ++id) {
        int ix = 0, iy = 0;
        gridPoint(id, row_major, ix, iy);
        perm->node_key[static_cast<std::size_t>(id)] =
            nodeKeyFromCoordinates(static_cast<double>(ix), static_cast<double>(iy), 0.0);
    }
    sys.factory = std::make_unique<FsilsFactory>(kDof, perm);

    const GlobalIndex n_global = kNodes * kDof;
    const IndexRange owned{first * kDof, last * kDof};
    DistributedSparsityPattern pattern(owned, owned, n_global, n_global);
    std::set<GlobalIndex> ghosts;
    for (GlobalIndex id = first; id < last; ++id) {
        int ix = 0, iy = 0;
        gridPoint(id, row_major, ix, iy);
        for (int r = 0; r < kDof; ++r) {
            for (const auto& [jx, jy] : stencil(ix, iy)) {
                const GlobalIndex jd = nodeId(jx, jy, row_major);
                if (jd < first || jd >= last) {
                    ghosts.insert(jd);
                }
                for (int c = 0; c < kDof; ++c) {
                    pattern.addEntry(id * kDof + r, jd * kDof + c);
                }
            }
        }
    }
    pattern.ensureDiagonal();
    pattern.finalize();
    // Ghost rows restricted to the local nodes (owned and ghost).
    auto local = [&](GlobalIndex jd) { return (jd >= first && jd < last) || ghosts.count(jd) > 0; };
    std::vector<GlobalIndex> ghost_rows, ghost_ptr{0}, ghost_cols;
    for (const auto gd : ghosts) {
        int ix = 0, iy = 0;
        gridPoint(gd, row_major, ix, iy);
        for (int r = 0; r < kDof; ++r) {
            ghost_rows.push_back(gd * kDof + r);
            for (const auto& [jx, jy] : stencil(ix, iy)) {
                const GlobalIndex jd = nodeId(jx, jy, row_major);
                if (!local(jd)) {
                    continue;
                }
                for (int c = 0; c < kDof; ++c) {
                    ghost_cols.push_back(jd * kDof + c);
                }
            }
            ghost_ptr.push_back(static_cast<GlobalIndex>(ghost_cols.size()));
        }
    }
    pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_ptr), std::move(ghost_cols));

    sys.A = sys.factory->createMatrix(pattern);
    sys.b = sys.factory->createVector(n_global);
    sys.x = sys.factory->createVector(n_global);
    auto viewA = sys.A->createAssemblyView();
    auto viewb = sys.b->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewb->beginAssemblyPhase();
    for (GlobalIndex id = first; id < last; ++id) {
        int ix = 0, iy = 0;
        gridPoint(id, row_major, ix, iy);
        const auto nb = stencil(ix, iy);
        std::vector<GlobalIndex> rows, cols;
        for (int r = 0; r < kDof; ++r) {
            rows.push_back(id * kDof + r);
        }
        for (const auto& [jx, jy] : nb) {
            for (int c = 0; c < kDof; ++c) {
                cols.push_back(nodeId(jx, jy, row_major) * kDof + c);
            }
        }
        std::vector<Real> values(rows.size() * cols.size(), 0.0);
        std::vector<Real> rhs(kDof, 0.0);
        Real B[kDof * kDof];
        for (std::size_t k = 0; k < nb.size(); ++k) {
            block(ix, iy, nb[k].first, nb[k].second, B);
            for (int r = 0; r < kDof; ++r) {
                for (int c = 0; c < kDof; ++c) {
                    values[static_cast<std::size_t>(r) * cols.size() + k * kDof + static_cast<std::size_t>(c)] =
                        B[r * kDof + c];
                    rhs[static_cast<std::size_t>(r)] += B[r * kDof + c] * exactValue(nb[k].first, nb[k].second, c);
                }
            }
        }
        viewA->addMatrixEntries(rows, cols, values, assembly::AddMode::Insert);
        viewb->addVectorEntries(rows, rhs, assembly::AddMode::Insert);
        // Dirichlet unknowns: component 0 on the left edge, component 1 on the bottom edge.
        if (ix == 0) {
            sys.dirichlet.push_back(id * kDof + 0);
        }
        if (iy == 0) {
            sys.dirichlet.push_back(id * kDof + 1);
        }
    }
    viewA->finalizeAssembly();
    viewb->finalizeAssembly();
    sys.A->finalizeAssembly();
    return sys;
}

SolverOptions gmresOptions(RightPreconditionerType pc)
{
    SolverOptions o;
    o.method = SolverMethod::GMRES;
    o.preconditioner = PreconditionerType::RowColumnScaling;
    o.fsils_use_rcs = true;
    o.rel_tol = 1e-10;
    o.abs_tol = 1e-14;
    o.max_iter = 200;
    o.krylov_dim = 50;
    o.right_preconditioner = pc;
    o.amg_coarse_nodes = 40;
    return o;
}

struct SolveResult {
    int iterations{0};
    bool converged{false};
    std::vector<Real> x{};              // physical order, kDof per point, on every rank
    std::vector<Real> exact{};          // same layout
    std::vector<long long> roots{};     // root key of each point's aggregate (Amg only)
};

SolveResult solveGrid(bool row_major, RightPreconditionerType pc, int rank, int size)
{
    auto sys = buildGrid(row_major, rank, size);
    FsilsLinearSolver solver(gmresOptions(pc));
    solver.setDirichletDofs(sys.dirichlet);
    const auto rep = solver.solve(*sys.A, *sys.x, *sys.b);
    SolveResult out;
    out.iterations = rep.iterations;
    out.converged = rep.converged;
    std::vector<Real> local(static_cast<std::size_t>(kNodes * kDof), 0.0);
    sys.x->updateGhosts();
    auto read = sys.x->createGhostedReadView();
    for (GlobalIndex id = sys.owned_nodes.first; id < sys.owned_nodes.last; ++id) {
        int ix = 0, iy = 0;
        gridPoint(id, row_major, ix, iy);
        for (int c = 0; c < kDof; ++c) {
            local[physical(ix, iy) * kDof + static_cast<std::size_t>(c)] = read->getVectorEntry(id * kDof + c);
        }
    }
    out.x.assign(local.size(), 0.0);
    out.exact.assign(local.size(), 0.0);
    for (int iy = 0; iy < kNy; ++iy) {
        for (int ix = 0; ix < kNx; ++ix) {
            for (int c = 0; c < kDof; ++c) {
                out.exact[physical(ix, iy) * kDof + static_cast<std::size_t>(c)] = exactValue(ix, iy, c);
            }
        }
    }
    MPI_Allreduce(local.data(), out.x.data(), static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (pc == RightPreconditionerType::Amg) {
        const auto* kp = solver.krylovPreconditioner();
        EXPECT_NE(kp, nullptr);
        if (kp != nullptr && kp->amgHierarchy() != nullptr) {
            const auto roots = kp->amgHierarchy()->finestAggregatesForTesting();
            std::vector<long long> lr(static_cast<std::size_t>(kNodes), 0);
            EXPECT_EQ(static_cast<GlobalIndex>(roots.size()), sys.owned_nodes.last - sys.owned_nodes.first);
            for (GlobalIndex id = sys.owned_nodes.first;
                 id < sys.owned_nodes.last && static_cast<std::size_t>(id - sys.owned_nodes.first) < roots.size();
                 ++id) {
                int ix = 0, iy = 0;
                gridPoint(id, row_major, ix, iy);
                lr[physical(ix, iy)] = roots[static_cast<std::size_t>(id - sys.owned_nodes.first)];
            }
            out.roots.assign(lr.size(), 0);
            MPI_Allreduce(lr.data(), out.roots.data(), static_cast<int>(lr.size()), MPI_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD);
        }
    }
    return out;
}

double maxRelDiff(const std::vector<Real>& a, const std::vector<Real>& b)
{
    double d = 0.0, s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(a[i] - b[i]));
        s = std::max(s, std::abs(a[i]));
    }
    return s > 0.0 ? d / s : d;
}

} // namespace

TEST(FsilsAmgMPI, AggregatesAndSolutionDoNotDependOnThePartition)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto amg_rows = solveGrid(true, RightPreconditionerType::Amg, rank, size);
    const auto amg_cols = solveGrid(false, RightPreconditionerType::Amg, rank, size);
    const auto rcs_rows = solveGrid(true, RightPreconditionerType::None, rank, size);

    EXPECT_TRUE(amg_rows.converged);
    EXPECT_TRUE(amg_cols.converged);
    EXPECT_TRUE(rcs_rows.converged);

    // Same aggregates for every grid point, whatever the numbering and split.
    ASSERT_EQ(amg_rows.roots.size(), static_cast<std::size_t>(kNodes));
    ASSERT_EQ(amg_cols.roots.size(), static_cast<std::size_t>(kNodes));
    std::size_t differing = 0, without = 0;
    for (std::size_t i = 0; i < amg_rows.roots.size(); ++i) {
        differing += amg_rows.roots[i] != amg_cols.roots[i] ? 1u : 0u;
        without += amg_rows.roots[i] == 0 ? 1u : 0u;
    }
    EXPECT_EQ(differing, 0u);
    // Only the corner point, whose unknowns are all Dirichlet, has no
    // couplings and therefore no aggregate (the smoother solves it).
    EXPECT_EQ(without, 1u);
    EXPECT_EQ(amg_rows.roots[physical(0, 0)], 0);

    // Same iterations and round-off-level agreement between the two splits.
    EXPECT_EQ(amg_rows.iterations, amg_cols.iterations);
    EXPECT_LT(maxRelDiff(amg_rows.x, amg_cols.x), 1e-11);
    // The exact solution, and the scaling-only solve, within the linear tolerance.
    EXPECT_LT(maxRelDiff(amg_rows.exact, amg_rows.x), 1e-7);
    EXPECT_LT(maxRelDiff(rcs_rows.exact, rcs_rows.x), 1e-7);
    EXPECT_LT(maxRelDiff(rcs_rows.x, amg_rows.x), 1e-7);
    // Far fewer iterations than the scaling alone.
    EXPECT_LT(3 * amg_rows.iterations, rcs_rows.iterations)
        << "amg " << amg_rows.iterations << " rcs " << rcs_rows.iterations;
    if (rank == 0) {
        std::cout << "[FsilsAmgMPI] ranks=" << size << " iterations amg_rows=" << amg_rows.iterations
                  << " amg_cols=" << amg_cols.iterations << " rcs=" << rcs_rows.iterations
                  << " rel_diff_splits=" << maxRelDiff(amg_rows.x, amg_cols.x)
                  << " rel_diff_rcs=" << maxRelDiff(rcs_rows.x, amg_rows.x)
                  << " rel_err_amg=" << maxRelDiff(amg_rows.exact, amg_rows.x) << std::endl;
    }
}

TEST(FsilsAmgMPI, SmoothedAggregationAndPowerBoundAlsoConverge)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto sys = buildGrid(true, rank, size);
    auto opts = gmresOptions(RightPreconditionerType::Amg);
    opts.amg_smooth_prolongator = true;
    opts.amg_lambda_iterations = 10;
    opts.amg_smoother_degree = 2;
    FsilsLinearSolver solver(opts);
    solver.setDirichletDofs(sys.dirichlet);
    const auto rep = solver.solve(*sys.A, *sys.x, *sys.b);
    EXPECT_TRUE(rep.converged) << rep.message;
    EXPECT_LT(rep.iterations, 40);
}

} // namespace svmp::FE::backends
