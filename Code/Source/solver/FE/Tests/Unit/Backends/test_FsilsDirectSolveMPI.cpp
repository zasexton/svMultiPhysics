/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Distributed checks of the FSILS gathered direct solve (SolverMethod::Direct)
// for any number of ranks: the distributed solve reproduces the solve of the
// same operator on one rank bit for bit, enforces Dirichlet DOFs given by any
// rank, keeps its symbolic analysis on every rank alike, and reports singular
// operators and unsupported operator updates collectively.

#include <gtest/gtest.h>

#include "Backends/FSILS/FsilsDirectSolver.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsLinearSolver.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/FEException.h"
#include "FsilsDirectSolveTestUtils.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace svmp::FE::backends {
namespace {

using namespace direct_test;

SolverOptions directOptions()
{
    SolverOptions o;
    o.method = SolverMethod::Direct;
    o.rel_tol = 1e-8;
    o.abs_tol = 1e-10;
    return o;
}

void commRankSize(int& rank, int& size)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

/// Values of x for every owned and ghost DOF of the local overlap set.
std::vector<std::pair<GlobalIndex, Real>> localValues(const ChainOptions& o, const ChainSystem& sys,
                                                      GenericVector& x)
{
    std::vector<std::pair<GlobalIndex, Real>> out;
    auto view = x.createGhostedReadView();
    const GlobalIndex lo = std::max<GlobalIndex>(0, sys.first - 1);
    const GlobalIndex hi = std::min<GlobalIndex>(o.n_nodes - 1, sys.last + 1);
    for (GlobalIndex node = lo; node <= hi; ++node) {
        for (int c = 0; c < kChainDof; ++c) {
            const GlobalIndex dof = node * kChainDof + c;
            out.emplace_back(dof, view->getVectorEntry(dof));
        }
    }
    return out;
}

/// Dirichlet DOFs of the local overlap set only (owned and ghost nodes).
std::vector<GlobalIndex> localDirichlet(const ChainOptions& o, const ChainSystem& sys)
{
    std::vector<GlobalIndex> out;
    for (const auto dof : o.dirichlet) {
        const auto node = dof / kChainDof;
        if (node >= sys.first - 1 && node <= sys.last + 1) {
            out.push_back(dof);
        }
    }
    return out;
}

TEST(FsilsDirectSolveMPI, DistributedSolveMatchesSingleRankSolveBitwise)
{
    int rank = 0;
    int size = 1;
    commRankSize(rank, size);

    ChainOptions o;
    o.n_nodes = 37;
    o.couple_10 = true;
    o.dirichlet = {0, 5 * kChainDof + 2, 10 * kChainDof + 1, 36 * kChainDof + 0, 36 * kChainDof + 1,
                   36 * kChainDof + 2};

    // Reference: the whole operator on this rank alone.
    FsilsFactory self_factory(kChainDof, {}, MPI_COMM_SELF);
    auto ref = buildSerialChain(self_factory, o);
    FsilsLinearSolver ref_solver(directOptions());
    ref_solver.setDirichletDofs(o.dirichlet);
    const auto ref_rep = ref_solver.solve(*ref.A, *ref.x, *ref.b);
    ASSERT_TRUE(ref_rep.converged) << ref_rep.message;

    // Distributed: each rank passes only the Dirichlet DOFs it can see.
    FsilsFactory factory(kChainDof);
    auto sys = buildDistributedChain(factory, o, rank, size);
    FsilsLinearSolver solver(directOptions());
    solver.setDirichletDofs(localDirichlet(o, sys));
    const auto rep = solver.solve(*sys.A, *sys.x, *sys.b);
    EXPECT_TRUE(rep.converged) << rep.message;
    EXPECT_EQ(rep.iterations, ref_rep.iterations);
    EXPECT_EQ(rep.final_residual_norm, ref_rep.final_residual_norm);

    auto ref_view = ref.x->createGhostedReadView();
    for (const auto& [dof, value] : localValues(o, sys, *sys.x)) {
        EXPECT_EQ(value, ref_view->getVectorEntry(dof)) << "dof=" << dof << " ranks=" << size;
        EXPECT_NEAR(value, exactValue(o, dof), 1e-12) << "dof=" << dof;
    }
}

TEST(FsilsDirectSolveMPI, AnalysisReuseIsIdenticalOnEveryRank)
{
    int rank = 0;
    int size = 1;
    commRankSize(rank, size);

    FsilsFactory factory(kChainDof);
    FsilsLinearSolver solver(directOptions());
    ChainOptions o;
    o.n_nodes = 31;
    o.dirichlet = {2, 14 * kChainDof + 0};
    solver.setDirichletDofs(o.dirichlet);  // full list on every rank

    std::vector<long long> analyses;
    for (int step = 0; step < 4; ++step) {
        o.scale = 1.0 + 0.1 * step;
        o.couple_10 = (step >= 2);  // step 2: new component pair
        auto sys = buildDistributedChain(factory, o, rank, size);
        const auto rep = solver.solve(*sys.A, *sys.x, *sys.b);
        EXPECT_TRUE(rep.converged) << rep.message;
        for (const auto& [dof, value] : localValues(o, sys, *sys.x)) {
            EXPECT_NEAR(value, exactValue(o, dof), 1e-12) << "dof=" << dof << " step=" << step;
        }
        analyses.push_back(static_cast<long long>(solver.directSolver()->stats().analyses));
    }
    EXPECT_EQ(analyses, (std::vector<long long>{1, 1, 2, 2}));

    long long local[3] = {static_cast<long long>(solver.directSolver()->stats().analyses),
                          static_cast<long long>(solver.directSolver()->stats().factorizations),
                          static_cast<long long>(solver.directSolver()->stats().failures)};
    long long lo[3] = {0, 0, 0};
    long long hi[3] = {0, 0, 0};
    MPI_Allreduce(local, lo, 3, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local, hi, 3, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(lo[k], hi[k]);
    }
}

TEST(FsilsDirectSolveMPI, MatchesFsilsGmresWithDirichletFaces)
{
    int rank = 0;
    int size = 1;
    commRankSize(rank, size);

    ChainOptions o;
    o.n_nodes = 26;
    o.couple_10 = true;
    o.dirichlet = {1, 7 * kChainDof + 0, 7 * kChainDof + 1, 12 * kChainDof + 2};

    SolverOptions gmres;
    gmres.method = SolverMethod::GMRES;
    gmres.preconditioner = PreconditionerType::RowColumnScaling;
    gmres.fsils_use_rcs = true;
    gmres.rel_tol = 1e-12;
    gmres.abs_tol = 1e-14;
    gmres.max_iter = 500;
    gmres.krylov_dim = 100;

    FsilsFactory factory(kChainDof);
    auto sys_k = buildDistributedChain(factory, o, rank, size);
    FsilsLinearSolver krylov(gmres);
    krylov.setDirichletDofs(localDirichlet(o, sys_k));
    const auto rep_k = krylov.solve(*sys_k.A, *sys_k.x, *sys_k.b);
    EXPECT_TRUE(rep_k.converged) << rep_k.message;

    FsilsFactory factory_d(kChainDof);
    auto sys_d = buildDistributedChain(factory_d, o, rank, size);
    FsilsLinearSolver direct(directOptions());
    direct.setDirichletDofs(localDirichlet(o, sys_d));
    const auto rep_d = direct.solve(*sys_d.A, *sys_d.x, *sys_d.b);
    EXPECT_TRUE(rep_d.converged) << rep_d.message;

    for (const auto& [dof, value] : localValues(o, sys_k, *sys_k.x)) {
        EXPECT_NEAR(value, exactValue(o, dof), 1e-8) << "gmres dof=" << dof;
    }
    for (const auto& [dof, value] : localValues(o, sys_d, *sys_d.x)) {
        EXPECT_NEAR(value, exactValue(o, dof), 1e-12) << "direct dof=" << dof;
    }
}

TEST(FsilsDirectSolveMPI, SingularOperatorAndUpdatesAreReportedOnEveryRank)
{
    int rank = 0;
    int size = 1;
    commRankSize(rank, size);

    FsilsFactory factory(kChainDof);
    FsilsLinearSolver solver(directOptions());
    ChainOptions o;
    o.n_nodes = 19;
    o.zero_row = 9 * kChainDof + 2;
    auto sys = buildDistributedChain(factory, o, rank, size);
    const auto rep = solver.solve(*sys.A, *sys.x, *sys.b);
    EXPECT_FALSE(rep.converged);
    EXPECT_TRUE(rep.numerical_breakdown);
    for (const auto v : sys.x->localSpan()) {
        EXPECT_EQ(v, 0.0);
    }

    // A rank-one update held by rank 0 only must be refused by every rank.
    o.zero_row = -1;
    auto sys2 = buildDistributedChain(factory, o, rank, size);
    RankOneUpdate update;
    update.sigma = 1.0;
    update.v = {{0, 1.0}};
    if (rank == 0) {
        solver.setRankOneUpdates(std::span<const RankOneUpdate>(&update, 1));
    }
    EXPECT_THROW((void)solver.solve(*sys2.A, *sys2.x, *sys2.b), NotImplementedException);
}

} // namespace
} // namespace svmp::FE::backends
