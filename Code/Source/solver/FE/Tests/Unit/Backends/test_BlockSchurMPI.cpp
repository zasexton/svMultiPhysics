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
    expectBlockSchurMetricsPresent(rep);
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
    auto* corr_fs = dynamic_cast<FsilsVector*>(corr.get());
    EXPECT_NE(corr_fs, nullptr);
    if (corr_fs != nullptr) {
        corr_fs->accumulateOverlap();
    }

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

std::vector<Real> collectRankOneDots(const GenericVector& x,
                                     std::span<const RankOneUpdate> updates,
                                     MPI_Comm comm)
{
    std::vector<Real> dots;
    dots.reserve(updates.size());
    for (const auto& update : updates) {
        dots.push_back(sparseRankOneDot(x, update, comm));
    }
    return dots;
}

std::vector<Real> gatherOwnedGlobalVector(const GenericVector& x,
                                          GlobalIndex n_global,
                                          MPI_Comm comm)
{
    const auto* x_fs = dynamic_cast<const FsilsVector*>(&x);
    EXPECT_NE(x_fs, nullptr);

    std::vector<Real> gathered(static_cast<std::size_t>(n_global), Real(0.0));
    if (x_fs == nullptr) {
        return gathered;
    }

    std::vector<GlobalIndex> dofs(static_cast<std::size_t>(n_global));
    for (GlobalIndex i = 0; i < n_global; ++i) {
        dofs[static_cast<std::size_t>(i)] = i;
    }

    std::vector<GlobalIndex> resolved(static_cast<std::size_t>(n_global), INVALID_GLOBAL_INDEX);
    x_fs->resolveEntriesCached(dofs, resolved);

    const auto xs = x_fs->localSpan();
    const auto* shared = x_fs->shared();
    const int dof = (shared != nullptr) ? shared->dof : 1;
    const int owned_node_count = (shared != nullptr) ? shared->owned_node_count
                                                     : static_cast<int>(n_global);

    std::vector<Real> local(static_cast<std::size_t>(n_global), Real(0.0));
    for (std::size_t i = 0; i < resolved.size(); ++i) {
        const auto local_dof = resolved[i];
        if (local_dof == INVALID_GLOBAL_INDEX ||
            local_dof < 0 ||
            local_dof >= static_cast<GlobalIndex>(xs.size())) {
            continue;
        }
        if (shared != nullptr) {
            const int old_node = static_cast<int>(local_dof / dof);
            if (old_node < 0 || old_node >= owned_node_count) {
                continue;
            }
        }
        local[i] = xs[static_cast<std::size_t>(local_dof)];
    }

    MPI_Allreduce(local.data(),
                  gathered.data(),
                  static_cast<int>(n_global),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);
    return gathered;
}

std::vector<Real> gatherOwnedGlobalVectorByOldNode(const GenericVector& x,
                                                   GlobalIndex n_global,
                                                   MPI_Comm comm)
{
    const auto* x_fs = dynamic_cast<const FsilsVector*>(&x);
    EXPECT_NE(x_fs, nullptr);

    std::vector<Real> gathered(static_cast<std::size_t>(n_global), Real(0.0));
    if (x_fs == nullptr) {
        return gathered;
    }

    const auto* shared = x_fs->shared();
    EXPECT_NE(shared, nullptr);
    if (shared == nullptr) {
        return gathered;
    }

    const int dof = shared->dof;
    const int owned_node_count = shared->owned_node_count;
    const auto xs = x_fs->localSpan();

    std::vector<Real> local(static_cast<std::size_t>(n_global), Real(0.0));
    for (int old = 0; old < owned_node_count; ++old) {
        const int global_node = shared->oldToGlobalNode(old);
        if (global_node < 0) {
            continue;
        }
        const std::size_t local_base =
            static_cast<std::size_t>(old) * static_cast<std::size_t>(dof);
        const std::size_t global_base =
            static_cast<std::size_t>(global_node) * static_cast<std::size_t>(dof);
        for (int c = 0; c < dof; ++c) {
            local[global_base + static_cast<std::size_t>(c)] =
                xs[local_base + static_cast<std::size_t>(c)];
        }
    }

    MPI_Allreduce(local.data(),
                  gathered.data(),
                  static_cast<int>(n_global),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);
    return gathered;
}

[[nodiscard]] bool solveDenseSystemInPlace(std::vector<Real>& a,
                                           std::vector<Real>& rhs,
                                           int n)
{
    if (static_cast<int>(a.size()) != n * n || static_cast<int>(rhs.size()) != n) {
        return false;
    }

    for (int pivot = 0; pivot < n; ++pivot) {
        int best = pivot;
        Real best_abs = std::abs(a[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(n) +
                                   static_cast<std::size_t>(pivot)]);
        for (int row = pivot + 1; row < n; ++row) {
            const Real candidate =
                std::abs(a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                           static_cast<std::size_t>(pivot)]);
            if (candidate > best_abs) {
                best_abs = candidate;
                best = row;
            }
        }

        if (!(best_abs > Real(1e-30))) {
            return false;
        }

        if (best != pivot) {
            for (int col = pivot; col < n; ++col) {
                std::swap(a[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(col)],
                          a[static_cast<std::size_t>(best) * static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(col)]);
            }
            std::swap(rhs[static_cast<std::size_t>(pivot)],
                      rhs[static_cast<std::size_t>(best)]);
        }

        const Real pivot_value =
            a[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(pivot)];
        for (int row = pivot + 1; row < n; ++row) {
            const Real factor =
                a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(pivot)] / pivot_value;
            if (std::abs(factor) <= Real(0.0)) {
                continue;
            }
            a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(pivot)] = Real(0.0);
            for (int col = pivot + 1; col < n; ++col) {
                a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(col)] -=
                    factor * a[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(n) +
                               static_cast<std::size_t>(col)];
            }
            rhs[static_cast<std::size_t>(row)] -= factor * rhs[static_cast<std::size_t>(pivot)];
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        Real accum = rhs[static_cast<std::size_t>(row)];
        for (int col = row + 1; col < n; ++col) {
            accum -= a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                       static_cast<std::size_t>(col)] *
                     rhs[static_cast<std::size_t>(col)];
        }
        const Real diag =
            a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(row)];
        if (!(std::abs(diag) > Real(1e-30))) {
            return false;
        }
        rhs[static_cast<std::size_t>(row)] = accum / diag;
    }

    return true;
}

Real denseRelativeResidual(std::span<const Real> a,
                           int n,
                           std::span<const Real> x,
                           std::span<const Real> b)
{
    EXPECT_EQ(static_cast<int>(a.size()), n * n);
    EXPECT_EQ(static_cast<int>(x.size()), n);
    EXPECT_EQ(static_cast<int>(b.size()), n);
    if (static_cast<int>(a.size()) != n * n ||
        static_cast<int>(x.size()) != n ||
        static_cast<int>(b.size()) != n) {
        return std::numeric_limits<Real>::infinity();
    }

    double num_sq = 0.0;
    double den_sq = 0.0;
    for (int row = 0; row < n; ++row) {
        double ax = 0.0;
        for (int col = 0; col < n; ++col) {
            ax += static_cast<double>(a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                                        static_cast<std::size_t>(col)]) *
                  static_cast<double>(x[static_cast<std::size_t>(col)]);
        }
        const double diff = static_cast<double>(b[static_cast<std::size_t>(row)]) - ax;
        num_sq += diff * diff;
        den_sq += static_cast<double>(b[static_cast<std::size_t>(row)]) *
                  static_cast<double>(b[static_cast<std::size_t>(row)]);
    }
    return static_cast<Real>(std::sqrt(num_sq / std::max(den_sq, 1e-60)));
}

std::vector<Real> denseMatVec(std::span<const Real> a,
                              int n,
                              std::span<const Real> x)
{
    std::vector<Real> y(static_cast<std::size_t>(n), Real(0.0));
    EXPECT_EQ(static_cast<int>(a.size()), n * n);
    EXPECT_EQ(static_cast<int>(x.size()), n);
    if (static_cast<int>(a.size()) != n * n || static_cast<int>(x.size()) != n) {
        return y;
    }

    for (int row = 0; row < n; ++row) {
        double ax = 0.0;
        for (int col = 0; col < n; ++col) {
            ax += static_cast<double>(a[static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
                                        static_cast<std::size_t>(col)]) *
                  static_cast<double>(x[static_cast<std::size_t>(col)]);
        }
        y[static_cast<std::size_t>(row)] = static_cast<Real>(ax);
    }
    return y;
}

std::vector<Real> denseRankOneDots(std::span<const Real> x,
                                   std::span<const std::vector<Real>> dense_modes)
{
    std::vector<Real> dots(dense_modes.size(), Real(0.0));
    for (std::size_t mode = 0; mode < dense_modes.size(); ++mode) {
        double dot = 0.0;
        const auto& dense_mode = dense_modes[mode];
        EXPECT_EQ(dense_mode.size(), x.size());
        const std::size_t limit = std::min(dense_mode.size(), x.size());
        for (std::size_t i = 0; i < limit; ++i) {
            dot += static_cast<double>(dense_mode[i]) *
                   static_cast<double>(x[i]);
        }
        dots[mode] = static_cast<Real>(dot);
    }
    return dots;
}

std::vector<Real> sampleDenseOperatorFromBackend(FsilsFactory& factory,
                                                 const GenericMatrix& A,
                                                 GlobalIndex n_global,
                                                 std::span<const RankOneUpdate> updates,
                                                 MPI_Comm comm,
                                                 bool accumulate_output = false)
{
    std::vector<Real> dense_operator(static_cast<std::size_t>(n_global * n_global), Real(0.0));
    for (GlobalIndex col = 0; col < n_global; ++col) {
        auto basis = factory.createVector(n_global);
        auto basis_view = basis->createAssemblyView();
        basis_view->beginAssemblyPhase();
        basis_view->addVectorEntry(col, Real(1.0), assembly::AddMode::Insert);
        basis_view->finalizeAssembly();
        basis->updateGhosts();

        auto y = factory.createVector(n_global);
        A.mult(*basis, *y);
        addRankOneContribution(factory, *y, *basis, updates, comm);
        if (accumulate_output) {
            auto* y_fs = dynamic_cast<FsilsVector*>(y.get());
            EXPECT_NE(y_fs, nullptr);
            if (y_fs != nullptr) {
                y_fs->accumulateOverlap();
            }
        }

        const auto y_global = gatherOwnedGlobalVector(*y, n_global, comm);
        for (GlobalIndex row = 0; row < n_global; ++row) {
            dense_operator[static_cast<std::size_t>(row) * static_cast<std::size_t>(n_global) +
                           static_cast<std::size_t>(col)] =
                y_global[static_cast<std::size_t>(row)];
        }
    }
    return dense_operator;
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
    auto* corr_fs = dynamic_cast<FsilsVector*>(corr.get());
    EXPECT_NE(corr_fs, nullptr);
    if (corr_fs != nullptr) {
        corr_fs->accumulateOverlap();
    }

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

TEST(FsilsBackendMPI, MultiModeNativeRankOneSolversTrackManufacturedModeResponse)
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
        {6.0, 0.7, 0.2,  1.1},
        {0.7, 5.4, 0.4, -0.3},
        {0.2, 0.4, 4.7,  0.5},
        {1.1,-0.3, 0.5,  1.4},
    };
    const Real C[4][4] = {
        {-1.4, 0.1,  0.0, -0.2},
        { 0.1,-1.1,  0.0,  0.2},
        { 0.0, 0.0, -1.0, -0.1},
        {-0.2, 0.2, -0.1,  0.3},
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

    std::array<RankOneUpdate, 2> updates{};
    for (auto& update : updates) {
        update.active_components = {0, 1, 2};
        update.prefer_native_face = true;
    }
    updates[0].sigma = 320.0;
    updates[1].sigma = 240.0;

    if (rank == 0) {
        updates[0].v = {
            {0,  0.06},
            {1, -0.02},
            {2,  0.04},
        };
        updates[1].v = {
            {0, -0.03},
            {1,  0.07},
            {2,  0.02},
        };
    } else {
        updates[0].v = {
            {8,  0.11},
            {9,  0.04},
            {10,-0.06},
        };
        updates[1].v = {
            {8, -0.05},
            {9,  0.09},
            {10, 0.08},
        };
    }

    const std::array<Real, n_global> x_exact_values{
         1.2, -0.7,  0.8, 40.0,
         0.6,  0.5, -1.0, 39.8,
        -1.4,  0.9,  0.7, 40.3,
    };

    auto x_exact = factory.createVector(n_global);
    auto exact_view = x_exact->createAssemblyView();
    exact_view->beginAssemblyPhase();
    for (GlobalIndex dof_idx = owned.first; dof_idx < owned.last; ++dof_idx) {
        exact_view->addVectorEntry(dof_idx,
                                   x_exact_values[static_cast<std::size_t>(dof_idx)],
                                   assembly::AddMode::Insert);
    }
    exact_view->finalizeAssembly();
    x_exact->updateGhosts();

    A->mult(*x_exact, *b);
    addRankOneContribution(factory,
                           *b,
                           *x_exact,
                           std::span<const RankOneUpdate>(updates.data(), updates.size()),
                           MPI_COMM_WORLD);

    auto x_ref = factory.createVector(n_global);
    SolverOptions ref_opts;
    ref_opts.method = SolverMethod::GMRES;
    ref_opts.preconditioner = PreconditionerType::Diagonal;
    ref_opts.rel_tol = 1e-10;
    ref_opts.abs_tol = 1e-14;
    ref_opts.max_iter = 400;
    ref_opts.krylov_dim = 120;
    ref_opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;

    auto ref_solver = factory.createLinearSolver(ref_opts);
    ASSERT_TRUE(ref_solver->supportsNativeRankOneUpdates());
    ref_solver->setRankOneUpdates(std::span<const RankOneUpdate>(updates.data(), updates.size()));
    ref_solver->setEffectiveTimeStep(1.0 / 300.0);

    const auto ref_rep = ref_solver->solve(*A, *x_ref, *b);
    EXPECT_TRUE(ref_rep.converged);
    expectSolverReportSane(ref_rep, ref_opts.max_iter);
    EXPECT_EQ(ref_rep.message.find("fallback"), std::string::npos);

    const Real ref_rel =
        fullOperatorRelativeResidual(factory,
                                     *A,
                                     *x_ref,
                                     *b,
                                     std::span<const RankOneUpdate>(updates.data(), updates.size()),
                                     MPI_COMM_WORLD);
    EXPECT_LE(ref_rel, 1e-10);

    auto x_bs = factory.createVector(n_global);
    auto bs_opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/4,
                                              /*primary_components=*/3,
                                              /*constraint_components=*/1,
                                              FsilsBlockSchurSchurPreconditioner::DiagL,
                                              FsilsBlockSchurMomentumApproximation::DiagK);
    bs_opts.rel_tol = 1e-8;
    bs_opts.abs_tol = 1e-12;
    bs_opts.max_iter = 30;
    bs_opts.krylov_dim = 40;
    bs_opts.fsils_blockschur_gm_max_iter = 160;
    bs_opts.fsils_blockschur_cg_max_iter = 160;
    bs_opts.fsils_blockschur_gm_rel_tol = 1e-10;
    bs_opts.fsils_blockschur_cg_rel_tol = 1e-10;

    auto bs_solver = factory.createLinearSolver(bs_opts);
    ASSERT_TRUE(bs_solver->supportsNativeRankOneUpdates());
    bs_solver->setRankOneUpdates(std::span<const RankOneUpdate>(updates.data(), updates.size()));
    bs_solver->setEffectiveTimeStep(1.0 / 300.0);

    const auto bs_rep = bs_solver->solve(*A, *x_bs, *b);
    EXPECT_TRUE(bs_rep.converged);
    expectSolverReportSane(bs_rep, bs_opts.max_iter);
    EXPECT_EQ(bs_rep.message.find("fallback"), std::string::npos);
    expectBlockSchurMetricsPresent(bs_rep);

    const Real bs_rel =
        fullOperatorRelativeResidual(factory,
                                     *A,
                                     *x_bs,
                                     *b,
                                     std::span<const RankOneUpdate>(updates.data(), updates.size()),
                                     MPI_COMM_WORLD);
    EXPECT_LE(bs_rel, 1e-6);

    const auto ref_dots =
        collectRankOneDots(*x_ref, std::span<const RankOneUpdate>(updates.data(), updates.size()), MPI_COMM_WORLD);
    const auto bs_dots =
        collectRankOneDots(*x_bs, std::span<const RankOneUpdate>(updates.data(), updates.size()), MPI_COMM_WORLD);

    ASSERT_EQ(ref_dots.size(), bs_dots.size());
    for (std::size_t i = 0; i < ref_dots.size(); ++i) {
        EXPECT_GT(std::abs(ref_dots[i]), 1e-3);
        const Real bs_tol = std::max<Real>(1e-6, std::abs(ref_dots[i]) * 2e-3);
        EXPECT_NEAR(bs_dots[i], ref_dots[i], bs_tol);
    }
}

// Targeted parity repro for the shared distributed low-rank / coarse-space path.
// Keep this disabled until BlockSchur recovers near-reference mode response on
// the near-dependent two-mode outlet system.
TEST(FsilsBackendMPI, DISABLED_NearDependentNativeRankOneBlockSchurMatchesReferenceModeResponse)
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
        {6.0, 0.5, 0.2,  1.0},
        {0.5, 5.0, 0.1, -0.8},
        {0.2, 0.1, 4.5,  0.6},
        {1.0,-0.8, 0.6,  1.0e-3},
    };
    const Real C[4][4] = {
        {-1.4, 0.0,  0.0, -0.15},
        { 0.0,-1.1,  0.0,  0.10},
        { 0.0, 0.0, -1.2, -0.08},
        {-0.15, 0.10,-0.08, 0.0},
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

    std::array<RankOneUpdate, 2> updates{};
    for (auto& update : updates) {
        update.active_components = {0, 1, 2};
        update.prefer_native_face = true;
    }
    updates[0].sigma = 400.0;
    updates[1].sigma = 350.0;

    if (rank == 0) {
        updates[0].v = {
            {0, 0.08},
            {1,-0.03},
            {2, 0.02},
        };
        updates[1].v = {
            {0, 0.07},
            {1, 0.04},
            {2,-0.01},
        };
    } else {
        updates[0].v = {
            {8,  0.12},
            {9,  0.05},
            {10,-0.07},
        };
        updates[1].v = {
            {8,  0.11},
            {9,  0.045},
            {10,-0.065},
        };
    }

    const std::array<Real, n_global> x_exact_values{
         1.3, -0.8,  0.9, 150.0,
         0.7,  0.4, -1.2, 149.7,
        -1.6,  1.1,  0.8, 150.4,
    };

    auto x_exact = factory.createVector(n_global);
    auto exact_view = x_exact->createAssemblyView();
    exact_view->beginAssemblyPhase();
    for (GlobalIndex dof_idx = owned.first; dof_idx < owned.last; ++dof_idx) {
        exact_view->addVectorEntry(dof_idx,
                                   x_exact_values[static_cast<std::size_t>(dof_idx)],
                                   assembly::AddMode::Insert);
    }
    exact_view->finalizeAssembly();
    x_exact->updateGhosts();

    A->mult(*x_exact, *b);
    addRankOneContribution(factory,
                           *b,
                           *x_exact,
                           std::span<const RankOneUpdate>(updates.data(), updates.size()),
                           MPI_COMM_WORLD);

    auto x_ref = factory.createVector(n_global);
    SolverOptions ref_opts;
    ref_opts.method = SolverMethod::GMRES;
    ref_opts.preconditioner = PreconditionerType::Diagonal;
    ref_opts.rel_tol = 1e-10;
    ref_opts.abs_tol = 1e-14;
    ref_opts.max_iter = 400;
    ref_opts.krylov_dim = 120;
    ref_opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;

    auto ref_solver = factory.createLinearSolver(ref_opts);
    ASSERT_TRUE(ref_solver->supportsNativeRankOneUpdates());
    ref_solver->setRankOneUpdates(std::span<const RankOneUpdate>(updates.data(), updates.size()));
    ref_solver->setEffectiveTimeStep(1.0 / 300.0);

    const auto ref_rep = ref_solver->solve(*A, *x_ref, *b);
    EXPECT_TRUE(ref_rep.converged);
    expectSolverReportSane(ref_rep, ref_opts.max_iter);
    EXPECT_EQ(ref_rep.message.find("fallback"), std::string::npos);

    const Real ref_rel =
        fullOperatorRelativeResidual(factory,
                                     *A,
                                     *x_ref,
                                     *b,
                                     std::span<const RankOneUpdate>(updates.data(), updates.size()),
                                     MPI_COMM_WORLD);
    EXPECT_LE(ref_rel, 1e-10);

    auto x_bs = factory.createVector(n_global);
    auto bs_opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/4,
                                              /*primary_components=*/3,
                                              /*constraint_components=*/1,
                                              FsilsBlockSchurSchurPreconditioner::DiagL,
                                              FsilsBlockSchurMomentumApproximation::DiagK);
    bs_opts.rel_tol = 1e-8;
    bs_opts.abs_tol = 1e-14;
    bs_opts.max_iter = 20;
    bs_opts.krylov_dim = 40;
    bs_opts.fsils_blockschur_gm_max_iter = 120;
    bs_opts.fsils_blockschur_cg_max_iter = 120;
    bs_opts.fsils_blockschur_gm_rel_tol = 1e-10;
    bs_opts.fsils_blockschur_cg_rel_tol = 1e-10;

    auto bs_solver = factory.createLinearSolver(bs_opts);
    ASSERT_TRUE(bs_solver->supportsNativeRankOneUpdates());
    bs_solver->setRankOneUpdates(std::span<const RankOneUpdate>(updates.data(), updates.size()));
    bs_solver->setEffectiveTimeStep(1.0 / 300.0);

    const auto bs_rep = bs_solver->solve(*A, *x_bs, *b);
    EXPECT_TRUE(bs_rep.converged);
    expectSolverReportSane(bs_rep, bs_opts.max_iter);
    EXPECT_EQ(bs_rep.message.find("fallback"), std::string::npos);
    expectBlockSchurMetricsPresent(bs_rep);

    const Real bs_rel =
        fullOperatorRelativeResidual(factory,
                                     *A,
                                     *x_bs,
                                     *b,
                                     std::span<const RankOneUpdate>(updates.data(), updates.size()),
                                     MPI_COMM_WORLD);
    EXPECT_LE(bs_rel, std::max<Real>(5e-10, ref_rel * 20.0));

    const auto ref_dots =
        collectRankOneDots(*x_ref, std::span<const RankOneUpdate>(updates.data(), updates.size()), MPI_COMM_WORLD);
    const auto bs_dots =
        collectRankOneDots(*x_bs, std::span<const RankOneUpdate>(updates.data(), updates.size()), MPI_COMM_WORLD);

    ASSERT_EQ(ref_dots.size(), bs_dots.size());
    for (std::size_t i = 0; i < ref_dots.size(); ++i) {
        EXPECT_GT(std::abs(ref_dots[i]), 1e-3);
        const Real bs_tol = std::max<Real>(1e-8, std::abs(ref_dots[i]) * 5e-4);
        EXPECT_NEAR(bs_dots[i], ref_dots[i], bs_tol);
    }
}

// Dense-collapse debug harness for the distributed multi-mode low-rank outlet path.
// This is useful for manual backend exploration, but it is not treated as the
// authoritative unit oracle for overlapped distributed semantics.
TEST(FsilsBackendMPI, DISABLED_DistributedRankOneLooseBlockSchurTracksReferenceModeResponse)
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
        {6.0, 0.5, 0.2,  1.0},
        {0.5, 5.0, 0.1, -0.8},
        {0.2, 0.1, 4.5,  0.6},
        {1.0,-0.8, 0.6,  1.0e-3},
    };
    const Real C[4][4] = {
        {-1.4, 0.0,  0.0, -0.15},
        { 0.0,-1.1,  0.0,  0.10},
        { 0.0, 0.0, -1.2, -0.08},
        {-0.15, 0.10,-0.08, 0.0},
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

    std::array<RankOneUpdate, 2> updates{};
    for (auto& update : updates) {
        update.active_components = {0, 1, 2};
        update.prefer_native_face = true;
    }
    updates[0].sigma = 400.0;
    updates[1].sigma = 350.0;

    if (rank == 0) {
        updates[0].v = {
            {0, 0.08},
            {1,-0.03},
            {2, 0.02},
        };
        updates[1].v = {
            {0, 0.07},
            {1, 0.04},
            {2,-0.01},
        };
    } else {
        updates[0].v = {
            {8,  0.12},
            {9,  0.05},
            {10,-0.07},
        };
        updates[1].v = {
            {8,  0.11},
            {9,  0.045},
            {10,-0.065},
        };
    }

    const std::array<Real, n_global> x_exact_values{
        1.3, -0.8,  0.9, 150.0,
        0.7,  0.4, -1.2, 149.7,
       -1.6,  1.1,  0.8, 150.4,
    };

    auto x_exact = factory.createVector(n_global);
    auto exact_view = x_exact->createAssemblyView();
    exact_view->beginAssemblyPhase();
    for (GlobalIndex dof_idx = owned.first; dof_idx < owned.last; ++dof_idx) {
        exact_view->addVectorEntry(dof_idx,
                                   x_exact_values[static_cast<std::size_t>(dof_idx)],
                                   assembly::AddMode::Insert);
    }
    exact_view->finalizeAssembly();
    x_exact->updateGhosts();

    A->mult(*x_exact, *b);
    addRankOneContribution(factory,
                           *b,
                           *x_exact,
                           std::span<const RankOneUpdate>(updates.data(), updates.size()),
                           MPI_COMM_WORLD);

    auto x_ref = factory.createVector(n_global);
    SolverOptions ref_opts;
    ref_opts.method = SolverMethod::GMRES;
    ref_opts.preconditioner = PreconditionerType::Diagonal;
    ref_opts.rel_tol = 1e-10;
    ref_opts.abs_tol = 1e-14;
    ref_opts.max_iter = 400;
    ref_opts.krylov_dim = 120;
    ref_opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;

    auto ref_solver = factory.createLinearSolver(ref_opts);
    ASSERT_TRUE(ref_solver->supportsNativeRankOneUpdates());
    ref_solver->setRankOneUpdates(std::span<const RankOneUpdate>(updates.data(), updates.size()));
    ref_solver->setEffectiveTimeStep(1.0 / 300.0);

    const auto ref_rep = ref_solver->solve(*A, *x_ref, *b);
    EXPECT_TRUE(ref_rep.converged);
    expectSolverReportSane(ref_rep, ref_opts.max_iter);

    const Real ref_rel =
        fullOperatorRelativeResidual(factory,
                                     *A,
                                     *x_ref,
                                     *b,
                                     std::span<const RankOneUpdate>(updates.data(), updates.size()),
                                     MPI_COMM_WORLD);
    EXPECT_LE(ref_rel, 1e-10);

    auto x_bs = factory.createVector(n_global);
    auto bs_opts = makeFsilsBlockSchurOptions(/*dof_per_node=*/4,
                                              /*primary_components=*/3,
                                              /*constraint_components=*/1,
                                              FsilsBlockSchurSchurPreconditioner::DiagL,
                                              FsilsBlockSchurMomentumApproximation::DiagK);
    bs_opts.rel_tol = 1e-3;
    bs_opts.abs_tol = 1e-14;
    bs_opts.max_iter = 15;
    bs_opts.krylov_dim = 40;
    bs_opts.fsils_blockschur_gm_max_iter = 120;
    bs_opts.fsils_blockschur_cg_max_iter = 120;
    bs_opts.fsils_blockschur_gm_rel_tol = 1e-10;
    bs_opts.fsils_blockschur_cg_rel_tol = 1e-10;

    auto bs_solver = factory.createLinearSolver(bs_opts);
    ASSERT_TRUE(bs_solver->supportsNativeRankOneUpdates());
    bs_solver->setRankOneUpdates(std::span<const RankOneUpdate>(updates.data(), updates.size()));
    bs_solver->setEffectiveTimeStep(1.0 / 300.0);

    const auto bs_rep = bs_solver->solve(*A, *x_bs, *b);
    EXPECT_TRUE(bs_rep.converged);
    expectSolverReportSane(bs_rep, bs_opts.max_iter);
    EXPECT_EQ(bs_rep.message.find("fallback"), std::string::npos);
    expectBlockSchurMetricsPresent(bs_rep);

    const Real bs_rel =
        fullOperatorRelativeResidual(factory,
                                     *A,
                                     *x_bs,
                                     *b,
                                     std::span<const RankOneUpdate>(updates.data(), updates.size()),
                                     MPI_COMM_WORLD);
    EXPECT_LE(bs_rel, 2e-3);

    std::vector<std::vector<Real>> dense_modes(updates.size(),
                                               std::vector<Real>(static_cast<std::size_t>(n_global),
                                                                 Real(0.0)));
    for (std::size_t mode = 0; mode < updates.size(); ++mode) {
        std::vector<Real> local_mode(static_cast<std::size_t>(n_global), Real(0.0));
        for (const auto& [dof_idx, value] : updates[mode].v) {
            local_mode[static_cast<std::size_t>(dof_idx)] = value;
        }
        MPI_Allreduce(local_mode.data(),
                      dense_modes[mode].data(),
                      static_cast<int>(n_global),
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
    }

    std::vector<Real> dense_operator(static_cast<std::size_t>(n_global * n_global), Real(0.0));
    auto addDenseElement = [&](GlobalIndex offset) {
        for (int r = 0; r < edof; ++r) {
            for (int c = 0; c < edof; ++c) {
                dense_operator[static_cast<std::size_t>(offset + r) * static_cast<std::size_t>(n_global) +
                               static_cast<std::size_t>(offset + c)] +=
                    Ke[static_cast<std::size_t>(r * edof + c)];
            }
        }
    };
    addDenseElement(0);
    addDenseElement(dof);
    for (std::size_t mode = 0; mode < updates.size(); ++mode) {
        for (int row = 0; row < n_global; ++row) {
            for (int col = 0; col < n_global; ++col) {
                dense_operator[static_cast<std::size_t>(row) * static_cast<std::size_t>(n_global) +
                               static_cast<std::size_t>(col)] +=
                    updates[mode].sigma *
                    dense_modes[mode][static_cast<std::size_t>(row)] *
                    dense_modes[mode][static_cast<std::size_t>(col)];
            }
        }
    }

    const auto backend_dense_operator =
        sampleDenseOperatorFromBackend(factory,
                                       *A,
                                       n_global,
                                       std::span<const RankOneUpdate>(updates.data(), updates.size()),
                                       MPI_COMM_WORLD);
    const auto backend_dense_operator_acc =
        sampleDenseOperatorFromBackend(factory,
                                       *A,
                                       n_global,
                                       std::span<const RankOneUpdate>(updates.data(), updates.size()),
                                       MPI_COMM_WORLD,
                                       /*accumulate_output=*/true);

    Real dense_operator_max_abs_diff = Real(0.0);
    GlobalIndex dense_operator_worst_row = 0;
    GlobalIndex dense_operator_worst_col = 0;
    Real dense_operator_acc_max_abs_diff = Real(0.0);
    GlobalIndex dense_operator_acc_worst_row = 0;
    GlobalIndex dense_operator_acc_worst_col = 0;
    for (GlobalIndex row = 0; row < n_global; ++row) {
        for (GlobalIndex col = 0; col < n_global; ++col) {
            const auto idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(n_global) +
                             static_cast<std::size_t>(col);
            const Real diff = std::abs(dense_operator[idx] - backend_dense_operator[idx]);
            if (diff > dense_operator_max_abs_diff) {
                dense_operator_max_abs_diff = diff;
                dense_operator_worst_row = row;
                dense_operator_worst_col = col;
            }
            const Real acc_diff =
                std::abs(backend_dense_operator_acc[idx] - backend_dense_operator[idx]);
            if (acc_diff > dense_operator_acc_max_abs_diff) {
                dense_operator_acc_max_abs_diff = acc_diff;
                dense_operator_acc_worst_row = row;
                dense_operator_acc_worst_col = col;
            }
        }
    }

    std::vector<Real> dense_exact(x_exact_values.begin(), x_exact_values.end());
    std::vector<Real> dense_rhs(static_cast<std::size_t>(n_global), Real(0.0));
    std::vector<Real> backend_dense_rhs(static_cast<std::size_t>(n_global), Real(0.0));
    std::vector<Real> backend_dense_rhs_acc(static_cast<std::size_t>(n_global), Real(0.0));
    for (int row = 0; row < n_global; ++row) {
        for (int col = 0; col < n_global; ++col) {
            dense_rhs[static_cast<std::size_t>(row)] +=
                dense_operator[static_cast<std::size_t>(row) * static_cast<std::size_t>(n_global) +
                               static_cast<std::size_t>(col)] *
                dense_exact[static_cast<std::size_t>(col)];
            backend_dense_rhs[static_cast<std::size_t>(row)] +=
                backend_dense_operator[static_cast<std::size_t>(row) * static_cast<std::size_t>(n_global) +
                                       static_cast<std::size_t>(col)] *
                dense_exact[static_cast<std::size_t>(col)];
            backend_dense_rhs_acc[static_cast<std::size_t>(row)] +=
                backend_dense_operator_acc[static_cast<std::size_t>(row) * static_cast<std::size_t>(n_global) +
                                           static_cast<std::size_t>(col)] *
                dense_exact[static_cast<std::size_t>(col)];
        }
    }

    auto dense_work = dense_operator;
    auto dense_solution = dense_rhs;
    ASSERT_TRUE(solveDenseSystemInPlace(dense_work, dense_solution, static_cast<int>(n_global)));
    for (int i = 0; i < n_global; ++i) {
        EXPECT_NEAR(dense_solution[static_cast<std::size_t>(i)],
                    dense_exact[static_cast<std::size_t>(i)],
                    1e-11);
    }

    auto backend_dense_work = backend_dense_operator;
    auto backend_dense_solution = backend_dense_rhs;
    ASSERT_TRUE(solveDenseSystemInPlace(
        backend_dense_work, backend_dense_solution, static_cast<int>(n_global)));
    for (int i = 0; i < n_global; ++i) {
        EXPECT_NEAR(backend_dense_solution[static_cast<std::size_t>(i)],
                    dense_exact[static_cast<std::size_t>(i)],
                    1e-11);
    }

    auto backend_dense_work_acc = backend_dense_operator_acc;
    auto backend_dense_solution_acc = backend_dense_rhs_acc;
    ASSERT_TRUE(solveDenseSystemInPlace(
        backend_dense_work_acc, backend_dense_solution_acc, static_cast<int>(n_global)));

    const auto x_exact_global = gatherOwnedGlobalVector(*x_exact, n_global, MPI_COMM_WORLD);
    for (int i = 0; i < n_global; ++i) {
        EXPECT_NEAR(x_exact_global[static_cast<std::size_t>(i)],
                    dense_exact[static_cast<std::size_t>(i)],
                    1e-12);
    }
    const auto b_global = gatherOwnedGlobalVector(*b, n_global, MPI_COMM_WORLD);
    auto b_acc = factory.createVector(n_global);
    b_acc->copyFrom(*b);
    auto* b_acc_fs = dynamic_cast<FsilsVector*>(b_acc.get());
    ASSERT_NE(b_acc_fs, nullptr);
    if (b_acc_fs != nullptr) {
        b_acc_fs->accumulateOverlap();
    }
    const auto b_acc_global = gatherOwnedGlobalVector(*b_acc, n_global, MPI_COMM_WORLD);
    for (int i = 0; i < n_global; ++i) {
        EXPECT_NEAR(b_global[static_cast<std::size_t>(i)],
                    dense_rhs[static_cast<std::size_t>(i)],
                    1e-10)
            << "backend distributed operator does not match dense assembled oracle at rhs entry " << i;
        EXPECT_NEAR(b_global[static_cast<std::size_t>(i)],
                    backend_dense_rhs[static_cast<std::size_t>(i)],
                    1e-10)
            << "backend distributed operator does not match sampled dense backend oracle at rhs entry "
            << i;
        EXPECT_NEAR(b_acc_global[static_cast<std::size_t>(i)],
                    backend_dense_rhs_acc[static_cast<std::size_t>(i)],
                    1e-10)
            << "backend accumulated rhs does not match accumulated sampled dense backend oracle at rhs entry "
            << i;
    }

    const auto x_ref_global = gatherOwnedGlobalVector(*x_ref, n_global, MPI_COMM_WORLD);
    const auto x_bs_global = gatherOwnedGlobalVector(*x_bs, n_global, MPI_COMM_WORLD);
    const auto x_ref_global_old_node =
        gatherOwnedGlobalVectorByOldNode(*x_ref, n_global, MPI_COMM_WORLD);
    const auto x_bs_global_old_node =
        gatherOwnedGlobalVectorByOldNode(*x_bs, n_global, MPI_COMM_WORLD);
    const Real ref_dense_rel =
        denseRelativeResidual(dense_operator, static_cast<int>(n_global), x_ref_global, dense_rhs);
    const Real bs_dense_rel =
        denseRelativeResidual(dense_operator, static_cast<int>(n_global), x_bs_global, dense_rhs);
    const Real ref_backend_dense_rel =
        denseRelativeResidual(backend_dense_operator,
                              static_cast<int>(n_global),
                              x_ref_global,
                              backend_dense_rhs);
    const Real bs_backend_dense_rel =
        denseRelativeResidual(backend_dense_operator,
                              static_cast<int>(n_global),
                              x_bs_global,
                              backend_dense_rhs);
    const Real ref_backend_dense_rel_accop =
        denseRelativeResidual(backend_dense_operator_acc,
                              static_cast<int>(n_global),
                              x_ref_global,
                              backend_dense_rhs_acc);
    const Real bs_backend_dense_rel_accop =
        denseRelativeResidual(backend_dense_operator_acc,
                              static_cast<int>(n_global),
                              x_bs_global,
                              backend_dense_rhs_acc);
    const Real exact_backend_dense_rel_vs_bacc =
        denseRelativeResidual(backend_dense_operator,
                              static_cast<int>(n_global),
                              dense_exact,
                              b_acc_global);
    const Real ref_backend_dense_rel_vs_bacc =
        denseRelativeResidual(backend_dense_operator,
                              static_cast<int>(n_global),
                              x_ref_global,
                              b_acc_global);
    const Real bs_backend_dense_rel_vs_bacc =
        denseRelativeResidual(backend_dense_operator,
                              static_cast<int>(n_global),
                              x_bs_global,
                              b_acc_global);

    auto Ax_ref = factory.createVector(n_global);
    A->mult(*x_ref, *Ax_ref);
    addRankOneContribution(factory,
                           *Ax_ref,
                           *x_ref,
                           std::span<const RankOneUpdate>(updates.data(), updates.size()),
                           MPI_COMM_WORLD);
    const auto Ax_ref_global = gatherOwnedGlobalVector(*Ax_ref, n_global, MPI_COMM_WORLD);
    const auto dense_Ax_ref = denseMatVec(backend_dense_operator, static_cast<int>(n_global), x_ref_global);
    const auto dense_Ax_ref_old_node =
        denseMatVec(backend_dense_operator, static_cast<int>(n_global), x_ref_global_old_node);

    auto Ax_bs = factory.createVector(n_global);
    A->mult(*x_bs, *Ax_bs);
    addRankOneContribution(factory,
                           *Ax_bs,
                           *x_bs,
                           std::span<const RankOneUpdate>(updates.data(), updates.size()),
                           MPI_COMM_WORLD);
    const auto Ax_bs_global = gatherOwnedGlobalVector(*Ax_bs, n_global, MPI_COMM_WORLD);
    const auto dense_Ax_bs = denseMatVec(backend_dense_operator, static_cast<int>(n_global), x_bs_global);
    const auto dense_Ax_bs_old_node =
        denseMatVec(backend_dense_operator, static_cast<int>(n_global), x_bs_global_old_node);

    Real ref_gather_oldnode_max_abs = Real(0.0);
    Real bs_gather_oldnode_max_abs = Real(0.0);
    Real ref_apply_dense_max_abs = Real(0.0);
    Real ref_apply_dense_oldnode_max_abs = Real(0.0);
    Real bs_apply_dense_max_abs = Real(0.0);
    Real bs_apply_dense_oldnode_max_abs = Real(0.0);
    for (int i = 0; i < n_global; ++i) {
        ref_gather_oldnode_max_abs =
            std::max(ref_gather_oldnode_max_abs,
                     std::abs(x_ref_global[static_cast<std::size_t>(i)] -
                              x_ref_global_old_node[static_cast<std::size_t>(i)]));
        bs_gather_oldnode_max_abs =
            std::max(bs_gather_oldnode_max_abs,
                     std::abs(x_bs_global[static_cast<std::size_t>(i)] -
                              x_bs_global_old_node[static_cast<std::size_t>(i)]));
        ref_apply_dense_max_abs =
            std::max(ref_apply_dense_max_abs,
                     std::abs(Ax_ref_global[static_cast<std::size_t>(i)] -
                              dense_Ax_ref[static_cast<std::size_t>(i)]));
        ref_apply_dense_oldnode_max_abs =
            std::max(ref_apply_dense_oldnode_max_abs,
                     std::abs(Ax_ref_global[static_cast<std::size_t>(i)] -
                              dense_Ax_ref_old_node[static_cast<std::size_t>(i)]));
        bs_apply_dense_max_abs =
            std::max(bs_apply_dense_max_abs,
                     std::abs(Ax_bs_global[static_cast<std::size_t>(i)] -
                              dense_Ax_bs[static_cast<std::size_t>(i)]));
        bs_apply_dense_oldnode_max_abs =
            std::max(bs_apply_dense_oldnode_max_abs,
                     std::abs(Ax_bs_global[static_cast<std::size_t>(i)] -
                              dense_Ax_bs_old_node[static_cast<std::size_t>(i)]));
    }

    auto denseResidualAfterSync = [&](const GenericVector& x_in,
                                      bool accumulate_first,
                                      bool update_after,
                                      std::span<const Real> dense_matrix,
                                      std::span<const Real> rhs) -> Real {
        auto tmp = factory.createVector(n_global);
        tmp->copyFrom(x_in);
        auto* tmp_fs = dynamic_cast<FsilsVector*>(tmp.get());
        EXPECT_NE(tmp_fs, nullptr);
        if (tmp_fs == nullptr) {
            return std::numeric_limits<Real>::infinity();
        }
        if (accumulate_first) {
            tmp_fs->accumulateOverlap();
        }
        if (update_after) {
            tmp_fs->updateGhosts();
        }
        const auto tmp_global = gatherOwnedGlobalVector(*tmp_fs, n_global, MPI_COMM_WORLD);
        return denseRelativeResidual(dense_matrix, static_cast<int>(n_global), tmp_global, rhs);
    };
    const Real ref_dense_rel_update = denseResidualAfterSync(
        *x_ref, /*accumulate_first=*/false, /*update_after=*/true, dense_operator, dense_rhs);
    const Real ref_dense_rel_acc = denseResidualAfterSync(
        *x_ref, /*accumulate_first=*/true, /*update_after=*/false, dense_operator, dense_rhs);
    const Real ref_dense_rel_acc_update = denseResidualAfterSync(
        *x_ref, /*accumulate_first=*/true, /*update_after=*/true, dense_operator, dense_rhs);
    const Real bs_dense_rel_update = denseResidualAfterSync(
        *x_bs, /*accumulate_first=*/false, /*update_after=*/true, dense_operator, dense_rhs);
    const Real bs_dense_rel_acc = denseResidualAfterSync(
        *x_bs, /*accumulate_first=*/true, /*update_after=*/false, dense_operator, dense_rhs);
    const Real bs_dense_rel_acc_update = denseResidualAfterSync(
        *x_bs, /*accumulate_first=*/true, /*update_after=*/true, dense_operator, dense_rhs);
    const Real ref_backend_dense_rel_update = denseResidualAfterSync(
        *x_ref,
        /*accumulate_first=*/false,
        /*update_after=*/true,
        backend_dense_operator,
        backend_dense_rhs);
    const Real ref_backend_dense_rel_sync_acc = denseResidualAfterSync(
        *x_ref,
        /*accumulate_first=*/true,
        /*update_after=*/false,
        backend_dense_operator,
        backend_dense_rhs);
    const Real ref_backend_dense_rel_acc_update = denseResidualAfterSync(
        *x_ref,
        /*accumulate_first=*/true,
        /*update_after=*/true,
        backend_dense_operator,
        backend_dense_rhs);
    const Real bs_backend_dense_rel_update = denseResidualAfterSync(
        *x_bs,
        /*accumulate_first=*/false,
        /*update_after=*/true,
        backend_dense_operator,
        backend_dense_rhs);
    const Real bs_backend_dense_rel_sync_acc = denseResidualAfterSync(
        *x_bs,
        /*accumulate_first=*/true,
        /*update_after=*/false,
        backend_dense_operator,
        backend_dense_rhs);
    const Real bs_backend_dense_rel_acc_update = denseResidualAfterSync(
        *x_bs,
        /*accumulate_first=*/true,
        /*update_after=*/true,
        backend_dense_operator,
        backend_dense_rhs);
    const Real ref_backend_dense_rel_accop_update = denseResidualAfterSync(
        *x_ref,
        /*accumulate_first=*/false,
        /*update_after=*/true,
        backend_dense_operator_acc,
        backend_dense_rhs_acc);
    const Real ref_backend_dense_rel_accop_acc = denseResidualAfterSync(
        *x_ref,
        /*accumulate_first=*/true,
        /*update_after=*/false,
        backend_dense_operator_acc,
        backend_dense_rhs_acc);
    const Real ref_backend_dense_rel_accop_acc_update = denseResidualAfterSync(
        *x_ref,
        /*accumulate_first=*/true,
        /*update_after=*/true,
        backend_dense_operator_acc,
        backend_dense_rhs_acc);
    const Real bs_backend_dense_rel_accop_update = denseResidualAfterSync(
        *x_bs,
        /*accumulate_first=*/false,
        /*update_after=*/true,
        backend_dense_operator_acc,
        backend_dense_rhs_acc);
    const Real bs_backend_dense_rel_accop_acc = denseResidualAfterSync(
        *x_bs,
        /*accumulate_first=*/true,
        /*update_after=*/false,
        backend_dense_operator_acc,
        backend_dense_rhs_acc);
    const Real bs_backend_dense_rel_accop_acc_update = denseResidualAfterSync(
        *x_bs,
        /*accumulate_first=*/true,
        /*update_after=*/true,
        backend_dense_operator_acc,
        backend_dense_rhs_acc);

    EXPECT_LE(ref_dense_rel, 1e-8)
        << "distributed GMRES reference solve deviates from dense/direct residual"
        << " raw=" << ref_dense_rel
        << " update=" << ref_dense_rel_update
        << " acc=" << ref_dense_rel_acc
        << " acc_update=" << ref_dense_rel_acc_update;
    EXPECT_LE(bs_dense_rel, 2e-3)
        << "distributed BlockSchur solve deviates from dense/direct residual"
        << " raw=" << bs_dense_rel
        << " update=" << bs_dense_rel_update
        << " acc=" << bs_dense_rel_acc
        << " acc_update=" << bs_dense_rel_acc_update;
    EXPECT_LE(ref_backend_dense_rel, 1e-8)
        << "distributed GMRES reference solve deviates from sampled dense backend residual"
        << " raw=" << ref_backend_dense_rel
        << " update=" << ref_backend_dense_rel_update
        << " acc=" << ref_backend_dense_rel_sync_acc
        << " acc_update=" << ref_backend_dense_rel_acc_update
        << " gather_vs_oldnode=" << ref_gather_oldnode_max_abs
        << " apply_vs_dense=" << ref_apply_dense_max_abs
        << " apply_vs_dense_oldnode=" << ref_apply_dense_oldnode_max_abs
        << " hand_vs_backend_max_abs=" << dense_operator_max_abs_diff
        << " worst_row=" << dense_operator_worst_row
        << " worst_col=" << dense_operator_worst_col;
    EXPECT_LE(bs_backend_dense_rel, 2e-3)
        << "distributed BlockSchur solve deviates from sampled dense backend residual"
        << " raw=" << bs_backend_dense_rel
        << " update=" << bs_backend_dense_rel_update
        << " acc=" << bs_backend_dense_rel_sync_acc
        << " acc_update=" << bs_backend_dense_rel_acc_update
        << " gather_vs_oldnode=" << bs_gather_oldnode_max_abs
        << " apply_vs_dense=" << bs_apply_dense_max_abs
        << " apply_vs_dense_oldnode=" << bs_apply_dense_oldnode_max_abs
        << " hand_vs_backend_max_abs=" << dense_operator_max_abs_diff
        << " worst_row=" << dense_operator_worst_row
        << " worst_col=" << dense_operator_worst_col;
    EXPECT_LE(ref_backend_dense_rel_accop, 1e-8)
        << "distributed GMRES reference solve deviates from accumulated sampled dense backend residual"
        << " raw=" << ref_backend_dense_rel_accop
        << " update=" << ref_backend_dense_rel_accop_update
        << " acc=" << ref_backend_dense_rel_accop_acc
        << " acc_update=" << ref_backend_dense_rel_accop_acc_update
        << " raw_vs_acc_op_max_abs=" << dense_operator_acc_max_abs_diff
        << " worst_row=" << dense_operator_acc_worst_row
        << " worst_col=" << dense_operator_acc_worst_col;
    EXPECT_LE(bs_backend_dense_rel_accop, 2e-3)
        << "distributed BlockSchur solve deviates from accumulated sampled dense backend residual"
        << " raw=" << bs_backend_dense_rel_accop
        << " update=" << bs_backend_dense_rel_accop_update
        << " acc=" << bs_backend_dense_rel_accop_acc
        << " acc_update=" << bs_backend_dense_rel_accop_acc_update
        << " raw_vs_acc_op_max_abs=" << dense_operator_acc_max_abs_diff
        << " worst_row=" << dense_operator_acc_worst_row
        << " worst_col=" << dense_operator_acc_worst_col;
    EXPECT_LE(std::abs(ref_dense_rel - ref_rel), std::max<Real>(1e-10, ref_dense_rel * 1e-1))
        << "distributed residual helper and dense residual disagree for GMRES";
    EXPECT_LE(std::abs(bs_dense_rel - bs_rel), std::max<Real>(1e-6, bs_dense_rel * 1e-1))
        << "distributed residual helper and dense residual disagree for BlockSchur";
    EXPECT_LE(std::abs(ref_backend_dense_rel - ref_rel),
              std::max<Real>(1e-10, ref_backend_dense_rel * 1e-1))
        << "distributed residual helper and sampled dense backend residual disagree for GMRES";
    EXPECT_LE(std::abs(bs_backend_dense_rel - bs_rel),
              std::max<Real>(1e-6, bs_backend_dense_rel * 1e-1))
        << "distributed residual helper and sampled dense backend residual disagree for BlockSchur";
    EXPECT_LE(std::abs(ref_backend_dense_rel_accop - ref_rel),
              std::max<Real>(1e-10, ref_backend_dense_rel_accop * 1e-1))
        << "distributed residual helper and accumulated sampled dense backend residual disagree for GMRES";
    EXPECT_LE(std::abs(bs_backend_dense_rel_accop - bs_rel),
              std::max<Real>(1e-6, bs_backend_dense_rel_accop * 1e-1))
        << "distributed residual helper and accumulated sampled dense backend residual disagree for BlockSchur";

    const auto exact_dots =
        collectRankOneDots(*x_exact, std::span<const RankOneUpdate>(updates.data(), updates.size()), MPI_COMM_WORLD);
    const auto ref_dots =
        collectRankOneDots(*x_ref, std::span<const RankOneUpdate>(updates.data(), updates.size()), MPI_COMM_WORLD);
    const auto bs_dots =
        collectRankOneDots(*x_bs, std::span<const RankOneUpdate>(updates.data(), updates.size()), MPI_COMM_WORLD);
    const auto dense_ref_dots =
        denseRankOneDots(x_ref_global, dense_modes);
    const auto dense_bs_dots =
        denseRankOneDots(x_bs_global, dense_modes);

    ASSERT_EQ(exact_dots.size(), ref_dots.size());
    ASSERT_EQ(exact_dots.size(), bs_dots.size());
    ASSERT_EQ(ref_dots.size(), dense_ref_dots.size());
    ASSERT_EQ(bs_dots.size(), dense_bs_dots.size());
    for (std::size_t i = 0; i < exact_dots.size(); ++i) {
        const Real ref_tol = std::max<Real>(1e-9, std::abs(exact_dots[i]) * 1e-8);
        EXPECT_NEAR(ref_dots[i], exact_dots[i], ref_tol);
        EXPECT_NEAR(ref_dots[i], dense_ref_dots[i], ref_tol);

        const Real bs_tol = std::max<Real>(2e-3, std::abs(ref_dots[i]) * 5e-2);
        EXPECT_NEAR(bs_dots[i], ref_dots[i], bs_tol);
        EXPECT_NEAR(bs_dots[i], dense_bs_dots[i], bs_tol);
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
