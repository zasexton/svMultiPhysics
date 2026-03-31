/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/FEException.h"
#include "Sparsity/SparsityPattern.h"

#include "Backends/FSILS/FsilsFactory.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace svmp::FE::backends {

namespace {

[[maybe_unused]] sparsity::SparsityPattern make_2x2_pattern()
{
    sparsity::SparsityPattern p(2, 2);
    p.addEntry(0, 0);
    p.addEntry(0, 1);
    p.addEntry(1, 0);
    p.addEntry(1, 1);
    p.finalize();
    return p;
}

[[maybe_unused]] sparsity::SparsityPattern make_dense_pattern(GlobalIndex n)
{
    sparsity::SparsityPattern p(n, n);
    for (GlobalIndex r = 0; r < n; ++r) {
        for (GlobalIndex c = 0; c < n; ++c) {
            p.addEntry(r, c);
        }
    }
    p.finalize();
    return p;
}

class ScopedEnvVar final {
public:
    ScopedEnvVar(const char* key, const char* value) : key_(key)
    {
        if (const char* prior = std::getenv(key_)) {
            prior_value_ = std::string(prior);
        }
        ::setenv(key_, value, 1);
    }

    ~ScopedEnvVar()
    {
        if (prior_value_) {
            ::setenv(key_, prior_value_->c_str(), 1);
        } else {
            ::unsetenv(key_);
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* key_{nullptr};
    std::optional<std::string> prior_value_{};
};

void addRankOneContributionSerial(FsilsFactory& factory,
                                  GenericVector& y,
                                  GenericVector& x,
                                  std::span<const RankOneUpdate> updates)
{
    if (updates.empty()) {
        return;
    }

    auto corr = factory.createVector(y.size());
    corr->zero();
    auto view = corr->createAssemblyView();
    ASSERT_NE(view, nullptr);
    view->beginAssemblyPhase();

    auto x_view = x.createAssemblyView();
    ASSERT_NE(x_view, nullptr);
    for (const auto& update : updates) {
        Real dot = 0.0;
        for (const auto& [dof, val] : update.v) {
            dot += val * x_view->getVectorEntry(dof);
        }
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
    ASSERT_EQ(ys.size(), cs.size());
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ys[i] += cs[i];
    }
}

void addReducedFieldContributionSerial(FsilsFactory& factory,
                                       GenericVector& y,
                                       GenericVector& x,
                                       std::span<const ReducedFieldUpdate> updates)
{
    if (updates.empty()) {
        return;
    }

    auto corr = factory.createVector(y.size());
    corr->zero();
    auto view = corr->createAssemblyView();
    ASSERT_NE(view, nullptr);
    view->beginAssemblyPhase();

    auto x_view = x.createAssemblyView();
    ASSERT_NE(x_view, nullptr);
    for (const auto& update : updates) {
        Real dot = 0.0;
        for (const auto& [dof, val] : update.right) {
            dot += val * x_view->getVectorEntry(dof);
        }
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
    ASSERT_EQ(ys.size(), cs.size());
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ys[i] += cs[i];
    }
}

Real fullOperatorRelativeResidualSerial(FsilsFactory& factory,
                                        const GenericMatrix& A,
                                        GenericVector& x,
                                        const GenericVector& b,
                                        std::span<const RankOneUpdate> updates)
{
    auto Ax = factory.createVector(b.size());
    A.mult(x, *Ax);
    addRankOneContributionSerial(factory, *Ax, x, updates);

    auto r = factory.createVector(b.size());
    auto rs = r->localSpan();
    const auto bs = b.localSpan();
    const auto axs = Ax->localSpan();
    EXPECT_EQ(rs.size(), bs.size());
    EXPECT_EQ(rs.size(), axs.size());
    if (rs.size() != bs.size() || rs.size() != axs.size()) {
        return std::numeric_limits<Real>::infinity();
    }
    for (std::size_t i = 0; i < rs.size(); ++i) {
        rs[i] = bs[i] - axs[i];
    }

    return r->norm() / std::max<Real>(b.norm(), 1e-30);
}

Real fullOperatorRelativeResidualSerial(FsilsFactory& factory,
                                        const GenericMatrix& A,
                                        GenericVector& x,
                                        const GenericVector& b,
                                        std::span<const ReducedFieldUpdate> updates)
{
    auto Ax = factory.createVector(b.size());
    A.mult(x, *Ax);
    addReducedFieldContributionSerial(factory, *Ax, x, updates);

    auto r = factory.createVector(b.size());
    auto rs = r->localSpan();
    const auto bs = b.localSpan();
    const auto axs = Ax->localSpan();
    EXPECT_EQ(rs.size(), bs.size());
    EXPECT_EQ(rs.size(), axs.size());
    if (rs.size() != bs.size() || rs.size() != axs.size()) {
        return std::numeric_limits<Real>::infinity();
    }
    for (std::size_t i = 0; i < rs.size(); ++i) {
        rs[i] = bs[i] - axs[i];
    }

    return r->norm() / std::max<Real>(b.norm(), 1e-30);
}

} // namespace

TEST(FsilsBackend, SolveCG2x2)
{
    FsilsFactory factory;
    const auto pattern = make_2x2_pattern();
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(2);
    auto x = factory.createVector(2);

    // A = [[4,1],[1,3]], b = [1,2] => x = [1/11, 7/11]
    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    const Real be[2] = {1.0, 2.0};
    viewb->addVectorEntries(dofs, be, assembly::AddMode::Insert);
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-12;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);
    EXPECT_EQ(rep.message.find("blockschur"), std::string::npos);
    EXPECT_EQ(rep.message.find("fallback"), std::string::npos);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-8);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-8);
}

TEST(FsilsBackend, SolveCGDof2SingleNode)
{
    FsilsFactory factory(/*dof_per_node=*/2);
    const auto pattern = make_dense_pattern(2);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(2);
    auto x = factory.createVector(2);

    // A = [[4,1],[1,3]], b = [1,2] => x = [1/11, 7/11]
    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    const Real be[2] = {1.0, 2.0};
    viewb->addVectorEntries(dofs, be, assembly::AddMode::Insert);
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-12;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-8);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-8);
}

TEST(FsilsBackend, SolveGMRESDof2SingleNode)
{
    FsilsFactory factory(/*dof_per_node=*/2);
    const auto pattern = make_dense_pattern(2);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(2);
    auto x = factory.createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    const Real be[2] = {1.0, 2.0};
    viewb->addVectorEntries(dofs, be, assembly::AddMode::Insert);
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-12;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);
    EXPECT_LE(rep.relative_residual, opts.rel_tol * 10.0);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-8);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-8);

    auto Ax = factory.createVector(2);
    A->mult(*x, *Ax);
    const auto ax = Ax->localSpan();
    const auto bb = b->localSpan();
    Real r2 = 0.0;
    Real b2 = 0.0;
    for (std::size_t i = 0; i < bb.size(); ++i) {
        const Real ri = bb[i] - ax[i];
        r2 += ri * ri;
        b2 += bb[i] * bb[i];
    }
    const Real rel = std::sqrt(r2) / std::max<Real>(std::sqrt(b2), 1e-30);
    EXPECT_LE(rel, opts.rel_tol * 10.0);
}

TEST(FsilsBackend, SolveBlockSchurDof3SingleNode)
{
    FsilsFactory factory(/*dof_per_node=*/3);
    const auto pattern = make_dense_pattern(3);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(3);
    auto x = factory.createVector(3);

    // 2D saddle-point layout per node: (u, v, p), with A = [K D; -G L] and G = -D^T
    // => (-G) = D^T in the assembled matrix.
    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[3] = {0, 1, 2};
    const Real Ke[9] = {4.0, 1.0, 1.0,
                        1.0, 3.0, 0.0,
                        1.0, 0.0, 1.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    const Real be[3] = {1.0, 2.0, 3.0};
    viewb->addVectorEntries(dofs, be, assembly::AddMode::Insert);
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::BlockSchur;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 0.4;
    opts.abs_tol = 0.0;
    opts.max_iter = 10;
    BlockLayout layout;
    layout.blocks.push_back({"u", 0, 2, BlockRole::PrimaryField});
    layout.blocks.push_back({"p", 2, 1, BlockRole::ConstraintField});
    layout.momentum_block = 0;
    layout.constraint_block = 1;
    opts.block_layout = layout;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 3u);

    // The FSILS NS solver is an iterative saddle-point routine; validate by residual reduction.
    auto Ax = factory.createVector(3);
    A->mult(*x, *Ax);
    const auto ax = Ax->localSpan();
    const auto bb = b->localSpan();
    ASSERT_EQ(ax.size(), bb.size());

    Real r2 = 0.0;
    Real b2 = 0.0;
    for (std::size_t i = 0; i < bb.size(); ++i) {
        const Real ri = bb[i] - ax[i];
        r2 += ri * ri;
        b2 += bb[i] * bb[i];
    }
    const Real denom = std::sqrt(b2);
    const Real rel = std::sqrt(r2) / ((denom > 1e-30) ? denom : 1e-30);
    EXPECT_LE(rel, opts.rel_tol + 1e-12);
}

TEST(FsilsBackend, RankOneUpdateSolversConverge)
{
    FsilsFactory factory(/*dof_per_node=*/3);
    const auto pattern = make_dense_pattern(3);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(3);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[3] = {0, 1, 2};
    const Real Ke[9] = {4.0, 1.0, 1.0,
                        1.0, 3.0, 0.0,
                        1.0, 0.0, 1.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    RankOneUpdate upd{};
    upd.sigma = 2000.0;
    upd.active_components = {0, 1};
    upd.v = {
        {0, 0.10},
        {1, 0.05},
    };

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    const Real dot_exact = 0.15;
    const Real scale_exact = upd.sigma * dot_exact;
    const Real be[3] = {
        6.0 + scale_exact * 0.10,
        4.0 + scale_exact * 0.05,
        2.0,
    };
    viewb->addVectorEntries(dofs, be, assembly::AddMode::Insert);
    viewb->finalizeAssembly();

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur" : "gmres");
        auto x = factory.createVector(3);

        SolverOptions opts;
        if (method == SolverMethod::BlockSchur) {
            opts.method = SolverMethod::BlockSchur;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 20;
            opts.krylov_dim = 40;
            opts.fsils_blockschur_gm_max_iter = 120;
            opts.fsils_blockschur_cg_max_iter = 120;
            opts.fsils_blockschur_gm_rel_tol = 1e-10;
            opts.fsils_blockschur_cg_rel_tol = 1e-10;
            BlockLayout layout;
            layout.blocks.push_back({"u", 0, 2, BlockRole::PrimaryField});
            layout.blocks.push_back({"p", 2, 1, BlockRole::ConstraintField});
            layout.momentum_block = 0;
            layout.constraint_block = 1;
            opts.block_layout = layout;
        } else {
            opts.method = SolverMethod::GMRES;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 200;
            opts.krylov_dim = 80;
            opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;
        }

        auto solver = factory.createLinearSolver(opts);
        ASSERT_TRUE(solver->supportsNativeRankOneUpdates());
        solver->setRankOneUpdates(std::span<const RankOneUpdate>(&upd, 1));

        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_TRUE(rep.converged);

        auto Ax = factory.createVector(3);
        A->mult(*x, *Ax);
        auto axs = Ax->localSpan();
        const auto xs = x->localSpan();
        const Real dot = static_cast<Real>(0.10) * xs[0] + static_cast<Real>(0.05) * xs[1];
        axs[0] += upd.sigma * dot * static_cast<Real>(0.10);
        axs[1] += upd.sigma * dot * static_cast<Real>(0.05);

        const auto bb = b->localSpan();
        Real r2 = 0.0;
        Real b2 = 0.0;
        for (std::size_t i = 0; i < bb.size(); ++i) {
            const Real ri = bb[i] - axs[i];
            r2 += ri * ri;
            b2 += bb[i] * bb[i];
        }
        const Real rel = std::sqrt(r2) / std::max<Real>(std::sqrt(b2), 1e-30);
        EXPECT_LE(rel, 1e-8);
    }
}

TEST(FsilsBackend, RankOneUpdateSolversConverge4DOF)
{
    constexpr int dof = 4;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    FsilsFactory factory(/*dof_per_node=*/dof);
    const auto pattern = make_dense_pattern(n_global);
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

    const std::array<GlobalIndex, 8> edofs0 = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::array<GlobalIndex, 8> edofs1 = {4, 5, 6, 7, 8, 9, 10, 11};
    viewA->addMatrixEntries(edofs0, Ke, assembly::AddMode::Add);
    viewA->addMatrixEntries(edofs1, Ke, assembly::AddMode::Add);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    RankOneUpdate upd{};
    upd.sigma = 1600.0;
    upd.active_components = {0, 1, 2};
    upd.v = {
        {8,  0.10},
        {9,  0.05},
        {10, -0.08},
    };

    auto x_exact = factory.createVector(n_global);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    A->mult(*x_exact, *b);
    addRankOneContributionSerial(factory, *b, *x_exact, std::span<const RankOneUpdate>(&upd, 1));

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur_4dof" : "gmres_4dof");
        auto x = factory.createVector(n_global);

        SolverOptions opts;
        if (method == SolverMethod::BlockSchur) {
            opts.method = SolverMethod::BlockSchur;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 30;
            opts.krylov_dim = 40;
            opts.fsils_blockschur_gm_max_iter = 160;
            opts.fsils_blockschur_cg_max_iter = 160;
            opts.fsils_blockschur_gm_rel_tol = 1e-10;
            opts.fsils_blockschur_cg_rel_tol = 1e-10;
            BlockLayout layout;
            layout.blocks.push_back({"u", 0, 3, BlockRole::PrimaryField});
            layout.blocks.push_back({"p", 3, 1, BlockRole::ConstraintField});
            layout.momentum_block = 0;
            layout.constraint_block = 1;
            opts.block_layout = layout;
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

        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_TRUE(rep.converged);
        EXPECT_EQ(rep.message.find("fallback"), std::string::npos);

        const Real rel = fullOperatorRelativeResidualSerial(factory, *A, *x, *b,
                                                            std::span<const RankOneUpdate>(&upd, 1));
        EXPECT_LE(rel, 1e-8);
    }
}

TEST(FsilsBackend, ReducedFieldUpdateSolversConverge)
{
    FsilsFactory factory(/*dof_per_node=*/3);
    const auto pattern = make_dense_pattern(3);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(3);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[3] = {0, 1, 2};
    const Real Ke[9] = {4.0, 1.0, 1.0,
                        1.0, 3.0, 0.0,
                        1.0, 0.0, 1.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    ReducedFieldUpdate upd{};
    upd.sigma = 800.0;
    upd.active_components = {0, 1};
    upd.left = {
        {0, 0.10},
        {1, -0.07},
    };
    upd.right = {
        {0, 0.05},
        {1, 0.12},
    };

    auto x_exact = factory.createVector(3);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    A->mult(*x_exact, *b);
    addReducedFieldContributionSerial(
        factory, *b, *x_exact, std::span<const ReducedFieldUpdate>(&upd, 1));

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur_reduced" : "gmres_reduced");
        auto x = factory.createVector(3);

        SolverOptions opts;
        if (method == SolverMethod::BlockSchur) {
            opts.method = SolverMethod::BlockSchur;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 20;
            opts.krylov_dim = 40;
            opts.fsils_blockschur_gm_max_iter = 120;
            opts.fsils_blockschur_cg_max_iter = 120;
            opts.fsils_blockschur_gm_rel_tol = 1e-10;
            opts.fsils_blockschur_cg_rel_tol = 1e-10;
            BlockLayout layout;
            layout.blocks.push_back({"u", 0, 2, BlockRole::PrimaryField});
            layout.blocks.push_back({"p", 2, 1, BlockRole::ConstraintField});
            layout.momentum_block = 0;
            layout.constraint_block = 1;
            opts.block_layout = layout;
        } else {
            opts.method = SolverMethod::GMRES;
            opts.preconditioner = PreconditionerType::Diagonal;
            opts.rel_tol = 1e-8;
            opts.abs_tol = 1e-12;
            opts.max_iter = 200;
            opts.krylov_dim = 80;
            opts.fsils_residual_check_policy = FsilsResidualCheckPolicy::RetryOnly;
        }

        auto solver = factory.createLinearSolver(opts);
        ASSERT_TRUE(solver->supportsNativeReducedFieldUpdates());
        solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(&upd, 1));

        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_TRUE(rep.converged);

        const Real rel = fullOperatorRelativeResidualSerial(
            factory, *A, *x, *b, std::span<const ReducedFieldUpdate>(&upd, 1));
        EXPECT_LE(rel, 1e-8);
    }
}

TEST(FsilsBackend, SolveRestartedGmresHardMatrix)
{
    ScopedEnvVar gmres_sd("SVMP_FSILS_GMRES_SD", "4");

    constexpr GlobalIndex n = 16;
    FsilsFactory factory;
    const auto pattern = make_dense_pattern(n);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(n);
    auto x = factory.createVector(n);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    std::array<GlobalIndex, n> dofs{};
    for (GlobalIndex i = 0; i < n; ++i) {
        dofs[static_cast<std::size_t>(i)] = i;
    }

    std::array<Real, n * n> Ke{};
    for (GlobalIndex i = 0; i < n; ++i) {
        Ke[static_cast<std::size_t>(i * n + i)] = 1.0;
        if (i > 0) {
            Ke[static_cast<std::size_t>(i * n + (i - 1))] = -1.0;
        }
        for (GlobalIndex j = i + 1; j <= std::min<GlobalIndex>(n - 1, i + 3); ++j) {
            Ke[static_cast<std::size_t>(i * n + j)] = 1.0;
        }
    }
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    std::array<Real, n> be{};
    for (GlobalIndex i = 0; i < n; ++i) {
        Real sum = 0.0;
        for (GlobalIndex j = 0; j < n; ++j) {
            sum += Ke[static_cast<std::size_t>(i * n + j)];
        }
        be[static_cast<std::size_t>(i)] = sum;
    }
    viewb->addVectorEntries(dofs, be, assembly::AddMode::Insert);
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-6;
    opts.abs_tol = 1e-10;
    opts.max_iter = 80;
    opts.krylov_dim = 4;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);
    EXPECT_NE(rep.message.find("fallback gmres"), std::string::npos);
    EXPECT_LT(rep.iterations, opts.max_iter);

    auto Ax = factory.createVector(n);
    A->mult(*x, *Ax);
    const auto ax = Ax->localSpan();
    const auto bb = b->localSpan();
    Real r2 = 0.0;
    Real b2 = 0.0;
    for (std::size_t i = 0; i < bb.size(); ++i) {
        const Real ri = bb[i] - ax[i];
        r2 += ri * ri;
        b2 += bb[i] * bb[i];
    }
    const Real rel = std::sqrt(r2) / std::max<Real>(std::sqrt(b2), 1e-30);
    EXPECT_LE(rel, opts.rel_tol * 10.0);
}

TEST(FsilsBackend, ResolvedMatrixEntriesContiguousBlockMatchesDirectAdd)
{
    FsilsFactory factory(/*dof_per_node=*/2);
    const auto pattern = make_dense_pattern(4); // two nodes, two components per node

    auto direct = factory.createMatrix(pattern);
    auto resolved = factory.createMatrix(pattern);

    auto* direct_fsils = dynamic_cast<FsilsMatrix*>(direct.get());
    auto* resolved_fsils = dynamic_cast<FsilsMatrix*>(resolved.get());
    ASSERT_NE(direct_fsils, nullptr);
    ASSERT_NE(resolved_fsils, nullptr);

    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    const std::array<Real, 16> local = {
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    };

    auto direct_view = direct->createAssemblyView();
    direct_view->beginAssemblyPhase();
    direct_view->addMatrixEntries(dofs, local, assembly::AddMode::Add);
    direct_view->finalizeAssembly();
    direct->finalizeAssembly();

    std::vector<GlobalIndex> slots(local.size(), INVALID_GLOBAL_INDEX);
    resolved_fsils->resolveMatrixEntrySlotsCached(dofs, dofs, std::span<GlobalIndex>(slots));
    resolved_fsils->addResolvedMatrixEntries(
        dofs, dofs, std::span<const GlobalIndex>(slots), local, assembly::AddMode::Add);
    resolved->finalizeAssembly();

    for (GlobalIndex row = 0; row < 4; ++row) {
        for (GlobalIndex col = 0; col < 4; ++col) {
            EXPECT_DOUBLE_EQ(direct_fsils->getEntry(row, col),
                             resolved_fsils->getEntry(row, col))
                << "row=" << row << " col=" << col;
        }
    }
}

TEST(FsilsBackend, ResolvedMatrixEntriesIrregularInsertMatchesDirectInsert)
{
    FsilsFactory factory(/*dof_per_node=*/2);
    const auto pattern = make_dense_pattern(4);

    auto direct = factory.createMatrix(pattern);
    auto resolved = factory.createMatrix(pattern);

    auto* direct_fsils = dynamic_cast<FsilsMatrix*>(direct.get());
    auto* resolved_fsils = dynamic_cast<FsilsMatrix*>(resolved.get());
    ASSERT_NE(direct_fsils, nullptr);
    ASSERT_NE(resolved_fsils, nullptr);

    const std::array<GlobalIndex, 3> row_dofs = {0, 2, 3};
    const std::array<GlobalIndex, 3> col_dofs = {1, 2, 99}; // invalid final column exercises fallback skipping
    const std::array<Real, 9> local = {
        1.5, 2.5, 3.5,
        4.5, 5.5, 6.5,
        7.5, 8.5, 9.5,
    };

    auto direct_view = direct->createAssemblyView();
    direct_view->beginAssemblyPhase();
    direct_view->addMatrixEntries(row_dofs, col_dofs, local, assembly::AddMode::Insert);
    direct_view->finalizeAssembly();
    direct->finalizeAssembly();

    std::vector<GlobalIndex> slots(local.size(), INVALID_GLOBAL_INDEX);
    resolved_fsils->resolveMatrixEntrySlotsCached(
        row_dofs, col_dofs, std::span<GlobalIndex>(slots));
    resolved_fsils->addResolvedMatrixEntries(
        row_dofs, col_dofs, std::span<const GlobalIndex>(slots), local,
        assembly::AddMode::Insert);
    resolved->finalizeAssembly();

    for (GlobalIndex row = 0; row < 4; ++row) {
        for (GlobalIndex col = 0; col < 4; ++col) {
            EXPECT_DOUBLE_EQ(direct_fsils->getEntry(row, col),
                             resolved_fsils->getEntry(row, col))
                << "row=" << row << " col=" << col;
        }
    }
}

TEST(FsilsBackend, ResolvedVectorEntriesContiguousBlockMatchesDirectAdd)
{
    FsilsFactory factory(/*dof_per_node=*/2);

    auto direct = factory.createVector(4);
    auto resolved = factory.createVector(4);

    auto* resolved_fsils = dynamic_cast<FsilsVector*>(resolved.get());
    ASSERT_NE(resolved_fsils, nullptr);

    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    const std::array<Real, 4> local = {1.0, 2.0, 3.0, 4.0};

    auto direct_view = direct->createAssemblyView();
    direct_view->beginAssemblyPhase();
    direct_view->addVectorEntries(dofs, local, assembly::AddMode::Add);
    direct_view->finalizeAssembly();

    std::vector<GlobalIndex> resolved_slots(dofs.size(), INVALID_GLOBAL_INDEX);
    resolved_fsils->resolveEntriesCached(dofs, std::span<GlobalIndex>(resolved_slots));

    auto resolved_view = resolved->createAssemblyView();
    resolved_view->beginAssemblyPhase();
    resolved_view->addVectorEntriesResolved(
        dofs,
        std::span<const GlobalIndex>(resolved_slots),
        local,
        assembly::AddMode::Add);
    resolved_view->finalizeAssembly();

    const auto direct_vals = direct->localSpan();
    const auto resolved_vals = resolved->localSpan();
    ASSERT_EQ(direct_vals.size(), resolved_vals.size());
    for (std::size_t i = 0; i < direct_vals.size(); ++i) {
        EXPECT_DOUBLE_EQ(resolved_vals[i], direct_vals[i]);
    }
}

TEST(FsilsBackend, ResolvedVectorEntriesIrregularInsertMatchesDirectInsert)
{
    FsilsFactory factory(/*dof_per_node=*/2);

    auto direct = factory.createVector(6);
    auto resolved = factory.createVector(6);

    auto* resolved_fsils = dynamic_cast<FsilsVector*>(resolved.get());
    ASSERT_NE(resolved_fsils, nullptr);

    const std::array<GlobalIndex, 4> dofs = {0, 3, 2, 5};
    const std::array<Real, 4> local = {9.0, 8.0, 7.0, 6.0};

    auto direct_view = direct->createAssemblyView();
    direct_view->beginAssemblyPhase();
    direct_view->addVectorEntries(dofs, local, assembly::AddMode::Insert);
    direct_view->finalizeAssembly();

    std::vector<GlobalIndex> resolved_slots(dofs.size(), INVALID_GLOBAL_INDEX);
    resolved_fsils->resolveEntriesCached(dofs, std::span<GlobalIndex>(resolved_slots));

    auto resolved_view = resolved->createAssemblyView();
    resolved_view->beginAssemblyPhase();
    resolved_view->addVectorEntriesResolved(
        dofs,
        std::span<const GlobalIndex>(resolved_slots),
        local,
        assembly::AddMode::Insert);
    resolved_view->finalizeAssembly();

    const auto direct_vals = direct->localSpan();
    const auto resolved_vals = resolved->localSpan();
    ASSERT_EQ(direct_vals.size(), resolved_vals.size());
    for (std::size_t i = 0; i < direct_vals.size(); ++i) {
        EXPECT_DOUBLE_EQ(resolved_vals[i], direct_vals[i]);
    }
}

} // namespace svmp::FE::backends
