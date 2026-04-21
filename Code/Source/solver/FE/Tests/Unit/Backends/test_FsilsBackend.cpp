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
#include "Backends/FSILS/liner_solver/block_schur_strategy_selector.h"

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

void expectVectorMatches(std::span<const Real> actual,
                         std::span<const Real> expected,
                         Real tol = 1e-12)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol) << "mismatch at dof " << i;
    }
}


TEST(FsilsBackendStrategy, SerialStructuredReducedCorrectionsUseExactScalarPath)
{
    fe_fsi_linear_solver::FSILS_lhsType lhs{};
    fe_fsi_linear_solver::distributed_low_rank_correction::Profile profile{};
    profile.distributed = false;
    profile.active_face_corrections = 0;
    profile.active_reduced_corrections = 2;
    profile.active_duplicate_reduced_corrections = 0;
    profile.active_nonduplicate_reduced_corrections = 2;
    profile.has_distinct_multi_reduced_corrections = true;
    profile.reduced_touches_constraint = false;
    profile.grouped_touches_constraint = false;

    const auto selection =
        fe_fsi_linear_solver::BlockSchurStrategySelector::select(lhs, profile, /*con_ncomp=*/1);

    EXPECT_TRUE(selection.require_exact_momentum_low_rank_path);
    EXPECT_FALSE(selection.use_legacy_scalar_schur());
}

TEST(FsilsBackendStrategy, SerialGroupedBorderedCorrectionsUseExactScalarPath)
{
    fe_fsi_linear_solver::FSILS_lhsType lhs{};
    fe_fsi_linear_solver::distributed_low_rank_correction::Profile profile{};
    profile.distributed = false;
    profile.has_grouped_bordered = true;
    profile.active_face_corrections = 0;
    profile.active_reduced_corrections = 2;
    profile.active_duplicate_reduced_corrections = 0;
    profile.active_nonduplicate_reduced_corrections = 2;
    profile.has_distinct_multi_reduced_corrections = true;
    profile.reduced_touches_constraint = false;
    profile.grouped_touches_constraint = false;

    const auto selection =
        fe_fsi_linear_solver::BlockSchurStrategySelector::select(lhs, profile, /*con_ncomp=*/1);

    EXPECT_TRUE(selection.require_exact_momentum_low_rank_path);
    EXPECT_FALSE(selection.use_legacy_scalar_schur());
}

TEST(FsilsBackendStrategy, SingleReducedScalarCorrectionStillUsesLegacyScalarPath)
{
    fe_fsi_linear_solver::FSILS_lhsType lhs{};
    fe_fsi_linear_solver::distributed_low_rank_correction::Profile profile{};
    profile.distributed = false;
    profile.active_face_corrections = 0;
    profile.active_reduced_corrections = 1;
    profile.active_duplicate_reduced_corrections = 0;
    profile.active_nonduplicate_reduced_corrections = 1;
    profile.has_distinct_multi_reduced_corrections = false;
    profile.reduced_touches_constraint = false;
    profile.grouped_touches_constraint = false;

    const auto selection =
        fe_fsi_linear_solver::BlockSchurStrategySelector::select(lhs, profile, /*con_ncomp=*/1);

    EXPECT_FALSE(selection.require_exact_momentum_low_rank_path);
    EXPECT_TRUE(selection.use_momentum_only_low_rank_legacy_scalar_schur);
}

} // namespace

TEST(FsilsBackend, SerialMatrixUsesExplicitOwnedRowLayout)
{
    const auto pattern = make_2x2_pattern();
    FsilsMatrix matrix(pattern);

    ASSERT_NE(matrix.shared(), nullptr);
    EXPECT_TRUE(matrix.usesOwnedRowOperator());
    EXPECT_TRUE(matrix.shared()->lhs.owned_row_operator);
    EXPECT_EQ(matrix.shared()->lhs.mynNo, matrix.shared()->lhs.nNo);
    EXPECT_EQ(matrix.shared()->owned_node_count, matrix.shared()->lhs.nNo);
}

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

TEST(FsilsBackend, CreateLinearSolverNormalizesMetadataDrivenBlockSchurOptions)
{
    FsilsFactory factory(/*dof_per_node=*/3);

    SolverOptions opts{};
    opts.method = SolverMethod::BlockSchur;

    BlockLayout field_layout{};
    field_layout.blocks.push_back({"velocity", 0, 2, BlockRole::PrimaryField});
    field_layout.blocks.push_back({"pressure", 2, 1, BlockRole::ConstraintField});
    field_layout.momentum_block = 0;
    field_layout.constraint_block = 1;
    opts.block_layout = field_layout;

    MixedBlockLayout mixed{};
    mixed.field_unknowns = 9;
    mixed.auxiliary_unknowns = 1;
    mixed.total_unknowns = 10;
    mixed.blocks.push_back({"velocity", 0, 8, BlockRole::PrimaryField, MixedBlockKind::Field});
    mixed.blocks.push_back({"pressure", 8, 1, BlockRole::ConstraintField, MixedBlockKind::Field});
    MixedBlockDescriptor stiff_aux{"stiff_aux",
                                   9,
                                   1,
                                   BlockRole::AuxiliaryField,
                                   MixedBlockKind::Auxiliary,
                                   /*block_diagonal_suitable=*/false,
                                   /*special_precondition=*/true};
    stiff_aux.assembly_mode = MixedBlockAssemblyMode::BorderedReduced;
    stiff_aux.row_ownership = MixedRowOwnershipPolicy::SingleOwner;
    stiff_aux.single_owner_rank = 0;
    mixed.blocks.push_back(stiff_aux);
    mixed.primary_block = 0;
    mixed.constraint_block = 1;
    opts.mixed_block_layout = mixed;

    auto solver = factory.createLinearSolver(opts);
    const auto& stored = solver->getOptions();
    EXPECT_TRUE(stored.fsils_use_rcs);
    EXPECT_EQ(stored.fsils_blockschur_schur_preconditioner,
              FsilsBlockSchurSchurPreconditioner::AlgebraicSchur);
    EXPECT_EQ(stored.fsils_blockschur_momentum_approximation,
              FsilsBlockSchurMomentumApproximation::ASM);
}

TEST(FsilsBackend, CreateLinearSolverRejectsInvalidNativeAuxiliaryPartition)
{
    FsilsFactory factory(/*dof_per_node=*/3);

    SolverOptions opts{};
    MixedBlockLayout mixed{};
    mixed.field_unknowns = 2;
    mixed.auxiliary_unknowns = 1;
    mixed.total_unknowns = 3;

    MixedBlockDescriptor velocity{"velocity", 0, 2, BlockRole::PrimaryField,
                                  MixedBlockKind::Field};
    velocity.node_component_start = 0;
    velocity.node_component_count = 2;
    mixed.blocks.push_back(velocity);

    MixedBlockDescriptor aux{"temperature_aux", 2, 1, BlockRole::AuxiliaryField,
                             MixedBlockKind::Auxiliary};
    aux.assembly_mode = MixedBlockAssemblyMode::NativeOwnedRows;
    aux.row_ownership = MixedRowOwnershipPolicy::BackendDofOwner;
    mixed.blocks.push_back(aux);
    opts.mixed_block_layout = mixed;

    EXPECT_THROW((void)factory.createLinearSolver(opts), InvalidArgumentException);

    mixed.blocks.back().node_component_start = 2;
    mixed.blocks.back().node_component_count = 1;
    opts.mixed_block_layout = mixed;
    EXPECT_NO_THROW((void)factory.createLinearSolver(opts));
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

TEST(FsilsBackend, ReducedFieldUpdateEmptyActiveComponentsMeansAllComponents)
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

    ReducedFieldUpdate upd_all{};
    upd_all.sigma = 650.0;
    upd_all.active_components = {0, 1, 2};
    upd_all.left = {
        {0, 0.10},
        {1, -0.07},
        {2, 0.09},
    };
    upd_all.right = {
        {0, 0.05},
        {1, 0.12},
        {2, -0.08},
    };

    ReducedFieldUpdate upd_empty = upd_all;
    upd_empty.active_components.clear();

    auto x_exact = factory.createVector(3);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    A->mult(*x_exact, *b);
    addReducedFieldContributionSerial(
        factory, *b, *x_exact, std::span<const ReducedFieldUpdate>(&upd_all, 1));

    SolverOptions opts;
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

    auto solve_with_update = [&](const ReducedFieldUpdate& upd)
        -> std::pair<std::shared_ptr<GenericVector>, Real> {
        auto solver = factory.createLinearSolver(opts);
        EXPECT_TRUE(solver->supportsNativeReducedFieldUpdates());
        solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(&upd, 1));
        auto x = factory.createVector(3);
        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_TRUE(rep.converged);
        const Real rel = fullOperatorRelativeResidualSerial(
            factory, *A, *x, *b, std::span<const ReducedFieldUpdate>(&upd_all, 1));
        return std::pair<std::shared_ptr<GenericVector>, Real>(std::move(x), rel);
    };

    const auto [x_all, rel_all] = solve_with_update(upd_all);
    const auto [x_empty, rel_empty] = solve_with_update(upd_empty);

    EXPECT_LE(rel_all, 1e-8);
    EXPECT_LE(rel_empty, 1e-8);

    const auto xs_all = x_all->localSpan();
    const auto xs_empty = x_empty->localSpan();
    ASSERT_EQ(xs_all.size(), xs_empty.size());
    for (std::size_t i = 0; i < xs_all.size(); ++i) {
        EXPECT_NEAR(xs_empty[i], xs_all[i], 1e-10);
    }
}

TEST(FsilsBackend, ReducedFieldUpdateFaceCacheAddBcMulMatchesSparse)
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

    SolverOptions opts;
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

    struct SolveCase {
        std::shared_ptr<GenericVector> x;
        Real rel{0.0};
        SolverReport rep{};
    };

    auto solve_with_face_cache = [&](bool disable_face_cache) {
        std::unique_ptr<ScopedEnvVar> face_cache_guard;
        if (disable_face_cache) {
            face_cache_guard = std::make_unique<ScopedEnvVar>(
                "SVMP_DISABLE_REDUCED_FACE_CACHE_ADD_BC_MUL", "1");
        }

        auto solver = factory.createLinearSolver(opts);
        EXPECT_TRUE(solver->supportsNativeReducedFieldUpdates());
        solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(&upd, 1));
        auto x = factory.createVector(3);
        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_TRUE(rep.converged);
        const Real rel = fullOperatorRelativeResidualSerial(
            factory, *A, *x, *b, std::span<const ReducedFieldUpdate>(&upd, 1));
        return SolveCase{std::move(x), rel, rep};
    };

    const auto cached = solve_with_face_cache(/*disable_face_cache=*/false);
    const auto sparse = solve_with_face_cache(/*disable_face_cache=*/true);

    EXPECT_LE(cached.rel, 1e-8);
    EXPECT_LE(sparse.rel, 1e-8);
    EXPECT_EQ(cached.rep.iterations, sparse.rep.iterations);
    EXPECT_EQ(cached.rep.blockschur_outer_iterations, sparse.rep.blockschur_outer_iterations);
    EXPECT_EQ(cached.rep.blockschur_schur_solve_calls, sparse.rep.blockschur_schur_solve_calls);
    EXPECT_EQ(cached.rep.blockschur_schur_iterations, sparse.rep.blockschur_schur_iterations);

    const auto xs_cached = cached.x->localSpan();
    const auto xs_sparse = sparse.x->localSpan();
    ASSERT_EQ(xs_cached.size(), xs_sparse.size());
    for (std::size_t i = 0; i < xs_cached.size(); ++i) {
        EXPECT_NEAR(xs_cached[i], xs_sparse[i], 1e-10);
    }
}

TEST(FsilsBackend, ReducedFieldUpdateDistributedShapeConverges)
{
    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    FsilsFactory factory(dof);
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

    {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    }
    {
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
    upd.left = {
        {0, 0.03},
        {1, -0.02},
        {6, 0.10},
        {7, -0.07},
    };
    upd.right = {
        {0, 0.05},
        {1, 0.12},
        {6, -0.02},
        {7, 0.08},
    };

    auto x_exact = factory.createVector(n_global);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    A->mult(*x_exact, *b);
    addReducedFieldContributionSerial(
        factory, *b, *x_exact, std::span<const ReducedFieldUpdate>(&upd, 1));

    SolverOptions opts;
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

    auto solver = factory.createLinearSolver(opts);
    ASSERT_TRUE(solver->supportsNativeReducedFieldUpdates());
    solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(&upd, 1));
    auto x = factory.createVector(n_global);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);
    const Real rel = fullOperatorRelativeResidualSerial(
        factory, *A, *x, *b, std::span<const ReducedFieldUpdate>(&upd, 1));
    EXPECT_LE(rel, 1e-8);
}

TEST(FsilsBackend, ReducedFieldUpdateDistributedShapeRhsMatchesReference)
{
    constexpr int dof = 3;
    constexpr GlobalIndex n_nodes = 3;
    constexpr GlobalIndex n_global = n_nodes * dof;

    FsilsFactory factory(dof);
    const auto pattern = make_dense_pattern(n_global);
    auto A = factory.createMatrix(pattern);
    auto b_matrix = factory.createVector(n_global);
    auto b_full = factory.createVector(n_global);

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

    {
        std::vector<GlobalIndex> idx(edof);
        for (int i = 0; i < edof; ++i) idx[static_cast<std::size_t>(i)] = i;
        viewA->addMatrixEntries(std::span<const GlobalIndex>(idx.data(), idx.size()),
                                std::span<const Real>(Ke.data(), Ke.size()),
                                assembly::AddMode::Add);
    }
    {
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
    upd.left = {
        {0, 0.03},
        {1, -0.02},
        {6, 0.10},
        {7, -0.07},
    };
    upd.right = {
        {0, 0.05},
        {1, 0.12},
        {6, -0.02},
        {7, 0.08},
    };

    auto x_exact = factory.createVector(n_global);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }

    A->mult(*x_exact, *b_matrix);
    b_full->copyFrom(*b_matrix);
    addReducedFieldContributionSerial(
        factory, *b_full, *x_exact, std::span<const ReducedFieldUpdate>(&upd, 1));

    static constexpr Real expected_matrix[] = {
        5.0, 3.0, 2.0,
        10.0, 6.0, 4.0,
        5.0, 3.0, 2.0,
    };
    static constexpr Real expected_full[] = {
        15.35, -3.9, 2.0,
        10.0, 6.0, 4.0,
        39.5, -21.15, 2.0,
    };

    expectVectorMatches(b_matrix->localSpan(), expected_matrix);
    expectVectorMatches(b_full->localSpan(), expected_full);
}

TEST(FsilsBackend, ReducedFieldUpdateGroupedWoodburySolversConverge)
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

    std::array<ReducedFieldUpdate, 2> updates{};
    updates[0].sigma = 650.0;
    updates[0].active_components = {0, 1};
    updates[0].left = {
        {0, 0.08},
        {1, -0.05},
    };
    updates[0].right = {
        {0, 0.05},
        {1, 0.11},
    };

    updates[1].sigma = -420.0;
    updates[1].active_components = {0, 1};
    updates[1].left = {
        {0, -0.04},
        {1, 0.09},
    };
    updates[1].right = {
        {0, 0.07},
        {1, -0.03},
    };

    auto x_exact = factory.createVector(3);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    A->mult(*x_exact, *b);
    addReducedFieldContributionSerial(
        factory, *b, *x_exact, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur_reduced_grouped"
                                                        : "gmres_reduced_grouped");
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
        solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));

        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_TRUE(rep.converged);

        const Real rel = fullOperatorRelativeResidualSerial(
            factory, *A, *x, *b, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
        EXPECT_LE(rel, 1e-8);
    }
}

TEST(FsilsBackend, GroupedBorderedFieldCouplingSolversConverge)
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

    // Condensed bordered coupling:
    //   K_eff = K - B * D^{-1} * C
    // The grouped bordered data keeps B, C, and D separate for the Schur-side
    // correction, while the operator itself still sees the exact condensed
    // reduced updates.
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

    std::array<ReducedFieldUpdate, 2> updates{};
    updates[0].sigma = -1.0;
    updates[0].active_components = {0, 1};
    updates[0].grouped_coupling_id = 0;
    updates[0].left = {
        {0, 0.08},
        {1, -0.05},
    };
    updates[0].right = {
        {0, dinv00 * 0.05 + dinv01 * 0.07},
        {1, dinv00 * 0.11 + dinv01 * -0.03},
    };

    updates[1].sigma = -1.0;
    updates[1].active_components = {0, 1};
    updates[1].grouped_coupling_id = 0;
    updates[1].left = {
        {0, -0.04},
        {1, 0.09},
    };
    updates[1].right = {
        {0, dinv10 * 0.05 + dinv11 * 0.07},
        {1, dinv10 * 0.11 + dinv11 * -0.03},
    };

    GroupedBorderedFieldCoupling grouped{};
    grouped.grouped_coupling_id = 0;
    grouped.aux_matrix = {d00, d01,
                          d10, d11};
    grouped.modes.resize(2);
    grouped.modes[0].active_components = {0, 1};
    grouped.modes[0].left = {
        {0, 0.08},
        {1, -0.05},
    };
    grouped.modes[0].right = {
        {0, 0.05},
        {1, 0.11},
    };
    grouped.modes[1].active_components = {0, 1};
    grouped.modes[1].left = {
        {0, -0.04},
        {1, 0.09},
    };
    grouped.modes[1].right = {
        {0, 0.07},
        {1, -0.03},
    };

    auto x_exact = factory.createVector(3);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    A->mult(*x_exact, *b);
    addReducedFieldContributionSerial(
        factory, *b, *x_exact, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur_grouped_bordered"
                                                        : "gmres_grouped_bordered");
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
        solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
        solver->setGroupedBorderedFieldCouplings(std::span<const GroupedBorderedFieldCoupling>(&grouped, 1));

        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_TRUE(rep.converged);

        const Real rel = fullOperatorRelativeResidualSerial(
            factory, *A, *x, *b, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
        EXPECT_LE(rel, 1e-8);
    }
}

TEST(FsilsBackend, GroupedBorderedFieldCouplingAsmMomentumConverges)
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

    std::array<ReducedFieldUpdate, 2> updates{};
    updates[0].sigma = -1.0;
    updates[0].active_components = {0, 1};
    updates[0].grouped_coupling_id = 0;
    updates[0].left = {{0, 0.08}, {1, -0.05}};
    updates[0].right = {
        {0, dinv00 * 0.05 + dinv01 * 0.07},
        {1, dinv00 * 0.11 + dinv01 * -0.03},
    };
    updates[1].sigma = -1.0;
    updates[1].active_components = {0, 1};
    updates[1].grouped_coupling_id = 0;
    updates[1].left = {{0, -0.04}, {1, 0.09}};
    updates[1].right = {
        {0, dinv10 * 0.05 + dinv11 * 0.07},
        {1, dinv10 * 0.11 + dinv11 * -0.03},
    };

    GroupedBorderedFieldCoupling grouped{};
    grouped.grouped_coupling_id = 0;
    grouped.aux_matrix = {d00, d01, d10, d11};
    grouped.modes.resize(2);
    grouped.modes[0].active_components = {0, 1};
    grouped.modes[0].left = {{0, 0.08}, {1, -0.05}};
    grouped.modes[0].right = {{0, 0.05}, {1, 0.11}};
    grouped.modes[1].active_components = {0, 1};
    grouped.modes[1].left = {{0, -0.04}, {1, 0.09}};
    grouped.modes[1].right = {{0, 0.07}, {1, -0.03}};

    auto x_exact = factory.createVector(3);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    A->mult(*x_exact, *b);
    addReducedFieldContributionSerial(
        factory, *b, *x_exact, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));

    SolverOptions opts;
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
    opts.fsils_blockschur_schur_preconditioner =
        FsilsBlockSchurSchurPreconditioner::AlgebraicSchur;
    opts.fsils_blockschur_momentum_approximation =
        FsilsBlockSchurMomentumApproximation::ASM;
    BlockLayout layout;
    layout.blocks.push_back({"u", 0, 2, BlockRole::PrimaryField});
    layout.blocks.push_back({"p", 2, 1, BlockRole::ConstraintField});
    layout.momentum_block = 0;
    layout.constraint_block = 1;
    opts.block_layout = layout;

    auto solver = factory.createLinearSolver(opts);
    ASSERT_TRUE(solver->supportsNativeReducedFieldUpdates());
    solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
    solver->setGroupedBorderedFieldCouplings(std::span<const GroupedBorderedFieldCoupling>(&grouped, 1));

    auto x = factory.createVector(3);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const Real rel = fullOperatorRelativeResidualSerial(
        factory, *A, *x, *b, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
    EXPECT_LE(rel, 1e-8);
}

TEST(FsilsBackend, GroupedBorderedFieldCouplingReusesSchurSetupAcrossRepeatedSolves)
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

    std::array<ReducedFieldUpdate, 2> updates{};
    updates[0].sigma = -1.0;
    updates[0].active_components = {0, 1};
    updates[0].grouped_coupling_id = 0;
    updates[0].left = {{0, 0.08}, {1, -0.05}};
    updates[0].right = {
        {0, dinv00 * 0.05 + dinv01 * 0.07},
        {1, dinv00 * 0.11 + dinv01 * -0.03},
    };
    updates[1].sigma = -1.0;
    updates[1].active_components = {0, 1};
    updates[1].grouped_coupling_id = 0;
    updates[1].left = {{0, -0.04}, {1, 0.09}};
    updates[1].right = {
        {0, dinv10 * 0.05 + dinv11 * 0.07},
        {1, dinv10 * 0.11 + dinv11 * -0.03},
    };

    GroupedBorderedFieldCoupling grouped{};
    grouped.grouped_coupling_id = 0;
    grouped.aux_matrix = {d00, d01, d10, d11};
    grouped.modes.resize(2);
    grouped.modes[0].active_components = {0, 1};
    grouped.modes[0].left = {{0, 0.08}, {1, -0.05}};
    grouped.modes[0].right = {{0, 0.05}, {1, 0.11}};
    grouped.modes[1].active_components = {0, 1};
    grouped.modes[1].left = {{0, -0.04}, {1, 0.09}};
    grouped.modes[1].right = {{0, 0.07}, {1, -0.03}};

    auto x_exact = factory.createVector(3);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    A->mult(*x_exact, *b);
    addReducedFieldContributionSerial(
        factory, *b, *x_exact, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));

    SolverOptions opts;
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

    auto solver = factory.createLinearSolver(opts);
    ASSERT_TRUE(solver->supportsNativeReducedFieldUpdates());
    solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
    solver->setGroupedBorderedFieldCouplings(std::span<const GroupedBorderedFieldCoupling>(&grouped, 1));

    auto x_first = factory.createVector(3);
    auto x_second = factory.createVector(3);
    const auto rep_first = solver->solve(*A, *x_first, *b);
    const auto rep_second = solver->solve(*A, *x_second, *b);

    EXPECT_TRUE(rep_first.converged);
    EXPECT_TRUE(rep_second.converged);
    EXPECT_GT(rep_first.blockschur_schur_solve_calls, 0);
    EXPECT_GT(rep_second.blockschur_schur_solve_calls, 0);
    EXPECT_LE(rep_second.blockschur_schur_setup_time_seconds,
              rep_first.blockschur_schur_setup_time_seconds + 1e-12);
    EXPECT_DOUBLE_EQ(rep_second.blockschur_schur_setup_time_seconds, 0.0);

    const Real rel_first = fullOperatorRelativeResidualSerial(
        factory, *A, *x_first, *b, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
    const Real rel_second = fullOperatorRelativeResidualSerial(
        factory, *A, *x_second, *b, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
    EXPECT_LE(rel_first, 1e-8);
    EXPECT_LE(rel_second, 1e-8);
}

TEST(FsilsBackend, GroupedBorderedCrossBlockCouplingConverges)
{
    FsilsFactory factory(/*dof_per_node=*/4);
    const auto pattern = make_dense_pattern(4);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(4);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[4] = {0, 1, 2, 3};
    const Real Ke[16] = {
        4.0, 0.6, 0.4, 0.2,
        0.6, 3.7, 0.3, 0.5,
        0.4, 0.3, 1.6, 0.2,
        0.2, 0.5, 0.2, 1.4,
    };
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    constexpr Real d00 = 1.8;
    constexpr Real d01 = -0.25;
    constexpr Real d10 = 0.35;
    constexpr Real d11 = 1.4;
    const Real det = d00 * d11 - d01 * d10;
    ASSERT_GT(std::abs(det), Real(1e-12));
    const Real dinv00 = d11 / det;
    const Real dinv01 = -d01 / det;
    const Real dinv10 = -d10 / det;
    const Real dinv11 = d00 / det;

    const std::array<std::array<Real, 4>, 2> c_rows{{
        {{0.03, -0.02, 0.11, 0.04}},
        {{-0.01, 0.05, 0.02, -0.09}},
    }};
    const std::array<std::array<Real, 4>, 2> b_cols{{
        {{0.07, 0.00, 0.05, 0.00}},
        {{0.00, -0.06, 0.00, 0.08}},
    }};

    std::array<ReducedFieldUpdate, 2> updates{};
    for (int i = 0; i < 2; ++i) {
        updates[static_cast<std::size_t>(i)].sigma = -1.0;
        updates[static_cast<std::size_t>(i)].grouped_coupling_id = 0;
        for (int dof_idx = 0; dof_idx < 4; ++dof_idx) {
            const Real left_val = b_cols[static_cast<std::size_t>(i)][static_cast<std::size_t>(dof_idx)];
            if (std::abs(left_val) > Real(1e-30)) {
                updates[static_cast<std::size_t>(i)].left.emplace_back(dof_idx, left_val);
            }
        }
    }
    for (int dof_idx = 0; dof_idx < 4; ++dof_idx) {
        const Real row0 = dinv00 * c_rows[0][static_cast<std::size_t>(dof_idx)] +
                          dinv01 * c_rows[1][static_cast<std::size_t>(dof_idx)];
        const Real row1 = dinv10 * c_rows[0][static_cast<std::size_t>(dof_idx)] +
                          dinv11 * c_rows[1][static_cast<std::size_t>(dof_idx)];
        if (std::abs(row0) > Real(1e-30)) {
            updates[0].right.emplace_back(dof_idx, row0);
        }
        if (std::abs(row1) > Real(1e-30)) {
            updates[1].right.emplace_back(dof_idx, row1);
        }
    }

    GroupedBorderedFieldCoupling grouped{};
    grouped.grouped_coupling_id = 0;
    grouped.aux_matrix = {d00, d01,
                          d10, d11};
    grouped.modes.resize(2);
    for (int i = 0; i < 2; ++i) {
        for (int dof_idx = 0; dof_idx < 4; ++dof_idx) {
            const Real left_val = b_cols[static_cast<std::size_t>(i)][static_cast<std::size_t>(dof_idx)];
            const Real right_val = c_rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(dof_idx)];
            if (std::abs(left_val) > Real(1e-30)) {
                grouped.modes[static_cast<std::size_t>(i)].left.emplace_back(dof_idx, left_val);
            }
            if (std::abs(right_val) > Real(1e-30)) {
                grouped.modes[static_cast<std::size_t>(i)].right.emplace_back(dof_idx, right_val);
            }
        }
    }

    auto x_exact = factory.createVector(4);
    {
        auto xs = x_exact->localSpan();
        std::fill(xs.begin(), xs.end(), Real(1.0));
    }
    A->mult(*x_exact, *b);
    addReducedFieldContributionSerial(
        factory, *b, *x_exact, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));

    const std::array<SolverMethod, 2> methods{
        SolverMethod::BlockSchur,
        SolverMethod::GMRES,
    };

    for (const auto method : methods) {
        SCOPED_TRACE(method == SolverMethod::BlockSchur ? "blockschur_grouped_crossblock"
                                                        : "gmres_grouped_crossblock");
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
            layout.blocks.push_back({"p", 2, 2, BlockRole::ConstraintField});
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
        solver->setReducedFieldUpdates(std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
        solver->setGroupedBorderedFieldCouplings(std::span<const GroupedBorderedFieldCoupling>(&grouped, 1));

        auto x = factory.createVector(4);
        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_TRUE(rep.converged);

        const Real rel = fullOperatorRelativeResidualSerial(
            factory, *A, *x, *b, std::span<const ReducedFieldUpdate>(updates.data(), updates.size()));
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
    EXPECT_FALSE(rep.converged);
    EXPECT_EQ(rep.message.find("fallback"), std::string::npos);
    EXPECT_EQ(rep.iterations, opts.max_iter);

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
    EXPECT_GT(rel, opts.rel_tol);
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
