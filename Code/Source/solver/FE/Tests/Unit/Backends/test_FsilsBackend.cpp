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
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/FEException.h"
#include "Sparsity/SparsityPattern.h"

#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsLinearSolver.h"
#include "Backends/FSILS/liner_solver/bcast.h"
#include "Backends/FSILS/liner_solver/block_schur_strategy_selector.h"
#include "Backends/FSILS/liner_solver/dot.h"
#include "Backends/FSILS/liner_solver/gmres.h"
#include "Backends/FSILS/liner_solver/norm.h"
#include "Backends/FSILS/liner_solver/ns_solver.h"
#include "Backends/FSILS/liner_solver/omp_la.h"
#include "Backends/FSILS/liner_solver/precond.h"
#include "Backends/FSILS/liner_solver/spar_mul.h"

#include "Array.h"
#include "Array3.h"
#include "Vector.h"

#include <algorithm>
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

class ScopedUnsetEnvVar final {
public:
    explicit ScopedUnsetEnvVar(const char* key) : key_(key)
    {
        if (const char* prior = std::getenv(key_)) {
            prior_value_ = std::string(prior);
        }
        ::unsetenv(key_);
    }

    ~ScopedUnsetEnvVar()
    {
        if (prior_value_) {
            ::setenv(key_, prior_value_->c_str(), 1);
        }
    }

    ScopedUnsetEnvVar(const ScopedUnsetEnvVar&) = delete;
    ScopedUnsetEnvVar& operator=(const ScopedUnsetEnvVar&) = delete;

private:
    const char* key_{nullptr};
    std::optional<std::string> prior_value_{};
};

template <typename T>
class ScopedVectorLegacyStatics final {
public:
    ScopedVectorLegacyStatics()
        : num_allocated_(Vector<T>::num_allocated)
        , active_(Vector<T>::active)
        , memory_in_use_(Vector<T>::memory_in_use)
        , memory_returned_(Vector<T>::memory_returned)
        , write_enabled_(Vector<T>::write_enabled)
        , show_index_check_message_(Vector<T>::show_index_check_message)
    {
    }

    ~ScopedVectorLegacyStatics()
    {
        Vector<T>::num_allocated = num_allocated_;
        Vector<T>::active = active_;
        Vector<T>::memory_in_use = memory_in_use_;
        Vector<T>::memory_returned = memory_returned_;
        Vector<T>::write_enabled = write_enabled_;
        Vector<T>::show_index_check_message = show_index_check_message_;
    }

    ScopedVectorLegacyStatics(const ScopedVectorLegacyStatics&) = delete;
    ScopedVectorLegacyStatics& operator=(const ScopedVectorLegacyStatics&) = delete;

private:
    int num_allocated_{0};
    int active_{0};
    double memory_in_use_{0.0};
    double memory_returned_{0.0};
    bool write_enabled_{false};
    bool show_index_check_message_{true};
};

template <typename T>
class ScopedArrayLegacyStatics final {
public:
    ScopedArrayLegacyStatics()
        : id_(Array<T>::id)
        , num_allocated_(Array<T>::num_allocated)
        , active_(Array<T>::active)
        , memory_in_use_(Array<T>::memory_in_use)
        , memory_returned_(Array<T>::memory_returned)
        , write_enabled_(Array<T>::write_enabled)
        , show_index_check_message_(Array<T>::show_index_check_message)
    {
    }

    ~ScopedArrayLegacyStatics()
    {
        Array<T>::id = id_;
        Array<T>::num_allocated = num_allocated_;
        Array<T>::active = active_;
        Array<T>::memory_in_use = memory_in_use_;
        Array<T>::memory_returned = memory_returned_;
        Array<T>::write_enabled = write_enabled_;
        Array<T>::show_index_check_message = show_index_check_message_;
    }

    ScopedArrayLegacyStatics(const ScopedArrayLegacyStatics&) = delete;
    ScopedArrayLegacyStatics& operator=(const ScopedArrayLegacyStatics&) = delete;

private:
    int id_{0};
    int num_allocated_{0};
    int active_{0};
    double memory_in_use_{0.0};
    double memory_returned_{0.0};
    bool write_enabled_{false};
    bool show_index_check_message_{true};
};

template <typename T>
class ScopedArray3LegacyStatics final {
public:
    ScopedArray3LegacyStatics()
        : num_allocated_(Array3<T>::num_allocated)
        , active_(Array3<T>::active)
        , memory_in_use_(Array3<T>::memory_in_use)
        , memory_returned_(Array3<T>::memory_returned)
        , write_enabled_(Array3<T>::write_enabled)
        , show_index_check_message_(Array3<T>::show_index_check_message)
    {
    }

    ~ScopedArray3LegacyStatics()
    {
        Array3<T>::num_allocated = num_allocated_;
        Array3<T>::active = active_;
        Array3<T>::memory_in_use = memory_in_use_;
        Array3<T>::memory_returned = memory_returned_;
        Array3<T>::write_enabled = write_enabled_;
        Array3<T>::show_index_check_message = show_index_check_message_;
    }

    ScopedArray3LegacyStatics(const ScopedArray3LegacyStatics&) = delete;
    ScopedArray3LegacyStatics& operator=(const ScopedArray3LegacyStatics&) = delete;

private:
    int num_allocated_{0};
    int active_{0};
    double memory_in_use_{0.0};
    double memory_returned_{0.0};
    bool write_enabled_{false};
    bool show_index_check_message_{true};
};

template <typename T>
void resetVectorLegacyStatics()
{
    Vector<T>::num_allocated = 0;
    Vector<T>::active = 0;
    Vector<T>::memory_in_use = 0.0;
    Vector<T>::memory_returned = 0.0;
    Vector<T>::write_enabled = false;
    Vector<T>::show_index_check_message = true;
}

template <typename T>
void resetArrayLegacyStatics()
{
    Array<T>::id = 0;
    Array<T>::num_allocated = 0;
    Array<T>::active = 0;
    Array<T>::memory_in_use = 0.0;
    Array<T>::memory_returned = 0.0;
    Array<T>::write_enabled = false;
    Array<T>::show_index_check_message = true;
}

template <typename T>
void resetArray3LegacyStatics()
{
    Array3<T>::num_allocated = 0;
    Array3<T>::active = 0;
    Array3<T>::memory_in_use = 0.0;
    Array3<T>::memory_returned = 0.0;
    Array3<T>::write_enabled = false;
    Array3<T>::show_index_check_message = true;
}

std::shared_ptr<DofPermutation> makeIdentityPermutation(GlobalIndex size)
{
    auto permutation = std::make_shared<DofPermutation>();
    permutation->forward.resize(static_cast<std::size_t>(size));
    permutation->inverse.resize(static_cast<std::size_t>(size));
    permutation->owner_rank.resize(static_cast<std::size_t>(size), 0);
    for (GlobalIndex i = 0; i < size; ++i) {
        permutation->forward[static_cast<std::size_t>(i)] = i;
        permutation->inverse[static_cast<std::size_t>(i)] = i;
    }
    return permutation;
}

std::shared_ptr<DofPermutation> makePermutation(std::vector<GlobalIndex> forward)
{
    auto permutation = std::make_shared<DofPermutation>();
    permutation->forward = std::move(forward);
    permutation->inverse.assign(permutation->forward.size(), INVALID_GLOBAL_INDEX);
    permutation->owner_rank.assign(permutation->forward.size(), 0);
    for (GlobalIndex fe_dof = 0; fe_dof < static_cast<GlobalIndex>(permutation->forward.size()); ++fe_dof) {
        const auto backend_dof = permutation->forward[static_cast<std::size_t>(fe_dof)];
        if (backend_dof < 0 ||
            static_cast<std::size_t>(backend_dof) >= permutation->inverse.size()) {
            continue;
        }
        permutation->inverse[static_cast<std::size_t>(backend_dof)] = fe_dof;
    }
    return permutation;
}

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

struct DenseFsilsCsrFixture {
    fe_fsi_linear_solver::FSILS_lhsType lhs{};
    Array<fe_fsi_linear_solver::fsils_int> row_ptr;
    Vector<fe_fsi_linear_solver::fsils_int> col_ptr;
    Vector<fe_fsi_linear_solver::fsils_int> diag_ptr;
    std::vector<fe_fsi_linear_solver::fsils_int> entry_row;
};

DenseFsilsCsrFixture makeDenseFsilsCsr(fe_fsi_linear_solver::fsils_int n_nodes,
                                       fe_fsi_linear_solver::fsils_int owned_nodes)
{
    DenseFsilsCsrFixture fixture{
        .row_ptr = Array<fe_fsi_linear_solver::fsils_int>(2, static_cast<int>(n_nodes)),
        .col_ptr = Vector<fe_fsi_linear_solver::fsils_int>(static_cast<int>(n_nodes * n_nodes)),
        .diag_ptr = Vector<fe_fsi_linear_solver::fsils_int>(static_cast<int>(n_nodes)),
        .entry_row = std::vector<fe_fsi_linear_solver::fsils_int>(static_cast<std::size_t>(n_nodes * n_nodes), 0),
    };

    fixture.lhs.gnNo = n_nodes;
    fixture.lhs.nNo = n_nodes;
    fixture.lhs.mynNo = owned_nodes;
    fixture.lhs.nnz = n_nodes * n_nodes;
    fixture.lhs.nFaces = 0;
    fixture.lhs.commu.task = 0;
    fixture.lhs.commu.master = 0;
    fixture.lhs.commu.masF = 1;
    fixture.lhs.commu.nTasks = 1;
    fixture.lhs.commu.comm = MPI_COMM_SELF;

    fe_fsi_linear_solver::fsils_int entry = 0;
    for (fe_fsi_linear_solver::fsils_int row = 0; row < n_nodes; ++row) {
        fixture.row_ptr(0, static_cast<int>(row)) = entry;
        for (fe_fsi_linear_solver::fsils_int col = 0; col < n_nodes; ++col) {
            fixture.col_ptr(static_cast<int>(entry)) = col;
            fixture.entry_row[static_cast<std::size_t>(entry)] = row;
            if (row == col) {
                fixture.diag_ptr(static_cast<int>(row)) = entry;
            }
            ++entry;
        }
        fixture.row_ptr(1, static_cast<int>(row)) = entry - 1;
    }

    return fixture;
}

std::vector<Real> denseVvReference(const DenseFsilsCsrFixture& fixture,
                                   int dof,
                                   const Array<Real>& values,
                                   const Array<Real>& input)
{
    std::vector<Real> expected(static_cast<std::size_t>(fixture.lhs.nNo * dof), 0.0);
    for (fe_fsi_linear_solver::fsils_int row = 0; row < fixture.lhs.mynNo; ++row) {
        for (fe_fsi_linear_solver::fsils_int entry = fixture.row_ptr(0, static_cast<int>(row));
             entry <= fixture.row_ptr(1, static_cast<int>(row));
             ++entry) {
            const auto col = fixture.col_ptr(static_cast<int>(entry));
            for (int out = 0; out < dof; ++out) {
                Real sum = 0.0;
                for (int in = 0; in < dof; ++in) {
                    sum += values(out * dof + in, static_cast<int>(entry)) *
                           input(in, static_cast<int>(col));
                }
                expected[static_cast<std::size_t>(row * dof + out)] += sum;
            }
        }
    }
    return expected;
}

void expectArrayMatchesFlat(const Array<Real>& actual,
                            std::span<const Real> expected,
                            Real tol = 1e-12)
{
    ASSERT_EQ(static_cast<std::size_t>(actual.size()), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual(static_cast<int>(i)), expected[i], tol) << "flat index " << i;
    }
}

fe_fsi_linear_solver::FSILS_commuType makeSelfCommu()
{
    fe_fsi_linear_solver::FSILS_commuType commu{};
    commu.task = 0;
    commu.master = 0;
    commu.masF = 1;
    commu.nTasks = 1;
    commu.comm = MPI_COMM_SELF;
    return commu;
}

Real arrayValue(int node, int comp, Real seed)
{
    const Real signed_scale = ((node + comp) % 2 == 0) ? Real(1.0) : Real(-1.0);
    return signed_scale * (seed + Real(0.37) * Real(node + 1) + Real(0.11) * Real(comp + 2));
}

Real referenceDotOwned(const Array<Real>& a,
                       const Array<Real>& b,
                       int dof,
                       int owned_nodes)
{
    Real sum = 0.0;
    for (int node = 0; node < owned_nodes; ++node) {
        for (int comp = 0; comp < dof; ++comp) {
            sum += a(comp, node) * b(comp, node);
        }
    }
    return sum;
}

Real referenceNormOwned(const Array<Real>& a, int dof, int owned_nodes)
{
    return std::sqrt(referenceDotOwned(a, a, dof, owned_nodes));
}

void fillGmresBasis(Array3<Real>& basis, int dof, int n_nodes, int n_slices)
{
    for (int s = 0; s < n_slices; ++s) {
        for (int node = 0; node < n_nodes; ++node) {
            for (int comp = 0; comp < dof; ++comp) {
                basis(comp, node, s) =
                    arrayValue(node, comp, Real(0.2) * Real(s + 1)) +
                    Real(0.03) * Real((s + 1) * (comp + 1));
            }
        }
    }
}

TEST(FsilsBackendLegacyKernels, EffectiveTimeStepStoresPositiveFiniteValuesAndFallsBack)
{
    SolverOptions opts;
    FsilsLinearSolver solver(opts);

    EXPECT_NEAR(solver.effectiveTimeStepForTesting(), 1.0, 0.0);

    solver.setEffectiveTimeStep(0.125);
    EXPECT_NEAR(solver.effectiveTimeStepForTesting(), 0.125, 0.0);

    solver.setEffectiveTimeStep(0.0);
    EXPECT_NEAR(solver.effectiveTimeStepForTesting(), 1.0, 0.0);

    solver.setEffectiveTimeStep(-2.0);
    EXPECT_NEAR(solver.effectiveTimeStepForTesting(), 1.0, 0.0);

    solver.setEffectiveTimeStep(std::numeric_limits<double>::infinity());
    EXPECT_NEAR(solver.effectiveTimeStepForTesting(), 1.0, 0.0);

    solver.setEffectiveTimeStep(std::numeric_limits<double>::quiet_NaN());
    EXPECT_NEAR(solver.effectiveTimeStepForTesting(), 1.0, 0.0);
}

TEST(FsilsBackendLegacyKernels, EffectiveTimeStepStageScalingPreservesExternalBlockSchurOperator)
{
    ScopedEnvVar enable_stage_scaling("SVMP_FSILS_ENABLE_BLOCKSCHUR_STAGE_SCALING", "1");
    ScopedUnsetEnvVar disable_stage_scaling("SVMP_FSILS_DISABLE_BLOCKSCHUR_STAGE_SCALING");

    constexpr Real dt_eff = 0.25;
    FsilsFactory factory(/*dof_per_node=*/3);
    const auto pattern = make_dense_pattern(3);
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(3);
    auto x = factory.createVector(3);
    auto x_exact = factory.createVector(3);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[3] = {0, 1, 2};
    const Real Ke[9] = {4.0, 1.0, 1.0,
                        1.0, 3.0, 0.0,
                        1.0, 0.0, 1.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    {
        auto xs = x_exact->localSpan();
        ASSERT_EQ(xs.size(), 3u);
        xs[0] = 1.0;
        xs[1] = -0.5;
        xs[2] = 0.25;
    }

    RankOneUpdate upd{};
    upd.sigma = 80.0;
    upd.active_components = {0, 1};
    upd.v = {
        {0, 0.20},
        {1, -0.10},
    };

    A->mult(*x_exact, *b);
    addRankOneContributionSerial(factory, *b, *x_exact, std::span<const RankOneUpdate>(&upd, 1));

    SolverOptions opts;
    opts.method = SolverMethod::BlockSchur;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-8;
    opts.abs_tol = 1e-12;
    opts.max_iter = 30;
    opts.krylov_dim = 50;
    opts.fsils_blockschur_gm_max_iter = 160;
    opts.fsils_blockschur_cg_max_iter = 160;
    opts.fsils_blockschur_gm_rel_tol = 1e-10;
    opts.fsils_blockschur_cg_rel_tol = 1e-10;
    BlockLayout layout;
    layout.blocks.push_back({"u", 0, 2, BlockRole::PrimaryField});
    layout.blocks.push_back({"p", 2, 1, BlockRole::ConstraintField});
    layout.momentum_block = 0;
    layout.constraint_block = 1;
    opts.block_layout = layout;

    auto solver = factory.createLinearSolver(opts);
    ASSERT_TRUE(solver->supportsNativeRankOneUpdates());
    solver->setEffectiveTimeStep(dt_eff);
    solver->setRankOneUpdates(std::span<const RankOneUpdate>(&upd, 1));

    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged) << rep.message;

    const Real unscaled_rel = fullOperatorRelativeResidualSerial(
        factory, *A, *x, *b, std::span<const RankOneUpdate>(&upd, 1));
    EXPECT_LE(unscaled_rel, 1e-8);

    RankOneUpdate incorrectly_scaled = upd;
    incorrectly_scaled.sigma *= dt_eff;
    const Real scaled_rel = fullOperatorRelativeResidualSerial(
        factory, *A, *x, *b, std::span<const RankOneUpdate>(&incorrectly_scaled, 1));
    EXPECT_GT(scaled_rel, unscaled_rel * Real(100.0));
}

TEST(FsilsBackendLegacyKernels, BcastSerialPathLeavesScalarAndVectorsUnchanged)
{
    auto commu = makeSelfCommu();

    double scalar = -3.25;
    bcast::fsils_bcast(scalar, commu);
    EXPECT_DOUBLE_EQ(scalar, -3.25);

    Vector<Real> legacy_vec(4);
    for (int i = 0; i < legacy_vec.size(); ++i) {
        legacy_vec(i) = Real(1.0 + 0.25 * i);
    }
    bcast::fsils_bcast_v(legacy_vec.size(), legacy_vec, commu);
    for (int i = 0; i < legacy_vec.size(); ++i) {
        EXPECT_DOUBLE_EQ(legacy_vec(i), Real(1.0 + 0.25 * i));
    }

    std::vector<Real> std_vec{2.0, -4.0, 8.0};
    bcast::fsils_bcast_v(static_cast<int>(std_vec.size()), std_vec, commu);
    ASSERT_EQ(std_vec.size(), 3u);
    EXPECT_DOUBLE_EQ(std_vec[0], 2.0);
    EXPECT_DOUBLE_EQ(std_vec[1], -4.0);
    EXPECT_DOUBLE_EQ(std_vec[2], 8.0);
}

TEST(FsilsBackendLegacyKernels, DotNormAndOmpSumVectorKernelsMatchReferences)
{
    auto commu = makeSelfCommu();
    constexpr int n_nodes = 5;
    constexpr int owned_nodes = 3;

    for (const int dof : {1, 2, 3, 4, 5}) {
        SCOPED_TRACE(::testing::Message() << "dof=" << dof);
        Array<Real> a(dof, n_nodes);
        Array<Real> b(dof, n_nodes);
        Array<Real> original(dof, n_nodes);

        for (int node = 0; node < n_nodes; ++node) {
            for (int comp = 0; comp < dof; ++comp) {
                a(comp, node) = arrayValue(node, comp, 0.5);
                b(comp, node) = arrayValue(node, comp, -0.25);
                original(comp, node) = a(comp, node);
            }
        }
        for (int node = owned_nodes; node < n_nodes; ++node) {
            for (int comp = 0; comp < dof; ++comp) {
                a(comp, node) = Real(1.0e6 + 100.0 * comp + node);
                b(comp, node) = Real(-2.0e6 - 50.0 * comp - node);
                original(comp, node) = a(comp, node);
            }
        }

        const Real dot_ref = referenceDotOwned(a, b, dof, owned_nodes);
        EXPECT_NEAR(dot::fsils_nc_dot_v(dof, owned_nodes, a, b), dot_ref, 1e-12);

        const Real norm_ref = referenceNormOwned(a, dof, owned_nodes);
        EXPECT_NEAR(norm::fsi_ls_normv(dof, owned_nodes, commu, a), norm_ref, 1e-12);

        constexpr Real scale = -0.375;
        omp_la::omp_sum_v(dof, n_nodes, scale, a, b);
        for (int node = 0; node < n_nodes; ++node) {
            for (int comp = 0; comp < dof; ++comp) {
                const Real expected = original(comp, node) + scale * b(comp, node);
                EXPECT_NEAR(a(comp, node), expected, 1e-14)
                    << "node=" << node << " comp=" << comp;
            }
        }

        omp_la::omp_sum_v(dof, n_nodes, 0.0, a, b);
        for (int node = 0; node < n_nodes; ++node) {
            for (int comp = 0; comp < dof; ++comp) {
                const Real expected = original(comp, node) + scale * b(comp, node);
                EXPECT_NEAR(a(comp, node), expected, 1e-14)
                    << "zero-scale node=" << node << " comp=" << comp;
            }
        }
    }
}

TEST(FsilsBackendLegacyKernels, FusedGmresDotAndUpdateMatchUnfusedReferences)
{
    constexpr int n_nodes = 6;
    constexpr int owned_nodes = 4;

    for (const int dof : {1, 2, 3, 4, 5}) {
        SCOPED_TRACE(::testing::Message() << "dof=" << dof);
        constexpr int last_i = 2;
        constexpr int n_slices = last_i + 1;
        Array3<Real> basis(dof, n_nodes, n_slices);
        Array<Real> candidate(dof, n_nodes);
        fillGmresBasis(basis, dof, n_nodes, n_slices);

        for (int node = 0; node < n_nodes; ++node) {
            for (int comp = 0; comp < dof; ++comp) {
                candidate(comp, node) = arrayValue(node, comp, 0.875);
            }
        }
        for (int node = owned_nodes; node < n_nodes; ++node) {
            for (int comp = 0; comp < dof; ++comp) {
                candidate(comp, node) = Real(1.0e5 + 10.0 * node + comp);
            }
        }

        std::vector<Real> h_col(static_cast<std::size_t>(last_i + 2), -999.0);
        const Real zz = gmres::test::fused_dot_zz_v_for_test(
            dof, owned_nodes, basis, last_i, candidate, h_col, /*num_threads=*/1);

        Real zz_ref = 0.0;
        for (int node = 0; node < owned_nodes; ++node) {
            for (int comp = 0; comp < dof; ++comp) {
                zz_ref += candidate(comp, node) * candidate(comp, node);
            }
        }
        EXPECT_NEAR(zz, zz_ref, 1e-12);

        for (int s = 0; s <= last_i; ++s) {
            Real ref = 0.0;
            for (int node = 0; node < owned_nodes; ++node) {
                for (int comp = 0; comp < dof; ++comp) {
                    ref += basis(comp, node, s) * candidate(comp, node);
                }
            }
            EXPECT_NEAR(h_col[static_cast<std::size_t>(s)], ref, 1e-12)
                << "slice=" << s;
        }
        EXPECT_DOUBLE_EQ(h_col[static_cast<std::size_t>(last_i + 1)], -999.0);

        for (const int depth : {0, 1, 2}) {
            SCOPED_TRACE(::testing::Message() << "depth=" << depth);
            Array<Real> update(dof, n_nodes);
            std::vector<Real> expected(static_cast<std::size_t>(dof * n_nodes), 0.0);
            std::vector<Real> h_factors(static_cast<std::size_t>(depth + 1), 0.0);
            for (int j = 0; j <= depth; ++j) {
                h_factors[static_cast<std::size_t>(j)] =
                    ((j % 2 == 0) ? Real(0.4) : Real(-0.2)) * Real(j + 1);
            }
            for (int node = 0; node < n_nodes; ++node) {
                for (int comp = 0; comp < dof; ++comp) {
                    update(comp, node) = arrayValue(node, comp, -1.125);
                    Real ref = update(comp, node);
                    for (int j = 0; j <= depth; ++j) {
                        ref -= h_factors[static_cast<std::size_t>(j)] *
                               basis(comp, node, j);
                    }
                    expected[static_cast<std::size_t>(node * dof + comp)] = ref;
                }
            }

            gmres::test::fused_update_v_inplace_for_test(
                dof, n_nodes, basis, depth, update, h_factors);
            expectArrayMatchesFlat(update, expected, 1e-13);
        }
    }
}

TEST(FsilsBackendLegacyKernels, GivensRotationAndHessenbergBacksolveMatchReferences)
{
    Array<Real> h(4, 4);
    Vector<Real> c(4);
    Vector<Real> s(4);
    Vector<Real> err(4);

    h = 0.0;
    h(0, 2) = 2.0;
    h(1, 2) = -1.5;
    h(2, 2) = 4.0;
    h(3, 2) = -3.0;
    c(0) = 0.8;
    s(0) = 0.6;
    c(1) = 5.0 / 13.0;
    s(1) = -12.0 / 13.0;
    err(0) = 10.0;
    err(1) = -2.0;
    err(2) = 7.5;
    err(3) = 0.0;

    Array<Real> h_ref(h);
    Vector<Real> err_ref(err);
    for (int j = 0; j <= 1; ++j) {
        const Real tmp = c(j) * h_ref(j, 2) + s(j) * h_ref(j + 1, 2);
        h_ref(j + 1, 2) = -s(j) * h_ref(j, 2) + c(j) * h_ref(j + 1, 2);
        h_ref(j, 2) = tmp;
    }
    const Real hypot_ref = std::hypot(h_ref(2, 2), h_ref(3, 2));
    const Real c2_ref = h_ref(2, 2) / hypot_ref;
    const Real s2_ref = h_ref(3, 2) / hypot_ref;
    h_ref(2, 2) = hypot_ref;
    h_ref(3, 2) = 0.0;
    err_ref(3) = -s2_ref * err_ref(2);
    err_ref(2) = c2_ref * err_ref(2);

    gmres::test::apply_givens_rotation_for_test(h, c, s, err, /*i=*/2);

    EXPECT_NEAR(c(2), c2_ref, 1e-15);
    EXPECT_NEAR(s(2), s2_ref, 1e-15);
    EXPECT_NEAR(c(2) * c(2) + s(2) * s(2), 1.0, 1e-15);
    for (int row = 0; row < 4; ++row) {
        EXPECT_NEAR(h(row, 2), h_ref(row, 2), 1e-15) << "h row=" << row;
        EXPECT_NEAR(err(row), err_ref(row), 1e-15) << "err row=" << row;
    }

    Array<Real> h_zero(2, 2);
    Vector<Real> c_zero(2);
    Vector<Real> s_zero(2);
    Vector<Real> err_zero(2);
    h_zero = 0.0;
    c_zero = 0.0;
    s_zero = 0.0;
    err_zero(0) = -4.0;
    err_zero(1) = 123.0;
    gmres::test::apply_givens_rotation_for_test(h_zero, c_zero, s_zero, err_zero, /*i=*/0);
    EXPECT_DOUBLE_EQ(c_zero(0), 1.0);
    EXPECT_DOUBLE_EQ(s_zero(0), 0.0);
    EXPECT_DOUBLE_EQ(h_zero(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(h_zero(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(err_zero(0), -4.0);
    EXPECT_DOUBLE_EQ(err_zero(1), 0.0);

    Array<Real> tri(4, 4);
    Vector<Real> rhs(4);
    Vector<Real> y(4);
    tri = 0.0;
    tri(0, 0) = 4.0;
    tri(0, 1) = -2.0;
    tri(0, 2) = 1.0;
    tri(1, 1) = -3.0;
    tri(1, 2) = 0.5;
    tri(2, 2) = 2.0;
    rhs(0) = 1.0;
    rhs(1) = -8.0;
    rhs(2) = 5.0;
    ASSERT_TRUE(gmres::test::backsolve_hessenberg_for_test(
        tri, rhs, /*last_i=*/2, /*initial_norm=*/10.0, /*eps=*/1e-14, y));
    EXPECT_NEAR(y(2), 2.5, 1e-15);
    EXPECT_NEAR(y(1), 3.0833333333333335, 1e-15);
    EXPECT_NEAR(y(0), 1.1666666666666667, 1e-15);

    tri(1, 1) = 0.0;
    rhs(1) = 0.0;
    rhs(0) = 0.0;
    y = 99.0;
    EXPECT_TRUE(gmres::test::backsolve_hessenberg_for_test(
        tri, rhs, /*last_i=*/1, /*initial_norm=*/1.0, /*eps=*/1e-12, y));
    EXPECT_DOUBLE_EQ(y(1), 0.0);

    rhs(1) = 1.0;
    EXPECT_FALSE(gmres::test::backsolve_hessenberg_for_test(
        tri, rhs, /*last_i=*/1, /*initial_norm=*/1.0, /*eps=*/1e-12, y));
}

TEST(FsilsBackendLegacyKernels, BoundaryPreprocessingComputesCoupledFaceNorms)
{
    fe_fsi_linear_solver::FSILS_lhsType lhs{};
    lhs.nFaces = 3;
    lhs.nNo = 4;
    lhs.mynNo = 2;
    lhs.commu = makeSelfCommu();
    lhs.face.resize(static_cast<std::size_t>(lhs.nFaces));

    lhs.face[0].coupledFlag = false;
    lhs.face[0].nS = 123.0;

    auto& face = lhs.face[1];
    face.coupledFlag = true;
    face.sharedFlag = false;
    face.nNo = 2;
    face.dof = 3;
    face.glob = Vector<int>(2);
    face.glob(0) = 0;
    face.glob(1) = 1;
    face.valM = Array<Real>(3, 2);
    face.valM(0, 0) = 1.0;
    face.valM(1, 0) = -2.0;
    face.valM(2, 0) = 50.0;
    face.valM(0, 1) = 3.0;
    face.valM(1, 1) = 4.0;
    face.valM(2, 1) = -60.0;

    auto& shared_face = lhs.face[2];
    shared_face.coupledFlag = true;
    shared_face.sharedFlag = true;
    shared_face.nNo = 3;
    shared_face.dof = 3;
    shared_face.glob = Vector<int>(3);
    shared_face.glob(0) = 0;
    shared_face.glob(1) = 2;
    shared_face.glob(2) = 3;
    shared_face.valM = Array<Real>(3, 3);
    shared_face.valM(0, 0) = -1.5;
    shared_face.valM(1, 0) = 2.5;
    shared_face.valM(2, 0) = 99.0;
    shared_face.valM(0, 1) = 10.0;
    shared_face.valM(1, 1) = 20.0;
    shared_face.valM(2, 1) = 30.0;
    shared_face.valM(0, 2) = -40.0;
    shared_face.valM(1, 2) = 50.0;
    shared_face.valM(2, 2) = -60.0;

    ns_solver::bc_pre(lhs, /*mom_ncomp=*/2, /*dof=*/3, lhs.nNo, lhs.mynNo);

    EXPECT_DOUBLE_EQ(lhs.face[0].nS, 123.0);
    EXPECT_NEAR(lhs.face[1].nS, 1.0 + 4.0 + 9.0 + 16.0, 1e-14);
    EXPECT_NEAR(lhs.face[2].nS, 2.25 + 6.25, 1e-14);
}

TEST(FsilsBackendLegacyKernels, PrecondDiagAppliesExactSymmetricDof3Scaling)
{
    auto fixture = makeDenseFsilsCsr(/*n_nodes=*/2, /*owned_nodes=*/2);
    constexpr int dof = 3;
    constexpr int block_size = dof * dof;
    Array<Real> values(block_size, static_cast<int>(fixture.lhs.nnz));
    Array<Real> rhs(dof, static_cast<int>(fixture.lhs.nNo));
    Array<Real> weights(dof, static_cast<int>(fixture.lhs.nNo));

    for (int entry = 0; entry < values.ncols(); ++entry) {
        for (int i = 0; i < block_size; ++i) {
            values(i, entry) = static_cast<Real>(0.25 * (1 + i) + 0.5 * (1 + entry));
        }
    }

    values(0, 0) = 4.0;
    values(4, 0) = -9.0;
    values(8, 0) = 0.0;
    values(0, 3) = 16.0;
    values(4, 3) = 25.0;
    values(8, 3) = -36.0;

    for (int node = 0; node < rhs.ncols(); ++node) {
        for (int c = 0; c < dof; ++c) {
            rhs(c, node) = static_cast<Real>(10.0 * (node + 1) + (c + 1));
        }
    }

    const Array<Real> original_values(values);
    const Array<Real> original_rhs(rhs);

    precond::precond_diag(fixture.lhs,
                          fixture.row_ptr,
                          fixture.col_ptr,
                          fixture.diag_ptr,
                          dof,
                          values,
                          rhs,
                          weights);

    const Real expected_weights[dof][2] = {
        {0.5, 0.25},
        {1.0 / 3.0, 0.2},
        {1.0, 1.0 / 6.0},
    };

    for (int node = 0; node < 2; ++node) {
        for (int c = 0; c < dof; ++c) {
            EXPECT_NEAR(weights(c, node), expected_weights[c][node], 1e-14)
                << "weight component " << c << " node " << node;
            EXPECT_NEAR(rhs(c, node),
                        original_rhs(c, node) * expected_weights[c][node],
                        1e-14)
                << "rhs component " << c << " node " << node;
        }
    }

    for (int entry = 0; entry < values.ncols(); ++entry) {
        const int row = static_cast<int>(fixture.entry_row[static_cast<std::size_t>(entry)]);
        const int col = static_cast<int>(fixture.col_ptr(entry));
        for (int out = 0; out < dof; ++out) {
            for (int in = 0; in < dof; ++in) {
                const int block_index = out * dof + in;
                const Real expected = original_values(block_index, entry) *
                                      expected_weights[out][row] *
                                      expected_weights[in][col];
                EXPECT_NEAR(values(block_index, entry), expected, 1e-14)
                    << "entry " << entry << " block(" << out << "," << in << ")";
            }
        }
    }
}

TEST(FsilsBackendLegacyKernels, SparseMulPublicVariantsMatchReferenceAndZeroGhostRows)
{
    auto fixture = makeDenseFsilsCsr(/*n_nodes=*/3, /*owned_nodes=*/2);
    constexpr int dof = 3;

    Vector<Real> scalar_values(static_cast<int>(fixture.lhs.nnz));
    Vector<Real> scalar_input(static_cast<int>(fixture.lhs.nNo));
    Vector<Real> scalar_output(static_cast<int>(fixture.lhs.nNo));
    Array<Real> block_values(dof, static_cast<int>(fixture.lhs.nnz));
    Array<Real> vector_input(dof, static_cast<int>(fixture.lhs.nNo));
    Array<Real> vector_output(dof, static_cast<int>(fixture.lhs.nNo));
    Array<Real> vv_values(dof * dof, static_cast<int>(fixture.lhs.nnz));
    Array<Real> vv_output(dof, static_cast<int>(fixture.lhs.nNo));

    for (int entry = 0; entry < static_cast<int>(fixture.lhs.nnz); ++entry) {
        scalar_values(entry) = static_cast<Real>(0.5 + 0.25 * entry);
        for (int c = 0; c < dof; ++c) {
            block_values(c, entry) = static_cast<Real>((c + 1) * (entry + 2) * 0.125);
        }
        for (int r = 0; r < dof; ++r) {
            for (int c = 0; c < dof; ++c) {
                vv_values(r * dof + c, entry) =
                    static_cast<Real>(0.05 * (entry + 1) + 0.1 * (r + 1) - 0.025 * c);
            }
        }
    }
    for (int node = 0; node < static_cast<int>(fixture.lhs.nNo); ++node) {
        scalar_input(node) = static_cast<Real>(1.0 + 0.5 * node);
        scalar_output(node) = -99.0;
        for (int c = 0; c < dof; ++c) {
            vector_input(c, node) = static_cast<Real>(2.0 + node - 0.25 * c);
            vector_output(c, node) = -99.0;
            vv_output(c, node) = -99.0;
        }
    }

    spar_mul::fsils_spar_mul_ss(fixture.lhs, fixture.row_ptr, fixture.col_ptr,
                                scalar_values, scalar_input, scalar_output);
    for (int row = 0; row < static_cast<int>(fixture.lhs.nNo); ++row) {
        Real expected = 0.0;
        if (row < fixture.lhs.mynNo) {
            for (auto entry = fixture.row_ptr(0, row); entry <= fixture.row_ptr(1, row); ++entry) {
                expected += scalar_values(static_cast<int>(entry)) *
                            scalar_input(static_cast<int>(fixture.col_ptr(static_cast<int>(entry))));
            }
        }
        EXPECT_NEAR(scalar_output(row), expected, 1e-14) << "SS row " << row;
    }

    spar_mul::fsils_spar_mul_sv(fixture.lhs, fixture.row_ptr, fixture.col_ptr,
                                dof, block_values, scalar_input, vector_output);
    for (int row = 0; row < static_cast<int>(fixture.lhs.nNo); ++row) {
        for (int c = 0; c < dof; ++c) {
            Real expected = 0.0;
            if (row < fixture.lhs.mynNo) {
                for (auto entry = fixture.row_ptr(0, row); entry <= fixture.row_ptr(1, row); ++entry) {
                    expected += block_values(c, static_cast<int>(entry)) *
                                scalar_input(static_cast<int>(fixture.col_ptr(static_cast<int>(entry))));
                }
            }
            EXPECT_NEAR(vector_output(c, row), expected, 1e-14)
                << "SV component " << c << " row " << row;
        }
    }

    scalar_output = -99.0;
    spar_mul::fsils_spar_mul_vs(fixture.lhs, fixture.row_ptr, fixture.col_ptr,
                                dof, block_values, vector_input, scalar_output);
    for (int row = 0; row < static_cast<int>(fixture.lhs.nNo); ++row) {
        Real expected = 0.0;
        if (row < fixture.lhs.mynNo) {
            for (auto entry = fixture.row_ptr(0, row); entry <= fixture.row_ptr(1, row); ++entry) {
                const int col = static_cast<int>(fixture.col_ptr(static_cast<int>(entry)));
                for (int c = 0; c < dof; ++c) {
                    expected += block_values(c, static_cast<int>(entry)) * vector_input(c, col);
                }
            }
        }
        EXPECT_NEAR(scalar_output(row), expected, 1e-14) << "VS row " << row;
    }

    spar_mul::fsils_spar_mul_vv(fixture.lhs, fixture.row_ptr, fixture.col_ptr,
                                dof, vv_values, vector_input, vv_output);
    expectArrayMatchesFlat(vv_output, denseVvReference(fixture, dof, vv_values, vector_input), 1e-14);
}

TEST(FsilsBackendLegacyKernels, SparseMulVectorVectorDynamicDofMatchesReference)
{
    auto fixture = makeDenseFsilsCsr(/*n_nodes=*/3, /*owned_nodes=*/2);
    constexpr int dof = 5;
    Array<Real> values(dof * dof, static_cast<int>(fixture.lhs.nnz));
    Array<Real> input(dof, static_cast<int>(fixture.lhs.nNo));
    Array<Real> output(dof, static_cast<int>(fixture.lhs.nNo));

    for (int entry = 0; entry < static_cast<int>(fixture.lhs.nnz); ++entry) {
        for (int r = 0; r < dof; ++r) {
            for (int c = 0; c < dof; ++c) {
                values(r * dof + c, entry) =
                    static_cast<Real>(0.02 * (entry + 3) + 0.03 * (r + 1) + 0.01 * (c + 2));
            }
        }
    }
    for (int node = 0; node < static_cast<int>(fixture.lhs.nNo); ++node) {
        for (int c = 0; c < dof; ++c) {
            input(c, node) = static_cast<Real>(-1.0 + 0.2 * node + 0.15 * c);
            output(c, node) = 77.0;
        }
    }

    spar_mul::fsils_spar_mul_vv(fixture.lhs, fixture.row_ptr, fixture.col_ptr,
                                dof, values, input, output);
    expectArrayMatchesFlat(output, denseVvReference(fixture, dof, values, input), 1e-14);
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

TEST(FsilsBackendStrategy, ForceSchurGmresBypassesLegacyScalarPath)
{
    ScopedEnvVar force_gmres("SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_GMRES", "1");
    ScopedUnsetEnvVar force_bicgstab("SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_BICGSTAB");
    ScopedUnsetEnvVar disable_face_legacy("SVMP_FSILS_BLOCKSCHUR_DISABLE_FACE_ONLY_LEGACY");

    fe_fsi_linear_solver::FSILS_lhsType lhs{};
    fe_fsi_linear_solver::distributed_low_rank_correction::Profile profile{};
    profile.distributed = true;
    profile.active_face_corrections = 1;
    profile.active_reduced_corrections = 0;
    profile.has_grouped_bordered = false;
    profile.native_face_duplicates_only = true;

    const auto selection =
        fe_fsi_linear_solver::BlockSchurStrategySelector::select(lhs, profile, /*con_ncomp=*/1);

    EXPECT_FALSE(selection.use_legacy_scalar_schur());
    EXPECT_TRUE(selection.prefer_schur_gmres);
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

TEST(FsilsBackend, FactoryCreateVectorWithoutMatrixReturnsStandaloneStorage)
{
    FsilsFactory factory(/*dof_per_node=*/3);

    auto standalone = factory.createVector(/*size=*/5);
    auto* standalone_fsils = dynamic_cast<FsilsVector*>(standalone.get());
    ASSERT_NE(standalone_fsils, nullptr);
    EXPECT_EQ(standalone->backendKind(), BackendKind::FSILS);
    EXPECT_EQ(standalone->size(), 5);
    EXPECT_EQ(standalone_fsils->shared(), nullptr);
    EXPECT_FALSE(standalone_fsils->usesOwnedRowLayout());

    auto overlap_fallback = factory.createVector(/*local_size=*/2, /*global_size=*/7);
    auto* overlap_fallback_fsils = dynamic_cast<FsilsVector*>(overlap_fallback.get());
    ASSERT_NE(overlap_fallback_fsils, nullptr);
    EXPECT_EQ(overlap_fallback->backendKind(), BackendKind::FSILS);
    EXPECT_EQ(overlap_fallback->size(), 7);
    EXPECT_EQ(overlap_fallback_fsils->shared(), nullptr);
    EXPECT_FALSE(overlap_fallback_fsils->usesOwnedRowLayout());
}

TEST(FsilsBackend, FactoryCachesSerialMatrixLayoutForSubsequentVectors)
{
    auto permutation = makeIdentityPermutation(/*size=*/4);
    FsilsFactory factory(/*dof_per_node=*/2, permutation);

    const auto pattern = make_dense_pattern(/*n=*/4);
    auto matrix = factory.createMatrix(pattern);
    auto* fsils_matrix = dynamic_cast<FsilsMatrix*>(matrix.get());
    ASSERT_NE(fsils_matrix, nullptr);

    const auto shared = fsils_matrix->shared();
    ASSERT_NE(shared, nullptr);
    EXPECT_EQ(shared->global_dofs, 4);
    EXPECT_EQ(shared->dof, 2);
    EXPECT_EQ(shared->gnNo, 2);
    EXPECT_EQ(shared->dof_permutation.get(), permutation.get());

    auto vector = factory.createVector(/*size=*/4);
    auto* fsils_vector = dynamic_cast<FsilsVector*>(vector.get());
    ASSERT_NE(fsils_vector, nullptr);
    EXPECT_EQ(fsils_vector->shared(), shared.get());
    EXPECT_TRUE(fsils_vector->usesOwnedRowLayout());
    EXPECT_TRUE(fsils_vector->ownsFeDof(0));
    EXPECT_TRUE(fsils_vector->ownsFeDof(3));
    EXPECT_EQ(fsils_vector->shared()->dof_permutation.get(), permutation.get());

    auto second_vector = factory.createVector(/*size=*/4);
    auto* second_fsils_vector = dynamic_cast<FsilsVector*>(second_vector.get());
    ASSERT_NE(second_fsils_vector, nullptr);
    EXPECT_EQ(second_fsils_vector->shared(), shared.get());

    EXPECT_THROW((void)factory.createVector(/*size=*/3), InvalidArgumentException);
}

TEST(FsilsBackend, BackendFactoryCreatesFsilsByKindAndNameAndRejectsInvalidArguments)
{
    auto permutation = makeIdentityPermutation(/*size=*/4);
    BackendFactory::CreateOptions opts{};
    opts.dof_per_node = 2;
    opts.dof_permutation = permutation;

    auto factory_by_kind = BackendFactory::create(BackendKind::FSILS, opts);
    ASSERT_NE(factory_by_kind, nullptr);
    EXPECT_EQ(factory_by_kind->backendKind(), BackendKind::FSILS);

    auto matrix_by_kind = factory_by_kind->createMatrix(make_dense_pattern(/*n=*/4));
    auto* kind_fsils_matrix = dynamic_cast<FsilsMatrix*>(matrix_by_kind.get());
    ASSERT_NE(kind_fsils_matrix, nullptr);
    ASSERT_NE(kind_fsils_matrix->shared(), nullptr);
    EXPECT_EQ(kind_fsils_matrix->shared()->dof, 2);
    EXPECT_EQ(kind_fsils_matrix->shared()->dof_permutation.get(), permutation.get());

    auto vector_by_kind = factory_by_kind->createVector(/*size=*/4);
    auto* kind_fsils_vector = dynamic_cast<FsilsVector*>(vector_by_kind.get());
    ASSERT_NE(kind_fsils_vector, nullptr);
    EXPECT_EQ(kind_fsils_vector->shared(), kind_fsils_matrix->shared().get());

    auto factory_by_name = BackendFactory::create("FSILS", opts);
    ASSERT_NE(factory_by_name, nullptr);
    EXPECT_EQ(factory_by_name->backendKind(), BackendKind::FSILS);

    auto matrix_by_name = factory_by_name->createMatrix(make_dense_pattern(/*n=*/4));
    auto* name_fsils_matrix = dynamic_cast<FsilsMatrix*>(matrix_by_name.get());
    ASSERT_NE(name_fsils_matrix, nullptr);
    ASSERT_NE(name_fsils_matrix->shared(), nullptr);
    EXPECT_EQ(name_fsils_matrix->shared()->dof, 2);
    EXPECT_EQ(name_fsils_matrix->shared()->dof_permutation.get(), permutation.get());

    BackendFactory::CreateOptions invalid_opts{};
    invalid_opts.dof_per_node = 0;
    EXPECT_THROW((void)BackendFactory::create(BackendKind::FSILS, invalid_opts), InvalidArgumentException);
    EXPECT_THROW((void)BackendFactory::create("unknown-backend", opts), InvalidArgumentException);
}

TEST(FsilsBackend, FsilsLegacyStaticsExposeIndependentDefaultStorage)
{
    ScopedVectorLegacyStatics<double> vector_double_scope;
    ScopedVectorLegacyStatics<int> vector_int_scope;
    ScopedArrayLegacyStatics<double> array_double_scope;
    ScopedArrayLegacyStatics<int> array_int_scope;
    ScopedArray3LegacyStatics<double> array3_double_scope;
    ScopedArray3LegacyStatics<int> array3_int_scope;

    resetVectorLegacyStatics<double>();
    resetVectorLegacyStatics<int>();
    resetArrayLegacyStatics<double>();
    resetArrayLegacyStatics<int>();
    resetArray3LegacyStatics<double>();
    resetArray3LegacyStatics<int>();

    EXPECT_TRUE(Vector<double>::show_index_check_message);
    EXPECT_FALSE(Vector<double>::write_enabled);
    EXPECT_EQ(Vector<double>::num_allocated, 0);
    EXPECT_EQ(Vector<double>::active, 0);
    EXPECT_DOUBLE_EQ(Vector<double>::memory_in_use, 0.0);
    EXPECT_DOUBLE_EQ(Vector<double>::memory_returned, 0.0);

    EXPECT_TRUE(Array<double>::show_index_check_message);
    EXPECT_FALSE(Array<double>::write_enabled);
    EXPECT_EQ(Array<double>::id, 0);
    EXPECT_EQ(Array<double>::num_allocated, 0);
    EXPECT_EQ(Array<double>::active, 0);
    EXPECT_DOUBLE_EQ(Array<double>::memory_in_use, 0.0);
    EXPECT_DOUBLE_EQ(Array<double>::memory_returned, 0.0);

    EXPECT_TRUE(Array3<double>::show_index_check_message);
    EXPECT_FALSE(Array3<double>::write_enabled);
    EXPECT_EQ(Array3<double>::num_allocated, 0);
    EXPECT_EQ(Array3<double>::active, 0);
    EXPECT_DOUBLE_EQ(Array3<double>::memory_in_use, 0.0);
    EXPECT_DOUBLE_EQ(Array3<double>::memory_returned, 0.0);

    Vector<double>::num_allocated = 11;
    Vector<double>::write_enabled = true;
    EXPECT_EQ(Vector<int>::num_allocated, 0);
    EXPECT_FALSE(Vector<int>::write_enabled);

    Array<double>::id = 5;
    Array<double>::active = 3;
    EXPECT_EQ(Array<int>::id, 0);
    EXPECT_EQ(Array<int>::active, 0);

    Array3<double>::num_allocated = 7;
    Array3<double>::memory_in_use = 96.0;
    EXPECT_EQ(Array3<int>::num_allocated, 0);
    EXPECT_DOUBLE_EQ(Array3<int>::memory_in_use, 0.0);
}

TEST(FsilsBackend, FsilsLegacyStaticsTrackLegacyContainerLifecycle)
{
    ScopedVectorLegacyStatics<double> vector_scope;
    ScopedArrayLegacyStatics<double> array_scope;
    ScopedArray3LegacyStatics<double> array3_scope;

    resetVectorLegacyStatics<double>();
    resetArrayLegacyStatics<double>();
    resetArray3LegacyStatics<double>();

    {
        Vector<double> values(3);
        EXPECT_EQ(Vector<double>::num_allocated, 1);
        EXPECT_EQ(Vector<double>::active, 1);
        EXPECT_DOUBLE_EQ(Vector<double>::memory_in_use, 3.0 * sizeof(double));
        EXPECT_DOUBLE_EQ(Vector<double>::memory_returned, 0.0);
    }
    EXPECT_EQ(Vector<double>::num_allocated, 1);
    EXPECT_EQ(Vector<double>::active, 0);
    EXPECT_DOUBLE_EQ(Vector<double>::memory_in_use, 0.0);
    EXPECT_DOUBLE_EQ(Vector<double>::memory_returned, 3.0 * sizeof(double));

    {
        Array<double> values;
        EXPECT_EQ(Array<double>::id, 1);
        EXPECT_EQ(Array<double>::num_allocated, 1);
        EXPECT_EQ(Array<double>::active, 1);
        values.resize(2, 3);
    }
    EXPECT_EQ(Array<double>::id, 0);
    EXPECT_EQ(Array<double>::num_allocated, 1);
#if Array_gather_stats
    EXPECT_EQ(Array<double>::active, 0);
    EXPECT_DOUBLE_EQ(Array<double>::memory_in_use, 0.0);
    EXPECT_DOUBLE_EQ(Array<double>::memory_returned, 6.0 * sizeof(double));
#else
    EXPECT_EQ(Array<double>::active, 1);
    EXPECT_DOUBLE_EQ(Array<double>::memory_in_use, 0.0);
    EXPECT_DOUBLE_EQ(Array<double>::memory_returned, 0.0);
#endif

    {
        Array3<double> tensor(2, 2, 2);
        EXPECT_EQ(Array3<double>::num_allocated, 1);
        EXPECT_EQ(Array3<double>::active, 1);
        EXPECT_DOUBLE_EQ(Array3<double>::memory_in_use, 8.0 * sizeof(double));
        EXPECT_DOUBLE_EQ(Array3<double>::memory_returned, 0.0);
    }
    EXPECT_EQ(Array3<double>::num_allocated, 1);
    EXPECT_EQ(Array3<double>::active, 0);
    EXPECT_DOUBLE_EQ(Array3<double>::memory_in_use, 0.0);
    EXPECT_DOUBLE_EQ(Array3<double>::memory_returned, 8.0 * sizeof(double));
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
    aux.row_owner_ranks = {0};
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

TEST(FsilsBackend, MatrixMultAddMatchesDenseReferenceAndRejectsLayoutMismatch)
{
    FsilsFactory factory(/*dof_per_node=*/2);
    const auto pattern = make_dense_pattern(4);
    auto A = factory.createMatrix(pattern);
    auto x = factory.createVector(4);
    auto y = factory.createVector(4);

    const std::array<Real, 16> local = {
        4.0,  1.0, -2.0,  0.5,
        1.5,  3.0,  0.0, -1.0,
       -2.0,  0.0,  5.0,  2.0,
        0.5, -1.5,  2.5,  6.0,
    };

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const std::array<GlobalIndex, 4> dofs = {0, 1, 2, 3};
    viewA->addMatrixEntries(dofs, local, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const std::array<Real, 4> x_init = {1.0, -2.0, 0.5, 3.0};
    const std::array<Real, 4> y_init = {4.0, -1.0, 2.5, 0.75};
    {
        auto xs = x->localSpan();
        auto ys = y->localSpan();
        ASSERT_EQ(xs.size(), x_init.size());
        ASSERT_EQ(ys.size(), y_init.size());
        std::copy(x_init.begin(), x_init.end(), xs.begin());
        std::copy(y_init.begin(), y_init.end(), ys.begin());
    }

    std::array<Real, 4> expected = y_init;
    for (std::size_t row = 0; row < expected.size(); ++row) {
        for (std::size_t col = 0; col < x_init.size(); ++col) {
            expected[row] += local[row * x_init.size() + col] * x_init[col];
        }
    }

    A->multAdd(*x, *y);

    const auto xs_after = x->localSpan();
    const auto ys_after = y->localSpan();
    ASSERT_EQ(xs_after.size(), x_init.size());
    ASSERT_EQ(ys_after.size(), expected.size());
    for (std::size_t i = 0; i < x_init.size(); ++i) {
        EXPECT_DOUBLE_EQ(xs_after[i], x_init[i]) << "x mutated at " << i;
        EXPECT_NEAR(ys_after[i], expected[i], 1e-14) << "y mismatch at " << i;
    }

    FsilsVector standalone(/*global_size=*/4);
    EXPECT_THROW(A->multAdd(standalone, *y), InvalidArgumentException);
}

TEST(FsilsBackend, CachedMatrixEntriesIrregularModesMatchDirectAssembly)
{
    FsilsFactory factory(/*dof_per_node=*/2);
    const auto pattern = make_dense_pattern(4);

    auto direct = factory.createMatrix(pattern);
    auto cached = factory.createMatrix(pattern);
    auto* direct_fsils = dynamic_cast<FsilsMatrix*>(direct.get());
    auto* cached_fsils = dynamic_cast<FsilsMatrix*>(cached.get());
    ASSERT_NE(direct_fsils, nullptr);
    ASSERT_NE(cached_fsils, nullptr);

    const std::array<GlobalIndex, 3> row_dofs = {0, 2, 3};
    const std::array<GlobalIndex, 3> col_dofs = {1, 2, 3};
    const std::array<Real, 9> local_add = {
        1.0,  2.0,  3.0,
       -4.0,  5.0,  6.0,
        7.0, -8.0,  9.0,
    };
    const std::array<Real, 9> local_max = {
        0.5, 4.0,  2.5,
       -5.0, 7.5,  1.0,
        3.5, -6.0, 12.0,
    };
    const std::array<Real, 9> local_min = {
        2.0,  3.0, -1.0,
       -6.0, 6.5,  2.0,
        8.0, -9.0, 10.0,
    };

    auto direct_view = direct->createAssemblyView();
    direct_view->beginAssemblyPhase();
    direct_view->addMatrixEntries(row_dofs, col_dofs, local_add, assembly::AddMode::Add);
    direct_view->addMatrixEntries(row_dofs, col_dofs, local_max, assembly::AddMode::Max);
    direct_view->addMatrixEntries(row_dofs, col_dofs, local_min, assembly::AddMode::Min);
    direct_view->finalizeAssembly();
    direct->finalizeAssembly();

    cached_fsils->addMatrixEntriesCached(row_dofs, col_dofs, local_add, assembly::AddMode::Add);
    cached_fsils->addMatrixEntriesCached(row_dofs, col_dofs, local_max, assembly::AddMode::Max);
    cached_fsils->addMatrixEntriesCached(row_dofs, col_dofs, local_min, assembly::AddMode::Min);
    cached->finalizeAssembly();

    for (GlobalIndex row = 0; row < 4; ++row) {
        for (GlobalIndex col = 0; col < 4; ++col) {
            EXPECT_DOUBLE_EQ(cached_fsils->getEntry(row, col),
                             direct_fsils->getEntry(row, col))
                << "row=" << row << " col=" << col;
        }
    }
}

TEST(FsilsBackend, AddBlockModesMatchDirectAssembly)
{
    FsilsFactory factory(/*dof_per_node=*/2);
    const auto pattern = make_dense_pattern(4);

    auto direct = factory.createMatrix(pattern);
    auto blocked = factory.createMatrix(pattern);
    auto* direct_fsils = dynamic_cast<FsilsMatrix*>(direct.get());
    auto* blocked_fsils = dynamic_cast<FsilsMatrix*>(blocked.get());
    ASSERT_NE(direct_fsils, nullptr);
    ASSERT_NE(blocked_fsils, nullptr);
    ASSERT_NE(blocked_fsils->shared(), nullptr);

    const std::array<GlobalIndex, 2> row_dofs = {0, 1};
    const std::array<GlobalIndex, 2> col_dofs = {2, 3};
    const std::array<Real, 4> block_add = {
        1.0, 2.0,
        3.0, 4.0,
    };
    const std::array<Real, 4> block_max = {
        0.5, 5.0,
        2.5, 3.5,
    };
    const std::array<Real, 4> block_min = {
        2.0, 4.0,
       -1.0, 6.0,
    };

    auto direct_view = direct->createAssemblyView();
    direct_view->beginAssemblyPhase();
    direct_view->addMatrixEntries(row_dofs, col_dofs, block_add, assembly::AddMode::Add);
    direct_view->addMatrixEntries(row_dofs, col_dofs, block_max, assembly::AddMode::Max);
    direct_view->addMatrixEntries(row_dofs, col_dofs, block_min, assembly::AddMode::Min);
    direct_view->finalizeAssembly();
    direct->finalizeAssembly();

    const auto* shared = blocked_fsils->shared().get();
    ASSERT_NE(shared, nullptr);
    const int row_internal = shared->globalNodeToInternal(/*global_node=*/0);
    const int col_internal = shared->globalNodeToInternal(/*global_node=*/1);
    ASSERT_GE(row_internal, 0);
    ASSERT_GE(col_internal, 0);

    blocked_fsils->addBlock(row_internal, col_internal, block_add.data(), /*dof=*/2, assembly::AddMode::Add);
    blocked_fsils->addBlock(row_internal, col_internal, block_max.data(), /*dof=*/2, assembly::AddMode::Max);
    blocked_fsils->addBlock(row_internal, col_internal, block_min.data(), /*dof=*/2, assembly::AddMode::Min);
    blocked->finalizeAssembly();

    for (GlobalIndex row = 0; row < 4; ++row) {
        for (GlobalIndex col = 0; col < 4; ++col) {
            EXPECT_DOUBLE_EQ(blocked_fsils->getEntry(row, col),
                             direct_fsils->getEntry(row, col))
                << "row=" << row << " col=" << col;
        }
    }

    EXPECT_DOUBLE_EQ(blocked_fsils->getEntry(-1, 0), 0.0);
    EXPECT_DOUBLE_EQ(blocked_fsils->getEntry(0, 8), 0.0);
}

TEST(FsilsBackend, VectorMutationAndCopyBehaveAsExpectedForStandaloneAndSharedLayouts)
{
    FsilsVector standalone(/*global_size=*/5);
    standalone.set(1.25);
    standalone.add(-0.25);
    standalone.scale(2.0);
    {
        const auto vals = standalone.localSpan();
        ASSERT_EQ(vals.size(), 5u);
        for (Real value : vals) {
            EXPECT_DOUBLE_EQ(value, 2.0);
        }
    }
    standalone.zero();
    {
        const auto vals = standalone.localSpan();
        for (Real value : vals) {
            EXPECT_DOUBLE_EQ(value, 0.0);
        }
    }

    FsilsVector standalone_src(/*global_size=*/5);
    {
        auto vals = standalone_src.localSpan();
        for (std::size_t i = 0; i < vals.size(); ++i) {
            vals[i] = static_cast<Real>(i + 1) * 0.5;
        }
    }
    standalone.copyFrom(standalone_src);
    {
        const auto vals = standalone.localSpan();
        ASSERT_EQ(vals.size(), 5u);
        for (std::size_t i = 0; i < vals.size(); ++i) {
            EXPECT_DOUBLE_EQ(vals[i], static_cast<Real>(i + 1) * 0.5);
        }
    }

    FsilsFactory factory(/*dof_per_node=*/2);
    auto matrix = factory.createMatrix(make_dense_pattern(4));
    auto shared_dst = factory.createVector(4);
    auto shared_src = factory.createVector(4);
    auto* shared_dst_fsils = dynamic_cast<FsilsVector*>(shared_dst.get());
    ASSERT_NE(shared_dst_fsils, nullptr);
    ASSERT_TRUE(shared_dst_fsils->usesOwnedRowLayout());

    shared_dst->set(3.0);
    shared_dst->add(-0.5);
    shared_dst->scale(-2.0);
    {
        const auto vals = shared_dst->localSpan();
        ASSERT_EQ(vals.size(), 4u);
        for (Real value : vals) {
            EXPECT_DOUBLE_EQ(value, -5.0);
        }
    }

    {
        auto vals = shared_src->localSpan();
        for (std::size_t i = 0; i < vals.size(); ++i) {
            vals[i] = static_cast<Real>(10 + i);
        }
    }
    shared_dst->copyFrom(*shared_src);
    {
        const auto vals = shared_dst->localSpan();
        ASSERT_EQ(vals.size(), 4u);
        for (std::size_t i = 0; i < vals.size(); ++i) {
            EXPECT_DOUBLE_EQ(vals[i], static_cast<Real>(10 + i));
        }
    }

    const auto owned_dofs = shared_dst_fsils->ownedFeDofs();
    EXPECT_EQ(owned_dofs, (std::vector<GlobalIndex>{0, 1, 2, 3}));
}

TEST(FsilsBackend, MatrixGetEntryRespectsPermutationAndSparseStructure)
{
    sparsity::SparsityPattern pattern(4, 4);
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 3);
    pattern.addEntry(1, 1);
    pattern.addEntry(2, 0);
    pattern.addEntry(2, 2);
    pattern.addEntry(3, 3);
    pattern.finalize();

    auto permutation = makePermutation({2, 0, 3, 1});
    FsilsFactory factory(/*dof_per_node=*/1, permutation);
    auto matrix = factory.createMatrix(pattern);
    auto* fsils = dynamic_cast<FsilsMatrix*>(matrix.get());
    ASSERT_NE(fsils, nullptr);

    auto view = matrix->createAssemblyView();
    EXPECT_EQ(view->backendName(), "FSILSMatrix");
    EXPECT_EQ(view->getPhase(), assembly::AssemblyPhase::NotStarted);
    EXPECT_TRUE(view->hasMatrix());
    EXPECT_FALSE(view->hasVector());
    EXPECT_TRUE(view->isDistributed());
    EXPECT_EQ(view->matrixLayoutHandle(), fsils->shared().get());
    const auto capabilities = view->insertionCapabilities();
    EXPECT_TRUE(capabilities.resolved_matrix_entries);
    EXPECT_FALSE(capabilities.resolved_vector_entries);
    EXPECT_TRUE(capabilities.contiguous_combined_matrix_insert);
    EXPECT_FALSE(capabilities.exact_rank_one_updates);

    view->beginAssemblyPhase();
    EXPECT_EQ(view->getPhase(), assembly::AssemblyPhase::Building);
    view->addMatrixEntry(0, 0, 1.25, assembly::AddMode::Insert);
    view->addMatrixEntry(0, 3, -2.5, assembly::AddMode::Insert);
    view->addMatrixEntry(1, 1, 3.75, assembly::AddMode::Insert);
    view->addMatrixEntry(2, 0, 4.5, assembly::AddMode::Insert);
    view->addMatrixEntry(2, 2, -5.25, assembly::AddMode::Insert);
    view->addMatrixEntry(3, 3, 6.0, assembly::AddMode::Insert);
    view->endAssemblyPhase();
    EXPECT_EQ(view->getPhase(), assembly::AssemblyPhase::Flushing);
    view->finalizeAssembly();
    EXPECT_EQ(view->getPhase(), assembly::AssemblyPhase::Finalized);
    matrix->finalizeAssembly();

    EXPECT_DOUBLE_EQ(fsils->getEntry(0, 0), 1.25);
    EXPECT_DOUBLE_EQ(fsils->getEntry(0, 3), -2.5);
    EXPECT_DOUBLE_EQ(fsils->getEntry(1, 1), 3.75);
    EXPECT_DOUBLE_EQ(fsils->getEntry(2, 0), 4.5);
    EXPECT_DOUBLE_EQ(fsils->getEntry(2, 2), -5.25);
    EXPECT_DOUBLE_EQ(fsils->getEntry(3, 3), 6.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(2, 2), -5.25);

    EXPECT_DOUBLE_EQ(fsils->getEntry(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(fsils->getEntry(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(fsils->getEntry(1, 3), 0.0);
    EXPECT_DOUBLE_EQ(fsils->getEntry(3, 2), 0.0);
    EXPECT_DOUBLE_EQ(fsils->getEntry(-1, 0), 0.0);
    EXPECT_DOUBLE_EQ(fsils->getEntry(0, 7), 0.0);
}

TEST(FsilsBackend, MatrixResolutionCacheIsStableAcrossRepeatedAndReorderedQueries)
{
    auto permutation = makePermutation({2, 0, 3, 1});
    FsilsFactory factory(/*dof_per_node=*/1, permutation);
    auto matrix = factory.createMatrix(make_dense_pattern(4));
    auto* fsils = dynamic_cast<FsilsMatrix*>(matrix.get());
    ASSERT_NE(fsils, nullptr);

    const std::array<GlobalIndex, 3> row_dofs_a = {0, 2, 3};
    const std::array<GlobalIndex, 3> col_dofs_a = {1, 2, 0};
    const std::array<GlobalIndex, 3> row_dofs_b = {3, 0, 2};
    const std::array<GlobalIndex, 3> col_dofs_b = {0, 1, 2};

    std::vector<GlobalIndex> slots_a_1(row_dofs_a.size() * col_dofs_a.size(), INVALID_GLOBAL_INDEX);
    std::vector<GlobalIndex> slots_a_2(row_dofs_a.size() * col_dofs_a.size(), INVALID_GLOBAL_INDEX);
    std::vector<GlobalIndex> slots_b(row_dofs_b.size() * col_dofs_b.size(), INVALID_GLOBAL_INDEX);

    fsils->resolveMatrixEntrySlotsCached(row_dofs_a, col_dofs_a, std::span<GlobalIndex>(slots_a_1));
    fsils->resolveMatrixEntrySlotsCached(row_dofs_b, col_dofs_b, std::span<GlobalIndex>(slots_b));
    fsils->resolveMatrixEntrySlotsCached(row_dofs_a, col_dofs_a, std::span<GlobalIndex>(slots_a_2));

    EXPECT_EQ(slots_a_2, slots_a_1);
    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(slots_a_1.size()); ++i) {
        EXPECT_GE(slots_a_1[static_cast<std::size_t>(i)], 0);
    }

    for (std::size_t i = 0; i < row_dofs_b.size(); ++i) {
        for (std::size_t j = 0; j < col_dofs_b.size(); ++j) {
            bool found = false;
            for (std::size_t ai = 0; ai < row_dofs_a.size() && !found; ++ai) {
                if (row_dofs_a[ai] != row_dofs_b[i]) {
                    continue;
                }
                for (std::size_t aj = 0; aj < col_dofs_a.size(); ++aj) {
                    if (col_dofs_a[aj] != col_dofs_b[j]) {
                        continue;
                    }
                    EXPECT_EQ(slots_b[i * col_dofs_b.size() + j],
                              slots_a_1[ai * col_dofs_a.size() + aj]);
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "Missing pair for row=" << row_dofs_b[i]
                               << " col=" << col_dofs_b[j];
        }
    }
}

TEST(FsilsBackend, VectorResolutionCacheAndResolvedViewAccessAreStable)
{
    auto permutation = makePermutation({2, 0, 3, 1});
    FsilsFactory factory(/*dof_per_node=*/1, permutation);
    auto matrix = factory.createMatrix(make_dense_pattern(4));
    auto vector = factory.createVector(4);
    auto* fsils = dynamic_cast<FsilsVector*>(vector.get());
    ASSERT_NE(fsils, nullptr);
    ASSERT_NE(fsils->shared(), nullptr);
    (void)matrix;

    auto view = vector->createAssemblyView();
    EXPECT_EQ(view->backendName(), "FSILSVector");
    EXPECT_EQ(view->getPhase(), assembly::AssemblyPhase::NotStarted);
    EXPECT_FALSE(view->hasMatrix());
    EXPECT_TRUE(view->hasVector());
    EXPECT_EQ(view->vectorLayoutHandle(), fsils->shared());
    const auto capabilities = view->insertionCapabilities();
    EXPECT_FALSE(capabilities.resolved_matrix_entries);
    EXPECT_TRUE(capabilities.resolved_vector_entries);
    EXPECT_FALSE(capabilities.contiguous_combined_matrix_insert);
    EXPECT_FALSE(capabilities.exact_rank_one_updates);

    const std::array<GlobalIndex, 4> dofs_a = {3, 0, 99, 2};
    const std::array<GlobalIndex, 4> dofs_b = {2, 99, 0, 3};
    std::vector<GlobalIndex> resolved_a_1(dofs_a.size(), INVALID_GLOBAL_INDEX);
    std::vector<GlobalIndex> resolved_a_2(dofs_a.size(), INVALID_GLOBAL_INDEX);
    std::vector<GlobalIndex> resolved_b(dofs_b.size(), INVALID_GLOBAL_INDEX);

    view->beginAssemblyPhase();
    EXPECT_EQ(view->getPhase(), assembly::AssemblyPhase::Building);
    fsils->resolveEntriesCached(dofs_a, std::span<GlobalIndex>(resolved_a_1));
    fsils->resolveEntriesCached(dofs_b, std::span<GlobalIndex>(resolved_b));
    fsils->resolveEntriesCached(dofs_a, std::span<GlobalIndex>(resolved_a_2));
    EXPECT_EQ(resolved_a_2, resolved_a_1);
    EXPECT_EQ(resolved_a_1[2], INVALID_GLOBAL_INDEX);

    for (std::size_t i = 0; i < dofs_b.size(); ++i) {
        bool found = false;
        for (std::size_t j = 0; j < dofs_a.size(); ++j) {
            if (dofs_a[j] != dofs_b[i]) {
                continue;
            }
            EXPECT_EQ(resolved_b[i], resolved_a_1[j]);
            found = true;
            break;
        }
        EXPECT_TRUE(found) << "Missing dof " << dofs_b[i];
    }

    const std::array<Real, 4> values = {10.0, 20.0, 30.0, 40.0};
    view->addVectorEntriesResolved(dofs_a,
                                   std::span<const GlobalIndex>(resolved_a_1),
                                   values,
                                   assembly::AddMode::Insert);

    std::array<Real, 4> gathered_by_resolved = {};
    view->getVectorEntriesResolved(std::span<const GlobalIndex>(resolved_a_1),
                                   std::span<Real>(gathered_by_resolved));
    EXPECT_DOUBLE_EQ(gathered_by_resolved[0], 10.0);
    EXPECT_DOUBLE_EQ(gathered_by_resolved[1], 20.0);
    EXPECT_DOUBLE_EQ(gathered_by_resolved[2], 0.0);
    EXPECT_DOUBLE_EQ(gathered_by_resolved[3], 40.0);

    std::array<Real, 4> gathered_by_dof = {};
    view->getVectorEntries(dofs_a, std::span<Real>(gathered_by_dof));
    EXPECT_EQ(gathered_by_dof, gathered_by_resolved);

    view->endAssemblyPhase();
    EXPECT_EQ(view->getPhase(), assembly::AssemblyPhase::Flushing);
    view->finalizeAssembly();
    EXPECT_EQ(view->getPhase(), assembly::AssemblyPhase::Finalized);
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
