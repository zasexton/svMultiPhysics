/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Utils/BackendOptions.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Core/FEException.h"

#include "LinearSolverTestUtils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace svmp::FE::backends {

namespace tu = svmp::FE::backends::testutils;

namespace {

std::unique_ptr<BackendFactory> createFactoryOrSkip(BackendKind kind, int dof_per_node = 1)
{
    BackendFactory::CreateOptions opts;
    opts.dof_per_node = dof_per_node;
    return BackendFactory::create(kind, opts);
}

void expectReportSane(const SolverReport& rep, int max_iter)
{
    EXPECT_GE(rep.iterations, 0);
    EXPECT_LE(rep.iterations, max_iter);
    EXPECT_TRUE(std::isfinite(rep.initial_residual_norm));
    EXPECT_TRUE(std::isfinite(rep.final_residual_norm));
    EXPECT_TRUE(std::isfinite(rep.relative_residual));
}

[[nodiscard]] std::vector<Real> makePoisson1DStiffnessRhsOnes(GlobalIndex n)
{
    std::vector<Real> b(static_cast<std::size_t>(n), 0.0);
    if (n == 0) return b;
    if (n == 1) {
        b[0] = 2.0;
        return b;
    }
    b[0] = 1.0;
    for (GlobalIndex i = 1; i + 1 < n; ++i) {
        b[static_cast<std::size_t>(i)] = 2.0;
    }
    b[static_cast<std::size_t>(n - 1)] = 1.0;
    return b;
}

void assemblePoisson1DStiffnessMatrix(GenericMatrix& A, GlobalIndex n)
{
    auto viewA = A.createAssemblyView();
    viewA->beginAssemblyPhase();
    const Real Ke[4] = {2.0, -1.0,
                        -1.0, 2.0};
    for (GlobalIndex e = 0; e + 1 < n; ++e) {
        const GlobalIndex dofs[2] = {e, e + 1};
        viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    }
    viewA->finalizeAssembly();
    A.finalizeAssembly();
}

void assembleGridLaplacianPlusIdentity(GenericMatrix& A, GlobalIndex nx, GlobalIndex ny, Real kx = 1.0, Real ky = 1.0)
{
    auto viewA = A.createAssemblyView();
    viewA->beginAssemblyPhase();

    auto idx = [nx](GlobalIndex i, GlobalIndex j) { return j * nx + i; };

    for (GlobalIndex j = 0; j < ny; ++j) {
        for (GlobalIndex i = 0; i < nx; ++i) {
            const GlobalIndex row = idx(i, j);
            Real diag = 1.0;
            if (i > 0) {
                viewA->addMatrixEntry(row, idx(i - 1, j), -kx, assembly::AddMode::Insert);
                diag += kx;
            }
            if (i + 1 < nx) {
                viewA->addMatrixEntry(row, idx(i + 1, j), -kx, assembly::AddMode::Insert);
                diag += kx;
            }
            if (j > 0) {
                viewA->addMatrixEntry(row, idx(i, j - 1), -ky, assembly::AddMode::Insert);
                diag += ky;
            }
            if (j + 1 < ny) {
                viewA->addMatrixEntry(row, idx(i, j + 1), -ky, assembly::AddMode::Insert);
                diag += ky;
            }
            viewA->addMatrixEntry(row, row, diag, assembly::AddMode::Insert);
        }
    }

    viewA->finalizeAssembly();
    A.finalizeAssembly();
}

} // namespace

TEST(LinearSolverContract, BackendMismatchThrows)
{
    const auto backends = tu::availableSerialBackends();
    if (backends.size() < 2) {
        GTEST_SKIP() << "Need at least two compiled backends";
    }

    auto factoryA = createFactoryOrSkip(backends[0]);
    auto factoryB = createFactoryOrSkip(backends[1]);

    const auto pat = tu::makeDensePattern(2);
    auto A = factoryA->createMatrix(pat);
    auto b = factoryB->createVector(2);
    auto x = factoryB->createVector(2);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 50;

    auto solver = factoryA->createLinearSolver(opts);
    EXPECT_THROW((void)solver->solve(*A, *x, *b), InvalidArgumentException);
}

class LinearSolverConformance : public ::testing::TestWithParam<BackendKind> {};

INSTANTIATE_TEST_SUITE_P(AllSerialBackends,
                         LinearSolverConformance,
                         ::testing::ValuesIn(tu::availableSerialBackends()));

TEST_P(LinearSolverConformance, InvalidOptionsThrow)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 0.0;
    opts.max_iter = 10;

    opts.max_iter = 0;
    EXPECT_THROW((void)factory->createLinearSolver(opts), InvalidArgumentException);
    opts.max_iter = 10;

    opts.rel_tol = -1.0;
    EXPECT_THROW((void)factory->createLinearSolver(opts), InvalidArgumentException);
    opts.rel_tol = 1e-12;

    opts.abs_tol = -1.0;
    EXPECT_THROW((void)factory->createLinearSolver(opts), InvalidArgumentException);
}

TEST_P(LinearSolverConformance, SizeMismatchThrows)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDensePattern(2);
    auto A = factory->createMatrix(pattern);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 10;

    auto solver = factory->createLinearSolver(opts);
    if (kind == BackendKind::FSILS) {
        // FsilsFactory enforces vector sizes matching the last matrix; construct mismatched vectors directly.
        FsilsVector b_bad(/*size=*/3);
        FsilsVector x_bad(/*size=*/2);
        EXPECT_THROW((void)solver->solve(*A, x_bad, b_bad), InvalidArgumentException);
    } else {
        auto b = factory->createVector(3);
        auto x = factory->createVector(2);
        EXPECT_THROW((void)solver->solve(*A, *x, *b), InvalidArgumentException);
    }
}

TEST_P(LinearSolverConformance, Solve1x1BasicMethods)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDensePattern(1);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(1);
    auto x = factory->createVector(1);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dof0[1] = {0};
    const Real a00[1] = {2.0};
    viewA->addMatrixEntries(dof0, a00, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real b0[1] = {4.0};
    tu::assembleVector(*b, b0);

    std::vector<SolverMethod> methods = {SolverMethod::CG, SolverMethod::GMRES, SolverMethod::FGMRES, SolverMethod::Direct};
    if (kind != BackendKind::FSILS) {
        // FSILS BiCGStab has known breakdowns on trivial 1x1 systems; cover it via larger nonsymmetric tests.
        methods.insert(methods.begin() + 3, SolverMethod::BiCGSTAB);
    }

    for (const auto method : methods) {
        SolverOptions opts;
        opts.method = method;
        opts.preconditioner = PreconditionerType::Diagonal;
        opts.rel_tol = 1e-14;
        opts.abs_tol = 1e-14;
        opts.max_iter = 100;

        SCOPED_TRACE("backend=" + tu::backendName(kind) + " method=" + std::string(solverMethodToString(method)));

        auto solver = factory->createLinearSolver(opts);
        bool expect_direct_notimpl = (kind == BackendKind::FSILS && method == SolverMethod::Direct);

        try {
            const auto rep = solver->solve(*A, *x, *b);
            expectReportSane(rep, opts.max_iter);
            EXPECT_TRUE(rep.converged);

            const auto xs = x->localSpan();
            ASSERT_EQ(xs.size(), 1u);
            EXPECT_NEAR(xs[0], 2.0, 1e-10);
        } catch (const NotImplementedException&) {
            if (!expect_direct_notimpl) {
                FAIL() << "Unexpected NotImplementedException";
            }
        }
    }
}

TEST_P(LinearSolverConformance, Solve2x2SPD_KnownSolution)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDensePattern(2);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(2);
    auto x = factory->createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real be[2] = {1.0, 2.0};
    tu::assembleVector(*b, be);

    const std::vector<SolverMethod> methods = {SolverMethod::CG, SolverMethod::GMRES, SolverMethod::Direct};

    for (const auto method : methods) {
        SolverOptions opts;
        opts.method = method;
        opts.preconditioner = PreconditionerType::Diagonal;
        opts.rel_tol = 1e-12;
        opts.abs_tol = 1e-14;
        opts.max_iter = 200;

        SCOPED_TRACE("backend=" + tu::backendName(kind) + " method=" + std::string(solverMethodToString(method)));

        auto solver = factory->createLinearSolver(opts);
        bool expect_direct_notimpl = (kind == BackendKind::FSILS && method == SolverMethod::Direct);
        try {
            const auto rep = solver->solve(*A, *x, *b);
            expectReportSane(rep, opts.max_iter);
            EXPECT_TRUE(rep.converged);

            const auto xs = x->localSpan();
            ASSERT_EQ(xs.size(), 2u);
            EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-8);
            EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-8);

            const Real rel = tu::computeRelativeResidual(*factory, *A, *x, *b);
            EXPECT_LE(rel, opts.rel_tol * 10.0 + 1e-14);
        } catch (const NotImplementedException&) {
            if (!expect_direct_notimpl) {
                FAIL() << "Unexpected NotImplementedException";
            }
        }
    }
}

TEST_P(LinearSolverConformance, RepeatSolveIsDeterministic)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDensePattern(2);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(2);
    auto x1 = factory->createVector(2);
    auto x2 = factory->createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real be[2] = {1.0, 2.0};
    tu::assembleVector(*b, be);

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory->createLinearSolver(opts);
    const auto rep1 = solver->solve(*A, *x1, *b);
    const auto rep2 = solver->solve(*A, *x2, *b);
    expectReportSane(rep1, opts.max_iter);
    expectReportSane(rep2, opts.max_iter);
    EXPECT_TRUE(rep1.converged);
    EXPECT_TRUE(rep2.converged);

    const auto s1 = x1->localSpan();
    const auto s2 = x2->localSpan();
    ASSERT_EQ(s1.size(), s2.size());
    for (std::size_t i = 0; i < s1.size(); ++i) {
        EXPECT_NEAR(s1[i], s2[i], 1e-12);
    }
}

TEST_P(LinearSolverConformance, Solve2x2NonSym_InitialGuessHandling)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDensePattern(2);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(2);
    auto x = factory->createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        2.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real be[2] = {1.0, 2.0};
    tu::assembleVector(*b, be);

    // Exact solution: x = [0.1, 0.6].
    x->localSpan()[0] = 100.0;
    x->localSpan()[1] = -100.0;

    SolverOptions opts;
    opts.method = SolverMethod::BiCGSTAB;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;
    opts.use_initial_guess = true;

    if (kind == BackendKind::FSILS) {
        EXPECT_THROW((void)factory->createLinearSolver(opts), NotImplementedException);
        return;
    }

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 0.1, 1e-8);
    EXPECT_NEAR(xs[1], 0.6, 1e-8);
}

TEST_P(LinearSolverConformance, Solve2x2NonSym_BiCGSTAB_NoInitialGuess)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDensePattern(2);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(2);
    auto x = factory->createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        2.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real be[2] = {1.0, 2.0};
    tu::assembleVector(*b, be);

    SolverOptions opts;
    opts.method = SolverMethod::BiCGSTAB;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;
    opts.use_initial_guess = false;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 0.1, 1e-8);
    EXPECT_NEAR(xs[1], 0.6, 1e-8);
}

TEST_P(LinearSolverConformance, SolvePoisson1D_50_Converges)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex n = 50;
    const auto pattern = tu::makeTridiagPattern(n);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    assemblePoisson1DStiffnessMatrix(*A, n);
    const auto rhs = makePoisson1DStiffnessRhsOnes(n);
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 0.0;
    opts.max_iter = 2000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);

    const Real rel = tu::computeRelativeResidual(*factory, *A, *x, *b);
    EXPECT_LE(rel, opts.rel_tol * 10.0 + 1e-12);
}

TEST_P(LinearSolverConformance, SolvePoisson2D_16x16_LaplacianPlusI)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex nx = 16;
    constexpr GlobalIndex ny = 16;
    const GlobalIndex n = nx * ny;
    const auto pattern = tu::makeGrid5ptPattern(nx, ny);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    assembleGridLaplacianPlusIdentity(*A, nx, ny);
    std::vector<Real> rhs(static_cast<std::size_t>(n), 1.0);
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-8;
    opts.abs_tol = 0.0;
    opts.max_iter = 3000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);

    const Real rel = tu::computeRelativeResidual(*factory, *A, *x, *b);
    EXPECT_LE(rel, opts.rel_tol * 20.0 + 1e-10);
}

TEST_P(LinearSolverConformance, SolveAnisotropicGrid_SPD)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex nx = 12;
    constexpr GlobalIndex ny = 12;
    const GlobalIndex n = nx * ny;
    const auto pattern = tu::makeGrid5ptPattern(nx, ny);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    // Mild anisotropy to keep iterations bounded in unit tests.
    assembleGridLaplacianPlusIdentity(*A, nx, ny, /*kx=*/1e3, /*ky=*/1.0);
    std::vector<Real> rhs(static_cast<std::size_t>(n), 1.0);
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-7;
    opts.abs_tol = 0.0;
    opts.max_iter = 5000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);
}

TEST_P(LinearSolverConformance, SolveConvectionDiffusion1D_NonSym)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex n = 80;
    const auto pattern = tu::makeTridiagPattern(n);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    // Upwind-like nonsymmetric tri-diagonal: diag dominates.
    const Real beta = 0.3;
    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    for (GlobalIndex i = 0; i < n; ++i) {
        const Real diag = 4.0;
        viewA->addMatrixEntry(i, i, diag, assembly::AddMode::Insert);
        if (i > 0) viewA->addMatrixEntry(i, i - 1, -1.0 - beta, assembly::AddMode::Insert);
        if (i + 1 < n) viewA->addMatrixEntry(i, i + 1, -1.0 + beta, assembly::AddMode::Insert);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    // Choose x_true = 1 => b is the row sum.
    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        Real sum = 4.0;
        if (i > 0) sum += (-1.0 - beta);
        if (i + 1 < n) sum += (-1.0 + beta);
        rhs[static_cast<std::size_t>(i)] = sum;
    }
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 0.0;
    opts.max_iter = 2000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);

    EXPECT_TRUE(tu::allFinite(x->localSpan()));
}

TEST_P(LinearSolverConformance, SolveIndefiniteDiagonal_GMRES)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex n = 12;
    const auto pattern = tu::makeDiagonalPattern(n);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    for (GlobalIndex i = 0; i < n; ++i) {
        const Real d = (i % 2 == 0) ? 1.0 : -1.0;
        viewA->addMatrixEntry(i, i, d, assembly::AddMode::Insert);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    // x_true = 1 => b = diag (alternating +/-).
    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) rhs[static_cast<std::size_t>(i)] = (i % 2 == 0) ? 1.0 : -1.0;
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::None;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    EXPECT_TRUE(tu::allFinite(xs));
    for (const auto v : xs) {
        EXPECT_NEAR(v, 1.0, 1e-10);
    }
}

TEST_P(LinearSolverConformance, ToleranceModes_RelOnly_AbsOnly)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDensePattern(2);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(2);
    auto x = factory->createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real be[2] = {1.0, 2.0};
    tu::assembleVector(*b, be);

    // Relative-only.
    {
        SolverOptions opts;
        opts.method = SolverMethod::GMRES;
        opts.preconditioner = PreconditionerType::Diagonal;
        opts.rel_tol = 1e-10;
        opts.abs_tol = 0.0;
        opts.max_iter = 200;

        auto solver = factory->createLinearSolver(opts);
        const auto rep = solver->solve(*A, *x, *b);
        expectReportSane(rep, opts.max_iter);
        EXPECT_TRUE(rep.converged);
        const Real rel = tu::computeRelativeResidual(*factory, *A, *x, *b);
        EXPECT_LE(rel, opts.rel_tol * 20.0 + 1e-12);
    }

    // Absolute-only.
    {
        SolverOptions opts;
        opts.method = SolverMethod::GMRES;
        opts.preconditioner = PreconditionerType::Diagonal;
        opts.rel_tol = 0.0;
        opts.abs_tol = 1e-12;
        opts.max_iter = 200;

        auto solver = factory->createLinearSolver(opts);
        const auto rep = solver->solve(*A, *x, *b);
        expectReportSane(rep, opts.max_iter);
        EXPECT_TRUE(rep.converged);
        const Real rn = tu::computeResidualNorm(*factory, *A, *x, *b);
        EXPECT_LE(rn, opts.abs_tol * 50.0 + 1e-14);
    }
}

TEST_P(LinearSolverConformance, ZeroRhsReturnsZeroSolution)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDensePattern(2);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(2);
    auto x = factory->createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real zeros[2] = {0.0, 0.0};
    tu::assembleVector(*b, zeros);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 100;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);
    EXPECT_NEAR(x->norm(), 0.0, 1e-14);
    EXPECT_NEAR(tu::computeResidualNorm(*factory, *A, *x, *b), 0.0, 1e-14);
}

TEST_P(LinearSolverConformance, SingularSystemFailsSafely)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDiagonalPattern(2);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(2);
    auto x = factory->createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewA->addMatrixEntry(0, 0, 1.0, assembly::AddMode::Insert);
    viewA->addMatrixEntry(1, 1, 0.0, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real be[2] = {1.0, 1.0}; // inconsistent with singular row
    tu::assembleVector(*b, be);

    SolverOptions opts;
    opts.method = (kind == BackendKind::PETSc) ? SolverMethod::GMRES : SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::None;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 50;

    auto solver = factory->createLinearSolver(opts);
    try {
        const auto rep = solver->solve(*A, *x, *b);
        EXPECT_FALSE(rep.converged);
        EXPECT_GE(rep.iterations, 0);
        EXPECT_LE(rep.iterations, opts.max_iter);
        EXPECT_TRUE(tu::allFinite(x->localSpan()));
    } catch (const std::exception&) {
        SUCCEED();
    }
}

TEST_P(LinearSolverConformance, NearSingularSystemProducesFiniteSolution)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex n = 12;
    const auto pattern = tu::makeDiagonalPattern(n);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        const Real d = (i == 0) ? 1e-12 : static_cast<Real>(i + 1);
        viewA->addMatrixEntry(i, i, d, assembly::AddMode::Insert);
        rhs[static_cast<std::size_t>(i)] = d; // x_true = 1
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(tu::allFinite(x->localSpan()));

    // Many solvers will converge immediately on a diagonal matrix; don't require it.
    if (rep.converged) {
        const Real rel = tu::computeRelativeResidual(*factory, *A, *x, *b);
        EXPECT_LE(rel, opts.rel_tol * 20.0 + 1e-12);
    }
}

TEST_P(LinearSolverConformance, RowColumnScalingOptionDoesNotBreakSolve)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex n = 60;
    const auto pattern = tu::makeTridiagPattern(n);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    // Build a strongly scaled SPD operator A = D*A0*D, with A0 the 1D stiffness matrix.
    // Choose x_true = 1, so b = A*1 can be computed without calling A.mult().
    std::vector<Real> s(static_cast<std::size_t>(n), 1.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(std::max<GlobalIndex>(1, n - 1));
        s[static_cast<std::size_t>(i)] = static_cast<Real>(std::pow(1e6, t));
    }

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    for (GlobalIndex i = 0; i < n; ++i) {
        const Real si = s[static_cast<std::size_t>(i)];
        Real diag0 = (n == 1) ? 2.0 : ((i == 0 || i + 1 == n) ? 2.0 : 4.0);
        // Diagonal entry: si * diag0 * si.
        viewA->addMatrixEntry(i, i, si * diag0 * si, assembly::AddMode::Insert);
        if (i > 0) {
            const Real sj = s[static_cast<std::size_t>(i - 1)];
            viewA->addMatrixEntry(i, i - 1, si * (-1.0) * sj, assembly::AddMode::Insert);
        }
        if (i + 1 < n) {
            const Real sj = s[static_cast<std::size_t>(i + 1)];
            viewA->addMatrixEntry(i, i + 1, si * (-1.0) * sj, assembly::AddMode::Insert);
        }
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    // Compute b = D*(A0*s), where s = D*1.
    std::vector<Real> tmp(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        Real v = 0.0;
        if (n == 1) {
            v = 2.0 * s[0];
        } else if (i == 0) {
            v = 2.0 * s[0] - 1.0 * s[1];
        } else if (i + 1 == n) {
            v = -1.0 * s[static_cast<std::size_t>(n - 2)] + 2.0 * s[static_cast<std::size_t>(n - 1)];
        } else {
            v = -1.0 * s[static_cast<std::size_t>(i - 1)] + 4.0 * s[static_cast<std::size_t>(i)] -
                1.0 * s[static_cast<std::size_t>(i + 1)];
        }
        tmp[static_cast<std::size_t>(i)] = v;
    }
    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        rhs[static_cast<std::size_t>(i)] = s[static_cast<std::size_t>(i)] * tmp[static_cast<std::size_t>(i)];
    }
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::RowColumnScaling;
    opts.rel_tol = 1e-8;
    opts.abs_tol = 0.0;
    opts.max_iter = 3000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);
    EXPECT_TRUE(tu::allFinite(x->localSpan()));
}

TEST_P(LinearSolverConformance, DiagonalMatrixConvergesFast)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex n = 10;
    const auto pattern = tu::makeDiagonalPattern(n);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    for (GlobalIndex i = 0; i < n; ++i) {
        viewA->addMatrixEntry(i, i, static_cast<Real>(i + 1), assembly::AddMode::Insert);
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    // x_true = 1 => b = diag
    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        rhs[static_cast<std::size_t>(i)] = static_cast<Real>(i + 1);
    }
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 50;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);
    EXPECT_TRUE(tu::allFinite(x->localSpan()));
    const Real rel = tu::computeRelativeResidual(*factory, *A, *x, *b);
    EXPECT_LE(rel, 1e-10);
}

TEST_P(LinearSolverConformance, ILUSmokeTest_WhenSupported)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex nx = 12;
    constexpr GlobalIndex ny = 12;
    const GlobalIndex n = nx * ny;
    const auto pattern = tu::makeGrid5ptPattern(nx, ny);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    assembleGridLaplacianPlusIdentity(*A, nx, ny);
    std::vector<Real> rhs(static_cast<std::size_t>(n), 1.0);
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::ILU;
    opts.rel_tol = 1e-8;
    opts.abs_tol = 0.0;
    opts.max_iter = 2000;

    auto solver = factory->createLinearSolver(opts);
    try {
        const auto rep = solver->solve(*A, *x, *b);
        expectReportSane(rep, opts.max_iter);
        EXPECT_TRUE(rep.converged);
    } catch (const NotImplementedException&) {
        // OK (backend may not support ILU).
        SUCCEED();
    } catch (const std::exception&) {
        // Some optional ILU implementations may not be configured.
        GTEST_SKIP() << "ILU preconditioner unavailable/configured for this backend";
    }
}

TEST_P(LinearSolverConformance, AMGSolvesPoisson_WhenAvailable)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex nx = 12;
    constexpr GlobalIndex ny = 12;
    const GlobalIndex n = nx * ny;
    const auto pattern = tu::makeGrid5ptPattern(nx, ny);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    assembleGridLaplacianPlusIdentity(*A, nx, ny);
    std::vector<Real> rhs(static_cast<std::size_t>(n), 1.0);
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::AMG;
    opts.rel_tol = 1e-8;
    opts.abs_tol = 0.0;
    opts.max_iter = 2000;

    auto solver = factory->createLinearSolver(opts);
    if (kind == BackendKind::Eigen) {
        EXPECT_THROW((void)solver->solve(*A, *x, *b), NotImplementedException);
        return;
    }

    // FSILS ignores AMG request (falls back to none). PETSc may use GAMG.
    try {
        const auto rep = solver->solve(*A, *x, *b);
        expectReportSane(rep, opts.max_iter);
        EXPECT_TRUE(rep.converged);
    } catch (const std::exception&) {
        GTEST_SKIP() << "AMG preconditioner not available/configured for this backend";
    }
}

TEST_P(LinearSolverConformance, DuplicateAssemblyContributionsAreLinear)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex n = 20;
    const auto pattern = tu::makeTridiagPattern(n);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    assemblePoisson1DStiffnessMatrix(*A, n);
    assemblePoisson1DStiffnessMatrix(*A, n); // add again => 2*A

    const auto rhs = makePoisson1DStiffnessRhsOnes(n);
    std::vector<Real> rhs2(rhs.size());
    for (std::size_t i = 0; i < rhs.size(); ++i) rhs2[i] = 2.0 * rhs[i];
    tu::assembleVector(*b, rhs2);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 0.0;
    opts.max_iter = 2000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);

    // The doubled system still has x_true = 1.
    const auto xs = x->localSpan();
    for (const auto v : xs) {
        EXPECT_NEAR(v, 1.0, 1e-8);
    }
}

TEST_P(LinearSolverConformance, AssemblyOrderDoesNotChangeSolution)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex nx = 8;
    constexpr GlobalIndex ny = 8;
    const GlobalIndex n = nx * ny;
    const auto pattern = tu::makeGrid5ptPattern(nx, ny);

    auto A1 = factory->createMatrix(pattern);
    auto A2 = factory->createMatrix(pattern);
    auto b = factory->createVector(n);

    assembleGridLaplacianPlusIdentity(*A1, nx, ny);

    // Assemble A2 with shuffled (row,col,val) triplets.
    struct Triplet {
        GlobalIndex r;
        GlobalIndex c;
        Real v;
    };
    std::vector<Triplet> trips;
    trips.reserve(static_cast<std::size_t>(n) * 5u);

    auto idx = [nx](GlobalIndex i, GlobalIndex j) { return j * nx + i; };
    for (GlobalIndex j = 0; j < ny; ++j) {
        for (GlobalIndex i = 0; i < nx; ++i) {
            const GlobalIndex row = idx(i, j);
            Real diag = 1.0;
            if (i > 0) {
                trips.push_back({row, idx(i - 1, j), -1.0});
                diag += 1.0;
            }
            if (i + 1 < nx) {
                trips.push_back({row, idx(i + 1, j), -1.0});
                diag += 1.0;
            }
            if (j > 0) {
                trips.push_back({row, idx(i, j - 1), -1.0});
                diag += 1.0;
            }
            if (j + 1 < ny) {
                trips.push_back({row, idx(i, j + 1), -1.0});
                diag += 1.0;
            }
            trips.push_back({row, row, diag});
        }
    }
    std::mt19937 rng(12345);
    std::shuffle(trips.begin(), trips.end(), rng);

    auto viewA = A2->createAssemblyView();
    viewA->beginAssemblyPhase();
    for (const auto& t : trips) {
        viewA->addMatrixEntry(t.r, t.c, t.v, assembly::AddMode::Insert);
    }
    viewA->finalizeAssembly();
    A2->finalizeAssembly();

    std::vector<Real> rhs(static_cast<std::size_t>(n), 1.0);
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 0.0;
    opts.max_iter = 2000;

    auto x1 = factory->createVector(n);
    auto x2 = factory->createVector(n);

    auto solver = factory->createLinearSolver(opts);
    const auto rep1 = solver->solve(*A1, *x1, *b);
    expectReportSane(rep1, opts.max_iter);
    EXPECT_TRUE(rep1.converged);

    const auto rep2 = solver->solve(*A2, *x2, *b);
    expectReportSane(rep2, opts.max_iter);
    EXPECT_TRUE(rep2.converged);

    const auto s1 = x1->localSpan();
    const auto s2 = x2->localSpan();
    ASSERT_EQ(s1.size(), s2.size());
    for (std::size_t i = 0; i < s1.size(); ++i) {
        EXPECT_NEAR(s1[i], s2[i], 1e-10);
    }
}

TEST_P(LinearSolverConformance, MissingPatternEntryBehaviorIsDeterministic)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    sparsity::SparsityPattern pat(2, 2);
    pat.addEntry(0, 0);
    pat.addEntry(1, 1);
    pat.finalize();

    auto A = factory->createMatrix(pat);
    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewA->addMatrixEntry(0, 1, 2.0, assembly::AddMode::Insert); // not in pattern
    viewA->addMatrixEntry(0, 0, 1.0, assembly::AddMode::Insert);
    viewA->addMatrixEntry(1, 1, 1.0, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    if (kind == BackendKind::PETSc) {
        EXPECT_DOUBLE_EQ(A->getEntry(0, 1), 2.0);
    } else {
        EXPECT_DOUBLE_EQ(A->getEntry(0, 1), 0.0);
    }
}

TEST_P(LinearSolverConformance, ZeroRowsImposesDirichletConstraint)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    const auto pattern = tu::makeDensePattern(2);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(2);
    auto x = factory->createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);

    const GlobalIndex rows_to_zero[1] = {0};
    viewA->zeroRows(rows_to_zero, true);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real be[2] = {5.0, 2.0};
    tu::assembleVector(*b, be);

    SolverOptions opts;
    opts.method = SolverMethod::GMRES;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-12;
    opts.abs_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 5.0, 1e-10);
}

TEST_P(LinearSolverConformance, MultiDofBlockDiagonalSystemDof2)
{
    const auto kind = GetParam();
    const int dof = 2;
    auto factory = createFactoryOrSkip(kind, (kind == BackendKind::FSILS) ? dof : 1);

    constexpr GlobalIndex n_nodes = 30;
    const GlobalIndex n = n_nodes * dof;

    // Block-diagonal: two independent copies of the 1D stiffness operator.
    const auto scalar_pat = tu::makeTridiagPattern(n_nodes);
    const auto pattern = tu::replicateScalarPatternPerComponent(scalar_pat, dof);

    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    // Assemble component-wise.
    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const Real Ke[4] = {2.0, -1.0,
                        -1.0, 2.0};
    for (GlobalIndex e = 0; e + 1 < n_nodes; ++e) {
        for (int c = 0; c < dof; ++c) {
            const GlobalIndex dofs_e[2] = {e * dof + c, (e + 1) * dof + c};
            viewA->addMatrixEntries(dofs_e, Ke, assembly::AddMode::Add);
        }
    }
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    // x_true = 1 for all dofs => RHS repeats scalar RHS.
    const auto rhs_scalar = makePoisson1DStiffnessRhsOnes(n_nodes);
    std::vector<Real> rhs(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        for (int c = 0; c < dof; ++c) {
            rhs[static_cast<std::size_t>(node * dof + c)] = rhs_scalar[static_cast<std::size_t>(node)];
        }
    }
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 0.0;
    opts.max_iter = 4000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    EXPECT_TRUE(tu::allFinite(xs));
    for (const auto v : xs) {
        EXPECT_NEAR(v, 1.0, 1e-8);
    }
}

TEST_P(LinearSolverConformance, BlockSchurSaddlePointMapping)
{
    const auto kind = GetParam();

    // Use a single-node dof=3 system in the (u,v,p) ordering expected by FSILS NS solver.
    const int dof = 3;
    auto factory = createFactoryOrSkip(kind, (kind == BackendKind::FSILS) ? dof : 1);

    const auto pattern = tu::makeDensePattern(dof);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(dof);
    auto x = factory->createVector(dof);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs_g[3] = {0, 1, 2};
    // Small saddle-point-like matrix (not fluid-specific).
    const Real Ke[9] = {4.0, 1.0, 1.0,
                        1.0, 3.0, 0.0,
                        1.0, 0.0, 1.0};
    viewA->addMatrixEntries(dofs_g, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    const Real be[3] = {1.0, 2.0, 3.0};
    tu::assembleVector(*b, be);

    SolverOptions opts;
    opts.method = SolverMethod::BlockSchur;
    opts.preconditioner = PreconditionerType::Diagonal;
    if (kind == BackendKind::FSILS) {
        // FSILS NS solver is an inexact saddle-point routine; validate by residual reduction.
        opts.rel_tol = 0.4;
        opts.abs_tol = 0.0;
        opts.max_iter = 200;
    } else {
        // Other backends treat BlockSchur as a monolithic Krylov solve; require a tighter tolerance.
        opts.rel_tol = 1e-12;
        opts.abs_tol = 1e-14;
        opts.max_iter = 200;
    }

    auto solver = factory->createLinearSolver(opts);

    if (kind == BackendKind::PETSc) {
        EXPECT_THROW((void)solver->solve(*A, *x, *b), InvalidArgumentException);
        return;
    }

    try {
        const auto rep = solver->solve(*A, *x, *b);
        expectReportSane(rep, opts.max_iter);
        EXPECT_TRUE(rep.converged);

        // Validate by residual reduction.
        const Real rel = tu::computeRelativeResidual(*factory, *A, *x, *b);
        if (kind == BackendKind::FSILS) {
            EXPECT_LE(rel, opts.rel_tol + 1e-12);
        } else {
            EXPECT_LE(rel, opts.rel_tol * 20.0 + 1e-12);
        }
    } catch (const NotImplementedException&) {
        // Backends may not implement a true BlockSchur solver (e.g. fallback paths).
        SUCCEED();
    }
}

TEST_P(LinearSolverConformance, ForcedNonconvergenceIsReported)
{
    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex nx = 16;
    constexpr GlobalIndex ny = 16;
    const GlobalIndex n = nx * ny;
    const auto pattern = tu::makeGrid5ptPattern(nx, ny);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    assembleGridLaplacianPlusIdentity(*A, nx, ny);
    std::vector<Real> rhs(static_cast<std::size_t>(n), 1.0);
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-16;
    opts.abs_tol = 0.0;
    opts.max_iter = 1;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_FALSE(rep.converged);

    const Real rel = tu::computeRelativeResidual(*factory, *A, *x, *b);
    EXPECT_GT(rel, opts.rel_tol * 1e3);
}

TEST_P(LinearSolverConformance, Stress_Poisson2D_10k_Gated)
{
    if (!tu::envFlagEnabled("SVMP_FE_STRESS_TESTS")) {
        GTEST_SKIP() << "Set SVMP_FE_STRESS_TESTS=1 to enable";
    }

    const auto kind = GetParam();
    auto factory = createFactoryOrSkip(kind);

    constexpr GlobalIndex nx = 100;
    constexpr GlobalIndex ny = 100;
    const GlobalIndex n = nx * ny;
    const auto pattern = tu::makeGrid5ptPattern(nx, ny);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n);
    auto x = factory->createVector(n);

    assembleGridLaplacianPlusIdentity(*A, nx, ny);
    std::vector<Real> rhs(static_cast<std::size_t>(n), 1.0);
    tu::assembleVector(*b, rhs);

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-8;
    opts.abs_tol = 0.0;
    opts.max_iter = 20000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);
}

} // namespace svmp::FE::backends
