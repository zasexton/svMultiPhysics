/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/FEException.h"

#include "LinearSolverTestUtils.h"

#include <cmath>
#include <string>
#include <vector>

namespace svmp::FE::backends {

#if !defined(FE_HAS_TRILINOS) || !FE_HAS_TRILINOS

TEST(TrilinosLinearSolverConformance, TrilinosNotEnabled)
{
    GTEST_SKIP() << "FE_HAS_TRILINOS not enabled";
}

#else

namespace tu = svmp::FE::backends::testutils;

namespace {

std::unique_ptr<BackendFactory> createTrilinosFactory()
{
    try {
        return BackendFactory::create(BackendKind::Trilinos);
    } catch (const std::exception& e) {
        (void)e;
        return nullptr;
    }
}

void expectReportSane(const SolverReport& rep, int max_iter)
{
    EXPECT_GE(rep.iterations, 0);
    EXPECT_LE(rep.iterations, max_iter);
    EXPECT_TRUE(std::isfinite(rep.initial_residual_norm));
    EXPECT_TRUE(std::isfinite(rep.final_residual_norm));
    EXPECT_TRUE(std::isfinite(rep.relative_residual));
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

[[nodiscard]] std::vector<Real> makePoisson1DStiffnessRhsOnes(GlobalIndex n)
{
    std::vector<Real> b(static_cast<std::size_t>(n), 0.0);
    if (n == 0) return b;
    if (n == 1) {
        b[0] = 2.0;
        return b;
    }
    b[0] = 1.0;
    for (GlobalIndex i = 1; i + 1 < n; ++i) b[static_cast<std::size_t>(i)] = 2.0;
    b[static_cast<std::size_t>(n - 1)] = 1.0;
    return b;
}

void assembleGridLaplacianPlusIdentity(GenericMatrix& A, GlobalIndex nx, GlobalIndex ny)
{
    auto viewA = A.createAssemblyView();
    viewA->beginAssemblyPhase();
    auto idx = [nx](GlobalIndex i, GlobalIndex j) { return j * nx + i; };
    for (GlobalIndex j = 0; j < ny; ++j) {
        for (GlobalIndex i = 0; i < nx; ++i) {
            const GlobalIndex row = idx(i, j);
            Real diag = 1.0;
            if (i > 0) {
                viewA->addMatrixEntry(row, idx(i - 1, j), -1.0, assembly::AddMode::Insert);
                diag += 1.0;
            }
            if (i + 1 < nx) {
                viewA->addMatrixEntry(row, idx(i + 1, j), -1.0, assembly::AddMode::Insert);
                diag += 1.0;
            }
            if (j > 0) {
                viewA->addMatrixEntry(row, idx(i, j - 1), -1.0, assembly::AddMode::Insert);
                diag += 1.0;
            }
            if (j + 1 < ny) {
                viewA->addMatrixEntry(row, idx(i, j + 1), -1.0, assembly::AddMode::Insert);
                diag += 1.0;
            }
            viewA->addMatrixEntry(row, row, diag, assembly::AddMode::Insert);
        }
    }
    viewA->finalizeAssembly();
    A.finalizeAssembly();
}

} // namespace

TEST(TrilinosLinearSolverConformance, InvalidOptionsThrow)
{
    auto factory = createTrilinosFactory();
    if (!factory) GTEST_SKIP() << "Trilinos backend unavailable";

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

TEST(TrilinosLinearSolverConformance, Solve2x2SPD_KnownSolution)
{
    auto factory = createTrilinosFactory();
    if (!factory) GTEST_SKIP() << "Trilinos backend unavailable";

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

    SolverOptions opts;
    opts.method = SolverMethod::CG;
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
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-8);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-8);

    const Real rel = tu::computeRelativeResidual(*factory, *A, *x, *b);
    EXPECT_LE(rel, opts.rel_tol * 50.0 + 1e-12);
}

TEST(TrilinosLinearSolverConformance, SolvePoisson1D_50_Converges)
{
    auto factory = createTrilinosFactory();
    if (!factory) GTEST_SKIP() << "Trilinos backend unavailable";

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
    opts.max_iter = 3000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);
}

TEST(TrilinosLinearSolverConformance, SolvePoisson2D_16x16_LaplacianPlusI)
{
    auto factory = createTrilinosFactory();
    if (!factory) GTEST_SKIP() << "Trilinos backend unavailable";

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
    opts.max_iter = 5000;

    auto solver = factory->createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    expectReportSane(rep, opts.max_iter);
    EXPECT_TRUE(rep.converged);
}

TEST(TrilinosLinearSolverConformance, ILUSmokeTest_WhenAvailable)
{
    auto factory = createTrilinosFactory();
    if (!factory) GTEST_SKIP() << "Trilinos backend unavailable";

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
    opts.max_iter = 3000;

    auto solver = factory->createLinearSolver(opts);
    try {
        const auto rep = solver->solve(*A, *x, *b);
        expectReportSane(rep, opts.max_iter);
        EXPECT_TRUE(rep.converged);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Ifpack2 ILU not available/configured: " << e.what();
    }
}

TEST(TrilinosLinearSolverConformance, AMGSolve_WhenAvailable)
{
    auto factory = createTrilinosFactory();
    if (!factory) GTEST_SKIP() << "Trilinos backend unavailable";

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
    opts.max_iter = 3000;

    auto solver = factory->createLinearSolver(opts);
    try {
        const auto rep = solver->solve(*A, *x, *b);
        expectReportSane(rep, opts.max_iter);
        EXPECT_TRUE(rep.converged);
    } catch (const NotImplementedException& e) {
        GTEST_SKIP() << e.what();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "AMG unavailable/configured: " << e.what();
    }
}

TEST(TrilinosLinearSolverConformance, BlockSchurIsNotImplemented)
{
    auto factory = createTrilinosFactory();
    if (!factory) GTEST_SKIP() << "Trilinos backend unavailable";

    const auto pattern = tu::makeDensePattern(3);
    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(3);
    auto x = factory->createVector(3);

    SolverOptions opts;
    opts.method = SolverMethod::BlockSchur;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-6;
    opts.abs_tol = 0.0;
    opts.max_iter = 20;

    auto solver = factory->createLinearSolver(opts);
    EXPECT_THROW((void)solver->solve(*A, *x, *b), NotImplementedException);
}

TEST(TrilinosLinearSolverConformance, ForcedNonconvergenceIsReported)
{
    auto factory = createTrilinosFactory();
    if (!factory) GTEST_SKIP() << "Trilinos backend unavailable";

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
}

#endif // FE_HAS_TRILINOS

} // namespace svmp::FE::backends
