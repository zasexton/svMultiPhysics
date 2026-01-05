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
#include "Sparsity/SparsityPattern.h"

#include <cmath>

#if defined(FE_HAS_EIGEN)
#include "Backends/Eigen/EigenFactory.h"
#endif

namespace svmp::FE::backends {

namespace {

sparsity::SparsityPattern make_2x2_pattern()
{
    sparsity::SparsityPattern p(2, 2);
    p.addEntry(0, 0);
    p.addEntry(0, 1);
    p.addEntry(1, 0);
    p.addEntry(1, 1);
    p.finalize();
    return p;
}

sparsity::SparsityPattern make_diag_2x2_pattern()
{
    sparsity::SparsityPattern p(2, 2);
    p.addEntry(0, 0);
    p.addEntry(1, 1);
    p.finalize();
    return p;
}

} // namespace

TEST(EigenBackend, MatrixViewInsertion)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_2x2_pattern();
    auto A = factory.createMatrix(pattern);
    auto view = A->createAssemblyView();

    view->beginAssemblyPhase();
    {
        const GlobalIndex dofs[2] = {0, 1};
        const Real Ke[4] = {4.0, 1.0,
                            1.0, 3.0};
        view->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    }
    view->endAssemblyPhase();
    view->finalizeAssembly();
    A->finalizeAssembly();

    EXPECT_DOUBLE_EQ(view->getMatrixEntry(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(1, 1), 3.0);

    // Add again, now doubles.
    view->beginAssemblyPhase();
    {
        const GlobalIndex dofs[2] = {0, 1};
        const Real Ke[4] = {4.0, 1.0,
                            1.0, 3.0};
        view->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    }
    view->finalizeAssembly();
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(0, 0), 8.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(1, 1), 6.0);
#endif
}

TEST(EigenBackend, VectorViewAndOps)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    auto v = factory.createVector(3);
    ASSERT_TRUE(v);
    EXPECT_EQ(v->size(), 3);

    v->set(2.0);
    for (const auto value : v->localSpan()) {
        EXPECT_DOUBLE_EQ(value, 2.0);
    }

    v->add(1.0);
    for (const auto value : v->localSpan()) {
        EXPECT_DOUBLE_EQ(value, 3.0);
    }

    v->scale(0.5);
    for (const auto value : v->localSpan()) {
        EXPECT_DOUBLE_EQ(value, 1.5);
    }

    auto w = factory.createVector(3);
    ASSERT_TRUE(w);
    w->set(1.0);
    EXPECT_NEAR(v->dot(*w), 4.5, 1e-14);
    EXPECT_NEAR(v->norm(), std::sqrt(3.0 * 1.5 * 1.5), 1e-14);

    auto view = v->createAssemblyView();
    ASSERT_TRUE(view);
    EXPECT_TRUE(view->hasVector());
    EXPECT_FALSE(view->hasMatrix());
    EXPECT_EQ(view->numRows(), 3);

    view->beginAssemblyPhase();
    {
        const GlobalIndex dofs[3] = {0, 1, 2};
        const Real vals[3] = {1.0, 2.0, 3.0};
        view->addVectorEntries(dofs, vals, assembly::AddMode::Insert);
    }

    view->addVectorEntry(1, 5.0, assembly::AddMode::Add); // 2 + 5 = 7
    view->addVectorEntry(2, 10.0, assembly::AddMode::Max); // max(3, 10) = 10
    view->addVectorEntry(0, -4.0, assembly::AddMode::Min); // min(1, -4) = -4
    view->finalizeAssembly();

    EXPECT_DOUBLE_EQ(view->getVectorEntry(0), -4.0);
    EXPECT_DOUBLE_EQ(view->getVectorEntry(1), 7.0);
    EXPECT_DOUBLE_EQ(view->getVectorEntry(2), 10.0);

    const GlobalIndex to_zero[2] = {0, 2};
    view->zeroVectorEntries(to_zero);
    EXPECT_DOUBLE_EQ(view->getVectorEntry(0), 0.0);
    EXPECT_DOUBLE_EQ(view->getVectorEntry(2), 0.0);
#endif
}

TEST(EigenBackend, MatrixMult2x2)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_2x2_pattern();
    auto A = factory.createMatrix(pattern);
    auto x = factory.createVector(2);
    auto y = factory.createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto xs = x->localSpan();
    xs[0] = 1.0;
    xs[1] = 2.0;

    A->mult(*x, *y);

    const auto ys = y->localSpan();
    ASSERT_EQ(ys.size(), 2u);
    EXPECT_DOUBLE_EQ(ys[0], 6.0);
    EXPECT_DOUBLE_EQ(ys[1], 7.0);
#endif
}

TEST(EigenBackend, SolveDirect2x2)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
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
    opts.method = SolverMethod::Direct;
    opts.rel_tol = 1e-14;
    opts.abs_tol = 1e-14;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);

    EXPECT_TRUE(rep.converged);
    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-10);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-10);
#endif
}

TEST(EigenBackend, SolveCG2x2)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_2x2_pattern();
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
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-14;
    opts.max_iter = 100;
    opts.use_initial_guess = false;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);
    EXPECT_GT(rep.iterations, 0);
    EXPECT_EQ(rep.message, "iterative");

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-10);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-10);
#endif
}

TEST(EigenBackend, SolveGMRES2x2)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_2x2_pattern();
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
    opts.rel_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-10);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-10);
#endif
}

TEST(EigenBackend, SolveFGMRES2x2)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_2x2_pattern();
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
    opts.method = SolverMethod::FGMRES;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-10);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-10);
#endif
}

TEST(EigenBackend, SolveBlockSchurFallback2x2)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_2x2_pattern();
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
    opts.method = SolverMethod::BlockSchur;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-14;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-10);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-10);
#endif
}

TEST(EigenBackend, SolveBiCGSTAB2x2NonSym)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_2x2_pattern();
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(2);
    auto x = factory.createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        2.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    const Real be[2] = {1.0, 2.0};
    viewb->addVectorEntries(dofs, be, assembly::AddMode::Insert);
    viewb->finalizeAssembly();

    auto xs0 = x->localSpan();
    xs0[0] = 100.0;
    xs0[1] = -100.0;

    SolverOptions opts;
    opts.method = SolverMethod::BiCGSTAB;
    opts.preconditioner = PreconditionerType::Diagonal;
    opts.rel_tol = 1e-14;
    opts.max_iter = 100;
    opts.use_initial_guess = true;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);
    EXPECT_GT(rep.iterations, 0);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 0.1, 1e-10);
    EXPECT_NEAR(xs[1], 0.6, 1e-10);
#endif
}

TEST(EigenBackend, SolveBiCGSTABILU2x2NonSym)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_2x2_pattern();
    auto A = factory.createMatrix(pattern);
    auto b = factory.createVector(2);
    auto x = factory.createVector(2);

    auto viewA = A->createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        2.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    viewA->finalizeAssembly();
    A->finalizeAssembly();

    auto viewb = b->createAssemblyView();
    viewb->beginAssemblyPhase();
    const Real be[2] = {1.0, 2.0};
    viewb->addVectorEntries(dofs, be, assembly::AddMode::Insert);
    viewb->finalizeAssembly();

    SolverOptions opts;
    opts.method = SolverMethod::BiCGSTAB;
    opts.preconditioner = PreconditionerType::ILU;
    opts.rel_tol = 1e-14;
    opts.max_iter = 100;
    opts.use_initial_guess = false;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 0.1, 1e-10);
    EXPECT_NEAR(xs[1], 0.6, 1e-10);
#endif
}

TEST(EigenBackend, StructurePreservingIgnoresMissingEntry)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_diag_2x2_pattern();
    auto A = factory.createMatrix(pattern);
    auto view = A->createAssemblyView();

    view->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {1.0, 2.0,
                        3.0, 4.0};
    view->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    view->finalizeAssembly();
    A->finalizeAssembly();

    EXPECT_DOUBLE_EQ(view->getMatrixEntry(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(1, 0), 0.0);
#endif
}

TEST(EigenBackend, ZeroRowsSetsDiagonal)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pattern = make_2x2_pattern();
    auto A = factory.createMatrix(pattern);
    auto view = A->createAssemblyView();

    view->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    view->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);

    const GlobalIndex row_to_zero[1] = {1};
    view->zeroRows(row_to_zero, true);
    view->finalizeAssembly();

    EXPECT_DOUBLE_EQ(view->getMatrixEntry(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(view->getMatrixEntry(1, 1), 1.0);
#endif
}

} // namespace svmp::FE::backends
