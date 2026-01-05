/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/Utils/BackendOptions.h"
#include "Sparsity/SparsityPattern.h"

#if defined(FE_HAS_TRILINOS)
#include "Backends/Trilinos/TrilinosFactory.h"
#endif

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

} // namespace

TEST(TrilinosBackend, SolveCG2x2)
{
#if !defined(FE_HAS_TRILINOS)
    GTEST_SKIP() << "FE_HAS_TRILINOS not enabled";
#else
    TrilinosFactory factory;
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
    opts.rel_tol = 1e-12;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-8);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-8);
#endif
}

TEST(TrilinosBackend, SolveGMRES2x2)
{
#if !defined(FE_HAS_TRILINOS)
    GTEST_SKIP() << "FE_HAS_TRILINOS not enabled";
#else
    TrilinosFactory factory;
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
    opts.rel_tol = 1e-12;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-8);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-8);
#endif
}

TEST(TrilinosBackend, SolveILUGMRES2x2)
{
#if !defined(FE_HAS_TRILINOS)
    GTEST_SKIP() << "FE_HAS_TRILINOS not enabled";
#else
    TrilinosFactory factory;
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
    opts.preconditioner = PreconditionerType::ILU;
    opts.rel_tol = 1e-12;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);

    SolverReport rep;
    try {
        rep = solver->solve(*A, *x, *b);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Ifpack2 ILU not available/configured: " << e.what();
    }
    EXPECT_TRUE(rep.converged);

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-8);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-8);
#endif
}

} // namespace svmp::FE::backends
