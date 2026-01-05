/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/Utils/BackendOptions.h"
#include "Core/FEException.h"
#include "Sparsity/SparsityPattern.h"

#include "Backends/FSILS/FsilsFactory.h"

#include <cmath>

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

    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-8);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-8);
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

} // namespace svmp::FE::backends
