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

#if defined(FE_HAS_PETSC)
#include "Backends/Interfaces/BlockMatrix.h"
#include "Backends/Interfaces/BlockVector.h"
#include "Backends/PETSc/PetscFactory.h"
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

TEST(PetscBackend, SolveCG2x2)
{
#if !defined(FE_HAS_PETSC)
    GTEST_SKIP() << "FE_HAS_PETSC not enabled";
#else
    PetscFactory factory;
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
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(*A, *x, *b);
    EXPECT_TRUE(rep.converged);

    // A = [[4,1],[1,3]], b = [1,2] => x = [1/11, 7/11]
    const auto xs = x->localSpan();
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[0], 1.0 / 11.0, 1e-10);
    EXPECT_NEAR(xs[1], 7.0 / 11.0, 1e-10);
#endif
}

TEST(PetscBackend, SolveGMRES2x2)
{
#if !defined(FE_HAS_PETSC)
    GTEST_SKIP() << "FE_HAS_PETSC not enabled";
#else
    PetscFactory factory;
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

TEST(PetscBackend, SolveILUGMRES2x2)
{
#if !defined(FE_HAS_PETSC)
    GTEST_SKIP() << "FE_HAS_PETSC not enabled";
#else
    PetscFactory factory;
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

TEST(PetscBackend, SolveFieldSplit2x2AsBlocks)
{
#if !defined(FE_HAS_PETSC)
    GTEST_SKIP() << "FE_HAS_PETSC not enabled";
#else
    PetscFactory factory;

    sparsity::SparsityPattern pat11(1, 1);
    pat11.addEntry(0, 0);
    pat11.finalize();

    std::vector<std::vector<std::unique_ptr<GenericMatrix>>> blocks(2);
    blocks[0].push_back(factory.createMatrix(pat11));
    blocks[0].push_back(factory.createMatrix(pat11));
    blocks[1].push_back(factory.createMatrix(pat11));
    blocks[1].push_back(factory.createMatrix(pat11));

    BlockMatrix A({1, 1}, {1, 1}, std::move(blocks));

    // Monolithic 2x2 system assembled into 2x2 blocks.
    auto viewA = A.createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {4.0, 1.0,
                        1.0, 3.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Add);
    viewA->finalizeAssembly();
    A.finalizeAssembly();

    std::vector<std::unique_ptr<GenericVector>> b_blocks;
    b_blocks.push_back(factory.createVector(1));
    b_blocks.push_back(factory.createVector(1));
    BlockVector b(std::move(b_blocks));
    b.block(0).localSpan()[0] = 1.0;
    b.block(1).localSpan()[0] = 2.0;

    std::vector<std::unique_ptr<GenericVector>> x_blocks;
    x_blocks.push_back(factory.createVector(1));
    x_blocks.push_back(factory.createVector(1));
    BlockVector x(std::move(x_blocks));

    SolverOptions opts;
    opts.method = SolverMethod::CG;
    opts.preconditioner = PreconditionerType::FieldSplit;
    opts.fieldsplit.kind = FieldSplitKind::Additive;
    opts.rel_tol = 1e-12;
    opts.max_iter = 200;

    auto solver = factory.createLinearSolver(opts);
    const auto rep = solver->solve(A, x, b);
    EXPECT_TRUE(rep.converged);

    // Solution matches monolithic solve.
    EXPECT_NEAR(x.block(0).localSpan()[0], 1.0 / 11.0, 1e-8);
    EXPECT_NEAR(x.block(1).localSpan()[0], 7.0 / 11.0, 1e-8);
#endif
}

} // namespace svmp::FE::backends
