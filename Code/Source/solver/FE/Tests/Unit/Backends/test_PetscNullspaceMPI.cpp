/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#if defined(FE_HAS_PETSC)

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/BlockMatrix.h"
#include "Backends/Interfaces/BlockVector.h"
#include "Backends/PETSc/PetscLinearSolver.h"
#include "Backends/PETSc/PetscMatrix.h"
#include "Backends/PETSc/PetscVector.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <utility>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Tests/Support/DistributedFieldCoefficients.h"
#endif

namespace svmp::FE::backends {
namespace {

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(PetscDistributedField, AssemblesHistoryAndInitializesOwnedVertices)
{
    int rank = 0, ranks = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
    svmp::FE::test::checkDistributedFieldCoefficients(rank, ranks,
        [](GlobalIndex, GlobalIndex count, const std::vector<GlobalIndex>& ghosts) {
            return std::make_unique<PetscVector>(count, 3, ghosts);
        });
}
#endif

using sparsity::IndexRange;

constexpr int n = 6;
constexpr std::array<Real, n> exact{2., 3., -1., -1., -1., -2.};

IndexRange ownership()
{
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size == 1) return {0, n};
    if (rank == 0) return {0, 1};
    return {1, rank == size - 1 ? n : 1};
}

sparsity::DistributedSparsityPattern graph(IndexRange owned, int stride)
{
    sparsity::DistributedSparsityPattern pattern(owned, owned, n, n);
    for (auto row = owned.first; row < owned.last; ++row) {
        pattern.addEntry(row, row);
        if (row >= stride) pattern.addEntry(row, row - stride);
        if (row + stride < n) pattern.addEntry(row, row + stride);
    }
    pattern.finalize();
    return pattern;
}

// A graph Laplacian on one path (stride 1) or two interleaved paths (stride 2).
struct Problem {
    IndexRange owned = ownership();
    PetscMatrix matrix;
    PetscVector x{owned.size(), n}, rhs{owned.size(), n};

    explicit Problem(int stride) : matrix(graph(owned, stride))
    {
        auto a = matrix.createAssemblyView();
        auto b = rhs.createAssemblyView();
        for (auto row = owned.first; row < owned.last; ++row) {
            Real degree = 0., load = 0.;
            for (auto neighbor : {row - stride, row + stride}) {
                if (neighbor < 0 || neighbor >= n) continue;
                a->addMatrixEntry(row, neighbor, -1.);
                ++degree;
                load += exact[row] - exact[neighbor];
            }
            a->addMatrixEntry(row, row, degree);
            b->addVectorEntry(row, load, assembly::AddMode::Insert);
        }
        a->finalizeAssembly();
        b->finalizeAssembly();
    }
};

SolverOptions options()
{
    SolverOptions result;
    result.method = SolverMethod::CG;
    result.preconditioner = PreconditionerType::None;
    result.rel_tol = 1e-12;
    result.use_initial_guess = true;
    return result;
}

void checkSolve(Problem& problem, const std::vector<std::vector<double>>& basis)
{
    PetscLinearSolver solver(options());
    solver.setNullspaceBasis(basis);
    // A nonzero constant is an exact null component in either problem.
    problem.x.set(7.);
    const auto report = solver.solve(problem.matrix, problem.x, problem.rhs);
    EXPECT_TRUE(report.converged) << report.message;
    const auto solution = std::as_const(problem.x).localSpan();
    for (auto row = problem.owned.first; row < problem.owned.last; ++row) {
        EXPECT_NEAR(solution[row - problem.owned.first], exact[row], 1e-11);
    }
    PetscVector residual(problem.owned.size(), n);
    problem.matrix.mult(problem.x, residual);
    FE_PETSC_CALL(VecAXPY(residual.petsc(), -1., problem.rhs.petsc()));
    EXPECT_LT(residual.norm(), 1e-11);

    MatNullSpace space = nullptr;
    FE_PETSC_CALL(MatGetNullSpace(problem.matrix.petsc(), &space));
    ASSERT_NE(space, nullptr);
    int relation;
    MPI_Comm_compare(PETSC_COMM_WORLD, PetscObjectComm(reinterpret_cast<PetscObject>(space)), &relation);
    ASSERT_TRUE(relation == MPI_IDENT || relation == MPI_CONGRUENT);
    PetscBool valid, constant;
    PetscInt count;
    const Vec* modes;
    FE_PETSC_CALL(MatNullSpaceTest(space, problem.matrix.petsc(), &valid));
    EXPECT_EQ(valid, PETSC_TRUE);
    FE_PETSC_CALL(MatNullSpaceGetVecs(space, &constant, &count, &modes));
    EXPECT_EQ(count + (constant ? 1 : 0), basis.size());
    for (PetscInt i = 0; i < count; ++i) {
        PetscInt first, last;
        FE_PETSC_CALL(VecGetOwnershipRange(modes[i], &first, &last));
        EXPECT_EQ(first, problem.owned.first);
        EXPECT_EQ(last, problem.owned.last);
        for (PetscInt j = 0; j <= i; ++j) {
            PetscScalar dot;
            FE_PETSC_CALL(VecDot(modes[i], modes[j], &dot));
            EXPECT_NEAR(PetscRealPart(dot), i == j ? 1. : 0., 1e-12);
        }
    }
    solver.setNullspaceBasis({});
    EXPECT_TRUE(solver.solve(problem.matrix, problem.x, problem.rhs).converged);
    FE_PETSC_CALL(MatGetNullSpace(problem.matrix.petsc(), &space));
    EXPECT_EQ(space, nullptr);
}

TEST(PetscNullspace, ConstantModeOnAnUnevenPartition)
{
    Problem problem(1);
    checkSolve(problem, {std::vector<double>(n, 2.)});
}

TEST(PetscNullspace, NonorthogonalModesOnInterleavedComponents)
{
    Problem problem(2);
    checkSolve(problem, {std::vector<double>(n, 2.), {3., 0., 3., 0., 3., 0.}});
}

TEST(PetscNullspace, CsrGraphStillRejectsAnUndeclaredColumn)
{
    Problem problem(1);
    if (!problem.owned.empty()) {
        auto view = problem.matrix.createAssemblyView();
        FE_PETSC_CALL(PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
        EXPECT_THROW(view->addMatrixEntry(problem.owned.first, (problem.owned.first + 3) % n, 1.), FEException);
        FE_PETSC_CALL(PetscPopErrorHandler());
    }
}

TEST(PetscNullspace, NestedModeUsesGlobalFieldOffsetsAcrossRanks)
{
    Problem first(1), second(1);
    std::vector<std::vector<std::unique_ptr<GenericMatrix>>> blocks(2);
    for (auto& row : blocks) row.resize(2);
    blocks[0][0] = std::make_unique<PetscMatrix>(std::move(first.matrix));
    blocks[1][1] = std::make_unique<PetscMatrix>(std::move(second.matrix));
    BlockMatrix matrix({n, n}, {n, n}, std::move(blocks));
    std::vector<std::unique_ptr<GenericVector>> guesses, loads;
    for (Problem* problem : {&first, &second}) {
        guesses.push_back(std::make_unique<PetscVector>(std::move(problem->x)));
        loads.push_back(std::make_unique<PetscVector>(std::move(problem->rhs)));
    }
    BlockVector x(std::move(guesses)), rhs(std::move(loads));
    x.block(0).set(7.);
    x.block(1).set(-3.);
    std::vector<std::vector<double>> modes(2, std::vector<double>(2 * n));
    for (int i = 0; i < n; ++i) {
        modes[0][i] = 1.;
        modes[1][n + i] = 1.;
    }
    PetscLinearSolver solver(options());
    solver.setNullspaceBasis(modes);
    EXPECT_TRUE(solver.solve(matrix, x, rhs).converged);
    for (std::size_t block = 0; block < x.numBlocks(); ++block) {
        const auto result = std::as_const(x).block(block).localSpan();
        for (auto row = first.owned.first; row < first.owned.last; ++row) {
            EXPECT_NEAR(result[row - first.owned.first], exact[row], 1e-11);
        }
    }
}

TEST(PetscNullspace, CollectivelyRejectsInvalidModesAndRecovers)
{
    Problem problem(1);
    PetscLinearSolver solver(options());
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    const std::vector<double> constant(n, 1.);
    auto nonfinite = constant;
    if (rank == 0) nonfinite[0] = std::numeric_limits<double>::quiet_NaN();
    auto short_mode = constant;
    if (rank == 0) short_mode.pop_back();
    std::vector<std::vector<std::vector<double>>> invalid{
        {short_mode}, {nonfinite}, {std::vector<double>(n, 0.)}, {constant, constant},
        {{1., 0., 0., 0., 0., 0.}}};
    if (size > 1) invalid.push_back(rank == 0 ? std::vector<std::vector<double>>{constant}
                                              : std::vector<std::vector<double>>{});
    for (const auto& modes : invalid) {
        EXPECT_THROW({
            solver.setNullspaceBasis(modes);
            solver.solve(problem.matrix, problem.x, problem.rhs);
        }, FEException);
    }
    solver.setNullspaceBasis(std::vector<std::vector<double>>{constant});
    EXPECT_TRUE(solver.solve(problem.matrix, problem.x, problem.rhs).converged);
}

TEST(PetscNullspace, ConfigurationOnOnlyOneRankFailsCollectively)
{
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    if (size == 1) GTEST_SKIP() << "requires multiple ranks";
    Problem problem(1);
    PetscLinearSolver solver(options());
    if (rank == 0) solver.setNullspaceBasis(std::vector<std::vector<double>>(1, std::vector<double>(n, 1.)));
    EXPECT_THROW((void)solver.solve(problem.matrix, problem.x, problem.rhs), FEException);
}

TEST(PetscNullspace, PreservesCallerAttachedSpaceWhenSolverConfigurationIsUnset)
{
    Problem problem(1);
    MatNullSpace external;
    FE_PETSC_CALL(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, nullptr, &external));
    FE_PETSC_CALL(MatSetNullSpace(problem.matrix.petsc(), external));
    PetscLinearSolver solver(options());
    EXPECT_TRUE(solver.solve(problem.matrix, problem.x, problem.rhs).converged);
    MatNullSpace attached;
    FE_PETSC_CALL(MatGetNullSpace(problem.matrix.petsc(), &attached));
    EXPECT_EQ(attached, external);
    FE_PETSC_CALL(MatNullSpaceDestroy(&external));
}

} // namespace
} // namespace svmp::FE::backends

#endif
