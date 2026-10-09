/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Distributed MUMPS solves (FE_ENABLE_MUMPS builds) on any rank count:
// symmetric positive definite and unsymmetric 2D grid operators against the
// exact solution and the single-rank solve, analysis reuse while the pattern
// is unchanged, and collective error reporting.

#include <gtest/gtest.h>

#include "Backends/MUMPS/MumpsDistributedSolver.h"

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace svmp::FE::backends {
namespace {

constexpr int kGrid = 23;  // kGrid x kGrid unknowns

struct Triplets {
    std::vector<GlobalIndex> rows;
    std::vector<GlobalIndex> cols;
    std::vector<Real> values;
};

/// 5-point operator on the grid: diagonal `diag`, west/east/south/north
/// couplings; `lower_only` keeps entries with col <= row.  Rows are dealt
/// to ranks by row % size.
Triplets gridOperator(int rank, int size, Real diag, Real west, Real east, Real south, Real north,
                      bool lower_only, Real scale = 1.0)
{
    Triplets t;
    const auto id = [](int i, int j) { return static_cast<GlobalIndex>(j * kGrid + i); };
    for (int j = 0; j < kGrid; ++j) {
        for (int i = 0; i < kGrid; ++i) {
            const auto row = id(i, j);
            if (row % size != rank) {
                continue;
            }
            const auto add = [&](GlobalIndex col, Real v) {
                if (lower_only && col > row) {
                    return;
                }
                t.rows.push_back(row);
                t.cols.push_back(col);
                t.values.push_back(scale * v);
            };
            add(row, diag);
            if (i > 0) add(id(i - 1, j), west);
            if (i + 1 < kGrid) add(id(i + 1, j), east);
            if (j > 0) add(id(i, j - 1), south);
            if (j + 1 < kGrid) add(id(i, j + 1), north);
        }
    }
    return t;
}

Real exactValue(GlobalIndex k)
{
    return 1.0 + std::sin(0.13 * static_cast<Real>(k)) + 0.01 * static_cast<Real>(k % 7);
}

/// b = A x_exact for the full 5-point operator.
std::vector<Real> rhsFor(Real diag, Real west, Real east, Real south, Real north, Real scale = 1.0)
{
    std::vector<Real> b(static_cast<std::size_t>(kGrid * kGrid), 0.0);
    for (int j = 0; j < kGrid; ++j) {
        for (int i = 0; i < kGrid; ++i) {
            const int row = j * kGrid + i;
            Real v = diag * exactValue(row);
            if (i > 0) v += west * exactValue(row - 1);
            if (i + 1 < kGrid) v += east * exactValue(row + 1);
            if (j > 0) v += south * exactValue(row - kGrid);
            if (j + 1 < kGrid) v += north * exactValue(row + kGrid);
            b[static_cast<std::size_t>(row)] = scale * v;
        }
    }
    return b;
}

Real maxError(const std::vector<Real>& x)
{
    Real err = 0.0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        err = std::max(err, std::abs(x[k] - exactValue(static_cast<GlobalIndex>(k))));
    }
    return err;
}

TEST(MumpsDistributedSolverMPI, SymmetricPositiveDefiniteAndUnsymmetricGridSolves)
{
    if (!mumpsAvailable()) {
        GTEST_SKIP() << "Built without FE_ENABLE_MUMPS";
    }
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const GlobalIndex n = kGrid * kGrid;

    // Symmetric positive definite: lower triangle only.
    MumpsDistributedSolver spd(MPI_COMM_WORLD, MumpsDistributedSolver::Symmetry::SymmetricPositiveDefinite);
    const auto ts = gridOperator(rank, size, 4.2, -1.0, -1.0, -1.0, -1.0, /*lower_only=*/true);
    ASSERT_TRUE(spd.factorize(n, ts.rows, ts.cols, ts.values)) << spd.lastError();
    std::vector<Real> x;
    ASSERT_TRUE(spd.solveReplicated(rhsFor(4.2, -1.0, -1.0, -1.0, -1.0), x)) << spd.lastError();
    EXPECT_LT(maxError(x), 1e-12);

    // Unsymmetric: all entries.
    MumpsDistributedSolver lu(MPI_COMM_WORLD, MumpsDistributedSolver::Symmetry::Unsymmetric);
    const auto tu = gridOperator(rank, size, 4.0, -1.3, -0.7, -1.1, -0.9, /*lower_only=*/false);
    ASSERT_TRUE(lu.factorize(n, tu.rows, tu.cols, tu.values)) << lu.lastError();
    std::vector<Real> y;
    ASSERT_TRUE(lu.solveReplicated(rhsFor(4.0, -1.3, -0.7, -1.1, -0.9), y)) << lu.lastError();
    EXPECT_LT(maxError(y), 1e-12);

    // The same systems on this rank alone.
    MumpsDistributedSolver self(MPI_COMM_SELF, MumpsDistributedSolver::Symmetry::Unsymmetric);
    const auto tself = gridOperator(0, 1, 4.0, -1.3, -0.7, -1.1, -0.9, false);
    ASSERT_TRUE(self.factorize(n, tself.rows, tself.cols, tself.values)) << self.lastError();
    std::vector<Real> z;
    ASSERT_TRUE(self.solveReplicated(rhsFor(4.0, -1.3, -0.7, -1.1, -0.9), z));
    Real spread = 0.0;
    for (std::size_t k = 0; k < z.size(); ++k) {
        spread = std::max(spread, std::abs(z[k] - y[k]));
    }
    EXPECT_LT(spread, 1e-13);

    // Every rank holds the same replicated solution.
    double local_sum = 0.0;
    for (const auto v : y) local_sum += v;
    double lo = 0.0;
    double hi = 0.0;
    MPI_Allreduce(&local_sum, &lo, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sum, &hi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(lo, hi);
}

TEST(MumpsDistributedSolverMPI, AnalysisIsKeptWhileThePatternIsUnchanged)
{
    if (!mumpsAvailable()) {
        GTEST_SKIP() << "Built without FE_ENABLE_MUMPS";
    }
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const GlobalIndex n = kGrid * kGrid;

    MumpsDistributedSolver solver(MPI_COMM_WORLD, MumpsDistributedSolver::Symmetry::Unsymmetric);
    std::vector<Real> first;
    std::vector<Real> again;
    for (int step = 0; step < 3; ++step) {
        const Real scale = 1.0 + 0.5 * step;
        const auto t = gridOperator(rank, size, 4.0, -1.3, -0.7, -1.1, -0.9, false, scale);
        ASSERT_TRUE(solver.factorize(n, t.rows, t.cols, t.values)) << solver.lastError();
        std::vector<Real> x;
        ASSERT_TRUE(solver.solveReplicated(rhsFor(4.0, -1.3, -0.7, -1.1, -0.9, scale), x));
        EXPECT_LT(maxError(x), 1e-12) << "step " << step;
        if (step == 0) {
            first = x;
        }
    }
    EXPECT_EQ(solver.statistics().analyses, 1u);
    EXPECT_EQ(solver.statistics().factorizations, 3u);
    EXPECT_GT(solver.statistics().factor_entries, 0);

    // Refactoring the first matrix reproduces the first solution.
    const auto t0 = gridOperator(rank, size, 4.0, -1.3, -0.7, -1.1, -0.9, false, 1.0);
    ASSERT_TRUE(solver.factorize(n, t0.rows, t0.cols, t0.values));
    ASSERT_TRUE(solver.solveReplicated(rhsFor(4.0, -1.3, -0.7, -1.1, -0.9, 1.0), again));
    Real diff = 0.0;
    for (std::size_t k = 0; k < again.size(); ++k) {
        diff = std::max(diff, std::abs(again[k] - first[k]));
    }
    EXPECT_LT(diff, 1e-14);
    if (rank == 0) {
        std::printf("MUMPS refactorization of the same matrix: %s (%d ranks)\n",
                    std::memcmp(again.data(), first.data(), again.size() * sizeof(Real)) == 0
                        ? "bitwise identical"
                        : "round-off differences",
                    size);
    }

    // A new pattern (one coupling dropped) triggers a new analysis.
    auto t1 = gridOperator(rank, size, 4.0, -1.3, -0.7, -1.1, 0.0, false, 1.0);
    std::vector<GlobalIndex> rows;
    std::vector<GlobalIndex> cols;
    std::vector<Real> values;
    for (std::size_t k = 0; k < t1.rows.size(); ++k) {
        if (t1.values[k] != 0.0) {
            rows.push_back(t1.rows[k]);
            cols.push_back(t1.cols[k]);
            values.push_back(t1.values[k]);
        }
    }
    ASSERT_TRUE(solver.factorize(n, rows, cols, values));
    EXPECT_EQ(solver.statistics().analyses, 2u);
}

TEST(MumpsDistributedSolverMPI, SingularMatrixIsReportedOnEveryRank)
{
    if (!mumpsAvailable()) {
        GTEST_SKIP() << "Built without FE_ENABLE_MUMPS";
    }
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const GlobalIndex n = kGrid * kGrid;
    // Row 5 empty except for a zero diagonal: structurally present, singular.
    auto t = gridOperator(rank, size, 4.0, -1.3, -0.7, -1.1, -0.9, false);
    for (std::size_t k = 0; k < t.rows.size(); ++k) {
        if (t.rows[k] == 5) {
            t.values[k] = 0.0;
        }
    }
    MumpsDistributedSolver solver(MPI_COMM_WORLD, MumpsDistributedSolver::Symmetry::Unsymmetric);
    const bool ok = solver.factorize(n, t.rows, t.cols, t.values);
    int local = ok ? 1 : 0;
    int any = 0;
    MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_FALSE(ok);
    EXPECT_EQ(any, 0);
    EXPECT_FALSE(solver.lastError().empty());
}

} // namespace
} // namespace svmp::FE::backends
