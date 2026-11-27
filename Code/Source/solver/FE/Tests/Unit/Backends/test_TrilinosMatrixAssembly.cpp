/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/SparseMatrixSummaryScanner.h"
#include "Backends/Trilinos/TrilinosMatrix.h"
#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include <gtest/gtest.h>
#include <array>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Backends/Trilinos/TrilinosVector.h"
#include "Tests/Support/DistributedFieldCoefficients.h"
#endif

namespace svmp::FE::backends {
namespace {

class TrilinosMatrixAssembly : public ::testing::Test {
protected:
    const int rank = Tpetra::getDefaultComm()->getRank();
    const int ranks = Tpetra::getDefaultComm()->getSize();
    const GlobalIndex first = 3 * rank;
    const GlobalIndex neighbor = 3 * ((rank + 1) % ranks);

    sparsity::DistributedSparsityPattern pattern(bool include_master = false) const
    {
        sparsity::DistributedSparsityPattern p(first, 3, first, 3, 3 * ranks, 3 * ranks);
        p.addEntry(first, first);
        p.addEntry(first + 1, first + 1);
        if (include_master) {
            p.addEntry(first, first + 1);
            p.addEntry(first + 1, first);
        }
        if (ranks > 1) p.addEntry(first + 1, neighbor);
        p.finalize();
        return p;
    }
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST_F(TrilinosMatrixAssembly, AssemblesHistoryAndInitializesOwnedVertices)
{
    svmp::FE::test::checkDistributedFieldCoefficients(rank, ranks,
        [](GlobalIndex first, GlobalIndex count, const std::vector<GlobalIndex>& ghosts) {
            return std::make_unique<TrilinosVector>(first, count, 3, ghosts);
        });
}
#endif

TEST_F(TrilinosMatrixAssembly, MissingOwnedCoefficientsFailOnEveryRank)
{
    TrilinosMatrix matrix(pattern());
    auto view = matrix.createAssemblyView();
    // Column 1 is in the local map but absent from row 0; column 2 is absent
    // from the map altogether. Only rank zero encounters either defect.
    for (GlobalIndex col : {1, 2}) {
        SCOPED_TRACE(col);
        EXPECT_THROW({
            if (rank == 0) view->addMatrixEntry(0, col, 2.5e-30, assembly::AddMode::Add);
            view->finalizeAssembly();
        }, FEException);
        matrix.zero();
    }
}

TEST_F(TrilinosMatrixAssembly, DenseBlockRejectsStaleGraphBeforeChangingAnyEntry)
{
    TrilinosMatrix matrix(pattern());
    auto view = matrix.createAssemblyView();
    const std::array<GlobalIndex, 2> dofs{first, first + 1};
    const std::array<Real, 4> reduced{4., 2.5, 3., 2.};
    EXPECT_THROW({
        if (rank == 0) view->addMatrixEntries(dofs, reduced, assembly::AddMode::Add);
        matrix.finalizeAssembly();
    }, FEException);
    EXPECT_DOUBLE_EQ(matrix.getEntry(first, first), 0.);

    TrilinosMatrix rebuilt(pattern(true));
    auto expanded = rebuilt.createAssemblyView();
    expanded->addMatrixEntries(dofs, reduced, assembly::AddMode::Add);
    expanded->finalizeAssembly();
    for (std::size_t i = 0; i < dofs.size(); ++i)
        for (std::size_t j = 0; j < dofs.size(); ++j)
            EXPECT_DOUBLE_EQ(rebuilt.getEntry(dofs[i], dofs[j]), reduced[2 * i + j]);

    trilinos::Vector ones(rebuilt.tpetra()->getDomainMap());
    trilinos::Vector result(rebuilt.tpetra()->getRangeMap());
    ones.putScalar(1.);
    rebuilt.tpetra()->apply(ones, result);
    const auto values = result.getData();
    EXPECT_DOUBLE_EQ(values[0], 6.5);
    EXPECT_DOUBLE_EQ(values[1], 5.);
}

TEST_F(TrilinosMatrixAssembly, StoredZerosAndStructuralZerosRespectAllInsertionModes)
{
    TrilinosMatrix matrix(pattern());
    auto view = matrix.createAssemblyView();
    const std::array<GlobalIndex, 2> dofs{first, first + 1};
    const std::array<Real, 4> diagonal{4., 0., 0., -2.};
    view->addMatrixEntries(dofs, diagonal, assembly::AddMode::Add);
    EXPECT_DOUBLE_EQ(matrix.getEntry(first, first), 4.);
    EXPECT_DOUBLE_EQ(matrix.getEntry(first + 1, first + 1), -2.);
    for (auto mode : {assembly::AddMode::Add, assembly::AddMode::Insert,
                      assembly::AddMode::Min, assembly::AddMode::Max}) {
        view->addMatrixEntry(first, first + 2, 0., mode);
    }
    view->addMatrixEntry(first, first, 0., assembly::AddMode::Min);
    view->addMatrixEntry(first + 1, first + 1, 0., assembly::AddMode::Max);
    EXPECT_DOUBLE_EQ(matrix.getEntry(first, first), 0.);
    EXPECT_DOUBLE_EQ(matrix.getEntry(first + 1, first + 1), 0.);
    view->addMatrixEntry(first, first, 8., assembly::AddMode::Insert);
    view->addMatrixEntry(first, first, 0., assembly::AddMode::Insert);
    view->finalizeAssembly();
    EXPECT_DOUBLE_EQ(matrix.getEntry(first, first), 0.);
}

TEST_F(TrilinosMatrixAssembly, MissingRequiredDiagonalPreservesEarlierRows)
{
    TrilinosMatrix matrix(pattern());
    auto view = matrix.createAssemblyView();
    view->setDiagonal(first, 7.);
    const std::array<GlobalIndex, 2> rows{first, first + 2};
    EXPECT_THROW({
        if (rank == 0) view->zeroRows(rows, true);
        view->finalizeAssembly();
    }, FEException);
    EXPECT_DOUBLE_EQ(matrix.getEntry(first, first), 7.);
    matrix.zero();
    view->setDiagonal(first, 4.);
    const std::array<Real, 2> values{2., 1.};
    EXPECT_THROW({
        if (rank == 0) view->setDiagonal(rows, values);
        view->finalizeAssembly();
    }, FEException);
    EXPECT_DOUBLE_EQ(matrix.getEntry(first, first), 4.);
}

TEST_F(TrilinosMatrixAssembly, OwnedRowsAcceptRemoteColumnsInTheirGraph)
{
    if (ranks < 2) GTEST_SKIP() << "Requires a distributed matrix";
    TrilinosMatrix matrix(pattern());
    auto view = matrix.createAssemblyView();
    view->addMatrixEntry(first + 1, neighbor, 3.5, assembly::AddMode::Add);
    view->finalizeAssembly();
    EXPECT_DOUBLE_EQ(matrix.getEntry(first + 1, neighbor), 3.5);
    EXPECT_THROW(view->addMatrixEntry(neighbor, neighbor, 1., assembly::AddMode::Add),
                 NotImplementedException);
}

TEST_F(TrilinosMatrixAssembly, SparseScannerVisitsEveryOwnedRowIncludingEmptyRows)
{
    TrilinosMatrix matrix(pattern());
    auto view = matrix.createAssemblyView();
    view->setDiagonal(first, 7.);
    view->finalizeAssembly();

    const auto scan = analysis::makeSparseRowScanSource(matrix);
    ASSERT_TRUE(scan);
    EXPECT_EQ(scan->hasCompleteGlobalRows(), ranks == 1);
    std::vector<GlobalIndex> rows;
    scan->forEachLocalRow([&](GlobalIndex row, const auto& entries, int owner) {
        rows.push_back(row);
        EXPECT_EQ(owner, rank);
        if (row == first) {
            ASSERT_EQ(entries.size(), 1u);
            EXPECT_EQ(entries.front().col, first);
            EXPECT_DOUBLE_EQ(entries.front().value, 7.);
        }
        if (row == first + 2) EXPECT_TRUE(entries.empty());
    });
    EXPECT_EQ(rows, (std::vector<GlobalIndex>{first, first + 1, first + 2}));
}

#ifdef _OPENMP
TEST_F(TrilinosMatrixAssembly, ParallelAssemblyOnDisjointRows)
{
    TrilinosMatrix matrix(pattern());
    auto view = matrix.createAssemblyView();
    // The colored element loop can write disjoint rows from different threads.
#pragma omp parallel for num_threads(2)
    for (int i = 0; i < 2; ++i) {
        for (int cell = 0; cell < 2000; ++cell) {
            view->addMatrixEntry(first + i, first + i, 1., assembly::AddMode::Add);
        }
    }
    view->finalizeAssembly();
    EXPECT_DOUBLE_EQ(matrix.getEntry(first, first), 2000.);
    EXPECT_DOUBLE_EQ(matrix.getEntry(first + 1, first + 1), 2000.);
}
#endif

} // namespace
} // namespace svmp::FE::backends
