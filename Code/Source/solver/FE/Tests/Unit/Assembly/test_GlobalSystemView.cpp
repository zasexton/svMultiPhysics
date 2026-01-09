/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GlobalSystemView.cpp
 * @brief Unit tests for GlobalSystemView implementations
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/RemappedSystemView.h"

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string_view>
#include <thread>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// DenseMatrixView Tests
// ============================================================================

class DenseMatrixViewTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_dofs_ = 4;
        matrix_ = std::make_unique<DenseMatrixView>(n_dofs_);
    }

    GlobalIndex n_dofs_;
    std::unique_ptr<DenseMatrixView> matrix_;
};

TEST_F(DenseMatrixViewTest, Construction) {
    EXPECT_EQ(matrix_->numRows(), n_dofs_);
    EXPECT_EQ(matrix_->numCols(), n_dofs_);
    EXPECT_TRUE(matrix_->hasMatrix());
    EXPECT_FALSE(matrix_->hasVector());
    EXPECT_EQ(matrix_->backendName(), "DenseMatrix");
}

TEST_F(DenseMatrixViewTest, RectangularConstruction) {
    DenseMatrixView rect(3, 5);
    EXPECT_EQ(rect.numRows(), 3);
    EXPECT_EQ(rect.numCols(), 5);
}

TEST_F(DenseMatrixViewTest, AddSingleEntry) {
    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntry(1, 2, 3.14);
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(1, 2), 3.14);
    EXPECT_DOUBLE_EQ((*matrix_)(0, 0), 0.0);
}

TEST_F(DenseMatrixViewTest, AddModeAdd) {
    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntry(1, 1, 1.0, AddMode::Add);
    matrix_->addMatrixEntry(1, 1, 2.0, AddMode::Add);
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(1, 1), 3.0);
}

TEST_F(DenseMatrixViewTest, AddModeInsert) {
    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntry(1, 1, 1.0, AddMode::Add);
    matrix_->addMatrixEntry(1, 1, 2.0, AddMode::Insert);
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(1, 1), 2.0);
}

TEST_F(DenseMatrixViewTest, AddModeMax) {
    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntry(1, 1, 5.0, AddMode::Insert);
    matrix_->addMatrixEntry(1, 1, 3.0, AddMode::Max);
    matrix_->addMatrixEntry(1, 1, 7.0, AddMode::Max);
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(1, 1), 7.0);
}

TEST_F(DenseMatrixViewTest, AddModeMin) {
    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntry(1, 1, 5.0, AddMode::Insert);
    matrix_->addMatrixEntry(1, 1, 3.0, AddMode::Min);
    matrix_->addMatrixEntry(1, 1, 7.0, AddMode::Min);
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(1, 1), 3.0);
}

TEST_F(DenseMatrixViewTest, AddModeTransitionsAddInsertAdd) {
    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntry(0, 0, 1.0, AddMode::Add);
    matrix_->addMatrixEntry(0, 0, 10.0, AddMode::Insert);
    matrix_->addMatrixEntry(0, 0, 2.0, AddMode::Add);
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(0, 0), 12.0);
}

TEST_F(DenseMatrixViewTest, AddMatrixEntriesEmptySpansNoOp) {
    const std::vector<GlobalIndex> empty;
    const std::vector<Real> empty_vals;

    matrix_->beginAssemblyPhase();
    EXPECT_NO_THROW(matrix_->addMatrixEntries(empty, empty, empty_vals));
    EXPECT_NO_THROW(matrix_->addMatrixEntries(std::vector<GlobalIndex>{0, 1}, empty, empty_vals));
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*matrix_)(1, 1), 0.0);
}

TEST_F(DenseMatrixViewTest, AddMatrixEntriesMismatchedSizesThrows) {
    const std::vector<GlobalIndex> dofs = {0, 1};
    const std::vector<Real> wrong = {1.0, 2.0, 3.0};

    matrix_->beginAssemblyPhase();
    EXPECT_THROW(matrix_->addMatrixEntries(dofs, wrong), std::invalid_argument);
    matrix_->endAssemblyPhase();
}

TEST_F(DenseMatrixViewTest, IsDistributedFalseForDenseViews) {
    EXPECT_FALSE(matrix_->isDistributed());
}

TEST(DenseMatrixViewThreadSafety, ConcurrentAddsToDifferentLocations) {
    DenseMatrixView matrix(200);
    matrix.beginAssemblyPhase();

    auto worker = [&matrix](GlobalIndex base) {
        for (GlobalIndex i = 0; i < 50; ++i) {
            const GlobalIndex idx = base + i;
            matrix.addMatrixEntry(idx, idx, 1.0, AddMode::Add);
        }
    };

    std::thread t0(worker, 0);
    std::thread t1(worker, 50);
    std::thread t2(worker, 100);
    std::thread t3(worker, 150);

    t0.join();
    t1.join();
    t2.join();
    t3.join();

    matrix.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(matrix.getMatrixEntry(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.getMatrixEntry(199, 199), 1.0);
    EXPECT_DOUBLE_EQ(matrix.getMatrixEntry(75, 75), 1.0);
}

TEST(DenseMatrixViewStress, VeryLargeMatrixConstructionOptional) {
    const char* v = std::getenv("SVMP_FE_RUN_STRESS_TESTS");
    if (v == nullptr || std::string_view(v) != "1") {
        GTEST_SKIP() << "Set SVMP_FE_RUN_STRESS_TESTS=1 to enable";
    }

    try {
        DenseMatrixView matrix(10000, 10000);
        EXPECT_EQ(matrix.numRows(), 10000);
        EXPECT_EQ(matrix.numCols(), 10000);
    } catch (const std::bad_alloc&) {
        GTEST_SKIP() << "Insufficient memory for 10000x10000 DenseMatrixView";
    }
}

TEST_F(DenseMatrixViewTest, AddSquareBlock) {
    std::vector<GlobalIndex> dofs = {0, 2};
    // 2x2 block: [[1, 2], [3, 4]]
    std::vector<Real> block = {1.0, 2.0, 3.0, 4.0};

    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntries(dofs, block);
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*matrix_)(0, 2), 2.0);
    EXPECT_DOUBLE_EQ((*matrix_)(2, 0), 3.0);
    EXPECT_DOUBLE_EQ((*matrix_)(2, 2), 4.0);
}

TEST_F(DenseMatrixViewTest, AddRectangularBlock) {
    std::vector<GlobalIndex> row_dofs = {0, 1};
    std::vector<GlobalIndex> col_dofs = {1, 2, 3};
    // 2x3 block
    std::vector<Real> block = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntries(row_dofs, col_dofs, block);
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(0, 1), 1.0);
    EXPECT_DOUBLE_EQ((*matrix_)(0, 2), 2.0);
    EXPECT_DOUBLE_EQ((*matrix_)(0, 3), 3.0);
    EXPECT_DOUBLE_EQ((*matrix_)(1, 1), 4.0);
    EXPECT_DOUBLE_EQ((*matrix_)(1, 2), 5.0);
    EXPECT_DOUBLE_EQ((*matrix_)(1, 3), 6.0);
}

TEST_F(DenseMatrixViewTest, SetDiagonal) {
    matrix_->beginAssemblyPhase();
    matrix_->setDiagonal(2, 5.0);
    matrix_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*matrix_)(2, 2), 5.0);
}

TEST_F(DenseMatrixViewTest, ZeroRows) {
    // First fill matrix
    matrix_->beginAssemblyPhase();
    for (GlobalIndex i = 0; i < n_dofs_; ++i) {
        for (GlobalIndex j = 0; j < n_dofs_; ++j) {
            matrix_->addMatrixEntry(i, j, static_cast<Real>(i * n_dofs_ + j));
        }
    }

    // Zero row 1 with diagonal = 1
    std::vector<GlobalIndex> rows = {1};
    matrix_->zeroRows(rows, true);
    matrix_->endAssemblyPhase();

    // Row 1 should be all zeros except diagonal
    EXPECT_DOUBLE_EQ((*matrix_)(1, 0), 0.0);
    EXPECT_DOUBLE_EQ((*matrix_)(1, 1), 1.0);  // diagonal set to 1
    EXPECT_DOUBLE_EQ((*matrix_)(1, 2), 0.0);
    EXPECT_DOUBLE_EQ((*matrix_)(1, 3), 0.0);

    // Other rows unchanged
    EXPECT_DOUBLE_EQ((*matrix_)(0, 0), 0.0);
    EXPECT_DOUBLE_EQ((*matrix_)(2, 0), 8.0);
}

TEST_F(DenseMatrixViewTest, IsSymmetric) {
    // Create symmetric matrix
    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntry(0, 0, 1.0);
    matrix_->addMatrixEntry(0, 1, 2.0);
    matrix_->addMatrixEntry(1, 0, 2.0);
    matrix_->addMatrixEntry(1, 1, 3.0);
    matrix_->endAssemblyPhase();

    EXPECT_TRUE(matrix_->isSymmetric());

    // Make it non-symmetric
    matrix_->addMatrixEntry(0, 1, 0.5, AddMode::Add);
    EXPECT_FALSE(matrix_->isSymmetric());
}

TEST_F(DenseMatrixViewTest, Clear) {
    matrix_->beginAssemblyPhase();
    matrix_->addMatrixEntry(0, 0, 5.0);
    matrix_->clear();

    EXPECT_DOUBLE_EQ((*matrix_)(0, 0), 0.0);
    EXPECT_EQ(matrix_->getPhase(), AssemblyPhase::NotStarted);
}

TEST_F(DenseMatrixViewTest, AssemblyPhase) {
    EXPECT_EQ(matrix_->getPhase(), AssemblyPhase::NotStarted);

    matrix_->beginAssemblyPhase();
    EXPECT_EQ(matrix_->getPhase(), AssemblyPhase::Building);

    matrix_->endAssemblyPhase();
    EXPECT_EQ(matrix_->getPhase(), AssemblyPhase::Flushing);

    matrix_->finalizeAssembly();
    EXPECT_EQ(matrix_->getPhase(), AssemblyPhase::Finalized);
}

// ============================================================================
// DenseVectorView Tests
// ============================================================================

class DenseVectorViewTest : public ::testing::Test {
protected:
    void SetUp() override {
        size_ = 5;
        vector_ = std::make_unique<DenseVectorView>(size_);
    }

    GlobalIndex size_;
    std::unique_ptr<DenseVectorView> vector_;
};

TEST_F(DenseVectorViewTest, Construction) {
    EXPECT_EQ(vector_->numRows(), size_);
    EXPECT_EQ(vector_->numCols(), 1);
    EXPECT_FALSE(vector_->hasMatrix());
    EXPECT_TRUE(vector_->hasVector());
}

TEST_F(DenseVectorViewTest, AddEntry) {
    vector_->beginAssemblyPhase();
    vector_->addVectorEntry(2, 3.14);
    vector_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*vector_)[2], 3.14);
    EXPECT_DOUBLE_EQ(vector_->getVectorEntry(2), 3.14);
}

TEST_F(DenseVectorViewTest, AddEntries) {
    std::vector<GlobalIndex> dofs = {0, 2, 4};
    std::vector<Real> values = {1.0, 2.0, 3.0};

    vector_->beginAssemblyPhase();
    vector_->addVectorEntries(dofs, values);
    vector_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*vector_)[0], 1.0);
    EXPECT_DOUBLE_EQ((*vector_)[2], 2.0);
    EXPECT_DOUBLE_EQ((*vector_)[4], 3.0);
}

TEST_F(DenseVectorViewTest, SetEntries) {
    vector_->beginAssemblyPhase();
    vector_->addVectorEntry(1, 5.0);

    std::vector<GlobalIndex> dofs = {1};
    std::vector<Real> values = {10.0};
    vector_->setVectorEntries(dofs, values);
    vector_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*vector_)[1], 10.0);
}

TEST_F(DenseVectorViewTest, ZeroEntries) {
    vector_->beginAssemblyPhase();
    vector_->addVectorEntry(1, 5.0);
    vector_->addVectorEntry(2, 3.0);

    std::vector<GlobalIndex> dofs = {1};
    vector_->zeroVectorEntries(dofs);
    vector_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ((*vector_)[1], 0.0);
    EXPECT_DOUBLE_EQ((*vector_)[2], 3.0);
}

TEST_F(DenseVectorViewTest, Norm) {
    vector_->beginAssemblyPhase();
    vector_->addVectorEntry(0, 3.0);
    vector_->addVectorEntry(1, 4.0);
    vector_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ(vector_->norm(), 5.0);
}

// ============================================================================
// DenseSystemView Tests
// ============================================================================

class DenseSystemViewTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_dofs_ = 4;
        system_ = std::make_unique<DenseSystemView>(n_dofs_);
    }

    GlobalIndex n_dofs_;
    std::unique_ptr<DenseSystemView> system_;
};

TEST_F(DenseSystemViewTest, Construction) {
    EXPECT_EQ(system_->numRows(), n_dofs_);
    EXPECT_EQ(system_->numCols(), n_dofs_);
    EXPECT_TRUE(system_->hasMatrix());
    EXPECT_TRUE(system_->hasVector());
}

TEST_F(DenseSystemViewTest, CombinedAssembly) {
    std::vector<GlobalIndex> dofs = {0, 1};
    std::vector<Real> matrix = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> vector = {5.0, 6.0};

    system_->beginAssemblyPhase();
    system_->addMatrixEntries(dofs, matrix);
    system_->addVectorEntries(dofs, vector);
    system_->endAssemblyPhase();

    EXPECT_DOUBLE_EQ(system_->matrixEntry(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(system_->matrixEntry(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(system_->matrixEntry(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(system_->matrixEntry(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(system_->vectorEntry(0), 5.0);
    EXPECT_DOUBLE_EQ(system_->vectorEntry(1), 6.0);
}

TEST_F(DenseSystemViewTest, ZeroRowsAffectsMatrixOnly) {
    system_->beginAssemblyPhase();
    system_->addMatrixEntry(1, 0, 5.0);
    system_->addMatrixEntry(1, 1, 5.0);
    system_->addVectorEntry(1, 10.0);

    std::vector<GlobalIndex> rows = {1};
    system_->zeroRows(rows, true);
    system_->endAssemblyPhase();

    // Matrix row zeroed
    EXPECT_DOUBLE_EQ(system_->matrixEntry(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(system_->matrixEntry(1, 1), 1.0);  // diagonal

    // Vector unchanged
    EXPECT_DOUBLE_EQ(system_->vectorEntry(1), 10.0);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST(GlobalSystemViewFactoryTest, CreateDenseMatrixView) {
    auto view = createDenseMatrixView(5);
    EXPECT_NE(view, nullptr);
    EXPECT_EQ(view->numRows(), 5);
    EXPECT_EQ(view->numCols(), 5);
    EXPECT_TRUE(view->hasMatrix());
}

TEST(GlobalSystemViewFactoryTest, CreateDenseVectorView) {
    auto view = createDenseVectorView(10);
    EXPECT_NE(view, nullptr);
    EXPECT_EQ(view->numRows(), 10);
    EXPECT_TRUE(view->hasVector());
}

TEST(GlobalSystemViewFactoryTest, CreateDenseSystemView) {
    auto view = createDenseSystemView(8);
    EXPECT_NE(view, nullptr);
    EXPECT_EQ(view->numRows(), 8);
    EXPECT_TRUE(view->hasMatrix());
    EXPECT_TRUE(view->hasVector());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST(GlobalSystemViewEdgeCases, OutOfRangeIgnored) {
    DenseMatrixView matrix(3);

    matrix.beginAssemblyPhase();
    // These should not crash - out of range entries are silently ignored
    matrix.addMatrixEntry(-1, 0, 1.0);
    matrix.addMatrixEntry(0, 10, 1.0);
    matrix.addMatrixEntry(5, 5, 1.0);
    matrix.endAssemblyPhase();

    // Matrix should still be valid and unchanged
    EXPECT_DOUBLE_EQ(matrix(0, 0), 0.0);
}

TEST(GlobalSystemViewEdgeCases, ZeroSizeConstruction) {
    // Zero-size should not crash
    DenseMatrixView matrix(0);
    EXPECT_EQ(matrix.numRows(), 0);
    EXPECT_EQ(matrix.data().size(), 0);
}

// ============================================================================
// DofRemapTable / RemappedSystemView Tests
// ============================================================================

TEST(DofRemapTableTest, SetAndMapBasic) {
    DofRemapTable table;
    table.set(5, 10);

    EXPECT_EQ(table.map(5).value_or(-1), 10);
    EXPECT_FALSE(table.map(3).has_value());
}

TEST(DofRemapTableTest, MultipleMappings) {
    DofRemapTable table;
    table.set(0, 10);
    table.set(1, 11);
    table.set(2, 12);

    EXPECT_EQ(table.map(0).value_or(-1), 10);
    EXPECT_EQ(table.map(1).value_or(-1), 11);
    EXPECT_EQ(table.map(2).value_or(-1), 12);
    EXPECT_FALSE(table.map(3).has_value());
}

TEST(DofRemapTableTest, SelfMapping) {
    DofRemapTable table;
    table.set(5, 5);
    EXPECT_EQ(table.map(5).value_or(-1), 5);
}

TEST(DofRemapTableTest, NegativeTargetStored) {
    DofRemapTable table;
    table.set(5, -1);
    EXPECT_EQ(table.map(5).value_or(0), -1);
}

TEST(DofRemapTableTest, OverwriteExistingMapping) {
    DofRemapTable table;
    table.set(5, 10);
    table.set(5, 20);
    EXPECT_EQ(table.map(5).value_or(-1), 20);
}

TEST(RemappedSystemViewTest, ConstructionDelegatesProperties) {
    DenseSystemView base(12);
    DofRemapTable table;
    RemappedSystemView view(base, table);

    EXPECT_TRUE(view.hasMatrix());
    EXPECT_TRUE(view.hasVector());
    EXPECT_EQ(view.numRows(), base.numRows());
    EXPECT_EQ(view.numCols(), base.numCols());
    EXPECT_EQ(view.backendName(), base.backendName());
}

TEST(RemappedSystemViewTest, AddMatrixEntryUnmappedPassthroughOnly) {
    DenseSystemView base(12);
    DofRemapTable table;
    RemappedSystemView view(base, table);

    view.beginAssemblyPhase();
    view.addMatrixEntry(0, 0, 3.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(base.getMatrixEntry(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(5, 0), 0.0);
}

TEST(RemappedSystemViewTest, AddMatrixEntryMappedRowDuplicates) {
    DenseSystemView base(12);
    DofRemapTable table;
    table.set(1, 5);
    RemappedSystemView view(base, table);

    view.beginAssemblyPhase();
    view.addMatrixEntry(1, 2, 3.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(base.getMatrixEntry(1, 2), 3.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(5, 2), 3.0);
}

TEST(RemappedSystemViewTest, AddMatrixEntryMappedRowAndColumnUsesMappedColumn) {
    DenseSystemView base(12);
    DofRemapTable table;
    table.set(1, 5);
    table.set(2, 6);
    RemappedSystemView view(base, table);

    view.beginAssemblyPhase();
    view.addMatrixEntry(1, 2, 4.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(base.getMatrixEntry(1, 2), 4.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(5, 6), 4.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(5, 2), 0.0);
}

TEST(RemappedSystemViewTest, AddMatrixEntrySelfMappedRowDoesNotDuplicate) {
    DenseSystemView base(12);
    DofRemapTable table;
    table.set(5, 5);
    RemappedSystemView view(base, table);

    view.beginAssemblyPhase();
    view.addMatrixEntry(5, 3, 1.0, AddMode::Add);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(base.getMatrixEntry(5, 3), 1.0);
}

TEST(RemappedSystemViewTest, AddMatrixEntryNegativeMappedRowSkipped) {
    DenseSystemView base(12);
    DofRemapTable table;
    table.set(1, -1);
    RemappedSystemView view(base, table);

    view.beginAssemblyPhase();
    view.addMatrixEntry(1, 2, 3.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(base.getMatrixEntry(1, 2), 3.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(0, 0), 0.0);
}

TEST(RemappedSystemViewTest, AddMatrixEntriesBatchMixedMappings) {
    DenseSystemView base(20);
    DofRemapTable table;
    table.set(1, 10);   // mapped row
    table.set(3, 11);   // mapped column
    RemappedSystemView view(base, table);

    const std::vector<GlobalIndex> row_dofs = {0, 1};
    const std::vector<GlobalIndex> col_dofs = {2, 3};
    const std::vector<Real> local = {
        1.0, 2.0,   // row 0
        3.0, 4.0    // row 1
    };

    view.beginAssemblyPhase();
    view.addMatrixEntries(row_dofs, col_dofs, local);
    view.endAssemblyPhase();

    // Original contribution
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(0, 2), 1.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(0, 3), 2.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(1, 2), 3.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(1, 3), 4.0);

    // Duplicated mapped row (1 -> 10), with mapped column (3 -> 11)
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(10, 2), 3.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(10, 11), 4.0);
}

TEST(RemappedSystemViewTest, AddVectorEntryMappedDuplicates) {
    DenseSystemView base(20);
    DofRemapTable table;
    table.set(2, 7);
    RemappedSystemView view(base, table);

    view.beginAssemblyPhase();
    view.addVectorEntry(2, 5.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(base.getVectorEntry(2), 5.0);
    EXPECT_DOUBLE_EQ(base.getVectorEntry(7), 5.0);
}

TEST(RemappedSystemViewTest, AddVectorEntriesBatchDuplicatesMappedOnly) {
    DenseSystemView base(20);
    DofRemapTable table;
    table.set(2, 7);
    table.set(3, 8);
    RemappedSystemView view(base, table);

    const std::vector<GlobalIndex> dofs = {1, 2, 3};
    const std::vector<Real> values = {1.0, 2.0, 3.0};

    view.beginAssemblyPhase();
    view.addVectorEntries(dofs, values);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(base.getVectorEntry(1), 1.0);
    EXPECT_DOUBLE_EQ(base.getVectorEntry(2), 2.0);
    EXPECT_DOUBLE_EQ(base.getVectorEntry(3), 3.0);
    EXPECT_DOUBLE_EQ(base.getVectorEntry(7), 2.0);
    EXPECT_DOUBLE_EQ(base.getVectorEntry(8), 3.0);
}

TEST(RemappedSystemViewTest, SetDiagonalPassthroughNoDuplication) {
    DenseSystemView base(20);
    DofRemapTable table;
    table.set(2, 7);
    RemappedSystemView view(base, table);

    view.beginAssemblyPhase();
    view.setDiagonal(2, 5.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(base.getMatrixEntry(2, 2), 5.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(7, 7), 0.0);
}

TEST(RemappedSystemViewTest, ZeroRowsPassthroughDoesNotAffectMappedRow) {
    DenseSystemView base(20);
    DofRemapTable table;
    table.set(1, 10);
    RemappedSystemView view(base, table);

    base.beginAssemblyPhase();
    base.addMatrixEntry(1, 0, 2.0);
    base.addMatrixEntry(10, 0, 3.0);
    base.endAssemblyPhase();

    view.beginAssemblyPhase();
    const std::vector<GlobalIndex> rows = {1};
    view.zeroRows(rows, true);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(base.getMatrixEntry(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(base.getMatrixEntry(10, 0), 3.0);
}

TEST(RemappedSystemViewTest, LifecycleDelegatesAndPhaseMatchesBase) {
    DenseSystemView base(5);
    DofRemapTable table;
    RemappedSystemView view(base, table);

    EXPECT_EQ(view.getPhase(), AssemblyPhase::NotStarted);
    view.beginAssemblyPhase();
    EXPECT_EQ(view.getPhase(), AssemblyPhase::Building);
    view.endAssemblyPhase();
    EXPECT_EQ(view.getPhase(), AssemblyPhase::Flushing);
    view.finalizeAssembly();
    EXPECT_EQ(view.getPhase(), AssemblyPhase::Finalized);
}

TEST(RemappedSystemViewTest, NullBasePropertiesAreSafeAfterMove) {
    DenseSystemView base(5);
    DofRemapTable table;
    RemappedSystemView view(base, table);

    RemappedSystemView moved(std::move(view));

    EXPECT_FALSE(view.hasMatrix());
    EXPECT_FALSE(view.hasVector());
    EXPECT_EQ(view.numRows(), 0);
    EXPECT_EQ(view.numCols(), 0);
    EXPECT_EQ(view.backendName(), "RemappedSystem");
    EXPECT_EQ(view.getPhase(), AssemblyPhase::NotStarted);

    EXPECT_TRUE(moved.hasMatrix());
    EXPECT_TRUE(moved.hasVector());
}

TEST(RemappedSystemViewTest, ZeroDelegates) {
    DenseSystemView base(10);
    DofRemapTable table;
    RemappedSystemView view(base, table);

    base.beginAssemblyPhase();
    base.addMatrixEntry(2, 3, 4.0);
    base.addVectorEntry(2, 5.0);
    base.endAssemblyPhase();

    view.zero();

    EXPECT_DOUBLE_EQ(base.getMatrixEntry(2, 3), 0.0);
    EXPECT_DOUBLE_EQ(base.getVectorEntry(2), 0.0);
}

TEST(RemappedSystemViewTest, GetEntryDelegatesToBase) {
    DenseSystemView base(10);
    DofRemapTable table;
    table.set(1, 5);
    RemappedSystemView view(base, table);

    view.beginAssemblyPhase();
    view.addMatrixEntry(1, 2, 3.0);
    view.addVectorEntry(1, 4.0);
    view.endAssemblyPhase();

    EXPECT_DOUBLE_EQ(view.getMatrixEntry(1, 2), 3.0);
    EXPECT_DOUBLE_EQ(view.getMatrixEntry(5, 2), 3.0);
    EXPECT_DOUBLE_EQ(view.getVectorEntry(1), 4.0);
    EXPECT_DOUBLE_EQ(view.getVectorEntry(5), 4.0);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
