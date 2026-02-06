/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>
#include "Sparsity/SparsityPattern.h"
#include <vector>
#include <set>
#include <algorithm>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST(SparsityPatternTest, DefaultConstruction) {
    SparsityPattern pattern;
    EXPECT_EQ(pattern.numRows(), 0);
    EXPECT_EQ(pattern.numCols(), 0);
    EXPECT_EQ(pattern.getNnz(), 0);
    EXPECT_FALSE(pattern.isFinalized());
    EXPECT_EQ(pattern.state(), SparsityState::Building);
}

TEST(SparsityPatternTest, SquareConstruction) {
    SparsityPattern pattern(10);
    EXPECT_EQ(pattern.numRows(), 10);
    EXPECT_EQ(pattern.numCols(), 10);
    EXPECT_TRUE(pattern.isSquare());
    EXPECT_FALSE(pattern.isFinalized());
}

TEST(SparsityPatternTest, RectangularConstruction) {
    SparsityPattern pattern(5, 8);
    EXPECT_EQ(pattern.numRows(), 5);
    EXPECT_EQ(pattern.numCols(), 8);
    EXPECT_FALSE(pattern.isSquare());
}

TEST(SparsityPatternTest, Resize) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    EXPECT_EQ(pattern.getNnz(), 1);

    pattern.resize(10, 10);
    EXPECT_EQ(pattern.numRows(), 10);
    EXPECT_EQ(pattern.numCols(), 10);
    EXPECT_EQ(pattern.getNnz(), 0);  // Cleared
}

// ============================================================================
// Entry Addition Tests
// ============================================================================

TEST(SparsityPatternTest, AddSingleEntry) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(2, 3);

    EXPECT_EQ(pattern.getNnz(), 1);
    EXPECT_TRUE(pattern.hasEntry(2, 3));
    EXPECT_FALSE(pattern.hasEntry(3, 2));
}

TEST(SparsityPatternTest, AddDuplicateEntry) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(2, 3);
    pattern.addEntry(2, 3);  // Duplicate
    pattern.addEntry(2, 3);  // Another duplicate

    EXPECT_EQ(pattern.getNnz(), 1);  // Should still be 1
}

TEST(SparsityPatternTest, AddMultipleEntries) {
    SparsityPattern pattern(5, 5);
    std::vector<GlobalIndex> cols = {0, 2, 4};
    pattern.addEntries(1, cols);

    EXPECT_EQ(pattern.getNnz(), 3);
    EXPECT_TRUE(pattern.hasEntry(1, 0));
    EXPECT_TRUE(pattern.hasEntry(1, 2));
    EXPECT_TRUE(pattern.hasEntry(1, 4));
    EXPECT_FALSE(pattern.hasEntry(1, 1));
}

TEST(SparsityPatternTest, AddBlock) {
    SparsityPattern pattern(5, 5);
    pattern.addBlock(1, 3, 1, 3);  // 2x2 block

    EXPECT_EQ(pattern.getNnz(), 4);
    EXPECT_TRUE(pattern.hasEntry(1, 1));
    EXPECT_TRUE(pattern.hasEntry(1, 2));
    EXPECT_TRUE(pattern.hasEntry(2, 1));
    EXPECT_TRUE(pattern.hasEntry(2, 2));
}

TEST(SparsityPatternTest, AddElementCouplings) {
    SparsityPattern pattern(10, 10);
    std::vector<GlobalIndex> dofs = {1, 3, 5};
    pattern.addElementCouplings(dofs);

    // Should create 3x3 = 9 couplings
    EXPECT_EQ(pattern.getNnz(), 9);

    // Check all pairs
    for (GlobalIndex row : dofs) {
        for (GlobalIndex col : dofs) {
            EXPECT_TRUE(pattern.hasEntry(row, col));
        }
    }
}

TEST(SparsityPatternTest, AddRectangularElementCouplings) {
    SparsityPattern pattern(10, 8);
    std::vector<GlobalIndex> row_dofs = {1, 3};
    std::vector<GlobalIndex> col_dofs = {0, 2, 4};
    pattern.addElementCouplings(row_dofs, col_dofs);

    // Should create 2x3 = 6 couplings
    EXPECT_EQ(pattern.getNnz(), 6);

    for (GlobalIndex row : row_dofs) {
        for (GlobalIndex col : col_dofs) {
            EXPECT_TRUE(pattern.hasEntry(row, col));
        }
    }
}

// ============================================================================
// Finalization Tests
// ============================================================================

TEST(SparsityPatternTest, Finalize) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 2);
    pattern.addEntry(1, 1);
    pattern.addEntry(2, 0);
    pattern.addEntry(2, 2);

    EXPECT_FALSE(pattern.isFinalized());
    pattern.finalize();
    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_EQ(pattern.state(), SparsityState::Finalized);
    EXPECT_EQ(pattern.getNnz(), 5);
}

TEST(SparsityPatternTest, CSRFormat) {
    SparsityPattern pattern(3, 3);
    // Row 0: columns 0, 2
    // Row 1: column 1
    // Row 2: columns 0, 1, 2
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 2);
    pattern.addEntry(1, 1);
    pattern.addEntry(2, 0);
    pattern.addEntry(2, 1);
    pattern.addEntry(2, 2);

    pattern.finalize();

    auto row_ptr = pattern.getRowPtr();
    auto col_idx = pattern.getColIndices();

    // Check row pointers
    ASSERT_EQ(row_ptr.size(), 4);
    EXPECT_EQ(row_ptr[0], 0);
    EXPECT_EQ(row_ptr[1], 2);  // Row 0 has 2 entries
    EXPECT_EQ(row_ptr[2], 3);  // Row 1 has 1 entry
    EXPECT_EQ(row_ptr[3], 6);  // Row 2 has 3 entries

    // Check column indices are sorted
    ASSERT_EQ(col_idx.size(), 6);
    EXPECT_EQ(col_idx[0], 0);  // Row 0
    EXPECT_EQ(col_idx[1], 2);
    EXPECT_EQ(col_idx[2], 1);  // Row 1
    EXPECT_EQ(col_idx[3], 0);  // Row 2
    EXPECT_EQ(col_idx[4], 1);
    EXPECT_EQ(col_idx[5], 2);
}

TEST(SparsityPatternTest, RowView) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 2);
    pattern.addEntry(0, 0);  // Added out of order
    pattern.addEntry(0, 1);
    pattern.finalize();

    auto row0 = pattern.getRowIndices(0);
    EXPECT_EQ(row0.size(), 3);

    // Should be sorted
    std::vector<GlobalIndex> expected = {0, 1, 2};
    std::vector<GlobalIndex> actual(row0.begin(), row0.end());
    EXPECT_EQ(actual, expected);
}

// ============================================================================
// Diagonal Tests
// ============================================================================

TEST(SparsityPatternTest, EnsureDiagonal) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 1);
    pattern.addEntry(2, 3);
    pattern.ensureDiagonal();

    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_TRUE(pattern.hasDiagonal(i)) << "Missing diagonal at row " << i;
    }
}

TEST(SparsityPatternTest, HasAllDiagonals) {
    SparsityPattern pattern(3, 3);
    EXPECT_FALSE(pattern.hasAllDiagonals());

    pattern.addEntry(0, 0);
    EXPECT_FALSE(pattern.hasAllDiagonals());

    pattern.addEntry(1, 1);
    pattern.addEntry(2, 2);
    EXPECT_TRUE(pattern.hasAllDiagonals());
}

TEST(SparsityPatternTest, EnsureNonEmptyRows) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(1, 2);
    pattern.addEntry(3, 4);
    pattern.ensureNonEmptyRows();

    // All rows should now have at least one entry
    pattern.finalize();
    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_GT(pattern.getRowNnz(i), 0) << "Empty row at " << i;
    }
}

// ============================================================================
// Rectangular Pattern Tests
// ============================================================================

TEST(SparsityPatternTest, RectangularPattern) {
    SparsityPattern pattern(3, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 4);
    pattern.addEntry(1, 2);
    pattern.addEntry(2, 1);
    pattern.addEntry(2, 3);

    EXPECT_EQ(pattern.getNnz(), 5);
    EXPECT_FALSE(pattern.isSquare());

    pattern.finalize();
    EXPECT_EQ(pattern.getRowNnz(0), 2);
    EXPECT_EQ(pattern.getRowNnz(1), 1);
    EXPECT_EQ(pattern.getRowNnz(2), 2);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(SparsityPatternTest, ComputeStats) {
    SparsityPattern pattern(4, 4);
    // Row 0: 3 entries
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 1);
    pattern.addEntry(0, 2);
    // Row 1: 2 entries
    pattern.addEntry(1, 1);
    pattern.addEntry(1, 2);
    // Row 2: 1 entry
    pattern.addEntry(2, 2);
    // Row 3: 4 entries
    pattern.addEntry(3, 0);
    pattern.addEntry(3, 1);
    pattern.addEntry(3, 2);
    pattern.addEntry(3, 3);

    auto stats = pattern.computeStats();
    EXPECT_EQ(stats.n_rows, 4);
    EXPECT_EQ(stats.n_cols, 4);
    EXPECT_EQ(stats.nnz, 10);
    EXPECT_EQ(stats.min_row_nnz, 1);
    EXPECT_EQ(stats.max_row_nnz, 4);
    EXPECT_EQ(stats.empty_rows, 0);
    EXPECT_NEAR(stats.avg_row_nnz, 2.5, 1e-10);
}

TEST(SparsityPatternTest, ComputeBandwidth) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 3);  // bandwidth 3
    pattern.addEntry(1, 1);
    pattern.addEntry(4, 1);  // bandwidth 3
    pattern.finalize();

    EXPECT_EQ(pattern.computeBandwidth(), 3);
}

TEST(SparsityPatternTest, IsSymmetric) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 1);
    pattern.addEntry(1, 0);  // Symmetric pair
    pattern.addEntry(1, 1);

    EXPECT_TRUE(pattern.isSymmetric());

    pattern.addEntry(0, 2);  // No (2,0) pair
    EXPECT_FALSE(pattern.isSymmetric());
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST(SparsityPatternTest, Transpose) {
    SparsityPattern pattern(3, 4);
    pattern.addEntry(0, 1);
    pattern.addEntry(0, 3);
    pattern.addEntry(1, 2);
    pattern.addEntry(2, 0);
    pattern.finalize();

    auto transposed = pattern.transpose();
    EXPECT_EQ(transposed.numRows(), 4);
    EXPECT_EQ(transposed.numCols(), 3);
    EXPECT_EQ(transposed.getNnz(), 4);

    EXPECT_TRUE(transposed.hasEntry(1, 0));
    EXPECT_TRUE(transposed.hasEntry(3, 0));
    EXPECT_TRUE(transposed.hasEntry(2, 1));
    EXPECT_TRUE(transposed.hasEntry(0, 2));
}

TEST(SparsityPatternTest, Permute) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 1);
    pattern.addEntry(1, 2);
    pattern.finalize();

    // Reverse permutation: 0->2, 1->1, 2->0
    std::vector<GlobalIndex> perm = {2, 1, 0};
    auto permuted = pattern.permute(perm, perm);

    EXPECT_TRUE(permuted.hasEntry(2, 2));  // (0,0) -> (2,2)
    EXPECT_TRUE(permuted.hasEntry(2, 1));  // (0,1) -> (2,1)
    EXPECT_TRUE(permuted.hasEntry(1, 0));  // (1,2) -> (1,0)
}

TEST(SparsityPatternTest, PermuteRejectsDuplicateRowPermutation) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 0);
    pattern.finalize();

    std::vector<GlobalIndex> bad_row = {0, 0, 2};  // duplicate 0
    std::vector<GlobalIndex> col = {0, 1, 2};

    EXPECT_THROW((void)pattern.permute(bad_row, col), svmp::FE::FEException);
}

TEST(SparsityPatternTest, PermuteRejectsDuplicateColumnPermutation) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 0);
    pattern.finalize();

    std::vector<GlobalIndex> row = {0, 1, 2};
    std::vector<GlobalIndex> bad_col = {0, 0, 2};  // duplicate 0

    EXPECT_THROW((void)pattern.permute(row, bad_col), svmp::FE::FEException);
}

TEST(SparsityPatternTest, Extract) {
    SparsityPattern pattern(5, 5);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            pattern.addEntry(i, j);
        }
    }
    pattern.finalize();

    std::vector<GlobalIndex> row_set = {1, 3};
    std::vector<GlobalIndex> col_set = {0, 2, 4};
    auto extracted = pattern.extract(row_set, col_set);

    EXPECT_EQ(extracted.numRows(), 2);
    EXPECT_EQ(extracted.numCols(), 3);
    EXPECT_EQ(extracted.getNnz(), 6);  // 2x3 full block
}

// ============================================================================
// Pattern Operations Tests
// ============================================================================

TEST(SparsityPatternTest, PatternUnion) {
    SparsityPattern a(3, 3);
    a.addEntry(0, 0);
    a.addEntry(0, 1);
    a.finalize();

    SparsityPattern b(3, 3);
    b.addEntry(0, 1);
    b.addEntry(1, 1);
    b.finalize();

    auto c = patternUnion(a, b);
    EXPECT_EQ(c.getNnz(), 3);  // Union has 3 unique entries
    EXPECT_TRUE(c.hasEntry(0, 0));
    EXPECT_TRUE(c.hasEntry(0, 1));
    EXPECT_TRUE(c.hasEntry(1, 1));
}

TEST(SparsityPatternTest, PatternIntersection) {
    SparsityPattern a(3, 3);
    a.addEntry(0, 0);
    a.addEntry(0, 1);
    a.addEntry(1, 1);
    a.finalize();

    SparsityPattern b(3, 3);
    b.addEntry(0, 1);
    b.addEntry(1, 1);
    b.addEntry(2, 2);
    b.finalize();

    auto c = patternIntersection(a, b);
    EXPECT_EQ(c.getNnz(), 2);  // Only (0,1) and (1,1) in both
    EXPECT_FALSE(c.hasEntry(0, 0));
    EXPECT_TRUE(c.hasEntry(0, 1));
    EXPECT_TRUE(c.hasEntry(1, 1));
    EXPECT_FALSE(c.hasEntry(2, 2));
}

TEST(SparsityPatternTest, Symmetrize) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 1);
    pattern.addEntry(1, 2);
    pattern.finalize();

    auto symmetric = symmetrize(pattern);
    EXPECT_EQ(symmetric.getNnz(), 4);
    EXPECT_TRUE(symmetric.hasEntry(0, 1));
    EXPECT_TRUE(symmetric.hasEntry(1, 0));  // Added
    EXPECT_TRUE(symmetric.hasEntry(1, 2));
    EXPECT_TRUE(symmetric.hasEntry(2, 1));  // Added
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(SparsityPatternTest, Validate) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 2);
    pattern.addEntry(2, 1);
    pattern.finalize();

    EXPECT_TRUE(pattern.validate());
    EXPECT_TRUE(pattern.validationError().empty());
}

TEST(SparsityPatternTest, Clear) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 1);
    pattern.finalize();

    pattern.clear();
    EXPECT_FALSE(pattern.isFinalized());
    EXPECT_EQ(pattern.getNnz(), 0);
    EXPECT_EQ(pattern.numRows(), 5);  // Dimensions preserved
}

// ============================================================================
// Move/Copy Semantics Tests
// ============================================================================

TEST(SparsityPatternTest, MoveConstruction) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 2);
    pattern.finalize();

    SparsityPattern moved(std::move(pattern));
    EXPECT_EQ(moved.numRows(), 5);
    EXPECT_EQ(moved.getNnz(), 2);
    EXPECT_TRUE(moved.isFinalized());

    // Original should be in valid but unspecified state
    EXPECT_EQ(pattern.numRows(), 0);
}

TEST(SparsityPatternTest, CopyConstruction) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 2);
    pattern.finalize();

    SparsityPattern copy(pattern);
    EXPECT_EQ(copy.numRows(), 5);
    EXPECT_TRUE(copy.isFinalized());
    EXPECT_TRUE(copy.hasEntry(0, 0));
    EXPECT_TRUE(copy.hasEntry(1, 2));
}

// ============================================================================
// Empty Pattern Tests
// ============================================================================

TEST(SparsityPatternTest, EmptyPattern) {
    SparsityPattern pattern(5, 5);
    pattern.finalize();

    EXPECT_EQ(pattern.getNnz(), 0);
    EXPECT_TRUE(pattern.validate());

    auto row_ptr = pattern.getRowPtr();
    EXPECT_EQ(row_ptr.size(), 6);
    for (size_t i = 0; i < row_ptr.size(); ++i) {
        EXPECT_EQ(row_ptr[i], 0);
    }
}

TEST(SparsityPatternTest, EmptyRow) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 0);
    pattern.addEntry(2, 2);
    // Row 1 is empty
    pattern.finalize();

    EXPECT_EQ(pattern.getRowNnz(0), 1);
    EXPECT_EQ(pattern.getRowNnz(1), 0);
    EXPECT_EQ(pattern.getRowNnz(2), 1);
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST(SparsityPatternTest, DeterministicOrdering) {
    // Add entries in random order, verify sorted output
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 4);
    pattern.addEntry(0, 1);
    pattern.addEntry(0, 3);
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 2);
    pattern.finalize();

    auto row0 = pattern.getRowIndices(0);
    std::vector<GlobalIndex> cols(row0.begin(), row0.end());
    std::vector<GlobalIndex> expected = {0, 1, 2, 3, 4};
    EXPECT_EQ(cols, expected);
}

TEST(SparsityPatternTest, ReproduciblePattern) {
    // Create same pattern twice, verify identical results
    auto create_pattern = []() {
        SparsityPattern pattern(10, 10);
        std::vector<GlobalIndex> elem1 = {0, 1, 2};
        std::vector<GlobalIndex> elem2 = {1, 2, 3};
        std::vector<GlobalIndex> elem3 = {2, 3, 4};

        pattern.addElementCouplings(elem1);
        pattern.addElementCouplings(elem2);
        pattern.addElementCouplings(elem3);
        pattern.finalize();
        return pattern;
    };

    auto p1 = create_pattern();
    auto p2 = create_pattern();

    EXPECT_EQ(p1.getNnz(), p2.getNnz());

    auto rp1 = p1.getRowPtr();
    auto rp2 = p2.getRowPtr();
    EXPECT_TRUE(std::equal(rp1.begin(), rp1.end(), rp2.begin()));

    auto ci1 = p1.getColIndices();
    auto ci2 = p2.getColIndices();
    EXPECT_TRUE(std::equal(ci1.begin(), ci1.end(), ci2.begin()));
}
