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
#include "Sparsity/DistributedSparsityPattern.h"
#include <vector>
#include <algorithm>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, DefaultConstruction) {
    DistributedSparsityPattern pattern;
    EXPECT_EQ(pattern.globalRows(), 0);
    EXPECT_EQ(pattern.globalCols(), 0);
    EXPECT_EQ(pattern.numOwnedRows(), 0);
    EXPECT_FALSE(pattern.isFinalized());
}

TEST(DistributedSparsityPatternTest, ConstructWithRanges) {
    IndexRange owned_rows{10, 20};  // Rows 10-19
    IndexRange owned_cols{10, 20};  // Cols 10-19
    DistributedSparsityPattern pattern(owned_rows, owned_cols, 100, 100);

    EXPECT_EQ(pattern.globalRows(), 100);
    EXPECT_EQ(pattern.globalCols(), 100);
    EXPECT_EQ(pattern.numOwnedRows(), 10);
    EXPECT_EQ(pattern.numOwnedCols(), 10);
    EXPECT_TRUE(pattern.isSquare());
}

TEST(DistributedSparsityPatternTest, ConstructWithOffsets) {
    DistributedSparsityPattern pattern(5, 10, 5, 10, 50, 50);

    EXPECT_EQ(pattern.globalRows(), 50);
    EXPECT_EQ(pattern.globalCols(), 50);
    EXPECT_EQ(pattern.numOwnedRows(), 10);
    EXPECT_EQ(pattern.numOwnedCols(), 10);
    EXPECT_EQ(pattern.ownedRows().first, 5);
    EXPECT_EQ(pattern.ownedRows().last, 15);
}

// ============================================================================
// Ownership Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, OwnershipQueries) {
    IndexRange owned_rows{10, 20};
    IndexRange owned_cols{5, 15};
    DistributedSparsityPattern pattern(owned_rows, owned_cols, 100, 100);

    EXPECT_TRUE(pattern.ownsRow(10));
    EXPECT_TRUE(pattern.ownsRow(19));
    EXPECT_FALSE(pattern.ownsRow(9));
    EXPECT_FALSE(pattern.ownsRow(20));

    EXPECT_TRUE(pattern.ownsCol(5));
    EXPECT_TRUE(pattern.ownsCol(14));
    EXPECT_FALSE(pattern.ownsCol(4));
    EXPECT_FALSE(pattern.ownsCol(15));

    EXPECT_FALSE(pattern.isGhostCol(10));
    EXPECT_TRUE(pattern.isGhostCol(0));
    EXPECT_TRUE(pattern.isGhostCol(20));
}

// ============================================================================
// Entry Addition Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, AddDiagonalEntry) {
    // Rank owns rows and cols 10-19
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Add entry in diagonal block
    pattern.addEntry(12, 15);  // Both in owned range

    EXPECT_TRUE(pattern.hasEntry(12, 15));
    EXPECT_EQ(pattern.getDiagNnz(), 1);
    EXPECT_EQ(pattern.getOffdiagNnz(), 0);
}

TEST(DistributedSparsityPatternTest, AddOffdiagonalEntry) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Add entry in off-diagonal block
    pattern.addEntry(12, 5);   // Row owned, col not owned
    pattern.addEntry(12, 25);  // Row owned, col not owned

    EXPECT_EQ(pattern.getDiagNnz(), 0);
    EXPECT_EQ(pattern.getOffdiagNnz(), 2);
}

TEST(DistributedSparsityPatternTest, AddMixedEntries) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Owned cols: 10-19
    pattern.addEntry(12, 10);  // Diagonal (col 10 is owned)
    pattern.addEntry(12, 15);  // Diagonal
    pattern.addEntry(12, 5);   // Off-diagonal (col 5 not owned)
    pattern.addEntry(12, 25);  // Off-diagonal

    EXPECT_EQ(pattern.getDiagNnz(), 2);
    EXPECT_EQ(pattern.getOffdiagNnz(), 2);
    EXPECT_EQ(pattern.getLocalNnz(), 4);
}

TEST(DistributedSparsityPatternTest, AddElementCouplings) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Element with DOFs crossing ownership boundary
    std::vector<GlobalIndex> dofs = {12, 13, 8, 22};  // 12,13 owned; 8,22 ghost

    pattern.addElementCouplings(dofs);

    // Only rows 12, 13 should be added (owned rows)
    // Row 12: couples with 12, 13 (diag) and 8, 22 (offdiag)
    // Row 13: couples with 12, 13 (diag) and 8, 22 (offdiag)
    EXPECT_EQ(pattern.getRowDiagNnz(2), 2);    // Local row 2 = global row 12
    EXPECT_EQ(pattern.getRowOffdiagNnz(2), 2);
    EXPECT_EQ(pattern.getRowDiagNnz(3), 2);    // Local row 3 = global row 13
    EXPECT_EQ(pattern.getRowOffdiagNnz(3), 2);
}

TEST(DistributedSparsityPatternTest, AddRectangularElementCouplings) {
    // Rows 0-9 owned, cols 10-19 owned
    DistributedSparsityPattern pattern(0, 10, 10, 10, 50, 50);

    std::vector<GlobalIndex> row_dofs = {2, 3, 15};  // 2,3 owned; 15 not owned
    std::vector<GlobalIndex> col_dofs = {10, 12, 5};  // 10,12 owned; 5 not owned

    pattern.addElementCouplings(row_dofs, col_dofs);

    // Only rows 2, 3 should be added
    EXPECT_EQ(pattern.getRowNnz(2), 3);  // 3 columns total
    EXPECT_EQ(pattern.getRowNnz(3), 3);
    EXPECT_EQ(pattern.getRowNnz(0), 0);  // Row 0 not touched
}

// ============================================================================
// Finalization Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, Finalize) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    pattern.addEntry(12, 10);  // Diag
    pattern.addEntry(12, 5);   // Offdiag
    pattern.addEntry(13, 15);  // Diag
    pattern.addEntry(13, 30);  // Offdiag

    EXPECT_FALSE(pattern.isFinalized());
    pattern.finalize();
    EXPECT_TRUE(pattern.isFinalized());
}

TEST(DistributedSparsityPatternTest, DiagOffdiagPatterns) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Row 12 (local 2): diag cols 10,12; offdiag col 5
    pattern.addEntry(12, 10);
    pattern.addEntry(12, 12);
    pattern.addEntry(12, 5);
    // Row 15 (local 5): diag col 15; offdiag cols 3, 40
    pattern.addEntry(15, 15);
    pattern.addEntry(15, 3);
    pattern.addEntry(15, 40);

    pattern.finalize();

    // Check diagonal pattern (uses local column indices: 0-9 for cols 10-19)
    const auto& diag = pattern.diagPattern();
    EXPECT_EQ(diag.numRows(), 10);
    EXPECT_EQ(diag.numCols(), 10);
    EXPECT_EQ(diag.getNnz(), 3);  // (12,10), (12,12), (15,15)

    // Row 2 (global 12) has local cols 0 and 2 (global 10 and 12)
    auto row2_diag = pattern.getRowDiagCols(2);
    EXPECT_EQ(row2_diag.size(), 2);

    // Check off-diagonal pattern
    const auto& offdiag = pattern.offdiagPattern();
    EXPECT_EQ(offdiag.numRows(), 10);
    EXPECT_EQ(offdiag.getNnz(), 3);  // (12,5), (15,3), (15,40)
}

TEST(DistributedSparsityPatternTest, GhostColumnMap) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    pattern.addEntry(12, 5);
    pattern.addEntry(12, 25);
    pattern.addEntry(13, 5);   // Duplicate ghost col
    pattern.addEntry(13, 40);

    pattern.finalize();

    EXPECT_EQ(pattern.numGhostCols(), 3);  // 5, 25, 40

    auto ghost_map = pattern.getGhostColMap();
    EXPECT_EQ(ghost_map.size(), 3);

    // Ghost cols should be sorted
    EXPECT_EQ(ghost_map[0], 5);
    EXPECT_EQ(ghost_map[1], 25);
    EXPECT_EQ(ghost_map[2], 40);
}

TEST(DistributedSparsityPatternTest, GhostIndexConversion) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    pattern.addEntry(12, 5);
    pattern.addEntry(12, 25);
    pattern.addEntry(12, 40);

    pattern.finalize();

    // Ghost map: [5, 25, 40] -> indices [0, 1, 2]
    EXPECT_EQ(pattern.ghostColToGlobal(0), 5);
    EXPECT_EQ(pattern.ghostColToGlobal(1), 25);
    EXPECT_EQ(pattern.ghostColToGlobal(2), 40);

    EXPECT_EQ(pattern.globalToGhostCol(5), 0);
    EXPECT_EQ(pattern.globalToGhostCol(25), 1);
    EXPECT_EQ(pattern.globalToGhostCol(40), 2);
    EXPECT_EQ(pattern.globalToGhostCol(10), -1);  // Not a ghost (owned)
    EXPECT_EQ(pattern.globalToGhostCol(99), -1);  // Not present
}

// ============================================================================
// Preallocation Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, PreallocationInfo) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Row 12 (local 2): 2 diag, 3 offdiag
    pattern.addEntry(12, 10);
    pattern.addEntry(12, 12);
    pattern.addEntry(12, 5);
    pattern.addEntry(12, 25);
    pattern.addEntry(12, 40);

    // Row 15 (local 5): 1 diag, 1 offdiag
    pattern.addEntry(15, 15);
    pattern.addEntry(15, 3);

    pattern.finalize();

    auto prealloc = pattern.getPreallocationInfo();

    EXPECT_EQ(prealloc.diag_nnz_per_row.size(), 10);
    EXPECT_EQ(prealloc.offdiag_nnz_per_row.size(), 10);

    EXPECT_EQ(prealloc.diag_nnz_per_row[2], 2);
    EXPECT_EQ(prealloc.offdiag_nnz_per_row[2], 3);
    EXPECT_EQ(prealloc.diag_nnz_per_row[5], 1);
    EXPECT_EQ(prealloc.offdiag_nnz_per_row[5], 1);

    EXPECT_EQ(prealloc.total_diag_nnz, 3);
    EXPECT_EQ(prealloc.total_offdiag_nnz, 4);
    EXPECT_EQ(prealloc.max_diag_nnz, 2);
    EXPECT_EQ(prealloc.max_offdiag_nnz, 3);
}

TEST(DistributedSparsityPatternTest, DiagOffdiagNnzPerRow) {
    DistributedSparsityPattern pattern(0, 5, 0, 5, 20, 20);

    // Row 0: 2 diag, 1 offdiag
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 2);
    pattern.addEntry(0, 10);

    // Row 2: 1 diag, 2 offdiag
    pattern.addEntry(2, 3);
    pattern.addEntry(2, 7);
    pattern.addEntry(2, 15);

    pattern.finalize();

    auto diag_nnz = pattern.getDiagNnzPerRow();
    auto offdiag_nnz = pattern.getOffdiagNnzPerRow();

    EXPECT_EQ(diag_nnz[0], 2);
    EXPECT_EQ(offdiag_nnz[0], 1);
    EXPECT_EQ(diag_nnz[2], 1);
    EXPECT_EQ(offdiag_nnz[2], 2);
}

// ============================================================================
// Index Conversion Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, RowIndexConversion) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Owned rows: 10-19 -> local 0-9
    EXPECT_EQ(pattern.globalRowToLocal(10), 0);
    EXPECT_EQ(pattern.globalRowToLocal(15), 5);
    EXPECT_EQ(pattern.globalRowToLocal(19), 9);
    EXPECT_EQ(pattern.globalRowToLocal(9), -1);   // Not owned
    EXPECT_EQ(pattern.globalRowToLocal(20), -1);  // Not owned

    EXPECT_EQ(pattern.localRowToGlobal(0), 10);
    EXPECT_EQ(pattern.localRowToGlobal(5), 15);
    EXPECT_EQ(pattern.localRowToGlobal(9), 19);
}

TEST(DistributedSparsityPatternTest, ColIndexConversion) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Owned cols: 10-19 -> local 0-9
    EXPECT_EQ(pattern.globalColToLocal(10), 0);
    EXPECT_EQ(pattern.globalColToLocal(15), 5);
    EXPECT_EQ(pattern.globalColToLocal(5), -1);   // Ghost
    EXPECT_EQ(pattern.globalColToLocal(25), -1);  // Ghost

    EXPECT_EQ(pattern.localColToGlobal(0), 10);
    EXPECT_EQ(pattern.localColToGlobal(5), 15);
}

// ============================================================================
// Rectangular Pattern Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, RectangularPattern) {
    // 30 rows, 50 cols globally
    // This rank owns rows 10-19 and cols 20-34
    IndexRange owned_rows{10, 20};
    IndexRange owned_cols{20, 35};
    DistributedSparsityPattern pattern(owned_rows, owned_cols, 30, 50);

    EXPECT_FALSE(pattern.isSquare());
    EXPECT_EQ(pattern.globalRows(), 30);
    EXPECT_EQ(pattern.globalCols(), 50);
    EXPECT_EQ(pattern.numOwnedRows(), 10);
    EXPECT_EQ(pattern.numOwnedCols(), 15);

    // Add entries
    pattern.addEntry(12, 25);  // Diagonal (col 25 is owned)
    pattern.addEntry(12, 10);  // Off-diagonal (col 10 not owned)
    pattern.addEntry(12, 45);  // Off-diagonal

    EXPECT_EQ(pattern.getDiagNnz(), 1);
    EXPECT_EQ(pattern.getOffdiagNnz(), 2);

    pattern.finalize();
    EXPECT_TRUE(pattern.validate());
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, ComputeStats) {
    DistributedSparsityPattern pattern(0, 5, 0, 5, 20, 20);

    // Row 0: 2 diag, 1 offdiag
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 2);
    pattern.addEntry(0, 10);

    // Row 2: 1 diag, 2 offdiag
    pattern.addEntry(2, 3);
    pattern.addEntry(2, 7);
    pattern.addEntry(2, 15);

    pattern.finalize();

    auto stats = pattern.computeStats();

    EXPECT_EQ(stats.n_owned_rows, 5);
    EXPECT_EQ(stats.n_owned_cols, 5);
    EXPECT_EQ(stats.n_ghost_cols, 3);  // 7, 10, 15
    EXPECT_EQ(stats.local_diag_nnz, 3);
    EXPECT_EQ(stats.local_offdiag_nnz, 3);
    EXPECT_EQ(stats.local_total_nnz, 6);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, Validate) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    pattern.addEntry(12, 10);
    pattern.addEntry(12, 5);
    pattern.addEntry(15, 15);
    pattern.addEntry(15, 40);

    EXPECT_TRUE(pattern.validate());  // Valid during building
    pattern.finalize();
    EXPECT_TRUE(pattern.validate());  // Valid after finalization
    EXPECT_TRUE(pattern.validationError().empty());
}

// ============================================================================
// EnsureDiagonal Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, EnsureDiagonal) {
    // Square pattern with matching row/col ownership
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    pattern.addEntry(12, 5);  // Just an off-diagonal entry
    pattern.ensureDiagonal();

    // All owned rows should now have their diagonal entry
    // Diagonal entries: (10,10), (11,11), ..., (19,19)
    // But 10-19 maps to local cols 0-9, so all should be diag
    pattern.finalize();

    EXPECT_EQ(pattern.getDiagNnz(), 10);  // All 10 diagonals

    for (GlobalIndex local_row = 0; local_row < 10; ++local_row) {
        EXPECT_GE(pattern.getRowDiagNnz(local_row), 1);
    }
}

TEST(DistributedSparsityPatternTest, EnsureNonEmptyRows) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    pattern.addEntry(12, 15);  // Only row 12 has an entry
    pattern.ensureNonEmptyRows();

    pattern.finalize();

    // All rows should have at least one entry
    for (GlobalIndex local_row = 0; local_row < 10; ++local_row) {
        EXPECT_GE(pattern.getRowNnz(local_row), 1)
            << "Row " << local_row << " is empty";
    }
}

// ============================================================================
// Move/Copy Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, MoveConstruction) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);
    pattern.addEntry(12, 10);
    pattern.addEntry(12, 5);
    pattern.finalize();

    DistributedSparsityPattern moved(std::move(pattern));

    EXPECT_EQ(moved.globalRows(), 50);
    EXPECT_EQ(moved.numOwnedRows(), 10);
    EXPECT_TRUE(moved.isFinalized());
    EXPECT_TRUE(moved.hasEntry(12, 10));
    EXPECT_TRUE(moved.hasEntry(12, 5));

    EXPECT_EQ(pattern.globalRows(), 0);
}

TEST(DistributedSparsityPatternTest, CopyConstruction) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);
    pattern.addEntry(12, 10);
    pattern.addEntry(12, 5);
    pattern.finalize();

    DistributedSparsityPattern copy(pattern);
    EXPECT_TRUE(copy.isFinalized());
    EXPECT_TRUE(copy.hasEntry(12, 10));
    EXPECT_TRUE(copy.hasEntry(12, 5));

    // Original unchanged
    EXPECT_TRUE(pattern.isFinalized());
}

// ============================================================================
// HasEntry Tests
// ============================================================================

TEST(DistributedSparsityPatternTest, HasEntry) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    pattern.addEntry(12, 10);  // Diag
    pattern.addEntry(12, 5);   // Offdiag

    // Before finalization
    EXPECT_TRUE(pattern.hasEntry(12, 10));
    EXPECT_TRUE(pattern.hasEntry(12, 5));
    EXPECT_FALSE(pattern.hasEntry(12, 15));
    EXPECT_FALSE(pattern.hasEntry(5, 10));  // Row not owned

    pattern.finalize();

    // After finalization
    EXPECT_TRUE(pattern.hasEntry(12, 10));
    EXPECT_TRUE(pattern.hasEntry(12, 5));
    EXPECT_FALSE(pattern.hasEntry(12, 15));
    EXPECT_FALSE(pattern.hasEntry(5, 10));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(DistributedSparsityPatternTest, EmptyPattern) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);
    pattern.finalize();

    EXPECT_EQ(pattern.getDiagNnz(), 0);
    EXPECT_EQ(pattern.getOffdiagNnz(), 0);
    EXPECT_EQ(pattern.numGhostCols(), 0);
    EXPECT_TRUE(pattern.validate());
}

TEST(DistributedSparsityPatternTest, AllDiagonalPattern) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Only add diagonal entries (all owned cols)
    pattern.addEntry(10, 10);
    pattern.addEntry(11, 11);
    pattern.addEntry(12, 12);

    pattern.finalize();

    EXPECT_EQ(pattern.getDiagNnz(), 3);
    EXPECT_EQ(pattern.getOffdiagNnz(), 0);
    EXPECT_EQ(pattern.numGhostCols(), 0);
}

TEST(DistributedSparsityPatternTest, AllOffdiagonalPattern) {
    DistributedSparsityPattern pattern(10, 10, 10, 10, 50, 50);

    // Only add off-diagonal entries (all ghost cols)
    pattern.addEntry(10, 5);
    pattern.addEntry(11, 25);
    pattern.addEntry(12, 40);

    pattern.finalize();

    EXPECT_EQ(pattern.getDiagNnz(), 0);
    EXPECT_EQ(pattern.getOffdiagNnz(), 3);
    EXPECT_EQ(pattern.numGhostCols(), 3);
}

TEST(DistributedSparsityPatternTest, SingleRowOwnership) {
    DistributedSparsityPattern pattern(5, 1, 5, 1, 10, 10);

    pattern.addEntry(5, 5);  // Diag
    pattern.addEntry(5, 0);  // Offdiag
    pattern.addEntry(5, 9);  // Offdiag

    pattern.finalize();

    EXPECT_EQ(pattern.numOwnedRows(), 1);
    EXPECT_EQ(pattern.getDiagNnz(), 1);
    EXPECT_EQ(pattern.getOffdiagNnz(), 2);
}
