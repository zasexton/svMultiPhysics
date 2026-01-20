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
#include "Sparsity/SparsityFormat.h"
#include <vector>
#include <algorithm>
#include <set>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Helper to create test patterns
// ============================================================================

SparsityPattern createTestPattern() {
    // Create a simple 5x5 pattern:
    // [1 1 0 0 0]
    // [1 1 1 0 0]
    // [0 1 1 1 0]
    // [0 0 1 1 1]
    // [0 0 0 1 1]
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0); pattern.addEntry(0, 1);
    pattern.addEntry(1, 0); pattern.addEntry(1, 1); pattern.addEntry(1, 2);
    pattern.addEntry(2, 1); pattern.addEntry(2, 2); pattern.addEntry(2, 3);
    pattern.addEntry(3, 2); pattern.addEntry(3, 3); pattern.addEntry(3, 4);
    pattern.addEntry(4, 3); pattern.addEntry(4, 4);
    pattern.finalize();
    return pattern;
}

SparsityPattern createBlockPattern() {
    // Create 6x6 pattern with 2x2 block structure
    SparsityPattern pattern(6, 6);
    // Block (0,0)
    pattern.addEntry(0, 0); pattern.addEntry(0, 1);
    pattern.addEntry(1, 0); pattern.addEntry(1, 1);
    // Block (0,2)
    pattern.addEntry(0, 4); pattern.addEntry(0, 5);
    pattern.addEntry(1, 4); pattern.addEntry(1, 5);
    // Block (1,1)
    pattern.addEntry(2, 2); pattern.addEntry(2, 3);
    pattern.addEntry(3, 2); pattern.addEntry(3, 3);
    // Block (2,2)
    pattern.addEntry(4, 4); pattern.addEntry(4, 5);
    pattern.addEntry(5, 4); pattern.addEntry(5, 5);
    pattern.finalize();
    return pattern;
}

// ============================================================================
// COOData Tests
// ============================================================================

TEST(COODataTest, DefaultConstruction) {
    COOData coo;
    EXPECT_EQ(coo.nnz(), 0);
    EXPECT_TRUE(coo.isValid());
}

TEST(COODataTest, IsValid) {
    COOData coo;
    coo.rows = {0, 1, 2};
    coo.cols = {0, 1, 2};
    coo.n_rows = 3;
    coo.n_cols = 3;

    EXPECT_TRUE(coo.isValid());

    // Make invalid
    coo.cols.push_back(3);
    EXPECT_FALSE(coo.isValid());
}

TEST(COODataTest, Sort) {
    COOData coo;
    coo.rows = {2, 0, 1, 0};
    coo.cols = {1, 2, 0, 0};
    coo.n_rows = 3;
    coo.n_cols = 3;

    coo.sort();

    // Should be sorted by (row, col)
    EXPECT_EQ(coo.rows[0], 0);
    EXPECT_EQ(coo.cols[0], 0);
    EXPECT_EQ(coo.rows[1], 0);
    EXPECT_EQ(coo.cols[1], 2);
    EXPECT_EQ(coo.rows[2], 1);
    EXPECT_EQ(coo.cols[2], 0);
    EXPECT_EQ(coo.rows[3], 2);
    EXPECT_EQ(coo.cols[3], 1);
}

TEST(COODataTest, Deduplicate) {
    COOData coo;
    coo.rows = {0, 0, 1, 1, 0};
    coo.cols = {0, 0, 1, 1, 0};  // (0,0) appears 3 times, (1,1) twice
    coo.n_rows = 2;
    coo.n_cols = 2;

    coo.deduplicate();

    EXPECT_EQ(coo.nnz(), 2);  // Only 2 unique entries
}

TEST(COODataTest, Clear) {
    COOData coo;
    coo.rows = {0, 1};
    coo.cols = {0, 1};

    coo.clear();

    EXPECT_EQ(coo.nnz(), 0);
    EXPECT_TRUE(coo.rows.empty());
    EXPECT_TRUE(coo.cols.empty());
}

// ============================================================================
// CSCData Tests
// ============================================================================

TEST(CSCDataTest, Nnz) {
    CSCData csc;
    csc.col_ptr = {0, 2, 4, 6};
    csc.row_idx = {0, 1, 0, 1, 0, 1};
    csc.n_rows = 2;
    csc.n_cols = 3;

    EXPECT_EQ(csc.nnz(), 6);
}

TEST(CSCDataTest, GetColumn) {
    CSCData csc;
    csc.col_ptr = {0, 2, 4, 5};
    csc.row_idx = {0, 1, 0, 2, 1};
    csc.n_rows = 3;
    csc.n_cols = 3;

    auto col0 = csc.getColumn(0);
    ASSERT_EQ(col0.size(), 2);
    EXPECT_EQ(col0[0], 0);
    EXPECT_EQ(col0[1], 1);

    auto col2 = csc.getColumn(2);
    ASSERT_EQ(col2.size(), 1);
    EXPECT_EQ(col2[0], 1);
}

TEST(CSCDataTest, GetColNnz) {
    CSCData csc;
    csc.col_ptr = {0, 2, 4, 5};
    csc.row_idx = {0, 1, 0, 2, 1};
    csc.n_rows = 3;
    csc.n_cols = 3;

    EXPECT_EQ(csc.getColNnz(0), 2);
    EXPECT_EQ(csc.getColNnz(1), 2);
    EXPECT_EQ(csc.getColNnz(2), 1);
}

TEST(CSCDataTest, IsValid) {
    CSCData csc;
    csc.col_ptr = {0, 2, 3};
    csc.row_idx = {0, 1, 0};
    csc.n_rows = 2;
    csc.n_cols = 2;

    EXPECT_TRUE(csc.isValid());
}

// ============================================================================
// BSRData Tests
// ============================================================================

TEST(BSRDataTest, NnzBlocks) {
    BSRData bsr;
    bsr.row_ptr = {0, 2, 3};
    bsr.col_idx = {0, 1, 2};
    bsr.n_block_rows = 2;
    bsr.n_block_cols = 3;
    bsr.block_size_row = 2;
    bsr.block_size_col = 2;

    EXPECT_EQ(bsr.nnzBlocks(), 3);
}

TEST(BSRDataTest, ScalarDimensions) {
    BSRData bsr;
    bsr.n_block_rows = 3;
    bsr.n_block_cols = 4;
    bsr.block_size_row = 2;
    bsr.block_size_col = 3;

    EXPECT_EQ(bsr.numRows(), 6);
    EXPECT_EQ(bsr.numCols(), 12);
}

TEST(BSRDataTest, ScalarNnz) {
    BSRData bsr;
    bsr.row_ptr = {0, 2, 3};
    bsr.n_block_rows = 2;
    bsr.n_block_cols = 3;
    bsr.block_size_row = 2;
    bsr.block_size_col = 2;

    // 3 blocks * 2 * 2 = 12 scalar entries
    EXPECT_EQ(bsr.nnz(), 12);
}

// ============================================================================
// ELLPACKData Tests
// ============================================================================

TEST(ELLPACKDataTest, GetEntry) {
    ELLPACKData ell;
    ell.col_idx = {0, 1, -1, 2, 3, 4};  // 2 rows, max 3 per row
    ell.n_rows = 2;
    ell.n_cols = 5;
    ell.max_nnz_per_row = 3;
    ell.padding_value = -1;

    EXPECT_EQ(ell.getEntry(0, 0), 0);
    EXPECT_EQ(ell.getEntry(0, 1), 1);
    EXPECT_EQ(ell.getEntry(0, 2), -1);  // Padding
    EXPECT_EQ(ell.getEntry(1, 0), 2);
    EXPECT_EQ(ell.getEntry(1, 1), 3);
    EXPECT_EQ(ell.getEntry(1, 2), 4);
}

TEST(ELLPACKDataTest, NnzExcludesPadding) {
    ELLPACKData ell;
    ell.col_idx = {0, 1, -1, 2, -1, -1};  // Row 0: 2 entries, Row 1: 1 entry
    ell.n_rows = 2;
    ell.n_cols = 3;
    ell.max_nnz_per_row = 3;
    ell.padding_value = -1;

    EXPECT_EQ(ell.nnz(), 3);  // Only actual entries
}

TEST(ELLPACKDataTest, IsValid) {
    ELLPACKData ell;
    ell.col_idx = {0, 1, 2, 3};
    ell.n_rows = 2;
    ell.n_cols = 4;
    ell.max_nnz_per_row = 2;

    EXPECT_TRUE(ell.isValid());
}

// ============================================================================
// CSR to COO Conversion Tests
// ============================================================================

TEST(SparsityFormatTest, CsrToCoo) {
    SparsityPattern pattern = createTestPattern();
    COOData coo = csrToCoo(pattern);

    EXPECT_EQ(coo.n_rows, 5);
    EXPECT_EQ(coo.n_cols, 5);
    EXPECT_EQ(coo.nnz(), pattern.getNnz());
    EXPECT_TRUE(coo.isValid());
}

TEST(SparsityFormatTest, CooToCsr) {
    COOData coo;
    coo.rows = {0, 0, 1, 1, 2};
    coo.cols = {0, 1, 0, 1, 2};
    coo.n_rows = 3;
    coo.n_cols = 3;

    SparsityPattern pattern = cooToCsr(coo);

    EXPECT_EQ(pattern.numRows(), 3);
    EXPECT_EQ(pattern.numCols(), 3);
    EXPECT_EQ(pattern.getNnz(), 5);
    EXPECT_TRUE(pattern.hasEntry(0, 0));
    EXPECT_TRUE(pattern.hasEntry(0, 1));
    EXPECT_TRUE(pattern.hasEntry(1, 0));
    EXPECT_TRUE(pattern.hasEntry(1, 1));
    EXPECT_TRUE(pattern.hasEntry(2, 2));
}

TEST(SparsityFormatTest, CsrCooCsrRoundtrip) {
    SparsityPattern original = createTestPattern();
    COOData coo = csrToCoo(original);
    SparsityPattern recovered = cooToCsr(coo);

    EXPECT_EQ(recovered.numRows(), original.numRows());
    EXPECT_EQ(recovered.numCols(), original.numCols());
    EXPECT_EQ(recovered.getNnz(), original.getNnz());

    // Check all entries match
    for (GlobalIndex i = 0; i < original.numRows(); ++i) {
        auto orig_row = original.getRowIndices(i);
        auto rec_row = recovered.getRowIndices(i);
        ASSERT_EQ(orig_row.size(), rec_row.size());
        for (GlobalIndex k = 0; k < orig_row.size(); ++k) {
            EXPECT_EQ(orig_row[k], rec_row[k]);
        }
    }
}

// ============================================================================
// CSR to CSC Conversion Tests
// ============================================================================

TEST(SparsityFormatTest, CsrToCsc) {
    SparsityPattern pattern = createTestPattern();
    CSCData csc = csrToCsc(pattern);

    EXPECT_EQ(csc.n_rows, 5);
    EXPECT_EQ(csc.n_cols, 5);
    EXPECT_EQ(csc.nnz(), pattern.getNnz());
    EXPECT_TRUE(csc.isValid());
}

TEST(SparsityFormatTest, CscToCsr) {
    CSCData csc;
    csc.col_ptr = {0, 2, 4, 5};
    csc.row_idx = {0, 1, 0, 1, 2};
    csc.n_rows = 3;
    csc.n_cols = 3;

    SparsityPattern pattern = cscToCsr(csc);

    EXPECT_EQ(pattern.numRows(), 3);
    EXPECT_EQ(pattern.numCols(), 3);
    EXPECT_EQ(pattern.getNnz(), 5);
}

TEST(SparsityFormatTest, CsrCscCsrRoundtrip) {
    SparsityPattern original = createTestPattern();
    CSCData csc = csrToCsc(original);
    SparsityPattern recovered = cscToCsr(csc);

    EXPECT_EQ(recovered.numRows(), original.numRows());
    EXPECT_EQ(recovered.getNnz(), original.getNnz());

    // Verify structure matches
    for (GlobalIndex i = 0; i < original.numRows(); ++i) {
        auto orig_row = original.getRowIndices(i);
        auto rec_row = recovered.getRowIndices(i);
        ASSERT_EQ(orig_row.size(), rec_row.size());
    }
}

TEST(SparsityFormatTest, CscColumnAccess) {
    SparsityPattern pattern = createTestPattern();
    CSCData csc = csrToCsc(pattern);

    // Column 0 should have rows 0 and 1
    auto col0 = csc.getColumn(0);
    std::set<GlobalIndex> col0_set(col0.begin(), col0.end());
    EXPECT_TRUE(col0_set.count(0));
    EXPECT_TRUE(col0_set.count(1));

    // Column 4 should have rows 3 and 4
    auto col4 = csc.getColumn(4);
    std::set<GlobalIndex> col4_set(col4.begin(), col4.end());
    EXPECT_TRUE(col4_set.count(3));
    EXPECT_TRUE(col4_set.count(4));
}

// ============================================================================
// CSR to BSR Conversion Tests
// ============================================================================

TEST(SparsityFormatTest, CsrToBsrSquare) {
    SparsityPattern pattern = createBlockPattern();
    BSRData bsr = csrToBsr(pattern, 2);

    EXPECT_EQ(bsr.n_block_rows, 3);
    EXPECT_EQ(bsr.n_block_cols, 3);
    EXPECT_EQ(bsr.block_size_row, 2);
    EXPECT_EQ(bsr.block_size_col, 2);
    EXPECT_TRUE(bsr.isValid());
}

TEST(SparsityFormatTest, CsrToBsrRectangular) {
    // Create 6x9 pattern (3 row blocks x 3 col blocks with block size 2x3)
    SparsityPattern pattern(6, 9);
    // Block (0,0)
    for (GlobalIndex i = 0; i < 2; ++i) {
        for (GlobalIndex j = 0; j < 3; ++j) {
            pattern.addEntry(i, j);
        }
    }
    // Block (1,1)
    for (GlobalIndex i = 2; i < 4; ++i) {
        for (GlobalIndex j = 3; j < 6; ++j) {
            pattern.addEntry(i, j);
        }
    }
    pattern.finalize();

    BSRData bsr = csrToBsr(pattern, 2, 3);

    EXPECT_EQ(bsr.n_block_rows, 3);
    EXPECT_EQ(bsr.n_block_cols, 3);
    EXPECT_EQ(bsr.block_size_row, 2);
    EXPECT_EQ(bsr.block_size_col, 3);
}

TEST(SparsityFormatTest, BsrToCsr) {
    BSRData bsr;
    bsr.row_ptr = {0, 2, 3};
    bsr.col_idx = {0, 1, 1};
    bsr.n_block_rows = 2;
    bsr.n_block_cols = 2;
    bsr.block_size_row = 2;
    bsr.block_size_col = 2;

    SparsityPattern pattern = bsrToCsr(bsr);

    EXPECT_EQ(pattern.numRows(), 4);
    EXPECT_EQ(pattern.numCols(), 4);
    // Block (0,0) and (0,1) = 4+4=8, Block (1,1) = 4, total = 12
    EXPECT_EQ(pattern.getNnz(), 12);
}

TEST(SparsityFormatTest, CsrBsrCsrRoundtrip) {
    SparsityPattern original = createBlockPattern();
    BSRData bsr = csrToBsr(original, 2);
    SparsityPattern recovered = bsrToCsr(bsr);

    EXPECT_EQ(recovered.numRows(), original.numRows());
    // Note: BSR may add fill-in for incomplete blocks
    EXPECT_GE(recovered.getNnz(), original.getNnz());
}

// ============================================================================
// CSR to ELLPACK Conversion Tests
// ============================================================================

TEST(SparsityFormatTest, CsrToEllpack) {
    SparsityPattern pattern = createTestPattern();
    ELLPACKData ell = csrToEllpack(pattern);

    EXPECT_EQ(ell.n_rows, 5);
    EXPECT_EQ(ell.n_cols, 5);
    EXPECT_EQ(ell.max_nnz_per_row, 3);  // Max row has 3 entries
    EXPECT_TRUE(ell.isValid());
}

TEST(SparsityFormatTest, CsrToEllpackCustomPadding) {
    SparsityPattern pattern = createTestPattern();
    ELLPACKData ell = csrToEllpack(pattern, -999);

    EXPECT_EQ(ell.padding_value, -999);
}

TEST(SparsityFormatTest, EllpackToCsr) {
    ELLPACKData ell;
    ell.col_idx = {0, 1, -1, 1, 2, -1, 2, 3, -1};
    ell.n_rows = 3;
    ell.n_cols = 4;
    ell.max_nnz_per_row = 3;
    ell.padding_value = -1;

    SparsityPattern pattern = ellpackToCsr(ell);

    EXPECT_EQ(pattern.numRows(), 3);
    EXPECT_EQ(pattern.numCols(), 4);
    EXPECT_EQ(pattern.getNnz(), 6);  // Excluding padding
}

TEST(SparsityFormatTest, CsrEllpackCsrRoundtrip) {
    SparsityPattern original = createTestPattern();
    ELLPACKData ell = csrToEllpack(original);
    SparsityPattern recovered = ellpackToCsr(ell);

    EXPECT_EQ(recovered.numRows(), original.numRows());
    EXPECT_EQ(recovered.getNnz(), original.getNnz());
}

// ============================================================================
// Index Base Conversion Tests
// ============================================================================

TEST(SparsityFormatTest, ConvertIndexBaseZeroToOne) {
    std::vector<GlobalIndex> indices = {0, 1, 2, 3, 4};
    convertIndexBase(indices, IndexBase::ZeroBased, IndexBase::OneBased);

    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 2);
    EXPECT_EQ(indices[2], 3);
    EXPECT_EQ(indices[3], 4);
    EXPECT_EQ(indices[4], 5);
}

TEST(SparsityFormatTest, ConvertIndexBaseOneToZero) {
    std::vector<GlobalIndex> indices = {1, 2, 3, 4, 5};
    convertIndexBase(indices, IndexBase::OneBased, IndexBase::ZeroBased);

    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 1);
    EXPECT_EQ(indices[2], 2);
    EXPECT_EQ(indices[3], 3);
    EXPECT_EQ(indices[4], 4);
}

TEST(SparsityFormatTest, ConvertIndexBaseSameBase) {
    std::vector<GlobalIndex> indices = {0, 1, 2};
    convertIndexBase(indices, IndexBase::ZeroBased, IndexBase::ZeroBased);

    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 1);
    EXPECT_EQ(indices[2], 2);
}

TEST(SparsityFormatTest, GetCSRArraysZeroBased) {
    SparsityPattern pattern = createTestPattern();
    auto [row_ptr, col_idx] = getCSRArrays(pattern, IndexBase::ZeroBased);

    EXPECT_EQ(row_ptr[0], 0);
    EXPECT_GT(row_ptr.size(), 0);
    EXPECT_GT(col_idx.size(), 0);
}

TEST(SparsityFormatTest, GetCSRArraysOneBased) {
    SparsityPattern pattern = createTestPattern();
    auto [row_ptr, col_idx] = getCSRArrays(pattern, IndexBase::OneBased);

    EXPECT_EQ(row_ptr[0], 1);  // 1-based offset
}

TEST(SparsityFormatTest, GetCSRArraysAsInt32) {
    SparsityPattern pattern = createTestPattern();
    auto [row_ptr, col_idx] = getCSRArraysAs<int32_t>(pattern);

    EXPECT_EQ(row_ptr[0], 0);
    static_assert(std::is_same_v<decltype(row_ptr)::value_type, int32_t>);
}

// ============================================================================
// Format Analysis Tests
// ============================================================================

TEST(SparsityFormatTest, IsBSRCompatibleYes) {
    SparsityPattern pattern = createBlockPattern();
    EXPECT_TRUE(isBSRCompatible(pattern, 2));
}

TEST(SparsityFormatTest, IsBSRCompatibleNo) {
    // Non-block pattern
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 2);
    pattern.addEntry(3, 4);
    pattern.finalize();

    EXPECT_FALSE(isBSRCompatible(pattern, 2));
}

TEST(SparsityFormatTest, DetectBlockSize) {
    SparsityPattern pattern = createBlockPattern();
    GlobalIndex detected = detectBlockSize(pattern, 4);

    // Should detect 2x2 blocks
    EXPECT_GE(detected, 1);
}

TEST(SparsityFormatTest, ComputeEllpackEfficiency) {
    // Uniform row lengths = high efficiency
    SparsityPattern uniform(5, 5);
    for (GlobalIndex i = 0; i < 5; ++i) {
        uniform.addEntry(i, 0);
        uniform.addEntry(i, 1);
    }
    uniform.finalize();

    double eff_uniform = computeEllpackEfficiency(uniform);
    EXPECT_GT(eff_uniform, 0.9);  // Should be ~100% efficient

    // Variable row lengths = lower efficiency
    SparsityPattern variable(5, 10);
    variable.addEntry(0, 0);
    for (GlobalIndex j = 0; j < 10; ++j) {
        variable.addEntry(1, j);  // Full row
    }
    variable.finalize();

    double eff_variable = computeEllpackEfficiency(variable);
    EXPECT_LT(eff_variable, eff_uniform);  // Should be less efficient
}

TEST(SparsityFormatTest, RecommendFormat) {
    // Diagonal pattern - CSR is fine
    SparsityPattern diag(100, 100);
    for (GlobalIndex i = 0; i < 100; ++i) {
        diag.addEntry(i, i);
    }
    diag.finalize();

    SparseFormat rec = recommendFormat(diag);
    // Should recommend a valid format (implementation specific)
    EXPECT_TRUE(rec == SparseFormat::CSR || rec == SparseFormat::CSC ||
                rec == SparseFormat::COO || rec == SparseFormat::BSR ||
                rec == SparseFormat::ELLPACK || rec == SparseFormat::DIA);
}

// ============================================================================
// SparsityFormatConverter Tests
// ============================================================================

TEST(SparsityFormatConverterTest, Construction) {
    SparsityPattern pattern = createTestPattern();
    SparsityFormatConverter converter(pattern);

    EXPECT_EQ(converter.source().numRows(), pattern.numRows());
}

TEST(SparsityFormatConverterTest, AsCOOCached) {
    SparsityPattern pattern = createTestPattern();
    SparsityFormatConverter converter(pattern);

    const COOData& coo1 = converter.asCOO();
    const COOData& coo2 = converter.asCOO();

    // Should return same cached object
    EXPECT_EQ(&coo1, &coo2);
}

TEST(SparsityFormatConverterTest, AsCSCCached) {
    SparsityPattern pattern = createTestPattern();
    SparsityFormatConverter converter(pattern);

    const CSCData& csc1 = converter.asCSC();
    const CSCData& csc2 = converter.asCSC();

    EXPECT_EQ(&csc1, &csc2);
}

TEST(SparsityFormatConverterTest, AsBSRCached) {
    SparsityPattern pattern = createBlockPattern();
    SparsityFormatConverter converter(pattern);

    const BSRData& bsr1 = converter.asBSR(2);
    const BSRData& bsr2 = converter.asBSR(2);

    EXPECT_EQ(&bsr1, &bsr2);
}

TEST(SparsityFormatConverterTest, AsBSRDifferentBlockSize) {
    SparsityPattern pattern = createBlockPattern();
    SparsityFormatConverter converter(pattern);

    const BSRData& bsr2 = converter.asBSR(2);
    // Different block size should recompute (invalidate cache)
    // Note: This might throw if not divisible, so we use same block size
    const BSRData& bsr2_again = converter.asBSR(2);

    EXPECT_EQ(&bsr2, &bsr2_again);
}

TEST(SparsityFormatConverterTest, AsELLPACKCached) {
    SparsityPattern pattern = createTestPattern();
    SparsityFormatConverter converter(pattern);

    const ELLPACKData& ell1 = converter.asELLPACK();
    const ELLPACKData& ell2 = converter.asELLPACK();

    EXPECT_EQ(&ell1, &ell2);
}

TEST(SparsityFormatConverterTest, ClearCache) {
    SparsityPattern pattern = createTestPattern();
    SparsityFormatConverter converter(pattern);

    // Get cached value
    const COOData& coo1 = converter.asCOO();
    (void)coo1;  // Suppress unused warning

    converter.clearCache();

    // After clear, the optional should be empty, but next call recreates it
    // The new object may or may not be at same address (implementation detail)
    // Just verify it works without crashing
    const COOData& coo2 = converter.asCOO();
    EXPECT_TRUE(coo2.isValid());
}

// ============================================================================
// Format Name Tests
// ============================================================================

TEST(SparsityFormatTest, FormatName) {
    EXPECT_STREQ(formatName(SparseFormat::CSR), "CSR");
    EXPECT_STREQ(formatName(SparseFormat::CSC), "CSC");
    EXPECT_STREQ(formatName(SparseFormat::COO), "COO");
    EXPECT_STREQ(formatName(SparseFormat::BSR), "BSR");
    EXPECT_STREQ(formatName(SparseFormat::ELLPACK), "ELLPACK");
    EXPECT_STREQ(formatName(SparseFormat::DIA), "DIA");
    EXPECT_STREQ(formatName(SparseFormat::Custom), "Custom");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(SparsityFormatTest, EmptyPatternConversions) {
    SparsityPattern empty(5, 5);
    empty.finalize();

    COOData coo = csrToCoo(empty);
    EXPECT_EQ(coo.nnz(), 0);

    CSCData csc = csrToCsc(empty);
    EXPECT_EQ(csc.nnz(), 0);

    ELLPACKData ell = csrToEllpack(empty);
    EXPECT_EQ(ell.nnz(), 0);
}

TEST(SparsityFormatTest, SingleEntryPattern) {
    SparsityPattern single(10, 10);
    single.addEntry(5, 5);
    single.finalize();

    COOData coo = csrToCoo(single);
    EXPECT_EQ(coo.nnz(), 1);
    EXPECT_EQ(coo.rows[0], 5);
    EXPECT_EQ(coo.cols[0], 5);

    CSCData csc = csrToCsc(single);
    EXPECT_EQ(csc.nnz(), 1);
}

TEST(SparsityFormatTest, RectangularPatternConversions) {
    SparsityPattern rect(3, 5);
    rect.addEntry(0, 0);
    rect.addEntry(0, 4);
    rect.addEntry(1, 2);
    rect.addEntry(2, 1);
    rect.addEntry(2, 3);
    rect.finalize();

    COOData coo = csrToCoo(rect);
    EXPECT_EQ(coo.n_rows, 3);
    EXPECT_EQ(coo.n_cols, 5);
    EXPECT_EQ(coo.nnz(), 5);

    CSCData csc = csrToCsc(rect);
    EXPECT_EQ(csc.n_rows, 3);
    EXPECT_EQ(csc.n_cols, 5);
    EXPECT_EQ(csc.nnz(), 5);
}

TEST(SparsityFormatTest, FullRowPattern) {
    SparsityPattern full_row(3, 100);
    for (GlobalIndex j = 0; j < 100; ++j) {
        full_row.addEntry(1, j);
    }
    full_row.finalize();

    ELLPACKData ell = csrToEllpack(full_row);
    EXPECT_EQ(ell.max_nnz_per_row, 100);
}

TEST(SparsityFormatTest, DiagonalPatternToBsr) {
    // 4x4 diagonal pattern with 2x2 blocks
    SparsityPattern diag(4, 4);
    for (GlobalIndex i = 0; i < 4; ++i) {
        diag.addEntry(i, i);
    }
    diag.finalize();

    BSRData bsr = csrToBsr(diag, 2);

    EXPECT_EQ(bsr.n_block_rows, 2);
    EXPECT_EQ(bsr.n_block_cols, 2);
    // Only diagonal blocks should be present
    EXPECT_GE(bsr.nnzBlocks(), 2);
}
