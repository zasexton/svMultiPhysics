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
#include "Sparsity/SparsityTwoPassBuilder.h"
#include "Sparsity/SparsityPattern.h"
#include <vector>
#include <algorithm>
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, DefaultConstruction) {
    SparsityTwoPassBuilder builder;
    EXPECT_EQ(builder.numRows(), 0);
    EXPECT_EQ(builder.numCols(), 0);
    EXPECT_EQ(builder.phase(), TwoPassPhase::Initial);
}

TEST(SparsityTwoPassBuilderTest, SquareConstruction) {
    SparsityTwoPassBuilder builder(10, 10);
    EXPECT_EQ(builder.numRows(), 10);
    EXPECT_EQ(builder.numCols(), 10);
    EXPECT_EQ(builder.phase(), TwoPassPhase::Initial);
}

TEST(SparsityTwoPassBuilderTest, RectangularConstruction) {
    SparsityTwoPassBuilder builder(5, 8);
    EXPECT_EQ(builder.numRows(), 5);
    EXPECT_EQ(builder.numCols(), 8);
}

TEST(SparsityTwoPassBuilderTest, ConstructionWithOptions) {
    TwoPassBuildOptions opts;
    opts.num_threads = 4;
    opts.approximate_count = true;

    SparsityTwoPassBuilder builder(100, 100, opts);
    EXPECT_EQ(builder.numRows(), 100);
}

// ============================================================================
// Counting Phase Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, CountEntry) {
    SparsityTwoPassBuilder builder(5, 5);

    builder.countEntry(0, 0);
    builder.countEntry(0, 1);
    builder.countEntry(0, 2);
    builder.countEntry(1, 0);
    builder.countEntry(1, 1);

    // After counting, estimated NNZ should be >= 5
    EXPECT_GE(builder.estimatedNnz(), 5);
}

TEST(SparsityTwoPassBuilderTest, CountEntries) {
    SparsityTwoPassBuilder builder(5, 5);

    std::vector<GlobalIndex> cols = {0, 1, 2};
    builder.countEntries(0, std::span<const GlobalIndex>(cols));

    EXPECT_GE(builder.estimatedNnz(), 3);
}

TEST(SparsityTwoPassBuilderTest, CountElementCouplings) {
    SparsityTwoPassBuilder builder(10, 10);

    // Element with DOFs 0, 1, 2
    std::vector<GlobalIndex> elem1 = {0, 1, 2};
    builder.countElementCouplings(std::span<const GlobalIndex>(elem1));

    // Each DOF gets 3 entries (couples with all others in element)
    EXPECT_GE(builder.estimatedNnz(), 9);
}

// ============================================================================
// Allocation Tests (finalizeCount)
// ============================================================================

TEST(SparsityTwoPassBuilderTest, FinalizeCountAfterCounting) {
    SparsityTwoPassBuilder builder(5, 5);

    std::vector<GlobalIndex> elem = {0, 1, 2};
    builder.countElementCouplings(std::span<const GlobalIndex>(elem));

    builder.finalizeCount();

    EXPECT_EQ(builder.phase(), TwoPassPhase::Allocated);
}

TEST(SparsityTwoPassBuilderTest, FinalizeCountWithOptions) {
    TwoPassBuildOptions opts;
    opts.overallocation_factor = 1.5;

    SparsityTwoPassBuilder builder(3, 3, opts);

    std::vector<GlobalIndex> elem = {0, 1, 2};
    builder.countElementCouplings(std::span<const GlobalIndex>(elem));

    builder.finalizeCount();

    // Should allocate with overallocation factor
    EXPECT_EQ(builder.phase(), TwoPassPhase::Allocated);
}

// ============================================================================
// Filling Phase Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, AddEntry) {
    // Disable default options that add extra entries
    TwoPassBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;

    SparsityTwoPassBuilder builder(3, 3, opts);

    // Count phase
    builder.countEntry(0, 0);
    builder.countEntry(0, 2);
    builder.countEntry(1, 0);
    builder.countEntry(1, 1);
    builder.countEntry(1, 2);
    builder.countEntry(2, 1);

    builder.finalizeCount();

    // Fill phase
    builder.addEntry(0, 0);
    builder.addEntry(0, 2);
    builder.addEntry(1, 0);
    builder.addEntry(1, 1);
    builder.addEntry(1, 2);
    builder.addEntry(2, 1);

    SparsityPattern pattern = builder.finalize();

    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_EQ(pattern.numRows(), 3);
    EXPECT_EQ(pattern.numCols(), 3);
    EXPECT_EQ(pattern.getNnz(), 6);

    EXPECT_TRUE(pattern.hasEntry(0, 0));
    EXPECT_TRUE(pattern.hasEntry(0, 2));
    EXPECT_TRUE(pattern.hasEntry(1, 0));
    EXPECT_TRUE(pattern.hasEntry(1, 1));
    EXPECT_TRUE(pattern.hasEntry(1, 2));
    EXPECT_TRUE(pattern.hasEntry(2, 1));
}

TEST(SparsityTwoPassBuilderTest, AddElementCouplings) {
    // Disable default options that add extra entries
    TwoPassBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;

    SparsityTwoPassBuilder builder(10, 10, opts);

    // Element with DOFs 0, 1, 2
    std::vector<GlobalIndex> elem = {0, 1, 2};

    // Count
    builder.countElementCouplings(std::span<const GlobalIndex>(elem));
    builder.finalizeCount();

    // Fill
    builder.addElementCouplings(std::span<const GlobalIndex>(elem));

    SparsityPattern pattern = builder.finalize();

    // Should have 9 entries (3x3 dense block)
    EXPECT_EQ(pattern.getNnz(), 9);

    // Verify all couplings
    for (GlobalIndex row : elem) {
        for (GlobalIndex col : elem) {
            EXPECT_TRUE(pattern.hasEntry(row, col));
        }
    }
}

TEST(SparsityTwoPassBuilderTest, AddMultipleElements) {
    SparsityTwoPassBuilder builder(10, 10);

    std::vector<GlobalIndex> elem1 = {0, 1, 2};
    std::vector<GlobalIndex> elem2 = {2, 3, 4};  // Shares DOF 2
    std::vector<GlobalIndex> elem3 = {4, 5};

    // Count all elements
    builder.countElementCouplings(std::span<const GlobalIndex>(elem1));
    builder.countElementCouplings(std::span<const GlobalIndex>(elem2));
    builder.countElementCouplings(std::span<const GlobalIndex>(elem3));

    builder.finalizeCount();

    // Fill all elements
    builder.addElementCouplings(std::span<const GlobalIndex>(elem1));
    builder.addElementCouplings(std::span<const GlobalIndex>(elem2));
    builder.addElementCouplings(std::span<const GlobalIndex>(elem3));

    SparsityPattern pattern = builder.finalize();

    // Verify structure
    EXPECT_TRUE(pattern.hasEntry(0, 0));
    EXPECT_TRUE(pattern.hasEntry(0, 1));
    EXPECT_TRUE(pattern.hasEntry(2, 3));  // From elem2
    EXPECT_TRUE(pattern.hasEntry(4, 5));  // From elem3
}

// ============================================================================
// Deduplication Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, DeduplicateDuringFinalize) {
    SparsityTwoPassBuilder builder(5, 5);

    // Count for overlapping elements
    std::vector<GlobalIndex> elem1 = {0, 1, 2};
    std::vector<GlobalIndex> elem2 = {1, 2, 3};  // Overlaps with elem1

    builder.countElementCouplings(std::span<const GlobalIndex>(elem1));
    builder.countElementCouplings(std::span<const GlobalIndex>(elem2));
    builder.finalizeCount();

    builder.addElementCouplings(std::span<const GlobalIndex>(elem1));
    builder.addElementCouplings(std::span<const GlobalIndex>(elem2));

    SparsityPattern pattern = builder.finalize();

    // Check that duplicates are removed
    // Row 1 should have entries {0, 1, 2, 3} - 4 unique entries
    EXPECT_EQ(pattern.getRowNnz(1), 4);

    // Verify sorted order
    auto row1 = pattern.getRowIndices(1);
    std::vector<GlobalIndex> expected = {0, 1, 2, 3};
    std::vector<GlobalIndex> actual(row1.begin(), row1.end());
    EXPECT_EQ(actual, expected);
}

// ============================================================================
// Rectangular Pattern Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, RectangularPattern) {
    // Disable default options that add extra entries
    // Note: ensure_diagonal does not apply for rectangular patterns (nRows != nCols)
    // but ensure_non_empty_rows would add entries to row 2
    TwoPassBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;

    SparsityTwoPassBuilder builder(3, 5, opts);

    // Count row DOFs x col DOFs
    std::vector<GlobalIndex> row_dofs = {0, 1};
    std::vector<GlobalIndex> col_dofs = {0, 2, 4};

    builder.countElementCouplings(std::span<const GlobalIndex>(row_dofs),
                                   std::span<const GlobalIndex>(col_dofs));
    builder.finalizeCount();

    builder.addElementCouplings(std::span<const GlobalIndex>(row_dofs),
                                 std::span<const GlobalIndex>(col_dofs));

    SparsityPattern pattern = builder.finalize();

    EXPECT_EQ(pattern.numRows(), 3);
    EXPECT_EQ(pattern.numCols(), 5);
    EXPECT_EQ(pattern.getNnz(), 6);  // 2 rows x 3 cols

    EXPECT_TRUE(pattern.hasEntry(0, 0));
    EXPECT_TRUE(pattern.hasEntry(0, 2));
    EXPECT_TRUE(pattern.hasEntry(0, 4));
    EXPECT_TRUE(pattern.hasEntry(1, 0));
    EXPECT_TRUE(pattern.hasEntry(1, 2));
    EXPECT_TRUE(pattern.hasEntry(1, 4));
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, GetBuildStats) {
    SparsityTwoPassBuilder builder(10, 10);

    std::vector<GlobalIndex> elem1 = {0, 1, 2};
    std::vector<GlobalIndex> elem2 = {3, 4, 5};

    builder.countElementCouplings(std::span<const GlobalIndex>(elem1));
    builder.countElementCouplings(std::span<const GlobalIndex>(elem2));
    builder.finalizeCount();

    builder.addElementCouplings(std::span<const GlobalIndex>(elem1));
    builder.addElementCouplings(std::span<const GlobalIndex>(elem2));

    static_cast<void>(builder.finalize());

    TwoPassBuildStats stats = builder.getStats();

    EXPECT_EQ(stats.n_rows, 10);
    EXPECT_EQ(stats.n_cols, 10);
    EXPECT_GT(stats.estimated_nnz, 0);
    EXPECT_GT(stats.actual_nnz, 0);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, EmptyPattern) {
    // Disable default options that add entries
    TwoPassBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;

    SparsityTwoPassBuilder builder(5, 5, opts);
    builder.finalizeCount();

    SparsityPattern pattern = builder.finalize();

    EXPECT_EQ(pattern.numRows(), 5);
    EXPECT_EQ(pattern.numCols(), 5);
    EXPECT_EQ(pattern.getNnz(), 0);
}

TEST(SparsityTwoPassBuilderTest, SingleEntry) {
    // Disable default options that add entries
    TwoPassBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;

    SparsityTwoPassBuilder builder(3, 3, opts);

    builder.countEntry(1, 2);
    builder.finalizeCount();
    builder.addEntry(1, 2);

    SparsityPattern pattern = builder.finalize();

    EXPECT_EQ(pattern.getNnz(), 1);
    EXPECT_TRUE(pattern.hasEntry(1, 2));
}

TEST(SparsityTwoPassBuilderTest, EmptyRow) {
    // Disable default options that add entries
    TwoPassBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;

    SparsityTwoPassBuilder builder(3, 3, opts);

    builder.countEntry(0, 0);
    builder.countEntry(0, 1);
    // Row 1 has no entries
    builder.countEntry(2, 2);

    builder.finalizeCount();

    builder.addEntry(0, 0);
    builder.addEntry(0, 1);
    builder.addEntry(2, 2);

    SparsityPattern pattern = builder.finalize();

    EXPECT_EQ(pattern.getRowNnz(0), 2);
    EXPECT_EQ(pattern.getRowNnz(1), 0);
    EXPECT_EQ(pattern.getRowNnz(2), 1);
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, DeterministicOutput) {
    auto create_pattern = []() {
        SparsityTwoPassBuilder builder(10, 10);

        std::vector<GlobalIndex> elem1 = {0, 1, 2};
        std::vector<GlobalIndex> elem2 = {1, 2, 3};
        std::vector<GlobalIndex> elem3 = {3, 4, 5};

        builder.countElementCouplings(std::span<const GlobalIndex>(elem1));
        builder.countElementCouplings(std::span<const GlobalIndex>(elem2));
        builder.countElementCouplings(std::span<const GlobalIndex>(elem3));

        builder.finalizeCount();

        builder.addElementCouplings(std::span<const GlobalIndex>(elem1));
        builder.addElementCouplings(std::span<const GlobalIndex>(elem2));
        builder.addElementCouplings(std::span<const GlobalIndex>(elem3));

        return builder.finalize();
    };

    auto p1 = create_pattern();
    auto p2 = create_pattern();

    // Same NNZ
    EXPECT_EQ(p1.getNnz(), p2.getNnz());

    // Same row structure
    for (GlobalIndex row = 0; row < 10; ++row) {
        EXPECT_EQ(p1.getRowNnz(row), p2.getRowNnz(row));
    }

    // Same column indices
    auto ci1 = p1.getColIndices();
    auto ci2 = p2.getColIndices();
    EXPECT_TRUE(std::equal(ci1.begin(), ci1.end(), ci2.begin()));
}

// ============================================================================
// Reset and Reuse Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, ResetAndReuse) {
    // Disable default options that add extra entries
    TwoPassBuildOptions opts;
    opts.ensure_diagonal = false;
    opts.ensure_non_empty_rows = false;

    SparsityTwoPassBuilder builder(5, 5, opts);

    std::vector<GlobalIndex> elem = {0, 1, 2};
    builder.countElementCouplings(std::span<const GlobalIndex>(elem));
    builder.finalizeCount();
    builder.addElementCouplings(std::span<const GlobalIndex>(elem));
    (void)builder.finalize();  // suppress nodiscard warning

    // Reset and build different pattern
    builder.reset();
    builder.resize(3, 4);

    EXPECT_EQ(builder.numRows(), 3);
    EXPECT_EQ(builder.numCols(), 4);
    EXPECT_EQ(builder.phase(), TwoPassPhase::Initial);

    builder.countEntry(0, 1);
    builder.countEntry(0, 3);
    builder.finalizeCount();
    builder.addEntry(0, 1);
    builder.addEntry(0, 3);

    SparsityPattern pattern = builder.finalize();

    EXPECT_EQ(pattern.numRows(), 3);
    EXPECT_EQ(pattern.numCols(), 4);
    EXPECT_EQ(pattern.getNnz(), 2);
}

// ============================================================================
// Large Pattern Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, LargerPattern) {
    const GlobalIndex n = 1000;
    const GlobalIndex bandwidth = 5;

    SparsityTwoPassBuilder builder(n, n);

    // Count for banded pattern
    for (GlobalIndex row = 0; row < n; ++row) {
        GlobalIndex start = std::max(GlobalIndex(0), row - bandwidth);
        GlobalIndex end = std::min(n, row + bandwidth + 1);
        for (GlobalIndex col = start; col < end; ++col) {
            builder.countEntry(row, col);
        }
    }

    builder.finalizeCount();

    // Fill banded pattern
    for (GlobalIndex row = 0; row < n; ++row) {
        GlobalIndex start = std::max(GlobalIndex(0), row - bandwidth);
        GlobalIndex end = std::min(n, row + bandwidth + 1);
        for (GlobalIndex col = start; col < end; ++col) {
            builder.addEntry(row, col);
        }
    }

    SparsityPattern pattern = builder.finalize();

    EXPECT_EQ(pattern.numRows(), n);
    EXPECT_EQ(pattern.numCols(), n);

    // Check bandwidth
    EXPECT_LE(pattern.computeBandwidth(), bandwidth);

    // Check diagonal
    for (GlobalIndex i = 0; i < n; ++i) {
        EXPECT_TRUE(pattern.hasEntry(i, i));
    }
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(SparsityTwoPassBuilderTest, BuildFromElementsFunction) {
    // Use the buildPatternTwoPassScalable template function
    auto element_dofs = [](GlobalIndex e) -> std::vector<GlobalIndex> {
        std::vector<GlobalIndex> dofs;
        for (GlobalIndex d = 0; d < 3; ++d) {
            dofs.push_back((e + d) % 10);
        }
        return dofs;
    };

    auto pattern = buildPatternTwoPassScalable(10, 5, element_dofs);

    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_EQ(pattern.numRows(), 10);

    // Verify some element couplings
    EXPECT_TRUE(pattern.hasEntry(0, 1));
    EXPECT_TRUE(pattern.hasEntry(1, 2));
}
