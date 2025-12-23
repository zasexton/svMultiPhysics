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
#include "Sparsity/CompressedSparsity.h"
#include <vector>
#include <algorithm>
#include <set>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// ThreadLocalBuffer Tests
// ============================================================================

TEST(ThreadLocalBufferTest, AddSingleEntry) {
    ThreadLocalBuffer buffer;
    buffer.addEntry(0, 1);

    EXPECT_EQ(buffer.size(), 1);
    EXPECT_EQ(buffer.entries()[0].first, 0);
    EXPECT_EQ(buffer.entries()[0].second, 1);
}

TEST(ThreadLocalBufferTest, AddMultipleEntries) {
    ThreadLocalBuffer buffer;
    buffer.addEntry(0, 1);
    buffer.addEntry(1, 2);
    buffer.addEntry(2, 3);

    EXPECT_EQ(buffer.size(), 3);
}

TEST(ThreadLocalBufferTest, AddEntriesBatch) {
    ThreadLocalBuffer buffer;
    std::vector<GlobalIndex> cols = {1, 2, 3, 4, 5};
    buffer.addEntries(0, cols);

    EXPECT_EQ(buffer.size(), 5);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(buffer.entries()[i].first, 0);
        EXPECT_EQ(buffer.entries()[i].second, static_cast<GlobalIndex>(i + 1));
    }
}

TEST(ThreadLocalBufferTest, Clear) {
    ThreadLocalBuffer buffer;
    buffer.addEntry(0, 1);
    buffer.addEntry(1, 2);

    buffer.clear();
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.entries().empty());
}

// ============================================================================
// CompressedSparsity Construction Tests
// ============================================================================

TEST(CompressedSparsityTest, DefaultConstruction) {
    CompressedSparsity cs;
    EXPECT_EQ(cs.numRows(), 0);
    EXPECT_EQ(cs.numCols(), 0);
}

TEST(CompressedSparsityTest, ConstructWithDimensions) {
    CompressedSparsity cs(100, 100);

    EXPECT_EQ(cs.numRows(), 100);
    EXPECT_EQ(cs.numCols(), 100);
}

TEST(CompressedSparsityTest, ConstructWithDimensionsSquare) {
    CompressedSparsity cs(100);  // Square by default

    EXPECT_EQ(cs.numRows(), 100);
    EXPECT_EQ(cs.numCols(), 100);
}

TEST(CompressedSparsityTest, ConstructWithOptions) {
    CompressedSparsityOptions opts;
    opts.mode = CompressionMode::TwoPass;
    opts.buffer_size_hint = 1024;

    CompressedSparsity cs(100, 100, opts);

    EXPECT_EQ(cs.numRows(), 100);
    EXPECT_EQ(cs.getOptions().mode, CompressionMode::TwoPass);
}

// Note: Move construction test removed because CompressedSparsity contains
// std::atomic and std::mutex members which are not movable. The class is
// designed to be non-copyable and non-movable for thread-safety reasons.

// ============================================================================
// Configuration Tests
// ============================================================================

TEST(CompressedSparsityTest, Resize) {
    CompressedSparsity cs(50, 50);
    cs.addEntry(0, 0);

    cs.resize(100, 100);

    EXPECT_EQ(cs.numRows(), 100);
    EXPECT_EQ(cs.numCols(), 100);
}

TEST(CompressedSparsityTest, Clear) {
    CompressedSparsity cs(100, 100);
    cs.addEntry(0, 0);
    cs.addEntry(1, 1);

    cs.clear();

    EXPECT_EQ(cs.getApproximateNnz(), 0);
}

TEST(CompressedSparsityTest, Reserve) {
    CompressedSparsity cs(100, 100);
    cs.reserve(10);  // 10 entries per row

    // Should not throw, just preallocate
    cs.addEntry(0, 0);
    EXPECT_EQ(cs.numRows(), 100);
}

TEST(CompressedSparsityTest, ReservePerRow) {
    CompressedSparsity cs(5, 5);
    std::vector<GlobalIndex> reserves = {2, 3, 1, 4, 2};
    cs.reserve(reserves);

    // Should not throw
    cs.addEntry(0, 0);
    cs.addEntry(0, 1);
    EXPECT_GE(cs.getApproximateNnz(), 2);
}

// ============================================================================
// Single-Pass Insertion Tests
// ============================================================================

TEST(CompressedSparsityTest, AddSingleEntry) {
    CompressedSparsity cs(10, 10);
    cs.addEntry(0, 1);

    EXPECT_GE(cs.getApproximateNnz(), 1);
}

TEST(CompressedSparsityTest, AddMultipleEntries) {
    CompressedSparsity cs(10, 10);
    cs.addEntry(0, 1);
    cs.addEntry(0, 2);
    cs.addEntry(1, 3);

    EXPECT_GE(cs.getApproximateNnz(), 3);
}

TEST(CompressedSparsityTest, AddEntriesBatch) {
    CompressedSparsity cs(10, 10);
    std::vector<GlobalIndex> cols = {0, 2, 4, 6, 8};
    cs.addEntries(0, cols);

    EXPECT_GE(cs.getApproximateNnz(), 5);
}

TEST(CompressedSparsityTest, AddElementCouplings) {
    CompressedSparsity cs(10, 10);
    std::vector<GlobalIndex> dofs = {0, 2, 5, 7};
    cs.addElementCouplings(dofs);

    // 4x4 = 16 entries
    EXPECT_GE(cs.getApproximateNnz(), 16);
}

TEST(CompressedSparsityTest, AddElementCouplingsRectangular) {
    CompressedSparsity cs(10, 10);
    std::vector<GlobalIndex> row_dofs = {0, 1, 2};
    std::vector<GlobalIndex> col_dofs = {3, 4};
    cs.addElementCouplings(row_dofs, col_dofs);

    // 3x2 = 6 entries
    EXPECT_GE(cs.getApproximateNnz(), 6);
}

TEST(CompressedSparsityTest, AddDuplicateEntries) {
    CompressedSparsity cs(10, 10);
    cs.addEntry(0, 1);
    cs.addEntry(0, 1);  // Duplicate
    cs.addEntry(0, 1);  // Duplicate

    // Duplicates should be merged in final pattern
    SparsityPattern pattern = cs.toSparsityPattern();
    EXPECT_EQ(pattern.getRowNnz(0), 1);  // Only 1 unique entry
}

// ============================================================================
// Two-Pass Construction Tests
// ============================================================================

TEST(CompressedSparsityTest, TwoPassConstruction) {
    CompressedSparsityOptions opts;
    opts.mode = CompressionMode::TwoPass;

    CompressedSparsity cs(10, 10, opts);

    EXPECT_TRUE(cs.isCountingPhase());

    // First pass: count
    cs.countEntry(0, 1);
    cs.countEntry(0, 2);
    cs.countEntry(1, 0);

    cs.finalizeCounting();

    EXPECT_FALSE(cs.isCountingPhase());

    // Second pass: fill
    cs.addEntry(0, 1);
    cs.addEntry(0, 2);
    cs.addEntry(1, 0);

    SparsityPattern pattern = cs.toSparsityPattern();
    EXPECT_EQ(pattern.getNnz(), 3);
}

TEST(CompressedSparsityTest, TwoPassCountEntries) {
    CompressedSparsityOptions opts;
    opts.mode = CompressionMode::TwoPass;

    CompressedSparsity cs(10, 10, opts);

    std::vector<GlobalIndex> cols = {1, 2, 3};
    cs.countEntries(0, cols);

    cs.finalizeCounting();

    cs.addEntries(0, cols);

    SparsityPattern pattern = cs.toSparsityPattern();
    EXPECT_EQ(pattern.getRowNnz(0), 3);
}

TEST(CompressedSparsityTest, TwoPassCountElementCouplings) {
    CompressedSparsityOptions opts;
    opts.mode = CompressionMode::TwoPass;

    CompressedSparsity cs(10, 10, opts);

    std::vector<GlobalIndex> dofs = {0, 1, 2};

    // First pass: count
    cs.countElementCouplings(dofs);
    cs.finalizeCounting();

    // Second pass: fill
    cs.addElementCouplings(dofs);

    SparsityPattern pattern = cs.toSparsityPattern();
    EXPECT_EQ(pattern.getNnz(), 9);  // 3x3
}

// ============================================================================
// Parallel Insertion Tests
// ============================================================================

TEST(CompressedSparsityTest, InitThreadBuffers) {
    CompressedSparsity cs(100, 100);
    cs.initThreadBuffers(4);

    // Should not throw
    cs.addEntryThreaded(0, 0, 1);
    cs.addEntryThreaded(1, 1, 2);
    cs.addEntryThreaded(2, 2, 3);
    cs.addEntryThreaded(3, 3, 4);
}

TEST(CompressedSparsityTest, AddEntryThreaded) {
    CompressedSparsity cs(100, 100);
    cs.initThreadBuffers(2);

    // Simulate two threads adding entries
    cs.addEntryThreaded(0, 0, 1);
    cs.addEntryThreaded(0, 0, 2);
    cs.addEntryThreaded(1, 1, 3);
    cs.addEntryThreaded(1, 1, 4);

    cs.mergeThreadBuffers();

    SparsityPattern pattern = cs.toSparsityPattern();
    EXPECT_EQ(pattern.getNnz(), 4);
}

TEST(CompressedSparsityTest, AddEntriesThreaded) {
    CompressedSparsity cs(100, 100);
    cs.initThreadBuffers(2);

    std::vector<GlobalIndex> cols1 = {1, 2, 3};
    std::vector<GlobalIndex> cols2 = {4, 5, 6};

    cs.addEntriesThreaded(0, 0, cols1);
    cs.addEntriesThreaded(1, 1, cols2);

    cs.mergeThreadBuffers();

    SparsityPattern pattern = cs.toSparsityPattern();
    EXPECT_GE(pattern.getNnz(), 6);
}

TEST(CompressedSparsityTest, AddElementCouplingsThreaded) {
    CompressedSparsity cs(100, 100);
    cs.initThreadBuffers(2);

    std::vector<GlobalIndex> dofs1 = {0, 1};
    std::vector<GlobalIndex> dofs2 = {2, 3};

    cs.addElementCouplingsThreaded(0, dofs1);
    cs.addElementCouplingsThreaded(1, dofs2);

    cs.mergeThreadBuffers();

    SparsityPattern pattern = cs.toSparsityPattern();
    EXPECT_GE(pattern.getNnz(), 8);  // 2x2 + 2x2
}

TEST(CompressedSparsityTest, MergeThreadBuffersDeterministic) {
    // Test that merging is deterministic
    CompressedSparsity cs1(100, 100);
    CompressedSparsity cs2(100, 100);

    cs1.initThreadBuffers(2);
    cs2.initThreadBuffers(2);

    // Same entries added in different order
    cs1.addEntryThreaded(0, 0, 5);
    cs1.addEntryThreaded(0, 0, 3);
    cs1.addEntryThreaded(1, 0, 1);
    cs1.addEntryThreaded(1, 0, 7);

    cs2.addEntryThreaded(1, 0, 7);
    cs2.addEntryThreaded(1, 0, 1);
    cs2.addEntryThreaded(0, 0, 3);
    cs2.addEntryThreaded(0, 0, 5);

    cs1.mergeThreadBuffers();
    cs2.mergeThreadBuffers();

    SparsityPattern p1 = cs1.toSparsityPattern();
    SparsityPattern p2 = cs2.toSparsityPattern();

    // Should produce identical patterns
    EXPECT_EQ(p1.getNnz(), p2.getNnz());

    // Row 0 should have same columns
    auto row1 = p1.getRowIndices(0);
    auto row2 = p2.getRowIndices(0);
    ASSERT_EQ(row1.size(), row2.size());
    for (std::size_t i = 0; i < row1.size(); ++i) {
        EXPECT_EQ(row1[i], row2[i]);
    }
}

// ============================================================================
// Compression Tests
// ============================================================================

TEST(CompressedSparsityTest, CompressRemovesDuplicates) {
    CompressedSparsity cs(10, 10);

    // Add many duplicates
    for (int i = 0; i < 100; ++i) {
        cs.addEntry(0, 1);
        cs.addEntry(0, 2);
    }

    cs.compress();

    SparsityPattern pattern = cs.toSparsityPattern();
    EXPECT_EQ(pattern.getRowNnz(0), 2);  // Only 2 unique
}

TEST(CompressedSparsityTest, CompressionModeIncremental) {
    CompressedSparsityOptions opts;
    opts.mode = CompressionMode::Incremental;
    opts.compression_threshold = 10;

    CompressedSparsity cs(10, 10, opts);

    // Add many entries to trigger incremental compression
    for (int i = 0; i < 100; ++i) {
        cs.addEntry(0, i % 5);
    }

    // Should have been compressed at some point
    SparsityPattern pattern = cs.toSparsityPattern();
    EXPECT_LE(pattern.getRowNnz(0), 5);
}

// ============================================================================
// Conversion Tests
// ============================================================================

TEST(CompressedSparsityTest, ToSparsityPattern) {
    CompressedSparsity cs(10, 10);
    cs.addEntry(0, 0);
    cs.addEntry(0, 1);
    cs.addEntry(1, 1);
    cs.addEntry(2, 0);

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.numRows(), 10);
    EXPECT_EQ(pattern.numCols(), 10);
    EXPECT_EQ(pattern.getNnz(), 4);
    EXPECT_TRUE(pattern.isFinalized());
}

TEST(CompressedSparsityTest, ToBuildingPattern) {
    CompressedSparsity cs(10, 10);
    cs.addEntry(0, 0);
    cs.addEntry(1, 1);

    SparsityPattern pattern = cs.toBuildingPattern();

    EXPECT_EQ(pattern.numRows(), 10);
    EXPECT_FALSE(pattern.isFinalized());  // Should be in building state

    // Can add more entries
    pattern.addEntry(2, 2);
    pattern.finalize();

    EXPECT_EQ(pattern.getNnz(), 3);
}

TEST(CompressedSparsityTest, ToSparsityPatternSorted) {
    CompressedSparsity cs(10, 10);

    // Add entries out of order
    cs.addEntry(0, 5);
    cs.addEntry(0, 1);
    cs.addEntry(0, 9);
    cs.addEntry(0, 3);

    SparsityPattern pattern = cs.toSparsityPattern();

    auto row = pattern.getRowIndices(0);
    ASSERT_EQ(row.size(), 4);

    // Should be sorted
    EXPECT_EQ(row[0], 1);
    EXPECT_EQ(row[1], 3);
    EXPECT_EQ(row[2], 5);
    EXPECT_EQ(row[3], 9);
}

// ============================================================================
// Query Tests
// ============================================================================

TEST(CompressedSparsityTest, GetApproximateNnz) {
    CompressedSparsity cs(10, 10);
    cs.addEntry(0, 0);
    cs.addEntry(0, 0);  // Duplicate
    cs.addEntry(1, 1);

    // Approximate includes duplicates
    EXPECT_GE(cs.getApproximateNnz(), 3);
}

TEST(CompressedSparsityTest, GetExactNnz) {
    CompressedSparsity cs(10, 10);
    cs.addEntry(0, 0);
    cs.addEntry(0, 0);  // Duplicate
    cs.addEntry(1, 1);

    // Exact removes duplicates
    EXPECT_EQ(cs.getExactNnz(), 2);
}

TEST(CompressedSparsityTest, GetStats) {
    CompressedSparsity cs(10, 10);

    for (int i = 0; i < 5; ++i) {
        cs.addEntry(0, i);
    }

    cs.compress();
    const auto& stats = cs.getStats();

    EXPECT_EQ(stats.n_rows, 10);
    EXPECT_EQ(stats.n_cols, 10);
    EXPECT_GE(stats.total_insertions, 5);
}

TEST(CompressedSparsityTest, CurrentMemoryBytes) {
    CompressedSparsity cs(100, 100);

    std::size_t initial = cs.currentMemoryBytes();

    // Add some entries
    for (int i = 0; i < 100; ++i) {
        cs.addEntry(i, i);
    }

    std::size_t after = cs.currentMemoryBytes();
    EXPECT_GE(after, initial);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(CompressedSparsityTest, BuildPatternEfficiently) {
    auto element_dofs = [](GlobalIndex e) -> std::vector<GlobalIndex> {
        // Simple element with 4 DOFs
        return {e * 3, e * 3 + 1, e * 3 + 2, e * 3 + 3};
    };

    SparsityPattern pattern = buildPatternEfficiently(100, element_dofs, 30);

    EXPECT_EQ(pattern.numRows(), 100);
    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_GT(pattern.getNnz(), 0);
}

TEST(CompressedSparsityTest, BuildPatternTwoPass) {
    auto element_dofs = [](GlobalIndex e) -> std::vector<GlobalIndex> {
        return {e * 2, e * 2 + 1, e * 2 + 2};
    };

    SparsityPattern pattern = buildPatternTwoPass(50, element_dofs, 20);

    EXPECT_EQ(pattern.numRows(), 50);
    EXPECT_TRUE(pattern.isFinalized());
    EXPECT_GT(pattern.getNnz(), 0);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(CompressedSparsityTest, EmptyPattern) {
    CompressedSparsity cs(10, 10);

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.numRows(), 10);
    EXPECT_EQ(pattern.getNnz(), 0);
}

TEST(CompressedSparsityTest, SingleEntry) {
    CompressedSparsity cs(100, 100);
    cs.addEntry(50, 50);

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.getNnz(), 1);
    EXPECT_TRUE(pattern.hasEntry(50, 50));
}

TEST(CompressedSparsityTest, AllSameEntry) {
    CompressedSparsity cs(10, 10);

    // Add same entry many times
    for (int i = 0; i < 1000; ++i) {
        cs.addEntry(5, 5);
    }

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.getNnz(), 1);
    EXPECT_TRUE(pattern.hasEntry(5, 5));
}

TEST(CompressedSparsityTest, FullRow) {
    CompressedSparsity cs(10, 10);

    // Fill entire row
    for (GlobalIndex j = 0; j < 10; ++j) {
        cs.addEntry(0, j);
    }

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.getRowNnz(0), 10);
}

TEST(CompressedSparsityTest, DiagonalPattern) {
    CompressedSparsity cs(100, 100);

    // Add diagonal
    for (GlobalIndex i = 0; i < 100; ++i) {
        cs.addEntry(i, i);
    }

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.getNnz(), 100);
    for (GlobalIndex i = 0; i < 100; ++i) {
        EXPECT_TRUE(pattern.hasEntry(i, i));
    }
}

TEST(CompressedSparsityTest, RectangularPattern) {
    CompressedSparsity cs(10, 20);

    cs.addEntry(0, 15);
    cs.addEntry(5, 19);

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.numRows(), 10);
    EXPECT_EQ(pattern.numCols(), 20);
    EXPECT_TRUE(pattern.hasEntry(0, 15));
    EXPECT_TRUE(pattern.hasEntry(5, 19));
}

// ============================================================================
// FEM-like Pattern Tests
// ============================================================================

TEST(CompressedSparsityTest, TriangularMeshPattern) {
    // Simulate building sparsity from a triangular mesh
    // Each element connects 3 nodes (DOFs)
    const GlobalIndex n_dofs = 50;
    const GlobalIndex n_elements = 80;

    CompressedSparsity cs(n_dofs, n_dofs);

    // Simulate element loop
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        // Pseudo-random element DOFs
        std::vector<GlobalIndex> dofs = {
            e % n_dofs,
            (e + 1) % n_dofs,
            (e + 7) % n_dofs
        };
        cs.addElementCouplings(dofs);
    }

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.numRows(), n_dofs);
    EXPECT_TRUE(pattern.isFinalized());

    // Each row should have at least diagonal (from self-coupling in elements)
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        EXPECT_GT(pattern.getRowNnz(i), 0);
    }
}

TEST(CompressedSparsityTest, QuadMeshPatternTwoPass) {
    // Two-pass construction for quad mesh
    const GlobalIndex n_dofs = 100;
    const GlobalIndex n_elements = 81;  // 9x9 grid of quads

    auto get_element_dofs = [n_dofs](GlobalIndex e) -> std::vector<GlobalIndex> {
        // Simple pattern: 4 consecutive DOFs
        GlobalIndex base = e % (n_dofs - 3);
        return {base, base + 1, base + 2, base + 3};
    };

    CompressedSparsityOptions opts;
    opts.mode = CompressionMode::TwoPass;

    CompressedSparsity cs(n_dofs, n_dofs, opts);

    // First pass: count
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        auto dofs = get_element_dofs(e);
        cs.countElementCouplings(dofs);
    }
    cs.finalizeCounting();

    // Second pass: fill
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        auto dofs = get_element_dofs(e);
        cs.addElementCouplings(dofs);
    }

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.numRows(), n_dofs);
    EXPECT_TRUE(pattern.isFinalized());
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

TEST(CompressedSparsityTest, LargePatternConstruction) {
    // Test that large patterns can be constructed
    const GlobalIndex n_dofs = 10000;
    const GlobalIndex n_elements = 9000;

    CompressedSparsity cs(n_dofs, n_dofs);

    // Element loop
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        std::vector<GlobalIndex> dofs = {
            e % n_dofs,
            (e + 1) % n_dofs,
            (e + 2) % n_dofs,
            (e + 3) % n_dofs
        };
        cs.addElementCouplings(dofs);
    }

    SparsityPattern pattern = cs.toSparsityPattern();

    EXPECT_EQ(pattern.numRows(), n_dofs);
    EXPECT_GT(pattern.getNnz(), 0);
}

TEST(CompressedSparsityTest, MemoryEfficiency) {
    // Verify that compressed sparsity uses less memory than naive approach
    CompressedSparsity cs(1000, 1000);

    // Add entries with many duplicates
    for (int round = 0; round < 10; ++round) {
        for (GlobalIndex i = 0; i < 100; ++i) {
            cs.addEntry(i, i);
            cs.addEntry(i, (i + 1) % 1000);
        }
    }

    std::size_t mem_before = cs.currentMemoryBytes();
    cs.compress();
    std::size_t mem_after = cs.currentMemoryBytes();

    // Memory should decrease or stay same after compression
    EXPECT_LE(mem_after, mem_before);
}

