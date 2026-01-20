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
#include "Sparsity/SparsityOps.h"
#include "Sparsity/SparsityPattern.h"
#include <vector>
#include <algorithm>
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Helper to create test patterns
// ============================================================================

SparsityPattern createTestPattern(GlobalIndex n, const std::vector<std::pair<GlobalIndex, GlobalIndex>>& entries) {
    SparsityPattern pattern(n, n);
    for (const auto& [row, col] : entries) {
        pattern.addEntry(row, col);
    }
    pattern.finalize();
    return pattern;
}

// ============================================================================
// Union Tests (using patternUnion from SparsityPattern.h)
// ============================================================================

TEST(SparsityOpsTest, UnionDisjoint) {
    SparsityPattern a = createTestPattern(5, {{0, 0}, {0, 1}});
    SparsityPattern b = createTestPattern(5, {{2, 2}, {3, 3}});

    auto result = patternUnion(a, b);

    EXPECT_EQ(result.getNnz(), 4);
    EXPECT_TRUE(result.hasEntry(0, 0));
    EXPECT_TRUE(result.hasEntry(0, 1));
    EXPECT_TRUE(result.hasEntry(2, 2));
    EXPECT_TRUE(result.hasEntry(3, 3));
}

TEST(SparsityOpsTest, UnionOverlapping) {
    SparsityPattern a = createTestPattern(5, {{0, 0}, {1, 1}, {2, 2}});
    SparsityPattern b = createTestPattern(5, {{1, 1}, {2, 2}, {3, 3}});

    auto result = patternUnion(a, b);

    EXPECT_EQ(result.getNnz(), 4);
    EXPECT_TRUE(result.hasEntry(0, 0));
    EXPECT_TRUE(result.hasEntry(1, 1));
    EXPECT_TRUE(result.hasEntry(2, 2));
    EXPECT_TRUE(result.hasEntry(3, 3));
}

TEST(SparsityOpsTest, UnionInPlace) {
    // unionInPlace requires target in Building state, so we need to build it differently
    SparsityPattern a(5, 5);
    a.addEntry(0, 0);
    a.addEntry(1, 1);
    // Do not finalize - keep in building state

    SparsityPattern b = createTestPattern(5, {{2, 2}, {3, 3}});

    unionInPlace(a, b);
    a.finalize();

    EXPECT_EQ(a.getNnz(), 4);
    EXPECT_TRUE(a.hasEntry(0, 0));
    EXPECT_TRUE(a.hasEntry(2, 2));
}

TEST(SparsityOpsTest, UnionEmpty) {
    SparsityPattern a = createTestPattern(5, {{0, 0}});
    SparsityPattern b(5, 5);
    b.finalize();

    auto result = patternUnion(a, b);
    EXPECT_EQ(result.getNnz(), 1);
}

// ============================================================================
// Intersection Tests
// ============================================================================

TEST(SparsityOpsTest, IntersectOverlapping) {
    SparsityPattern a = createTestPattern(5, {{0, 0}, {1, 1}, {2, 2}});
    SparsityPattern b = createTestPattern(5, {{1, 1}, {2, 2}, {3, 3}});

    auto result = intersect(a, b);

    EXPECT_EQ(result.getNnz(), 2);
    EXPECT_FALSE(result.hasEntry(0, 0));
    EXPECT_TRUE(result.hasEntry(1, 1));
    EXPECT_TRUE(result.hasEntry(2, 2));
    EXPECT_FALSE(result.hasEntry(3, 3));
}

TEST(SparsityOpsTest, IntersectDisjoint) {
    SparsityPattern a = createTestPattern(5, {{0, 0}, {1, 1}});
    SparsityPattern b = createTestPattern(5, {{2, 2}, {3, 3}});

    auto result = intersect(a, b);

    EXPECT_EQ(result.getNnz(), 0);
}

TEST(SparsityOpsTest, IntersectIdentical) {
    SparsityPattern a = createTestPattern(5, {{0, 0}, {1, 1}, {2, 2}});
    SparsityPattern b = createTestPattern(5, {{0, 0}, {1, 1}, {2, 2}});

    auto result = intersect(a, b);

    EXPECT_EQ(result.getNnz(), 3);
}

// ============================================================================
// Difference Tests
// ============================================================================

TEST(SparsityOpsTest, Difference) {
    SparsityPattern a = createTestPattern(5, {{0, 0}, {1, 1}, {2, 2}});
    SparsityPattern b = createTestPattern(5, {{1, 1}, {2, 2}, {3, 3}});

    auto result = difference(a, b);

    EXPECT_EQ(result.getNnz(), 1);
    EXPECT_TRUE(result.hasEntry(0, 0));
    EXPECT_FALSE(result.hasEntry(1, 1));
    EXPECT_FALSE(result.hasEntry(2, 2));
}

TEST(SparsityOpsTest, DifferenceDisjoint) {
    SparsityPattern a = createTestPattern(5, {{0, 0}, {1, 1}});
    SparsityPattern b = createTestPattern(5, {{2, 2}, {3, 3}});

    auto result = difference(a, b);

    EXPECT_EQ(result.getNnz(), 2);
}

TEST(SparsityOpsTest, SymmetricDifference) {
    SparsityPattern a = createTestPattern(5, {{0, 0}, {1, 1}, {2, 2}});
    SparsityPattern b = createTestPattern(5, {{1, 1}, {2, 2}, {3, 3}});

    auto result = symmetricDifference(a, b);

    EXPECT_EQ(result.getNnz(), 2);
    EXPECT_TRUE(result.hasEntry(0, 0));
    EXPECT_FALSE(result.hasEntry(1, 1));
    EXPECT_FALSE(result.hasEntry(2, 2));
    EXPECT_TRUE(result.hasEntry(3, 3));
}

// ============================================================================
// Extract Tests
// ============================================================================

TEST(SparsityOpsTest, ExtractBlock) {
    // Create 5x5 pattern with dense 3x3 block in upper left
    SparsityPattern pattern(5, 5);
    for (GlobalIndex i = 0; i < 3; ++i) {
        for (GlobalIndex j = 0; j < 3; ++j) {
            pattern.addEntry(i, j);
        }
    }
    pattern.finalize();

    std::vector<GlobalIndex> rows = {0, 1, 2};
    std::vector<GlobalIndex> cols = {0, 1, 2};

    auto block = extractBlock(pattern,
                              std::span<const GlobalIndex>(rows),
                              std::span<const GlobalIndex>(cols));

    EXPECT_EQ(block.numRows(), 3);
    EXPECT_EQ(block.numCols(), 3);
    EXPECT_EQ(block.getNnz(), 9);
}

TEST(SparsityOpsTest, ExtractRows) {
    SparsityPattern pattern = createTestPattern(5, {
        {0, 0}, {0, 1},
        {1, 1}, {1, 2},
        {2, 2}, {2, 3},
        {3, 3}, {3, 4},
        {4, 0}, {4, 4}
    });

    std::vector<GlobalIndex> rows = {1, 3};

    auto extracted = extractRows(pattern, std::span<const GlobalIndex>(rows));

    EXPECT_EQ(extracted.numRows(), 2);
    EXPECT_EQ(extracted.numCols(), 5);
    EXPECT_EQ(extracted.getNnz(), 4);

    EXPECT_TRUE(extracted.hasEntry(0, 1));
    EXPECT_TRUE(extracted.hasEntry(0, 2));
    EXPECT_TRUE(extracted.hasEntry(1, 3));
    EXPECT_TRUE(extracted.hasEntry(1, 4));
}

TEST(SparsityOpsTest, ExtractColumns) {
    SparsityPattern pattern = createTestPattern(5, {
        {0, 0}, {0, 2},
        {1, 1}, {1, 3},
        {2, 2}, {2, 4},
        {3, 0}, {3, 2},
        {4, 1}, {4, 3}
    });

    std::vector<GlobalIndex> cols = {0, 2};

    auto extracted = extractColumns(pattern, std::span<const GlobalIndex>(cols));

    EXPECT_EQ(extracted.numRows(), 5);
    EXPECT_EQ(extracted.numCols(), 2);
    // Entries in columns 0 or 2: (0,0), (0,2), (2,2), (3,0), (3,2) = 5
    EXPECT_EQ(extracted.getNnz(), 5);
}

// ============================================================================
// Permutation Tests
// ============================================================================

TEST(SparsityOpsTest, PermuteSymmetric) {
    // Diagonal pattern
    SparsityPattern pattern = createTestPattern(3, {{0, 0}, {1, 1}, {2, 2}});

    // Reverse permutation
    std::vector<GlobalIndex> perm = {2, 1, 0};

    auto permuted = permutePattern(pattern,
                                   std::span<const GlobalIndex>(perm),
                                   std::span<const GlobalIndex>(perm));

    // Should still be diagonal
    EXPECT_EQ(permuted.getNnz(), 3);
    EXPECT_TRUE(permuted.hasEntry(0, 0));
    EXPECT_TRUE(permuted.hasEntry(1, 1));
    EXPECT_TRUE(permuted.hasEntry(2, 2));
}

TEST(SparsityOpsTest, PermuteArrow) {
    // Arrow pattern: dense first row and column
    SparsityPattern pattern(4, 4);
    for (GlobalIndex i = 0; i < 4; ++i) {
        pattern.addEntry(0, i);
        pattern.addEntry(i, 0);
    }
    pattern.finalize();

    // Reverse permutation
    std::vector<GlobalIndex> perm = {3, 2, 1, 0};

    auto permuted = permutePattern(pattern,
                                   std::span<const GlobalIndex>(perm),
                                   std::span<const GlobalIndex>(perm));

    // Should now have dense last row and column
    EXPECT_EQ(permuted.getNnz(), pattern.getNnz());
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_TRUE(permuted.hasEntry(3, i));
        EXPECT_TRUE(permuted.hasEntry(i, 3));
    }
}

TEST(SparsityOpsTest, PermuteSymmetricFunction) {
    SparsityPattern pattern = createTestPattern(3, {
        {0, 0}, {0, 1},
        {1, 0}, {1, 1}, {1, 2},
        {2, 1}, {2, 2}
    });

    std::vector<GlobalIndex> perm = {2, 1, 0};

    auto permuted = permuteSymmetric(pattern, std::span<const GlobalIndex>(perm));

    EXPECT_EQ(permuted.getNnz(), 7);
    EXPECT_TRUE(permuted.isSymmetric());
}

TEST(SparsityOpsTest, InvertPermutation) {
    std::vector<GlobalIndex> perm = {2, 0, 1};
    auto inv = invertPermutation(std::span<const GlobalIndex>(perm));

    // Apply both should give identity
    std::vector<GlobalIndex> result(3);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = inv[static_cast<std::size_t>(perm[i])];
    }

    std::vector<GlobalIndex> identity = {0, 1, 2};
    EXPECT_EQ(result, identity);
}

TEST(SparsityOpsTest, IsValidPermutation) {
    std::vector<GlobalIndex> p1 = {0, 1, 2};
    std::vector<GlobalIndex> p2 = {2, 0, 1};
    std::vector<GlobalIndex> p3 = {0, 0, 1};
    std::vector<GlobalIndex> p4 = {0, 1, 3};

    EXPECT_TRUE(isValidPermutation(std::span<const GlobalIndex>(p1), 3));
    EXPECT_TRUE(isValidPermutation(std::span<const GlobalIndex>(p2), 3));
    EXPECT_FALSE(isValidPermutation(std::span<const GlobalIndex>(p3), 3));  // Duplicate
    EXPECT_FALSE(isValidPermutation(std::span<const GlobalIndex>(p4), 3));  // Out of range
}

// ============================================================================
// Triangular Extraction Tests
// ============================================================================

TEST(SparsityOpsTest, LowerTriangular) {
    // Full 3x3 pattern
    SparsityPattern pattern(3, 3);
    for (GlobalIndex i = 0; i < 3; ++i) {
        for (GlobalIndex j = 0; j < 3; ++j) {
            pattern.addEntry(i, j);
        }
    }
    pattern.finalize();

    auto lower = lowerTriangular(pattern);  // Includes diagonal by default

    EXPECT_EQ(lower.getNnz(), 6);  // 3 + 2 + 1 = 6
    EXPECT_TRUE(lower.hasEntry(0, 0));
    EXPECT_FALSE(lower.hasEntry(0, 1));
    EXPECT_TRUE(lower.hasEntry(1, 0));
    EXPECT_TRUE(lower.hasEntry(1, 1));
    EXPECT_TRUE(lower.hasEntry(2, 0));
    EXPECT_TRUE(lower.hasEntry(2, 1));
    EXPECT_TRUE(lower.hasEntry(2, 2));
}

TEST(SparsityOpsTest, UpperTriangular) {
    // Full 3x3 pattern
    SparsityPattern pattern(3, 3);
    for (GlobalIndex i = 0; i < 3; ++i) {
        for (GlobalIndex j = 0; j < 3; ++j) {
            pattern.addEntry(i, j);
        }
    }
    pattern.finalize();

    auto upper = upperTriangular(pattern);

    EXPECT_EQ(upper.getNnz(), 6);
    EXPECT_TRUE(upper.hasEntry(0, 0));
    EXPECT_TRUE(upper.hasEntry(0, 1));
    EXPECT_TRUE(upper.hasEntry(0, 2));
    EXPECT_FALSE(upper.hasEntry(1, 0));
    EXPECT_TRUE(upper.hasEntry(1, 1));
    EXPECT_TRUE(upper.hasEntry(1, 2));
}

TEST(SparsityOpsTest, ExtractDiagonalEntries) {
    SparsityPattern pattern = createTestPattern(4, {
        {0, 0}, {0, 1},
        {1, 1}, {1, 2},
        {2, 2},
        {3, 3}, {3, 0}
    });

    auto diag = extractDiagonal(pattern);

    EXPECT_EQ(diag.getNnz(), 4);
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_TRUE(diag.hasEntry(i, i));
    }
    EXPECT_FALSE(diag.hasEntry(0, 1));
}

TEST(SparsityOpsTest, StrictLowerTriangular) {
    SparsityPattern pattern(3, 3);
    for (GlobalIndex i = 0; i < 3; ++i) {
        for (GlobalIndex j = 0; j < 3; ++j) {
            pattern.addEntry(i, j);
        }
    }
    pattern.finalize();

    auto strict_lower = strictLowerTriangular(pattern);

    EXPECT_EQ(strict_lower.getNnz(), 3);  // 2 + 1 = 3
    EXPECT_FALSE(strict_lower.hasEntry(0, 0));
    EXPECT_TRUE(strict_lower.hasEntry(1, 0));
    EXPECT_FALSE(strict_lower.hasEntry(1, 1));
}

// ============================================================================
// Standard Pattern Creation Tests
// ============================================================================

TEST(SparsityOpsTest, IdentityPattern) {
    auto identity = identityPattern(5);

    EXPECT_EQ(identity.numRows(), 5);
    EXPECT_EQ(identity.numCols(), 5);
    EXPECT_EQ(identity.getNnz(), 5);

    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_TRUE(identity.hasEntry(i, i));
        EXPECT_EQ(identity.getRowNnz(i), 1);
    }
}

TEST(SparsityOpsTest, FullPattern) {
    auto full = fullPattern(3, 4);

    EXPECT_EQ(full.numRows(), 3);
    EXPECT_EQ(full.numCols(), 4);
    EXPECT_EQ(full.getNnz(), 12);

    for (GlobalIndex i = 0; i < 3; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_TRUE(full.hasEntry(i, j));
        }
    }
}

TEST(SparsityOpsTest, TridiagonalPattern) {
    auto tri = tridiagonalPattern(5);

    EXPECT_EQ(tri.numRows(), 5);
    EXPECT_EQ(tri.numCols(), 5);
    EXPECT_EQ(tri.getNnz(), 13);  // 5 + 4 + 4 = 13

    // Check structure
    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_TRUE(tri.hasEntry(i, i));
        if (i > 0) {
            EXPECT_TRUE(tri.hasEntry(i, i-1));
        }
        if (i < 4) {
            EXPECT_TRUE(tri.hasEntry(i, i+1));
        }
    }

    // Bandwidth should be 1
    EXPECT_EQ(tri.computeBandwidth(), 1);
}

TEST(SparsityOpsTest, ArrowPattern) {
    auto arrow = arrowPattern(5);

    EXPECT_EQ(arrow.numRows(), 5);
    EXPECT_EQ(arrow.numCols(), 5);

    // First row dense
    for (GlobalIndex j = 0; j < 5; ++j) {
        EXPECT_TRUE(arrow.hasEntry(0, j));
    }

    // First column dense
    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_TRUE(arrow.hasEntry(i, 0));
    }

    // Diagonal
    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_TRUE(arrow.hasEntry(i, i));
    }
}

TEST(SparsityOpsTest, FilterToBand) {
    // Create full pattern then filter to band
    auto full = fullPattern(5, 5);
    auto band = filterToBand(full, 1, 2);

    EXPECT_EQ(band.numRows(), 5);

    // Check row 2: should have columns 1, 2, 3, 4
    EXPECT_TRUE(band.hasEntry(2, 1));
    EXPECT_TRUE(band.hasEntry(2, 2));
    EXPECT_TRUE(band.hasEntry(2, 3));
    EXPECT_TRUE(band.hasEntry(2, 4));
    EXPECT_FALSE(band.hasEntry(2, 0));
}

// ============================================================================
// Transpose Tests
// ============================================================================

TEST(SparsityOpsTest, Transpose) {
    SparsityPattern pattern = createTestPattern(3, {
        {0, 1}, {0, 2},
        {1, 2},
        {2, 0}
    });

    auto transposed = transposePattern(pattern);

    EXPECT_EQ(transposed.numRows(), 3);
    EXPECT_EQ(transposed.numCols(), 3);
    EXPECT_EQ(transposed.getNnz(), 4);

    EXPECT_TRUE(transposed.hasEntry(1, 0));
    EXPECT_TRUE(transposed.hasEntry(2, 0));
    EXPECT_TRUE(transposed.hasEntry(2, 1));
    EXPECT_TRUE(transposed.hasEntry(0, 2));
}

TEST(SparsityOpsTest, TransposeRectangular) {
    SparsityPattern pattern(3, 5);
    pattern.addEntry(0, 1);
    pattern.addEntry(0, 4);
    pattern.addEntry(1, 2);
    pattern.addEntry(2, 0);
    pattern.finalize();

    auto transposed = transposePattern(pattern);

    EXPECT_EQ(transposed.numRows(), 5);
    EXPECT_EQ(transposed.numCols(), 3);
    EXPECT_EQ(transposed.getNnz(), 4);
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST(SparsityOpsTest, PatternsEqual) {
    SparsityPattern a = createTestPattern(3, {{0, 0}, {1, 1}, {2, 2}});
    SparsityPattern b = createTestPattern(3, {{0, 0}, {1, 1}, {2, 2}});
    SparsityPattern c = createTestPattern(3, {{0, 0}, {1, 1}});

    EXPECT_TRUE(patternsEqual(a, b));
    EXPECT_FALSE(patternsEqual(a, c));
}

TEST(SparsityOpsTest, IsSubset) {
    SparsityPattern subset = createTestPattern(3, {{0, 0}, {1, 1}});
    SparsityPattern superset = createTestPattern(3, {{0, 0}, {1, 1}, {2, 2}});

    EXPECT_TRUE(isSubset(subset, superset));
    EXPECT_FALSE(isSubset(superset, subset));
}

// ============================================================================
// Counting Tests
// ============================================================================

TEST(SparsityOpsTest, CountDifference) {
    SparsityPattern a = createTestPattern(5, {{0, 0}, {1, 1}, {2, 2}});
    SparsityPattern b = createTestPattern(5, {{1, 1}, {2, 2}, {3, 3}});

    // countDifference counts entries in a not in b
    EXPECT_EQ(countDifference(a, b), 1);  // Only {0,0}
}

// ============================================================================
// Row/Column Operations Tests
// ============================================================================

TEST(SparsityOpsTest, RemoveEmptyRows) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    // Row 1 empty
    pattern.addEntry(2, 2);
    // Row 3 empty
    pattern.addEntry(4, 4);
    pattern.finalize();

    std::vector<GlobalIndex> row_map;
    auto result = removeEmptyRows(pattern, &row_map);

    EXPECT_EQ(result.numRows(), 3);
    EXPECT_EQ(result.getNnz(), 3);

    EXPECT_EQ(row_map[0], 0);
    EXPECT_EQ(row_map[2], 1);
    EXPECT_EQ(row_map[4], 2);
}

TEST(SparsityOpsTest, RemoveEmptyColumns) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 2);
    pattern.addEntry(2, 4);
    pattern.finalize();

    std::vector<GlobalIndex> col_map;
    auto result = removeEmptyColumns(pattern, &col_map);

    EXPECT_EQ(result.numCols(), 3);
    EXPECT_EQ(result.getNnz(), 3);
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST(SparsityOpsTest, DeterministicUnion) {
    auto create_and_union = []() {
        SparsityPattern a = createTestPattern(5, {{0, 0}, {1, 1}, {2, 2}});
        SparsityPattern b = createTestPattern(5, {{1, 1}, {3, 3}, {4, 4}});
        return patternUnion(a, b);
    };

    auto r1 = create_and_union();
    auto r2 = create_and_union();

    EXPECT_EQ(r1.getNnz(), r2.getNnz());

    auto ci1 = r1.getColIndices();
    auto ci2 = r2.getColIndices();
    EXPECT_TRUE(std::equal(ci1.begin(), ci1.end(), ci2.begin()));
}
