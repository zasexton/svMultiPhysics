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

#include "SparsityOps.h"
#include <algorithm>
#include <unordered_set>
#include <numeric>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Set Operations
// ============================================================================

void unionInPlace(SparsityPattern& target, const SparsityPattern& source) {
    FE_CHECK_ARG(target.numRows() == source.numRows() &&
                 target.numCols() == source.numCols(),
                 "Pattern dimensions must match for union");
    FE_CHECK_ARG(!target.isFinalized(),
                 "Target pattern must be in Building state for unionInPlace");

    for (GlobalIndex row = 0; row < source.numRows(); ++row) {
        if (source.isFinalized()) {
            target.addEntries(row, source.getRowSpan(row));
        } else {
            // Source is in building state - iterate differently
            for (GlobalIndex col = 0; col < source.numCols(); ++col) {
                if (source.hasEntry(row, col)) {
                    target.addEntry(row, col);
                }
            }
        }
    }
}

SparsityPattern intersect(const SparsityPattern& a, const SparsityPattern& b) {
    FE_CHECK_ARG(a.numRows() == b.numRows() && a.numCols() == b.numCols(),
                 "Pattern dimensions must match for intersection");

    SparsityPattern result(a.numRows(), a.numCols());

    // Both patterns should be finalized for efficient intersection
    const bool a_final = a.isFinalized();
    const bool b_final = b.isFinalized();

    for (GlobalIndex row = 0; row < a.numRows(); ++row) {
        if (a_final && b_final) {
            auto a_span = a.getRowSpan(row);
            auto b_span = b.getRowSpan(row);

            // Merge-style intersection of sorted arrays
            auto a_it = a_span.begin();
            auto b_it = b_span.begin();

            while (a_it != a_span.end() && b_it != b_span.end()) {
                if (*a_it < *b_it) {
                    ++a_it;
                } else if (*b_it < *a_it) {
                    ++b_it;
                } else {
                    result.addEntry(row, *a_it);
                    ++a_it;
                    ++b_it;
                }
            }
        } else {
            // Fallback for non-finalized patterns
            for (GlobalIndex col = 0; col < a.numCols(); ++col) {
                if (a.hasEntry(row, col) && b.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern difference(const SparsityPattern& a, const SparsityPattern& b) {
    FE_CHECK_ARG(a.numRows() == b.numRows() && a.numCols() == b.numCols(),
                 "Pattern dimensions must match for difference");

    SparsityPattern result(a.numRows(), a.numCols());

    for (GlobalIndex row = 0; row < a.numRows(); ++row) {
        if (a.isFinalized()) {
            for (GlobalIndex col : a.getRowSpan(row)) {
                if (!b.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        } else {
            for (GlobalIndex col = 0; col < a.numCols(); ++col) {
                if (a.hasEntry(row, col) && !b.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern symmetricDifference(const SparsityPattern& a, const SparsityPattern& b) {
    FE_CHECK_ARG(a.numRows() == b.numRows() && a.numCols() == b.numCols(),
                 "Pattern dimensions must match for symmetric difference");

    SparsityPattern result(a.numRows(), a.numCols());

    for (GlobalIndex row = 0; row < a.numRows(); ++row) {
        // Entries in a but not b
        if (a.isFinalized()) {
            for (GlobalIndex col : a.getRowSpan(row)) {
                if (!b.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }

        // Entries in b but not a
        if (b.isFinalized()) {
            for (GlobalIndex col : b.getRowSpan(row)) {
                if (!a.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

// ============================================================================
// Extraction Operations
// ============================================================================

SparsityPattern extractBlock(const SparsityPattern& pattern,
                             std::span<const GlobalIndex> row_indices,
                             std::span<const GlobalIndex> col_indices) {
    // Build column index map: old_col -> new_col (or -1 if not in set)
    std::vector<GlobalIndex> col_map(static_cast<std::size_t>(pattern.numCols()), -1);
    for (std::size_t i = 0; i < col_indices.size(); ++i) {
        GlobalIndex col = col_indices[i];
        if (col >= 0 && col < pattern.numCols()) {
            col_map[static_cast<std::size_t>(col)] = static_cast<GlobalIndex>(i);
        }
    }

    SparsityPattern result(static_cast<GlobalIndex>(row_indices.size()),
                          static_cast<GlobalIndex>(col_indices.size()));

    for (std::size_t new_row = 0; new_row < row_indices.size(); ++new_row) {
        GlobalIndex old_row = row_indices[new_row];
        if (old_row < 0 || old_row >= pattern.numRows()) continue;

        if (pattern.isFinalized()) {
            for (GlobalIndex old_col : pattern.getRowSpan(old_row)) {
                GlobalIndex new_col = col_map[static_cast<std::size_t>(old_col)];
                if (new_col >= 0) {
                    result.addEntry(static_cast<GlobalIndex>(new_row), new_col);
                }
            }
        } else {
            for (GlobalIndex old_col : col_indices) {
                if (pattern.hasEntry(old_row, old_col)) {
                    GlobalIndex new_col = col_map[static_cast<std::size_t>(old_col)];
                    if (new_col >= 0) {
                        result.addEntry(static_cast<GlobalIndex>(new_row), new_col);
                    }
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern extractRows(const SparsityPattern& pattern,
                            std::span<const GlobalIndex> row_indices) {
    SparsityPattern result(static_cast<GlobalIndex>(row_indices.size()), pattern.numCols());

    for (std::size_t new_row = 0; new_row < row_indices.size(); ++new_row) {
        GlobalIndex old_row = row_indices[new_row];
        if (old_row < 0 || old_row >= pattern.numRows()) continue;

        if (pattern.isFinalized()) {
            result.addEntries(static_cast<GlobalIndex>(new_row), pattern.getRowSpan(old_row));
        } else {
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(old_row, col)) {
                    result.addEntry(static_cast<GlobalIndex>(new_row), col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern extractColumns(const SparsityPattern& pattern,
                               std::span<const GlobalIndex> col_indices) {
    // Build column index map
    std::vector<GlobalIndex> col_map(static_cast<std::size_t>(pattern.numCols()), -1);
    for (std::size_t i = 0; i < col_indices.size(); ++i) {
        GlobalIndex col = col_indices[i];
        if (col >= 0 && col < pattern.numCols()) {
            col_map[static_cast<std::size_t>(col)] = static_cast<GlobalIndex>(i);
        }
    }

    SparsityPattern result(pattern.numRows(), static_cast<GlobalIndex>(col_indices.size()));

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex old_col : pattern.getRowSpan(row)) {
                GlobalIndex new_col = col_map[static_cast<std::size_t>(old_col)];
                if (new_col >= 0) {
                    result.addEntry(row, new_col);
                }
            }
        } else {
            for (GlobalIndex old_col : col_indices) {
                if (pattern.hasEntry(row, old_col)) {
                    GlobalIndex new_col = col_map[static_cast<std::size_t>(old_col)];
                    if (new_col >= 0) {
                        result.addEntry(row, new_col);
                    }
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern extractDiagonalBlock(const SparsityPattern& pattern,
                                      std::span<const GlobalIndex> indices) {
    return extractBlock(pattern, indices, indices);
}

SparsityPattern extractOffDiagonalBlock(const SparsityPattern& pattern,
                                         std::span<const GlobalIndex> row_indices,
                                         std::span<const GlobalIndex> col_indices) {
    return extractBlock(pattern, row_indices, col_indices);
}

// ============================================================================
// Permutation Operations
// ============================================================================

SparsityPattern permutePattern(const SparsityPattern& pattern,
                               std::span<const GlobalIndex> row_perm,
                               std::span<const GlobalIndex> col_perm) {
    FE_CHECK_ARG(static_cast<GlobalIndex>(row_perm.size()) == pattern.numRows(),
                 "Row permutation size mismatch");
    FE_CHECK_ARG(static_cast<GlobalIndex>(col_perm.size()) == pattern.numCols(),
                 "Column permutation size mismatch");

    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex old_row = 0; old_row < pattern.numRows(); ++old_row) {
        GlobalIndex new_row = row_perm[static_cast<std::size_t>(old_row)];
        FE_CHECK_ARG(new_row >= 0 && new_row < pattern.numRows(),
                     "Invalid row permutation value");

        if (pattern.isFinalized()) {
            for (GlobalIndex old_col : pattern.getRowSpan(old_row)) {
                GlobalIndex new_col = col_perm[static_cast<std::size_t>(old_col)];
                result.addEntry(new_row, new_col);
            }
        } else {
            for (GlobalIndex old_col = 0; old_col < pattern.numCols(); ++old_col) {
                if (pattern.hasEntry(old_row, old_col)) {
                    GlobalIndex new_col = col_perm[static_cast<std::size_t>(old_col)];
                    result.addEntry(new_row, new_col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern permuteSymmetric(const SparsityPattern& pattern,
                                  std::span<const GlobalIndex> perm) {
    FE_CHECK_ARG(pattern.isSquare(), "Symmetric permutation requires square pattern");
    return permutePattern(pattern, perm, perm);
}

SparsityPattern permuteInverse(const SparsityPattern& pattern,
                                std::span<const GlobalIndex> row_perm,
                                std::span<const GlobalIndex> col_perm) {
    auto inv_row = invertPermutation(row_perm);
    auto inv_col = invertPermutation(col_perm);
    return permutePattern(pattern, inv_row, inv_col);
}

std::vector<GlobalIndex> invertPermutation(std::span<const GlobalIndex> perm) {
    std::vector<GlobalIndex> inv(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        auto idx = static_cast<std::size_t>(perm[i]);
        FE_CHECK_ARG(idx < perm.size(), "Invalid permutation index");
        inv[idx] = static_cast<GlobalIndex>(i);
    }
    return inv;
}

bool isValidPermutation(std::span<const GlobalIndex> perm, GlobalIndex n) {
    if (static_cast<GlobalIndex>(perm.size()) != n) return false;

    std::vector<bool> seen(static_cast<std::size_t>(n), false);
    for (GlobalIndex val : perm) {
        if (val < 0 || val >= n) return false;
        if (seen[static_cast<std::size_t>(val)]) return false;
        seen[static_cast<std::size_t>(val)] = true;
    }
    return true;
}

// ============================================================================
// Transpose Operations
// ============================================================================

SparsityPattern transposePattern(const SparsityPattern& pattern) {
    SparsityPattern result(pattern.numCols(), pattern.numRows());

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex col : pattern.getRowSpan(row)) {
                result.addEntry(col, row);
            }
        } else {
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col)) {
                    result.addEntry(col, row);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern symmetrizePattern(const SparsityPattern& pattern) {
    FE_CHECK_ARG(pattern.isSquare(), "Can only symmetrize square patterns");

    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex col : pattern.getRowSpan(row)) {
                result.addEntry(row, col);
                result.addEntry(col, row);
            }
        } else {
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col)) {
                    result.addEntry(row, col);
                    result.addEntry(col, row);
                }
            }
        }
    }

    result.finalize();
    return result;
}

bool isStructurallySymmetric(const SparsityPattern& pattern) {
    if (!pattern.isSquare()) return false;
    return pattern.isSymmetric();
}

// ============================================================================
// Pattern Multiplication (Symbolic)
// ============================================================================

SparsityPattern multiplyPatterns(const SparsityPattern& a, const SparsityPattern& b) {
    FE_CHECK_ARG(a.numCols() == b.numRows(),
                 "Inner dimensions must match for pattern multiplication");

    SparsityPattern result(a.numRows(), b.numCols());

    // For each row of A
    for (GlobalIndex i = 0; i < a.numRows(); ++i) {
        std::vector<GlobalIndex> result_cols;

        // For each k where A[i,k] is non-zero
        if (a.isFinalized()) {
            for (GlobalIndex k : a.getRowSpan(i)) {
                // Add all j where B[k,j] is non-zero
                if (b.isFinalized()) {
                    for (GlobalIndex j : b.getRowSpan(k)) {
                        result_cols.push_back(j);
                    }
                } else {
                    for (GlobalIndex j = 0; j < b.numCols(); ++j) {
                        if (b.hasEntry(k, j)) {
                            result_cols.push_back(j);
                        }
                    }
                }
            }
        } else {
            for (GlobalIndex k = 0; k < a.numCols(); ++k) {
                if (a.hasEntry(i, k)) {
                    if (b.isFinalized()) {
                        for (GlobalIndex j : b.getRowSpan(k)) {
                            result_cols.push_back(j);
                        }
                    } else {
                        for (GlobalIndex j = 0; j < b.numCols(); ++j) {
                            if (b.hasEntry(k, j)) {
                                result_cols.push_back(j);
                            }
                        }
                    }
                }
            }
        }

        std::sort(result_cols.begin(), result_cols.end());
        result_cols.erase(std::unique(result_cols.begin(), result_cols.end()), result_cols.end());

        for (GlobalIndex j : result_cols) {
            result.addEntry(i, j);
        }
    }

    result.finalize();
    return result;
}

SparsityPattern computeAAt(const SparsityPattern& pattern) {
    auto at = transposePattern(pattern);
    return multiplyPatterns(pattern, at);
}

SparsityPattern computeAtA(const SparsityPattern& pattern) {
    auto at = transposePattern(pattern);
    return multiplyPatterns(at, pattern);
}

// ============================================================================
// Row/Column Operations
// ============================================================================

SparsityPattern removeEmptyRows(const SparsityPattern& pattern,
                                 std::vector<GlobalIndex>* row_map) {
    std::vector<GlobalIndex> non_empty_rows;
    non_empty_rows.reserve(static_cast<std::size_t>(pattern.numRows()));

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.getRowNnz(row) > 0) {
            non_empty_rows.push_back(row);
        }
    }

    if (row_map) {
        row_map->assign(static_cast<std::size_t>(pattern.numRows()), -1);
        for (std::size_t i = 0; i < non_empty_rows.size(); ++i) {
            (*row_map)[static_cast<std::size_t>(non_empty_rows[i])] =
                static_cast<GlobalIndex>(i);
        }
    }

    return extractRows(pattern, non_empty_rows);
}

SparsityPattern removeEmptyColumns(const SparsityPattern& pattern,
                                    std::vector<GlobalIndex>* col_map) {
    // Find non-empty columns
    std::vector<bool> col_has_entry(static_cast<std::size_t>(pattern.numCols()), false);

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex col : pattern.getRowSpan(row)) {
                col_has_entry[static_cast<std::size_t>(col)] = true;
            }
        } else {
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col)) {
                    col_has_entry[static_cast<std::size_t>(col)] = true;
                }
            }
        }
    }

    std::vector<GlobalIndex> non_empty_cols;
    for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
        if (col_has_entry[static_cast<std::size_t>(col)]) {
            non_empty_cols.push_back(col);
        }
    }

    if (col_map) {
        col_map->assign(static_cast<std::size_t>(pattern.numCols()), -1);
        for (std::size_t i = 0; i < non_empty_cols.size(); ++i) {
            (*col_map)[static_cast<std::size_t>(non_empty_cols[i])] =
                static_cast<GlobalIndex>(i);
        }
    }

    return extractColumns(pattern, non_empty_cols);
}

SparsityPattern addDiagonal(const SparsityPattern& pattern) {
    FE_CHECK_ARG(pattern.isSquare(), "addDiagonal requires square pattern");

    SparsityPattern result(pattern.numRows(), pattern.numCols());

    // Copy existing entries
    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            result.addEntries(row, pattern.getRowSpan(row));
        } else {
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
        // Add diagonal
        result.addEntry(row, row);
    }

    result.finalize();
    return result;
}

SparsityPattern removeDiagonal(const SparsityPattern& pattern) {
    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex col : pattern.getRowSpan(row)) {
                if (col != row) {
                    result.addEntry(row, col);
                }
            }
        } else {
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (col != row && pattern.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern extractDiagonal(const SparsityPattern& pattern) {
    FE_CHECK_ARG(pattern.isSquare(), "extractDiagonal requires square pattern");

    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex i = 0; i < pattern.numRows(); ++i) {
        if (pattern.hasEntry(i, i)) {
            result.addEntry(i, i);
        }
    }

    result.finalize();
    return result;
}

SparsityPattern lowerTriangular(const SparsityPattern& pattern) {
    FE_CHECK_ARG(pattern.isSquare(), "lowerTriangular requires square pattern");

    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex col : pattern.getRowSpan(row)) {
                if (col <= row) {
                    result.addEntry(row, col);
                }
            }
        } else {
            for (GlobalIndex col = 0; col <= row; ++col) {
                if (pattern.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern upperTriangular(const SparsityPattern& pattern) {
    FE_CHECK_ARG(pattern.isSquare(), "upperTriangular requires square pattern");

    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex col : pattern.getRowSpan(row)) {
                if (col >= row) {
                    result.addEntry(row, col);
                }
            }
        } else {
            for (GlobalIndex col = row; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern strictLowerTriangular(const SparsityPattern& pattern) {
    FE_CHECK_ARG(pattern.isSquare(), "strictLowerTriangular requires square pattern");

    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex col : pattern.getRowSpan(row)) {
                if (col < row) {
                    result.addEntry(row, col);
                }
            }
        } else {
            for (GlobalIndex col = 0; col < row; ++col) {
                if (pattern.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern strictUpperTriangular(const SparsityPattern& pattern) {
    FE_CHECK_ARG(pattern.isSquare(), "strictUpperTriangular requires square pattern");

    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex col : pattern.getRowSpan(row)) {
                if (col > row) {
                    result.addEntry(row, col);
                }
            }
        } else {
            for (GlobalIndex col = row + 1; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

// ============================================================================
// Filtering Operations
// ============================================================================

SparsityPattern filterEntries(const SparsityPattern& pattern,
                               std::function<bool(GlobalIndex, GlobalIndex)> predicate) {
    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            for (GlobalIndex col : pattern.getRowSpan(row)) {
                if (predicate(row, col)) {
                    result.addEntry(row, col);
                }
            }
        } else {
            for (GlobalIndex col = 0; col < pattern.numCols(); ++col) {
                if (pattern.hasEntry(row, col) && predicate(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern filterToBand(const SparsityPattern& pattern,
                              GlobalIndex lower_bandwidth,
                              GlobalIndex upper_bandwidth) {
    FE_CHECK_ARG(lower_bandwidth >= 0, "Lower bandwidth must be non-negative");
    FE_CHECK_ARG(upper_bandwidth >= 0, "Upper bandwidth must be non-negative");

    return filterEntries(pattern, [=](GlobalIndex row, GlobalIndex col) {
        return (col >= row - lower_bandwidth) && (col <= row + upper_bandwidth);
    });
}

// ============================================================================
// Comparison Operations
// ============================================================================

bool patternsEqual(const SparsityPattern& a, const SparsityPattern& b) {
    if (a.numRows() != b.numRows() || a.numCols() != b.numCols()) {
        return false;
    }
    if (a.getNnz() != b.getNnz()) {
        return false;
    }

    for (GlobalIndex row = 0; row < a.numRows(); ++row) {
        if (a.getRowNnz(row) != b.getRowNnz(row)) {
            return false;
        }

        if (a.isFinalized() && b.isFinalized()) {
            auto a_span = a.getRowSpan(row);
            auto b_span = b.getRowSpan(row);
            if (!std::equal(a_span.begin(), a_span.end(), b_span.begin())) {
                return false;
            }
        } else {
            for (GlobalIndex col = 0; col < a.numCols(); ++col) {
                if (a.hasEntry(row, col) != b.hasEntry(row, col)) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool isSubset(const SparsityPattern& a, const SparsityPattern& b) {
    if (a.numRows() != b.numRows() || a.numCols() != b.numCols()) {
        return false;
    }

    for (GlobalIndex row = 0; row < a.numRows(); ++row) {
        if (a.isFinalized()) {
            for (GlobalIndex col : a.getRowSpan(row)) {
                if (!b.hasEntry(row, col)) {
                    return false;
                }
            }
        } else {
            for (GlobalIndex col = 0; col < a.numCols(); ++col) {
                if (a.hasEntry(row, col) && !b.hasEntry(row, col)) {
                    return false;
                }
            }
        }
    }

    return true;
}

GlobalIndex countDifference(const SparsityPattern& a, const SparsityPattern& b) {
    FE_CHECK_ARG(a.numRows() == b.numRows() && a.numCols() == b.numCols(),
                 "Pattern dimensions must match");

    GlobalIndex count = 0;

    for (GlobalIndex row = 0; row < a.numRows(); ++row) {
        if (a.isFinalized()) {
            for (GlobalIndex col : a.getRowSpan(row)) {
                if (!b.hasEntry(row, col)) {
                    ++count;
                }
            }
        } else {
            for (GlobalIndex col = 0; col < a.numCols(); ++col) {
                if (a.hasEntry(row, col) && !b.hasEntry(row, col)) {
                    ++count;
                }
            }
        }
    }

    return count;
}

// ============================================================================
// Utility Functions
// ============================================================================

SparsityPattern identityPattern(GlobalIndex n) {
    SparsityPattern result(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        result.addEntry(i, i);
    }
    result.finalize();
    return result;
}

SparsityPattern fullPattern(GlobalIndex n_rows, GlobalIndex n_cols) {
    SparsityPattern result(n_rows, n_cols);
    for (GlobalIndex row = 0; row < n_rows; ++row) {
        for (GlobalIndex col = 0; col < n_cols; ++col) {
            result.addEntry(row, col);
        }
    }
    result.finalize();
    return result;
}

SparsityPattern tridiagonalPattern(GlobalIndex n) {
    SparsityPattern result(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        if (i > 0) result.addEntry(i, i - 1);
        result.addEntry(i, i);
        if (i < n - 1) result.addEntry(i, i + 1);
    }
    result.finalize();
    return result;
}

SparsityPattern blockDiagonalPattern(std::span<const GlobalIndex> block_sizes) {
    GlobalIndex total = 0;
    for (GlobalIndex size : block_sizes) {
        total += size;
    }

    SparsityPattern result(total, total);

    GlobalIndex offset = 0;
    for (GlobalIndex block_size : block_sizes) {
        for (GlobalIndex i = 0; i < block_size; ++i) {
            for (GlobalIndex j = 0; j < block_size; ++j) {
                result.addEntry(offset + i, offset + j);
            }
        }
        offset += block_size;
    }

    result.finalize();
    return result;
}

SparsityPattern arrowPattern(GlobalIndex n) {
    SparsityPattern result(n, n);

    // First row is dense
    for (GlobalIndex j = 0; j < n; ++j) {
        result.addEntry(0, j);
    }

    // First column is dense
    for (GlobalIndex i = 1; i < n; ++i) {
        result.addEntry(i, 0);
    }

    // Diagonal
    for (GlobalIndex i = 1; i < n; ++i) {
        result.addEntry(i, i);
    }

    result.finalize();
    return result;
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
