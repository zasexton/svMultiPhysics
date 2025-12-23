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

#ifndef SVMP_FE_SPARSITY_OPS_H
#define SVMP_FE_SPARSITY_OPS_H

/**
 * @file SparsityOps.h
 * @brief Pattern algebra and manipulation operations for sparsity patterns
 *
 * This header provides functions for manipulating sparsity patterns:
 * - Set operations: union, intersection, difference
 * - In-place modifications
 * - Extraction of sub-blocks
 * - Permutation operations
 * - Structural transpose
 * - Pattern multiplication (symbolic only)
 *
 * All operations preserve determinism through sorted column indices.
 *
 * Complexity notes:
 * - unionInPlace: O(NNZ_A + NNZ_B)
 * - intersect: O(min(NNZ_A, NNZ_B))
 * - extract: O(NNZ) with index mapping overhead
 * - permute: O(NNZ + n_rows * log(n_cols))
 * - transpose: O(NNZ)
 * - multiply (symbolic): O(n_rows * avg_nnz_A * avg_nnz_B)
 *
 * @see SparsityPattern for the pattern representation
 */

#include "SparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <functional>
#include <optional>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Set Operations
// ============================================================================

/**
 * @brief Add entries from source pattern to target (union in-place)
 *
 * @param target Pattern to modify (must be in Building state)
 * @param source Pattern to add from (may be in any state)
 * @throws FEException if dimensions don't match or target is finalized
 *
 * Complexity: O(NNZ_source * log(row_nnz))
 *
 * @note This modifies target in-place. For a new pattern, use patternUnion().
 */
void unionInPlace(SparsityPattern& target, const SparsityPattern& source);

/**
 * @brief Compute intersection of two patterns
 *
 * @param a First pattern
 * @param b Second pattern
 * @return New pattern containing only entries in both a and b
 * @throws FEException if dimensions don't match
 *
 * Complexity: O(NNZ_a + NNZ_b) with merge-style iteration
 */
[[nodiscard]] SparsityPattern intersect(const SparsityPattern& a,
                                        const SparsityPattern& b);

/**
 * @brief Compute difference a - b (entries in a but not in b)
 *
 * @param a First pattern
 * @param b Second pattern
 * @return New pattern with entries in a that are not in b
 * @throws FEException if dimensions don't match
 *
 * Complexity: O(NNZ_a + NNZ_b)
 */
[[nodiscard]] SparsityPattern difference(const SparsityPattern& a,
                                         const SparsityPattern& b);

/**
 * @brief Compute symmetric difference (entries in exactly one of a or b)
 *
 * @param a First pattern
 * @param b Second pattern
 * @return New pattern with entries in a XOR b
 * @throws FEException if dimensions don't match
 *
 * Complexity: O(NNZ_a + NNZ_b)
 */
[[nodiscard]] SparsityPattern symmetricDifference(const SparsityPattern& a,
                                                   const SparsityPattern& b);

// ============================================================================
// Extraction Operations
// ============================================================================

/**
 * @brief Extract sub-block of pattern
 *
 * @param pattern Source pattern
 * @param row_indices Rows to extract (must be sorted, unique)
 * @param col_indices Columns to extract (must be sorted, unique)
 * @return New pattern with dimensions (row_indices.size() x col_indices.size())
 *
 * The result contains (i, j) if pattern contains (row_indices[i], col_indices[j]).
 * Indices in the result are 0-based.
 *
 * Complexity: O(NNZ) with O(n_rows + n_cols) mapping overhead
 */
[[nodiscard]] SparsityPattern extractBlock(
    const SparsityPattern& pattern,
    std::span<const GlobalIndex> row_indices,
    std::span<const GlobalIndex> col_indices);

/**
 * @brief Extract rows from pattern
 *
 * @param pattern Source pattern
 * @param row_indices Rows to extract (must be sorted, unique)
 * @return New pattern with all columns, selected rows
 *
 * Complexity: O(sum of row NNZ for selected rows)
 */
[[nodiscard]] SparsityPattern extractRows(
    const SparsityPattern& pattern,
    std::span<const GlobalIndex> row_indices);

/**
 * @brief Extract columns from pattern
 *
 * @param pattern Source pattern
 * @param col_indices Columns to extract (must be sorted, unique)
 * @return New pattern with all rows, selected columns
 *
 * Complexity: O(NNZ) - must scan all entries
 */
[[nodiscard]] SparsityPattern extractColumns(
    const SparsityPattern& pattern,
    std::span<const GlobalIndex> col_indices);

/**
 * @brief Extract diagonal block (for square patterns)
 *
 * @param pattern Source pattern (must be square)
 * @param indices Indices for both rows and columns
 * @return Square sub-pattern
 *
 * Equivalent to extractBlock(pattern, indices, indices).
 */
[[nodiscard]] SparsityPattern extractDiagonalBlock(
    const SparsityPattern& pattern,
    std::span<const GlobalIndex> indices);

/**
 * @brief Extract off-diagonal block
 *
 * @param pattern Source pattern
 * @param row_indices Rows to extract
 * @param col_indices Columns to extract (should not overlap with row_indices for square)
 * @return Rectangular sub-pattern
 */
[[nodiscard]] SparsityPattern extractOffDiagonalBlock(
    const SparsityPattern& pattern,
    std::span<const GlobalIndex> row_indices,
    std::span<const GlobalIndex> col_indices);

// ============================================================================
// Permutation Operations
// ============================================================================

/**
 * @brief Apply row and column permutations
 *
 * @param pattern Source pattern
 * @param row_perm Row permutation: new_row = row_perm[old_row]
 * @param col_perm Column permutation: new_col = col_perm[old_col]
 * @return New pattern with permuted indices
 *
 * Complexity: O(NNZ + n_rows * log(row_nnz)) for sorting after permutation
 *
 * @note Permutation vectors must be valid bijections on [0, n).
 */
[[nodiscard]] SparsityPattern permutePattern(
    const SparsityPattern& pattern,
    std::span<const GlobalIndex> row_perm,
    std::span<const GlobalIndex> col_perm);

/**
 * @brief Apply symmetric permutation (same for rows and columns)
 *
 * @param pattern Source pattern (must be square)
 * @param perm Permutation vector
 * @return New pattern with symmetric permutation applied
 */
[[nodiscard]] SparsityPattern permuteSymmetric(
    const SparsityPattern& pattern,
    std::span<const GlobalIndex> perm);

/**
 * @brief Apply inverse permutation
 *
 * @param pattern Source pattern
 * @param row_perm Row permutation to invert and apply
 * @param col_perm Column permutation to invert and apply
 * @return New pattern with inverse permutations applied
 */
[[nodiscard]] SparsityPattern permuteInverse(
    const SparsityPattern& pattern,
    std::span<const GlobalIndex> row_perm,
    std::span<const GlobalIndex> col_perm);

/**
 * @brief Compute inverse permutation vector
 *
 * @param perm Permutation vector
 * @return Inverse permutation: inv[perm[i]] = i
 */
[[nodiscard]] std::vector<GlobalIndex> invertPermutation(
    std::span<const GlobalIndex> perm);

/**
 * @brief Check if vector is a valid permutation
 *
 * @param perm Potential permutation
 * @param n Expected size
 * @return true if perm is a bijection on [0, n)
 */
[[nodiscard]] bool isValidPermutation(std::span<const GlobalIndex> perm, GlobalIndex n);

// ============================================================================
// Transpose Operations
// ============================================================================

/**
 * @brief Compute structural transpose
 *
 * @param pattern Source pattern
 * @return Transposed pattern (n_cols x n_rows)
 *
 * Complexity: O(NNZ)
 */
[[nodiscard]] SparsityPattern transposePattern(const SparsityPattern& pattern);

/**
 * @brief Compute transpose and add to original (A + A^T)
 *
 * @param pattern Source pattern (must be square)
 * @return Symmetrized pattern
 *
 * Useful for making patterns structurally symmetric.
 */
[[nodiscard]] SparsityPattern symmetrizePattern(const SparsityPattern& pattern);

/**
 * @brief Check if pattern equals its transpose (structurally symmetric)
 *
 * @param pattern Pattern to check
 * @return true if pattern is structurally symmetric
 */
[[nodiscard]] bool isStructurallySymmetric(const SparsityPattern& pattern);

// ============================================================================
// Pattern Multiplication (Symbolic)
// ============================================================================

/**
 * @brief Compute symbolic product pattern C = A * B
 *
 * @param a Left pattern (m x k)
 * @param b Right pattern (k x n)
 * @return Product pattern (m x n)
 * @throws FEException if inner dimensions don't match
 *
 * C[i,j] is non-zero if there exists some k such that A[i,k] and B[k,j] are non-zero.
 *
 * Complexity: O(n_rows_A * avg_nnz_A * avg_nnz_B)
 *
 * @note This computes the structural pattern only, not actual matrix values.
 */
[[nodiscard]] SparsityPattern multiplyPatterns(const SparsityPattern& a,
                                                const SparsityPattern& b);

/**
 * @brief Compute A * A^T pattern
 *
 * @param pattern Input pattern (m x n)
 * @return Square pattern (m x m)
 *
 * Useful for normal equations pattern: (A^T A)^{-1} A^T b
 */
[[nodiscard]] SparsityPattern computeAAt(const SparsityPattern& pattern);

/**
 * @brief Compute A^T * A pattern
 *
 * @param pattern Input pattern (m x n)
 * @return Square pattern (n x n)
 */
[[nodiscard]] SparsityPattern computeAtA(const SparsityPattern& pattern);

// ============================================================================
// Row/Column Operations
// ============================================================================

/**
 * @brief Remove empty rows from pattern
 *
 * @param pattern Source pattern
 * @param[out] row_map Optional output: old_row -> new_row (-1 if removed)
 * @return Compressed pattern without empty rows
 */
[[nodiscard]] SparsityPattern removeEmptyRows(
    const SparsityPattern& pattern,
    std::vector<GlobalIndex>* row_map = nullptr);

/**
 * @brief Remove empty columns from pattern
 *
 * @param pattern Source pattern
 * @param[out] col_map Optional output: old_col -> new_col (-1 if removed)
 * @return Compressed pattern without empty columns
 */
[[nodiscard]] SparsityPattern removeEmptyColumns(
    const SparsityPattern& pattern,
    std::vector<GlobalIndex>* col_map = nullptr);

/**
 * @brief Add diagonal entries to all rows
 *
 * @param pattern Source pattern (must be square)
 * @return Pattern with all diagonal entries present
 */
[[nodiscard]] SparsityPattern addDiagonal(const SparsityPattern& pattern);

/**
 * @brief Remove diagonal entries
 *
 * @param pattern Source pattern
 * @return Pattern with diagonal entries removed
 */
[[nodiscard]] SparsityPattern removeDiagonal(const SparsityPattern& pattern);

/**
 * @brief Keep only diagonal entries
 *
 * @param pattern Source pattern (must be square)
 * @return Diagonal-only pattern
 */
[[nodiscard]] SparsityPattern extractDiagonal(const SparsityPattern& pattern);

/**
 * @brief Keep only lower triangular part (including diagonal)
 *
 * @param pattern Source pattern (must be square)
 * @return Lower triangular pattern
 */
[[nodiscard]] SparsityPattern lowerTriangular(const SparsityPattern& pattern);

/**
 * @brief Keep only upper triangular part (including diagonal)
 *
 * @param pattern Source pattern (must be square)
 * @return Upper triangular pattern
 */
[[nodiscard]] SparsityPattern upperTriangular(const SparsityPattern& pattern);

/**
 * @brief Keep only strictly lower triangular part (excluding diagonal)
 *
 * @param pattern Source pattern (must be square)
 * @return Strictly lower triangular pattern
 */
[[nodiscard]] SparsityPattern strictLowerTriangular(const SparsityPattern& pattern);

/**
 * @brief Keep only strictly upper triangular part (excluding diagonal)
 *
 * @param pattern Source pattern (must be square)
 * @return Strictly upper triangular pattern
 */
[[nodiscard]] SparsityPattern strictUpperTriangular(const SparsityPattern& pattern);

// ============================================================================
// Filtering Operations
// ============================================================================

/**
 * @brief Filter entries by predicate
 *
 * @param pattern Source pattern
 * @param predicate Function returning true for entries to keep
 * @return Filtered pattern
 */
[[nodiscard]] SparsityPattern filterEntries(
    const SparsityPattern& pattern,
    std::function<bool(GlobalIndex row, GlobalIndex col)> predicate);

/**
 * @brief Filter to band pattern
 *
 * @param pattern Source pattern
 * @param lower_bandwidth Lower bandwidth (>=0)
 * @param upper_bandwidth Upper bandwidth (>=0)
 * @return Band pattern: keep (i,j) where i - lower_bw <= j <= i + upper_bw
 */
[[nodiscard]] SparsityPattern filterToBand(
    const SparsityPattern& pattern,
    GlobalIndex lower_bandwidth,
    GlobalIndex upper_bandwidth);

// ============================================================================
// Comparison Operations
// ============================================================================

/**
 * @brief Check if two patterns are identical
 *
 * @param a First pattern
 * @param b Second pattern
 * @return true if patterns have same dimensions and entries
 */
[[nodiscard]] bool patternsEqual(const SparsityPattern& a, const SparsityPattern& b);

/**
 * @brief Check if a is a subset of b
 *
 * @param a First pattern
 * @param b Second pattern
 * @return true if every entry in a is also in b
 */
[[nodiscard]] bool isSubset(const SparsityPattern& a, const SparsityPattern& b);

/**
 * @brief Count entries in a that are not in b
 *
 * @param a First pattern
 * @param b Second pattern
 * @return Number of entries in a not present in b
 */
[[nodiscard]] GlobalIndex countDifference(const SparsityPattern& a, const SparsityPattern& b);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Create identity pattern
 *
 * @param n Size of square pattern
 * @return Diagonal pattern with all diagonal entries
 */
[[nodiscard]] SparsityPattern identityPattern(GlobalIndex n);

/**
 * @brief Create full/dense pattern
 *
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @return Pattern with all (n_rows * n_cols) entries
 *
 * @warning Memory usage is O(n_rows * n_cols)!
 */
[[nodiscard]] SparsityPattern fullPattern(GlobalIndex n_rows, GlobalIndex n_cols);

/**
 * @brief Create tridiagonal pattern
 *
 * @param n Size of square pattern
 * @return Tridiagonal pattern (bandwidth 1)
 */
[[nodiscard]] SparsityPattern tridiagonalPattern(GlobalIndex n);

/**
 * @brief Create block diagonal pattern from block sizes
 *
 * @param block_sizes Size of each diagonal block
 * @return Block diagonal pattern
 */
[[nodiscard]] SparsityPattern blockDiagonalPattern(std::span<const GlobalIndex> block_sizes);

/**
 * @brief Create arrow pattern (dense first row/col + diagonal)
 *
 * @param n Size of square pattern
 * @return Arrow pattern
 */
[[nodiscard]] SparsityPattern arrowPattern(GlobalIndex n);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_OPS_H
