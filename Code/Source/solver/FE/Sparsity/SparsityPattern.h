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

#ifndef SVMP_FE_SPARSITY_SPARSITY_PATTERN_H
#define SVMP_FE_SPARSITY_SPARSITY_PATTERN_H

/**
 * @file SparsityPattern.h
 * @brief Core sparsity pattern representation for FEM matrices
 *
 * This header provides the fundamental SparsityPattern class for representing
 * the non-zero structure of sparse matrices. Key features:
 *
 * - Rectangular support: (nRows x nCols) dimensions, nRows != nCols is valid
 * - Two-phase lifecycle: Building (insertion) -> Finalized (compressed CSR)
 * - Deterministic finalization: sorted column indices for reproducibility
 * - Thread-safe read access after finalization
 * - Memory-efficient: set-based building, CSR storage after compression
 *
 * Complexity notes:
 * - addEntry(): O(log n) per row during building (set insertion)
 * - finalize(): O(NNZ * log(NNZ/nRows)) for sorting
 * - hasEntry(): O(log row_nnz) after finalization (binary search)
 * - getRowIndices(): O(1) access to column indices
 *
 * @see DistributedSparsityPattern for MPI-parallel extension
 * @see SparsityBuilder for mesh-based construction
 */

#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <set>
#include <span>
#include <cstdint>
#include <atomic>
#include <string>
#include <algorithm>
#include <functional>

namespace svmp {
namespace FE {
namespace sparsity {

/**
 * @brief Lifecycle state for SparsityPattern
 *
 * The pattern follows a build -> finalize -> use lifecycle:
 * - Building: Mutable state, entries can be added
 * - Finalized: Immutable CSR format, thread-safe access
 */
enum class SparsityState : std::uint8_t {
    Building,    ///< Mutable state - can add entries
    Finalized    ///< Immutable CSR format - thread-safe reads
};

/**
 * @brief Statistics about a sparsity pattern
 */
struct SparsityStats {
    GlobalIndex n_rows{0};           ///< Number of rows
    GlobalIndex n_cols{0};           ///< Number of columns
    GlobalIndex nnz{0};              ///< Total non-zeros
    GlobalIndex empty_rows{0};       ///< Number of empty rows
    GlobalIndex min_row_nnz{0};      ///< Minimum NNZ in any row
    GlobalIndex max_row_nnz{0};      ///< Maximum NNZ in any row
    double avg_row_nnz{0.0};         ///< Average NNZ per row
    double fill_ratio{0.0};          ///< NNZ / (nRows * nCols)
    GlobalIndex bandwidth{0};        ///< Max |row - col| for any entry
    bool has_diagonal{false};        ///< All diagonal entries present (square only)
    bool is_symmetric_structure{false}; ///< Structural symmetry (A[i,j] implies A[j,i])
};

/**
 * @brief Row iterator for accessing column indices in a finalized pattern
 *
 * Provides random access iteration over the column indices of a single row.
 */
class RowIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = GlobalIndex;
    using difference_type = std::ptrdiff_t;
    using pointer = const GlobalIndex*;
    using reference = const GlobalIndex&;

    RowIterator() = default;
    explicit RowIterator(pointer ptr) : ptr_(ptr) {}

    reference operator*() const { return *ptr_; }
    pointer operator->() const { return ptr_; }

    RowIterator& operator++() { ++ptr_; return *this; }
    RowIterator operator++(int) { RowIterator tmp = *this; ++ptr_; return tmp; }
    RowIterator& operator--() { --ptr_; return *this; }
    RowIterator operator--(int) { RowIterator tmp = *this; --ptr_; return tmp; }

    RowIterator& operator+=(difference_type n) { ptr_ += n; return *this; }
    RowIterator& operator-=(difference_type n) { ptr_ -= n; return *this; }

    RowIterator operator+(difference_type n) const { return RowIterator(ptr_ + n); }
    RowIterator operator-(difference_type n) const { return RowIterator(ptr_ - n); }
    difference_type operator-(const RowIterator& other) const { return ptr_ - other.ptr_; }

    reference operator[](difference_type n) const { return ptr_[n]; }

    bool operator==(const RowIterator& other) const { return ptr_ == other.ptr_; }
    bool operator!=(const RowIterator& other) const { return ptr_ != other.ptr_; }
    bool operator<(const RowIterator& other) const { return ptr_ < other.ptr_; }
    bool operator>(const RowIterator& other) const { return ptr_ > other.ptr_; }
    bool operator<=(const RowIterator& other) const { return ptr_ <= other.ptr_; }
    bool operator>=(const RowIterator& other) const { return ptr_ >= other.ptr_; }

private:
    pointer ptr_{nullptr};
};

/**
 * @brief View of a single row in the sparsity pattern
 *
 * Provides begin/end iterators and size for range-based access.
 */
class RowView {
public:
    RowView() = default;
    RowView(const GlobalIndex* begin, const GlobalIndex* end)
        : begin_(begin), end_(end) {}

    [[nodiscard]] RowIterator begin() const { return RowIterator(begin_); }
    [[nodiscard]] RowIterator end() const { return RowIterator(end_); }
    [[nodiscard]] GlobalIndex size() const { return static_cast<GlobalIndex>(end_ - begin_); }
    [[nodiscard]] bool empty() const { return begin_ == end_; }
    [[nodiscard]] const GlobalIndex& operator[](GlobalIndex i) const { return begin_[i]; }

    [[nodiscard]] const GlobalIndex* data() const { return begin_; }

private:
    const GlobalIndex* begin_{nullptr};
    const GlobalIndex* end_{nullptr};
};

/**
 * @brief Core sparsity pattern representation
 *
 * Represents the structural non-zero pattern of a sparse matrix in CSR format
 * after finalization. Supports rectangular matrices (nRows != nCols).
 *
 * Usage pattern:
 * @code
 * SparsityPattern pattern(nRows, nCols);
 * // Insert entries during building phase
 * pattern.addEntry(0, 0);
 * pattern.addEntry(0, 1);
 * pattern.addEntry(1, 1);
 * // Compress to CSR format
 * pattern.finalize();
 * // Now read-only access
 * auto row0 = pattern.getRowIndices(0);
 * for (GlobalIndex col : row0) { ... }
 * @endcode
 *
 * @note After finalization, the pattern is immutable and thread-safe for reads.
 * @note For very large patterns, consider SparsityTwoPassBuilder to avoid
 *       set-of-sets memory overhead during building.
 */
class SparsityPattern {
public:
    // =========================================================================
    // Construction and lifecycle
    // =========================================================================

    /**
     * @brief Default constructor - creates empty pattern
     *
     * Creates an empty pattern in Building state. Use resize() to set dimensions.
     */
    SparsityPattern() = default;

    /**
     * @brief Construct pattern with given dimensions
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns (defaults to n_rows for square matrices)
     *
     * Complexity: O(n_rows) for set allocation
     * Memory: O(n_rows) initially, grows with entries
     */
    explicit SparsityPattern(GlobalIndex n_rows, GlobalIndex n_cols = -1);

    /// Move constructor
    SparsityPattern(SparsityPattern&& other) noexcept;

    /// Move assignment
    SparsityPattern& operator=(SparsityPattern&& other) noexcept;

    /// Copy constructor (deep copy, resets to Building state)
    SparsityPattern(const SparsityPattern& other);

    /// Copy assignment (deep copy, resets to Building state)
    SparsityPattern& operator=(const SparsityPattern& other);

    /// Destructor
    ~SparsityPattern() = default;

    /**
     * @brief Clone the pattern preserving Finalized state when possible
     *
     * Unlike the copy constructor (which always resets to Building state),
     * this method returns a pattern in Finalized state when the source is
     * finalized by directly copying CSR storage. If the source is in Building
     * state, the returned pattern is finalized via finalize().
     *
     * This is useful for caching and sharing immutable patterns efficiently.
     */
    [[nodiscard]] SparsityPattern cloneFinalized() const;

    // =========================================================================
    // Setup methods (Building state only)
    // =========================================================================

    /**
     * @brief Resize the pattern dimensions
     *
     * @param n_rows New number of rows
     * @param n_cols New number of columns (defaults to n_rows)
     * @throws FEException if already finalized
     *
     * Note: Resizing clears all existing entries.
     * Complexity: O(n_rows)
     */
    void resize(GlobalIndex n_rows, GlobalIndex n_cols = -1);

    /**
     * @brief Clear all entries, return to empty Building state
     *
     * Keeps current dimensions but removes all entries.
     * If finalized, returns to Building state.
     */
    void clear();

    /**
     * @brief Reserve approximate capacity per row
     *
     * @param entries_per_row Expected average entries per row
     *
     * Hint for memory preallocation during building phase.
     * Has no effect if already finalized.
     */
    void reservePerRow(GlobalIndex entries_per_row);

    /**
     * @brief Add a single entry to the pattern
     *
     * @param row Row index (0-based)
     * @param col Column index (0-based)
     * @throws FEException if already finalized or indices out of range
     *
     * Complexity: O(log row_nnz) for set insertion
     *
     * @note Duplicate entries are automatically ignored (set semantics).
     */
    void addEntry(GlobalIndex row, GlobalIndex col);

    /**
     * @brief Add multiple entries to a single row
     *
     * @param row Row index
     * @param cols Column indices to add
     * @throws FEException if already finalized or indices out of range
     *
     * More efficient than individual addEntry() calls for bulk insertion.
     * Complexity: O(n * log row_nnz) where n = cols.size()
     */
    void addEntries(GlobalIndex row, std::span<const GlobalIndex> cols);

    /**
     * @brief Add a dense block of entries
     *
     * @param row_begin First row of block
     * @param row_end One past last row of block
     * @param col_begin First column of block
     * @param col_end One past last column of block
     * @throws FEException if already finalized or indices out of range
     *
     * Adds all (row, col) pairs in [row_begin, row_end) x [col_begin, col_end).
     * Useful for element-wise assembly patterns.
     */
    void addBlock(GlobalIndex row_begin, GlobalIndex row_end,
                  GlobalIndex col_begin, GlobalIndex col_end);

    /**
     * @brief Add couplings for an element's DOFs
     *
     * @param dofs Array of DOF indices (both row and column indices)
     *
     * Creates all (dof_i, dof_j) couplings for the given DOFs.
     * This is the typical pattern for single-field FEM assembly.
     * Complexity: O(n^2 * log n) where n = dofs.size()
     */
    void addElementCouplings(std::span<const GlobalIndex> dofs);

    /**
     * @brief Add couplings between two sets of DOFs (rectangular)
     *
     * @param row_dofs Row DOF indices
     * @param col_dofs Column DOF indices
     *
     * Creates all (row_dof, col_dof) couplings.
     * Useful for mixed/multi-field problems or rectangular operators.
     */
    void addElementCouplings(std::span<const GlobalIndex> row_dofs,
                             std::span<const GlobalIndex> col_dofs);

    /**
     * @brief Ensure diagonal entries exist (square patterns only)
     *
     * @throws FEException if pattern is not square
     *
     * Adds (i, i) for all rows i that don't have a diagonal entry.
     * Important for iterative solvers and preconditioning.
     */
    void ensureDiagonal();

    /**
     * @brief Ensure no rows are empty
     *
     * Rows without any entries get a diagonal entry (if square) or
     * entry at column 0 (if rectangular).
     *
     * Important for handling Dirichlet elimination patterns where
     * some rows may have been cleared.
     */
    void ensureNonEmptyRows();

    /**
     * @brief Finalize the pattern - compress to CSR format
     *
     * After finalization:
     * - All setup methods will throw
     * - Read methods are thread-safe
     * - Pattern is in sorted CSR format
     *
     * @throws FEException if already finalized
     *
     * Complexity: O(NNZ * log(NNZ/nRows)) for sorting columns per row
     *
     * @note Column indices within each row are sorted in ascending order
     *       for deterministic behavior and efficient binary search.
     */
    void finalize();

    // =========================================================================
    // Query methods (safe in any state, thread-safe after finalize)
    // =========================================================================

    /**
     * @brief Check if pattern is finalized (immutable CSR format)
     */
    [[nodiscard]] bool isFinalized() const noexcept {
        return state_.load(std::memory_order_acquire) == SparsityState::Finalized;
    }

    /**
     * @brief Get current lifecycle state
     */
    [[nodiscard]] SparsityState state() const noexcept {
        return state_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get number of rows
     */
    [[nodiscard]] GlobalIndex numRows() const noexcept { return n_rows_; }

    /**
     * @brief Get number of columns
     */
    [[nodiscard]] GlobalIndex numCols() const noexcept { return n_cols_; }

    /**
     * @brief Check if pattern is square (nRows == nCols)
     */
    [[nodiscard]] bool isSquare() const noexcept { return n_rows_ == n_cols_; }

    /**
     * @brief Get total number of non-zero entries
     *
     * @note During building phase, this iterates over all row sets.
     *       After finalization, this is O(1).
     */
    [[nodiscard]] GlobalIndex getNnz() const;

    /**
     * @brief Get number of entries in a specific row
     *
     * @param row Row index
     * @return Number of non-zeros in the row
     * @throws FEException if row out of range
     */
    [[nodiscard]] GlobalIndex getRowNnz(GlobalIndex row) const;

    /**
     * @brief Get column indices for a row (finalized patterns only)
     *
     * @param row Row index
     * @return View of sorted column indices
     * @throws FEException if not finalized or row out of range
     *
     * Complexity: O(1)
     */
    [[nodiscard]] RowView getRowIndices(GlobalIndex row) const;

    /**
     * @brief Get column indices as a span (finalized patterns only)
     *
     * @param row Row index
     * @return Span of sorted column indices
     * @throws FEException if not finalized or row out of range
     */
    [[nodiscard]] std::span<const GlobalIndex> getRowSpan(GlobalIndex row) const;

    /**
     * @brief Check if an entry exists in the pattern
     *
     * @param row Row index
     * @param col Column index
     * @return true if (row, col) is in the pattern
     *
     * Complexity: O(log row_nnz) for finalized, O(log row_nnz) for building
     */
    [[nodiscard]] bool hasEntry(GlobalIndex row, GlobalIndex col) const;

    /**
     * @brief Check if diagonal entry exists for a row
     *
     * @param row Row index
     * @return true if (row, row) is in the pattern
     */
    [[nodiscard]] bool hasDiagonal(GlobalIndex row) const;

    /**
     * @brief Check if all diagonal entries exist (square patterns)
     *
     * @return true if pattern is square and all (i,i) entries exist
     */
    [[nodiscard]] bool hasAllDiagonals() const;

    // =========================================================================
    // CSR format access (finalized patterns only)
    // =========================================================================

    /**
     * @brief Get CSR row pointer array
     *
     * @return Span of [n_rows+1] offsets into column indices
     * @throws FEException if not finalized
     *
     * row_ptr[i] is the index of the first column entry for row i.
     * row_ptr[n_rows] is the total NNZ.
     */
    [[nodiscard]] std::span<const GlobalIndex> getRowPtr() const;

    /**
     * @brief Get CSR column indices array
     *
     * @return Span of column indices (sorted within each row)
     * @throws FEException if not finalized
     */
    [[nodiscard]] std::span<const GlobalIndex> getColIndices() const;

    /**
     * @brief Get raw pointer to row pointer array (for backend interop)
     *
     * @throws FEException if not finalized
     */
    [[nodiscard]] const GlobalIndex* rowPtrData() const;

    /**
     * @brief Get raw pointer to column indices array (for backend interop)
     *
     * @throws FEException if not finalized
     */
    [[nodiscard]] const GlobalIndex* colIndicesData() const;

    // =========================================================================
    // Statistics and analysis
    // =========================================================================

    /**
     * @brief Compute detailed statistics about the pattern
     *
     * @return Statistics structure with various metrics
     *
     * @note For building patterns, some statistics may be expensive to compute.
     */
    [[nodiscard]] SparsityStats computeStats() const;

    /**
     * @brief Compute bandwidth of the pattern
     *
     * @return Maximum |row - col| for any (row, col) in the pattern
     *
     * Bandwidth is 0 for diagonal-only patterns.
     */
    [[nodiscard]] GlobalIndex computeBandwidth() const;

    /**
     * @brief Check if pattern is structurally symmetric
     *
     * @return true if for every (i,j) in pattern, (j,i) is also in pattern
     *
     * @note Only meaningful for square patterns.
     * @note This is O(NNZ) check.
     */
    [[nodiscard]] bool isSymmetric() const;

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate internal consistency
     *
     * Checks:
     * - Row pointers are monotonically increasing
     * - All column indices are in valid range [0, n_cols)
     * - Column indices are sorted within each row (if finalized)
     * - No duplicate entries (if finalized)
     *
     * @return true if valid, false otherwise
     */
    [[nodiscard]] bool validate() const noexcept;

    /**
     * @brief Get detailed validation error message
     *
     * @return Error message, or empty string if valid
     */
    [[nodiscard]] std::string validationError() const;

    // =========================================================================
    // Utility methods
    // =========================================================================

    /**
     * @brief Apply a permutation to rows and columns
     *
     * @param row_perm Row permutation: new_row = row_perm[old_row]
     * @param col_perm Column permutation: new_col = col_perm[old_col]
     * @return New pattern with permuted indices
     *
     * Useful for reordering algorithms (RCM, AMD, etc.)
     */
    [[nodiscard]] SparsityPattern permute(std::span<const GlobalIndex> row_perm,
                                          std::span<const GlobalIndex> col_perm) const;

    /**
     * @brief Create a transpose of the pattern
     *
     * @return New pattern where (i,j) becomes (j,i)
     *
     * Dimensions are swapped: result has (n_cols x n_rows).
     */
    [[nodiscard]] SparsityPattern transpose() const;

    /**
     * @brief Extract a sub-block of the pattern
     *
     * @param row_set Rows to extract (sorted)
     * @param col_set Columns to extract (sorted)
     * @return New pattern with only entries in (row_set x col_set)
     *
     * Indices in the result are renumbered to 0-based.
     */
    [[nodiscard]] SparsityPattern extract(std::span<const GlobalIndex> row_set,
                                          std::span<const GlobalIndex> col_set) const;

    /**
     * @brief Memory usage in bytes
     *
     * @return Approximate memory consumption
     */
    [[nodiscard]] std::size_t memoryUsageBytes() const noexcept;

private:
    friend class ConstraintSparsityAugmenter;
    friend SparsityPattern patternUnion(const SparsityPattern& a, const SparsityPattern& b);
    friend SparsityPattern patternIntersection(const SparsityPattern& a, const SparsityPattern& b);
    friend SparsityPattern symmetrize(const SparsityPattern& pattern);

    // Internal helpers
    void checkNotFinalized() const;
    void checkFinalized() const;
    void checkRowIndex(GlobalIndex row) const;
    void checkColIndex(GlobalIndex col) const;
    void checkIndices(GlobalIndex row, GlobalIndex col) const;

    // Dimensions
    GlobalIndex n_rows_{0};
    GlobalIndex n_cols_{0};

    // Building phase storage: set of column indices per row
    // Using std::set for deterministic ordering (sorted column indices)
    std::vector<std::set<GlobalIndex>> row_sets_;

    // Finalized CSR storage
    std::vector<GlobalIndex> row_ptr_;    // [n_rows + 1] offsets
    std::vector<GlobalIndex> col_idx_;    // [nnz] column indices

    // Cached statistics (computed on finalize)
    mutable GlobalIndex cached_nnz_{-1};

    // State management
    std::atomic<SparsityState> state_{SparsityState::Building};
};

// ============================================================================
// Free functions for pattern operations
// ============================================================================

/**
 * @brief Compute the union of two sparsity patterns
 *
 * @param a First pattern
 * @param b Second pattern (must have same dimensions)
 * @return Pattern containing entries from both a and b
 * @throws FEException if dimensions don't match
 */
SparsityPattern patternUnion(const SparsityPattern& a, const SparsityPattern& b);

/**
 * @brief Compute the intersection of two sparsity patterns
 *
 * @param a First pattern
 * @param b Second pattern (must have same dimensions)
 * @return Pattern containing only entries in both a and b
 * @throws FEException if dimensions don't match
 */
SparsityPattern patternIntersection(const SparsityPattern& a, const SparsityPattern& b);

/**
 * @brief Compute A + A^T pattern (symmetrize)
 *
 * @param pattern Input pattern (must be square)
 * @return Pattern with (i,j) if (i,j) or (j,i) in original
 * @throws FEException if pattern is not square
 */
SparsityPattern symmetrize(const SparsityPattern& pattern);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_SPARSITY_PATTERN_H
