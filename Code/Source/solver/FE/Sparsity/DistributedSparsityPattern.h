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

#ifndef SVMP_FE_SPARSITY_DISTRIBUTED_SPARSITY_PATTERN_H
#define SVMP_FE_SPARSITY_DISTRIBUTED_SPARSITY_PATTERN_H

/**
 * @file DistributedSparsityPattern.h
 * @brief First-class distributed sparsity artifact with diag/offdiag split
 *
 * This header provides the DistributedSparsityPattern class for representing
 * sparsity patterns in distributed-memory (MPI) parallel environments. The
 * pattern explicitly separates diagonal (owned x owned) and off-diagonal
 * (owned x ghost) couplings to match common HPC backend requirements.
 *
 * Key features:
 * - Diag/offdiag split matching PETSc MatMPIAIJ, Trilinos Tpetra::CrsGraph
 * - Local column index remapping for off-diagonal entries
 * - Preallocation descriptors for backend matrix creation
 * - Thread-safe read access after finalization
 * - Deterministic ordering for reproducibility
 *
 * Design follows:
 * - PETSc: MatMPIAIJSetPreallocation(mat, d_nz, d_nnz, o_nz, o_nnz)
 * - Trilinos: Tpetra::CrsGraph with row/column maps
 * - deal.II: DynamicSparsityPattern with distributed constraints
 *
 * @see SparsityPattern for sequential pattern representation
 * @see ParallelSparsity for MPI communication and construction
 */

#include "SparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <span>
#include <cstdint>
#include <atomic>
#include <functional>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace sparsity {

class ConstraintSparsityAugmenter;

/**
 * @brief Index range representing contiguous ownership
 *
 * Represents a half-open range [first, last) of global indices
 * owned by this rank.
 */
struct IndexRange {
    GlobalIndex first{0};   ///< First owned index (inclusive)
    GlobalIndex last{0};    ///< One past last owned index (exclusive)

    [[nodiscard]] GlobalIndex size() const noexcept { return last - first; }
    [[nodiscard]] bool empty() const noexcept { return first >= last; }
    [[nodiscard]] bool contains(GlobalIndex idx) const noexcept {
        return idx >= first && idx < last;
    }
};

/**
 * @brief Preallocation information for backend matrix creation
 *
 * Provides per-row NNZ counts for diagonal and off-diagonal blocks,
 * suitable for PETSc MatMPIAIJSetPreallocation and similar APIs.
 */
struct PreallocationInfo {
    std::vector<GlobalIndex> diag_nnz_per_row;     ///< NNZ in diag block per row
    std::vector<GlobalIndex> offdiag_nnz_per_row;  ///< NNZ in offdiag block per row

    GlobalIndex max_diag_nnz{0};      ///< Maximum NNZ in any diag row
    GlobalIndex max_offdiag_nnz{0};   ///< Maximum NNZ in any offdiag row
    GlobalIndex total_diag_nnz{0};    ///< Total diagonal entries
    GlobalIndex total_offdiag_nnz{0}; ///< Total off-diagonal entries

    /**
     * @brief Get raw pointers for PETSc-style preallocation
     *
     * @param[out] d_nnz Pointer to diagonal NNZ array
     * @param[out] o_nnz Pointer to off-diagonal NNZ array
     * @param[out] n_local Number of local rows
     */
    void getPetscArrays(const GlobalIndex*& d_nnz,
                        const GlobalIndex*& o_nnz,
                        GlobalIndex& n_local) const {
        d_nnz = diag_nnz_per_row.data();
        o_nnz = offdiag_nnz_per_row.data();
        n_local = static_cast<GlobalIndex>(diag_nnz_per_row.size());
    }
};

/**
 * @brief Statistics for distributed sparsity pattern
 */
struct DistributedSparsityStats {
    // Local statistics
    GlobalIndex n_owned_rows{0};       ///< Number of locally owned rows
    GlobalIndex n_owned_cols{0};       ///< Number of locally owned columns
    GlobalIndex n_ghost_cols{0};       ///< Number of ghost columns (off-proc)
    GlobalIndex n_relevant_cols{0};    ///< Total columns appearing locally

    GlobalIndex local_diag_nnz{0};     ///< NNZ in diagonal block
    GlobalIndex local_offdiag_nnz{0};  ///< NNZ in off-diagonal block
    GlobalIndex local_total_nnz{0};    ///< Total local NNZ

    // Per-row statistics
    GlobalIndex min_diag_row_nnz{0};
    GlobalIndex max_diag_row_nnz{0};
    GlobalIndex min_offdiag_row_nnz{0};
    GlobalIndex max_offdiag_row_nnz{0};

    double avg_diag_row_nnz{0.0};
    double avg_offdiag_row_nnz{0.0};

    // Global statistics (requires MPI reduction)
    GlobalIndex global_rows{0};
    GlobalIndex global_cols{0};
    GlobalIndex global_nnz{0};
};

/**
 * @brief First-class distributed sparsity pattern artifact
 *
 * Represents the structural non-zero pattern of a distributed sparse matrix,
 * explicitly separating diagonal and off-diagonal couplings. This matches
 * how HPC backends conceptualize MPI-distributed matrices.
 *
 * Terminology:
 * - **Owned rows**: Rows assigned to this MPI rank
 * - **Owned columns**: Columns corresponding to DOFs owned by this rank
 * - **Ghost columns**: Columns from DOFs owned by other ranks
 * - **Relevant columns**: All columns that appear in owned rows (owned + ghost)
 * - **Diagonal block**: (owned row, owned col) couplings
 * - **Off-diagonal block**: (owned row, ghost col) couplings
 *
 * The off-diagonal block uses local column indices that map to global IDs
 * via the offdiagColMap.
 *
 * Usage pattern:
 * @code
 * // Typically constructed via ParallelSparsity or SparsityBuilder
 * DistributedSparsityPattern dist_pattern(owned_row_range, owned_col_range);
 *
 * // Add entries (global indices)
 * dist_pattern.addEntry(global_row, global_col);
 *
 * // Finalize - separates diag/offdiag, builds local maps
 * dist_pattern.finalize();
 *
 * // Get preallocation for backend
 * auto prealloc = dist_pattern.getPreallocationInfo();
 * @endcode
 *
 * @note Row indices must be in the owned range. Column indices can be any
 *       valid global column index.
 */
class DistributedSparsityPattern {
public:
    // =========================================================================
    // Construction and lifecycle
    // =========================================================================

    /**
     * @brief Default constructor - creates empty pattern
     */
    DistributedSparsityPattern() = default;

    /**
     * @brief Construct with ownership ranges
     *
     * @param owned_rows Range of globally owned row indices
     * @param owned_cols Range of globally owned column indices
     * @param global_rows Total number of global rows
     * @param global_cols Total number of global columns (defaults to global_rows)
     *
     * For square matrices with matching row/column distributions, owned_rows
     * and owned_cols are typically the same range.
     */
    DistributedSparsityPattern(IndexRange owned_rows,
                               IndexRange owned_cols,
                               GlobalIndex global_rows,
                               GlobalIndex global_cols = -1);

    /**
     * @brief Construct with contiguous ownership starting from first_owned
     *
     * @param first_owned_row First owned row index
     * @param n_owned_rows Number of owned rows
     * @param first_owned_col First owned column index
     * @param n_owned_cols Number of owned columns
     * @param global_rows Total global rows
     * @param global_cols Total global columns
     */
    DistributedSparsityPattern(GlobalIndex first_owned_row,
                               GlobalIndex n_owned_rows,
                               GlobalIndex first_owned_col,
                               GlobalIndex n_owned_cols,
                               GlobalIndex global_rows,
                               GlobalIndex global_cols = -1);

    /// Move constructor
    DistributedSparsityPattern(DistributedSparsityPattern&& other) noexcept;

    /// Move assignment
    DistributedSparsityPattern& operator=(DistributedSparsityPattern&& other) noexcept;

    /// Copy constructor
    DistributedSparsityPattern(const DistributedSparsityPattern& other);

    /// Copy assignment
    DistributedSparsityPattern& operator=(const DistributedSparsityPattern& other);

    /// Destructor
    ~DistributedSparsityPattern() = default;

    // =========================================================================
    // Setup methods (Building state only)
    // =========================================================================

    /**
     * @brief Clear all entries, return to Building state
     */
    void clear();

    /**
     * @brief Add a single entry to the pattern
     *
     * @param global_row Global row index (must be owned)
     * @param global_col Global column index (any valid index)
     * @throws FEException if already finalized or row not owned
     *
     * Column indices are automatically classified as diagonal (owned)
     * or off-diagonal (ghost) based on the column ownership range.
     */
    void addEntry(GlobalIndex global_row, GlobalIndex global_col);

    /**
     * @brief Add multiple entries to a row
     *
     * @param global_row Global row index (must be owned)
     * @param global_cols Global column indices
     * @throws FEException if already finalized or row not owned
     */
    void addEntries(GlobalIndex global_row, std::span<const GlobalIndex> global_cols);

    /**
     * @brief Add element couplings (square)
     *
     * @param global_dofs Array of global DOF indices
     *
     * Creates all (dof_i, dof_j) couplings where dof_i is owned.
     */
    void addElementCouplings(std::span<const GlobalIndex> global_dofs);

    /**
     * @brief Add element couplings (rectangular)
     *
     * @param row_dofs Row DOF indices (only owned rows are added)
     * @param col_dofs Column DOF indices (any valid indices)
     */
    void addElementCouplings(std::span<const GlobalIndex> row_dofs,
                             std::span<const GlobalIndex> col_dofs);

    /**
     * @brief Ensure diagonal entries exist for owned rows
     *
     * Only meaningful for square patterns where owned rows == owned cols.
     */
    void ensureDiagonal();

    /**
     * @brief Ensure no owned rows are empty
     */
    void ensureNonEmptyRows();

    /**
     * @brief Finalize the pattern - separate diag/offdiag and compress
     *
     * After finalization:
     * - Ghost column map is built
     * - Diag and offdiag patterns are compressed to CSR
     * - Pattern is immutable and thread-safe
     *
     * @throws FEException if already finalized
     */
    void finalize();

    /**
     * @brief Clone the pattern preserving finalization state
     *
     * Unlike the copy constructor (which reconstructs to Building state),
     * this method returns a pattern that is finalized when the source is
     * finalized, enabling safe caching of distributed artifacts.
     */
    [[nodiscard]] DistributedSparsityPattern cloneFinalized() const;

    // =========================================================================
    // Query methods (safe in any state, thread-safe after finalize)
    // =========================================================================

    /**
     * @brief Check if pattern is finalized
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

    // --- Dimension queries ---

    /**
     * @brief Get total number of global rows
     */
    [[nodiscard]] GlobalIndex globalRows() const noexcept { return global_rows_; }

    /**
     * @brief Get total number of global columns
     */
    [[nodiscard]] GlobalIndex globalCols() const noexcept { return global_cols_; }

    /**
     * @brief Get number of locally owned rows
     */
    [[nodiscard]] GlobalIndex numOwnedRows() const noexcept { return owned_rows_.size(); }

    /**
     * @brief Get number of locally owned columns
     */
    [[nodiscard]] GlobalIndex numOwnedCols() const noexcept { return owned_cols_.size(); }

    /**
     * @brief Get owned row index range
     */
    [[nodiscard]] const IndexRange& ownedRows() const noexcept { return owned_rows_; }

    /**
     * @brief Get owned column index range
     */
    [[nodiscard]] const IndexRange& ownedCols() const noexcept { return owned_cols_; }

    /**
     * @brief Check if pattern is square (global dimensions)
     */
    [[nodiscard]] bool isSquare() const noexcept { return global_rows_ == global_cols_; }

    // --- Ownership queries ---

    /**
     * @brief Check if a row is owned by this rank
     */
    [[nodiscard]] bool ownsRow(GlobalIndex global_row) const noexcept {
        return owned_rows_.contains(global_row);
    }

    /**
     * @brief Check if a column is owned by this rank
     */
    [[nodiscard]] bool ownsCol(GlobalIndex global_col) const noexcept {
        return owned_cols_.contains(global_col);
    }

    /**
     * @brief Check if a column is a ghost column (off-proc)
     */
    [[nodiscard]] bool isGhostCol(GlobalIndex global_col) const noexcept {
        return !owned_cols_.contains(global_col);
    }

    // --- NNZ queries ---

    /**
     * @brief Get total NNZ in diagonal block
     */
    [[nodiscard]] GlobalIndex getDiagNnz() const;

    /**
     * @brief Get total NNZ in off-diagonal block
     */
    [[nodiscard]] GlobalIndex getOffdiagNnz() const;

    /**
     * @brief Get total local NNZ (diag + offdiag)
     */
    [[nodiscard]] GlobalIndex getLocalNnz() const {
        return getDiagNnz() + getOffdiagNnz();
    }

    /**
     * @brief Get NNZ in diagonal block for a row
     *
     * @param local_row Local row index (0 to numOwnedRows-1)
     */
    [[nodiscard]] GlobalIndex getRowDiagNnz(GlobalIndex local_row) const;

    /**
     * @brief Get NNZ in off-diagonal block for a row
     *
     * @param local_row Local row index
     */
    [[nodiscard]] GlobalIndex getRowOffdiagNnz(GlobalIndex local_row) const;

    /**
     * @brief Get total NNZ for a row (diag + offdiag)
     */
    [[nodiscard]] GlobalIndex getRowNnz(GlobalIndex local_row) const {
        return getRowDiagNnz(local_row) + getRowOffdiagNnz(local_row);
    }

    // --- Ghost column information (finalized only) ---

    /**
     * @brief Get number of ghost columns
     */
    [[nodiscard]] GlobalIndex numGhostCols() const noexcept {
        return static_cast<GlobalIndex>(ghost_col_map_.size());
    }

    /**
     * @brief Get total number of relevant columns (owned + ghost)
     */
    [[nodiscard]] GlobalIndex numRelevantCols() const noexcept {
        return numOwnedCols() + numGhostCols();
    }

    /**
     * @brief Get global ID for a ghost column
     *
     * @param local_ghost_idx Local ghost column index (0 to numGhostCols-1)
     * @return Global column ID
     * @throws FEException if not finalized or index out of range
     */
    [[nodiscard]] GlobalIndex ghostColToGlobal(GlobalIndex local_ghost_idx) const;

    /**
     * @brief Get local ghost index for a global column
     *
     * @param global_col Global column ID
     * @return Local ghost index, or -1 if not a ghost column
     */
    [[nodiscard]] GlobalIndex globalToGhostCol(GlobalIndex global_col) const;

    /**
     * @brief Get the sorted list of ghost column global IDs
     *
     * @return Span of ghost column global IDs in sorted order
     * @throws FEException if not finalized
     */
    [[nodiscard]] std::span<const GlobalIndex> getGhostColMap() const;

    // --- Ghost row information (optional, finalized only) ---

    /**
     * @brief Get number of stored ghost rows
     *
     * Ghost rows are non-owned rows whose full sparsity has been imported
     * (typically via MPI exchange) for overlapping methods.
     */
    [[nodiscard]] GlobalIndex numGhostRows() const noexcept {
        return static_cast<GlobalIndex>(ghost_row_map_.size());
    }

    /**
     * @brief Get the sorted list of ghost row global IDs
     *
     * @throws FEException if no ghost rows are stored
     */
    [[nodiscard]] std::span<const GlobalIndex> getGhostRowMap() const;

    /**
     * @brief Get local ghost-row index for a global row ID
     *
     * @return Local ghost-row index, or -1 if not present
     */
    [[nodiscard]] GlobalIndex globalToGhostRow(GlobalIndex global_row) const;

    /**
     * @brief Get global column indices for a stored ghost row
     *
     * @param local_ghost_row Local ghost-row index (0..numGhostRows-1)
     * @return Span of global column indices (sorted unique)
     * @throws FEException if index out of range or ghost rows not stored
     */
    [[nodiscard]] std::span<const GlobalIndex> getGhostRowCols(GlobalIndex local_ghost_row) const;

    /**
     * @brief Replace stored ghost-row sparsity (advanced / MPI exchange)
     *
     * @param ghost_rows Sorted list of ghost row global IDs
     * @param row_ptr CSR row pointer array of size ghost_rows.size()+1
     * @param col_idx Flat CSR global column indices (sorted unique per row)
     *
     * @note This does not affect owned-row diag/offdiag patterns.
     */
    void setGhostRows(std::vector<GlobalIndex> ghost_rows,
                      std::vector<GlobalIndex> row_ptr,
                      std::vector<GlobalIndex> col_idx);

    /**
     * @brief Clear any stored ghost-row sparsity
     */
    void clearGhostRows();

    // --- Pattern access (finalized only) ---

    /**
     * @brief Get the diagonal block pattern
     *
     * @return Reference to diagonal pattern (owned rows x owned cols)
     * @throws FEException if not finalized
     *
     * Column indices are **local** (0 to numOwnedCols-1), offset from
     * owned_cols_.first.
     */
    [[nodiscard]] const SparsityPattern& diagPattern() const;

    /**
     * @brief Get the off-diagonal block pattern
     *
     * @return Reference to off-diagonal pattern (owned rows x ghost cols)
     * @throws FEException if not finalized
     *
     * Column indices are **local ghost indices** (0 to numGhostCols-1),
     * corresponding to entries in the ghost column map.
     */
    [[nodiscard]] const SparsityPattern& offdiagPattern() const;

    /**
     * @brief Get diagonal column indices for a row (local indices)
     *
     * @param local_row Local row index
     * @return Span of local column indices in diagonal block
     */
    [[nodiscard]] std::span<const GlobalIndex> getRowDiagCols(GlobalIndex local_row) const;

    /**
     * @brief Get off-diagonal column indices for a row (local ghost indices)
     *
     * @param local_row Local row index
     * @return Span of local ghost column indices in off-diagonal block
     */
    [[nodiscard]] std::span<const GlobalIndex> getRowOffdiagCols(GlobalIndex local_row) const;

    // =========================================================================
    // Preallocation for backends
    // =========================================================================

    /**
     * @brief Get preallocation information for backend matrix creation
     *
     * @return PreallocationInfo with per-row NNZ arrays
     * @throws FEException if not finalized
     *
     * Suitable for:
     * - PETSc: MatMPIAIJSetPreallocation(mat, 0, d_nnz, 0, o_nnz)
     * - Trilinos: Tpetra::CrsGraph constructor with nnz arrays
     */
    [[nodiscard]] PreallocationInfo getPreallocationInfo() const;

    /**
     * @brief Get diagonal NNZ array for preallocation
     *
     * @return Vector of NNZ per row in diagonal block
     */
    [[nodiscard]] std::vector<GlobalIndex> getDiagNnzPerRow() const;

    /**
     * @brief Get off-diagonal NNZ array for preallocation
     *
     * @return Vector of NNZ per row in off-diagonal block
     */
    [[nodiscard]] std::vector<GlobalIndex> getOffdiagNnzPerRow() const;

    // =========================================================================
    // Statistics and validation
    // =========================================================================

    /**
     * @brief Compute detailed statistics
     *
     * @return Statistics structure (local statistics only, no MPI)
     */
    [[nodiscard]] DistributedSparsityStats computeStats() const;

    /**
     * @brief Validate internal consistency
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

    /**
     * @brief Memory usage in bytes
     */
    [[nodiscard]] std::size_t memoryUsageBytes() const noexcept;

    // =========================================================================
    // Conversion and export
    // =========================================================================

    /**
     * @brief Convert global row index to local row index
     *
     * @param global_row Global row index
     * @return Local row index (0-based), or -1 if not owned
     */
    [[nodiscard]] GlobalIndex globalRowToLocal(GlobalIndex global_row) const noexcept {
        if (!owned_rows_.contains(global_row)) return -1;
        return global_row - owned_rows_.first;
    }

    /**
     * @brief Convert local row index to global row index
     *
     * @param local_row Local row index
     * @return Global row index
     */
    [[nodiscard]] GlobalIndex localRowToGlobal(GlobalIndex local_row) const noexcept {
        return local_row + owned_rows_.first;
    }

    /**
     * @brief Convert global owned column to local column index
     *
     * @param global_col Global column index
     * @return Local column index (0-based), or -1 if not owned
     */
    [[nodiscard]] GlobalIndex globalColToLocal(GlobalIndex global_col) const noexcept {
        if (!owned_cols_.contains(global_col)) return -1;
        return global_col - owned_cols_.first;
    }

    /**
     * @brief Convert local column index to global column index
     *
     * @param local_col Local column index
     * @return Global column index
     */
    [[nodiscard]] GlobalIndex localColToGlobal(GlobalIndex local_col) const noexcept {
        return local_col + owned_cols_.first;
    }

    /**
     * @brief Check if an entry exists (using global indices)
     *
     * @param global_row Global row index
     * @param global_col Global column index
     * @return true if entry exists and row is owned
     */
    [[nodiscard]] bool hasEntry(GlobalIndex global_row, GlobalIndex global_col) const;

    /**
     * @brief Get the global column indices for an owned row (finalized only)
     *
     * @param global_row Global row index (must be owned)
     * @return Sorted list of global column indices for that row
     */
    [[nodiscard]] std::vector<GlobalIndex> getOwnedRowGlobalCols(GlobalIndex global_row) const;

private:
    friend class ConstraintSparsityAugmenter;

    // Internal helpers
    void checkNotFinalized() const;
    void checkFinalized() const;
    void checkOwnedRow(GlobalIndex global_row) const;
    void checkLocalRow(GlobalIndex local_row) const;

    // Global dimensions
    GlobalIndex global_rows_{0};
    GlobalIndex global_cols_{0};

    // Ownership ranges
    IndexRange owned_rows_;
    IndexRange owned_cols_;

    // Building phase: store global column indices per owned row
    // Entry classification (diag vs offdiag) happens at finalization
    std::vector<std::set<GlobalIndex>> building_rows_;

    // Finalized patterns (local indices)
    SparsityPattern diag_pattern_;     // n_owned_rows x n_owned_cols
    SparsityPattern offdiag_pattern_;  // n_owned_rows x n_ghost_cols

    // Ghost column mapping: local ghost index -> global column ID
    std::vector<GlobalIndex> ghost_col_map_;  // Sorted global IDs

    // Reverse map: global column ID -> local ghost index
    std::unordered_map<GlobalIndex, GlobalIndex> global_to_ghost_;

    // Optional ghost-row storage (CSR with global column indices)
    std::vector<GlobalIndex> ghost_row_map_;  // Sorted global row IDs
    std::vector<GlobalIndex> ghost_row_ptr_;  // CSR row ptr [n_ghost_rows+1]
    std::vector<GlobalIndex> ghost_row_cols_; // CSR col idx (global IDs)
    std::unordered_map<GlobalIndex, GlobalIndex> global_to_ghost_row_;

    // State management
    std::atomic<SparsityState> state_{SparsityState::Building};
};

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_DISTRIBUTED_SPARSITY_PATTERN_H
