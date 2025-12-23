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

#ifndef SVMP_FE_SPARSITY_TWO_PASS_BUILDER_H
#define SVMP_FE_SPARSITY_TWO_PASS_BUILDER_H

/**
 * @file SparsityTwoPassBuilder.h
 * @brief Scalable two-pass sparsity pattern construction for massive NNZ counts
 *
 * This header provides the SparsityTwoPassBuilder class for constructing
 * sparsity patterns in two passes:
 *
 * Pass 1: Count row NNZ (exact or approximate) - minimal memory
 * Pass 2: Fill column indices - single allocation
 *
 * Key advantages over standard SparsityPattern:
 * - Avoids "set-of-sets" memory explosion (40+ bytes per entry in std::set)
 * - Single allocation after counting eliminates reallocation overhead
 * - Deterministic through sorted merge, not insertion order
 * - Suitable for billion-NNZ problems
 *
 * Trade-offs:
 * - Requires two traversals of element connectivity
 * - No hasEntry() during construction
 * - Counting pass may overcount due to duplicate detection
 *
 * Complexity:
 * - Pass 1 (count): O(n_elements * dofs_per_elem^2)
 * - Pass 2 (fill): O(n_elements * dofs_per_elem^2)
 * - finalize(): O(NNZ * log(avg_row_nnz)) for sorting
 *
 * Memory:
 * - Pass 1: O(n_rows) for row counts
 * - Pass 2: O(NNZ) for column indices (single allocation)
 * - Final: O(n_rows + NNZ) CSR format
 *
 * @see SparsityPattern for standard construction
 * @see CompressedSparsity for similar memory-efficient approach
 */

#include "SparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <functional>
#include <atomic>
#include <mutex>
#include <memory>

namespace svmp {
namespace FE {
namespace sparsity {

/**
 * @brief Options for two-pass construction
 */
struct TwoPassBuildOptions {
    /**
     * @brief Use approximate counting (faster, may over-allocate)
     *
     * If true, counting assumes all element DOF pairs are unique.
     * This is faster but may allocate more memory than needed.
     * If false, uses hash sets for exact counting (slower but precise).
     */
    bool approximate_count{true};

    /**
     * @brief Over-allocation factor for approximate counting
     *
     * When using approximate counting, multiply by this factor
     * to handle edge effects at domain boundaries.
     */
    double overallocation_factor{1.0};

    /**
     * @brief Enable parallel counting/filling
     */
    bool enable_parallel{false};

    /**
     * @brief Number of threads for parallel mode
     */
    int num_threads{1};

    /**
     * @brief Force diagonal entries in final pattern
     */
    bool ensure_diagonal{true};

    /**
     * @brief Force non-empty rows in final pattern
     */
    bool ensure_non_empty_rows{true};

    /**
     * @brief Memory budget hint in bytes (0 = unlimited)
     *
     * If set, builder may reduce over-allocation or use
     * incremental strategies when approaching budget.
     */
    std::size_t memory_budget_bytes{0};
};

/**
 * @brief Statistics from two-pass construction
 */
struct TwoPassBuildStats {
    GlobalIndex n_rows{0};
    GlobalIndex n_cols{0};

    // Counting phase stats
    GlobalIndex estimated_nnz{0};      ///< Estimated NNZ after counting
    GlobalIndex allocated_nnz{0};      ///< Allocated storage (may differ)
    std::size_t count_memory_bytes{0}; ///< Peak memory during counting

    // Fill phase stats
    GlobalIndex actual_nnz{0};         ///< Actual NNZ after deduplication
    GlobalIndex duplicates_removed{0}; ///< Duplicates removed in finalize
    std::size_t fill_memory_bytes{0};  ///< Peak memory during fill

    // Final stats
    std::size_t final_memory_bytes{0}; ///< Final CSR memory
    double memory_efficiency{1.0};     ///< actual/allocated ratio
};

/**
 * @brief Phase of two-pass construction
 */
enum class TwoPassPhase : std::uint8_t {
    Initial,     ///< Not started
    Counting,    ///< Pass 1: counting entries
    Allocated,   ///< Storage allocated, ready for fill
    Filling,     ///< Pass 2: filling entries
    Finalized    ///< Pattern complete
};

/**
 * @brief Two-pass sparsity pattern builder for massive problems
 *
 * SparsityTwoPassBuilder provides a memory-efficient approach to constructing
 * sparsity patterns for very large problems (billions of NNZ). The two-pass
 * approach minimizes memory overhead by:
 *
 * 1. First counting the number of entries per row
 * 2. Allocating exactly the required storage
 * 3. Then filling the column indices
 *
 * Usage:
 * @code
 * SparsityTwoPassBuilder builder(n_dofs, n_dofs);
 *
 * // Pass 1: Count
 * for (element : mesh) {
 *     auto dofs = dof_map.getCellDofs(element);
 *     builder.countElementCouplings(dofs);
 * }
 * builder.finalizeCount();
 *
 * // Pass 2: Fill
 * for (element : mesh) {
 *     auto dofs = dof_map.getCellDofs(element);
 *     builder.addElementCouplings(dofs);
 * }
 *
 * SparsityPattern pattern = builder.finalize();
 * @endcode
 *
 * Parallel usage:
 * @code
 * TwoPassBuildOptions opts;
 * opts.enable_parallel = true;
 * opts.num_threads = 4;
 * SparsityTwoPassBuilder builder(n_dofs, n_dofs, opts);
 *
 * // Pass 1: Parallel counting
 * builder.countParallel([&](int thread_id, TwoPassCounter& counter) {
 *     for (element : thread_elements[thread_id]) {
 *         auto dofs = dof_map.getCellDofs(element);
 *         counter.countElementCouplings(dofs);
 *     }
 * });
 * builder.finalizeCount();
 *
 * // Pass 2: Parallel fill
 * builder.fillParallel([&](int thread_id, TwoPassFiller& filler) {
 *     for (element : thread_elements[thread_id]) {
 *         auto dofs = dof_map.getCellDofs(element);
 *         filler.addElementCouplings(dofs);
 *     }
 * });
 *
 * SparsityPattern pattern = builder.finalize();
 * @endcode
 */
class SparsityTwoPassBuilder {
public:
    // Forward declarations for parallel helpers
    class Counter;
    class Filler;

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor - empty builder
     */
    SparsityTwoPassBuilder() = default;

    /**
     * @brief Construct with dimensions
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns (-1 for square)
     */
    explicit SparsityTwoPassBuilder(GlobalIndex n_rows, GlobalIndex n_cols = -1);

    /**
     * @brief Construct with dimensions and options
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @param options Construction options
     */
    SparsityTwoPassBuilder(GlobalIndex n_rows, GlobalIndex n_cols,
                           const TwoPassBuildOptions& options);

    /// Destructor
    ~SparsityTwoPassBuilder() = default;

    // Non-copyable due to internal state
    SparsityTwoPassBuilder(const SparsityTwoPassBuilder&) = delete;
    SparsityTwoPassBuilder& operator=(const SparsityTwoPassBuilder&) = delete;

    // Movable
    SparsityTwoPassBuilder(SparsityTwoPassBuilder&&) noexcept = default;
    SparsityTwoPassBuilder& operator=(SparsityTwoPassBuilder&&) noexcept = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set pattern dimensions
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     */
    void resize(GlobalIndex n_rows, GlobalIndex n_cols = -1);

    /**
     * @brief Reset to initial state
     */
    void reset();

    /**
     * @brief Set construction options
     */
    void setOptions(const TwoPassBuildOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const TwoPassBuildOptions& getOptions() const noexcept {
        return options_;
    }

    // =========================================================================
    // Pass 1: Counting
    // =========================================================================

    /**
     * @brief Count single entry (may be duplicate)
     *
     * @param row Row index
     * @param col Column index
     */
    void countEntry(GlobalIndex row, GlobalIndex col);

    /**
     * @brief Count multiple entries in a row
     *
     * @param row Row index
     * @param cols Column indices
     */
    void countEntries(GlobalIndex row, std::span<const GlobalIndex> cols);

    /**
     * @brief Count element couplings (all DOF pairs)
     *
     * @param dofs DOF indices
     */
    void countElementCouplings(std::span<const GlobalIndex> dofs);

    /**
     * @brief Count element couplings (row x col pairs)
     *
     * @param row_dofs Row DOF indices
     * @param col_dofs Column DOF indices
     */
    void countElementCouplings(std::span<const GlobalIndex> row_dofs,
                               std::span<const GlobalIndex> col_dofs);

    /**
     * @brief Finalize counting phase and allocate storage
     *
     * Must be called after counting and before filling.
     * Allocates column index storage based on row counts.
     */
    void finalizeCount();

    // =========================================================================
    // Pass 2: Filling
    // =========================================================================

    /**
     * @brief Add single entry
     *
     * @param row Row index
     * @param col Column index
     */
    void addEntry(GlobalIndex row, GlobalIndex col);

    /**
     * @brief Add multiple entries to a row
     *
     * @param row Row index
     * @param cols Column indices
     */
    void addEntries(GlobalIndex row, std::span<const GlobalIndex> cols);

    /**
     * @brief Add element couplings (all DOF pairs)
     *
     * @param dofs DOF indices
     */
    void addElementCouplings(std::span<const GlobalIndex> dofs);

    /**
     * @brief Add element couplings (row x col pairs)
     *
     * @param row_dofs Row DOF indices
     * @param col_dofs Column DOF indices
     */
    void addElementCouplings(std::span<const GlobalIndex> row_dofs,
                             std::span<const GlobalIndex> col_dofs);

    // =========================================================================
    // Parallel operations
    // =========================================================================

    /**
     * @brief Parallel counting callback type
     */
    using CountCallback = std::function<void(int thread_id, Counter& counter)>;

    /**
     * @brief Parallel filling callback type
     */
    using FillCallback = std::function<void(int thread_id, Filler& filler)>;

    /**
     * @brief Execute parallel counting
     *
     * @param callback Function called for each thread with Counter helper
     */
    void countParallel(CountCallback callback);

    /**
     * @brief Execute parallel filling
     *
     * @param callback Function called for each thread with Filler helper
     */
    void fillParallel(FillCallback callback);

    // =========================================================================
    // Finalization
    // =========================================================================

    /**
     * @brief Finalize and return sparsity pattern
     *
     * Sorts column indices, removes duplicates, and returns
     * the finalized SparsityPattern.
     *
     * @return Finalized sparsity pattern
     */
    [[nodiscard]] SparsityPattern finalize();

    /**
     * @brief Check if pattern can be finalized
     */
    [[nodiscard]] bool canFinalize() const noexcept;

    // =========================================================================
    // Query methods
    // =========================================================================

    /**
     * @brief Get current phase
     */
    [[nodiscard]] TwoPassPhase phase() const noexcept { return phase_; }

    /**
     * @brief Get number of rows
     */
    [[nodiscard]] GlobalIndex numRows() const noexcept { return n_rows_; }

    /**
     * @brief Get number of columns
     */
    [[nodiscard]] GlobalIndex numCols() const noexcept { return n_cols_; }

    /**
     * @brief Get estimated NNZ after counting
     */
    [[nodiscard]] GlobalIndex estimatedNnz() const noexcept;

    /**
     * @brief Get estimated NNZ for a specific row
     */
    [[nodiscard]] GlobalIndex getRowEstimate(GlobalIndex row) const;

    /**
     * @brief Get construction statistics
     */
    [[nodiscard]] const TwoPassBuildStats& getStats() const noexcept {
        return stats_;
    }

    /**
     * @brief Get current memory usage in bytes
     */
    [[nodiscard]] std::size_t currentMemoryBytes() const noexcept;

    // =========================================================================
    // Helper classes for parallel construction
    // =========================================================================

    /**
     * @brief Thread-local counter for parallel counting phase
     */
    class Counter {
    public:
        Counter(SparsityTwoPassBuilder& builder, int thread_id);

        void countEntry(GlobalIndex row, GlobalIndex col);
        void countEntries(GlobalIndex row, std::span<const GlobalIndex> cols);
        void countElementCouplings(std::span<const GlobalIndex> dofs);
        void countElementCouplings(std::span<const GlobalIndex> row_dofs,
                                   std::span<const GlobalIndex> col_dofs);

    private:
        friend class SparsityTwoPassBuilder;
        SparsityTwoPassBuilder& builder_;
        int thread_id_;
        std::vector<GlobalIndex> local_counts_;
    };

    /**
     * @brief Thread-local filler for parallel fill phase
     */
    class Filler {
    public:
        Filler(SparsityTwoPassBuilder& builder, int thread_id);

        void addEntry(GlobalIndex row, GlobalIndex col);
        void addEntries(GlobalIndex row, std::span<const GlobalIndex> cols);
        void addElementCouplings(std::span<const GlobalIndex> dofs);
        void addElementCouplings(std::span<const GlobalIndex> row_dofs,
                                 std::span<const GlobalIndex> col_dofs);

    private:
        friend class SparsityTwoPassBuilder;
        SparsityTwoPassBuilder& builder_;
        int thread_id_;
        std::vector<std::pair<GlobalIndex, GlobalIndex>> local_entries_;
    };

private:
    // Internal helpers
    void checkPhase(TwoPassPhase expected, const char* operation) const;
    void checkPhaseForCounting(const char* operation) const;
    void checkRowIndex(GlobalIndex row) const;
    void checkColIndex(GlobalIndex col) const;
    void addEntryInternal(GlobalIndex row, GlobalIndex col);
    void mergeThreadCounts();
    void mergeThreadEntries();
    void sortAndDeduplicate();
    void updateStats();

    // Dimensions
    GlobalIndex n_rows_{0};
    GlobalIndex n_cols_{0};

    // Options
    TwoPassBuildOptions options_;

    // Phase tracking
    TwoPassPhase phase_{TwoPassPhase::Initial};

    // Pass 1 storage: row counts
    std::vector<GlobalIndex> row_counts_;

    // Pass 2 storage: CSR-like arrays
    std::vector<GlobalIndex> row_offsets_;  // Size: n_rows + 1
    std::vector<GlobalIndex> col_indices_;  // Size: allocated_nnz
    std::vector<GlobalIndex> row_positions_; // Current fill position per row

    // Thread-local buffers for parallel mode
    std::vector<std::unique_ptr<std::vector<GlobalIndex>>> thread_counts_;
    std::vector<std::unique_ptr<std::vector<std::pair<GlobalIndex, GlobalIndex>>>> thread_entries_;
    std::mutex merge_mutex_;

    // Statistics
    mutable TwoPassBuildStats stats_;
};

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * @brief Build pattern using two-pass for minimum memory
 *
 * @param n_dofs Total DOFs
 * @param n_elements Number of elements
 * @param element_dofs Function returning DOFs for element i
 * @return Finalized sparsity pattern
 */
template<typename DofGetter>
[[nodiscard]] SparsityPattern buildPatternTwoPassScalable(
    GlobalIndex n_dofs,
    GlobalIndex n_elements,
    DofGetter&& element_dofs)
{
    SparsityTwoPassBuilder builder(n_dofs, n_dofs);

    // Pass 1: Count
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        auto dofs = element_dofs(e);
        builder.countElementCouplings(dofs);
    }
    builder.finalizeCount();

    // Pass 2: Fill
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        auto dofs = element_dofs(e);
        builder.addElementCouplings(dofs);
    }

    return builder.finalize();
}

/**
 * @brief Build rectangular pattern using two-pass
 *
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param n_elements Number of elements
 * @param row_dofs Function returning row DOFs for element i
 * @param col_dofs Function returning column DOFs for element i
 * @return Finalized sparsity pattern
 */
template<typename RowDofGetter, typename ColDofGetter>
[[nodiscard]] SparsityPattern buildPatternTwoPassRectangular(
    GlobalIndex n_rows,
    GlobalIndex n_cols,
    GlobalIndex n_elements,
    RowDofGetter&& row_dofs,
    ColDofGetter&& col_dofs)
{
    SparsityTwoPassBuilder builder(n_rows, n_cols);

    // Pass 1: Count
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        builder.countElementCouplings(row_dofs(e), col_dofs(e));
    }
    builder.finalizeCount();

    // Pass 2: Fill
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        builder.addElementCouplings(row_dofs(e), col_dofs(e));
    }

    return builder.finalize();
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_TWO_PASS_BUILDER_H
