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

#ifndef SVMP_FE_SPARSITY_COMPRESSED_SPARSITY_H
#define SVMP_FE_SPARSITY_COMPRESSED_SPARSITY_H

/**
 * @file CompressedSparsity.h
 * @brief Memory-efficient construction phase for sparsity patterns
 *
 * This header provides the CompressedSparsity class for memory-efficient
 * construction of sparsity patterns, especially for very large problems.
 * Unlike SparsityPattern which uses set-of-sets during building (high memory
 * overhead from tree nodes), CompressedSparsity uses:
 *
 * 1. Sorted vectors with deferred deduplication
 * 2. Two-pass counting option for exact preallocation
 * 3. Thread-local buffers for parallel insertion
 * 4. Memory pool for reduced allocation overhead
 *
 * Key features:
 * - Lower memory overhead than std::set per row
 * - Efficient batch insertion
 * - Parallel-safe construction with thread-local buffers
 * - Deterministic merge for reproducibility
 * - Two-pass mode for minimal memory footprint
 *
 * Trade-offs vs SparsityPattern:
 * + 2-4x lower memory during construction
 * + Better cache performance for batch insertion
 * - O(n log n) finalization per row vs O(n) for sets
 * - No efficient hasEntry() during construction
 *
 * Use CompressedSparsity when:
 * - Memory is constrained
 * - Pattern is built via batch insertions (element loops)
 * - Problem size is very large (millions of DOFs)
 *
 * Use SparsityPattern when:
 * - Random insertion order with immediate deduplication
 * - Need hasEntry() during construction
 * - Problem size is moderate
 *
 * @see SparsityPattern for standard construction
 */

#include "SparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <mutex>
#include <thread>
#include <atomic>
#include <algorithm>

namespace svmp {
namespace FE {
namespace sparsity {

/**
 * @brief Construction mode for compressed sparsity
 */
enum class CompressionMode : std::uint8_t {
    /**
     * @brief Single-pass construction
     *
     * Entries are collected in unsorted buffers, then sorted and
     * deduplicated during finalization. Good balance of speed and memory.
     */
    SinglePass,

    /**
     * @brief Two-pass construction
     *
     * First pass counts entries per row, second pass fills pre-allocated
     * arrays. Minimum memory footprint but requires two traversals.
     */
    TwoPass,

    /**
     * @brief Incremental compression
     *
     * Compresses when buffer exceeds threshold. Good for very long
     * construction phases where periodic memory reclamation helps.
     */
    Incremental
};

/**
 * @brief Options for compressed sparsity construction
 */
struct CompressedSparsityOptions {
    CompressionMode mode{CompressionMode::SinglePass};
    std::size_t buffer_size_hint{4096};     ///< Initial buffer size per row
    std::size_t compression_threshold{1000}; ///< Compress when duplicates exceed this
    bool enable_parallel{false};            ///< Enable thread-local buffers
    int num_threads{1};                     ///< Number of threads if parallel
    bool verify_sorted{false};              ///< Debug: verify sorting after finalize
};

/**
 * @brief Statistics about compressed construction
 */
struct CompressionStats {
    GlobalIndex n_rows{0};
    GlobalIndex n_cols{0};
    GlobalIndex total_insertions{0};    ///< Total insertions (with duplicates)
    GlobalIndex unique_entries{0};      ///< Unique entries after compression
    GlobalIndex duplicates_removed{0};  ///< Duplicates removed
    std::size_t peak_memory_bytes{0};   ///< Peak memory during construction
    std::size_t final_memory_bytes{0};  ///< Final memory after compression
    double compression_ratio{1.0};      ///< insertions / unique
    int num_compressions{0};            ///< Number of compression passes
};

/**
 * @brief Thread-local insertion buffer
 */
class ThreadLocalBuffer {
public:
    ThreadLocalBuffer() = default;

    /**
     * @brief Add entry to thread-local buffer
     */
    void addEntry(GlobalIndex row, GlobalIndex col);

    /**
     * @brief Add batch of entries
     */
    void addEntries(GlobalIndex row, std::span<const GlobalIndex> cols);

    /**
     * @brief Clear buffer
     */
    void clear();

    /**
     * @brief Get buffer contents
     */
    [[nodiscard]] const std::vector<std::pair<GlobalIndex, GlobalIndex>>& entries() const {
        return entries_;
    }

    /**
     * @brief Get buffer size
     */
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

private:
    std::vector<std::pair<GlobalIndex, GlobalIndex>> entries_;
};

/**
 * @brief Memory-efficient sparsity pattern construction
 *
 * CompressedSparsity provides a memory-efficient alternative to SparsityPattern
 * for the building phase. It uses sorted vectors instead of sets, with
 * deferred sorting and deduplication.
 *
 * Usage:
 * @code
 * // Single-pass construction
 * CompressedSparsity cs(n_dofs, n_dofs);
 * for (element : mesh) {
 *     auto dofs = dof_map.getCellDofs(element);
 *     cs.addElementCouplings(dofs);
 * }
 * SparsityPattern pattern = cs.toSparsityPattern();
 *
 * // Two-pass construction (minimum memory)
 * CompressedSparsityOptions opts;
 * opts.mode = CompressionMode::TwoPass;
 *
 * // First pass: count
 * CompressedSparsity cs1(n_dofs, n_dofs, opts);
 * for (element : mesh) {
 *     auto dofs = dof_map.getCellDofs(element);
 *     cs1.countElementCouplings(dofs);
 * }
 * cs1.finalizeCounting();
 *
 * // Second pass: fill
 * for (element : mesh) {
 *     auto dofs = dof_map.getCellDofs(element);
 *     cs1.addElementCouplings(dofs);
 * }
 * SparsityPattern pattern = cs1.toSparsityPattern();
 *
 * // Parallel construction
 * opts.enable_parallel = true;
 * opts.num_threads = 4;
 * CompressedSparsity cs_par(n_dofs, n_dofs, opts);
 * #pragma omp parallel for
 * for (int t = 0; t < n_threads; ++t) {
 *     int thread_id = omp_get_thread_num();
 *     for (element : thread_elements[t]) {
 *         cs_par.addElementCouplingsThreaded(thread_id, dofs);
 *     }
 * }
 * cs_par.mergeThreadBuffers();
 * SparsityPattern pattern = cs_par.toSparsityPattern();
 * @endcode
 */
class CompressedSparsity {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor - empty pattern
     */
    CompressedSparsity() = default;

    /**
     * @brief Construct with dimensions
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns (-1 for square)
     */
    explicit CompressedSparsity(GlobalIndex n_rows, GlobalIndex n_cols = -1);

    /**
     * @brief Construct with dimensions and options
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @param options Construction options
     */
    CompressedSparsity(GlobalIndex n_rows, GlobalIndex n_cols,
                       const CompressedSparsityOptions& options);

    /// Destructor
    ~CompressedSparsity() = default;

    // Non-copyable and non-movable due to atomic and mutex members
    CompressedSparsity(const CompressedSparsity&) = delete;
    CompressedSparsity& operator=(const CompressedSparsity&) = delete;
    CompressedSparsity(CompressedSparsity&&) = delete;
    CompressedSparsity& operator=(CompressedSparsity&&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Resize pattern
     *
     * @param n_rows New number of rows
     * @param n_cols New number of columns
     */
    void resize(GlobalIndex n_rows, GlobalIndex n_cols = -1);

    /**
     * @brief Clear all data
     */
    void clear();

    /**
     * @brief Reserve memory for expected entries
     *
     * @param entries_per_row Expected average entries per row
     */
    void reserve(GlobalIndex entries_per_row);

    /**
     * @brief Reserve memory for specific rows
     *
     * @param row_reserves Expected entries per row
     */
    void reserve(std::span<const GlobalIndex> row_reserves);

    // =========================================================================
    // Single-pass insertion
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
     * @brief Add element couplings (all pairs)
     *
     * @param dofs DOF indices
     */
    void addElementCouplings(std::span<const GlobalIndex> dofs);

    /**
     * @brief Add element couplings (row x col)
     *
     * @param row_dofs Row DOF indices
     * @param col_dofs Column DOF indices
     */
    void addElementCouplings(std::span<const GlobalIndex> row_dofs,
                             std::span<const GlobalIndex> col_dofs);

    // =========================================================================
    // Two-pass construction (counting phase)
    // =========================================================================

    /**
     * @brief Count entry (first pass of two-pass mode)
     */
    void countEntry(GlobalIndex row, GlobalIndex col);

    /**
     * @brief Count entries (first pass)
     */
    void countEntries(GlobalIndex row, std::span<const GlobalIndex> cols);

    /**
     * @brief Count element couplings (first pass)
     */
    void countElementCouplings(std::span<const GlobalIndex> dofs);

    /**
     * @brief Finalize counting phase and allocate storage
     */
    void finalizeCounting();

    // =========================================================================
    // Parallel insertion
    // =========================================================================

    /**
     * @brief Initialize thread-local buffers
     *
     * @param num_threads Number of threads
     */
    void initThreadBuffers(int num_threads);

    /**
     * @brief Add entry from specific thread
     *
     * @param thread_id Thread identifier
     * @param row Row index
     * @param col Column index
     */
    void addEntryThreaded(int thread_id, GlobalIndex row, GlobalIndex col);

    /**
     * @brief Add entries from specific thread
     *
     * @param thread_id Thread identifier
     * @param row Row index
     * @param cols Column indices
     */
    void addEntriesThreaded(int thread_id, GlobalIndex row,
                            std::span<const GlobalIndex> cols);

    /**
     * @brief Add element couplings from specific thread
     *
     * @param thread_id Thread identifier
     * @param dofs DOF indices
     */
    void addElementCouplingsThreaded(int thread_id,
                                      std::span<const GlobalIndex> dofs);

    /**
     * @brief Merge all thread buffers into main storage
     *
     * Must be called after parallel insertion completes.
     */
    void mergeThreadBuffers();

    // =========================================================================
    // Compression and finalization
    // =========================================================================

    /**
     * @brief Compress current buffers (remove duplicates, sort)
     *
     * Can be called periodically during construction to reclaim memory.
     */
    void compress();

    /**
     * @brief Convert to finalized SparsityPattern
     *
     * @return Finalized sparsity pattern
     */
    [[nodiscard]] SparsityPattern toSparsityPattern();

    /**
     * @brief Convert to SparsityPattern without finalizing
     *
     * Returns pattern in Building state for further modification.
     *
     * @return SparsityPattern in Building state
     */
    [[nodiscard]] SparsityPattern toBuildingPattern() const;

    // =========================================================================
    // Query methods
    // =========================================================================

    /**
     * @brief Get number of rows
     */
    [[nodiscard]] GlobalIndex numRows() const noexcept { return n_rows_; }

    /**
     * @brief Get number of columns
     */
    [[nodiscard]] GlobalIndex numCols() const noexcept { return n_cols_; }

    /**
     * @brief Get approximate NNZ (may include duplicates)
     */
    [[nodiscard]] GlobalIndex getApproximateNnz() const;

    /**
     * @brief Get exact NNZ (requires compression)
     */
    [[nodiscard]] GlobalIndex getExactNnz();

    /**
     * @brief Check if in two-pass counting phase
     */
    [[nodiscard]] bool isCountingPhase() const noexcept {
        return in_counting_phase_;
    }

    /**
     * @brief Get construction statistics
     */
    [[nodiscard]] const CompressionStats& getStats() const noexcept {
        return stats_;
    }

    /**
     * @brief Get current memory usage in bytes
     */
    [[nodiscard]] std::size_t currentMemoryBytes() const noexcept;

    // =========================================================================
    // Options
    // =========================================================================

    /**
     * @brief Get current options
     */
    [[nodiscard]] const CompressedSparsityOptions& getOptions() const noexcept {
        return options_;
    }

private:
    // Internal helpers
    void checkRowIndex(GlobalIndex row) const;
    void checkColIndex(GlobalIndex col) const;
    void sortAndDeduplicate(std::vector<GlobalIndex>& cols);
    void compressRow(GlobalIndex row);
    void updateStats();

    // Dimensions
    GlobalIndex n_rows_{0};
    GlobalIndex n_cols_{0};

    // Options
    CompressedSparsityOptions options_;

    // Main storage: row-indexed vectors of column indices (may have duplicates)
    std::vector<std::vector<GlobalIndex>> row_cols_;

    // Two-pass mode: count per row (first pass), then row_cols_ (second pass)
    std::vector<GlobalIndex> row_counts_;
    bool in_counting_phase_{false};
    bool counting_finalized_{false};

    // Thread-local buffers for parallel insertion
    std::vector<std::unique_ptr<ThreadLocalBuffer>> thread_buffers_;

    // Compression state
    std::vector<bool> row_compressed_;  // Track which rows are compressed
    std::atomic<std::size_t> total_insertions_{0};

    // Statistics
    mutable CompressionStats stats_;
    mutable std::mutex stats_mutex_;
};

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * @brief Build pattern efficiently from element loop
 *
 * @param n_dofs Total DOFs
 * @param element_dofs Function returning DOFs for element i
 * @param n_elements Number of elements
 * @return Finalized sparsity pattern
 */
template<typename DofGetter>
[[nodiscard]] SparsityPattern buildPatternEfficiently(
    GlobalIndex n_dofs,
    DofGetter&& element_dofs,
    GlobalIndex n_elements)
{
    CompressedSparsity cs(n_dofs, n_dofs);

    for (GlobalIndex e = 0; e < n_elements; ++e) {
        auto dofs = element_dofs(e);
        cs.addElementCouplings(dofs);
    }

    return cs.toSparsityPattern();
}

/**
 * @brief Build pattern with two-pass for minimum memory
 *
 * @param n_dofs Total DOFs
 * @param element_dofs Function returning DOFs for element i
 * @param n_elements Number of elements
 * @return Finalized sparsity pattern
 */
template<typename DofGetter>
[[nodiscard]] SparsityPattern buildPatternTwoPass(
    GlobalIndex n_dofs,
    DofGetter&& element_dofs,
    GlobalIndex n_elements)
{
    CompressedSparsityOptions opts;
    opts.mode = CompressionMode::TwoPass;
    CompressedSparsity cs(n_dofs, n_dofs, opts);

    // First pass: count
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        auto dofs = element_dofs(e);
        cs.countElementCouplings(dofs);
    }
    cs.finalizeCounting();

    // Second pass: fill
    for (GlobalIndex e = 0; e < n_elements; ++e) {
        auto dofs = element_dofs(e);
        cs.addElementCouplings(dofs);
    }

    return cs.toSparsityPattern();
}

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPARSITY_COMPRESSED_SPARSITY_H
