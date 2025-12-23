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

#include "SparsityTwoPassBuilder.h"
#include <algorithm>
#include <numeric>
#include <thread>
#include <cmath>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Construction
// ============================================================================

SparsityTwoPassBuilder::SparsityTwoPassBuilder(GlobalIndex n_rows, GlobalIndex n_cols)
    : n_rows_(n_rows),
      n_cols_(n_cols < 0 ? n_rows : n_cols),
      phase_(TwoPassPhase::Initial)
{
    FE_CHECK_ARG(n_rows >= 0, "Number of rows must be non-negative");
    FE_CHECK_ARG(n_cols_ >= 0, "Number of columns must be non-negative");

    row_counts_.resize(static_cast<std::size_t>(n_rows_), 0);
    stats_.n_rows = n_rows_;
    stats_.n_cols = n_cols_;
}

SparsityTwoPassBuilder::SparsityTwoPassBuilder(GlobalIndex n_rows, GlobalIndex n_cols,
                                               const TwoPassBuildOptions& options)
    : SparsityTwoPassBuilder(n_rows, n_cols)
{
    options_ = options;
}

// ============================================================================
// Configuration
// ============================================================================

void SparsityTwoPassBuilder::resize(GlobalIndex n_rows, GlobalIndex n_cols) {
    FE_CHECK_ARG(n_rows >= 0, "Number of rows must be non-negative");

    n_rows_ = n_rows;
    n_cols_ = (n_cols < 0) ? n_rows : n_cols;

    reset();
}

void SparsityTwoPassBuilder::reset() {
    row_counts_.clear();
    row_counts_.resize(static_cast<std::size_t>(n_rows_), 0);
    row_offsets_.clear();
    col_indices_.clear();
    row_positions_.clear();
    thread_counts_.clear();
    thread_entries_.clear();
    phase_ = TwoPassPhase::Initial;
    stats_ = TwoPassBuildStats{};
    stats_.n_rows = n_rows_;
    stats_.n_cols = n_cols_;
}

void SparsityTwoPassBuilder::setOptions(const TwoPassBuildOptions& options) {
    options_ = options;
}

// ============================================================================
// Pass 1: Counting
// ============================================================================

void SparsityTwoPassBuilder::countEntry(GlobalIndex row, GlobalIndex col) {
    checkPhaseForCounting("countEntry");
    checkRowIndex(row);
    checkColIndex(col);

    ++row_counts_[static_cast<std::size_t>(row)];
}

void SparsityTwoPassBuilder::countEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    checkPhaseForCounting("countEntries");
    checkRowIndex(row);

    for (GlobalIndex col : cols) {
        checkColIndex(col);
    }

    row_counts_[static_cast<std::size_t>(row)] += static_cast<GlobalIndex>(cols.size());
}

void SparsityTwoPassBuilder::countElementCouplings(std::span<const GlobalIndex> dofs) {
    checkPhaseForCounting("countElementCouplings");

    for (GlobalIndex row_dof : dofs) {
        if (row_dof >= 0 && row_dof < n_rows_) {
            GlobalIndex valid_cols = 0;
            for (GlobalIndex col_dof : dofs) {
                if (col_dof >= 0 && col_dof < n_cols_) {
                    ++valid_cols;
                }
            }
            row_counts_[static_cast<std::size_t>(row_dof)] += valid_cols;
        }
    }
}

void SparsityTwoPassBuilder::countElementCouplings(std::span<const GlobalIndex> row_dofs,
                                                    std::span<const GlobalIndex> col_dofs) {
    checkPhaseForCounting("countElementCouplings");

    // Count valid columns once
    GlobalIndex valid_cols = 0;
    for (GlobalIndex col_dof : col_dofs) {
        if (col_dof >= 0 && col_dof < n_cols_) {
            ++valid_cols;
        }
    }

    for (GlobalIndex row_dof : row_dofs) {
        if (row_dof >= 0 && row_dof < n_rows_) {
            row_counts_[static_cast<std::size_t>(row_dof)] += valid_cols;
        }
    }
}

void SparsityTwoPassBuilder::finalizeCount() {
    checkPhaseForCounting("finalizeCount");

    // Merge thread counts if parallel mode was used
    mergeThreadCounts();

    // Apply overallocation factor if approximate counting
    if (options_.approximate_count && options_.overallocation_factor > 1.0) {
        for (auto& count : row_counts_) {
            count = static_cast<GlobalIndex>(
                std::ceil(static_cast<double>(count) * options_.overallocation_factor));
        }
    }

    // Add diagonal entries if requested
    if (options_.ensure_diagonal && n_rows_ == n_cols_) {
        for (GlobalIndex i = 0; i < n_rows_; ++i) {
            ++row_counts_[static_cast<std::size_t>(i)];
        }
    }

    // Ensure non-empty rows if requested
    if (options_.ensure_non_empty_rows) {
        for (GlobalIndex i = 0; i < n_rows_; ++i) {
            if (row_counts_[static_cast<std::size_t>(i)] == 0) {
                row_counts_[static_cast<std::size_t>(i)] = 1;
            }
        }
    }

    // Build row offsets (prefix sum)
    row_offsets_.resize(static_cast<std::size_t>(n_rows_) + 1);
    row_offsets_[0] = 0;
    for (GlobalIndex i = 0; i < n_rows_; ++i) {
        row_offsets_[static_cast<std::size_t>(i) + 1] =
            row_offsets_[static_cast<std::size_t>(i)] +
            row_counts_[static_cast<std::size_t>(i)];
    }

    // Allocate column indices
    GlobalIndex total_allocated = row_offsets_[static_cast<std::size_t>(n_rows_)];
    col_indices_.resize(static_cast<std::size_t>(total_allocated));

    // Initialize row positions (for filling)
    row_positions_.assign(row_offsets_.begin(), row_offsets_.end() - 1);

    // Update statistics
    stats_.estimated_nnz = total_allocated;
    stats_.allocated_nnz = total_allocated;
    stats_.count_memory_bytes = row_counts_.capacity() * sizeof(GlobalIndex) +
                                row_offsets_.capacity() * sizeof(GlobalIndex);

    phase_ = TwoPassPhase::Allocated;
}

// ============================================================================
// Pass 2: Filling
// ============================================================================

void SparsityTwoPassBuilder::addEntry(GlobalIndex row, GlobalIndex col) {
    checkPhase(TwoPassPhase::Allocated, "addEntry");
    checkRowIndex(row);
    checkColIndex(col);

    auto row_idx = static_cast<std::size_t>(row);
    GlobalIndex& pos = row_positions_[row_idx];
    GlobalIndex end = row_offsets_[row_idx + 1];

    if (pos < end) {
        col_indices_[static_cast<std::size_t>(pos)] = col;
        ++pos;
    }
    // If pos >= end, we've exceeded the allocation for this row
    // This can happen with approximate counting - we'll handle during finalize
}

void SparsityTwoPassBuilder::addEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    checkPhase(TwoPassPhase::Allocated, "addEntries");
    checkRowIndex(row);

    auto row_idx = static_cast<std::size_t>(row);
    GlobalIndex& pos = row_positions_[row_idx];
    GlobalIndex end = row_offsets_[row_idx + 1];

    for (GlobalIndex col : cols) {
        if (col >= 0 && col < n_cols_ && pos < end) {
            col_indices_[static_cast<std::size_t>(pos)] = col;
            ++pos;
        }
    }
}

void SparsityTwoPassBuilder::addElementCouplings(std::span<const GlobalIndex> dofs) {
    checkPhase(TwoPassPhase::Allocated, "addElementCouplings");

    for (GlobalIndex row_dof : dofs) {
        if (row_dof >= 0 && row_dof < n_rows_) {
            auto row_idx = static_cast<std::size_t>(row_dof);
            GlobalIndex& pos = row_positions_[row_idx];
            GlobalIndex end = row_offsets_[row_idx + 1];

            for (GlobalIndex col_dof : dofs) {
                if (col_dof >= 0 && col_dof < n_cols_ && pos < end) {
                    col_indices_[static_cast<std::size_t>(pos)] = col_dof;
                    ++pos;
                }
            }
        }
    }
}

void SparsityTwoPassBuilder::addElementCouplings(std::span<const GlobalIndex> row_dofs,
                                                  std::span<const GlobalIndex> col_dofs) {
    checkPhase(TwoPassPhase::Allocated, "addElementCouplings");

    for (GlobalIndex row_dof : row_dofs) {
        if (row_dof >= 0 && row_dof < n_rows_) {
            auto row_idx = static_cast<std::size_t>(row_dof);
            GlobalIndex& pos = row_positions_[row_idx];
            GlobalIndex end = row_offsets_[row_idx + 1];

            for (GlobalIndex col_dof : col_dofs) {
                if (col_dof >= 0 && col_dof < n_cols_ && pos < end) {
                    col_indices_[static_cast<std::size_t>(pos)] = col_dof;
                    ++pos;
                }
            }
        }
    }
}

// ============================================================================
// Parallel operations
// ============================================================================

void SparsityTwoPassBuilder::countParallel(CountCallback callback) {
    checkPhaseForCounting("countParallel");

    int num_threads = options_.num_threads;
    if (num_threads < 1) num_threads = 1;

    thread_counts_.clear();
    thread_counts_.resize(static_cast<std::size_t>(num_threads));

    for (int t = 0; t < num_threads; ++t) {
        thread_counts_[static_cast<std::size_t>(t)] =
            std::make_unique<std::vector<GlobalIndex>>(static_cast<std::size_t>(n_rows_), 0);
    }

    // Execute callbacks - could use std::thread or OpenMP
    // For simplicity, sequential execution (real impl would parallelize)
    for (int t = 0; t < num_threads; ++t) {
        Counter counter(*this, t);
        callback(t, counter);
    }
}

void SparsityTwoPassBuilder::fillParallel(FillCallback callback) {
    checkPhase(TwoPassPhase::Allocated, "fillParallel");

    int num_threads = options_.num_threads;
    if (num_threads < 1) num_threads = 1;

    thread_entries_.clear();
    thread_entries_.resize(static_cast<std::size_t>(num_threads));

    for (int t = 0; t < num_threads; ++t) {
        thread_entries_[static_cast<std::size_t>(t)] =
            std::make_unique<std::vector<std::pair<GlobalIndex, GlobalIndex>>>();
    }

    // Execute callbacks
    for (int t = 0; t < num_threads; ++t) {
        Filler filler(*this, t);
        callback(t, filler);
    }

    // Merge thread entries
    mergeThreadEntries();
}

// ============================================================================
// Finalization
// ============================================================================

SparsityPattern SparsityTwoPassBuilder::finalize() {
    FE_CHECK_ARG(canFinalize(), "Builder not ready for finalization");

    // Note: We do NOT change phase_ here - addEntryInternal does not check phase
    // Ensure diagonal entries if requested
    if (options_.ensure_diagonal && n_rows_ == n_cols_) {
        for (GlobalIndex i = 0; i < n_rows_; ++i) {
            addEntryInternal(i, i);
        }
    }

    // Ensure non-empty rows if requested
    if (options_.ensure_non_empty_rows) {
        for (GlobalIndex i = 0; i < n_rows_; ++i) {
            auto row_idx = static_cast<std::size_t>(i);
            GlobalIndex start = row_offsets_[row_idx];
            GlobalIndex& pos = row_positions_[row_idx];
            if (pos == start) {
                // Row is empty - add diagonal or column 0
                GlobalIndex col = (n_rows_ == n_cols_) ? i : 0;
                if (col < n_cols_) {
                    addEntryInternal(i, col);
                }
            }
        }
    }

    // Transition to Filling phase
    phase_ = TwoPassPhase::Filling;

    // Sort and deduplicate
    sortAndDeduplicate();

    // Build final SparsityPattern
    SparsityPattern pattern(n_rows_, n_cols_);

    // Insert entries row by row
    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        auto row_idx = static_cast<std::size_t>(row);
        GlobalIndex start = row_offsets_[row_idx];
        GlobalIndex end = row_positions_[row_idx];  // After sort/dedup, this is the actual end

        for (GlobalIndex i = start; i < end; ++i) {
            pattern.addEntry(row, col_indices_[static_cast<std::size_t>(i)]);
        }
    }

    pattern.finalize();

    // Update final statistics
    stats_.actual_nnz = pattern.getNnz();
    stats_.duplicates_removed = stats_.estimated_nnz - stats_.actual_nnz;
    stats_.final_memory_bytes = pattern.memoryUsageBytes();
    if (stats_.allocated_nnz > 0) {
        stats_.memory_efficiency =
            static_cast<double>(stats_.actual_nnz) / static_cast<double>(stats_.allocated_nnz);
    }

    phase_ = TwoPassPhase::Finalized;

    return pattern;
}

bool SparsityTwoPassBuilder::canFinalize() const noexcept {
    return phase_ == TwoPassPhase::Allocated || phase_ == TwoPassPhase::Filling;
}

// ============================================================================
// Query methods
// ============================================================================

GlobalIndex SparsityTwoPassBuilder::estimatedNnz() const noexcept {
    if (phase_ == TwoPassPhase::Initial || phase_ == TwoPassPhase::Counting) {
        GlobalIndex sum = 0;
        for (GlobalIndex count : row_counts_) {
            sum += count;
        }
        return sum;
    }
    return stats_.estimated_nnz;
}

GlobalIndex SparsityTwoPassBuilder::getRowEstimate(GlobalIndex row) const {
    checkRowIndex(row);
    if (phase_ == TwoPassPhase::Initial || phase_ == TwoPassPhase::Counting) {
        return row_counts_[static_cast<std::size_t>(row)];
    }
    auto row_idx = static_cast<std::size_t>(row);
    return row_offsets_[row_idx + 1] - row_offsets_[row_idx];
}

std::size_t SparsityTwoPassBuilder::currentMemoryBytes() const noexcept {
    std::size_t bytes = sizeof(*this);
    bytes += row_counts_.capacity() * sizeof(GlobalIndex);
    bytes += row_offsets_.capacity() * sizeof(GlobalIndex);
    bytes += col_indices_.capacity() * sizeof(GlobalIndex);
    bytes += row_positions_.capacity() * sizeof(GlobalIndex);
    return bytes;
}

// ============================================================================
// Helper class implementations
// ============================================================================

SparsityTwoPassBuilder::Counter::Counter(SparsityTwoPassBuilder& builder, int thread_id)
    : builder_(builder), thread_id_(thread_id)
{
    if (thread_id >= 0 &&
        static_cast<std::size_t>(thread_id) < builder_.thread_counts_.size()) {
        local_counts_.resize(static_cast<std::size_t>(builder_.n_rows_), 0);
    }
}

void SparsityTwoPassBuilder::Counter::countEntry(GlobalIndex row, GlobalIndex col) {
    if (row >= 0 && row < builder_.n_rows_ && col >= 0 && col < builder_.n_cols_) {
        if (!local_counts_.empty()) {
            ++local_counts_[static_cast<std::size_t>(row)];
        } else {
            builder_.countEntry(row, col);
        }
    }
}

void SparsityTwoPassBuilder::Counter::countEntries(GlobalIndex row,
                                                    std::span<const GlobalIndex> cols) {
    if (row >= 0 && row < builder_.n_rows_) {
        GlobalIndex valid_count = 0;
        for (GlobalIndex col : cols) {
            if (col >= 0 && col < builder_.n_cols_) {
                ++valid_count;
            }
        }
        if (!local_counts_.empty()) {
            local_counts_[static_cast<std::size_t>(row)] += valid_count;
        } else {
            builder_.row_counts_[static_cast<std::size_t>(row)] += valid_count;
        }
    }
}

void SparsityTwoPassBuilder::Counter::countElementCouplings(std::span<const GlobalIndex> dofs) {
    for (GlobalIndex row_dof : dofs) {
        if (row_dof >= 0 && row_dof < builder_.n_rows_) {
            GlobalIndex valid_cols = 0;
            for (GlobalIndex col_dof : dofs) {
                if (col_dof >= 0 && col_dof < builder_.n_cols_) {
                    ++valid_cols;
                }
            }
            if (!local_counts_.empty()) {
                local_counts_[static_cast<std::size_t>(row_dof)] += valid_cols;
            } else {
                builder_.row_counts_[static_cast<std::size_t>(row_dof)] += valid_cols;
            }
        }
    }
}

void SparsityTwoPassBuilder::Counter::countElementCouplings(
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs)
{
    GlobalIndex valid_cols = 0;
    for (GlobalIndex col_dof : col_dofs) {
        if (col_dof >= 0 && col_dof < builder_.n_cols_) {
            ++valid_cols;
        }
    }

    for (GlobalIndex row_dof : row_dofs) {
        if (row_dof >= 0 && row_dof < builder_.n_rows_) {
            if (!local_counts_.empty()) {
                local_counts_[static_cast<std::size_t>(row_dof)] += valid_cols;
            } else {
                builder_.row_counts_[static_cast<std::size_t>(row_dof)] += valid_cols;
            }
        }
    }
}

SparsityTwoPassBuilder::Filler::Filler(SparsityTwoPassBuilder& builder, int thread_id)
    : builder_(builder), thread_id_(thread_id)
{
}

void SparsityTwoPassBuilder::Filler::addEntry(GlobalIndex row, GlobalIndex col) {
    if (row >= 0 && row < builder_.n_rows_ && col >= 0 && col < builder_.n_cols_) {
        local_entries_.emplace_back(row, col);
    }
}

void SparsityTwoPassBuilder::Filler::addEntries(GlobalIndex row,
                                                 std::span<const GlobalIndex> cols) {
    if (row >= 0 && row < builder_.n_rows_) {
        for (GlobalIndex col : cols) {
            if (col >= 0 && col < builder_.n_cols_) {
                local_entries_.emplace_back(row, col);
            }
        }
    }
}

void SparsityTwoPassBuilder::Filler::addElementCouplings(std::span<const GlobalIndex> dofs) {
    for (GlobalIndex row_dof : dofs) {
        if (row_dof >= 0 && row_dof < builder_.n_rows_) {
            for (GlobalIndex col_dof : dofs) {
                if (col_dof >= 0 && col_dof < builder_.n_cols_) {
                    local_entries_.emplace_back(row_dof, col_dof);
                }
            }
        }
    }
}

void SparsityTwoPassBuilder::Filler::addElementCouplings(
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs)
{
    for (GlobalIndex row_dof : row_dofs) {
        if (row_dof >= 0 && row_dof < builder_.n_rows_) {
            for (GlobalIndex col_dof : col_dofs) {
                if (col_dof >= 0 && col_dof < builder_.n_cols_) {
                    local_entries_.emplace_back(row_dof, col_dof);
                }
            }
        }
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

void SparsityTwoPassBuilder::checkPhase(TwoPassPhase expected, const char* operation) const {
    if (phase_ != expected) {
        std::string msg = std::string(operation) + " called in wrong phase. Expected: ";
        switch (expected) {
            case TwoPassPhase::Initial: msg += "Initial"; break;
            case TwoPassPhase::Counting: msg += "Counting"; break;
            case TwoPassPhase::Allocated: msg += "Allocated"; break;
            case TwoPassPhase::Filling: msg += "Filling"; break;
            case TwoPassPhase::Finalized: msg += "Finalized"; break;
        }
        msg += ", Current: ";
        switch (phase_) {
            case TwoPassPhase::Initial: msg += "Initial"; break;
            case TwoPassPhase::Counting: msg += "Counting"; break;
            case TwoPassPhase::Allocated: msg += "Allocated"; break;
            case TwoPassPhase::Filling: msg += "Filling"; break;
            case TwoPassPhase::Finalized: msg += "Finalized"; break;
        }
        FE_THROW(InvalidArgumentException, msg);
    }
}

void SparsityTwoPassBuilder::checkPhaseForCounting(const char* operation) const {
    if (phase_ != TwoPassPhase::Initial && phase_ != TwoPassPhase::Counting) {
        std::string msg = std::string(operation) + " called in wrong phase. Expected: Initial or Counting, Current: ";
        switch (phase_) {
            case TwoPassPhase::Initial: msg += "Initial"; break;
            case TwoPassPhase::Counting: msg += "Counting"; break;
            case TwoPassPhase::Allocated: msg += "Allocated"; break;
            case TwoPassPhase::Filling: msg += "Filling"; break;
            case TwoPassPhase::Finalized: msg += "Finalized"; break;
        }
        FE_THROW(InvalidArgumentException, msg);
    }
}

void SparsityTwoPassBuilder::checkRowIndex(GlobalIndex row) const {
    FE_CHECK_ARG(row >= 0 && row < n_rows_,
                 "Row index " + std::to_string(row) + " out of range [0, " +
                 std::to_string(n_rows_) + ")");
}

void SparsityTwoPassBuilder::checkColIndex(GlobalIndex col) const {
    FE_CHECK_ARG(col >= 0 && col < n_cols_,
                 "Column index " + std::to_string(col) + " out of range [0, " +
                 std::to_string(n_cols_) + ")");
}

void SparsityTwoPassBuilder::addEntryInternal(GlobalIndex row, GlobalIndex col) {
    // Internal version that skips phase check - for use in finalize()
    if (row >= 0 && row < n_rows_ && col >= 0 && col < n_cols_) {
        auto row_idx = static_cast<std::size_t>(row);
        GlobalIndex& pos = row_positions_[row_idx];
        GlobalIndex end = row_offsets_[row_idx + 1];

        if (pos < end) {
            col_indices_[static_cast<std::size_t>(pos)] = col;
            ++pos;
        }
    }
}

void SparsityTwoPassBuilder::mergeThreadCounts() {
    if (thread_counts_.empty()) return;

    for (auto& thread_count_ptr : thread_counts_) {
        if (thread_count_ptr) {
            const auto& thread_counts = *thread_count_ptr;
            for (std::size_t i = 0; i < thread_counts.size(); ++i) {
                row_counts_[i] += thread_counts[i];
            }
        }
    }
    thread_counts_.clear();
}

void SparsityTwoPassBuilder::mergeThreadEntries() {
    if (thread_entries_.empty()) return;

    // Sort entries by row for efficient insertion
    for (auto& entry_ptr : thread_entries_) {
        if (entry_ptr && !entry_ptr->empty()) {
            auto& entries = *entry_ptr;
            // Insert into main storage
            for (const auto& [row, col] : entries) {
                addEntry(row, col);
            }
        }
    }
    thread_entries_.clear();
}

void SparsityTwoPassBuilder::sortAndDeduplicate() {
    // Sort and deduplicate each row
    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        auto row_idx = static_cast<std::size_t>(row);
        GlobalIndex start = row_offsets_[row_idx];
        GlobalIndex end = row_positions_[row_idx];

        if (end <= start) continue;

        auto begin_it = col_indices_.begin() + start;
        auto end_it = col_indices_.begin() + end;

        // Sort
        std::sort(begin_it, end_it);

        // Remove duplicates
        auto new_end = std::unique(begin_it, end_it);
        GlobalIndex new_count = static_cast<GlobalIndex>(new_end - begin_it);

        // Update position to reflect deduplicated count
        row_positions_[row_idx] = start + new_count;
    }
}

void SparsityTwoPassBuilder::updateStats() {
    stats_.fill_memory_bytes = currentMemoryBytes();
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
