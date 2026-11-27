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
    overflow_entries_.clear();
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

    addEntryInternal(row, col);
}

void SparsityTwoPassBuilder::addEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    checkPhase(TwoPassPhase::Allocated, "addEntries");
    checkRowIndex(row);

    for (GlobalIndex col : cols) {
        if (col >= 0 && col < n_cols_) addEntryInternal(row, col);
    }
}

void SparsityTwoPassBuilder::addElementCouplings(std::span<const GlobalIndex> dofs) {
    addElementCouplings(dofs, dofs);
}

void SparsityTwoPassBuilder::addElementCouplings(std::span<const GlobalIndex> row_dofs,
                                                  std::span<const GlobalIndex> col_dofs) {
    checkPhase(TwoPassPhase::Allocated, "addElementCouplings");

    for (GlobalIndex row_dof : row_dofs) {
        if (row_dof >= 0 && row_dof < n_rows_) addEntries(row_dof, col_dofs);
    }
}

// ============================================================================
// Parallel operations
// ============================================================================

void SparsityTwoPassBuilder::countParallel(CountCallback callback) {
    checkPhaseForCounting("countParallel");

    const int workers = std::max(1, options_.num_threads);
    std::vector<std::vector<GlobalIndex>> counts(static_cast<std::size_t>(workers));
    for (int t = 0; t < workers; ++t) {
        Counter counter(*this, t);
        counter.local_counts_.resize(static_cast<std::size_t>(n_rows_));
        callback(t, counter);
        counts[t] = std::move(counter.local_counts_);
    }
    for (const auto& local : counts) {
        for (std::size_t row = 0; row < local.size(); ++row) row_counts_[row] += local[row];
    }
}

void SparsityTwoPassBuilder::fillParallel(FillCallback callback) {
    checkPhase(TwoPassPhase::Allocated, "fillParallel");
    const int workers = std::max(1, options_.num_threads);
    std::vector<std::vector<std::pair<GlobalIndex, GlobalIndex>>> entries(static_cast<std::size_t>(workers));
    for (int t = 0; t < workers; ++t) {
        Filler filler(*this, t);
        filler.buffered_ = true;
        callback(t, filler);
        entries[t] = std::move(filler.local_entries_);
    }
    for (const auto& local : entries) {
        for (const auto& [row, col] : local) addEntryInternal(row, col);
    }
}

// ============================================================================
// Finalization
// ============================================================================

SparsityPattern SparsityTwoPassBuilder::finalize() {
    FE_CHECK_ARG(canFinalize(), "Builder not ready for finalization");

    SparsityPattern pattern(n_rows_, n_cols_);
    GlobalIndex inserted = static_cast<GlobalIndex>(overflow_entries_.size());
    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        const auto i = static_cast<std::size_t>(row);
        const auto begin = row_offsets_[i];
        const auto count = row_positions_[i] - begin;
        inserted += count;
        pattern.addEntries(row, std::span<const GlobalIndex>(col_indices_).subspan(begin, count));
    }
    for (const auto& [row, col] : overflow_entries_) pattern.addEntry(row, col);
    if (options_.ensure_diagonal && n_rows_ == n_cols_) {
        pattern.ensureDiagonal();
        inserted += n_rows_;
    }
    if (options_.ensure_non_empty_rows) {
        const auto before = pattern.getNnz();
        pattern.ensureNonEmptyRows();
        inserted += pattern.getNnz() - before;
    }
    pattern.finalize();
    stats_.actual_nnz = pattern.getNnz();
    stats_.duplicates_removed = inserted - stats_.actual_nnz;
    stats_.allocated_nnz = stats_.estimated_nnz + static_cast<GlobalIndex>(overflow_entries_.capacity());
    stats_.fill_memory_bytes = currentMemoryBytes();
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
    bytes += overflow_entries_.capacity() * sizeof(overflow_entries_[0]);
    return bytes;
}

// ============================================================================
// Helper class implementations
// ============================================================================

SparsityTwoPassBuilder::Counter::Counter(SparsityTwoPassBuilder& builder, int)
    : builder_(builder)
{}

void SparsityTwoPassBuilder::Counter::countEntry(GlobalIndex row, GlobalIndex col) {
    countEntries(row, std::span<const GlobalIndex>(&col, 1));
}

void SparsityTwoPassBuilder::Counter::countEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    countElementCouplings(std::span<const GlobalIndex>(&row, 1), cols);
}

void SparsityTwoPassBuilder::Counter::countElementCouplings(std::span<const GlobalIndex> dofs) {
    countElementCouplings(dofs, dofs);
}

void SparsityTwoPassBuilder::Counter::countElementCouplings(
    std::span<const GlobalIndex> rows, std::span<const GlobalIndex> cols)
{
    builder_.checkPhaseForCounting("countElementCouplings");
    const auto valid = std::count_if(cols.begin(), cols.end(),
        [&](GlobalIndex col) { return col >= 0 && col < builder_.n_cols_; });
    auto& counts = local_counts_.empty() ? builder_.row_counts_ : local_counts_;
    for (auto row : rows) {
        if (row >= 0 && row < builder_.n_rows_) counts[static_cast<std::size_t>(row)] += valid;
    }
}

SparsityTwoPassBuilder::Filler::Filler(SparsityTwoPassBuilder& builder, int)
    : builder_(builder)
{}

void SparsityTwoPassBuilder::Filler::addEntry(GlobalIndex row, GlobalIndex col) {
    if (row < 0 || row >= builder_.n_rows_ || col < 0 || col >= builder_.n_cols_) return;
    if (buffered_) local_entries_.emplace_back(row, col);
    else builder_.addEntry(row, col);
}

void SparsityTwoPassBuilder::Filler::addEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    for (auto col : cols) addEntry(row, col);
}

void SparsityTwoPassBuilder::Filler::addElementCouplings(std::span<const GlobalIndex> dofs) {
    addElementCouplings(dofs, dofs);
}

void SparsityTwoPassBuilder::Filler::addElementCouplings(
    std::span<const GlobalIndex> rows, std::span<const GlobalIndex> cols)
{
    for (auto row : rows) addEntries(row, cols);
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
    const auto i = static_cast<std::size_t>(row);
    auto& position = row_positions_[i];
    if (position < row_offsets_[i + 1]) {
        col_indices_[static_cast<std::size_t>(position++)] = col;
    } else {
        overflow_entries_.emplace_back(row, col);
    }
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
