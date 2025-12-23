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

#include "CompressedSparsity.h"
#include <algorithm>
#include <numeric>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// ThreadLocalBuffer implementation
// ============================================================================

void ThreadLocalBuffer::addEntry(GlobalIndex row, GlobalIndex col) {
    entries_.emplace_back(row, col);
}

void ThreadLocalBuffer::addEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    for (GlobalIndex col : cols) {
        entries_.emplace_back(row, col);
    }
}

void ThreadLocalBuffer::clear() {
    entries_.clear();
}

// ============================================================================
// CompressedSparsity construction
// ============================================================================

CompressedSparsity::CompressedSparsity(GlobalIndex n_rows, GlobalIndex n_cols)
    : n_rows_(n_rows),
      n_cols_(n_cols < 0 ? n_rows : n_cols),
      row_cols_(static_cast<std::size_t>(n_rows)),
      row_compressed_(static_cast<std::size_t>(n_rows), false)
{
    FE_CHECK_ARG(n_rows >= 0, "Number of rows must be non-negative");
    stats_.n_rows = n_rows_;
    stats_.n_cols = n_cols_;
}

CompressedSparsity::CompressedSparsity(GlobalIndex n_rows, GlobalIndex n_cols,
                                       const CompressedSparsityOptions& options)
    : n_rows_(n_rows),
      n_cols_(n_cols < 0 ? n_rows : n_cols),
      options_(options),
      row_cols_(static_cast<std::size_t>(n_rows)),
      row_compressed_(static_cast<std::size_t>(n_rows), false)
{
    FE_CHECK_ARG(n_rows >= 0, "Number of rows must be non-negative");
    stats_.n_rows = n_rows_;
    stats_.n_cols = n_cols_;

    if (options_.mode == CompressionMode::TwoPass) {
        in_counting_phase_ = true;
        row_counts_.resize(static_cast<std::size_t>(n_rows), 0);
    }

    if (options_.enable_parallel && options_.num_threads > 1) {
        initThreadBuffers(options_.num_threads);
    }
}

// ============================================================================
// Configuration
// ============================================================================

void CompressedSparsity::resize(GlobalIndex n_rows, GlobalIndex n_cols) {
    n_rows_ = n_rows;
    n_cols_ = n_cols < 0 ? n_rows : n_cols;
    row_cols_.clear();
    row_cols_.resize(static_cast<std::size_t>(n_rows_));
    row_compressed_.clear();
    row_compressed_.resize(static_cast<std::size_t>(n_rows_), false);
    row_counts_.clear();
    if (options_.mode == CompressionMode::TwoPass) {
        row_counts_.resize(static_cast<std::size_t>(n_rows_), 0);
        in_counting_phase_ = true;
    }
    total_insertions_ = 0;
    stats_ = CompressionStats{};
    stats_.n_rows = n_rows_;
    stats_.n_cols = n_cols_;
}

void CompressedSparsity::clear() {
    for (auto& row : row_cols_) {
        row.clear();
    }
    std::fill(row_compressed_.begin(), row_compressed_.end(), false);
    if (!row_counts_.empty()) {
        std::fill(row_counts_.begin(), row_counts_.end(), GlobalIndex{0});
    }
    in_counting_phase_ = (options_.mode == CompressionMode::TwoPass);
    counting_finalized_ = false;
    total_insertions_ = 0;
    stats_ = CompressionStats{};
    stats_.n_rows = n_rows_;
    stats_.n_cols = n_cols_;
}

void CompressedSparsity::reserve(GlobalIndex entries_per_row) {
    for (auto& row : row_cols_) {
        row.reserve(static_cast<std::size_t>(entries_per_row));
    }
}

void CompressedSparsity::reserve(std::span<const GlobalIndex> row_reserves) {
    FE_CHECK_ARG(static_cast<GlobalIndex>(row_reserves.size()) == n_rows_,
                 "Row reserves size mismatch");
    for (GlobalIndex i = 0; i < n_rows_; ++i) {
        row_cols_[static_cast<std::size_t>(i)].reserve(
            static_cast<std::size_t>(row_reserves[static_cast<std::size_t>(i)]));
    }
}

// ============================================================================
// Single-pass insertion
// ============================================================================

void CompressedSparsity::addEntry(GlobalIndex row, GlobalIndex col) {
    checkRowIndex(row);
    checkColIndex(col);

    if (in_counting_phase_ && !counting_finalized_) {
        countEntry(row, col);
        return;
    }

    row_cols_[static_cast<std::size_t>(row)].push_back(col);
    row_compressed_[static_cast<std::size_t>(row)] = false;
    total_insertions_.fetch_add(1, std::memory_order_relaxed);

    // Incremental compression check
    if (options_.mode == CompressionMode::Incremental) {
        auto& rc = row_cols_[static_cast<std::size_t>(row)];
        if (rc.size() > options_.compression_threshold) {
            compressRow(row);
        }
    }
}

void CompressedSparsity::addEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    checkRowIndex(row);

    if (in_counting_phase_ && !counting_finalized_) {
        countEntries(row, cols);
        return;
    }

    auto& rc = row_cols_[static_cast<std::size_t>(row)];
    for (GlobalIndex col : cols) {
        if (col >= 0 && col < n_cols_) {
            rc.push_back(col);
        }
    }
    row_compressed_[static_cast<std::size_t>(row)] = false;
    total_insertions_.fetch_add(cols.size(), std::memory_order_relaxed);

    // Incremental compression check
    if (options_.mode == CompressionMode::Incremental) {
        if (rc.size() > options_.compression_threshold) {
            compressRow(row);
        }
    }
}

void CompressedSparsity::addElementCouplings(std::span<const GlobalIndex> dofs) {
    if (in_counting_phase_ && !counting_finalized_) {
        countElementCouplings(dofs);
        return;
    }

    for (GlobalIndex row_dof : dofs) {
        if (row_dof >= 0 && row_dof < n_rows_) {
            addEntries(row_dof, dofs);
        }
    }
}

void CompressedSparsity::addElementCouplings(std::span<const GlobalIndex> row_dofs,
                                              std::span<const GlobalIndex> col_dofs) {
    for (GlobalIndex row_dof : row_dofs) {
        if (row_dof >= 0 && row_dof < n_rows_) {
            addEntries(row_dof, col_dofs);
        }
    }
}

// ============================================================================
// Two-pass construction
// ============================================================================

void CompressedSparsity::countEntry(GlobalIndex row, GlobalIndex col) {
    checkRowIndex(row);
    if (col >= 0 && col < n_cols_) {
        row_counts_[static_cast<std::size_t>(row)]++;
    }
}

void CompressedSparsity::countEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    checkRowIndex(row);
    for (GlobalIndex col : cols) {
        if (col >= 0 && col < n_cols_) {
            row_counts_[static_cast<std::size_t>(row)]++;
        }
    }
}

void CompressedSparsity::countElementCouplings(std::span<const GlobalIndex> dofs) {
    for (GlobalIndex row_dof : dofs) {
        if (row_dof >= 0 && row_dof < n_rows_) {
            for (GlobalIndex col_dof : dofs) {
                if (col_dof >= 0 && col_dof < n_cols_) {
                    row_counts_[static_cast<std::size_t>(row_dof)]++;
                }
            }
        }
    }
}

void CompressedSparsity::finalizeCounting() {
    FE_THROW_IF(!in_counting_phase_, InvalidArgumentException,
                "Not in counting phase");

    // Reserve based on counts
    for (GlobalIndex i = 0; i < n_rows_; ++i) {
        row_cols_[static_cast<std::size_t>(i)].reserve(
            static_cast<std::size_t>(row_counts_[static_cast<std::size_t>(i)]));
    }

    counting_finalized_ = true;
    in_counting_phase_ = false;  // Now in filling phase
}

// ============================================================================
// Parallel insertion
// ============================================================================

void CompressedSparsity::initThreadBuffers(int num_threads) {
    FE_CHECK_ARG(num_threads > 0, "Number of threads must be positive");
    thread_buffers_.clear();
    thread_buffers_.reserve(static_cast<std::size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) {
        thread_buffers_.push_back(std::make_unique<ThreadLocalBuffer>());
    }
}

void CompressedSparsity::addEntryThreaded(int thread_id, GlobalIndex row, GlobalIndex col) {
    FE_CHECK_ARG(thread_id >= 0 &&
                 static_cast<std::size_t>(thread_id) < thread_buffers_.size(),
                 "Invalid thread ID");
    thread_buffers_[static_cast<std::size_t>(thread_id)]->addEntry(row, col);
}

void CompressedSparsity::addEntriesThreaded(int thread_id, GlobalIndex row,
                                             std::span<const GlobalIndex> cols) {
    FE_CHECK_ARG(thread_id >= 0 &&
                 static_cast<std::size_t>(thread_id) < thread_buffers_.size(),
                 "Invalid thread ID");
    thread_buffers_[static_cast<std::size_t>(thread_id)]->addEntries(row, cols);
}

void CompressedSparsity::addElementCouplingsThreaded(int thread_id,
                                                      std::span<const GlobalIndex> dofs) {
    FE_CHECK_ARG(thread_id >= 0 &&
                 static_cast<std::size_t>(thread_id) < thread_buffers_.size(),
                 "Invalid thread ID");

    auto* buffer = thread_buffers_[static_cast<std::size_t>(thread_id)].get();
    for (GlobalIndex row_dof : dofs) {
        if (row_dof >= 0 && row_dof < n_rows_) {
            for (GlobalIndex col_dof : dofs) {
                if (col_dof >= 0 && col_dof < n_cols_) {
                    buffer->addEntry(row_dof, col_dof);
                }
            }
        }
    }
}

void CompressedSparsity::mergeThreadBuffers() {
    // Collect all entries from thread buffers
    std::vector<std::pair<GlobalIndex, GlobalIndex>> all_entries;

    for (auto& buffer : thread_buffers_) {
        const auto& entries = buffer->entries();
        all_entries.insert(all_entries.end(), entries.begin(), entries.end());
        buffer->clear();
    }

    // Sort by row for deterministic insertion
    std::sort(all_entries.begin(), all_entries.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first ||
                         (a.first == b.first && a.second < b.second);
              });

    // Insert into row storage
    for (const auto& [row, col] : all_entries) {
        if (row >= 0 && row < n_rows_ && col >= 0 && col < n_cols_) {
            row_cols_[static_cast<std::size_t>(row)].push_back(col);
            row_compressed_[static_cast<std::size_t>(row)] = false;
        }
    }

    total_insertions_.fetch_add(all_entries.size(), std::memory_order_relaxed);
}

// ============================================================================
// Compression and finalization
// ============================================================================

void CompressedSparsity::compress() {
    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        compressRow(row);
    }
    stats_.num_compressions++;
    updateStats();
}

void CompressedSparsity::sortAndDeduplicate(std::vector<GlobalIndex>& cols) {
    if (cols.empty()) return;

    // Sort
    std::sort(cols.begin(), cols.end());

    // Remove duplicates
    auto last = std::unique(cols.begin(), cols.end());
    cols.erase(last, cols.end());
}

void CompressedSparsity::compressRow(GlobalIndex row) {
    auto& cols = row_cols_[static_cast<std::size_t>(row)];
    if (cols.empty() || row_compressed_[static_cast<std::size_t>(row)]) {
        return;
    }

    std::size_t before = cols.size();
    sortAndDeduplicate(cols);
    std::size_t after = cols.size();

    row_compressed_[static_cast<std::size_t>(row)] = true;

    // Track duplicates removed (thread-safe update)
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.duplicates_removed += static_cast<GlobalIndex>(before - after);
}

SparsityPattern CompressedSparsity::toSparsityPattern() {
    // Compress all rows
    compress();

    SparsityPattern pattern(n_rows_, n_cols_);

    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        const auto& cols = row_cols_[static_cast<std::size_t>(row)];
        if (!cols.empty()) {
            pattern.addEntries(row, cols);
        }
    }

    pattern.finalize();

    // Update final stats
    stats_.unique_entries = pattern.getNnz();
    stats_.total_insertions = static_cast<GlobalIndex>(total_insertions_.load());
    if (stats_.unique_entries > 0) {
        stats_.compression_ratio = static_cast<double>(stats_.total_insertions) /
                                   static_cast<double>(stats_.unique_entries);
    }
    stats_.final_memory_bytes = pattern.memoryUsageBytes();

    return pattern;
}

SparsityPattern CompressedSparsity::toBuildingPattern() const {
    SparsityPattern pattern(n_rows_, n_cols_);

    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        const auto& cols = row_cols_[static_cast<std::size_t>(row)];
        if (!cols.empty()) {
            // Need to deduplicate if not already compressed
            if (row_compressed_[static_cast<std::size_t>(row)]) {
                pattern.addEntries(row, cols);
            } else {
                // Sort and dedupe a copy
                std::vector<GlobalIndex> sorted_cols(cols);
                std::sort(sorted_cols.begin(), sorted_cols.end());
                auto last = std::unique(sorted_cols.begin(), sorted_cols.end());
                sorted_cols.erase(last, sorted_cols.end());
                pattern.addEntries(row, sorted_cols);
            }
        }
    }

    return pattern;
}

// ============================================================================
// Query methods
// ============================================================================

GlobalIndex CompressedSparsity::getApproximateNnz() const {
    GlobalIndex total = 0;
    for (const auto& row : row_cols_) {
        total += static_cast<GlobalIndex>(row.size());
    }
    return total;
}

GlobalIndex CompressedSparsity::getExactNnz() {
    compress();

    GlobalIndex total = 0;
    for (const auto& row : row_cols_) {
        total += static_cast<GlobalIndex>(row.size());
    }
    return total;
}

std::size_t CompressedSparsity::currentMemoryBytes() const noexcept {
    std::size_t bytes = sizeof(*this);
    bytes += row_cols_.capacity() * sizeof(std::vector<GlobalIndex>);
    for (const auto& row : row_cols_) {
        bytes += row.capacity() * sizeof(GlobalIndex);
    }
    bytes += row_compressed_.capacity() * sizeof(bool);
    bytes += row_counts_.capacity() * sizeof(GlobalIndex);
    bytes += thread_buffers_.capacity() * sizeof(std::unique_ptr<ThreadLocalBuffer>);
    for (const auto& buf : thread_buffers_) {
        if (buf) {
            bytes += sizeof(ThreadLocalBuffer);
            bytes += buf->entries().capacity() * sizeof(std::pair<GlobalIndex, GlobalIndex>);
        }
    }
    return bytes;
}

// ============================================================================
// Private helpers
// ============================================================================

void CompressedSparsity::checkRowIndex(GlobalIndex row) const {
    FE_CHECK_ARG(row >= 0 && row < n_rows_,
                 "Row index " + std::to_string(row) + " out of range [0, " +
                 std::to_string(n_rows_) + ")");
}

void CompressedSparsity::checkColIndex(GlobalIndex col) const {
    FE_CHECK_ARG(col >= 0 && col < n_cols_,
                 "Column index " + std::to_string(col) + " out of range [0, " +
                 std::to_string(n_cols_) + ")");
}

void CompressedSparsity::updateStats() {
    stats_.n_rows = n_rows_;
    stats_.n_cols = n_cols_;
    stats_.total_insertions = static_cast<GlobalIndex>(total_insertions_.load(std::memory_order_relaxed));
    stats_.peak_memory_bytes = std::max(stats_.peak_memory_bytes, currentMemoryBytes());

    // Compute unique entries
    GlobalIndex unique = 0;
    for (const auto& row : row_cols_) {
        unique += static_cast<GlobalIndex>(row.size());
    }
    stats_.unique_entries = unique;
    stats_.duplicates_removed = stats_.total_insertions - unique;
    if (unique > 0) {
        stats_.compression_ratio = static_cast<double>(stats_.total_insertions) / static_cast<double>(unique);
    }
    stats_.final_memory_bytes = currentMemoryBytes();
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
