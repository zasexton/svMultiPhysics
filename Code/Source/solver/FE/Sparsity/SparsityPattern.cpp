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

#include "SparsityPattern.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Construction and lifecycle
// ============================================================================

SparsityPattern::SparsityPattern(GlobalIndex n_rows, GlobalIndex n_cols)
    : n_rows_(n_rows),
      n_cols_(n_cols < 0 ? n_rows : n_cols),
      row_sets_(static_cast<std::size_t>(n_rows)),
      state_(SparsityState::Building)
{
    FE_CHECK_ARG(n_rows >= 0, "Number of rows must be non-negative");
    FE_CHECK_ARG(n_cols_ >= 0, "Number of columns must be non-negative");
}

SparsityPattern::SparsityPattern(SparsityPattern&& other) noexcept
    : n_rows_(other.n_rows_),
      n_cols_(other.n_cols_),
      row_sets_(std::move(other.row_sets_)),
      row_ptr_(std::move(other.row_ptr_)),
      col_idx_(std::move(other.col_idx_)),
      cached_nnz_(other.cached_nnz_),
      state_(other.state_.load(std::memory_order_acquire))
{
    other.n_rows_ = 0;
    other.n_cols_ = 0;
    other.cached_nnz_ = -1;
    other.state_.store(SparsityState::Building, std::memory_order_release);
}

SparsityPattern& SparsityPattern::operator=(SparsityPattern&& other) noexcept {
    if (this != &other) {
        n_rows_ = other.n_rows_;
        n_cols_ = other.n_cols_;
        row_sets_ = std::move(other.row_sets_);
        row_ptr_ = std::move(other.row_ptr_);
        col_idx_ = std::move(other.col_idx_);
        cached_nnz_ = other.cached_nnz_;
        state_.store(other.state_.load(std::memory_order_acquire), std::memory_order_release);

        other.n_rows_ = 0;
        other.n_cols_ = 0;
        other.cached_nnz_ = -1;
        other.state_.store(SparsityState::Building, std::memory_order_release);
    }
    return *this;
}

SparsityPattern::SparsityPattern(const SparsityPattern& other)
    : n_rows_(other.n_rows_),
      n_cols_(other.n_cols_),
      state_(SparsityState::Building)
{
    // Always copy to building state - reconstruct from CSR if needed
    if (other.isFinalized()) {
        row_sets_.resize(static_cast<std::size_t>(n_rows_));
        for (GlobalIndex row = 0; row < n_rows_; ++row) {
            auto span = other.getRowSpan(row);
            row_sets_[static_cast<std::size_t>(row)].insert(span.begin(), span.end());
        }
    } else {
        row_sets_ = other.row_sets_;
    }
    cached_nnz_ = -1;
}

SparsityPattern& SparsityPattern::operator=(const SparsityPattern& other) {
    if (this != &other) {
        n_rows_ = other.n_rows_;
        n_cols_ = other.n_cols_;
        row_ptr_.clear();
        col_idx_.clear();
        cached_nnz_ = -1;
        state_.store(SparsityState::Building, std::memory_order_release);

        if (other.isFinalized()) {
            row_sets_.clear();
            row_sets_.resize(static_cast<std::size_t>(n_rows_));
            for (GlobalIndex row = 0; row < n_rows_; ++row) {
                auto span = other.getRowSpan(row);
                row_sets_[static_cast<std::size_t>(row)].insert(span.begin(), span.end());
            }
        } else {
            row_sets_ = other.row_sets_;
        }
    }
    return *this;
}

SparsityPattern SparsityPattern::cloneFinalized() const {
    if (!isFinalized()) {
        SparsityPattern copy(*this);
        copy.finalize();
        return copy;
    }

    SparsityPattern copy;
    copy.n_rows_ = n_rows_;
    copy.n_cols_ = n_cols_;
    copy.row_ptr_ = row_ptr_;
    copy.col_idx_ = col_idx_;
    copy.cached_nnz_ = cached_nnz_;
    copy.state_.store(SparsityState::Finalized, std::memory_order_release);
    return copy;
}

// ============================================================================
// Setup methods (Building state only)
// ============================================================================

void SparsityPattern::resize(GlobalIndex n_rows, GlobalIndex n_cols) {
    checkNotFinalized();
    FE_CHECK_ARG(n_rows >= 0, "Number of rows must be non-negative");

    n_rows_ = n_rows;
    n_cols_ = (n_cols < 0) ? n_rows : n_cols;

    row_sets_.clear();
    row_sets_.resize(static_cast<std::size_t>(n_rows_));
    row_ptr_.clear();
    col_idx_.clear();
    cached_nnz_ = -1;
}

void SparsityPattern::clear() {
    row_sets_.clear();
    row_sets_.resize(static_cast<std::size_t>(n_rows_));
    row_ptr_.clear();
    col_idx_.clear();
    cached_nnz_ = -1;
    state_.store(SparsityState::Building, std::memory_order_release);
}

void SparsityPattern::reservePerRow(GlobalIndex /*entries_per_row*/) {
    // std::set doesn't have reserve(), but this is a hint for future optimization
    // Could switch to sorted vector with reserve
}

void SparsityPattern::addEntry(GlobalIndex row, GlobalIndex col) {
    checkNotFinalized();
    checkIndices(row, col);

    row_sets_[static_cast<std::size_t>(row)].insert(col);
    cached_nnz_ = -1;  // Invalidate cache
}

void SparsityPattern::addEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    checkNotFinalized();
    checkRowIndex(row);

    auto& row_set = row_sets_[static_cast<std::size_t>(row)];
    for (GlobalIndex col : cols) {
        checkColIndex(col);
        row_set.insert(col);
    }
    cached_nnz_ = -1;
}

void SparsityPattern::addBlock(GlobalIndex row_begin, GlobalIndex row_end,
                               GlobalIndex col_begin, GlobalIndex col_end) {
    checkNotFinalized();
    FE_CHECK_ARG(row_begin >= 0 && row_begin <= row_end && row_end <= n_rows_,
                 "Invalid row range");
    FE_CHECK_ARG(col_begin >= 0 && col_begin <= col_end && col_end <= n_cols_,
                 "Invalid column range");

    for (GlobalIndex row = row_begin; row < row_end; ++row) {
        auto& row_set = row_sets_[static_cast<std::size_t>(row)];
        for (GlobalIndex col = col_begin; col < col_end; ++col) {
            row_set.insert(col);
        }
    }
    cached_nnz_ = -1;
}

void SparsityPattern::addElementCouplings(std::span<const GlobalIndex> dofs) {
    checkNotFinalized();

    for (GlobalIndex row_dof : dofs) {
        if (row_dof < 0 || row_dof >= n_rows_) continue;  // Skip out-of-range
        auto& row_set = row_sets_[static_cast<std::size_t>(row_dof)];
        for (GlobalIndex col_dof : dofs) {
            if (col_dof >= 0 && col_dof < n_cols_) {
                row_set.insert(col_dof);
            }
        }
    }
    cached_nnz_ = -1;
}

void SparsityPattern::addElementCouplings(std::span<const GlobalIndex> row_dofs,
                                          std::span<const GlobalIndex> col_dofs) {
    checkNotFinalized();

    for (GlobalIndex row_dof : row_dofs) {
        if (row_dof < 0 || row_dof >= n_rows_) continue;
        auto& row_set = row_sets_[static_cast<std::size_t>(row_dof)];
        for (GlobalIndex col_dof : col_dofs) {
            if (col_dof >= 0 && col_dof < n_cols_) {
                row_set.insert(col_dof);
            }
        }
    }
    cached_nnz_ = -1;
}

void SparsityPattern::ensureDiagonal() {
    checkNotFinalized();
    FE_CHECK_ARG(isSquare(), "ensureDiagonal() requires square pattern");

    for (GlobalIndex i = 0; i < n_rows_; ++i) {
        row_sets_[static_cast<std::size_t>(i)].insert(i);
    }
    cached_nnz_ = -1;
}

void SparsityPattern::ensureNonEmptyRows() {
    checkNotFinalized();

    for (GlobalIndex i = 0; i < n_rows_; ++i) {
        if (row_sets_[static_cast<std::size_t>(i)].empty()) {
            // Add diagonal for square, column 0 for rectangular
            GlobalIndex col = isSquare() ? i : 0;
            if (col < n_cols_) {
                row_sets_[static_cast<std::size_t>(i)].insert(col);
            }
        }
    }
    cached_nnz_ = -1;
}

void SparsityPattern::finalize() {
    checkNotFinalized();

    // Count total NNZ
    GlobalIndex nnz = 0;
    for (const auto& row_set : row_sets_) {
        nnz += static_cast<GlobalIndex>(row_set.size());
    }

    // Allocate CSR storage
    row_ptr_.resize(static_cast<std::size_t>(n_rows_) + 1);
    col_idx_.resize(static_cast<std::size_t>(nnz));

    // Build CSR arrays - sets are already sorted
    row_ptr_[0] = 0;
    GlobalIndex idx = 0;
    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        const auto& row_set = row_sets_[static_cast<std::size_t>(row)];
        for (GlobalIndex col : row_set) {
            col_idx_[static_cast<std::size_t>(idx++)] = col;
        }
        row_ptr_[static_cast<std::size_t>(row) + 1] = idx;
    }

    // Cache NNZ
    cached_nnz_ = nnz;

    // Clear building storage to free memory
    row_sets_.clear();
    row_sets_.shrink_to_fit();

    // Transition to finalized state
    state_.store(SparsityState::Finalized, std::memory_order_release);
}

// ============================================================================
// Query methods
// ============================================================================

GlobalIndex SparsityPattern::getNnz() const {
    if (isFinalized()) {
        return cached_nnz_;
    }

    // During building, count from sets
    GlobalIndex nnz = 0;
    for (const auto& row_set : row_sets_) {
        nnz += static_cast<GlobalIndex>(row_set.size());
    }
    return nnz;
}

GlobalIndex SparsityPattern::getRowNnz(GlobalIndex row) const {
    checkRowIndex(row);

    if (isFinalized()) {
        return row_ptr_[static_cast<std::size_t>(row) + 1] -
               row_ptr_[static_cast<std::size_t>(row)];
    }
    return static_cast<GlobalIndex>(row_sets_[static_cast<std::size_t>(row)].size());
}

RowView SparsityPattern::getRowIndices(GlobalIndex row) const {
    checkFinalized();
    checkRowIndex(row);

    GlobalIndex start = row_ptr_[static_cast<std::size_t>(row)];
    GlobalIndex end = row_ptr_[static_cast<std::size_t>(row) + 1];

    return RowView(col_idx_.data() + start, col_idx_.data() + end);
}

std::span<const GlobalIndex> SparsityPattern::getRowSpan(GlobalIndex row) const {
    checkFinalized();
    checkRowIndex(row);

    GlobalIndex start = row_ptr_[static_cast<std::size_t>(row)];
    GlobalIndex end = row_ptr_[static_cast<std::size_t>(row) + 1];

    return std::span<const GlobalIndex>(col_idx_.data() + start,
                                        static_cast<std::size_t>(end - start));
}

bool SparsityPattern::hasEntry(GlobalIndex row, GlobalIndex col) const {
    if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_) {
        return false;
    }

    if (isFinalized()) {
        auto row_span = getRowSpan(row);
        return std::binary_search(row_span.begin(), row_span.end(), col);
    }

    const auto& row_set = row_sets_[static_cast<std::size_t>(row)];
    return row_set.find(col) != row_set.end();
}

bool SparsityPattern::hasDiagonal(GlobalIndex row) const {
    if (!isSquare() || row < 0 || row >= n_rows_) {
        return false;
    }
    return hasEntry(row, row);
}

bool SparsityPattern::hasAllDiagonals() const {
    if (!isSquare()) {
        return false;
    }
    for (GlobalIndex i = 0; i < n_rows_; ++i) {
        if (!hasDiagonal(i)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// CSR format access
// ============================================================================

std::span<const GlobalIndex> SparsityPattern::getRowPtr() const {
    checkFinalized();
    return std::span<const GlobalIndex>(row_ptr_.data(), row_ptr_.size());
}

std::span<const GlobalIndex> SparsityPattern::getColIndices() const {
    checkFinalized();
    return std::span<const GlobalIndex>(col_idx_.data(), col_idx_.size());
}

const GlobalIndex* SparsityPattern::rowPtrData() const {
    checkFinalized();
    return row_ptr_.data();
}

const GlobalIndex* SparsityPattern::colIndicesData() const {
    checkFinalized();
    return col_idx_.data();
}

// ============================================================================
// Statistics and analysis
// ============================================================================

SparsityStats SparsityPattern::computeStats() const {
    SparsityStats stats;
    stats.n_rows = n_rows_;
    stats.n_cols = n_cols_;
    stats.nnz = getNnz();

    if (n_rows_ == 0) {
        return stats;
    }

    stats.min_row_nnz = std::numeric_limits<GlobalIndex>::max();
    stats.max_row_nnz = 0;
    stats.empty_rows = 0;
    stats.bandwidth = 0;
    stats.has_diagonal = isSquare();

    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        GlobalIndex row_nnz = getRowNnz(row);

        if (row_nnz == 0) {
            stats.empty_rows++;
            stats.min_row_nnz = 0;
        } else {
            stats.min_row_nnz = std::min(stats.min_row_nnz, row_nnz);
        }
        stats.max_row_nnz = std::max(stats.max_row_nnz, row_nnz);

        // Check bandwidth and diagonal
        if (isFinalized()) {
            auto row_span = getRowSpan(row);
            bool has_diag = false;
            for (GlobalIndex col : row_span) {
                GlobalIndex bw = std::abs(row - col);
                stats.bandwidth = std::max(stats.bandwidth, bw);
                if (col == row) has_diag = true;
            }
            if (isSquare() && !has_diag) {
                stats.has_diagonal = false;
            }
        } else {
            const auto& row_set = row_sets_[static_cast<std::size_t>(row)];
            for (GlobalIndex col : row_set) {
                GlobalIndex bw = std::abs(row - col);
                stats.bandwidth = std::max(stats.bandwidth, bw);
            }
            if (isSquare() && row_set.find(row) == row_set.end()) {
                stats.has_diagonal = false;
            }
        }
    }

    if (stats.min_row_nnz == std::numeric_limits<GlobalIndex>::max()) {
        stats.min_row_nnz = 0;
    }

    stats.avg_row_nnz = n_rows_ > 0 ? static_cast<double>(stats.nnz) / static_cast<double>(n_rows_) : 0.0;

    if (n_rows_ > 0 && n_cols_ > 0) {
        stats.fill_ratio = static_cast<double>(stats.nnz) /
                          (static_cast<double>(n_rows_) * static_cast<double>(n_cols_));
    }

    stats.is_symmetric_structure = isSymmetric();

    return stats;
}

GlobalIndex SparsityPattern::computeBandwidth() const {
    GlobalIndex bandwidth = 0;

    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        if (isFinalized()) {
            auto row_span = getRowSpan(row);
            for (GlobalIndex col : row_span) {
                bandwidth = std::max(bandwidth, std::abs(row - col));
            }
        } else {
            const auto& row_set = row_sets_[static_cast<std::size_t>(row)];
            for (GlobalIndex col : row_set) {
                bandwidth = std::max(bandwidth, std::abs(row - col));
            }
        }
    }

    return bandwidth;
}

bool SparsityPattern::isSymmetric() const {
    if (!isSquare()) {
        return false;
    }

    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        if (isFinalized()) {
            auto row_span = getRowSpan(row);
            for (GlobalIndex col : row_span) {
                if (!hasEntry(col, row)) {
                    return false;
                }
            }
        } else {
            const auto& row_set = row_sets_[static_cast<std::size_t>(row)];
            for (GlobalIndex col : row_set) {
                if (!hasEntry(col, row)) {
                    return false;
                }
            }
        }
    }

    return true;
}

// ============================================================================
// Validation
// ============================================================================

bool SparsityPattern::validate() const noexcept {
    try {
        // Check dimensions
        if (n_rows_ < 0 || n_cols_ < 0) {
            return false;
        }

        if (isFinalized()) {
            // Check row_ptr
            if (row_ptr_.size() != static_cast<std::size_t>(n_rows_) + 1) {
                return false;
            }
            if (row_ptr_[0] != 0) {
                return false;
            }

            // Check monotonicity and column indices
            for (GlobalIndex row = 0; row < n_rows_; ++row) {
                GlobalIndex start = row_ptr_[static_cast<std::size_t>(row)];
                GlobalIndex end = row_ptr_[static_cast<std::size_t>(row) + 1];

                if (end < start) {
                    return false;  // Not monotonic
                }

                // Check column indices are valid and sorted
                GlobalIndex prev_col = -1;
                for (GlobalIndex i = start; i < end; ++i) {
                    GlobalIndex col = col_idx_[static_cast<std::size_t>(i)];
                    if (col < 0 || col >= n_cols_) {
                        return false;
                    }
                    if (col <= prev_col) {
                        return false;  // Not strictly increasing (duplicate or unsorted)
                    }
                    prev_col = col;
                }
            }
        } else {
            // Check set sizes
            if (row_sets_.size() != static_cast<std::size_t>(n_rows_)) {
                return false;
            }

            // Check column indices are valid
            for (const auto& row_set : row_sets_) {
                for (GlobalIndex col : row_set) {
                    if (col < 0 || col >= n_cols_) {
                        return false;
                    }
                }
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

std::string SparsityPattern::validationError() const {
    std::ostringstream oss;

    if (n_rows_ < 0) {
        oss << "Negative number of rows: " << n_rows_;
        return oss.str();
    }
    if (n_cols_ < 0) {
        oss << "Negative number of columns: " << n_cols_;
        return oss.str();
    }

    if (isFinalized()) {
        if (row_ptr_.size() != static_cast<std::size_t>(n_rows_) + 1) {
            oss << "row_ptr size mismatch: " << row_ptr_.size()
                << " vs expected " << (n_rows_ + 1);
            return oss.str();
        }
        if (row_ptr_[0] != 0) {
            oss << "row_ptr[0] is not 0: " << row_ptr_[0];
            return oss.str();
        }

        for (GlobalIndex row = 0; row < n_rows_; ++row) {
            GlobalIndex start = row_ptr_[static_cast<std::size_t>(row)];
            GlobalIndex end = row_ptr_[static_cast<std::size_t>(row) + 1];

            if (end < start) {
                oss << "row_ptr not monotonic at row " << row
                    << ": row_ptr[" << row << "]=" << start
                    << ", row_ptr[" << (row+1) << "]=" << end;
                return oss.str();
            }

            GlobalIndex prev_col = -1;
            for (GlobalIndex i = start; i < end; ++i) {
                GlobalIndex col = col_idx_[static_cast<std::size_t>(i)];
                if (col < 0 || col >= n_cols_) {
                    oss << "Invalid column index " << col << " at row " << row
                        << " (valid range: [0, " << n_cols_ << "))";
                    return oss.str();
                }
                if (col <= prev_col) {
                    oss << "Column indices not strictly increasing at row " << row
                        << ": col[" << (i-1) << "]=" << prev_col
                        << ", col[" << i << "]=" << col;
                    return oss.str();
                }
                prev_col = col;
            }
        }
    } else {
        if (row_sets_.size() != static_cast<std::size_t>(n_rows_)) {
            oss << "row_sets size mismatch: " << row_sets_.size()
                << " vs expected " << n_rows_;
            return oss.str();
        }

        for (GlobalIndex row = 0; row < n_rows_; ++row) {
            for (GlobalIndex col : row_sets_[static_cast<std::size_t>(row)]) {
                if (col < 0 || col >= n_cols_) {
                    oss << "Invalid column index " << col << " at row " << row
                        << " (valid range: [0, " << n_cols_ << "))";
                    return oss.str();
                }
            }
        }
    }

    return "";  // Valid
}

// ============================================================================
// Utility methods
// ============================================================================

SparsityPattern SparsityPattern::permute(std::span<const GlobalIndex> row_perm,
                                         std::span<const GlobalIndex> col_perm) const {
    FE_CHECK_ARG(static_cast<GlobalIndex>(row_perm.size()) == n_rows_,
                 "Row permutation size mismatch");
    FE_CHECK_ARG(static_cast<GlobalIndex>(col_perm.size()) == n_cols_,
                 "Column permutation size mismatch");

    // Validate permutations are bijections (no duplicates).
    std::vector<std::uint8_t> seen_rows(static_cast<std::size_t>(n_rows_), 0u);
    for (GlobalIndex old_row = 0; old_row < n_rows_; ++old_row) {
        GlobalIndex new_row = row_perm[static_cast<std::size_t>(old_row)];
        FE_CHECK_ARG(new_row >= 0 && new_row < n_rows_, "Invalid row permutation");
        FE_CHECK_ARG(seen_rows[static_cast<std::size_t>(new_row)] == 0u,
                     "Row permutation must be a bijection");
        seen_rows[static_cast<std::size_t>(new_row)] = 1u;
    }

    std::vector<std::uint8_t> seen_cols(static_cast<std::size_t>(n_cols_), 0u);
    for (GlobalIndex old_col = 0; old_col < n_cols_; ++old_col) {
        GlobalIndex new_col = col_perm[static_cast<std::size_t>(old_col)];
        FE_CHECK_ARG(new_col >= 0 && new_col < n_cols_, "Invalid column permutation");
        FE_CHECK_ARG(seen_cols[static_cast<std::size_t>(new_col)] == 0u,
                     "Column permutation must be a bijection");
        seen_cols[static_cast<std::size_t>(new_col)] = 1u;
    }

    SparsityPattern result(n_rows_, n_cols_);

    for (GlobalIndex old_row = 0; old_row < n_rows_; ++old_row) {
        GlobalIndex new_row = row_perm[static_cast<std::size_t>(old_row)];

        if (isFinalized()) {
            auto row_span = getRowSpan(old_row);
            for (GlobalIndex old_col : row_span) {
                GlobalIndex new_col = col_perm[static_cast<std::size_t>(old_col)];
                result.addEntry(new_row, new_col);
            }
        } else {
            const auto& row_set = row_sets_[static_cast<std::size_t>(old_row)];
            for (GlobalIndex old_col : row_set) {
                GlobalIndex new_col = col_perm[static_cast<std::size_t>(old_col)];
                result.addEntry(new_row, new_col);
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern SparsityPattern::transpose() const {
    SparsityPattern result(n_cols_, n_rows_);

    for (GlobalIndex row = 0; row < n_rows_; ++row) {
        if (isFinalized()) {
            auto row_span = getRowSpan(row);
            for (GlobalIndex col : row_span) {
                result.addEntry(col, row);
            }
        } else {
            const auto& row_set = row_sets_[static_cast<std::size_t>(row)];
            for (GlobalIndex col : row_set) {
                result.addEntry(col, row);
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern SparsityPattern::extract(std::span<const GlobalIndex> row_set,
                                         std::span<const GlobalIndex> col_set) const {
    // Build index maps
    std::vector<GlobalIndex> row_map(static_cast<std::size_t>(n_rows_), -1);
    std::vector<GlobalIndex> col_map(static_cast<std::size_t>(n_cols_), -1);

    for (std::size_t i = 0; i < row_set.size(); ++i) {
        GlobalIndex row = row_set[i];
        if (row >= 0 && row < n_rows_) {
            row_map[static_cast<std::size_t>(row)] = static_cast<GlobalIndex>(i);
        }
    }

    for (std::size_t i = 0; i < col_set.size(); ++i) {
        GlobalIndex col = col_set[i];
        if (col >= 0 && col < n_cols_) {
            col_map[static_cast<std::size_t>(col)] = static_cast<GlobalIndex>(i);
        }
    }

    SparsityPattern result(static_cast<GlobalIndex>(row_set.size()),
                          static_cast<GlobalIndex>(col_set.size()));

    for (std::size_t i = 0; i < row_set.size(); ++i) {
        GlobalIndex old_row = row_set[i];
        if (old_row < 0 || old_row >= n_rows_) continue;

        GlobalIndex new_row = static_cast<GlobalIndex>(i);

        if (isFinalized()) {
            auto row_span = getRowSpan(old_row);
            for (GlobalIndex old_col : row_span) {
                GlobalIndex new_col = col_map[static_cast<std::size_t>(old_col)];
                if (new_col >= 0) {
                    result.addEntry(new_row, new_col);
                }
            }
        } else {
            const auto& rs = row_sets_[static_cast<std::size_t>(old_row)];
            for (GlobalIndex old_col : rs) {
                GlobalIndex new_col = col_map[static_cast<std::size_t>(old_col)];
                if (new_col >= 0) {
                    result.addEntry(new_row, new_col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

std::size_t SparsityPattern::memoryUsageBytes() const noexcept {
    std::size_t bytes = sizeof(*this);

    if (isFinalized()) {
        bytes += row_ptr_.capacity() * sizeof(GlobalIndex);
        bytes += col_idx_.capacity() * sizeof(GlobalIndex);
    } else {
        bytes += row_sets_.capacity() * sizeof(std::set<GlobalIndex>);
        // Approximate set overhead: ~40 bytes per node on 64-bit systems
        for (const auto& row_set : row_sets_) {
            bytes += row_set.size() * (sizeof(GlobalIndex) + 40);
        }
    }

    return bytes;
}

// ============================================================================
// Internal helpers
// ============================================================================

void SparsityPattern::checkNotFinalized() const {
    FE_THROW_IF(isFinalized(), InvalidArgumentException,
                "SparsityPattern is finalized - modification not allowed");
}

void SparsityPattern::checkFinalized() const {
    FE_THROW_IF(!isFinalized(), InvalidArgumentException,
                "SparsityPattern is not finalized - CSR access not available");
}

void SparsityPattern::checkRowIndex(GlobalIndex row) const {
    FE_CHECK_ARG(row >= 0 && row < n_rows_,
                 "Row index " + std::to_string(row) + " out of range [0, " +
                 std::to_string(n_rows_) + ")");
}

void SparsityPattern::checkColIndex(GlobalIndex col) const {
    FE_CHECK_ARG(col >= 0 && col < n_cols_,
                 "Column index " + std::to_string(col) + " out of range [0, " +
                 std::to_string(n_cols_) + ")");
}

void SparsityPattern::checkIndices(GlobalIndex row, GlobalIndex col) const {
    checkRowIndex(row);
    checkColIndex(col);
}

// ============================================================================
// Free functions
// ============================================================================

SparsityPattern patternUnion(const SparsityPattern& a, const SparsityPattern& b) {
    FE_CHECK_ARG(a.numRows() == b.numRows() && a.numCols() == b.numCols(),
                 "Pattern dimensions must match for union");

    SparsityPattern result(a.numRows(), a.numCols());

    // Add entries from a
    for (GlobalIndex row = 0; row < a.numRows(); ++row) {
        if (a.isFinalized()) {
            result.addEntries(row, a.getRowSpan(row));
        } else {
            const auto& row_set = a.row_sets_[static_cast<std::size_t>(row)];
            for (GlobalIndex col : row_set) {
                result.addEntry(row, col);
            }
        }
    }

    // Add entries from b
    for (GlobalIndex row = 0; row < b.numRows(); ++row) {
        if (b.isFinalized()) {
            result.addEntries(row, b.getRowSpan(row));
        } else {
            const auto& row_set = b.row_sets_[static_cast<std::size_t>(row)];
            for (GlobalIndex col : row_set) {
                result.addEntry(row, col);
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern patternIntersection(const SparsityPattern& a, const SparsityPattern& b) {
    FE_CHECK_ARG(a.numRows() == b.numRows() && a.numCols() == b.numCols(),
                 "Pattern dimensions must match for intersection");

    SparsityPattern result(a.numRows(), a.numCols());

    for (GlobalIndex row = 0; row < a.numRows(); ++row) {
        if (a.isFinalized()) {
            auto row_span = a.getRowSpan(row);
            for (GlobalIndex col : row_span) {
                if (b.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        } else {
            const auto& row_set = a.row_sets_[static_cast<std::size_t>(row)];
            for (GlobalIndex col : row_set) {
                if (b.hasEntry(row, col)) {
                    result.addEntry(row, col);
                }
            }
        }
    }

    result.finalize();
    return result;
}

SparsityPattern symmetrize(const SparsityPattern& pattern) {
    FE_CHECK_ARG(pattern.isSquare(), "Can only symmetrize square patterns");

    SparsityPattern result(pattern.numRows(), pattern.numCols());

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.isFinalized()) {
            auto row_span = pattern.getRowSpan(row);
            for (GlobalIndex col : row_span) {
                result.addEntry(row, col);
                result.addEntry(col, row);
            }
        } else {
            const auto& row_set = pattern.row_sets_[static_cast<std::size_t>(row)];
            for (GlobalIndex col : row_set) {
                result.addEntry(row, col);
                result.addEntry(col, row);
            }
        }
    }

    result.finalize();
    return result;
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
