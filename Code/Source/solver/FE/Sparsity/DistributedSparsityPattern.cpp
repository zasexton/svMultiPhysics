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

#include "DistributedSparsityPattern.h"
#include <algorithm>
#include <iterator>
#include <sstream>
#include <numeric>

namespace svmp {
namespace FE {
namespace sparsity {

namespace {

inline void sortUnique(std::vector<GlobalIndex>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

inline void insertSortedUnique(std::vector<GlobalIndex>& row, GlobalIndex col) {
    auto it = std::lower_bound(row.begin(), row.end(), col);
    if (it == row.end() || *it != col) {
        row.insert(it, col);
    }
}

inline void unionSortedUniqueInto(std::vector<GlobalIndex>& row,
                                  const std::vector<GlobalIndex>& add_sorted_unique,
                                  std::vector<GlobalIndex>& scratch) {
    if (add_sorted_unique.empty()) {
        return;
    }
    if (row.empty()) {
        row = add_sorted_unique;
        return;
    }

    scratch.clear();
    scratch.reserve(row.size() + add_sorted_unique.size());
    std::set_union(row.begin(), row.end(),
                   add_sorted_unique.begin(), add_sorted_unique.end(),
                   std::back_inserter(scratch));
    row.swap(scratch);
}

} // namespace

// ============================================================================
// Construction and lifecycle
// ============================================================================

DistributedSparsityPattern::DistributedSparsityPattern(IndexRange owned_rows,
                                                       IndexRange owned_cols,
                                                       GlobalIndex global_rows,
                                                       GlobalIndex global_cols)
    : global_rows_(global_rows),
      global_cols_(global_cols < 0 ? global_rows : global_cols),
      owned_rows_(owned_rows),
      owned_cols_(owned_cols),
      building_rows_(static_cast<std::size_t>(owned_rows.size())),
      state_(SparsityState::Building)
{
    FE_CHECK_ARG(global_rows >= 0, "Global rows must be non-negative");
    FE_CHECK_ARG(global_cols_ >= 0, "Global cols must be non-negative");
    FE_CHECK_ARG(owned_rows.first >= 0 && owned_rows.last <= global_rows,
                 "Owned row range must be within global rows");
    FE_CHECK_ARG(owned_cols.first >= 0 && owned_cols.last <= global_cols_,
                 "Owned column range must be within global columns");
}

DistributedSparsityPattern::DistributedSparsityPattern(GlobalIndex first_owned_row,
                                                       GlobalIndex n_owned_rows,
                                                       GlobalIndex first_owned_col,
                                                       GlobalIndex n_owned_cols,
                                                       GlobalIndex global_rows,
                                                       GlobalIndex global_cols)
    : DistributedSparsityPattern(
          IndexRange{first_owned_row, first_owned_row + n_owned_rows},
          IndexRange{first_owned_col, first_owned_col + n_owned_cols},
          global_rows,
          global_cols < 0 ? global_rows : global_cols)
{
}

DistributedSparsityPattern::DistributedSparsityPattern(DistributedSparsityPattern&& other) noexcept
    : global_rows_(other.global_rows_),
      global_cols_(other.global_cols_),
      owned_rows_(other.owned_rows_),
      owned_cols_(other.owned_cols_),
      dof_indexing_(other.dof_indexing_),
      building_rows_(std::move(other.building_rows_)),
      diag_pattern_(std::move(other.diag_pattern_)),
      offdiag_pattern_(std::move(other.offdiag_pattern_)),
      ghost_col_map_(std::move(other.ghost_col_map_)),
      global_to_ghost_(std::move(other.global_to_ghost_)),
      ghost_row_map_(std::move(other.ghost_row_map_)),
      ghost_row_ptr_(std::move(other.ghost_row_ptr_)),
      ghost_row_cols_(std::move(other.ghost_row_cols_)),
      global_to_ghost_row_(std::move(other.global_to_ghost_row_)),
      state_(other.state_.load(std::memory_order_acquire))
{
    other.global_rows_ = 0;
    other.global_cols_ = 0;
    other.owned_rows_ = IndexRange{};
    other.owned_cols_ = IndexRange{};
    other.ghost_row_map_.clear();
    other.ghost_row_ptr_.clear();
    other.ghost_row_cols_.clear();
    other.global_to_ghost_row_.clear();
    other.state_.store(SparsityState::Building, std::memory_order_release);
}

DistributedSparsityPattern& DistributedSparsityPattern::operator=(DistributedSparsityPattern&& other) noexcept {
    if (this != &other) {
        global_rows_ = other.global_rows_;
        global_cols_ = other.global_cols_;
        owned_rows_ = other.owned_rows_;
        owned_cols_ = other.owned_cols_;
        dof_indexing_ = other.dof_indexing_;
        building_rows_ = std::move(other.building_rows_);
        diag_pattern_ = std::move(other.diag_pattern_);
        offdiag_pattern_ = std::move(other.offdiag_pattern_);
        ghost_col_map_ = std::move(other.ghost_col_map_);
        global_to_ghost_ = std::move(other.global_to_ghost_);
        ghost_row_map_ = std::move(other.ghost_row_map_);
        ghost_row_ptr_ = std::move(other.ghost_row_ptr_);
        ghost_row_cols_ = std::move(other.ghost_row_cols_);
        global_to_ghost_row_ = std::move(other.global_to_ghost_row_);
        state_.store(other.state_.load(std::memory_order_acquire), std::memory_order_release);

        other.global_rows_ = 0;
        other.global_cols_ = 0;
        other.owned_rows_ = IndexRange{};
        other.owned_cols_ = IndexRange{};
        other.dof_indexing_ = DofIndexing::Natural;
        other.ghost_row_map_.clear();
        other.ghost_row_ptr_.clear();
        other.ghost_row_cols_.clear();
        other.global_to_ghost_row_.clear();
        other.state_.store(SparsityState::Building, std::memory_order_release);
    }
    return *this;
}

DistributedSparsityPattern::DistributedSparsityPattern(const DistributedSparsityPattern& other)
    : global_rows_(other.global_rows_),
      global_cols_(other.global_cols_),
      owned_rows_(other.owned_rows_),
      owned_cols_(other.owned_cols_),
      dof_indexing_(other.dof_indexing_),
      ghost_row_map_(other.ghost_row_map_),
      ghost_row_ptr_(other.ghost_row_ptr_),
      ghost_row_cols_(other.ghost_row_cols_),
      global_to_ghost_row_(other.global_to_ghost_row_),
      state_(SparsityState::Building)
{
    if (other.isFinalized()) {
        building_rows_.clear();
        diag_pattern_ = other.diag_pattern_;
        offdiag_pattern_ = other.offdiag_pattern_;
        ghost_col_map_ = other.ghost_col_map_;
        global_to_ghost_ = other.global_to_ghost_;
        state_.store(SparsityState::Finalized, std::memory_order_release);
    } else {
        building_rows_ = other.building_rows_;
        diag_pattern_ = SparsityPattern();
        offdiag_pattern_ = SparsityPattern();
        ghost_col_map_.clear();
        global_to_ghost_.clear();
        state_.store(SparsityState::Building, std::memory_order_release);
    }
}

DistributedSparsityPattern& DistributedSparsityPattern::operator=(const DistributedSparsityPattern& other) {
    if (this != &other) {
        global_rows_ = other.global_rows_;
        global_cols_ = other.global_cols_;
        owned_rows_ = other.owned_rows_;
        owned_cols_ = other.owned_cols_;
        dof_indexing_ = other.dof_indexing_;
        ghost_row_map_ = other.ghost_row_map_;
        ghost_row_ptr_ = other.ghost_row_ptr_;
        ghost_row_cols_ = other.ghost_row_cols_;
        global_to_ghost_row_ = other.global_to_ghost_row_;

        if (other.isFinalized()) {
            building_rows_.clear();
            diag_pattern_ = other.diag_pattern_;
            offdiag_pattern_ = other.offdiag_pattern_;
            ghost_col_map_ = other.ghost_col_map_;
            global_to_ghost_ = other.global_to_ghost_;
            state_.store(SparsityState::Finalized, std::memory_order_release);
        } else {
            building_rows_ = other.building_rows_;
            diag_pattern_ = SparsityPattern();
            offdiag_pattern_ = SparsityPattern();
            ghost_col_map_.clear();
            global_to_ghost_.clear();
            state_.store(SparsityState::Building, std::memory_order_release);
        }
    }
    return *this;
}

DistributedSparsityPattern DistributedSparsityPattern::cloneFinalized() const {
    if (!isFinalized()) {
        DistributedSparsityPattern copy(*this);
        copy.finalize();
        return copy;
    }

    DistributedSparsityPattern copy;
    copy.global_rows_ = global_rows_;
    copy.global_cols_ = global_cols_;
    copy.owned_rows_ = owned_rows_;
    copy.owned_cols_ = owned_cols_;
    copy.dof_indexing_ = dof_indexing_;

    copy.building_rows_.clear();

    copy.diag_pattern_ = diag_pattern_.cloneFinalized();
    copy.offdiag_pattern_ = offdiag_pattern_.cloneFinalized();

    copy.ghost_col_map_ = ghost_col_map_;
    copy.global_to_ghost_ = global_to_ghost_;

    copy.ghost_row_map_ = ghost_row_map_;
    copy.ghost_row_ptr_ = ghost_row_ptr_;
    copy.ghost_row_cols_ = ghost_row_cols_;
    copy.global_to_ghost_row_ = global_to_ghost_row_;

    copy.state_.store(SparsityState::Finalized, std::memory_order_release);
    return copy;
}

// ============================================================================
// Setup methods
// ============================================================================

void DistributedSparsityPattern::clear() {
    building_rows_.clear();
    building_rows_.resize(static_cast<std::size_t>(owned_rows_.size()));
    diag_pattern_ = SparsityPattern();
    offdiag_pattern_ = SparsityPattern();
    ghost_col_map_.clear();
    global_to_ghost_.clear();
    clearGhostRows();
    state_.store(SparsityState::Building, std::memory_order_release);
}

void DistributedSparsityPattern::addEntry(GlobalIndex global_row, GlobalIndex global_col) {
    checkNotFinalized();
    checkOwnedRow(global_row);
    FE_CHECK_ARG(global_col >= 0 && global_col < global_cols_,
                 "Column index out of range");

    GlobalIndex local_row = global_row - owned_rows_.first;
    insertSortedUnique(building_rows_[static_cast<std::size_t>(local_row)], global_col);
}

void DistributedSparsityPattern::addEntries(GlobalIndex global_row,
                                            std::span<const GlobalIndex> global_cols) {
    checkNotFinalized();
    checkOwnedRow(global_row);

    GlobalIndex local_row = global_row - owned_rows_.first;
    std::vector<GlobalIndex> cols_sorted;
    cols_sorted.reserve(global_cols.size());
    for (GlobalIndex col : global_cols) {
        FE_CHECK_ARG(col >= 0 && col < global_cols_, "Column index out of range");
        cols_sorted.push_back(col);
    }
    sortUnique(cols_sorted);

    std::vector<GlobalIndex> scratch;
    unionSortedUniqueInto(building_rows_[static_cast<std::size_t>(local_row)], cols_sorted, scratch);
}

void DistributedSparsityPattern::addElementCouplings(std::span<const GlobalIndex> global_dofs) {
    checkNotFinalized();

    std::vector<GlobalIndex> cols_sorted;
    cols_sorted.reserve(global_dofs.size());
    for (GlobalIndex col_dof : global_dofs) {
        if (col_dof >= 0 && col_dof < global_cols_) {
            cols_sorted.push_back(col_dof);
        }
    }
    sortUnique(cols_sorted);

    std::vector<GlobalIndex> scratch;
    for (GlobalIndex row_dof : global_dofs) {
        // Only add entries for owned rows
        if (!owned_rows_.contains(row_dof)) continue;

        GlobalIndex local_row = row_dof - owned_rows_.first;
        unionSortedUniqueInto(building_rows_[static_cast<std::size_t>(local_row)], cols_sorted, scratch);
    }
}

void DistributedSparsityPattern::addElementCouplings(std::span<const GlobalIndex> row_dofs,
                                                     std::span<const GlobalIndex> col_dofs) {
    checkNotFinalized();

    std::vector<GlobalIndex> cols_sorted;
    cols_sorted.reserve(col_dofs.size());
    for (GlobalIndex col_dof : col_dofs) {
        if (col_dof >= 0 && col_dof < global_cols_) {
            cols_sorted.push_back(col_dof);
        }
    }
    sortUnique(cols_sorted);

    std::vector<GlobalIndex> scratch;
    for (GlobalIndex row_dof : row_dofs) {
        if (!owned_rows_.contains(row_dof)) continue;

        GlobalIndex local_row = row_dof - owned_rows_.first;
        unionSortedUniqueInto(building_rows_[static_cast<std::size_t>(local_row)], cols_sorted, scratch);
    }
}

void DistributedSparsityPattern::ensureDiagonal() {
    checkNotFinalized();

    // For square matrices where row and col ownership match
    for (GlobalIndex local_row = 0; local_row < numOwnedRows(); ++local_row) {
        GlobalIndex global_row = local_row + owned_rows_.first;
        // Only add diagonal if this column exists
        if (global_row < global_cols_) {
            insertSortedUnique(building_rows_[static_cast<std::size_t>(local_row)], global_row);
        }
    }
}

void DistributedSparsityPattern::ensureNonEmptyRows() {
    checkNotFinalized();

    for (GlobalIndex local_row = 0; local_row < numOwnedRows(); ++local_row) {
        if (building_rows_[static_cast<std::size_t>(local_row)].empty()) {
            GlobalIndex global_row = local_row + owned_rows_.first;
            // Add diagonal for square, or first owned column, or column 0
            if (global_row < global_cols_) {
                building_rows_[static_cast<std::size_t>(local_row)].push_back(global_row);
            } else if (owned_cols_.size() > 0) {
                building_rows_[static_cast<std::size_t>(local_row)].push_back(owned_cols_.first);
            } else if (global_cols_ > 0) {
                building_rows_[static_cast<std::size_t>(local_row)].push_back(0);
            }
        }
    }
}

void DistributedSparsityPattern::finalize() {
    checkNotFinalized();

    GlobalIndex n_owned_rows = numOwnedRows();
    GlobalIndex n_owned_cols = numOwnedCols();

    // Step 1: Collect all ghost columns (columns not in owned range)
    std::vector<GlobalIndex> ghost_cols;
    for (const auto& row_cols : building_rows_) {
        for (GlobalIndex col : row_cols) {
            if (!owned_cols_.contains(col)) {
                ghost_cols.push_back(col);
            }
        }
    }
    sortUnique(ghost_cols);

    // Step 2: Build ghost column map (sorted for determinism)
    ghost_col_map_ = std::move(ghost_cols);
    GlobalIndex n_ghost_cols = static_cast<GlobalIndex>(ghost_col_map_.size());

    // Build reverse map
    global_to_ghost_.clear();
    for (GlobalIndex i = 0; i < n_ghost_cols; ++i) {
        global_to_ghost_[ghost_col_map_[static_cast<std::size_t>(i)]] = i;
    }

    // Step 3: Create diagonal and off-diagonal patterns
    diag_pattern_ = SparsityPattern(n_owned_rows, n_owned_cols);
    offdiag_pattern_ = SparsityPattern(n_owned_rows, n_ghost_cols);

    // Step 4: Classify entries into diag/offdiag patterns
    for (GlobalIndex local_row = 0; local_row < n_owned_rows; ++local_row) {
        const auto& row_cols = building_rows_[static_cast<std::size_t>(local_row)];

        for (GlobalIndex global_col : row_cols) {
            if (owned_cols_.contains(global_col)) {
                // Diagonal entry: convert global col to local col
                GlobalIndex local_col = global_col - owned_cols_.first;
                diag_pattern_.addEntry(local_row, local_col);
            } else {
                // Off-diagonal entry: convert global col to ghost index
                GlobalIndex ghost_idx = global_to_ghost_.at(global_col);
                offdiag_pattern_.addEntry(local_row, ghost_idx);
            }
        }
    }

    // Step 5: Finalize sub-patterns
    diag_pattern_.finalize();
    offdiag_pattern_.finalize();

    // Step 6: Clear building storage
    building_rows_.clear();
    building_rows_.shrink_to_fit();

    // Transition to finalized state
    state_.store(SparsityState::Finalized, std::memory_order_release);
}

// ============================================================================
// Query methods
// ============================================================================

GlobalIndex DistributedSparsityPattern::getDiagNnz() const {
    if (isFinalized()) {
        return diag_pattern_.getNnz();
    }

    // During building, count owned columns
    GlobalIndex nnz = 0;
    for (const auto& row_set : building_rows_) {
        for (GlobalIndex col : row_set) {
            if (owned_cols_.contains(col)) {
                ++nnz;
            }
        }
    }
    return nnz;
}

GlobalIndex DistributedSparsityPattern::getOffdiagNnz() const {
    if (isFinalized()) {
        return offdiag_pattern_.getNnz();
    }

    // During building, count ghost columns
    GlobalIndex nnz = 0;
    for (const auto& row_set : building_rows_) {
        for (GlobalIndex col : row_set) {
            if (!owned_cols_.contains(col)) {
                ++nnz;
            }
        }
    }
    return nnz;
}

GlobalIndex DistributedSparsityPattern::getRowDiagNnz(GlobalIndex local_row) const {
    checkLocalRow(local_row);

    if (isFinalized()) {
        return diag_pattern_.getRowNnz(local_row);
    }

    const auto& row_set = building_rows_[static_cast<std::size_t>(local_row)];
    GlobalIndex count = 0;
    for (GlobalIndex col : row_set) {
        if (owned_cols_.contains(col)) {
            ++count;
        }
    }
    return count;
}

GlobalIndex DistributedSparsityPattern::getRowOffdiagNnz(GlobalIndex local_row) const {
    checkLocalRow(local_row);

    if (isFinalized()) {
        return offdiag_pattern_.getRowNnz(local_row);
    }

    const auto& row_set = building_rows_[static_cast<std::size_t>(local_row)];
    GlobalIndex count = 0;
    for (GlobalIndex col : row_set) {
        if (!owned_cols_.contains(col)) {
            ++count;
        }
    }
    return count;
}

GlobalIndex DistributedSparsityPattern::ghostColToGlobal(GlobalIndex local_ghost_idx) const {
    checkFinalized();
    FE_CHECK_ARG(local_ghost_idx >= 0 &&
                 local_ghost_idx < static_cast<GlobalIndex>(ghost_col_map_.size()),
                 "Ghost column index out of range");
    return ghost_col_map_[static_cast<std::size_t>(local_ghost_idx)];
}

GlobalIndex DistributedSparsityPattern::globalToGhostCol(GlobalIndex global_col) const {
    if (!isFinalized()) {
        return -1;
    }
    auto it = global_to_ghost_.find(global_col);
    return (it != global_to_ghost_.end()) ? it->second : -1;
}

std::span<const GlobalIndex> DistributedSparsityPattern::getGhostColMap() const {
    checkFinalized();
    return std::span<const GlobalIndex>(ghost_col_map_.data(), ghost_col_map_.size());
}

std::span<const GlobalIndex> DistributedSparsityPattern::getGhostRowMap() const {
    checkFinalized();
    FE_CHECK_ARG(!ghost_row_map_.empty(), "No ghost rows are stored");
    return std::span<const GlobalIndex>(ghost_row_map_.data(), ghost_row_map_.size());
}

GlobalIndex DistributedSparsityPattern::globalToGhostRow(GlobalIndex global_row) const {
    if (!isFinalized() || ghost_row_map_.empty()) {
        return -1;
    }
    auto it = global_to_ghost_row_.find(global_row);
    return (it != global_to_ghost_row_.end()) ? it->second : -1;
}

std::span<const GlobalIndex> DistributedSparsityPattern::getGhostRowCols(
    GlobalIndex local_ghost_row) const {
    checkFinalized();
    FE_CHECK_ARG(!ghost_row_map_.empty(), "No ghost rows are stored");
    FE_CHECK_ARG(local_ghost_row >= 0 &&
                     local_ghost_row < static_cast<GlobalIndex>(ghost_row_map_.size()),
                 "Ghost row index out of range");
    FE_CHECK_ARG(ghost_row_ptr_.size() == ghost_row_map_.size() + 1,
                 "Ghost row CSR pointer array size mismatch");

    const auto start = static_cast<std::size_t>(ghost_row_ptr_[static_cast<std::size_t>(local_ghost_row)]);
    const auto end = static_cast<std::size_t>(ghost_row_ptr_[static_cast<std::size_t>(local_ghost_row) + 1]);
    FE_CHECK_ARG(start <= end && end <= ghost_row_cols_.size(),
                 "Ghost row CSR pointers out of range");
    return std::span<const GlobalIndex>(ghost_row_cols_.data() + start, end - start);
}

void DistributedSparsityPattern::setGhostRows(std::vector<GlobalIndex> ghost_rows,
                                              std::vector<GlobalIndex> row_ptr,
                                              std::vector<GlobalIndex> col_idx) {
    checkFinalized();

    FE_CHECK_ARG(row_ptr.size() == ghost_rows.size() + 1,
                 "Ghost row CSR pointer array must have size ghost_rows.size()+1");
    FE_CHECK_ARG(!row_ptr.empty() ? row_ptr.front() == 0 : ghost_rows.empty(),
                 "Ghost row CSR pointer array must start at 0");
    FE_CHECK_ARG(!row_ptr.empty() ? row_ptr.back() == static_cast<GlobalIndex>(col_idx.size()) : col_idx.empty(),
                 "Ghost row CSR pointer array end must equal col_idx size");

    for (std::size_t i = 1; i < ghost_rows.size(); ++i) {
        FE_CHECK_ARG(ghost_rows[i] > ghost_rows[i - 1], "Ghost row map must be sorted and unique");
    }
    for (GlobalIndex row : ghost_rows) {
        FE_CHECK_ARG(row >= 0 && row < global_rows_, "Ghost row out of global range");
        FE_CHECK_ARG(!owned_rows_.contains(row), "Ghost row must not be owned");
    }
    for (std::size_t i = 1; i < row_ptr.size(); ++i) {
        FE_CHECK_ARG(row_ptr[i] >= row_ptr[i - 1], "Ghost row CSR pointers must be non-decreasing");
    }
    for (GlobalIndex col : col_idx) {
        FE_CHECK_ARG(col >= 0 && col < global_cols_, "Ghost row column out of global range");
    }

    ghost_row_map_ = std::move(ghost_rows);
    ghost_row_ptr_ = std::move(row_ptr);
    ghost_row_cols_ = std::move(col_idx);

    global_to_ghost_row_.clear();
    global_to_ghost_row_.reserve(ghost_row_map_.size());
    for (std::size_t i = 0; i < ghost_row_map_.size(); ++i) {
        global_to_ghost_row_[ghost_row_map_[i]] = static_cast<GlobalIndex>(i);
    }
}

void DistributedSparsityPattern::clearGhostRows() {
    ghost_row_map_.clear();
    ghost_row_ptr_.clear();
    ghost_row_cols_.clear();
    global_to_ghost_row_.clear();
}

std::vector<GlobalIndex> DistributedSparsityPattern::getOwnedRowGlobalCols(GlobalIndex global_row) const {
    checkFinalized();
    checkOwnedRow(global_row);

    const GlobalIndex local_row = global_row - owned_rows_.first;
    const auto diag_cols = getRowDiagCols(local_row);
    const auto offdiag_cols = getRowOffdiagCols(local_row);

    std::vector<GlobalIndex> cols;
    cols.reserve(static_cast<std::size_t>(diag_cols.size() + offdiag_cols.size()));
    for (GlobalIndex local_col : diag_cols) {
        cols.push_back(local_col + owned_cols_.first);
    }
    for (GlobalIndex ghost_idx : offdiag_cols) {
        cols.push_back(ghostColToGlobal(ghost_idx));
    }
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    return cols;
}

const SparsityPattern& DistributedSparsityPattern::diagPattern() const {
    checkFinalized();
    return diag_pattern_;
}

const SparsityPattern& DistributedSparsityPattern::offdiagPattern() const {
    checkFinalized();
    return offdiag_pattern_;
}

std::span<const GlobalIndex> DistributedSparsityPattern::getRowDiagCols(GlobalIndex local_row) const {
    checkFinalized();
    checkLocalRow(local_row);
    return diag_pattern_.getRowSpan(local_row);
}

std::span<const GlobalIndex> DistributedSparsityPattern::getRowOffdiagCols(GlobalIndex local_row) const {
    checkFinalized();
    checkLocalRow(local_row);
    return offdiag_pattern_.getRowSpan(local_row);
}

bool DistributedSparsityPattern::hasEntry(GlobalIndex global_row, GlobalIndex global_col) const {
    if (!owned_rows_.contains(global_row)) {
        return false;
    }

    GlobalIndex local_row = global_row - owned_rows_.first;

    if (isFinalized()) {
        if (owned_cols_.contains(global_col)) {
            GlobalIndex local_col = global_col - owned_cols_.first;
            return diag_pattern_.hasEntry(local_row, local_col);
        } else {
            GlobalIndex ghost_idx = globalToGhostCol(global_col);
            if (ghost_idx < 0) return false;
            return offdiag_pattern_.hasEntry(local_row, ghost_idx);
        }
    } else {
        const auto& row_cols = building_rows_[static_cast<std::size_t>(local_row)];
        return std::binary_search(row_cols.begin(), row_cols.end(), global_col);
    }
}

// ============================================================================
// Preallocation
// ============================================================================

PreallocationInfo DistributedSparsityPattern::getPreallocationInfo() const {
    checkFinalized();

    PreallocationInfo info;
    GlobalIndex n_rows = numOwnedRows();

    info.diag_nnz_per_row.resize(static_cast<std::size_t>(n_rows));
    info.offdiag_nnz_per_row.resize(static_cast<std::size_t>(n_rows));

    info.max_diag_nnz = 0;
    info.max_offdiag_nnz = 0;
    info.total_diag_nnz = 0;
    info.total_offdiag_nnz = 0;

    for (GlobalIndex local_row = 0; local_row < n_rows; ++local_row) {
        GlobalIndex diag_nnz = diag_pattern_.getRowNnz(local_row);
        GlobalIndex offdiag_nnz = offdiag_pattern_.getRowNnz(local_row);

        info.diag_nnz_per_row[static_cast<std::size_t>(local_row)] = diag_nnz;
        info.offdiag_nnz_per_row[static_cast<std::size_t>(local_row)] = offdiag_nnz;

        info.max_diag_nnz = std::max(info.max_diag_nnz, diag_nnz);
        info.max_offdiag_nnz = std::max(info.max_offdiag_nnz, offdiag_nnz);
        info.total_diag_nnz += diag_nnz;
        info.total_offdiag_nnz += offdiag_nnz;
    }

    return info;
}

std::vector<GlobalIndex> DistributedSparsityPattern::getDiagNnzPerRow() const {
    checkFinalized();

    GlobalIndex n_rows = numOwnedRows();
    std::vector<GlobalIndex> result(static_cast<std::size_t>(n_rows));

    for (GlobalIndex local_row = 0; local_row < n_rows; ++local_row) {
        result[static_cast<std::size_t>(local_row)] = diag_pattern_.getRowNnz(local_row);
    }

    return result;
}

std::vector<GlobalIndex> DistributedSparsityPattern::getOffdiagNnzPerRow() const {
    checkFinalized();

    GlobalIndex n_rows = numOwnedRows();
    std::vector<GlobalIndex> result(static_cast<std::size_t>(n_rows));

    for (GlobalIndex local_row = 0; local_row < n_rows; ++local_row) {
        result[static_cast<std::size_t>(local_row)] = offdiag_pattern_.getRowNnz(local_row);
    }

    return result;
}

// ============================================================================
// Statistics and validation
// ============================================================================

DistributedSparsityStats DistributedSparsityPattern::computeStats() const {
    DistributedSparsityStats stats;

    stats.n_owned_rows = numOwnedRows();
    stats.n_owned_cols = numOwnedCols();
    stats.n_ghost_cols = numGhostCols();
    stats.n_relevant_cols = numRelevantCols();
    stats.global_rows = global_rows_;
    stats.global_cols = global_cols_;

    stats.local_diag_nnz = getDiagNnz();
    stats.local_offdiag_nnz = getOffdiagNnz();
    stats.local_total_nnz = stats.local_diag_nnz + stats.local_offdiag_nnz;

    if (stats.n_owned_rows == 0) {
        return stats;
    }

    stats.min_diag_row_nnz = std::numeric_limits<GlobalIndex>::max();
    stats.max_diag_row_nnz = 0;
    stats.min_offdiag_row_nnz = std::numeric_limits<GlobalIndex>::max();
    stats.max_offdiag_row_nnz = 0;

    for (GlobalIndex local_row = 0; local_row < stats.n_owned_rows; ++local_row) {
        GlobalIndex diag_nnz = getRowDiagNnz(local_row);
        GlobalIndex offdiag_nnz = getRowOffdiagNnz(local_row);

        stats.min_diag_row_nnz = std::min(stats.min_diag_row_nnz, diag_nnz);
        stats.max_diag_row_nnz = std::max(stats.max_diag_row_nnz, diag_nnz);
        stats.min_offdiag_row_nnz = std::min(stats.min_offdiag_row_nnz, offdiag_nnz);
        stats.max_offdiag_row_nnz = std::max(stats.max_offdiag_row_nnz, offdiag_nnz);
    }

    if (stats.min_diag_row_nnz == std::numeric_limits<GlobalIndex>::max()) {
        stats.min_diag_row_nnz = 0;
    }
    if (stats.min_offdiag_row_nnz == std::numeric_limits<GlobalIndex>::max()) {
        stats.min_offdiag_row_nnz = 0;
    }

    stats.avg_diag_row_nnz = static_cast<double>(stats.local_diag_nnz) /
                             static_cast<double>(stats.n_owned_rows);
    stats.avg_offdiag_row_nnz = static_cast<double>(stats.local_offdiag_nnz) /
                                static_cast<double>(stats.n_owned_rows);

    return stats;
}

bool DistributedSparsityPattern::validate() const noexcept {
    try {
        // Check dimension consistency
        if (global_rows_ < 0 || global_cols_ < 0) return false;
        if (owned_rows_.first < 0 || owned_rows_.last > global_rows_) return false;
        if (owned_cols_.first < 0 || owned_cols_.last > global_cols_) return false;

        if (isFinalized()) {
            // Check patterns
            if (!diag_pattern_.validate()) return false;
            if (!offdiag_pattern_.validate()) return false;

            // Check dimensions match
            if (diag_pattern_.numRows() != numOwnedRows()) return false;
            if (diag_pattern_.numCols() != numOwnedCols()) return false;
            if (offdiag_pattern_.numRows() != numOwnedRows()) return false;
            if (offdiag_pattern_.numCols() != numGhostCols()) return false;

            // Check ghost column map is sorted and unique
            for (std::size_t i = 1; i < ghost_col_map_.size(); ++i) {
                if (ghost_col_map_[i] <= ghost_col_map_[i-1]) return false;
            }

            // Check ghost columns are not owned
            for (GlobalIndex ghost_col : ghost_col_map_) {
                if (owned_cols_.contains(ghost_col)) return false;
            }

            // Optional ghost rows
            if (!ghost_row_map_.empty()) {
                if (ghost_row_ptr_.size() != ghost_row_map_.size() + 1) return false;
                if (!ghost_row_ptr_.empty() && ghost_row_ptr_.front() != 0) return false;
                if (!ghost_row_ptr_.empty() &&
                    ghost_row_ptr_.back() != static_cast<GlobalIndex>(ghost_row_cols_.size())) {
                    return false;
                }
                for (std::size_t i = 1; i < ghost_row_map_.size(); ++i) {
                    if (ghost_row_map_[i] <= ghost_row_map_[i - 1]) return false;
                }
                for (GlobalIndex row : ghost_row_map_) {
                    if (row < 0 || row >= global_rows_) return false;
                    if (owned_rows_.contains(row)) return false;
                }
                for (std::size_t i = 1; i < ghost_row_ptr_.size(); ++i) {
                    if (ghost_row_ptr_[i] < ghost_row_ptr_[i - 1]) return false;
                }
                for (GlobalIndex col : ghost_row_cols_) {
                    if (col < 0 || col >= global_cols_) return false;
                }
            }
        } else {
            // Check building_rows_ size
            if (building_rows_.size() != static_cast<std::size_t>(numOwnedRows())) {
                return false;
            }

            // Check column indices
            for (const auto& row_set : building_rows_) {
                for (GlobalIndex col : row_set) {
                    if (col < 0 || col >= global_cols_) return false;
                }
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

std::string DistributedSparsityPattern::validationError() const {
    std::ostringstream oss;

    if (global_rows_ < 0) {
        oss << "Negative global rows: " << global_rows_;
        return oss.str();
    }
    if (global_cols_ < 0) {
        oss << "Negative global cols: " << global_cols_;
        return oss.str();
    }
    if (owned_rows_.first < 0 || owned_rows_.last > global_rows_) {
        oss << "Owned row range [" << owned_rows_.first << ", " << owned_rows_.last
            << ") outside global range [0, " << global_rows_ << ")";
        return oss.str();
    }
    if (owned_cols_.first < 0 || owned_cols_.last > global_cols_) {
        oss << "Owned col range [" << owned_cols_.first << ", " << owned_cols_.last
            << ") outside global range [0, " << global_cols_ << ")";
        return oss.str();
    }

    if (isFinalized()) {
        std::string diag_err = diag_pattern_.validationError();
        if (!diag_err.empty()) {
            oss << "Diagonal pattern: " << diag_err;
            return oss.str();
        }

        std::string offdiag_err = offdiag_pattern_.validationError();
        if (!offdiag_err.empty()) {
            oss << "Off-diagonal pattern: " << offdiag_err;
            return oss.str();
        }

        if (diag_pattern_.numRows() != numOwnedRows()) {
            oss << "Diagonal pattern row count mismatch";
            return oss.str();
        }
    }

    return "";  // Valid
}

std::size_t DistributedSparsityPattern::memoryUsageBytes() const noexcept {
    std::size_t bytes = sizeof(*this);

    if (isFinalized()) {
        bytes += diag_pattern_.memoryUsageBytes();
        bytes += offdiag_pattern_.memoryUsageBytes();
        bytes += ghost_col_map_.capacity() * sizeof(GlobalIndex);
        // Approximate hash map overhead
        bytes += global_to_ghost_.size() * (sizeof(GlobalIndex) * 2 + 32);
        bytes += ghost_row_map_.capacity() * sizeof(GlobalIndex);
        bytes += ghost_row_ptr_.capacity() * sizeof(GlobalIndex);
        bytes += ghost_row_cols_.capacity() * sizeof(GlobalIndex);
        bytes += global_to_ghost_row_.size() * (sizeof(GlobalIndex) * 2 + 32);
    } else {
        bytes += building_rows_.capacity() * sizeof(std::vector<GlobalIndex>);
        for (const auto& row_cols : building_rows_) {
            bytes += row_cols.capacity() * sizeof(GlobalIndex);
        }
    }

    return bytes;
}

// ============================================================================
// Internal helpers
// ============================================================================

void DistributedSparsityPattern::checkNotFinalized() const {
    FE_THROW_IF(isFinalized(), InvalidArgumentException,
                "DistributedSparsityPattern is finalized - modification not allowed");
}

void DistributedSparsityPattern::checkFinalized() const {
    FE_THROW_IF(!isFinalized(), InvalidArgumentException,
                "DistributedSparsityPattern is not finalized");
}

void DistributedSparsityPattern::checkOwnedRow(GlobalIndex global_row) const {
    FE_CHECK_ARG(owned_rows_.contains(global_row),
                 "Row " + std::to_string(global_row) + " not in owned range [" +
                 std::to_string(owned_rows_.first) + ", " +
                 std::to_string(owned_rows_.last) + ")");
}

void DistributedSparsityPattern::checkLocalRow(GlobalIndex local_row) const {
    FE_CHECK_ARG(local_row >= 0 && local_row < numOwnedRows(),
                 "Local row " + std::to_string(local_row) + " out of range [0, " +
                 std::to_string(numOwnedRows()) + ")");
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
