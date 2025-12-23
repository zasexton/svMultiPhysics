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

#include "AdaptiveSparsity.h"
#include <algorithm>
#include <sstream>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// PatternDelta Implementation
// ============================================================================

void PatternDelta::clear() {
    added_entries_.clear();
    removed_entries_.clear();
    added_rows_.clear();
    removed_rows_.clear();
    added_cols_.clear();
    removed_cols_.clear();
}

void PatternDelta::addEntry(GlobalIndex row, GlobalIndex col) {
    // Check if this was previously removed
    auto rem_it = removed_entries_.find(row);
    if (rem_it != removed_entries_.end()) {
        auto& rem_cols = rem_it->second;
        auto col_it = rem_cols.find(col);
        if (col_it != rem_cols.end()) {
            // Cancel out the removal
            rem_cols.erase(col_it);
            if (rem_cols.empty()) {
                removed_entries_.erase(rem_it);
            }
            return;
        }
    }

    // Add to added entries
    added_entries_[row].insert(col);
}

void PatternDelta::addEntries(GlobalIndex row, std::span<const GlobalIndex> cols) {
    for (GlobalIndex col : cols) {
        addEntry(row, col);
    }
}

void PatternDelta::removeEntry(GlobalIndex row, GlobalIndex col) {
    // Check if this was previously added
    auto add_it = added_entries_.find(row);
    if (add_it != added_entries_.end()) {
        auto& add_cols = add_it->second;
        auto col_it = add_cols.find(col);
        if (col_it != add_cols.end()) {
            // Cancel out the addition
            add_cols.erase(col_it);
            if (add_cols.empty()) {
                added_entries_.erase(add_it);
            }
            return;
        }
    }

    // Add to removed entries
    removed_entries_[row].insert(col);
}

void PatternDelta::addRow(GlobalIndex row) {
    // Check if previously removed
    if (removed_rows_.count(row)) {
        removed_rows_.erase(row);
        return;
    }
    added_rows_.insert(row);
}

void PatternDelta::removeRow(GlobalIndex row) {
    // Check if previously added
    if (added_rows_.count(row)) {
        added_rows_.erase(row);
        // Also remove any added entries for this row
        added_entries_.erase(row);
        return;
    }
    removed_rows_.insert(row);

    // Remove any added entries for this row
    added_entries_.erase(row);
}

void PatternDelta::addColumn(GlobalIndex col) {
    if (removed_cols_.count(col)) {
        removed_cols_.erase(col);
        return;
    }
    added_cols_.insert(col);
}

void PatternDelta::removeColumn(GlobalIndex col) {
    if (added_cols_.count(col)) {
        added_cols_.erase(col);
        return;
    }
    removed_cols_.insert(col);
}

GlobalIndex PatternDelta::numAddedEntries() const {
    GlobalIndex count = 0;
    for (const auto& [row, cols] : added_entries_) {
        count += static_cast<GlobalIndex>(cols.size());
    }
    return count;
}

GlobalIndex PatternDelta::numRemovedEntries() const {
    GlobalIndex count = 0;
    for (const auto& [row, cols] : removed_entries_) {
        count += static_cast<GlobalIndex>(cols.size());
    }
    return count;
}

void PatternDelta::merge(const PatternDelta& other) {
    // Merge added entries
    for (const auto& [row, cols] : other.added_entries_) {
        for (GlobalIndex col : cols) {
            addEntry(row, col);
        }
    }

    // Merge removed entries
    for (const auto& [row, cols] : other.removed_entries_) {
        for (GlobalIndex col : cols) {
            removeEntry(row, col);
        }
    }

    // Merge added/removed rows
    for (GlobalIndex row : other.added_rows_) {
        addRow(row);
    }
    for (GlobalIndex row : other.removed_rows_) {
        removeRow(row);
    }

    // Merge added/removed cols
    for (GlobalIndex col : other.added_cols_) {
        addColumn(col);
    }
    for (GlobalIndex col : other.removed_cols_) {
        removeColumn(col);
    }
}

// ============================================================================
// AdaptiveSparsity Implementation
// ============================================================================

AdaptiveSparsity::AdaptiveSparsity()
    : pattern_()
{
}

AdaptiveSparsity::AdaptiveSparsity(SparsityPattern pattern,
                                   const AdaptiveSparsityConfig& config)
    : pattern_(std::move(pattern)),
      config_(config)
{
}

AdaptiveSparsity::AdaptiveSparsity(GlobalIndex n_rows, GlobalIndex n_cols,
                                   const AdaptiveSparsityConfig& config)
    : pattern_(n_rows, n_cols),
      config_(config)
{
}

SparsityPattern AdaptiveSparsity::releasePattern() {
    // Apply any pending changes first
    if (hasPendingChanges()) {
        finalize();
    }

    SparsityPattern result = std::move(pattern_);
    pattern_ = SparsityPattern();
    delta_.clear();
    return result;
}

// ============================================================================
// Refinement Operations
// ============================================================================

void AdaptiveSparsity::applyRefinement(const RefinementEvent& event) {
    if (!event.isValid()) {
        return;
    }

    prepareForModification();

    // Expand pattern dimensions if needed
    GlobalIndex max_new_dof = 0;
    for (GlobalIndex dof : event.new_dofs) {
        max_new_dof = std::max(max_new_dof, dof);
    }
    for (GlobalIndex dof : event.added_dofs) {
        max_new_dof = std::max(max_new_dof, dof);
    }

    resizeIfNeeded(max_new_dof + 1, max_new_dof + 1);

    // Add couplings for new DOFs
    // New DOFs couple with all DOFs on child elements
    for (GlobalIndex new_dof : event.added_dofs) {
        delta_.addRow(new_dof);
        ++stats_.total_dof_additions;

        // Add coupling to all new_dofs (including self for diagonal)
        for (GlobalIndex other_dof : event.new_dofs) {
            trackAddedEntry(new_dof, other_dof);
            if (new_dof != other_dof) {
                trackAddedEntry(other_dof, new_dof);
            }
        }
    }

    // Also add couplings between existing DOFs on child elements
    // (they may have new couplings through the refined element)
    for (std::size_t i = 0; i < event.new_dofs.size(); ++i) {
        for (std::size_t j = i; j < event.new_dofs.size(); ++j) {
            GlobalIndex dof_i = event.new_dofs[i];
            GlobalIndex dof_j = event.new_dofs[j];

            // Skip already tracked new DOFs
            bool i_is_new = std::find(event.added_dofs.begin(),
                                       event.added_dofs.end(),
                                       dof_i) != event.added_dofs.end();
            bool j_is_new = std::find(event.added_dofs.begin(),
                                       event.added_dofs.end(),
                                       dof_j) != event.added_dofs.end();

            if (!i_is_new && !j_is_new) {
                // Both existing DOFs - add coupling if not present
                trackAddedEntry(dof_i, dof_j);
                if (dof_i != dof_j) {
                    trackAddedEntry(dof_j, dof_i);
                }
            }
        }
    }

    ++stats_.total_refinements;
}

void AdaptiveSparsity::applyRefinements(std::span<const RefinementEvent> events) {
    for (const auto& event : events) {
        applyRefinement(event);
    }
}

void AdaptiveSparsity::addDofCouplings(std::span<const GlobalIndex> new_dofs,
                                       std::span<const GlobalIndex> coupled_dofs) {
    prepareForModification();

    // Expand dimensions if needed
    GlobalIndex max_dof = 0;
    for (GlobalIndex dof : new_dofs) {
        max_dof = std::max(max_dof, dof);
    }
    for (GlobalIndex dof : coupled_dofs) {
        max_dof = std::max(max_dof, dof);
    }
    resizeIfNeeded(max_dof + 1, max_dof + 1);

    // Add couplings between new_dofs and coupled_dofs
    for (GlobalIndex new_dof : new_dofs) {
        delta_.addRow(new_dof);

        for (GlobalIndex coupled_dof : coupled_dofs) {
            trackAddedEntry(new_dof, coupled_dof);
            trackAddedEntry(coupled_dof, new_dof);
        }

        // Also add self-coupling (diagonal)
        trackAddedEntry(new_dof, new_dof);
    }

    // Add couplings within new_dofs
    for (std::size_t i = 0; i < new_dofs.size(); ++i) {
        for (std::size_t j = i + 1; j < new_dofs.size(); ++j) {
            trackAddedEntry(new_dofs[i], new_dofs[j]);
            trackAddedEntry(new_dofs[j], new_dofs[i]);
        }
    }
}

void AdaptiveSparsity::addElementCouplings(std::span<const GlobalIndex> element_dofs) {
    prepareForModification();

    // Expand dimensions if needed
    GlobalIndex max_dof = 0;
    for (GlobalIndex dof : element_dofs) {
        max_dof = std::max(max_dof, dof);
    }
    resizeIfNeeded(max_dof + 1, max_dof + 1);

    // Add full element coupling
    for (GlobalIndex row : element_dofs) {
        for (GlobalIndex col : element_dofs) {
            trackAddedEntry(row, col);
        }
    }
}

// ============================================================================
// Coarsening Operations
// ============================================================================

void AdaptiveSparsity::applyCoarsening(const CoarseningEvent& event) {
    if (!event.isValid()) {
        return;
    }

    prepareForModification();

    // Remove DOFs
    for (GlobalIndex dof : event.removed_dofs) {
        delta_.removeRow(dof);
        ++stats_.total_dof_removals;

        // Track removal of all entries in this row
        if (pattern_.isFinalized()) {
            auto row_span = pattern_.getRowSpan(dof);
            for (GlobalIndex col : row_span) {
                trackRemovedEntry(dof, col);
            }
        }

        // Track removal of all entries in this column
        for (GlobalIndex row = 0; row < pattern_.numRows(); ++row) {
            if (pattern_.hasEntry(row, dof)) {
                trackRemovedEntry(row, dof);
            }
        }
    }

    // Add couplings for parent element DOFs
    for (GlobalIndex row : event.new_dofs) {
        for (GlobalIndex col : event.new_dofs) {
            trackAddedEntry(row, col);
        }
    }

    ++stats_.total_coarsenings;
}

void AdaptiveSparsity::applyCoarsenings(std::span<const CoarseningEvent> events) {
    for (const auto& event : events) {
        applyCoarsening(event);
    }
}

void AdaptiveSparsity::removeDofs(std::span<const GlobalIndex> dofs) {
    prepareForModification();

    for (GlobalIndex dof : dofs) {
        delta_.removeRow(dof);
        ++stats_.total_dof_removals;

        // Track removal of all entries involving this DOF
        if (pattern_.isFinalized() && dof < pattern_.numRows()) {
            auto row_span = pattern_.getRowSpan(dof);
            for (GlobalIndex col : row_span) {
                trackRemovedEntry(dof, col);
            }
        }

        // Remove column entries
        for (GlobalIndex row = 0; row < pattern_.numRows(); ++row) {
            if (pattern_.hasEntry(row, dof)) {
                trackRemovedEntry(row, dof);
            }
        }
    }
}

void AdaptiveSparsity::eliminateDofs(std::span<const GlobalIndex> dofs) {
    prepareForModification();

    for (GlobalIndex dof : dofs) {
        if (dof >= pattern_.numRows()) continue;

        // Remove all entries except diagonal
        if (pattern_.isFinalized()) {
            auto row_span = pattern_.getRowSpan(dof);
            for (GlobalIndex col : row_span) {
                if (col != dof) {
                    trackRemovedEntry(dof, col);
                }
            }
        }

        // Remove column entries except diagonal
        for (GlobalIndex row = 0; row < pattern_.numRows(); ++row) {
            if (row != dof && pattern_.hasEntry(row, dof)) {
                trackRemovedEntry(row, dof);
            }
        }

        // Ensure diagonal is present
        trackAddedEntry(dof, dof);
    }
}

// ============================================================================
// Renumbering Operations
// ============================================================================

void AdaptiveSparsity::applyRenumbering(const DofRenumbering& renumbering) {
    if (!renumbering.isValid()) {
        return;
    }

    // If it's an identity mapping, nothing to do
    if (renumbering.isIdentity()) {
        return;
    }

    // Force a rebuild with the new numbering
    needs_rebuild_ = true;

    // Create new pattern with renumbered indices
    SparsityPattern new_pattern(renumbering.new_n_dofs, renumbering.new_n_dofs);

    for (GlobalIndex old_row = 0; old_row < pattern_.numRows(); ++old_row) {
        if (static_cast<std::size_t>(old_row) >= renumbering.old_to_new.size()) {
            continue;
        }

        GlobalIndex new_row = renumbering.old_to_new[static_cast<std::size_t>(old_row)];
        if (new_row < 0) continue;  // DOF removed

        if (pattern_.isFinalized()) {
            auto row_span = pattern_.getRowSpan(old_row);
            for (GlobalIndex old_col : row_span) {
                if (static_cast<std::size_t>(old_col) >= renumbering.old_to_new.size()) {
                    continue;
                }

                GlobalIndex new_col = renumbering.old_to_new[static_cast<std::size_t>(old_col)];
                if (new_col < 0) continue;  // DOF removed

                new_pattern.addEntry(new_row, new_col);
            }
        }
    }

    // Also apply delta entries with renumbering
    for (const auto& [old_row, cols] : delta_.getAddedEntries()) {
        if (static_cast<std::size_t>(old_row) >= renumbering.old_to_new.size()) {
            continue;
        }

        GlobalIndex new_row = renumbering.old_to_new[static_cast<std::size_t>(old_row)];
        if (new_row < 0) continue;

        for (GlobalIndex old_col : cols) {
            if (static_cast<std::size_t>(old_col) >= renumbering.old_to_new.size()) {
                continue;
            }

            GlobalIndex new_col = renumbering.old_to_new[static_cast<std::size_t>(old_col)];
            if (new_col < 0) continue;

            new_pattern.addEntry(new_row, new_col);
        }
    }

    new_pattern.finalize();
    pattern_ = std::move(new_pattern);
    delta_.clear();
    needs_rebuild_ = false;
}

void AdaptiveSparsity::compact(std::vector<GlobalIndex>* old_to_new) {
    // Find non-empty rows
    std::vector<GlobalIndex> non_empty_rows;
    for (GlobalIndex row = 0; row < pattern_.numRows(); ++row) {
        if (pattern_.getRowNnz(row) > 0) {
            non_empty_rows.push_back(row);
        }
    }

    if (non_empty_rows.size() == static_cast<std::size_t>(pattern_.numRows())) {
        // Already compact
        if (old_to_new) {
            old_to_new->resize(static_cast<std::size_t>(pattern_.numRows()));
            for (GlobalIndex i = 0; i < pattern_.numRows(); ++i) {
                (*old_to_new)[static_cast<std::size_t>(i)] = i;
            }
        }
        return;
    }

    // Create renumbering
    DofRenumbering renumbering;
    renumbering.old_to_new.resize(static_cast<std::size_t>(pattern_.numRows()), -1);
    renumbering.new_to_old.resize(non_empty_rows.size());
    renumbering.new_n_dofs = static_cast<GlobalIndex>(non_empty_rows.size());

    for (std::size_t i = 0; i < non_empty_rows.size(); ++i) {
        GlobalIndex old_idx = non_empty_rows[i];
        renumbering.old_to_new[static_cast<std::size_t>(old_idx)] = static_cast<GlobalIndex>(i);
        renumbering.new_to_old[i] = old_idx;
    }

    if (old_to_new) {
        *old_to_new = renumbering.old_to_new;
    }

    applyRenumbering(renumbering);
}

// ============================================================================
// Pattern Merging
// ============================================================================

void AdaptiveSparsity::merge(const SparsityPattern& other) {
    prepareForModification();

    // Expand dimensions if needed
    resizeIfNeeded(std::max(pattern_.numRows(), other.numRows()),
                   std::max(pattern_.numCols(), other.numCols()));

    // Add all entries from other pattern
    for (GlobalIndex row = 0; row < other.numRows(); ++row) {
        if (other.isFinalized()) {
            auto row_span = other.getRowSpan(row);
            for (GlobalIndex col : row_span) {
                trackAddedEntry(row, col);
            }
        }
    }
}

void AdaptiveSparsity::merge(const AdaptiveSparsity& other) {
    merge(other.pattern_);
    delta_.merge(other.delta_);
}

// ============================================================================
// Finalization
// ============================================================================

void AdaptiveSparsity::finalize() {
    if (!hasPendingChanges()) {
        if (!pattern_.isFinalized()) {
            pattern_.finalize();
        }
        return;
    }

    if (shouldRebuild()) {
        // Full rebuild
        ++stats_.total_rebuilds;

        // Create new pattern with all entries
        SparsityPattern new_pattern(
            std::max(pattern_.numRows(), pending_new_rows_),
            std::max(pattern_.numCols(), pending_new_cols_));

        // Add existing entries (minus removals)
        const auto& removed = delta_.getRemovedEntries();
        const auto& removed_rows = delta_.getRemovedRows();

        for (GlobalIndex row = 0; row < pattern_.numRows(); ++row) {
            if (removed_rows.count(row)) continue;

            std::set<GlobalIndex> row_removed;
            auto rem_it = removed.find(row);
            if (rem_it != removed.end()) {
                row_removed = rem_it->second;
            }

            if (pattern_.isFinalized()) {
                auto row_span = pattern_.getRowSpan(row);
                for (GlobalIndex col : row_span) {
                    if (!row_removed.count(col)) {
                        new_pattern.addEntry(row, col);
                    }
                }
            }
        }

        // Add new entries
        for (const auto& [row, cols] : delta_.getAddedEntries()) {
            for (GlobalIndex col : cols) {
                new_pattern.addEntry(row, col);
            }
        }

        new_pattern.finalize();
        pattern_ = std::move(new_pattern);
    } else {
        // Incremental update
        ++stats_.total_incremental_updates;
        applyDelta();
    }

    delta_.clear();
    needs_rebuild_ = false;
    pending_new_rows_ = 0;
    pending_new_cols_ = 0;

    if (config_.verify_after_update) {
        if (!validate()) {
            throw FEException("Pattern validation failed after adaptive update",
                              __FILE__, __LINE__);
        }
    }
}

void AdaptiveSparsity::clearDelta() {
    delta_.clear();
    pending_new_rows_ = 0;
    pending_new_cols_ = 0;
}

// ============================================================================
// Hierarchical Support
// ============================================================================

SparsityPattern AdaptiveSparsity::createCoarsePattern(
    std::span<const GlobalIndex> fine_to_coarse) const {

    // Determine number of coarse DOFs
    GlobalIndex n_coarse = 0;
    for (GlobalIndex coarse : fine_to_coarse) {
        if (coarse >= 0) {
            n_coarse = std::max(n_coarse, coarse + 1);
        }
    }

    SparsityPattern coarse_pattern(n_coarse, n_coarse);

    // Create coarse pattern: coarse DOFs couple if any of their
    // corresponding fine DOFs couple
    for (GlobalIndex fine_row = 0; fine_row < pattern_.numRows(); ++fine_row) {
        if (static_cast<std::size_t>(fine_row) >= fine_to_coarse.size()) {
            continue;
        }

        GlobalIndex coarse_row = fine_to_coarse[static_cast<std::size_t>(fine_row)];
        if (coarse_row < 0) continue;

        if (pattern_.isFinalized()) {
            auto row_span = pattern_.getRowSpan(fine_row);
            for (GlobalIndex fine_col : row_span) {
                if (static_cast<std::size_t>(fine_col) >= fine_to_coarse.size()) {
                    continue;
                }

                GlobalIndex coarse_col = fine_to_coarse[static_cast<std::size_t>(fine_col)];
                if (coarse_col < 0) continue;

                coarse_pattern.addEntry(coarse_row, coarse_col);
            }
        }
    }

    coarse_pattern.finalize();
    return coarse_pattern;
}

SparsityPattern AdaptiveSparsity::createRestrictionPattern(
    std::span<const GlobalIndex> fine_to_coarse) const {

    // Restriction: coarse <- fine
    // Determine dimensions
    GlobalIndex n_coarse = 0;
    for (GlobalIndex coarse : fine_to_coarse) {
        if (coarse >= 0) {
            n_coarse = std::max(n_coarse, coarse + 1);
        }
    }

    GlobalIndex n_fine = static_cast<GlobalIndex>(fine_to_coarse.size());

    SparsityPattern restriction(n_coarse, n_fine);

    // Each coarse DOF receives from the fine DOFs that map to it
    for (GlobalIndex fine = 0; fine < n_fine; ++fine) {
        GlobalIndex coarse = fine_to_coarse[static_cast<std::size_t>(fine)];
        if (coarse >= 0) {
            restriction.addEntry(coarse, fine);
        }
    }

    restriction.finalize();
    return restriction;
}

SparsityPattern AdaptiveSparsity::createProlongationPattern(
    std::span<const GlobalIndex> fine_to_coarse) const {

    // Prolongation: fine <- coarse
    // Determine dimensions
    GlobalIndex n_coarse = 0;
    for (GlobalIndex coarse : fine_to_coarse) {
        if (coarse >= 0) {
            n_coarse = std::max(n_coarse, coarse + 1);
        }
    }

    GlobalIndex n_fine = static_cast<GlobalIndex>(fine_to_coarse.size());

    SparsityPattern prolongation(n_fine, n_coarse);

    // Each fine DOF receives from its corresponding coarse DOF
    for (GlobalIndex fine = 0; fine < n_fine; ++fine) {
        GlobalIndex coarse = fine_to_coarse[static_cast<std::size_t>(fine)];
        if (coarse >= 0) {
            prolongation.addEntry(fine, coarse);
        }
    }

    prolongation.finalize();
    return prolongation;
}

// ============================================================================
// Validation and Debugging
// ============================================================================

bool AdaptiveSparsity::validate() const {
    return pattern_.validate();
}

std::string AdaptiveSparsity::validationError() const {
    return pattern_.validationError();
}

// ============================================================================
// Internal Implementation
// ============================================================================

void AdaptiveSparsity::prepareForModification() {
    // If pattern is finalized and we need to modify, we'll track changes
    // and apply them during finalize
    if (!config_.track_deltas) {
        // If not tracking deltas, we need to rebuild
        needs_rebuild_ = true;
    }
}

void AdaptiveSparsity::applyDelta() {
    // Apply incremental changes to pattern
    // First, need to unfinalize if finalized
    if (pattern_.isFinalized()) {
        // Create new pattern in building state
        SparsityPattern new_pattern(pattern_.numRows(), pattern_.numCols());

        // Copy existing entries minus removals
        const auto& removed = delta_.getRemovedEntries();
        const auto& removed_rows = delta_.getRemovedRows();

        for (GlobalIndex row = 0; row < pattern_.numRows(); ++row) {
            if (removed_rows.count(row)) continue;

            std::set<GlobalIndex> row_removed;
            auto rem_it = removed.find(row);
            if (rem_it != removed.end()) {
                row_removed = rem_it->second;
            }

            auto row_span = pattern_.getRowSpan(row);
            for (GlobalIndex col : row_span) {
                if (!row_removed.count(col)) {
                    new_pattern.addEntry(row, col);
                }
            }
        }

        // Add new entries
        for (const auto& [row, cols] : delta_.getAddedEntries()) {
            for (GlobalIndex col : cols) {
                new_pattern.addEntry(row, col);
            }
        }

        new_pattern.finalize();
        pattern_ = std::move(new_pattern);
    } else {
        // Pattern is in building state - can modify directly
        // (though this path is less common)
        for (const auto& [row, cols] : delta_.getAddedEntries()) {
            for (GlobalIndex col : cols) {
                pattern_.addEntry(row, col);
            }
        }
        // Note: removal is harder in building state, may need rebuild
        if (!delta_.getRemovedEntries().empty() ||
            !delta_.getRemovedRows().empty()) {
            needs_rebuild_ = true;
        }
    }
}

bool AdaptiveSparsity::shouldRebuild() const {
    if (needs_rebuild_) return true;

    // Check if change exceeds threshold
    GlobalIndex current_nnz = pattern_.getNnz();
    if (current_nnz == 0) return true;

    GlobalIndex added = delta_.numAddedEntries();
    GlobalIndex removed = delta_.numRemovedEntries();
    GlobalIndex changed = added + removed;

    double change_ratio = static_cast<double>(changed) /
                          static_cast<double>(current_nnz);

    return change_ratio > config_.rebuild_threshold;
}

void AdaptiveSparsity::trackAddedEntry(GlobalIndex row, GlobalIndex col) {
    if (config_.track_deltas) {
        delta_.addEntry(row, col);
    }
}

void AdaptiveSparsity::trackRemovedEntry(GlobalIndex row, GlobalIndex col) {
    if (config_.track_deltas) {
        delta_.removeEntry(row, col);
    }
}

void AdaptiveSparsity::resizeIfNeeded(GlobalIndex new_n_rows, GlobalIndex new_n_cols) {
    if (new_n_rows > pattern_.numRows() || new_n_cols > pattern_.numCols()) {
        pending_new_rows_ = std::max(pending_new_rows_, new_n_rows);
        pending_new_cols_ = std::max(pending_new_cols_, new_n_cols);

        // If pattern is empty or in building state, resize now
        if (pattern_.getNnz() == 0 || !pattern_.isFinalized()) {
            pattern_.resize(std::max(pattern_.numRows(), new_n_rows),
                           std::max(pattern_.numCols(), new_n_cols));
        } else {
            // Will resize during finalize
            needs_rebuild_ = true;
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

AdaptiveSparsity makeAdaptive(SparsityPattern pattern) {
    return AdaptiveSparsity(std::move(pattern));
}

void applyRefinement(SparsityPattern& pattern, const RefinementEvent& event) {
    AdaptiveSparsity adaptive(std::move(pattern));
    adaptive.applyRefinement(event);
    adaptive.finalize();
    pattern = adaptive.releasePattern();
}

void applyCoarsening(SparsityPattern& pattern, const CoarseningEvent& event) {
    AdaptiveSparsity adaptive(std::move(pattern));
    adaptive.applyCoarsening(event);
    adaptive.finalize();
    pattern = adaptive.releasePattern();
}

SparsityPattern compactPattern(
    const SparsityPattern& pattern,
    std::vector<GlobalIndex>* old_to_new) {

    // Find non-empty rows and columns
    std::set<GlobalIndex> non_empty_rows, used_cols;

    for (GlobalIndex row = 0; row < pattern.numRows(); ++row) {
        if (pattern.getRowNnz(row) > 0) {
            non_empty_rows.insert(row);
            if (pattern.isFinalized()) {
                auto row_span = pattern.getRowSpan(row);
                for (GlobalIndex col : row_span) {
                    used_cols.insert(col);
                }
            }
        }
    }

    // Create mapping
    std::vector<GlobalIndex> row_map(static_cast<std::size_t>(pattern.numRows()), -1);
    GlobalIndex new_idx = 0;
    for (GlobalIndex old_idx : non_empty_rows) {
        row_map[static_cast<std::size_t>(old_idx)] = new_idx++;
    }

    if (old_to_new) {
        *old_to_new = row_map;
    }

    // Create compact pattern
    GlobalIndex new_n = static_cast<GlobalIndex>(non_empty_rows.size());
    SparsityPattern compact(new_n, new_n);

    for (GlobalIndex old_row : non_empty_rows) {
        GlobalIndex new_row = row_map[static_cast<std::size_t>(old_row)];

        if (pattern.isFinalized()) {
            auto row_span = pattern.getRowSpan(old_row);
            for (GlobalIndex old_col : row_span) {
                if (static_cast<std::size_t>(old_col) < row_map.size() &&
                    row_map[static_cast<std::size_t>(old_col)] >= 0) {
                    GlobalIndex new_col = row_map[static_cast<std::size_t>(old_col)];
                    compact.addEntry(new_row, new_col);
                }
            }
        }
    }

    compact.finalize();
    return compact;
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
