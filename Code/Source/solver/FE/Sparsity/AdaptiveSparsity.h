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

#ifndef SVMP_FE_ADAPTIVE_SPARSITY_H
#define SVMP_FE_ADAPTIVE_SPARSITY_H

/**
 * @file AdaptiveSparsity.h
 * @brief Dynamic sparsity pattern support for adaptive mesh refinement
 *
 * This header provides the AdaptiveSparsity class for managing sparsity
 * patterns that evolve during mesh adaptivity (h-refinement, p-refinement,
 * or coarsening). Key features:
 *
 * - Incremental pattern updates without full rebuild
 * - Refinement support: adding new DOFs from element subdivision
 * - Coarsening support: removing DOFs during mesh coarsening
 * - Pattern merging for parallel redistribution
 * - Hierarchical pattern representation for multigrid
 * - Change tracking for efficient matrix updates
 *
 * Integration with FE adaptivity workflows:
 * - Consumes refinement/coarsening events from an external adaptivity driver
 * - Respects hanging node / MPC constraints (via ConstraintSparsityAugmenter)
 * - Supports incremental DOF renumbering
 *
 * Performance considerations:
 * - Incremental update: O(affected_nnz) vs O(total_nnz) for rebuild
 * - Memory: maintains delta structures for efficient updates
 * - Determinism: updates are deterministic given ordered input
 *
 * @see SparsityPattern for the base pattern representation
 * @see SparsityBuilder for initial pattern construction
 * @see ConstraintSparsityAugmenter for hanging node constraints
 */

#include "SparsityPattern.h"
#include "DistributedSparsityPattern.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>
#include <functional>
#include <optional>
#include <set>
#include <map>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// Adaptivity Events and Actions
// ============================================================================

/**
 * @brief Type of adaptivity operation
 */
enum class AdaptivityAction : std::uint8_t {
    None,           ///< No action
    Refine,         ///< Element refinement (subdivision)
    Coarsen,        ///< Element coarsening (merging)
    PRefine,        ///< Polynomial order increase
    PCoarsen,       ///< Polynomial order decrease
    Redistribute    ///< Parallel redistribution
};

/**
 * @brief Element refinement event
 *
 * Describes how an element was refined and the resulting DOF changes.
 */
struct RefinementEvent {
    GlobalIndex parent_element{-1};           ///< Original element ID
    std::vector<GlobalIndex> child_elements;  ///< New child element IDs
    std::vector<GlobalIndex> old_dofs;        ///< DOFs on parent element
    std::vector<GlobalIndex> new_dofs;        ///< All DOFs on children (includes new)
    std::vector<GlobalIndex> added_dofs;      ///< Newly created DOFs

    /**
     * @brief Check if event is valid
     */
    [[nodiscard]] bool isValid() const noexcept {
        return parent_element >= 0 && !child_elements.empty();
    }
};

/**
 * @brief Element coarsening event
 *
 * Describes how elements were coarsened and the resulting DOF changes.
 */
struct CoarseningEvent {
    std::vector<GlobalIndex> child_elements;  ///< Elements being coarsened
    GlobalIndex parent_element{-1};           ///< Resulting parent element
    std::vector<GlobalIndex> old_dofs;        ///< All DOFs on children
    std::vector<GlobalIndex> new_dofs;        ///< DOFs on parent element
    std::vector<GlobalIndex> removed_dofs;    ///< DOFs that will be removed

    /**
     * @brief Check if event is valid
     */
    [[nodiscard]] bool isValid() const noexcept {
        return parent_element >= 0 && !child_elements.empty();
    }
};

/**
 * @brief DOF renumbering map
 */
struct DofRenumbering {
    std::vector<GlobalIndex> old_to_new;  ///< old_to_new[old_dof] = new_dof
    std::vector<GlobalIndex> new_to_old;  ///< new_to_old[new_dof] = old_dof
    GlobalIndex new_n_dofs{0};            ///< Total DOFs after renumbering

    /**
     * @brief Check if renumbering is valid
     */
    [[nodiscard]] bool isValid() const noexcept {
        return !old_to_new.empty() && new_n_dofs > 0;
    }

    /**
     * @brief Check if this is an identity mapping
     */
    [[nodiscard]] bool isIdentity() const noexcept {
        for (std::size_t i = 0; i < old_to_new.size(); ++i) {
            if (old_to_new[i] != static_cast<GlobalIndex>(i)) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Pattern Delta Tracking
// ============================================================================

/**
 * @brief Tracks changes to sparsity pattern
 *
 * Maintains added and removed entries for efficient incremental updates.
 */
class PatternDelta {
public:
    /**
     * @brief Default constructor
     */
    PatternDelta() = default;

    /**
     * @brief Clear all tracked changes
     */
    void clear();

    /**
     * @brief Track entry addition
     */
    void addEntry(GlobalIndex row, GlobalIndex col);

    /**
     * @brief Track multiple entry additions
     */
    void addEntries(GlobalIndex row, std::span<const GlobalIndex> cols);

    /**
     * @brief Track entry removal
     */
    void removeEntry(GlobalIndex row, GlobalIndex col);

    /**
     * @brief Track row addition (empty row)
     */
    void addRow(GlobalIndex row);

    /**
     * @brief Track row removal
     */
    void removeRow(GlobalIndex row);

    /**
     * @brief Track column addition
     */
    void addColumn(GlobalIndex col);

    /**
     * @brief Track column removal
     */
    void removeColumn(GlobalIndex col);

    /**
     * @brief Get added entries
     */
    [[nodiscard]] const std::map<GlobalIndex, std::set<GlobalIndex>>& getAddedEntries() const {
        return added_entries_;
    }

    /**
     * @brief Get removed entries
     */
    [[nodiscard]] const std::map<GlobalIndex, std::set<GlobalIndex>>& getRemovedEntries() const {
        return removed_entries_;
    }

    /**
     * @brief Get added rows
     */
    [[nodiscard]] const std::set<GlobalIndex>& getAddedRows() const {
        return added_rows_;
    }

    /**
     * @brief Get removed rows
     */
    [[nodiscard]] const std::set<GlobalIndex>& getRemovedRows() const {
        return removed_rows_;
    }

    /**
     * @brief Get number of added entries
     */
    [[nodiscard]] GlobalIndex numAddedEntries() const;

    /**
     * @brief Get number of removed entries
     */
    [[nodiscard]] GlobalIndex numRemovedEntries() const;

    /**
     * @brief Check if there are any changes
     */
    [[nodiscard]] bool hasChanges() const noexcept {
        return !added_entries_.empty() || !removed_entries_.empty() ||
               !added_rows_.empty() || !removed_rows_.empty();
    }

    /**
     * @brief Merge another delta into this one
     */
    void merge(const PatternDelta& other);

private:
    std::map<GlobalIndex, std::set<GlobalIndex>> added_entries_;
    std::map<GlobalIndex, std::set<GlobalIndex>> removed_entries_;
    std::set<GlobalIndex> added_rows_;
    std::set<GlobalIndex> removed_rows_;
    std::set<GlobalIndex> added_cols_;
    std::set<GlobalIndex> removed_cols_;
};

// ============================================================================
// Adaptive Sparsity Configuration
// ============================================================================

/**
 * @brief Configuration for adaptive sparsity management
 */
struct AdaptiveSparsityConfig {
    /**
     * @brief Rebuild threshold: fraction of NNZ change triggering full rebuild
     *
     * If more than this fraction of entries change, rebuild from scratch
     * instead of incremental update.
     */
    double rebuild_threshold{0.5};

    /**
     * @brief Enable delta tracking
     *
     * If false, always rebuild pattern after changes.
     */
    bool track_deltas{true};

    /**
     * @brief Reserve extra capacity for growth
     *
     * Factor by which to over-allocate for anticipated growth.
     */
    double growth_factor{1.5};

    /**
     * @brief Keep removed DOF slots as empty rows
     *
     * If true, removed DOFs leave empty rows in pattern.
     * If false, pattern is compacted.
     */
    bool keep_empty_rows{false};

    /**
     * @brief Verify pattern integrity after updates
     */
    bool verify_after_update{false};
};

// ============================================================================
// AdaptiveSparsity Class
// ============================================================================

/**
 * @brief Manages dynamic sparsity patterns for adaptive simulations
 *
 * AdaptiveSparsity provides efficient incremental updates to sparsity
 * patterns during mesh adaptivity, avoiding expensive full rebuilds.
 *
 * Usage:
 * @code
 * // Create adaptive sparsity from initial pattern
 * SparsityPattern initial_pattern = buildPattern(...);
 * AdaptiveSparsity adaptive(std::move(initial_pattern));
 *
 * // Handle mesh refinement
 * RefinementEvent event;
 * event.parent_element = elem_id;
 * event.child_elements = {child1, child2, child3, child4};
 * event.old_dofs = getElementDofs(elem_id);
 * event.new_dofs = getAllChildDofs(event.child_elements);
 * event.added_dofs = getNewlyCreatedDofs(...);
 *
 * adaptive.applyRefinement(event);
 *
 * // Handle mesh coarsening
 * CoarseningEvent coarsen_event;
 * coarsen_event.child_elements = {child1, child2, child3, child4};
 * coarsen_event.parent_element = parent_id;
 * coarsen_event.removed_dofs = getDofsToRemove(...);
 *
 * adaptive.applyCoarsening(coarsen_event);
 *
 * // Apply accumulated changes and get updated pattern
 * adaptive.finalize();
 * const SparsityPattern& updated = adaptive.getPattern();
 *
 * // Or get the changes for incremental matrix update
 * const PatternDelta& delta = adaptive.getDelta();
 * @endcode
 */
class AdaptiveSparsity {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (empty pattern)
     */
    AdaptiveSparsity();

    /**
     * @brief Construct with initial pattern
     *
     * @param pattern Initial sparsity pattern (moved)
     * @param config Configuration options
     */
    explicit AdaptiveSparsity(SparsityPattern pattern,
                              const AdaptiveSparsityConfig& config = {});

    /**
     * @brief Construct with dimensions
     *
     * @param n_rows Number of rows
     * @param n_cols Number of columns
     * @param config Configuration options
     */
    AdaptiveSparsity(GlobalIndex n_rows, GlobalIndex n_cols,
                     const AdaptiveSparsityConfig& config = {});

    /// Destructor
    ~AdaptiveSparsity() = default;

    // Non-copyable (owns pattern)
    AdaptiveSparsity(const AdaptiveSparsity&) = delete;
    AdaptiveSparsity& operator=(const AdaptiveSparsity&) = delete;

    // Movable
    AdaptiveSparsity(AdaptiveSparsity&&) = default;
    AdaptiveSparsity& operator=(AdaptiveSparsity&&) = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set configuration
     */
    void setConfig(const AdaptiveSparsityConfig& config) {
        config_ = config;
    }

    /**
     * @brief Get current configuration
     */
    [[nodiscard]] const AdaptiveSparsityConfig& getConfig() const noexcept {
        return config_;
    }

    // =========================================================================
    // Pattern Access
    // =========================================================================

    /**
     * @brief Get current pattern (const reference)
     *
     * @note Pattern may be in Building state if changes are pending.
     */
    [[nodiscard]] const SparsityPattern& getPattern() const noexcept {
        return pattern_;
    }

    /**
     * @brief Get current pattern (mutable, for direct modification)
     *
     * @warning Direct modifications bypass delta tracking.
     */
    [[nodiscard]] SparsityPattern& getPatternMutable() noexcept {
        return pattern_;
    }

    /**
     * @brief Get pattern and release ownership
     *
     * After this call, AdaptiveSparsity is empty.
     */
    [[nodiscard]] SparsityPattern releasePattern();

    /**
     * @brief Get accumulated delta
     */
    [[nodiscard]] const PatternDelta& getDelta() const noexcept {
        return delta_;
    }

    // =========================================================================
    // Refinement Operations
    // =========================================================================

    /**
     * @brief Apply refinement event
     *
     * Adds new DOF couplings for refined elements.
     *
     * @param event Refinement event describing the change
     */
    void applyRefinement(const RefinementEvent& event);

    /**
     * @brief Apply multiple refinement events
     *
     * @param events Vector of refinement events
     */
    void applyRefinements(std::span<const RefinementEvent> events);

    /**
     * @brief Add couplings for new DOFs
     *
     * @param new_dofs DOFs to add
     * @param coupled_dofs Existing DOFs that new DOFs couple to
     *
     * Creates full coupling between new_dofs and coupled_dofs.
     */
    void addDofCouplings(std::span<const GlobalIndex> new_dofs,
                         std::span<const GlobalIndex> coupled_dofs);

    /**
     * @brief Add element coupling for new element
     *
     * @param element_dofs DOFs on the new element
     *
     * Creates full element coupling matrix.
     */
    void addElementCouplings(std::span<const GlobalIndex> element_dofs);

    // =========================================================================
    // Coarsening Operations
    // =========================================================================

    /**
     * @brief Apply coarsening event
     *
     * Removes DOF couplings for coarsened elements.
     *
     * @param event Coarsening event describing the change
     */
    void applyCoarsening(const CoarseningEvent& event);

    /**
     * @brief Apply multiple coarsening events
     *
     * @param events Vector of coarsening events
     */
    void applyCoarsenings(std::span<const CoarseningEvent> events);

    /**
     * @brief Remove DOFs from pattern
     *
     * @param dofs DOFs to remove
     *
     * Removes all couplings involving these DOFs.
     */
    void removeDofs(std::span<const GlobalIndex> dofs);

    /**
     * @brief Mark DOFs as eliminated (keep rows but clear entries)
     *
     * @param dofs DOFs to eliminate
     *
     * Keeps diagonal entry only for eliminated DOFs.
     */
    void eliminateDofs(std::span<const GlobalIndex> dofs);

    // =========================================================================
    // Renumbering Operations
    // =========================================================================

    /**
     * @brief Apply DOF renumbering
     *
     * @param renumbering Renumbering map
     *
     * Updates all row/column indices according to the renumbering.
     */
    void applyRenumbering(const DofRenumbering& renumbering);

    /**
     * @brief Compact pattern by removing empty rows/columns
     *
     * @param[out] old_to_new Optional output: mapping from old to new indices
     */
    void compact(std::vector<GlobalIndex>* old_to_new = nullptr);

    // =========================================================================
    // Pattern Merging
    // =========================================================================

    /**
     * @brief Merge another pattern into this one
     *
     * @param other Pattern to merge (union of entries)
     *
     * Used for parallel redistribution or pattern combination.
     */
    void merge(const SparsityPattern& other);

    /**
     * @brief Merge another adaptive sparsity into this one
     *
     * @param other AdaptiveSparsity to merge
     *
     * Merges both pattern and delta.
     */
    void merge(const AdaptiveSparsity& other);

    // =========================================================================
    // Finalization
    // =========================================================================

    /**
     * @brief Apply all pending changes
     *
     * Finalizes the pattern if needed.
     */
    void finalize();

    /**
     * @brief Clear delta tracking (keep pattern)
     */
    void clearDelta();

    /**
     * @brief Check if there are pending changes
     */
    [[nodiscard]] bool hasPendingChanges() const noexcept {
        return delta_.hasChanges() || needs_rebuild_;
    }

    /**
     * @brief Check if full rebuild is needed
     */
    [[nodiscard]] bool needsRebuild() const noexcept {
        return needs_rebuild_;
    }

    /**
     * @brief Force full rebuild on next finalize
     */
    void requestRebuild() {
        needs_rebuild_ = true;
    }

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * @brief Get current number of rows
     */
    [[nodiscard]] GlobalIndex numRows() const noexcept {
        return pattern_.numRows();
    }

    /**
     * @brief Get current number of columns
     */
    [[nodiscard]] GlobalIndex numCols() const noexcept {
        return pattern_.numCols();
    }

    /**
     * @brief Get current NNZ
     */
    [[nodiscard]] GlobalIndex getNnz() const noexcept {
        return pattern_.getNnz();
    }

    /**
     * @brief Get number of DOF additions since last finalize
     */
    [[nodiscard]] GlobalIndex numAddedDofs() const noexcept {
        return static_cast<GlobalIndex>(delta_.getAddedRows().size());
    }

    /**
     * @brief Get number of DOF removals since last finalize
     */
    [[nodiscard]] GlobalIndex numRemovedDofs() const noexcept {
        return static_cast<GlobalIndex>(delta_.getRemovedRows().size());
    }

    // =========================================================================
    // Hierarchical Support (for Multigrid)
    // =========================================================================

    /**
     * @brief Create coarse level pattern
     *
     * @param fine_to_coarse Mapping from fine DOFs to coarse DOFs
     * @return Coarse level pattern
     *
     * Used to build coarse level operators for multigrid.
     */
    [[nodiscard]] SparsityPattern createCoarsePattern(
        std::span<const GlobalIndex> fine_to_coarse) const;

    /**
     * @brief Create restriction operator pattern
     *
     * @param fine_to_coarse Mapping from fine DOFs to coarse DOFs
     * @return Restriction pattern (coarse x fine)
     */
    [[nodiscard]] SparsityPattern createRestrictionPattern(
        std::span<const GlobalIndex> fine_to_coarse) const;

    /**
     * @brief Create prolongation operator pattern
     *
     * @param fine_to_coarse Mapping from fine DOFs to coarse DOFs
     * @return Prolongation pattern (fine x coarse)
     */
    [[nodiscard]] SparsityPattern createProlongationPattern(
        std::span<const GlobalIndex> fine_to_coarse) const;

    // =========================================================================
    // Validation and Debugging
    // =========================================================================

    /**
     * @brief Validate pattern integrity
     */
    [[nodiscard]] bool validate() const;

    /**
     * @brief Get validation error message
     */
    [[nodiscard]] std::string validationError() const;

    /**
     * @brief Get statistics about adaptivity operations
     */
    struct AdaptivityStats {
        GlobalIndex total_refinements{0};
        GlobalIndex total_coarsenings{0};
        GlobalIndex total_dof_additions{0};
        GlobalIndex total_dof_removals{0};
        GlobalIndex total_rebuilds{0};
        GlobalIndex total_incremental_updates{0};
    };

    [[nodiscard]] AdaptivityStats getStats() const noexcept {
        return stats_;
    }

private:
    // Internal implementation
    void prepareForModification();
    void applyDelta();
    bool shouldRebuild() const;
    void trackAddedEntry(GlobalIndex row, GlobalIndex col);
    void trackRemovedEntry(GlobalIndex row, GlobalIndex col);
    void resizeIfNeeded(GlobalIndex new_n_rows, GlobalIndex new_n_cols);

    // Data members
    SparsityPattern pattern_;
    PatternDelta delta_;
    AdaptiveSparsityConfig config_;
    AdaptivityStats stats_;
    bool needs_rebuild_{false};
    GlobalIndex pending_new_rows_{0};
    GlobalIndex pending_new_cols_{0};
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Create adaptive sparsity from pattern
 */
[[nodiscard]] AdaptiveSparsity makeAdaptive(SparsityPattern pattern);

/**
 * @brief Apply refinement to pattern in-place
 */
void applyRefinement(SparsityPattern& pattern, const RefinementEvent& event);

/**
 * @brief Apply coarsening to pattern in-place
 */
void applyCoarsening(SparsityPattern& pattern, const CoarseningEvent& event);

/**
 * @brief Compact pattern by removing empty rows
 */
[[nodiscard]] SparsityPattern compactPattern(
    const SparsityPattern& pattern,
    std::vector<GlobalIndex>* old_to_new = nullptr);

} // namespace sparsity
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ADAPTIVE_SPARSITY_H
