/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_DOFINDEXSET_H
#define SVMP_FE_DOFS_DOFINDEXSET_H

/**
 * @file DofIndexSet.h
 * @brief First-class partition abstraction for DOF index sets
 *
 * The DofIndexSet class provides an efficient representation for sets of
 * DOF indices, supporting the key parallel FE concepts:
 *  - Locally owned DOFs (this rank solves for these)
 *  - Locally relevant DOFs (owned + ghost, needed for assembly)
 *  - Ghost DOFs (not owned, but needed for computation)
 *
 * The implementation uses compressed interval lists for contiguous ranges
 * and falls back to sorted vectors for non-contiguous sets. This balances
 * memory efficiency with O(log n) lookup performance.
 *
 * Key design decisions:
 *  - Immutable after construction (thread-safe)
 *  - Backend-agnostic (provides hints for PETSc/Trilinos conversion)
 *  - Supports set operations (union, intersection, difference)
 *  - Stable iteration order (sorted by index)
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <cstdint>
#include <optional>
#include <algorithm>
#include <iterator>

namespace svmp {
namespace FE {
namespace dofs {

/**
 * @brief Compressed representation of an index interval
 */
struct IndexInterval {
    GlobalIndex begin{0};  ///< First index (inclusive)
    GlobalIndex end{0};    ///< Past-last index (exclusive)

    [[nodiscard]] GlobalIndex size() const noexcept { return end - begin; }
    [[nodiscard]] bool empty() const noexcept { return begin >= end; }
    [[nodiscard]] bool contains(GlobalIndex idx) const noexcept {
        return idx >= begin && idx < end;
    }

    bool operator==(const IndexInterval& other) const noexcept {
        return begin == other.begin && end == other.end;
    }
};

/**
 * @brief Hints for backend index set conversion
 */
struct BackendMapHints {
    bool is_contiguous{false};       ///< True if indices form a single range
    GlobalIndex range_begin{0};      ///< Start of range (if contiguous)
    GlobalIndex range_end{0};        ///< End of range (if contiguous)
    GlobalIndex global_size{0};      ///< Total DOFs in system
    int owning_rank{0};              ///< MPI rank for this set
};

/**
 * @brief Iterator for IndexSet
 *
 * Iterates over all indices in sorted order, handling both
 * interval-compressed and explicit storage.
 */
class IndexSetIterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = GlobalIndex;
    using difference_type = std::ptrdiff_t;
    using pointer = const GlobalIndex*;
    using reference = GlobalIndex;

    IndexSetIterator() = default;

    // For interval-based iteration
    IndexSetIterator(const std::vector<IndexInterval>* intervals,
                     std::size_t interval_idx, GlobalIndex pos)
        : intervals_(intervals)
        , interval_idx_(interval_idx)
        , current_pos_(pos)
    {}

    // For explicit index iteration
    IndexSetIterator(const std::vector<GlobalIndex>* indices, std::size_t pos)
        : explicit_indices_(indices)
        , explicit_pos_(pos)
    {}

    reference operator*() const {
        if (intervals_) {
            return current_pos_;
        }
        return (*explicit_indices_)[explicit_pos_];
    }

    IndexSetIterator& operator++() {
        if (intervals_) {
            ++current_pos_;
            // Check if we need to move to next interval
            while (interval_idx_ < intervals_->size() &&
                   current_pos_ >= (*intervals_)[interval_idx_].end) {
                ++interval_idx_;
                if (interval_idx_ < intervals_->size()) {
                    current_pos_ = (*intervals_)[interval_idx_].begin;
                }
            }
        } else if (explicit_indices_) {
            ++explicit_pos_;
        }
        return *this;
    }

    IndexSetIterator operator++(int) {
        IndexSetIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const IndexSetIterator& other) const {
        if (intervals_) {
            return interval_idx_ == other.interval_idx_ &&
                   current_pos_ == other.current_pos_;
        }
        return explicit_pos_ == other.explicit_pos_;
    }

    bool operator!=(const IndexSetIterator& other) const {
        return !(*this == other);
    }

private:
    // Interval-based storage
    const std::vector<IndexInterval>* intervals_{nullptr};
    std::size_t interval_idx_{0};
    GlobalIndex current_pos_{0};

    // Explicit storage
    const std::vector<GlobalIndex>* explicit_indices_{nullptr};
    std::size_t explicit_pos_{0};
};

/**
 * @brief Efficient set of DOF indices with compressed storage
 *
 * IndexSet stores DOF indices in a compressed format optimized for
 * the common case of contiguous or nearly-contiguous index ranges.
 *
 * Storage modes:
 * 1. Contiguous: Single interval [begin, end)
 * 2. Interval-compressed: Multiple non-overlapping intervals
 * 3. Explicit: Sorted vector of individual indices
 *
 * The class automatically selects the most efficient representation.
 */
class IndexSet {
public:
    using iterator = IndexSetIterator;
    using const_iterator = IndexSetIterator;

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor - creates empty set
     */
    IndexSet() = default;

    /**
     * @brief Construct contiguous range [begin, end)
     */
    IndexSet(GlobalIndex begin, GlobalIndex end);

    /**
     * @brief Construct from interval
     */
    explicit IndexSet(IndexInterval interval);

    /**
     * @brief Construct from multiple intervals
     *
     * Intervals will be sorted and merged if overlapping.
     */
    explicit IndexSet(std::vector<IndexInterval> intervals);

    /**
     * @brief Construct from explicit indices
     *
     * Indices will be sorted and deduplicated.
     */
    explicit IndexSet(std::vector<GlobalIndex> indices);

    /**
     * @brief Construct from span of indices
     */
    explicit IndexSet(std::span<const GlobalIndex> indices);

    // =========================================================================
    // Query methods
    // =========================================================================

    /**
     * @brief Check if set is empty
     */
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /**
     * @brief Get number of indices in the set
     */
    [[nodiscard]] GlobalIndex size() const noexcept { return size_; }

    /**
     * @brief Check if index is in the set
     *
     * O(log n) where n is the number of intervals or explicit indices.
     */
    [[nodiscard]] bool contains(GlobalIndex idx) const noexcept;

    /**
     * @brief Check if set is a contiguous range
     */
    [[nodiscard]] bool isContiguous() const noexcept {
        return intervals_.size() == 1;
    }

    /**
     * @brief Get the range if contiguous
     *
     * @return Interval if contiguous, nullopt otherwise
     */
    [[nodiscard]] std::optional<IndexInterval> contiguousRange() const noexcept {
        if (isContiguous()) {
            return intervals_.front();
        }
        return std::nullopt;
    }

    /**
     * @brief Get minimum index in set
     * @return Minimum index, or -1 if empty
     */
    [[nodiscard]] GlobalIndex minIndex() const noexcept;

    /**
     * @brief Get maximum index in set
     * @return Maximum index, or -1 if empty
     */
    [[nodiscard]] GlobalIndex maxIndex() const noexcept;

    // =========================================================================
    // Iteration
    // =========================================================================

    /**
     * @brief Get iterator to first index
     */
    [[nodiscard]] iterator begin() const;

    /**
     * @brief Get iterator past last index
     */
    [[nodiscard]] iterator end() const;

    /**
     * @brief Convert to explicit vector of indices
     *
     * This allocates and returns a new vector. Use iterators for
     * memory-efficient traversal.
     */
    [[nodiscard]] std::vector<GlobalIndex> toVector() const;

    // =========================================================================
    // Set operations
    // =========================================================================

    /**
     * @brief Union of two index sets
     */
    [[nodiscard]] IndexSet unionWith(const IndexSet& other) const;

    /**
     * @brief Intersection of two index sets
     */
    [[nodiscard]] IndexSet intersectionWith(const IndexSet& other) const;

    /**
     * @brief Difference: indices in this but not in other
     */
    [[nodiscard]] IndexSet difference(const IndexSet& other) const;

    /**
     * @brief Add a single index
     *
     * @note Returns a new IndexSet (this class is immutable)
     */
    [[nodiscard]] IndexSet add(GlobalIndex idx) const;

    /**
     * @brief Remove a single index
     *
     * @note Returns a new IndexSet (this class is immutable)
     */
    [[nodiscard]] IndexSet remove(GlobalIndex idx) const;

    // =========================================================================
    // Backend conversion helpers
    // =========================================================================

    /**
     * @brief Get hints for backend index set creation
     */
    [[nodiscard]] BackendMapHints getBackendMapHints() const noexcept;

    /**
     * @brief Set global size (total DOFs in system)
     *
     * This is metadata used by backends, doesn't affect the set contents.
     */
    void setGlobalSize(GlobalIndex global_size) { global_size_ = global_size; }

    /**
     * @brief Set owning rank
     */
    void setOwningRank(int rank) { owning_rank_ = rank; }

    // =========================================================================
    // Raw access (for advanced use/serialization)
    // =========================================================================

    /**
     * @brief Get intervals (may be empty if using explicit storage)
     */
    [[nodiscard]] std::span<const IndexInterval> intervals() const noexcept {
        return intervals_;
    }

    /**
     * @brief Check if using explicit index storage
     */
    [[nodiscard]] bool usesExplicitStorage() const noexcept {
        return use_explicit_storage_;
    }

    /**
     * @brief Get explicit indices (may be empty if using interval storage)
     */
    [[nodiscard]] std::span<const GlobalIndex> explicitIndices() const noexcept {
        return explicit_indices_;
    }

    // =========================================================================
    // Comparison
    // =========================================================================

    bool operator==(const IndexSet& other) const;
    bool operator!=(const IndexSet& other) const { return !(*this == other); }

private:
    // Helper to compress indices into intervals
    void compressToIntervals();

    // Helper to merge overlapping intervals
    void mergeIntervals();

    // Interval-compressed storage
    std::vector<IndexInterval> intervals_;

    // Explicit index storage (used when compression doesn't help)
    std::vector<GlobalIndex> explicit_indices_;
    bool use_explicit_storage_{false};

    // Cached size
    GlobalIndex size_{0};

    // Metadata for backends
    GlobalIndex global_size_{0};
    int owning_rank_{0};
};

/**
 * @brief DOF partition with owned, relevant, and ghost sets
 *
 * This class bundles the three key index sets for parallel FE:
 * - Locally owned: DOFs this rank is responsible for
 * - Ghost: DOFs owned by other ranks but needed locally
 * - Locally relevant: owned + ghost (all DOFs needed for assembly)
 *
 * Invariant: locally_relevant = locally_owned ∪ ghost
 */
class DofPartition {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    DofPartition() = default;

    /**
     * @brief Construct from owned and ghost sets
     *
     * Locally relevant is computed automatically as union.
     */
    DofPartition(IndexSet owned, IndexSet ghost);

    /**
     * @brief Construct from owned range and ghost indices
     *
     * Common case for contiguous ownership.
     */
    DofPartition(GlobalIndex owned_begin, GlobalIndex owned_end,
                 std::span<const GlobalIndex> ghost_indices);

    // =========================================================================
    // Access
    // =========================================================================

    /**
     * @brief Get locally owned DOFs
     */
    [[nodiscard]] const IndexSet& locallyOwned() const noexcept { return owned_; }

    /**
     * @brief Get locally relevant DOFs (owned + ghost)
     */
    [[nodiscard]] const IndexSet& locallyRelevant() const noexcept { return relevant_; }

    /**
     * @brief Get ghost DOFs
     */
    [[nodiscard]] const IndexSet& ghost() const noexcept { return ghost_; }

    // =========================================================================
    // Size queries
    // =========================================================================

    /**
     * @brief Get total DOFs in the global system
     */
    [[nodiscard]] GlobalIndex globalSize() const noexcept { return global_size_; }

    /**
     * @brief Get number of locally owned DOFs
     */
    [[nodiscard]] GlobalIndex localOwnedSize() const noexcept { return owned_.size(); }

    /**
     * @brief Get number of locally relevant DOFs (owned + ghost)
     */
    [[nodiscard]] GlobalIndex localRelevantSize() const noexcept { return relevant_.size(); }

    /**
     * @brief Get number of ghost DOFs
     */
    [[nodiscard]] GlobalIndex ghostSize() const noexcept { return ghost_.size(); }

    /**
     * @brief Set global system size
     */
    void setGlobalSize(GlobalIndex size);

    // =========================================================================
    // Queries
    // =========================================================================

    /**
     * @brief Check if a DOF is locally owned
     */
    [[nodiscard]] bool isOwned(GlobalIndex dof) const noexcept {
        return owned_.contains(dof);
    }

    /**
     * @brief Check if a DOF is a ghost
     */
    [[nodiscard]] bool isGhost(GlobalIndex dof) const noexcept {
        return ghost_.contains(dof);
    }

    /**
     * @brief Check if a DOF is locally relevant (owned or ghost)
     */
    [[nodiscard]] bool isRelevant(GlobalIndex dof) const noexcept {
        return relevant_.contains(dof);
    }

private:
    IndexSet owned_;
    IndexSet ghost_;
    IndexSet relevant_;
    GlobalIndex global_size_{0};
};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_DOFINDEXSET_H
