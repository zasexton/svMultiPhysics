/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "DofIndexSet.h"
#include <algorithm>
#include <numeric>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// IndexSet Construction
// =============================================================================

IndexSet::IndexSet(GlobalIndex begin, GlobalIndex end) {
    if (end > begin) {
        intervals_.push_back({begin, end});
        size_ = end - begin;
    }
}

IndexSet::IndexSet(IndexInterval interval) {
    if (!interval.empty()) {
        intervals_.push_back(interval);
        size_ = interval.size();
    }
}

IndexSet::IndexSet(std::vector<IndexInterval> intervals)
    : intervals_(std::move(intervals))
{
    mergeIntervals();
    size_ = std::accumulate(intervals_.begin(), intervals_.end(), GlobalIndex{0},
        [](GlobalIndex sum, const IndexInterval& iv) { return sum + iv.size(); });
}

IndexSet::IndexSet(std::vector<GlobalIndex> indices)
    : explicit_indices_(std::move(indices))
{
    // Sort and deduplicate
    std::sort(explicit_indices_.begin(), explicit_indices_.end());
    explicit_indices_.erase(
        std::unique(explicit_indices_.begin(), explicit_indices_.end()),
        explicit_indices_.end());

    size_ = static_cast<GlobalIndex>(explicit_indices_.size());

    // Try to compress to intervals
    compressToIntervals();
}

IndexSet::IndexSet(std::span<const GlobalIndex> indices)
    : IndexSet(std::vector<GlobalIndex>(indices.begin(), indices.end()))
{
}

// =============================================================================
// IndexSet Query Methods
// =============================================================================

bool IndexSet::contains(GlobalIndex idx) const noexcept {
    if (use_explicit_storage_) {
        return std::binary_search(explicit_indices_.begin(),
                                   explicit_indices_.end(), idx);
    }

    // Binary search through intervals
    auto it = std::lower_bound(intervals_.begin(), intervals_.end(), idx,
        [](const IndexInterval& iv, GlobalIndex val) {
            return iv.end <= val;
        });

    return it != intervals_.end() && it->contains(idx);
}

GlobalIndex IndexSet::minIndex() const noexcept {
    if (empty()) return -1;

    if (use_explicit_storage_) {
        return explicit_indices_.front();
    }
    return intervals_.front().begin;
}

GlobalIndex IndexSet::maxIndex() const noexcept {
    if (empty()) return -1;

    if (use_explicit_storage_) {
        return explicit_indices_.back();
    }
    return intervals_.back().end - 1;
}

// =============================================================================
// IndexSet Iteration
// =============================================================================

IndexSet::iterator IndexSet::begin() const {
    if (empty()) return end();

    if (use_explicit_storage_) {
        return IndexSetIterator(&explicit_indices_, 0);
    }
    return IndexSetIterator(&intervals_, 0, intervals_.front().begin);
}

IndexSet::iterator IndexSet::end() const {
    if (use_explicit_storage_) {
        return IndexSetIterator(&explicit_indices_, explicit_indices_.size());
    }
    // For interval-based iteration, we treat the end iterator as "past the last
    // interval" and set the position to the end of the last interval so that
    // a fully-advanced iterator compares equal to end().
    GlobalIndex end_pos = 0;
    if (!intervals_.empty()) {
        end_pos = intervals_.back().end;
    }
    return IndexSetIterator(&intervals_, intervals_.size(), end_pos);
}

std::vector<GlobalIndex> IndexSet::toVector() const {
    std::vector<GlobalIndex> result;
    result.reserve(static_cast<std::size_t>(size_));

    if (use_explicit_storage_) {
        return explicit_indices_;
    }

    for (const auto& interval : intervals_) {
        for (GlobalIndex i = interval.begin; i < interval.end; ++i) {
            result.push_back(i);
        }
    }

    return result;
}

// =============================================================================
// IndexSet Set Operations
// =============================================================================

IndexSet IndexSet::unionWith(const IndexSet& other) const {
    if (empty()) return other;
    if (other.empty()) return *this;

    // Collect all indices and rebuild
    std::vector<GlobalIndex> combined;
    combined.reserve(static_cast<std::size_t>(size_ + other.size_));

    for (auto idx : *this) combined.push_back(idx);
    for (auto idx : other) combined.push_back(idx);

    return IndexSet(std::move(combined));
}

IndexSet IndexSet::intersectionWith(const IndexSet& other) const {
    if (empty() || other.empty()) return IndexSet();

    std::vector<GlobalIndex> result;

    // Use smaller set for iteration
    const IndexSet& smaller = (size_ <= other.size_) ? *this : other;
    const IndexSet& larger = (size_ <= other.size_) ? other : *this;

    for (auto idx : smaller) {
        if (larger.contains(idx)) {
            result.push_back(idx);
        }
    }

    return IndexSet(std::move(result));
}

IndexSet IndexSet::difference(const IndexSet& other) const {
    if (empty() || other.empty()) return *this;

    std::vector<GlobalIndex> result;

    for (auto idx : *this) {
        if (!other.contains(idx)) {
            result.push_back(idx);
        }
    }

    return IndexSet(std::move(result));
}

IndexSet IndexSet::add(GlobalIndex idx) const {
    if (contains(idx)) return *this;

    std::vector<GlobalIndex> indices = toVector();
    indices.push_back(idx);
    return IndexSet(std::move(indices));
}

IndexSet IndexSet::remove(GlobalIndex idx) const {
    if (!contains(idx)) return *this;

    std::vector<GlobalIndex> indices;
    indices.reserve(static_cast<std::size_t>(size_ - 1));

    for (auto i : *this) {
        if (i != idx) {
            indices.push_back(i);
        }
    }

    return IndexSet(std::move(indices));
}

// =============================================================================
// IndexSet Backend Helpers
// =============================================================================

BackendMapHints IndexSet::getBackendMapHints() const noexcept {
    BackendMapHints hints;
    hints.is_contiguous = isContiguous();
    hints.global_size = global_size_;
    hints.owning_rank = owning_rank_;

    if (hints.is_contiguous && !intervals_.empty()) {
        hints.range_begin = intervals_.front().begin;
        hints.range_end = intervals_.front().end;
    } else if (!empty()) {
        hints.range_begin = minIndex();
        hints.range_end = maxIndex() + 1;
    }

    return hints;
}

// =============================================================================
// IndexSet Comparison
// =============================================================================

bool IndexSet::operator==(const IndexSet& other) const {
    if (size_ != other.size_) return false;

    // Compare index by index
    auto it1 = begin();
    auto it2 = other.begin();
    auto end1 = end();

    while (it1 != end1) {
        if (*it1 != *it2) return false;
        ++it1;
        ++it2;
    }

    return true;
}

// =============================================================================
// IndexSet Private Helpers
// =============================================================================

void IndexSet::compressToIntervals() {
    if (explicit_indices_.empty()) return;

    // Try to find contiguous runs
    std::vector<IndexInterval> intervals;
    GlobalIndex run_start = explicit_indices_.front();
    GlobalIndex run_end = run_start + 1;

    for (std::size_t i = 1; i < explicit_indices_.size(); ++i) {
        if (explicit_indices_[i] == run_end) {
            // Extend current run
            ++run_end;
        } else {
            // End current run, start new one
            intervals.push_back({run_start, run_end});
            run_start = explicit_indices_[i];
            run_end = run_start + 1;
        }
    }
    // Don't forget last run
    intervals.push_back({run_start, run_end});

    // Decide which storage is more efficient
    // Intervals use 2 * sizeof(GlobalIndex) each
    // Explicit uses 1 * sizeof(GlobalIndex) per index
    std::size_t interval_cost = intervals.size() * 2 * sizeof(GlobalIndex);
    std::size_t explicit_cost = explicit_indices_.size() * sizeof(GlobalIndex);

    // Use interval storage if it saves memory or results in few intervals
    if (interval_cost < explicit_cost || intervals.size() <= 3) {
        intervals_ = std::move(intervals);
        explicit_indices_.clear();
        explicit_indices_.shrink_to_fit();
        use_explicit_storage_ = false;
    } else {
        use_explicit_storage_ = true;
    }
}

void IndexSet::mergeIntervals() {
    if (intervals_.size() <= 1) return;

    // Sort by begin
    std::sort(intervals_.begin(), intervals_.end(),
        [](const IndexInterval& a, const IndexInterval& b) {
            return a.begin < b.begin;
        });

    // Merge overlapping or adjacent intervals
    std::vector<IndexInterval> merged;
    merged.push_back(intervals_.front());

    for (std::size_t i = 1; i < intervals_.size(); ++i) {
        auto& last = merged.back();
        const auto& curr = intervals_[i];

        if (curr.begin <= last.end) {
            // Overlapping or adjacent - merge
            last.end = std::max(last.end, curr.end);
        } else {
            // Disjoint - add new interval
            merged.push_back(curr);
        }
    }

    intervals_ = std::move(merged);
}

// =============================================================================
// DofPartition Implementation
// =============================================================================

DofPartition::DofPartition(IndexSet owned, IndexSet ghost)
    : owned_(std::move(owned))
    , ghost_(std::move(ghost))
    , relevant_(owned_.unionWith(ghost_))
{
}

DofPartition::DofPartition(GlobalIndex owned_begin, GlobalIndex owned_end,
                           std::span<const GlobalIndex> ghost_indices)
    : owned_(owned_begin, owned_end)
    , ghost_(ghost_indices)
    , relevant_(owned_.unionWith(ghost_))
{
}

void DofPartition::setGlobalSize(GlobalIndex size) {
    global_size_ = size;
    // Propagate to contained sets
    owned_.setGlobalSize(size);
    ghost_.setGlobalSize(size);
    relevant_.setGlobalSize(size);
}

} // namespace dofs
} // namespace FE
} // namespace svmp
