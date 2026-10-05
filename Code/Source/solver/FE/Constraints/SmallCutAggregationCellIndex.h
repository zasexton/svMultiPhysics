/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_SMALLCUTAGGREGATIONCELLINDEX_H
#define SVMP_FE_CONSTRAINTS_SMALLCUTAGGREGATIONCELLINDEX_H

/**
 * @file SmallCutAggregationCellIndex.h
 * @brief Sorted cell-key tables used by one small-cut aggregation refresh
 *
 * SmallCutAggregationConstraint identifies cells across ranks by their sorted
 * system-global field-DOF support (the cell key). These tables replace
 * node-based std::map containers keyed by cell keys: they iterate in the same
 * ascending key order and answer the same lookups, without a tree node and a
 * key copy per cell. Both are built and discarded within one apply() call.
 */

#include "Core/Types.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace detail {

using SmallCutAggregationCellKey = std::vector<GlobalIndex>;

/**
 * Rank-local cells keyed by their cell key. Entries are appended in any
 * order; finalize() sorts them by key and reports whether two cells share a
 * key. After finalize(), iteration is in ascending key order and find()
 * returns the entry of a key or end().
 */
class SmallCutAggregationLocalCellTable {
public:
    using Key = SmallCutAggregationCellKey;
    using Entry = std::pair<Key, GlobalIndex>;
    using const_iterator = std::vector<Entry>::const_iterator;

    void reserve(std::size_t count) { entries_.reserve(count); }

    void append(Key key, GlobalIndex cell)
    {
        entries_.emplace_back(std::move(key), cell);
    }

    /// Sorts the appended entries; false when two cells share one key.
    [[nodiscard]] bool finalize()
    {
        std::sort(entries_.begin(), entries_.end(),
                  [](const Entry& a, const Entry& b) { return a.first < b.first; });
        return std::adjacent_find(entries_.begin(), entries_.end(),
                                  [](const Entry& a, const Entry& b) {
                                      return a.first == b.first;
                                  }) == entries_.end();
    }

    [[nodiscard]] const_iterator begin() const noexcept { return entries_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return entries_.end(); }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    [[nodiscard]] const_iterator find(const Key& key) const
    {
        const auto it = std::lower_bound(
            entries_.begin(), entries_.end(), key,
            [](const Entry& entry, const Key& k) { return entry.first < k; });
        return it != entries_.end() && it->first == key ? it : entries_.end();
    }

private:
    std::vector<Entry> entries_;
};

/**
 * Communicator-global classified cells in ascending key order. A cell's
 * position is its index into the per-cell arrays of one refresh, so
 * ascending index order is ascending key order.
 */
class SmallCutAggregationCellIndex {
public:
    using Key = SmallCutAggregationCellKey;
    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

    SmallCutAggregationCellIndex() = default;

    /// @param sorted_unique_keys keys in strictly ascending order
    explicit SmallCutAggregationCellIndex(std::vector<Key> sorted_unique_keys)
        : keys_(std::move(sorted_unique_keys)) {}

    [[nodiscard]] std::size_t size() const noexcept { return keys_.size(); }
    [[nodiscard]] bool empty() const noexcept { return keys_.empty(); }
    [[nodiscard]] const Key& key(std::size_t cell) const { return keys_[cell]; }

    /// Index of a key, or npos.
    [[nodiscard]] std::size_t find(const Key& key) const
    {
        const auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
        return it != keys_.end() && *it == key
                   ? static_cast<std::size_t>(it - keys_.begin())
                   : npos;
    }

private:
    std::vector<Key> keys_;
};

} // namespace detail
} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_SMALLCUTAGGREGATIONCELLINDEX_H
