/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "SubspaceView.h"
#include <algorithm>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// Construction
// =============================================================================

SubspaceView::SubspaceView(IndexSet dofs, std::string name, int block_index)
    : dofs_(std::move(dofs))
    , name_(std::move(name))
    , block_index_(block_index) {}

SubspaceView::SubspaceView(GlobalIndex start, GlobalIndex end, std::string name,
                           int block_index)
    : name_(std::move(name))
    , block_index_(block_index) {
    if (start < end) {
        std::vector<GlobalIndex> indices;
        indices.reserve(static_cast<std::size_t>(end - start));
        for (GlobalIndex i = start; i < end; ++i) {
            indices.push_back(i);
        }
        dofs_ = IndexSet(std::move(indices));
    }
}

// =============================================================================
// Range Information
// =============================================================================

bool SubspaceView::isContiguous() const noexcept {
    return dofs_.contiguousRange().has_value();
}

std::optional<std::pair<GlobalIndex, GlobalIndex>>
SubspaceView::contiguousRange() const noexcept {
    auto range = dofs_.contiguousRange();
    if (range) {
        return std::make_pair(range->begin, range->end);
    }
    return std::nullopt;
}

// =============================================================================
// Subvector Operations
// =============================================================================

std::vector<double> SubspaceView::extractSubvector(
    std::span<const double> full_vector) const {

    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(dofs_.size()));

    for (auto dof : dofs_) {
        if (dof >= 0 && static_cast<std::size_t>(dof) < full_vector.size()) {
            result.push_back(full_vector[static_cast<std::size_t>(dof)]);
        } else {
            result.push_back(0.0);
        }
    }

    return result;
}

void SubspaceView::extractSubvector(std::span<const double> full_vector,
                                     std::span<double> output) const {
    if (output.size() < static_cast<std::size_t>(dofs_.size())) {
        throw FEException("SubspaceView::extractSubvector: output buffer too small");
    }

    std::size_t idx = 0;
    for (auto dof : dofs_) {
        if (dof >= 0 && static_cast<std::size_t>(dof) < full_vector.size()) {
            output[idx] = full_vector[static_cast<std::size_t>(dof)];
        } else {
            output[idx] = 0.0;
        }
        ++idx;
    }
}

void SubspaceView::scatterToFull(std::span<const double> subvector,
                                  std::span<double> full_vector) const {
    if (subvector.size() < static_cast<std::size_t>(dofs_.size())) {
        throw FEException("SubspaceView::scatterToFull: subvector too small");
    }

    std::size_t idx = 0;
    for (auto dof : dofs_) {
        if (dof >= 0 && static_cast<std::size_t>(dof) < full_vector.size()) {
            full_vector[static_cast<std::size_t>(dof)] = subvector[idx];
        }
        ++idx;
    }
}

void SubspaceView::scatterToFullAdd(std::span<const double> subvector,
                                     std::span<double> full_vector) const {
    if (subvector.size() < static_cast<std::size_t>(dofs_.size())) {
        throw FEException("SubspaceView::scatterToFullAdd: subvector too small");
    }

    std::size_t idx = 0;
    for (auto dof : dofs_) {
        if (dof >= 0 && static_cast<std::size_t>(dof) < full_vector.size()) {
            full_vector[static_cast<std::size_t>(dof)] += subvector[idx];
        }
        ++idx;
    }
}

// =============================================================================
// Mapping
// =============================================================================

GlobalIndex SubspaceView::localToGlobal(GlobalIndex local_idx) const {
    buildLocalToGlobalIfNeeded();

    if (local_idx < 0 || static_cast<std::size_t>(local_idx) >= local_to_global_.size()) {
        throw FEException("SubspaceView::localToGlobal: index out of range");
    }

    return local_to_global_[static_cast<std::size_t>(local_idx)];
}

GlobalIndex SubspaceView::globalToLocal(GlobalIndex global_dof) const {
    buildLocalToGlobalIfNeeded();

    // Binary search in sorted local_to_global
    auto it = std::lower_bound(local_to_global_.begin(), local_to_global_.end(), global_dof);
    if (it != local_to_global_.end() && *it == global_dof) {
        return static_cast<GlobalIndex>(std::distance(local_to_global_.begin(), it));
    }

    return -1;  // Not found
}

std::vector<GlobalIndex> SubspaceView::buildLocalToGlobalMap() const {
    buildLocalToGlobalIfNeeded();
    return local_to_global_;
}

void SubspaceView::buildLocalToGlobalIfNeeded() const {
    if (local_to_global_built_) return;

    local_to_global_.clear();
    local_to_global_.reserve(static_cast<std::size_t>(dofs_.size()));

    for (auto dof : dofs_) {
        local_to_global_.push_back(dof);
    }

    // IndexSet is already sorted, but ensure for binary search
    std::sort(local_to_global_.begin(), local_to_global_.end());

    local_to_global_built_ = true;
}

// =============================================================================
// Set Operations
// =============================================================================

SubspaceView SubspaceView::intersection_with(const SubspaceView& other) const {
    auto result_set = dofs_.intersectionWith(other.dofs_);
    return SubspaceView(std::move(result_set), name_ + "_intersect_" + other.name_);
}

SubspaceView SubspaceView::union_with(const SubspaceView& other) const {
    auto result_set = dofs_.unionWith(other.dofs_);
    return SubspaceView(std::move(result_set), name_ + "_union_" + other.name_);
}

SubspaceView SubspaceView::difference(const SubspaceView& other) const {
    auto result_set = dofs_.difference(other.dofs_);
    return SubspaceView(std::move(result_set), name_ + "_minus_" + other.name_);
}

SubspaceView SubspaceView::complement(GlobalIndex total_dofs) const {
    std::vector<GlobalIndex> comp_indices;
    comp_indices.reserve(static_cast<std::size_t>(total_dofs - dofs_.size()));

    for (GlobalIndex i = 0; i < total_dofs; ++i) {
        if (!dofs_.contains(i)) {
            comp_indices.push_back(i);
        }
    }

    return SubspaceView(IndexSet(std::move(comp_indices)), name_ + "_complement");
}

} // namespace dofs
} // namespace FE
} // namespace svmp
