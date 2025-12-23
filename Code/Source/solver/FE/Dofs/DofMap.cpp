/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "DofMap.h"
#include <algorithm>
#include <sstream>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// Construction
// =============================================================================

DofMap::DofMap(GlobalIndex n_cells, GlobalIndex n_dofs_total, LocalIndex dofs_per_cell)
    : n_cells_(n_cells)
    , n_dofs_total_(n_dofs_total)
    , n_dofs_local_(n_dofs_total)  // Default: all local
    , state_(DofMapState::Building)
{
    if (n_cells > 0) {
        cell_dof_offsets_.resize(static_cast<std::size_t>(n_cells) + 1, 0);
        if (dofs_per_cell > 0) {
            cell_dofs_.reserve(static_cast<std::size_t>(n_cells * dofs_per_cell));
        }
    }
}

DofMap::DofMap(DofMap&& other) noexcept
    : cell_dof_offsets_(std::move(other.cell_dof_offsets_))
    , cell_dofs_(std::move(other.cell_dofs_))
    , n_cells_(other.n_cells_)
    , n_dofs_total_(other.n_dofs_total_)
    , n_dofs_local_(other.n_dofs_local_)
    , owner_func_(std::move(other.owner_func_))
    , my_rank_(other.my_rank_)
    , state_(other.state_.load(std::memory_order_acquire))
{
    other.n_cells_ = 0;
    other.n_dofs_total_ = 0;
    other.n_dofs_local_ = 0;
    other.state_.store(DofMapState::Building, std::memory_order_release);
}

DofMap& DofMap::operator=(DofMap&& other) noexcept {
    if (this != &other) {
        cell_dof_offsets_ = std::move(other.cell_dof_offsets_);
        cell_dofs_ = std::move(other.cell_dofs_);
        n_cells_ = other.n_cells_;
        n_dofs_total_ = other.n_dofs_total_;
        n_dofs_local_ = other.n_dofs_local_;
        owner_func_ = std::move(other.owner_func_);
        my_rank_ = other.my_rank_;
        state_.store(other.state_.load(std::memory_order_acquire), std::memory_order_release);

        other.n_cells_ = 0;
        other.n_dofs_total_ = 0;
        other.n_dofs_local_ = 0;
        other.state_.store(DofMapState::Building, std::memory_order_release);
    }
    return *this;
}

DofMap::DofMap(const DofMap& other)
    : cell_dof_offsets_(other.cell_dof_offsets_)
    , cell_dofs_(other.cell_dofs_)
    , n_cells_(other.n_cells_)
    , n_dofs_total_(other.n_dofs_total_)
    , n_dofs_local_(other.n_dofs_local_)
    , owner_func_(other.owner_func_)
    , my_rank_(other.my_rank_)
    , state_(DofMapState::Building)  // Copy always starts in Building state
{
}

DofMap& DofMap::operator=(const DofMap& other) {
    if (this != &other) {
        cell_dof_offsets_ = other.cell_dof_offsets_;
        cell_dofs_ = other.cell_dofs_;
        n_cells_ = other.n_cells_;
        n_dofs_total_ = other.n_dofs_total_;
        n_dofs_local_ = other.n_dofs_local_;
        owner_func_ = other.owner_func_;
        my_rank_ = other.my_rank_;
        state_.store(DofMapState::Building, std::memory_order_release);  // Reset to building
    }
    return *this;
}

// =============================================================================
// Setup methods
// =============================================================================

void DofMap::reserve(GlobalIndex n_cells, LocalIndex dofs_per_cell) {
    checkNotFinalized();

    n_cells_ = n_cells;
    cell_dof_offsets_.resize(static_cast<std::size_t>(n_cells) + 1, 0);
    if (dofs_per_cell > 0) {
        cell_dofs_.reserve(static_cast<std::size_t>(n_cells * dofs_per_cell));
    }
}

void DofMap::setCellDofs(GlobalIndex cell_id, std::span<const GlobalIndex> dof_ids) {
    checkNotFinalized();

    if (cell_id < 0) {
        throw FEException("DofMap::setCellDofs: negative cell_id");
    }

    // Ensure offsets array is large enough
    auto cell_idx = static_cast<std::size_t>(cell_id);
    if (cell_idx >= cell_dof_offsets_.size()) {
        // Need to grow - this is an incremental build case
        cell_dof_offsets_.resize(cell_idx + 2, 0);
        n_cells_ = std::max(n_cells_, cell_id + 1);
    }

    // For incremental builds, we append DOFs and track offsets
    // This assumes cells are set in order or we're rebuilding
    auto current_offset = static_cast<GlobalIndex>(cell_dofs_.size());
    cell_dof_offsets_[cell_idx] = current_offset;

    // Append DOFs
    cell_dofs_.insert(cell_dofs_.end(), dof_ids.begin(), dof_ids.end());

    // Update next offset (for the last cell)
    if (cell_idx + 1 < cell_dof_offsets_.size()) {
        cell_dof_offsets_[cell_idx + 1] = static_cast<GlobalIndex>(cell_dofs_.size());
    }

    n_cells_ = std::max(n_cells_, cell_id + 1);
}

void DofMap::setCellDofsBatch(std::span<const GlobalIndex> cell_ids,
                              std::span<const GlobalIndex> offsets,
                              std::span<const GlobalIndex> dof_ids) {
    checkNotFinalized();

    if (cell_ids.empty()) return;

    // Find max cell ID to size arrays
    GlobalIndex max_cell = *std::max_element(cell_ids.begin(), cell_ids.end());

    if (max_cell >= static_cast<GlobalIndex>(cell_dof_offsets_.size()) - 1) {
        cell_dof_offsets_.resize(static_cast<std::size_t>(max_cell) + 2, 0);
    }

    // Copy data directly (assumes this is a fresh build)
    cell_dofs_ = std::vector<GlobalIndex>(dof_ids.begin(), dof_ids.end());

    // Set offsets for each cell
    for (std::size_t i = 0; i < cell_ids.size(); ++i) {
        auto cell_idx = static_cast<std::size_t>(cell_ids[i]);
        cell_dof_offsets_[cell_idx] = offsets[i];
        if (i + 1 < offsets.size()) {
            cell_dof_offsets_[cell_idx + 1] = offsets[i + 1];
        }
    }

    // Set final offset
    cell_dof_offsets_.back() = static_cast<GlobalIndex>(cell_dofs_.size());

    n_cells_ = max_cell + 1;
}

void DofMap::setNumDofs(GlobalIndex n_dofs) {
    checkNotFinalized();
    n_dofs_total_ = n_dofs;
}

void DofMap::setNumLocalDofs(GlobalIndex n_local) {
    checkNotFinalized();
    n_dofs_local_ = n_local;
}

void DofMap::setDofOwnership(std::function<int(GlobalIndex)> owner_func) {
    checkNotFinalized();
    owner_func_ = std::move(owner_func);
}

void DofMap::finalize() {
    checkNotFinalized();

    // Ensure offsets array is properly sized
    if (n_cells_ > 0 && cell_dof_offsets_.size() != static_cast<std::size_t>(n_cells_) + 1) {
        cell_dof_offsets_.resize(static_cast<std::size_t>(n_cells_) + 1);
    }

    // Ensure final offset is set
    if (!cell_dof_offsets_.empty()) {
        cell_dof_offsets_.back() = static_cast<GlobalIndex>(cell_dofs_.size());
    }

    // Validate before finalizing
    auto error = validationError();
    if (!error.empty()) {
        throw FEException("DofMap::finalize: validation failed - " + error);
    }

    // Transition to finalized state
    state_.store(DofMapState::Finalized, std::memory_order_release);
}

// =============================================================================
// Query methods
// =============================================================================

std::span<const GlobalIndex> DofMap::getCellDofs(GlobalIndex cell_id) const {
    checkCellId(cell_id);

    auto cell_idx = static_cast<std::size_t>(cell_id);
    auto start = static_cast<std::size_t>(cell_dof_offsets_[cell_idx]);
    auto end = static_cast<std::size_t>(cell_dof_offsets_[cell_idx + 1]);

    return {cell_dofs_.data() + start, end - start};
}

GlobalIndex DofMap::localToGlobal(GlobalIndex cell_id, LocalIndex local_dof) const {
    auto dofs = getCellDofs(cell_id);

    if (local_dof >= static_cast<LocalIndex>(dofs.size())) {
        throw FEException("DofMap::localToGlobal: local_dof " +
                          std::to_string(local_dof) + " out of range for cell " +
                          std::to_string(cell_id) + " (has " +
                          std::to_string(dofs.size()) + " DOFs)");
    }

    return dofs[local_dof];
}

LocalIndex DofMap::getNumCellDofs(GlobalIndex cell_id) const {
    checkCellId(cell_id);

    auto cell_idx = static_cast<std::size_t>(cell_id);
    return static_cast<LocalIndex>(
        cell_dof_offsets_[cell_idx + 1] - cell_dof_offsets_[cell_idx]);
}

int DofMap::getDofOwner(GlobalIndex global_dof) const {
    if (owner_func_) {
        return owner_func_(global_dof);
    }
    // Default: all DOFs owned by rank 0
    return 0;
}

bool DofMap::isOwnedDof(GlobalIndex global_dof) const {
    return getDofOwner(global_dof) == my_rank_;
}

// =============================================================================
// Device support
// =============================================================================

DofMapDeviceView DofMap::getDeviceView() const {
    checkFinalized();

    DofMapDeviceView view;
    view.cell_dof_offsets = cell_dof_offsets_.data();
    view.cell_dofs = cell_dofs_.data();
    view.n_cells = n_cells_;
    view.n_dofs_total = n_dofs_total_;
    view.n_dofs_local = n_dofs_local_;

    return view;
}

// =============================================================================
// Validation
// =============================================================================

bool DofMap::validate() const noexcept {
    return validationError().empty();
}

std::string DofMap::validationError() const {
    // Check offsets array size
    if (n_cells_ > 0 && cell_dof_offsets_.size() != static_cast<std::size_t>(n_cells_) + 1) {
        return "Offsets array size mismatch: expected " +
               std::to_string(n_cells_ + 1) + ", got " +
               std::to_string(cell_dof_offsets_.size());
    }

    // CSR must start at 0 when non-empty.
    if (n_cells_ > 0 && !cell_dof_offsets_.empty() && cell_dof_offsets_.front() != 0) {
        return "Offsets[0] must be 0 for non-empty CSR (got " +
               std::to_string(cell_dof_offsets_.front()) + ")";
    }

    // Check offsets are monotonically increasing
    for (std::size_t i = 1; i < cell_dof_offsets_.size(); ++i) {
        if (cell_dof_offsets_[i] < cell_dof_offsets_[i - 1]) {
            return "Offsets not monotonically increasing at index " + std::to_string(i);
        }
    }

    // Ensure each cell has a non-empty DOF range. This catches out-of-order
    // incremental builds that overwrite offsets and silently drop cell DOFs.
    if (n_cells_ > 0 && cell_dof_offsets_.size() >= static_cast<std::size_t>(n_cells_) + 1) {
        for (GlobalIndex c = 0; c < n_cells_; ++c) {
            const auto i = static_cast<std::size_t>(c);
            if (cell_dof_offsets_[i + 1] <= cell_dof_offsets_[i]) {
                return "Cell " + std::to_string(c) + " has empty DOF range (offsets[" +
                       std::to_string(c) + "]=" + std::to_string(cell_dof_offsets_[i]) +
                       ", offsets[" + std::to_string(c + 1) + "]=" +
                       std::to_string(cell_dof_offsets_[i + 1]) + ")";
            }
        }
    }

    // Check final offset matches DOF array size
    if (!cell_dof_offsets_.empty()) {
        auto expected = cell_dof_offsets_.back();
        auto actual = static_cast<GlobalIndex>(cell_dofs_.size());
        if (expected != actual) {
            return "Final offset (" + std::to_string(expected) +
                   ") doesn't match DOF array size (" + std::to_string(actual) + ")";
        }
    }

    // Check DOF indices are valid (non-negative and within range)
    for (std::size_t i = 0; i < cell_dofs_.size(); ++i) {
        if (cell_dofs_[i] < 0) {
            return "Negative DOF index at position " + std::to_string(i);
        }
        if (n_dofs_total_ > 0 && cell_dofs_[i] >= n_dofs_total_) {
            return "DOF index " + std::to_string(cell_dofs_[i]) +
                   " at position " + std::to_string(i) +
                   " exceeds total DOF count " + std::to_string(n_dofs_total_);
        }
    }

    return "";  // No errors
}

// =============================================================================
// Internal helpers
// =============================================================================

void DofMap::checkNotFinalized() const {
    if (isFinalized()) {
        throw FEException("DofMap: operation not allowed after finalization");
    }
}

void DofMap::checkFinalized() const {
    if (!isFinalized()) {
        throw FEException("DofMap: operation requires finalized state");
    }
}

void DofMap::checkCellId(GlobalIndex cell_id) const {
    if (cell_id < 0 || cell_id >= n_cells_) {
        throw FEException("DofMap: cell_id " + std::to_string(cell_id) +
                          " out of range [0, " + std::to_string(n_cells_) + ")");
    }
}

} // namespace dofs
} // namespace FE
} // namespace svmp
