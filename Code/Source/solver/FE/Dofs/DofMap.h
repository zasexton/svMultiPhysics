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

#ifndef SVMP_FE_DOFS_DOFMAP_H
#define SVMP_FE_DOFS_DOFMAP_H

/**
 * @file DofMap.h
 * @brief Core DOF mapping infrastructure for finite element assembly
 *
 * The DofMap class provides the fundamental mapping between element-local
 * DOF indices and global system DOF indices. It is the backbone of the
 * DOF management system and is designed for:
 *  - O(1) access to element DOF arrays via getCellDofs()
 *  - Thread-safe read access after finalization
 *  - GPU-compatible flat data layouts for matrix-free kernels
 *  - Canonical ordering on shared entities for orientation consistency
 *
 * This class does NOT own mesh topology - it queries connectivity from
 * an external mesh and stores the computed DOF assignments.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include <algorithm>
#include <vector>
#include <span>
#include <cstdint>
#include <atomic>
#include <functional>

namespace svmp {
namespace FE {
namespace dofs {

/**
 * @brief Lifecycle state for DofMap
 *
 * The DofMap follows a build -> finalize -> use lifecycle:
 * - Building: Mutable state, DOFs can be distributed and modified
 * - Finalized: Immutable state, thread-safe and device-copyable
 */
enum class DofMapState : std::uint8_t {
    Building,   ///< Mutable state - can add/modify DOFs
    Finalized   ///< Immutable state - thread-safe, device-ready
};

/**
 * @brief Read-only view of DOF map data suitable for device kernels
 *
 * This struct contains raw pointers to the DOF data arrays in a flat,
 * GPU-copyable format. It is valid only while the parent DofMap exists
 * and remains finalized.
 */
struct DofMapDeviceView {
    const GlobalIndex* cell_dof_offsets{nullptr};  ///< CSR offsets [n_cells+1]
    const GlobalIndex* cell_dofs{nullptr};         ///< Flat DOF array
    GlobalIndex n_cells{0};                        ///< Number of cells
    GlobalIndex n_dofs_total{0};                   ///< Total DOF count
    GlobalIndex n_dofs_local{0};                   ///< Locally owned DOFs
};

/**
 * @brief Core DOF mapping from element-local to global indices
 *
 * The DofMap stores the mapping from cells to their global DOF indices
 * in a CSR-like format for efficient, cache-friendly access during assembly.
 *
 * Key design decisions:
 * - No global_to_local() API (ambiguous - DOF appears in many elements)
 * - getCellDofs() returns span for zero-copy element assembly
 * - Canonical ordering based on global vertex IDs for shared entities
 * - Finalize/immutable lifecycle for thread safety
 *
 * @note This class does not own mesh data. The mesh must remain valid
 *       for the lifetime of the DofMap.
 */
class DofMap {
public:
    // =========================================================================
    // Construction and lifecycle
    // =========================================================================

    /**
     * @brief Default constructor - creates empty map in Building state
     */
    DofMap() = default;

    /**
     * @brief Construct with known capacity
     * @param n_cells Expected number of cells
     * @param n_dofs_total Expected total DOF count
     * @param dofs_per_cell Typical DOFs per cell (for preallocation)
     */
    DofMap(GlobalIndex n_cells, GlobalIndex n_dofs_total, LocalIndex dofs_per_cell = 0);

    /// Move constructor
    DofMap(DofMap&& other) noexcept;

    /// Move assignment
    DofMap& operator=(DofMap&& other) noexcept;

    /// Copy constructor (deep copy, resets to Building state)
    DofMap(const DofMap& other);

    /// Copy assignment (deep copy, resets to Building state)
    DofMap& operator=(const DofMap& other);

    /// Destructor
    ~DofMap() = default;

    // =========================================================================
    // Setup methods (Building state only)
    // =========================================================================

    /**
     * @brief Reserve storage for expected cell count
     * @param n_cells Number of cells to reserve for
     * @param dofs_per_cell Typical DOFs per cell
     * @throws FEException if already finalized
     */
    void reserve(GlobalIndex n_cells, LocalIndex dofs_per_cell = 8);

    /**
     * @brief Set the DOFs for a single cell
     *
     * @param cell_id Local cell index
     * @param dof_ids Global DOF indices for this cell
     * @throws FEException if already finalized or cell_id out of range
     *
     * @note DOFs should be ordered consistently with the element's local
     *       numbering convention (vertex DOFs, edge DOFs, face DOFs, cell DOFs).
     */
    void setCellDofs(GlobalIndex cell_id, std::span<const GlobalIndex> dof_ids);

    /**
     * @brief Set DOFs for multiple cells at once (batch version)
     *
     * @param cell_ids Cell indices
     * @param offsets CSR offsets into dof_ids
     * @param dof_ids Flat array of DOF indices
     * @throws FEException if already finalized
     */
    void setCellDofsBatch(std::span<const GlobalIndex> cell_ids,
                          std::span<const GlobalIndex> offsets,
                          std::span<const GlobalIndex> dof_ids);

    /**
     * @brief Set the total number of DOFs in the system
     * @param n_dofs Total DOF count (global across all ranks)
     * @throws FEException if already finalized
     */
    void setNumDofs(GlobalIndex n_dofs);

    /**
     * @brief Set the number of locally owned DOFs
     * @param n_local Number of DOFs owned by this rank
     * @throws FEException if already finalized
     */
    void setNumLocalDofs(GlobalIndex n_local);

    /**
     * @brief Set DOF ownership information
     * @param owner_func Function mapping global DOF -> owning MPI rank
     * @throws FEException if already finalized
     */
    void setDofOwnership(std::function<int(GlobalIndex)> owner_func);

    /**
     * @brief Set local MPI rank used by isOwnedDof()
     *
     * For mesh-independent workflows, this must be set by the caller (typically DofHandler)
     * when running in parallel.
     */
    void setMyRank(int my_rank) noexcept { my_rank_ = my_rank; }

    /**
     * @brief Transition to immutable state
     *
     * After finalization:
     * - All setup methods will throw
     * - Read methods are thread-safe
     * - Device views can be created
     *
     * @throws FEException if already finalized or if data is inconsistent
     */
    void finalize();

    // =========================================================================
    // Query methods (safe in any state, thread-safe after finalize)
    // =========================================================================

    /**
     * @brief Check if map is finalized (immutable)
     */
    [[nodiscard]] bool isFinalized() const noexcept {
        return state_.load(std::memory_order_acquire) == DofMapState::Finalized;
    }

    /**
     * @brief Get current lifecycle state
     */
    [[nodiscard]] DofMapState state() const noexcept {
        return state_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get global DOFs for a cell (primary assembly interface)
     *
     * @param cell_id Local cell index
     * @return Span of global DOF indices for this cell
     * @throws FEException if cell_id out of range
     *
     * This is the primary method for assembly - returns a view into
     * the internal storage with no copying.
     */
    [[nodiscard]] std::span<const GlobalIndex> getCellDofs(GlobalIndex cell_id) const;

    /**
     * @brief Get a single DOF for a cell (convenience method)
     *
     * @param cell_id Local cell index
     * @param local_dof Local DOF index within cell
     * @return Global DOF index
     * @throws FEException if indices out of range
     *
     * Prefer getCellDofs() for assembly loops - this is for occasional access.
     */
    [[nodiscard]] GlobalIndex localToGlobal(GlobalIndex cell_id, LocalIndex local_dof) const;

    /**
     * @brief Get number of DOFs for a cell
     * @param cell_id Local cell index
     * @return Number of DOFs on this cell
     */
    [[nodiscard]] LocalIndex getNumCellDofs(GlobalIndex cell_id) const;

    /**
     * @brief Get total number of DOFs in the system
     */
    [[nodiscard]] GlobalIndex getNumDofs() const noexcept { return n_dofs_total_; }

    /**
     * @brief Get number of locally owned DOFs
     */
    [[nodiscard]] GlobalIndex getNumLocalDofs() const noexcept { return n_dofs_local_; }

    /**
     * @brief Get number of cells in the map
     */
    [[nodiscard]] GlobalIndex getNumCells() const noexcept { return n_cells_; }

    /**
     * @brief Get maximum DOFs per cell (for preallocation)
     *
     * Returns the maximum number of DOFs associated with any single cell in
     * this map, computed from the CSR offsets. This is intended for sizing
     * temporary assembly buffers.
     */
    [[nodiscard]] LocalIndex getMaxDofsPerCell() const noexcept {
        LocalIndex max_dofs = 0;
        if (cell_dof_offsets_.size() < 2) {
            return max_dofs;
        }
        for (std::size_t i = 0; i + 1 < cell_dof_offsets_.size(); ++i) {
            const GlobalIndex span = cell_dof_offsets_[i + 1] - cell_dof_offsets_[i];
            if (span <= 0) {
                continue;
            }
            const auto count = static_cast<std::uint64_t>(span);
            if (count >= static_cast<std::uint64_t>(std::numeric_limits<LocalIndex>::max())) {
                return std::numeric_limits<LocalIndex>::max();
            }
            max_dofs = std::max(max_dofs, static_cast<LocalIndex>(count));
        }
        return max_dofs;
    }

    /**
     * @brief Get the MPI rank that owns a DOF
     *
     * @param global_dof Global DOF index
     * @return Owning MPI rank, or -1 if unknown
     */
    [[nodiscard]] int getDofOwner(GlobalIndex global_dof) const;

    /**
     * @brief Check if a DOF is locally owned
     * @param global_dof Global DOF index
     * @return true if owned by this rank
     */
    [[nodiscard]] bool isOwnedDof(GlobalIndex global_dof) const;

    // =========================================================================
    // Device/GPU support
    // =========================================================================

    /**
     * @brief Get a device-compatible view of the DOF data
     *
     * The returned view contains raw pointers valid only while:
     * - This DofMap exists
     * - This DofMap remains finalized
     *
     * @return Device view struct with raw data pointers
     * @throws FEException if not finalized
     */
    [[nodiscard]] DofMapDeviceView getDeviceView() const;

    // =========================================================================
    // Raw data access (for advanced use/serialization)
    // =========================================================================

    /**
     * @brief Get CSR offsets array
     * @return Span of offsets [n_cells+1]
     */
    [[nodiscard]] std::span<const GlobalIndex> getOffsets() const noexcept {
        return {cell_dof_offsets_.data(), cell_dof_offsets_.size()};
    }

    /**
     * @brief Get flat DOF indices array
     * @return Span of all DOF indices
     */
    [[nodiscard]] std::span<const GlobalIndex> getDofIndices() const noexcept {
        return {cell_dofs_.data(), cell_dofs_.size()};
    }

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate internal consistency
     *
     * Checks:
     * - Offsets are monotonically increasing
     * - All DOF indices are in valid range [0, n_dofs_total)
     * - No negative DOF indices (unless explicitly allowed)
     *
     * @return true if valid, false otherwise
     */
    [[nodiscard]] bool validate() const noexcept;

    /**
     * @brief Get detailed validation error message
     * @return Error message, or empty string if valid
     */
    [[nodiscard]] std::string validationError() const;

private:
    // Internal helpers
    void checkNotFinalized() const;
    void checkFinalized() const;
    void checkCellId(GlobalIndex cell_id) const;

    // CSR storage for cell -> DOF mapping
    std::vector<GlobalIndex> cell_dof_offsets_;  ///< [n_cells+1] offsets
    std::vector<GlobalIndex> cell_dofs_;          ///< Flat DOF array

    // Metadata
    GlobalIndex n_cells_{0};
    GlobalIndex n_dofs_total_{0};
    GlobalIndex n_dofs_local_{0};

    // Ownership function (for parallel)
    std::function<int(GlobalIndex)> owner_func_;
    int my_rank_{0};

    // State management
    std::atomic<DofMapState> state_{DofMapState::Building};
};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_DOFMAP_H
