#ifndef SVMP_FE_AUXILIARY_STATE_STORAGE_H
#define SVMP_FE_AUXILIARY_STATE_STORAGE_H

/**
 * @file AuxiliaryStateStorage.h
 * @brief Block-level storage backend for auxiliary state variables.
 *
 * Each `AuxiliaryBlockStorage` instance manages the contiguous memory for
 * one registered auxiliary block.  It provides three storage tiers:
 *
 * - **work** — the current mutable state used during assembly and
 *   nonlinear iterations.
 * - **committed** — the last committed time-step state.
 * - **history** — older committed snapshots managed by AuxiliaryHistoryBuffer.
 *
 * Both fixed-stride and ragged layout modes are supported:
 * - **FixedStride**: `size = entity_count * component_stride`
 * - **Ragged**: `size = entity_offsets.back()` with per-entity offsets
 *
 * Read-only and writable views are returned as `std::span` without copies.
 */

#include "Core/Types.h"
#include "Core/Alignment.h"
#include "Core/AlignedAllocator.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Auxiliary/AuxiliaryHistoryBuffer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Block-level storage for one auxiliary state block.
 *
 * Manages contiguous work and committed buffers plus a history buffer
 * for one named auxiliary block.  All data is SIMD-aligned.
 *
 * ## Layout modes
 *
 * - **FixedStride**: Every entity has the same number of components
 *   (`component_stride`).  Total size = `entity_count * component_stride`.
 *   Entity `i`, component `c` is at index `i * stride + c` (ByEntityThenComponent)
 *   or `c * entity_count + i` (ByComponentThenEntity).
 *
 * - **Ragged**: Each entity may have a different number of components.
 *   The caller provides per-entity offsets during setup.  Entity `i`
 *   data spans `[offsets[i], offsets[i+1])`.
 */
class AuxiliaryBlockStorage {
public:
    using AlignedVec = std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>>;

    AuxiliaryBlockStorage() = default;

    // -----------------------------------------------------------------
    //  Setup
    // -----------------------------------------------------------------

    /**
     * @brief Set up fixed-stride storage for the given entity count.
     *
     * @param spec         Block specification (name, size = component count, etc.)
     * @param entity_count Number of entities in this block on this rank.
     *
     * Both work and committed buffers are allocated and zero-initialized.
     * The history buffer is configured according to `spec.history_mode`
     * and `spec.history_depth`.
     */
    void setupFixedStride(const AuxiliaryStateSpec& spec, std::size_t entity_count);

    /**
     * @brief Set up ragged storage with per-entity offsets.
     *
     * @param spec    Block specification.
     * @param offsets Offsets array of size `entity_count + 1`.
     *               Entity `i` has `offsets[i+1] - offsets[i]` components.
     *               `offsets[0]` must be 0.
     */
    void setupRagged(const AuxiliaryStateSpec& spec, std::span<const std::size_t> offsets);

    /**
     * @brief Resize for a new entity count (fixed-stride only).
     *
     * Preserves existing data up to `min(old_size, new_size)`.
     * New entries are zero-initialized.
     */
    void resize(std::size_t new_entity_count);

    /**
     * @brief Override the owned-entity prefix count for distributed layouts.
     *
     * Storage keeps owned entities first and ghosts afterward.  Serial blocks
     * leave this equal to `entityCount()`.  Distributed node blocks can lower
     * it so `blockLayout()` reports the real owned/ghost split.
     */
    void setOwnedEntityCount(std::size_t owned_entity_count);

    // -----------------------------------------------------------------
    //  Properties
    // -----------------------------------------------------------------

    [[nodiscard]] bool isSetup() const noexcept { return is_setup_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] AuxiliaryStateScope scope() const noexcept { return scope_; }
    [[nodiscard]] AuxiliaryLayoutMode layoutMode() const noexcept { return layout_mode_; }
    [[nodiscard]] AuxiliaryEntityOrdering ordering() const noexcept { return ordering_; }
    [[nodiscard]] int componentStride() const noexcept { return component_stride_; }
    [[nodiscard]] std::size_t entityCount() const noexcept { return entity_count_; }
    [[nodiscard]] std::size_t ownedEntityCount() const noexcept { return owned_entity_count_; }
    [[nodiscard]] std::size_t storageSize() const noexcept { return storage_size_; }

    /// Entity offsets for ragged layout (size = entity_count + 1).
    [[nodiscard]] std::span<const std::size_t> entityOffsets() const noexcept
    {
        return entity_offsets_;
    }

    // -----------------------------------------------------------------
    //  Work buffer access
    // -----------------------------------------------------------------

    /// Full writable view of the work buffer.
    [[nodiscard]] std::span<Real> work() noexcept { return work_; }

    /// Full read-only view of the work buffer.
    [[nodiscard]] std::span<const Real> work() const noexcept { return work_; }

    /// View of entity `i` in the work buffer when the entity is contiguous.
    [[nodiscard]] std::span<Real> workEntity(std::size_t entity_idx)
    {
        validateContiguousEntityView(entity_idx, "AuxiliaryBlockStorage::workEntity");
        const auto off = entityOffset(entity_idx);
        const auto len = entityLength(entity_idx);
        return {work_.data() + off, len};
    }

    /// Read-only view of entity `i` in the work buffer when the entity is contiguous.
    [[nodiscard]] std::span<const Real> workEntity(std::size_t entity_idx) const
    {
        validateContiguousEntityView(entity_idx, "AuxiliaryBlockStorage::workEntity");
        const auto off = entityOffset(entity_idx);
        const auto len = entityLength(entity_idx);
        return {work_.data() + off, len};
    }

    // -----------------------------------------------------------------
    //  Committed buffer access
    // -----------------------------------------------------------------

    /// Full read-only view of the committed buffer.
    [[nodiscard]] std::span<const Real> committed() const noexcept { return committed_; }

    /// Writable committed buffer access for manager-level synchronization and restore paths.
    [[nodiscard]] std::span<Real> committedMutable() noexcept { return committed_; }

    /// View of entity `i` in the committed buffer when the entity is contiguous.
    [[nodiscard]] std::span<const Real> committedEntity(std::size_t entity_idx) const
    {
        validateContiguousEntityView(entity_idx, "AuxiliaryBlockStorage::committedEntity");
        const auto off = entityOffset(entity_idx);
        const auto len = entityLength(entity_idx);
        return {committed_.data() + off, len};
    }

    // -----------------------------------------------------------------
    //  History access (delegated to AuxiliaryHistoryBuffer)
    // -----------------------------------------------------------------

    /// Access the history buffer.
    [[nodiscard]] const AuxiliaryHistoryBuffer& history() const noexcept { return history_; }
    [[nodiscard]] AuxiliaryHistoryBuffer& history() noexcept { return history_; }

    // -----------------------------------------------------------------
    //  Layout-aware entity gather/scatter
    //
    //  These work correctly for ALL layout modes:
    //  - FixedStride + ByEntityThenComponent (contiguous, fast)
    //  - FixedStride + ByComponentThenEntity (strided gather)
    //  - Ragged (offset-based gather)
    // -----------------------------------------------------------------

    /// Gather entity `i` work values into a contiguous vector.
    [[nodiscard]] std::vector<Real> gatherEntityWork(std::size_t entity_idx) const;

    /// Gather entity `i` committed values into a contiguous vector.
    [[nodiscard]] std::vector<Real> gatherEntityCommitted(std::size_t entity_idx) const;

    /// Gather entity `i` from a history snapshot into a contiguous vector.
    [[nodiscard]] std::vector<Real> gatherEntityHistory(
        std::size_t snapshot_idx, std::size_t entity_idx) const;

    /// Scatter contiguous values back into entity `i` of the work buffer.
    void scatterEntityWork(std::size_t entity_idx, std::span<const Real> values);

    // -----------------------------------------------------------------
    //  Lifecycle operations
    // -----------------------------------------------------------------

    /**
     * @brief Reset work buffer to committed state.
     *
     * Does not modify committed or history buffers.
     */
    void resetToCommitted();

    /**
     * @brief Reset work buffer to committed state and optionally refresh ghosts.
     *
     * For ghosted layouts, only the owned prefix is restored authoritatively.
     * If `sync_ghosts` is provided, it is called afterward to repopulate ghost entries.
     */
    void resetToCommitted(const std::function<void(std::span<Real>)>& sync_ghosts);

    /**
     * @brief Commit work buffer as new state at the given time.
     *
     * Pushes the current committed buffer into history (with timestamp),
     * then copies work → committed.
     */
    void commitTimeStep(Real time);

    /**
     * @brief Commit work buffer as new state at the given time and optionally refresh ghosts.
     *
     * For ghosted layouts, only the owned prefix is authoritative. When `sync_ghosts`
     * is provided, ghost entries in committed/history snapshots are refreshed from the
     * owned prefix after the local copy.
     */
    void commitTimeStep(Real time, const std::function<void(std::span<Real>)>& sync_ghosts);

    /**
     * @brief Rollback: restore work buffer from committed state.
     *
     * Equivalent to resetToCommitted(); included for semantic clarity
     * in rollback workflows.
     */
    void rollback();

    /**
     * @brief Rollback with optional ghost refresh.
     */
    void rollback(const std::function<void(std::span<Real>)>& sync_ghosts);

    /**
     * @brief Initialize both work and committed buffers with the given values.
     *
     * @param values Must have exactly `storageSize()` elements.
     */
    void initialize(std::span<const Real> values);

    /**
     * @brief Clear all storage and reset to uninitialized state.
     */
    void clear();

    // -----------------------------------------------------------------
    //  Summary
    // -----------------------------------------------------------------

    [[nodiscard]] AuxiliaryStateBlockLayout blockLayout() const noexcept;

private:
    void validateContiguousEntityView(std::size_t entity_idx, const char* caller) const
    {
        FE_THROW_IF(!is_setup_, InvalidStateException,
                    std::string(caller) + ": storage is not set up");
        FE_THROW_IF(entity_idx >= entity_count_, InvalidArgumentException,
                    std::string(caller) + ": entity index " +
                        std::to_string(entity_idx) + " exceeds entity count " +
                        std::to_string(entity_count_));
        FE_THROW_IF(layout_mode_ == AuxiliaryLayoutMode::FixedStride &&
                        ordering_ == AuxiliaryEntityOrdering::ByComponentThenEntity,
                    InvalidStateException,
                    std::string(caller) +
                        ": entity data is strided for ByComponentThenEntity; "
                        "use gatherEntityWork()/gatherEntityCommitted() instead");
    }

    [[nodiscard]] std::size_t entityOffset(std::size_t entity_idx) const noexcept
    {
        if (layout_mode_ == AuxiliaryLayoutMode::Ragged) {
            return entity_offsets_[entity_idx];
        }
        if (ordering_ == AuxiliaryEntityOrdering::ByEntityThenComponent) {
            return entity_idx * static_cast<std::size_t>(component_stride_);
        }
        // ByComponentThenEntity: caller uses component * entity_count + entity_idx
        // For entity view, we return offset of first component
        return entity_idx;
    }

    [[nodiscard]] std::size_t entityLength(std::size_t entity_idx) const noexcept
    {
        if (layout_mode_ == AuxiliaryLayoutMode::Ragged) {
            return entity_offsets_[entity_idx + 1] - entity_offsets_[entity_idx];
        }
        return static_cast<std::size_t>(component_stride_);
    }

    bool is_setup_{false};
    std::string name_{};
    AuxiliaryStateScope scope_{AuxiliaryStateScope::Global};
    AuxiliaryLayoutMode layout_mode_{AuxiliaryLayoutMode::FixedStride};
    AuxiliaryEntityOrdering ordering_{AuxiliaryEntityOrdering::ByEntityThenComponent};
    int component_stride_{0};
    std::size_t entity_count_{0};
    std::size_t owned_entity_count_{0};
    std::size_t storage_size_{0};

    /// Per-entity offsets for ragged layout (size = entity_count_ + 1).
    std::vector<std::size_t> entity_offsets_{};

    AlignedVec work_{};
    AlignedVec committed_{};
    AuxiliaryHistoryBuffer history_{};

    /// Internal block id assigned during registration (setup-stable).
    std::uint32_t block_id_{0};

    friend class AuxiliaryBlockStorageRegistry;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_STATE_STORAGE_H
