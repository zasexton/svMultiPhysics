#ifndef SVMP_FE_SYSTEMS_AUXILIARY_STATE_MANAGER_H
#define SVMP_FE_SYSTEMS_AUXILIARY_STATE_MANAGER_H

/**
 * @file AuxiliaryStateManager.h
 * @brief Central manager for auxiliary state blocks within an FESystem.
 *
 * `AuxiliaryStateManager` owns the generalized `AuxiliaryState` container,
 * provides distributed ownership and synchronization semantics, and exposes
 * pack/unpack APIs for checkpoint/restart without coupling to a specific
 * I/O format.
 *
 * ## Distributed ownership rules (MPI)
 *
 * Each auxiliary scope has well-defined ownership and ghost semantics:
 *
 * | Scope             | Ownership rule                                       |
 * |-------------------|------------------------------------------------------|
 * | `Global`          | Replicated on every rank.  No ghosts.                |
 * | `Node`            | Owned by the rank that owns the mesh vertex.         |
 * |                   | Ghosts appended after owned in canonical order.      |
 * | `Cell`            | Owned by the rank that owns the mesh cell.           |
 * |                   | No ghosts by default (cell data is rank-local).      |
 * | `QuadraturePoint` | Inherits cell ownership.  No ghosts.                 |
 * | `Boundary`        | Single instance on a named boundary, no distribution |
 * | `Facet`           | Owned by the rank that owns the underlying entity    |
 * |                   | (face/edge).  Stable within mesh topology.           |
 *
 * ## Synchronization
 *
 * - `None`: no MPI communication (Global, serial runs).
 * - `OwnedOnly`: owned entities are authoritative; no ghost exchange.
 * - `OwnedAndGhost`: owned values are scattered to ghost layers after
 *   each commit or explicit sync call.
 *
 * The formulation selects the sync policy per block at registration time
 * via `AuxiliaryStateSpec::sync_policy`.
 *
 * ## Monolithic auxiliary unknowns
 *
 * Monolithic auxiliary blocks use auxiliary-specific unknown layouts
 * rather than reusing FE field DOF maps.  Ownership follows the same
 * scope rules; the manager contributes layout metadata so `FESystem`
 * can compose field + auxiliary unknowns into one mixed system.
 *
 * ## Checkpoint / restart
 *
 * `packBlock()` / `unpackBlock()` serialize block data to/from flat
 * byte buffers without assuming a particular I/O format.  The caller
 * is responsible for actual file I/O.
 *
 * ## Mesh adaptation / repartitioning
 *
 * `TransferHook` callbacks allow formulations to define custom remap
 * logic for scope-specific data during remeshing or repartitioning.
 * The manager invokes these hooks and handles the boilerplate of
 * resizing storage and reindexing entities.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Systems/AuxiliaryStateTypes.h"
#include "Systems/AuxiliaryState.h"
#include "Systems/AuxiliaryStateIndexing.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Central manager for auxiliary state blocks within an FESystem.
 *
 * Owns the block-based `AuxiliaryState` container and provides:
 * - Block registration with scope-specific indexing
 * - Distributed ownership and ghost synchronization
 * - Pack/unpack for checkpoint/restart
 * - Transfer hooks for mesh adaptation
 * - Debug validation
 */
class AuxiliaryStateManager {
public:
    // -----------------------------------------------------------------
    //  Transfer / remap hooks
    // -----------------------------------------------------------------

    /**
     * @brief Callback invoked during mesh transfer for a block.
     *
     * Arguments: (old_data, old_entity_count, new_entity_count, output)
     * The hook writes remapped data into `output`.
     */
    using TransferHook = std::function<void(
        std::span<const Real> old_data,
        std::size_t old_entity_count,
        std::size_t new_entity_count,
        std::span<Real> output)>;

    /**
     * @brief Callback for ghost synchronization (scatter owned → ghost).
     *
     * Arguments: (block_name, work_buffer)
     * The hook performs MPI scatter of owned data to ghost positions.
     */
    using GhostSyncHook = std::function<void(
        std::string_view block_name,
        std::span<Real> work_buffer)>;

    AuxiliaryStateManager() = default;

    // -----------------------------------------------------------------
    //  Block registration
    // -----------------------------------------------------------------

    /**
     * @brief Register a new auxiliary block with scope-specific indexing.
     *
     * @param spec         Block specification.
     * @param entity_count Number of entities (1 for Global, mesh-derived for others).
     * @param initial_values Optional initial values.
     * @return Block index in the underlying AuxiliaryState.
     */
    std::size_t registerBlock(const AuxiliaryStateSpec& spec,
                              std::size_t entity_count,
                              std::span<const Real> initial_values = {});

    /**
     * @brief Register a QuadraturePoint-scoped block with explicit per-cell QP offsets.
     */
    std::size_t registerBlockWithQPOffsets(const AuxiliaryStateSpec& spec,
                                           std::span<const std::size_t> qp_offsets,
                                           std::span<const Real> initial_values = {});

    /**
     * @brief Register a block with ragged layout.
     */
    std::size_t registerBlockRagged(const AuxiliaryStateSpec& spec,
                                    std::span<const std::size_t> offsets,
                                    std::span<const Real> initial_values = {});

    // -----------------------------------------------------------------
    //  Access
    // -----------------------------------------------------------------

    /// The underlying auxiliary state container.
    [[nodiscard]] AuxiliaryState& state() noexcept { return state_; }
    [[nodiscard]] const AuxiliaryState& state() const noexcept { return state_; }

    /// Number of registered blocks.
    [[nodiscard]] std::size_t blockCount() const noexcept { return state_.blockCount(); }

    /// Whether a block with the given name exists.
    [[nodiscard]] bool hasBlock(std::string_view name) const noexcept
    {
        return state_.hasBlock(name);
    }

    /// Get block storage by name.
    [[nodiscard]] AuxiliaryBlockStorage& getBlock(std::string_view name)
    {
        return state_.getBlock(name);
    }
    [[nodiscard]] const AuxiliaryBlockStorage& getBlock(std::string_view name) const
    {
        return state_.getBlock(name);
    }

    /// Get indexing descriptor for a block.
    [[nodiscard]] const AuxiliaryBlockIndexing& getIndexing(std::string_view name) const;

    /// Get the spec used to register a block.
    [[nodiscard]] const AuxiliaryStateSpec& getSpec(std::string_view name) const;

    // -----------------------------------------------------------------
    //  Ghost synchronization
    // -----------------------------------------------------------------

    /**
     * @brief Set a ghost synchronization hook for a block.
     *
     * The hook is called by `syncGhosts()` for blocks with
     * `OwnedAndGhost` sync policy.
     */
    void setGhostSyncHook(std::string_view block_name, GhostSyncHook hook);

    /**
     * @brief Synchronize ghost values for all blocks that need it.
     *
     * Invokes the registered ghost sync hook for each block with
     * `OwnedAndGhost` sync policy.  No-op in serial or when no
     * hooks are registered.
     */
    void syncGhosts();

    /**
     * @brief Synchronize ghost values for a single block.
     */
    void syncGhosts(std::string_view block_name);

    // -----------------------------------------------------------------
    //  Lifecycle
    // -----------------------------------------------------------------

    /**
     * @brief Reset all blocks to committed state.
     */
    void resetAllToCommitted();

    /**
     * @brief Commit all blocks at the given time.
     */
    void commitAll(Real time);

    /**
     * @brief Rollback all blocks.
     */
    void rollbackAll();

    /**
     * @brief Clear all blocks and registrations.
     */
    void clear();

    /**
     * @brief Called when FESystem::invalidateSetup() runs.
     *
     * Clears setup-time state (indexing, sync hooks) but preserves
     * block definitions and data.
     */
    void invalidateSetup();

    // -----------------------------------------------------------------
    //  Checkpoint / restart (pack / unpack)
    // -----------------------------------------------------------------

    /**
     * @brief Pack a block's work and committed data into a byte buffer.
     *
     * Format: [committed_data | work_data] as contiguous Real arrays.
     * History is NOT packed (it can be rebuilt from checkpoint sequence).
     *
     * @param block_name Block to pack.
     * @return Packed data (2 × storage_size Reals).
     */
    [[nodiscard]] std::vector<Real> packBlock(std::string_view block_name) const;

    /**
     * @brief Unpack a block's data from a byte buffer.
     *
     * @param block_name Block to unpack into.
     * @param packed_data Must have 2 × storage_size elements.
     */
    void unpackBlock(std::string_view block_name, std::span<const Real> packed_data);

    /**
     * @brief Pack all blocks into a single buffer.
     *
     * Format: for each block in registration order,
     * [n_reals (uint64) | committed | work].
     *
     * @return Packed data.
     */
    [[nodiscard]] std::vector<Real> packAll() const;

    /**
     * @brief Unpack all blocks from a single buffer.
     */
    void unpackAll(std::span<const Real> packed_data);

    // -----------------------------------------------------------------
    //  Mesh adaptation / transfer
    // -----------------------------------------------------------------

    /**
     * @brief Register a transfer hook for a block.
     *
     * Called during `transferBlock()` to remap data when entity counts change.
     */
    void setTransferHook(std::string_view block_name, TransferHook hook);

    /**
     * @brief Transfer (remap) a block to a new entity count.
     *
     * If a transfer hook is registered, it is invoked.
     * Otherwise, the block is resized and:
     * - For fixed-stride: existing data preserved up to min(old, new).
     * - For ragged: the caller must provide new offsets.
     *
     * @param block_name     Block to transfer.
     * @param new_entity_count New entity count.
     */
    void transferBlock(std::string_view block_name, std::size_t new_entity_count);

    /**
     * @brief Called when quadrature-point or boundary-entity data
     *        cannot be meaningfully remapped during remeshing.
     *
     * Re-initializes the block to zero with the new entity count.
     * A diagnostic message is emitted.
     */
    void reinitializeBlock(std::string_view block_name, std::size_t new_entity_count);

    // -----------------------------------------------------------------
    //  Validation
    // -----------------------------------------------------------------

    /**
     * @brief Validate that all block storage sizes and indexing are
     *        internally consistent.
     *
     * Checks:
     * - Storage size matches indexing total storage size.
     * - Owned entity count is consistent.
     * - Committed and work buffers have the same size.
     *
     * @throws InvalidStateException if any inconsistency is found.
     */
    void validate() const;

    // -----------------------------------------------------------------
    //  Summary
    // -----------------------------------------------------------------

    /// Aggregate storage summary across all blocks.
    [[nodiscard]] AuxiliaryStateStorageSummary storageSummary() const noexcept
    {
        return state_.storageSummary();
    }

private:
    AuxiliaryState state_{};

    /// Per-block metadata (indexed parallel to state_.blocks_)
    struct BlockMeta {
        AuxiliaryStateSpec spec{};
        AuxiliaryBlockIndexing indexing{};
        TransferHook transfer_hook{};
        GhostSyncHook ghost_sync_hook{};
    };

    std::vector<BlockMeta> block_meta_{};
    std::unordered_map<std::string, std::size_t> meta_name_to_index_{};

    [[nodiscard]] std::size_t metaIndex(std::string_view name) const;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_STATE_MANAGER_H
