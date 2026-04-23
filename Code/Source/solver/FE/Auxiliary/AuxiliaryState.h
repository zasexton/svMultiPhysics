#ifndef SVMP_FE_AUXILIARY_STATE_H
#define SVMP_FE_AUXILIARY_STATE_H

/**
 * @file AuxiliaryState.h
 * @brief Generalized auxiliary state container for the FE library
 *
 * This module provides the core runtime container for auxiliary (non-PDE)
 * state variables managed by the FE library.  Auxiliary state is
 * FE-library infrastructure — not a boundary-condition feature and not a
 * physics-specific concept.  Boundary functionals, EP-like ionic models,
 * metabolism models, reduced models, and future coupled subsystems all
 * use the same neutral AuxiliaryState infrastructure.
 *
 * ## Public API surface
 *
 * - `AuxiliaryStateTypes.h`: Core enums and specification structs.
 * - `AuxiliaryStateStorage.h`: Per-block storage with committed/work/history.
 * - `AuxiliaryHistoryBuffer.h`: Time-stamped history snapshots.
 * - `AuxiliaryStateIndexing.h`: Scope-specific entity indexing.
 * - `AuxiliaryState`: Mutable runtime container with a block-based API
 *   supporting multiple blocks with distinct scopes.
 *
 * The block-based `AuxiliaryStateManager` path is the authoritative API.
 *
 * See `AuxiliaryStateTypes.h` for the full vocabulary of scopes, solve
 * modes, derivative policies, and layout options.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Auxiliary/AuxiliaryStateStorage.h"

#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Mutable runtime container for named auxiliary state variables.
 *
 * - `registerBlock()` — register a typed block with a specific scope.
 * - `getBlock()` / `hasBlock()` — block lookup by name.
 * - Each block is an `AuxiliaryBlockStorage` with its own work/committed
 *   buffers, history, and scope-specific indexing.
 * - Blocks may have different scopes in the same `AuxiliaryState`.
 *
 * Block names are the durable public handles.  Numeric indices are
 * internal, setup-stable identifiers.
 */
class AuxiliaryState {
public:
    AuxiliaryState() = default;

    void clear()
    {
        blocks_.clear();
        block_name_to_index_.clear();
    }

    // =================================================================
    //  Block-based API (generalized)
    // =================================================================

    /**
     * @brief Register a new typed auxiliary block.
     *
     * @param spec         Block specification.
     * @param entity_count Number of entities (1 for Global, mesh count for others).
     * @param initial_values Optional initial values (must be entity_count * spec.size).
     *
     * @return Index of the newly registered block.
     */
    std::size_t registerBlock(const AuxiliaryStateSpec& spec,
                              std::size_t entity_count,
                              std::span<const Real> initial_values = {})
    {
        FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                    "AuxiliaryState::registerBlock: empty block name");
        FE_THROW_IF(block_name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                    "AuxiliaryState::registerBlock: duplicate block '" + spec.name + "'");
        FE_THROW_IF(spec.layout_mode != AuxiliaryLayoutMode::FixedStride,
                    NotImplementedException,
                    "AuxiliaryState::registerBlock: ragged layout requires "
                    "registerBlockRagged()");

        AuxiliaryBlockStorage block;
        block.setupFixedStride(spec, entity_count);

        if (!initial_values.empty()) {
            block.initialize(initial_values);
        }

        const auto idx = blocks_.size();
        blocks_.push_back(std::move(block));
        block_name_to_index_.emplace(spec.name, idx);
        return idx;
    }

    /**
     * @brief Register a new block with ragged per-entity layout.
     *
     * @param spec    Block specification.
     * @param offsets Per-entity offsets (size = entity_count + 1).
     * @param initial_values Optional initial values (must be offsets.back()).
     *
     * @return Index of the newly registered block.
     */
    std::size_t registerBlockRagged(const AuxiliaryStateSpec& spec,
                                    std::span<const std::size_t> offsets,
                                    std::span<const Real> initial_values = {})
    {
        FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                    "AuxiliaryState::registerBlockRagged: empty block name");
        FE_THROW_IF(block_name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                    "AuxiliaryState::registerBlockRagged: duplicate block '" + spec.name + "'");
        FE_THROW_IF(spec.layout_mode != AuxiliaryLayoutMode::Ragged,
                    InvalidArgumentException,
                    "AuxiliaryState::registerBlockRagged: spec layout_mode must be Ragged");

        AuxiliaryBlockStorage block;
        block.setupRagged(spec, offsets);

        if (!initial_values.empty()) {
            block.initialize(initial_values);
        }

        const auto idx = blocks_.size();
        blocks_.push_back(std::move(block));
        block_name_to_index_.emplace(spec.name, idx);
        return idx;
    }

    /// Number of registered blocks.
    [[nodiscard]] std::size_t blockCount() const noexcept { return blocks_.size(); }

    /// Whether a block with the given name exists.
    [[nodiscard]] bool hasBlock(std::string_view name) const noexcept
    {
        return block_name_to_index_.find(std::string(name)) !=
               block_name_to_index_.end();
    }

    /// Get block index by name (throws if not found).
    /// Setup-stable internal handle; prefer `getBlock(name)` in user-facing code.
    [[nodiscard]] std::size_t blockIndex(std::string_view name) const
    {
        auto it = block_name_to_index_.find(std::string(name));
        FE_THROW_IF(it == block_name_to_index_.end(), InvalidArgumentException,
                    "AuxiliaryState: unknown block '" + std::string(name) + "'");
        return it->second;
    }

    /// Get block by index.
    /// Setup-stable internal access path; prefer `getBlock(name)` in user-facing code.
    [[nodiscard]] AuxiliaryBlockStorage& block(std::size_t idx)
    {
        FE_THROW_IF(idx >= blocks_.size(), InvalidArgumentException,
                    "AuxiliaryState::block: index out of range");
        return blocks_[idx];
    }
    [[nodiscard]] const AuxiliaryBlockStorage& block(std::size_t idx) const
    {
        FE_THROW_IF(idx >= blocks_.size(), InvalidArgumentException,
                    "AuxiliaryState::block: index out of range");
        return blocks_[idx];
    }

    /// Get block by name.
    /// This is the preferred public access path for auxiliary blocks.
    [[nodiscard]] AuxiliaryBlockStorage& getBlock(std::string_view name)
    {
        return blocks_[blockIndex(name)];
    }
    [[nodiscard]] const AuxiliaryBlockStorage& getBlock(std::string_view name) const
    {
        return blocks_[blockIndex(name)];
    }

    /// Get all block names (in registration order).
    [[nodiscard]] std::vector<std::string> blockNames() const
    {
        std::vector<std::string> names;
        names.reserve(blocks_.size());
        for (const auto& b : blocks_) {
            names.push_back(b.name());
        }
        return names;
    }

    /**
     * @brief Reset all blocks to their committed state.
     */
    void resetAllBlocks()
    {
        for (auto& b : blocks_) {
            b.resetToCommitted();
        }
    }

    /**
     * @brief Commit all blocks at the given time.
     */
    void commitAllBlocks(Real time)
    {
        for (auto& b : blocks_) {
            b.commitTimeStep(time);
        }
    }

    /**
     * @brief Rollback all blocks to their committed state.
     */
    void rollbackAllBlocks()
    {
        for (auto& b : blocks_) {
            b.rollback();
        }
    }

    /**
     * @brief Get a storage summary across all blocks.
     */
    [[nodiscard]] AuxiliaryStateStorageSummary storageSummary() const noexcept
    {
        AuxiliaryStateStorageSummary summary;
        summary.block_count = blocks_.size();
        for (const auto& b : blocks_) {
            summary.total_work_storage += b.storageSize();
            summary.total_committed_storage += b.storageSize();
            summary.total_history_storage += b.history().totalHistoryStorage();
        }
        return summary;
    }

private:
    std::vector<AuxiliaryBlockStorage> blocks_{};
    std::unordered_map<std::string, std::size_t> block_name_to_index_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_STATE_H
