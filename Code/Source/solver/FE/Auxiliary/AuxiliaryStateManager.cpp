#include "Auxiliary/AuxiliaryStateManager.h"

#include <algorithm>
#include <cstring>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Internal helpers
// ---------------------------------------------------------------------------

std::size_t AuxiliaryStateManager::metaIndex(std::string_view name) const
{
    auto it = meta_name_to_index_.find(std::string(name));
    FE_THROW_IF(it == meta_name_to_index_.end(), InvalidArgumentException,
                "AuxiliaryStateManager: unknown block '" + std::string(name) + "'");
    return it->second;
}

// ---------------------------------------------------------------------------
//  Block registration
// ---------------------------------------------------------------------------

std::size_t AuxiliaryStateManager::registerBlock(
    const AuxiliaryStateSpec& spec,
    std::size_t entity_count,
    std::span<const Real> initial_values)
{
    return registerBlock(spec, entity_count, entity_count, initial_values);
}

std::size_t AuxiliaryStateManager::registerBlock(
    const AuxiliaryStateSpec& spec,
    std::size_t entity_count,
    std::size_t owned_entity_count,
    std::span<const Real> initial_values)
{
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "AuxiliaryStateManager::registerBlock: empty block name");
    FE_THROW_IF(meta_name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                "AuxiliaryStateManager::registerBlock: duplicate block '" + spec.name + "'");
    FE_THROW_IF(owned_entity_count > entity_count, InvalidArgumentException,
                "AuxiliaryStateManager::registerBlock: owned_entity_count " +
                    std::to_string(owned_entity_count) +
                    " exceeds entity_count " + std::to_string(entity_count));

    // Create indexing descriptor for the scope.
    AuxiliaryBlockIndexing indexing;
    switch (spec.scope) {
        case AuxiliaryStateScope::Global:
            FE_THROW_IF(entity_count != 1u, InvalidArgumentException,
                        "AuxiliaryStateManager::registerBlock: Global scope requires entity_count == 1");
            FE_THROW_IF(owned_entity_count != entity_count, InvalidArgumentException,
                        "AuxiliaryStateManager::registerBlock: Global scope does not support ghosts");
            indexing = AuxiliaryBlockIndexing::createGlobal(spec.size);
            break;
        case AuxiliaryStateScope::Node:
            indexing = AuxiliaryBlockIndexing::createNode(
                owned_entity_count,
                entity_count - owned_entity_count,
                spec.size);
            break;
        case AuxiliaryStateScope::Cell:
            FE_THROW_IF(owned_entity_count != entity_count, InvalidArgumentException,
                        "AuxiliaryStateManager::registerBlock: Cell scope does not support ghosts");
            indexing = AuxiliaryBlockIndexing::createCell(
                entity_count, spec.size);
            break;
        case AuxiliaryStateScope::QuadraturePoint:
            FE_THROW_IF(owned_entity_count != entity_count, InvalidArgumentException,
                        "AuxiliaryStateManager::registerBlock: QuadraturePoint scope does not support ghosts");
            // QP scope with uniform QP count not known here; use flat entity.
            // Caller should use registerBlockWithQPOffsets for proper indexing.
            indexing = AuxiliaryBlockIndexing::createCell(
                entity_count, spec.size);
            break;
        case AuxiliaryStateScope::Region:
            FE_THROW_IF(owned_entity_count != entity_count, InvalidArgumentException,
                        "AuxiliaryStateManager::registerBlock: Region scope does not support ghosts");
            indexing = AuxiliaryBlockIndexing::createRegion(entity_count, spec.size);
            break;
        case AuxiliaryStateScope::Boundary:
            FE_THROW_IF(entity_count != 1u, InvalidArgumentException,
                        "AuxiliaryStateManager::registerBlock: Boundary scope requires entity_count == 1");
            FE_THROW_IF(owned_entity_count != entity_count, InvalidArgumentException,
                        "AuxiliaryStateManager::registerBlock: Boundary scope does not support ghosts");
            indexing = AuxiliaryBlockIndexing::createBoundary(spec.size);
            break;
        case AuxiliaryStateScope::Facet:
            FE_THROW_IF(owned_entity_count != entity_count, InvalidArgumentException,
                        "AuxiliaryStateManager::registerBlock: Facet scope does not support ghosts");
            indexing = AuxiliaryBlockIndexing::createFacet(
                entity_count, spec.size);
            break;
    }

    const auto block_idx = state_.registerBlock(spec, entity_count, initial_values);
    state_.block(block_idx).setOwnedEntityCount(indexing.ownedEntityCount());

    const auto meta_idx = block_meta_.size();
    block_meta_.push_back(BlockMeta{spec, indexing, {}, {}});
    meta_name_to_index_.emplace(spec.name, meta_idx);

    return block_idx;
}

std::size_t AuxiliaryStateManager::registerBlockWithQPOffsets(
    const AuxiliaryStateSpec& spec,
    std::span<const std::size_t> qp_offsets,
    std::span<const Real> initial_values)
{
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "AuxiliaryStateManager::registerBlockWithQPOffsets: empty block name");
    FE_THROW_IF(meta_name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                "AuxiliaryStateManager::registerBlockWithQPOffsets: duplicate block '"
                    + spec.name + "'");
    FE_THROW_IF(spec.scope != AuxiliaryStateScope::QuadraturePoint, InvalidArgumentException,
                "AuxiliaryStateManager::registerBlockWithQPOffsets: scope must be QuadraturePoint");
    FE_THROW_IF(qp_offsets.size() < 2u, InvalidArgumentException,
                "AuxiliaryStateManager::registerBlockWithQPOffsets: qp_offsets must have >= 2 entries");

    const auto total_qps = qp_offsets.back();
    const auto block_idx = state_.registerBlock(spec, total_qps, initial_values);

    const auto meta_idx = block_meta_.size();
    block_meta_.push_back(BlockMeta{
        spec,
        AuxiliaryBlockIndexing::createQuadraturePoint(qp_offsets, spec.size),
        {},
        {},
    });
    meta_name_to_index_.emplace(spec.name, meta_idx);

    return block_idx;
}

std::size_t AuxiliaryStateManager::registerBlockRagged(
    const AuxiliaryStateSpec& spec,
    std::span<const std::size_t> offsets,
    std::span<const Real> initial_values)
{
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "AuxiliaryStateManager::registerBlockRagged: empty block name");
    FE_THROW_IF(meta_name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                "AuxiliaryStateManager::registerBlockRagged: duplicate '" + spec.name + "'");

    const auto block_idx = state_.registerBlockRagged(spec, offsets, initial_values);

    // Ragged blocks use a simple entity-count indexing (no stride).
    AuxiliaryBlockIndexing indexing;
    const auto n_entities = offsets.size() - 1;
    switch (spec.scope) {
        case AuxiliaryStateScope::Cell:
            indexing = AuxiliaryBlockIndexing::createCell(n_entities, 1);
            break;
        case AuxiliaryStateScope::Region:
            indexing = AuxiliaryBlockIndexing::createRegion(n_entities, 1);
            break;
        default:
            // For ragged layout with other scopes, use a generic index.
            indexing = AuxiliaryBlockIndexing::createCell(n_entities, 1);
            break;
    }

    const auto meta_idx = block_meta_.size();
    block_meta_.push_back(BlockMeta{spec, indexing, {}, {}});
    meta_name_to_index_.emplace(spec.name, meta_idx);

    return block_idx;
}

// ---------------------------------------------------------------------------
//  Access
// ---------------------------------------------------------------------------

const AuxiliaryBlockIndexing& AuxiliaryStateManager::getIndexing(
    std::string_view name) const
{
    return block_meta_[metaIndex(name)].indexing;
}

const AuxiliaryStateSpec& AuxiliaryStateManager::getSpec(
    std::string_view name) const
{
    return block_meta_[metaIndex(name)].spec;
}

// ---------------------------------------------------------------------------
//  Ghost synchronization
// ---------------------------------------------------------------------------

void AuxiliaryStateManager::setGhostSyncHook(
    std::string_view block_name, GhostSyncHook hook)
{
    auto idx = metaIndex(block_name);
    block_meta_[idx].ghost_sync_hook = std::move(hook);
}

void AuxiliaryStateManager::syncGhosts()
{
    for (auto& meta : block_meta_) {
        if (meta.spec.sync_policy == AuxiliarySyncPolicy::OwnedAndGhost &&
            meta.ghost_sync_hook) {
            auto& blk = state_.getBlock(meta.spec.name);
            meta.ghost_sync_hook(meta.spec.name, blk.work());
        }
    }
}

void AuxiliaryStateManager::syncGhosts(std::string_view block_name)
{
    auto idx = metaIndex(block_name);
    auto& meta = block_meta_[idx];
    if (meta.spec.sync_policy == AuxiliarySyncPolicy::OwnedAndGhost &&
        meta.ghost_sync_hook) {
        auto& blk = state_.getBlock(block_name);
        meta.ghost_sync_hook(block_name, blk.work());
    }
}

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

void AuxiliaryStateManager::resetAllToCommitted()
{
    for (auto& meta : block_meta_) {
        auto& blk = state_.getBlock(meta.spec.name);
        if (meta.spec.sync_policy == AuxiliarySyncPolicy::OwnedAndGhost &&
            meta.ghost_sync_hook) {
            blk.resetToCommitted([&meta](std::span<Real> values) {
                meta.ghost_sync_hook(meta.spec.name, values);
            });
        } else {
            blk.resetToCommitted();
        }
    }
}

void AuxiliaryStateManager::commitAll(Real time)
{
    for (auto& meta : block_meta_) {
        auto& blk = state_.getBlock(meta.spec.name);
        if (meta.spec.sync_policy == AuxiliarySyncPolicy::OwnedAndGhost &&
            meta.ghost_sync_hook) {
            blk.commitTimeStep(time, [&meta](std::span<Real> values) {
                meta.ghost_sync_hook(meta.spec.name, values);
            });
        } else {
            blk.commitTimeStep(time);
        }
    }
}

void AuxiliaryStateManager::rollbackAll()
{
    for (auto& meta : block_meta_) {
        auto& blk = state_.getBlock(meta.spec.name);
        if (meta.spec.sync_policy == AuxiliarySyncPolicy::OwnedAndGhost &&
            meta.ghost_sync_hook) {
            blk.rollback([&meta](std::span<Real> values) {
                meta.ghost_sync_hook(meta.spec.name, values);
            });
        } else {
            blk.rollback();
        }
    }
}

void AuxiliaryStateManager::clear()
{
    state_.clear();
    block_meta_.clear();
    meta_name_to_index_.clear();
}

void AuxiliaryStateManager::invalidateSetup()
{
    // Clear setup-time hooks but preserve block definitions and data.
    // Indexing may need to be rebuilt if mesh changes.
    for (auto& meta : block_meta_) {
        meta.ghost_sync_hook = {};
        meta.transfer_hook = {};
    }
}

// ---------------------------------------------------------------------------
//  Checkpoint / restart
// ---------------------------------------------------------------------------

std::vector<Real> AuxiliaryStateManager::packBlock(std::string_view block_name) const
{
    const auto& blk = state_.getBlock(block_name);
    const auto sz = blk.storageSize();

    std::vector<Real> packed(sz * 2);
    auto committed = blk.committed();
    auto work = blk.work();

    std::copy(committed.begin(), committed.end(), packed.begin());
    std::copy(work.begin(), work.end(), packed.begin() + static_cast<std::ptrdiff_t>(sz));

    return packed;
}

void AuxiliaryStateManager::unpackBlock(
    std::string_view block_name,
    std::span<const Real> packed_data)
{
    const auto meta_idx = metaIndex(block_name);
    auto& meta = block_meta_[meta_idx];
    auto& blk = state_.getBlock(block_name);
    const auto sz = blk.storageSize();

    FE_THROW_IF(packed_data.size() != sz * 2, InvalidArgumentException,
                "AuxiliaryStateManager::unpackBlock: expected " +
                    std::to_string(sz * 2) + " values, got " +
                    std::to_string(packed_data.size()));

    // Restore committed via initialize (sets both work and committed),
    // then overwrite work with the packed work values.
    blk.initialize(packed_data.subspan(0, sz));
    auto work = blk.work();
    std::copy(packed_data.begin() + static_cast<std::ptrdiff_t>(sz),
              packed_data.end(), work.begin());

    if (meta.spec.sync_policy == AuxiliarySyncPolicy::OwnedAndGhost &&
        meta.ghost_sync_hook) {
        meta.ghost_sync_hook(block_name, blk.committedMutable());
        meta.ghost_sync_hook(block_name, work);
    }
}

std::vector<Real> AuxiliaryStateManager::packAll() const
{
    // First pass: compute total size.
    std::size_t total = 0;
    for (std::size_t i = 0; i < state_.blockCount(); ++i) {
        total += 1; // header: storage_size encoded as Real
        total += state_.block(i).storageSize() * 2;
    }

    std::vector<Real> packed;
    packed.reserve(total);

    for (std::size_t i = 0; i < state_.blockCount(); ++i) {
        const auto& blk = state_.block(i);
        const auto sz = blk.storageSize();

        // Encode size as Real (lossless for sizes < 2^53).
        packed.push_back(static_cast<Real>(sz));

        auto committed = blk.committed();
        auto work = blk.work();
        packed.insert(packed.end(), committed.begin(), committed.end());
        packed.insert(packed.end(), work.begin(), work.end());
    }

    return packed;
}

void AuxiliaryStateManager::unpackAll(std::span<const Real> packed_data)
{
    std::size_t offset = 0;

    for (std::size_t i = 0; i < state_.blockCount(); ++i) {
        FE_THROW_IF(offset >= packed_data.size(), InvalidArgumentException,
                    "AuxiliaryStateManager::unpackAll: premature end of data");

        const auto sz = static_cast<std::size_t>(packed_data[offset]);
        ++offset;

        FE_THROW_IF(offset + sz * 2 > packed_data.size(), InvalidArgumentException,
                    "AuxiliaryStateManager::unpackAll: insufficient data for block " +
                        std::to_string(i));

        auto& blk = state_.block(i);
        FE_THROW_IF(blk.storageSize() != sz, InvalidArgumentException,
                    "AuxiliaryStateManager::unpackAll: size mismatch for block '" +
                        blk.name() + "'");

        blk.initialize(packed_data.subspan(offset, sz));
        offset += sz;

        auto work = blk.work();
        std::copy(packed_data.begin() + static_cast<std::ptrdiff_t>(offset),
                  packed_data.begin() + static_cast<std::ptrdiff_t>(offset + sz),
                  work.begin());
        offset += sz;

        const auto& meta = block_meta_[i];
        if (meta.spec.sync_policy == AuxiliarySyncPolicy::OwnedAndGhost &&
            meta.ghost_sync_hook) {
            const auto name = blk.name();
            meta.ghost_sync_hook(name, blk.committedMutable());
            meta.ghost_sync_hook(name, work);
        }
    }
}

// ---------------------------------------------------------------------------
//  Transfer
// ---------------------------------------------------------------------------

void AuxiliaryStateManager::setTransferHook(
    std::string_view block_name, TransferHook hook)
{
    auto idx = metaIndex(block_name);
    block_meta_[idx].transfer_hook = std::move(hook);
}

void AuxiliaryStateManager::transferBlock(
    std::string_view block_name, std::size_t new_entity_count)
{
    auto meta_idx = metaIndex(block_name);
    auto& meta = block_meta_[meta_idx];
    auto& blk = state_.getBlock(block_name);

    if (meta.transfer_hook) {
        // Use the hook for custom remap.
        const auto old_count = blk.entityCount();
        const auto new_sz = new_entity_count *
                            static_cast<std::size_t>(blk.componentStride());

        std::vector<Real> new_data(new_sz, 0.0);
        meta.transfer_hook(blk.work(), old_count, new_entity_count, new_data);

        blk.resize(new_entity_count);
        blk.initialize(new_data);
    } else {
        // Default: resize and preserve existing data.
        blk.resize(new_entity_count);
    }
}

void AuxiliaryStateManager::reinitializeBlock(
    std::string_view block_name, std::size_t new_entity_count)
{
    auto& blk = state_.getBlock(block_name);
    blk.resize(new_entity_count);

    // Zero-fill both buffers.
    auto work = blk.work();
    std::fill(work.begin(), work.end(), Real{0.0});
    blk.initialize(work);
}

// ---------------------------------------------------------------------------
//  Validation
// ---------------------------------------------------------------------------

void AuxiliaryStateManager::validate() const
{
    FE_THROW_IF(block_meta_.size() != state_.blockCount(), InvalidStateException,
                "AuxiliaryStateManager::validate: meta count (" +
                    std::to_string(block_meta_.size()) +
                    ") != block count (" +
                    std::to_string(state_.blockCount()) + ")");

    for (std::size_t i = 0; i < state_.blockCount(); ++i) {
        const auto& blk = state_.block(i);
        const auto& meta = block_meta_[i];

        FE_THROW_IF(blk.name() != meta.spec.name, InvalidStateException,
                    "AuxiliaryStateManager::validate: block name mismatch at index " +
                        std::to_string(i));

        FE_THROW_IF(blk.work().size() != blk.committed().size(), InvalidStateException,
                    "AuxiliaryStateManager::validate: work/committed size mismatch "
                    "for block '" + blk.name() + "'");
        FE_THROW_IF(blk.entityCount() != meta.indexing.totalEntityCount(), InvalidStateException,
                    "AuxiliaryStateManager::validate: entity count mismatch for block '" +
                        blk.name() + "'");
        FE_THROW_IF(blk.ownedEntityCount() != meta.indexing.ownedEntityCount(), InvalidStateException,
                    "AuxiliaryStateManager::validate: owned entity count mismatch for block '" +
                        blk.name() + "'");

        if (blk.layoutMode() == AuxiliaryLayoutMode::FixedStride) {
            const auto expected = blk.entityCount() *
                                  static_cast<std::size_t>(blk.componentStride());
            FE_THROW_IF(blk.storageSize() != expected, InvalidStateException,
                        "AuxiliaryStateManager::validate: storage size mismatch "
                        "for block '" + blk.name() + "'");
            FE_THROW_IF(blk.storageSize() != meta.indexing.totalStorageSize(), InvalidStateException,
                        "AuxiliaryStateManager::validate: indexing storage mismatch "
                        "for block '" + blk.name() + "'");
        }
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
