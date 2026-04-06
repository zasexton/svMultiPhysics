#include "Auxiliary/AuxiliaryStateStorage.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Setup
// ---------------------------------------------------------------------------

void AuxiliaryBlockStorage::setupFixedStride(
    const AuxiliaryStateSpec& spec,
    std::size_t entity_count)
{
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "AuxiliaryBlockStorage::setupFixedStride: empty block name");
    FE_THROW_IF(spec.size <= 0, InvalidArgumentException,
                "AuxiliaryBlockStorage::setupFixedStride: spec.size must be > 0");

    name_ = spec.name;
    scope_ = spec.scope;
    layout_mode_ = AuxiliaryLayoutMode::FixedStride;
    ordering_ = spec.ordering;
    component_stride_ = spec.size;
    entity_count_ = entity_count;
    storage_size_ = entity_count * static_cast<std::size_t>(spec.size);

    entity_offsets_.clear();

    work_.assign(storage_size_, Real{0.0});
    committed_.assign(storage_size_, Real{0.0});

    // Configure history
    std::size_t max_history = 0;
    switch (spec.history_mode) {
        case AuxiliaryHistoryMode::None:
            max_history = 0;
            break;
        case AuxiliaryHistoryMode::SingleStep:
            max_history = 1;
            break;
        case AuxiliaryHistoryMode::MultiStep:
            max_history = static_cast<std::size_t>(
                spec.history_depth > 0 ? spec.history_depth : 1);
            break;
    }
    history_.setup(storage_size_, max_history, spec.history_interpolation);

    is_setup_ = true;
}

void AuxiliaryBlockStorage::setupRagged(
    const AuxiliaryStateSpec& spec,
    std::span<const std::size_t> offsets)
{
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "AuxiliaryBlockStorage::setupRagged: empty block name");
    FE_THROW_IF(offsets.size() < 2u, InvalidArgumentException,
                "AuxiliaryBlockStorage::setupRagged: offsets must have >= 2 entries");
    FE_THROW_IF(offsets[0] != 0u, InvalidArgumentException,
                "AuxiliaryBlockStorage::setupRagged: offsets[0] must be 0");

    name_ = spec.name;
    scope_ = spec.scope;
    layout_mode_ = AuxiliaryLayoutMode::Ragged;
    ordering_ = AuxiliaryEntityOrdering::ByEntityThenComponent; // Only valid for ragged
    component_stride_ = 0; // Not applicable for ragged
    entity_count_ = offsets.size() - 1;
    storage_size_ = offsets.back();

    entity_offsets_.assign(offsets.begin(), offsets.end());

    work_.assign(storage_size_, Real{0.0});
    committed_.assign(storage_size_, Real{0.0});

    // Configure history
    std::size_t max_history = 0;
    switch (spec.history_mode) {
        case AuxiliaryHistoryMode::None:
            max_history = 0;
            break;
        case AuxiliaryHistoryMode::SingleStep:
            max_history = 1;
            break;
        case AuxiliaryHistoryMode::MultiStep:
            max_history = static_cast<std::size_t>(
                spec.history_depth > 0 ? spec.history_depth : 1);
            break;
    }
    history_.setup(storage_size_, max_history, spec.history_interpolation);

    is_setup_ = true;
}

void AuxiliaryBlockStorage::resize(std::size_t new_entity_count)
{
    FE_THROW_IF(!is_setup_, InvalidStateException,
                "AuxiliaryBlockStorage::resize: not set up");
    FE_THROW_IF(layout_mode_ != AuxiliaryLayoutMode::FixedStride, InvalidStateException,
                "AuxiliaryBlockStorage::resize: only supported for FixedStride layout");

    entity_count_ = new_entity_count;
    storage_size_ = new_entity_count * static_cast<std::size_t>(component_stride_);

    work_.resize(storage_size_, Real{0.0});
    committed_.resize(storage_size_, Real{0.0});
}

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

void AuxiliaryBlockStorage::resetToCommitted()
{
    FE_THROW_IF(!is_setup_, InvalidStateException,
                "AuxiliaryBlockStorage::resetToCommitted: not set up");
    work_ = committed_;
}

void AuxiliaryBlockStorage::commitTimeStep(Real time)
{
    FE_THROW_IF(!is_setup_, InvalidStateException,
                "AuxiliaryBlockStorage::commitTimeStep: not set up");

    // Push current committed into history (with its timestamp or the new time)
    if (history_.maxDepth() > 0 && !committed_.empty()) {
        // Use the time of the state being pushed (the previous committed state).
        // If history is empty, this is the initial state — use NaN or 0 as sentinel.
        history_.push(time, committed_);
    }

    committed_ = work_;
}

void AuxiliaryBlockStorage::rollback()
{
    resetToCommitted();
}

void AuxiliaryBlockStorage::initialize(std::span<const Real> values)
{
    FE_THROW_IF(!is_setup_, InvalidStateException,
                "AuxiliaryBlockStorage::initialize: not set up");
    FE_THROW_IF(values.size() != storage_size_, InvalidArgumentException,
                "AuxiliaryBlockStorage::initialize: size mismatch (" +
                    std::to_string(values.size()) + " vs " +
                    std::to_string(storage_size_) + ")");

    std::copy(values.begin(), values.end(), work_.begin());
    std::copy(values.begin(), values.end(), committed_.begin());
}

// ---------------------------------------------------------------------------
//  Layout-aware gather/scatter
// ---------------------------------------------------------------------------

namespace {
void gatherFromBuffer(const std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>>& buf,
                       std::size_t entity_idx, int stride, std::size_t entity_count,
                       AuxiliaryLayoutMode layout, AuxiliaryEntityOrdering ordering,
                       const std::vector<std::size_t>& offsets,
                       std::vector<Real>& out)
{
    const auto s = static_cast<std::size_t>(stride);
    out.resize(s);
    if (layout == AuxiliaryLayoutMode::Ragged) {
        const auto off = offsets[entity_idx];
        const auto len = offsets[entity_idx + 1] - off;
        out.resize(len);
        std::copy(buf.begin() + static_cast<std::ptrdiff_t>(off),
                  buf.begin() + static_cast<std::ptrdiff_t>(off + len),
                  out.begin());
    } else if (ordering == AuxiliaryEntityOrdering::ByEntityThenComponent) {
        const auto off = entity_idx * s;
        std::copy(buf.begin() + static_cast<std::ptrdiff_t>(off),
                  buf.begin() + static_cast<std::ptrdiff_t>(off + s),
                  out.begin());
    } else {
        // ByComponentThenEntity: components are strided by entity_count.
        for (std::size_t c = 0; c < s; ++c) {
            out[c] = buf[c * entity_count + entity_idx];
        }
    }
}
} // namespace

std::vector<Real> AuxiliaryBlockStorage::gatherEntityWork(std::size_t entity_idx) const
{
    std::vector<Real> out;
    gatherFromBuffer(work_, entity_idx, component_stride_, entity_count_,
                     layout_mode_, ordering_, entity_offsets_, out);
    return out;
}

std::vector<Real> AuxiliaryBlockStorage::gatherEntityCommitted(std::size_t entity_idx) const
{
    std::vector<Real> out;
    gatherFromBuffer(committed_, entity_idx, component_stride_, entity_count_,
                     layout_mode_, ordering_, entity_offsets_, out);
    return out;
}

std::vector<Real> AuxiliaryBlockStorage::gatherEntityHistory(
    std::size_t snapshot_idx, std::size_t entity_idx) const
{
    auto snap = history_.snapshot(snapshot_idx);
    // Treat the snapshot as a flat buffer with the same layout.
    // Convert span to a vector for the helper.
    using AlignedVec = std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>>;
    AlignedVec snap_vec(snap.begin(), snap.end());
    std::vector<Real> out;
    gatherFromBuffer(snap_vec, entity_idx, component_stride_, entity_count_,
                     layout_mode_, ordering_, entity_offsets_, out);
    return out;
}

void AuxiliaryBlockStorage::scatterEntityWork(std::size_t entity_idx,
                                                std::span<const Real> values)
{
    const auto s = static_cast<std::size_t>(component_stride_);
    if (layout_mode_ == AuxiliaryLayoutMode::Ragged) {
        const auto off = entity_offsets_[entity_idx];
        const auto len = entity_offsets_[entity_idx + 1] - off;
        std::copy(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(len),
                  work_.begin() + static_cast<std::ptrdiff_t>(off));
    } else if (ordering_ == AuxiliaryEntityOrdering::ByEntityThenComponent) {
        const auto off = entity_idx * s;
        std::copy(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(s),
                  work_.begin() + static_cast<std::ptrdiff_t>(off));
    } else {
        // ByComponentThenEntity
        for (std::size_t c = 0; c < s; ++c) {
            work_[c * entity_count_ + entity_idx] = values[c];
        }
    }
}

void AuxiliaryBlockStorage::clear()
{
    is_setup_ = false;
    name_.clear();
    component_stride_ = 0;
    entity_count_ = 0;
    storage_size_ = 0;
    entity_offsets_.clear();
    work_.clear();
    committed_.clear();
    history_.clear();
}

// ---------------------------------------------------------------------------
//  Summary
// ---------------------------------------------------------------------------

AuxiliaryStateBlockLayout AuxiliaryBlockStorage::blockLayout() const noexcept
{
    AuxiliaryStateBlockLayout layout;
    layout.block_id = block_id_;
    layout.component_stride = component_stride_;
    layout.entity_count = entity_count_;
    layout.local_storage_size = storage_size_;
    layout.owned_entity_count = entity_count_; // Phase 2: no MPI distinction yet
    layout.owned_storage_size = storage_size_;
    layout.history_storage_size = history_.totalHistoryStorage();
    return layout;
}

} // namespace systems
} // namespace FE
} // namespace svmp
