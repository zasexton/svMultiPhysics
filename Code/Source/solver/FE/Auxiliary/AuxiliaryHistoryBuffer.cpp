#include "Auxiliary/AuxiliaryHistoryBuffer.h"

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Setup
// ---------------------------------------------------------------------------

void AuxiliaryHistoryBuffer::setup(
    std::size_t storage_size,
    std::size_t max_depth,
    AuxiliaryHistoryInterpolationPolicy interp_policy)
{
    storage_size_ = storage_size;
    max_depth_ = max_depth;
    interp_policy_ = interp_policy;
    snapshots_.clear();
    snapshots_.reserve(max_depth);
    is_setup_ = true;
}

void AuxiliaryHistoryBuffer::setInterpolationHook(InterpolationHook hook)
{
    FE_THROW_IF(!is_setup_, InvalidStateException,
                "AuxiliaryHistoryBuffer::setInterpolationHook: not set up");
    interp_hook_ = std::move(hook);
}

// ---------------------------------------------------------------------------
//  Step-back access
// ---------------------------------------------------------------------------

std::span<const Real> AuxiliaryHistoryBuffer::snapshot(std::size_t steps_back) const
{
    FE_THROW_IF(steps_back >= snapshots_.size(), InvalidArgumentException,
                "AuxiliaryHistoryBuffer::snapshot: steps_back=" +
                    std::to_string(steps_back) + " >= depth=" +
                    std::to_string(snapshots_.size()));
    return snapshots_[steps_back].data;
}

Real AuxiliaryHistoryBuffer::snapshotTime(std::size_t steps_back) const
{
    FE_THROW_IF(steps_back >= snapshots_.size(), InvalidArgumentException,
                "AuxiliaryHistoryBuffer::snapshotTime: steps_back out of range");
    return snapshots_[steps_back].time;
}

const AuxiliaryHistorySnapshot& AuxiliaryHistoryBuffer::snapshotRecord(
    std::size_t steps_back) const
{
    FE_THROW_IF(steps_back >= snapshots_.size(), InvalidArgumentException,
                "AuxiliaryHistoryBuffer::snapshotRecord: steps_back out of range");
    return snapshots_[steps_back];
}

// ---------------------------------------------------------------------------
//  Time-based access
// ---------------------------------------------------------------------------

std::size_t AuxiliaryHistoryBuffer::findSnapshotAtTime(Real t, Real tol) const
{
    for (std::size_t i = 0; i < snapshots_.size(); ++i) {
        if (std::abs(snapshots_[i].time - t) <= tol) {
            return i;
        }
    }
    return static_cast<std::size_t>(-1);
}

void AuxiliaryHistoryBuffer::interpolate(Real t, std::span<Real> output) const
{
    FE_THROW_IF(!is_setup_, InvalidStateException,
                "AuxiliaryHistoryBuffer::interpolate: not set up");
    FE_THROW_IF(interp_policy_ == AuxiliaryHistoryInterpolationPolicy::None,
                InvalidStateException,
                "AuxiliaryHistoryBuffer::interpolate: interpolation not enabled");
    FE_THROW_IF(output.size() != storage_size_, InvalidArgumentException,
                "AuxiliaryHistoryBuffer::interpolate: output size mismatch");
    FE_THROW_IF(snapshots_.size() < 2u, InvalidStateException,
                "AuxiliaryHistoryBuffer::interpolate: need at least 2 snapshots");

    // Find bracketing snapshots.
    // snapshots_ is newest-first, so times are in decreasing order.
    // Find i such that snapshots_[i].time >= t >= snapshots_[i+1].time
    std::size_t idx_after = static_cast<std::size_t>(-1);  // newer (higher time)
    std::size_t idx_before = static_cast<std::size_t>(-1); // older (lower time)

    for (std::size_t i = 0; i + 1 < snapshots_.size(); ++i) {
        if (snapshots_[i].time >= t && t >= snapshots_[i + 1].time) {
            idx_after = i;
            idx_before = i + 1;
            break;
        }
    }

    FE_THROW_IF(idx_after == static_cast<std::size_t>(-1), InvalidArgumentException,
                "AuxiliaryHistoryBuffer::interpolate: t=" + std::to_string(t) +
                    " outside stored time range [" +
                    std::to_string(snapshots_.back().time) + ", " +
                    std::to_string(snapshots_.front().time) + "]");

    const auto& snap_before = snapshots_[idx_before];
    const auto& snap_after = snapshots_[idx_after];

    if (interp_policy_ == AuxiliaryHistoryInterpolationPolicy::FormulationDefined) {
        FE_THROW_IF(!interp_hook_, InvalidStateException,
                    "AuxiliaryHistoryBuffer::interpolate: FormulationDefined "
                    "but no hook registered");
        interp_hook_(t, snap_before.data, snap_before.time,
                     snap_after.data, snap_after.time, output);
        return;
    }

    // Linear interpolation
    const Real dt = snap_after.time - snap_before.time;
    if (std::abs(dt) < 1.0e-30) {
        // Degenerate: same time, just copy
        std::copy(snap_before.data.begin(), snap_before.data.end(), output.begin());
        return;
    }
    const Real alpha = (t - snap_before.time) / dt; // 0 at before, 1 at after
    for (std::size_t j = 0; j < storage_size_; ++j) {
        output[j] = (1.0 - alpha) * snap_before.data[j] + alpha * snap_after.data[j];
    }
}

// ---------------------------------------------------------------------------
//  Mutation
// ---------------------------------------------------------------------------

void AuxiliaryHistoryBuffer::push(Real time, std::span<const Real> data)
{
    FE_THROW_IF(!is_setup_, InvalidStateException,
                "AuxiliaryHistoryBuffer::push: not set up");
    FE_THROW_IF(data.size() != storage_size_, InvalidArgumentException,
                "AuxiliaryHistoryBuffer::push: data size " +
                    std::to_string(data.size()) + " != storage_size " +
                    std::to_string(storage_size_));

    if (max_depth_ == 0) {
        return; // History disabled
    }

    // Drop oldest if at capacity
    if (snapshots_.size() >= max_depth_) {
        snapshots_.pop_back();
    }

    // Insert at front (newest-first)
    snapshots_.emplace(snapshots_.begin());
    snapshots_.front().time = time;
    snapshots_.front().data.assign(data.begin(), data.end());
}

void AuxiliaryHistoryBuffer::popNewest()
{
    FE_THROW_IF(snapshots_.empty(), InvalidStateException,
                "AuxiliaryHistoryBuffer::popNewest: buffer is empty");
    snapshots_.erase(snapshots_.begin());
}

void AuxiliaryHistoryBuffer::clear()
{
    snapshots_.clear();
}

// ---------------------------------------------------------------------------
//  Summary
// ---------------------------------------------------------------------------

std::size_t AuxiliaryHistoryBuffer::totalHistoryStorage() const noexcept
{
    return snapshots_.size() * storage_size_;
}

} // namespace systems
} // namespace FE
} // namespace svmp
