#ifndef SVMP_FE_AUXILIARY_HISTORY_BUFFER_H
#define SVMP_FE_AUXILIARY_HISTORY_BUFFER_H

/**
 * @file AuxiliaryHistoryBuffer.h
 * @brief Time-stamped history snapshots for auxiliary state blocks.
 *
 * Manages a rolling window of committed auxiliary state snapshots, each
 * tagged with the simulation time at which it was committed.  Supports:
 *
 * - Fixed step-back access: `snapshot(k)` returns the k-th most recent
 *   committed snapshot (k=0 is the most recently pushed).
 * - Time-stamped queries: `snapshotAtTime(t)` returns the snapshot
 *   whose timestamp matches `t`.
 * - Interpolation: `interpolate(t)` returns a linearly interpolated
 *   state between adjacent snapshots (when enabled), or delegates to a
 *   formulation-defined interpolation hook.
 *
 * History depth is block-wide and configured at setup time.
 */

#include "Core/Types.h"
#include "Core/Alignment.h"
#include "Core/AlignedAllocator.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief A single time-stamped snapshot of auxiliary state data.
 */
struct AuxiliaryHistorySnapshot {
    using AlignedVec = std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>>;

    Real time{std::numeric_limits<Real>::quiet_NaN()};
    AlignedVec data{};
};

/**
 * @brief Rolling history buffer for one auxiliary block.
 *
 * Stores up to `max_depth` time-stamped snapshots.  When a new snapshot
 * is pushed and the buffer is full, the oldest snapshot is discarded.
 *
 * Snapshots are stored newest-first: index 0 is the most recently pushed.
 */
class AuxiliaryHistoryBuffer {
public:
    /**
     * @brief Callback type for formulation-defined interpolation.
     *
     * Arguments: (time, snapshot_before, time_before, snapshot_after, time_after, output)
     * The callback writes the interpolated result into `output`.
     */
    using InterpolationHook = std::function<void(
        Real t,
        std::span<const Real> before, Real t_before,
        std::span<const Real> after, Real t_after,
        std::span<Real> output)>;

    AuxiliaryHistoryBuffer() = default;

    // -----------------------------------------------------------------
    //  Setup
    // -----------------------------------------------------------------

    /**
     * @brief Configure the history buffer.
     *
     * @param storage_size   Number of Real values per snapshot.
     * @param max_depth      Maximum number of snapshots to retain.
     * @param interp_policy  Interpolation policy for off-grid access.
     */
    void setup(std::size_t storage_size, std::size_t max_depth,
               AuxiliaryHistoryInterpolationPolicy interp_policy =
                   AuxiliaryHistoryInterpolationPolicy::None);

    /**
     * @brief Set a formulation-defined interpolation hook.
     *
     * Only meaningful when `interp_policy` is `FormulationDefined`.
     */
    void setInterpolationHook(InterpolationHook hook);

    // -----------------------------------------------------------------
    //  Properties
    // -----------------------------------------------------------------

    /// Whether setup() has been called.
    [[nodiscard]] bool isSetup() const noexcept { return is_setup_; }

    /// Maximum number of snapshots that can be stored.
    [[nodiscard]] std::size_t maxDepth() const noexcept { return max_depth_; }

    /// Current number of stored snapshots.
    [[nodiscard]] std::size_t depth() const noexcept { return snapshots_.size(); }

    /// Number of Real values per snapshot.
    [[nodiscard]] std::size_t storageSize() const noexcept { return storage_size_; }

    /// Whether any snapshots are available.
    [[nodiscard]] bool empty() const noexcept { return snapshots_.empty(); }

    /// Whether the buffer has at least `k` snapshots (0-based).
    [[nodiscard]] bool hasSnapshot(std::size_t steps_back) const noexcept
    {
        return steps_back < snapshots_.size();
    }

    // -----------------------------------------------------------------
    //  Step-back access
    // -----------------------------------------------------------------

    /**
     * @brief Get the k-th most recent snapshot (0 = most recent).
     *
     * @throws InvalidArgumentException if `steps_back >= depth()`.
     */
    [[nodiscard]] std::span<const Real> snapshot(std::size_t steps_back) const;

    /**
     * @brief Get the timestamp of the k-th most recent snapshot.
     */
    [[nodiscard]] Real snapshotTime(std::size_t steps_back) const;

    /**
     * @brief Get the full snapshot record (data + time) at step-back k.
     */
    [[nodiscard]] const AuxiliaryHistorySnapshot& snapshotRecord(std::size_t steps_back) const;

    // -----------------------------------------------------------------
    //  Time-based access
    // -----------------------------------------------------------------

    /**
     * @brief Find the snapshot whose timestamp matches `t` (within tolerance).
     *
     * @return Index (step-back) if found, or `std::size_t(-1)` if not found.
     */
    [[nodiscard]] std::size_t findSnapshotAtTime(Real t, Real tol = 1.0e-14) const;

    /**
     * @brief Interpolate state at time `t` between stored snapshots.
     *
     * For `Linear` policy: performs linear interpolation between the two
     * adjacent snapshots bracketing `t`.
     *
     * For `FormulationDefined` policy: delegates to the registered hook.
     *
     * @param t      Target time.
     * @param output Buffer to write interpolated result (must be `storageSize()`).
     *
     * @throws InvalidStateException if interpolation is not configured
     *         or if `t` is outside the stored time range.
     */
    void interpolate(Real t, std::span<Real> output) const;

    // -----------------------------------------------------------------
    //  Mutation
    // -----------------------------------------------------------------

    /**
     * @brief Push a new snapshot with the given timestamp.
     *
     * If the buffer is at max depth, the oldest snapshot is discarded.
     *
     * @param time Simulation time for this snapshot.
     * @param data Snapshot data (must have `storageSize()` elements).
     */
    void push(Real time, std::span<const Real> data);

    /**
     * @brief Remove the most recent snapshot (undo last push).
     *
     * Useful for rollback workflows.
     */
    void popNewest();

    /**
     * @brief Clear all snapshots.
     */
    void clear();

    // -----------------------------------------------------------------
    //  Summary
    // -----------------------------------------------------------------

    /// Total memory used by history snapshots in Real values.
    [[nodiscard]] std::size_t totalHistoryStorage() const noexcept;

private:
    bool is_setup_{false};
    std::size_t storage_size_{0};
    std::size_t max_depth_{0};
    AuxiliaryHistoryInterpolationPolicy interp_policy_{
        AuxiliaryHistoryInterpolationPolicy::None};

    /// Newest-first: snapshots_[0] is most recently pushed.
    std::vector<AuxiliaryHistorySnapshot> snapshots_{};

    InterpolationHook interp_hook_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_HISTORY_BUFFER_H
