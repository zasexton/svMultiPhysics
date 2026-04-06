#ifndef SVMP_FE_AUXILIARY_HISTORY_SERVICE_H
#define SVMP_FE_AUXILIARY_HISTORY_SERVICE_H

/**
 * @file AuxiliaryHistoryService.h
 * @brief Rich history access, delay evaluation, and retention management
 *        for auxiliary state blocks.
 *
 * Wraps `AuxiliaryHistoryBuffer` with higher-level services:
 * - Retention policies (short window, long window, formulation-defined)
 * - Time-query APIs with interpolation and extrapolation control
 * - Delay-evaluation helpers for constant and variable delays
 * - Checkpoint pruning and compression for long histories
 * - Diagnostics for missing history, extrapolation, and invalid queries
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryHistoryBuffer.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Retention policy
// ---------------------------------------------------------------------------

/**
 * @brief How history is retained over time.
 */
enum class HistoryRetentionPolicy : std::uint8_t {
    /// Keep a fixed number of most recent snapshots (FIFO).
    FixedWindow,

    /// Keep all snapshots within a time window [t - T_window, t].
    TimeWindow,

    /// Keep snapshots at logarithmically spaced intervals.
    Logarithmic,

    /// Formulation provides a custom retention predicate.
    FormulationDefined
};

/**
 * @brief How to handle queries outside the stored history range.
 */
enum class ExtrapolationPolicy : std::uint8_t {
    /// Throw an error for out-of-range queries.
    Error,

    /// Clamp to the nearest available snapshot.
    Clamp,

    /// Extrapolate linearly from the boundary snapshots.
    LinearExtrapolation,

    /// Return zero for out-of-range queries.
    Zero
};

/**
 * @brief Configuration for the history service.
 */
struct HistoryServiceConfig {
    HistoryRetentionPolicy retention{HistoryRetentionPolicy::FixedWindow};
    ExtrapolationPolicy extrapolation{ExtrapolationPolicy::Error};

    /// For FixedWindow: max snapshots.
    std::size_t max_snapshots{8};

    /// For TimeWindow: window size in time units.
    Real time_window{1.0};

    /// For Logarithmic: number of levels (each level halves the resolution).
    int log_levels{4};

    /// Custom retention predicate (for FormulationDefined).
    /// Arguments: (snapshot_time, current_time, snapshot_index) → keep?
    std::function<bool(Real, Real, std::size_t)> retention_predicate{};
};

// ---------------------------------------------------------------------------
//  Delay query descriptor
// ---------------------------------------------------------------------------

/**
 * @brief Descriptor for a delay evaluation query.
 */
struct DelayQuery {
    /// The delay amount (positive, in time units).
    Real delay{0.0};

    /// Whether the delay is constant or state-dependent.
    bool constant_delay{true};

    /// For state-dependent delays: callback returning the delay value.
    std::function<Real(std::span<const Real> x, Real t)> delay_fn{};
};

// ---------------------------------------------------------------------------
//  History service
// ---------------------------------------------------------------------------

/**
 * @brief High-level history service wrapping AuxiliaryHistoryBuffer.
 */
class AuxiliaryHistoryService {
public:
    AuxiliaryHistoryService() = default;

    /**
     * @brief Configure the service for a block.
     *
     * @param storage_size  Number of Real values per snapshot.
     * @param config        Service configuration.
     */
    void setup(std::size_t storage_size, const HistoryServiceConfig& config);

    /// Whether setup() has been called.
    [[nodiscard]] bool isSetup() const noexcept { return buffer_.isSetup(); }

    // -----------------------------------------------------------------
    //  Push / commit
    // -----------------------------------------------------------------

    /**
     * @brief Push a new snapshot and apply retention policy.
     */
    void push(Real time, std::span<const Real> data);

    // -----------------------------------------------------------------
    //  Time-based queries
    // -----------------------------------------------------------------

    /**
     * @brief Query state at a specific time.
     *
     * Uses interpolation between stored snapshots.  Handles out-of-range
     * queries according to the extrapolation policy.
     *
     * @param t       Query time.
     * @param output  Buffer for result (must be storage_size).
     */
    void queryAtTime(Real t, std::span<Real> output) const;

    /**
     * @brief Evaluate a delayed state: x(t - delay).
     *
     * @param current_time  Current time t.
     * @param query         Delay descriptor.
     * @param current_state Current state x(t) (for state-dependent delays).
     * @param output        Buffer for result.
     */
    void evaluateDelay(Real current_time,
                        const DelayQuery& query,
                        std::span<const Real> current_state,
                        std::span<Real> output) const;

    // -----------------------------------------------------------------
    //  Step-back access (delegate to buffer)
    // -----------------------------------------------------------------

    [[nodiscard]] std::size_t depth() const noexcept { return buffer_.depth(); }
    [[nodiscard]] std::span<const Real> snapshot(std::size_t k) const
    {
        return buffer_.snapshot(k);
    }
    [[nodiscard]] Real snapshotTime(std::size_t k) const
    {
        return buffer_.snapshotTime(k);
    }

    // -----------------------------------------------------------------
    //  Maintenance
    // -----------------------------------------------------------------

    /**
     * @brief Apply retention policy to prune old snapshots.
     *
     * Called automatically after push().  Can also be called manually.
     */
    void applyRetention(Real current_time);

    /**
     * @brief Clear all snapshots.
     */
    void clear() { buffer_.clear(); }

    /// Access the underlying buffer.
    [[nodiscard]] const AuxiliaryHistoryBuffer& buffer() const noexcept
    {
        return buffer_;
    }

    // -----------------------------------------------------------------
    //  Diagnostics
    // -----------------------------------------------------------------

    /**
     * @brief Check if a time query would require extrapolation.
     */
    [[nodiscard]] bool wouldExtrapolate(Real t) const;

    /**
     * @brief Get the stored time range [oldest, newest].
     */
    [[nodiscard]] std::pair<Real, Real> timeRange() const;

    /// Diagnostic messages from the last query.
    [[nodiscard]] const std::vector<std::string>& diagnostics() const noexcept
    {
        return diagnostics_;
    }

private:
    AuxiliaryHistoryBuffer buffer_{};
    HistoryServiceConfig config_{};
    std::size_t storage_size_{0};
    mutable std::vector<std::string> diagnostics_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_HISTORY_SERVICE_H
