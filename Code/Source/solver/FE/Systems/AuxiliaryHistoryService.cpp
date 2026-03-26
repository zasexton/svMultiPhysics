#include "Systems/AuxiliaryHistoryService.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace systems {

void AuxiliaryHistoryService::setup(std::size_t storage_size,
                                     const HistoryServiceConfig& config)
{
    config_ = config;
    storage_size_ = storage_size;

    std::size_t max_depth = config.max_snapshots;
    if (config.retention == HistoryRetentionPolicy::Logarithmic) {
        max_depth = std::max(max_depth, static_cast<std::size_t>(config.log_levels * 4));
    }

    auto interp = AuxiliaryHistoryInterpolationPolicy::Linear;
    buffer_.setup(storage_size, max_depth, interp);
}

void AuxiliaryHistoryService::push(Real time, std::span<const Real> data)
{
    buffer_.push(time, data);
    applyRetention(time);
}

void AuxiliaryHistoryService::queryAtTime(Real t, std::span<Real> output) const
{
    diagnostics_.clear();

    if (buffer_.empty()) {
        if (config_.extrapolation == ExtrapolationPolicy::Error) {
            FE_THROW(InvalidStateException, "AuxiliaryHistoryService: no history");
        }
        std::fill(output.begin(), output.end(), Real{0.0});
        diagnostics_.push_back("No history available; returned zeros");
        return;
    }

    // Check for extrapolation.
    if (wouldExtrapolate(t)) {
        switch (config_.extrapolation) {
            case ExtrapolationPolicy::Error:
                FE_THROW(InvalidArgumentException,
                         "AuxiliaryHistoryService: query time " +
                             std::to_string(t) + " outside stored range");
            case ExtrapolationPolicy::Zero:
                std::fill(output.begin(), output.end(), Real{0.0});
                diagnostics_.push_back("Extrapolation: returned zeros");
                return;
            case ExtrapolationPolicy::Clamp: {
                auto [t_old, t_new] = timeRange();
                Real t_clamped = std::clamp(t, t_old, t_new);
                if (buffer_.depth() == 1 || std::abs(t_clamped - t_new) < 1e-30) {
                    auto snap = buffer_.snapshot(0);
                    std::copy(snap.begin(), snap.end(), output.begin());
                    return;
                }
                // Fall through to interpolation with clamped time.
                t = t_clamped;
                break;
            }
            case ExtrapolationPolicy::LinearExtrapolation:
                // Fall through to interpolation which handles linear extrap.
                break;
        }
    }

    // Single snapshot: just return it.
    if (buffer_.depth() == 1) {
        auto snap = buffer_.snapshot(0);
        std::copy(snap.begin(), snap.end(), output.begin());
        return;
    }

    // Interpolation via the buffer.
    buffer_.interpolate(t, output);
}

void AuxiliaryHistoryService::evaluateDelay(
    Real current_time,
    const DelayQuery& query,
    std::span<const Real> current_state,
    std::span<Real> output) const
{
    Real delay = query.delay;
    if (!query.constant_delay && query.delay_fn) {
        delay = query.delay_fn(current_state, current_time);
    }

    FE_THROW_IF(delay < 0.0, InvalidArgumentException,
                "AuxiliaryHistoryService: negative delay");

    if (delay < 1e-30) {
        // Zero delay: return current state.
        std::copy(current_state.begin(), current_state.end(), output.begin());
        return;
    }

    queryAtTime(current_time - delay, output);
}

void AuxiliaryHistoryService::applyRetention(Real current_time)
{
    if (buffer_.depth() <= 1) return;

    switch (config_.retention) {
        case HistoryRetentionPolicy::FixedWindow:
            // Buffer handles this automatically via max_depth.
            break;

        case HistoryRetentionPolicy::TimeWindow: {
            Real cutoff = current_time - config_.time_window;
            while (buffer_.depth() > 1) {
                Real oldest_time = buffer_.snapshotTime(buffer_.depth() - 1);
                if (oldest_time < cutoff) {
                    // Remove oldest by popping from the back.
                    // AuxiliaryHistoryBuffer only supports popNewest.
                    // For time-window pruning, we'd need popOldest.
                    // For now, rely on the max_depth mechanism.
                    break;
                }
                break;
            }
            break;
        }

        case HistoryRetentionPolicy::Logarithmic:
            // Logarithmic pruning: keep snapshots at exponentially
            // increasing intervals from the present.
            // Implementation deferred — max_depth mechanism handles basic cases.
            break;

        case HistoryRetentionPolicy::FormulationDefined:
            // Custom retention via predicate — would need popOldest support.
            break;
    }
}

bool AuxiliaryHistoryService::wouldExtrapolate(Real t) const
{
    if (buffer_.empty()) return true;
    if (buffer_.depth() == 1) return false; // Single point — no interpolation needed.

    auto [t_old, t_new] = timeRange();
    return (t < t_old - 1e-14) || (t > t_new + 1e-14);
}

std::pair<Real, Real> AuxiliaryHistoryService::timeRange() const
{
    if (buffer_.empty()) {
        return {std::numeric_limits<Real>::quiet_NaN(),
                std::numeric_limits<Real>::quiet_NaN()};
    }
    Real newest = buffer_.snapshotTime(0);
    Real oldest = buffer_.snapshotTime(buffer_.depth() - 1);
    return {oldest, newest};
}

} // namespace systems
} // namespace FE
} // namespace svmp
