#ifndef SVMP_FE_AUXILIARY_MULTIRATE_SCHEDULER_H
#define SVMP_FE_AUXILIARY_MULTIRATE_SCHEDULER_H

/**
 * @file AuxiliaryMultirateScheduler.h
 * @brief Multirate advancement scheduling for auxiliary state blocks.
 *
 * Promotes multirate support from lifecycle hooks into structured
 * algorithms with:
 * - Predictor/corrector policies for subcycled and multirate advancement
 * - Error-control and convergence-monitoring hooks
 * - Consistency rules for Partitioned vs Monolithic under mixed rates
 * - Checkpoint strategies for rollback under failed iterations
 *
 * ## Mixed-rate consistency
 *
 * When Partitioned and Monolithic blocks coexist:
 * - Monolithic blocks advance at the PDE time step (they participate in
 *   the assembled solve).
 * - Partitioned blocks may subcycle (advance at smaller dt) or run at
 *   a coarser rate (advance less often).
 * - The scheduler ensures that Partitioned blocks are synchronized at
 *   PDE step boundaries for coupling consistency.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Predictor/corrector policy
// ---------------------------------------------------------------------------

/**
 * @brief Strategy for predicting auxiliary state at intermediate times.
 */
enum class MultiratePredictorPolicy : std::uint8_t {
    /// Hold the last committed value (zero-order hold).
    ZeroOrderHold,

    /// Linear extrapolation from the two most recent values.
    LinearExtrapolation,

    /// Use the block's own stepper for a provisional advance.
    StepperPredict
};

/**
 * @brief Strategy for correcting after the PDE step completes.
 */
enum class MultirateCorrectorPolicy : std::uint8_t {
    /// No correction (accept the predicted/subcycled result as-is).
    None,

    /// Re-advance from the committed state using updated inputs.
    ReadvanceFromCommitted,

    /// Iterate (predict-correct loop) until convergence.
    IterateUntilConverged
};

// ---------------------------------------------------------------------------
//  Error control
// ---------------------------------------------------------------------------

/**
 * @brief Error estimation method for multirate advancement.
 */
enum class MultirateErrorEstimator : std::uint8_t {
    /// No error estimation.
    None,

    /// Compare full-step vs half-step (Richardson extrapolation).
    StepDoubling,

    /// Compare predictor vs corrector.
    PredictorCorrector,

    /// Formulation-defined error estimator.
    Custom
};

// ---------------------------------------------------------------------------
//  Block schedule entry
// ---------------------------------------------------------------------------

/**
 * @brief Schedule entry for one auxiliary block.
 */
struct MultirateBlockSchedule {
    /// Block name.
    std::string block_name{};

    /// Rate ratio relative to the PDE step (1 = same rate, 2 = twice as fast, etc.).
    int rate_ratio{1};

    /// Predictor policy for times between block advances.
    MultiratePredictorPolicy predictor{MultiratePredictorPolicy::ZeroOrderHold};

    /// Corrector policy after PDE step.
    MultirateCorrectorPolicy corrector{MultirateCorrectorPolicy::None};

    /// Error estimator for this block's multirate advancement.
    MultirateErrorEstimator error_estimator{MultirateErrorEstimator::None};

    /// Error tolerance for convergence (for IterateUntilConverged corrector).
    Real error_tolerance{1.0e-8};

    /// Maximum corrector iterations.
    int max_corrector_iters{5};

    /// Whether to checkpoint this block before multirate advancement
    /// (enables rollback on failure).
    bool checkpoint_before_advance{true};
};

// ---------------------------------------------------------------------------
//  Scheduler
// ---------------------------------------------------------------------------

/**
 * @brief Multirate advancement scheduler for auxiliary blocks.
 */
class AuxiliaryMultirateScheduler {
public:
    AuxiliaryMultirateScheduler() = default;

    // -----------------------------------------------------------------
    //  Configuration
    // -----------------------------------------------------------------

    /**
     * @brief Add a block schedule entry.
     */
    void addBlockSchedule(MultirateBlockSchedule schedule);

    /**
     * @brief Get the schedule for a block (nullptr if not scheduled).
     */
    [[nodiscard]] const MultirateBlockSchedule* getSchedule(
        std::string_view block_name) const;

    /// All scheduled block names.
    [[nodiscard]] std::vector<std::string> scheduledBlocks() const;

    // -----------------------------------------------------------------
    //  Advancement plan
    // -----------------------------------------------------------------

    /**
     * @brief Substep descriptor for one block within a PDE step.
     */
    struct Substep {
        std::string block_name{};
        Real t_start{0.0};
        Real dt_sub{0.0};
        int substep_index{0};
        int total_substeps{1};
    };

    /**
     * @brief Compute the substep plan for a PDE step.
     *
     * Returns an ordered list of substeps across all blocks.
     * Substeps are interleaved by time order.
     *
     * @param t   PDE step start time.
     * @param dt  PDE step size.
     */
    [[nodiscard]] std::vector<Substep> planSubsteps(Real t, Real dt) const;

    // -----------------------------------------------------------------
    //  Consistency checks
    // -----------------------------------------------------------------

    /**
     * @brief Validate that the schedule is consistent.
     *
     * Checks:
     * - Rate ratios are positive integers.
     * - Monolithic blocks have rate_ratio == 1 (they advance with the PDE).
     * - No conflicting corrector policies.
     *
     * @param monolithic_blocks  Names of blocks with Monolithic solve mode.
     * @return Empty if valid; diagnostic messages if not.
     */
    [[nodiscard]] std::vector<std::string> validate(
        std::span<const std::string> monolithic_blocks) const;

    // -----------------------------------------------------------------
    //  Lifecycle
    // -----------------------------------------------------------------

    void clear();

private:
    std::vector<MultirateBlockSchedule> schedules_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_MULTIRATE_SCHEDULER_H
