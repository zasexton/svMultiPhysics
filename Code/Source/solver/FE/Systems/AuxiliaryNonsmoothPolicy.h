#ifndef SVMP_FE_SYSTEMS_AUXILIARY_NONSMOOTH_POLICY_H
#define SVMP_FE_SYSTEMS_AUXILIARY_NONSMOOTH_POLICY_H

/**
 * @file AuxiliaryNonsmoothPolicy.h
 * @brief Policy types for nonsmooth, active-set, complementarity,
 *        and hybrid switching auxiliary models.
 *
 * These policies configure how the event manager and stepper handle
 * nonsmooth behavior during auxiliary state advancement.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Strategy for resolving active-set changes.
 */
enum class ActiveSetStrategy : std::uint8_t {
    /// Detect active-set changes via event functions, then re-solve.
    EventDriven,

    /// Fixed-point iteration over active-set membership.
    FixedPointIteration,

    /// Semismooth Newton (treats complementarity directly).
    SemismoothNewton
};

/**
 * @brief Strategy for handling complementarity conditions.
 */
enum class ComplementarityStrategy : std::uint8_t {
    /// NCP function reformulation (e.g., Fischer-Burmeister).
    NCPFunction,

    /// Active-set method with inequality switching.
    ActiveSet,

    /// Penalty/regularization approach.
    Penalty
};

/**
 * @brief Strategy for hybrid (multi-mode) state transitions.
 */
enum class HybridTransitionStrategy : std::uint8_t {
    /// Hard switching at event crossing.
    InstantSwitch,

    /// Smooth blending over a transition zone.
    SmoothBlend,

    /// Hysteresis with guard bands.
    Hysteresis
};

/**
 * @brief Configuration for nonsmooth auxiliary model behavior.
 */
struct AuxiliaryNonsmoothPolicy {
    /// Active-set resolution strategy.
    ActiveSetStrategy active_set{ActiveSetStrategy::EventDriven};

    /// Complementarity resolution strategy.
    ComplementarityStrategy complementarity{ComplementarityStrategy::NCPFunction};

    /// Hybrid transition strategy.
    HybridTransitionStrategy transition{HybridTransitionStrategy::InstantSwitch};

    /// Maximum active-set iterations per step.
    int max_active_set_iters{20};

    /// Tolerance for complementarity residual.
    double complementarity_tol{1.0e-10};

    /// Smoothing parameter for blended transitions.
    double blend_width{0.0};

    /// Hysteresis guard band width.
    double hysteresis_band{0.0};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_NONSMOOTH_POLICY_H
