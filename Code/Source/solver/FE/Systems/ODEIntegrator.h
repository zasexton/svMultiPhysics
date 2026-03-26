#ifndef SVMP_FE_SYSTEMS_ODE_INTEGRATOR_H
#define SVMP_FE_SYSTEMS_ODE_INTEGRATOR_H

/**
 * @file ODEIntegrator.h
 * @brief Legacy scalar ODE integrator for AuxiliaryState variables.
 *
 * This is the original scalar integrator used by the coupled-boundary path.
 * It advances a single named scalar variable in the flat `AuxiliaryState`
 * buffer using a FormExpr-based RHS function.
 *
 * ## Relationship to the generalized stepper interface
 *
 * The new `AuxiliaryStateStepper` interface (in `AuxiliaryStateStepper.h`)
 * provides a block-level, residual-based, DAE-capable stepper API that
 * supports multi-component blocks, implicit/explicit methods, and
 * automatic derivative infrastructure.
 *
 * `ODEIntegrator` remains available for backward compatibility with the
 * coupled-boundary ODE path (scalar FormExpr RHS + `PointEvaluator`).
 * New auxiliary models should use `AuxiliaryStateStepper` instead.
 */

#include "Core/Types.h"

#include "Forms/FormExpr.h"

#include <cstdint>
#include <string_view>
#include <span>
#include <optional>

namespace svmp {
namespace FE {

namespace systems {

class AuxiliaryState;

enum class ODEMethod : std::uint8_t {
    ForwardEuler,
    BackwardEuler,
    RK4,
    BDF2,
};

/// Generic scalar ODE integrator.
class ODEIntegrator {
public:
    static void advance(ODEMethod method,
                        std::uint32_t state_slot,
                        AuxiliaryState& state,
                        const forms::FormExpr& rhs,
                        const std::optional<forms::FormExpr>& d_rhs_dX,
                        std::span<const Real> integrals,
                        std::span<const Real> params,
                        Real t,
                        Real dt);
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_ODE_INTEGRATOR_H
