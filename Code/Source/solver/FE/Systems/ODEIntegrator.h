#ifndef SVMP_FE_SYSTEMS_ODE_INTEGRATOR_H
#define SVMP_FE_SYSTEMS_ODE_INTEGRATOR_H

/**
 * @file ODEIntegrator.h
 * @brief Generic scalar ODE integrator for AuxiliaryState variables
 *
 * This integrator is physics-agnostic: it advances a named scalar variable
 * in `systems::AuxiliaryState` using a user-provided RHS function and a chosen
 * time-integration method.
 *
 * The RHS function has access to:
 * - the full AuxiliaryState (so coupled 0D models can reference other aux vars),
 * - the current BoundaryFunctionalResults (non-local boundary integrals),
 * - the current time `t`.
 *
 * Notes:
 * - Current scope: scalar variable integration (one named component at a time).
 * - Implicit methods (BackwardEuler/BDF2) use a robust scalar Newton solve with
 *   finite-difference derivative.
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
