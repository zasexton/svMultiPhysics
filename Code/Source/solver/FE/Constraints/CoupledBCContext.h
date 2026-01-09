#ifndef SVMP_FE_CONSTRAINTS_COUPLED_BC_CONTEXT_H
#define SVMP_FE_CONSTRAINTS_COUPLED_BC_CONTEXT_H

/**
 * @file CoupledBCContext.h
 * @brief Context for coupled boundary-condition evaluators (integrals + aux state)
 */

#include "Core/Types.h"
#include "Forms/BoundaryFunctional.h"
#include "Systems/AuxiliaryState.h"

#include <array>
#include <functional>
#include <span>

namespace svmp {
namespace FE {

namespace constraints {

struct CoupledBCContext {
    const forms::BoundaryFunctionalResults& integrals;
    const systems::AuxiliaryState& aux_state;
    Real t{0.0};
    Real dt{0.0};

    [[nodiscard]] std::span<const Real> integralsValues() const noexcept { return integrals.all(); }
    [[nodiscard]] std::span<const Real> auxValues() const noexcept { return aux_state.values(); }
};

using CoupledBCEvaluator =
    std::function<Real(const CoupledBCContext& ctx, Real x, Real y, Real z)>;

using CoupledVectorBCEvaluator =
    std::function<std::array<Real, 3>(const CoupledBCContext& ctx,
                                      Real x, Real y, Real z,
                                      const std::array<Real, 3>& normal)>;
// NOTE: These callback-based evaluators are supported for interpreter/fallback paths,
// but are not "JIT-fast" (opaque call boundary; no inlining/vectorization).

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_COUPLED_BC_CONTEXT_H
