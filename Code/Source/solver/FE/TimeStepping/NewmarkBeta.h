/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_NEWMARK_BETA_H
#define SVMP_FE_TIMESTEPPING_NEWMARK_BETA_H

#include "Systems/TimeIntegrator.h"

namespace svmp {
namespace FE {
namespace timestepping {

struct NewmarkBetaIntegratorOptions {
    double beta{0.25};
    double gamma{0.5};
};

/**
 * @brief Newmark-β time integrator metadata for 2nd-order structural dynamics using displacement-only unknown.
 *
 * This integrator assumes the time loop solves for u_{n+1} and encodes
 * scheme-specific constants into the first two history slots:
 * - u_prev  holds the constant term for dt(u,2) -> a_{n+1}
 * - u_prev2 holds the constant term for dt(u)   -> v_{n+1}
 *
 * The resulting lowering is affine in the step unknown:
 *   dt(u)   = (γ/(βΔt)) * u_{n+1} + const_v
 *   dt(u,2) = (1/(βΔt^2)) * u_{n+1} + const_a
 */
class NewmarkBetaIntegrator final : public systems::TimeIntegrator {
public:
    explicit NewmarkBetaIntegrator(NewmarkBetaIntegratorOptions options);

    [[nodiscard]] std::string name() const override { return "NewmarkBeta"; }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 2; }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const override;

private:
    NewmarkBetaIntegratorOptions options_{};
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_NEWMARK_BETA_H

