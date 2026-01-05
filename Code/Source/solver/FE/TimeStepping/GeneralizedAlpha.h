/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_GENERALIZED_ALPHA_H
#define SVMP_FE_TIMESTEPPING_GENERALIZED_ALPHA_H

#include "Systems/TimeIntegrator.h"

namespace svmp {
namespace FE {
namespace timestepping {

struct GeneralizedAlphaFirstOrderIntegratorOptions {
    double alpha_m{0.0};
    double alpha_f{0.0};
    double gamma{0.0};

    // Order used to reconstruct u_dot^n from history without storing rate vectors.
    // 1 -> (u^n - u^{n-1})/dt
    // 2 -> 2nd-order backward difference (variable step supported)
    int history_rate_order{2};
};

struct GeneralizedAlphaSecondOrderIntegratorOptions {
    double alpha_m{0.0};
    double alpha_f{0.0};
    double beta{0.0};
    double gamma{0.0};
};

/**
 * @brief Generalized-α time integrator for first-order systems (JWH-style) using only solution history.
 *
 * This targets systems written in the standard first-order form:
 *   M(u,t) * u̇ = R(u,t)
 *
 * Spatial terms are evaluated at the stage state u_{n+α_f} and u̇_{n+α_m} is
 * approximated by combining:
 * - the stage increment (u_{n+α_f} - u_n), and
 * - a reconstructed u̇_n from solution history.
 */
class GeneralizedAlphaFirstOrderIntegrator final : public systems::TimeIntegrator {
public:
    explicit GeneralizedAlphaFirstOrderIntegrator(GeneralizedAlphaFirstOrderIntegratorOptions options);

    [[nodiscard]] std::string name() const override { return "GeneralizedAlpha(1stOrder)"; }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 1; }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const override;

private:
    GeneralizedAlphaFirstOrderIntegratorOptions options_{};
};

/**
 * @brief Generalized-α integrator metadata for 2nd-order structural dynamics using stage displacement.
 *
 * This integrator assumes the time loop solves for the stage displacement
 *   u_{n+α_f}
 * and encodes scheme-specific constant vectors into the first two history slots:
 * - u_prev  holds the constant term for dt(u,2) -> a_{n+α_m}
 * - u_prev2 holds the constant term for dt(u)   -> v_{n+α_f}
 *
 * The resulting lowering is affine in the stage unknown:
 *   dt(u)   = (γ/(βΔt)) * u_stage + const_v
 *   dt(u,2) = (α_m/(α_f β Δt^2)) * u_stage + const_a
 */
class GeneralizedAlphaSecondOrderIntegrator final : public systems::TimeIntegrator {
public:
    explicit GeneralizedAlphaSecondOrderIntegrator(GeneralizedAlphaSecondOrderIntegratorOptions options);

    [[nodiscard]] std::string name() const override { return "GeneralizedAlpha(2ndOrder)"; }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 2; }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const override;

private:
    GeneralizedAlphaSecondOrderIntegratorOptions options_{};
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_GENERALIZED_ALPHA_H
