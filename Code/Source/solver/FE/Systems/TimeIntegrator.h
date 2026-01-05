/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_TIME_INTEGRATOR_H
#define SVMP_FE_SYSTEMS_TIME_INTEGRATOR_H

/**
 * @file TimeIntegrator.h
 * @brief Systems-owned time-integration interface for lowering symbolic `dt(·,k)` operators
 *
 * This file intentionally defines *only* time discretization responsibilities.
 * It does not belong to FE/Forms.
 */

#include "Assembly/TimeIntegrationContext.h"
#include "Systems/SystemState.h"

#include <memory>
#include <string>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Abstract interface for time integration metadata and stencils
 *
 * This interface is consumed by Systems-level transient orchestration to:
 * - validate derivative orders,
 * - produce an `assembly::TimeIntegrationContext` that enables assembly-time
 *   lowering of symbolic `dt(·,k)` nodes.
 */
class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Maximum continuous-time derivative order supported by this integrator
     */
    [[nodiscard]] virtual int maxSupportedDerivativeOrder() const noexcept = 0;

    /**
     * @brief Build an assembly-time context for lowering symbolic `dt(·,k)`
     *
     * @param max_time_derivative_order Maximum dt order observed in the system
     * @param state Current/historical state view (must provide dt and history vectors)
     */
    [[nodiscard]] virtual assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const SystemStateView& state) const = 0;
};

/**
 * @brief Simple backward-difference integrator (minimal reference implementation)
 *
 * - For dt(u): backward Euler stencil (BDF1)  (u^n - u^{n-1})/dt
 * - For dt(u,2): second-order backward difference (u^n - 2u^{n-1} + u^{n-2})/dt^2
 *
 * This integrator is intentionally limited to orders 1 and 2.
 */
class BackwardDifferenceIntegrator final : public TimeIntegrator {
public:
    [[nodiscard]] std::string name() const override { return "BackwardDifference"; }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 2; }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const SystemStateView& state) const override;
};

/**
 * @brief BDF2 integrator for first-order systems (dt(u))
 *
 * Supports constant- and variable-step BDF2 via `SystemStateView::dt_prev`.
 *
 * Discretization (variable step):
 *   u' ≈ a0*u^n + a1*u^{n-1} + a2*u^{n-2}
 */
class BDF2Integrator final : public TimeIntegrator {
public:
    [[nodiscard]] std::string name() const override { return "BDF2"; }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 1; }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const SystemStateView& state) const override;
};

/**
 * @brief Variable-step backward-difference integrator (orders 1..5).
 *
 * Coefficients are computed from the actual step history using finite-difference
 * weights at the end of the step:
 *   u'(t_{n+1})  ≈ sum_{j} a1[j] * u^{n+1-j}
 *   u''(t_{n+1}) ≈ sum_{j} a2[j] * u^{n+1-j}
 *
 * History requirements:
 * - dt(u):   requires at least `p` history entries (u^n ... u^{n-p+1}).
 * - dt(u,2): requires at least `p+1` history entries (u^n ... u^{n-p}).
 */
class BDFIntegrator final : public TimeIntegrator {
public:
    explicit BDFIntegrator(int order);

    [[nodiscard]] std::string name() const override { return "BDF" + std::to_string(order_); }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 2; }
    [[nodiscard]] int order() const noexcept { return order_; }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const SystemStateView& state) const override;

private:
    int order_{1};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_TIME_INTEGRATOR_H
