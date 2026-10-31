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

#include <algorithm>
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
     * @brief Whether a one-based history slot carries a rate-like quantity.
     *
     * Most integrators store solution values in every history slot.  Some
     * stage formulations temporarily inject a rate into one of those slots;
     * affine MPC projection must then use the homogeneous relation (no value
     * inhomogeneity) for that slot.
     */
    [[nodiscard]] virtual bool
    historySlotStoresRate(int /*one_based_slot*/) const noexcept
    {
        return false;
    }

    /**
     * @brief Whether a one-based history slot holds a previous solution during the solve.
     *
     * `previousSolution(k)` reads slot k.  A slot that holds an injected rate
     * or a scheme constant does not hold u^{n+1-k}; TransientSystem refuses
     * forms that read it.
     */
    [[nodiscard]] virtual bool historySlotIsSolution(int one_based_slot) const noexcept
    {
        return !historySlotStoresRate(one_based_slot);
    }

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
 * This integrator supports dt orders up to 8 with a uniform-step backward stencil.
 */
class BackwardDifferenceIntegrator final : public TimeIntegrator {
public:
    [[nodiscard]] std::string name() const override { return "BackwardDifference"; }
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 8; }

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
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return std::min(8, 9 - order_); }
    [[nodiscard]] int order() const noexcept { return order_; }

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const SystemStateView& state) const override;

private:
    int order_{1};
};

/**
 * @brief Integrator whose time-derivative stencils are identically zero.
 *
 * Assembling with it evaluates the steady residual of a transient system
 * (every dt(u) and dt(u,2) term vanishes); used to certify static states.
 */
class ZeroTimeDerivativeIntegrator final : public svmp::FE::systems::TimeIntegrator {
public:
  [[nodiscard]] std::string name() const override { return "ZeroTimeDerivative"; }
  [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override { return 2; }

  [[nodiscard]] svmp::FE::assembly::TimeIntegrationContext buildContext(
      int max_time_derivative_order, const svmp::FE::systems::SystemStateView& /*state*/) const override
  {
    svmp::FE::assembly::TimeIntegrationContext ctx;
    ctx.integrator_name = name();
    if (max_time_derivative_order <= 0) {
      return ctx;
    }

    if (max_time_derivative_order >= 1) {
      svmp::FE::assembly::TimeDerivativeStencil s;
      s.order = 1;
      s.a.assign(1, static_cast<svmp::FE::Real>(0.0));
      ctx.dt1 = std::move(s);
    }
    if (max_time_derivative_order >= 2) {
      svmp::FE::assembly::TimeDerivativeStencil s;
      s.order = 2;
      s.a.assign(1, static_cast<svmp::FE::Real>(0.0));
      ctx.dt2 = std::move(s);
    }

    return ctx;
  }
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_TIME_INTEGRATOR_H
