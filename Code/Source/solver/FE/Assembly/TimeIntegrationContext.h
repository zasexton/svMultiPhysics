/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_TIME_INTEGRATION_CONTEXT_H
#define SVMP_FE_ASSEMBLY_TIME_INTEGRATION_CONTEXT_H

/**
 * @file TimeIntegrationContext.h
 * @brief Backend-provided data for interpreting symbolic time-derivative operators
 *
 * This header defines a small, backend-facing data structure that allows
 * FE/Forms to keep `dt(·,k)` purely symbolic while enabling Systems/TimeStepping
 * to lower it during assembly.
 *
 * Design boundary:
 * - FE/Forms: detects and represents dt() as a symbolic node; does not choose schemes.
 * - FE/Systems / time integration: selects schemes, computes coefficients, manages history.
 * - FE/Assembly: transports the chosen coefficients/history pointers into kernels.
 */

#include "Core/Types.h"

#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

/**
 * @brief Discrete coefficient stencil for a k-th time derivative
 *
 * The stencil is expressed as a linear combination of the current and
 * historical states:
 *
 *   d^k u / dt^k ≈ a0 * u^n + a1 * u^{n-1} + a2 * u^{n-2}
 *
 * where `u^n` is the "current" state during assembly and `u^{n-1}`, `u^{n-2}`
 * are historical states (if needed).
 *
 * Note: This structure is intentionally minimal and only supports up to two
 * history states because Systems currently exposes `u_prev` and `u_prev2`.
 */
struct TimeDerivativeStencil {
    int order{0};

    // Coefficients for a linear combination of current and historical states:
    //   d^k u / dt^k ≈ sum_{j=0..m} a[j] * u^{n-j}
    // where:
    //   j=0 is the current state u^n,
    //   j=1 is u^{n-1}, etc.
    std::vector<Real> a{};

    [[nodiscard]] int requiredHistoryStates() const noexcept
    {
        if (order <= 0 || a.empty()) {
            return 0;
        }

        int required = 0;
        for (std::size_t i = 1; i < a.size(); ++i) {
            if (a[i] != 0.0) {
                required = static_cast<int>(i);
            }
        }
        return required;
    }

    [[nodiscard]] Real coeff(int history_index) const noexcept
    {
        if (history_index < 0) {
            return 0.0;
        }
        const auto idx = static_cast<std::size_t>(history_index);
        if (idx >= a.size()) {
            return 0.0;
        }
        return a[idx];
    }
};

/**
 * @brief Context object required to assemble forms containing symbolic `dt(·,k)`
 *
 * The presence of a non-null pointer to this object in the AssemblyContext
 * is the sole signal that assembly is occurring under a transient (time
 * integration) context.
 */
struct TimeIntegrationContext {
    std::string integrator_name{"<unset>"};

    std::optional<TimeDerivativeStencil> dt1{};
    std::optional<TimeDerivativeStencil> dt2{};
    // Optional higher-order derivative stencils:
    // - index 0 corresponds to order 3, index 1 -> order 4, etc.
    std::vector<std::optional<TimeDerivativeStencil>> dt_extra{};

    // Optional per-term scaling, used by some time-stepping schemes (e.g. θ-method)
    // to weight dt-containing vs dt-free contributions without re-compiling kernels.
    Real time_derivative_term_weight{1.0};
    Real non_time_derivative_term_weight{1.0};

    // Optional per-derivative scaling (enables splitting dt(·) vs dt(·,2) terms).
    Real dt1_term_weight{1.0};
    Real dt2_term_weight{1.0};
    // Optional per-derivative scaling for dt(·,k), k>=3 (same indexing convention as dt_extra).
    std::vector<Real> dt_extra_term_weight{};

    [[nodiscard]] const TimeDerivativeStencil* stencil(int order) const noexcept
    {
        switch (order) {
            case 1: return dt1 ? &(*dt1) : nullptr;
            case 2: return dt2 ? &(*dt2) : nullptr;
            default:
                break;
        }
        if (order < 3) {
            return nullptr;
        }
        const auto idx = static_cast<std::size_t>(order - 3);
        if (idx >= dt_extra.size()) {
            return nullptr;
        }
        return dt_extra[idx] ? &(*dt_extra[idx]) : nullptr;
    }

    [[nodiscard]] Real derivativeTermWeight(int order) const noexcept
    {
        if (order == 1) return dt1_term_weight;
        if (order == 2) return dt2_term_weight;
        if (order < 3) return Real(1.0);
        const auto idx = static_cast<std::size_t>(order - 3);
        if (idx >= dt_extra_term_weight.size()) {
            return Real(1.0);
        }
        return dt_extra_term_weight[idx];
    }
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_TIME_INTEGRATION_CONTEXT_H
