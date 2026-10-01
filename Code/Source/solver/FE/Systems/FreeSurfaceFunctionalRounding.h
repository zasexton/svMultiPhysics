/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FREESURFACEFUNCTIONALROUNDING_H
#define SVMP_FE_SYSTEMS_FREESURFACEFUNCTIONALROUNDING_H

/**
 * @file FreeSurfaceFunctionalRounding.h
 * @brief Rounding bound for comparing two floating-point sums of the same terms.
 *
 * Accepted free-surface functional records hold some quantities summed in two
 * ways, for example the liquid volume summed per retained rule and summed over
 * every quadrature weight of those rules.  The two sums agree in exact
 * arithmetic and differ in floating point by rounding that grows with the
 * number of terms, so a fixed number of units in the last place cannot
 * separate rounding from a genuine inconsistency once the rules hold enough
 * points.
 *
 * The bound below follows from the standard model of floating-point
 * arithmetic, fl(x + y) = (x + y)(1 + d) with |d| <= u, where u is the unit
 * roundoff of round-to-nearest (u = 2^-53 for binary64, half the machine
 * epsilon).  Any summation that repeatedly adds two partial sums (recursive
 * summation, per-rule sums added to a total, MPI reductions) passes each term
 * through at most n - 1 additions that combine it with other terms; adding a
 * partial sum that is exactly zero, such as that of an MPI rank without terms,
 * is exact.  The computed sum s of the exact sum S of n terms therefore obeys
 *
 *     |s - S| <= gamma_{n-1} * sum_i |x_i|,    gamma_k = k u / (1 - k u)
 *
 * (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., sec. 4.2),
 * and two sums a and b of the same terms, with n_a and n_b terms, obey
 *
 *     |a - b| <= (gamma_{n_a - 1} + gamma_{n_b - 1}) * sum_i |x_i|.
 *
 * For nonnegative terms sum_i |x_i| = S <= s / (1 - gamma_{n-1}), so either
 * computed sum bounds the absolute sum.  The bound has no free parameter.  The
 * tolerance itself is evaluated in floating point; its relative rounding of a
 * few u is second order next to gamma.
 */

#include "Core/Types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace svmp {
namespace FE {
namespace systems {

/// Unit roundoff u of round-to-nearest Real arithmetic (half the machine epsilon).
inline constexpr Real kFreeSurfaceFunctionalUnitRoundoff =
    std::numeric_limits<Real>::epsilon() / Real{2.0};

/**
 * gamma_{n-1} for a sum of n terms: zero for at most one term (such a sum is
 * exact) and +infinity once (n - 1) u reaches one, where the model gives no
 * bound.
 */
[[nodiscard]] inline Real freeSurfaceSummationErrorFactor(
    std::uint64_t term_count) noexcept
{
    if (term_count <= 1u) {
        return Real{0.0};
    }
    const Real ku = static_cast<Real>(term_count - 1u) *
                    kFreeSurfaceFunctionalUnitRoundoff;
    if (!(ku < Real{1.0})) {
        return std::numeric_limits<Real>::infinity();
    }
    return ku / (Real{1.0} - ku);
}

/**
 * Largest |a - b| that rounding allows when a sums a_terms terms and b sums
 * b_terms terms of the same values, given absolute_sum >= sum_i |x_i|.
 */
[[nodiscard]] inline Real freeSurfaceSummationTolerance(
    std::uint64_t a_terms,
    std::uint64_t b_terms,
    Real absolute_sum) noexcept
{
    if (absolute_sum == Real{0.0}) {
        return Real{0.0};
    }
    return (freeSurfaceSummationErrorFactor(a_terms) +
            freeSurfaceSummationErrorFactor(b_terms)) *
           absolute_sum;
}

/// True when two sums of the same terms agree within the rounding bound.
[[nodiscard]] inline bool freeSurfaceSummationsAgree(
    Real a,
    std::uint64_t a_terms,
    Real b,
    std::uint64_t b_terms,
    Real absolute_sum) noexcept
{
    return std::abs(a - b) <=
           freeSurfaceSummationTolerance(a_terms, b_terms, absolute_sum);
}

/**
 * Bound on sum_i |x_i| for nonnegative terms from the two computed sums:
 * each computed sum s of n terms gives S <= s / (1 - gamma_{n-1}); the larger
 * of the two bounds is returned.
 */
[[nodiscard]] inline Real freeSurfaceNonnegativeSummationAbsoluteBound(
    Real a,
    std::uint64_t a_terms,
    Real b,
    std::uint64_t b_terms) noexcept
{
    const auto bound = [](Real sum, std::uint64_t terms) {
        const Real factor = freeSurfaceSummationErrorFactor(terms);
        if (!(factor < Real{1.0})) {
            return std::numeric_limits<Real>::infinity();
        }
        return std::abs(sum) / (Real{1.0} - factor);
    };
    return std::max(bound(a, a_terms), bound(b, b_terms));
}

/// True when two sums of the same nonnegative terms agree within the bound.
[[nodiscard]] inline bool freeSurfaceNonnegativeSummationsAgree(
    Real a,
    std::uint64_t a_terms,
    Real b,
    std::uint64_t b_terms) noexcept
{
    return freeSurfaceSummationsAgree(
        a,
        a_terms,
        b,
        b_terms,
        freeSurfaceNonnegativeSummationAbsoluteBound(
            a, a_terms, b, b_terms));
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FREESURFACEFUNCTIONALROUNDING_H
