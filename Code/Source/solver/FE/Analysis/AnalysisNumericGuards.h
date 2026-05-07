/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_ANALYSIS_NUMERIC_GUARDS_H
#define SVMP_FE_ANALYSIS_ANALYSIS_NUMERIC_GUARDS_H

#include "Core/Types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace svmp {
namespace FE {
namespace analysis {
namespace numeric {

[[nodiscard]] inline bool finite(Real value) noexcept
{
    if constexpr (std::numeric_limits<Real>::is_iec559 &&
                  sizeof(Real) == sizeof(std::uint64_t)) {
        std::uint64_t bits{};
        std::memcpy(&bits, &value, sizeof(bits));
        constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ULL;
        return (bits & exponent_mask) != exponent_mask;
    } else if constexpr (std::numeric_limits<Real>::is_iec559 &&
                         sizeof(Real) == sizeof(std::uint32_t)) {
        std::uint32_t bits{};
        std::memcpy(&bits, &value, sizeof(bits));
        constexpr std::uint32_t exponent_mask = 0x7f800000U;
        return (bits & exponent_mask) != exponent_mask;
    } else {
        using std::isfinite;
        return isfinite(value);
    }
}

[[nodiscard]] inline bool finiteNonnegative(Real value) noexcept
{
    return finite(value) && value >= Real{};
}

[[nodiscard]] inline bool finitePositive(Real value) noexcept
{
    return finite(value) && value > Real{};
}

[[nodiscard]] inline bool finiteTolerance(Real value) noexcept
{
    return finiteNonnegative(value);
}

[[nodiscard]] inline bool finiteDeclaredTolerance(Real value) noexcept
{
    return finitePositive(value);
}

[[nodiscard]] inline bool finiteResidual(Real value) noexcept
{
    return finite(value);
}

[[nodiscard]] inline bool finiteAbsWithin(Real value,
                                          Real tolerance) noexcept
{
    return finiteResidual(value) &&
           finiteTolerance(tolerance) &&
           std::abs(value) <= tolerance;
}

[[nodiscard]] inline bool finiteAbsResidualTriple(Real a,
                                                 Real b,
                                                 Real c) noexcept
{
    return finiteResidual(a) && finiteResidual(b) && finiteResidual(c);
}

[[nodiscard]] inline Real maxAbsTriple(Real a, Real b, Real c) noexcept
{
    const Real absa = std::abs(a);
    const Real absb = std::abs(b);
    const Real absc = std::abs(c);
    return std::max(absa, std::max(absb, absc));
}

[[nodiscard]] inline bool finiteOrdered(Real lower, Real upper) noexcept
{
    return finite(lower) && finite(upper) && lower <= upper;
}

[[nodiscard]] inline bool finitePositiveOrdered(Real lower,
                                                Real upper) noexcept
{
    return finitePositive(lower) && finite(upper) && lower <= upper;
}

[[nodiscard]] inline bool finiteNonnegativeBounded(Real value,
                                                   Real upper) noexcept
{
    return finiteNonnegative(value) && finite(upper) && value <= upper;
}

} // namespace numeric
} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_ANALYSIS_NUMERIC_GUARDS_H
