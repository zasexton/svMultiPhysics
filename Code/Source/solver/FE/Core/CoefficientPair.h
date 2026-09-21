#pragma once

#include "Core/Types.h"

#include <cmath>
#include <limits>

namespace svmp::FE {

struct CoefficientPair {
    Real high{0};
    Real low{0};

    [[nodiscard]] long double value() const noexcept
    {
        return low == Real{0} ? static_cast<long double>(high)
                             : static_cast<long double>(high) +
                                   static_cast<long double>(low);
    }
};

// Commit only an exact representation in the supported binary formats.
[[nodiscard]] inline bool encodeCoefficientPair(
    long double value, CoefficientPair& output) noexcept
{
    using Work = long double;
    if constexpr (std::numeric_limits<Real>::radix != 2 ||
                  std::numeric_limits<Work>::radix != 2 ||
                  !std::numeric_limits<Real>::is_iec559 ||
                  !std::numeric_limits<Work>::is_iec559 ||
                  std::numeric_limits<Work>::digits < std::numeric_limits<Real>::digits ||
                  std::numeric_limits<Work>::digits > 2 * std::numeric_limits<Real>::digits ||
                  std::numeric_limits<Work>::max_exponent < std::numeric_limits<Real>::max_exponent ||
                  std::numeric_limits<Work>::min_exponent > std::numeric_limits<Real>::min_exponent) {
        return false;
    }
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<Real>::max()) {
        return false;
    }
    const Real high = static_cast<Real>(value);
    if (!std::isfinite(high) || (value != Work{0} && high == Real{0})) return false;
    const Work tail = value - static_cast<Work>(high);
    const Real low = static_cast<Real>(tail);
    const CoefficientPair candidate{high, low};
    if (!std::isfinite(low) || static_cast<Work>(low) != tail ||
        candidate.value() != value ||
        (value == Work{0} && std::signbit(candidate.value()) != std::signbit(value))) {
        return false;
    }
    output = candidate;
    return true;
}

[[nodiscard]] inline bool isCanonicalCoefficientPair(
    const CoefficientPair& pair) noexcept
{
    if (!std::isfinite(pair.high) || !std::isfinite(pair.low)) return false;
    CoefficientPair canonical;
    return encodeCoefficientPair(pair.value(), canonical) &&
           canonical.high == pair.high && canonical.low == pair.low;
}

} // namespace svmp::FE
