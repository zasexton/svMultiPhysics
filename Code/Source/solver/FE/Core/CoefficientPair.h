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
//
// The work format is the target's long double: x87 extended precision (64
// digits) on x86-64, IEEE binary128 (113 digits) on AArch64 Linux.  With at
// most twice the digits of Real, every value in range has a Real tail and is
// encodable.  With more, a value is encodable only when its tail is a Real,
// and the exactness checks below reject the others, so a pair is accepted on
// either target exactly when it is the canonical split of a long double.
[[nodiscard]] inline bool encodeCoefficientPair(
    long double value, CoefficientPair& output) noexcept
{
    using Work = long double;
    if constexpr (std::numeric_limits<Real>::radix != 2 ||
                  std::numeric_limits<Work>::radix != 2 ||
                  !std::numeric_limits<Real>::is_iec559 ||
                  !std::numeric_limits<Work>::is_iec559 ||
                  std::numeric_limits<Work>::digits < std::numeric_limits<Real>::digits ||
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
