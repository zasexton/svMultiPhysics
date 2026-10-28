/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CORE_NUMERICVALIDATION_H
#define SVMP_FE_CORE_NUMERICVALIDATION_H

#include "Core/Types.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace svmp::FE::numeric {

/// The standard floating-point finiteness check, without a sign restriction.
[[nodiscard]] inline bool isFinite(Real value) noexcept { return std::isfinite(value); }

/// Sign checks alone: positive infinity passes; NaN fails both comparisons.
/// Call isFinite separately when the model also requires a finite value.
[[nodiscard]] inline bool isNonnegative(Real value) noexcept { return value >= Real{0.0}; }
[[nodiscard]] inline bool isPositive(Real value) noexcept { return value > Real{0.0}; }

/// Checks a raw value with a predicate accepting Real. The caller owns the
/// complete diagnostic and the order of independent checks. The message is
/// copied into the exception on failure; it is never retained as a view.
template <class Predicate>
inline void validate(Real value, Predicate&& predicate, std::string_view message)
{
    if (!std::forward<Predicate>(predicate)(value)) {
        throw std::invalid_argument(std::string(message));
    }
}

} // namespace svmp::FE::numeric

#endif // SVMP_FE_CORE_NUMERICVALIDATION_H
