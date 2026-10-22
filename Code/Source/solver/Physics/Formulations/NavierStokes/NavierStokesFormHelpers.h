/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_FORM_HELPERS_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_FORM_HELPERS_H

/**
 * @file NavierStokesFormHelpers.h
 * @brief Small form-building helpers of the Navier-Stokes formulation
 */
#include "FE/Core/Types.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/FormExpr.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

/// a + b, or b when a is empty.
[[nodiscard]] inline FE::forms::FormExpr sumOf(const FE::forms::FormExpr& a, const FE::forms::FormExpr& b)
{
    return a.isValid() ? a + b : b;
}

/// The first @p n components of @p values as a constant vector.
[[nodiscard]] inline FE::forms::FormExpr constantVector(const std::array<FE::Real, 3>& values, int n)
{
    std::vector<FE::forms::FormExpr> c;
    for (int d = 0; d < n; ++d) {
        c.push_back(FE::forms::FormExpr::constant(values[static_cast<std::size_t>(d)]));
    }
    return FE::forms::FormExpr::asVector(std::move(c));
}

/// The first @p n components of @p values; component d is named "<prefix><d>".
[[nodiscard]] inline FE::forms::FormExpr scalarVector(const std::array<FE::forms::bc::ScalarValue, 3>& values,
                                                      int n, std::string_view prefix)
{
    std::vector<FE::forms::FormExpr> c;
    for (int d = 0; d < n; ++d) {
        c.push_back(FE::forms::bc::toScalarExpr(values[static_cast<std::size_t>(d)],
                                                std::string(prefix) + std::to_string(d)));
    }
    return FE::forms::FormExpr::asVector(std::move(c));
}

template <class T>
void appendUnique(std::vector<T>& values, const T& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_FORM_HELPERS_H
