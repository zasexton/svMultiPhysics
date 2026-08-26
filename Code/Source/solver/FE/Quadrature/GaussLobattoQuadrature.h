// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SVMP_FE_GAUSS_LOBATTO_QUADRATURE_H
#define SVMP_FE_GAUSS_LOBATTO_QUADRATURE_H

/**
 * @file GaussLobattoQuadrature.h
 * @brief Bounded Gauss-Lobatto-Legendre quadrature generation on the reference line.
 * @ingroup FE_Quadrature
 */

#include "FE/Quadrature/QuadratureRule.h"

namespace svmp::FE::quadrature {

/** @addtogroup FE_Quadrature
 * @{
 */

/**
 * @brief Return the largest supported Gauss-Lobatto-Legendre point count.
 * @details The 128 total points include both endpoints. This project support
 * bound limits generator work and downstream product-rule growth while
 * providing endpoint-inclusive line exactness through degree 253; it is not a
 * mathematical or convergence limit.
 * @return The inclusive point-count limit, 128.
 */
[[nodiscard]] constexpr int max_gauss_lobatto_points() noexcept
{
    return 128;
}

/**
 * @brief Generate an @p num_points Gauss-Lobatto-Legendre rule on
 *        @f$[-1,1]@f$.
 *
 * @details The returned line rule has exactly @f$-1@f$ as its first point and
 * exactly @f$+1@f$ as its last point. When present, its @f$n-2@f$ interior
 * points are the roots of @f$P'_{n-1}@f$, where @f$n@f$ is @p num_points.
 * Points are strictly increasing, weights are positive and aligned with their
 * points, and the rule has polynomial exactness @f$2n-3@f$.
 *
 * @param num_points Signed number of quadrature points; must be in the
 *        inclusive range
 *        @f$[2,\texttt{max\_gauss\_lobatto\_points()}]@f$.
 * @return A complete QuadratureRule value for CellFamily::Line.
 * @throws InvalidArgumentException If @p num_points is outside the supported
 *         range.
 * @throws ConvergenceException If root refinement or final numerical
 *         validation fails.
 */
[[nodiscard]] QuadratureRule
make_gauss_lobatto_rule(int num_points);

/** @} */

} // namespace svmp::FE::quadrature

#endif // SVMP_FE_GAUSS_LOBATTO_QUADRATURE_H
