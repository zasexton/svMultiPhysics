// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SVMP_FE_GAUSS_QUADRATURE_H
#define SVMP_FE_GAUSS_QUADRATURE_H

/**
 * @file GaussQuadrature.h
 * @brief Bounded Gauss-Legendre quadrature generation on the reference line.
 * @ingroup FE_Quadrature
 */

#include "FE/Quadrature/QuadratureRule.h"

namespace svmp::FE::quadrature {

/** @addtogroup FE_Quadrature
 * @{
 */

/**
 * @brief Return the largest supported Gauss-Legendre point count.
 * @details The 128-point ceiling is a project support bound that limits
 * generator work and downstream product-rule growth while providing line
 * exactness through degree 255.
 * @return The inclusive point-count limit, 128.
 */
[[nodiscard]] constexpr int max_gauss_legendre_points() noexcept
{
    return 128;
}

/**
 * @brief Generate an @p num_points Gauss-Legendre rule on @f$[-1,1]@f$.
 *
 * @details The returned line rule contains the roots of @f$P_n@f$ in strictly
 * increasing order, where @f$n@f$ is @p num_points. Every point lies strictly
 * inside @f$(-1,1)@f$, so neither endpoint is included. The rule has polynomial
 * exactness @f$2n-1@f$ and positive weights aligned with its points.
 *
 * @param num_points Number of quadrature points; must be in
 *        @f$[1,\texttt{max\_gauss\_legendre\_points()}]@f$.
 * @return A complete QuadratureRule value for CellFamily::Line.
 * @throws InvalidArgumentException If @p num_points is outside the supported
 *         range.
 * @throws ConvergenceException If root refinement or final numerical
 *         validation fails.
 */
[[nodiscard]] QuadratureRule
make_gauss_legendre_rule(int num_points);

/** @} */

} // namespace svmp::FE::quadrature

#endif // SVMP_FE_GAUSS_QUADRATURE_H
