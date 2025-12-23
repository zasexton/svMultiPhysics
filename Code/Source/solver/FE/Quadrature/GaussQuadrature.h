/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_GAUSSQUADRATURE_H
#define SVMP_FE_QUADRATURE_GAUSSQUADRATURE_H

/**
 * @file GaussQuadrature.h
 * @brief Gauss-Legendre quadrature rules for 1D reference elements
 *
 * Provides optimal (n-point) Gauss quadrature on the interval [-1, 1],
 * achieving exact integration of polynomials up to degree 2n-1. Rules are
 * computed on demand using stable Newton refinement of Legendre polynomial
 * roots with symmetry exploitation.
 */

#include "QuadratureRule.h"
#include <utility>

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Gauss-Legendre quadrature on [-1, 1]
 *
 * The rule is generated at construction time; points and weights are stored in
 * contiguous arrays suitable for cache-friendly access during assembly.
 */
class GaussQuadrature1D : public QuadratureRule {
public:
    /// Construct rule with @p num_points Gauss nodes (num_points >= 1)
    explicit GaussQuadrature1D(int num_points);

    /// Expose underlying abscissae/weights for reuse by other rules
    static std::pair<std::vector<Real>, std::vector<Real>>
    generate_raw(int num_points);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_GAUSSQUADRATURE_H
