/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_GAUSSLOBATTOQUADRATURE_H
#define SVMP_FE_QUADRATURE_GAUSSLOBATTOQUADRATURE_H

/**
 * @file GaussLobattoQuadrature.h
 * @brief Gauss-Lobatto rules including interval endpoints
 *
 * Gauss-Lobatto rules are optimal quadrature formulas that include the interval
 * endpoints, making them attractive for spectral elements and mortar coupling.
 * An n-point Gauss-Lobatto rule exactly integrates polynomials of degree
 * 2n-3 on [-1, 1].
 */

#include "QuadratureRule.h"
#include <utility>

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Gauss-Lobatto quadrature on [-1, 1] including endpoints
 */
class GaussLobattoQuadrature1D : public QuadratureRule {
public:
    explicit GaussLobattoQuadrature1D(int num_points);

    static std::pair<std::vector<Real>, std::vector<Real>>
    generate_raw(int num_points);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_GAUSSLOBATTOQUADRATURE_H
