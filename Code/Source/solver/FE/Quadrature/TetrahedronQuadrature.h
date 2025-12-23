/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_TETRAHEDRONQUADRATURE_H
#define SVMP_FE_QUADRATURE_TETRAHEDRONQUADRATURE_H

/**
 * @file TetrahedronQuadrature.h
 * @brief Quadrature rules for the reference tetrahedron
 *
 * Implements tetrahedral quadrature via a three-parameter Duffy transform from
 * [0,1]^3 to the reference simplex with vertices (0,0,0), (1,0,0), (0,1,0),
 * (0,0,1). The construction delivers arbitrary-order rules without external
 * tables while preserving positive weights and symmetry.
 */

#include "QuadratureRule.h"
#include "GaussQuadrature.h"

namespace svmp {
namespace FE {
namespace quadrature {

class TetrahedronQuadrature : public QuadratureRule {
public:
    /**
     * @brief Construct a tetrahedron quadrature of at least the requested order
     * @param requested_order Desired total-degree polynomial exactness (order >= 1)
     *
     * The underlying construction uses an n×n×n tensor-product Gauss rule on
     * [0,1]^3 composed with a Duffy transform. With this mapping, the guaranteed
     * total-degree polynomial exactness is:
     *
     *   order() = 2*n - 3
     *
     * The implementation chooses the smallest n such that `order() >= requested_order`.
     */
    explicit TetrahedronQuadrature(int requested_order);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_TETRAHEDRONQUADRATURE_H
