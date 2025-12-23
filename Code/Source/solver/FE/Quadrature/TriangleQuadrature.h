/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_TRIANGLEQUADRATURE_H
#define SVMP_FE_QUADRATURE_TRIANGLEQUADRATURE_H

/**
 * @file TriangleQuadrature.h
 * @brief Symmetric quadrature rules for reference triangles
 *
 * Implements triangle quadrature by transforming tensor-product Gauss rules on
 * [0,1]^2 via a Duffy map to the reference simplex with vertices
 * (0,0), (1,0), (0,1). This approach provides arbitrary-order rules without
 * relying on large tabulated datasets while preserving symmetry.
 */

#include "QuadratureRule.h"
#include "GaussQuadrature.h"

namespace svmp {
namespace FE {
namespace quadrature {

class TriangleQuadrature : public QuadratureRule {
public:
    /**
     * @brief Construct a triangle quadrature of at least the requested order
     * @param requested_order Desired polynomial exactness (order >= 1)
     *
     * The underlying construction uses an n×n tensor-product Gauss rule on
     * [0,1]^2 composed with a Duffy transform. With this mapping, the guaranteed
     * total-degree polynomial exactness is:
     *
     *   order() = 2*n - 2
     *
     * The implementation chooses the smallest n such that `order() >= requested_order`.
     */
    explicit TriangleQuadrature(int requested_order);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_TRIANGLEQUADRATURE_H
