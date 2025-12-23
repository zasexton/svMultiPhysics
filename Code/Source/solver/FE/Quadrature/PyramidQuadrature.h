/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_PYRAMIDQUADRATURE_H
#define SVMP_FE_QUADRATURE_PYRAMIDQUADRATURE_H

/**
 * @file PyramidQuadrature.h
 * @brief Quadrature rules for reference pyramid elements
 *
 * Reference pyramid: square base on z=0 with coordinates (±1, ±1, 0) and apex
 * at (0, 0, 1). The mapping used here is x = (1 - t) a, y = (1 - t) b, z = t
 * with (a,b) ∈ [-1,1]^2 and t ∈ [0,1].
 */

#include "QuadratureRule.h"
#include "GaussQuadrature.h"

namespace svmp {
namespace FE {
namespace quadrature {

class PyramidQuadrature : public QuadratureRule {
public:
    explicit PyramidQuadrature(int requested_order);

    PyramidQuadrature(int order_ab, int order_t);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_PYRAMIDQUADRATURE_H
