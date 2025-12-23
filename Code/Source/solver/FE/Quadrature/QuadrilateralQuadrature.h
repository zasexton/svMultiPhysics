/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_QUADRILATERALQUADRATURE_H
#define SVMP_FE_QUADRATURE_QUADRILATERALQUADRATURE_H

/**
 * @file QuadrilateralQuadrature.h
 * @brief Tensor-product quadrature for reference quadrilaterals
 */

#include "QuadratureRule.h"
#include "GaussQuadrature.h"
#include "GaussLobattoQuadrature.h"

namespace svmp {
namespace FE {
namespace quadrature {

class QuadrilateralQuadrature : public QuadratureRule {
public:
    explicit QuadrilateralQuadrature(int order,
                                     QuadratureType type = QuadratureType::GaussLegendre);

    QuadrilateralQuadrature(int order_x, int order_y,
                            QuadratureType type = QuadratureType::GaussLegendre);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_QUADRILATERALQUADRATURE_H
