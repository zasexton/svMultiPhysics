/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_HEXAHEDRONQUADRATURE_H
#define SVMP_FE_QUADRATURE_HEXAHEDRONQUADRATURE_H

/**
 * @file HexahedronQuadrature.h
 * @brief Tensor-product quadrature for reference hexahedra
 */

#include "QuadratureRule.h"
#include "GaussQuadrature.h"
#include "GaussLobattoQuadrature.h"

namespace svmp {
namespace FE {
namespace quadrature {

class HexahedronQuadrature : public QuadratureRule {
public:
    explicit HexahedronQuadrature(int order,
                                  QuadratureType type = QuadratureType::GaussLegendre);

    HexahedronQuadrature(int order_x, int order_y, int order_z,
                         QuadratureType type = QuadratureType::GaussLegendre);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_HEXAHEDRONQUADRATURE_H
