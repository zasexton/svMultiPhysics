/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_WEDGEQUADRATURE_H
#define SVMP_FE_QUADRATURE_WEDGEQUADRATURE_H

/**
 * @file WedgeQuadrature.h
 * @brief Quadrature rules for wedge/prism reference elements
 */

#include "QuadratureRule.h"
#include "TriangleQuadrature.h"
#include "GaussQuadrature.h"
#include "GaussLobattoQuadrature.h"

namespace svmp {
namespace FE {
namespace quadrature {

class WedgeQuadrature : public QuadratureRule {
public:
    explicit WedgeQuadrature(int order,
                             QuadratureType type = QuadratureType::GaussLegendre);

    WedgeQuadrature(int triangle_order, int line_order,
                    QuadratureType type = QuadratureType::GaussLegendre);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_WEDGEQUADRATURE_H
