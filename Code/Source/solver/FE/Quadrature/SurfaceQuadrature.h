/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_SURFACEQUADRATURE_H
#define SVMP_FE_QUADRATURE_SURFACEQUADRATURE_H

/**
 * @file SurfaceQuadrature.h
 * @brief Face and edge quadrature helpers for reference elements
 */

#include "QuadratureRule.h"
#include "GaussQuadrature.h"
#include "GaussLobattoQuadrature.h"
#include "TriangleQuadrature.h"
#include "QuadrilateralQuadrature.h"
#include <memory>

namespace svmp {
namespace FE {
namespace quadrature {

class SurfaceQuadrature {
public:
    /**
     * @brief Create a face quadrature rule for a parent element
     * @param elem_type FE element type (from FE/Core/Types.h)
     * @param face_id Zero-based face index (orientation assumed canonical)
     * @param order Requested polynomial order
     * @param type Quadrature rule type
     */
    static std::unique_ptr<QuadratureRule> face_rule(ElementType elem_type,
                                                     int face_id,
                                                     int order,
                                                     QuadratureType type = QuadratureType::GaussLegendre);

    /**
     * @brief Create an edge quadrature rule for a surface element
     * @param face_family Cell family of the face (Line, Quad, Triangle)
     * @param edge_id Zero-based edge index (canonical orientation)
     * @param order Requested order
     */
    static std::unique_ptr<QuadratureRule> edge_rule(svmp::CellFamily face_family,
                                                     int edge_id,
                                                     int order,
                                                     QuadratureType type = QuadratureType::GaussLegendre);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_SURFACEQUADRATURE_H
