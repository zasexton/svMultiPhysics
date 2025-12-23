/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_GEOMETRYQUADRATURE_H
#define SVMP_FE_GEOMETRY_GEOMETRYQUADRATURE_H

/**
 * @file GeometryQuadrature.h
 * @brief Mapping-aware quadrature utilities (mapped points and scaled weights)
 */

#include "GeometryMapping.h"
#include "Quadrature/QuadratureRule.h"
#include <vector>

namespace svmp {
namespace FE {
namespace geometry {

struct GeometryQuadratureData {
    std::vector<math::Vector<Real, 3>> physical_points;
    std::vector<Real> scaled_weights;
    std::vector<Real> detJ;
};

class GeometryQuadrature {
public:
    static GeometryQuadratureData evaluate(const GeometryMapping& mapping,
                                           const quadrature::QuadratureRule& quad);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_GEOMETRYQUADRATURE_H
