/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "GeometryQuadrature.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {

GeometryQuadratureData GeometryQuadrature::evaluate(const GeometryMapping& mapping,
                                                    const quadrature::QuadratureRule& quad) {
    GeometryQuadratureData data;
    const auto& pts = quad.points();
    data.physical_points.resize(pts.size());
    data.scaled_weights.resize(pts.size());
    data.detJ.resize(pts.size());

    for (std::size_t i = 0; i < pts.size(); ++i) {
        data.physical_points[i] = mapping.map_to_physical(pts[i]);
        data.detJ[i] = mapping.jacobian_determinant(pts[i]);
        data.scaled_weights[i] = quad.weight(i) * std::abs(data.detJ[i]);
    }
    return data;
}

} // namespace geometry
} // namespace FE
} // namespace svmp
