/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SurfaceGeometry.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {

SurfaceData SurfaceGeometry::evaluate(const GeometryMapping& mapping,
                                      const math::Vector<Real, 3>& xi) {
    if (mapping.dimension() != 2) {
        throw FEException("SurfaceGeometry expects a 2D reference surface",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    auto J = mapping.jacobian(xi);
    SurfaceData data;
    // Tangents are columns of J for parametric directions
    data.tangent_u = math::Vector<Real, 3>{J(0,0), J(1,0), J(2,0)};
    data.tangent_v = math::Vector<Real, 3>{J(0,1), J(1,1), J(2,1)};
    data.normal = data.tangent_u.cross(data.tangent_v);
    data.area_element = data.normal.norm();
    if (data.area_element > Real(0)) {
        data.normal /= data.area_element; // unit normal
    }
    return data;
}

} // namespace geometry
} // namespace FE
} // namespace svmp

