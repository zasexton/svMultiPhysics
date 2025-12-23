/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_SURFACEGEOMETRY_H
#define SVMP_FE_GEOMETRY_SURFACEGEOMETRY_H

/**
 * @file SurfaceGeometry.h
 * @brief Surface geometric quantities (normals, tangents, area elements)
 */

#include "GeometryMapping.h"
#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace geometry {

struct SurfaceData {
    math::Vector<Real, 3> tangent_u{};
    math::Vector<Real, 3> tangent_v{};
    math::Vector<Real, 3> normal{};
    Real area_element{0};
};

class SurfaceGeometry {
public:
    /// Evaluate surface tangents and normal for a 2D reference surface in 3D
    static SurfaceData evaluate(const GeometryMapping& mapping,
                                const math::Vector<Real, 3>& xi);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_SURFACEGEOMETRY_H
