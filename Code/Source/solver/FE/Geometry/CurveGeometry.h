/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_CURVEGEOMETRY_H
#define SVMP_FE_GEOMETRY_CURVEGEOMETRY_H

/**
 * @file CurveGeometry.h
 * @brief Geometric quantities for 1D reference curves embedded in 3D.
 */

#include "GeometryMapping.h"
#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace geometry {

struct CurveData {
    math::Vector<Real, 3> tangent{};
    math::Vector<Real, 3> unit_tangent{};
    math::Vector<Real, 3> normal_1{};
    math::Vector<Real, 3> normal_2{};
    Real line_element{0};
};

class CurveGeometry {
public:
    /// Evaluate tangent and line element for a 1D reference curve (edge) in 3D.
    static CurveData evaluate(const GeometryMapping& mapping,
                              const math::Vector<Real, 3>& xi);
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_CURVEGEOMETRY_H

