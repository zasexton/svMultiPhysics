/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_GEOMETRYFRAMEUTILS_H
#define SVMP_FE_GEOMETRY_GEOMETRYFRAMEUTILS_H

/**
 * @file GeometryFrameUtils.h
 * @brief Internal utilities for completing Jacobian frames in embedded geometry.
 */

#include "Core/Types.h"
#include "Math/Vector.h"
#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {
namespace detail {

#ifndef FE_GEOMETRY_DEGENERATE_TOL
constexpr Real kDegenerateTol = Real(1e-14);
#else
constexpr Real kDegenerateTol = Real(FE_GEOMETRY_DEGENERATE_TOL);
#endif

/**
 * @brief Build an orthonormal complement for a 3D curve tangent.
 *
 * The returned (n1,n2) form a right-handed frame with the unit tangent:
 *   { t_unit, n1, n2 }.
 *
 * For nearly-degenerate tangents, falls back to a fixed coordinate frame.
 */
inline void complete_curve_frame(const math::Vector<Real, 3>& tangent,
                                 math::Vector<Real, 3>& n1,
                                 math::Vector<Real, 3>& n2,
                                 Real tol = kDegenerateTol) {
    const Real tnorm = tangent.norm();
    if (tnorm < tol) {
        n1 = math::Vector<Real, 3>{Real(0), Real(1), Real(0)};
        n2 = math::Vector<Real, 3>{Real(0), Real(0), Real(1)};
        return;
    }

    const math::Vector<Real, 3> t_unit = tangent / tnorm;

    math::Vector<Real, 3> a{Real(1), Real(0), Real(0)};
    if (std::abs(t_unit[0]) > Real(0.9)) {
        a = math::Vector<Real, 3>{Real(0), Real(1), Real(0)};
        if (std::abs(t_unit[1]) > Real(0.9)) {
            a = math::Vector<Real, 3>{Real(0), Real(0), Real(1)};
        }
    }

    n1 = t_unit.cross(a);
    Real n1_norm = n1.norm();
    if (n1_norm < tol) {
        a = math::Vector<Real, 3>{Real(0), Real(0), Real(1)};
        n1 = t_unit.cross(a);
        n1_norm = n1.norm();
    }
    if (n1_norm < tol) {
        n1 = math::Vector<Real, 3>{Real(0), Real(1), Real(0)};
        n2 = math::Vector<Real, 3>{Real(0), Real(0), Real(1)};
        return;
    }

    n1 /= n1_norm;
    n2 = t_unit.cross(n1);
}

} // namespace detail
} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_GEOMETRYFRAMEUTILS_H

