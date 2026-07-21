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
#include <algorithm>
#include <cmath>
#include <limits>

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
 * @brief Overflow/underflow-resistant Euclidean norm for a three-vector.
 */
[[nodiscard]] inline Real stable_vector_norm(
    const math::Vector<Real, 3>& value) noexcept
{
    const Real scale = std::max(
        {std::abs(value[0]), std::abs(value[1]), std::abs(value[2])});
    if (scale == Real(0)) {
        return Real(0);
    }
    if (!std::isfinite(scale)) {
        return std::numeric_limits<Real>::infinity();
    }
    const math::Vector<Real, 3> normalized = value / scale;
    return scale * std::sqrt(normalized.dot(normalized));
}

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
    const Real tnorm = stable_vector_norm(tangent);
    if (!std::isfinite(tnorm) || tnorm <= Real(0)) {
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
    Real n1_norm = stable_vector_norm(n1);
    if (n1_norm < tol) {
        a = math::Vector<Real, 3>{Real(0), Real(0), Real(1)};
        n1 = t_unit.cross(a);
        n1_norm = stable_vector_norm(n1);
    }
    if (!std::isfinite(n1_norm) || n1_norm < tol) {
        n1 = math::Vector<Real, 3>{Real(0), Real(1), Real(0)};
        n2 = math::Vector<Real, 3>{Real(0), Real(0), Real(1)};
        return;
    }

    n1 /= n1_norm;
    n2 = t_unit.cross(n1);
}

/**
 * @brief Complete two surface tangents with a unit normal.
 *
 * Degeneracy is decided from the sine of the angle between separately
 * normalized tangents, so the decision is independent of physical units and
 * of positive scaling of either tangent column.
 */
[[nodiscard]] inline bool complete_surface_frame(
    const math::Vector<Real, 3>& tangent_u,
    const math::Vector<Real, 3>& tangent_v,
    math::Vector<Real, 3>& normal,
    Real tol = kDegenerateTol) noexcept
{
    const Real norm_u = stable_vector_norm(tangent_u);
    const Real norm_v = stable_vector_norm(tangent_v);
    if (!std::isfinite(norm_u) || !std::isfinite(norm_v) ||
        norm_u <= Real(0) || norm_v <= Real(0)) {
        normal = {};
        return false;
    }

    const math::Vector<Real, 3> unit_u = tangent_u / norm_u;
    const math::Vector<Real, 3> unit_v = tangent_v / norm_v;
    const auto cross = unit_u.cross(unit_v);
    const Real sine = stable_vector_norm(cross);
    if (!std::isfinite(sine) || sine <= tol) {
        normal = {};
        return false;
    }
    normal = cross / sine;
    return std::isfinite(normal[0]) && std::isfinite(normal[1]) &&
           std::isfinite(normal[2]);
}

} // namespace detail
} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_GEOMETRYFRAMEUTILS_H
