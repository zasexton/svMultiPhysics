/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "CurveGeometry.h"

namespace svmp {
namespace FE {
namespace geometry {

CurveData CurveGeometry::evaluate(const GeometryMapping& mapping,
                                  const math::Vector<Real, 3>& xi) {
    if (mapping.dimension() != 1) {
        throw FEException("CurveGeometry expects a 1D reference curve",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const auto J = mapping.jacobian(xi);
    CurveData data;
    data.tangent = math::Vector<Real, 3>{J(0, 0), J(1, 0), J(2, 0)};
    data.line_element = data.tangent.norm();
    if (data.line_element > Real(0)) {
        data.unit_tangent = data.tangent / data.line_element;
    }

    // For 1D mappings, GeometryMapping implementations return a full 3x3 frame:
    // col(1) and col(2) complete the orthonormal complement to the tangent.
    data.normal_1 = math::Vector<Real, 3>{J(0, 1), J(1, 1), J(2, 1)};
    data.normal_2 = math::Vector<Real, 3>{J(0, 2), J(1, 2), J(2, 2)};
    return data;
}

} // namespace geometry
} // namespace FE
} // namespace svmp

