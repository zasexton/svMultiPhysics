/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/ElementValidator.h"

namespace svmp {
namespace FE {
namespace elements {

using svmp::FE::geometry::GeometryValidator;

ElementQuality ElementValidator::validate(const Element& element,
                                          const geometry::GeometryMapping& mapping) {
    ElementQuality out;

    auto quad = element.quadrature();
    if (!quad) {
        out.positive_jacobian = false;
        out.min_detJ = Real(0);
        out.max_condition_number = Real(0);
        return out;
    }

    const auto& pts = quad->points();
    if (pts.empty()) {
        out.min_detJ = Real(0);
        out.max_condition_number = Real(0);
        out.positive_jacobian = false;
        return out;
    }

    out.min_detJ = std::numeric_limits<Real>::max();
    out.max_condition_number = Real(0);
    out.positive_jacobian = true;

    for (std::size_t i = 0; i < pts.size(); ++i) {
        auto q = GeometryValidator::evaluate(mapping, pts[i]);
        out.min_detJ = std::min(out.min_detJ, q.detJ);
        out.max_condition_number = std::max(out.max_condition_number, q.condition_number);
        if (!q.positive_jacobian) {
            out.positive_jacobian = false;
        }
    }

    return out;
}

} // namespace elements
} // namespace FE
} // namespace svmp

