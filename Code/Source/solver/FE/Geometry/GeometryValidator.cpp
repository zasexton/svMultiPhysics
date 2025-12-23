/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "GeometryValidator.h"
#include "MetricTensor.h"
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace geometry {

GeometryQuality GeometryValidator::evaluate(const GeometryMapping& mapping,
                                            const math::Vector<Real, 3>& xi) {
    const Real det = mapping.jacobian_determinant(xi);
    const int dim = mapping.dimension();

    GeometryQuality q;
    q.detJ = det;
    q.positive_jacobian = det > Real(0);

    if (dim == 1) {
        q.condition_number = (std::abs(det) > Real(0))
                                 ? Real(1)
                                 : std::numeric_limits<Real>::infinity();
        return q;
    }

    auto J = mapping.jacobian(xi);
    if (dim == 2) {
        // Use the metric tensor eigenvalues: cond(J) = sqrt(cond(J^T J)).
        const auto G = MetricTensor::covariant(J, 2);
        const Real a = G(0, 0);
        const Real b = G(0, 1);
        const Real c = G(1, 1);
        const Real tr = a + c;
        const Real detG = a * c - b * b;
        const Real disc = std::sqrt(std::max(Real(0), (a - c) * (a - c) + Real(4) * b * b));
        const Real lambda_max = (tr + disc) / Real(2);
        // Avoid catastrophic cancellation in lambda_min = (tr - disc)/2 for extreme aspect ratios.
        // For SPD 2x2, det(G) = lambda_min * lambda_max, so lambda_min = det(G) / lambda_max.
        const Real lambda_min = (lambda_max > Real(0)) ? (detG / lambda_max) : Real(0);
        if (lambda_min <= Real(0)) {
            q.condition_number = std::numeric_limits<Real>::infinity();
        } else {
            q.condition_number = std::sqrt(lambda_max / lambda_min);
        }
        return q;
    }

    // Condition number (infinity norm approximation) for 3D volumetric mappings.
    const auto Jinv = J.inverse();
    Real normJ = Real(0);
    Real normJinv = Real(0);
    for (std::size_t i = 0; i < 3; ++i) {
        Real row_sum_J = Real(0);
        Real row_sum_Jinv = Real(0);
        for (std::size_t j = 0; j < 3; ++j) {
            row_sum_J += std::abs(J(i, j));
            row_sum_Jinv += std::abs(Jinv(i, j));
        }
        normJ = std::max(normJ, row_sum_J);
        normJinv = std::max(normJinv, row_sum_Jinv);
    }
    q.condition_number = normJ * normJinv;
    return q;
}

} // namespace geometry
} // namespace FE
} // namespace svmp
