/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "TriangleQuadrature.h"
#include <algorithm>

namespace svmp {
namespace FE {
namespace quadrature {

TriangleQuadrature::TriangleQuadrature(int requested_order)
    : QuadratureRule(svmp::CellFamily::Triangle, 2) {
    if (requested_order < 1) {
        throw FEException("TriangleQuadrature: requested_order must be >= 1",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // With Duffy: a total-degree p monomial maps to u^a (1-u)^{b+1} v^b.
    // The u-degree increases by 1, so an n-point Gauss rule guarantees
    // total-degree exactness up to 2n-2.
    const int n = std::max(1, (requested_order + 3) / 2);  // ensures 2n-2 >= requested
    GaussQuadrature1D base(n);
    const auto& line_pts = base.points();
    const auto& line_wts = base.weights();

    std::vector<QuadPoint> pts;
    std::vector<Real> wts;
    pts.reserve(static_cast<std::size_t>(n * n));
    wts.reserve(static_cast<std::size_t>(n * n));

    for (int i = 0; i < n; ++i) {
        const Real u = Real(0.5) * (line_pts[static_cast<std::size_t>(i)][0] + Real(1)); // map to [0,1]
        const Real wu = line_wts[static_cast<std::size_t>(i)] * Real(0.5);
        const Real one_minus_u = Real(1) - u;

        for (int j = 0; j < n; ++j) {
            const Real v = Real(0.5) * (line_pts[static_cast<std::size_t>(j)][0] + Real(1));
            const Real wv = line_wts[static_cast<std::size_t>(j)] * Real(0.5);

            QuadPoint qp{u, one_minus_u * v, Real(0)};
            Real weight = wu * wv * one_minus_u;  // Jacobian of Duffy transform
            pts.push_back(qp);
            wts.push_back(weight);
        }
    }

    set_order(std::max(1, 2 * n - 2));
    set_data(std::move(pts), std::move(wts));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
