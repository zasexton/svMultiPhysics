/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "TetrahedronQuadrature.h"
#include <algorithm>

namespace svmp {
namespace FE {
namespace quadrature {

TetrahedronQuadrature::TetrahedronQuadrature(int requested_order)
    : QuadratureRule(svmp::CellFamily::Tetra, 3) {
    if (requested_order < 1) {
        throw FEException("TetrahedronQuadrature: requested_order must be >= 1",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // With the 3D Duffy map, a total-degree p monomial maps to:
    //   u^a (1-u)^{b+c+2} v^b (1-v)^{c+1} w^c
    // so the u-degree increases by 2, which yields guaranteed total-degree
    // exactness up to 2n-3 for an n-point Gauss rule per dimension.
    const int n = std::max(1, (requested_order + 4) / 2);  // ensures 2n-3 >= requested
    GaussQuadrature1D base(n);
    const auto& line_pts = base.points();
    const auto& line_wts = base.weights();

    std::vector<QuadPoint> pts;
    std::vector<Real> wts;
    pts.reserve(static_cast<std::size_t>(n * n * n));
    wts.reserve(static_cast<std::size_t>(n * n * n));

    for (int i = 0; i < n; ++i) {
        const Real u = Real(0.5) * (line_pts[static_cast<std::size_t>(i)][0] + Real(1));
        const Real wu = line_wts[static_cast<std::size_t>(i)] * Real(0.5);
        const Real one_minus_u = Real(1) - u;

        for (int j = 0; j < n; ++j) {
            const Real v = Real(0.5) * (line_pts[static_cast<std::size_t>(j)][0] + Real(1));
            const Real wv = line_wts[static_cast<std::size_t>(j)] * Real(0.5);
            const Real one_minus_v = Real(1) - v;

            for (int k = 0; k < n; ++k) {
                const Real w = Real(0.5) * (line_pts[static_cast<std::size_t>(k)][0] + Real(1));
                const Real ww = line_wts[static_cast<std::size_t>(k)] * Real(0.5);

                QuadPoint qp{
                    u,
                    one_minus_u * v,
                    one_minus_u * one_minus_v * w
                };

                const Real jac = one_minus_u * one_minus_u * one_minus_v;
                pts.push_back(qp);
                wts.push_back(wu * wv * ww * jac);
            }
        }
    }

    set_order(std::max(1, 2 * n - 3));
    set_data(std::move(pts), std::move(wts));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
