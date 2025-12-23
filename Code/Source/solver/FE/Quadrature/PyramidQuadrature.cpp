/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "PyramidQuadrature.h"
#include <algorithm>

namespace svmp {
namespace FE {
namespace quadrature {

namespace {

int points_for_order(int requested) {
    return std::max(1, (requested + 2) / 2);  // 2n-1 >= requested
}

int points_for_order_with_weight(int requested) {
    // Need to integrate p(t)*(1-t)^2 -> require order p+2 before mapping
    return std::max(1, (requested + 5) / 2);  // ensure 2n-1 >= requested+2
}

} // namespace

PyramidQuadrature::PyramidQuadrature(int requested_order)
    : PyramidQuadrature(requested_order, requested_order) {}

PyramidQuadrature::PyramidQuadrature(int order_ab, int order_t)
    : QuadratureRule(svmp::CellFamily::Pyramid, 3) {
    if (order_ab < 1 || order_t < 1) {
        throw FEException("PyramidQuadrature: orders must be >= 1",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const int n_ab = points_for_order(order_ab);
    const int n_t = points_for_order_with_weight(order_t);

    GaussQuadrature1D rule_ab(n_ab);
    GaussQuadrature1D rule_t(n_t);

    const auto& pts_ab = rule_ab.points();
    const auto& wts_ab = rule_ab.weights();
    const auto& pts_t = rule_t.points();
    const auto& wts_t = rule_t.weights();

    std::vector<QuadPoint> pts;
    std::vector<Real> wts;
    pts.reserve(static_cast<std::size_t>(n_ab * n_ab * n_t));
    wts.reserve(pts.capacity());

    for (int i = 0; i < n_ab; ++i) {
        const Real a = pts_ab[static_cast<std::size_t>(i)][0];
        const Real wa = wts_ab[static_cast<std::size_t>(i)];
        for (int j = 0; j < n_ab; ++j) {
            const Real b = pts_ab[static_cast<std::size_t>(j)][0];
            const Real wb = wts_ab[static_cast<std::size_t>(j)];

            for (int k = 0; k < n_t; ++k) {
                const Real t_ref = pts_t[static_cast<std::size_t>(k)][0];
                const Real wt = wts_t[static_cast<std::size_t>(k)];

                const Real t = Real(0.5) * (t_ref + Real(1));   // map to [0,1]
                const Real w_t_mapped = wt * Real(0.5);
                const Real scale = Real(1) - t;

                QuadPoint qp{scale * a, scale * b, t};
                pts.push_back(qp);
                wts.push_back(wa * wb * w_t_mapped * scale * scale);
            }
        }
    }

    const int order_estimate = std::min(rule_ab.order(), rule_t.order() - 2);
    set_order(std::max(1, order_estimate));
    set_data(std::move(pts), std::move(wts));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
