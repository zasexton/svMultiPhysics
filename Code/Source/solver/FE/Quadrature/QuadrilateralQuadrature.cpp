/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "QuadrilateralQuadrature.h"
#include <algorithm>
#include <memory>

namespace svmp {
namespace FE {
namespace quadrature {

namespace {

int points_for_order(int requested, QuadratureType type) {
    if (requested < 1) {
        requested = 1;
    }
    switch (type) {
        case QuadratureType::GaussLobatto:
            // Need 2n-3 >= requested  ->  n >= ceil((requested+3)/2)
            return std::max(2, (requested + 4) / 2);
        default:
            return std::max(1, (requested + 2) / 2);  // 2n-1 >= requested
    }
}

int order_for_points(int points, QuadratureType type) {
    switch (type) {
        case QuadratureType::GaussLobatto:
            return 2 * points - 3;
        default:
            return 2 * points - 1;
    }
}

std::unique_ptr<QuadratureRule> make_line_rule(int points, QuadratureType type) {
    switch (type) {
        case QuadratureType::GaussLobatto:
            return std::make_unique<GaussLobattoQuadrature1D>(points);
        default:
            return std::make_unique<GaussQuadrature1D>(points);
    }
}

} // namespace

QuadrilateralQuadrature::QuadrilateralQuadrature(int order, QuadratureType type)
    : QuadrilateralQuadrature(order, order, type) {}

QuadrilateralQuadrature::QuadrilateralQuadrature(int order_x, int order_y, QuadratureType type)
    : QuadratureRule(svmp::CellFamily::Quad, 2) {
    const bool reduced = type == QuadratureType::Reduced;
    QuadratureType base_type = (type == QuadratureType::GaussLobatto)
                                   ? QuadratureType::GaussLobatto
                                   : QuadratureType::GaussLegendre;

    const int effective_order_x = reduced ? std::max(1, order_x - 1) : order_x;
    const int effective_order_y = reduced ? std::max(1, order_y - 1) : order_y;

    const int points_x = points_for_order(effective_order_x, base_type);
    const int points_y = points_for_order(effective_order_y, base_type);

    auto rule_x = make_line_rule(points_x, base_type);
    auto rule_y = make_line_rule(points_y, base_type);

    const int order_estimate = std::min(order_for_points(points_x, base_type),
                                        order_for_points(points_y, base_type));
    set_order(order_estimate);

    std::vector<QuadPoint> pts;
    std::vector<Real> wts;
    pts.reserve(static_cast<std::size_t>(rule_x->num_points() * rule_y->num_points()));
    wts.reserve(pts.capacity());

    for (std::size_t i = 0; i < rule_x->num_points(); ++i) {
        for (std::size_t j = 0; j < rule_y->num_points(); ++j) {
            QuadPoint qp{rule_x->point(i)[0], rule_y->point(j)[0], Real(0)};
            pts.push_back(qp);
            wts.push_back(rule_x->weight(i) * rule_y->weight(j));
        }
    }

    set_data(std::move(pts), std::move(wts));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
