/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "HexahedronQuadrature.h"
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
            return std::max(1, (requested + 2) / 2);
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

HexahedronQuadrature::HexahedronQuadrature(int order, QuadratureType type)
    : HexahedronQuadrature(order, order, order, type) {}

HexahedronQuadrature::HexahedronQuadrature(int order_x, int order_y, int order_z,
                                           QuadratureType type)
    : QuadratureRule(svmp::CellFamily::Hex, 3) {
    const bool reduced = type == QuadratureType::Reduced;
    QuadratureType base_type = (type == QuadratureType::GaussLobatto)
                                   ? QuadratureType::GaussLobatto
                                   : QuadratureType::GaussLegendre;

    const int ex = reduced ? std::max(1, order_x - 1) : order_x;
    const int ey = reduced ? std::max(1, order_y - 1) : order_y;
    const int ez = reduced ? std::max(1, order_z - 1) : order_z;

    const int px = points_for_order(ex, base_type);
    const int py = points_for_order(ey, base_type);
    const int pz = points_for_order(ez, base_type);

    auto rule_x = make_line_rule(px, base_type);
    auto rule_y = make_line_rule(py, base_type);
    auto rule_z = make_line_rule(pz, base_type);

    const int order_estimate = std::min({order_for_points(px, base_type),
                                         order_for_points(py, base_type),
                                         order_for_points(pz, base_type)});
    set_order(order_estimate);

    std::vector<QuadPoint> pts;
    std::vector<Real> wts;
    pts.reserve(rule_x->num_points() * rule_y->num_points() * rule_z->num_points());
    wts.reserve(pts.capacity());

    for (std::size_t i = 0; i < rule_x->num_points(); ++i) {
        for (std::size_t j = 0; j < rule_y->num_points(); ++j) {
            for (std::size_t k = 0; k < rule_z->num_points(); ++k) {
                QuadPoint qp{rule_x->point(i)[0], rule_y->point(j)[0], rule_z->point(k)[0]};
                pts.push_back(qp);
                wts.push_back(rule_x->weight(i) * rule_y->weight(j) * rule_z->weight(k));
            }
        }
    }

    set_data(std::move(pts), std::move(wts));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
