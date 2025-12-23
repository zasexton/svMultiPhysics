/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "WedgeQuadrature.h"
#include <memory>

namespace svmp {
namespace FE {
namespace quadrature {

namespace {

std::unique_ptr<QuadratureRule> make_line_rule(int order, QuadratureType type) {
    QuadratureType base = (type == QuadratureType::GaussLobatto)
                              ? QuadratureType::GaussLobatto
                              : QuadratureType::GaussLegendre;

    int points = (base == QuadratureType::GaussLobatto)
                     ? std::max(2, (order + 4) / 2)
                     : std::max(1, (order + 2) / 2);

    switch (base) {
        case QuadratureType::GaussLobatto:
            return std::make_unique<GaussLobattoQuadrature1D>(points);
        default:
            return std::make_unique<GaussQuadrature1D>(points);
    }
}

} // namespace

WedgeQuadrature::WedgeQuadrature(int order, QuadratureType type)
    : WedgeQuadrature(order, order, type) {}

WedgeQuadrature::WedgeQuadrature(int triangle_order, int line_order, QuadratureType type)
    : QuadratureRule(svmp::CellFamily::Wedge, 3) {
    const bool reduced = type == QuadratureType::Reduced;
    const int tri_eff  = reduced ? std::max(1, triangle_order - 1) : triangle_order;
    const int line_eff = reduced ? std::max(1, line_order - 1) : line_order;

    TriangleQuadrature tri_rule(tri_eff);
    auto line_rule = make_line_rule(line_eff, type);

    const int estimated_order = std::min(tri_rule.order(), line_rule->order());
    set_order(estimated_order);

    std::vector<QuadPoint> pts;
    std::vector<Real> wts;
    pts.reserve(tri_rule.num_points() * line_rule->num_points());
    wts.reserve(pts.capacity());

    for (std::size_t i = 0; i < tri_rule.num_points(); ++i) {
        for (std::size_t j = 0; j < line_rule->num_points(); ++j) {
            QuadPoint qp{
                tri_rule.point(i)[0],
                tri_rule.point(i)[1],
                line_rule->point(j)[0]
            };
            pts.push_back(qp);
            wts.push_back(tri_rule.weight(i) * line_rule->weight(j));
        }
    }

    set_data(std::move(pts), std::move(wts));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
