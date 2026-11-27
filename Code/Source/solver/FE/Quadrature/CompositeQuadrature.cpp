/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "CompositeQuadrature.h"
#include "PyramidQuadrature.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace svmp {
namespace FE {
namespace quadrature {

namespace {

void ensure_positive(std::array<int, 3>& s) {
    for (int& v : s) {
        if (v < 1) {
            v = 1;
        }
    }
}

// Map the base rule onto each interval, rectangle, or box.
void build_tensor(const QuadratureRule& base,
                  const TensorPartitionQuadrature::AxisBreaks& breaks,
                  std::vector<QuadPoint>& points,
                  std::vector<Real>& weights) {
    if (base.num_points() == 0) return;
    std::size_t boxes = 1;
    for (const auto& axis : breaks) {
        const std::size_t spans = axis.size() - 1;
        FE_CHECK_ARG(boxes <= points.max_size() / spans, "Tensor partition exceeds capacity");
        boxes *= spans;
    }
    FE_CHECK_ARG(boxes <= points.max_size() / base.num_points(), "Tensor quadrature exceeds capacity");
    points.reserve(boxes * base.num_points());
    weights.reserve(boxes * base.num_points());
    for (std::size_t box = 0; box < boxes; ++box) {
        auto index = box;
        QuadPoint lower{}, scale{};
        Real jacobian = 1;
        for (int d = base.dimension() - 1; d >= 0; --d) {
            const auto& axis = breaks[d];
            const auto span = index % (axis.size() - 1);
            index /= axis.size() - 1;
            lower[d] = axis[span];
            scale[d] = (axis[span + 1] - axis[span]) / 2;
            jacobian *= scale[d];
        }
        for (std::size_t q = 0; q < base.num_points(); ++q) {
            QuadPoint mapped{};
            const auto point = base.point(q);
            for (int d = 0; d < base.dimension(); ++d) {
                mapped[d] = lower[d] + scale[d] * (point[d] + 1);
            }
            points.push_back(mapped);
            weights.push_back(base.weight(q) * jacobian);
        }
    }
}

// Add a mapped triangle defined by v0,v1,v2 to output using base rule
void add_subtriangle(const QuadratureRule& base,
                     const QuadPoint& v0,
                     const QuadPoint& v1,
                     const QuadPoint& v2,
                     std::vector<QuadPoint>& out_pts,
                     std::vector<Real>& out_wts) {
    const auto& pts = base.points();
    const auto& wts = base.weights();

    const Real jx0 = v1[0] - v0[0];
    const Real jy0 = v1[1] - v0[1];
    const Real jx1 = v2[0] - v0[0];
    const Real jy1 = v2[1] - v0[1];
    const Real jac = std::abs(jx0 * jy1 - jx1 * jy0); // twice the subtriangle area

    for (std::size_t i = 0; i < pts.size(); ++i) {
        const Real r = pts[i][0];
        const Real s = pts[i][1];

        QuadPoint mapped{
            v0[0] + r * jx0 + s * jx1,
            v0[1] + r * jy0 + s * jy1,
            Real(0)
        };

        out_pts.push_back(mapped);
        out_wts.push_back(wts[i] * jac);
    }
}

void build_triangle(const QuadratureRule& base,
                    int subdivisions,
                    std::vector<QuadPoint>& out_pts,
                    std::vector<Real>& out_wts) {
    const int n = std::max(1, subdivisions);
    const Real h = Real(1) / Real(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - i; ++j) {
            QuadPoint v0{Real(i) * h, Real(j) * h, Real(0)};
            QuadPoint v1{Real(i + 1) * h, Real(j) * h, Real(0)};
            QuadPoint v2{Real(i) * h, Real(j + 1) * h, Real(0)};
            add_subtriangle(base, v0, v1, v2, out_pts, out_wts);

            if (i + j + 1 < n) {
                QuadPoint v3{Real(i + 1) * h, Real(j + 1) * h, Real(0)};
                add_subtriangle(base, v1, v3, v2, out_pts, out_wts);
            }
        }
    }
}

void build_tetrahedron(const QuadratureRule& base, int n,
                       std::vector<QuadPoint>& points, std::vector<Real>& weights) {
    const Real h = Real{1} / n;
    // The ordered cube coordinates 0 <= k <= j <= i <= n map onto the
    // reference tetrahedron by (i,j,k) -> (i-j,j-k,k)/n. Each unit cube
    // splits into six tetrahedra along its body diagonal. Retain the ones
    // whose vertices satisfy the ordering, including along the sloping faces.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            for (int k = 0; k <= j; ++k) {
                const QuadPoint origin{(i - j) * h, (j - k) * h, k * h};
                std::array<int, 3> order{0, 1, 2};
                do {
                    std::array<int, 3> corner{i, j, k};
                    std::array<QuadPoint, 3> edges;
                    bool inside = true;
                    for (int e = 0; e < 3; ++e) {
                        ++corner[order[e]];
                        if (corner[0] < corner[1] || corner[1] < corner[2]) {
                            inside = false;
                            break;
                        }
                        edges[e] = QuadPoint{(corner[0] - corner[1]) * h,
                                             (corner[1] - corner[2]) * h, corner[2] * h} - origin;
                    }
                    if (!inside) continue;
                    for (std::size_t q = 0; q < base.num_points(); ++q) {
                        const auto& p = base.point(q);
                        points.push_back(origin + edges[0] * p[0] + edges[1] * p[1] + edges[2] * p[2]);
                        weights.push_back(base.weight(q) * h * h * h);
                    }
                } while (std::next_permutation(order.begin(), order.end()));
            }
        }
    }
}

void build_pyramid(const PyramidQuadrature& base, const std::array<int, 3>& sub,
                   std::vector<QuadPoint>& points, std::vector<Real>& weights) {
    const Real cell_scale = Real{1} / sub[0] / sub[1] / sub[2];
    // Refine the two lateral axes and the height in collapsed-cube
    // coordinates. Replace the base point's (1-z)^2 volume factor with
    // the factor at the new height. The Gauss points are below the apex.
    for (int i = 0; i < sub[0]; ++i) {
        for (int j = 0; j < sub[1]; ++j) {
            for (int k = 0; k < sub[2]; ++k) {
                for (std::size_t q = 0; q < base.num_points(); ++q) {
                    const auto& p = base.point(q);
                    const Real width = 1 - p[2];
                    const Real z = (k + p[2]) / sub[2];
                    const Real a = -1 + (2.0 * i + 1 + p[0] / width) / sub[0];
                    const Real b = -1 + (2.0 * j + 1 + p[1] / width) / sub[1];
                    const Real ratio = (1 - z) / width;
                    points.push_back({(1 - z) * a, (1 - z) * b, z});
                    weights.push_back(base.weight(q) * cell_scale * ratio * ratio);
                }
            }
        }
    }
}

void build_wedge(const QuadratureRule& base,
                 const std::array<int, 3>& sub,
                 std::vector<QuadPoint>& out_pts,
                 std::vector<Real>& out_wts) {
    const int tri_sub = std::max(1, sub[0]);
    const int line_sub = std::max(1, sub[2]);
    const Real h_t = Real(2) / Real(line_sub);

    const auto& pts = base.points();
    const auto& wts = base.weights();

    const Real h = Real(1) / Real(tri_sub);

    for (int i = 0; i < tri_sub; ++i) {
        for (int j = 0; j < tri_sub - i; ++j) {
            QuadPoint v0{Real(i) * h, Real(j) * h, Real(0)};
            QuadPoint v1{Real(i + 1) * h, Real(j) * h, Real(0)};
            QuadPoint v2{Real(i) * h, Real(j + 1) * h, Real(0)};
            QuadPoint v3{Real(i + 1) * h, Real(j + 1) * h, Real(0)};

            auto add_subwedge = [&](const QuadPoint& a,
                                    const QuadPoint& b,
                                    const QuadPoint& c) {
                const Real jx0 = b[0] - a[0];
                const Real jy0 = b[1] - a[1];
                const Real jx1 = c[0] - a[0];
                const Real jy1 = c[1] - a[1];
                const Real jac_tri = std::abs(jx0 * jy1 - jx1 * jy0); // twice area of subtriangle

                for (int lt = 0; lt < line_sub; ++lt) {
                    const Real ot = Real(-1) + h_t * Real(lt);
                    const Real st = h_t / Real(2);

                    for (std::size_t p = 0; p < pts.size(); ++p) {
                        const Real r = pts[p][0];
                        const Real s = pts[p][1];
                        const Real t = pts[p][2];

                        out_pts.push_back(QuadPoint{
                            a[0] + r * jx0 + s * jx1,
                            a[1] + r * jy0 + s * jy1,
                            ot + st * (t + Real(1))
                        });
                        out_wts.push_back(wts[p] * jac_tri * st);
                    }
                }
            };

            // Always add the “lower” subtriangle (v0,v1,v2).
            add_subwedge(v0, v1, v2);

            // Only add the “upper” subtriangle (v1,v3,v2) when it lies inside
            // the reference triangle grid (matches build_triangle()).
            if (i + j + 1 < tri_sub) {
                add_subwedge(v1, v3, v2);
            }
        }
    }
}

} // namespace

CompositeQuadrature::CompositeQuadrature(const QuadratureRule& base_rule, int subdivisions)
    : CompositeQuadrature(base_rule, {subdivisions, subdivisions, subdivisions}) {}

CompositeQuadrature::CompositeQuadrature(const QuadratureRule& base_rule,
                                         const std::array<int, 3>& subdivisions)
    : QuadratureRule(base_rule.cell_family(), base_rule.dimension(), base_rule.order()) {
    std::array<int, 3> sub = subdivisions;
    ensure_positive(sub);

    std::vector<QuadPoint> pts;
    std::vector<Real> wts;
    if (sub == std::array<int, 3>{1, 1, 1}) {
        set_data(base_rule.points(), base_rule.weights());
        return;
    }
    const auto family = base_rule.cell_family();
    if (family == svmp::CellFamily::Triangle || family == svmp::CellFamily::Wedge ||
        family == svmp::CellFamily::Tetra) sub[1] = sub[0];
    if (family == svmp::CellFamily::Tetra) sub[2] = sub[0];
    std::size_t count = base_rule.num_points();
    for (int d = 0; d < base_rule.dimension(); ++d) {
        FE_CHECK_ARG(count <= pts.max_size() / sub[d], "Composite quadrature point count exceeds capacity");
        count *= sub[d];
    }
    pts.reserve(count);
    wts.reserve(count);

    switch (base_rule.cell_family()) {
        case svmp::CellFamily::Line:
        case svmp::CellFamily::Quad:
        case svmp::CellFamily::Hex: {
            TensorPartitionQuadrature::AxisBreaks breaks(base_rule.dimension());
            for (int d = 0; d < base_rule.dimension(); ++d) {
                auto& axis = breaks[d];
                const auto n = static_cast<std::size_t>(sub[d]);
                axis.reserve(n + 1);
                for (std::size_t i = 0; i <= n; ++i) {
                    axis.push_back(Real(-1) + (Real(2) * i) / n);
                }
            }
            build_tensor(base_rule, breaks, pts, wts);
            break;
        }
        case svmp::CellFamily::Triangle:
            build_triangle(base_rule, sub[0], pts, wts);
            break;
        case svmp::CellFamily::Wedge:
            build_wedge(base_rule, sub, pts, wts);
            break;
        case svmp::CellFamily::Tetra:
            build_tetrahedron(base_rule, sub[0], pts, wts);
            break;
        case svmp::CellFamily::Pyramid:
            if (const auto* pyramid = dynamic_cast<const PyramidQuadrature*>(&base_rule)) {
                build_pyramid(*pyramid, sub, pts, wts);
            } else {
                build_pyramid(PyramidQuadrature(std::max(1, base_rule.order())), sub, pts, wts);
            }
            break;
        default:
            FE_THROW(NotImplementedException, "Composite quadrature does not support refinement for this cell family");
    }

    set_data(std::move(pts), std::move(wts));
}

TensorPartitionQuadrature::TensorPartitionQuadrature(const QuadratureRule& base_rule,
                                                     AxisBreaks axis_breaks)
    : QuadratureRule(base_rule.cell_family(), base_rule.dimension(), base_rule.order()),
      axis_breaks_(std::move(axis_breaks)) {
    const auto family = cell_family();
    FE_CHECK_ARG(family == svmp::CellFamily::Line || family == svmp::CellFamily::Quad ||
                 family == svmp::CellFamily::Hex, "Tensor partitions require a line, quad or hex");
    FE_CHECK_ARG(axis_breaks_.size() == static_cast<std::size_t>(dimension()),
                 "Tensor partition must describe every active axis");
    for (const auto& axis : axis_breaks_) {
        FE_CHECK_ARG(axis.size() >= 2 && axis.front() == -1 && axis.back() == 1,
                     "Tensor partition must cover the full reference interval");
        for (std::size_t i = 1; i < axis.size(); ++i) {
            FE_CHECK_ARG(axis[i] > axis[i - 1], "Tensor partition endpoints must increase");
        }
    }
    std::vector<QuadPoint> points;
    std::vector<Real> weights;
    build_tensor(base_rule, axis_breaks_, points, weights);
    set_data(std::move(points), std::move(weights));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
