/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "CompositeQuadrature.h"
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

// Tensor-product subdivision for line/quad/hex
void build_tensor(const QuadratureRule& base,
                  const std::array<int, 3>& sub,
                  std::vector<QuadPoint>& out_pts,
                  std::vector<Real>& out_wts) {
    const int dim = base.dimension();
    const Real h_x = Real(2) / Real(sub[0]);
    const Real h_y = dim >= 2 ? Real(2) / Real(sub[1]) : Real(0);
    const Real h_z = dim == 3 ? Real(2) / Real(sub[2]) : Real(0);

    const auto& pts = base.points();
    const auto& wts = base.weights();

    auto map_coord = [](Real pt, Real origin, Real scale) {
        return origin + scale * (pt + Real(1));
    };

    for (int ix = 0; ix < sub[0]; ++ix) {
        const Real ox = Real(-1) + h_x * Real(ix);
        const Real sx = h_x / Real(2);

        for (int iy = 0; iy < (dim >= 2 ? sub[1] : 1); ++iy) {
            const Real oy = dim >= 2 ? (Real(-1) + h_y * Real(iy)) : Real(0);
            const Real sy = dim >= 2 ? (h_y / Real(2)) : Real(0);

            for (int iz = 0; iz < (dim == 3 ? sub[2] : 1); ++iz) {
                const Real oz = dim == 3 ? (Real(-1) + h_z * Real(iz)) : Real(0);
                const Real sz = dim == 3 ? (h_z / Real(2)) : Real(0);

                for (std::size_t p = 0; p < pts.size(); ++p) {
                    Real x = map_coord(pts[p][0], ox, sx);
                    Real y = dim >= 2 ? map_coord(pts[p][1], oy, sy) : Real(0);
                    Real z = dim == 3 ? map_coord(pts[p][2], oz, sz) : Real(0);

                    out_pts.push_back(QuadPoint{x, y, z});

                    Real weight_scale = sx;
                    if (dim >= 2) weight_scale *= sy;
                    if (dim == 3) weight_scale *= sz;
                    out_wts.push_back(wts[p] * weight_scale);
                }
            }
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
    pts.reserve(base_rule.num_points() * static_cast<std::size_t>(sub[0] * sub[1] * sub[2]));
    wts.reserve(pts.capacity());

    switch (base_rule.cell_family()) {
        case svmp::CellFamily::Line:
        case svmp::CellFamily::Quad:
        case svmp::CellFamily::Hex:
            build_tensor(base_rule, sub, pts, wts);
            break;
        case svmp::CellFamily::Triangle:
            build_triangle(base_rule, sub[0], pts, wts);
            break;
        case svmp::CellFamily::Wedge:
            build_wedge(base_rule, sub, pts, wts);
            break;
        default:
            // Fallback: no subdivision support for this family yet
            pts.assign(base_rule.points().begin(), base_rule.points().end());
            wts.assign(base_rule.weights().begin(), base_rule.weights().end());
            break;
    }

    set_data(std::move(pts), std::move(wts));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
