/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "SingularQuadrature.h"
#include <algorithm>

namespace svmp {
namespace FE {
namespace quadrature {

namespace {
class WrappedRule : public QuadratureRule {
public:
    WrappedRule(svmp::CellFamily family, int dim, int order,
                std::vector<QuadPoint> pts, std::vector<Real> wts)
        : QuadratureRule(family, dim, order) {
        set_data(std::move(pts), std::move(wts));
    }
};
} // namespace

std::unique_ptr<QuadratureRule> SingularQuadrature::duffy_triangle(int order) {
    // Use an n×n Gauss rule on [0,1]^2 and map to the reference triangle
    // (0,0)-(1,0)-(0,1) with a vertex-focused Duffy transform:
    //   x = r(1-s), y = r s,  (r,s) in [0,1]^2,  J = r.
    // This guarantees total-degree exactness up to 2n-2 for smooth polynomials.
    const int n = std::max(1, (order + 3) / 2);  // ensure 2n-2 >= order
    GaussQuadrature1D base(n);
    const auto& pts = base.points();
    const auto& wts = base.weights();

    std::vector<QuadPoint> out_pts;
    std::vector<Real> out_wts;
    out_pts.reserve(static_cast<std::size_t>(n * n));
    out_wts.reserve(out_pts.capacity());

    for (int i = 0; i < n; ++i) {
        const Real r = Real(0.5) * (pts[static_cast<std::size_t>(i)][0] + Real(1));
        const Real wr = wts[static_cast<std::size_t>(i)] * Real(0.5);
        for (int j = 0; j < n; ++j) {
            const Real s = Real(0.5) * (pts[static_cast<std::size_t>(j)][0] + Real(1));
            const Real ws = wts[static_cast<std::size_t>(j)] * Real(0.5);

            out_pts.push_back(QuadPoint{r * (Real(1) - s), r * s, Real(0)});
            out_wts.push_back(wr * ws * r); // Jacobian J = r
        }
    }

    return std::make_unique<WrappedRule>(svmp::CellFamily::Triangle, 2, std::max(1, 2 * n - 2),
                                         std::move(out_pts), std::move(out_wts));
}

std::unique_ptr<QuadratureRule> SingularQuadrature::duffy_tetrahedron(int order) {
    // Vertex-focused Duffy transform for the reference tetrahedron
    // (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1):
    //   x = r(1-s)
    //   y = r s (1-t)
    //   z = r s t
    // with (r,s,t) in [0,1]^3 and J = r^2 s.
    // This guarantees total-degree exactness up to 2n-3 for smooth polynomials.
    const int n = std::max(1, (order + 4) / 2);  // ensure 2n-3 >= order
    GaussQuadrature1D base(n);
    const auto& pts = base.points();
    const auto& wts = base.weights();

    std::vector<QuadPoint> out_pts;
    std::vector<Real> out_wts;
    out_pts.reserve(static_cast<std::size_t>(n * n * n));
    out_wts.reserve(out_pts.capacity());

    for (int i = 0; i < n; ++i) {
        const Real r = Real(0.5) * (pts[static_cast<std::size_t>(i)][0] + Real(1));
        const Real wr = wts[static_cast<std::size_t>(i)] * Real(0.5);
        for (int j = 0; j < n; ++j) {
            const Real s = Real(0.5) * (pts[static_cast<std::size_t>(j)][0] + Real(1));
            const Real ws = wts[static_cast<std::size_t>(j)] * Real(0.5);
            for (int k = 0; k < n; ++k) {
                const Real t = Real(0.5) * (pts[static_cast<std::size_t>(k)][0] + Real(1));
                const Real wt = wts[static_cast<std::size_t>(k)] * Real(0.5);

                out_pts.push_back(QuadPoint{r * (Real(1) - s),
                                            r * s * (Real(1) - t),
                                            r * s * t});
                out_wts.push_back(wr * ws * wt * r * r * s); // Jacobian J = r^2 s
            }
        }
    }

    return std::make_unique<WrappedRule>(svmp::CellFamily::Tetra, 3, std::max(1, 2 * n - 3),
                                         std::move(out_pts), std::move(out_wts));
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
