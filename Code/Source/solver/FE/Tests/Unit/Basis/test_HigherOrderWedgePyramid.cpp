/**
 * @file test_HigherOrderWedgePyramid.cpp
 * @brief Tests for higher-order RT and Nedelec on Wedge and Pyramid
 *
 * NOTE: Higher-order (k >= 1) RT and Nedelec elements on Wedge and Pyramid
 * are not yet fully implemented. The moment-based construction produces
 * singular matrices because the naive hierarchical modal basis is not
 * linearly independent under the DOF functionals.
 *
 * This file documents the expected behavior and serves as a test harness
 * for when the implementation is completed.
 *
 * Required for full implementation:
 * - Option A: Direct DOF-based construction (like MFEM)
 * - Option B: Symbolic/CAS-derived nodal basis (like RT0 on pyramid)
 * - Option C: Hierarchical basis with careful orthogonalization
 *
 * See: Bergot & Duruffle (2013), Falk-Gatto-Monk (2011), Nigam-Phillips (2012)
 */

#include <gtest/gtest.h>
#include "FE/Basis/VectorBasis.h"
#include "FE/Core/FEException.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::basis;

// =============================================================================
// Local helpers for moment-based DOF integration
// =============================================================================

/// Integrate N_i . n_hat over a face of the reference wedge.
/// For RT(k), the DOF functional for face f is:
///   integral_f (N_i . n) * q dA = delta_{i,dof}
/// where q is a polynomial test function on the face.
/// With q=1 (mode 0), this gives the zeroth-order face moment.
static double integrate_wedge_face_flux(const RaviartThomasBasis& basis,
                                        int face_id,
                                        int func_id,
                                        int tri_order = 6,
                                        int quad_order = 6) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    double flux = 0.0;

    // Faces 0,1 are triangular (bottom z=-1, top z=+1)
    if (face_id == 0 || face_id == 1) {
        TriangleQuadrature tri(tri_order);
        const double z = (face_id == 0) ? -1.0 : 1.0;
        const double nz = (face_id == 0) ? -1.0 : 1.0;
        for (std::size_t q = 0; q < tri.num_points(); ++q) {
            const auto& pt = tri.point(q);
            Vector<Real,3> xi{pt[0], pt[1], static_cast<Real>(z)};
            std::vector<Vector<Real,3>> vals;
            basis.evaluate_vector_values(xi, vals);
            const auto& v = vals[static_cast<std::size_t>(func_id)];
            const double integrand = static_cast<double>(v[2]) * nz;
            flux += static_cast<double>(tri.weight(q)) * integrand;
        }
        return flux;
    }

    // Faces 2,3,4 are quadrilateral, parameterized over [-1,1]^2
    QuadrilateralQuadrature quad(quad_order, quad_order);
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& pt = quad.point(q);
        const double xi_q = pt[0];
        const double eta = pt[1];

        Vector<Real,3> x_ref{};
        Vector<Real,3> cross{};

        if (face_id == 2) {        // y=0 face: nodes 0,1,4,3
            const double t = 0.5 * (xi_q + 1.0);
            const double z = eta;
            x_ref = Vector<Real,3>{static_cast<Real>(t), Real(0), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(0), Real(-0.5), Real(0)};
        } else if (face_id == 3) { // x=0 face: nodes 1,2,5,4
            const double t = 0.5 * (xi_q + 1.0);
            const double z = eta;
            x_ref = Vector<Real,3>{Real(0), static_cast<Real>(t), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(0.5), Real(0), Real(0)};
        } else {                   // face_id == 4, x+y=1 face: nodes 2,0,3,5
            const double t = 0.5 * (xi_q + 1.0);
            const double z = eta;
            const double x = t;
            const double y = 1.0 - t;
            x_ref = Vector<Real,3>{static_cast<Real>(x), static_cast<Real>(y), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(-0.5), Real(-0.5), Real(0)};
        }

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(x_ref, vals);
        const auto& v = vals[static_cast<std::size_t>(func_id)];
        const double integrand =
            static_cast<double>(v[0]) * cross[0] +
            static_cast<double>(v[1]) * cross[1] +
            static_cast<double>(v[2]) * cross[2];
        flux += static_cast<double>(quad.weight(q)) * integrand;
    }

    return flux;
}

/// Integrate N_i . n_hat over a face of the reference pyramid.
static double integrate_pyramid_face_flux(const RaviartThomasBasis& basis,
                                          int face_id,
                                          int func_id,
                                          int tri_order = 6,
                                          int quad_order = 6) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    double flux = 0.0;

    const Vector<Real,3> v0{Real(-1), Real(-1), Real(0)};
    const Vector<Real,3> v1{Real(1),  Real(-1), Real(0)};
    const Vector<Real,3> v2{Real(1),  Real(1),  Real(0)};
    const Vector<Real,3> v3{Real(-1), Real(1),  Real(0)};
    const Vector<Real,3> v4{Real(0),  Real(0),  Real(1)};

    // Face 0: quad base {0,1,2,3}, outward normal = -z
    if (face_id == 0) {
        QuadrilateralQuadrature quad(quad_order, quad_order);
        for (std::size_t q = 0; q < quad.num_points(); ++q) {
            const auto& pt = quad.point(q);
            Vector<Real,3> xi{pt[0], pt[1], Real(0)};
            std::vector<Vector<Real,3>> vals;
            basis.evaluate_vector_values(xi, vals);
            const auto& v = vals[static_cast<std::size_t>(func_id)];
            const double integrand = -static_cast<double>(v[2]); // outward -z
            flux += static_cast<double>(quad.weight(q)) * integrand;
        }
        return flux;
    }

    // Faces 1-4: triangular faces to apex
    TriangleQuadrature tri(tri_order);
    const Vector<Real,3>* a = nullptr;
    const Vector<Real,3>* b = nullptr;
    const Vector<Real,3>* c = nullptr;
    switch (face_id) {
        case 1: a = &v0; b = &v1; c = &v4; break;
        case 2: a = &v1; b = &v2; c = &v4; break;
        case 3: a = &v2; b = &v3; c = &v4; break;
        default: a = &v3; b = &v0; c = &v4; break;
    }
    Vector<Real,3> e1 = *b - *a;
    Vector<Real,3> e2 = *c - *a;
    Vector<Real,3> cross = e1.cross(e2);

    for (std::size_t q = 0; q < tri.num_points(); ++q) {
        const auto& pt = tri.point(q);
        const double r = pt[0];
        const double s = pt[1];
        const double l0 = 1.0 - r - s;
        const double l1 = r;
        const double l2 = s;
        Vector<Real,3> xi =
            (*a) * static_cast<Real>(l0) +
            (*b) * static_cast<Real>(l1) +
            (*c) * static_cast<Real>(l2);

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& v = vals[static_cast<std::size_t>(func_id)];
        const double integrand =
            static_cast<double>(v[0]) * cross[0] +
            static_cast<double>(v[1]) * cross[1] +
            static_cast<double>(v[2]) * cross[2];
        flux += static_cast<double>(tri.weight(q)) * integrand;
    }

    return flux;
}

/// Integrate N_i . t along an edge of the reference wedge (for Nedelec DOFs).
/// The edge tangent t = (b - a), not normalized, matching the edge DOF definition.
static double integrate_wedge_edge_tangent(const NedelecBasis& basis,
                                           int edge_id,
                                           int func_id,
                                           int quad_order = 6) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    const Vector<Real,3> v0{Real(0), Real(0), Real(-1)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(-1)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(-1)};
    const Vector<Real,3> v3{Real(0), Real(0), Real(1)};
    const Vector<Real,3> v4{Real(1), Real(0), Real(1)};
    const Vector<Real,3> v5{Real(0), Real(1), Real(1)};

    struct Edge { Vector<Real,3> a, b; };
    const Edge edges[9] = {
        {v0, v1}, {v1, v2}, {v2, v0}, // bottom tri
        {v3, v4}, {v4, v5}, {v5, v3}, // top tri
        {v0, v3}, {v1, v4}, {v2, v5}  // vertical
    };

    const Edge& e = edges[static_cast<std::size_t>(edge_id)];
    const Vector<Real,3> t_vec = e.b - e.a;

    GaussQuadrature1D quad(quad_order);
    double dof = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real tq = quad.point(q)[0]; // in [-1,1]
        const Real s = (tq + Real(1)) * Real(0.5); // map to [0,1]
        Vector<Real,3> xi = e.a * (Real(1) - s) + e.b * s;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        const double integrand = static_cast<double>(
            w[0]*t_vec[0] + w[1]*t_vec[1] + w[2]*t_vec[2]);
        dof += 0.5 * quad.weight(q) * integrand;
    }
    return dof;
}

/// Integrate N_i . t along an edge of the reference pyramid (for Nedelec DOFs).
static double integrate_pyramid_edge_tangent(const NedelecBasis& basis,
                                             int edge_id,
                                             int func_id,
                                             int quad_order = 6) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    const Vector<Real,3> v0{Real(-1), Real(-1), Real(0)};
    const Vector<Real,3> v1{Real(1),  Real(-1), Real(0)};
    const Vector<Real,3> v2{Real(1),  Real(1),  Real(0)};
    const Vector<Real,3> v3{Real(-1), Real(1),  Real(0)};
    const Vector<Real,3> v4{Real(0),  Real(0),  Real(1)};

    struct Edge { Vector<Real,3> a, b; };
    const Edge edges[8] = {
        {v0, v1}, {v1, v2}, {v2, v3}, {v3, v0}, // base
        {v0, v4}, {v1, v4}, {v2, v4}, {v3, v4}  // vertical
    };

    const Edge& e = edges[static_cast<std::size_t>(edge_id)];
    const Vector<Real,3> t_vec = e.b - e.a;

    GaussQuadrature1D quad(quad_order);
    double dof = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real tq = quad.point(q)[0];
        const Real s = (tq + Real(1)) * Real(0.5);
        Vector<Real,3> xi = e.a * (Real(1) - s) + e.b * s;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        const double integrand = static_cast<double>(
            w[0]*t_vec[0] + w[1]*t_vec[1] + w[2]*t_vec[2]);
        dof += 0.5 * quad.weight(q) * integrand;
    }
    return dof;
}

// =============================================================================
// RT(0) on Wedge and Pyramid - WORKING
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT0Construction) {
    // RT(0) on Wedge should work with 5 DOFs (5 faces)
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 5u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidRT0Construction) {
    // RT(0) on Pyramid should work with 5 DOFs (5 faces)
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 5u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeRT0Evaluation) {
    RaviartThomasBasis basis(ElementType::Wedge6, 0);

    // Test point inside the wedge
    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 5u);

    // All values should be finite
    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT0Evaluation) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);

    // Test point inside the pyramid
    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 5u);

    // All values should be finite
    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

// =============================================================================
// Nedelec(0) on Wedge and Pyramid - WORKING
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND0Construction) {
    // Nedelec(0) on Wedge should work with 9 DOFs (9 edges)
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 9u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidND0Construction) {
    // Nedelec(0) on Pyramid should work with 8 DOFs (8 edges)
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 8u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeND0Evaluation) {
    NedelecBasis basis(ElementType::Wedge6, 0);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 9u);

    // All values should be finite
    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND0Evaluation) {
    NedelecBasis basis(ElementType::Pyramid5, 0);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 8u);

    // All values should be finite
    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

// =============================================================================
// RT(k>=1) on Wedge and Pyramid - NOW IMPLEMENTED via Direct DOF Construction
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT1Implemented) {
    // RT(1) on Wedge is implemented via direct DOF-based construction
    // Expected DOFs: face = 6 + 12 = 18, interior = 6, total = 24
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 1);
        EXPECT_EQ(basis.size(), 24u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidRT1Implemented) {
    // RT(1) on Pyramid is implemented via direct DOF-based construction
    // Expected DOFs: face = 4 + 12 = 16, interior = 3, total = 19
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 1);
        EXPECT_EQ(basis.size(), 19u);
    });
}

// =============================================================================
// Nedelec(k>=1) on Wedge and Pyramid - NOW IMPLEMENTED via Direct DOF Construction
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND1Implemented) {
    // Nedelec(1) on Wedge is implemented via direct DOF-based construction
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 1);
        EXPECT_EQ(basis.size(), 34u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidND1Implemented) {
    // Nedelec(1) on Pyramid is constructed (evaluation throws NotImplemented for now)
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 1);
        EXPECT_EQ(basis.size(), 28u);
    });
}

// =============================================================================
// Dimension Formula Tests (Theoretical Values)
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRTDimensionFormulas) {
    // Document the correct dimension formulas for wedge RT(k)
    // Face DOFs: 2*(k+1)(k+2)/2 + 3*(k+1)^2 = (k+1)(k+2) + 3(k+1)^2
    // Interior DOFs: 3*k*(k+1)^2/2 for k >= 1

    // RT(0): 5 faces, 0 interior = 5
    {
        RaviartThomasBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.size(), 5u);
    }

    // RT(1): theoretical = 18 face + 6 interior = 24
    // (not yet working, but size() returns the expected value before construction fails)

    // RT(2): theoretical = 39 face + 27 interior = 66
}

TEST(HigherOrderWedgePyramid, PyramidRTDimensionFormulas) {
    // Document the correct dimension formulas for pyramid RT(k)
    // Face DOFs: (k+1)^2 + 4*(k+1)(k+2)/2 = (k+1)^2 + 2(k+1)(k+2)
    // Interior DOFs: 3*k^3 for k >= 1

    // RT(0): 5 faces, 0 interior = 5
    {
        RaviartThomasBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.size(), 5u);
    }

    // RT(1): theoretical = 16 face + 3 interior = 19
    // RT(2): theoretical = 33 face + 24 interior = 57
}

TEST(HigherOrderWedgePyramid, WedgeNedelecDimensionFormulas) {
    // Document the correct dimension formulas for wedge Nedelec(k)
    // Edge DOFs: 9*(k+1)
    // Face DOFs: varies with k
    // Interior DOFs: varies with k

    // Nedelec(0): 9 edges = 9
    {
        NedelecBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.size(), 9u);
    }
}

TEST(HigherOrderWedgePyramid, PyramidNedelecDimensionFormulas) {
    // Document the correct dimension formulas for pyramid Nedelec(k)
    // Edge DOFs: 8*(k+1)
    // Face DOFs: varies with k
    // Interior DOFs: varies with k

    // Nedelec(0): 8 edges = 8
    {
        NedelecBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.size(), 8u);
    }
}

// =============================================================================
// RT(0) Divergence Tests - Verify Constant Divergence
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT0DivergenceIsConstant) {
    RaviartThomasBasis basis(ElementType::Wedge6, 0);

    // Two different interior points
    math::Vector<Real, 3> xi1{Real(0.2), Real(0.3), Real(-0.4)};
    math::Vector<Real, 3> xi2{Real(0.1), Real(0.1), Real(0.5)};

    std::vector<Real> div1, div2;
    basis.evaluate_divergence(xi1, div1);
    basis.evaluate_divergence(xi2, div2);

    ASSERT_EQ(div1.size(), 5u);
    ASSERT_EQ(div2.size(), 5u);

    // For RT(0), divergence should be constant over the element
    for (std::size_t i = 0; i < div1.size(); ++i) {
        EXPECT_NEAR(div1[i], div2[i], 1e-12);
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT0DivergenceIsConstant) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);

    // Two different interior points
    math::Vector<Real, 3> xi1{Real(0.1), Real(0.2), Real(0.3)};
    math::Vector<Real, 3> xi2{Real(-0.3), Real(0.1), Real(0.5)};

    std::vector<Real> div1, div2;
    basis.evaluate_divergence(xi1, div1);
    basis.evaluate_divergence(xi2, div2);

    ASSERT_EQ(div1.size(), 5u);
    ASSERT_EQ(div2.size(), 5u);

    // For RT(0), divergence should be constant over the element
    for (std::size_t i = 0; i < div1.size(); ++i) {
        EXPECT_NEAR(div1[i], div2[i], 1e-12);
    }
}

// =============================================================================
// Nedelec(0) Curl Tests - Verify curl(phi) is divergence-free
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND0CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Wedge6, 0);

    const Real h = Real(1e-3);
    math::Vector<Real, 3> xc{Real(0.25), Real(0.25), Real(0.0)};

    for (std::size_t i = 0; i < basis.size(); ++i) {
        // Approximate divergence of curl via finite differences
        auto eval_curl = [&](const math::Vector<Real, 3>& p) {
            std::vector<math::Vector<Real, 3>> c;
            basis.evaluate_curl(p, c);
            return c[i];
        };

        math::Vector<Real, 3> pxp{xc[0] + h, xc[1], xc[2]};
        math::Vector<Real, 3> pxm{xc[0] - h, xc[1], xc[2]};
        math::Vector<Real, 3> pyp{xc[0], xc[1] + h, xc[2]};
        math::Vector<Real, 3> pym{xc[0], xc[1] - h, xc[2]};
        math::Vector<Real, 3> pzp{xc[0], xc[1], xc[2] + h};
        math::Vector<Real, 3> pzm{xc[0], xc[1], xc[2] - h};

        auto cxp = eval_curl(pxp);
        auto cxm = eval_curl(pxm);
        auto cyp = eval_curl(pyp);
        auto cym = eval_curl(pym);
        auto czp = eval_curl(pzp);
        auto czm = eval_curl(pzm);

        double dCx_dx = (static_cast<double>(cxp[0] - cxm[0]) / (2.0 * static_cast<double>(h)));
        double dCy_dy = (static_cast<double>(cyp[1] - cym[1]) / (2.0 * static_cast<double>(h)));
        double dCz_dz = (static_cast<double>(czp[2] - czm[2]) / (2.0 * static_cast<double>(h)));

        double div_curl = dCx_dx + dCy_dy + dCz_dz;
        EXPECT_NEAR(div_curl, 0.0, 1e-5) << "Basis function " << i << " curl is not divergence-free";
    }
}

TEST(HigherOrderWedgePyramid, PyramidND0CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Pyramid5, 0);

    const Real h = Real(1e-3);
    math::Vector<Real, 3> xc{Real(0.0), Real(0.0), Real(0.25)};

    for (std::size_t i = 0; i < basis.size(); ++i) {
        // Approximate divergence of curl via finite differences
        auto eval_curl = [&](const math::Vector<Real, 3>& p) {
            std::vector<math::Vector<Real, 3>> c;
            basis.evaluate_curl(p, c);
            return c[i];
        };

        math::Vector<Real, 3> pxp{xc[0] + h, xc[1], xc[2]};
        math::Vector<Real, 3> pxm{xc[0] - h, xc[1], xc[2]};
        math::Vector<Real, 3> pyp{xc[0], xc[1] + h, xc[2]};
        math::Vector<Real, 3> pym{xc[0], xc[1] - h, xc[2]};
        math::Vector<Real, 3> pzp{xc[0], xc[1], xc[2] + h};
        math::Vector<Real, 3> pzm{xc[0], xc[1], xc[2] - h};

        auto cxp = eval_curl(pxp);
        auto cxm = eval_curl(pxm);
        auto cyp = eval_curl(pyp);
        auto cym = eval_curl(pym);
        auto czp = eval_curl(pzp);
        auto czm = eval_curl(pzm);

        double dCx_dx = (static_cast<double>(cxp[0] - cxm[0]) / (2.0 * static_cast<double>(h)));
        double dCy_dy = (static_cast<double>(cyp[1] - cym[1]) / (2.0 * static_cast<double>(h)));
        double dCz_dz = (static_cast<double>(czp[2] - czm[2]) / (2.0 * static_cast<double>(h)));

        double div_curl = dCx_dx + dCy_dy + dCz_dz;
        EXPECT_NEAR(div_curl, 0.0, 1e-5) << "Basis function " << i << " curl is not divergence-free";
    }
}

// =============================================================================
// RT(1) and Nedelec(1) Evaluation Tests - NOW FULLY WORKING
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT1Evaluation) {
    RaviartThomasBasis basis(ElementType::Wedge6, 1);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 24u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT1Evaluation) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 19u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, WedgeND1Evaluation) {
    NedelecBasis basis(ElementType::Wedge6, 1);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 34u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND1Evaluation) {
    NedelecBasis basis(ElementType::Pyramid5, 1);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 28u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND1CurlEvaluation) {
    NedelecBasis basis(ElementType::Pyramid5, 1);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 28u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

// =============================================================================
// RT(2) and Nedelec(2) on Wedge and Pyramid - NOW IMPLEMENTED
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT2Construction) {
    // RT(2) on Wedge: 66 DOFs
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 2);
        EXPECT_EQ(basis.size(), 66u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidRT2Construction) {
    // RT(2) on Pyramid: 57 DOFs
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 2);
        EXPECT_EQ(basis.size(), 57u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeND2Construction) {
    // Nedelec(2) on Wedge: 84 DOFs
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 2);
        EXPECT_EQ(basis.size(), 84u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidND2Construction) {
    // Nedelec(2) on Pyramid: 63 DOFs
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 2);
        EXPECT_EQ(basis.size(), 63u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeRT2Evaluation) {
    RaviartThomasBasis basis(ElementType::Wedge6, 2);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 66u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT2Evaluation) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 2);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 57u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, WedgeND2Evaluation) {
    NedelecBasis basis(ElementType::Wedge6, 2);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 84u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND2Evaluation) {
    NedelecBasis basis(ElementType::Pyramid5, 2);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 63u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

// =============================================================================
// RT(k) Divergence Tests for k=1,2
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT1DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Wedge6, 1);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 24u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT1DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 19u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT2DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Wedge6, 2);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 66u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT2DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 2);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 57u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

// =============================================================================
// Nedelec(k) Curl Tests for k=1,2
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND1CurlFinite) {
    NedelecBasis basis(ElementType::Wedge6, 1);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 34u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

TEST(HigherOrderWedgePyramid, WedgeND2CurlFinite) {
    NedelecBasis basis(ElementType::Wedge6, 2);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 84u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND2CurlFinite) {
    NedelecBasis basis(ElementType::Pyramid5, 2);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 63u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

// =============================================================================
// Moment-Based DOF Verification: RT Face-Normal Integrals
// =============================================================================
// For RT(k), the face DOF functional is:
//   dof_f(N_i) = integral_f (N_i . n) * q_m dA
// where q_m is the m-th polynomial test function on the face.
// For the zeroth mode (q_0 = 1), this should give delta_{i,f} for RT(0) DOFs,
// and for RT(1) the first face DOF per face should integrate to 1 while
// higher-order face moments and interior DOFs should integrate to 0.
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT0FaceNormalMoment) {
    // Verify RT(0) on wedge: integral of N_i . n over face j = delta_ij
    RaviartThomasBasis basis(ElementType::Wedge6, 0);
    ASSERT_EQ(basis.size(), 5u);

    constexpr int n_faces = 5;
    for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
        for (int f = 0; f < n_faces; ++f) {
            double flux = integrate_wedge_face_flux(basis, f, i);
            if (i == f) {
                EXPECT_NEAR(flux, 1.0, 1e-10)
                    << "RT(0) wedge: flux of N_" << i << " over face " << f
                    << " should be 1";
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10)
                    << "RT(0) wedge: flux of N_" << i << " over face " << f
                    << " should be 0";
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT0FaceNormalMoment) {
    // Verify RT(0) on pyramid: integral of N_i . n over face j = delta_ij
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);
    ASSERT_EQ(basis.size(), 5u);

    constexpr int n_faces = 5;
    for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
        for (int f = 0; f < n_faces; ++f) {
            double flux = integrate_pyramid_face_flux(basis, f, i);
            if (i == f) {
                EXPECT_NEAR(flux, 1.0, 1e-10)
                    << "RT(0) pyramid: flux of N_" << i << " over face " << f
                    << " should be 1";
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10)
                    << "RT(0) pyramid: flux of N_" << i << " over face " << f
                    << " should be 0";
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT1FaceNormalMoment) {
    // RT(1) on wedge has 24 DOFs: 18 face + 6 interior.
    // Verify that the RT(0)-subset DOFs (the first face DOF per face,
    // zeroth moment q=1) satisfy:
    //   integral_f (N_i . n) dA = delta_{i,f}  for i in first-face-DOFs
    // and interior DOFs have zero face flux for q=1.
    RaviartThomasBasis basis(ElementType::Wedge6, 1);
    ASSERT_EQ(basis.size(), 24u);

    auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), 24u);

    constexpr int n_faces = 5;

    // Collect the zeroth-moment face DOF index for each face.
    // These are face DOFs with moment_index == 0.
    std::vector<int> zeroth_moment_dof(n_faces, -1);
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        if (assoc[i].entity_type == DofEntity::Face &&
            assoc[i].moment_index == 0 &&
            assoc[i].entity_id >= 0 && assoc[i].entity_id < n_faces) {
            zeroth_moment_dof[static_cast<std::size_t>(assoc[i].entity_id)]
                = static_cast<int>(i);
        }
    }

    // Every face should have a zeroth-moment DOF
    for (int f = 0; f < n_faces; ++f) {
        ASSERT_GE(zeroth_moment_dof[static_cast<std::size_t>(f)], 0)
            << "Face " << f << " missing zeroth-moment DOF";
    }

    // Check that the zeroth-moment face DOFs have the correct face flux:
    // integral_f (N_i . n) dA should be 1 on its own face, 0 on others.
    for (int f = 0; f < n_faces; ++f) {
        int dof_idx = zeroth_moment_dof[static_cast<std::size_t>(f)];
        for (int g = 0; g < n_faces; ++g) {
            double flux = integrate_wedge_face_flux(basis, g, dof_idx);
            if (f == g) {
                EXPECT_NEAR(flux, 1.0, 1e-8)
                    << "Wedge RT(1): zeroth-moment DOF " << dof_idx
                    << " (face " << f << ") flux over face " << g
                    << " should be 1";
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-8)
                    << "Wedge RT(1): zeroth-moment DOF " << dof_idx
                    << " (face " << f << ") flux over face " << g
                    << " should be 0";
            }
        }
    }

    // Interior DOFs should have zero zeroth-moment face flux on all faces
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        if (assoc[i].entity_type == DofEntity::Interior) {
            for (int f = 0; f < n_faces; ++f) {
                double flux = integrate_wedge_face_flux(
                    basis, f, static_cast<int>(i));
                EXPECT_NEAR(flux, 0.0, 1e-8)
                    << "Wedge RT(1): interior DOF " << i
                    << " should have zero flux on face " << f;
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT1FaceNormalMoment) {
    // RT(1) on pyramid has 19 DOFs: 16 face + 3 interior.
    // Same verification strategy as wedge.
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);
    ASSERT_EQ(basis.size(), 19u);

    auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), 19u);

    constexpr int n_faces = 5;

    // Find zeroth-moment face DOF for each face
    std::vector<int> zeroth_moment_dof(n_faces, -1);
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        if (assoc[i].entity_type == DofEntity::Face &&
            assoc[i].moment_index == 0 &&
            assoc[i].entity_id >= 0 && assoc[i].entity_id < n_faces) {
            zeroth_moment_dof[static_cast<std::size_t>(assoc[i].entity_id)]
                = static_cast<int>(i);
        }
    }

    for (int f = 0; f < n_faces; ++f) {
        ASSERT_GE(zeroth_moment_dof[static_cast<std::size_t>(f)], 0)
            << "Face " << f << " missing zeroth-moment DOF";
    }

    // Zeroth-moment DOFs: flux over own face = 1, other faces = 0
    for (int f = 0; f < n_faces; ++f) {
        int dof_idx = zeroth_moment_dof[static_cast<std::size_t>(f)];
        for (int g = 0; g < n_faces; ++g) {
            double flux = integrate_pyramid_face_flux(basis, g, dof_idx);
            if (f == g) {
                EXPECT_NEAR(flux, 1.0, 1e-8)
                    << "Pyramid RT(1): zeroth-moment DOF " << dof_idx
                    << " (face " << f << ") flux over face " << g
                    << " should be 1";
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-8)
                    << "Pyramid RT(1): zeroth-moment DOF " << dof_idx
                    << " (face " << f << ") flux over face " << g
                    << " should be 0";
            }
        }
    }

    // Interior DOFs should have zero zeroth-moment flux on all faces
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        if (assoc[i].entity_type == DofEntity::Interior) {
            for (int f = 0; f < n_faces; ++f) {
                double flux = integrate_pyramid_face_flux(
                    basis, f, static_cast<int>(i));
                EXPECT_NEAR(flux, 0.0, 1e-8)
                    << "Pyramid RT(1): interior DOF " << i
                    << " should have zero flux on face " << f;
            }
        }
    }
}

// =============================================================================
// Moment-Based DOF Verification: Nedelec Edge-Tangent Integrals
// =============================================================================
// For Nedelec(k), the edge DOF functional is:
//   dof_e(N_i) = integral_e (N_i . t) * q_m ds
// where t is the edge tangent and q_m is the m-th polynomial mode.
// For k=0 (zeroth mode q=1), this should give delta_{i,e}.
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND0EdgeTangentMoment) {
    // Verify Nedelec(0) on wedge: integral of N_i . t along edge j = delta_ij
    NedelecBasis basis(ElementType::Wedge6, 0);
    ASSERT_EQ(basis.size(), 9u);

    constexpr int n_edges = 9;
    for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
        for (int e = 0; e < n_edges; ++e) {
            double tangent_integral = integrate_wedge_edge_tangent(basis, e, i);
            if (i == e) {
                EXPECT_NEAR(tangent_integral, 1.0, 1e-10)
                    << "ND(0) wedge: tangent integral of N_" << i
                    << " along edge " << e << " should be 1";
            } else {
                EXPECT_NEAR(tangent_integral, 0.0, 1e-10)
                    << "ND(0) wedge: tangent integral of N_" << i
                    << " along edge " << e << " should be 0";
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidND0EdgeTangentMoment) {
    // Verify Nedelec(0) on pyramid: integral of N_i . t along edge j = delta_ij
    NedelecBasis basis(ElementType::Pyramid5, 0);
    ASSERT_EQ(basis.size(), 8u);

    constexpr int n_edges = 8;
    for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
        for (int e = 0; e < n_edges; ++e) {
            double tangent_integral = integrate_pyramid_edge_tangent(basis, e, i);
            if (i == e) {
                EXPECT_NEAR(tangent_integral, 1.0, 1e-10)
                    << "ND(0) pyramid: tangent integral of N_" << i
                    << " along edge " << e << " should be 1";
            } else {
                EXPECT_NEAR(tangent_integral, 0.0, 1e-10)
                    << "ND(0) pyramid: tangent integral of N_" << i
                    << " along edge " << e << " should be 0";
            }
        }
    }
}

// =============================================================================
// RT(1) Divergence Consistency with Face Flux
// =============================================================================
// For any RT(k) basis function N_i, the divergence theorem gives:
//   integral_K div(N_i) dV = sum_f integral_f (N_i . n) dA
// Verify this identity numerically for RT(1) on both wedge and pyramid.
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT1DivergenceFluxConsistency) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    RaviartThomasBasis basis(ElementType::Wedge6, 1);
    ASSERT_EQ(basis.size(), 24u);

    constexpr int n_faces = 5;

    // Use a wedge quadrature rule to integrate divergence over the volume.
    // The reference wedge is a triangular prism: (x,y) in triangle [0,1]^2
    // with x+y<=1, z in [-1,1]. Use tensor product of TriangleQuadrature
    // and GaussQuadrature1D.
    TriangleQuadrature tri(6);
    GaussQuadrature1D gz(4);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        // Volume integral of divergence
        double vol_div = 0.0;
        for (std::size_t qt = 0; qt < tri.num_points(); ++qt) {
            for (std::size_t qz = 0; qz < gz.num_points(); ++qz) {
                const auto& pt_tri = tri.point(qt);
                const Real z = gz.point(qz)[0];
                Vector<Real,3> xi{pt_tri[0], pt_tri[1], z};

                std::vector<Real> div;
                basis.evaluate_divergence(xi, div);
                // Weight: tri_weight * gz_weight (tri already has Jacobian
                // for the Duffy transform, gz is on [-1,1])
                vol_div += static_cast<double>(tri.weight(qt))
                         * gz.weight(qz)
                         * static_cast<double>(div[i]);
            }
        }

        // Sum of face fluxes
        double face_sum = 0.0;
        for (int f = 0; f < n_faces; ++f) {
            face_sum += integrate_wedge_face_flux(
                basis, f, static_cast<int>(i));
        }

        EXPECT_NEAR(vol_div, face_sum, 1e-8)
            << "Wedge RT(1): divergence theorem mismatch for DOF " << i;
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT1DivergenceFluxConsistency) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    RaviartThomasBasis basis(ElementType::Pyramid5, 1);
    ASSERT_EQ(basis.size(), 19u);

    constexpr int n_faces = 5;

    // For the pyramid, use a simple sub-tet decomposition for volume
    // quadrature. The reference pyramid with vertices
    //   v0=(-1,-1,0), v1=(1,-1,0), v2=(1,1,0), v3=(-1,1,0), v4=(0,0,1)
    // can be split into 2 tetrahedra for integration purposes.
    // Tet A: v0, v1, v2, v4
    // Tet B: v0, v2, v3, v4
    struct Tet {
        Vector<Real,3> a, b, c, d;
    };
    const Tet tets[2] = {
        {{Real(-1),Real(-1),Real(0)}, {Real(1),Real(-1),Real(0)},
         {Real(1),Real(1),Real(0)},   {Real(0),Real(0),Real(1)}},
        {{Real(-1),Real(-1),Real(0)}, {Real(1),Real(1),Real(0)},
         {Real(-1),Real(1),Real(0)},  {Real(0),Real(0),Real(1)}}
    };

    // 4-point quadrature on the reference tetrahedron [0,1]^3 simplex
    // (barycentric coordinates). We use a simple 4-point rule.
    struct QPoint { double l0, l1, l2, l3, w; };
    const QPoint tet_qpts[4] = {
        {0.5854101966, 0.1381966011, 0.1381966011, 0.1381966011, 1.0/24.0},
        {0.1381966011, 0.5854101966, 0.1381966011, 0.1381966011, 1.0/24.0},
        {0.1381966011, 0.1381966011, 0.5854101966, 0.1381966011, 1.0/24.0},
        {0.1381966011, 0.1381966011, 0.1381966011, 0.5854101966, 1.0/24.0}
    };

    for (std::size_t i = 0; i < basis.size(); ++i) {
        double vol_div = 0.0;

        for (int t = 0; t < 2; ++t) {
            const Tet& tet = tets[t];
            // Jacobian determinant = 6 * volume of tet
            Vector<Real,3> e1 = tet.b - tet.a;
            Vector<Real,3> e2 = tet.c - tet.a;
            Vector<Real,3> e3 = tet.d - tet.a;
            double detJ = std::abs(static_cast<double>(
                e1[0]*(e2[1]*e3[2] - e2[2]*e3[1]) -
                e1[1]*(e2[0]*e3[2] - e2[2]*e3[0]) +
                e1[2]*(e2[0]*e3[1] - e2[1]*e3[0])));

            for (int q = 0; q < 4; ++q) {
                const auto& qp = tet_qpts[q];
                Vector<Real,3> xi =
                    tet.a * static_cast<Real>(qp.l0) +
                    tet.b * static_cast<Real>(qp.l1) +
                    tet.c * static_cast<Real>(qp.l2) +
                    tet.d * static_cast<Real>(qp.l3);

                std::vector<Real> div;
                basis.evaluate_divergence(xi, div);
                vol_div += qp.w * detJ * static_cast<double>(div[i]);
            }
        }

        // Sum of face fluxes
        double face_sum = 0.0;
        for (int f = 0; f < n_faces; ++f) {
            face_sum += integrate_pyramid_face_flux(
                basis, f, static_cast<int>(i));
        }

        EXPECT_NEAR(vol_div, face_sum, 1e-6)
            << "Pyramid RT(1): divergence theorem mismatch for DOF " << i;
    }
}

// =============================================================================
// Summary of supported orders
// =============================================================================
// RT on Wedge:   k=0 (5 DOFs), k=1 (24 DOFs), k=2 (66 DOFs)
// RT on Pyramid: k=0 (5 DOFs), k=1 (19 DOFs), k=2 (57 DOFs)
// ND on Wedge:   k=0 (9 DOFs), k=1 (34 DOFs), k=2 (84 DOFs)
// ND on Pyramid: k=0 (8 DOFs), k=1 (28 DOFs), k=2 (63 DOFs)
