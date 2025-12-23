/**
 * @file test_VectorBases.cpp
 * @brief Tests for H(div)/H(curl) vector bases
 */

#include <gtest/gtest.h>
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Core/FEException.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include <algorithm>
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::basis;

// Forward declarations for helper integrators used in edge-based DOF tests.
static double integrate_wedge_edge_dof(const NedelecBasis& basis,
                                       int edge_id,
                                       int func_id,
                                       int quad_order = 4);

static double integrate_pyramid_edge_dof(const NedelecBasis& basis,
                                         int edge_id,
                                         int func_id,
                                         int quad_order = 4);

static double integrate_triangle_edge_tangential_moment(const NedelecBasis& basis,
                                                        int edge_id,
                                                        int mode_id,
                                                        int func_id,
                                                        int order,
                                                        int quad_order = 8);

static double integrate_triangle_edge_normal_moment(const RaviartThomasBasis& basis,
                                                    int edge_id,
                                                    int mode_id,
                                                    int func_id,
                                                    int order,
                                                    int quad_order = 8);

static double integrate_tetra_edge_tangential_moment(const NedelecBasis& basis,
                                                     int edge_id,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order = 8);

static double integrate_tetra_face_tangential_moment(const NedelecBasis& basis,
                                                     int face_id,
                                                     int dir_id,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order = 8);

TEST(RaviartThomasBasis, DivergenceConstants) {
    RaviartThomasBasis basis(ElementType::Quad4, 0);
    svmp::FE::math::Vector<Real, 3> xi{0.0, 0.0, 0.0};
    std::vector<svmp::FE::math::Vector<Real, 3>> values;
    basis.evaluate_vector_values(xi, values);
    ASSERT_EQ(values.size(), 4u);
    EXPECT_NEAR(values[0][0], 0.5, 1e-12);

    std::vector<Real> div;
    basis.evaluate_divergence(xi, div);
    ASSERT_EQ(div.size(), 4u);
    EXPECT_NEAR(div[0], 0.5, 1e-12);
    EXPECT_NEAR(div[1], -0.5, 1e-12);
}

TEST(NedelecBasis, CurlSigns) {
    NedelecBasis basis(ElementType::Quad4, 0);
    svmp::FE::math::Vector<Real, 3> xi{0.1, -0.2, 0.0};
    EXPECT_TRUE(basis.is_vector_valued());

    std::vector<svmp::FE::math::Vector<Real, 3>> curl;
    basis.evaluate_curl(xi, curl);
    ASSERT_EQ(curl.size(), 4u);
    for (const auto& c : curl) {
        EXPECT_NEAR(c[0], 0.0, 1e-14);
        EXPECT_NEAR(c[1], 0.0, 1e-14);
        EXPECT_NEAR(c[2], 0.25, 1e-12);
    }
}

TEST(BDMBasis, DivergenceLinear) {
    BDMBasis basis(ElementType::Quad4, 1);
    svmp::FE::math::Vector<Real, 3> xi{0.3, -0.4, 0.0};
    std::vector<Real> div;
    basis.evaluate_divergence(xi, div);
    ASSERT_EQ(div.size(), 8u);
    EXPECT_NEAR(div[0], 0.5, 1e-12);
    EXPECT_NEAR(div[1], 0.5 * xi[0], 1e-12);
    EXPECT_NEAR(div[2], 0.5, 1e-12);
    EXPECT_NEAR(div[3], 0.5 * xi[1], 1e-12);
    EXPECT_NEAR(div[4], 0.5, 1e-12);
    EXPECT_NEAR(div[5], 0.5 * xi[0], 1e-12);
    EXPECT_NEAR(div[6], 0.5, 1e-12);
    EXPECT_NEAR(div[7], 0.5 * xi[1], 1e-12);
}

TEST(BDMBasis, TriangleDivergenceLinear) {
    BDMBasis basis(ElementType::Triangle3, 1);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.4), Real(0)};
    std::vector<Real> div;
    basis.evaluate_divergence(xi, div);
    ASSERT_EQ(div.size(), 6u);
    EXPECT_NEAR(div[0], 2.0, 1e-12);
    EXPECT_NEAR(div[1], 0.0, 1e-12);
    EXPECT_NEAR(div[2], 2.0, 1e-12);
    EXPECT_NEAR(div[3], 0.0, 1e-12);
    EXPECT_NEAR(div[4], 2.0, 1e-12);
    EXPECT_NEAR(div[5], 0.0, 1e-12);
}

TEST(VectorBasis, ScalarEvaluateThrows) {
    RaviartThomasBasis basis(ElementType::Quad4, 0);
    svmp::FE::math::Vector<Real, 3> xi{0.0, 0.0, 0.0};
    std::vector<Real> vals;
    EXPECT_THROW(basis.evaluate_values(xi, vals), svmp::FE::FEException);
}

TEST(VectorBasis, HexahedronInstantiationAndSize) {
    RaviartThomasBasis rt_hex(ElementType::Hex8, 0);
    NedelecBasis ned_hex(ElementType::Hex8, 0);
    EXPECT_EQ(rt_hex.dimension(), 3);
    EXPECT_EQ(ned_hex.dimension(), 3);
    EXPECT_EQ(rt_hex.size(), 6u);
    EXPECT_EQ(ned_hex.size(), 12u);
}

TEST(VectorBasis, WedgeAndPyramidInstantiationAndSize) {
    RaviartThomasBasis rt_wedge(ElementType::Wedge6, 0);
    EXPECT_EQ(rt_wedge.dimension(), 3);
    EXPECT_EQ(rt_wedge.size(), 5u);

    NedelecBasis ned_wedge(ElementType::Wedge6, 0);
    EXPECT_EQ(ned_wedge.dimension(), 3);
    EXPECT_EQ(ned_wedge.size(), 9u);

    RaviartThomasBasis rt_pyr(ElementType::Pyramid5, 0);
    EXPECT_EQ(rt_pyr.dimension(), 3);
    EXPECT_EQ(rt_pyr.size(), 5u);

    NedelecBasis ned_pyr(ElementType::Pyramid5, 0);
    EXPECT_EQ(ned_pyr.dimension(), 3);
    EXPECT_EQ(ned_pyr.size(), 8u);
}

TEST(RaviartThomasBasis, PyramidRT0DivergenceConstant) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);
    svmp::FE::math::Vector<Real, 3> xi1{Real(0.1), Real(0.2), Real(0.3)};
    svmp::FE::math::Vector<Real, 3> xi2{Real(-0.3), Real(0.1), Real(0.5)};
    std::vector<Real> div1, div2;
    basis.evaluate_divergence(xi1, div1);
    basis.evaluate_divergence(xi2, div2);
    ASSERT_EQ(div1.size(), 5u);
    ASSERT_EQ(div2.size(), 5u);
    for (std::size_t i = 0; i < div1.size(); ++i) {
        EXPECT_NEAR(div1[i], div2[i], 1e-14);
    }
}

TEST(VectorBasis, TriangleAndTetraInstantiationAndSize) {
    RaviartThomasBasis rt_tri(ElementType::Triangle3, 0);
    EXPECT_EQ(rt_tri.dimension(), 2);
    EXPECT_EQ(rt_tri.size(), 3u);

    RaviartThomasBasis rt_tet(ElementType::Tetra4, 0);
    EXPECT_EQ(rt_tet.dimension(), 3);
    EXPECT_EQ(rt_tet.size(), 4u);

    NedelecBasis ned_tri(ElementType::Triangle3, 0);
    EXPECT_EQ(ned_tri.dimension(), 2);
    EXPECT_EQ(ned_tri.size(), 3u);

    NedelecBasis ned_tet(ElementType::Tetra4, 0);
    EXPECT_EQ(ned_tet.dimension(), 3);
    EXPECT_EQ(ned_tet.size(), 6u);
}

TEST(RaviartThomasBasis, TriangleAndTetraDivergenceConstants) {
    RaviartThomasBasis rt_tri(ElementType::Triangle3, 0);
    svmp::FE::math::Vector<Real, 3> xi_tri1{Real(0.2), Real(0.3), Real(0)};
    svmp::FE::math::Vector<Real, 3> xi_tri2{Real(0.1), Real(0.2), Real(0)};
    std::vector<Real> div_tri1, div_tri2;
    rt_tri.evaluate_divergence(xi_tri1, div_tri1);
    rt_tri.evaluate_divergence(xi_tri2, div_tri2);
    ASSERT_EQ(div_tri1.size(), 3u);
    ASSERT_EQ(div_tri2.size(), 3u);
    for (std::size_t i = 0; i < div_tri1.size(); ++i) {
        EXPECT_NEAR(div_tri1[i], div_tri2[i], 1e-14);
    }

    RaviartThomasBasis rt_tet(ElementType::Tetra4, 0);
    svmp::FE::math::Vector<Real, 3> xi_tet1{Real(0.1), Real(0.2), Real(0.1)};
    svmp::FE::math::Vector<Real, 3> xi_tet2{Real(0.2), Real(0.1), Real(0.2)};
    std::vector<Real> div_tet1, div_tet2;
    rt_tet.evaluate_divergence(xi_tet1, div_tet1);
    rt_tet.evaluate_divergence(xi_tet2, div_tet2);
    ASSERT_EQ(div_tet1.size(), 4u);
    ASSERT_EQ(div_tet2.size(), 4u);
    for (std::size_t i = 0; i < div_tet1.size(); ++i) {
        EXPECT_NEAR(div_tet1[i], div_tet2[i], 1e-14);
    }
}

TEST(RaviartThomasBasis, WedgeDivergenceConstants) {
    RaviartThomasBasis basis(ElementType::Wedge6, 0);
    svmp::FE::math::Vector<Real, 3> xi1{Real(0.2), Real(0.3), Real(-0.4)};
    svmp::FE::math::Vector<Real, 3> xi2{Real(0.1), Real(0.1), Real(0.5)};
    std::vector<Real> div1, div2;
    basis.evaluate_divergence(xi1, div1);
    basis.evaluate_divergence(xi2, div2);
    ASSERT_EQ(div1.size(), 5u);
    ASSERT_EQ(div2.size(), 5u);
    for (std::size_t i = 0; i < div1.size(); ++i) {
        EXPECT_NEAR(div1[i], div2[i], 1e-14);
    }
}

TEST(NedelecBasis, TriangleAndTetraCurls) {
    NedelecBasis ned_tri(ElementType::Triangle3, 0);
    svmp::FE::math::Vector<Real, 3> xi_tri{Real(0.2), Real(0.3), Real(0)};
    std::vector<svmp::FE::math::Vector<Real, 3>> curl_tri;
    ned_tri.evaluate_curl(xi_tri, curl_tri);
    ASSERT_EQ(curl_tri.size(), 3u);
    for (const auto& c : curl_tri) {
        EXPECT_GT(c[2], 1.5); // all have positive z-curl
    }

    NedelecBasis ned_tet(ElementType::Tetra4, 0);
    svmp::FE::math::Vector<Real, 3> xi_tet{Real(0.1), Real(0.2), Real(0.1)};
    std::vector<svmp::FE::math::Vector<Real, 3>> curl_tet;
    ned_tet.evaluate_curl(xi_tet, curl_tet);
    ASSERT_EQ(curl_tet.size(), 6u);
    // Reuse existing expectations on hex-like patterns: non-zero components present
    int nonzero = 0;
    for (const auto& c : curl_tet) {
        if (std::abs(c[0]) > 1e-14 || std::abs(c[1]) > 1e-14 || std::abs(c[2]) > 1e-14) {
            ++nonzero;
        }
    }
    EXPECT_GE(nonzero, 3);
}

TEST(NedelecBasis, TriangleOrderOneEdgeDofsAreKronecker) {
    constexpr int k = 1;
    NedelecBasis basis(ElementType::Triangle3, k);
    ASSERT_EQ(basis.size(), 8u);

    const int edge_dofs = 3 * (k + 1);
    for (int e = 0; e < 3; ++e) {
        for (int a = 0; a <= k; ++a) {
            const int dof_idx = e * (k + 1) + a;
            for (std::size_t j = 0; j < basis.size(); ++j) {
                const double dof = integrate_triangle_edge_tangential_moment(
                    basis, e, a, static_cast<int>(j), k);
                if (static_cast<int>(j) == dof_idx) {
                    EXPECT_NEAR(dof, 1.0, 1e-10);
                } else if (static_cast<int>(j) < edge_dofs) {
                    EXPECT_NEAR(dof, 0.0, 1e-10);
                }
            }
        }
    }
}

TEST(NedelecBasis, TetraOrderOneEdgeDofsAreKronecker) {
    constexpr int k = 1;
    NedelecBasis basis(ElementType::Tetra4, k);
    ASSERT_EQ(basis.size(), 20u);

    const int edge_dofs = 6 * (k + 1);
    for (int e = 0; e < 6; ++e) {
        for (int a = 0; a <= k; ++a) {
            const int dof_idx = e * (k + 1) + a;
            for (std::size_t j = 0; j < basis.size(); ++j) {
                const double dof = integrate_tetra_edge_tangential_moment(
                    basis, e, a, static_cast<int>(j), k);
                if (static_cast<int>(j) == dof_idx) {
                    EXPECT_NEAR(dof, 1.0, 5e-10);
                } else if (static_cast<int>(j) < edge_dofs) {
                    EXPECT_NEAR(dof, 0.0, 5e-10);
                }
            }
        }
    }
}

TEST(NedelecBasis, TetraOrderOneFaceDofsAreKronecker) {
    constexpr int k = 1;
    NedelecBasis basis(ElementType::Tetra4, k);
    ASSERT_EQ(basis.size(), 20u);

    const int edge_dofs = 6 * (k + 1);
    const int face_dofs = 4 * 2; // 4 faces, 2 tangential DOFs each for k=1

    for (int f = 0; f < 4; ++f) {
        for (int dir = 0; dir < 2; ++dir) {
            const int dof_idx = edge_dofs + 2 * f + dir;
            for (std::size_t j = 0; j < basis.size(); ++j) {
                const double dof = integrate_tetra_face_tangential_moment(
                    basis, f, dir, /*mode_id=*/0, static_cast<int>(j), k);
                if (static_cast<int>(j) == dof_idx) {
                    EXPECT_NEAR(dof, 1.0, 5e-9);
                } else if (static_cast<int>(j) >= edge_dofs &&
                           static_cast<int>(j) < edge_dofs + face_dofs) {
                    EXPECT_NEAR(dof, 0.0, 5e-9);
                }
            }
        }
    }
}

TEST(RaviartThomasBasis, TriangleOrderOneEdgeDofsAreKronecker) {
    constexpr int k = 1;
    RaviartThomasBasis basis(ElementType::Triangle3, k);
    ASSERT_EQ(basis.size(), 8u);

    const int edge_dofs = 3 * (k + 1);
    for (int e = 0; e < 3; ++e) {
        for (int a = 0; a <= k; ++a) {
            const int dof_idx = e * (k + 1) + a;
            for (std::size_t j = 0; j < basis.size(); ++j) {
                const double dof = integrate_triangle_edge_normal_moment(
                    basis, e, a, static_cast<int>(j), k);
                if (static_cast<int>(j) == dof_idx) {
                    EXPECT_NEAR(dof, 1.0, 1e-10);
                } else if (static_cast<int>(j) < edge_dofs) {
                    EXPECT_NEAR(dof, 0.0, 1e-10);
                }
            }
        }
    }
}

TEST(RaviartThomasBasis, HexDivergenceConstants) {
    RaviartThomasBasis basis(ElementType::Hex8, 0);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(-0.1), Real(0.3)};

    std::vector<Real> div;
    basis.evaluate_divergence(xi, div);
    ASSERT_EQ(div.size(), 6u);
    EXPECT_NEAR(div[0], 0.25, 1e-12);
    EXPECT_NEAR(div[1], -0.25, 1e-12);
    EXPECT_NEAR(div[2], 0.25, 1e-12);
    EXPECT_NEAR(div[3], -0.25, 1e-12);
    EXPECT_NEAR(div[4], 0.25, 1e-12);
    EXPECT_NEAR(div[5], -0.25, 1e-12);

    std::vector<svmp::FE::math::Vector<Real, 3>> vals;
    basis.evaluate_vector_values(xi, vals);
    ASSERT_EQ(vals.size(), 6u);
    EXPECT_NEAR(vals[0][0], 0.25 * (Real(1) + xi[0]), 1e-12);
    EXPECT_NEAR(vals[1][0], 0.25 * (Real(1) - xi[0]), 1e-12);
    EXPECT_NEAR(vals[2][1], 0.25 * (Real(1) + xi[1]), 1e-12);
    EXPECT_NEAR(vals[3][1], 0.25 * (Real(1) - xi[1]), 1e-12);
    EXPECT_NEAR(vals[4][2], 0.25 * (Real(1) + xi[2]), 1e-12);
    EXPECT_NEAR(vals[5][2], 0.25 * (Real(1) - xi[2]), 1e-12);
    EXPECT_NEAR(vals[0][1], 0.0, 1e-12);
    EXPECT_NEAR(vals[0][2], 0.0, 1e-12);
    EXPECT_NEAR(vals[1][1], 0.0, 1e-12);
    EXPECT_NEAR(vals[1][2], 0.0, 1e-12);
    EXPECT_NEAR(vals[2][0], 0.0, 1e-12);
    EXPECT_NEAR(vals[2][2], 0.0, 1e-12);
    EXPECT_NEAR(vals[3][0], 0.0, 1e-12);
    EXPECT_NEAR(vals[3][2], 0.0, 1e-12);
    EXPECT_NEAR(vals[4][0], 0.0, 1e-12);
    EXPECT_NEAR(vals[4][1], 0.0, 1e-12);
    EXPECT_NEAR(vals[5][0], 0.0, 1e-12);
    EXPECT_NEAR(vals[5][1], 0.0, 1e-12);
}

TEST(NedelecBasis, HexCurlConstants) {
    NedelecBasis basis(ElementType::Hex8, 0);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.15), Real(-0.05), Real(0.4)};

    std::vector<svmp::FE::math::Vector<Real, 3>> curl;
    basis.evaluate_curl(xi, curl);
    ASSERT_EQ(curl.size(), 12u);
    // Spot-check a few curls against the analytical expressions.
    // Edge 0-1: v = (1/8)(1-y)(1-z) e_x
    EXPECT_NEAR(curl[0][1], -0.125 * (Real(1) - xi[1]), 1e-12);
    EXPECT_NEAR(curl[0][2],  0.125 * (Real(1) - xi[2]), 1e-12);
    // Edge 1-2: v = (1/8)(1+x)(1-z) e_y
    EXPECT_NEAR(curl[1][0],  0.125 * (Real(1) + xi[0]), 1e-12);
    EXPECT_NEAR(curl[1][2],  0.125 * (Real(1) - xi[2]), 1e-12);
    // Vertical edge 0-4: v = (1/8)(1-x)(1-y) e_z
    EXPECT_NEAR(curl[8][0],  0.125 * (Real(1) - xi[0]), 1e-12);
    EXPECT_NEAR(curl[8][1], -0.125 * (Real(1) - xi[1]), 1e-12);

    std::vector<svmp::FE::math::Vector<Real, 3>> vals;
    basis.evaluate_vector_values(xi, vals);
    ASSERT_EQ(vals.size(), 12u);
    // Each edge function has exactly one nonzero component.
    for (std::size_t i = 0; i < vals.size(); ++i) {
        const int nonzero =
            (std::abs(vals[i][0]) > Real(1e-14)) +
            (std::abs(vals[i][1]) > Real(1e-14)) +
            (std::abs(vals[i][2]) > Real(1e-14));
        EXPECT_EQ(nonzero, 1);
    }
}

TEST(NedelecBasis, WedgeMinimalEdgeDofs) {
    NedelecBasis basis(ElementType::Wedge6, 0);

    for (int i = 0; i < 9; ++i) {
        for (int e = 0; e < 9; ++e) {
            double dof = integrate_wedge_edge_dof(basis, e, i);
            if (i == e) {
                EXPECT_NEAR(dof, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(dof, 0.0, 1e-10);
            }
        }
    }
}

TEST(NedelecBasis, PyramidMinimalEdgeDofs) {
    NedelecBasis basis(ElementType::Pyramid5, 0);

    for (int i = 0; i < 8; ++i) {
        for (int e = 0; e < 8; ++e) {
            double dof = integrate_pyramid_edge_dof(basis, e, i);
            if (i == e) {
                EXPECT_NEAR(dof, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(dof, 0.0, 1e-10);
            }
        }
    }
}

TEST(NedelecBasis, PyramidCurlsAreDivergenceFree) {
    std::unique_ptr<NedelecBasis> basis = std::make_unique<NedelecBasis>(ElementType::Pyramid5, 0);

    using svmp::FE::math::Vector;
    const Real h = Real(1e-3);
    // Choose an interior point well inside the pyramid
    Vector<Real,3> xc{Real(0), Real(0), Real(0.25)};

    for (int i = 0; i < 8; ++i) {
        // Approximate divergence of curl φ_i at xc via finite differences
        // ∂curl_x/∂x + ∂curl_y/∂y + ∂curl_z/∂z
        auto eval_curl = [&](const Vector<Real,3>& p) {
            std::vector<Vector<Real,3>> c;
            basis->evaluate_curl(p, c);
            return c[static_cast<std::size_t>(i)];
        };

        Vector<Real,3> c0 = eval_curl(xc);

        Vector<Real,3> pxp{xc[0] + h, xc[1], xc[2]};
        Vector<Real,3> pxm{xc[0] - h, xc[1], xc[2]};
        Vector<Real,3> cxp = eval_curl(pxp);
        Vector<Real,3> cxm = eval_curl(pxm);

        Vector<Real,3> pyp{xc[0], xc[1] + h, xc[2]};
        Vector<Real,3> pym{xc[0], xc[1] - h, xc[2]};
        Vector<Real,3> cyp = eval_curl(pyp);
        Vector<Real,3> cym = eval_curl(pym);

        Vector<Real,3> pzp{xc[0], xc[1], xc[2] + h};
        Vector<Real,3> pzm{xc[0], xc[1], xc[2] - h};
        Vector<Real,3> czp = eval_curl(pzp);
        Vector<Real,3> czm = eval_curl(pzm);

        double dCx_dx = (static_cast<double>(cxp[0] - cxm[0]) / (2.0 * static_cast<double>(h)));
        double dCy_dy = (static_cast<double>(cyp[1] - cym[1]) / (2.0 * static_cast<double>(h)));
        double dCz_dz = (static_cast<double>(czp[2] - czm[2]) / (2.0 * static_cast<double>(h)));

        double div_curl = dCx_dx + dCy_dy + dCz_dz;
        EXPECT_NEAR(div_curl, 0.0, 1e-6);
        (void)c0;
    }
}

static double integrate_edge_normal_flux(const RaviartThomasBasis& basis,
                                         int edge_id,
                                         const svmp::FE::math::Vector<Real, 3>& xi_fixed,
                                         int varying_dim) {
    using svmp::FE::quadrature::GaussQuadrature1D;
    GaussQuadrature1D quad(4);
    double flux = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        auto p = xi_fixed;
        p[static_cast<std::size_t>(varying_dim)] = quad.point(q)[0];

        std::vector<svmp::FE::math::Vector<Real, 3>> v;
        basis.evaluate_vector_values(p, v);
        const svmp::FE::math::Vector<Real, 3>& f = v[static_cast<std::size_t>(edge_id)];

        // Outward normals for quad edges: +x, -x, +y, -y
        svmp::FE::math::Vector<Real, 3> n{};
        if (edge_id == 0) n[0] = 1.0;
        else if (edge_id == 1) n[0] = -1.0;
        else if (edge_id == 2) n[1] = 1.0;
        else n[1] = -1.0;

        flux += quad.weight(q) * (f[0] * n[0] + f[1] * n[1]);
    }
    return flux;
}

TEST(RaviartThomasBasis, QuadEdgeFluxes) {
    RaviartThomasBasis basis(ElementType::Quad4, 0);

    // Edge parameterizations: x=+/-1 vary y; y=+/-1 vary x
    struct Edge {
        int id;
        svmp::FE::math::Vector<Real, 3> xi;
        int varying_dim;
    };
    std::vector<Edge> edges = {
        {0, {Real(1), Real(0), Real(0)}, 1},   // x=+1
        {1, {Real(-1), Real(0), Real(0)}, 1},  // x=-1
        {2, {Real(0), Real(1), Real(0)}, 0},   // y=+1
        {3, {Real(0), Real(-1), Real(0)}, 0},  // y=-1
    };

    for (const auto& e : edges) {
        double own_flux = integrate_edge_normal_flux(basis, e.id, e.xi, e.varying_dim);
        EXPECT_GT(std::abs(own_flux), 1.5); // dominant on its edge

        for (int other = 0; other < 4; ++other) {
            if (other == e.id) continue;
            double flux = integrate_edge_normal_flux(basis, other, e.xi, e.varying_dim);
            EXPECT_LT(std::abs(flux), std::abs(own_flux));
        }
    }
}

static double integrate_wedge_face_flux(const RaviartThomasBasis& basis,
                                        int face_id,
                                        int func_id,
                                        int tri_order = 4,
                                        int quad_order = 4) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    double flux = 0.0;

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

    QuadrilateralQuadrature quad(quad_order, quad_order);
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& pt = quad.point(q);
        const double xi = pt[0];
        const double eta = pt[1];

        Vector<Real,3> x_ref{};
        Vector<Real,3> cross{};

        if (face_id == 2) { // y=0
            const double t = 0.5 * (xi + 1.0);
            const double z = eta;
            x_ref = Vector<Real,3>{static_cast<Real>(t), Real(0), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(0), Real(-0.5), Real(0)};
        } else if (face_id == 3) { // x=0
            const double t = 0.5 * (xi + 1.0);
            const double z = eta;
            x_ref = Vector<Real,3>{Real(0), static_cast<Real>(t), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(0.5), Real(0), Real(0)};
        } else { // face_id == 4, x+y=1
            const double t = 0.5 * (xi + 1.0);
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

static double integrate_triangle_edge_tangential_moment(const NedelecBasis& basis,
                                                        int edge_id,
                                                        int mode_id,
                                                        int func_id,
                                                        int order,
                                                        int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    const Vector<Real,3> v0{Real(0), Real(0), Real(0)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(0)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(0)};

    struct Edge { Vector<Real,3> a; Vector<Real,3> b; };
    const Edge edges[3] = {{v0, v1}, {v1, v2}, {v2, v0}};
    const Edge& e = edges[static_cast<std::size_t>(edge_id)];

    const Vector<Real,3> tvec = e.b - e.a;
    const Real len = tvec.norm();
    EXPECT_GT(len, Real(0));
    if (len <= Real(0)) {
        return 0.0;
    }
    const Vector<Real,3> t = tvec / len;
    const Real J = len * Real(0.5);

    LagrangeBasis line(ElementType::Line2, order);
    GaussQuadrature1D quad(quad_order);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        std::vector<Real> lvals;
        line.evaluate_values(Vector<Real,3>{s, Real(0), Real(0)}, lvals);
        const Real l = lvals[static_cast<std::size_t>(mode_id)];

        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real,3> xi = e.a * (Real(1) - tpar) + e.b * tpar;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];
        const Real dot = w.dot(t);

        moment += static_cast<double>(quad.weight(q) * (J * l * dot));
    }
    return moment;
}

static double integrate_triangle_edge_normal_moment(const RaviartThomasBasis& basis,
                                                    int edge_id,
                                                    int mode_id,
                                                    int func_id,
                                                    int order,
                                                    int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    const Vector<Real,3> v0{Real(0), Real(0), Real(0)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(0)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(0)};

    struct Edge { Vector<Real,3> a; Vector<Real,3> b; };
    const Edge edges[3] = {{v0, v1}, {v1, v2}, {v2, v0}};
    const Edge& e = edges[static_cast<std::size_t>(edge_id)];

    const Vector<Real,3> tvec = e.b - e.a;
    const Real len = tvec.norm();
    EXPECT_GT(len, Real(0));
    if (len <= Real(0)) {
        return 0.0;
    }
    const Vector<Real,3> t = tvec / len;
    const Vector<Real,3> n{t[1], -t[0], Real(0)}; // outward for CCW edge ordering
    const Real J = len * Real(0.5);

    LagrangeBasis line(ElementType::Line2, order);
    GaussQuadrature1D quad(quad_order);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        std::vector<Real> lvals;
        line.evaluate_values(Vector<Real,3>{s, Real(0), Real(0)}, lvals);
        const Real l = lvals[static_cast<std::size_t>(mode_id)];

        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real,3> xi = e.a * (Real(1) - tpar) + e.b * tpar;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];
        const Real dot = w.dot(n);

        moment += static_cast<double>(quad.weight(q) * (J * l * dot));
    }
    return moment;
}

static double integrate_tetra_edge_tangential_moment(const NedelecBasis& basis,
                                                     int edge_id,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    const Vector<Real,3> v0{Real(0), Real(0), Real(0)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(0)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(0)};
    const Vector<Real,3> v3{Real(0), Real(0), Real(1)};

    struct Edge { Vector<Real,3> a; Vector<Real,3> b; };
    const Edge edges[6] = {
        {v0, v1}, {v1, v2}, {v2, v0},
        {v0, v3}, {v1, v3}, {v2, v3}
    };
    const Edge& e = edges[static_cast<std::size_t>(edge_id)];

    const Vector<Real,3> tvec = e.b - e.a;
    const Real len = tvec.norm();
    EXPECT_GT(len, Real(0));
    if (len <= Real(0)) {
        return 0.0;
    }
    const Vector<Real,3> t = tvec / len;
    const Real J = len * Real(0.5);

    const int q_order = std::max(2 * order + 2, quad_order);
    const auto quad = QuadratureFactory::create(
        ElementType::Line2, q_order, QuadratureType::GaussLegendre, /*use_cache=*/false);

    LagrangeBasis line(ElementType::Line2, order);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const Real s = quad->point(q)[0];
        std::vector<Real> lvals;
        line.evaluate_values(Vector<Real,3>{s, Real(0), Real(0)}, lvals);
        const Real l = lvals[static_cast<std::size_t>(mode_id)];

        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real,3> xi = e.a * (Real(1) - tpar) + e.b * tpar;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];
        const Real dot = w.dot(t);

        moment += static_cast<double>(quad->weight(q) * (J * l * dot));
    }
    return moment;
}

static double integrate_tetra_face_tangential_moment(const NedelecBasis& basis,
                                                     int face_id,
                                                     int dir_id,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    const Vector<Real,3> v0{Real(0), Real(0), Real(0)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(0)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(0)};
    const Vector<Real,3> v3{Real(0), Real(0), Real(1)};

    struct Face { Vector<Real,3> a; Vector<Real,3> b; Vector<Real,3> c; };
    const Face faces[4] = {
        {v0, v1, v2},
        {v0, v1, v3},
        {v1, v2, v3},
        {v0, v2, v3}
    };
    const Face& f = faces[static_cast<std::size_t>(face_id)];

    const Vector<Real,3> tu = f.b - f.a;
    const Vector<Real,3> tv = f.c - f.a;

    Vector<Real,3> cross{};
    cross[0] = tu[1] * tv[2] - tu[2] * tv[1];
    cross[1] = tu[2] * tv[0] - tu[0] * tv[2];
    cross[2] = tu[0] * tv[1] - tu[1] * tv[0];
    const Real scale = cross.norm();
    EXPECT_GT(scale, Real(0));
    if (scale <= Real(0)) {
        return 0.0;
    }

    const int q_order = std::max(2 * order + 2, quad_order);
    const auto quad = QuadratureFactory::create(
        ElementType::Triangle3, q_order, QuadratureType::GaussLegendre, /*use_cache=*/false);

    LagrangeBasis face_basis(ElementType::Triangle3, order - 1);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto pt = quad->point(q);
        const Real u = pt[0];
        const Real v = pt[1];

        std::vector<Real> bvals;
        face_basis.evaluate_values(Vector<Real,3>{u, v, Real(0)}, bvals);
        const Real phi = bvals[static_cast<std::size_t>(mode_id)];

        const Vector<Real,3> xi = f.a + tu * u + tv * v;
        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        const Vector<Real,3>& t = (dir_id == 0) ? tu : tv;
        const Real dot = w.dot(t);

        moment += static_cast<double>(quad->weight(q) * (scale * phi * dot));
    }
    return moment;
}

static double integrate_wedge_edge_dof(const NedelecBasis& basis,
                                       int edge_id,
                                       int func_id,
                                       int quad_order) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    // Reference wedge nodes (matching NodeOrderingConventions and RT0 wedge tests)
    const Vector<Real,3> v0{Real(0), Real(0), Real(-1)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(-1)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(-1)};
    const Vector<Real,3> v3{Real(0), Real(0), Real(1)};
    const Vector<Real,3> v4{Real(1), Real(0), Real(1)};
    const Vector<Real,3> v5{Real(0), Real(1), Real(1)};

    struct Edge {
        Vector<Real,3> a;
        Vector<Real,3> b;
    };
    const Edge edges[9] = {
        {v0, v1}, // bottom
        {v1, v2},
        {v2, v0},
        {v3, v4}, // top
        {v4, v5},
        {v5, v3},
        {v0, v3}, // vertical
        {v1, v4},
        {v2, v5}
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

        const Real wx = w[0];
        const Real wy = w[1];
        const Real wz = w[2];
        const Real tx = t_vec[0];
        const Real ty = t_vec[1];
        const Real tz = t_vec[2];

        const double integrand = static_cast<double>(wx*tx + wy*ty + wz*tz);
        const double ds = 0.5 * quad.weight(q);
        dof += ds * integrand;
    }
    return dof;
}

static double integrate_pyramid_edge_dof(const NedelecBasis& basis,
                                         int edge_id,
                                         int func_id,
                                         int quad_order) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    // Reference Pyramid5 vertices (matching NodeOrderingConventions)
    const Vector<Real,3> v0{Real(-1), Real(-1), Real(0)};
    const Vector<Real,3> v1{Real(1),  Real(-1), Real(0)};
    const Vector<Real,3> v2{Real(1),  Real(1),  Real(0)};
    const Vector<Real,3> v3{Real(-1), Real(1),  Real(0)};
    const Vector<Real,3> v4{Real(0),  Real(0),  Real(1)};

    struct Edge {
        Vector<Real,3> a;
        Vector<Real,3> b;
    };

    const Edge edges[8] = {
        {v0, v1}, // base edges
        {v1, v2},
        {v2, v3},
        {v3, v0},
        {v0, v4}, // vertical edges
        {v1, v4},
        {v2, v4},
        {v3, v4}
    };

    const Edge& e = edges[static_cast<std::size_t>(edge_id)];
    const Vector<Real,3> t_vec = e.b - e.a;

    GaussQuadrature1D quad(quad_order);
    double dof = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real tq = quad.point(q)[0];   // [-1,1]
        const Real s  = (tq + Real(1)) * Real(0.5); // [0,1]
        Vector<Real,3> xi = e.a * (Real(1) - s) + e.b * s;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        const Real wx = w[0];
        const Real wy = w[1];
        const Real wz = w[2];
        const Real tx = t_vec[0];
        const Real ty = t_vec[1];
        const Real tz = t_vec[2];

        const double integrand = static_cast<double>(wx*tx + wy*ty + wz*tz);
        const double dt = 0.5 * quad.weight(q);
        dof += dt * integrand;
    }

    return dof;
}

TEST(RaviartThomasBasis, WedgeMinimalFaceFluxDofs) {
    RaviartThomasBasis basis(ElementType::Wedge6, 0);
    constexpr int n_faces = 5;
    constexpr int n_funcs = 5;

    for (int i = 0; i < n_funcs; ++i) {
        for (int j = 0; j < n_faces; ++j) {
            double flux = integrate_wedge_face_flux(basis, j, i);
            if (i == j) {
                EXPECT_NEAR(flux, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10);
            }
        }
    }
}


static double integrate_pyramid_face_flux(const RaviartThomasBasis& basis,
                                          int face_id,
                                          int func_id,
                                          int tri_order = 4,
                                          int quad_order = 4) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    double flux = 0.0;

    // Reference pyramid vertices
    const Vector<Real,3> v0{Real(-1), Real(-1), Real(0)};
    const Vector<Real,3> v1{Real(1),  Real(-1), Real(0)};
    const Vector<Real,3> v2{Real(1),  Real(1),  Real(0)};
    const Vector<Real,3> v3{Real(-1), Real(1),  Real(0)};
    const Vector<Real,3> v4{Real(0),  Real(0),  Real(1)};

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

TEST(RaviartThomasBasis, PyramidMinimalFaceFluxDofs) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);
    constexpr int n_faces = 5;
    constexpr int n_funcs = 5;

    for (int i = 0; i < n_funcs; ++i) {
        for (int j = 0; j < n_faces; ++j) {
            double flux = integrate_pyramid_face_flux(basis, j, i);
            if (i == j) {
                EXPECT_NEAR(flux, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10);
            }
        }
    }
}

// =============================================================================
// Higher-order Wedge/Pyramid Vector Element Tests
// =============================================================================

TEST(RaviartThomasBasis, WedgeHigherOrderConstruction) {
    // Test that RT(k) for wedge can be constructed
    // k=0 works with analytical formulas
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 5u);
    });

    // k=1 now works with direct DOF-based construction
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 1);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 1);
        EXPECT_EQ(basis.size(), 24u);  // RT(1) on wedge: 18 face + 6 interior
    });
}

TEST(RaviartThomasBasis, PyramidHigherOrderConstruction) {
    // Test that RT(k) for pyramid can be constructed
    // k=0 works with analytical formulas
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 5u);
    });

    // k=1 now works with direct DOF-based construction
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 1);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 1);
        EXPECT_EQ(basis.size(), 19u);  // RT(1) on pyramid: 16 face + 3 interior
    });
}

TEST(RaviartThomasBasis, WedgeRT1Size) {
    // Wedge RT(k) DOF counts
    // RT(0): 5 faces (2 tri + 3 quad with 1 DOF each)
    RaviartThomasBasis rt0(ElementType::Wedge6, 0);
    EXPECT_EQ(rt0.size(), 5u);

    // RT(1): 24 DOFs (18 face + 6 interior)
    RaviartThomasBasis rt1(ElementType::Wedge6, 1);
    EXPECT_EQ(rt1.size(), 24u);
}

TEST(RaviartThomasBasis, PyramidRT1Size) {
    // Pyramid RT(k) DOF counts
    // RT(0): 5 faces (1 quad + 4 tri)
    RaviartThomasBasis rt0(ElementType::Pyramid5, 0);
    EXPECT_EQ(rt0.size(), 5u);

    // RT(1): 19 DOFs (16 face + 3 interior)
    RaviartThomasBasis rt1(ElementType::Pyramid5, 1);
    EXPECT_EQ(rt1.size(), 19u);
}

TEST(RaviartThomasBasis, WedgeRT1Evaluation) {
    // Test that RT(1) on wedge can be evaluated
    RaviartThomasBasis basis(ElementType::Wedge6, 1);

    // Evaluate at a test point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};
    std::vector<svmp::FE::math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 24u);

    // Test divergence evaluation
    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 24u);
}

TEST(RaviartThomasBasis, PyramidRT1Evaluation) {
    // Test that RT(1) on pyramid can be evaluated
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);

    // Evaluate at a test point (inside pyramid)
    svmp::FE::math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};
    std::vector<svmp::FE::math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 19u);

    // Test divergence evaluation
    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 19u);
}

TEST(NedelecBasis, WedgeHigherOrderConstruction) {
    // Test that Nedelec(k) for wedge can be constructed
    // k=0 works with analytical formulas
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 9u);  // 9 edges
    });

    // k=1 now works with direct DOF-based construction
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 1);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 1);
        EXPECT_EQ(basis.size(), 34u);  // 18 edge + 16 face
    });
}

TEST(NedelecBasis, PyramidHigherOrderConstruction) {
    // Test that Nedelec(k) for pyramid can be constructed
    // k=0 works with analytical formulas
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 8u);  // 8 edges
    });

    // k=1 construction works but evaluation throws NotImplemented
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 1);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 1);
        // Note: evaluation will throw NotImplemented for now
    });
}

TEST(NedelecBasis, WedgeND1Size) {
    // Wedge Nedelec(k) DOF counts
    // ND(0): 9 edges
    NedelecBasis nd0(ElementType::Wedge6, 0);
    EXPECT_EQ(nd0.size(), 9u);

    // ND(1): 34 DOFs (18 edge + 16 face)
    NedelecBasis nd1(ElementType::Wedge6, 1);
    EXPECT_EQ(nd1.size(), 34u);
}

TEST(NedelecBasis, PyramidND1Size) {
    // Pyramid Nedelec(k) DOF counts
    // ND(0): 8 edges
    NedelecBasis nd0(ElementType::Pyramid5, 0);
    EXPECT_EQ(nd0.size(), 8u);

    // ND(1) construction works (evaluation throws NotImplemented)
    NedelecBasis nd1(ElementType::Pyramid5, 1);
    EXPECT_EQ(nd1.size(), 28u);  // 16 edge + 12 face
}

TEST(NedelecBasis, WedgeND1Evaluation) {
    // Test that Nedelec(1) on wedge can be evaluated
    NedelecBasis basis(ElementType::Wedge6, 1);

    // Evaluate at a test point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};
    std::vector<svmp::FE::math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 34u);

    // Test curl evaluation (via finite differences)
    std::vector<svmp::FE::math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 34u);
}

TEST(NedelecBasis, PyramidND1Evaluation) {
    // Test that Nedelec(1) on pyramid can be evaluated (direct DOF construction)
    NedelecBasis basis(ElementType::Pyramid5, 1);

    // Evaluate at a test point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};
    std::vector<svmp::FE::math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 28u);  // Pyramid ND(1) has 28 DOFs

    // Test curl evaluation (via finite differences)
    std::vector<svmp::FE::math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 28u);
}

// =============================================================================
// Dimension Verification Tests
// =============================================================================
// The dimension formulas for higher-order wedge/pyramid RT and Nedelec elements.
// NOTE: Higher-order (k>=1) tests now enabled via direct DOF construction.
// See test_HigherOrderWedgePyramid.cpp for comprehensive k=0,1,2 coverage.

TEST(RaviartThomasBasis, WedgeDimensionFormulas) {
    // Wedge RT(k) dimensions (theoretical):
    // Face DOFs: 2*(k+1)(k+2)/2 + 3*(k+1)^2 = (k+1)(k+2) + 3(k+1)^2
    // Interior DOFs: 3*k*(k+1)^2/2 for k >= 1
    // RT(0): 5 (5 faces, no interior)
    // RT(1): 18 face + 6 interior = 24 (construction still fails)
    // RT(2): 39 face + 24 interior = 63 (construction still fails)

    RaviartThomasBasis rt0(ElementType::Wedge6, 0);
    EXPECT_EQ(rt0.size(), 5u);
    // Higher-order tests disabled until singular matrix issue is fixed
}

TEST(RaviartThomasBasis, PyramidDimensionFormulas) {
    // Pyramid RT(k) dimensions (theoretical):
    // Face DOFs: (k+1)^2 + 4*(k+1)(k+2)/2 = (k+1)^2 + 2(k+1)(k+2)
    // Interior DOFs: 3*k^3 for k >= 1
    // RT(0): 5 (5 faces, no interior)
    // RT(1): 16 face + 3 interior = 19 (construction still fails)
    // RT(2): 33 face + 24 interior = 57 (construction still fails)

    RaviartThomasBasis rt0(ElementType::Pyramid5, 0);
    EXPECT_EQ(rt0.size(), 5u);
    // Higher-order tests disabled until singular matrix issue is fixed
}

TEST(NedelecBasis, WedgeDimensionFormulas) {
    // Wedge Nedelec(k) dimensions (theoretical):
    // Edge DOFs: 9*(k+1)
    // Face DOFs: 2*k(k+1) + 3*2k(k+1) = 8k(k+1) for k >= 1
    // Interior DOFs: 3*k*(k-1)*(k+1)/2 for k >= 2
    // ND(0): 9 edges
    // ND(1): 18 edge + 16 face + 0 interior = 34 (construction still fails)
    // ND(2): 27 edge + 48 face + 9 interior = 84 (construction still fails)

    NedelecBasis nd0(ElementType::Wedge6, 0);
    EXPECT_EQ(nd0.size(), 9u);
    // Higher-order tests disabled until singular matrix issue is fixed
}

TEST(NedelecBasis, PyramidDimensionFormulas) {
    // Pyramid Nedelec(k) dimensions (theoretical):
    // Edge DOFs: 8*(k+1)
    // Face DOFs: 2k(k+1) + 4*k(k+1) = 6k(k+1) for k >= 1
    // Interior DOFs: 3*k*(k-1)*(k+1)/6 for k >= 2
    // ND(0): 8 edges
    // ND(1): 16 edge + 12 face + 0 interior = 28 (construction still fails)
    // ND(2): 24 edge + 36 face + 3 interior = 63 (construction still fails)

    NedelecBasis nd0(ElementType::Pyramid5, 0);
    EXPECT_EQ(nd0.size(), 8u);
    // Higher-order tests disabled until singular matrix issue is fixed
}

// =============================================================================
// DOF Association Tests
// =============================================================================

TEST(RaviartThomasBasis, DofAssociationsTriangle) {
    RaviartThomasBasis basis(ElementType::Triangle3, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());

    // For RT(1) on triangle: 3 edges x 2 DOFs per edge + interior
    int edge_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Edge) {
            ++edge_dofs;
        } else if (a.entity_type == DofEntity::Interior) {
            ++interior_dofs;
        }
    }
    EXPECT_EQ(edge_dofs, 6); // 3 edges x 2 DOFs
    EXPECT_GE(interior_dofs, 0);
}

TEST(NedelecBasis, DofAssociationsTriangle) {
    NedelecBasis basis(ElementType::Triangle3, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());

    // For Nedelec(1) on triangle: 3 edges x 2 DOFs per edge + interior
    int edge_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Edge) {
            ++edge_dofs;
        }
    }
    EXPECT_EQ(edge_dofs, 6); // 3 edges x 2 DOFs
}

TEST(RaviartThomasBasis, DofAssociationsTetra) {
    RaviartThomasBasis basis(ElementType::Tetra4, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());

    // For RT(1) on tetrahedron: 4 faces with face DOFs
    int face_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Face) {
            ++face_dofs;
        }
    }
    EXPECT_GT(face_dofs, 0);
}

TEST(NedelecBasis, DofAssociationsTetra) {
    NedelecBasis basis(ElementType::Tetra4, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());

    // For Nedelec(1) on tetrahedron: 6 edges with edge DOFs + face DOFs
    int edge_dofs = 0;
    int face_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Edge) {
            ++edge_dofs;
        } else if (a.entity_type == DofEntity::Face) {
            ++face_dofs;
        }
    }
    EXPECT_EQ(edge_dofs, 12); // 6 edges x 2 DOFs
    EXPECT_GT(face_dofs, 0);
}

TEST(BDMBasis, DofAssociationsTriangle) {
    BDMBasis basis(ElementType::Triangle3, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 6u); // BDM1 on triangle has 6 DOFs

    // All should be edge DOFs for BDM1
    for (const auto& a : assoc) {
        EXPECT_EQ(a.entity_type, DofEntity::Edge);
    }
}

TEST(RaviartThomasBasis, DofAssociationsWedge) {
    // Test RT(0) on wedge instead of RT(1) due to known singular matrix issue
    RaviartThomasBasis basis(ElementType::Wedge6, 0);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 5u); // RT(0) on wedge has 5 face DOFs

    int face_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Face) face_dofs++;
        else if (a.entity_type == DofEntity::Interior) interior_dofs++;
    }
    EXPECT_EQ(face_dofs, 5);
    EXPECT_EQ(interior_dofs, 0);
}

TEST(NedelecBasis, DofAssociationsPyramid) {
    // Test ND(0) on pyramid instead of ND(1) due to known singular matrix issue
    NedelecBasis basis(ElementType::Pyramid5, 0);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 8u); // ND(0) on pyramid has 8 edge DOFs

    int edge_dofs = 0;
    int face_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Edge) edge_dofs++;
        else if (a.entity_type == DofEntity::Face) face_dofs++;
        else if (a.entity_type == DofEntity::Interior) interior_dofs++;
    }
    EXPECT_EQ(edge_dofs, 8); // 8 edges * 1 DOF each
    EXPECT_EQ(face_dofs, 0);
    EXPECT_EQ(interior_dofs, 0);
}

