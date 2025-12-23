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
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::basis;

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
// Summary of supported orders
// =============================================================================
// RT on Wedge:   k=0 (5 DOFs), k=1 (24 DOFs), k=2 (66 DOFs)
// RT on Pyramid: k=0 (5 DOFs), k=1 (19 DOFs), k=2 (57 DOFs)
// ND on Wedge:   k=0 (9 DOFs), k=1 (34 DOFs), k=2 (84 DOFs)
// ND on Pyramid: k=0 (8 DOFs), k=1 (28 DOFs), k=2 (63 DOFs)
