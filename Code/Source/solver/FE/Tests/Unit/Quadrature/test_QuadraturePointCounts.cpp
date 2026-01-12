/**
 * @file test_QuadraturePointCounts.cpp
 * @brief Regression tests for quadrature point counts and efficiency.
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/GaussLobattoQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/HexahedronQuadrature.h"
#include "FE/Quadrature/WedgeQuadrature.h"
#include "FE/Quadrature/PyramidQuadrature.h"
#include "FE/Quadrature/SymmetricTriangleQuadrature.h"
#include "FE/Quadrature/SymmetricTetrahedronQuadrature.h"

#include <algorithm>
#include <map>

using namespace svmp::FE;
using namespace svmp::FE::quadrature;

namespace {
int points_for_order(int order, QuadratureType type) {
    if (order < 1) {
        order = 1;
    }
    switch (type) {
        case QuadratureType::GaussLobatto:
            return std::max(2, (order + 4) / 2);
        default:
            return std::max(1, (order + 2) / 2);
    }
}
} // namespace

TEST(QuadraturePointCounts, MatchExpected) {
    // 1D
    for (int order = 1; order <= 15; ++order) {
        GaussQuadrature1D line(points_for_order(order, QuadratureType::GaussLegendre));
        EXPECT_EQ(line.num_points(), static_cast<std::size_t>(points_for_order(order, QuadratureType::GaussLegendre)));

        GaussLobattoQuadrature1D lob(points_for_order(order, QuadratureType::GaussLobatto));
        EXPECT_EQ(lob.num_points(), static_cast<std::size_t>(points_for_order(order, QuadratureType::GaussLobatto)));
    }

    // Triangle / tetrahedron (Duffy)
    for (int order = 1; order <= 12; ++order) {
        TriangleQuadrature tri(order);
        const int n_tri = std::max(1, (order + 3) / 2);
        EXPECT_EQ(tri.num_points(), static_cast<std::size_t>(n_tri * n_tri));

        TetrahedronQuadrature tet(order);
        const int n_tet = std::max(1, (order + 4) / 2);
        EXPECT_EQ(tet.num_points(), static_cast<std::size_t>(n_tet * n_tet * n_tet));
    }

    // Tensor-product quad/hex
    for (int order = 1; order <= 12; ++order) {
        {
            QuadrilateralQuadrature quad(order, QuadratureType::GaussLegendre);
            const int n = points_for_order(order, QuadratureType::GaussLegendre);
            EXPECT_EQ(quad.num_points(), static_cast<std::size_t>(n * n));
        }
        {
            QuadrilateralQuadrature quad(order, QuadratureType::GaussLobatto);
            const int n = points_for_order(order, QuadratureType::GaussLobatto);
            EXPECT_EQ(quad.num_points(), static_cast<std::size_t>(n * n));
        }
        {
            HexahedronQuadrature hex(order, QuadratureType::GaussLegendre);
            const int n = points_for_order(order, QuadratureType::GaussLegendre);
            EXPECT_EQ(hex.num_points(), static_cast<std::size_t>(n * n * n));
        }
        {
            HexahedronQuadrature hex(order, QuadratureType::GaussLobatto);
            const int n = points_for_order(order, QuadratureType::GaussLobatto);
            EXPECT_EQ(hex.num_points(), static_cast<std::size_t>(n * n * n));
        }
    }

    // Wedge: triangle x line (isotropic order)
    for (int order = 1; order <= 12; ++order) {
        WedgeQuadrature wedge(order, QuadratureType::GaussLegendre);
        const int n_tri = std::max(1, (order + 3) / 2);
        const int n_line = std::max(1, (order + 2) / 2);
        EXPECT_EQ(wedge.num_points(), static_cast<std::size_t>(n_tri * n_tri * n_line));
    }

    // Pyramid: (a,b) Gauss-Legendre x weighted t Gauss-Legendre
    for (int order = 1; order <= 12; ++order) {
        PyramidQuadrature pyr(order);
        const int n_ab = std::max(1, (order + 2) / 2);
        const int n_t  = std::max(1, (order + 5) / 2);
        EXPECT_EQ(pyr.num_points(), static_cast<std::size_t>(n_ab * n_ab * n_t));
    }

    // Symmetric rules: spot-check canonical point counts.
    const std::map<int, std::size_t> expected_dunavant = {
        {1, 1},   {2, 3},   {3, 4},   {4, 6},   {5, 7},
        {6, 12},  {7, 13},  {8, 16},  {9, 19},  {10, 25},
        {11, 27}, {12, 33}, {13, 37}, {14, 42}, {15, 48},
        {16, 52}, {17, 61}, {18, 70}, {19, 73}, {20, 79}
    };
    for (const auto& [order, expected] : expected_dunavant) {
        SymmetricTriangleQuadrature quad(order);
        EXPECT_EQ(quad.num_points(), expected);
    }

    const std::map<int, std::size_t> expected_keast = {
        {1, 1}, {2, 4}, {3, 5}, {4, 11}, {5, 15}, {6, 24}, {7, 31}, {8, 45}
    };
    for (const auto& [order, expected] : expected_keast) {
        SymmetricTetrahedronQuadrature quad(order);
        EXPECT_EQ(quad.num_points(), expected);
    }
}

TEST(SymmetricRules, FewerPointsThanTensorProduct) {
    // Triangles: Dunavant should be at most as large as Duffy tensor rules.
    for (int order = 1; order <= 20; ++order) {
        SymmetricTriangleQuadrature sym(order);
        TriangleQuadrature tensor(order);
        EXPECT_LE(sym.num_points(), tensor.num_points()) << "order=" << order;
    }

    // Tetrahedra: Keast tabulated rules (1-8) should be strictly smaller.
    for (int order = 2; order <= 8; ++order) {
        SymmetricTetrahedronQuadrature sym(order);
        TetrahedronQuadrature tensor(order);
        EXPECT_LT(sym.num_points(), tensor.num_points()) << "order=" << order;
    }
}

