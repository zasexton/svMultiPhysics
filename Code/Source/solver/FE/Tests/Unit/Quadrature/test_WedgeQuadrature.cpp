/**
 * @file test_WedgeQuadrature.cpp
 * @brief Unit tests for wedge/prism quadrature
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/WedgeQuadrature.h"
#include <numeric>

using namespace svmp::FE::quadrature;

namespace {
bool point_inside_wedge(const QuadPoint& p) {
    // Triangle in (x,y): x>=0, y>=0, x+y<=1; z in [-1,1]
    return p[0] >= -1e-12 && p[1] >= -1e-12 && (p[0] + p[1] <= 1.0 + 1e-12) &&
           (p[2] >= -1.0 - 1e-12) && (p[2] <= 1.0 + 1e-12);
}
}

TEST(WedgeQuadrature, WeightSumMatchesVolume) {
    WedgeQuadrature quad(3);
    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12); // area 0.5 * length 2
}

TEST(WedgeQuadrature, IntegratesLinearMonomials) {
    WedgeQuadrature quad(4);
    double ix = 0.0;
    double iz = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        double w = quad.weight(i);
        ix += w * p[0];
        iz += w * p[2];
    }
    // ∫_triangle x dA = 1/6, times line length 2 = 1/3
    EXPECT_NEAR(ix, 1.0 / 3.0, 1e-12);
    // z is odd over [-1,1], integral should be zero
    EXPECT_NEAR(iz, 0.0, 1e-14);
}

TEST(WedgeQuadrature, PointsInsideReferenceElement) {
    WedgeQuadrature quad(3);
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        EXPECT_TRUE(point_inside_wedge(quad.point(i)));
    }
}
