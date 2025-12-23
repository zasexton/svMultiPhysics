/**
 * @file test_BernsteinBasis.cpp
 * @brief Unit tests for Bernstein bases
 */

#include <gtest/gtest.h>
#include "FE/Basis/BernsteinBasis.h"
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::basis;

TEST(BernsteinBasis, LinePartitionOfUnity) {
    BernsteinBasis basis(ElementType::Line2, 3);
    svmp::FE::math::Vector<Real, 3> xi{0.1, 0.0, 0.0};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(BernsteinBasis, TrianglePartitionOfUnity) {
    BernsteinBasis basis(ElementType::Triangle3, 2);
    svmp::FE::math::Vector<Real, 3> xi{0.3, 0.2, 0.0};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(BernsteinBasis, HexPartitionOfUnity) {
    BernsteinBasis basis(ElementType::Hex8, 2);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.3)};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    ASSERT_EQ(vals.size(), 27u); // (order+1)^3
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(BernsteinBasis, WedgePartitionOfUnity) {
    BernsteinBasis basis(ElementType::Wedge6, 2);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.1), Real(-0.3)};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(BernsteinBasis, PyramidPartitionOfUnity) {
    BernsteinBasis basis(ElementType::Pyramid5, 2);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.4)};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}
