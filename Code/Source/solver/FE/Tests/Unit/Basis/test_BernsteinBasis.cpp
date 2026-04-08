/**
 * @file test_BernsteinBasis.cpp
 * @brief Unit tests for Bernstein bases
 */

#include <gtest/gtest.h>
#include "FE/Basis/BernsteinBasis.h"
#include <cmath>
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::basis;

namespace {

using Point = svmp::FE::math::Vector<Real, 3>;

std::vector<Point> boundary_stress_points_for(ElementType type) {
    switch (type) {
        case ElementType::Line2:
            return {
                Point{Real(-0.98), Real(0), Real(0)},
                Point{Real(-0.2), Real(0), Real(0)},
                Point{Real(0.94), Real(0), Real(0)}
            };
        case ElementType::Triangle3:
            return {
                Point{Real(0.01), Real(0.01), Real(0)},
                Point{Real(0.97), Real(0.02), Real(0)},
                Point{Real(0.03), Real(0.94), Real(0)},
                Point{Real(0.49), Real(0.49), Real(0)}
            };
        case ElementType::Quad4:
            return {
                Point{Real(-0.97), Real(-0.95), Real(0)},
                Point{Real(0.96), Real(-0.9), Real(0)},
                Point{Real(-0.92), Real(0.94), Real(0)},
                Point{Real(0.93), Real(0.91), Real(0)}
            };
        case ElementType::Hex8:
            return {
                Point{Real(-0.95), Real(-0.94), Real(-0.92)},
                Point{Real(0.93), Real(-0.91), Real(0.88)},
                Point{Real(-0.9), Real(0.89), Real(0.86)},
                Point{Real(0.88), Real(0.9), Real(-0.87)}
            };
        case ElementType::Wedge6:
            return {
                Point{Real(0.01), Real(0.01), Real(-0.95)},
                Point{Real(0.96), Real(0.02), Real(0.9)},
                Point{Real(0.03), Real(0.93), Real(-0.85)},
                Point{Real(0.49), Real(0.49), Real(0.8)}
            };
        case ElementType::Pyramid5:
            return {
                Point{Real(0.0), Real(0.0), Real(0.98)},
                Point{Real(0.02), Real(-0.015), Real(0.95)},
                Point{Real(0.92), Real(-0.88), Real(-0.9)},
                Point{Real(-0.85), Real(0.9), Real(-0.85)}
            };
        default:
            return {Point{Real(0), Real(0), Real(0)}};
    }
}

void expect_gradient_entries_finite(const std::vector<Gradient>& grads, int dimension) {
    for (const auto& grad : grads) {
        for (int d = 0; d < dimension; ++d) {
            EXPECT_TRUE(std::isfinite(grad[static_cast<std::size_t>(d)]));
        }
    }
}

} // namespace

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

TEST(BernsteinBasis, DeterministicBoundarySweepMaintainsPartitionNonnegativeAndFiniteGradients) {
    const struct Case {
        ElementType type;
        int order;
    } cases[] = {
        {ElementType::Line2, 4},
        {ElementType::Triangle3, 3},
        {ElementType::Quad4, 3},
        {ElementType::Hex8, 2},
        {ElementType::Wedge6, 2},
        {ElementType::Pyramid5, 2},
    };

    for (const auto& c : cases) {
        BernsteinBasis basis(c.type, c.order);
        for (const auto& xi : boundary_stress_points_for(c.type)) {
            std::vector<Real> values;
            std::vector<Gradient> grads;
            basis.evaluate_values(xi, values);
            basis.evaluate_gradients(xi, grads);

            ASSERT_EQ(values.size(), basis.size());
            ASSERT_EQ(grads.size(), basis.size());

            Real sum = Real(0);
            Gradient grad_sum{};
            for (std::size_t i = 0; i < values.size(); ++i) {
                EXPECT_TRUE(std::isfinite(values[i]));
                EXPECT_GE(values[i], Real(-1e-12))
                    << "type=" << static_cast<int>(c.type)
                    << ", order=" << c.order << ", i=" << i;
                sum += values[i];
                for (int d = 0; d < basis.dimension(); ++d) {
                    grad_sum[static_cast<std::size_t>(d)] += grads[i][static_cast<std::size_t>(d)];
                }
            }

            expect_gradient_entries_finite(grads, basis.dimension());
            EXPECT_NEAR(sum, Real(1), c.type == ElementType::Pyramid5 ? Real(1e-9) : Real(1e-12));
            for (int d = 0; d < basis.dimension(); ++d) {
                EXPECT_NEAR(grad_sum[static_cast<std::size_t>(d)], Real(0),
                            c.type == ElementType::Pyramid5 ? Real(1e-8) : Real(1e-10));
            }
        }
    }
}
