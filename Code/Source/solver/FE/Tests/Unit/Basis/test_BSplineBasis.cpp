/**
 * @file test_BSplineBasis.cpp
 * @brief Unit tests for BSplineBasis
 */

#include <gtest/gtest.h>

#include "FE/Basis/BSplineBasis.h"

#include <numeric>
#include <random>

using namespace svmp::FE;
using namespace svmp::FE::basis;

namespace {

std::vector<Real> make_open_uniform_knots(int degree,
                                          int num_basis,
                                          Real u0 = Real(0),
                                          Real u1 = Real(1)) {
    std::vector<Real> knots;
    knots.reserve(static_cast<std::size_t>(num_basis + degree + 1));

    for (int i = 0; i < degree + 1; ++i) {
        knots.push_back(u0);
    }

    const int interior = num_basis - degree - 1;
    for (int j = 1; j <= interior; ++j) {
        knots.push_back(u0 + (u1 - u0) * Real(j) / Real(num_basis - degree));
    }

    for (int i = 0; i < degree + 1; ++i) {
        knots.push_back(u1);
    }

    return knots;
}

} // namespace

TEST(BSplineBasis, PartitionOfUnity) {
    struct Case {
        int degree;
        std::vector<Real> knots;
    };

    std::vector<Case> cases;
    for (int degree = 1; degree <= 4; ++degree) {
        cases.push_back({degree, make_open_uniform_knots(degree, /*num_basis=*/degree + 4)});
    }
    cases.push_back({2, {Real(0), Real(0), Real(0), Real(0.2), Real(0.8), Real(1), Real(1), Real(1)}});

    std::mt19937 rng(12345);
    std::uniform_real_distribution<Real> dist(Real(-0.999), Real(0.999));

    const Real tol = Real(1e-12);
    for (const auto& c : cases) {
        BSplineBasis basis(c.degree, c.knots);
        for (int sample = 0; sample < 50; ++sample) {
            const Real xi0 = dist(rng);
            math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};

            std::vector<Real> values;
            basis.evaluate_values(xi, values);
            ASSERT_EQ(values.size(), basis.size());

            const Real sum = std::accumulate(values.begin(), values.end(), Real(0));
            EXPECT_NEAR(sum, Real(1), tol);
        }
    }
}

TEST(BSplineBasis, LocalSupport) {
    struct Case {
        int degree;
        std::vector<Real> knots;
    };

    const std::vector<Case> cases = {
        {1, make_open_uniform_knots(1, /*num_basis=*/6)},
        {2, make_open_uniform_knots(2, /*num_basis=*/8)},
        {3, make_open_uniform_knots(3, /*num_basis=*/9)},
        {2, {Real(0),
             Real(0),
             Real(0),
             Real(0.2),
             Real(0.5),
             Real(0.7),
             Real(1),
             Real(1),
             Real(1)}},
    };

    const Real tol = Real(1e-14);
    for (const auto& c : cases) {
        BSplineBasis basis(c.degree, c.knots);
        const auto& knots = basis.knots();
        const int degree = basis.order();
        const int num_basis = static_cast<int>(basis.size());

        const Real u_min = knots[static_cast<std::size_t>(degree)];
        const Real u_max = knots[static_cast<std::size_t>(num_basis)];
        ASSERT_GT(num_basis, degree + 1);
        ASSERT_GT(u_max, u_min);
        const auto u_to_xi = [&](Real u) -> Real {
            return Real(2) * (u - u_min) / (u_max - u_min) - Real(1);
        };

        for (int i = 0; i < num_basis; ++i) {
            const Real support_start = knots[static_cast<std::size_t>(i)];
            const Real support_end = knots[static_cast<std::size_t>(i + degree + 1)];

            bool tested = false;
            if (support_start > u_min) {
                const Real u = (u_min + support_start) * Real(0.5);
                math::Vector<Real, 3> xi{u_to_xi(u), Real(0), Real(0)};

                std::vector<Real> values;
                basis.evaluate_values(xi, values);
                ASSERT_EQ(values.size(), basis.size());
                EXPECT_NEAR(values[static_cast<std::size_t>(i)], Real(0), tol);
                tested = true;
            }

            if (support_end < u_max) {
                const Real u = (support_end + u_max) * Real(0.5);
                math::Vector<Real, 3> xi{u_to_xi(u), Real(0), Real(0)};

                std::vector<Real> values;
                basis.evaluate_values(xi, values);
                ASSERT_EQ(values.size(), basis.size());
                EXPECT_NEAR(values[static_cast<std::size_t>(i)], Real(0), tol);
                tested = true;
            }

            ASSERT_TRUE(tested) << "No domain points outside support for i=" << i
                                << " (degree=" << degree << ")";
        }
    }
}
