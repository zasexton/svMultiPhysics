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

TEST(BSplineBasis, GradientMatchesFiniteDifference) {
    const std::vector<std::pair<int, std::vector<Real>>> cases = {
        {2, make_open_uniform_knots(2, /*num_basis=*/6)},
        {3, make_open_uniform_knots(3, /*num_basis=*/8)},
        {2, {Real(0), Real(0), Real(0), Real(0.2), Real(0.8), Real(1), Real(1), Real(1)}},
    };

    const Real eps = Real(1e-6);
    for (const auto& [degree, knots] : cases) {
        BSplineBasis basis(degree, knots);
        const Real xi_pts[] = {Real(-0.7), Real(-0.2), Real(0.0), Real(0.3), Real(0.8)};

        for (Real xi0 : xi_pts) {
            math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
            math::Vector<Real, 3> xi_p{xi0 + eps, Real(0), Real(0)};
            math::Vector<Real, 3> xi_m{xi0 - eps, Real(0), Real(0)};

            std::vector<Gradient> grads;
            basis.evaluate_gradients(xi, grads);

            std::vector<Real> vals_p, vals_m;
            basis.evaluate_values(xi_p, vals_p);
            basis.evaluate_values(xi_m, vals_m);

            ASSERT_EQ(grads.size(), basis.size());
            for (std::size_t i = 0; i < basis.size(); ++i) {
                const Real fd = (vals_p[i] - vals_m[i]) / (Real(2) * eps);
                EXPECT_NEAR(grads[i][0], fd, 1e-5)
                    << "degree=" << degree << ", i=" << i << ", xi=" << xi0;
            }
        }
    }
}

TEST(BSplineBasis, GradientSumZero) {
    // Partition of unity implies sum of gradients = 0
    for (int degree = 1; degree <= 4; ++degree) {
        BSplineBasis basis(degree, make_open_uniform_knots(degree, degree + 4));

        const Real xi_pts[] = {Real(-0.5), Real(0.0), Real(0.5)};
        for (Real xi0 : xi_pts) {
            math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
            std::vector<Gradient> grads;
            basis.evaluate_gradients(xi, grads);

            Real sum = Real(0);
            for (const auto& g : grads) {
                sum += g[0];
            }
            EXPECT_NEAR(sum, 0.0, 1e-10)
                << "degree=" << degree << ", xi=" << xi0;
        }
    }
}

TEST(BSplineBasis, BoundaryEvaluation) {
    // At xi=-1 and xi=+1 (the domain boundaries), open B-splines should have
    // the first/last basis function equal to 1 and all others 0.
    for (int degree = 1; degree <= 3; ++degree) {
        BSplineBasis basis(degree, make_open_uniform_knots(degree, degree + 3));

        {
            math::Vector<Real, 3> xi{Real(-1), Real(0), Real(0)};
            std::vector<Real> vals;
            basis.evaluate_values(xi, vals);
            EXPECT_NEAR(vals.front(), 1.0, 1e-12) << "degree=" << degree;
            for (std::size_t i = 1; i < vals.size(); ++i) {
                EXPECT_NEAR(vals[i], 0.0, 1e-12) << "degree=" << degree << ", i=" << i;
            }
        }
        {
            math::Vector<Real, 3> xi{Real(1), Real(0), Real(0)};
            std::vector<Real> vals;
            basis.evaluate_values(xi, vals);
            EXPECT_NEAR(vals.back(), 1.0, 1e-12) << "degree=" << degree;
            for (std::size_t i = 0; i + 1 < vals.size(); ++i) {
                EXPECT_NEAR(vals[i], 0.0, 1e-12) << "degree=" << degree << ", i=" << i;
            }
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

// =============================================================================
// NURBS (rational) B-spline tests
// =============================================================================

TEST(BSplineBasis, NURBSUniformWeightsMatchNonRational) {
    const int degree = 2;
    auto knots = make_open_uniform_knots(degree, /*num_basis=*/6);
    BSplineBasis bspline(degree, knots);

    std::vector<Real> weights(6, Real(1)); // uniform weights
    BSplineBasis nurbs(degree, knots, weights);

    const Real xi_pts[] = {Real(-0.7), Real(-0.2), Real(0.0), Real(0.3), Real(0.8)};
    for (Real xi0 : xi_pts) {
        math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};

        std::vector<Real> vals_bs, vals_nurbs;
        bspline.evaluate_values(xi, vals_bs);
        nurbs.evaluate_values(xi, vals_nurbs);

        ASSERT_EQ(vals_bs.size(), vals_nurbs.size());
        for (std::size_t i = 0; i < vals_bs.size(); ++i) {
            EXPECT_NEAR(vals_bs[i], vals_nurbs[i], 1e-14)
                << "xi=" << xi0 << ", i=" << i;
        }

        std::vector<Gradient> grads_bs, grads_nurbs;
        bspline.evaluate_gradients(xi, grads_bs);
        nurbs.evaluate_gradients(xi, grads_nurbs);
        for (std::size_t i = 0; i < grads_bs.size(); ++i) {
            EXPECT_NEAR(grads_bs[i][0], grads_nurbs[i][0], 1e-12)
                << "gradient xi=" << xi0 << ", i=" << i;
        }
    }
}

TEST(BSplineBasis, NURBSPartitionOfUnity) {
    const int degree = 2;
    auto knots = make_open_uniform_knots(degree, /*num_basis=*/5);
    std::vector<Real> weights = {Real(1), Real(0.5), Real(2), Real(0.7), Real(1)};
    BSplineBasis nurbs(degree, knots, weights);

    const Real xi_pts[] = {Real(-0.9), Real(-0.3), Real(0.0), Real(0.5), Real(0.9)};
    for (Real xi0 : xi_pts) {
        math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
        std::vector<Real> vals;
        nurbs.evaluate_values(xi, vals);
        const Real sum = std::accumulate(vals.begin(), vals.end(), Real(0));
        EXPECT_NEAR(sum, Real(1), 1e-12) << "xi=" << xi0;
    }
}

TEST(BSplineBasis, NURBSGradientMatchesFiniteDifference) {
    const int degree = 2;
    auto knots = make_open_uniform_knots(degree, /*num_basis=*/5);
    std::vector<Real> weights = {Real(1), Real(0.5), Real(2), Real(0.7), Real(1)};
    BSplineBasis nurbs(degree, knots, weights);

    const Real eps = Real(1e-6);
    const Real xi_pts[] = {Real(-0.5), Real(0.0), Real(0.4)};
    for (Real xi0 : xi_pts) {
        math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
        math::Vector<Real, 3> xi_p{xi0 + eps, Real(0), Real(0)};
        math::Vector<Real, 3> xi_m{xi0 - eps, Real(0), Real(0)};

        std::vector<Gradient> grads;
        nurbs.evaluate_gradients(xi, grads);

        std::vector<Real> vp, vm;
        nurbs.evaluate_values(xi_p, vp);
        nurbs.evaluate_values(xi_m, vm);

        for (std::size_t i = 0; i < nurbs.size(); ++i) {
            const Real fd = (vp[i] - vm[i]) / (Real(2) * eps);
            EXPECT_NEAR(grads[i][0], fd, 1e-5)
                << "NURBS gradient xi=" << xi0 << ", i=" << i;
        }
    }
}

TEST(BSplineBasis, NURBSGradientSumZero) {
    const int degree = 3;
    auto knots = make_open_uniform_knots(degree, /*num_basis=*/7);
    std::vector<Real> weights = {Real(1), Real(0.3), Real(1.5), Real(2),
                                 Real(0.8), Real(1.2), Real(1)};
    BSplineBasis nurbs(degree, knots, weights);

    const Real xi_pts[] = {Real(-0.5), Real(0.0), Real(0.5)};
    for (Real xi0 : xi_pts) {
        math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
        std::vector<Gradient> grads;
        nurbs.evaluate_gradients(xi, grads);

        Real sum = Real(0);
        for (const auto& g : grads) {
            sum += g[0];
        }
        EXPECT_NEAR(sum, 0.0, 1e-10) << "xi=" << xi0;
    }
}

TEST(BSplineBasis, NURBSWeightsSizeMismatchThrows) {
    const int degree = 2;
    auto knots = make_open_uniform_knots(degree, /*num_basis=*/5);
    std::vector<Real> bad_weights = {Real(1), Real(1)}; // wrong size
    EXPECT_THROW(BSplineBasis(degree, knots, bad_weights), svmp::FE::FEException);
}
