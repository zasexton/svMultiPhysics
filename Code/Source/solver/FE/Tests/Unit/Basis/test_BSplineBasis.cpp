/**
 * @file test_BSplineBasis.cpp
 * @brief Unit tests for BSplineBasis
 */

#include <gtest/gtest.h>

#include "FE/Basis/BSplineBasis.h"
#include "FE/Basis/TensorBasis.h"

#include <cmath>
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

Real map_u_to_xi(const BSplineBasis& basis, Real u) {
    const auto& knots = basis.knots();
    const Real u_min = knots[static_cast<std::size_t>(basis.order())];
    const Real u_max = knots[basis.size()];
    return Real(2) * (u - u_min) / (u_max - u_min) - Real(1);
}

template <typename BasisLike>
void numerical_gradient_helper(const BasisLike& basis,
                               const math::Vector<Real, 3>& xi,
                               std::vector<Gradient>& grads,
                               Real eps = Real(1e-6)) {
    std::vector<Real> vals_p;
    std::vector<Real> vals_m;
    grads.assign(basis.size(), Gradient{});

    for (int d = 0; d < basis.dimension(); ++d) {
        math::Vector<Real, 3> xi_p = xi;
        math::Vector<Real, 3> xi_m = xi;
        xi_p[static_cast<std::size_t>(d)] += eps;
        xi_m[static_cast<std::size_t>(d)] -= eps;

        basis.evaluate_values(xi_p, vals_p);
        basis.evaluate_values(xi_m, vals_m);
        ASSERT_EQ(vals_p.size(), basis.size());
        ASSERT_EQ(vals_m.size(), basis.size());

        for (std::size_t i = 0; i < basis.size(); ++i) {
            grads[i][static_cast<std::size_t>(d)] =
                (vals_p[i] - vals_m[i]) / (Real(2) * eps);
        }
    }
}

template <typename BasisLike>
void expect_gradients_match_numerical(const BasisLike& basis,
                                      const math::Vector<Real, 3>& xi,
                                      Real tol) {
    std::vector<Gradient> analytical;
    std::vector<Gradient> numerical;
    basis.evaluate_gradients(xi, analytical);
    numerical_gradient_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t i = 0; i < analytical.size(); ++i) {
        for (int d = 0; d < basis.dimension(); ++d) {
            const std::size_t sd = static_cast<std::size_t>(d);
            EXPECT_NEAR(analytical[i][sd], numerical[i][sd], tol)
                << "basis=" << i << ", dim=" << d
                << ", xi=(" << xi[0] << "," << xi[1] << "," << xi[2] << ")";
        }
    }
}

template <typename BasisLike>
void expect_partition_of_unity_and_finite(const BasisLike& basis,
                                          const math::Vector<Real, 3>& xi,
                                          Real tol = Real(1e-12)) {
    std::vector<Real> values;
    std::vector<Gradient> grads;
    basis.evaluate_values(xi, values);
    basis.evaluate_gradients(xi, grads);

    ASSERT_EQ(values.size(), basis.size());
    ASSERT_EQ(grads.size(), basis.size());

    Real sum = Real(0);
    for (Real value : values) {
        EXPECT_TRUE(std::isfinite(value));
        sum += value;
    }
    EXPECT_NEAR(sum, Real(1), tol);

    for (const auto& grad : grads) {
        for (int d = 0; d < basis.dimension(); ++d) {
            EXPECT_TRUE(std::isfinite(grad[static_cast<std::size_t>(d)]));
        }
    }
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

TEST(BSplineBasis, NonUniformKnotsMatchReferenceValues) {
    // Linear non-uniform B-spline with open knot vector U = [0,0,0.25,0.75,1,1].
    // The piecewise basis values are simple enough to verify independently:
    // on [0,0.25):  [1-4u, 4u, 0, 0]
    // on [0.25,0.75): [0, 1.5-2u, 2u-0.5, 0]
    // on [0.75,1]: [0, 0, 4(1-u), 4u-3]
    BSplineBasis basis(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});

    struct Case {
        Real u;
        std::vector<Real> expected;
    };

    const Case cases[] = {
        {Real(0.125), {Real(0.5), Real(0.5), Real(0), Real(0)}},
        {Real(0.5),   {Real(0), Real(0.5), Real(0.5), Real(0)}},
        {Real(0.875), {Real(0), Real(0), Real(0.5), Real(0.5)}},
    };

    for (const auto& c : cases) {
        const Real xi0 = Real(2) * c.u - Real(1);
        std::vector<Real> values;
        basis.evaluate_values(math::Vector<Real, 3>{xi0, Real(0), Real(0)}, values);

        ASSERT_EQ(values.size(), c.expected.size());
        for (std::size_t i = 0; i < c.expected.size(); ++i) {
            EXPECT_NEAR(values[i], c.expected[i], 1e-14)
                << "u=" << c.u << ", xi=" << xi0 << ", i=" << i;
        }
    }
}

TEST(BSplineBasis, NonUniformKnotsMatchReferenceGradientsAndHessians) {
    BSplineBasis basis(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});

    struct Case {
        Real u;
        std::vector<Real> expected_gradients;
    };

    const Case cases[] = {
        {Real(0.125), {Real(-2), Real(2), Real(0), Real(0)}},
        {Real(0.5),   {Real(0), Real(-1), Real(1), Real(0)}},
        {Real(0.875), {Real(0), Real(0), Real(-2), Real(2)}},
    };

    for (const auto& c : cases) {
        const Real xi0 = map_u_to_xi(basis, c.u);
        std::vector<Gradient> grads;
        std::vector<Hessian> hessians;
        basis.evaluate_gradients(math::Vector<Real, 3>{xi0, Real(0), Real(0)}, grads);
        basis.evaluate_hessians(math::Vector<Real, 3>{xi0, Real(0), Real(0)}, hessians);

        ASSERT_EQ(grads.size(), c.expected_gradients.size());
        ASSERT_EQ(hessians.size(), c.expected_gradients.size());
        for (std::size_t i = 0; i < c.expected_gradients.size(); ++i) {
            EXPECT_NEAR(grads[i][0], c.expected_gradients[i], 1e-12)
                << "u=" << c.u << ", xi=" << xi0 << ", i=" << i;
            EXPECT_NEAR(hessians[i](0, 0), Real(0), 1e-6)
                << "u=" << c.u << ", xi=" << xi0 << ", i=" << i;
        }
    }
}

TEST(BSplineBasis, RepeatedInteriorKnotsPreserveExpectedContinuityDrop) {
    // Quadratic spline with a repeated interior knot at u=0.5 (multiplicity 2).
    // Values remain continuous, but first derivatives exhibit the expected jump
    // because the continuity drops from C1 to C0 at the repeated knot.
    BSplineBasis basis(2, {Real(0), Real(0), Real(0), Real(0.5), Real(0.5),
                           Real(1), Real(1), Real(1)});

    const Real eps = Real(1e-6);
    std::vector<Real> values_left, values_right;
    std::vector<Gradient> grads_left, grads_right;

    basis.evaluate_values(math::Vector<Real, 3>{-eps, Real(0), Real(0)}, values_left);
    basis.evaluate_values(math::Vector<Real, 3>{eps, Real(0), Real(0)}, values_right);
    basis.evaluate_gradients(math::Vector<Real, 3>{-eps, Real(0), Real(0)}, grads_left);
    basis.evaluate_gradients(math::Vector<Real, 3>{eps, Real(0), Real(0)}, grads_right);

    ASSERT_EQ(values_left.size(), values_right.size());
    ASSERT_EQ(grads_left.size(), grads_right.size());

    Real max_value_gap = Real(0);
    Real max_gradient_jump = Real(0);
    for (std::size_t i = 0; i < values_left.size(); ++i) {
        max_value_gap = std::max(max_value_gap, std::abs(values_left[i] - values_right[i]));
        max_gradient_jump = std::max(
            max_gradient_jump,
            std::abs(grads_left[i][0] - grads_right[i][0]));
    }

    EXPECT_LT(max_value_gap, Real(1e-4));
    EXPECT_GT(max_gradient_jump, Real(1.0));
}

TEST(BSplineBasis, ClampedBoundaryOneSidedDerivativesMatchFiniteDifference) {
    BSplineBasis basis(2, make_open_uniform_knots(2, /*num_basis=*/6));
    const Real h = Real(1e-4);

    const math::Vector<Real, 3> xi_left{Real(-1), Real(0), Real(0)};
    const math::Vector<Real, 3> xi_left_h{Real(-1) + h, Real(0), Real(0)};
    const math::Vector<Real, 3> xi_left_2h{Real(-1) + Real(2) * h, Real(0), Real(0)};

    const math::Vector<Real, 3> xi_right{Real(1), Real(0), Real(0)};
    const math::Vector<Real, 3> xi_right_h{Real(1) - h, Real(0), Real(0)};
    const math::Vector<Real, 3> xi_right_2h{Real(1) - Real(2) * h, Real(0), Real(0)};

    std::vector<Real> vals_left, vals_left_h, vals_left_2h;
    std::vector<Real> vals_right, vals_right_h, vals_right_2h;
    std::vector<Gradient> grads_left, grads_right;

    basis.evaluate_values(xi_left, vals_left);
    basis.evaluate_values(xi_left_h, vals_left_h);
    basis.evaluate_values(xi_left_2h, vals_left_2h);
    basis.evaluate_gradients(xi_left, grads_left);

    basis.evaluate_values(xi_right, vals_right);
    basis.evaluate_values(xi_right_h, vals_right_h);
    basis.evaluate_values(xi_right_2h, vals_right_2h);
    basis.evaluate_gradients(xi_right, grads_right);

    ASSERT_EQ(vals_left.size(), basis.size());
    ASSERT_EQ(vals_right.size(), basis.size());
    ASSERT_EQ(grads_left.size(), basis.size());
    ASSERT_EQ(grads_right.size(), basis.size());

    for (std::size_t i = 0; i < basis.size(); ++i) {
        const Real fd_left =
            (-Real(3) * vals_left[i] + Real(4) * vals_left_h[i] - vals_left_2h[i]) /
            (Real(2) * h);
        const Real fd_right =
            (Real(3) * vals_right[i] - Real(4) * vals_right_h[i] + vals_right_2h[i]) /
            (Real(2) * h);

        EXPECT_NEAR(grads_left[i][0], fd_left, 5e-4) << "left i=" << i;
        EXPECT_NEAR(grads_right[i][0], fd_right, 5e-4) << "right i=" << i;
    }
}

TEST(BSplineBasis, HigherMultiplicityInteriorKnotsPreserveContinuityAndDerivativeJump) {
    BSplineBasis basis(3, {Real(0), Real(0), Real(0), Real(0), Real(0.5), Real(0.5), Real(0.5),
                           Real(1), Real(1), Real(1), Real(1)});

    const Real eps = Real(1e-6);
    std::vector<Real> values_left, values_right;
    std::vector<Gradient> grads_left, grads_right;

    basis.evaluate_values(math::Vector<Real, 3>{-eps, Real(0), Real(0)}, values_left);
    basis.evaluate_values(math::Vector<Real, 3>{eps, Real(0), Real(0)}, values_right);
    basis.evaluate_gradients(math::Vector<Real, 3>{-eps, Real(0), Real(0)}, grads_left);
    basis.evaluate_gradients(math::Vector<Real, 3>{eps, Real(0), Real(0)}, grads_right);

    ASSERT_EQ(values_left.size(), values_right.size());
    ASSERT_EQ(grads_left.size(), grads_right.size());

    Real max_value_gap = Real(0);
    Real max_gradient_jump = Real(0);
    for (std::size_t i = 0; i < values_left.size(); ++i) {
        max_value_gap = std::max(max_value_gap, std::abs(values_left[i] - values_right[i]));
        max_gradient_jump = std::max(
            max_gradient_jump, std::abs(grads_left[i][0] - grads_right[i][0]));
    }

    EXPECT_LT(max_value_gap, Real(1e-4));
    EXPECT_GT(max_gradient_jump, Real(4.0));
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

TEST(BSplineBasis, NURBSUniformWeightScalingLeavesValuesAndGradientsInvariant) {
    const int degree = 3;
    auto knots = make_open_uniform_knots(degree, /*num_basis=*/7);
    const std::vector<Real> weights = {Real(1), Real(0.3), Real(1.5), Real(2),
                                       Real(0.8), Real(1.2), Real(1)};
    std::vector<Real> scaled_weights = weights;
    for (Real& weight : scaled_weights) {
        weight *= Real(7.5);
    }

    BSplineBasis nurbs(degree, knots, weights);
    BSplineBasis scaled(degree, knots, scaled_weights);

    const Real xi_pts[] = {Real(-0.8), Real(-0.3), Real(0.0), Real(0.45), Real(0.9)};
    for (Real xi0 : xi_pts) {
        const math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};

        std::vector<Real> values, scaled_values;
        nurbs.evaluate_values(xi, values);
        scaled.evaluate_values(xi, scaled_values);
        ASSERT_EQ(values.size(), scaled_values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_NEAR(values[i], scaled_values[i], 1e-13)
                << "value xi=" << xi0 << ", i=" << i;
        }

        std::vector<Gradient> grads, scaled_grads;
        nurbs.evaluate_gradients(xi, grads);
        scaled.evaluate_gradients(xi, scaled_grads);
        ASSERT_EQ(grads.size(), scaled_grads.size());
        for (std::size_t i = 0; i < grads.size(); ++i) {
            EXPECT_NEAR(grads[i][0], scaled_grads[i][0], 1e-11)
                << "gradient xi=" << xi0 << ", i=" << i;
        }
    }
}

TEST(BSplineBasis, AnisotropicTensorQuadGradientsMatchFiniteDifference) {
    const BSplineBasis bx(2, {Real(0), Real(0), Real(0), Real(0.35), Real(1), Real(1), Real(1)});
    const BSplineBasis by(1, {Real(0), Real(0), Real(0.2), Real(0.8), Real(1), Real(1)});
    const TensorProductBasis<BSplineBasis> basis(bx, by);

    const math::Vector<Real, 3> points[] = {
        {Real(-0.82), Real(-0.65), Real(0)},
        {Real(-0.15), Real(0.2), Real(0)},
        {Real(0.7), Real(0.75), Real(0)},
    };

    for (const auto& xi : points) {
        expect_partition_of_unity_and_finite(basis, xi);
        expect_gradients_match_numerical(basis, xi, Real(3e-5));
    }
}

TEST(BSplineBasis, AnisotropicTensorHexGradientsMatchFiniteDifference) {
    const BSplineBasis bx(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});
    const BSplineBasis by(2, {Real(0), Real(0), Real(0), Real(0.4), Real(1), Real(1), Real(1)});
    const BSplineBasis bz(3, make_open_uniform_knots(3, /*num_basis=*/7));
    const TensorProductBasis<BSplineBasis> basis(bx, by, bz);

    const math::Vector<Real, 3> points[] = {
        {Real(-0.8), Real(-0.55), Real(-0.35)},
        {Real(-0.1), Real(0.2), Real(0.15)},
        {Real(0.72), Real(0.68), Real(0.55)},
    };

    for (const auto& xi : points) {
        expect_partition_of_unity_and_finite(basis, xi, Real(2e-12));
        expect_gradients_match_numerical(basis, xi, Real(6e-5));
    }
}

TEST(BSplineBasis, NURBSWeightsSizeMismatchThrows) {
    const int degree = 2;
    auto knots = make_open_uniform_knots(degree, /*num_basis=*/5);
    std::vector<Real> bad_weights = {Real(1), Real(1)}; // wrong size
    EXPECT_THROW(BSplineBasis(degree, knots, bad_weights), svmp::FE::basis::BasisConfigurationException);
}

TEST(BSplineBasis, DegreeZeroBehavior) {
    BSplineBasis basis(0, make_open_uniform_knots(0, /*num_basis=*/4));

    const Real xi_pts[] = {Real(-0.75), Real(-0.1), Real(0.1), Real(0.75)};
    for (Real xi0 : xi_pts) {
        math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
        std::vector<Real> values;
        basis.evaluate_values(xi, values);

        ASSERT_EQ(values.size(), basis.size());
        const Real sum = std::accumulate(values.begin(), values.end(), Real(0));
        EXPECT_NEAR(sum, Real(1), 1e-14);

        int active = 0;
        for (Real value : values) {
            if (std::abs(value) > Real(1e-14)) {
                ++active;
            }
        }
        EXPECT_EQ(active, 1);
    }
}

TEST(BSplineBasis, OutOfRangeXiClampsToBoundary) {
    BSplineBasis basis(2, make_open_uniform_knots(2, /*num_basis=*/5));

    std::vector<Real> vals_left, vals_right, vals_below, vals_above;
    basis.evaluate_values(math::Vector<Real, 3>{Real(-1), Real(0), Real(0)}, vals_left);
    basis.evaluate_values(math::Vector<Real, 3>{Real(1), Real(0), Real(0)}, vals_right);
    basis.evaluate_values(math::Vector<Real, 3>{Real(-1.5), Real(0), Real(0)}, vals_below);
    basis.evaluate_values(math::Vector<Real, 3>{Real(1.5), Real(0), Real(0)}, vals_above);

    ASSERT_EQ(vals_left.size(), vals_below.size());
    ASSERT_EQ(vals_right.size(), vals_above.size());
    for (std::size_t i = 0; i < vals_left.size(); ++i) {
        EXPECT_NEAR(vals_left[i], vals_below[i], 1e-14);
        EXPECT_NEAR(vals_right[i], vals_above[i], 1e-14);
    }
}

TEST(BSplineBasis, RationalBasisReportsCorrectSemanticType) {
    const int degree = 2;
    auto knots = make_open_uniform_knots(degree, /*num_basis=*/5);
    std::vector<Real> weights = {Real(1), Real(0.5), Real(2), Real(0.7), Real(1)};
    BSplineBasis bspline(degree, knots);
    BSplineBasis nurbs(degree, knots, weights);

    EXPECT_EQ(bspline.basis_type(), BasisType::BSpline);
    EXPECT_EQ(nurbs.basis_type(), BasisType::NURBS);
    EXPECT_FALSE(bspline.is_rational());
    EXPECT_TRUE(nurbs.is_rational());
}
