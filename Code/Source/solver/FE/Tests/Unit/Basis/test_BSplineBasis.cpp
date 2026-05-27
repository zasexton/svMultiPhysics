/**
 * @file test_BSplineBasis.cpp
 * @brief Unit tests for BSplineBasis
 */

#include <gtest/gtest.h>

#include "FE/Basis/BSplineBasis.h"
#include "FE/Basis/NURBSTensorBasis.h"
#include "FE/Basis/TensorBasis.h"

#include <cmath>
#include <limits>
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

TEST(BSplineBasis, FusedValuesAndGradientsMatchIndependentCalls) {
    const struct Case {
        BSplineBasis basis;
        Real tolerance;
    } cases[] = {
        {BSplineBasis(3, make_open_uniform_knots(3, /*num_basis=*/7)), Real(1e-12)},
        {BSplineBasis(2,
                      make_open_uniform_knots(2, /*num_basis=*/6),
                      {Real(1), Real(0.8), Real(1.4), Real(0.9), Real(1.1), Real(1)}),
         Real(1e-12)}
    };

    const Real points[] = {Real(-0.73), Real(-0.2), Real(0.41), Real(0.88)};
    for (const auto& c : cases) {
        for (Real xi0 : points) {
            const math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
            std::vector<Real> expected_values;
            std::vector<Gradient> expected_gradients;
            std::vector<Real> fused_values;
            std::vector<Gradient> fused_gradients;

            c.basis.evaluate_values(xi, expected_values);
            c.basis.evaluate_gradients(xi, expected_gradients);
            c.basis.evaluate_values_and_gradients(xi, fused_values, fused_gradients);

            ASSERT_EQ(fused_values.size(), expected_values.size());
            ASSERT_EQ(fused_gradients.size(), expected_gradients.size());
            for (std::size_t i = 0; i < c.basis.size(); ++i) {
                EXPECT_NEAR(fused_values[i], expected_values[i], c.tolerance)
                    << "xi=" << xi0 << ", i=" << i;
                EXPECT_NEAR(fused_gradients[i][0], expected_gradients[i][0], c.tolerance)
                    << "xi=" << xi0 << ", i=" << i;
            }
        }
    }
}

TEST(BSplineBasis, ActiveSupportMatchesDenseEvaluation) {
    const struct Case {
        BSplineBasis basis;
        Real tolerance;
    } cases[] = {
        {BSplineBasis(3, make_open_uniform_knots(3, /*num_basis=*/7)), Real(1e-12)},
        {BSplineBasis(2,
                      make_open_uniform_knots(2, /*num_basis=*/6),
                      {Real(1), Real(0.8), Real(1.4), Real(0.9), Real(1.1), Real(1)}),
         Real(1e-12)}
    };

    const Real points[] = {Real(-0.61), Real(0.13), Real(0.72)};
    for (const auto& c : cases) {
        for (Real xi0 : points) {
            const math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
            std::vector<Real> dense_values;
            std::vector<Gradient> dense_gradients;
            std::vector<Hessian> dense_hessians;
            c.basis.evaluate_all(xi, dense_values, dense_gradients, dense_hessians);

            std::vector<Real> active_values;
            std::vector<Real> active_first;
            std::vector<Real> active_second;
            const auto range =
                c.basis.evaluate_active_support(xi, active_values, &active_first, &active_second);

            ASSERT_EQ(active_values.size(), range.count);
            ASSERT_EQ(active_first.size(), range.count);
            ASSERT_EQ(active_second.size(), range.count);
            EXPECT_LE(range.count, static_cast<std::size_t>(c.basis.order() + 1));
            ASSERT_LE(range.first_index + range.count, c.basis.size());

            for (std::size_t offset = 0; offset < range.count; ++offset) {
                const std::size_t global = range.first_index + offset;
                EXPECT_NEAR(active_values[offset], dense_values[global], c.tolerance)
                    << "xi=" << xi0 << ", i=" << global;
                EXPECT_NEAR(active_first[offset], dense_gradients[global][0], c.tolerance)
                    << "xi=" << xi0 << ", i=" << global;
                EXPECT_NEAR(active_second[offset], dense_hessians[global](0, 0), Real(1e-10))
                    << "xi=" << xi0 << ", i=" << global;
            }
        }
    }
}

TEST(BSplineBasis, EvaluateAllRawAndQuadratureOutputsMatchIndependentCalls) {
    auto check_basis = [](const BSplineBasis& basis,
                          const std::vector<Real>& xi_values,
                          Real tolerance) {
        std::vector<math::Vector<Real, 3>> points;
        points.reserve(xi_values.size());
        for (Real xi : xi_values) {
            points.push_back({xi, Real(0), Real(0)});
        }

        const std::size_t num_dofs = basis.size();
        const std::size_t num_qpts = points.size();

        std::vector<Real> expected_values;
        std::vector<Gradient> expected_gradients;
        std::vector<Hessian> expected_hessians;
        std::vector<Real> fused_values;
        std::vector<Gradient> fused_gradients;
        std::vector<Hessian> fused_hessians;

        basis.evaluate_values(points.front(), expected_values);
        basis.evaluate_gradients(points.front(), expected_gradients);
        basis.evaluate_hessians(points.front(), expected_hessians);
        basis.evaluate_all(points.front(), fused_values, fused_gradients, fused_hessians);

        ASSERT_EQ(fused_values.size(), expected_values.size());
        ASSERT_EQ(fused_gradients.size(), expected_gradients.size());
        ASSERT_EQ(fused_hessians.size(), expected_hessians.size());
        for (std::size_t i = 0; i < num_dofs; ++i) {
            EXPECT_NEAR(fused_values[i], expected_values[i], tolerance) << "fused value i=" << i;
            EXPECT_NEAR(fused_gradients[i][0], expected_gradients[i][0], tolerance)
                << "fused gradient i=" << i;
            EXPECT_NEAR(fused_hessians[i](0, 0), expected_hessians[i](0, 0), tolerance)
                << "fused hessian i=" << i;
        }

        std::vector<Real> raw_values(num_dofs);
        std::vector<Real> raw_gradients(num_dofs * 3u);
        std::vector<Real> raw_hessians(num_dofs * 9u);
        basis.evaluate_values_to(points.front(), raw_values.data());
        basis.evaluate_gradients_to(points.front(), raw_gradients.data());
        basis.evaluate_hessians_to(points.front(), raw_hessians.data());
        for (std::size_t i = 0; i < num_dofs; ++i) {
            EXPECT_NEAR(raw_values[i], expected_values[i], tolerance) << "raw value i=" << i;
            EXPECT_NEAR(raw_gradients[i * 3u], expected_gradients[i][0], tolerance)
                << "raw gradient i=" << i;
            EXPECT_NEAR(raw_hessians[i * 9u], expected_hessians[i](0, 0), tolerance)
                << "raw hessian i=" << i;
        }

        std::vector<Real> q_values(num_dofs * num_qpts);
        std::vector<Real> q_gradients(num_dofs * 3u * num_qpts);
        std::vector<Real> q_hessians(num_dofs * 9u * num_qpts);
        basis.evaluate_at_quadrature_points(points,
                                            q_values.data(),
                                            q_gradients.data(),
                                            q_hessians.data());
        for (std::size_t q = 0; q < num_qpts; ++q) {
            basis.evaluate_all(points[q], fused_values, fused_gradients, fused_hessians);
            for (std::size_t i = 0; i < num_dofs; ++i) {
                EXPECT_NEAR(q_values[i * num_qpts + q], fused_values[i], tolerance)
                    << "q value dof=" << i << " q=" << q;
                EXPECT_NEAR(q_gradients[(i * 3u) * num_qpts + q], fused_gradients[i][0], tolerance)
                    << "q gradient dof=" << i << " q=" << q;
                EXPECT_NEAR(q_hessians[(i * 9u) * num_qpts + q], fused_hessians[i](0, 0), tolerance)
                    << "q hessian dof=" << i << " q=" << q;
            }
        }
    };

    BSplineBasis bspline(3, make_open_uniform_knots(3, /*num_basis=*/7));
    check_basis(bspline, {Real(-0.55), Real(-0.1), Real(0.42)}, Real(1e-11));

    BSplineBasis rational(2,
                          make_open_uniform_knots(2, /*num_basis=*/6),
                          {Real(1), Real(0.7), Real(1.5), Real(0.9), Real(1.2), Real(1)});
    check_basis(rational, {Real(-0.48), Real(0.18), Real(0.67)}, Real(1e-10));
}

TEST(BSplineBasis, RawOutputsOverwriteDirtyBuffersCompletely) {
    BSplineBasis basis(2, make_open_uniform_knots(2, /*num_basis=*/6));
    const std::vector<math::Vector<Real, 3>> points = {
        {Real(-0.72), Real(0), Real(0)},
        {Real(0.15), Real(0), Real(0)},
        {Real(0.68), Real(0), Real(0)}
    };
    const std::size_t num_dofs = basis.size();
    const std::size_t num_qpts = points.size();
    constexpr Real sentinel = Real(913.25);

    std::vector<Real> values(num_dofs, sentinel);
    std::vector<Real> gradients(num_dofs * 3u, sentinel);
    std::vector<Real> hessians(num_dofs * 9u, sentinel);
    basis.evaluate_values_to(points[1], values.data());
    basis.evaluate_gradients_to(points[1], gradients.data());
    basis.evaluate_hessians_to(points[1], hessians.data());

    std::vector<Real> expected_values;
    std::vector<Gradient> expected_gradients;
    std::vector<Hessian> expected_hessians;
    basis.evaluate_all(points[1], expected_values, expected_gradients, expected_hessians);
    for (std::size_t dof = 0; dof < num_dofs; ++dof) {
        EXPECT_NEAR(values[dof], expected_values[dof], Real(1e-12));
        for (std::size_t component = 0; component < 3u; ++component) {
            EXPECT_NEAR(gradients[dof * 3u + component],
                        expected_gradients[dof][component],
                        Real(1e-12));
        }
        for (std::size_t component = 0; component < 9u; ++component) {
            const std::size_t row = component / 3u;
            const std::size_t col = component % 3u;
            EXPECT_NEAR(hessians[dof * 9u + component],
                        expected_hessians[dof](row, col),
                        Real(1e-10));
        }
    }

    const std::size_t stride = num_qpts + 2u;
    std::vector<Real> q_values(num_dofs * stride, sentinel);
    std::vector<Real> q_gradients(num_dofs * 3u * stride, sentinel);
    std::vector<Real> q_hessians(num_dofs * 9u * stride, sentinel);
    basis.evaluate_at_quadrature_points_strided(points,
                                                stride,
                                                q_values.data(),
                                                q_gradients.data(),
                                                q_hessians.data());
    for (std::size_t q = 0; q < num_qpts; ++q) {
        basis.evaluate_all(points[q], expected_values, expected_gradients, expected_hessians);
        for (std::size_t dof = 0; dof < num_dofs; ++dof) {
            EXPECT_NEAR(q_values[dof * stride + q], expected_values[dof], Real(1e-12));
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_NEAR(q_gradients[(dof * 3u + component) * stride + q],
                            expected_gradients[dof][component],
                            Real(1e-12));
            }
            for (std::size_t component = 0; component < 9u; ++component) {
                const std::size_t row = component / 3u;
                const std::size_t col = component % 3u;
                EXPECT_NEAR(q_hessians[(dof * 9u + component) * stride + q],
                            expected_hessians[dof](row, col),
                            Real(1e-10));
            }
        }
    }
    for (std::size_t dof = 0; dof < num_dofs; ++dof) {
        for (std::size_t pad = num_qpts; pad < stride; ++pad) {
            EXPECT_DOUBLE_EQ(q_values[dof * stride + pad], sentinel);
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

TEST(BSplineBasis, NearlyRepeatedInteriorKnotsRemainFinite) {
    const Real tiny_gap = std::numeric_limits<Real>::epsilon() * Real(8);
    BSplineBasis basis(2, {Real(0), Real(0), Real(0), Real(0.5), Real(0.5) + tiny_gap,
                           Real(1), Real(1), Real(1)});

    const Real xi_mid = map_u_to_xi(basis, Real(0.5));
    expect_partition_of_unity_and_finite(basis, {xi_mid, Real(0), Real(0)}, Real(1e-12));
    expect_partition_of_unity_and_finite(basis, {xi_mid + Real(1e-10), Real(0), Real(0)}, Real(1e-12));
    expect_partition_of_unity_and_finite(basis, {xi_mid - Real(1e-10), Real(0), Real(0)}, Real(1e-12));
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

TEST(BSplineBasis, NURBSTensorQuadPartitionOfUnityAndScalingInvariance) {
    const BSplineBasis bx(2, {Real(0), Real(0), Real(0), Real(0.45), Real(1), Real(1), Real(1)});
    const BSplineBasis by(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});
    const std::vector<Real> weights = {
        Real(1.0), Real(0.9), Real(1.2), Real(1.1),
        Real(0.8), Real(1.3), Real(1.6), Real(0.95),
        Real(1.4), Real(0.7), Real(1.05), Real(1.2),
        Real(0.85), Real(1.1), Real(0.92), Real(1.5),
    };
    std::vector<Real> scaled_weights = weights;
    for (Real& weight : scaled_weights) {
        weight *= Real(4.25);
    }

    const NURBSTensorBasis basis(bx, by, weights, {4, 4});
    const NURBSTensorBasis scaled(bx, by, scaled_weights, {4, 4});

    const math::Vector<Real, 3> points[] = {
        {Real(-0.8), Real(-0.6), Real(0)},
        {Real(-0.2), Real(0.1), Real(0)},
        {Real(0.65), Real(0.75), Real(0)},
    };

    for (const auto& xi : points) {
        expect_partition_of_unity_and_finite(basis, xi);

        std::vector<Real> values, scaled_values;
        std::vector<Gradient> grads, scaled_grads;
        basis.evaluate_values(xi, values);
        scaled.evaluate_values(xi, scaled_values);
        basis.evaluate_gradients(xi, grads);
        scaled.evaluate_gradients(xi, scaled_grads);

        ASSERT_EQ(values.size(), scaled_values.size());
        ASSERT_EQ(grads.size(), scaled_grads.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_NEAR(values[i], scaled_values[i], 1e-13);
            EXPECT_NEAR(grads[i][0], scaled_grads[i][0], 1e-11);
            EXPECT_NEAR(grads[i][1], scaled_grads[i][1], 1e-11);
        }
    }
}

TEST(BSplineBasis, NURBSTensorEvaluateAllMatchesIndependentCalls) {
    const BSplineBasis bx(2, {Real(0), Real(0), Real(0), Real(0.45), Real(1), Real(1), Real(1)});
    const BSplineBasis by(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});
    const NURBSTensorBasis basis(
        bx,
        by,
        {
            Real(1.0), Real(0.9), Real(1.2), Real(1.1),
            Real(0.8), Real(1.3), Real(1.6), Real(0.95),
            Real(1.4), Real(0.7), Real(1.05), Real(1.2),
            Real(0.85), Real(1.1), Real(0.92), Real(1.5),
        },
        {4, 4});

    const math::Vector<Real, 3> xi{Real(0.18), Real(-0.22), Real(0)};
    std::vector<Real> values_ref, values_all;
    std::vector<Gradient> gradients_ref, gradients_all;
    std::vector<Hessian> hessians_ref, hessians_all;

    basis.evaluate_values(xi, values_ref);
    basis.evaluate_gradients(xi, gradients_ref);
    basis.evaluate_hessians(xi, hessians_ref);
    basis.evaluate_all(xi, values_all, gradients_all, hessians_all);

    ASSERT_EQ(values_all.size(), values_ref.size());
    ASSERT_EQ(gradients_all.size(), gradients_ref.size());
    ASSERT_EQ(hessians_all.size(), hessians_ref.size());
    for (std::size_t i = 0; i < values_ref.size(); ++i) {
        EXPECT_NEAR(values_all[i], values_ref[i], Real(1e-14));
        for (std::size_t d = 0; d < 3u; ++d) {
            EXPECT_NEAR(gradients_all[i][d], gradients_ref[i][d], Real(1e-12));
            for (std::size_t e = 0; e < 3u; ++e) {
                EXPECT_NEAR(hessians_all[i](d, e), hessians_ref[i](d, e), Real(1e-10));
            }
        }
    }
}

TEST(BSplineBasis, NURBSTensorActiveSupportMatchesDenseEvaluation) {
    const BSplineBasis bx(2, {Real(0), Real(0), Real(0), Real(0.45), Real(1), Real(1), Real(1)});
    const BSplineBasis by(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});
    const NURBSTensorBasis basis(
        bx,
        by,
        {
            Real(1.0), Real(0.9), Real(1.2), Real(1.1),
            Real(0.8), Real(1.3), Real(1.6), Real(0.95),
            Real(1.4), Real(0.7), Real(1.05), Real(1.2),
            Real(0.85), Real(1.1), Real(0.92), Real(1.5),
        },
        {4, 4});

    const math::Vector<Real, 3> xi{Real(0.18), Real(-0.22), Real(0)};
    std::vector<Real> dense_values;
    std::vector<Gradient> dense_gradients;
    std::vector<Hessian> dense_hessians;
    basis.evaluate_all(xi, dense_values, dense_gradients, dense_hessians);

    std::vector<std::size_t> active_indices;
    std::vector<Real> active_values;
    std::vector<Gradient> active_gradients;
    std::vector<Hessian> active_hessians;
    const auto range = basis.evaluate_active_support(
        xi, active_indices, &active_values, &active_gradients, &active_hessians);

    ASSERT_EQ(active_indices.size(), active_values.size());
    ASSERT_EQ(active_indices.size(), active_gradients.size());
    ASSERT_EQ(active_indices.size(), active_hessians.size());
    ASSERT_LE(active_indices.size(), range.compact_size(basis.dimension()));
    EXPECT_GT(range.counts[0], 0u);
    EXPECT_GT(range.counts[1], 0u);

    std::vector<bool> seen(basis.size(), false);
    for (std::size_t pos = 0; pos < active_indices.size(); ++pos) {
        const std::size_t global = active_indices[pos];
        ASSERT_LT(global, basis.size());
        EXPECT_FALSE(seen[global]) << "duplicate active index " << global;
        seen[global] = true;
        EXPECT_NEAR(active_values[pos], dense_values[global], Real(1e-14))
            << "active value i=" << global;
        for (std::size_t d = 0; d < 3u; ++d) {
            EXPECT_NEAR(active_gradients[pos][d], dense_gradients[global][d], Real(1e-12))
                << "active gradient i=" << global << " d=" << d;
            for (std::size_t e = 0; e < 3u; ++e) {
                EXPECT_NEAR(active_hessians[pos](d, e), dense_hessians[global](d, e), Real(1e-10))
                    << "active hessian i=" << global << " (" << d << "," << e << ")";
            }
        }
    }

    std::vector<std::size_t> range_only_indices;
    const auto range_only = basis.evaluate_active_support(xi, range_only_indices);
    EXPECT_EQ(range_only.first_indices, range.first_indices);
    EXPECT_EQ(range_only.counts, range.counts);
    EXPECT_EQ(range_only_indices.size(), range_only.compact_size(basis.dimension()));
}

TEST(BSplineBasis, NURBSTensorRawAndQuadratureOutputsMatchVectorEvaluation) {
    const BSplineBasis bx(2, {Real(0), Real(0), Real(0), Real(0.45), Real(1), Real(1), Real(1)});
    const BSplineBasis by(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});
    const NURBSTensorBasis basis(
        bx,
        by,
        {
            Real(1.0), Real(0.9), Real(1.2), Real(1.1),
            Real(0.8), Real(1.3), Real(1.6), Real(0.95),
            Real(1.4), Real(0.7), Real(1.05), Real(1.2),
            Real(0.85), Real(1.1), Real(0.92), Real(1.5),
        },
        {4, 4});

    const std::vector<math::Vector<Real, 3>> points = {
        {Real(0.18), Real(-0.22), Real(0)},
        {Real(-0.42), Real(0.35), Real(0)},
        {Real(0.61), Real(0.58), Real(0)}
    };
    const std::size_t num_dofs = basis.size();
    const std::size_t num_qpts = points.size();

    std::vector<Real> expected_values;
    std::vector<Gradient> expected_gradients;
    std::vector<Hessian> expected_hessians;
    basis.evaluate_all(points.front(), expected_values, expected_gradients, expected_hessians);

    std::vector<Real> raw_values(num_dofs);
    std::vector<Real> raw_gradients(num_dofs * 3u);
    std::vector<Real> raw_hessians(num_dofs * 9u);
    basis.evaluate_values_to(points.front(), raw_values.data());
    basis.evaluate_gradients_to(points.front(), raw_gradients.data());
    basis.evaluate_hessians_to(points.front(), raw_hessians.data());

    for (std::size_t i = 0; i < num_dofs; ++i) {
        EXPECT_NEAR(raw_values[i], expected_values[i], Real(1e-14)) << "raw value i=" << i;
        for (int a = 0; a < 3; ++a) {
            const auto sa = static_cast<std::size_t>(a);
            EXPECT_NEAR(raw_gradients[i * 3u + sa], expected_gradients[i][sa], Real(1e-12))
                << "raw gradient i=" << i << " component=" << a;
            for (int b = 0; b < 3; ++b) {
                const auto sb = static_cast<std::size_t>(b);
                EXPECT_NEAR(raw_hessians[i * 9u + static_cast<std::size_t>(a * 3 + b)],
                            expected_hessians[i](sa, sb),
                            Real(1e-10))
                    << "raw hessian i=" << i << " component=(" << a << "," << b << ")";
            }
        }
    }

    std::vector<Real> q_values(num_dofs * num_qpts);
    std::vector<Real> q_gradients(num_dofs * 3u * num_qpts);
    std::vector<Real> q_hessians(num_dofs * 9u * num_qpts);
    basis.evaluate_at_quadrature_points(points,
                                        q_values.data(),
                                        q_gradients.data(),
                                        q_hessians.data());

    for (std::size_t q = 0; q < num_qpts; ++q) {
        basis.evaluate_all(points[q], expected_values, expected_gradients, expected_hessians);
        for (std::size_t i = 0; i < num_dofs; ++i) {
            EXPECT_NEAR(q_values[i * num_qpts + q], expected_values[i], Real(1e-14))
                << "q value dof=" << i << " q=" << q;
            for (int a = 0; a < 3; ++a) {
                const auto sa = static_cast<std::size_t>(a);
                EXPECT_NEAR(q_gradients[(i * 3u + sa) * num_qpts + q],
                            expected_gradients[i][sa],
                            Real(1e-12))
                    << "q gradient dof=" << i << " component=" << a << " q=" << q;
                for (int b = 0; b < 3; ++b) {
                    const auto sb = static_cast<std::size_t>(b);
                    EXPECT_NEAR(q_hessians[(i * 9u + static_cast<std::size_t>(a * 3 + b)) * num_qpts + q],
                                expected_hessians[i](sa, sb),
                                Real(1e-10))
                        << "q hessian dof=" << i << " component=(" << a << "," << b << ") q=" << q;
                }
            }
        }
    }

    const std::size_t stride = num_qpts + 2u;
    std::vector<Real> strided_values(num_dofs * stride, Real(-7));
    std::vector<Real> strided_gradients(num_dofs * 3u * stride, Real(-7));
    std::vector<Real> strided_hessians(num_dofs * 9u * stride, Real(-7));
    basis.evaluate_at_quadrature_points_strided(points,
                                                stride,
                                                strided_values.data(),
                                                strided_gradients.data(),
                                                strided_hessians.data());

    for (std::size_t q = 0; q < num_qpts; ++q) {
        basis.evaluate_all(points[q], expected_values, expected_gradients, expected_hessians);
        for (std::size_t i = 0; i < num_dofs; ++i) {
            EXPECT_NEAR(strided_values[i * stride + q], expected_values[i], Real(1e-14))
                << "strided value dof=" << i << " q=" << q;
            for (int a = 0; a < 3; ++a) {
                const auto sa = static_cast<std::size_t>(a);
                EXPECT_NEAR(strided_gradients[(i * 3u + sa) * stride + q],
                            expected_gradients[i][sa],
                            Real(1e-12))
                    << "strided gradient dof=" << i << " component=" << a << " q=" << q;
                for (int b = 0; b < 3; ++b) {
                    const auto sb = static_cast<std::size_t>(b);
                    EXPECT_NEAR(strided_hessians[(i * 9u + static_cast<std::size_t>(a * 3 + b)) * stride + q],
                                expected_hessians[i](sa, sb),
                                Real(1e-10))
                        << "strided hessian dof=" << i << " component=(" << a << "," << b << ") q=" << q;
                }
            }
        }
    }

    for (std::size_t i = 0; i < num_dofs; ++i) {
        for (std::size_t pad = num_qpts; pad < stride; ++pad) {
            EXPECT_DOUBLE_EQ(strided_values[i * stride + pad], Real(-7));
        }
    }
}

TEST(BSplineBasis, NURBSTensorQuadGradientsMatchFiniteDifference) {
    const BSplineBasis bx(2, {Real(0), Real(0), Real(0), Real(0.45), Real(1), Real(1), Real(1)});
    const BSplineBasis by(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});
    const NURBSTensorBasis basis(
        bx,
        by,
        {
            Real(1.0), Real(0.9), Real(1.2), Real(1.1),
            Real(0.8), Real(1.3), Real(1.6), Real(0.95),
            Real(1.4), Real(0.7), Real(1.05), Real(1.2),
            Real(0.85), Real(1.1), Real(0.92), Real(1.5),
        },
        {4, 4});

    const math::Vector<Real, 3> points[] = {
        {Real(-0.75), Real(-0.55), Real(0)},
        {Real(-0.1), Real(0.2), Real(0)},
        {Real(0.72), Real(0.68), Real(0)},
    };

    for (const auto& xi : points) {
        expect_partition_of_unity_and_finite(basis, xi, Real(2e-12));
        expect_gradients_match_numerical(basis, xi, Real(7e-5));
    }
}

TEST(BSplineBasis, NURBSTensorHexGradientsMatchFiniteDifference) {
    const BSplineBasis bx(1, {Real(0), Real(0), Real(0.2), Real(0.8), Real(1), Real(1)});
    const BSplineBasis by(2, {Real(0), Real(0), Real(0), Real(0.4), Real(1), Real(1), Real(1)});
    const BSplineBasis bz(1, {Real(0), Real(0), Real(0.35), Real(0.7), Real(1), Real(1)});
    std::vector<Real> weights;
    weights.reserve(64);
    for (int k = 0; k < 4; ++k) {
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                weights.push_back(Real(0.8) + Real(0.1) * Real((i + 2 * j + 3 * k) % 7));
            }
        }
    }

    const NURBSTensorBasis basis(bx, by, bz, weights, {4, 4, 4});
    const math::Vector<Real, 3> points[] = {
        {Real(-0.8), Real(-0.55), Real(-0.35)},
        {Real(-0.1), Real(0.2), Real(0.15)},
        {Real(0.72), Real(0.68), Real(0.55)},
    };

    for (const auto& xi : points) {
        expect_partition_of_unity_and_finite(basis, xi, Real(2e-12));
        expect_gradients_match_numerical(basis, xi, Real(9e-5));
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

TEST(BSplineBasis, TensorProductStridedOutputsMatchVectorEvaluation) {
    const BSplineBasis bx(2, {Real(0), Real(0), Real(0), Real(0.35), Real(1), Real(1), Real(1)});
    const BSplineBasis by(1, {Real(0), Real(0), Real(0.2), Real(0.8), Real(1), Real(1)});
    const TensorProductBasis<BSplineBasis> basis(bx, by);
    const std::vector<math::Vector<Real, 3>> points = {
        {Real(-0.82), Real(-0.65), Real(0)},
        {Real(-0.15), Real(0.2), Real(0)},
        {Real(0.7), Real(0.75), Real(0)}
    };
    const std::size_t num_dofs = basis.size();
    const std::size_t num_qpts = points.size();
    const std::size_t stride = num_qpts + 3u;

    std::vector<Real> values(num_dofs * stride, Real(-11));
    std::vector<Real> gradients(num_dofs * 3u * stride, Real(-11));
    std::vector<Real> hessians(num_dofs * 9u * stride, Real(-11));
    basis.evaluate_at_quadrature_points_strided(points,
                                                stride,
                                                values.data(),
                                                gradients.data(),
                                                hessians.data());

    std::vector<Real> expected_values;
    std::vector<Gradient> expected_gradients;
    std::vector<Hessian> expected_hessians;
    for (std::size_t q = 0; q < num_qpts; ++q) {
        basis.evaluate_all(points[q], expected_values, expected_gradients, expected_hessians);
        for (std::size_t i = 0; i < num_dofs; ++i) {
            EXPECT_NEAR(values[i * stride + q], expected_values[i], Real(1e-14));
            for (int a = 0; a < 3; ++a) {
                const auto sa = static_cast<std::size_t>(a);
                EXPECT_NEAR(gradients[(i * 3u + sa) * stride + q],
                            expected_gradients[i][sa],
                            Real(1e-12));
                for (int b = 0; b < 3; ++b) {
                    const auto sb = static_cast<std::size_t>(b);
                    EXPECT_NEAR(hessians[(i * 9u + static_cast<std::size_t>(a * 3 + b)) * stride + q],
                                expected_hessians[i](sa, sb),
                                Real(1e-10));
                }
            }
        }
    }

    for (std::size_t i = 0; i < num_dofs; ++i) {
        for (std::size_t pad = num_qpts; pad < stride; ++pad) {
            EXPECT_DOUBLE_EQ(values[i * stride + pad], Real(-11));
        }
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

TEST(BSplineBasis, NURBSTensorWeightsSizeMismatchThrows) {
    const BSplineBasis bx(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});
    const BSplineBasis by(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});
    EXPECT_THROW(NURBSTensorBasis(bx, by, {Real(1), Real(2)}),
                 svmp::FE::basis::BasisConfigurationException);
}

TEST(BSplineBasis, DegreeZeroBehavior) {
    BSplineBasis basis(0, make_open_uniform_knots(0, /*num_basis=*/4));

    const Real xi_pts[] = {Real(-0.75), Real(-0.1), Real(0.1), Real(0.75)};
    for (Real xi0 : xi_pts) {
        math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
        std::vector<Real> values;
        std::vector<Gradient> gradients;
        std::vector<Hessian> hessians;
        std::vector<Real> fused_values;
        std::vector<Gradient> fused_gradients;
        std::vector<Hessian> fused_hessians;
        basis.evaluate_values(xi, values);
        basis.evaluate_gradients(xi, gradients);
        basis.evaluate_hessians(xi, hessians);
        basis.evaluate_all(xi, fused_values, fused_gradients, fused_hessians);

        ASSERT_EQ(values.size(), basis.size());
        ASSERT_EQ(gradients.size(), basis.size());
        ASSERT_EQ(hessians.size(), basis.size());
        ASSERT_EQ(fused_values.size(), basis.size());
        ASSERT_EQ(fused_gradients.size(), basis.size());
        ASSERT_EQ(fused_hessians.size(), basis.size());
        const Real sum = std::accumulate(values.begin(), values.end(), Real(0));
        EXPECT_NEAR(sum, Real(1), 1e-14);

        int active = 0;
        for (std::size_t i = 0; i < basis.size(); ++i) {
            const Real value = values[i];
            if (std::abs(value) > Real(1e-14)) {
                ++active;
            }
            EXPECT_NEAR(fused_values[i], value, Real(1e-14));
            EXPECT_EQ(gradients[i][0], Real(0));
            EXPECT_EQ(hessians[i](0, 0), Real(0));
            EXPECT_EQ(fused_gradients[i][0], Real(0));
            EXPECT_EQ(fused_hessians[i](0, 0), Real(0));
        }
        EXPECT_EQ(active, 1);
    }
}

TEST(BSplineBasis, DegreeOneEvaluateAllKeepsSecondDerivativesZero) {
    BSplineBasis basis(1, {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)});

    const Real xi_pts[] = {map_u_to_xi(basis, Real(0.125)),
                           map_u_to_xi(basis, Real(0.5)),
                           map_u_to_xi(basis, Real(0.875))};
    for (Real xi0 : xi_pts) {
        math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
        std::vector<Real> values;
        std::vector<Gradient> gradients;
        std::vector<Hessian> hessians;
        std::vector<Real> fused_values;
        std::vector<Gradient> fused_gradients;
        std::vector<Hessian> fused_hessians;

        basis.evaluate_values(xi, values);
        basis.evaluate_gradients(xi, gradients);
        basis.evaluate_hessians(xi, hessians);
        basis.evaluate_all(xi, fused_values, fused_gradients, fused_hessians);

        ASSERT_EQ(fused_values.size(), basis.size());
        ASSERT_EQ(fused_gradients.size(), basis.size());
        ASSERT_EQ(fused_hessians.size(), basis.size());
        for (std::size_t i = 0; i < basis.size(); ++i) {
            EXPECT_NEAR(fused_values[i], values[i], Real(1e-14));
            EXPECT_NEAR(fused_gradients[i][0], gradients[i][0], Real(1e-14));
            EXPECT_NEAR(hessians[i](0, 0), Real(0), Real(1e-14));
            EXPECT_NEAR(fused_hessians[i](0, 0), Real(0), Real(1e-14));
        }
    }
}

TEST(BSplineBasis, RationalDegreeOneHessiansMatchGradientFiniteDifference) {
    BSplineBasis basis(1,
                       {Real(0), Real(0), Real(0.25), Real(0.75), Real(1), Real(1)},
                       {Real(1), Real(1.4), Real(0.8), Real(1.2)});

    const Real eps = Real(1e-6);
    const Real xi_pts[] = {map_u_to_xi(basis, Real(0.125)),
                           map_u_to_xi(basis, Real(0.5)),
                           map_u_to_xi(basis, Real(0.875))};
    for (Real xi0 : xi_pts) {
        math::Vector<Real, 3> xi{xi0, Real(0), Real(0)};
        std::vector<Real> values;
        std::vector<Gradient> gradients;
        std::vector<Hessian> hessians;
        std::vector<Real> fused_values;
        std::vector<Gradient> fused_gradients;
        std::vector<Hessian> fused_hessians;
        basis.evaluate_values(xi, values);
        basis.evaluate_gradients(xi, gradients);
        basis.evaluate_hessians(xi, hessians);
        basis.evaluate_all(xi, fused_values, fused_gradients, fused_hessians);

        math::Vector<Real, 3> xi_p{xi0 + eps, Real(0), Real(0)};
        math::Vector<Real, 3> xi_m{xi0 - eps, Real(0), Real(0)};
        std::vector<Gradient> gradients_p;
        std::vector<Gradient> gradients_m;
        basis.evaluate_gradients(xi_p, gradients_p);
        basis.evaluate_gradients(xi_m, gradients_m);

        Real hessian_sum = Real(0);
        ASSERT_EQ(values.size(), basis.size());
        ASSERT_EQ(gradients.size(), basis.size());
        ASSERT_EQ(hessians.size(), basis.size());
        for (std::size_t i = 0; i < basis.size(); ++i) {
            EXPECT_NEAR(fused_values[i], values[i], Real(1e-12));
            EXPECT_NEAR(fused_gradients[i][0], gradients[i][0], Real(1e-12));
            EXPECT_NEAR(fused_hessians[i](0, 0), hessians[i](0, 0), Real(1e-12));
            const Real fd = (gradients_p[i][0] - gradients_m[i][0]) / (Real(2) * eps);
            EXPECT_NEAR(hessians[i](0, 0), fd, Real(1e-5));
            hessian_sum += hessians[i](0, 0);
        }
        EXPECT_NEAR(hessian_sum, Real(0), Real(1e-11));
    }
}

TEST(BSplineBasis, OutOfRangeXiThrowsAndRoundoffEndpointSnaps) {
    BSplineBasis basis(2, make_open_uniform_knots(2, /*num_basis=*/5));

    std::vector<Real> vals_left, vals_right, vals_below_tol, vals_above_tol;
    basis.evaluate_values(math::Vector<Real, 3>{Real(-1), Real(0), Real(0)}, vals_left);
    basis.evaluate_values(math::Vector<Real, 3>{Real(1), Real(0), Real(0)}, vals_right);
    basis.evaluate_values(math::Vector<Real, 3>{Real(-1) - std::numeric_limits<Real>::epsilon(), Real(0), Real(0)},
                          vals_below_tol);
    basis.evaluate_values(math::Vector<Real, 3>{Real(1) + std::numeric_limits<Real>::epsilon(), Real(0), Real(0)},
                          vals_above_tol);

    ASSERT_EQ(vals_left.size(), vals_below_tol.size());
    ASSERT_EQ(vals_right.size(), vals_above_tol.size());
    for (std::size_t i = 0; i < vals_left.size(); ++i) {
        EXPECT_NEAR(vals_left[i], vals_below_tol[i], 1e-14);
        EXPECT_NEAR(vals_right[i], vals_above_tol[i], 1e-14);
    }

    std::vector<Real> values;
    EXPECT_THROW(basis.evaluate_values(math::Vector<Real, 3>{Real(-1.5), Real(0), Real(0)}, values),
                 BasisEvaluationException);
    EXPECT_THROW(basis.evaluate_values(math::Vector<Real, 3>{Real(1.5), Real(0), Real(0)}, values),
                 BasisEvaluationException);
    std::vector<Gradient> gradients;
    EXPECT_THROW(basis.evaluate_gradients(math::Vector<Real, 3>{Real(1.5), Real(0), Real(0)}, gradients),
                 BasisEvaluationException);
    std::vector<Hessian> hessians;
    EXPECT_THROW(basis.evaluate_hessians(math::Vector<Real, 3>{Real(-1.5), Real(0), Real(0)}, hessians),
                 BasisEvaluationException);
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

TEST(BSplineBasis, RationalWeightsMustBeFiniteAndPositive) {
    const int degree = 2;
    auto knots = make_open_uniform_knots(degree, /*num_basis=*/5);

    EXPECT_THROW(BSplineBasis(degree, knots, {Real(1), Real(0), Real(1), Real(1), Real(1)}),
                 BasisConfigurationException);
    EXPECT_THROW(BSplineBasis(degree, knots, {Real(1), Real(-0.5), Real(1), Real(1), Real(1)}),
                 BasisConfigurationException);
    EXPECT_THROW(BSplineBasis(degree,
                              knots,
                              {Real(1), std::numeric_limits<Real>::infinity(), Real(1), Real(1), Real(1)}),
                 BasisConfigurationException);
}

TEST(BSplineBasis, NURBSTensorWeightsMustBeFiniteAndPositive) {
    const BSplineBasis bx(1, {Real(0), Real(0), Real(0.5), Real(1), Real(1)});
    const BSplineBasis by(1, {Real(0), Real(0), Real(0.5), Real(1), Real(1)});

    EXPECT_THROW(NURBSTensorBasis(bx, by, {Real(1), Real(1), Real(0), Real(1), Real(1), Real(1),
                                           Real(1), Real(1), Real(1)}),
                 BasisConfigurationException);
    EXPECT_THROW(NURBSTensorBasis(bx, by, {Real(1), Real(1), Real(1), Real(1), Real(-1), Real(1),
                                           Real(1), Real(1), Real(1)}),
                 BasisConfigurationException);
}
