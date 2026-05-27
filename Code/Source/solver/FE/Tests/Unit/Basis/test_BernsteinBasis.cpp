/**
 * @file test_BernsteinBasis.cpp
 * @brief Unit tests for Bernstein bases
 */

#include <gtest/gtest.h>
#include "FE/Basis/BernsteinBasis.h"
#include <array>
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
        case ElementType::Tetra4:
            return {
                Point{Real(0.02), Real(0.03), Real(0.01)},
                Point{Real(0.9), Real(0.03), Real(0.02)},
                Point{Real(0.05), Real(0.82), Real(0.03)},
                Point{Real(0.08), Real(0.1), Real(0.74)}
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

void expect_basis_outputs_near(const std::vector<Real>& expected_values,
                               const std::vector<Gradient>& expected_gradients,
                               const std::vector<Hessian>& expected_hessians,
                               const std::vector<Real>& values,
                               const std::vector<Gradient>& gradients,
                               const std::vector<Hessian>& hessians,
                               Real tolerance) {
    ASSERT_EQ(values.size(), expected_values.size());
    ASSERT_EQ(gradients.size(), expected_gradients.size());
    ASSERT_EQ(hessians.size(), expected_hessians.size());

    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(values[i], expected_values[i], tolerance) << "value i=" << i;
        for (int a = 0; a < 3; ++a) {
            const auto sa = static_cast<std::size_t>(a);
            EXPECT_NEAR(gradients[i][sa], expected_gradients[i][sa], tolerance)
                << "gradient i=" << i << " component=" << a;
            for (int b = 0; b < 3; ++b) {
                const auto sb = static_cast<std::size_t>(b);
                EXPECT_NEAR(hessians[i](sa, sb), expected_hessians[i](sa, sb), tolerance)
                    << "hessian i=" << i << " component=(" << a << "," << b << ")";
            }
        }
    }
}

std::vector<Real> values_at(const BernsteinBasis& basis, Point point) {
    std::vector<Real> values;
    basis.evaluate_values(point, values);
    return values;
}

Point shifted(Point point, int axis, Real delta) {
    point[static_cast<std::size_t>(axis)] += delta;
    return point;
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

TEST(BernsteinBasis, TetraPartitionOfUnity) {
    BernsteinBasis basis(ElementType::Tetra4, 2);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.15), Real(0.2), Real(0.25)};
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

TEST(BernsteinBasis, FusedEvaluateAllMatchesSeparateEvaluation) {
    const struct Case {
        ElementType type;
        int order;
        Point point;
        Real tolerance;
    } cases[] = {
        {ElementType::Line2, 4, Point{Real(0.17), Real(0), Real(0)}, Real(1e-12)},
        {ElementType::Triangle3, 3, Point{Real(0.23), Real(0.31), Real(0)}, Real(1e-12)},
        {ElementType::Tetra4, 3, Point{Real(0.11), Real(0.22), Real(0.19)}, Real(1e-12)},
        {ElementType::Quad4, 3, Point{Real(-0.27), Real(0.36), Real(0)}, Real(1e-12)},
        {ElementType::Hex8, 2, Point{Real(-0.22), Real(0.31), Real(-0.41)}, Real(1e-12)},
        {ElementType::Wedge6, 2, Point{Real(0.21), Real(0.18), Real(-0.37)}, Real(1e-12)},
        {ElementType::Pyramid5, 2, Point{Real(0.09), Real(-0.14), Real(0.35)}, Real(1e-10)},
    };

    for (const auto& c : cases) {
        BernsteinBasis basis(c.type, c.order);
        std::vector<Real> expected_values;
        std::vector<Gradient> expected_gradients;
        std::vector<Hessian> expected_hessians;
        basis.evaluate_values(c.point, expected_values);
        basis.evaluate_gradients(c.point, expected_gradients);
        basis.evaluate_hessians(c.point, expected_hessians);

        std::vector<Real> values;
        std::vector<Gradient> gradients;
        std::vector<Hessian> hessians;
        basis.evaluate_all(c.point, values, gradients, hessians);

        expect_basis_outputs_near(expected_values,
                                  expected_gradients,
                                  expected_hessians,
                                  values,
                                  gradients,
                                  hessians,
                                  c.tolerance);
    }
}

TEST(BernsteinBasis, RawAndQuadratureOutputsMatchVectorEvaluation) {
    const struct Case {
        ElementType type;
        int order;
        std::vector<Point> points;
        Real tolerance;
    } cases[] = {
        {ElementType::Line2, 4,
         {Point{Real(-0.71), Real(0), Real(0)}, Point{Real(0.17), Real(0), Real(0)}},
         Real(1e-12)},
        {ElementType::Triangle3, 3,
         {Point{Real(0.23), Real(0.31), Real(0)}, Point{Real(0.12), Real(0.52), Real(0)}},
         Real(1e-12)},
        {ElementType::Tetra4, 3,
         {Point{Real(0.11), Real(0.22), Real(0.19)}, Point{Real(0.31), Real(0.14), Real(0.18)}},
         Real(1e-12)},
        {ElementType::Quad4, 3,
         {Point{Real(-0.27), Real(0.36), Real(0)}, Point{Real(0.41), Real(-0.22), Real(0)}},
         Real(1e-12)},
        {ElementType::Hex8, 2,
         {Point{Real(-0.22), Real(0.31), Real(-0.41)}, Point{Real(0.37), Real(-0.28), Real(0.19)}},
         Real(1e-12)},
        {ElementType::Wedge6, 2,
         {Point{Real(0.21), Real(0.18), Real(-0.37)}, Point{Real(0.11), Real(0.42), Real(0.29)}},
         Real(1e-12)},
        {ElementType::Pyramid5, 2,
         {Point{Real(0.09), Real(-0.14), Real(0.35)}, Point{Real(-0.18), Real(0.21), Real(0.22)}},
         Real(1e-10)},
    };

    for (const auto& c : cases) {
        BernsteinBasis basis(c.type, c.order);
        const std::size_t num_dofs = basis.size();
        const std::size_t num_qpts = c.points.size();

        std::vector<Real> raw_values(num_dofs);
        std::vector<Real> raw_gradients(num_dofs * 3u);
        std::vector<Real> raw_hessians(num_dofs * 9u);

        std::vector<Real> expected_values;
        std::vector<Gradient> expected_gradients;
        std::vector<Hessian> expected_hessians;
        basis.evaluate_all(c.points.front(), expected_values, expected_gradients, expected_hessians);

        basis.evaluate_values_to(c.points.front(), raw_values.data());
        basis.evaluate_gradients_to(c.points.front(), raw_gradients.data());
        basis.evaluate_hessians_to(c.points.front(), raw_hessians.data());

        for (std::size_t i = 0; i < num_dofs; ++i) {
            EXPECT_NEAR(raw_values[i], expected_values[i], c.tolerance) << "raw value i=" << i;
            for (int a = 0; a < 3; ++a) {
                const auto sa = static_cast<std::size_t>(a);
                EXPECT_NEAR(raw_gradients[i * 3u + sa], expected_gradients[i][sa], c.tolerance)
                    << "raw gradient i=" << i << " component=" << a;
                for (int b = 0; b < 3; ++b) {
                    const auto sb = static_cast<std::size_t>(b);
                    EXPECT_NEAR(raw_hessians[i * 9u + static_cast<std::size_t>(a * 3 + b)],
                                expected_hessians[i](sa, sb),
                                c.tolerance)
                        << "raw hessian i=" << i << " component=(" << a << "," << b << ")";
                }
            }
        }

        std::vector<Real> q_values(num_dofs * num_qpts);
        std::vector<Real> q_gradients(num_dofs * 3u * num_qpts);
        std::vector<Real> q_hessians(num_dofs * 9u * num_qpts);
        basis.evaluate_at_quadrature_points(c.points,
                                            q_values.data(),
                                            q_gradients.data(),
                                            q_hessians.data());

        for (std::size_t q = 0; q < num_qpts; ++q) {
            basis.evaluate_all(c.points[q], expected_values, expected_gradients, expected_hessians);
            for (std::size_t i = 0; i < num_dofs; ++i) {
                EXPECT_NEAR(q_values[i * num_qpts + q], expected_values[i], c.tolerance)
                    << "q value dof=" << i << " q=" << q;
                for (int a = 0; a < 3; ++a) {
                    const auto sa = static_cast<std::size_t>(a);
                    EXPECT_NEAR(q_gradients[(i * 3u + sa) * num_qpts + q],
                                expected_gradients[i][sa],
                                c.tolerance)
                        << "q gradient dof=" << i << " component=" << a << " q=" << q;
                    for (int b = 0; b < 3; ++b) {
                        const auto sb = static_cast<std::size_t>(b);
                        EXPECT_NEAR(q_hessians[(i * 9u + static_cast<std::size_t>(a * 3 + b)) * num_qpts + q],
                                    expected_hessians[i](sa, sb),
                                    c.tolerance)
                            << "q hessian dof=" << i << " component=(" << a << "," << b << ") q=" << q;
                    }
                }
            }
        }

        const std::size_t stride = num_qpts + 2u;
        const Real sentinel = Real(-987654.25);
        std::vector<Real> strided_values(num_dofs * stride, sentinel);
        std::vector<Real> strided_gradients(num_dofs * 3u * stride, sentinel);
        std::vector<Real> strided_hessians(num_dofs * 9u * stride, sentinel);
        basis.evaluate_at_quadrature_points_strided(c.points,
                                                    stride,
                                                    strided_values.data(),
                                                    strided_gradients.data(),
                                                    strided_hessians.data());

        for (std::size_t q = 0; q < num_qpts; ++q) {
            basis.evaluate_all(c.points[q], expected_values, expected_gradients, expected_hessians);
            for (std::size_t i = 0; i < num_dofs; ++i) {
                EXPECT_NEAR(strided_values[i * stride + q], expected_values[i], c.tolerance)
                    << "strided value dof=" << i << " q=" << q;
                for (int a = 0; a < 3; ++a) {
                    const auto sa = static_cast<std::size_t>(a);
                    EXPECT_NEAR(strided_gradients[(i * 3u + sa) * stride + q],
                                expected_gradients[i][sa],
                                c.tolerance)
                        << "strided gradient dof=" << i << " component=" << a << " q=" << q;
                    for (int b = 0; b < 3; ++b) {
                        const auto sb = static_cast<std::size_t>(b);
                        EXPECT_NEAR(strided_hessians[(i * 9u + static_cast<std::size_t>(a * 3 + b)) * stride + q],
                                    expected_hessians[i](sa, sb),
                                    c.tolerance)
                            << "strided hessian dof=" << i << " component=(" << a << "," << b << ") q=" << q;
                    }
                }
            }
        }

        for (std::size_t i = 0; i < num_dofs; ++i) {
            for (std::size_t q = num_qpts; q < stride; ++q) {
                EXPECT_EQ(strided_values[i * stride + q], sentinel)
                    << "value padding dof=" << i << " q=" << q;
            }
            for (std::size_t component = 0; component < 3u; ++component) {
                for (std::size_t q = num_qpts; q < stride; ++q) {
                    EXPECT_EQ(strided_gradients[(i * 3u + component) * stride + q], sentinel)
                        << "gradient padding dof=" << i << " component=" << component << " q=" << q;
                }
            }
            for (std::size_t component = 0; component < 9u; ++component) {
                for (std::size_t q = num_qpts; q < stride; ++q) {
                    EXPECT_EQ(strided_hessians[(i * 9u + component) * stride + q], sentinel)
                        << "hessian padding dof=" << i << " component=" << component << " q=" << q;
                }
            }
        }
    }
}

TEST(BernsteinBasis, PyramidDerivativesMatchFiniteDifferences) {
    BernsteinBasis basis(ElementType::Pyramid5, 3);
    const Point point{Real(0.12), Real(-0.19), Real(0.2)};

    std::vector<Gradient> gradients;
    std::vector<Hessian> hessians;
    basis.evaluate_gradients(point, gradients);
    basis.evaluate_hessians(point, hessians);

    const Real grad_step = Real(1e-6);
    const Real hess_step = Real(2e-4);
    const auto center = values_at(basis, point);

    ASSERT_EQ(gradients.size(), basis.size());
    ASSERT_EQ(hessians.size(), basis.size());

    for (int axis = 0; axis < 3; ++axis) {
        const auto plus = values_at(basis, shifted(point, axis, grad_step));
        const auto minus = values_at(basis, shifted(point, axis, -grad_step));
        for (std::size_t i = 0; i < basis.size(); ++i) {
            const Real fd = (plus[i] - minus[i]) / (Real(2) * grad_step);
            EXPECT_NEAR(gradients[i][static_cast<std::size_t>(axis)], fd, Real(1e-7))
                << "gradient dof=" << i << " axis=" << axis;
        }
    }

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            std::vector<Real> fd_values;
            if (a == b) {
                const auto plus = values_at(basis, shifted(point, a, hess_step));
                const auto minus = values_at(basis, shifted(point, a, -hess_step));
                fd_values.resize(basis.size());
                for (std::size_t i = 0; i < basis.size(); ++i) {
                    fd_values[i] = (plus[i] - Real(2) * center[i] + minus[i]) / (hess_step * hess_step);
                }
            } else {
                Point pp = shifted(shifted(point, a, hess_step), b, hess_step);
                Point pm = shifted(shifted(point, a, hess_step), b, -hess_step);
                Point mp = shifted(shifted(point, a, -hess_step), b, hess_step);
                Point mm = shifted(shifted(point, a, -hess_step), b, -hess_step);
                const auto fpp = values_at(basis, pp);
                const auto fpm = values_at(basis, pm);
                const auto fmp = values_at(basis, mp);
                const auto fmm = values_at(basis, mm);
                fd_values.resize(basis.size());
                for (std::size_t i = 0; i < basis.size(); ++i) {
                    fd_values[i] = (fpp[i] - fpm[i] - fmp[i] + fmm[i]) /
                                   (Real(4) * hess_step * hess_step);
                }
            }

            for (std::size_t i = 0; i < basis.size(); ++i) {
                EXPECT_NEAR(hessians[i](static_cast<std::size_t>(a), static_cast<std::size_t>(b)),
                            fd_values[i],
                            Real(2e-5))
                    << "hessian dof=" << i << " component=(" << a << "," << b << ")";
            }
        }
    }
}

TEST(BernsteinBasis, DeterministicBoundarySweepMaintainsPartitionNonnegativeAndFiniteGradients) {
    const struct Case {
        ElementType type;
        int order;
    } cases[] = {
        {ElementType::Line2, 4},
        {ElementType::Triangle3, 3},
        {ElementType::Tetra4, 3},
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
