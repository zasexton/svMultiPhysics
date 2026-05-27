/**
 * @file test_HierarchicalSpectralBasis.cpp
 * @brief Tests for hierarchical and spectral bases
 */

#include <gtest/gtest.h>
#include "FE/Basis/HierarchicalBasis.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/OrthogonalPolynomials.h"
#include "FE/Basis/SpectralBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/GaussLobattoQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/WedgeQuadrature.h"
#include "FE/Quadrature/PyramidQuadrature.h"
#include <array>
#include <cmath>
#include <exception>
#include <numeric>
#include <thread>

using namespace svmp::FE;
using namespace svmp::FE::basis;
using namespace svmp::FE::quadrature;

namespace {

using Point = svmp::FE::math::Vector<Real, 3>;

std::vector<Point> boundary_stress_points_for(ElementType type) {
    switch (type) {
        case ElementType::Line2:
            return {
                Point{Real(-0.98), Real(0), Real(0)},
                Point{Real(-0.15), Real(0), Real(0)},
                Point{Real(0.95), Real(0), Real(0)}
            };
        case ElementType::Triangle3:
            return {
                Point{Real(0.01), Real(0.01), Real(0)},
                Point{Real(0.97), Real(0.01), Real(0)},
                Point{Real(0.02), Real(0.95), Real(0)},
                Point{Real(0.45), Real(0.5), Real(0)}
            };
        case ElementType::Quad4:
            return {
                Point{Real(-0.96), Real(-0.94), Real(0)},
                Point{Real(0.95), Real(-0.91), Real(0)},
                Point{Real(-0.93), Real(0.92), Real(0)},
                Point{Real(0.9), Real(0.94), Real(0)}
            };
        case ElementType::Tetra4:
            return {
                Point{Real(0.01), Real(0.01), Real(0.01)},
                Point{Real(0.96), Real(0.01), Real(0.01)},
                Point{Real(0.02), Real(0.94), Real(0.01)},
                Point{Real(0.03), Real(0.04), Real(0.9)}
            };
        case ElementType::Hex8:
            return {
                Point{Real(-0.95), Real(-0.94), Real(-0.92)},
                Point{Real(0.93), Real(-0.9), Real(0.88)},
                Point{Real(-0.89), Real(0.91), Real(0.86)},
                Point{Real(0.9), Real(0.89), Real(-0.87)}
            };
        case ElementType::Wedge6:
            return {
                Point{Real(0.01), Real(0.01), Real(-0.95)},
                Point{Real(0.97), Real(0.01), Real(0.88)},
                Point{Real(0.02), Real(0.95), Real(-0.82)},
                Point{Real(0.46), Real(0.48), Real(0.9)}
            };
        case ElementType::Pyramid5:
            return {
                Point{Real(0.0), Real(0.0), Real(0.98)},
                Point{Real(0.02), Real(-0.015), Real(0.95)},
                Point{Real(0.94), Real(-0.9), Real(-0.92)},
                Point{Real(-0.86), Real(0.88), Real(-0.85)}
            };
        default:
            return {Point{Real(0), Real(0), Real(0)}};
    }
}

void expect_values_and_gradients_finite(const std::vector<Real>& values,
                                        const std::vector<Gradient>& grads,
                                        int dimension) {
    for (Real value : values) {
        EXPECT_TRUE(std::isfinite(value));
    }
    for (const auto& grad : grads) {
        for (int d = 0; d < dimension; ++d) {
            EXPECT_TRUE(std::isfinite(grad[static_cast<std::size_t>(d)]));
        }
    }
}

bool same_point(const Point& a, const Point& b, Real tol = Real(1e-10)) {
    return std::abs(a[0] - b[0]) <= tol &&
           std::abs(a[1] - b[1]) <= tol &&
           std::abs(a[2] - b[2]) <= tol;
}

void expect_simplex_nodes_unique(const std::vector<Point>& nodes, Real tol = Real(1e-10)) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (std::size_t j = i + 1; j < nodes.size(); ++j) {
            EXPECT_FALSE(same_point(nodes[i], nodes[j], tol))
                << "Duplicate interpolation nodes at " << i << " and " << j;
        }
    }
}

void expect_triangle_nodes_in_reference(const std::vector<Point>& nodes,
                                        int order,
                                        Real tol = Real(1e-10)) {
    ASSERT_EQ(nodes.size(), static_cast<std::size_t>((order + 1) * (order + 2) / 2));
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        EXPECT_GE(node[0], -tol) << "node " << i;
        EXPECT_GE(node[1], -tol) << "node " << i;
        EXPECT_NEAR(node[2], Real(0), tol) << "node " << i;
        EXPECT_LE(node[0] + node[1], Real(1) + tol) << "node " << i;
    }
}

void expect_tetra_nodes_in_reference(const std::vector<Point>& nodes,
                                     int order,
                                     Real tol = Real(1e-10)) {
    ASSERT_EQ(nodes.size(),
              static_cast<std::size_t>((order + 1) * (order + 2) * (order + 3) / 6));
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        EXPECT_GE(node[0], -tol) << "node " << i;
        EXPECT_GE(node[1], -tol) << "node " << i;
        EXPECT_GE(node[2], -tol) << "node " << i;
        EXPECT_LE(node[0] + node[1] + node[2], Real(1) + tol) << "node " << i;
    }
}

void expect_nodal_interpolation_identity(const SpectralBasis& basis, Real tol) {
    const auto& nodes = basis.interpolation_nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        std::vector<Real> values;
        basis.evaluate_values(nodes[i], values);
        ASSERT_EQ(values.size(), nodes.size());
        for (std::size_t j = 0; j < values.size(); ++j) {
            EXPECT_NEAR(values[j], i == j ? Real(1) : Real(0), tol)
                << "node " << i << ", basis " << j;
        }
    }
}

Point map_triangle_to_pyramid_front(const Point& tri_node) {
    return {tri_node[1] - tri_node[0],
            -tri_node[0] - tri_node[1],
            Real(1) - tri_node[0] - tri_node[1]};
}

struct SpectralStridedRequest {
    bool values;
    bool gradients;
    bool hessians;
};

void expect_spectral_strided_matches_pointwise(const SpectralBasis& basis,
                                               const std::vector<Point>& points,
                                               const SpectralStridedRequest& request,
                                               Real tol) {
    const std::size_t stride = points.size() + 4u;
    constexpr Real sentinel = Real(-719.25);

    std::vector<Real> values(request.values ? basis.size() * stride : 0u, sentinel);
    std::vector<Real> gradients(request.gradients ? basis.size() * 3u * stride : 0u, sentinel);
    std::vector<Real> hessians(request.hessians ? basis.size() * 9u * stride : 0u, sentinel);

    basis.evaluate_at_quadrature_points_strided(
        points,
        stride,
        request.values ? values.data() : nullptr,
        request.gradients ? gradients.data() : nullptr,
        request.hessians ? hessians.data() : nullptr);

    for (std::size_t q = 0; q < points.size(); ++q) {
        if (request.values) {
            std::vector<Real> expected;
            basis.evaluate_values(points[q], expected);
            ASSERT_EQ(expected.size(), basis.size());
            for (std::size_t i = 0; i < basis.size(); ++i) {
                EXPECT_NEAR(values[i * stride + q], expected[i], tol);
            }
        }
        if (request.gradients) {
            std::vector<Gradient> expected;
            basis.evaluate_gradients(points[q], expected);
            ASSERT_EQ(expected.size(), basis.size());
            for (std::size_t i = 0; i < basis.size(); ++i) {
                for (std::size_t c = 0; c < 3u; ++c) {
                    EXPECT_NEAR(gradients[(i * 3u + c) * stride + q],
                                expected[i][c],
                                Real(4) * tol);
                }
            }
        }
        if (request.hessians) {
            std::vector<Hessian> expected;
            basis.evaluate_hessians(points[q], expected);
            ASSERT_EQ(expected.size(), basis.size());
            for (std::size_t i = 0; i < basis.size(); ++i) {
                for (std::size_t r = 0; r < 3u; ++r) {
                    for (std::size_t c = 0; c < 3u; ++c) {
                        EXPECT_NEAR(hessians[(i * 9u + r * 3u + c) * stride + q],
                                    expected[i](r, c),
                                    Real(8) * tol);
                    }
                }
            }
        }
    }

    const auto expect_padding_untouched = [&](const std::vector<Real>& buffer,
                                              std::size_t rows) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t q = points.size(); q < stride; ++q) {
                EXPECT_EQ(buffer[row * stride + q], sentinel);
            }
        }
    };

    if (request.values) {
        expect_padding_untouched(values, basis.size());
    }
    if (request.gradients) {
        expect_padding_untouched(gradients, basis.size() * 3u);
    }
    if (request.hessians) {
        expect_padding_untouched(hessians, basis.size() * 9u);
    }
}

} // namespace

TEST(HierarchicalBasis, LegendreOrthogonalityOnLine) {
    const int order = 3;
    HierarchicalBasis basis(ElementType::Line2, order);
    GaussQuadrature1D quad(8); // integrates up to degree 15

    for (int i = 0; i <= order; ++i) {
        for (int j = 0; j <= order; ++j) {
            double inner = 0.0;
            for (std::size_t q = 0; q < quad.num_points(); ++q) {
                std::vector<Real> vals;
                basis.evaluate_values(quad.point(q), vals);
                inner += quad.weight(q) * vals[static_cast<std::size_t>(i)] *
                         vals[static_cast<std::size_t>(j)];
            }
            if (i == j) {
                double expected = 2.0 / (2 * i + 1);
                EXPECT_NEAR(inner, expected, 1e-10);
            } else {
                EXPECT_NEAR(inner, 0.0, 1e-10);
            }
        }
    }
}

TEST(SpectralBasis, GLLKroneckerProperty) {
    const int order = 4;
    SpectralBasis basis(ElementType::Line2, order);
    const auto& nodes = basis.nodes_1d();

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        svmp::FE::math::Vector<Real, 3> xi{nodes[i], 0.0, 0.0};
        std::vector<Real> vals;
        basis.evaluate_values(xi, vals);
        for (std::size_t j = 0; j < vals.size(); ++j) {
            if (i == j) {
                EXPECT_NEAR(vals[j], 1.0, 1e-12);
            } else {
                EXPECT_NEAR(vals[j], 0.0, 1e-12);
            }
        }
    }

    EXPECT_NEAR(nodes.front(), -1.0, 1e-14);
    EXPECT_NEAR(nodes.back(), 1.0, 1e-14);
}

TEST(SpectralBasis, QuadKroneckerAndPartitionOfUnity) {
    const int order = 3;
    SpectralBasis basis(ElementType::Quad4, order);
    const auto& nodes = basis.nodes_1d();
    const std::size_t n1d = nodes.size();
    const std::size_t n = basis.size();
    ASSERT_EQ(n, n1d * n1d);

    // Kronecker property at tensor-product nodes
    for (std::size_t j = 0; j < n1d; ++j) {
        for (std::size_t i = 0; i < n1d; ++i) {
            svmp::FE::math::Vector<Real, 3> xi{nodes[i], nodes[j], Real(0)};
            std::vector<Real> vals;
            basis.evaluate_values(xi, vals);
            ASSERT_EQ(vals.size(), n);
            const std::size_t active = j * n1d + i;
            for (std::size_t k = 0; k < n; ++k) {
                if (k == active) {
                    EXPECT_NEAR(vals[k], 1.0, 1e-12);
                } else {
                    EXPECT_NEAR(vals[k], 0.0, 1e-12);
                }
            }
        }
    }

    // Partition of unity at a generic interior point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(-0.3), Real(0)};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = 0.0;
    for (double v : vals) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(SpectralBasis, GradientsMatchFiniteDifference1D) {
    const int order = 4;
    SpectralBasis basis(ElementType::Line2, order);
    const Real h = Real(1e-6);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0), Real(0)};

    std::vector<Real> vals_p, vals_m;
    svmp::FE::math::Vector<Real, 3> xi_p{xi[0] + h, Real(0), Real(0)};
    svmp::FE::math::Vector<Real, 3> xi_m{xi[0] - h, Real(0), Real(0)};
    basis.evaluate_values(xi_p, vals_p);
    basis.evaluate_values(xi_m, vals_m);

    std::vector<svmp::FE::basis::Gradient> grads;
    basis.evaluate_gradients(xi, grads);

    ASSERT_EQ(vals_p.size(), grads.size());
    for (std::size_t i = 0; i < grads.size(); ++i) {
        const Real fd = (vals_p[i] - vals_m[i]) / (Real(2) * h);
        EXPECT_NEAR(grads[i][0], fd, 1e-6);
        EXPECT_NEAR(grads[i][1], 0.0, 1e-12);
        EXPECT_NEAR(grads[i][2], 0.0, 1e-12);
    }
}

TEST(SpectralBasis, ValuesGradientsAndHessiansAreFiniteAtGLLNodes) {
    const int order = 6;
    SpectralBasis basis(ElementType::Line2, order);
    const auto& nodes = basis.nodes_1d();

    for (Real node : nodes) {
        svmp::FE::math::Vector<Real, 3> xi{node, Real(0), Real(0)};
        std::vector<Real> values;
        std::vector<svmp::FE::basis::Gradient> gradients;
        std::vector<svmp::FE::basis::Hessian> hessians;
        basis.evaluate_values(xi, values);
        basis.evaluate_gradients(xi, gradients);
        basis.evaluate_hessians(xi, hessians);

        ASSERT_EQ(values.size(), gradients.size());
        ASSERT_EQ(values.size(), hessians.size());

        Real value_sum = Real(0);
        Real gradient_sum = Real(0);
        Real hessian_sum = Real(0);
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_TRUE(std::isfinite(values[i]));
            EXPECT_TRUE(std::isfinite(gradients[i][0]));
            EXPECT_TRUE(std::isfinite(hessians[i](0, 0)));
            value_sum += values[i];
            gradient_sum += gradients[i][0];
            hessian_sum += hessians[i](0, 0);
        }
        EXPECT_NEAR(value_sum, Real(1), 1e-12);
        EXPECT_NEAR(gradient_sum, Real(0), 1e-10);
        EXPECT_NEAR(hessian_sum, Real(0), 1e-8);
    }
}

TEST(SpectralBasis, NodalDerivativeMatricesDifferentiatePolynomialsAtGLLNodes) {
    const int order = 6;
    SpectralBasis basis(ElementType::Line2, order);
    const auto& nodes = basis.nodes_1d();

    for (Real node : nodes) {
        const Point xi{node, Real(0), Real(0)};
        std::vector<Gradient> gradients;
        std::vector<Hessian> hessians;
        basis.evaluate_gradients(xi, gradients);
        basis.evaluate_hessians(xi, hessians);
        ASSERT_EQ(gradients.size(), nodes.size());
        ASSERT_EQ(hessians.size(), nodes.size());

        for (int degree = 0; degree <= order; ++degree) {
            Real interpolated_first = Real(0);
            Real interpolated_second = Real(0);
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                const Real nodal_value = std::pow(nodes[i], degree);
                interpolated_first += nodal_value * gradients[i][0];
                interpolated_second += nodal_value * hessians[i](0, 0);
            }

            const Real expected_first =
                (degree == 0) ? Real(0) :
                    static_cast<Real>(degree) * std::pow(node, degree - 1);
            const Real expected_second =
                (degree < 2) ? Real(0) :
                    static_cast<Real>(degree * (degree - 1)) * std::pow(node, degree - 2);
            EXPECT_NEAR(interpolated_first, expected_first, Real(1.0e-9))
                << "degree=" << degree << " node=" << node;
            EXPECT_NEAR(interpolated_second, expected_second, Real(1.0e-7))
                << "degree=" << degree << " node=" << node;
        }
    }
}

TEST(SpectralBasis, HighOrderNodalSecondDerivativesDifferentiatePolynomials) {
    const int order = 12;
    SpectralBasis basis(ElementType::Line2, order);
    const auto& nodes = basis.nodes_1d();

    for (Real node : nodes) {
        const Point xi{node, Real(0), Real(0)};
        std::vector<Hessian> hessians;
        basis.evaluate_hessians(xi, hessians);
        ASSERT_EQ(hessians.size(), nodes.size());

        for (int degree = 0; degree <= order; ++degree) {
            Real interpolated_second = Real(0);
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                interpolated_second += std::pow(nodes[i], degree) * hessians[i](0, 0);
            }

            const Real expected_second =
                (degree < 2) ? Real(0) :
                    static_cast<Real>(degree * (degree - 1)) * std::pow(node, degree - 2);
            EXPECT_NEAR(interpolated_second, expected_second, Real(1.0e-6))
                << "degree=" << degree << " node=" << node;
        }
    }
}

TEST(SpectralBasis, EvaluateAllMatchesIndependentTensorCalls) {
    SpectralBasis basis(ElementType::Hex8, 4);
    const Point xi{Real(0.17), Real(-0.31), Real(0.46)};

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
        EXPECT_NEAR(values_all[i], values_ref[i], Real(1e-13));
        for (std::size_t d = 0; d < 3u; ++d) {
            EXPECT_NEAR(gradients_all[i][d], gradients_ref[i][d], Real(1e-12));
            for (std::size_t e = 0; e < 3u; ++e) {
                EXPECT_NEAR(hessians_all[i](d, e), hessians_ref[i](d, e), Real(1e-10));
            }
        }
    }
}

TEST(SpectralBasis, EvaluateAllMatchesIndependentSimplexCalls) {
    const struct Case {
        ElementType type;
        int order;
        Point point;
        Real tolerance;
    } cases[] = {
        {ElementType::Triangle3, 3, Point{Real(0.21), Real(0.34), Real(0)}, Real(1e-9)},
        {ElementType::Tetra4, 3, Point{Real(0.17), Real(0.23), Real(0.19)}, Real(1e-9)},
    };

    for (const auto& c : cases) {
        SpectralBasis basis(c.type, c.order);
        std::vector<Real> values_ref, values_all;
        std::vector<Gradient> gradients_ref, gradients_all;
        std::vector<Hessian> hessians_ref, hessians_all;
        basis.evaluate_values(c.point, values_ref);
        basis.evaluate_gradients(c.point, gradients_ref);
        basis.evaluate_hessians(c.point, hessians_ref);
        basis.evaluate_all(c.point, values_all, gradients_all, hessians_all);

        ASSERT_EQ(values_all.size(), values_ref.size());
        ASSERT_EQ(gradients_all.size(), gradients_ref.size());
        ASSERT_EQ(hessians_all.size(), hessians_ref.size());
        for (std::size_t i = 0; i < values_ref.size(); ++i) {
            EXPECT_NEAR(values_all[i], values_ref[i], c.tolerance) << "value i=" << i;
            for (std::size_t d = 0; d < 3u; ++d) {
                EXPECT_NEAR(gradients_all[i][d], gradients_ref[i][d], c.tolerance)
                    << "gradient i=" << i << " d=" << d;
                for (std::size_t e = 0; e < 3u; ++e) {
                    EXPECT_NEAR(hessians_all[i](d, e), hessians_ref[i](d, e), Real(1e-7))
                        << "hessian i=" << i << " d=" << d << " e=" << e;
                }
            }
        }
    }
}

TEST(SpectralBasis, VandermondeDiagnosticsAcceptSupportedOrders) {
    EXPECT_NO_THROW((void)SpectralBasis(ElementType::Triangle3, 6));
    EXPECT_NO_THROW((void)SpectralBasis(ElementType::Tetra4, 4));
    EXPECT_NO_THROW((void)SpectralBasis(ElementType::Pyramid5, 3));
    EXPECT_NO_THROW((void)LagrangeBasis(ElementType::Pyramid5, 5));
}

TEST(SpectralBasis, CachedVandermondeConstructionIsThreadSafe) {
    constexpr std::size_t num_threads = 8;
    const struct Case {
        ElementType type;
        int order;
        Point point;
    } cases[] = {
        {ElementType::Triangle3, 4, Point{Real(0.21), Real(0.32), Real(0)}},
        {ElementType::Tetra4, 3, Point{Real(0.17), Real(0.21), Real(0.19)}},
        {ElementType::Pyramid5, 3, Point{Real(0.08), Real(-0.13), Real(0.24)}},
    };

    std::vector<Real> baseline;
    for (const auto& c : cases) {
        SpectralBasis basis(c.type, c.order);
        std::vector<Real> values;
        basis.evaluate_values(c.point, values);
        baseline.insert(baseline.end(), values.begin(), values.end());
    }

    std::array<std::vector<Real>, num_threads> observed;
    std::array<std::exception_ptr, num_threads> errors{};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (std::size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
        threads.emplace_back([&, thread_index]() {
            try {
                for (const auto& c : cases) {
                    SpectralBasis basis(c.type, c.order);
                    std::vector<Real> values;
                    basis.evaluate_values(c.point, values);
                    observed[thread_index].insert(observed[thread_index].end(),
                                                  values.begin(),
                                                  values.end());
                }
            } catch (...) {
                errors[thread_index] = std::current_exception();
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    for (std::size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
        ASSERT_FALSE(errors[thread_index]);
        ASSERT_EQ(observed[thread_index].size(), baseline.size());
        for (std::size_t i = 0; i < baseline.size(); ++i) {
            EXPECT_NEAR(observed[thread_index][i], baseline[i], Real(1e-10))
                << "thread=" << thread_index << " entry=" << i;
        }
    }
}

TEST(SpectralBasis, RawAndQuadratureOutputsMatchVectorEvaluation) {
    const struct Case {
        ElementType type;
        int order;
        std::vector<Point> points;
        Real tolerance;
    } cases[] = {
        {ElementType::Line2, 4,
         {Point{Real(-0.37), Real(0), Real(0)}, Point{Real(0.42), Real(0), Real(0)}},
         Real(1e-10)},
        {ElementType::Triangle3, 2,
         {Point{Real(0.22), Real(0.31), Real(0)}, Point{Real(0.12), Real(0.47), Real(0)}},
         Real(1e-9)},
        {ElementType::Tetra4, 2,
         {Point{Real(0.14), Real(0.21), Real(0.18)}, Point{Real(0.27), Real(0.16), Real(0.23)}},
         Real(1e-9)},
        {ElementType::Quad4, 3,
         {Point{Real(-0.24), Real(0.33), Real(0)}, Point{Real(0.41), Real(-0.19), Real(0)}},
         Real(1e-9)},
        {ElementType::Hex8, 2,
         {Point{Real(-0.22), Real(0.31), Real(-0.41)}, Point{Real(0.37), Real(-0.28), Real(0.19)}},
         Real(1e-9)},
        {ElementType::Wedge6, 2,
         {Point{Real(0.21), Real(0.18), Real(-0.37)}, Point{Real(0.11), Real(0.42), Real(0.29)}},
         Real(1e-9)},
        {ElementType::Pyramid5, 2,
         {Point{Real(0.09), Real(-0.14), Real(0.35)}, Point{Real(-0.18), Real(0.21), Real(0.22)}},
         Real(1e-8)},
    };

    for (const auto& c : cases) {
        SpectralBasis basis(c.type, c.order);
        const std::size_t num_dofs = basis.size();
        const std::size_t num_qpts = c.points.size();

        std::vector<Real> expected_values;
        std::vector<Gradient> expected_gradients;
        std::vector<Hessian> expected_hessians;
        basis.evaluate_all(c.points.front(), expected_values, expected_gradients, expected_hessians);

        std::vector<Real> raw_values(num_dofs);
        std::vector<Real> raw_gradients(num_dofs * 3u);
        std::vector<Real> raw_hessians(num_dofs * 9u);
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
        std::vector<Real> strided_values(num_dofs * stride, Real(-13));
        std::vector<Real> strided_gradients(num_dofs * 3u * stride, Real(-13));
        std::vector<Real> strided_hessians(num_dofs * 9u * stride, Real(-13));
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
            for (std::size_t pad = num_qpts; pad < stride; ++pad) {
                EXPECT_DOUBLE_EQ(strided_values[i * stride + pad], Real(-13));
            }
        }
    }
}

TEST(SpectralBasis, TensorProductStridedOutputCombinationsMatchPointwise) {
    const std::vector<SpectralStridedRequest> requests = {
        {true, false, false},
        {false, true, false},
        {false, false, true},
        {true, true, false},
        {true, false, true},
        {false, true, true},
        {true, true, true},
    };

    SpectralBasis quad(ElementType::Quad4, 4);
    const std::vector<Point> quad_points = {
        {Real(-0.42), Real(0.33), Real(0)},
        {Real(0.17), Real(-0.51), Real(0)},
        {Real(0.68), Real(0.44), Real(0)},
    };
    SpectralBasis hex(ElementType::Hex8, 3);
    const std::vector<Point> hex_points = {
        {Real(-0.42), Real(0.33), Real(-0.25)},
        {Real(0.17), Real(-0.51), Real(0.58)},
        {Real(0.68), Real(0.44), Real(-0.62)},
    };
    SpectralBasis wedge(ElementType::Wedge6, 3);
    const std::vector<Point> wedge_points = {
        {Real(0.12), Real(0.23), Real(-0.41)},
        {Real(0.31), Real(0.17), Real(0.26)},
        {Real(0.18), Real(0.45), Real(0.67)},
    };
    SpectralBasis triangle(ElementType::Triangle3, 3);
    const std::vector<Point> triangle_points = {
        {Real(0.12), Real(0.23), Real(0)},
        {Real(0.31), Real(0.17), Real(0)},
        {Real(0.18), Real(0.45), Real(0)},
    };
    SpectralBasis tetra(ElementType::Tetra4, 2);
    const std::vector<Point> tetra_points = {
        {Real(0.12), Real(0.23), Real(0.11)},
        {Real(0.31), Real(0.17), Real(0.22)},
        {Real(0.18), Real(0.45), Real(0.08)},
    };
    SpectralBasis pyramid(ElementType::Pyramid5, 2);
    const std::vector<Point> pyramid_points = {
        {Real(0.10), Real(-0.16), Real(0.25)},
        {Real(-0.21), Real(0.18), Real(0.36)},
        {Real(0.03), Real(0.07), Real(0.62)},
    };

    for (const auto& request : requests) {
        SCOPED_TRACE(request.values ? "values" : "no values");
        SCOPED_TRACE(request.gradients ? "gradients" : "no gradients");
        SCOPED_TRACE(request.hessians ? "hessians" : "no hessians");
        expect_spectral_strided_matches_pointwise(quad, quad_points, request, Real(1e-10));
        expect_spectral_strided_matches_pointwise(hex, hex_points, request, Real(1e-10));
        expect_spectral_strided_matches_pointwise(wedge, wedge_points, request, Real(1e-9));
        expect_spectral_strided_matches_pointwise(triangle, triangle_points, request, Real(1e-9));
        expect_spectral_strided_matches_pointwise(tetra, tetra_points, request, Real(1e-9));
        expect_spectral_strided_matches_pointwise(pyramid, pyramid_points, request, Real(1e-8));
    }
}

TEST(HierarchicalBasis, TriangleModalGramPositiveDefinite) {
    const int order = 3;
    HierarchicalBasis basis(ElementType::Triangle3, order);
    // Need enough quadrature accuracy to integrate products up to degree 2*order
    TriangleQuadrature quad(2 * order + 2);

    const std::size_t n = basis.size();
    std::vector<std::vector<double>> G(n, std::vector<double>(n, 0.0));
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        std::vector<Real> vals;
        basis.evaluate_values(quad.point(q), vals);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                G[i][j] += quad.weight(q) * vals[i] * vals[j];
            }
        }
    }

    // Simple Cholesky-like check for positive definiteness
    for (std::size_t k = 0; k < n; ++k) {
        double diag = G[k][k];
        for (std::size_t j = 0; j < k; ++j) {
            diag -= G[k][j] * G[k][j];
        }
        EXPECT_GT(diag, 1e-10);
        double inv = 1.0 / std::sqrt(std::max(diag, 1e-12));
        for (std::size_t i = k + 1; i < n; ++i) {
            double val = G[i][k];
            for (std::size_t j = 0; j < k; ++j) {
                val -= G[i][j] * G[k][j];
            }
            G[i][k] = val * inv;
        }
    }
}

TEST(HierarchicalBasis, TetraModalGramPositiveDefinite) {
    const int order = 2;
    HierarchicalBasis basis(ElementType::Tetra4, order);
    quadrature::TetrahedronQuadrature quad(order * 2 + 2);

    const std::size_t n = basis.size();
    std::vector<std::vector<double>> G(n, std::vector<double>(n, 0.0));
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        std::vector<Real> vals;
        basis.evaluate_values(quad.point(q), vals);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                G[i][j] += quad.weight(q) * vals[i] * vals[j];
            }
        }
    }

    // Simple Cholesky-like positive definiteness check
    for (std::size_t k = 0; k < n; ++k) {
        double diag = G[k][k];
        for (std::size_t j = 0; j < k; ++j) {
            diag -= G[k][j] * G[k][j];
        }
        EXPECT_GT(diag, 1e-8);
        double inv = 1.0 / std::sqrt(std::max(diag, 1e-12));
        for (std::size_t i = k + 1; i < n; ++i) {
            double val = G[i][k];
            for (std::size_t j = 0; j < k; ++j) {
                val -= G[i][j] * G[k][j];
            }
            G[i][k] = val * inv;
        }
    }
}

TEST(HierarchicalBasis, WedgeModalGramPositiveDefinite) {
    const int order = 2;
    HierarchicalBasis basis(ElementType::Wedge6, order);
    quadrature::WedgeQuadrature quad(2 * order + 2);

    const std::size_t n = basis.size();
    std::vector<std::vector<double>> G(n, std::vector<double>(n, 0.0));
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        std::vector<Real> vals;
        basis.evaluate_values(quad.point(q), vals);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                G[i][j] += quad.weight(q) * vals[i] * vals[j];
            }
        }
    }

    for (std::size_t k = 0; k < n; ++k) {
        double diag = G[k][k];
        for (std::size_t j = 0; j < k; ++j) {
            diag -= G[k][j] * G[k][j];
        }
        EXPECT_GT(diag, 1e-8);
        double inv = 1.0 / std::sqrt(std::max(diag, 1e-12));
        for (std::size_t i = k + 1; i < n; ++i) {
            double val = G[i][k];
            for (std::size_t j = 0; j < k; ++j) {
                val -= G[i][j] * G[k][j];
            }
            G[i][k] = val * inv;
        }
    }
}

TEST(HierarchicalBasis, PyramidModalGramPositiveDefinite) {
    const int order = 2;
    HierarchicalBasis basis(ElementType::Pyramid5, order);
    quadrature::PyramidQuadrature quad(2 * order + 2);

    const std::size_t n = basis.size();
    std::vector<std::vector<double>> G(n, std::vector<double>(n, 0.0));
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        std::vector<Real> vals;
        basis.evaluate_values(quad.point(q), vals);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                G[i][j] += quad.weight(q) * vals[i] * vals[j];
            }
        }
    }

    for (std::size_t k = 0; k < n; ++k) {
        double diag = G[k][k];
        for (std::size_t j = 0; j < k; ++j) {
            diag -= G[k][j] * G[k][j];
        }
        EXPECT_GT(diag, 1e-8);
        double inv = 1.0 / std::sqrt(std::max(diag, 1e-12));
        for (std::size_t i = k + 1; i < n; ++i) {
            double val = G[i][k];
            for (std::size_t j = 0; j < k; ++j) {
                val -= G[i][j] * G[k][j];
            }
            G[i][k] = val * inv;
        }
    }
}

TEST(HierarchicalBasis, BoundarySweepValuesAndGradientsRemainFinite) {
    const struct Case {
        ElementType type;
        int order;
    } cases[] = {
        {ElementType::Line2, 5},
        {ElementType::Triangle3, 4},
        {ElementType::Quad4, 4},
        {ElementType::Tetra4, 3},
        {ElementType::Hex8, 3},
        {ElementType::Wedge6, 3},
        {ElementType::Pyramid5, 3},
    };

    for (const auto& c : cases) {
        HierarchicalBasis basis(c.type, c.order);
        for (const auto& xi : boundary_stress_points_for(c.type)) {
            std::vector<Real> values;
            std::vector<Gradient> grads;
            basis.evaluate_values(xi, values);
            basis.evaluate_gradients(xi, grads);

            ASSERT_EQ(values.size(), basis.size());
            ASSERT_EQ(grads.size(), basis.size());
            expect_values_and_gradients_finite(values, grads, basis.dimension());
        }
    }
}

TEST(SpectralBasis, HexKroneckerAndPartitionOfUnity) {
    const int order = 3;
    SpectralBasis basis(ElementType::Hex8, order);
    const auto& nodes = basis.nodes_1d();
    const std::size_t n1d = nodes.size();
    const std::size_t n = basis.size();
    ASSERT_EQ(n, n1d * n1d * n1d);

    // Kronecker property at tensor-product nodes in 3D
    for (std::size_t k = 0; k < n1d; ++k) {
        for (std::size_t j = 0; j < n1d; ++j) {
            for (std::size_t i = 0; i < n1d; ++i) {
                svmp::FE::math::Vector<Real, 3> xi{nodes[i], nodes[j], nodes[k]};
                std::vector<Real> vals;
                basis.evaluate_values(xi, vals);
                ASSERT_EQ(vals.size(), n);
                const std::size_t active = k * n1d * n1d + j * n1d + i;
                for (std::size_t idx = 0; idx < n; ++idx) {
                    if (idx == active) {
                        EXPECT_NEAR(vals[idx], 1.0, 1e-12);
                    } else {
                        EXPECT_NEAR(vals[idx], 0.0, 1e-12);
                    }
                }
            }
        }
    }

    // Partition of unity at a generic interior point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.3)};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(SpectralBasis, WedgeKroneckerAndPartitionOfUnity) {
    const int order = 3;
    SpectralBasis basis(ElementType::Wedge6, order);
    const auto& nodes = basis.interpolation_nodes();
    ASSERT_EQ(nodes.size(), basis.size());

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        std::vector<Real> vals;
        basis.evaluate_values(nodes[i], vals);
        ASSERT_EQ(vals.size(), basis.size());
        for (std::size_t j = 0; j < vals.size(); ++j) {
            EXPECT_NEAR(vals[j], (i == j) ? Real(1) : Real(0), 1e-10)
                << "i=" << i << ", j=" << j;
        }
    }

    std::vector<Real> vals;
    basis.evaluate_values({Real(0.2), Real(0.25), Real(0.3)}, vals);
    EXPECT_NEAR(std::accumulate(vals.begin(), vals.end(), Real(0)), Real(1), 1e-10);
}

TEST(SpectralBasis, PyramidKroneckerAndPartitionOfUnity) {
    const int order = 3;
    SpectralBasis basis(ElementType::Pyramid5, order);
    const auto& nodes = basis.interpolation_nodes();
    ASSERT_EQ(nodes.size(), basis.size());

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        std::vector<Real> vals;
        basis.evaluate_values(nodes[i], vals);
        ASSERT_EQ(vals.size(), basis.size());
        for (std::size_t j = 0; j < vals.size(); ++j) {
            EXPECT_NEAR(vals[j], (i == j) ? Real(1) : Real(0), 5e-8)
                << "i=" << i << ", j=" << j;
        }
    }

    std::vector<Real> vals;
    basis.evaluate_values({Real(0.12), Real(-0.08), Real(0.3)}, vals);
    EXPECT_NEAR(std::accumulate(vals.begin(), vals.end(), Real(0)), Real(1), 1e-9);
}

TEST(SpectralBasis, WedgeAndPyramidTraceNodesMatchFaceFamilies) {
    const int order = 3;

    SpectralBasis wedge(ElementType::Wedge6, order);
    SpectralBasis tri(ElementType::Triangle3, order);
    std::size_t bottom_count = 0;
    std::size_t top_count = 0;
    for (const auto& node : wedge.interpolation_nodes()) {
        if (std::abs(node[2] + Real(1)) <= Real(1e-12)) {
            ++bottom_count;
            const bool found = std::any_of(
                tri.interpolation_nodes().begin(), tri.interpolation_nodes().end(),
                [&](const Point& tri_node) {
                    return same_point(node, {tri_node[0], tri_node[1], Real(-1)});
                });
            EXPECT_TRUE(found);
        }
        if (std::abs(node[2] - Real(1)) <= Real(1e-12)) {
            ++top_count;
        }
    }
    EXPECT_EQ(bottom_count, tri.size());
    EXPECT_EQ(top_count, tri.size());

    SpectralBasis pyramid(ElementType::Pyramid5, order);
    const auto& gll = pyramid.nodes_1d();
    std::size_t base_count = 0;
    for (const auto& node : pyramid.interpolation_nodes()) {
        if (std::abs(node[2]) <= Real(1e-12)) {
            ++base_count;
            const bool x_match = std::any_of(gll.begin(), gll.end(),
                                             [&](Real gx) { return std::abs(node[0] - gx) <= Real(1e-12); });
            const bool y_match = std::any_of(gll.begin(), gll.end(),
                                             [&](Real gy) { return std::abs(node[1] - gy) <= Real(1e-12); });
            EXPECT_TRUE(x_match);
            EXPECT_TRUE(y_match);
        }
    }
    EXPECT_EQ(base_count, static_cast<std::size_t>((order + 1) * (order + 1)));

    std::size_t front_count = 0;
    for (const auto& node : pyramid.interpolation_nodes()) {
        if (std::abs(node[1] - (node[2] - Real(1))) <= Real(1e-10)) {
            ++front_count;
        }
    }
    EXPECT_EQ(front_count, tri.size());

    std::size_t front_interior_count = 0;
    for (const auto& tri_node : tri.interpolation_nodes()) {
        if (tri_node[0] <= Real(1e-10) ||
            tri_node[1] <= Real(1e-10) ||
            tri_node[0] + tri_node[1] >= Real(1) - Real(1e-10)) {
            continue;
        }
        const Point mapped = map_triangle_to_pyramid_front(tri_node);
        const bool found = std::any_of(
            pyramid.interpolation_nodes().begin(), pyramid.interpolation_nodes().end(),
            [&](const Point& node) { return same_point(node, mapped, Real(1e-10)); });
        EXPECT_TRUE(found);
        ++front_interior_count;
    }
    EXPECT_GT(front_interior_count, 0u);
}

TEST(SpectralBasis, BoundarySweepMaintainsPartitionAndFiniteGradients) {
    const struct Case {
        ElementType type;
        int order;
    } cases[] = {
        {ElementType::Line2, 5},
        {ElementType::Triangle3, 4},
        {ElementType::Quad4, 4},
        {ElementType::Tetra4, 3},
        {ElementType::Hex8, 3},
        {ElementType::Wedge6, 3},
        {ElementType::Pyramid5, 3},
    };

    for (const auto& c : cases) {
        SpectralBasis basis(c.type, c.order);
        for (const auto& xi : boundary_stress_points_for(c.type)) {
            std::vector<Real> values;
            std::vector<Gradient> grads;
            basis.evaluate_values(xi, values);
            basis.evaluate_gradients(xi, grads);

            ASSERT_EQ(values.size(), basis.size());
            ASSERT_EQ(grads.size(), basis.size());
            expect_values_and_gradients_finite(values, grads, basis.dimension());

            Real sum = Real(0);
            Gradient grad_sum{};
            for (std::size_t i = 0; i < values.size(); ++i) {
                sum += values[i];
                for (int d = 0; d < basis.dimension(); ++d) {
                    grad_sum[static_cast<std::size_t>(d)] += grads[i][static_cast<std::size_t>(d)];
                }
            }

            EXPECT_NEAR(sum, Real(1), Real(1e-10))
                << "type=" << static_cast<int>(c.type) << ", order=" << c.order;
            for (int d = 0; d < basis.dimension(); ++d) {
                EXPECT_NEAR(grad_sum[static_cast<std::size_t>(d)], Real(0), Real(1e-8))
                    << "type=" << static_cast<int>(c.type)
                    << ", order=" << c.order << ", dim=" << d;
            }
        }
    }
}

TEST(SpectralBasis, GLLNodeAccuracy) {
    // Compare 1D GLL nodes against known reference values for low orders.
    // GLL nodes are roots of (1-x^2)*P'_n(x) = 0 plus endpoints.
    {
        SpectralBasis basis(ElementType::Line2, 2);
        const auto& nodes = basis.nodes_1d();
        ASSERT_EQ(nodes.size(), 3u);
        EXPECT_NEAR(nodes[0], -1.0, 1e-14);
        EXPECT_NEAR(nodes[1], 0.0, 1e-14);
        EXPECT_NEAR(nodes[2], 1.0, 1e-14);
    }
    {
        SpectralBasis basis(ElementType::Line2, 3);
        const auto& nodes = basis.nodes_1d();
        ASSERT_EQ(nodes.size(), 4u);
        EXPECT_NEAR(nodes[0], -1.0, 1e-14);
        EXPECT_NEAR(nodes[1], -std::sqrt(1.0 / 5.0), 1e-12);
        EXPECT_NEAR(nodes[2], std::sqrt(1.0 / 5.0), 1e-12);
        EXPECT_NEAR(nodes[3], 1.0, 1e-14);
    }
    {
        SpectralBasis basis(ElementType::Line2, 4);
        const auto& nodes = basis.nodes_1d();
        ASSERT_EQ(nodes.size(), 5u);
        EXPECT_NEAR(nodes[0], -1.0, 1e-14);
        EXPECT_NEAR(nodes[1], -std::sqrt(3.0 / 7.0), 1e-12);
        EXPECT_NEAR(nodes[2], 0.0, 1e-12);
        EXPECT_NEAR(nodes[3], std::sqrt(3.0 / 7.0), 1e-12);
        EXPECT_NEAR(nodes[4], 1.0, 1e-14);
    }
}

TEST(SpectralBasis, GLLNodesMatchQuadratureRawNodes) {
    for (int num_points : {2, 3, 4, 7, 10}) {
        const auto from_orthopoly = orthopoly::gll_nodes(num_points);
        const auto from_quadrature = GaussLobattoQuadrature1D::generate_raw(num_points);
        ASSERT_EQ(from_orthopoly.size(), from_quadrature.first.size());
        for (std::size_t i = 0; i < from_orthopoly.size(); ++i) {
            EXPECT_NEAR(from_orthopoly[i], from_quadrature.first[i], 1e-14)
                << "num_points=" << num_points << ", node=" << i;
        }
    }
}

TEST(SpectralBasis, GradientsMatchFiniteDifference2D) {
    const int order = 3;
    SpectralBasis basis(ElementType::Quad4, order);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.3), Real(-0.2), Real(0)};
    const Real eps = Real(1e-6);

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), basis.size());

    for (int d = 0; d < 2; ++d) {
        auto xi_p = xi, xi_m = xi;
        xi_p[static_cast<std::size_t>(d)] += eps;
        xi_m[static_cast<std::size_t>(d)] -= eps;

        std::vector<Real> vals_p, vals_m;
        basis.evaluate_values(xi_p, vals_p);
        basis.evaluate_values(xi_m, vals_m);

        for (std::size_t i = 0; i < basis.size(); ++i) {
            const Real fd = (vals_p[i] - vals_m[i]) / (Real(2) * eps);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], fd, 1e-5)
                << "Basis " << i << ", dim " << d;
        }
    }
}

TEST(SpectralBasis, GradientsMatchFiniteDifference3D) {
    const int order = 2;
    SpectralBasis basis(ElementType::Hex8, order);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(-0.3), Real(0.15)};
    const Real eps = Real(1e-6);

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), basis.size());

    for (int d = 0; d < 3; ++d) {
        auto xi_p = xi, xi_m = xi;
        xi_p[static_cast<std::size_t>(d)] += eps;
        xi_m[static_cast<std::size_t>(d)] -= eps;

        std::vector<Real> vals_p, vals_m;
        basis.evaluate_values(xi_p, vals_p);
        basis.evaluate_values(xi_m, vals_m);

        for (std::size_t i = 0; i < basis.size(); ++i) {
            const Real fd = (vals_p[i] - vals_m[i]) / (Real(2) * eps);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], fd, 1e-5)
                << "Basis " << i << ", dim " << d;
        }
    }
}

TEST(SpectralBasis, WedgeGradientMatchesFiniteDifference) {
    const int order = 3;
    SpectralBasis basis(ElementType::Wedge6, order);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.25), Real(0.2)};
    const Real eps = Real(1e-6);

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), basis.size());

    for (int d = 0; d < 3; ++d) {
        auto xi_p = xi, xi_m = xi;
        xi_p[static_cast<std::size_t>(d)] += eps;
        xi_m[static_cast<std::size_t>(d)] -= eps;

        std::vector<Real> vals_p, vals_m;
        basis.evaluate_values(xi_p, vals_p);
        basis.evaluate_values(xi_m, vals_m);

        for (std::size_t i = 0; i < basis.size(); ++i) {
            const Real fd = (vals_p[i] - vals_m[i]) / (Real(2) * eps);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], fd, 1e-5)
                << "Basis " << i << ", dim " << d;
        }
    }
}

TEST(SpectralBasis, PyramidGradientMatchesFiniteDifference) {
    const int order = 2;
    SpectralBasis basis(ElementType::Pyramid5, order);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.12), Real(-0.08), Real(0.28)};
    const Real eps = Real(1e-6);

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), basis.size());

    for (int d = 0; d < 3; ++d) {
        auto xi_p = xi, xi_m = xi;
        xi_p[static_cast<std::size_t>(d)] += eps;
        xi_m[static_cast<std::size_t>(d)] -= eps;

        std::vector<Real> vals_p, vals_m;
        basis.evaluate_values(xi_p, vals_p);
        basis.evaluate_values(xi_m, vals_m);

        for (std::size_t i = 0; i < basis.size(); ++i) {
            const Real fd = (vals_p[i] - vals_m[i]) / (Real(2) * eps);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], fd, 2e-4)
                << "Basis " << i << ", dim " << d;
        }
    }
}

// =============================================================================
// Simplex SpectralBasis tests
// =============================================================================

TEST(SpectralBasis, TriangleNodesStayInReferenceSimplexAndRemainUnisolvent) {
    for (int order = 1; order <= 10; ++order) {
        SpectralBasis basis(ElementType::Triangle3, order);
        const auto& nodes = basis.interpolation_nodes();
        expect_triangle_nodes_in_reference(nodes, order);
        expect_simplex_nodes_unique(nodes);
        expect_nodal_interpolation_identity(basis, Real(2e-9));
    }
}

TEST(SpectralBasis, TriangleKroneckerAndPartitionOfUnity) {
    for (int order = 1; order <= 5; ++order) {
        SpectralBasis basis(ElementType::Triangle3, order);
        EXPECT_TRUE(basis.is_simplex());
        EXPECT_EQ(basis.dimension(), 2);

        const std::size_t expected_size =
            static_cast<std::size_t>((order + 1) * (order + 2) / 2);
        EXPECT_EQ(basis.size(), expected_size);

        // Partition of unity at interior point
        svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(0)};
        std::vector<Real> vals;
        basis.evaluate_values(xi, vals);
        ASSERT_EQ(vals.size(), expected_size);

        const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-10)
            << "Partition of unity failed for triangle order " << order;
    }
}

TEST(SpectralBasis, TriangleNodalKronecker) {
    // Verify partition of unity holds at each vertex (vertices are always nodes)
    for (int order = 1; order <= 4; ++order) {
        SpectralBasis basis(ElementType::Triangle3, order);
        const std::size_t n = basis.size();

        const svmp::FE::math::Vector<Real, 3> verts[] = {
            {Real(0), Real(0), Real(0)},
            {Real(1), Real(0), Real(0)},
            {Real(0), Real(1), Real(0)},
        };

        for (const auto& v : verts) {
            std::vector<Real> vals;
            basis.evaluate_values(v, vals);
            ASSERT_EQ(vals.size(), n);
            const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
            EXPECT_NEAR(sum, 1.0, 1e-10)
                << "Partition of unity at vertex for order " << order;
        }
    }
}

TEST(SpectralBasis, TriangleGradientMatchesFiniteDifference) {
    const int order = 3;
    SpectralBasis basis(ElementType::Triangle3, order);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.25), Real(0)};
    const Real eps = Real(1e-6);

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), basis.size());

    for (int d = 0; d < 2; ++d) {
        auto xi_p = xi, xi_m = xi;
        xi_p[static_cast<std::size_t>(d)] += eps;
        xi_m[static_cast<std::size_t>(d)] -= eps;

        std::vector<Real> vals_p, vals_m;
        basis.evaluate_values(xi_p, vals_p);
        basis.evaluate_values(xi_m, vals_m);

        for (std::size_t i = 0; i < basis.size(); ++i) {
            const Real fd = (vals_p[i] - vals_m[i]) / (Real(2) * eps);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], fd, 1e-4)
                << "Triangle order=" << order << ", basis " << i << ", dim " << d;
        }
    }
}

TEST(SpectralBasis, TriangleGradientsSumToZero) {
    for (int order = 2; order <= 4; ++order) {
        SpectralBasis basis(ElementType::Triangle3, order);
        svmp::FE::math::Vector<Real, 3> xi{Real(0.15), Real(0.25), Real(0)};
        std::vector<Gradient> grads;
        basis.evaluate_gradients(xi, grads);

        Gradient sum{};
        for (const auto& g : grads) {
            sum[0] += g[0];
            sum[1] += g[1];
        }
        EXPECT_NEAR(sum[0], 0.0, 1e-8) << "Triangle order=" << order;
        EXPECT_NEAR(sum[1], 0.0, 1e-8) << "Triangle order=" << order;
    }
}

TEST(SpectralBasis, TetraNodesStayInReferenceSimplexAndRemainUnisolvent) {
    for (int order = 1; order <= 4; ++order) {
        SpectralBasis basis(ElementType::Tetra4, order);
        const auto& nodes = basis.interpolation_nodes();
        expect_tetra_nodes_in_reference(nodes, order);
        expect_simplex_nodes_unique(nodes);
        expect_nodal_interpolation_identity(basis, Real(1e-8));
    }
}

TEST(SpectralBasis, TetraKroneckerAndPartitionOfUnity) {
    for (int order = 1; order <= 3; ++order) {
        SpectralBasis basis(ElementType::Tetra4, order);
        EXPECT_TRUE(basis.is_simplex());
        EXPECT_EQ(basis.dimension(), 3);

        const std::size_t expected_size =
            static_cast<std::size_t>((order + 1) * (order + 2) * (order + 3) / 6);
        EXPECT_EQ(basis.size(), expected_size);

        svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(0.2), Real(0.15)};
        std::vector<Real> vals;
        basis.evaluate_values(xi, vals);
        ASSERT_EQ(vals.size(), expected_size);

        const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-10)
            << "Partition of unity failed for tet order " << order;

        const svmp::FE::math::Vector<Real, 3> verts[] = {
            {Real(0), Real(0), Real(0)},
            {Real(1), Real(0), Real(0)},
            {Real(0), Real(1), Real(0)},
            {Real(0), Real(0), Real(1)},
        };
        for (const auto& v : verts) {
            basis.evaluate_values(v, vals);
            const double vsum = std::accumulate(vals.begin(), vals.end(), 0.0);
            EXPECT_NEAR(vsum, 1.0, 1e-10)
                << "Partition of unity at vertex for tet order " << order;
        }
    }
}

TEST(SpectralBasis, TetraGradientMatchesFiniteDifference) {
    const int order = 2;
    SpectralBasis basis(ElementType::Tetra4, order);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.15), Real(0.2), Real(0.1)};
    const Real eps = Real(1e-6);

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), basis.size());

    for (int d = 0; d < 3; ++d) {
        auto xi_p = xi, xi_m = xi;
        xi_p[static_cast<std::size_t>(d)] += eps;
        xi_m[static_cast<std::size_t>(d)] -= eps;

        std::vector<Real> vals_p, vals_m;
        basis.evaluate_values(xi_p, vals_p);
        basis.evaluate_values(xi_m, vals_m);

        for (std::size_t i = 0; i < basis.size(); ++i) {
            const Real fd = (vals_p[i] - vals_m[i]) / (Real(2) * eps);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], fd, 1e-4)
                << "Tet order=" << order << ", basis " << i << ", dim " << d;
        }
    }
}

TEST(SpectralBasis, TetraGradientsSumToZero) {
    for (int order = 1; order <= 3; ++order) {
        SpectralBasis basis(ElementType::Tetra4, order);
        svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(0.15), Real(0.1)};
        std::vector<Gradient> grads;
        basis.evaluate_gradients(xi, grads);

        Gradient sum{};
        for (const auto& g : grads) {
            sum[0] += g[0];
            sum[1] += g[1];
            sum[2] += g[2];
        }
        EXPECT_NEAR(sum[0], 0.0, 1e-8) << "Tet order=" << order;
        EXPECT_NEAR(sum[1], 0.0, 1e-8) << "Tet order=" << order;
        EXPECT_NEAR(sum[2], 0.0, 1e-8) << "Tet order=" << order;
    }
}
