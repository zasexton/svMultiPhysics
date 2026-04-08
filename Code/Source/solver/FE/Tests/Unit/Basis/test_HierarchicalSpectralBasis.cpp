/**
 * @file test_HierarchicalSpectralBasis.cpp
 * @brief Tests for hierarchical and spectral bases
 */

#include <gtest/gtest.h>
#include "FE/Basis/HierarchicalBasis.h"
#include "FE/Basis/SpectralBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/WedgeQuadrature.h"
#include "FE/Quadrature/PyramidQuadrature.h"
#include <cmath>
#include <numeric>

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

TEST(SpectralBasis, WedgeAndPyramidSupportContractIsExplicit) {
    const int order = 2;
    try {
        (void)SpectralBasis(ElementType::Wedge6, order);
        FAIL() << "Expected FEException";
    } catch (const svmp::FE::FEException& e) {
        EXPECT_EQ(e.status(), svmp::FE::FEStatus::NotImplemented);
    }

    try {
        (void)SpectralBasis(ElementType::Pyramid5, order);
        FAIL() << "Expected FEException";
    } catch (const svmp::FE::FEException& e) {
        EXPECT_EQ(e.status(), svmp::FE::FEStatus::NotImplemented);
    }
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

// =============================================================================
// Simplex SpectralBasis (Warp & Blend) tests
// =============================================================================

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
