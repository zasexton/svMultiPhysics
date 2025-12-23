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

TEST(SpectralBasis, WedgeAndPyramidThrowNotImplemented) {
    const int order = 2;
    EXPECT_THROW(SpectralBasis(ElementType::Wedge6, order), svmp::FE::FEException);
    EXPECT_THROW(SpectralBasis(ElementType::Pyramid5, order), svmp::FE::FEException);
}
