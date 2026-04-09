/**
 * @file test_BasisHessians.cpp
 * @brief Tests for analytical Hessian evaluation on scalar bases
 */

#include <gtest/gtest.h>
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/BernsteinBasis.h"
#include "FE/Basis/HierarchicalBasis.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Basis/TensorBasis.h"
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::basis;

TEST(BasisHessians, LagrangeQuadraticLineHasZeroSecondDerivativesAtCenter) {
    LagrangeBasis basis(ElementType::Line2, 2); // quadratic 1D basis on [-1,1]
    svmp::FE::math::Vector<Real, 3> xi{0.0, 0.0, 0.0};

    std::vector<Hessian> hess;
    basis.evaluate_hessians(xi, hess);

    ASSERT_EQ(hess.size(), basis.size());
    // For quadratic Lagrange on [-1,1], second derivatives are constant and sum to 0
    double sum_second = 0.0;
    for (const auto& H : hess) {
        sum_second += H(0,0);
    }
    EXPECT_NEAR(sum_second, 0.0, 1e-8);
}

TEST(BasisHessians, BernsteinQuadraticQuadGradientAndHessianSumZero) {
    BernsteinBasis basis(ElementType::Quad4, 2);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0)};

    std::vector<svmp::FE::basis::Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), basis.size());

    svmp::FE::basis::Gradient gsum{};
    for (const auto& g : grads) {
        gsum[0] += g[0];
        gsum[1] += g[1];
        gsum[2] += g[2];
    }
    EXPECT_NEAR(gsum[0], 0.0, 1e-8);
    EXPECT_NEAR(gsum[1], 0.0, 1e-8);
    EXPECT_NEAR(gsum[2], 0.0, 1e-8);

    std::vector<Hessian> hess;
    basis.evaluate_hessians(xi, hess);
    ASSERT_EQ(hess.size(), basis.size());
    svmp::FE::math::Matrix<Real, 3, 3> hsum{};
    for (const auto& H : hess) {
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                hsum(i, j) += H(i, j);
            }
        }
    }
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(hsum(i, j), 0.0, 5e-6);
        }
    }
}

// ============================================================================
// LagrangeBasis Hessian tests - verify analytical vs numerical
// ============================================================================

namespace {
// Helper to compute numerical Hessians using central differences
void numerical_hessian_helper(const BasisFunction& basis,
                              const math::Vector<Real, 3>& xi,
                              std::vector<Hessian>& hess,
                              Real eps = 1e-5) {
    hess.assign(basis.size(), Hessian{});
    const int dim = basis.dimension();

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            // Use central difference on gradients
            math::Vector<Real, 3> xi_p = xi, xi_m = xi;
            const std::size_t sj = static_cast<std::size_t>(j);
            xi_p[sj] += eps;
            xi_m[sj] -= eps;

            std::vector<Gradient> g_p, g_m;
            basis.evaluate_gradients(xi_p, g_p);
            basis.evaluate_gradients(xi_m, g_m);

            for (std::size_t n = 0; n < basis.size(); ++n) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj2 = static_cast<std::size_t>(j);
                hess[n](si, sj2) = (g_p[n][si] - g_m[n][si]) / (2 * eps);
            }
        }
    }
}

void numerical_gradient_helper(const BasisFunction& basis,
                               const math::Vector<Real, 3>& xi,
                               std::vector<Gradient>& grads,
                               Real eps);

void expect_gradients_match_numerical(const BasisFunction& basis,
                                      const math::Vector<Real, 3>& xi,
                                      Real tol) {
    std::vector<Gradient> analytical, numerical;
    basis.evaluate_gradients(xi, analytical);
    numerical_gradient_helper(basis, xi, numerical, Real(1e-6));

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int d = 0; d < basis.dimension(); ++d) {
            const std::size_t sd = static_cast<std::size_t>(d);
            EXPECT_NEAR(analytical[n][sd], numerical[n][sd], tol)
                << "Mismatch at basis " << n << ", dim " << d;
        }
    }
}

std::vector<math::Vector<Real, 3>> sample_points_for(ElementType type) {
    switch (type) {
        case ElementType::Line2:
            return {
                {Real(-0.65), Real(0), Real(0)},
                {Real(0.1), Real(0), Real(0)},
                {Real(0.7), Real(0), Real(0)}
            };
        case ElementType::Triangle3:
            return {
                {Real(0.15), Real(0.2), Real(0)},
                {Real(0.25), Real(0.1), Real(0)},
                {Real(0.2), Real(0.3), Real(0)}
            };
        case ElementType::Quad4:
            return {
                {Real(0.2), Real(-0.3), Real(0)},
                {Real(-0.45), Real(0.25), Real(0)},
                {Real(0.55), Real(0.15), Real(0)}
            };
        case ElementType::Tetra4:
            return {
                {Real(0.12), Real(0.18), Real(0.16)},
                {Real(0.2), Real(0.1), Real(0.18)},
                {Real(0.15), Real(0.2), Real(0.1)}
            };
        case ElementType::Hex8:
            return {
                {Real(0.1), Real(-0.2), Real(0.3)},
                {Real(-0.35), Real(0.25), Real(-0.15)},
                {Real(0.45), Real(0.1), Real(0.55)}
            };
        case ElementType::Wedge6:
            return {
                {Real(0.18), Real(0.22), Real(-0.2)},
                {Real(0.12), Real(0.16), Real(0.1)},
                {Real(0.25), Real(0.15), Real(0.45)}
            };
        case ElementType::Pyramid5:
            return {
                {Real(0.0), Real(0.0), Real(0.2)},
                {Real(0.12), Real(-0.08), Real(0.24)},
                {Real(-0.1), Real(0.15), Real(0.35)}
            };
        default:
            return {{Real(0), Real(0), Real(0)}};
    }
}

void expect_hessians_symmetric(const BasisFunction& basis,
                               const math::Vector<Real, 3>& xi,
                               Real tol) {
    std::vector<Hessian> hessians;
    basis.evaluate_hessians(xi, hessians);

    ASSERT_EQ(hessians.size(), basis.size());
    for (std::size_t n = 0; n < hessians.size(); ++n) {
        for (int i = 0; i < basis.dimension(); ++i) {
            for (int j = i + 1; j < basis.dimension(); ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(hessians[n](si, sj), hessians[n](sj, si), tol)
                    << "Mismatch at basis " << n << ", components (" << i << "," << j
                    << "), element " << static_cast<int>(basis.element_type())
                    << ", order " << basis.order();
            }
        }
    }
}

void expect_hessians_match_numerical(const BasisFunction& basis,
                                     const std::vector<math::Vector<Real, 3>>& points,
                                     Real tol,
                                     Real eps = Real(1e-5)) {
    for (const auto& xi : points) {
        std::vector<Hessian> analytical;
        std::vector<Hessian> numerical;
        basis.evaluate_hessians(xi, analytical);
        numerical_hessian_helper(basis, xi, numerical, eps);

        ASSERT_EQ(analytical.size(), numerical.size());
        for (std::size_t n = 0; n < analytical.size(); ++n) {
            for (int i = 0; i < basis.dimension(); ++i) {
                for (int j = 0; j < basis.dimension(); ++j) {
                    const std::size_t si = static_cast<std::size_t>(i);
                    const std::size_t sj = static_cast<std::size_t>(j);
                    EXPECT_NEAR(analytical[n](si, sj), numerical[n](si, sj), tol)
                        << "Mismatch at basis " << n
                        << ", component (" << i << "," << j << ")"
                        << ", element " << static_cast<int>(basis.element_type())
                        << ", order " << basis.order()
                        << ", xi=(" << xi[0] << "," << xi[1] << "," << xi[2] << ")";
                }
            }
        }
    }
}
} // namespace

TEST(BasisHessians, LagrangeTriangleAnalyticalMatchesNumerical) {
    LagrangeBasis basis(ElementType::Triangle3, 3); // cubic triangle
    math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(0)};

    std::vector<Hessian> analytical, numerical;
    basis.evaluate_hessians(xi, analytical);
    numerical_hessian_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(analytical[n](si, sj), numerical[n](si, sj), 1e-6)
                    << "Mismatch at basis " << n << ", component (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, LagrangeTetrahedronAnalyticalMatchesNumerical) {
    LagrangeBasis basis(ElementType::Tetra4, 2); // quadratic tet
    math::Vector<Real, 3> xi{Real(0.15), Real(0.2), Real(0.1)};

    std::vector<Hessian> analytical, numerical;
    basis.evaluate_hessians(xi, analytical);
    numerical_hessian_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(analytical[n](si, sj), numerical[n](si, sj), 1e-6)
                    << "Mismatch at basis " << n << ", component (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, LagrangeHexahedronAnalyticalMatchesNumerical) {
    LagrangeBasis basis(ElementType::Hex8, 2); // quadratic hex
    math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.3)};

    std::vector<Hessian> analytical, numerical;
    basis.evaluate_hessians(xi, analytical);
    numerical_hessian_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(analytical[n](si, sj), numerical[n](si, sj), 1e-6)
                    << "Mismatch at basis " << n << ", component (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, LagrangeWedgeAnalyticalMatchesNumerical) {
    LagrangeBasis basis(ElementType::Wedge6, 2); // quadratic wedge
    math::Vector<Real, 3> xi{Real(0.2), Real(0.15), Real(-0.3)};

    std::vector<Hessian> analytical, numerical;
    basis.evaluate_hessians(xi, analytical);
    numerical_hessian_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(analytical[n](si, sj), numerical[n](si, sj), 1e-5)
                    << "Mismatch at basis " << n << ", component (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, LagrangePyramid5AnalyticalMatchesNumerical) {
    LagrangeBasis basis(ElementType::Pyramid5, 1); // rational Pyramid5 basis
    math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.3)};

    std::vector<Hessian> analytical, numerical;
    basis.evaluate_hessians(xi, analytical);
    numerical_hessian_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(analytical[n](si, sj), numerical[n](si, sj), 1e-6)
                    << "Mismatch at basis " << n << ", component (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, LagrangeQuadAnalyticalMatchesNumerical) {
    LagrangeBasis basis(ElementType::Quad4, 3); // cubic quad
    math::Vector<Real, 3> xi{Real(0.3), Real(-0.4), Real(0)};

    std::vector<Hessian> analytical, numerical;
    basis.evaluate_hessians(xi, analytical);
    numerical_hessian_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(analytical[n](si, sj), numerical[n](si, sj), 1e-6)
                    << "Mismatch at basis " << n << ", component (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, LagrangeHessiansSumToZero) {
    // Partition of unity implies sum of Hessians = 0
    const std::vector<std::pair<ElementType, int>> cases = {
        {ElementType::Line2, 3},
        {ElementType::Quad4, 2},
        {ElementType::Hex8, 2},
        {ElementType::Triangle3, 2},
        {ElementType::Tetra4, 2},
        {ElementType::Wedge6, 2},
        {ElementType::Pyramid5, 1}
    };

    for (const auto& [etype, order] : cases) {
        LagrangeBasis basis(etype, order);
        math::Vector<Real, 3> xi{Real(0.2), Real(0.15), Real(0.1)};

        std::vector<Hessian> hess;
        basis.evaluate_hessians(xi, hess);

        Hessian sum{};
        for (const auto& H : hess) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const std::size_t si = static_cast<std::size_t>(i);
                    const std::size_t sj = static_cast<std::size_t>(j);
                    sum(si, sj) += H(si, sj);
                }
            }
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(sum(si, sj), 0.0, 1e-10)
                    << "Element type " << static_cast<int>(etype) << ", order " << order
                    << ", component (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, LagrangeHessiansAreExplicitlySymmetric) {
    const struct Case {
        ElementType type;
        int order;
        math::Vector<Real, 3> xi;
        Real tol;
    } cases[] = {
        {ElementType::Line2, 3, {Real(0.15), Real(0), Real(0)}, Real(1e-12)},
        {ElementType::Triangle3, 3, {Real(0.2), Real(0.25), Real(0)}, Real(1e-10)},
        {ElementType::Quad4, 3, {Real(0.3), Real(-0.2), Real(0)}, Real(1e-12)},
        {ElementType::Tetra4, 2, {Real(0.15), Real(0.2), Real(0.1)}, Real(1e-10)},
        {ElementType::Hex8, 2, {Real(0.1), Real(-0.2), Real(0.3)}, Real(1e-12)},
        {ElementType::Wedge6, 2, {Real(0.2), Real(0.15), Real(-0.3)}, Real(1e-10)},
        {ElementType::Pyramid5, 1, {Real(0.1), Real(-0.2), Real(0.3)}, Real(1e-8)},
    };

    for (const auto& c : cases) {
        LagrangeBasis basis(c.type, c.order);
        expect_hessians_symmetric(basis, c.xi, c.tol);
    }
}

TEST(BasisHessians, LagrangeHigherOrderCanonicalPathsMatchNumericalAtMultiplePoints) {
    const struct Case {
        ElementType type;
        int order;
        Real tol;
        Real eps;
    } cases[] = {
        {ElementType::Line2, 5, Real(1e-6), Real(1e-5)},
        {ElementType::Triangle3, 4, Real(2e-6), Real(1e-5)},
        {ElementType::Quad4, 4, Real(1e-6), Real(1e-5)},
        {ElementType::Tetra4, 3, Real(3e-6), Real(1e-5)},
        {ElementType::Hex8, 3, Real(2e-6), Real(1e-5)},
        {ElementType::Wedge6, 3, Real(2e-5), Real(1e-5)},
        {ElementType::Pyramid5, 3, Real(3e-4), Real(2e-5)},
    };

    for (const auto& c : cases) {
        LagrangeBasis basis(c.type, c.order);
        expect_hessians_match_numerical(basis, sample_points_for(c.type), c.tol, c.eps);
    }
}

TEST(BasisHessians, LagrangeHigherOrderPyramidNearApexMatchesNumericalAndIsSymmetric) {
    LagrangeBasis basis(ElementType::Pyramid5, 4);
    const std::vector<math::Vector<Real, 3>> points = {
        {Real(0.0), Real(0.0), Real(0.82)},
        {Real(0.015), Real(-0.01), Real(0.9)},
        {Real(0.01), Real(0.005), Real(0.95)},
    };

    expect_hessians_match_numerical(basis, points, Real(8e-4), Real(2e-5));
    for (const auto& xi : points) {
        expect_hessians_symmetric(basis, xi, Real(5e-8));
    }
}

TEST(BasisHessians, SerendipityHessiansSumToZero) {
    const struct Case {
        ElementType type;
        int order;
        math::Vector<Real, 3> xi;
    } cases[] = {
        {ElementType::Hex20, 2, {Real(0.2), Real(-0.1), Real(0.3)}},
        {ElementType::Wedge15, 2, {Real(0.2), Real(0.3), Real(0.1)}},
        {ElementType::Pyramid13, 2, {Real(0.1), Real(-0.2), Real(0.4)}},
    };

    for (const auto& c : cases) {
        SerendipityBasis basis(c.type, c.order);
        std::vector<Hessian> hess;
        basis.evaluate_hessians(c.xi, hess);
        ASSERT_EQ(hess.size(), basis.size());

        Hessian sum{};
        for (const auto& H : hess) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const std::size_t si = static_cast<std::size_t>(i);
                    const std::size_t sj = static_cast<std::size_t>(j);
                    sum(si, sj) += H(si, sj);
                }
            }
        }

        for (int i = 0; i < basis.dimension(); ++i) {
            for (int j = 0; j < basis.dimension(); ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(sum(si, sj), 0.0, 1e-6)
                    << "Element type " << static_cast<int>(c.type)
                    << ", order " << c.order
                    << ", component (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, SerendipityHessiansAreExplicitlySymmetric) {
    const struct Case {
        ElementType type;
        int order;
        math::Vector<Real, 3> xi;
    } cases[] = {
        {ElementType::Quad8, 2, {Real(0.2), Real(-0.15), Real(0)}},
        {ElementType::Hex20, 2, {Real(0.2), Real(-0.1), Real(0.3)}},
        {ElementType::Wedge15, 2, {Real(0.2), Real(0.3), Real(0.1)}},
        {ElementType::Pyramid13, 2, {Real(0.1), Real(-0.2), Real(0.4)}},
    };

    for (const auto& c : cases) {
        SerendipityBasis basis(c.type, c.order);
        expect_hessians_symmetric(basis, c.xi, Real(5e-6));
    }
}

// ============================================================================
// HierarchicalBasis gradient tests - verify analytical vs numerical
// ============================================================================

namespace {
void numerical_gradient_helper(const BasisFunction& basis,
                               const math::Vector<Real, 3>& xi,
                               std::vector<Gradient>& grads,
                               Real eps = 1e-6) {
    grads.assign(basis.size(), Gradient{});
    const int dim = basis.dimension();

    for (int d = 0; d < dim; ++d) {
        math::Vector<Real, 3> xi_p = xi, xi_m = xi;
        const std::size_t sd = static_cast<std::size_t>(d);
        xi_p[sd] += eps;
        xi_m[sd] -= eps;

        std::vector<Real> v_p, v_m;
        basis.evaluate_values(xi_p, v_p);
        basis.evaluate_values(xi_m, v_m);

        for (std::size_t n = 0; n < basis.size(); ++n) {
            grads[n][sd] = (v_p[n] - v_m[n]) / (2 * eps);
        }
    }
}
} // namespace

TEST(BasisGradients, HierarchicalLineAnalyticalMatchesNumerical) {
    HierarchicalBasis basis(ElementType::Line2, 4);
    math::Vector<Real, 3> xi{Real(0.3), Real(0), Real(0)};

    std::vector<Gradient> analytical, numerical;
    basis.evaluate_gradients(xi, analytical);
    numerical_gradient_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        EXPECT_NEAR(analytical[n][0], numerical[n][0], 1e-5)
            << "Mismatch at basis " << n;
    }
}

TEST(BasisGradients, HierarchicalQuadAnalyticalMatchesNumerical) {
    HierarchicalBasis basis(ElementType::Quad4, 3);
    math::Vector<Real, 3> xi{Real(0.2), Real(-0.3), Real(0)};

    expect_gradients_match_numerical(basis, xi, Real(1e-5));
}

TEST(BasisGradients, HierarchicalTriangleAnalyticalMatchesNumerical) {
    HierarchicalBasis basis(ElementType::Triangle3, 4);
    math::Vector<Real, 3> xi{Real(0.21), Real(0.27), Real(0)};

    expect_gradients_match_numerical(basis, xi, Real(1e-5));
}

TEST(BasisGradients, HierarchicalHexAnalyticalMatchesNumerical) {
    HierarchicalBasis basis(ElementType::Hex8, 2);
    math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.3)};

    expect_gradients_match_numerical(basis, xi, Real(1e-5));
}

TEST(BasisGradients, HierarchicalWedgeAnalyticalMatchesNumerical) {
    HierarchicalBasis basis(ElementType::Wedge6, 3);
    math::Vector<Real, 3> xi{Real(0.18), Real(0.22), Real(0.15)};

    expect_gradients_match_numerical(basis, xi, Real(2e-5));
}

TEST(BasisGradients, HierarchicalTetraAnalyticalMatchesNumerical) {
    HierarchicalBasis basis(ElementType::Tetra4, 3);
    math::Vector<Real, 3> xi{Real(0.16), Real(0.19), Real(0.12)};

    expect_gradients_match_numerical(basis, xi, Real(2e-5));
}

TEST(BasisGradients, HierarchicalPyramidAnalyticalMatchesNumerical) {
    HierarchicalBasis basis(ElementType::Pyramid5, 3);
    math::Vector<Real, 3> xi{Real(0.12), Real(-0.08), Real(0.24)};

    expect_gradients_match_numerical(basis, xi, Real(2e-5));
}

TEST(BasisGradients, HierarchicalMultipleInteriorPointsMatchNumerical) {
    const struct Case {
        ElementType type;
        int order;
        Real tol;
    } cases[] = {
        {ElementType::Line2, 4, Real(1e-5)},
        {ElementType::Triangle3, 4, Real(1e-5)},
        {ElementType::Quad4, 3, Real(1e-5)},
        {ElementType::Tetra4, 3, Real(2e-5)},
        {ElementType::Hex8, 2, Real(1e-5)},
        {ElementType::Wedge6, 3, Real(2e-5)},
        {ElementType::Pyramid5, 3, Real(2e-5)},
    };

    for (const auto& c : cases) {
        HierarchicalBasis basis(c.type, c.order);
        for (const auto& xi : sample_points_for(c.type)) {
            expect_gradients_match_numerical(basis, xi, c.tol);
        }
    }
}

// ============================================================================
// BernsteinBasis gradient tests - verify analytical vs numerical
// ============================================================================

TEST(BasisGradients, BernsteinLineAnalyticalMatchesNumerical) {
    BernsteinBasis basis(ElementType::Line2, 4);
    math::Vector<Real, 3> xi{Real(0.3), Real(0), Real(0)};

    std::vector<Gradient> analytical, numerical;
    basis.evaluate_gradients(xi, analytical);
    numerical_gradient_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        EXPECT_NEAR(analytical[n][0], numerical[n][0], 1e-5)
            << "Mismatch at basis " << n;
    }
}

TEST(BasisGradients, BernsteinQuadAnalyticalMatchesNumerical) {
    BernsteinBasis basis(ElementType::Quad4, 3);
    math::Vector<Real, 3> xi{Real(0.2), Real(-0.3), Real(0)};

    std::vector<Gradient> analytical, numerical;
    basis.evaluate_gradients(xi, analytical);
    numerical_gradient_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int d = 0; d < 2; ++d) {
            const std::size_t sd = static_cast<std::size_t>(d);
            EXPECT_NEAR(analytical[n][sd], numerical[n][sd], 1e-5)
                << "Mismatch at basis " << n << ", dim " << d;
        }
    }
}

TEST(BasisGradients, BernsteinTriangleGradientsSumToZero) {
    // For triangle, verify partition of unity property: sum of gradients = 0
    BernsteinBasis basis(ElementType::Triangle3, 3);
    math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(0)};

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);

    Gradient sum{};
    for (const auto& g : grads) {
        sum[0] += g[0];
        sum[1] += g[1];
    }

    EXPECT_NEAR(sum[0], 0.0, 1e-10);
    EXPECT_NEAR(sum[1], 0.0, 1e-10);
}

TEST(BasisGradients, BernsteinHexAnalyticalMatchesNumerical) {
    BernsteinBasis basis(ElementType::Hex8, 2);
    math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.3)};

    std::vector<Gradient> analytical, numerical;
    basis.evaluate_gradients(xi, analytical);
    numerical_gradient_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int d = 0; d < 3; ++d) {
            const std::size_t sd = static_cast<std::size_t>(d);
            EXPECT_NEAR(analytical[n][sd], numerical[n][sd], 1e-5)
                << "Mismatch at basis " << n << ", dim " << d;
        }
    }
}

TEST(BasisGradients, BernsteinWedgeGradientsSumToZero) {
    // For wedge, verify partition of unity property: sum of gradients = 0
    BernsteinBasis basis(ElementType::Wedge6, 2);
    math::Vector<Real, 3> xi{Real(0.2), Real(0.15), Real(-0.3)};

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);

    Gradient sum{};
    for (const auto& g : grads) {
        sum[0] += g[0];
        sum[1] += g[1];
        sum[2] += g[2];
    }

    EXPECT_NEAR(sum[0], 0.0, 1e-10);
    EXPECT_NEAR(sum[1], 0.0, 1e-10);
    EXPECT_NEAR(sum[2], 0.0, 1e-10);
}

TEST(BasisGradients, BernsteinTriangleAnalyticalMatchesNumericalAtMultiplePoints) {
    BernsteinBasis basis(ElementType::Triangle3, 3);
    for (const auto& xi : sample_points_for(ElementType::Triangle3)) {
        expect_gradients_match_numerical(basis, xi, Real(1e-5));
    }
}

TEST(BasisGradients, BernsteinTetraAnalyticalMatchesNumericalAtMultiplePoints) {
    BernsteinBasis basis(ElementType::Tetra4, 3);
    for (const auto& xi : sample_points_for(ElementType::Tetra4)) {
        expect_gradients_match_numerical(basis, xi, Real(2e-5));
    }
}

TEST(BasisGradients, BernsteinWedgeAnalyticalMatchesNumericalAtMultiplePoints) {
    BernsteinBasis basis(ElementType::Wedge6, 2);
    for (const auto& xi : sample_points_for(ElementType::Wedge6)) {
        expect_gradients_match_numerical(basis, xi, Real(2e-5));
    }
}

TEST(BasisGradients, BernsteinPyramidAnalyticalMatchesNumericalAtMultiplePoints) {
    BernsteinBasis basis(ElementType::Pyramid5, 2);
    for (const auto& xi : sample_points_for(ElementType::Pyramid5)) {
        expect_gradients_match_numerical(basis, xi, Real(2e-5));
    }
}

TEST(BasisGradients, GradientsSumToZeroForPartitionOfUnity) {
    // Test that analytical gradients sum to zero (partition of unity property)
    // Note: HierarchicalBasis uses orthogonal polynomials (Legendre, Dubiner, Proriol)
    // which do NOT satisfy partition of unity, so we only test BernsteinBasis here.
    const std::vector<std::pair<ElementType, int>> cases = {
        {ElementType::Line2, 3},
        {ElementType::Quad4, 2},
        {ElementType::Hex8, 2},
        {ElementType::Triangle3, 3},
        {ElementType::Tetra4, 3},
        {ElementType::Wedge6, 2}
    };

    math::Vector<Real, 3> xi{Real(0.2), Real(0.15), Real(0.1)};

    for (const auto& [etype, order] : cases) {
        // Test BernsteinBasis (satisfies partition of unity)
        BernsteinBasis basis(etype, order);
        std::vector<Gradient> grads;
        basis.evaluate_gradients(xi, grads);

        Gradient sum{};
        for (const auto& g : grads) {
            sum[0] += g[0];
            sum[1] += g[1];
            sum[2] += g[2];
        }

        for (int d = 0; d < basis.dimension(); ++d) {
            const std::size_t sd = static_cast<std::size_t>(d);
            EXPECT_NEAR(sum[sd], 0.0, 1e-10)
                << "BernsteinBasis element " << static_cast<int>(etype)
                << ", order " << order << ", dim " << d;
        }
    }
}

// ============================================================================
// TensorProductBasis Hessian tests
// ============================================================================

TEST(BasisHessians, TensorProductQuadHessianMatchesNumerical) {
    LagrangeBasis basis_1d(ElementType::Line2, 2);
    TensorProductBasis<LagrangeBasis> basis(basis_1d, 2);
    math::Vector<Real, 3> xi{Real(0.3), Real(-0.2), Real(0)};

    std::vector<Hessian> analytical, numerical;
    basis.evaluate_hessians(xi, analytical);
    numerical_hessian_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(analytical[n](si, sj), numerical[n](si, sj), 1e-6)
                    << "TensorProduct Quad, basis " << n << ", (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, TensorProductHexHessianMatchesNumerical) {
    LagrangeBasis basis_1d(ElementType::Line2, 2);
    TensorProductBasis<LagrangeBasis> basis(basis_1d, 3);
    math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.3)};

    std::vector<Hessian> analytical, numerical;
    basis.evaluate_hessians(xi, analytical);
    numerical_hessian_helper(basis, xi, numerical);

    ASSERT_EQ(analytical.size(), numerical.size());
    for (std::size_t n = 0; n < analytical.size(); ++n) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const std::size_t si = static_cast<std::size_t>(i);
                const std::size_t sj = static_cast<std::size_t>(j);
                EXPECT_NEAR(analytical[n](si, sj), numerical[n](si, sj), 1e-6)
                    << "TensorProduct Hex, basis " << n << ", (" << i << "," << j << ")";
            }
        }
    }
}

TEST(BasisHessians, TensorProductHessiansSumToZero) {
    // TensorProductBasis of Lagrange bases satisfies partition of unity,
    // so sum of Hessians must be zero.
    LagrangeBasis basis_1d(ElementType::Line2, 3);
    TensorProductBasis<LagrangeBasis> basis(basis_1d, 2);
    math::Vector<Real, 3> xi{Real(0.15), Real(-0.25), Real(0)};

    std::vector<Hessian> hess;
    basis.evaluate_hessians(xi, hess);

    Hessian sum{};
    for (const auto& H : hess) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                sum(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) +=
                    H(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(sum(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), 0.0, 1e-10);
        }
    }
}

TEST(BasisHessians, TensorProductHessiansAreExplicitlySymmetric) {
    LagrangeBasis basis_1d(ElementType::Line2, 2);

    {
        TensorProductBasis<LagrangeBasis> basis(basis_1d, 2);
        expect_hessians_symmetric(
            basis, math::Vector<Real, 3>{Real(0.3), Real(-0.2), Real(0)}, Real(1e-12));
    }

    {
        TensorProductBasis<LagrangeBasis> basis(basis_1d, 3);
        expect_hessians_symmetric(
            basis, math::Vector<Real, 3>{Real(0.1), Real(-0.2), Real(0.3)}, Real(1e-12));
    }
}
