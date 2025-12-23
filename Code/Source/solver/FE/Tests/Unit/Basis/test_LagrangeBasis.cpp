/**
 * @file test_LagrangeBasis.cpp
 * @brief Unit tests for Lagrange basis functions
 */

#include <gtest/gtest.h>
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include <numeric>

using svmp::FE::basis::LagrangeBasis;
using svmp::FE::ElementType;
using svmp::FE::Real;
using svmp::FE::basis::Gradient;
using svmp::FE::basis::NodeOrdering;

namespace {

void expect_nodes_match_node_ordering(ElementType canonical_type,
                                      int order,
                                      ElementType node_ordering_type) {
    LagrangeBasis basis(canonical_type, order);
    const auto& nodes = basis.nodes();

    ASSERT_EQ(nodes.size(), NodeOrdering::num_nodes(node_ordering_type));
    ASSERT_EQ(nodes.size(), basis.size());

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto expected = NodeOrdering::get_node_coords(node_ordering_type, i);
        EXPECT_NEAR(nodes[i][0], expected[0], 1e-14);
        EXPECT_NEAR(nodes[i][1], expected[1], 1e-14);
        EXPECT_NEAR(nodes[i][2], expected[2], 1e-14);

        std::vector<Real> vals;
        basis.evaluate_values(expected, vals);
        ASSERT_EQ(vals.size(), nodes.size());
        for (std::size_t j = 0; j < vals.size(); ++j) {
            const double expected_delta = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(vals[j], expected_delta, 1e-12);
        }
    }
}

} // namespace

TEST(LagrangeBasis, QuadPartitionOfUnity) {
    LagrangeBasis basis(ElementType::Quad4, 1);
    svmp::FE::math::Vector<Real, 3> xi{0.2, -0.3, 0.0};

    std::vector<Real> values;
    basis.evaluate_values(xi, values);

    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(LagrangeBasis, LineGradientLinear) {
    LagrangeBasis basis(ElementType::Line2, 1);
    svmp::FE::math::Vector<Real, 3> xi{0.0, 0.0, 0.0};
    std::vector<Gradient> grad;
    basis.evaluate_gradients(xi, grad);

    ASSERT_EQ(grad.size(), 2u);
    EXPECT_NEAR(grad[0][0], -0.5, 1e-12);
    EXPECT_NEAR(grad[1][0], 0.5, 1e-12);
}

TEST(LagrangeBasis, TrianglePartitionOfUnity) {
    LagrangeBasis basis(ElementType::Triangle3, 1);
    svmp::FE::math::Vector<Real, 3> xi{0.2, 0.3, 0.0};
    std::vector<Real> values;
    basis.evaluate_values(xi, values);

    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(LagrangeBasis, SizeFormulasPerElement) {
    for (int order = 0; order <= 3; ++order) {
        {
            LagrangeBasis line(ElementType::Line2, order);
            EXPECT_EQ(line.size(), static_cast<std::size_t>(order + 1));
        }
        {
            LagrangeBasis quad(ElementType::Quad4, order);
            const std::size_t n1d = static_cast<std::size_t>(order + 1);
            EXPECT_EQ(quad.size(), n1d * n1d);
        }
        {
            LagrangeBasis hex(ElementType::Hex8, order);
            const std::size_t n1d = static_cast<std::size_t>(order + 1);
            EXPECT_EQ(hex.size(), n1d * n1d * n1d);
        }
        {
            LagrangeBasis tri(ElementType::Triangle3, order);
            const std::size_t expected =
                static_cast<std::size_t>(order + 1) *
                static_cast<std::size_t>(order + 2) / 2;
            EXPECT_EQ(tri.size(), expected);
        }
        {
            LagrangeBasis tet(ElementType::Tetra4, order);
            const std::size_t expected =
                static_cast<std::size_t>(order + 1) *
                static_cast<std::size_t>(order + 2) *
                static_cast<std::size_t>(order + 3) / 6;
            EXPECT_EQ(tet.size(), expected);
        }
    }
}

TEST(LagrangeBasis, KroneckerDeltaAtNodes) {
    const std::vector<std::pair<ElementType, int>> cases = {
        {ElementType::Line2, 1},
        {ElementType::Quad4, 1},
        {ElementType::Triangle3, 1},
        {ElementType::Tetra4, 1},
        {ElementType::Hex8, 1},
        {ElementType::Triangle3, 2},
        {ElementType::Tetra4, 2},
        {ElementType::Quad4, 2},
        {ElementType::Hex8, 2},
        {ElementType::Wedge6, 2}
    };

    for (const auto& c : cases) {
        LagrangeBasis basis(c.first, c.second);
        const auto& nodes = basis.nodes();
        ASSERT_EQ(nodes.size(), basis.size());

        for (std::size_t i = 0; i < nodes.size(); ++i) {
            std::vector<Real> vals;
            basis.evaluate_values(nodes[i], vals);
            ASSERT_EQ(vals.size(), nodes.size());
            for (std::size_t j = 0; j < nodes.size(); ++j) {
                if (i == j) {
                    EXPECT_NEAR(vals[j], 1.0, 1e-12);
                } else {
                    EXPECT_NEAR(vals[j], 0.0, 1e-12);
                }
            }
        }
    }
}

TEST(LagrangeBasis, MatchesNodeOrderingConventionsForLinearAndQuadratic) {
    // Tensor-product elements
    expect_nodes_match_node_ordering(ElementType::Line2, 1, ElementType::Line2);
    expect_nodes_match_node_ordering(ElementType::Line2, 2, ElementType::Line3);
    expect_nodes_match_node_ordering(ElementType::Quad4, 1, ElementType::Quad4);
    expect_nodes_match_node_ordering(ElementType::Quad4, 2, ElementType::Quad9);
    expect_nodes_match_node_ordering(ElementType::Hex8, 1, ElementType::Hex8);
    expect_nodes_match_node_ordering(ElementType::Hex8, 2, ElementType::Hex27);

    // Simplex elements
    expect_nodes_match_node_ordering(ElementType::Triangle3, 1, ElementType::Triangle3);
    expect_nodes_match_node_ordering(ElementType::Triangle3, 2, ElementType::Triangle6);
    expect_nodes_match_node_ordering(ElementType::Tetra4, 1, ElementType::Tetra4);
    expect_nodes_match_node_ordering(ElementType::Tetra4, 2, ElementType::Tetra10);

    // Mixed topology
    expect_nodes_match_node_ordering(ElementType::Wedge6, 1, ElementType::Wedge6);
    expect_nodes_match_node_ordering(ElementType::Wedge6, 2, ElementType::Wedge18);

    // Pyramid
    expect_nodes_match_node_ordering(ElementType::Pyramid5, 1, ElementType::Pyramid5);
    expect_nodes_match_node_ordering(ElementType::Pyramid14, 2, ElementType::Pyramid14);
}

TEST(LagrangeBasis, WedgeAndPyramidPartitionOfUnity) {
    {
        LagrangeBasis wedge(ElementType::Wedge6, 1);
        svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.1), Real(0.3)};
        std::vector<Real> vals;
        wedge.evaluate_values(xi, vals);
        const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }

    {
        LagrangeBasis wedge_q(ElementType::Wedge18, 2);
        svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.1), Real(-0.25)};
        std::vector<Real> vals;
        wedge_q.evaluate_values(xi, vals);
        const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-12);

        // Wedge18 should report 18 nodes in NodeOrdering
        EXPECT_EQ(NodeOrdering::num_nodes(ElementType::Wedge18), 18u);
        // Corner nodes should match Wedge6 vertices
        auto v0 = NodeOrdering::get_node_coords(ElementType::Wedge18, 0);
        auto v1 = NodeOrdering::get_node_coords(ElementType::Wedge18, 1);
        auto v2 = NodeOrdering::get_node_coords(ElementType::Wedge18, 2);
        EXPECT_NEAR(v0[0], Real(0), 1e-14);
        EXPECT_NEAR(v0[1], Real(0), 1e-14);
        EXPECT_NEAR(v0[2], Real(-1), 1e-14);
        EXPECT_NEAR(v1[0], Real(1), 1e-14);
        EXPECT_NEAR(v1[1], Real(0), 1e-14);
        EXPECT_NEAR(v1[2], Real(-1), 1e-14);
        EXPECT_NEAR(v2[0], Real(0), 1e-14);
        EXPECT_NEAR(v2[1], Real(1), 1e-14);
        EXPECT_NEAR(v2[2], Real(-1), 1e-14);
    }

    {
        LagrangeBasis pyr(ElementType::Pyramid5, 1);
        svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.4)};
        std::vector<Real> vals;
        pyr.evaluate_values(xi, vals);
        const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST(LagrangeBasis, Pyramid14RationalNodalAndPartition) {
    using svmp::FE::basis::NodeOrdering;

    LagrangeBasis basis(ElementType::Pyramid14, 2);
    EXPECT_EQ(basis.dimension(), 3);
    EXPECT_EQ(basis.size(), 14u);

    // Kronecker nodal property at all Pyramid14 nodes
    for (std::size_t i = 0; i < basis.size(); ++i) {
        auto xi = NodeOrdering::get_node_coords(ElementType::Pyramid14, i);
        std::vector<Real> vals;
        basis.evaluate_values(xi, vals);
        ASSERT_EQ(vals.size(), basis.size());
        for (std::size_t j = 0; j < basis.size(); ++j) {
            const double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(vals[j], expected, 1e-12);
        }
    }

    // Partition of unity at an interior point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.3)};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(LagrangeBasis, Pyramid14GradientSumZero) {
    LagrangeBasis basis(ElementType::Pyramid14, 2);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.15), Real(-0.1), Real(0.3)};

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), basis.size());

    Gradient sum{};
    for (const auto& g : grads) {
        sum[0] += g[0];
        sum[1] += g[1];
        sum[2] += g[2];
    }
    EXPECT_NEAR(sum[0], 0.0, 1e-8);
    EXPECT_NEAR(sum[1], 0.0, 1e-8);
    EXPECT_NEAR(sum[2], 0.0, 1e-8);
}

TEST(LagrangeBasis, HigherOrderP4KroneckerAndPartition) {
    struct Case {
        ElementType type;
        int order;
        svmp::FE::math::Vector<Real, 3> xi;
    };

    const std::vector<Case> cases = {
        {ElementType::Line2, 4, {Real(0.11), Real(0), Real(0)}},
        {ElementType::Quad4, 4, {Real(0.2), Real(-0.3), Real(0)}},
        {ElementType::Triangle3, 4, {Real(0.2), Real(0.3), Real(0)}},
        {ElementType::Hex8, 4, {Real(0.2), Real(-0.3), Real(0.4)}},
    };

    for (const auto& c : cases) {
        LagrangeBasis basis(c.type, c.order);

        // Partition of unity at an interior point
        std::vector<Real> values;
        basis.evaluate_values(c.xi, values);
        const double sum = std::accumulate(values.begin(), values.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-12);

        // Kronecker delta property at all nodes
        const auto& nodes = basis.nodes();
        ASSERT_EQ(nodes.size(), basis.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            basis.evaluate_values(nodes[i], values);
            ASSERT_EQ(values.size(), nodes.size());
            for (std::size_t j = 0; j < nodes.size(); ++j) {
                const double expected = (i == j) ? 1.0 : 0.0;
                EXPECT_NEAR(values[j], expected, 1e-12);
            }
        }
    }
}

TEST(LagrangeBasis, Pyramid14InterpolatesQuadraticPolynomials) {
    using svmp::FE::basis::NodeOrdering;

    LagrangeBasis basis(ElementType::Pyramid14, 2);
    const std::size_t n = basis.size();

    // Precompute nodal coordinates
    std::vector<svmp::FE::math::Vector<Real,3>> nodes;
    nodes.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        nodes.push_back(NodeOrdering::get_node_coords(ElementType::Pyramid14, i));
    }

    auto interpolate_and_check = [&](auto f, Real tol) {
        // Nodal coefficients
        std::vector<Real> coeffs(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& x = nodes[i];
            coeffs[i] = f(x[0], x[1], x[2]);
        }

        // Test at a few interior points
        const svmp::FE::math::Vector<Real,3> test_pts[] = {
            {Real(0.1), Real(-0.2), Real(0.2)},
            {Real(-0.2), Real(0.15), Real(0.4)},
            {Real(0.05), Real(0.05), Real(0.3)}
        };

        for (const auto& xi : test_pts) {
            std::vector<Real> vals;
            basis.evaluate_values(xi, vals);
            ASSERT_EQ(vals.size(), n);

            Real u_interp = Real(0);
            for (std::size_t i = 0; i < n; ++i) {
                u_interp += coeffs[i] * vals[i];
            }

            const Real u_exact = f(xi[0], xi[1], xi[2]);
            EXPECT_NEAR(u_interp, u_exact, tol);
        }
    };

    // Constant, linear and quadratic monomials
    interpolate_and_check([](Real, Real, Real) { return Real(1); }, Real(1e-12));
    interpolate_and_check([](Real x, Real, Real) { return x; }, Real(1e-11));
    interpolate_and_check([](Real, Real y, Real) { return y; }, Real(1e-11));
    interpolate_and_check([](Real, Real, Real z) { return z; }, Real(1e-11));
    interpolate_and_check([](Real x, Real y, Real) { return x * y; }, Real(1e-10));
    interpolate_and_check([](Real x, Real, Real z) { return x * z; }, Real(1e-10));
    interpolate_and_check([](Real, Real y, Real z) { return y * z; }, Real(1e-10));
    interpolate_and_check([](Real x, Real, Real) { return x * x; }, Real(1e-10));
    interpolate_and_check([](Real, Real y, Real) { return y * y; }, Real(1e-10));
    interpolate_and_check([](Real, Real, Real z) { return z * z; }, Real(1e-10));
}

TEST(LagrangeBasis, Pyramid14GradientMatchesLinearFunctionGradient) {
    using svmp::FE::basis::NodeOrdering;

    LagrangeBasis basis(ElementType::Pyramid14, 2);
    const std::size_t n = basis.size();

    // Nodal coordinates and coefficients for f(x,y,z) = ax + by + cz
    const Real a = Real(1.2);
    const Real b = Real(-0.7);
    const Real c = Real(0.5);

    std::vector<Real> coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto x = NodeOrdering::get_node_coords(ElementType::Pyramid14, i);
        coeffs[i] = a * x[0] + b * x[1] + c * x[2];
    }

    const svmp::FE::math::Vector<Real,3> xi{Real(0.1), Real(-0.15), Real(0.35)};

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), n);

    Gradient g_interp{};
    for (std::size_t i = 0; i < n; ++i) {
        g_interp[0] += coeffs[i] * grads[i][0];
        g_interp[1] += coeffs[i] * grads[i][1];
        g_interp[2] += coeffs[i] * grads[i][2];
    }

    EXPECT_NEAR(g_interp[0], a, 1e-6);
    EXPECT_NEAR(g_interp[1], b, 1e-6);
    EXPECT_NEAR(g_interp[2], c, 1e-6);
}

TEST(LagrangeBasis, GradientSumZeroQuadAndTet) {
    const std::vector<std::pair<ElementType, svmp::FE::math::Vector<Real, 3>>> cases = {
        {ElementType::Quad4, svmp::FE::math::Vector<Real, 3>{Real(0.2), Real(-0.1), Real(0)}},
        {ElementType::Tetra4, svmp::FE::math::Vector<Real, 3>{Real(0.1), Real(0.2), Real(0.1)}}
    };

    for (const auto& c : cases) {
        LagrangeBasis basis(c.first, 1);
        std::vector<Gradient> grads;
        basis.evaluate_gradients(c.second, grads);

        ASSERT_EQ(grads.size(), basis.size());
        Gradient sum{};
        for (const auto& g : grads) {
            sum[0] += g[0];
            sum[1] += g[1];
            sum[2] += g[2];
        }
        EXPECT_NEAR(sum[0], 0.0, 1e-12);
        EXPECT_NEAR(sum[1], 0.0, 1e-12);
        EXPECT_NEAR(sum[2], 0.0, 1e-12);
    }
}

TEST(LagrangeBasis, HexPartitionAndGradientSumZeroOrderThree) {
    LagrangeBasis basis(ElementType::Hex8, 3);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.25)};

    std::vector<Real> values;
    basis.evaluate_values(xi, values);
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-12);

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    Gradient gsum{};
    for (const auto& g : grads) {
        gsum[0] += g[0];
        gsum[1] += g[1];
        gsum[2] += g[2];
    }
    EXPECT_NEAR(gsum[0], 0.0, 1e-10);
    EXPECT_NEAR(gsum[1], 0.0, 1e-10);
    EXPECT_NEAR(gsum[2], 0.0, 1e-10);
}
