/**
 * @file test_LinearMapping.cpp
 * @brief Tests for affine linear geometry mappings
 */

#include <gtest/gtest.h>
#include "FE/Geometry/LinearMapping.h"
#include "FE/Quadrature/GaussQuadrature.h"

#include <thread>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

TEST(LinearMapping, LineMapsEndpointsAndMidpoint) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);

    math::Vector<Real, 3> x0 = map.map_to_physical({Real(-1), Real(0), Real(0)});
    math::Vector<Real, 3> x1 = map.map_to_physical({Real(1), Real(0), Real(0)});
    math::Vector<Real, 3> xm = map.map_to_physical({Real(0), Real(0), Real(0)});

    EXPECT_NEAR(x0[0], 0.0, 1e-14);
    EXPECT_NEAR(x1[0], 1.0, 1e-14);
    EXPECT_NEAR(xm[0], 0.5, 1e-14);

    auto J = map.jacobian({});
    EXPECT_NEAR(J(0,0), 0.5, 1e-14);
    EXPECT_NEAR(map.jacobian_determinant({}), 0.5, 1e-14);
}

TEST(LinearMapping, TriangleMapsBarycentricCenter) {
    // Node ordering must match LagrangeBasis/NodeOrderingConventions for Triangle3:
    // (0,0), (1,0), (0,1) on the reference unit right triangle.
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)}, // ref node (0,0)
        {Real(1), Real(0), Real(0)}, // ref node (1,0)
        {Real(0), Real(1), Real(0)}  // ref node (0,1)
    };
    LinearMapping map(ElementType::Triangle3, nodes);
    math::Vector<Real, 3> xi{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], 1.0/3.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0/3.0, 1e-12);

    auto det = map.jacobian_determinant({});
    EXPECT_NEAR(det, 1.0, 1e-12); // matches reference triangle measure
}

TEST(LinearMapping, TetraMapsBarycenterAndDeterminant) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)},
        {Real(0), Real(1), Real(0)},
        {Real(0), Real(0), Real(1)}
    };
    LinearMapping map(ElementType::Tetra4, nodes);
    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.25)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], 0.25, 1e-12);
    EXPECT_NEAR(x[1], 0.25, 1e-12);
    EXPECT_NEAR(x[2], 0.25, 1e-12);

    auto det = map.jacobian_determinant(xi);
    EXPECT_NEAR(det, 1.0, 1e-12);

    // Roundtrip reference mapping
    auto xi_back = map.map_to_reference(x);
    EXPECT_NEAR(xi_back[0], xi[0], 1e-8);
    EXPECT_NEAR(xi_back[1], xi[1], 1e-8);
    EXPECT_NEAR(xi_back[2], xi[2], 1e-8);
}

TEST(LinearMapping, LineJacobianInverseAndTransformGradient) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);
    auto J = map.jacobian({});
    EXPECT_NEAR(J(0,0), 1.0, 1e-12);
    auto Jinv = map.jacobian_inverse({});
    EXPECT_NEAR(Jinv(0,0), 1.0, 1e-12);

    math::Vector<Real,3> grad_ref{Real(2), Real(0), Real(0)};
    auto grad_phys = map.transform_gradient(grad_ref, math::Vector<Real,3>{});
    EXPECT_NEAR(grad_phys[0], 2.0, 1e-12);
}

TEST(LinearMapping, LineAlongYAxisHasValidJacobianFrame) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(0), Real(2), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);

    auto J = map.jacobian({});
    EXPECT_NEAR(J(0,0), 0.0, 1e-12);
    EXPECT_NEAR(J(1,0), 1.0, 1e-12); // dy/dxi = length/2
    EXPECT_NEAR(map.jacobian_determinant({}), 1.0, 1e-12);

    // Roundtrip reference mapping at the midpoint.
    const auto x_mid = map.map_to_physical(math::Vector<Real,3>{Real(0), Real(0), Real(0)});
    const auto xi_back = map.map_to_reference(x_mid);
    EXPECT_NEAR(xi_back[0], 0.0, 1e-8);
}

// Every LinearMapping of one simplex type shares one immutable linear basis.
// Mappings must evaluate exactly as with a basis of their own, also when used
// from several threads at once.
TEST(LinearMapping, SharedBasisEvaluatesAsAnOwnBasis) {
    struct Case {
        ElementType type;
        std::vector<math::Vector<Real, 3>> nodes;
        math::Vector<Real, 3> xi;
    };
    const std::vector<Case> cases = {
        {ElementType::Line2,
         {{Real(0.3), Real(-0.2), Real(0.1)}, {Real(1.7), Real(0.4), Real(-0.6)}},
         {Real(0.37), Real(0), Real(0)}},
        {ElementType::Triangle3,
         {{Real(0.1), Real(0.2), Real(0)}, {Real(1.3), Real(0.1), Real(0)}, {Real(0.4), Real(0.9), Real(0)}},
         {Real(0.21), Real(0.33), Real(0)}},
        {ElementType::Tetra4,
         {{Real(0.1), Real(0.2), Real(0.05)}, {Real(0.9), Real(0.15), Real(0.1)},
          {Real(0.3), Real(0.8), Real(0.2)}, {Real(0.25), Real(0.3), Real(0.95)}},
         {Real(0.2), Real(0.3), Real(0.1)}},
    };
    for (const auto& c : cases) {
        const LinearMapping a(c.type, c.nodes);
        const LinearMapping b(c.type, c.nodes);
        EXPECT_EQ(&a.geometryBasis(), &b.geometryBasis());

        // The same evaluation with a basis of its own.
        const basis::LagrangeBasis own(c.type, 1);
        std::vector<Real> values;
        own.evaluate_values(c.xi, values);
        math::Vector<Real, 3> expected{};
        for (std::size_t n = 0; n < values.size(); ++n) {
            expected += c.nodes[n] * values[n];
        }
        const auto mapped = a.map_to_physical(c.xi);
        for (std::size_t k = 0; k < 3; ++k) {
            EXPECT_EQ(mapped[k], expected[k]);
        }

        const auto reference_jacobian = a.jacobian(c.xi);
        std::vector<math::Matrix<Real, 3, 3>> jacobians(8);
        std::vector<std::thread> threads;
        for (std::size_t t = 0; t < jacobians.size(); ++t) {
            threads.emplace_back([&, t] {
                const LinearMapping local(c.type, c.nodes);
                jacobians[t] = local.jacobian(c.xi);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (const auto& jacobian : jacobians) {
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    EXPECT_EQ(jacobian(i, j), reference_jacobian(i, j));
                }
            }
        }
    }
}
