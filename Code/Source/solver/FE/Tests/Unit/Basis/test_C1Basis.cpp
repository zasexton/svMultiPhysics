/**
 * @file test_C1Basis.cpp
 * @brief Tests for C¹ Hermite basis family (1D and 2D) and factory integration
 */

#include <gtest/gtest.h>

#include "FE/Basis/HermiteBasis.h"
#include "FE/Basis/BasisFactory.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Basis/NodeOrderingConventions.h"

using namespace svmp::FE;
using namespace svmp::FE::basis;
using namespace svmp::FE::quadrature;

TEST(C1Basis, LineMetadataAndSize) {
    HermiteBasis basis(ElementType::Line2, 3);

    EXPECT_EQ(basis.element_type(), ElementType::Line2);
    EXPECT_EQ(basis.dimension(), 1);
    EXPECT_EQ(basis.order(), 3);
    EXPECT_EQ(basis.size(), 4u);
    EXPECT_EQ(basis.basis_type(), BasisType::Custom);
}

TEST(C1Basis, LineNodalAndDerivativeConditionsAtEndpoints) {
    HermiteBasis basis(ElementType::Line2, 3);

    math::Vector<Real, 3> xi_left{Real(-1), Real(0), Real(0)};
    math::Vector<Real, 3> xi_right{Real(1), Real(0), Real(0)};

    std::vector<Real> vals;

    basis.evaluate_values(xi_left, vals);
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_NEAR(vals[0], 1.0, 1e-14); // value at left node
    EXPECT_NEAR(vals[1], 0.0, 1e-14); // value at right node
    EXPECT_NEAR(vals[2], 0.0, 1e-14); // slope mode at left
    EXPECT_NEAR(vals[3], 0.0, 1e-14); // slope mode at right

    basis.evaluate_values(xi_right, vals);
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_NEAR(vals[0], 0.0, 1e-14); // value at left node
    EXPECT_NEAR(vals[1], 1.0, 1e-14); // value at right node
    EXPECT_NEAR(vals[2], 0.0, 1e-14); // slope mode at left
    EXPECT_NEAR(vals[3], 0.0, 1e-14); // slope mode at right
}

TEST(C1Basis, LinePartitionOfUnityForValueModes) {
    HermiteBasis basis(ElementType::Line2, 3);

    GaussQuadrature1D quad(5);
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        auto xi = quad.point(q);
        std::vector<Real> vals;
        basis.evaluate_values(xi, vals);
        ASSERT_EQ(vals.size(), 4u);

        const Real sum_values = vals[0] + vals[1];
        EXPECT_NEAR(sum_values, 1.0, 1e-12);
    }
}

TEST(C1Basis, LineGradientsMatchFiniteDifference) {
    HermiteBasis basis(ElementType::Line2, 3);

    math::Vector<Real, 3> xi{Real(0.2), Real(0), Real(0)};

    std::vector<Real> vals_plus, vals_minus;
    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), 4u);

    const Real eps = Real(1e-6);
    math::Vector<Real, 3> xi_plus = xi;
    math::Vector<Real, 3> xi_minus = xi;
    xi_plus[0] += eps;
    xi_minus[0] -= eps;

    basis.evaluate_values(xi_plus, vals_plus);
    basis.evaluate_values(xi_minus, vals_minus);
    ASSERT_EQ(vals_plus.size(), 4u);
    ASSERT_EQ(vals_minus.size(), 4u);

    for (std::size_t i = 0; i < 4; ++i) {
        const Real fd = (vals_plus[i] - vals_minus[i]) / (Real(2) * eps);
        EXPECT_NEAR(grads[i][0], fd, 1e-5);
    }
}

TEST(C1Basis, BasisFactoryCreatesHermiteForC1Line2) {
    BasisRequest req{
        ElementType::Line2,
        BasisType::Custom,
        3,
        Continuity::C1,
        FieldType::Scalar
    };

    auto basis_ptr = BasisFactory::create(req);
    ASSERT_TRUE(basis_ptr);
    EXPECT_EQ(basis_ptr->element_type(), ElementType::Line2);
    EXPECT_EQ(basis_ptr->dimension(), 1);
    EXPECT_EQ(basis_ptr->order(), 3);
    EXPECT_EQ(basis_ptr->size(), 4u);
    EXPECT_EQ(basis_ptr->basis_type(), BasisType::Custom);

    // Dynamic type should be HermiteBasis
    auto* hermite = dynamic_cast<HermiteBasis*>(basis_ptr.get());
    EXPECT_NE(hermite, nullptr);
}

TEST(C1Basis, QuadMetadataAndSize) {
    HermiteBasis basis(ElementType::Quad4, 3);

    EXPECT_EQ(basis.element_type(), ElementType::Quad4);
    EXPECT_EQ(basis.dimension(), 2);
    EXPECT_EQ(basis.order(), 3);
    EXPECT_EQ(basis.size(), 16u);
    EXPECT_EQ(basis.basis_type(), BasisType::Custom);
}

TEST(C1Basis, QuadValueModesKroneckerAtCorners) {
    HermiteBasis basis(ElementType::Quad4, 3);

    for (std::size_t c = 0; c < 4; ++c) {
        auto xi = NodeOrdering::get_node_coords(ElementType::Quad4, c);
        math::Vector<Real, 3> pt;
        pt[0] = xi[0];
        pt[1] = xi[1];
        pt[2] = Real(0);
        std::vector<Real> vals;
        basis.evaluate_values(pt, vals);
        ASSERT_EQ(vals.size(), 16u);

        const std::size_t base = 4 * c;

        // Value DOF at this corner is 1, all others 0
        for (std::size_t corner = 0; corner < 4; ++corner) {
            const std::size_t val_index = 4 * corner;
            const Real expected = (corner == c) ? Real(1) : Real(0);
            EXPECT_NEAR(vals[val_index], expected, 1e-14);
        }

        // All derivative DOFs vanish at corners
        for (std::size_t i = 0; i < 16; ++i) {
            if (i == base) {
                continue; // already checked value DOF
            }
            EXPECT_NEAR(vals[i], 0.0, 1e-14);
        }
    }
}

TEST(C1Basis, QuadValueModesPartitionOfUnity) {
    HermiteBasis basis(ElementType::Quad4, 3);

    // Use a tensor-product Gauss rule to probe interior points
    GaussQuadrature1D quad_1d(3);
    for (std::size_t qx = 0; qx < quad_1d.num_points(); ++qx) {
        for (std::size_t qy = 0; qy < quad_1d.num_points(); ++qy) {
            math::Vector<Real, 3> xi{
                quad_1d.point(qx)[0],
                quad_1d.point(qy)[0],
                Real(0)
            };
            std::vector<Real> vals;
            basis.evaluate_values(xi, vals);
            ASSERT_EQ(vals.size(), 16u);

            // Constant field is represented by setting all corner values to 1
            // and all derivative DOFs to 0, so the sum of value modes must be 1.
            const Real sum_values =
                vals[0] + vals[4] + vals[8] + vals[12];
            EXPECT_NEAR(sum_values, 1.0, 1e-12);
        }
    }
}

TEST(C1Basis, QuadInterpolatesBicubicPolynomial) {
    HermiteBasis basis(ElementType::Quad4, 3);

    // Bicubic-ish polynomial (not fully general but nontrivial)
    auto poly = [](Real x, Real y) {
        return Real(1) + Real(0.5) * x + Real(0.3) * y
             + Real(0.2) * x * x + Real(0.1) * y * y
             + Real(0.4) * x * y
             + Real(0.05) * x * x * x
             + Real(0.07) * y * y * y;
    };

    auto dpx = [](Real x, Real y) {
        return Real(0.5)
             + Real(0.4) * x
             + Real(0.4) * y
             + Real(0.15) * x * x;
    };

    auto dpy = [](Real x, Real y) {
        return Real(0.3)
             + Real(0.2) * y
             + Real(0.4) * x
             + Real(0.21) * y * y;
    };

    auto dpxy = [](Real, Real) {
        return Real(0.4);
    };

    // Build DOFs from corner values and derivatives
    std::vector<Real> dofs(16, Real(0));
    for (std::size_t c = 0; c < 4; ++c) {
        auto node = NodeOrdering::get_node_coords(ElementType::Quad4, c);
        const Real x = node[0];
        const Real y = node[1];

        const std::size_t base = 4 * c;
        dofs[base + 0] = poly(x, y);   // value
        dofs[base + 1] = dpx(x, y);    // d/dx
        dofs[base + 2] = dpy(x, y);    // d/dy
        dofs[base + 3] = dpxy(x, y);   // d2/(dx dy)
    }

    // Check interpolation at a few interior points
    const math::Vector<Real, 3> test_pts[] = {
        {Real(0.0), Real(0.0), Real(0)},
        {Real(0.3), Real(-0.2), Real(0)},
        {Real(-0.4), Real(0.5), Real(0)}
    };

    for (const auto& xi : test_pts) {
        std::vector<Real> vals;
        basis.evaluate_values(xi, vals);
        ASSERT_EQ(vals.size(), 16u);

        Real u_interp = Real(0);
        for (std::size_t i = 0; i < 16; ++i) {
            u_interp += dofs[i] * vals[i];
        }

        const Real u_exact = poly(xi[0], xi[1]);
        EXPECT_NEAR(u_interp, u_exact, 1e-10);
    }
}
