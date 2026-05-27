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

namespace {

void expect_hermite_raw_and_strided_match(
    ElementType type,
    const std::vector<math::Vector<Real, 3>>& points) {
    HermiteBasis basis(type, 3);
    const std::size_t n = basis.size();

    for (const auto& xi : points) {
        std::vector<Real> values;
        std::vector<Gradient> gradients;
        std::vector<Hessian> hessians;
        basis.evaluate_all(xi, values, gradients, hessians);
        ASSERT_EQ(values.size(), n);
        ASSERT_EQ(gradients.size(), n);
        ASSERT_EQ(hessians.size(), n);

        std::vector<Real> raw_values(n, Real(0));
        std::vector<Real> raw_gradients(n * 3u, Real(0));
        std::vector<Real> raw_hessians(n * 9u, Real(0));
        basis.evaluate_all_to(
            xi, raw_values.data(), raw_gradients.data(), raw_hessians.data());

        std::vector<Real> separate_values(n, Real(0));
        std::vector<Real> separate_gradients(n * 3u, Real(0));
        std::vector<Real> separate_hessians(n * 9u, Real(0));
        basis.evaluate_values_to(xi, separate_values.data());
        basis.evaluate_gradients_to(xi, separate_gradients.data());
        basis.evaluate_hessians_to(xi, separate_hessians.data());

        for (std::size_t dof = 0; dof < n; ++dof) {
            EXPECT_NEAR(raw_values[dof], values[dof], 1e-14);
            EXPECT_NEAR(separate_values[dof], values[dof], 1e-14);
            for (std::size_t component = 0; component < 3u; ++component) {
                const std::size_t idx = dof * 3u + component;
                EXPECT_NEAR(raw_gradients[idx], gradients[dof][component], 1e-14);
                EXPECT_NEAR(separate_gradients[idx], gradients[dof][component], 1e-14);
            }
            for (std::size_t row = 0; row < 3u; ++row) {
                for (std::size_t col = 0; col < 3u; ++col) {
                    const std::size_t idx = dof * 9u + row * 3u + col;
                    EXPECT_NEAR(raw_hessians[idx], hessians[dof](row, col), 1e-14);
                    EXPECT_NEAR(separate_hessians[idx], hessians[dof](row, col), 1e-14);
                }
            }
        }
    }

    const std::size_t stride = points.size() + 2u;
    const Real sentinel = Real(-9876.5);
    std::vector<Real> strided_values(n * stride, sentinel);
    std::vector<Real> strided_gradients(n * 3u * stride, sentinel);
    std::vector<Real> strided_hessians(n * 9u * stride, sentinel);
    basis.evaluate_at_quadrature_points_strided(points,
                                                stride,
                                                strided_values.data(),
                                                strided_gradients.data(),
                                                strided_hessians.data());

    for (std::size_t q = 0; q < points.size(); ++q) {
        std::vector<Real> values;
        std::vector<Gradient> gradients;
        std::vector<Hessian> hessians;
        basis.evaluate_all(points[q], values, gradients, hessians);

        for (std::size_t dof = 0; dof < n; ++dof) {
            EXPECT_NEAR(strided_values[dof * stride + q], values[dof], 1e-14);
            for (std::size_t component = 0; component < 3u; ++component) {
                const std::size_t row = dof * 3u + component;
                EXPECT_NEAR(strided_gradients[row * stride + q],
                            gradients[dof][component],
                            1e-14);
            }
            for (std::size_t row = 0; row < 3u; ++row) {
                for (std::size_t col = 0; col < 3u; ++col) {
                    const std::size_t hrow = dof * 9u + row * 3u + col;
                    EXPECT_NEAR(strided_hessians[hrow * stride + q],
                                hessians[dof](row, col),
                                1e-14);
                }
            }
        }
    }

    for (std::size_t row = 0; row < n; ++row) {
        EXPECT_EQ(strided_values[row * stride + points.size()], sentinel);
    }
    for (std::size_t row = 0; row < n * 3u; ++row) {
        EXPECT_EQ(strided_gradients[row * stride + points.size()], sentinel);
    }
    for (std::size_t row = 0; row < n * 9u; ++row) {
        EXPECT_EQ(strided_hessians[row * stride + points.size()], sentinel);
    }
}

} // namespace

TEST(C1Basis, LineMetadataAndSize) {
    HermiteBasis basis(ElementType::Line2, 3);

    EXPECT_EQ(basis.element_type(), ElementType::Line2);
    EXPECT_EQ(basis.dimension(), 1);
    EXPECT_EQ(basis.order(), 3);
    EXPECT_EQ(basis.size(), 4u);
    EXPECT_EQ(basis.basis_type(), BasisType::Hermite);
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
        BasisType::Hermite,
        3,
        Continuity::C1,
        FieldType::Scalar
    };

    auto basis_ptr = basis_factory::create(req);
    ASSERT_TRUE(basis_ptr);
    EXPECT_EQ(basis_ptr->element_type(), ElementType::Line2);
    EXPECT_EQ(basis_ptr->dimension(), 1);
    EXPECT_EQ(basis_ptr->order(), 3);
    EXPECT_EQ(basis_ptr->size(), 4u);
    EXPECT_EQ(basis_ptr->basis_type(), BasisType::Hermite);

    // Dynamic type should be HermiteBasis
    auto* hermite = dynamic_cast<HermiteBasis*>(basis_ptr.get());
    EXPECT_NE(hermite, nullptr);
}

TEST(C1Basis, BasisFactoryCreatesHermiteForC1Quad4) {
    BasisRequest req{
        ElementType::Quad4,
        BasisType::Hermite,
        3,
        Continuity::C1,
        FieldType::Scalar
    };

    auto basis_ptr = basis_factory::create(req);
    ASSERT_TRUE(basis_ptr);
    EXPECT_EQ(basis_ptr->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis_ptr->dimension(), 2);
    EXPECT_EQ(basis_ptr->order(), 3);
    EXPECT_EQ(basis_ptr->size(), 16u);
    EXPECT_EQ(basis_ptr->basis_type(), BasisType::Hermite);
}

TEST(C1Basis, QuadMetadataAndSize) {
    HermiteBasis basis(ElementType::Quad4, 3);

    EXPECT_EQ(basis.element_type(), ElementType::Quad4);
    EXPECT_EQ(basis.dimension(), 2);
    EXPECT_EQ(basis.order(), 3);
    EXPECT_EQ(basis.size(), 16u);
    EXPECT_EQ(basis.basis_type(), BasisType::Hermite);
}

TEST(C1Basis, QuadValueModesKroneckerAtCorners) {
    HermiteBasis basis(ElementType::Quad4, 3);

    for (std::size_t c = 0; c < 4; ++c) {
        auto xi = ReferenceNodeLayout::get_node_coords(ElementType::Quad4, c);
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
        auto node = ReferenceNodeLayout::get_node_coords(ElementType::Quad4, c);
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

TEST(C1Basis, BasisFactoryCreatesHermiteForC1Hex8) {
    BasisRequest req{
        ElementType::Hex8,
        BasisType::Hermite,
        3,
        Continuity::C1,
        FieldType::Scalar
    };

    auto basis_ptr = basis_factory::create(req);
    ASSERT_TRUE(basis_ptr);
    EXPECT_EQ(basis_ptr->element_type(), ElementType::Hex8);
    EXPECT_EQ(basis_ptr->dimension(), 3);
    EXPECT_EQ(basis_ptr->order(), 3);
    EXPECT_EQ(basis_ptr->size(), 64u);
    EXPECT_EQ(basis_ptr->basis_type(), BasisType::Hermite);
}

// =============================================================================
// Hex8 (3D tricubic Hermite) tests
// =============================================================================

TEST(C1Basis, HexMetadataAndSize) {
    HermiteBasis basis(ElementType::Hex8, 3);
    EXPECT_EQ(basis.element_type(), ElementType::Hex8);
    EXPECT_EQ(basis.dimension(), 3);
    EXPECT_EQ(basis.order(), 3);
    EXPECT_EQ(basis.size(), 64u);
    EXPECT_EQ(basis.basis_type(), BasisType::Hermite);
}

TEST(C1Basis, HexValueModesKroneckerAtCorners) {
    HermiteBasis basis(ElementType::Hex8, 3);

    for (std::size_t c = 0; c < 8; ++c) {
        auto xi = ReferenceNodeLayout::get_node_coords(ElementType::Hex8, c);
        std::vector<Real> vals;
        basis.evaluate_values(xi, vals);
        ASSERT_EQ(vals.size(), 64u);

        // Value DOF at this corner (index 8*c) should be 1
        for (std::size_t corner = 0; corner < 8; ++corner) {
            const std::size_t val_index = 8 * corner;
            const Real expected = (corner == c) ? Real(1) : Real(0);
            EXPECT_NEAR(vals[val_index], expected, 1e-14)
                << "Corner " << c << ", checking value DOF at corner " << corner;
        }

        // All derivative DOFs at this corner should be 0
        const std::size_t base = 8 * c;
        for (std::size_t i = 0; i < 64; ++i) {
            if (i == base) continue; // skip the value DOF
            EXPECT_NEAR(vals[i], 0.0, 1e-14)
                << "Corner " << c << ", DOF " << i;
        }
    }
}

TEST(C1Basis, HexValueModesPartitionOfUnity) {
    HermiteBasis basis(ElementType::Hex8, 3);

    const math::Vector<Real, 3> test_pts[] = {
        {Real(0), Real(0), Real(0)},
        {Real(0.3), Real(-0.2), Real(0.5)},
        {Real(-0.7), Real(0.4), Real(-0.3)},
    };

    for (const auto& xi : test_pts) {
        std::vector<Real> vals;
        basis.evaluate_values(xi, vals);
        ASSERT_EQ(vals.size(), 64u);

        // Sum of value modes at all 8 corners should be 1
        Real sum = Real(0);
        for (std::size_t c = 0; c < 8; ++c) {
            sum += vals[8 * c];
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST(C1Basis, HexGradientMatchesFiniteDifference) {
    HermiteBasis basis(ElementType::Hex8, 3);
    math::Vector<Real, 3> xi{Real(0.2), Real(-0.3), Real(0.15)};
    const Real eps = Real(1e-6);

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);
    ASSERT_EQ(grads.size(), 64u);

    for (int d = 0; d < 3; ++d) {
        auto xi_p = xi, xi_m = xi;
        xi_p[static_cast<std::size_t>(d)] += eps;
        xi_m[static_cast<std::size_t>(d)] -= eps;

        std::vector<Real> vp, vm;
        basis.evaluate_values(xi_p, vp);
        basis.evaluate_values(xi_m, vm);

        for (std::size_t i = 0; i < 64; ++i) {
            const Real fd = (vp[i] - vm[i]) / (Real(2) * eps);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], fd, 1e-5)
                << "DOF " << i << ", dim " << d;
        }
    }
}

TEST(C1Basis, HermiteRawAndStridedOutputsMatchVectorEvaluation) {
    expect_hermite_raw_and_strided_match(
        ElementType::Line2,
        {{Real(-0.4), Real(0), Real(0)}, {Real(0.35), Real(0), Real(0)}});
    expect_hermite_raw_and_strided_match(
        ElementType::Quad4,
        {{Real(-0.2), Real(0.3), Real(0)}, {Real(0.45), Real(-0.55), Real(0)}});
    expect_hermite_raw_and_strided_match(
        ElementType::Hex8,
        {{Real(-0.2), Real(0.3), Real(-0.4)},
         {Real(0.45), Real(-0.55), Real(0.25)}});
}
