/**
 * @file test_FunctionSpaceGradients.cpp
 * @brief Unit tests for FunctionSpace::evaluate_gradient
 */

#include <gtest/gtest.h>

#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/HCurlSpace.h"
#include "FE/Spaces/HDivSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

namespace {

FunctionSpace::Value xi1(Real x) {
    FunctionSpace::Value xi{};
    xi[0] = x;
    xi[1] = Real(0);
    xi[2] = Real(0);
    return xi;
}

} // namespace

TEST(FunctionSpaceGradients, H1LineP1LinearGradientIsConstant) {
    H1Space space(ElementType::Line2, 1);

    // u(x) = 1 + x on [-1, 1] -> nodal values are u(-1)=0, u(1)=2
    std::vector<Real> coeffs = {Real(0), Real(2)};
    ASSERT_EQ(coeffs.size(), space.dofs_per_element());

    const std::vector<Real> eval_pts = {Real(-1), Real(-0.25), Real(0), Real(0.75), Real(1)};
    for (Real x : eval_pts) {
        const auto g = space.evaluate_gradient(xi1(x), coeffs);
        EXPECT_NEAR(g[0], 1.0, 1e-12);
        EXPECT_NEAR(g[1], 0.0, 1e-12);
        EXPECT_NEAR(g[2], 0.0, 1e-12);
    }
}

TEST(FunctionSpaceGradients, ScalarJacobianMatchesGradient) {
    H1Space space(ElementType::Line2, 1);
    std::vector<Real> coeffs = {Real(0), Real(2)};

    const auto xi = xi1(Real(0.2));
    const auto g = space.evaluate_gradient(xi, coeffs);
    const auto J = space.evaluate_jacobian(xi, coeffs);

    EXPECT_NEAR(J(0, 0), g[0], 1e-12);
    EXPECT_NEAR(J(0, 1), g[1], 1e-12);
    EXPECT_NEAR(J(0, 2), g[2], 1e-12);
    EXPECT_NEAR(J(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(J(2, 0), 0.0, 1e-12);
}

TEST(FunctionSpaceGradients, VectorValuedSpacesExposeJacobianWithoutChangingGradientSemantics) {
    HCurlSpace space(ElementType::Quad4, 0);
    std::vector<Real> coeffs(space.dofs_per_element(), Real(1));
    EXPECT_THROW(space.evaluate_gradient(FunctionSpace::Value{}, coeffs), svmp::FE::FEException);

    const FunctionSpace::Value xi{Real(0.1), Real(-0.2), Real(0)};
    const auto J = space.evaluate_jacobian(xi, coeffs);

    const Real eps = Real(1e-6);
    for (int d = 0; d < 2; ++d) {
        FunctionSpace::Value xf = xi;
        FunctionSpace::Value xb = xi;
        xf[static_cast<std::size_t>(d)] += eps;
        xb[static_cast<std::size_t>(d)] -= eps;
        const auto vf = space.evaluate(xf, coeffs);
        const auto vb = space.evaluate(xb, coeffs);
        const auto fd = (vf - vb) / (Real(2) * eps);
        for (int comp = 0; comp < 3; ++comp) {
            EXPECT_NEAR(J(static_cast<std::size_t>(comp), static_cast<std::size_t>(d)),
                        fd[static_cast<std::size_t>(comp)],
                        1e-8);
        }
    }
}

TEST(FunctionSpaceGradients, HCurlJacobianUsesAnalyticVectorBasisDerivatives) {
    HCurlSpace space(ElementType::Quad4, 0, BasisType::Nedelec);
    const std::vector<Real> coeffs = {Real(2), Real(-3), Real(5), Real(7)};
    ASSERT_EQ(coeffs.size(), space.dofs_per_element());

    const FunctionSpace::Value xi{Real(0.1), Real(-0.2), Real(0)};
    const auto J = space.evaluate_jacobian(xi, coeffs);

    EXPECT_NEAR(J(0, 0), 0.0, 1e-15);
    EXPECT_NEAR(J(0, 1), -1.75, 1e-15);
    EXPECT_NEAR(J(1, 0), 1.0, 1e-15);
    EXPECT_NEAR(J(1, 1), 0.0, 1e-15);
    EXPECT_NEAR(J(2, 0), 0.0, 1e-15);
    EXPECT_NEAR(J(2, 1), 0.0, 1e-15);
}

TEST(FunctionSpaceGradients, HDivJacobianMatchesCoefficientWeightedBasisJacobians) {
    HDivSpace space(ElementType::Triangle3, 0, BasisType::RaviartThomas);
    const std::vector<Real> coeffs = {Real(1.25), Real(-0.5), Real(2.0)};
    ASSERT_EQ(coeffs.size(), space.dofs_per_element());

    const FunctionSpace::Value xi{Real(0.2), Real(0.3), Real(0)};
    const auto J = space.evaluate_jacobian(xi, coeffs);

    std::vector<basis::VectorJacobian> basis_jacobians;
    space.element().basis().evaluate_vector_jacobians(xi, basis_jacobians);
    ASSERT_EQ(basis_jacobians.size(), coeffs.size());

    FunctionSpace::Jacobian expected{};
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                expected(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) +=
                    coeffs[i] * basis_jacobians[i](static_cast<std::size_t>(r), static_cast<std::size_t>(c));
            }
        }
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(J(static_cast<std::size_t>(r), static_cast<std::size_t>(c)),
                        expected(static_cast<std::size_t>(r), static_cast<std::size_t>(c)),
                        1e-15);
        }
    }
}

TEST(FunctionSpaceGradients, BDMJacobianMatchesAnalyticReferenceFormula) {
    HDivSpace space(ElementType::Quad4, 1, BasisType::BDM);
    std::vector<Real> coeffs(space.dofs_per_element(), Real(0));
    ASSERT_EQ(coeffs.size(), 8u);
    coeffs[1] = Real(2.0);
    coeffs[3] = Real(-0.5);
    coeffs[5] = Real(1.5);
    coeffs[7] = Real(-3.0);

    const FunctionSpace::Value xi{Real(0.25), Real(-0.4), Real(0)};
    const auto J = space.evaluate_jacobian(xi, coeffs);

    const Real x = xi[0];
    const Real y = xi[1];
    const Real expected_x_x = coeffs[3] * Real(0.5) * y + coeffs[7] * Real(0.5) * y;
    const Real expected_x_y = coeffs[3] * Real(0.5) * (Real(1) + x) +
                              coeffs[7] * Real(0.5) * (x - Real(1));
    const Real expected_y_x = coeffs[1] * Real(0.5) * (y - Real(1)) +
                              coeffs[5] * Real(0.5) * (Real(1) + y);
    const Real expected_y_y = coeffs[1] * Real(0.5) * x + coeffs[5] * Real(0.5) * x;

    EXPECT_NEAR(J(0, 0), expected_x_x, 1e-15);
    EXPECT_NEAR(J(0, 1), expected_x_y, 1e-15);
    EXPECT_NEAR(J(1, 0), expected_y_x, 1e-15);
    EXPECT_NEAR(J(1, 1), expected_y_y, 1e-15);
    EXPECT_NEAR(J(2, 0), 0.0, 1e-15);
    EXPECT_NEAR(J(2, 1), 0.0, 1e-15);
}

TEST(FunctionSpaceGradients, CompatibleTensorVectorSpaceJacobianUsesAnalyticBasisDerivatives) {
    elements::ElementRequest req;
    req.element_type = ElementType::Quad4;
    req.basis_type = BasisType::BSpline;
    req.field_type = FieldType::Vector;
    req.continuity = Continuity::H_div;
    req.order = 2;
    req.axis_orders = {2, 2};
    req.axis_knot_vectors = {
        {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)},
        {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)}
    };

    HDivSpace space(req);
    std::vector<Real> coeffs(space.dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = Real(0.1) * static_cast<Real>(i + 1);
    }

    const FunctionSpace::Value xi{Real(0.15), Real(-0.25), Real(0)};
    const auto J = space.evaluate_jacobian(xi, coeffs);

    std::vector<basis::VectorJacobian> basis_jacobians;
    space.element().basis().evaluate_vector_jacobians(xi, basis_jacobians);
    ASSERT_EQ(basis_jacobians.size(), coeffs.size());

    FunctionSpace::Jacobian expected{};
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                expected(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) +=
                    coeffs[i] * basis_jacobians[i](static_cast<std::size_t>(r), static_cast<std::size_t>(c));
            }
        }
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(J(static_cast<std::size_t>(r), static_cast<std::size_t>(c)),
                        expected(static_cast<std::size_t>(r), static_cast<std::size_t>(c)),
                        1e-13);
        }
    }
}
