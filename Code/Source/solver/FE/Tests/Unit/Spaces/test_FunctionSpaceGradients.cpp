/**
 * @file test_FunctionSpaceGradients.cpp
 * @brief Unit tests for FunctionSpace::evaluate_gradient
 */

#include <gtest/gtest.h>

#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/HCurlSpace.h"

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

TEST(FunctionSpaceGradients, VectorValuedSpacesThrow) {
    HCurlSpace space(ElementType::Quad4, 0);
    std::vector<Real> coeffs(space.dofs_per_element(), Real(1));
    EXPECT_THROW(space.evaluate_gradient(FunctionSpace::Value{}, coeffs), svmp::FE::FEException);
}

