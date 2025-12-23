/**
 * @file test_VectorSpaceOperators.cpp
 * @brief Unit tests for vector-space differential operators (curl/div)
 */

#include <gtest/gtest.h>

#include "FE/Spaces/HCurlSpace.h"
#include "FE/Spaces/HDivSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

namespace {

FunctionSpace::Value xi2(Real x, Real y) {
    FunctionSpace::Value xi{};
    xi[0] = x;
    xi[1] = y;
    xi[2] = Real(0);
    return xi;
}

} // namespace

TEST(VectorSpaceOperators, HDivEvaluateDivergenceMatchesBasisCombination) {
    HDivSpace space(ElementType::Quad4, 0);
    ASSERT_EQ(space.dofs_per_element(), 4u);

    // For Quad4 RT0, the basis divergences are {+0.5, -0.5, +0.5, -0.5}.
    std::vector<Real> coeffs = {Real(1), Real(2), Real(3), Real(4)};
    const Real expected = Real(0.5) * (coeffs[0] - coeffs[1] + coeffs[2] - coeffs[3]);

    const Real div0 = space.evaluate_divergence(xi2(Real(0), Real(0)), coeffs);
    const Real div1 = space.evaluate_divergence(xi2(Real(0.25), Real(-0.5)), coeffs);
    EXPECT_NEAR(div0, expected, 1e-12);
    EXPECT_NEAR(div1, expected, 1e-12);
}

TEST(VectorSpaceOperators, HCurlEvaluateCurlMatchesBasisCombination) {
    HCurlSpace space(ElementType::Quad4, 0);
    ASSERT_EQ(space.dofs_per_element(), 4u);

    // For Quad4 Nedelec0, each basis curl is (0,0,0.25).
    std::vector<Real> coeffs = {Real(1), Real(2), Real(3), Real(4)};
    const Real expected_z = Real(0.25) * (coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3]);

    const auto curl = space.evaluate_curl(xi2(Real(0), Real(0)), coeffs);
    EXPECT_NEAR(curl[0], 0.0, 1e-12);
    EXPECT_NEAR(curl[1], 0.0, 1e-12);
    EXPECT_NEAR(curl[2], expected_z, 1e-12);
}

