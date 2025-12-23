/**
 * @file test_MixedSpaceEvaluation.cpp
 * @brief Unit tests for MixedSpace component-wise evaluation helpers
 */

#include <gtest/gtest.h>

#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/L2Space.h"
#include "FE/Spaces/MixedSpace.h"

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

TEST(MixedSpace, EvaluateComponentsOnLine) {
    MixedSpace mixed;
    auto u = std::make_shared<H1Space>(ElementType::Line2, 1);
    auto p = std::make_shared<L2Space>(ElementType::Line2, 1);
    mixed.add_component("u", u);
    mixed.add_component("p", p);

    const auto dofs_u = u->dofs_per_element();
    const auto dofs_p = p->dofs_per_element();
    ASSERT_EQ(dofs_u, 2u);
    ASSERT_EQ(dofs_p, 2u);

    // u(x) = 1 + x -> nodal values [u(-1), u(1)] = [0, 2]
    // p(x) = 3 -> nodal values [3, 3]
    std::vector<Real> coeffs;
    coeffs.insert(coeffs.end(), {Real(0), Real(2)});
    coeffs.insert(coeffs.end(), {Real(3), Real(3)});
    ASSERT_EQ(coeffs.size(), mixed.dofs_per_element());

    const auto xi = xi1(Real(0));
    const auto v_u = mixed.evaluate_component(0, xi, coeffs);
    const auto v_p = mixed.evaluate_component(1, xi, coeffs);

    EXPECT_NEAR(v_u[0], 1.0, 1e-12);
    EXPECT_NEAR(v_p[0], 3.0, 1e-12);

    const auto all = mixed.evaluate_components(xi, coeffs);
    ASSERT_EQ(all.size(), 2u);
    EXPECT_NEAR(all[0][0], 1.0, 1e-12);
    EXPECT_NEAR(all[1][0], 3.0, 1e-12);

    EXPECT_THROW(mixed.evaluate(xi, coeffs), svmp::FE::FEException);
}

TEST(MixedSpace, InterpolateThrowsForAggregatedField) {
    MixedSpace mixed;
    mixed.add_component("u", std::make_shared<H1Space>(ElementType::Line2, 1));
    mixed.add_component("p", std::make_shared<L2Space>(ElementType::Line2, 1));

    std::vector<Real> coeffs;
    EXPECT_THROW(
        mixed.interpolate(
            [](const FunctionSpace::Value& xi) {
                FunctionSpace::Value out{};
                out[0] = Real(1) + xi[0];
                return out;
            },
            coeffs),
        svmp::FE::FEException);
}
