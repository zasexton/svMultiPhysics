/**
 * @file test_AuxiliaryStateBuilder.cpp
 * @brief Unit tests for AuxiliaryStateBuilder convenience API
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Systems/AuxiliaryStateBuilder.h"

using svmp::FE::Real;

TEST(AuxiliaryStateBuilder, BuildsScalarRegistration)
{
    using namespace svmp::FE;

    forms::BoundaryFunctional Q;
    Q.integrand = forms::FormExpr::constant(1.0);
    Q.boundary_marker = 3;
    Q.name = "Q";
    Q.reduction = forms::BoundaryFunctional::Reduction::Sum;

    auto reg = systems::auxiliaryODE("X", Real(2.0))
                   .requiresIntegral(Q)
                   .withRHS(forms::FormExpr::boundaryIntegralValue("Q") - forms::FormExpr::auxiliaryState("X"))
                   .withIntegrator(systems::ODEMethod::BackwardEuler)
                   .build();

    EXPECT_EQ(reg.spec.size, 1);
    EXPECT_EQ(reg.spec.name, "X");
    ASSERT_EQ(reg.initial_values.size(), 1u);
    EXPECT_DOUBLE_EQ(reg.initial_values[0], 2.0);
    ASSERT_EQ(reg.required_integrals.size(), 1u);
    EXPECT_EQ(reg.required_integrals[0].name, "Q");
    EXPECT_EQ(reg.integrator, systems::ODEMethod::BackwardEuler);
    EXPECT_TRUE(reg.rhs.isValid());
}
