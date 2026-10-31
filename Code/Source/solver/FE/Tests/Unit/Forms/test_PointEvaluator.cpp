/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/TimeIntegrationContext.h"
#include "Forms/PointEvaluator.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(PointEvaluatorTest, EvaluatesTimeAndTimeStep)
{
    PointEvalContext ctx;
    ctx.x = {1.0, 2.0, 3.0};
    ctx.time = 2.5;
    ctx.dt = 0.1;

    EXPECT_DOUBLE_EQ(evaluateScalarAt(t(), ctx), 2.5);
    EXPECT_DOUBLE_EQ(evaluateScalarAt(deltat(), ctx), 0.1);
    EXPECT_DOUBLE_EQ(evaluateScalarAt(t() + deltat(), ctx), 2.6);
}

TEST(PointEvaluatorTest, EffectiveTimeStepIsOneOverA0OfTheStencil)
{
    PointEvalContext ctx;
    ctx.dt = 0.1;
    EXPECT_DOUBLE_EQ(evaluateScalarAt(deltat_eff(), ctx), 0.1);   // no stencil: the step

    // First-order generalized-alpha, rho_inf = 0.5: a0 = alpha_m / (gamma alpha_f dt) = 18.75.
    assembly::TimeIntegrationContext time_ctx;
    assembly::TimeDerivativeStencil dt1;
    dt1.order = 1;
    dt1.a = {18.75, -18.75, 0.0};
    time_ctx.dt1 = dt1;
    ctx.time_integration = &time_ctx;
    EXPECT_DOUBLE_EQ(evaluateScalarAt(deltat_eff(), ctx), 1.0 / 18.75);
    EXPECT_DOUBLE_EQ(evaluateScalarAt(deltat_eff(), ctx), assembly::effectiveTimeStep(&time_ctx, ctx.dt));
}

TEST(PointEvaluatorTest, EvaluatesSpatialAndTimeCoefficients)
{
    const auto g = FormExpr::coefficient("g", [](Real x, Real y, Real z) { return x + 2.0 * y + 3.0 * z; });
    const auto h = FormExpr::coefficient("h", [](Real x, Real, Real, Real t) { return x + t; });

    PointEvalContext ctx;
    ctx.x = {1.0, 2.0, 3.0};
    ctx.time = 4.0;
    ctx.dt = 0.0;

    EXPECT_DOUBLE_EQ(evaluateScalarAt(g, ctx), 14.0);
    EXPECT_DOUBLE_EQ(evaluateScalarAt(h, ctx), 5.0);
}

TEST(PointEvaluatorTest, DetectsTimeDependence)
{
    EXPECT_FALSE(isTimeDependent(FormExpr::constant(1.0)));
    EXPECT_TRUE(isTimeDependent(t()));

    const auto spatial = FormExpr::coefficient("g", [](Real x, Real, Real) { return x; });
    const auto temporal = FormExpr::coefficient("h", [](Real x, Real, Real, Real t) { return x + t; });

    EXPECT_FALSE(isTimeDependent(spatial));
    EXPECT_TRUE(isTimeDependent(temporal));
    EXPECT_TRUE(isTimeDependent(temporal + deltat()));
}

TEST(PointEvaluatorTest, RejectsTestTrialFunctions)
{
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto v = TestFunction(space, "v");

    PointEvalContext ctx;
    ctx.x = {0.0, 0.0, 0.0};
    ctx.time = 0.0;

    EXPECT_THROW((void)evaluateScalarAt(v, ctx), std::invalid_argument);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

