/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"
#include "Physics/Materials/Fluid/NewtonianViscosity.h"

#include "FE/Forms/Dual.h"
#include "FE/Forms/PointEvaluator.h"
#include "FE/Forms/Value.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace svmp {
namespace Physics {
namespace test {

TEST(NewtonianViscosity, ExpandsToTheConstantAndPublishesItsMetadata)
{
    const auto law = materials::fluid::newtonianViscosity(/*mu=*/0.01);
    EXPECT_EQ(law.model()->kindId(),
              FE::forms::InlinableConstitutiveModel::fnv1a64("Physics::materials::fluid::NewtonianViscosity"));
    const std::vector<FE::forms::FormExpr> inputs{FE::forms::FormExpr::constant(5.0)};
    const auto expansion = law.model()->inlineExpand(inputs, FE::forms::InlinableConstitutiveContext{.dim = 3});
    ASSERT_EQ(expansion.outputs.size(), 1u);
    EXPECT_EQ(expansion.outputs.front().toString(), FE::forms::FormExpr::constant(0.01).toString());
    const auto metadata = law.model()->constitutiveLawMetadata(0u);
    ASSERT_TRUE(metadata.has_value());
    EXPECT_EQ(metadata->role, FE::analysis::ConstitutiveLawRole::DynamicViscosity);
    EXPECT_TRUE(metadata->constant_value_available);
    EXPECT_EQ(metadata->constant_value, 0.01);
    EXPECT_THROW((void)materials::fluid::newtonianViscosity(0.0), std::invalid_argument);
}

// The opt-in Carreau-Yasuda law written as forms matches the class to round-off.
TEST(CarreauYasudaViscosity, LawWrittenAsFormsMatchesTheClass)
{
    const materials::fluid::CarreauYasudaViscosity model(0.056, 0.0035, 3.313, 0.3568, 2.0);
    const auto law = materials::fluid::carreauYasudaViscosityLaw(0.056, 0.0035, 3.313, 0.3568, 2.0);
    ASSERT_TRUE(law.model()->constitutiveLawMetadata(0u).has_value());
    EXPECT_EQ(law.model()->constitutiveLawMetadata(0u)->role, FE::analysis::ConstitutiveLawRole::DynamicViscosity);
    for (const FE::Real gamma : {0.0, 0.5, 12.5, 400.0}) {
        SCOPED_TRACE(gamma);
        const std::vector<FE::forms::FormExpr> inputs{FE::forms::FormExpr::constant(gamma)};
        const auto expansion = law.model()->inlineExpand(inputs, FE::forms::InlinableConstitutiveContext{.dim = 3});
        const FE::Real from_law = FE::forms::evaluateScalarAt(expansion.outputs.front(), FE::forms::PointEvalContext{});
        const FE::Real from_class =
            model.evaluate(FE::forms::Value<FE::Real>{FE::forms::Value<FE::Real>::Kind::Scalar, gamma}, 3).s;
        EXPECT_NEAR(from_law, from_class, 1e-14);
    }
}

TEST(CarreauYasudaViscosity, DerivativeMatchesFiniteDifference)
{
    materials::fluid::CarreauYasudaViscosity model(
        /*mu0=*/0.056, /*mu_inf=*/0.0035, /*lambda=*/3.313, /*n=*/0.3568, /*a=*/2.0);

    const FE::Real gamma = 12.5;
    const FE::Real eps = 1e-6;

    FE::forms::Value<FE::Real> in;
    in.kind = FE::forms::Value<FE::Real>::Kind::Scalar;
    in.s = gamma;

    const auto mu = model.evaluate(in, /*dim=*/3).s;
    const auto mu_p = model.evaluate(FE::forms::Value<FE::Real>{FE::forms::Value<FE::Real>::Kind::Scalar, gamma + eps}, 3).s;
    const auto mu_m = model.evaluate(FE::forms::Value<FE::Real>{FE::forms::Value<FE::Real>::Kind::Scalar, gamma - eps}, 3).s;
    const FE::Real fd = (mu_p - mu_m) / (2.0 * eps);

    FE::forms::DualWorkspace ws;
    ws.reset(1);

    FE::forms::Value<FE::forms::Dual> in_dual;
    in_dual.kind = FE::forms::Value<FE::forms::Dual>::Kind::Scalar;
    in_dual.s = FE::forms::makeDual(gamma, ws.alloc());
    in_dual.s.deriv[0] = 1.0;

    const auto out_dual = model.evaluate(in_dual, /*dim=*/3, ws);
    ASSERT_EQ(out_dual.kind, FE::forms::Value<FE::forms::Dual>::Kind::Scalar);
    EXPECT_NEAR(out_dual.s.value, mu, 1e-14);

    const FE::Real ad = out_dual.s.deriv[0];
    EXPECT_NEAR(ad, fd, std::max(1e-10, 5e-6 * std::abs(fd)));
}

} // namespace test
} // namespace Physics
} // namespace svmp

