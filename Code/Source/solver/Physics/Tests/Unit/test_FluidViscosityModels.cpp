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
#include "FE/Forms/Value.h"

#include <cmath>

namespace svmp {
namespace Physics {
namespace test {

TEST(NewtonianViscosity, ReturnsConstantAndZeroDerivative)
{
    materials::fluid::NewtonianViscosity model(/*mu=*/0.01);

    FE::forms::Value<FE::Real> in;
    in.kind = FE::forms::Value<FE::Real>::Kind::Scalar;
    in.s = 5.0;

    const auto out = model.evaluate(in, /*dim=*/3);
    ASSERT_EQ(out.kind, FE::forms::Value<FE::Real>::Kind::Scalar);
    EXPECT_NEAR(out.s, 0.01, 0.0);

    FE::forms::DualWorkspace ws;
    ws.reset(1);

    FE::forms::Value<FE::forms::Dual> in_dual;
    in_dual.kind = FE::forms::Value<FE::forms::Dual>::Kind::Scalar;
    in_dual.s = FE::forms::makeDual(in.s, ws.alloc());
    in_dual.s.deriv[0] = 1.0;

    const auto out_dual = model.evaluate(in_dual, /*dim=*/3, ws);
    ASSERT_EQ(out_dual.kind, FE::forms::Value<FE::forms::Dual>::Kind::Scalar);
    EXPECT_NEAR(out_dual.s.value, 0.01, 0.0);
    ASSERT_EQ(out_dual.s.deriv.size(), 1u);
    EXPECT_NEAR(out_dual.s.deriv[0], 0.0, 0.0);
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

