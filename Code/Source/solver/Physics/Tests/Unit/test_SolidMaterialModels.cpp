/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Materials/Solid/LinearElasticStress.h"
#if defined(SVMP_TEST_HAS_ISOCHORIC_NEOHOOKEAN_PK1)
#include "Physics/Materials/Solid/IsochoricNeoHookeanPK1.h"
#endif
#include "Physics/Materials/Solid/NeoHookeanPK1.h"

#include "FE/Forms/Dual.h"
#include "FE/Forms/Value.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/SystemManifest.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace svmp {
namespace Physics {
namespace test {

TEST(LinearElasticStress, MatchesClosedFormAndDerivative)
{
    const FE::Real lambda = 1.0;
    const FE::Real mu = 2.0;
    materials::solid::LinearElasticStress model(lambda, mu);

    FE::forms::Value<FE::Real> eps;
    eps.kind = FE::forms::Value<FE::Real>::Kind::SymmetricMatrix;
    eps.resizeMatrix(3, 3);
    eps.matrixAt(0, 0) = 0.10;
    eps.matrixAt(1, 1) = -0.05;
    eps.matrixAt(2, 2) = 0.02;
    eps.matrixAt(0, 1) = 0.03;
    eps.matrixAt(1, 0) = 0.03;

    const auto sig = model.evaluate(eps, /*dim=*/3);
    ASSERT_EQ(sig.kind, FE::forms::Value<FE::Real>::Kind::SymmetricMatrix);

    const FE::Real tr = eps.matrixAt(0, 0) + eps.matrixAt(1, 1) + eps.matrixAt(2, 2);
    const FE::Real expected00 = lambda * tr + 2.0 * mu * eps.matrixAt(0, 0);
    const FE::Real expected11 = lambda * tr + 2.0 * mu * eps.matrixAt(1, 1);
    const FE::Real expected01 = 2.0 * mu * eps.matrixAt(0, 1);

    EXPECT_NEAR(sig.matrixAt(0, 0), expected00, 1e-14);
    EXPECT_NEAR(sig.matrixAt(1, 1), expected11, 1e-14);
    EXPECT_NEAR(sig.matrixAt(0, 1), expected01, 1e-14);

    FE::forms::DualWorkspace ws;
    ws.reset(1);

    FE::forms::Value<FE::forms::Dual> eps_dual;
    eps_dual.kind = FE::forms::Value<FE::forms::Dual>::Kind::SymmetricMatrix;
    eps_dual.resizeMatrix(3, 3);

    // Seed derivative w.r.t eps_00.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const FE::Real val = eps.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            eps_dual.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                FE::forms::makeDualConstant(val, ws.alloc());
        }
    }
    eps_dual.matrixAt(0, 0).deriv[0] = 1.0;

    const auto sig_dual = model.evaluate(eps_dual, /*dim=*/3, ws);
    ASSERT_EQ(sig_dual.kind, FE::forms::Value<FE::forms::Dual>::Kind::SymmetricMatrix);

    // d sigma_00 / d eps_00 = lambda + 2 mu
    EXPECT_NEAR(sig_dual.matrixAt(0, 0).deriv[0], lambda + 2.0 * mu, 1e-12);
    // d sigma_11 / d eps_00 = lambda
    EXPECT_NEAR(sig_dual.matrixAt(1, 1).deriv[0], lambda, 1e-12);
    // d sigma_01 / d eps_00 = 0
    EXPECT_NEAR(sig_dual.matrixAt(0, 1).deriv[0], 0.0, 1e-12);
}

TEST(NeoHookeanPK1, IdentityDeformationGivesZeroStress)
{
    materials::solid::NeoHookeanPK1 model(/*lambda=*/10.0, /*mu=*/2.0);

    FE::forms::Value<FE::Real> F;
    F.kind = FE::forms::Value<FE::Real>::Kind::Matrix;
    F.resizeMatrix(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            F.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = (i == j) ? 1.0 : 0.0;
        }
    }

    const auto P = model.evaluate(F, /*dim=*/3);
    ASSERT_EQ(P.kind, FE::forms::Value<FE::Real>::Kind::Matrix);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(P.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), 0.0, 1e-14);
        }
    }
}

TEST(NeoHookeanPK1, DerivativeMatchesFiniteDifferenceForSingleEntry)
{
    materials::solid::NeoHookeanPK1 model(/*lambda=*/10.0, /*mu=*/2.0);

    FE::forms::Value<FE::Real> F;
    F.kind = FE::forms::Value<FE::Real>::Kind::Matrix;
    F.resizeMatrix(3, 3);
    // A mildly deformed, positive-J configuration.
    F.matrixAt(0, 0) = 1.10;
    F.matrixAt(0, 1) = 0.02;
    F.matrixAt(0, 2) = -0.01;
    F.matrixAt(1, 0) = 0.03;
    F.matrixAt(1, 1) = 0.95;
    F.matrixAt(1, 2) = 0.00;
    F.matrixAt(2, 0) = 0.00;
    F.matrixAt(2, 1) = -0.02;
    F.matrixAt(2, 2) = 1.05;

    const FE::Real eps = 1e-6;
    auto Fp = F;
    auto Fm = F;
    Fp.matrixAt(0, 0) += eps;
    Fm.matrixAt(0, 0) -= eps;

    const FE::Real P00 = model.evaluate(F, /*dim=*/3).matrixAt(0, 0);
    const FE::Real P00p = model.evaluate(Fp, /*dim=*/3).matrixAt(0, 0);
    const FE::Real P00m = model.evaluate(Fm, /*dim=*/3).matrixAt(0, 0);
    const FE::Real fd = (P00p - P00m) / (2.0 * eps);

    FE::forms::DualWorkspace ws;
    ws.reset(1);

    FE::forms::Value<FE::forms::Dual> Fd;
    Fd.kind = FE::forms::Value<FE::forms::Dual>::Kind::Matrix;
    Fd.resizeMatrix(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const FE::Real val = F.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            Fd.matrixAt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                FE::forms::makeDualConstant(val, ws.alloc());
        }
    }
    Fd.matrixAt(0, 0).deriv[0] = 1.0;

    const auto Pd = model.evaluate(Fd, /*dim=*/3, ws);
    const FE::Real ad = Pd.matrixAt(0, 0).deriv[0];

    EXPECT_NEAR(Pd.matrixAt(0, 0).value, P00, 1e-14);
    EXPECT_NEAR(ad, fd, std::max(1e-9, 2e-5 * std::abs(fd)));
}

#if defined(SVMP_TEST_HAS_ISOCHORIC_NEOHOOKEAN_PK1)
// The law written as forms expands to the tree of the former class's
// inlineExpand (written out here as it was) with the same kind id, so the
// kernels that inline it are unchanged.
TEST(IsochoricNeoHookeanPK1, ExpandsToTheTreeAndKindIdOfTheFormerClass)
{
    using FE::forms::FormExpr;
    const FE::Real mu = 2.0;
    const auto law = materials::solid::isochoricNeoHookeanPK1(mu);
    EXPECT_EQ(law.model()->kindId(),
              FE::forms::InlinableConstitutiveModel::fnv1a64("Physics::materials::solid::IsochoricNeoHookeanPK1"));
    EXPECT_EQ(law.model()->expectedInputKind(), FE::forms::ConstitutiveModel::ValueKind::Matrix);
    EXPECT_EQ(law.model()->outputSpec(0u).kind, FE::forms::ConstitutiveModel::ValueKind::Matrix);
    EXPECT_THROW((void)materials::solid::isochoricNeoHookeanPK1(0.0), std::invalid_argument);

    auto space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, 1);
    FE::spaces::ProductSpace vector(space, 3);
    const auto u = FormExpr::trialFunction(vector, "u");
    for (const int dim : {2, 3}) {
        SCOPED_TRACE(dim);
        const auto F = FormExpr::identity(dim) + FE::forms::grad(u);
        const std::vector<FormExpr> inputs{F};
        const auto expansion = law.model()->inlineExpand(inputs, FE::forms::InlinableConstitutiveContext{.dim = dim});
        ASSERT_EQ(expansion.outputs.size(), 1u);

        const auto constant = [](FE::Real value) {
            return FormExpr::constant(value);
        };
        const auto C = FE::forms::transpose(F) * F;
        const auto Cinv = FE::forms::inv(C);
        const auto J = FE::forms::det(F);
        const auto Jm2d = FE::forms::pow(J, constant(-2.0 / static_cast<FE::Real>(dim)));
        const auto I = FormExpr::identity(dim);
        const auto Siso =
            constant(mu) * Jm2d *
            (I - (FE::forms::trace(C) / constant(static_cast<FE::Real>(dim))) * Cinv);
        const auto former = F * Siso;
        EXPECT_EQ(FE::systems::formExprTreeHash(expansion.outputs.front()), FE::systems::formExprTreeHash(former));
        EXPECT_EQ(expansion.outputs.front().toString(), former.toString());
    }
}
#endif

} // namespace test
} // namespace Physics
} // namespace svmp
