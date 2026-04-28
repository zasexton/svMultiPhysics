/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"
#include "Physics/Materials/Solid/LinearElasticStress.h"
#include "Physics/Materials/Solid/NeoHookeanPK1.h"
#include "Physics/Tests/Unit/FiniteDeformationSolidTestSupport.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Forms/Vocabulary.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <memory>
#include <vector>

namespace svmp {
namespace Physics {
namespace test {

TEST(FormIntegrationJacobian, NonlinearViscosityForm_MatchesFiniteDifference)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, /*order=*/1);

    FE::systems::FieldSpec spec;
    spec.name = "u";
    spec.space = space;
    const FE::FieldId u_id = system.addField(std::move(spec));

    system.addOperator("residual");
    system.addOperator("jacobian");

    using namespace svmp::FE::forms;

    auto u = FormExpr::stateField(u_id, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    auto mu_model = std::make_shared<materials::fluid::CarreauYasudaViscosity>(
        /*mu0=*/0.056, /*mu_inf=*/0.0035, /*lambda=*/3.313, /*n=*/0.3568, /*a=*/2.0);
    auto gamma = norm(grad(u));
    auto mu = FormExpr::constitutive(mu_model, gamma);

    const auto residual = (mu * inner(grad(u), grad(v))).dx();

    FE::systems::installFormulation(system, "residual", {u_id}, residual);
    FE::systems::installFormulation(system, "jacobian", {u_id}, residual);

    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 4);

    std::vector<FE::Real> uvec(static_cast<std::size_t>(n));
    uvec[0] = 0.00;
    uvec[1] = 0.20;
    uvec[2] = -0.10;
    uvec[3] = 0.05;

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(uvec);

    expectJacobianMatchesCentralFD(system, state, /*eps=*/1e-6, /*rtol=*/2e-4, /*atol=*/1e-9);
}

TEST(FormIntegrationJacobian, LinearElasticityForm_MatchesFiniteDifference)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space =
        FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, /*order=*/1, /*components=*/3);

    FE::systems::FieldSpec spec;
    spec.name = "u";
    spec.space = space;
    spec.components = 3;
    const FE::FieldId u_id = system.addField(std::move(spec));

    system.addOperator("residual");
    system.addOperator("jacobian");

    using namespace svmp::FE::forms;

    auto u = FormExpr::stateField(u_id, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    auto model = std::make_shared<materials::solid::LinearElasticStress>(/*lambda=*/1.0, /*mu=*/2.0);
    auto eps_u = sym(grad(u));
    auto eps_v = sym(grad(v));
    auto sigma = FormExpr::constitutive(model, eps_u);

    const auto residual = inner(sigma, eps_v).dx();

    FE::systems::installFormulation(system, "residual", {u_id}, residual);
    FE::systems::installFormulation(system, "jacobian", {u_id}, residual);

    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 12);

    std::vector<FE::Real> uvec(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < uvec.size(); ++i) {
        uvec[i] = static_cast<FE::Real>(0.01 * (static_cast<int>(i) - 6));
    }

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(uvec);

    expectJacobianMatchesCentralFD(system, state, /*eps=*/1e-6, /*rtol=*/1e-6, /*atol=*/1e-10);
}

TEST(FormIntegrationJacobian, NeoHookeanForm_MatchesFiniteDifference)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space =
        FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, /*order=*/1, /*components=*/3);

    FE::systems::FieldSpec spec;
    spec.name = "u";
    spec.space = space;
    spec.components = 3;
    const FE::FieldId u_id = system.addField(std::move(spec));

    system.addOperator("residual");
    system.addOperator("jacobian");

    using namespace svmp::FE::forms;

    auto u = FormExpr::stateField(u_id, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    auto model = std::make_shared<materials::solid::NeoHookeanPK1>(/*lambda=*/10.0, /*mu=*/2.0);
    const auto residual =
        test_support::solid::totalLagrangianPK1Residual(u, v, model, 3);

    FE::systems::installFormulation(system, "residual", {u_id}, residual);
    FE::systems::installFormulation(system, "jacobian", {u_id}, residual);

    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 12);

    std::vector<FE::Real> uvec(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < uvec.size(); ++i) {
        uvec[i] = static_cast<FE::Real>(0.001 * (static_cast<int>(i) - 5));
    }

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(uvec);

    expectJacobianMatchesCentralFD(system, state, /*eps=*/1e-6, /*rtol=*/5e-4, /*atol=*/1e-9);
}

TEST(FormIntegrationJacobian, SolidFiniteDeformationHelpersBuildFollowerAndGeometricStiffnessTerms)
{
    auto space =
        FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, /*order=*/1, /*components=*/3);

    using namespace svmp::FE::forms;

    const FE::FieldId u_id{0};
    auto u = FormExpr::stateField(u_id, *space, "u");
    auto du = FormExpr::trialFunction(*space, "du");
    auto v = FormExpr::testFunction(*space, "v");
    auto F = finite_deformation::deformationGradient(u, 3);
    auto pressure = FormExpr::constant(2.0);
    auto initial_stress = FormExpr::identity(3);

    const auto follower =
        test_support::solid::followerPressureReferenceResidual(pressure, v, F, 2);
    const auto geometric_stiffness =
        test_support::solid::initialStressGeometricStiffnessResidual(initial_stress,
                                                                    du,
                                                                    v);

    EXPECT_TRUE(follower.isValid());
    EXPECT_TRUE(geometric_stiffness.isValid());
}

} // namespace test
} // namespace Physics
} // namespace svmp
