/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/Poisson/PoissonModule.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Systems/FESystem.h"

#include <memory>
#include <vector>

namespace svmp {
namespace Physics {
namespace test {

TEST(PoissonModule, AssembledJacobianMatchesFiniteDifference)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.field_name = "u";
    opts.diffusion = 1.0;
    opts.source = 0.25;

    formulations::poisson::PoissonModule module(space, opts);
    module.registerOn(system);

    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 4);

    std::vector<FE::Real> u(static_cast<std::size_t>(n));
    u[0] = 0.10;
    u[1] = -0.05;
    u[2] = 0.20;
    u[3] = 0.03;

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);

    FE::assembly::DenseMatrixView J(n);
    const auto jr = system.assembleJacobian(state, J);
    ASSERT_TRUE(jr.success) << jr.error_message;
    EXPECT_TRUE(J.isSymmetric(1e-12));

    expectJacobianMatchesCentralFD(system, state, /*eps=*/1e-6, /*rtol=*/1e-6, /*atol=*/1e-10);
}

} // namespace test
} // namespace Physics
} // namespace svmp

